""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- Compute the fitness of a model_net using the evolved networks.


"""
import logging
import os
import time
import numpy as np
import torch
from medmnist import INFO, Evaluator
import torch.nn as nn
from torch.amp import GradScaler
from tqdm.notebook import tqdm
from typing import Dict, List, Union, Any
from sklearn.metrics import confusion_matrix
from cnn import model, input, metrics
from util import create_info_file, init_log, load_yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR, MultiStepLR

# Console output always on. The file handler is attached lazily per experiment,
# since the experiment path is only known at call time - see
# _attach_experiment_log_file(). Mirrors cnn/train.py's LOGGER setup.
LOGGER = init_log("INFO", name=__name__)
_log_file_experiment_path = None


def _attach_experiment_log_file(experiment_path: str):
    """Write retrain logs into <experiment_path>/retrain.log instead of a fixed
    repo-wide location, so each experiment's logs live alongside its other
    artifacts. A no-op if already attached for this experiment_path (all
    num_repetitions of one retrain run share the same root experiment_path).
    """
    global _log_file_experiment_path
    if _log_file_experiment_path == experiment_path:
        return

    log_path = os.path.join(experiment_path, 'retrain.log')
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(module)s: %(asctime)s.%(msecs)03d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    LOGGER.addHandler(handler)
    _log_file_experiment_path = experiment_path

def release_gpu_memory(gpu_name='cuda:0'):
    """
    Release GPU memory.
    
    Args:
        gpu_name (str): The name of the GPU device (default is 'cuda').
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPU memory to release.")
        return
    
    if gpu_name == 'cuda':
        gpu_name = 'cuda:0'

    device = torch.device(gpu_name)
    torch.cuda.set_device(device)

    # Get memory usage before clearing the cache
    memory_allocated_before = torch.cuda.memory_allocated(device)
    memory_reserved_before = torch.cuda.memory_reserved(device)

    # Clear the cache
    torch.cuda.empty_cache()

    # Check if there was a significant change
    memory_allocated_after = torch.cuda.memory_allocated(device)
    memory_reserved_after = torch.cuda.memory_reserved(device)

    # Verificar si hubo un cambio significativo
    if memory_allocated_before != memory_allocated_after or memory_reserved_before != memory_reserved_after:
        print("Cache was cleared.")
    else:
        print("Cache was already empty.")

def compute_metrics(model, data_loader, params):
    model.eval()
    all_labels = []
    all_predictions = []
    auc, acc = 0, 0
    y_score = torch.tensor([]).to(params['device'])
    memory_format = torch.channels_last if params.get('channels_last', False) \
        else torch.contiguous_format

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(params['device'], memory_format=memory_format)
            labels = labels.to(params['device'])
            y_logits = model(inputs)
            _, predicted = y_logits.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            if params['task'] == 'multi-class':
                output = y_logits.softmax(dim=-1)
                y_score = torch.cat((y_score, output), 0)

        if params['task'] == 'multi-class':
            y_score = y_score.cpu().detach().numpy()
            evaluator = Evaluator(params['dataset'], split='test', root=params['data_path'])
            metrics = evaluator.evaluate(y_score)
            auc, acc = metrics

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    return conf_matrix, auc, acc

def reset_and_load_best_model(params, best_model_path):
    # Reinitialize the original model
    
    best_model = model.NetworkGraph(num_classes=params["num_classes"],
                                    network_config=params['network_config'], 
                                    network_gap=params['network_gap'])
    filtered_dict = {key: item for key, item in params['fn_dict'].items() if key in params['net_list']}
    best_model.create_functions(fn_dict=filtered_dict, net_list=params['net_list'])

    input_random = torch.randn(params['input_shape'])
    with torch.no_grad():
        _ = best_model(input_random)
    # Load the state dictionary of the best model into the new model
    best_model.load_state_dict(torch.load(best_model_path))
    memory_format = torch.channels_last if params.get('channels_last', False) \
        else torch.contiguous_format
    best_model.to(params['device'], memory_format=memory_format)

    return best_model

def train_epoch(model, criterion, optimizer, data_loader, params, scaler):
    model.train()
    total = 0
    device = torch.device(params['device'])
    amp_device = device.type  # 'cuda' or 'cpu'
    # NHWC memory format lets cuDNN use its fastest Tensor Core conv kernels on
    # Ampere+ under AMP (typical 1.2-1.4x speedup) - the tensor's logical NCHW
    # shape is unchanged, only its physical memory layout.
    memory_format = torch.channels_last if params.get('channels_last', False) \
        else torch.contiguous_format
    # Accumulate on-device and only sync to host once, after the loop - a per-batch
    # .item() call forces a CUDA sync and stalls the async training pipeline (mirrors
    # cnn/train.py's search-phase train_epoch).
    train_loss_t = torch.zeros((), device=device)
    correct_t = torch.zeros((), device=device)

    for inputs, labels in data_loader:
        inputs = inputs.to(device, memory_format=memory_format)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=params['mixed_precision']):
            y_logits = model(inputs)
            if params['task'] == 'multi-class':
                labels = labels.squeeze().long() # medmnist
            loss = criterion(y_logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss_t += loss.detach()
        _, predicted = y_logits.max(1)
        total += labels.size(0)
        correct_t += predicted.eq(labels).sum()

    accuracy = 100 * correct_t.item() / total
    train_loss = train_loss_t.item() / len(data_loader)
    return train_loss, accuracy

def evaluate(model, criterion, data_loader, params, test=False):
    model.eval()
    total = 0
    device = torch.device(params['device'])
    amp_device = device.type  # 'cuda' or 'cpu'
    memory_format = torch.channels_last if params.get('channels_last', False) \
        else torch.contiguous_format
    eval_loss_t = torch.zeros((), device=device)
    correct_t = torch.zeros((), device=device)

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, memory_format=memory_format)
            labels = labels.to(device)
            with torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=params['mixed_precision']):
                y_logits = model(inputs)
                if params['task'] == 'multi-class':
                    labels = labels.squeeze().long() # medmnist
                loss = criterion(y_logits, labels)
            eval_loss_t += loss.detach()
            _, predicted = y_logits.max(1)
            total += labels.size(0)
            correct_t += predicted.eq(labels).sum()

    accuracy = 100 * correct_t.item() / total
    eval_loss = eval_loss_t.item() / len(data_loader)

    if test:
        confusion_matrix, auc, acc = compute_metrics(model, data_loader, params)
        return eval_loss, accuracy, auc, acc , confusion_matrix

    return eval_loss, accuracy

def train(model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        params: Dict[str, Union[int, float, str]]) -> Dict[str, Union[List[float], float]]:
    """
    Retrain a convolutional neural network model.

    Args:
        model (Module): Model to be trained.
        criterion (Module): Loss function.
        optimizer (Optimizer): Optimization algorithm.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        params (Dict[str, Union[int, float, str]]): Dictionary with parameters necessary for training.
            - 'max_epochs' (int): Number of epochs to train.
            - 'model_path' (str): Path to save the trained model.
            - 'lr_scheduler' (str): Learning rate scheduler to use.
            - 'experiment_path' (str): Path to save the experiment.
            - 'device' (str): Device to use for training.

    Returns:
        Dict[str, Union[List[float], float]]: Dictionary with the training results.
        
        - 'training_losses' (List[float]): List of training losses for each epoch.
        - 'training_accuracies' (List[float]): List of training accuracies for each epoch.
        - 'validation_losses' (List[float]): List of validation losses for each epoch.
        - 'validation_accuracies' (List[float]): List of validation accuracies for each epoch.
        - 'best_accuracy' (float): Best validation accuracy achieved.
        - 'test_loss' (float): Loss on the test set.
        - 'test_accuracy' (float): Accuracy on the test set.
        - 'auc_score' (float): AUC score on the test set.
        - 'acc_medmnist' (float): Accuracy on the test set.
        - 'confusion_matrix' (numpy.ndarray): Confusion matrix on the test set.
        - 'total_trainable_params' (int): Total number of trainable parameters in the model.
    """
    model.train()
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []
    best_accuracy = 0.0
    best_validation_loss = float('inf')
    auc_value = 0.0
    acc_med = 0.0
    training_results = {}
    max_epochs = params['max_epochs']
    milestones = [0.5 * max_epochs, 0.75 * max_epochs]

    # Early stopping: stop retraining once validation loss plateaus instead of
    # always running the full max_epochs (mirrors cnn/train.py's search-phase
    # early stopping). Disabled by default - opt in via --early_stopping_enabled.
    es_enabled = params.get('early_stopping_enabled', False)
    es_patience = params.get('early_stopping_patience', 10)
    es_min_delta = params.get('early_stopping_min_delta', 0.0)
    es_counter = 0
    stopped_early_at = None

    best_model_path = os.path.join(params['model_path'], 'best_model.pth')

    amp_device = torch.device(params['device']).type
    scaler = GradScaler(amp_device, enabled=params['mixed_precision'])

    if params['lr_scheduler'] == 'exponential':
        lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif params['lr_scheduler'] == 'reduce_on_plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.1)
    elif params['lr_scheduler'] == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0, last_epoch=-1)
    elif params['lr_scheduler'] == 'multistep':
        lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        lr_scheduler = None
    #for epoch in tqdm(range(1, max_epochs + 1), desc="Retrain Scheme"):
    for epoch in range(1, max_epochs + 1):
        train_loss, train_accuracy = train_epoch(model, criterion, optimizer, train_loader, params, scaler)
        training_losses.append(train_loss)
        training_accuracies.append(train_accuracy)
        
        validation_loss, accuracy = evaluate(model, criterion, val_loader, params)
        validation_losses.append(validation_loss)
        validation_accuracies.append(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            create_info_file(params['model_path'], {'best_accuracy': best_accuracy}, 'best_accuracy.txt')

        if validation_loss < best_validation_loss - es_min_delta:
            best_validation_loss = validation_loss
            es_counter = 0
        else:
            es_counter += 1

        if epoch % 25 == 0:
            LOGGER.info(f"Experiment: {params['experiment_path']} - Epoch [{epoch}/{max_epochs}] - Training loss: {train_loss:.2f} - Validation loss: {validation_loss:.2f} - Validation accuracy: {accuracy:.2f}%")
            #print(f"Epoch [{epoch}/{max_epochs}] - Training loss: {train_loss} - Validation loss: {validation_loss} - Validation accuracy: {accuracy}%")

        if lr_scheduler is not None:
            if params['lr_scheduler'] == 'reduce_on_plateau':
                lr_scheduler.step(validation_loss)
            else:
                lr_scheduler.step()

        if es_enabled and es_counter >= es_patience:
            stopped_early_at = epoch
            LOGGER.info(f"Experiment: {params['experiment_path']} - Early stopping at epoch "
                        f"{epoch}/{max_epochs} (no val_loss improvement for {es_patience} epochs)")
            break

    best_model_loaded = reset_and_load_best_model(params, best_model_path)
    test_loss, test_accuracy, auc_value, acc_med, confusion_matrix = evaluate(best_model_loaded, criterion, test_loader, params, test=True)
    
    LOGGER.info(f"Experiment: {params['experiment_path']} - Test loss: {test_loss:.2f} - Test accuracy: {test_accuracy:.2f}%")
    #print(f"Test loss: {test_loss} - Test accuracy: {test_accuracy}%")
            
    params['t1'] = time.time()
    params['stopped_early'] = stopped_early_at is not None
    if stopped_early_at is not None:
        params['stopped_early_at_epoch'] = stopped_early_at

    create_info_file(params['model_path'], params, 'retraining_params.txt')
    
    model_metrics = metrics.ModelMetrics(best_model_loaded, device=params['device'])

    memory_format = torch.channels_last if params.get('channels_last', False) \
        else torch.contiguous_format
    inference_images = next(iter(val_loader))[0][:10].to(params['device'], memory_format=memory_format)
    input_shape = params['input_shape']
    
    cuda_inference_time = model_metrics.measure_inference_time(inference_images)
    model_memory_usage = model_metrics.measure_memory(input_shape) / (1024 ** 2)  # Convert bytes to MB
    total_trainable_params = model_metrics.measure_parameters()
    total_flops = model_metrics.measure_flops(input_shape)
        
    training_results['total_trainable_params'] = total_trainable_params
    training_results['cuda_inference_time'] = cuda_inference_time
    training_results['total_flops'] = total_flops
    training_results['model_memory_usage'] = model_memory_usage
    training_results['training_losses'] = training_losses
    training_results['training_accuracies'] = training_accuracies
    training_results['validation_losses'] = validation_losses
    training_results['validation_accuracies'] = validation_accuracies
    training_results['best_accuracy'] = best_accuracy
    training_results['test_loss'] = test_loss
    training_results['test_accuracy'] = test_accuracy
    training_results['auc_score'] = auc_value
    training_results['acc_medmnist'] = acc_med
    training_results['confusion_matrix'] = confusion_matrix.tolist()
            
    return training_results


def train_and_eval(params: Dict[str, Any], 
                    fn_dict: Dict[str, Any],net_list:List[str],
                    train_loader:torch.utils.data.DataLoader, 
                    val_loader:torch.utils.data.DataLoader,
                    test_loader:torch.utils.data.DataLoader) -> Dict[str, Union[List[float], float]]:
    """
    This function retrains and evaluates a convolutional neural network model using the specified
    configuration.

    Args:
        params (Dict[str, Any]): A dictionary with parameters necessary for training, including.
        fn_dict (Dict[str, Any]): A dictionary with definitions of the possible layers, including
            their names and parameters.
        net_list (List[str]): A list with names of layers defining the network, in the order they appear.

    Returns:
        Dict[str, Union[List[float], float]]: Dictionary with the training results.
        
        - 'training_losses' (List[float]): List of training losses for each epoch.
        - 'training_accuracies' (List[float]): List of training accuracies for each epoch.
        - 'validation_losses' (List[float]): List of validation losses for each epoch.
        - 'validation_accuracies' (List[float]): List of validation accuracies for each epoch.
        - 'best_accuracy' (float): Best validation accuracy achieved.
        - 'test_loss' (float): Loss on the test set.
        - 'test_accuracy' (float): Accuracy on the test set.
        - 'auc_score' (float): AUC score on the test set.
        - 'acc_medmnist' (float): Accuracy on the test set.
        - 'confusion_matrix' (numpy.ndarray): Confusion matrix on the test set.
        - 'total_trainable_params' (int): Total number of trainable parameters in the model.
    """
    
    _attach_experiment_log_file(params.get('experiment_path_root', params['experiment_path']))

    device = params['device']
    model_path = os.path.join(params['experiment_path'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    params['model_path'] = model_path
    
    LOGGER.info(f"Start retraining of the experiment: {params['experiment_path']}")
    # Load data information
    if params['dataset'].lower() in input.available_datasets:
        dataset_info = input.available_datasets[params['dataset'].lower()]
    else:
        dataset_info = load_yaml(os.path.join(params['data_path'], 'data_info.txt'))
    
    # Create the model
    model_net = model.NetworkGraph(num_classes=dataset_info['num_classes'], 
                                   network_config=params['network_config'], 
                                   network_gap=params['network_gap'])
    filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    model_net.create_functions(fn_dict=filtered_dict, net_list=net_list)

    #params['model_net'] = model_net
    params['net_list'] = net_list
    params['fn_dict'] = fn_dict
    params['num_classes'] = dataset_info["num_classes"]
    params['task'] = dataset_info["task"]
    
    # Add the fully connected layer to the model
    input_shape =  [params['batch_size']] + dataset_info['shape']
    inputs = torch.randn(input_shape)
    _ = model_net(inputs)
    memory_format = torch.channels_last if params.get('channels_last', False) \
        else torch.contiguous_format
    model_net.to(device, memory_format=memory_format)
    
    params['input_shape'] = input_shape
    
    criterion = nn.CrossEntropyLoss()
    
    if params['optimizer'] == 'RMSProp':
        optimizer = torch.optim.RMSprop(model_net.parameters())
    elif params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model_net.parameters())
    elif params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model_net.parameters())
    elif params['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model_net.parameters(), lr=params['learning_rate'])
    else:
        raise ValueError(f"Invalid optimizer: {params['optimizer']}")
        

    # Training time start counting here.
    params['t0'] = time.time()
    
    try:
        results_dict = train(model_net, criterion, optimizer, train_loader, val_loader, test_loader, params)
    except RuntimeError as e:
        if "out of memory" in str(e):
            LOGGER.error(f"Out of memory error: {e}")
            results_dict = None
            release_gpu_memory(gpu_name=params['device'])
        else:
            LOGGER.error(f"Runtime error during training: {e}")
            raise
    except Exception as e:
        LOGGER.error(f"An unexpected error occurred during training: {e}")
        raise
    
    release_gpu_memory(gpu_name=params['device'])
    
    return results_dict
