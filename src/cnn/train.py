""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- Compute the fitness of a model_net using the evolved networks.

Documentation:

    - Automatic mixed precision training (AMP): 
        - https://pytorch.org/docs/stable/amp.html, 
        - https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#all-together-automatic-mixed-precision
    
"""
import os
import time
from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler

from cnn import model, input, metrics, fitness_utils
from cnn.mixed_model import MixedNetworkGraph
from util import create_info_file, init_log, load_yaml


TRAIN_TIMEOUT = 5400

current_directory = os.path.dirname(os.path.dirname(__file__))
log_directory = os.path.join(current_directory, 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file = os.path.join(log_directory, 'train.log')
LOGGER = init_log("INFO", name=__name__, file_path=log_file)



def train_epoch(model, criterion, optimizer, data_loader, params, scaler):
    model.train()
    total = 0
    device = torch.device(params['device'])
    amp_device = device.type  # 'cuda' or 'cpu'
    # Accumulate on-device and only sync to host once, after the loop - a per-batch
    # .item() call forces a CUDA sync and stalls the async training pipeline.
    train_loss_t = torch.zeros((), device=device)
    correct_t = torch.zeros((), device=device)

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=params['mixed_precision']):
            y_logits = model(inputs)
            if params['task'] == 'multi-class':
                labels = labels.squeeze().long()
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

def evaluate(model, criterion, data_loader, params):
    model.eval()
    total = 0
    device = torch.device(params['device'])
    amp_device = device.type  # 'cuda' or 'cpu'
    validation_loss_t = torch.zeros((), device=device)
    correct_t = torch.zeros((), device=device)

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=params['mixed_precision']):
                y_logits = model(inputs)
                if params['task'] == 'multi-class':
                    labels = labels.squeeze().long() # medmnist
                loss = criterion(y_logits, labels)
            validation_loss_t += loss.detach()
            _, predicted = y_logits.max(1)
            total += labels.size(0)
            correct_t += predicted.eq(labels).sum()

    accuracy = 100 * correct_t.item() / total
    validation_loss = validation_loss_t.item() / len(data_loader)

    return validation_loss, accuracy

def train(model:torch.nn.Module, criterion:torch.nn.Module, optimizer:torch.optim.Optimizer, 
        train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader, 
        params:Dict, debug=False) -> Dict:
    """
    Train a neural network model.

    Args:
        model: Model to be trained.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        params: Dictionary with parameters necessary for training
            - max_epochs: Number of epochs to train.
            - epochs_to_eval: Number of epochs before starting validation.
            - t0: Time when the training started.
        device: Device to run the training on (CPU or GPU).

    Returns:
        training_results: Dictionary with the training results.
        
            -training_losses: List of training losses for each epoch.
            -validation_losses: List of validation losses for each epoch.
            -best_accuracy: Best validation accuracy achieved.
    """
    model.train()
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []
    best_accuracy = 0.0
    best_validation_loss = float('inf')
    
    training_results = {}
    max_epochs = params['max_epochs']
    epochs_to_eval = params['epochs_to_eval']
    start_eval = max_epochs - epochs_to_eval
    
    # Automatic mixed precision training (AMP)
    amp_device = torch.device(params['device']).type
    scaler = GradScaler(amp_device, enabled=params['mixed_precision'])

    for epoch in range(1, max_epochs + 1):
        train_loss, train_accuracy = train_epoch(model, criterion, optimizer, train_loader, params,scaler)
        training_losses.append(train_loss)
        training_accuracies.append(train_accuracy)

        if epoch < start_eval and (time.time() - params['t0']) > TRAIN_TIMEOUT:
            print("Timeout reached")
            raise TimeoutError()
        
        if epoch > start_eval:
            validation_loss, accuracy = evaluate(model, criterion, val_loader, params)
            validation_losses.append(validation_loss)
            validation_accuracies.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                create_info_file(params['model_path'], {'best_accuracy': best_accuracy}, 'best_accuracy.txt')
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                create_info_file(params['model_path'], {'best_validation_loss': best_validation_loss}, 'best_validation_loss.txt')
    if debug:
        if epoch >= start_eval:
            print(f"Epoch [{epoch}/{max_epochs}] - Training Loss: {train_loss:.4f} - Validation Loss: {validation_loss:.4f} - Validation Accuracy: {accuracy:.2f}%")
        elif epoch % 5 == 0:
            print(f"Epoch [{epoch}/{max_epochs}] - Training Loss: {train_loss:.4f}")
            
    params['t1'] = time.time()
    params['training_time'] = params['t1'] - params['t0']
    
    model_metrics = metrics.ModelMetrics(model, device=params['device'])
    
    inference_images = next(iter(val_loader))[0][:10].to(params['device'])
    input_shape = params['input_shape']
    
    cuda_inference_time = model_metrics.measure_inference_time(inference_images)
    model_memory_usage = model_metrics.measure_memory(input_shape) / (1024 ** 2)  # Convert bytes to MB
    total_trainable_params = model_metrics.measure_parameters()
    total_flops = model_metrics.measure_flops(input_shape)
        
    fitness_metric = params['fitness_metric']
    mo_base_metric = params['mo_metric_base']

    if fitness_metric == 'best_accuracy' or (fitness_metric == 'scalar_multi_objective' and mo_base_metric == 'accuracy'):
        metric_value = best_accuracy
        metric_type = 'accuracy'
    elif fitness_metric == 'best_loss' or (fitness_metric == 'scalar_multi_objective' and mo_base_metric == 'loss'):
        metric_value = best_validation_loss
        metric_type = 'loss'
    else:
        raise ValueError(f"Invalid fitness_metric: {fitness_metric}")   
        
    # Scalarized multi-objective function
    scalar_multi_objective = fitness_utils.mofitness(metric_value=metric_value,params=total_trainable_params,inference_time=cuda_inference_time,
                                    T_p=params['max_params'], T_t=params['max_inference_time'],metric_type=metric_type)
        
    fitness_val_loss = (1 / (1 + best_validation_loss))*100.0 # Lower loss leads to higher fitness - Reciprocal Transformation
    
    params['total_trainable_params'] = total_trainable_params
    params['cuda_inference_time'] = cuda_inference_time
    params['model_memory_usage'] = model_memory_usage
    params['total_flops'] = total_flops
    params['best_accuracy'] = best_accuracy
    params['best_validation_loss'] = best_validation_loss
    params['fitness_val_loss'] = fitness_val_loss
    params['scalar_multi_objective'] = scalar_multi_objective

    
    LOGGER.info(f"Cuda Inference time: {cuda_inference_time} microseconds")
    LOGGER.info(f"Total trainable parameters: {round(total_trainable_params / 1e6,2)}M")
    
    create_info_file(params['model_path'], params, 'training_params.txt')
    
    training_results['training_losses'] = training_losses
    training_results['training_accuracies'] = training_accuracies
    training_results['validation_losses'] = validation_losses
    training_results['validation_accuracies'] = validation_accuracies
    training_results['cuda_inference_time'] = cuda_inference_time # in microseconds
    training_results['model_memory_usage'] = model_memory_usage # in MB
    training_results['total_trainable_params'] = total_trainable_params / 1e6 # in millions
    training_results['total_flops'] = total_flops / 1e6  # Convert to MFLOPs
    training_results['best_accuracy'] = best_accuracy
    training_results['fitness_val_loss'] = fitness_val_loss
    training_results['scalar_multi_objective'] = scalar_multi_objective        
    return training_results


def fitness_calculation(id_num: str, params: Dict[str, Any],
                        fn_dict: Dict[str, Any], net_list: List[str],
                        train_loader: torch.utils.data.DataLoader,
                        val_loader: torch.utils.data.DataLoader,
                        return_val, debug: bool = False,
                        weight_bank=None,
                        alpha_vec: Optional[np.ndarray] = None) -> None:
    """Train and evaluate a model using evolved hyperparameters.

    This function trains and evaluates a convolutional neural network model using the specified
    configuration and evolved hyperparameters.

    Args:
        id_num (str): A string identifying the generation number and the individual number.
        params (Dict[str, Any]): A dictionary with parameters necessary for training, including
            the evolved hyperparameters.
        fn_dict (Dict[str, Any]): A dictionary with definitions of the possible layers, including
            their names and parameters.
        net_list (List[str]): A list with names of layers defining the network, in the order they appear.

    Returns:
        Dict[str, Union[List[float], float]]: A dictionary containing the training results.

        - 'training_losses' (List[float]): List of training losses for each epoch.
        - 'validation_losses' (List[float]): List of validation losses for each epoch.
        - 'best_accuracy' (float): Best validation accuracy achieved.

    Raises:
        TimeoutError: If the training process takes too long to complete.
    """

    # Fixed input/batch shape per run - let cudnn autotune the fastest conv algorithms.
    torch.backends.cudnn.benchmark = True

    device = params['device']
    params['net_list'] = net_list
    model_path = os.path.join(params['experiment_path'], id_num)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    params['model_path'] = model_path
    params['generation'] = id_num.split('_')[0]
    params['individual'] = id_num.split('_')[1]

    LOGGER.info(f"Training model {id_num} on device {device} ...")
    # Load data info
    if params['dataset'].lower() in input.available_datasets:
        dataset_info = input.available_datasets[params['dataset'].lower()]
    else:
        dataset_info = load_yaml(os.path.join(params['data_path'], 'data_info.txt'))

    params['num_classes'] = dataset_info['num_classes']
    params['task'] = dataset_info['task']

    # check if cbam is a key in the fn_dict
    has_cbam_key = any(key.startswith('cbam') for key in fn_dict)

    # Create the model
    model_net = model.NetworkGraph(num_classes=dataset_info['num_classes'],
                                   network_config=params['network_config'],
                                   network_gap=params['network_gap'])
    filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    model_net.create_functions(fn_dict=filtered_dict, net_list=net_list, cbam=has_cbam_key)

    # Add the fully connected layer to the model
    input_shape =  [params['batch_size']] + dataset_info['shape']
    inputs = torch.randn(input_shape)
    with torch.no_grad():
        _ = model_net(inputs)
    model_net.to(device)

    params['input_shape'] = input_shape

    # Weight reuse: load weights from a close architecture if available
    if weight_bank is not None and alpha_vec is not None:
        closest_path = weight_bank.find_close_match(alpha_vec)
        if closest_path is not None:
            loaded = weight_bank.load_weights_into_model(model_net, closest_path, device)
            if loaded:
                finetune_epochs = params.get('weight_reuse_finetune_epochs', params['max_epochs'])
                params['max_epochs'] = finetune_epochs
                params['weight_reuse_applied'] = True
                LOGGER.info(f"Weight reuse applied for {id_num} from {closest_path}")

    criterion = nn.CrossEntropyLoss()

    if params['optimizer'] == 'RMSProp':
        optimizer = torch.optim.RMSprop(model_net.parameters())
    elif params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model_net.parameters())
    elif params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model_net.parameters())
    else:
        optimizer = torch.optim.SGD(model_net.parameters(), lr=params['learning_rate'])

    # Training time start counting here.
    params['t0'] = time.time()

    # Train the model in fitness scheme
    try:
        results_dict = train(model_net, criterion, optimizer, train_loader, val_loader, params, debug)
        if debug:
            result = results_dict
            return result
        else:
            if params['fitness_metric'] == 'best_accuracy':
                return_val[0] = results_dict['best_accuracy']
                return_val[1] = results_dict['total_trainable_params']
                return_val[2] = results_dict['cuda_inference_time']
            elif params['fitness_metric'] == 'best_loss':
                return_val[0] = results_dict['fitness_val_loss']
                return_val[1] = results_dict['total_trainable_params']
                return_val[2] = results_dict['cuda_inference_time']
            elif params['fitness_metric'] == 'scalar_multi_objective':
                return_val[0] = results_dict['scalar_multi_objective']
                return_val[1] = results_dict['total_trainable_params']
                return_val[2] = results_dict['cuda_inference_time']
            else:
                raise ValueError(f"Invalid fitness metric: {params['fitness_metric']}")

        LOGGER.info(f"Training of model {id_num} finished, best {params['fitness_metric']}: {round(return_val[0], 2)}")

        # Save weights and alpha signature for weight bank registration (main process reads these)
        best_model_path = os.path.join(model_path, 'best_model.pth')
        torch.save(model_net.state_dict(), best_model_path)
        with open(os.path.join(model_path, 'weights_path.txt'), 'w') as _f:
            _f.write(best_model_path)
        if alpha_vec is not None:
            np.save(os.path.join(model_path, 'alpha_signature.npy'), alpha_vec)

    except (TimeoutError, MemoryError) as e:
        LOGGER.error(f"Exception: {e}")
        return_val[:] = [0.0, 0.0, 0.0]
    except Exception as e:
        if "out of memory" in str(e):
            LOGGER.error(f"CUDA out of memory exception, error: {e}")
            return_val[:] = [0.0, 0.0, 0.0]
        else:
            LOGGER.error(f"Exception: {e}")
            return_val[:] = [0.0, 0.0, 0.0]
        raise e


def fitness_calculation_mixedop(
    id_num: str,
    params: Dict[str, Any],
    fn_dict: Dict[str, Any],
    fn_list: List[str],
    alpha_matrix: np.ndarray,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    return_val,
    debug: bool = False,
    weight_bank=None,
) -> None:
    """Train and evaluate a MixedNetworkGraph (DARTS-style) individual.

    Each node runs all candidate operations in parallel, combined via softmax(alpha)
    weights. Alpha is an nn.Parameter updated by gradient descent alongside the network
    weights. After training, the trained alpha matrix is saved to disk so the main
    process can update the quantum alpha_logits.

    Args:
        id_num: "{generation}_{individual}" identifier string.
        params: training parameters dict.
        fn_dict: full operation config dict.
        fn_list: ordered list of operation names (defines alpha column order).
        alpha_matrix: np.ndarray of shape [num_nodes, num_ops], initial alpha logits
                      derived from quantum alpha_logits for this individual.
        train_loader / val_loader: data loaders.
        return_val: mp.Array('f', 3) for returning (fitness, params_M, infer_us).
        weight_bank: optional WeightBank snapshot for weight reuse lookup.
    """
    # Fixed input/batch shape per run - let cudnn autotune the fastest conv algorithms.
    torch.backends.cudnn.benchmark = True

    device = params['device']
    model_path = os.path.join(params['experiment_path'], id_num)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    params['model_path'] = model_path
    params['generation'] = id_num.split('_')[0]
    params['individual'] = id_num.split('_')[1]

    LOGGER.info(f"Training MixedOp model {id_num} on device {device} ...")

    if params['dataset'].lower() in input.available_datasets:
        dataset_info = input.available_datasets[params['dataset'].lower()]
    else:
        dataset_info = load_yaml(os.path.join(params['data_path'], 'data_info.txt'))

    params['num_classes'] = dataset_info['num_classes']
    params['task'] = dataset_info['task']

    # Build MixedNetworkGraph
    model_net = MixedNetworkGraph(
        num_classes=dataset_info['num_classes'],
        in_channels=dataset_info['shape'][0],
        network_gap=params.get('network_gap', False),
    )
    model_net.create_mixed_ops(fn_dict=fn_dict, fn_list=fn_list, alpha_matrix=alpha_matrix,
                                max_channels=params.get('mixedop_max_channels'))

    # Warm-up forward pass to initialise fc layer
    input_shape = [params['batch_size']] + dataset_info['shape']
    with torch.no_grad():
        _ = model_net(torch.randn(input_shape))
    model_net.to(device)
    params['input_shape'] = input_shape

    # Weight reuse: load closest trained architecture weights
    alpha_flat = alpha_matrix.flatten()
    if weight_bank is not None:
        closest_path = weight_bank.find_close_match(alpha_flat)
        if closest_path is not None:
            loaded = weight_bank.load_weights_into_model(model_net, closest_path, device)
            if loaded:
                finetune_epochs = params.get('weight_reuse_finetune_epochs', params['max_epochs'])
                params['max_epochs'] = finetune_epochs
                params['weight_reuse_applied'] = True
                LOGGER.info(f"Weight reuse applied for {id_num} from {closest_path}")

    criterion = nn.CrossEntropyLoss()

    # Optimizer covers ALL parameters including alpha (registered as nn.Parameter)
    if params['optimizer'] == 'RMSProp':
        optimizer = torch.optim.RMSprop(model_net.parameters())
    elif params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model_net.parameters())
    elif params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model_net.parameters())
    else:
        optimizer = torch.optim.SGD(model_net.parameters(), lr=params['learning_rate'])

    params['t0'] = time.time()

    try:
        results_dict = train(model_net, criterion, optimizer, train_loader, val_loader, params, debug)
        if debug:
            return results_dict

        if params['fitness_metric'] == 'best_accuracy':
            return_val[0] = results_dict['best_accuracy']
        elif params['fitness_metric'] == 'best_loss':
            return_val[0] = results_dict['fitness_val_loss']
        elif params['fitness_metric'] == 'scalar_multi_objective':
            return_val[0] = results_dict['scalar_multi_objective']
        else:
            raise ValueError(f"Invalid fitness metric: {params['fitness_metric']}")
        return_val[1] = results_dict['total_trainable_params']
        return_val[2] = results_dict['cuda_inference_time']

        LOGGER.info(f"MixedOp model {id_num} finished, best {params['fitness_metric']}: {round(return_val[0], 2)}")

        # Save trained alpha for quantum update feedback (main process reads this)
        trained_alpha = model_net.get_trained_alpha()
        np.save(os.path.join(model_path, 'trained_alpha.npy'), trained_alpha)

        # Save weights and alpha signature for weight bank registration
        best_model_path = os.path.join(model_path, 'best_model.pth')
        torch.save(model_net.state_dict(), best_model_path)
        with open(os.path.join(model_path, 'weights_path.txt'), 'w') as _f:
            _f.write(best_model_path)
        # Use trained softmax weights (not logits) as the bank signature
        alpha_t = torch.tensor(trained_alpha, dtype=torch.float32)
        trained_probs = torch.softmax(alpha_t, dim=-1).numpy()
        np.save(os.path.join(model_path, 'alpha_signature.npy'), trained_probs.flatten())

    except (TimeoutError, MemoryError) as e:
        LOGGER.error(f"Exception: {e}")
        return_val[:] = [0.0, 0.0, 0.0]
    except Exception as e:
        if "out of memory" in str(e):
            LOGGER.error(f"CUDA out of memory exception for {id_num}: {e}")
            return_val[:] = [0.0, 0.0, 0.0]
        else:
            LOGGER.error(f"Exception for {id_num}: {e}")
            return_val[:] = [0.0, 0.0, 0.0]
        raise e