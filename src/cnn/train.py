""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- Compute the fitness of a model_net using the evolved networks.

Documentation:

    - Automatic mixed precision training (AMP): 
        - https://pytorch.org/docs/stable/amp.html, 
        - https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#all-together-automatic-mixed-precision
    
"""
import logging
import os
import time
from typing import Dict, List, Union, Any

import torch
import torch.nn as nn
from torch.amp import GradScaler

from cnn import model, input, metrics, fitness_utils
from util import create_info_file, init_log, load_yaml


TRAIN_TIMEOUT = 5400

# Console output always on (also how individual training progress reaches the
# terminal live - see the per-epoch print() in train()). The file handler is
# attached lazily per experiment_path (see _attach_experiment_log_file), since
# the experiment path is only known at call time, not at import time.
LOGGER = init_log("INFO", name=__name__)
_log_file_experiment_path = None


def _attach_experiment_log_file(experiment_path: str):
    """Write training logs into <experiment_path>/train.log instead of a fixed
    repo-wide location, so each experiment's logs live alongside its other
    artifacts. A no-op if already attached for this experiment_path (workers
    train many individuals in the same process across a run).
    """
    global _log_file_experiment_path
    if _log_file_experiment_path == experiment_path:
        return

    log_path = os.path.join(experiment_path, 'train.log')
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(module)s: %(asctime)s.%(msecs)03d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    LOGGER.addHandler(handler)
    _log_file_experiment_path = experiment_path



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
    # .item() call forces a CUDA sync and stalls the async training pipeline.
    train_loss_t = torch.zeros((), device=device)
    correct_t = torch.zeros((), device=device)

    for inputs, labels in data_loader:
        inputs = inputs.to(device, memory_format=memory_format)
        labels = labels.to(device)
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
    memory_format = torch.channels_last if params.get('channels_last', False) \
        else torch.contiguous_format
    validation_loss_t = torch.zeros((), device=device)
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

    # Per-individual early stopping: stop THIS model's training once its monitored
    # validation metric stops improving, instead of always running the full
    # max_epochs. Distinct from QNAS-level early_stopping (src/qnas.py), which stops
    # the whole generational search - this stops one individual's training loop.
    #
    # Monitors whichever metric actually drives fitness (see fitness_metric handling
    # below, ~line 243) instead of always watching val_loss: otherwise a model whose
    # accuracy is still climbing can get killed by a stalled/noisy loss that isn't
    # even what's being optimized for.
    individual_es_enabled = params.get('early_stopping_enabled', False)
    individual_es_patience = params.get('early_stopping_patience', 5)
    individual_es_min_delta = params.get('early_stopping_min_delta', 0.0)
    individual_es_counter = 0
    stopped_early_at = None
    fitness_metric = params['fitness_metric']
    mo_base_metric = params.get('mo_metric_base')
    es_monitor_accuracy = (
        fitness_metric == 'best_accuracy'
        or (fitness_metric == 'scalar_multi_objective' and mo_base_metric == 'accuracy')
    )

    # Automatic mixed precision training (AMP)
    amp_device = torch.device(params['device']).type
    scaler = GradScaler(amp_device, enabled=params['mixed_precision'])

    id_num = f"{params.get('generation', '?')}_{params.get('individual', '?')}"

    for epoch in range(1, max_epochs + 1):
        train_loss, train_accuracy = train_epoch(model, criterion, optimizer, train_loader, params,scaler)
        training_losses.append(train_loss)
        training_accuracies.append(train_accuracy)

        if epoch < start_eval and (time.time() - params['t0']) > TRAIN_TIMEOUT:
            print("Timeout reached")
            raise TimeoutError()

        progress = (f"[{id_num}] epoch {epoch}/{max_epochs} - "
                   f"train_loss={train_loss:.4f} train_acc={train_accuracy:.2f}%")

        # With early stopping enabled, validate every epoch (needed to actually detect
        # a plateau) instead of only the last epochs_to_eval epochs.
        if epoch > start_eval or individual_es_enabled:
            validation_loss, accuracy = evaluate(model, criterion, val_loader, params)
            validation_losses.append(validation_loss)
            validation_accuracies.append(accuracy)

            # min_delta gates only the patience-counter reset below, not this
            # best-value bookkeeping, so best_accuracy/best_validation_loss always
            # reflect the true best seen regardless of the early-stopping threshold.
            prev_best_accuracy = best_accuracy
            prev_best_validation_loss = best_validation_loss
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                create_info_file(params['model_path'], {'best_accuracy': best_accuracy}, 'best_accuracy.txt')
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                create_info_file(params['model_path'], {'best_validation_loss': best_validation_loss}, 'best_validation_loss.txt')

            if es_monitor_accuracy:
                monitored_improved = accuracy > prev_best_accuracy + individual_es_min_delta
            else:
                monitored_improved = validation_loss < prev_best_validation_loss - individual_es_min_delta
            if monitored_improved:
                individual_es_counter = 0
            else:
                individual_es_counter += 1

            progress += f" - val_loss={validation_loss:.4f} val_acc={accuracy:.2f}%"

        # Plain print (not LOGGER, which is file-only) so this shows up live in the
        # terminal - individual training can take minutes/epoch with nothing else
        # printed in between otherwise.
        print(progress, flush=True)

        if individual_es_enabled and individual_es_counter >= individual_es_patience:
            stopped_early_at = epoch
            LOGGER.info(f"Early stopping {id_num} at epoch {epoch}/{max_epochs} "
                        f"(no val_loss improvement for {individual_es_patience} epochs)")
            print(f"[{id_num}] early stopped at epoch {epoch}/{max_epochs}", flush=True)
            break
    if debug:
        if epoch >= start_eval:
            print(f"Epoch [{epoch}/{max_epochs}] - Training Loss: {train_loss:.4f} - Validation Loss: {validation_loss:.4f} - Validation Accuracy: {accuracy:.2f}%")
        elif epoch % 5 == 0:
            print(f"Epoch [{epoch}/{max_epochs}] - Training Loss: {train_loss:.4f}")
            
    params['t1'] = time.time()
    params['training_time'] = params['t1'] - params['t0']
    
    model_metrics = metrics.ModelMetrics(model, device=params['device'])
    
    memory_format = torch.channels_last if params.get('channels_last', False) \
        else torch.contiguous_format
    inference_images = next(iter(val_loader))[0][:10].to(params['device'], memory_format=memory_format)
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
    params['stopped_early'] = stopped_early_at is not None
    if stopped_early_at is not None:
        params['stopped_early_at_epoch'] = stopped_early_at


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
                        architecture_cache=None) -> None:
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

    _attach_experiment_log_file(params['experiment_path'])

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

    # Fitness cache: an architecture (net_list) that was already evaluated - by this
    # or an earlier generation, e.g. via elitism - reuses its cached fitness instead of
    # retraining. No weights are stored or reused, only the fitness/params/inference
    # time values. Checked BEFORE building any model, so a hit skips model construction
    # and the warm-up forward pass too, not just training.
    if architecture_cache is not None:
        cache_hit = architecture_cache.find_cached_result(net_list)
        if cache_hit is not None:
            return_val[0] = cache_hit['fitness']
            return_val[1] = cache_hit['params_m']
            return_val[2] = cache_hit['inference_us']
            return_val[3] = 1.0  # cache hit - NOT a new architecture this generation
            params['weight_reuse_applied'] = True
            params['cache_hit'] = True
            params['training_time'] = 0.0
            params['total_trainable_params'] = cache_hit['params_m'] * 1e6
            params['cuda_inference_time'] = cache_hit['inference_us']
            params['cache_hit_count'] = cache_hit.get('hit_count', 0)
            LOGGER.info(f"Cache hit for {id_num}: reusing fitness {cache_hit['fitness']:.3f} "
                        f"for architecture {net_list} (hit #{cache_hit.get('hit_count', 0)})")
            create_info_file(model_path, params, 'training_params.txt')
            return

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
            return_val[3] = 0.0  # actually trained - a new architecture this generation

        LOGGER.info(f"Training of model {id_num} finished, best {params['fitness_metric']}: {round(return_val[0], 2)}")

        # Save weights (used by the retrain step / infographic, not by the fitness cache)
        best_model_path = os.path.join(model_path, 'best_model.pth')
        torch.save(model_net.state_dict(), best_model_path)

        # Update the fitness cache immediately, right after this individual finishes -
        # not batched at the end of the generation - so any other individual (in this
        # or a later generation) with the exact same architecture can hit the cache
        # right away. Safe under concurrent workers: ArchitectureCache.register() does
        # a locked read-modify-write of cache.json (see architecture_cache.py).
        if architecture_cache is not None:
            architecture_cache.register(
                net_list, fitness=return_val[0],
                params_m=results_dict['total_trainable_params'],  # already in millions
                inference_us=results_dict['cuda_inference_time'],
            )

    except (TimeoutError, MemoryError) as e:
        LOGGER.error(f"Exception: {e}")
        return_val[:] = [0.0, 0.0, 0.0, 0.0]
    except Exception as e:
        if "out of memory" in str(e):
            LOGGER.error(f"CUDA out of memory exception, error: {e}")
            return_val[:] = [0.0, 0.0, 0.0, 0.0]
        else:
            LOGGER.error(f"Exception: {e}")
            return_val[:] = [0.0, 0.0, 0.0, 0.0]
        raise e