import logging
import yaml
import pickle as pkl
import os
import re
import json
import copy

import time
import matplotlib.pyplot as plt
from shutil import rmtree
import numpy as np
import seaborn as sns
import pandas as pd
import torchvision.datasets
from torchvision.transforms import ToTensor
import medmnist
from medmnist import INFO

import gc
import torch
import torch.cuda as cuda
from cnn import model, input
import GPUtil


def natural_key(string):
    """ Key to use with sort() in order to sort string lists in natural order.
        Example: [1_1, 1_2, 1_5, 1_10, 1_13].
    """

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def load_yaml(file_path):
    """ Wrapper to load a yaml file.

    Args:
        file_path: (str) path to the file to load.

    Returns:
        dict with loaded parameters.
    """

    with open(file_path, 'r') as f:
        file = yaml.safe_load(f)

    return file

def load_pkl(file_path):
    """ Load a pickle file.

    Args:
        file_path: (str) path to the file to load.

    Returns:
        loaded data.
    """

    with open(file_path, 'rb') as f:
        file = pkl.load(f)

    return file

def create_info_file(out_path, info_dict, file_name='data_info.txt'):
    """ Saves info in *info_dict* in a txt file.

    Args:
        out_path: (str) path to the directory where to save info file.
        info_dict: dict with all relevant info the user wants to save in the info file.
    """
    
    with open(os.path.join(out_path, file_name), 'w') as f:
        yaml.dump(info_dict, f)

def save_results_file(out_path, results_dict, file_name='retrain_results.txt'):
    """ Saves results in *results_dict* in a txt file.

    Args:
        out_path: (str) path to the directory where to save results file.
        results_dict: dict with all relevant results the user wants to save in the results file.
    """

    with open(os.path.join(out_path, file_name), 'w') as f:
        json.dump(results_dict, f, indent=4)

def check_file_exists(file_path):
    """ Check if a file exists.
    
    Args:
        file_path: (str) path to the file to check.
        
    Returns:
        True if the file exists, False otherwise.
    """
    if os.path.exists(file_path):
        return True
    else:
        return False

def load_retrain_results(experiment_path, retrain_file_name):
    file_path = os.path.join(experiment_path, retrain_file_name)
    with open(file_path, 'r') as f:
        retrain_data = json.load(f)    
    return retrain_data
    
def plot_confusion_matrix(confusion_matrix, labels):
    confusion_matrix= np.array(confusion_matrix)

    df_cm = pd.DataFrame(confusion_matrix, index = labels, columns = labels)
    plt.figure(figsize = (7,6))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', cbar=False, fmt='g')
    plt.title('Confusion matrix - Retrained model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
        
def test_acc_mean_std(experiment_path, retrain_file_name):
    retrain_data = load_retrain_results(experiment_path, retrain_file_name)
    test_acc_mean = np.mean([retrain_data[key]['test_accuracy'] for key in retrain_data.keys()])
    test_acc_std = np.std([retrain_data[key]['test_accuracy'] for key in retrain_data.keys()])
    
    return test_acc_mean, test_acc_std    

def agg_results(results_dict):
    # Create an empty dictionary to store the mean and std for each variable
    
    agg_results_dict = {
        "training_losses": [],
        "validation_losses": [],
        "training_accuracies": [],
        "validation_accuracies": [],
        # Add other variables as needed
    }
    # Loop through each dictionary and aggregate the results
    for key in results_dict.keys():
        current_dict = results_dict[key]  # Replace 'results_dicts' with the actual list of dictionaries
        agg_results_dict["training_losses"].append(current_dict["training_losses"])
        agg_results_dict["validation_losses"].append(current_dict["validation_losses"])
        agg_results_dict["training_accuracies"].append(current_dict["training_accuracies"])
        agg_results_dict["validation_accuracies"].append(current_dict["validation_accuracies"])
    
    # Convert the lists to NumPy arrays
    agg_results_dict["training_losses"] = np.array(agg_results_dict["training_losses"])
    agg_results_dict["validation_losses"] = np.array(agg_results_dict["validation_losses"])
    agg_results_dict["training_accuracies"] = np.array(agg_results_dict["training_accuracies"])
    agg_results_dict["validation_accuracies"] = np.array(agg_results_dict["validation_accuracies"])

    # Calculate the mean and std across the first axis (axis=0)
    agg_results_dict["mean_training_losses"] = np.mean(agg_results_dict["training_losses"], axis=0)
    agg_results_dict["std_training_losses"] = np.std(agg_results_dict["training_losses"], axis=0)
    agg_results_dict["mean_validation_losses"] = np.mean(agg_results_dict["validation_losses"], axis=0)
    agg_results_dict["std_validation_losses"] = np.std(agg_results_dict["validation_losses"], axis=0)
    agg_results_dict["mean_training_accuracies"] = np.mean(agg_results_dict["training_accuracies"], axis=0)
    agg_results_dict["std_training_accuracies"] = np.std(agg_results_dict["training_accuracies"], axis=0)
    agg_results_dict["mean_validation_accuracies"] = np.mean(agg_results_dict["validation_accuracies"], axis=0)
    agg_results_dict["std_validation_accuracies"] = np.std(agg_results_dict["validation_accuracies"], axis=0)
    
    return agg_results_dict
        
def plot_training_history(results_dict:dict, params:dict=None, retrain:bool=False, title:str=''):
    """ Plot the training history of a model.
    
    Args:
        results_dict: (dict) dictionary with the training history.
    """
    num_keys = len(results_dict.keys())
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    if retrain:
        if num_keys > 1:
            keys = list(results_dict.keys())
            total_epochs = len(results_dict[keys[0]]['training_losses'])
            epochs = range(1, total_epochs + 1)
            test_acc_mean = np.mean([results_dict[key]['test_accuracy'] for key in results_dict.keys()])
            test_acc_std = np.std([results_dict[key]['test_accuracy'] for key in results_dict.keys()])
            agg_results_dict = agg_results(results_dict)
            ax[0].plot(epochs, agg_results_dict["mean_training_losses"], label='Training', color='blue')
            ax[0].fill_between(epochs, 
                                agg_results_dict["mean_training_losses"] - agg_results_dict["std_training_losses"], 
                                agg_results_dict["mean_training_losses"] + agg_results_dict["std_training_losses"], 
                                color='blue', alpha=0.2)
            ax[0].plot(epochs, agg_results_dict["mean_validation_losses"], label='Validation', color='red')
            ax[0].fill_between(epochs, 
                                agg_results_dict["mean_validation_losses"] - agg_results_dict["std_validation_losses"], 
                                agg_results_dict["mean_validation_losses"] + agg_results_dict["std_validation_losses"], 
                                color='red', alpha=0.2)
            ax[0].set_title('Loss')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Loss')
            ax[0].legend(fontsize=12)
            ax[0].grid(True)
            ax[0].set_xlim([1, total_epochs])
            ax[0].set_ylim([0, 1.5])
            
            ax[1].plot(epochs, agg_results_dict["mean_training_accuracies"], label='Training', color='blue')
            ax[1].fill_between(epochs, 
                                agg_results_dict["mean_training_accuracies"] - agg_results_dict["std_training_accuracies"], 
                                agg_results_dict["mean_training_accuracies"] + agg_results_dict["std_training_accuracies"], 
                                color='blue', alpha=0.2)
            ax[1].plot(epochs, agg_results_dict["mean_validation_accuracies"], label='Validation', color='red')
            ax[1].fill_between(epochs, 
                                agg_results_dict["mean_validation_accuracies"] - agg_results_dict["std_validation_accuracies"], 
                                agg_results_dict["mean_validation_accuracies"] + agg_results_dict["std_validation_accuracies"], 
                                color='red', alpha=0.2)
            
            ax[1].axhline(y=test_acc_mean, color='green', linestyle='--', label='Test Accuracy')
            ax[1].text(epochs[-2], test_acc_mean+1, f'{test_acc_mean:.2f} ± {test_acc_std:.2f}', ha='right', va='center', color='black', fontsize=14)
            
            ax[1].set_title('Accuracy')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy')
            ax[1].legend(loc='lower right', fontsize=14)
            ax[1].grid(True)
            ax[1].set_xlim([1, total_epochs])
            # add plt title
            plt.suptitle(f'Training History: {title}', fontsize=16)
            plt.show()
        else:
            results_dict = results_dict[list(results_dict.keys())[0]]
            epochs = range(1, len(results_dict['training_losses']) + 1)
            ax[0].plot(epochs, results_dict["training_losses"], 'b', label='Training loss')
            ax[0].plot(epochs, results_dict["validation_losses"], 'r', label='Validation loss')
            ax[0].set_title('Loss')
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Loss')
            ax[0].legend()
            ax[0].grid(True)
            
            ax[1].plot(epochs, results_dict["training_accuracies"], 'b', label='Training Acc')
            ax[1].plot(epochs, results_dict["validation_accuracies"], 'r', label='Validation Acc')
            max_acc, index = max(results_dict["validation_accuracies"]), results_dict["validation_accuracies"].index(max(results_dict["validation_accuracies"]))
            ax[1].plot(index+1, max_acc, 'go', label='Max Acc')
            ax[1].text(index+1, max_acc+0.1, f'{max_acc:.2f}', fontsize=12)
            ax[1].set_title('Accuracy')
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Accuracy')
            ax[1].legend()
            ax[1].grid(True)
    else:
        epochs = range(1, len(results_dict['training_losses']) + 1)
        eval_starts = params["max_epochs"] - params["epochs_to_eval"]
        epochs_val = range(eval_starts+1, max(epochs)+1)
    
        ax[0].plot(epochs, results_dict["training_losses"], 'b', label='Training loss')
        ax[0].plot(epochs_val, results_dict["validation_losses"], 'r', label='Validation loss')
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[0].grid(True)
        

        ax[1].plot(epochs, results_dict["training_accuracies"], 'b', label='Training Acc')
        ax[1].plot(epochs_val, results_dict["validation_accuracies"], 'r', label='Validation Acc')
        max_acc, index = max(results_dict["validation_accuracies"]), results_dict["validation_accuracies"].index(max(results_dict["validation_accuracies"]))
        ax[1].plot(index+1, max_acc, 'go', label='Max Acc')
        ax[1].text(index+1, max_acc+0.1, f'{max_acc:.2f}', fontsize=12)
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        ax[1].grid(True)
    
    plt.show()

def delete_old_dirs(path, keep_best=False, best_id=''):
    """ Delete directories with old training files (models, checkpoints...). Assumes the
        directories' names start with digits.

    Args:
        path: (str) path to the experiment folder.
        keep_best: (bool) True if user wants to keep files from the best individual.
        best_id: (str) id of the best individual.
    """

    folders = [os.path.join(path, d) for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d)) and d[0].isdigit()]
    folders.sort(key=natural_key)

    if keep_best and best_id:
        folders = [d for d in folders if os.path.basename(d) != best_id]

    for f in folders:
        rmtree(f)

def check_files(exp_path):
    """ Check if exp_path exists and if it does, check if log_file is valid.

    Args:
        exp_path: (str) path to the experiment folder.
    """
    if not os.path.exists(exp_path):
        raise OSError('User must provide a valid \"--experiment_path\" to continue '
                    'evolution or to retrain a model.')
    experiment_folders = [f.name for f in os.scandir(exp_path) if f.is_dir()]
    best_result_folder = [name for name in experiment_folders if name[0].isdigit()]
    best_result_folder = os.path.join(exp_path, best_result_folder[0])
    # identify the index of the folder that begins with a number
    file_path = os.path.join(best_result_folder, 'training_params.txt')

    if os.path.exists(file_path):
        if os.stat(file_path).st_size == 0:
            raise OSError('User must provide an \"--experiment_path\" with a valid data file to '
                        'continue evolution or to retrain a model.')
    else:
        raise OSError('training_params.txt not found!')

    file_path = os.path.join(exp_path, 'log_params_evolution.txt')

    if os.path.exists(file_path):
        if os.stat(file_path).st_size == 0:
            raise OSError('User must provide an \"--experiment_path\" with a valid config_file '
                        'to continue evolution or to retrain a model.')
    else:
        raise OSError('log_params_evolution.txt not found!')
    
def init_log(log_level, name, file_path=None):
    """ Initialize a logging.Logger with level *log_level* and name *name*.

    Args:
        log_level: (str) one of 'NONE', 'INFO' or 'DEBUG'.
        name: (str) name of the module initiating the logger (will be the logger name).
        file_path: (str) path to the log file. If None, stdout is used.

    Returns:
        logging.Logger object.
    """

    logger = logging.getLogger(name)

    if file_path is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(file_path)

    formatter = logging.Formatter('%(levelname)s: %(module)s: %(asctime)s.%(msecs)03d '
                                '- %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_level == 'INFO':
        logger.setLevel(logging.INFO)
    elif log_level == 'DEBUG':
        logger.setLevel(logging.DEBUG)

    return logger

def load_evolved_data(experiment_path: str):
        """
        Loads evolved data from the specified experiment path.

        Parameters:
        - experiment_path (str): The path to the experiment folder containing evolved data.

        Returns:
        None

        This method reads the evolved data from the best-performing experiment folder within the specified path.
        It extracts information such as neural network details, generation, and individual from the 'training_params.txt' file.

        If the data is in an old format (generation and individual not specified in 'training_params.txt'),
        it attempts to extract them from the folder name using a regular expression.

        The extracted information is stored in the 'evolved_params' attribute of the class.

        Note: This method assumes a specific folder and file structure for evolved data.
        """

        experiment_folders = [f.name for f in os.scandir(experiment_path) if f.is_dir()]
        best_result_folder = [name for name in experiment_folders if name[0].isdigit()]
        best_result_folder = os.path.join(experiment_path, best_result_folder[0])
        with open(os.path.join(best_result_folder, 'training_params.txt'), 'r') as file:
                best_individual_info = yaml.safe_load(file)
        net_list = best_individual_info.get('net_list', [])
        generation = best_individual_info.get('generation', 0)
        individual = best_individual_info.get('individual', 0)
        best_acc = best_individual_info.get('best_accuracy', 0.0)
        
        if generation == 0 and individual == 0: # only for old format
                matches = re.search(r'(\d+)_(\d+)$', best_result_folder)
                generation = int(matches.group(1))
                individual = int(matches.group(2))

        return {'net': net_list, 'generation': generation, 'individual': individual, 'best_accuracy': best_acc}
    
def load_retrain_results(experiment_path, retrain_file_name='retrain_results_F13_multistep.txt'):
    """
    Load and identify the best retrained model from a JSON results file, then
    return the directory path, best model path, and its network definition.

    Args:
        experiment_path (str):
            The path to the experiment folder containing retraining results.
        retrain_file_name (str, optional):
            The name of the JSON file that stores retrain results 
            (keys map to experiment runs, values include test metrics).
            Defaults to 'retrain_results_F13_multistep.txt'.

    Returns:
        dict:
            A dictionary with:
                - 'net': (list) the network layer definitions from the best run.
                - 'retrain_path': (str) the folder where the best retraining logs 
                and files are stored.
                - 'best_model_path': (str) the file path to the best model checkpoint 
                (`best_model.pth`) in the best retraining folder.

    Raises:
        FileNotFoundError:
            If the determined best retrain folder does not exist, or if the JSON results file 
            or `retraining_params.txt` file are missing or unreadable.
    """
    file_path = os.path.join(experiment_path, retrain_file_name)
    with open(file_path, 'r') as f:
        retrain_data = json.load(f)
        
    # Determine the key with the highest test accuracy
    best_key = max(retrain_data, key=lambda x: retrain_data[x]['test_accuracy'])
    
    # Convert key naming (e.g., "multistep_F13_retrain_1" -> "retrain_F13_1")
    parts = best_key.split("_")
    best_key = f"{parts[2]}_{parts[1]}_{parts[3]}"
    
    # Construct path to the folder for the best retraining run
    retrain_path = os.path.join(experiment_path, best_key)
    if not os.path.exists(retrain_path):
        raise FileNotFoundError(f"Could not find the retrain folder at {retrain_path}")
    
    # Load retraining params (YAML) within the best retraining folder
    with open(os.path.join(retrain_path, 'retraining_params.txt'), 'r') as file:
        best_retrain_info = yaml.safe_load(file)
    
    net_list = best_retrain_info.get('net_list', [])
    
    # Build the path to the best model file
    best_model_path = os.path.join(retrain_path, 'best_model.pth')
        
    return {'net': net_list, 'retrain_path': retrain_path, 'best_model_path': best_model_path}

    
def load_log_params_evolution(experiment_path: str):
    """
    Loads the log parameters for the evolution process from the specified experiment path.

    Parameters:
    - experiment_path (str): The path to the experiment folder containing evolved data.

    Returns:
    dict: A dictionary containing the log parameters for the evolution process.

    This method reads the log parameters for the evolution process from the
    'log_params_evolution.txt' file. These typically include:
        - train_spec  (dict)
        - QNAS_spec   (dict)
        - fn_dict     (dict)
    among other possible keys like population size, generations, mutation rate, etc.
    """

    log_file = os.path.join(experiment_path, 'log_params_evolution.txt')
    if not os.path.isfile(log_file):
        raise FileNotFoundError(f"Could not find log_params_evolution.txt at {log_file}")

    with open(log_file, 'r') as file:
        log_params = yaml.safe_load(file)
    
    # Extract the subsets you need: train, QNAS, fn_dict
    train_spec = dict(log_params['train'])
    QNAS_spec = dict(log_params['QNAS'])    
    fn_dict = log_params['fn_dict']

    # Return them together in a dictionary (you can rename or restructure as you prefer):
    return {
        'train_spec': train_spec,
        'QNAS_spec': QNAS_spec,
        'fn_dict': fn_dict
    }
    
def calculate_time(start_time, elapse_time,current_gen:int=0, max_generations:int=300, end_evol = True):
    """
    Calculate the elapsed time and the estimated remaining time in the evolution process.

    Parameters:
    start_time (int): The start time of the evolution process.
    elapse_time (int): The current time in the evolution process.
    current_gen (int): The current generation number. Default is 0.
    max_generations (int): The maximum number of generations. Default is 300.
    end_evol (bool): If True, only calculate the elapsed time. If False, also calculate the estimated remaining time. Default is True.

    Returns:
    tuple: If end_evol is True, returns a tuple (hours, minutes) representing the elapsed time.
        If end_evol is False, returns a tuple (hours, minutes, remaining_total_hours, remaining_total_minutes) representing the elapsed time and the estimated remaining time.
    """
    
    total_time = elapse_time - start_time
    hours = int(total_time / 3600)
    minutes = int((total_time - hours * 3600) / 60)
    
    if end_evol:
        return hours, minutes
    else:
        avg_time_per_gen = total_time / current_gen if current_gen != 0 else 0
        remaining_total_time = avg_time_per_gen * (max_generations - current_gen)
        remaining_total_hours = int(remaining_total_time / 3600)
        remaining_total_minutes = int((remaining_total_time - remaining_total_hours * 3600) / 60)
        
        return hours, minutes, remaining_total_hours, remaining_total_minutes
    
def download_dataset(params: dict):
    """
    Downloads the specified dataset if it is not already available locally.

    Parameters:
    - params (dict): A dictionary containing the parameters for the dataset.
        - 'data_path' (str): The path where the dataset should be stored.
        - 'dataset' (str): The name of the dataset to be downloaded.

    If the dataset directory specified by 'data_path' does not exist, it will be created, 
    and the dataset will be downloaded. The function supports downloading datasets from 
    torchvision and MedMNIST. If the dataset already exists, it will print a message and 
    skip the download.

    Raises:
    - ValueError: If the dataset is not found in torchvision.datasets or MedMNIST INFO.
    """
    data_path = params['data_path']
    dataset_name = params['dataset'].lower()

    download_status = not os.path.exists(data_path)
    
    if download_status:
        os.makedirs(data_path)

        if hasattr(torchvision.datasets, dataset_name.upper()):
            dataset_family = "pytorch"
            dataset_class = getattr(torchvision.datasets, dataset_name.upper())
            dataset_class(data_path, download=True, transform=ToTensor())
        elif dataset_name in INFO:
            dataset_family = "medmnist"
            general_info = INFO[dataset_name]
            dataset_class = getattr(medmnist, general_info['python_class'])
            dataset_class(root=data_path, split='train', download=True, transform=ToTensor(), as_rgb=True)
        else:
            raise ValueError(f"Dataset class {dataset_name} not found in torchvision.datasets or available_datasets.")
        return False
    else:
        return True

def get_gpu_memory():
    """
    Retrieve GPU memory usage using GPUtil.
    
    Returns:
    - Used memory in MB.
    """
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[0].memoryUsed  # Assuming single-GPU use; modify if using multiple GPUs
    return None