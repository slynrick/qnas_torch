"""
Copyright (c) 2024, Diego R. Páez Ardila
Licensed under The MIT License [see LICENSE for details]

This script shows how to fine-tune a previously trained model (best_model.pth)
with a new or extended dataset.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Union, Any, List
import time

from cnn import model, input
from cnn.train_detailed import evaluate
from util import init_log, load_yaml


# Global cache dictionary
dataset_info_cache = {}

# Function to get dataset info with caching
def get_dataset_info(dataset_name, data_path):
    if dataset_name in dataset_info_cache:
        #print(f"Using cached dataset info for {dataset_name}")
        return dataset_info_cache[dataset_name]

    dataset_info_path = os.path.join(data_path, 'data_info.txt')
    dataset_info = load_yaml(dataset_info_path)

    if dataset_info is not None:
        dataset_info_cache[dataset_name] = dataset_info
    return dataset_info

def load_trained_model(decoded_net: List[str], train_params: Dict[str, Any]) -> nn.Module:
    """
    Build the model based on the decoded network architecture and training parameters.
    Load pretrained weights, remove the existing FC layer, and add a new FC layer
    for the new dataset.

    Parameters:
    - decoded_net (List[str]): The network architecture definition.
    - train_params (Dict[str, Any]): Dictionary containing training parameters.

    Returns:
    - model_instance (nn.Module): The constructed model ready for fine-tuning.
    """
    # Load dataset information
    dataset_name = train_params['dataset'].lower()

    if dataset_name in input.available_datasets:
        dataset_info = input.available_datasets[dataset_name]
    else:
        dataset_info = get_dataset_info(dataset_name, train_params['data_path'])
    
    if dataset_info is None:
        raise ValueError(
            f"Failed to load dataset information for {dataset_name}. "
            "Check if the dataset is available or if 'data_info.txt' exists and is correctly formatted."
        )

    # Update train_params with dataset info
    train_params['task'] = dataset_info['task']
    train_params['input_shape'] = [train_params['batch_size']] + dataset_info['shape']

    # Check if 'cbam' is a key in the fn_dict
    has_cbam_key = any(key.startswith('cbam') for key in train_params['fn_dict'])

    # Filter fn_dict to include only keys present in decoded_net
    filtered_fn_dict = {
        key: item for key, item in train_params['fn_dict'].items() if key in decoded_net
    }

    # Create the model instance
    model_instance = model.NetworkGraph(num_classes=dataset_info['num_classes'])
    model_instance.create_functions(
        fn_dict=filtered_fn_dict, net_list=decoded_net, cbam=has_cbam_key
    )

    # Initialize the model with a random input to finalize shapes
    input_random = torch.randn(train_params['input_shape'])
    with torch.no_grad():
        _ = model_instance(input_random)

    # Load pretrained weights
    state_dict = torch.load(train_params['best_model_path'], map_location=train_params['device'])
    model_instance.load_state_dict(state_dict)
    

    # **Remove the existing FC layer and add a new one**
    # Assuming the final layer is named 'fc' and it's an instance of 'FullyConnected'
    if hasattr(model_instance, 'fc'):
        # Access the inner Linear layer within the FullyConnected module
        if hasattr(model_instance.fc, 'fc'):
            in_features = model_instance.fc.fc.in_features
            # Replace the existing FC layer with a new one for the new number of classes
            model_instance.fc.fc = nn.Linear(in_features, train_params['num_classes_new'])
            print("Replaced the final FC layer successfully.")
        else:
            raise AttributeError(
                "The 'fc' module does not contain an attribute named 'fc'. "
                "Please verify the structure of the 'FullyConnected' module."
            )
    else:
        raise AttributeError(
            "The model does not have an attribute named 'fc'. "
            "Please adjust the layer replacement accordingly."
        )
        
    model_instance.to(train_params['device'])

    return model_instance

def freeze_layers(model: nn.Module, freeze_pattern: str = None) -> None:
    """
    Freeze certain layers of the model for fine-tuning. For example, you can:
    - Freeze all but the last layer
    - Freeze only the first few convolution layers
    - Or do not freeze at all (freeze_pattern=None)

    Args:
        model (nn.Module): The model to partially freeze.
        freeze_pattern (str): Some pattern to decide what to freeze.
            E.g., "all_but_last" or "first_conv" or None for no freezing.
    """
    if freeze_pattern == "all_but_last":
        # Freeze all layers except the final classification layer
        for name, param in model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False

    elif freeze_pattern == "first_conv":
        # Freeze only the first convolutional layer
        for name, param in model.named_parameters():
            if "conv1" in name:
                param.requires_grad = False


def fine_tune_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    params: Dict[str, Any],
    max_epochs: int = 10
) -> str:
    """
    Fine-tune the model with the new classification layer.

    Args:
        model (nn.Module): The loaded and modified model.
        train_loader (DataLoader): DataLoader for the fine-tuning train dataset.
        val_loader (DataLoader): DataLoader for the fine-tuning validation dataset.
        params (Dict[str, Any]): Dictionary with training hyperparameters.
        max_epochs (int, optional): Number of epochs to run for fine-tuning. Defaults to 10.

    Returns:
        str: Path to the best fine-tuned model checkpoint.
    """
    criterion = nn.CrossEntropyLoss()

    # Only parameters that require gradients are updated
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    
    # change the optimizer to AdamW
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()))

    best_val_accuracy = 0.0
    best_model_path = os.path.join(params['fine_tune_path'], 'best_finetuned_model.pth')

    for epoch in range(1, max_epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(params['device']), labels.to(params['device'])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_accuracy = 100 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(params['device']), labels.to(params['device'])
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_accuracy = 100 * correct / total

        print(f"[Epoch {epoch}/{max_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Val Acc: {val_accuracy:.2f}%")
            # (Optional) Evaluate on a test set
            test_loss, test_accuracy, auc_value, acc_med, confusion_matrix = evaluate(model, criterion, test_loader, params, True)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

    print("Fine-tuning completed.")
    return best_model_path, test_accuracy, auc_value, acc_med, confusion_matrix

