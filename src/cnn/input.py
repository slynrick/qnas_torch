""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- Input module - Generic data loader for PyTorch, supporting various datasets and data augmentation.

"""
import torch
import random
import util
import os
import numpy as np
from time import time
import torchvision.datasets
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, TrivialAugmentWide
from sklearn.model_selection import StratifiedShuffleSplit
import medmnist
from medmnist import INFO

cifar10_info = {
  'dataset': 'CIFAR10',
  'mean': [0.491400808095932, 0.48215898871421814, 0.44653093814849854],
  'std': [0.24703224003314972, 0.24348513782024384, 0.26158785820007324],
  'shape': [3, 32, 32], 
  'num_classes': 10,
  'task': 'classification',
  'balanced_train': True
}

cifar100_info = {
  'dataset': 'CIFAR100',
  'mean': [0.5070757865905762, 0.48655030131340027, 0.4409191310405731],
  'std': [0.2673342823982239, 0.2564384639263153, 0.2761504650115967],
  'shape': [3, 32, 32],
  'num_classes': 100,
  'task': 'classification',
  'balanced_train': True
}

atletasaxial_info = {
  'dataset': 'ATLETA_AXIAL',
  'mean': [0.485, 0.456, 0.406],
  'std': [0.229, 0.224, 0.225],
  'shape': [3, 128, 128],
  'num_classes': 3,
  'task': 'classification',
  'balanced_train': False
}

atletascoronal_info = {
  'dataset': 'ATLETA_CORONAL',
  'mean': [0.485, 0.456, 0.406],
  'std': [0.229, 0.224, 0.225],
  'shape': [3, 128, 128],
  'num_classes': 3,
  'task': 'classification',
  'balanced_train': False
}

available_datasets = {
  'cifar10': cifar10_info,
  'cifar100': cifar100_info,
  'atleta_axial': atletasaxial_info,
  'atleta_coronal': atletascoronal_info
}

class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
          x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)
        
class GenericDataLoader:
  """A generic data loader for PyTorch, supporting various datasets and data augmentation."""
  def __init__(self, params: dict, train_split=0.9, seed=None, info: dict = {}):
    """
    Initialize the GenericDataLoader.
    
    Parameters:
      params (dict): Dictionary containing parameters.
      train_split (float): Split ratio for training data.
      seed (int): Seed for randomization.
      info (dict): Additional information about the dataset.
      
    Returns:
      None
    """
    self.params = params
    self.train_split = train_split
    error_msg = "[!] train_split should be in the range [0, 1]."
    assert 0 <= self.train_split <= 1, error_msg    
    if seed is None:
        seed = int(time())
        random.seed(seed)
        torch.manual_seed(seed)
    self.info_dict = {'dataset': f'{self.params["dataset"]}'}
    self.info_dict['seed'] = seed
    self.download_status = not os.path.exists(self.params['data_path'])
    
    if self.download_status:
      os.makedirs(self.params['data_path'])

    if not info:
        # Check if the dataset is available in the available_datasets dict
        if self.params['dataset'].lower() in available_datasets.keys():
          dataset_family = "local"
          dataset_info = available_datasets[self.params['dataset'].lower()]
          mean = dataset_info['mean']
          std = dataset_info['std']
          channels, height, width = dataset_info['shape']
          self.num_classes = dataset_info['num_classes']
          self.task = dataset_info['task']
          
        # check if the dataset is in the torchvision datasets and compute the parameters
        elif hasattr(torchvision.datasets, self.params['dataset'].upper()):
          dataset_family = "pytorch"
          if util.check_file_exists(os.path.join(self.params['data_path'], 'data_info.txt')):
            info_dataset = util.load_yaml(os.path.join(self.params['data_path'], 'data_info.txt'))
            mean = info_dataset['mean']
            std = info_dataset['std']
            channels, height, width = info_dataset['shape']
            self.num_classes = info_dataset['num_classes']
            self.task = info_dataset['task']
          else:
            dataset_class = getattr(torchvision.datasets, self.params['dataset'].upper())
            dataset_ = dataset_class(self.params['data_path'], download=self.download_status, transform=ToTensor())
            loader = DataLoader(dataset_, batch_size=len(dataset_), num_workers=0, shuffle=False)
            data = next(iter(loader))
            mean = data[0].mean(dim=(0, 2, 3)).tolist()
            std = data[0].std(dim=(0, 2, 3)).tolist()
            channels, height, width = dataset_[0][0].shape
            self.num_classes = len(dataset_.classes)
            self.task = 'classification'
          
        elif self.params['dataset'].lower() in INFO and hasattr(medmnist, INFO[self.params['dataset'].lower()]['python_class']):
          dataset_family = "medmnist"
          general_info = INFO[self.params['dataset'].lower()]
          if util.check_file_exists(os.path.join(self.params['data_path'], 'data_info.txt')):
            info_dataset = util.load_yaml(os.path.join(self.params['data_path'], 'data_info.txt'))
            mean = info_dataset['mean']
            std = info_dataset['std']
            channels, height, width = info_dataset['shape']
            self.num_classes = info_dataset['num_classes']
            self.task = info_dataset['task']
          else:
            dataset_class = getattr(medmnist, general_info['python_class'])
            dataset_ = dataset_class(root=self.params['data_path'], split='train', download=self.download_status, transform=ToTensor(), as_rgb=True)
            loader = DataLoader(dataset_, batch_size=len(dataset_), num_workers=0, shuffle=False)
            data = next(iter(loader))
            mean = data[0].mean(dim=(0, 2, 3)).tolist()
            std = data[0].std(dim=(0, 2, 3)).tolist()
            channels, height, width = dataset_[0][0].shape
            self.num_classes = len(general_info['label'])
            self.task = general_info['task']
        else:
          raise ValueError(f"Dataset class {self.params['dataset']} not found in torchvision.datasets or available_datasets.")
        
    else:
      raise NotImplementedError('Custom dataset is not implemented yet.')             
    
    self.info_dict['shape'] = [channels, height, width]  
    self.info_dict['mean'] = mean
    self.info_dict['std'] = std
    self.info_dict['num_classes'] = self.num_classes
    self.info_dict['task'] = self.task
    
    if not util.check_file_exists(os.path.join(self.params['data_path'], 'data_info.txt')):
      util.create_info_file(out_path=self.params['data_path'], info_dict=self.info_dict)
    
    
    # Define common transformations
    basic_transform = [ToTensor(), Normalize(mean=mean, std=std)]
    resize_transform = [Resize((height, width))] + basic_transform

    # Set the self.transform
    self.transform = Compose(resize_transform if 'atleta' in self.params['dataset'].lower() else basic_transform)

    # Check for data augmentation
    if self.params['data_augmentation']:
        augmentation_transform = [TrivialAugmentWide(num_magnitude_bins=31)] + basic_transform

        if dataset_family == "medmnist":
            self.train_transform = Compose(augmentation_transform)
        else:
            # Optionally resize for 'atleta' datasets or other datasets
            if 'atleta' in self.params['dataset'].lower():
                self.train_transform = Compose([Resize((height, width))] + augmentation_transform)
            else:
                self.train_transform = Compose(augmentation_transform)
    else:
        # Use self.transform if no data augmentation
        self.train_transform = self.transform
    
      
  def get_loader(self, for_train=True, pin_memory_device="cuda"):
    """
    Get data loader for training or validation/testing.

    Parameters:
      for_train (bool): If True, returns the training and val loader; otherwise, returns the testing loader.

    Returns:
      DataLoader: PyTorch DataLoader.
    """
    # create the dataset
    if hasattr(torchvision.datasets, self.params['dataset'].upper()):
      dataset_class = getattr(torchvision.datasets, self.params['dataset'].upper())
      full_dataset = dataset_class(self.params['data_path'], train=True, download=self.download_status)
      test_dataset = dataset_class(self.params['data_path'], train=False, download=self.download_status,transform=self.transform)
      self.download_status = not os.path.exists(self.params['data_path'])
      
    elif self.params['dataset'].lower() in INFO and hasattr(medmnist, INFO[self.params['dataset'].lower()]['python_class']):
      dataset_class = getattr(medmnist, INFO[self.params['dataset'].lower()]['python_class'])
      train_dataset = dataset_class(root=self.params['data_path'], split='train', download=self.download_status, transform=self.train_transform, as_rgb=True)
      valid_dataset = dataset_class(root=self.params['data_path'], split='val', download=self.download_status, transform=self.transform, as_rgb=True)
      test_dataset = dataset_class(root=self.params['data_path'], split='test', download=self.download_status, transform=self.transform, as_rgb=True)
      self.download_status = not os.path.exists(self.params['data_path'])
    elif self.params['dataset'].lower() == 'atleta_axial' or self.params['dataset'].lower() == 'atleta_coronal':
      try:
        train_dataset =torchvision.datasets.ImageFolder(root=f"{self.params['data_path']}/train", transform=self.train_transform)
        valid_dataset = torchvision.datasets.ImageFolder(root=f"{self.params['data_path']}/val", transform=self.transform)
        test_dataset = torchvision.datasets.ImageFolder(root=f"{self.params['data_path']}/test", transform=self.transform)
      except:
        raise ValueError(f"Dataset is not available in the path {self.params['data_path']}")
    else:
      raise NotImplementedError('Custom dataset is not implemented yet.')
    
    drop_last = True if self.params['dataset'].lower() == 'organamnist' else False
        
    if not for_train:
      test_loader = DataLoader(
        test_dataset,
        batch_size=self.params['eval_batch_size'],
        num_workers=self.params['num_workers'],
        shuffle=False,
        pin_memory=True,
        pin_memory_device=pin_memory_device)
      return test_loader
    
    if hasattr(torchvision.datasets, self.params['dataset'].upper()):
      val_split = 1 - self.train_split
      # Get the labels and create StratifiedShuffleSplit
      labels = full_dataset.targets
      stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=val_split)

      # Get the training and validation indices
      train_idx, val_idx = next(stratified_split.split(labels, labels))
      
      num_train = len(full_dataset)
      
      # split the dataset into train and validation
      if self.params['limit_data'] and self.params['limit_data_value'] < num_train:
        train_samples = (int(self.train_split * self.params['limit_data_value'])) 
        val_samples = (int(self.params['limit_data_value'] - train_samples))
        
        train_count_per_class = train_samples // self.num_classes
        val_count_per_class = val_samples // self.num_classes
        
        train_indices = []
        val_indices = []
        for label in set(labels):
          label_indices = [i for i in train_idx if labels[i] == label]
          train_indices.extend(label_indices[:train_count_per_class])

          label_indices = [i for i in val_idx if labels[i] == label]
          val_indices.extend(label_indices[:val_count_per_class])
        
        train_subset = Subset(full_dataset, train_indices)
        valid_subset = Subset(full_dataset, val_indices)

      else:
        train_subset = Subset(full_dataset, train_idx)
        valid_subset = Subset(full_dataset, val_idx)
      
      # Apply transformations to the datasets
      train_dataset = MyDataset(train_subset, transform=self.train_transform)
      valid_dataset = MyDataset(valid_subset, transform=self.transform)
      
    elif self.params['dataset'].lower() in INFO and hasattr(medmnist, INFO[self.params['dataset'].lower()]['python_class']):
      
      num_train = len(train_dataset)
      num_val = len(valid_dataset)
      train_labels = train_dataset.labels.squeeze().tolist()
      val_labels = valid_dataset.labels.squeeze().tolist()
      
      train_samples = (int(self.train_split * self.params['limit_data_value'])) 
      val_samples = (int(self.params['limit_data_value'] - train_samples))
      
      # split the dataset into train and validation
      if self.params['limit_data'] and train_samples < num_train and val_samples < num_val:
        
        train_count_per_class = train_samples // self.num_classes
        val_count_per_class = val_samples // self.num_classes
        train_indices = []
        val_indices = []

        # Iterate through each class and pick 1000 samples
        for label in set(train_labels):
            label_indices = [i for i, l in enumerate(train_labels) if l == label]
            train_indices.extend(label_indices[:train_count_per_class])

        for label in set(val_labels):
            label_indices = [i for i, l in enumerate(val_labels) if l == label]
            val_indices.extend(label_indices[:val_count_per_class])
      
        train_dataset = Subset(train_dataset, train_indices)
        valid_dataset = Subset(valid_dataset, val_indices)
    elif 'atleta' in self.params['dataset'].lower():
      pass
    else:
      raise NotImplementedError('Custom dataset is not implemented yet.')

    train_loader = DataLoader(
      train_dataset,
      batch_size=self.params['batch_size'],
      num_workers=self.params['num_workers'],
      shuffle=True,
      drop_last=drop_last,
      pin_memory=True, 
      pin_memory_device=pin_memory_device)

    val_loader = DataLoader(
      valid_dataset,
      batch_size=self.params['eval_batch_size'],
      num_workers=self.params['num_workers'],
      shuffle=False,
      pin_memory=True, 
      pin_memory_device=pin_memory_device)
  
    self.info_dict['train_records'] = len(train_dataset)
    self.info_dict['valid_records'] = len(valid_dataset)
    self.info_dict['test_records'] = len(test_dataset)
            
    util.create_info_file(out_path=self.params['data_path'], info_dict=self.info_dict)
    
    return train_loader, val_loader