""" Copyright (c) 2023, Diego Páez
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute and Evaluate the population using multiple processes.
"""

import torch.multiprocessing as mp
from typing import Dict, Any, List
import numpy as np
from cnn import train
from util import init_log
import torch
from cnn import input
import time

class EvalPopulation(object):
    """
    Evaluate a population using multiple processes.

    This class is designed to distribute the evaluation of a population of models
    using multiple processes.
    
    Parameters
    ----------
    params : dict
        A dictionary containing parameters for the evaluation process.
    fn_dict : dict
        A dictionary containing definitions of the functions.
    log_level : str, optional
        The logging level for the internal logger (default is 'INFO').

    Attributes
    ----------
    train_params : dict
        Parameters for the training and evaluation process.
    fn_dict : dict
        Definitions of the functions used in the evaluation.
    timeout : int
        Timeout value for the Dask operations.
    logger : logger
        Internal logger for logging messages.
    gpus : list
        List of GPU devices available for evaluation.
    client : Client
        Dask client for managing the distributed computation.

    Methods
    -------
    __call__(decoded_params, decoded_nets, generation)
        Perform the evaluation of the population.
    
    """
    def __init__(self, params: dict, fn_dict: dict, log_level: str = 'INFO'):
        """
        Initialize the EvalPopulation object.

        Arguments:
        params : dict
            A dictionary containing parameters for the evaluation process.
        fn_dict : dict
            A dictionary containing definitions of the functions.
        log_level : str, optional
            The logging level for the internal logger (default is 'INFO').
        """
        
        self.train_params = params
        self.fn_dict = fn_dict
        self.logger = init_log(log_level, name=__name__)
        self.gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        self.loader = input.GenericDataLoader(params=self.train_params)
        #mp.set_start_method('spawn') # This is necessary for the multiprocessing to work on Windows
        self.logger.info(f"Evaluation process initialized with {len(self.gpus)} GPUs")        
        
    def __call__(self, decoded_params: list, decoded_nets: list, generation: int):
        """
        Evaluate the population.

        Parameters
        ----------
        decoded_params : list
            List of dictionaries containing the parameters for each model.
        decoded_nets : list
            List of lists containing the network architectures for each model.
        generation : int
            The generation number for tracking purposes.

        Returns
        -------
        np.ndarray
            An array containing the evaluations for each model.

        Raises
        ------
        TimeoutError
            If the Dask operations exceed the specified timeout.
        """
        pop_size = len(decoded_nets)
        evaluations = np.empty(shape=(pop_size, ))
        
        #variables = [mp.Value('f', 0.0000) for _ in range(pop_size)]
        variables = [mp.Array('f', 3) for _ in range(pop_size)]
        
        
        # Temporal solution to distribute the individuals in the threads
        selected_thread = 0
        individual_per_thread = []
        for idx in range(len(variables)):
            individual_per_thread.append((idx, selected_thread, decoded_nets[idx], decoded_params[idx], variables[idx]))
            selected_thread += 1
            if selected_thread >= self.train_params['threads']:
                selected_thread = selected_thread % self.train_params['threads']
        
        processes = []
        
        print("\n")
        self.logger.info(f"Starting the Generation {generation} with {pop_size} individuals")
        evol_time_start = time.perf_counter()

        for idx in range(self.train_params['threads']):
            individuals_selected_thread = list(filter(lambda x: x[1]==idx, individual_per_thread))
            gpu_device = self.gpus[idx%len(self.gpus)]
            process = mp.Process(target=self.run_individuals, args=(generation,
                                                self.train_params,
                                                self.fn_dict,
                                                individuals_selected_thread,
                                                gpu_device))
            process.start()
            processes.append(process)

        for p in processes:
            p.join()
                    
        for idx, val in enumerate(variables):
            evaluations[idx] = val[0] # Accuracy - Best Metric
            #evaluations[idx] = val.value
            
        evol_end = time.perf_counter()
        time_elapsed_min = (evol_end-evol_time_start)/60
        time_elapsed_sec = (evol_end-evol_time_start)%60
        self.logger.info(f"Time elapsed for {pop_size} individuals: {time_elapsed_min:.0f}m {time_elapsed_sec:.0f}s")
        return evaluations
            
            
    def run_individuals(self, generation,  train_params, fn_dict, individuals_selected_thread, gpu_device):
        train_loader, val_loader = self.loader.get_loader(pin_memory_device=gpu_device)
        for individual, selected_thread, decoded_net, decoded_params, return_val in individuals_selected_thread:
            self.train_params['device'] = gpu_device
            train.fitness_calculation(f"{generation}_{individual}",
                                        {**train_params},
                                        fn_dict,
                                        decoded_net,
                                        train_loader,
                                        val_loader, 
                                        return_val)
            self.logger.info(f"Calculated fitness of individual {individual} on thread {selected_thread} with "
                            f"Best Metric: {round(return_val[0], 3)}, Params: {round(return_val[1], 2)}M, "
                            f"Inference Time: {round(return_val[2], 3)} uS")