""" Copyright (c) 2023, Diego Páez
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute and Evaluate the population using multiple processes.
"""

import os
import time
from typing import List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp

from cnn import train, input
from util import init_log
from architecture_cache import ArchitectureCache

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
    def __init__(self, params: dict, fn_dict: dict, log_level: str = 'INFO',
                 fn_list: Optional[List[str]] = None):
        """
        Initialize the EvalPopulation object.

        Arguments:
        params : dict
            A dictionary containing parameters for the evaluation process.
        fn_dict : dict
            A dictionary containing definitions of the functions.
        log_level : str, optional
            The logging level for the internal logger (default is 'INFO').
        fn_list : list, optional
            Ordered list of operation names for the current progressive stage.
        """
        self.train_params = params
        self.fn_dict = fn_dict
        self.fn_list = fn_list or []
        # Same train.log that cnn/train.py's fitness_calculation writes to (via
        # _attach_experiment_log_file), so per-generation/per-individual messages
        # logged here (e.g. "Starting Generation N ...", "Individual X: Best Metric=
        # ...") land in experiment_path alongside everything else instead of only
        # appearing in stdout.
        self.logger = init_log(log_level, name=__name__,
                               file_path=os.path.join(params['experiment_path'], 'train.log'))
        self.gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        self.loader = input.GenericDataLoader(params=self.train_params)

        # Single cache.json shared for the whole experiment (all progressive stages
        # included - the net_list key already disambiguates by architecture, so
        # different stages naturally never collide). ArchitectureCache only holds a
        # path (no open file handles), so it's safe to hand the same instance directly
        # to every worker process - each one reads/writes the shared file under a lock
        # (see architecture_cache.py), updating it live as soon as an individual
        # finishes rather than batching updates until the whole generation completes.
        cache_path = os.path.join(params['experiment_path'], 'cache.json')
        self.architecture_cache = ArchitectureCache(cache_path=cache_path)

        self.logger.info(f"Evaluation process initialized with {len(self.gpus)} GPUs")

    def __call__(self, decoded_params: list, decoded_nets: list, generation: int
                 ) -> "tuple[np.ndarray, np.ndarray]":
        """
        Evaluate the population.

        Parameters
        ----------
        decoded_params : list
            List of dicts with hyperparameters for each model.
        decoded_nets : list
            List of operation-name lists.
        generation : int
            Current generation number.

        Returns
        -------
        evaluations : np.ndarray, shape [pop_size]
        cache_hits : np.ndarray of bool, shape [pop_size]
            True for individuals whose fitness was reused from the architecture
            cache (§9) instead of actually being trained this generation.
        """
        pop_size = len(decoded_nets)
        evaluations = np.empty(shape=(pop_size,))
        # 4th slot: 1.0 if this individual was a cache hit, 0.0 if it was actually
        # trained - written by cnn/train.py's fitness_calculation.
        variables = [mp.Array('f', 4) for _ in range(pop_size)]

        # Distribute individuals across threads
        selected_thread = 0
        individual_per_thread = []
        for idx in range(pop_size):
            individual_per_thread.append(
                (idx, selected_thread, decoded_nets[idx], decoded_params[idx], variables[idx])
            )
            selected_thread = (selected_thread + 1) % self.train_params['threads']

        processes = []
        print("\n")
        self.logger.info(f"Starting Generation {generation} with {pop_size} individuals")
        evol_time_start = time.perf_counter()

        for idx in range(self.train_params['threads']):
            individuals_selected_thread = [x for x in individual_per_thread if x[1] == idx]
            gpu_device = self.gpus[idx % len(self.gpus)]
            process = mp.Process(
                target=self.run_individuals,
                args=(generation, self.train_params, self.fn_dict, self.fn_list,
                      individuals_selected_thread, gpu_device, self.architecture_cache),
            )
            process.start()
            processes.append(process)

        for p in processes:
            p.join()

        cache_hits = np.empty(shape=(pop_size,), dtype=bool)
        for idx, val in enumerate(variables):
            evaluations[idx] = val[0]
            cache_hits[idx] = bool(val[3])

        evol_end = time.perf_counter()
        elapsed_min = (evol_end - evol_time_start) / 60
        elapsed_sec = (evol_end - evol_time_start) % 60
        self.logger.info(
            f"Time elapsed for {pop_size} individuals: {elapsed_min:.0f}m {elapsed_sec:.0f}s"
        )

        return evaluations, cache_hits


    def run_individuals(self, generation, train_params, fn_dict, fn_list,
                        individuals_selected_thread, gpu_device, architecture_cache):
        train_loader, val_loader = self.loader.get_loader(pin_memory_device=gpu_device)
        for individual, selected_thread, decoded_net, decoded_params, return_val in \
                individuals_selected_thread:
            p = {**train_params, 'device': gpu_device}
            id_num = f"{generation}_{individual}"
            train.fitness_calculation(
                id_num, p, fn_dict, decoded_net,
                train_loader, val_loader, return_val,
                architecture_cache=architecture_cache,
            )
            cache_hit_note = " (cache hit)" if return_val[3] else ""
            self.logger.info(
                f"Individual {individual} on thread {selected_thread}: "
                f"Best Metric={round(return_val[0], 3)}, "
                f"Params={round(return_val[1], 2)}M, "
                f"Inference Time={round(return_val[2], 3)}uS{cache_hit_note}"
            )