""" Copyright (c) 2023, Diego Páez
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute and Evaluate the population using multiple processes.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

from cnn import train, input
from util import init_log
from weight_bank import WeightBank

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
                 mixedop_mode: bool = False, fn_list: Optional[List[str]] = None,
                 weight_reuse_enabled: bool = False):
        """
        Initialize the EvalPopulation object.

        Arguments:
        params : dict
            A dictionary containing parameters for the evaluation process.
        fn_dict : dict
            A dictionary containing definitions of the functions.
        log_level : str, optional
            The logging level for the internal logger (default is 'INFO').
        mixedop_mode : bool
            If True, evaluate individuals as MixedNetworkGraph (DARTS-style).
        fn_list : list, optional
            Ordered list of operation names for MixedOp mode (alpha column order).
        weight_reuse_enabled : bool
            If True, use a WeightBank to reuse weights from similar architectures.
        """
        self.train_params = params
        self.fn_dict = fn_dict
        self.mixedop_mode = mixedop_mode
        self.fn_list = fn_list or []
        self.weight_reuse_enabled = weight_reuse_enabled
        self.logger = init_log(log_level, name=__name__)
        self.gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        self.loader = input.GenericDataLoader(params=self.train_params)

        # Initialize WeightBank (main process owns it; workers get read-only snapshots)
        self.weight_bank: Optional[WeightBank] = None
        if weight_reuse_enabled:
            bank_dir = params.get('weight_bank_dir', '') or os.path.join(
                params['experiment_path'], 'weight_bank'
            )
            cosine_threshold = params.get('cosine_threshold', 0.05)
            self.weight_bank = WeightBank(bank_dir=bank_dir,
                                          cosine_threshold=cosine_threshold)
            params['weight_bank_dir'] = bank_dir

        self.logger.info(
            f"Evaluation process initialized with {len(self.gpus)} GPUs, "
            f"mixedop_mode={mixedop_mode}, weight_reuse={weight_reuse_enabled}"
        )
        
    def __call__(self, decoded_params: list, decoded_nets: list, generation: int,
                 alpha_matrices: Optional[List[np.ndarray]] = None
                 ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Evaluate the population.

        Parameters
        ----------
        decoded_params : list
            List of dicts with hyperparameters for each model.
        decoded_nets : list
            In discrete mode: list of operation-name lists.
            In mixedop_mode: ignored (alpha_matrices used instead).
        generation : int
            Current generation number.
        alpha_matrices : list of np.ndarray, optional
            In mixedop_mode: [num_ind * repetition] arrays of shape
            [num_nodes, num_ops], initial alpha logits per individual.
            In discrete mode with weight_reuse: flat alpha prob vectors per individual
            (used only for weight bank lookup).

        Returns
        -------
        evaluations : np.ndarray, shape [pop_size]
        trained_alphas : list of np.ndarray or None
            In mixedop_mode: trained alpha matrices read back after training.
            Otherwise None.
        """
        pop_size = len(decoded_nets)
        evaluations = np.empty(shape=(pop_size,))
        variables = [mp.Array('f', 3) for _ in range(pop_size)]

        # Build per-individual weight bank lookup vectors
        if self.weight_bank is not None:
            bank_snapshot = self.weight_bank.snapshot()
        else:
            bank_snapshot = None

        # Distribute individuals across threads
        selected_thread = 0
        individual_per_thread = []
        for idx in range(pop_size):
            alpha = alpha_matrices[idx] if alpha_matrices is not None else None
            individual_per_thread.append(
                (idx, selected_thread, decoded_nets[idx], decoded_params[idx],
                 variables[idx], alpha)
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
                      individuals_selected_thread, gpu_device, bank_snapshot,
                      self.mixedop_mode),
            )
            process.start()
            processes.append(process)

        for p in processes:
            p.join()

        for idx, val in enumerate(variables):
            evaluations[idx] = val[0]

        evol_end = time.perf_counter()
        elapsed_min = (evol_end - evol_time_start) / 60
        elapsed_sec = (evol_end - evol_time_start) % 60
        self.logger.info(
            f"Time elapsed for {pop_size} individuals: {elapsed_min:.0f}m {elapsed_sec:.0f}s"
        )

        # Collect results from worker files (main process only)
        trained_alphas = None
        if self.mixedop_mode:
            trained_alphas = self._collect_trained_alphas(decoded_nets, generation)

        if self.weight_bank is not None:
            self._register_weights_in_bank(decoded_nets, generation, alpha_matrices)

        return evaluations, trained_alphas
            
            
    def run_individuals(self, generation, train_params, fn_dict, fn_list,
                        individuals_selected_thread, gpu_device, bank_snapshot,
                        mixedop_mode: bool):
        train_loader, val_loader = self.loader.get_loader(pin_memory_device=gpu_device)
        for individual, selected_thread, decoded_net, decoded_params, return_val, alpha in \
                individuals_selected_thread:
            p = {**train_params, 'device': gpu_device}
            id_num = f"{generation}_{individual}"
            if mixedop_mode:
                train.fitness_calculation_mixedop(
                    id_num, p, fn_dict, fn_list, alpha,
                    train_loader, val_loader, return_val,
                    weight_bank=bank_snapshot,
                )
            else:
                train.fitness_calculation(
                    id_num, p, fn_dict, decoded_net,
                    train_loader, val_loader, return_val,
                    weight_bank=bank_snapshot,
                    alpha_vec=alpha,
                )
            self.logger.info(
                f"Individual {individual} on thread {selected_thread}: "
                f"Best Metric={round(return_val[0], 3)}, "
                f"Params={round(return_val[1], 2)}M, "
                f"Inference Time={round(return_val[2], 3)}uS"
            )

    def _collect_trained_alphas(self, decoded_nets: list,
                                 generation: int) -> List[Optional[np.ndarray]]:
        """Read trained_alpha.npy files written by workers (mixedop_mode only)."""
        trained_alphas = []
        for idx in range(len(decoded_nets)):
            model_path = os.path.join(
                self.train_params['experiment_path'], f"{generation}_{idx}"
            )
            alpha_path = os.path.join(model_path, 'trained_alpha.npy')
            if os.path.exists(alpha_path):
                trained_alphas.append(np.load(alpha_path))
            else:
                trained_alphas.append(None)
        return trained_alphas

    def _register_weights_in_bank(self, decoded_nets: list, generation: int,
                                   alpha_matrices: Optional[List[np.ndarray]]):
        """Register each individual's trained weights in the WeightBank (main process)."""
        for idx in range(len(decoded_nets)):
            model_path = os.path.join(
                self.train_params['experiment_path'], f"{generation}_{idx}"
            )
            weights_path_file = os.path.join(model_path, 'weights_path.txt')
            alpha_sig_path = os.path.join(model_path, 'alpha_signature.npy')

            if not os.path.exists(weights_path_file):
                continue

            with open(weights_path_file) as _f:
                weights_path = _f.read().strip()

            if not os.path.exists(weights_path):
                continue

            # Determine alpha vector for bank key
            if os.path.exists(alpha_sig_path):
                alpha_vec = np.load(alpha_sig_path)
            elif alpha_matrices is not None and idx < len(alpha_matrices) and \
                    alpha_matrices[idx] is not None:
                alpha_vec = alpha_matrices[idx].flatten()
            else:
                continue

            self.weight_bank.save_weights(alpha_vec, weights_path)