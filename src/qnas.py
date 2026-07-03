""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Q-NAS algorithm class.
"""

import datetime
import os
from pickle import dump, HIGHEST_PROTOCOL

import numpy as np
import time

from population import QPopulationNetwork, QPopulationParams
from util import delete_old_dirs, init_log, load_pkl, calculate_time


class QNAS(object):
    """ Quantum Inspired Neural Architecture Search """

    def __init__(self, eval_func, experiment_path, log_file, log_level, data_file):
        """ Initialize QNAS.

        Args:
            eval_func: function that will be used to evaluate individuals.
            experiment_path: (str) path to the folder where logs and models will be saved.
            log_file: (str) path to the file to keep logs.
            log_level: (str) one of "INFO", "DEBUG" or "NONE".
        """

        self.dtype = np.float64                 # Type of all arrays excluding fitnesses
        self.tolerance = 1.e-15                 # Tolerance to compare floating point

        self.best_so_far = 0.0                  # Best fitness so far
        self.best_so_far_id = [0, 0]            # id = [generation, position in the population]
        self.current_best_id = [0, 0]
        self.current_gen = 0                    # Current generation number
        self.data_file = data_file
        self.eval_func = eval_func
        self.experiment_path = experiment_path
        self.fitnesses = None                   # TF calculates accuracy with float32 precision
        self.generations = None
        self.update_quantum_gen = None
        self.logger = init_log(log_level, name=__name__, file_path=log_file)
        self.penalties = None
        self.penalize_number = None
        self.random = 0.0
        self.raw_fitnesses = None
        self.reducing_fns_list = []
        self.replace_method = None
        self.save_data_freq = np.inf
        self.total_eval = 0
        self.early_stopping_counter = 0

        self.qpop_params = None
        self.qpop_net = None
        self.best_alpha = None  # trained alpha matrix of the best individual so far (mixedop_mode)
        self.best_alpha_fn_list = None  # fn_list active when best_alpha was captured -
                                         # progressive stages can prune fn_list after the
                                         # best individual's generation, so best_alpha's op
                                         # indices must be decoded against ITS fn_list, not
                                         # whatever fn_list is active at collapse time.

        # P-DARTS-style progressive depth growth + op pruning (mixedop_mode only, optional)
        self.progressive_stages = None
        self.current_stage_idx = 0

    def initialize_qnas(self, num_quantum_ind, params_ranges, repetition, max_generations,
                        crossover_rate, update_quantum_gen, replace_method, fn_list,
                        initial_probs, update_quantum_rate, max_num_nodes, reducing_fns_list,
                        patience, early_stopping, save_data_freq=0, penalize_number=0,
                        crossover_frequency=5, en_pop_crossover=False,
                        pop_crossover_rate=0.25, pop_crossover_method='hux',
                        mixedop_mode=False, alpha_noise_std=0.1, progressive_stages=None):

        """ Initialize algorithm with several parameter values.

        Args:
            num_quantum_ind: (int) number of quantum individuals.
            params_ranges: {'parameter_name': [parameter_lower_limit, parameter_upper_limit]}.
            repetition: (int) ratio between the number of classic individuals in the classic
                population and the quantum individuals in the quantum population.
            max_generations: (int) number of generations to run the evolution.
            crossover_rate: (float) crossover rate for numerical part of the chromosomes.
            update_quantum_gen: (int) the width of the quantum genes will be updated in a
                interval of *update_quantum_gen* generations.
            replace_method: (str) one of 'best' or 'elitism', indicating which method to
                substitute the population.
            fn_list: list of possible functions.
            initial_probs: list defining the initial probabilities for each function; if empty,
                the algorithm will give the same probability for each function.
            update_quantum_rate: (float) probability that a quantum gene will be updated,
                if using update_center() and/or update_width_decay().
            max_num_nodes: (int) initial number of nodes in the network to be evolved (the 
                classifier fc layer is always included).
            save_data_freq: generation frequency in which train loss and accuracy of the best
                model (of current generation) will be extracted from events.out.tfevents file
                and saved in a csv file.
            penalize_number: (int) defines the minimum number of reducing layers an individual
                can have without being penalized. The penalty is proportional to the number of
                exceeding reducing layers. If 0, no penalization will be applied.
            reducing_fns_list: (list) list of reducing functions (stride > 2) names.
            patience: (int) number of generations without improvement in the best fitness to
                stop the evolution.
            early_stopping: (bool) if True, the evolution will stop if the best fitness does not
                improve by at least 0.005 (0.5%) for *patience* generations.
            en_pop_crossover: (bool) if True, there will be crossover between the best individuals
                of the current and new populations.
            crossover_frequency: (int) frequency of crossover in the population of networks.
            pop_crossover_rate: (float) crossover rate for the population of networks, used to select
                the number of offspring to generate in the crossover [0 - 1].
            pop_crossover_method: (str) one of 'hux' or 'uniform', indicating the method to
                apply crossover in the population of networks.
        """

        self.generations = max_generations
        self.update_quantum_gen = update_quantum_gen
        self.replace_method = replace_method
        self.penalize_number = penalize_number
        self.patience = patience
        self.early_stopping = early_stopping
        self.en_pop_crossover = en_pop_crossover
        self.pop_crossover_rate = pop_crossover_rate
        self.crossover_frequency = crossover_frequency
        self.mixedop_mode = mixedop_mode
        self.alpha_noise_std = alpha_noise_std
        self._last_trained_alphas = None  # set after each eval in mixedop mode
        self.weight_reuse_enabled = False  # set by run_evolution.py via eval_func

        # P-DARTS-style progressive growth: stage 0 always starts from the full fn_list
        # (op pruning only makes sense once there's a trained alpha to rank ops by, at
        # the end of stage 0) - later stages narrow fn_list and grow num_nodes.
        self.progressive_stages = progressive_stages
        self.current_stage_idx = 0
        if progressive_stages:
            if progressive_stages[0]['num_ops'] != len(fn_list):
                raise ValueError(
                    "progressive_stages[0].num_ops must equal len(fn_list) - the first "
                    "stage has no trained alpha yet to rank/prune ops by."
                )
            max_num_nodes = progressive_stages[0]['num_nodes']

        if reducing_fns_list:
            self.penalties = np.zeros(shape=(num_quantum_ind * repetition))
            self.reducing_fns_list = [i for i in range(len(fn_list))
                                    if fn_list[i] in reducing_fns_list]

        if save_data_freq:
            self.save_data_freq = save_data_freq

        self.qpop_params = QPopulationParams(num_quantum_ind=num_quantum_ind,
                                            params_ranges=params_ranges,
                                            repetition=repetition,
                                            crossover_rate=crossover_rate,
                                            update_quantum_rate=update_quantum_rate)

        self.qpop_net = QPopulationNetwork(num_quantum_ind=num_quantum_ind,
                                            max_num_nodes=max_num_nodes,
                                            repetition=repetition,
                                            update_quantum_rate=update_quantum_rate,
                                            fn_list=fn_list,
                                            initial_probs=initial_probs,
                                            crossover_method=pop_crossover_method)

        if self.mixedop_mode:
            self.qpop_net.initialize_alpha_logits()
                                            

    def replace_pop(self, new_pop_params, new_pop_net, new_fitnesses, raw_fitnesses):
        """ Replace the individuals of old population using one of two methods: elitism or
            replace the worst. In *elitism*, only the best individual of the old population is
            maintained, while all the others are replaced by the new population. In *best*,
            only the best of the union of both populations individuals are kept.

        Args:
            new_pop_params: float ndarray representing a classical population of parameters.
            new_pop_net: int ndarray representing a classical population of networks.
            new_fitnesses: float numpy array representing the fitness of each individual in
                *new_pop*.
            raw_fitnesses: float numpy array representing the fitness of each individual in
                *new_pop* before the penalization method. Note that, if no penalization method
                is applied, *raw_fitnesses* = *new_fitnesses*.
        """

        if self.current_gen == 0:
            # In the 1st generation, the current population is the one that was just generated.
            self.qpop_params.current_pop = new_pop_params
            self.qpop_net.current_pop = new_pop_net

            self.fitnesses = new_fitnesses
            self.raw_fitnesses = raw_fitnesses
            self.update_best_id(new_fitnesses)
        else:
            # Checking if the best so far individual has changed in the current generation
            self.update_best_id(new_fitnesses)

            if self.replace_method == 'elitism':
                select_new = range(new_fitnesses.shape[0] - 1)
                new_fitnesses, raw_fitnesses, new_pop_params, \
                    new_pop_net = self.order_pop(new_fitnesses,
                                                new_pop_params,
                                                new_pop_net,
                                                select_new)
                selected = range(1)
            elif self.replace_method == 'best':
                selected = range(self.fitnesses.shape[0])
                
            # Concatenate populations
            self.fitnesses = np.concatenate((self.fitnesses[selected], new_fitnesses))
            self.raw_fitnesses = np.concatenate((self.raw_fitnesses[selected], raw_fitnesses))
            self.qpop_params.current_pop = np.concatenate(
                    (self.qpop_params.current_pop[selected], new_pop_params))
            self.qpop_net.current_pop = np.concatenate(
                    (self.qpop_net.current_pop[selected], new_pop_net))
        
        ## TODO: Here we have the the last and new population Multi objective operation
        
            
        # Order the population based on fitness
        num_classic = self.qpop_params.num_ind * self.qpop_params.repetition
        self.fitnesses, self.raw_fitnesses, self.qpop_params.current_pop, \
            self.qpop_net.current_pop = self.order_pop(self.fitnesses,
                                                        self.raw_fitnesses,
                                                        self.qpop_params.current_pop,
                                                        self.qpop_net.current_pop,
                                                        selection=range(num_classic))       
        
        self.best_so_far = self.fitnesses[0]

    @staticmethod
    def order_pop(fitnesses, raw_fitnesses, pop_params, pop_net, selection=None):
        """ Order the population based on *fitnesses*.

        Args:
            fitnesses: ndarray with fitnesses values.
            raw_fitnesses: float ndarray representing the fitness of each individual before the
                penalization method.
            pop_params: ndarray with population of parameters.
            pop_net: ndarray with population of networks.
            selection: range to select elements from the population.

        Returns:
            ordered population and fitnesses.
        """

        if selection is None:
            selection = range(fitnesses.shape[0])
        idx = np.argsort(fitnesses)[::-1]
        pop_params = pop_params[idx][selection]
        pop_net = pop_net[idx][selection]
        fitnesses = fitnesses[idx][selection]
        raw_fitnesses = raw_fitnesses[idx][selection]

        return fitnesses, raw_fitnesses, pop_params, pop_net

    def update_best_id(self, new_fitnesses):
        """ Checks if the new population contains the best individual so far and updates
            *self.best_so_far_id*.

        Args:
            new_fitnesses: float numpy array representing the fitness of each individual in
                *new_pop*.
        """

        idx = np.argsort(new_fitnesses)[::-1]
        self.current_best_id = [self.current_gen, int(idx[0])]
        if new_fitnesses[idx[0]] > self.best_so_far:
            self.best_so_far_id = self.current_best_id

            if self.mixedop_mode and self._last_trained_alphas is not None:
                best_idx = int(idx[0])
                if best_idx < len(self._last_trained_alphas) and \
                        self._last_trained_alphas[best_idx] is not None:
                    self.best_alpha = np.copy(self._last_trained_alphas[best_idx])
                    self.best_alpha_fn_list = list(self.qpop_net.chromosome.fn_list)

    def generate_classical(self):
        """ Generate a specific number of classical individuals from the observation of quantum
            individuals. This number is equal to (*num_ind* x *repetition*). The new classic
            individuals will be evaluated and ordered according to their fitness values.
        """

        # Generate distance for crossover and quantum updates every generation
        self.random = np.random.rand()

        # Generate classical pop for hyperparameters
        new_pop_params = self.qpop_params.generate_classical()
        if self.current_gen > 0:
            new_pop_params = self.qpop_params.classic_crossover(new_pop=new_pop_params,
                                                                distance=self.random)

        if self.mixedop_mode:
            # MixedOp mode: individuals are alpha logit matrices (PDF representation)
            alpha_matrices = self.qpop_net.generate_classical_mixedop(sigma=self.alpha_noise_std)
            self.logger.info("new MixedOp population generated, shape=%s", alpha_matrices.shape)

            # Evaluate: alpha_matrices[i] is passed to fitness_calculation_mixedop
            new_fitnesses, raw_fitnesses, trained_alphas = self.eval_pop(
                new_pop_params, alpha_matrices, mixedop_mode=True
            )
            self._last_trained_alphas = trained_alphas

            # In mixedop mode, current_pop stores individual indices (0..total_ind-1)
            total_ind = new_pop_params.shape[0]
            new_pop_net = np.arange(total_ind, dtype=np.int32).reshape(-1, 1)
        else:
            # Discrete mode: classical individuals are operation index arrays
            new_pop_net = self.qpop_net.generate_classical()

            if self.current_gen > 0 and self.en_pop_crossover:
                if self.current_gen % self.crossover_frequency == 0:
                    num_offspring = int(len(new_pop_net) * self.pop_crossover_rate)
                    best_current_pop = self.qpop_net.current_pop[:num_offspring]
                    new_pop_net[:num_offspring] = self.qpop_net.apply_crossover(
                        best_current_pop, new_pop_net[:num_offspring]
                    )

            self.logger.info("new population created =%s", new_pop_net)
            new_fitnesses, raw_fitnesses, _ = self.eval_pop(
                new_pop_params, new_pop_net, mixedop_mode=False
            )

        self.replace_pop(new_pop_params, new_pop_net, new_fitnesses, raw_fitnesses)

    def decode_pop(self, pop_params, pop_net):
        """ Decode a population of parameters and networks.

        Args:
            pop_params: float numpy array with a classic population of hyperparameters.
            pop_net: int numpy array with a classic population of networks.

        Returns:
            list of decoded params and list of decoded networks.
        """

        num_individuals = pop_net.shape[0]

        decoded_params = [None] * num_individuals
        decoded_nets = [None] * num_individuals

        for i in range(num_individuals):
            decoded_params[i] = self.qpop_params.chromosome.decode(pop_params[i])
            decoded_nets[i] = self.qpop_net.chromosome.decode(pop_net[i, :])

        return decoded_params, decoded_nets

    def eval_pop(self, pop_params, pop_net_or_alpha, mixedop_mode=False):
        """ Decode and evaluate a population of networks and hyperparameters.

        Args:
            pop_params: float numpy array with a classic population of hyperparameters.
            pop_net_or_alpha: in discrete mode, int ndarray of operation indices;
                              in mixedop_mode, float ndarray [total_ind, num_nodes, num_ops].
            mixedop_mode: bool, whether to evaluate as MixedNetworkGraph.

        Returns:
            (penalized_fitnesses, raw_fitnesses, trained_alphas)
            trained_alphas is a list of np.ndarray or None (mixedop_mode only).
        """
        if mixedop_mode:
            # pop_net_or_alpha is [total_ind, num_nodes, num_ops] alpha matrices
            decoded_params = [
                self.qpop_params.chromosome.decode(pop_params[i])
                for i in range(pop_params.shape[0])
            ]
            alpha_matrices = [pop_net_or_alpha[i] for i in range(pop_net_or_alpha.shape[0])]

            self.logger.info('Evaluating MixedOp population ...')
            fitnesses, trained_alphas = self.eval_func(
                decoded_params, alpha_matrices,
                generation=self.current_gen,
                alpha_matrices=alpha_matrices,
            )
            penalized_fitnesses = np.copy(fitnesses)
        else:
            decoded_params, decoded_nets = self.decode_pop(pop_params, pop_net_or_alpha)

            # For discrete mode with weight reuse: provide quantum prob vectors as alpha
            alpha_matrices = None
            if getattr(self, 'weight_reuse_enabled', False):
                total_ind = pop_params.shape[0]
                alpha_matrices = []
                for i in range(total_ind):
                    q_idx = i % self.qpop_net.num_ind
                    alpha_matrices.append(self.qpop_net.probabilities[q_idx].flatten())

            self.logger.info('Evaluating new population ...')
            fitnesses, _ = self.eval_func(
                decoded_params, decoded_nets,
                generation=self.current_gen,
                alpha_matrices=alpha_matrices,
            )
            trained_alphas = None
            penalized_fitnesses = np.copy(fitnesses)

            if self.penalize_number:
                penalties = self.get_penalties(pop_net_or_alpha)
                penalized_fitnesses -= penalties

        # Update the total evaluation counter
        self.total_eval = self.total_eval + np.size(pop_params, axis=0)

        return penalized_fitnesses, fitnesses, trained_alphas

    def get_penalties(self, pop_net, penalty_factor=0.01):
        """ Penalize individuals with more than *self.penalize_number* reducing layers. The
            penalty is proportional (default factor of 1%) to the number of exceeding layers.

        Args:
            pop_net: ndarray representing the encoded population of networks (just evaluated).
            penalty_factor: (float) the factor to multiply the penalties for all networks.

        Returns:
            penalties for each network in pop_net.
        """

        penalties = np.zeros(shape=pop_net.shape[0])

        for i, net in enumerate(pop_net):
            unique, counts = np.unique(net, return_counts=True)
            reducing_fns_count = np.sum([counts[i] for i in range(len(unique))
                                        if unique[i] in self.reducing_fns_list])
            # Penalize individual only if number of reducing layers exceed the maximum allowed
            if reducing_fns_count > self.penalize_number:
                penalties[i] = reducing_fns_count - self.penalize_number

        penalties = penalty_factor * penalties

        return penalties

    def log_data(self):
        """ Log QNAS evolution info into a log file. """

        np.set_printoptions(precision=4)

        stage_line = ''
        if self.progressive_stages:
            stage = self.progressive_stages[self.current_stage_idx]
            stage_line = (
                f'- Stage: {self.current_stage_idx} '
                f'(gen_start={stage["gen_start"]}, num_nodes={stage["num_nodes"]}, '
                f'num_ops={stage["num_ops"]})\n'
            )

        self.logger.info(f'New generation finished running!\n'
                        f'- Generation: {self.current_gen}\n'
                        f'{stage_line}'
                        f'- Best so far: {self.best_so_far_id} --> {self.best_so_far:.5f}\n'
                        f'- Fitnesses: {self.fitnesses}\n'
                        f'- Fitnesses without penalties: {self.raw_fitnesses}\n')

    def save_data(self):
        """ Save QNAS data in a pickle file for logging and reloading purposes, including
            chromosomes, generation number, evaluation score and number of evaluations. Note
            that the data in the file is loaded and updated with the current generation, so that
            we keep track of the entire evolutionary process.
        """

        if self.current_gen == 0:
            data = dict()
        else:
            data = load_pkl(self.data_file)

        entry = {'time': str(datetime.datetime.now()),
                 'total_eval': self.total_eval,
                 'best_so_far': self.best_so_far,
                 'best_so_far_id': self.best_so_far_id,
                 'fitnesses': self.fitnesses,
                 'raw_fitnesses': self.raw_fitnesses,
                 'lower': self.qpop_params.lower,
                 'upper': self.qpop_params.upper,
                 'params_pop': self.qpop_params.current_pop,
                 'net_probs': self.qpop_net.probabilities,
                 'num_net_nodes': self.qpop_net.chromosome.num_genes,
                 'net_pop': self.qpop_net.current_pop}

        if self.mixedop_mode and hasattr(self.qpop_net, 'alpha_logits'):
            entry['alpha_logits'] = self.qpop_net.alpha_logits

        if self.progressive_stages:
            entry['current_stage_idx'] = self.current_stage_idx
            entry['fn_list'] = list(self.qpop_net.chromosome.fn_list)

        if self.mixedop_mode and self.best_alpha is not None:
            # Needed to resume correctly and to collapse the right network later -
            # best_alpha's shape/fn_list may predate later progressive-stage prunings,
            # so it must be restored alongside the fn_list that was active when it was
            # captured (see collapse_best_network()).
            entry['best_alpha'] = self.best_alpha
            entry['best_alpha_fn_list'] = self.best_alpha_fn_list

        data[self.current_gen] = entry

        self.dump_pkl_data(data)

    def dump_pkl_data(self, new_data):
        """ Saves *new_data* into *self.data_file* pickle file.

        Args:
            new_data: dict containing data to save.
        """

        with open(self.data_file, 'wb') as f:
            dump(new_data, f, protocol=HIGHEST_PROTOCOL)

    def load_qnas_data(self, file_path):
        """ Read pkl data in *file_path* and load its information to current QNAS. It also saves
            its info into the new pkl data file *self.data_file*.

        Args:
            file_path: (str) path to the pkl data file.
        """

        log_data = load_pkl(file_path)

        if not os.path.exists(self.data_file):
            self.dump_pkl_data(log_data)

        generation = max(log_data.keys())
        log_data = log_data[generation]

        self.current_gen = generation
        self.total_eval = log_data['total_eval']
        self.best_so_far = log_data['best_so_far']
        self.best_so_far_id = log_data['best_so_far_id']
        self.qpop_net.chromosome.set_num_genes(log_data['num_net_nodes'])

        self.fitnesses = log_data['fitnesses']
        self.raw_fitnesses = log_data['raw_fitnesses']
        self.qpop_params.lower = log_data['lower']
        self.qpop_params.upper = log_data['upper']
        self.qpop_net.probabilities = log_data['net_probs']

        self.qpop_params.current_pop = log_data['params_pop']
        self.qpop_net.current_pop = log_data['net_pop']

        if self.mixedop_mode and 'alpha_logits' in log_data:
            self.qpop_net.alpha_logits = log_data['alpha_logits']

        if self.progressive_stages and 'current_stage_idx' in log_data:
            self.current_stage_idx = log_data['current_stage_idx']
            self.qpop_net.chromosome.fn_list = log_data['fn_list']
            self.qpop_net.chromosome.num_functions = len(log_data['fn_list'])
            self.eval_func.fn_list = log_data['fn_list']
            self.eval_func.set_stage_weight_bank(self.current_stage_idx)

        if self.mixedop_mode and 'best_alpha' in log_data:
            self.best_alpha = log_data['best_alpha']
            self.best_alpha_fn_list = log_data['best_alpha_fn_list']

    def check_early_stopping(self):
        """
        Compute the early stopping of the evolution. If the best fitness does not improve 
        by at least 0.005 (0.5%) for `patience` generations, the evolution stops.
        """
        if self.current_gen > 1:
            improvement = (self.best_so_far - self.last_best_so_far) / self.last_best_so_far
            if improvement > 0.005:
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            self.logger.info(f"Early stopping counter: {self.early_stopping_counter}")
            if self.early_stopping_counter >= self.patience:
                self.logger.info(f"Early stopping at generation {self.current_gen}!")
                return True

        self.last_best_so_far = self.best_so_far
        return False

    def update_quantum(self):
        """ Update quantum populations of networks and hyperparameters. """

        if np.remainder(self.current_gen,
                        self.update_quantum_gen) == 0 and self.current_gen > 0:

            self.qpop_params.update_quantum(intensity=self.random)

            if self.mixedop_mode and self._last_trained_alphas is not None:
                # Use trained alpha values to update quantum alpha_logits (PDF mode)
                valid_alphas = [a for a in self._last_trained_alphas if a is not None]
                if valid_alphas:
                    num_ind = self.qpop_net.num_ind
                    # best_indices: indices of best-ranked individuals (already ordered)
                    best_indices = np.arange(
                        min(num_ind, len(valid_alphas)), dtype=np.int32
                    )
                    trained_arr = np.stack(valid_alphas, axis=0)
                    self.qpop_net.update_quantum_from_alpha(
                        trained_arr, best_indices, intensity=self.random
                    )
            else:
                self.qpop_net.update_quantum(intensity=self.random)
    
    def go_next_gen(self):
        """ Go to the next generation --> update quantum genes, log data, delete unnecessary
            training files and update generation counter.
        """

        self.update_quantum()

        self.save_data()
        self.log_data()
        #self.save_train_data()

        # Remove Tensorflow models files
        delete_old_dirs(self.experiment_path, keep_best=True,
                        best_id=f'{self.best_so_far_id[0]}_{self.best_so_far_id[1]}')
        self.current_gen += 1

    def collapse_best_network(self):
        """ Collapse the best alpha found during the evolution (mixedop_mode) into a discrete
            network: for each node, keep only the single operation with the highest alpha
            weight. Saves the result to *self.experiment_path*.

        Returns:
            list of function names representing the collapsed (discrete) network, or None if
            not running in mixedop_mode / no alpha is available.
        """

        if not self.mixedop_mode:
            return None

        alpha = self.best_alpha
        fn_list = self.best_alpha_fn_list
        if alpha is None:
            self.logger.info("No best_alpha tracked; falling back to quantum alpha_logits "
                            "centroid for collapsing the network.")
            alpha = self.qpop_net.alpha_logits[0]
            fn_list = self.qpop_net.chromosome.fn_list

        op_indices = np.argmax(alpha, axis=-1).astype(np.int32)
        # Decode against fn_list as it was when best_alpha was captured, NOT the
        # current chromosome's fn_list - progressive stages may have pruned it since
        # (op_indices would then point past the end of a shorter, later fn_list).
        net_list = [fn_list[gene] for gene in op_indices]

        self.logger.info(f"Collapsed best network (one operation per node): {net_list}")

        result = {
            'best_so_far_id': self.best_so_far_id,
            'best_so_far': self.best_so_far,
            'op_indices': op_indices.tolist(),
            'net_list': net_list,
        }

        out_path = os.path.join(self.experiment_path, 'best_network_collapsed.pkl')
        with open(out_path, 'wb') as f:
            dump(result, f, protocol=HIGHEST_PROTOCOL)

        return net_list

    def _rank_and_prune_ops(self, old_fn_list, num_ops):
        """Rank ops by mean trained softmax(alpha) weight (across the population and all
        nodes) from the stage that just finished, and keep the top *num_ops*.

        Falls back to keeping the first *num_ops* of *old_fn_list* if no trained alphas
        are available yet (shouldn't normally happen, since stage transitions only occur
        after at least one generation has been evaluated).
        """
        if num_ops >= len(old_fn_list):
            return list(old_fn_list)

        valid_alphas = [a for a in (self._last_trained_alphas or []) if a is not None]
        if not valid_alphas:
            self.logger.info("No trained alphas available for op pruning; keeping first "
                            f"{num_ops} ops of {old_fn_list}.")
            return list(old_fn_list[:num_ops])

        stacked = np.stack(valid_alphas, axis=0)  # [ind, nodes, ops]
        exp = np.exp(stacked - stacked.max(axis=-1, keepdims=True))
        softmax = exp / exp.sum(axis=-1, keepdims=True)
        mean_weight = softmax.mean(axis=(0, 1))  # [ops]

        ranked = sorted(range(len(old_fn_list)), key=lambda i: mean_weight[i], reverse=True)
        kept_indices = sorted(ranked[:num_ops])
        new_fn_list = [old_fn_list[i] for i in kept_indices]
        self.logger.info(f"Op pruning: {old_fn_list} -> {new_fn_list}")
        return new_fn_list

    def _transition_stage(self, new_stage_idx):
        """Grow depth / prune ops / warm-start weights at a progressive-stage boundary."""
        stage = self.progressive_stages[new_stage_idx]
        old_fn_list = list(self.qpop_net.chromosome.fn_list)
        old_num_nodes = self.qpop_net.chromosome.num_genes

        new_fn_list = self._rank_and_prune_ops(old_fn_list, stage['num_ops'])
        self.qpop_net.grow_and_prune(stage['num_nodes'], new_fn_list)

        self.eval_func.fn_list = new_fn_list
        self.eval_func.set_stage_weight_bank(new_stage_idx)

        best_gen, best_ind = self.best_so_far_id
        src_path = os.path.join(self.experiment_path, f'{best_gen}_{best_ind}', 'best_model.pth')
        if os.path.exists(src_path):
            self.eval_func.warm_start_info = {
                'state_dict_path': src_path,
                'old_fn_list': old_fn_list,
                'old_num_nodes': old_num_nodes,
            }
        else:
            self.logger.info(f"No best model found at {src_path}; stage {new_stage_idx} "
                            "will train from scratch instead of warm-starting.")
            self.eval_func.warm_start_info = None

        self.current_stage_idx = new_stage_idx
        self.logger.info(
            f"Progressive stage transition -> stage {new_stage_idx} "
            f"(gen_start={stage['gen_start']}, num_nodes={stage['num_nodes']}, "
            f"num_ops={stage['num_ops']}): "
            f"{old_num_nodes} -> {stage['num_nodes']} nodes, "
            f"{len(old_fn_list)} -> {len(new_fn_list)} ops."
        )

    def evolve(self):
        """ Run the evolution. """
        start_evolution = time.time()
        max_generations = self.generations
        #print(f"early stopping enable?: {self.early_stopping}")
        #print(f"population crossover enable?: {self.en_pop_crossover}")
        
        # Update maximum number of generations if continue previous evolution process
        if self.current_gen > 0:
            # Resuming (crash/restart via --continue_path): current_gen is the last
            # COMPLETED generation restored from the log, so pick up at the next one.
            # max_generations stays the original absolute target from config, so a
            # resumed run finishes the SAME run instead of tacking on extra generations
            # every time it's resumed. To deliberately extend a finished run, raise
            # max_generations in the config before resuming.
            self.current_gen += 1

        while self.current_gen < max_generations:
            # Estimate time to finish the evolution
            if self.current_gen % 5 == 0 and self.current_gen > 0:
                int_time = time.time()
                int_hours, int_mins, est_hours, est_mins = calculate_time(start_evolution,int_time,self.current_gen, max_generations, end_evol=False)
                self.logger.info(f"Current evolution time at generation {self.current_gen}: {int_hours} hours and {int_mins} mins")
                self.logger.info(f"Estimated time to finish the evolution: {est_hours} hours and {est_mins} mins")

            if self.mixedop_mode and self.progressive_stages:
                target_stage_idx = max(
                    i for i, s in enumerate(self.progressive_stages)
                    if s['gen_start'] <= self.current_gen
                )
                if target_stage_idx != self.current_stage_idx:
                    self._transition_stage(target_stage_idx)

            self.generate_classical()
            self.go_next_gen()
            
            if self.early_stopping and self.check_early_stopping(): break
        
        end_evolution = time.time()
        evolution_hours, evolution_minutes = calculate_time(start_evolution,end_evolution)
        self.logger.info(f"Total evolution time: {evolution_hours} hours and {evolution_minutes} minutes")

        if self.mixedop_mode:
            self.collapse_best_network()