""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Q-NAS algorithm class.
"""

import datetime
import os
import shutil
from pickle import dump, load, HIGHEST_PROTOCOL, UnpicklingError

import numpy as np
import time

import surrogate
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
        self.reducing_fns_names = set()
        self.replace_method = None
        self.save_data_freq = np.inf
        self.total_eval = 0
        self.early_stopping_counter = 0
        self.last_best_so_far = None            # best fitness at the previous early-stopping check

        # Quantum update engine selection - see initialize_qnas() docstring.
        self.quantum_update_engine = 'default'
        self.quantum_update_age_decay = 0.5
        # Per-classical-individual lineage tracking (row-aligned with current_pop),
        # used only by the 'ancestor_decay' engine: how many generations each
        # individual has survived, and which quantum individual originally produced
        # it. Populated by replace_pop().
        self.classic_age = None
        self.classic_ancestor = None

        # Rank-based network update and op-pruning criterion - see initialize_qnas().
        # quantum_lr None keeps the original update rules untouched.
        self.quantum_lr = None
        self.quantum_rank_weighting = 'uniform'
        self.quantum_top_k = 1
        self.quantum_negative_lr = 0.0
        self.quantum_prob_floor = 0.0
        self.prune_criterion = 'pmf'
        # Per-node mean PMF at the start of the current stage (reference for the 'lift'
        # criterion and for the KL logged each generation), and the individuals evaluated
        # since then with their fitnesses (for the 'empirical' criterion).
        self._stage_start_probs = None
        self._stage_eval_net = []
        self._stage_eval_fit = []

        self.qpop_params = None
        self.qpop_net = None

        # P-DARTS-style progressive depth growth + op pruning (optional)
        self.progressive_stages = None
        self.progressive_mode = None
        self.current_stage_idx = 0
        self._just_transitioned = False  # True for the generation right after a stage
                                          # transition, when current_pop has already been
                                          # filtered/remapped by _transition_stage and
                                          # replace_pop should keep all survivors rather
                                          # than apply elitism's "keep only 1" slice.

        # How many of the current generation's individuals were actually trained
        # (i.e. did NOT hit the architecture cache) - see eval_pop().
        self.new_architectures_count = 0

        # Options of docs/QNAS_SEARCH_ENGINE_IMPROVEMENT_PLAN_V2.md - see initialize_qnas().
        # All off by default (the original search).
        self.surrogate_enabled = False
        self.surrogate_rng = None
        self.sampling_tau0 = None
        self.niche_max_age = None
        self.niche_restart_patience = None
        self.niche_restart_kl = None
        self.niche_best = None              # best fitness ever produced by each niche
        self.niche_last_improve = None      # generation of that best
        self.fidelity_enabled = False
        self.selection_capacity_lambda = 0.0
        self._stage_start_gen = 0           # generation at which the current stage began
        self.last_screening = None          # (pool size, evaluated, tau) of the last screening

    def initialize_qnas(self, num_quantum_ind, params_ranges, repetition, max_generations,
                        crossover_rate, update_quantum_gen, replace_method, fn_list,
                        initial_probs, update_quantum_rate, max_num_nodes, reducing_fns_list,
                        patience, early_stopping, save_data_freq=0, penalize_number=0,
                        crossover_frequency=5, en_pop_crossover=False,
                        pop_crossover_rate=0.25, pop_crossover_method='hux',
                        progressive_stages=None, noop_fn_name=None,
                        reset_probs_on_stage_change=False, global_op_pruning=False,
                        progressive_mode=None, dynamic_initial_num_nodes=None,
                        dynamic_max_num_nodes=None, dynamic_probability_threshold=0.8,
                        dynamic_min_ops=2, dynamic_flatness_epsilon=0.0,
                        dynamic_check_every_gen=None, dynamic_growth_patience=1,
                        dynamic_node_growth_amount=1, quantum_update_engine='default',
                        quantum_update_age_decay=0.5, quantum_lr=None,
                        quantum_rank_weighting='uniform', quantum_top_k=1,
                        quantum_negative_lr=0.0, quantum_prob_floor=0.0,
                        quantum_max_update=0.05, quantum_max_prob=0.99,
                        prune_criterion='pmf', surrogate_enabled=False,
                        surrogate_oversample=1, surrogate_n_eval=None,
                        surrogate_warmup_evals=200, surrogate_tau0=1.0, surrogate_tau_min=0.1,
                        surrogate_tau_decay_gens=20, surrogate_beta=0.5,
                        surrogate_random_frac=0.2, surrogate_ensemble=5,
                        surrogate_ridge_lambda=3.0, surrogate_seed=0, sampling_tau0=None,
                        sampling_tau_min=0.3, sampling_tau_decay_gens=30, niche_max_age=None,
                        niche_restart_patience=None, niche_restart_kl=None,
                        fidelity_enabled=False, fidelity_low_epochs=10, fidelity_oversample=2,
                        selection_capacity_lambda=0.0):

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
            reset_probs_on_stage_change: (bool) if True, every existing node's quantum
                probabilities are reset to uniform at each progressive-stage transition
                instead of carrying over/renormalizing the pre-transition PMF - i.e. the
                search "forgets" what it learned so far and restarts exploration on the
                new (pruned) op menu. Ignored if progressive_stages is not set.
            global_op_pruning: (bool) if True, op pruning at each stage transition ranks
                ops ONCE using the mean quantum probability mass across every node and
                individual, and applies that single surviving op list to every node
                (the pre-per-node-pruning behavior - all nodes always share one op
                menu). If False (default), each node ranks and prunes its own ops
                independently, so different nodes can end up with different op
                subsets. Applies to both progressive modes.
            progressive_mode: (str or None) 'deterministic' or 'dynamic', set only when
                progressive mode is active (None otherwise). 'deterministic' uses
                progressive_stages (gen_start-triggered, fixed num_ops per stage).
                'dynamic' has no stage list - see the dynamic_* args below.
            dynamic_initial_num_nodes: (int) starting depth for dynamic mode - the
                dynamic-mode equivalent of progressive_stages[0]['num_nodes'].
            dynamic_max_num_nodes: (int) growth ceiling for dynamic mode.
            dynamic_probability_threshold: (float) nucleus mass to keep per node (or
                network-wide under global_op_pruning) when dynamic-pruning ops.
            dynamic_min_ops: (int) floor on ops kept per node (NoOp included) when
                dynamic-pruning.
            dynamic_flatness_epsilon: (float) if the rankable (non-NoOp) ops' mean-PMF
                spread (max - min) is <= this, dynamic pruning skips this check
                entirely - the distribution hasn't differentiated enough to prune yet.
            dynamic_check_every_gen: (int or None) how often (generations) dynamic mode
                re-ranks/cuts; must be a positive multiple of update_quantum_gen.
                Defaults to update_quantum_gen if None.
            dynamic_growth_patience: (int) consecutive stable (no-cut) checks required
                before dynamic mode grows depth.
            dynamic_node_growth_amount: (int) nodes added per dynamic growth event,
                capped at dynamic_max_num_nodes.
            quantum_update_engine: (str) one of 'default' or 'ancestor_decay' -
                selects how the network quantum population is rotated toward
                observed classical individuals. See update_quantum() and
                QPopulationNetwork.update_quantum_ancestor_decay() for details.
            quantum_update_age_decay: (float) decay rate used by the
                'ancestor_decay' engine (ignored by 'default').
            quantum_lr: (float or None) None (default) keeps the original network update
                rules (random intensity * quantum_max_update). A value in (0, 1] switches
                to the rank-based update (QPopulationNetwork.update_quantum_ranked):
                every update moves each quantum individual this fraction of the way
                toward its targets. quantum_update_engine still picks the targets:
                'default' -> ranks q, q + num_ind, ... of current_pop; 'ancestor_decay'
                -> the best rows of q's own lineage, weighted by exp(-decay * age).
            quantum_rank_weighting: (str) 'uniform', 'linear' or 'nes' - how the
                quantum_top_k targets of one quantum individual are weighted.
            quantum_top_k: (int) targets per quantum individual (rank-based update).
            quantum_negative_lr: (float) in [0, 1) - rank-based update only: fraction of
                probability removed from the op of a poor individual (the worst of the
                lineage for 'ancestor_decay', rank n-1-q for 'default').
            quantum_prob_floor: (float) in [0, 1) - rank-based update only: every op
                keeps at least quantum_prob_floor / n_ops of probability.
            quantum_max_update: (float) max step of the original update rules.
            quantum_max_prob: (float) ceiling on any op's probability (all update rules).
            prune_criterion: (str) how progressive op pruning ranks a node's ops
                (deterministic mode only): 'pmf' (default) - mean quantum probability;
                'lift' - mean probability divided by its value at the start of the
                stage, so ops with a large prior (pooling, no_op) don't survive on the
                prior alone; 'empirical' - mean fitness percentile of the individuals
                evaluated in the stage that used the op at that node; 'surrogate' - the
                op's coefficient in a ridge-ensemble fit of the stage's evaluations (its
                marginal effect on fitness, controlling for the other nodes).

            The remaining arguments are the options of
            docs/QNAS_SEARCH_ENGINE_IMPROVEMENT_PLAN_V2.md (defaults in
            src/option_defaults.py; every default is the original search):
            surrogate_*: D1, surrogate-assisted screening - each generation samples
                surrogate_oversample x surrogate_n_eval candidates and trains only
                surrogate_n_eval of them, picked by Boltzmann sampling on a ridge-ensemble
                prediction (see generate_classical / src/surrogate.py).
            sampling_tau0 / sampling_tau_min / sampling_tau_decay_gens: D3, temperature of
                the PMF sampling, annealed within each stage.
            niche_max_age / niche_restart_patience / niche_restart_kl: D4, used with
                replace_method 'niche' (one quota of the classical population per quantum
                individual's lineage).
            fidelity_enabled / fidelity_low_epochs / fidelity_oversample: D2, two-rung
                evaluation - fidelity_oversample x the evaluated individuals train for
                fidelity_low_epochs, and only the best continue to the full budget.
            selection_capacity_lambda: D2, adds lambda * log(params in millions) to the
                fitness used for selection (the raw fitness is unchanged).
        """

        self.generations = max_generations
        self.update_quantum_gen = update_quantum_gen
        self.replace_method = replace_method
        self.quantum_update_engine = quantum_update_engine
        self.quantum_update_age_decay = quantum_update_age_decay
        self._set_update_and_prune_options(
            quantum_lr, quantum_rank_weighting, quantum_top_k, quantum_negative_lr,
            quantum_prob_floor, quantum_max_prob, prune_criterion, progressive_mode)
        self._set_v2_options(
            num_quantum_ind, repetition, replace_method, progressive_mode,
            dict(enabled=surrogate_enabled, oversample=surrogate_oversample,
                 n_eval=surrogate_n_eval, warmup_evals=surrogate_warmup_evals,
                 tau0=surrogate_tau0, tau_min=surrogate_tau_min,
                 tau_decay_gens=surrogate_tau_decay_gens, beta=surrogate_beta,
                 random_frac=surrogate_random_frac, ensemble=surrogate_ensemble,
                 ridge_lambda=surrogate_ridge_lambda, seed=surrogate_seed),
            sampling_tau0, sampling_tau_min, sampling_tau_decay_gens,
            niche_max_age, niche_restart_patience, niche_restart_kl,
            fidelity_enabled, fidelity_low_epochs, fidelity_oversample,
            selection_capacity_lambda)
        self.penalize_number = penalize_number
        self.patience = patience
        self.early_stopping = early_stopping
        self.en_pop_crossover = en_pop_crossover
        self.pop_crossover_rate = pop_crossover_rate
        self.crossover_frequency = crossover_frequency

        # P-DARTS-style progressive growth: stage 0 always starts from the full fn_list
        # (op pruning ranks ops by the quantum PMF, which only reflects a pruned fn_list
        # once a stage transition has narrowed it) - later stages narrow fn_list and grow
        # num_nodes.
        self.progressive_stages = progressive_stages
        self.current_stage_idx = 0
        self.noop_fn_name = noop_fn_name
        self.reset_probs_on_stage_change = reset_probs_on_stage_change
        self.global_op_pruning = global_op_pruning
        self.progressive_mode = progressive_mode
        self._stable_streak = None  # dynamic mode only - initialized below, once
                                     # self.qpop_net (and its num_genes) exists.
        if progressive_stages:
            if progressive_stages[0]['num_ops'] != len(fn_list):
                raise ValueError(
                    "progressive_stages[0].num_ops must equal len(fn_list) - the first "
                    "stage starts from the full op menu."
                )
            for stage in progressive_stages:
                if stage['num_nodes'] < 1:
                    raise ValueError("progressive_stages[*].num_nodes must be >= 1.")
            if not noop_fn_name or noop_fn_name not in fn_list:
                raise ValueError(
                    "progressive_stages requires noop_fn_name to be set and present in "
                    "fn_list - per-node pruning force-keeps it forever so grown nodes "
                    "can extend surviving classical individuals with a no-op."
                )
            max_num_nodes = progressive_stages[0]['num_nodes']
        elif progressive_mode == 'dynamic':
            if not noop_fn_name or noop_fn_name not in fn_list:
                raise ValueError(
                    "progressive dynamic mode requires noop_fn_name to be set and "
                    "present in fn_list - per-node pruning force-keeps it forever so "
                    "grown nodes can extend surviving classical individuals with a "
                    "no-op."
                )
            self.dynamic_max_num_nodes = dynamic_max_num_nodes
            self.dynamic_probability_threshold = dynamic_probability_threshold
            self.dynamic_min_ops = dynamic_min_ops
            self.dynamic_flatness_epsilon = dynamic_flatness_epsilon
            self.dynamic_check_every_gen = dynamic_check_every_gen or update_quantum_gen
            self.dynamic_growth_patience = dynamic_growth_patience
            self.dynamic_node_growth_amount = dynamic_node_growth_amount
            max_num_nodes = dynamic_initial_num_nodes

        # Per-node reducing-ops names (used by get_penalties). Kept as names, not
        # global indices: once fn_list is per-node (progressive stages), the same
        # gene integer can mean a different op at different node positions, so
        # index-based matching would silently misclassify ops.
        self.reducing_fns_names = set()
        if reducing_fns_list:
            self.penalties = np.zeros(shape=(num_quantum_ind * repetition))
            self.reducing_fns_names = set(reducing_fns_list)

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
                                            crossover_method=pop_crossover_method,
                                            max_update=quantum_max_update,
                                            max_prob=quantum_max_prob)

        if self.progressive_mode == 'dynamic':
            self._init_stability_streaks()
        self._snapshot_stage_start_probs()
        self.niche_best = np.full(num_quantum_ind, -np.inf)
        self.niche_last_improve = np.zeros(num_quantum_ind, dtype=np.int64)

    REPLACE_METHODS = ('best', 'elitism', 'niche')

    def _set_v2_options(self, num_quantum_ind, repetition, replace_method, progressive_mode,
                        surrogate, sampling_tau0, sampling_tau_min, sampling_tau_decay_gens,
                        niche_max_age, niche_restart_patience, niche_restart_kl,
                        fidelity_enabled, fidelity_low_epochs, fidelity_oversample,
                        selection_capacity_lambda):
        """ Validate and store the options of docs/QNAS_SEARCH_ENGINE_IMPROVEMENT_PLAN_V2.md. """
        if replace_method not in self.REPLACE_METHODS:
            raise ValueError(f"replace_method must be one of {self.REPLACE_METHODS}, "
                             f"got {replace_method!r}")
        num_classic = num_quantum_ind * repetition

        if surrogate['enabled']:
            if progressive_mode == 'dynamic':
                raise ValueError("surrogate screening is not supported with progressive "
                                 "mode 'dynamic'.")
            if int(surrogate['oversample']) < 1:
                raise ValueError("surrogate oversample must be >= 1.")
            if surrogate['n_eval'] is not None and not \
                    num_quantum_ind <= surrogate['n_eval'] <= num_classic:
                raise ValueError("surrogate n_eval must be between num_quantum_ind and "
                                 "num_quantum_ind * repetition.")
            if surrogate['tau0'] <= 0 or surrogate['tau_min'] <= 0:
                raise ValueError("surrogate tau0 and tau_min must be > 0.")
            if surrogate['tau_decay_gens'] <= 0:
                raise ValueError("surrogate tau_decay_gens must be > 0.")
            if not 0.0 <= surrogate['random_frac'] <= 1.0:
                raise ValueError("surrogate random_frac must be in [0, 1].")
            if surrogate['warmup_evals'] < 0 or surrogate['ensemble'] < 1:
                raise ValueError("surrogate warmup_evals must be >= 0 and ensemble >= 1.")
        self.surrogate_enabled = bool(surrogate['enabled'])
        self.surrogate_oversample = int(surrogate['oversample'])
        self.surrogate_n_eval = surrogate['n_eval']
        self.surrogate_warmup_evals = surrogate['warmup_evals']
        self.surrogate_tau0 = surrogate['tau0']
        self.surrogate_tau_min = surrogate['tau_min']
        self.surrogate_tau_decay_gens = surrogate['tau_decay_gens']
        self.surrogate_beta = surrogate['beta']
        self.surrogate_random_frac = surrogate['random_frac']
        self.surrogate_ensemble = surrogate['ensemble']
        self.surrogate_ridge_lambda = surrogate['ridge_lambda']
        # Own RNG: screening must never draw from the global numpy RNG that the rest of
        # the search consumes.
        self.surrogate_rng = np.random.default_rng(surrogate['seed'])

        if sampling_tau0 is not None and (sampling_tau0 <= 0 or sampling_tau_min <= 0
                                          or sampling_tau_decay_gens <= 0):
            raise ValueError("sampling_temperature tau0, tau_min and tau_decay_gens must be > 0.")
        self.sampling_tau0 = sampling_tau0
        self.sampling_tau_min = sampling_tau_min
        self.sampling_tau_decay_gens = sampling_tau_decay_gens

        niche_opts = (niche_max_age, niche_restart_patience, niche_restart_kl)
        if replace_method != 'niche' and any(v is not None for v in niche_opts):
            raise ValueError("niche max_age / restart_patience / restart_kl require "
                             "replace_method: niche.")
        if niche_max_age is not None and niche_max_age < 1:
            raise ValueError("niche max_age must be >= 1.")
        if niche_restart_patience is not None and niche_restart_patience < 1:
            raise ValueError("niche restart_patience must be >= 1.")
        if niche_restart_kl is not None and niche_restart_kl <= 0:
            raise ValueError("niche restart_kl must be > 0.")
        self.niche_max_age = niche_max_age
        self.niche_restart_patience = niche_restart_patience
        self.niche_restart_kl = niche_restart_kl

        if fidelity_enabled and (int(fidelity_oversample) < 1 or fidelity_low_epochs < 1):
            raise ValueError("fidelity oversample and low_epochs must be >= 1.")
        self.fidelity_enabled = bool(fidelity_enabled)
        self.fidelity_low_epochs = fidelity_low_epochs
        self.fidelity_oversample = int(fidelity_oversample)

        if selection_capacity_lambda < 0:
            raise ValueError("selection_capacity_lambda must be >= 0.")
        self.selection_capacity_lambda = float(selection_capacity_lambda)

    def replace_pop(self, new_pop_params, new_pop_net, new_fitnesses, raw_fitnesses,
                    new_ancestor=None):
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
            new_ancestor: int ndarray, shape (new_fitnesses.shape[0],), the quantum
                individual index (0..num_quantum_ind-1) that generated each row of
                *new_pop_net*/*new_pop_params*. Used only by the 'ancestor_decay'
                quantum update engine to track lineage/age of classical individuals
                across generations. If None, falls back to a positional index
                (row % num_quantum_ind).
        """

        if new_ancestor is None:
            new_ancestor = np.arange(new_fitnesses.shape[0]) % self.qpop_net.num_ind
        new_ancestor = new_ancestor.astype(np.int64)
        if self.replace_method == 'niche':
            self._track_niche_progress(new_fitnesses, new_ancestor)

        if self.current_gen == 0 or (self._just_transitioned and self.qpop_net.current_pop is None):
            # In the 1st generation, and right after a progressive-stage transition that
            # left no surviving classical individual (_transition_stage already filtered
            # out anyone whose ops didn't survive the prune), the current population is
            # the one that was just generated.
            self.qpop_params.current_pop = new_pop_params
            self.qpop_net.current_pop = new_pop_net

            self.fitnesses = new_fitnesses
            self.raw_fitnesses = raw_fitnesses
            self.classic_age = np.zeros(new_fitnesses.shape[0], dtype=np.int64)
            self.classic_ancestor = new_ancestor
            self.update_best_id(new_fitnesses)
        else:
            # Checking if the best so far individual has changed in the current generation
            self.update_best_id(new_fitnesses)

            # Lineage/age tracking can go stale (e.g. right after a progressive-stage
            # transition externally filtered/remapped current_pop without knowing about
            # these arrays) - auto-heal instead of crashing on a shape mismatch. This
            # just means age tracking restarts from that point.
            if (self.classic_age is None
                    or self.classic_age.shape[0] != self.fitnesses.shape[0]):
                self.classic_age = np.zeros(self.fitnesses.shape[0], dtype=np.int64)
                self.classic_ancestor = np.arange(
                    self.fitnesses.shape[0]) % self.qpop_net.num_ind

            if self._just_transitioned:
                # _transition_stage already filtered/remapped current_pop to only the
                # individuals whose fitness is still valid post-transition - keep all of
                # them (like 'best') rather than applying elitism's "keep only 1" slice,
                # which assumes an untouched prior generation.
                selected = range(self.fitnesses.shape[0])
            elif self.replace_method == 'elitism':
                select_new = range(new_fitnesses.shape[0] - 1)
                new_fitnesses, raw_fitnesses, new_pop_params, new_pop_net, \
                    (new_ancestor,) = self.order_pop(new_fitnesses,
                                                    raw_fitnesses,
                                                    new_pop_params,
                                                    new_pop_net,
                                                    select_new,
                                                    extra=[new_ancestor])
                selected = range(1)
            elif self.replace_method in ('best', 'niche'):
                # 'niche' keeps every old individual here too; the per-lineage quotas are
                # applied to the union below (_niche_selection).
                selected = range(self.fitnesses.shape[0])

            # Concatenate populations
            self.fitnesses = np.concatenate((self.fitnesses[selected], new_fitnesses))
            self.raw_fitnesses = np.concatenate((self.raw_fitnesses[selected], raw_fitnesses))
            self.qpop_params.current_pop = np.concatenate(
                    (self.qpop_params.current_pop[selected], new_pop_params))
            self.qpop_net.current_pop = np.concatenate(
                    (self.qpop_net.current_pop[selected], new_pop_net))
            # Individuals that survive into the next generation get older; freshly
            # generated ones start at age 0 - this is what lets the 'ancestor_decay'
            # engine nudge less and less toward a stale, long-surviving elite.
            self.classic_age = np.concatenate(
                (self.classic_age[selected] + 1,
                np.zeros(new_fitnesses.shape[0], dtype=np.int64)))
            self.classic_ancestor = np.concatenate(
                (self.classic_ancestor[selected], new_ancestor))

        ## TODO: Here we have the the last and new population Multi objective operation


        # Order the population based on fitness
        num_classic = self.qpop_params.num_ind * self.qpop_params.repetition
        if self.replace_method == 'niche':
            keep = self._niche_selection(num_classic)
            self.fitnesses = self.fitnesses[keep]
            self.raw_fitnesses = self.raw_fitnesses[keep]
            self.qpop_params.current_pop = self.qpop_params.current_pop[keep]
            self.qpop_net.current_pop = self.qpop_net.current_pop[keep]
            self.classic_age = self.classic_age[keep]
            self.classic_ancestor = self.classic_ancestor[keep]
        self.fitnesses, self.raw_fitnesses, self.qpop_params.current_pop, \
            self.qpop_net.current_pop, (self.classic_age, self.classic_ancestor) = \
            self.order_pop(self.fitnesses,
                            self.raw_fitnesses,
                            self.qpop_params.current_pop,
                            self.qpop_net.current_pop,
                            # the population can start below num_classic when fewer
                            # individuals are evaluated per generation (surrogate n_eval)
                            selection=range(min(num_classic, self.fitnesses.shape[0])),
                            extra=[self.classic_age, self.classic_ancestor])

        # self.fitnesses[0] is normally the best-ever individual (elitism/best always
        # concatenate history in), except right after a progressive-stage transition
        # that left no surviving individual, where the population is a fresh start with
        # no history concatenated - guard with max() so best_so_far can never regress.
        self.best_so_far = max(self.best_so_far, self.fitnesses[0])
        self._just_transitioned = False

    @staticmethod
    def order_pop(fitnesses, raw_fitnesses, pop_params, pop_net, selection=None, extra=None):
        """ Order the population based on *fitnesses*.

        Args:
            fitnesses: ndarray with fitnesses values.
            raw_fitnesses: float ndarray representing the fitness of each individual before the
                penalization method.
            pop_params: ndarray with population of parameters.
            pop_net: ndarray with population of networks.
            selection: range to select elements from the population.
            extra: optional list of additional 1D ndarrays, aligned with *fitnesses*,
                to reorder/select in lockstep (e.g. the per-individual age/ancestor
                lineage tracking used by the 'ancestor_decay' quantum update engine).

        Returns:
            ordered population and fitnesses, plus a tuple of the reordered *extra*
            arrays as a 5th return value if *extra* was given.
        """

        if selection is None:
            selection = range(fitnesses.shape[0])
        idx = np.argsort(fitnesses)[::-1]
        pop_params = pop_params[idx][selection]
        pop_net = pop_net[idx][selection]
        fitnesses = fitnesses[idx][selection]
        raw_fitnesses = raw_fitnesses[idx][selection]

        if extra is not None:
            extra = tuple(arr[idx][selection] for arr in extra)
            return fitnesses, raw_fitnesses, pop_params, pop_net, extra

        return fitnesses, raw_fitnesses, pop_params, pop_net

    def _niche_selection(self, num_classic):
        """ replace_method 'niche' (plan v2, D4): row indices of the union population to keep.

        Each quantum individual's lineage (classic_ancestor) gets num_classic // num_ind
        slots, filled with its best members; free slots (a lineage short of members) go to
        the best remaining individuals of any lineage. With niche_max_age, individuals that
        survived more than that many generations are only used once nobody else is left -
        except the best individual of the union, which is always kept.
        """
        fit = np.nan_to_num(np.asarray(self.fitnesses, dtype=np.float64), nan=-np.inf)
        n = fit.shape[0]
        if n <= num_classic and self.niche_max_age is None:
            return np.arange(n)
        num_ind = self.qpop_net.num_ind
        eligible = np.ones(n, dtype=bool)
        if self.niche_max_age is not None:
            eligible = self.classic_age <= self.niche_max_age
            eligible[int(np.argmax(fit))] = True

        quota = num_classic // num_ind
        keep = []
        for q in range(num_ind):
            members = np.where(eligible & (self.classic_ancestor == q))[0]
            keep.extend(members[np.argsort(-fit[members], kind='stable')][:quota].tolist())
        taken = np.zeros(n, dtype=bool)
        taken[keep] = True
        for pool in (eligible & ~taken, ~eligible & ~taken):
            if len(keep) >= num_classic:
                break
            rest = np.where(pool)[0]
            rest = rest[np.argsort(-fit[rest], kind='stable')]
            keep.extend(rest[:num_classic - len(keep)].tolist())
        return np.array(sorted(keep), dtype=np.int64)

    def _track_niche_progress(self, new_fitnesses, new_ancestor):
        """ Record each niche's best fitness and when it last improved (niche restarts). """
        fit = np.nan_to_num(np.asarray(new_fitnesses, dtype=np.float64), nan=-np.inf)
        for q in np.unique(new_ancestor):
            best = fit[new_ancestor == q].max()
            if best > self.niche_best[q] + self.tolerance:
                self.niche_best[q] = best
                self.niche_last_improve[q] = self.current_gen

    def _maybe_restart_niches(self):
        """ replace_method 'niche' (plan v2, D4): reset a quantum individual's PMF to uniform
            when its niche stalls (no improvement for niche_restart_patience generations)
            or collapses onto the best niche (mean KL to it below niche_restart_kl) - the
            per-stage reset that helped in the experiments, applied to one niche at a time.
            The niche holding the best fitness is never restarted. """
        if self.replace_method != 'niche' or (self.niche_restart_patience is None
                                              and self.niche_restart_kl is None):
            return
        best_niche = int(np.argmax(self.niche_best))
        for q in range(self.qpop_net.num_ind):
            if q == best_niche:
                continue
            stalled = (self.niche_restart_patience is not None
                       and self.current_gen - self.niche_last_improve[q] >= self.niche_restart_patience)
            collapsed = False
            if self.niche_restart_kl is not None:
                kls = []
                for probs in self.qpop_net.probabilities:
                    p = np.clip(probs[q], 1e-12, None)
                    r = np.clip(probs[best_niche], 1e-12, None)
                    kls.append(float((p * (np.log(p) - np.log(r))).sum()))
                collapsed = float(np.mean(kls)) < self.niche_restart_kl
            if stalled or collapsed:
                for probs in self.qpop_net.probabilities:
                    probs[q] = 1.0 / probs.shape[1]
                self.niche_last_improve[q] = self.current_gen
                self.logger.info(f"Niche restart: quantum individual {q} reset to uniform "
                                 f"({'stalled' if stalled else 'collapsed onto niche ' + str(best_niche)}).")

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

    def generate_classical(self):
        """ Generate a specific number of classical individuals from the observation of quantum
            individuals. This number is equal to (*num_ind* x *repetition*). The new classic
            individuals will be evaluated and ordered according to their fitness values.
        """

        # Generate distance for crossover and quantum updates every generation
        self.random = np.random.rand()
        self.qpop_net.sampling_tau = self._sampling_tau()

        if self.surrogate_enabled or self.fidelity_enabled:
            new_pop_params, new_pop_net, new_ancestor, new_fitnesses, raw_fitnesses = \
                self._generate_screened()
            self._record_stage_evaluations(new_pop_net, new_fitnesses)
            self.replace_pop(new_pop_params, new_pop_net, new_fitnesses, raw_fitnesses,
                             new_ancestor)
            return

        new_pop_params, new_pop_net, new_ancestor = self._sample_block()

        self.logger.info("new population created =%s", new_pop_net)
        new_fitnesses, raw_fitnesses = self.eval_pop(new_pop_params, new_pop_net)
        self._record_stage_evaluations(new_pop_net, new_fitnesses)

        self.replace_pop(new_pop_params, new_pop_net, new_fitnesses, raw_fitnesses,
                        new_ancestor)

    def _sample_block(self):
        """ One classical population (num_ind x repetition rows) observed from the quantum
            populations, with the hyperparameter and population crossovers - the original
            generate_classical() body. Returns (params, net, ancestor). """
        # Generate classical pop for hyperparameters
        new_pop_params = self.qpop_params.generate_classical()
        if self.current_gen > 0:
            new_pop_params = self.qpop_params.classic_crossover(new_pop=new_pop_params,
                                                                distance=self.random)

        new_pop_net = self.qpop_net.generate_classical()
        # Row i of a freshly generated population was sampled from quantum individual
        # i % num_ind (see QPopulationNetwork.generate_classical) - record this before
        # any crossover/reordering so the 'ancestor_decay' quantum update engine can
        # later trace each classical individual back to the quantum individual that
        # produced it, instead of relying on its current rank in current_pop.
        new_ancestor = np.arange(new_pop_net.shape[0]) % self.qpop_net.num_ind

        if (self.current_gen > 0 and self.en_pop_crossover
                and self.qpop_net.current_pop is not None):
            if self.current_gen % self.crossover_frequency == 0:
                num_offspring = int(len(new_pop_net) * self.pop_crossover_rate)
                # current_pop can be smaller than the full population right after a
                # dynamic prune/growth step (individuals that didn't survive op
                # pruning are dropped - see _apply_fn_list_change), so clamp to what
                # is actually available to avoid a shape mismatch in apply_crossover.
                num_offspring = min(num_offspring, len(self.qpop_net.current_pop))
                best_current_pop = self.qpop_net.current_pop[:num_offspring]
                new_pop_net[:num_offspring] = self.qpop_net.apply_crossover(
                    best_current_pop, new_pop_net[:num_offspring]
                )

        return new_pop_params, new_pop_net, new_ancestor

    def _record_stage_evaluations(self, pop_net, fitnesses):
        """ Keep the current stage's evaluations in memory for the criteria/screening that
            learn from them ('empirical' / 'surrogate' pruning, surrogate screening). """
        if self.prune_criterion in ('empirical', 'surrogate') or self.surrogate_enabled:
            self._stage_eval_net.append(np.array(pop_net))
            self._stage_eval_fit.append(np.array(fitnesses, dtype=np.float64))

    def _stage_training_data(self):
        """ (net, fitness) of every finite evaluation of the current stage: the in-memory
            record, or cache.json after a resume. (None, None) if there is none. """
        if self._stage_eval_net:
            net = np.concatenate(self._stage_eval_net)
            fit = np.concatenate(self._stage_eval_fit)
        else:
            net, fit = self._stage_evaluations_from_cache()
        if net is None:
            return None, None
        finite = np.isfinite(fit)
        return net[finite], fit[finite]

    def _gens_in_stage(self):
        return max(0, self.current_gen - self._stage_start_gen)

    def _sampling_tau(self):
        """ D3: temperature of this generation's PMF sampling (1.0 = unchanged PMF). """
        if self.sampling_tau0 is None:
            return 1.0
        return max(self.sampling_tau_min,
                   self.sampling_tau0 / (1.0 + self._gens_in_stage() / self.sampling_tau_decay_gens))

    def _surrogate_tau(self):
        return max(self.surrogate_tau_min,
                   self.surrogate_tau0 / (1.0 + self._gens_in_stage() / self.surrogate_tau_decay_gens))

    def _generate_screened(self):
        """ Plan v2, D1 + D2: build a candidate pool by repeating the original sampling,
            keep n_eval of them (surrogate screening, once the stage has
            surrogate_warmup_evals evaluations) and evaluate - in two fidelity rungs when
            fidelity is enabled. Returns (params, net, ancestor, fitness, raw_fitness) of
            the individuals that get a final fitness. """
        num_classic = self.qpop_net.num_ind * self.qpop_net.repetition
        n_eval = (self.surrogate_n_eval or num_classic) if self.surrogate_enabled else num_classic
        n_rung = n_eval * (self.fidelity_oversample if self.fidelity_enabled else 1)

        train_net = train_fit = None
        screening = False
        if self.surrogate_enabled:
            train_net, train_fit = self._stage_training_data()
            screening = train_net is not None and train_net.shape[0] >= self.surrogate_warmup_evals
        pool_size = n_rung * (self.surrogate_oversample if screening else 1)

        blocks = [self._sample_block() for _ in range(int(np.ceil(pool_size / num_classic)))]
        params = np.concatenate([b[0] for b in blocks])
        net = np.concatenate([b[1] for b in blocks])
        anc = np.concatenate([b[2] for b in blocks])
        params, net, anc = self._dedupe_pool(params, net, anc, train_net, pool_size)

        if screening and net.shape[0] > n_rung:
            tau = self._surrogate_tau()
            idx, _ = surrogate.screen(
                net, self.qpop_net.chromosome.num_functions, train_net, train_fit, n_rung,
                tau, self.surrogate_beta, self.surrogate_random_frac, self.surrogate_rng,
                num_models=self.surrogate_ensemble, ridge_lambda=self.surrogate_ridge_lambda)
            idx = np.sort(idx)
            self.last_screening = (int(net.shape[0]), int(idx.shape[0]), tau)
        else:
            idx = np.arange(min(n_rung, net.shape[0]))
            self.last_screening = None
        params, net, anc = params[idx], net[idx], anc[idx]

        self.logger.info("new population created =%s", net)
        if not self.fidelity_enabled:
            fit, raw = self.eval_pop(params, net)
            return params, net, anc, fit, raw

        # Cheap rung for everyone (folders gen_<offset + i>, cleaned at the end of the
        # generation), then the best n_eval resume from their checkpoint to the full budget
        # in folders gen_0..gen_<n_eval - 1>, so best_so_far_id names a real folder.
        low_fit, _ = self.eval_pop(params, net, fidelity_rung='low',
                                   fidelity_epochs=self.fidelity_low_epochs,
                                   id_offset=self.FIDELITY_ID_OFFSET)
        order = np.argsort(-np.nan_to_num(low_fit, nan=-np.inf), kind='stable')
        keep = np.sort(order[:n_eval])
        self.logger.info(f"Fidelity: {net.shape[0]} individuals trained for "
                         f"{self.fidelity_low_epochs} epochs, {keep.shape[0]} continue.")
        fit, raw = self.eval_pop(params[keep], net[keep], fidelity_rung='resume',
                                 resume_ids=[self.FIDELITY_ID_OFFSET + int(i) for i in keep])
        return params[keep], net[keep], anc[keep], fit, raw

    # Folder offset of the cheap fidelity rung (gen_1000, gen_1001, ...).
    FIDELITY_ID_OFFSET = 1000

    @staticmethod
    def _dedupe_pool(params, net, anc, seen_net, needed):
        """ Order the pool so architectures that are new to the stage come first, then
            repeats (of the pool itself or of the stage's evaluations); repeats are dropped
            when there are at least *needed* new ones. """
        seen = set() if seen_net is None else {tuple(r) for r in seen_net}
        in_pool = set()
        fresh, repeated = [], []
        for i, row in enumerate(map(tuple, net)):
            (repeated if row in in_pool or row in seen else fresh).append(i)
            in_pool.add(row)
        order = np.array(fresh if len(fresh) >= needed else fresh + repeated, dtype=np.int64)
        return params[order], net[order], anc[order]

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

    def eval_pop(self, pop_params, pop_net, **fidelity_kwargs):
        """ Decode and evaluate a population of networks and hyperparameters.

        Args:
            pop_params: float numpy array with a classic population of hyperparameters.
            pop_net: int ndarray of operation indices.
            fidelity_kwargs: two-rung evaluation only (plan v2, D2) - fidelity_rung
                ('low' or 'resume'), fidelity_epochs, id_offset, resume_ids; forwarded to
                eval_func. Not passed at all on the original path.

        Returns:
            (penalized_fitnesses, raw_fitnesses)
        """
        decoded_params, decoded_nets = self.decode_pop(pop_params, pop_net)

        self.logger.info('Evaluating new population ...')
        fitnesses, cache_hits = self.eval_func(
            decoded_params, decoded_nets,
            generation=self.current_gen,
            **fidelity_kwargs,
        )
        # "New architecture discovered" = did NOT hit the fitness cache this
        # generation, i.e. it was actually trained (see architecture_cache.py).
        self.new_architectures_count = int(np.sum(~cache_hits))
        penalized_fitnesses = np.copy(fitnesses)

        if self.penalize_number:
            penalties = self.get_penalties(pop_net)
            penalized_fitnesses -= penalties

        if self.selection_capacity_lambda:
            # Plan v2, D2: the retrain rewards capacity that the short search proxy does
            # not see - bias selection (never the reported raw fitness) toward bigger nets.
            params_m = getattr(self.eval_func, 'last_params_m', None)
            if params_m is None or len(params_m) != len(penalized_fitnesses):
                raise RuntimeError("selection_capacity_lambda needs an evaluator that "
                                   "exposes last_params_m (EvalPopulation does).")
            penalized_fitnesses += self.selection_capacity_lambda * np.log(
                np.clip(np.nan_to_num(np.asarray(params_m, dtype=np.float64), nan=1e-3),
                        1e-3, None))

        # Update the total evaluation counter
        self.total_eval = self.total_eval + np.size(pop_params, axis=0)

        return penalized_fitnesses, fitnesses

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
        fn_list = self.qpop_net.chromosome.fn_list

        for i, net in enumerate(pop_net):
            # Gene value meaning is per-node (fn_list[node]), so a gene must be matched
            # against the reducing-ops names node-by-node, not compared network-wide by
            # raw integer value - the same integer can be a different op at different
            # node positions once fn_list is per-node (progressive stages).
            reducing_fns_count = sum(
                1 for node, gene in enumerate(net)
                if gene >= 0 and fn_list[node][gene] in self.reducing_fns_names
            )
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
        elif self.progressive_mode == 'dynamic':
            streak = self._stable_streak if self.global_op_pruning else min(self._stable_streak)
            stage_line = (
                f'- Dynamic progressive: num_nodes={self.qpop_net.chromosome.num_genes}, '
                f'stable_streak={streak}/{self.dynamic_growth_patience}\n'
            )

        entropy, kl = self._pmf_stats()
        v2_lines = ''
        if self.qpop_net.sampling_tau != 1.0:
            v2_lines += f'- Sampling temperature: {self.qpop_net.sampling_tau:.4f}\n'
        if self.last_screening is not None:
            pool, kept, tau = self.last_screening
            v2_lines += f'- Surrogate screening: {kept}/{pool} candidates evaluated (tau={tau:.4f})\n'
        gpu_seconds = getattr(self.eval_func, 'gpu_seconds_total', None)
        if gpu_seconds:
            v2_lines += f'- GPU time so far: {gpu_seconds / 3600:.2f} h\n'
        self.logger.info(f'New generation finished running!\n'
                        f'- Generation: {self.current_gen}\n'
                        f'{stage_line}'
                        f'- New architectures discovered: {self.new_architectures_count}\n'
                        f'- PMF: normalized entropy={entropy:.4f}, '
                        f'KL to stage start={kl:.4f}\n'
                        f'{v2_lines}'
                        f'- Best so far: {self.best_so_far_id} --> {self.best_so_far:.5f}\n'
                        f'- Fitnesses: {self.fitnesses}\n'
                        f'- Fitnesses without penalties: {self.raw_fitnesses}\n')

    def save_data(self):
        """ Save this generation's QNAS data (chromosomes, generation number, evaluation
            score, number of evaluations, ...) as one incremental record appended to
            self.data_file - previous generations' records are never read back or rewritten
            (see append_pkl_entry / util.load_pkl, which reassembles the full per-generation
            history on read by merging every appended record).
        """

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
                 'net_pop': self.qpop_net.current_pop,
                 # Lineage of current_pop (row-aligned): which quantum individual produced
                 # each classical individual and for how many generations it survived.
                 # Used by the ancestor_decay engine and restored on resume.
                 'classic_age': self.classic_age,
                 'classic_ancestor': self.classic_ancestor,
                 # Per-node mean PMF at the start of the current stage ('lift' pruning
                 # reference, restored on resume) and how far the PMF has moved.
                 'stage_start_probs': self._stage_start_probs,
                 # Plan v2 state (absent from older checkpoints; read back with .get()).
                 'stage_start_gen': self._stage_start_gen,
                 'niche_best': self.niche_best,
                 'niche_last_improve': self.niche_last_improve,
                 'sampling_tau': self.qpop_net.sampling_tau,
                 'screening': self.last_screening,
                 'gpu_seconds': getattr(self.eval_func, 'gpu_seconds_total', None)}
        entry['pmf_entropy'], entry['pmf_kl_stage_start'] = self._pmf_stats()

        if self.progressive_stages:
            entry['current_stage_idx'] = self.current_stage_idx
            # List of per-node op-name lists (ragged) - deep-copied one level deeper
            # than a flat list so later in-place mutation can't corrupt this entry.
            entry['fn_list'] = [list(names) for names in self.qpop_net.chromosome.fn_list]
        elif self.progressive_mode == 'dynamic':
            entry['progressive_mode'] = 'dynamic'
            entry['fn_list'] = [list(names) for names in self.qpop_net.chromosome.fn_list]
            entry['dynamic_stable_streak'] = (
                self._stable_streak if self.global_op_pruning else list(self._stable_streak)
            )

        self.append_pkl_entry(self.current_gen, entry)

    @staticmethod
    def _sidecar_path(data_file):
        """ Path to the small 'latest generation only' companion file kept next to
        *data_file* - see append_pkl_entry/load_qnas_data.
        """

        root, ext = os.path.splitext(data_file)
        return f'{root}_latest{ext}'

    def append_pkl_entry(self, generation, entry):
        """ Appends one generation's *entry* to *self.data_file* without touching
        whatever earlier generations' records are already there. Also overwrites a
        small sidecar file with just this record, so a later resume (load_qnas_data)
        can read the latest generation in O(1) instead of replaying the whole history
        through util.load_pkl.

        Args:
            generation: (int) generation number this entry belongs to.
            entry: dict of this generation's data.
        """

        with open(self.data_file, 'ab') as f:
            dump({generation: entry}, f, protocol=HIGHEST_PROTOCOL)

        with open(self._sidecar_path(self.data_file), 'wb') as f:
            dump({generation: entry}, f, protocol=HIGHEST_PROTOCOL)

    def dump_pkl_data(self, new_data):
        """ Overwrites *self.data_file* with *new_data* in one shot - used to seed a new
        data file with a previously resumed run's full history (load_qnas_data) or to
        write a complete, already-assembled dict; not part of the per-generation path
        (see append_pkl_entry/save_data), so this does not reintroduce a full
        read-modify-write on every generation.

        Args:
            new_data: dict containing data to save.
        """

        with open(self.data_file, 'wb') as f:
            dump(new_data, f, protocol=HIGHEST_PROTOCOL)

        # This overwrite makes *new_data* the sole authority on data_file's content -
        # drop the sidecar (rather than risk it going stale) so load_qnas_data falls
        # back to reading data_file itself next time.
        sidecar_path = self._sidecar_path(self.data_file)
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)

    def load_qnas_data(self, file_path):
        """ Read pkl data in *file_path* and load its information to current QNAS. It also saves
            its info into the new pkl data file *self.data_file*.

            Only the latest generation's record is actually needed here (unlike
            generate_infographic.py, which needs every generation) - tries the small
            '_latest' sidecar first (a single O(1) record) instead of replaying the
            whole history through util.load_pkl. Falls back to the full merge for
            experiment directories written before this sidecar existed.

        Args:
            file_path: (str) path to the pkl data file.
        """

        sidecar_path = self._sidecar_path(file_path)
        log_data = None
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, 'rb') as f:
                    sidecar_record = load(f)
                generation = max(sidecar_record.keys())
                log_data = sidecar_record[generation]
            except (EOFError, UnpicklingError, OSError):
                log_data = None

        if log_data is not None:
            if not os.path.exists(self.data_file):
                # Same on-disk format either way (an appendable pickle stream), so a
                # raw byte copy preserves the whole history for generate_infographic.py
                # without deserializing/reserializing any of it.
                shutil.copyfile(file_path, self.data_file)
                shutil.copyfile(sidecar_path, self._sidecar_path(self.data_file))
        else:
            full_data = load_pkl(file_path)
            generation = max(full_data.keys())
            log_data = full_data[generation]

            if not os.path.exists(self.data_file):
                self.dump_pkl_data(full_data)

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

        # Checkpoints written before lineage was saved have no such keys: leave them
        # None and replace_pop() re-seeds them positionally, as resume always did.
        self.classic_age = log_data.get('classic_age')
        self.classic_ancestor = log_data.get('classic_ancestor')

        if self.progressive_stages and 'current_stage_idx' in log_data:
            fn_list = log_data['fn_list']
            if not fn_list or isinstance(fn_list[0], str):
                raise RuntimeError(
                    "Checkpoint predates the per-node progressive-pruning refactor "
                    "(fn_list is a flat list, not one list per node) - re-run from "
                    "scratch instead of resuming with progressive_stages configured."
                )
            self.current_stage_idx = log_data['current_stage_idx']
            self.qpop_net.chromosome.fn_list = fn_list
            self.qpop_net.chromosome.num_functions = [len(names) for names in fn_list]
            self.eval_func.fn_list = fn_list
        elif self.progressive_mode == 'dynamic' and 'fn_list' in log_data:
            fn_list = log_data['fn_list']
            if not fn_list or isinstance(fn_list[0], str):
                raise RuntimeError(
                    "Checkpoint predates the per-node progressive-pruning refactor "
                    "(fn_list is a flat list, not one list per node) - re-run from "
                    "scratch instead of resuming with progressive dynamic mode "
                    "configured."
                )
            self.qpop_net.chromosome.fn_list = fn_list
            self.qpop_net.chromosome.num_functions = [len(names) for names in fn_list]
            self.eval_func.fn_list = fn_list
            self._stable_streak = log_data.get('dynamic_stable_streak')
            if self._stable_streak is None:
                self._init_stability_streaks()

        # Checkpoints written before the stage-start PMF was recorded get a rebuilt one.
        # The stage's in-memory evaluation record is not checkpointed: 'empirical'
        # pruning falls back to cache.json after a resume.
        stage_start_probs = log_data.get('stage_start_probs')
        if stage_start_probs is None or len(stage_start_probs) != len(self.qpop_net.probabilities):
            stage_start_probs = self._reconstruct_stage_start_probs()
        self._stage_start_probs = stage_start_probs
        self._stage_eval_net = []
        self._stage_eval_fit = []

        # Plan v2 state. Checkpoints that predate it: the stage's first generation comes from
        # the stage list (deterministic) or is unknown (0); niche records restart empty.
        stage_start_gen = log_data.get('stage_start_gen')
        if stage_start_gen is None:
            stage_start_gen = (self.progressive_stages[self.current_stage_idx]['gen_start']
                               if self.progressive_stages else 0)
        self._stage_start_gen = stage_start_gen
        if log_data.get('niche_best') is not None:
            self.niche_best = np.array(log_data['niche_best'], dtype=np.float64)
            self.niche_last_improve = np.array(log_data['niche_last_improve'], dtype=np.int64)
        gpu_seconds = log_data.get('gpu_seconds')
        if gpu_seconds is not None and hasattr(self.eval_func, 'gpu_seconds_total'):
            self.eval_func.gpu_seconds_total = gpu_seconds

    def check_early_stopping(self):
        """
        Compute the early stopping of the evolution. If the best fitness does not improve 
        by at least 0.005 (0.5%) for `patience` generations, the evolution stops.
        """
        # last_best_so_far is None on the first check of a process - e.g. right after
        # resuming from a checkpoint at current_gen > 1 - so that check only sets the
        # baseline instead of comparing against a value that was never recorded.
        if self.current_gen > 1 and self.last_best_so_far is not None:
            if self.last_best_so_far > 0:
                improvement = (self.best_so_far - self.last_best_so_far) / self.last_best_so_far
            else:
                improvement = np.inf if self.best_so_far > self.last_best_so_far else 0.0
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
        """ Update quantum populations of networks and hyperparameters.

            The hyperparameter population always uses its single update rule. The
            network (architecture) population uses whichever rule
            *self.quantum_update_engine* selects:
              - 'default': rotate quantum individual i toward whichever classical
                individual currently sits at rank i in current_pop (top num_ind,
                sorted by fitness) - fires unconditionally every update_quantum_gen
                generations regardless of whether that ranking actually changed.
              - 'ancestor_decay': rotate each quantum individual toward every
                classical individual it actually produced (tracked lineage), with
                the nudge intensity decaying by how long that classical individual
                has survived, so a stale elite stops forcing convergence over time.
        """

        if np.remainder(self.current_gen,
                        self.update_quantum_gen) == 0 and self.current_gen > 0:

            self.qpop_params.update_quantum(intensity=self.random)

            if self.quantum_lr is not None:
                self.qpop_net.update_quantum_ranked(
                    lr=self.quantum_lr,
                    targets=self._ranked_update_targets(),
                    negative_lr=self.quantum_negative_lr,
                    prob_floor=self.quantum_prob_floor,
                )
            elif self.quantum_update_engine == 'ancestor_decay':
                self.qpop_net.update_quantum_ancestor_decay(
                    intensity=self.random,
                    ages=self.classic_age,
                    ancestors=self.classic_ancestor,
                    decay_rate=self.quantum_update_age_decay,
                )
            else:
                self.qpop_net.update_quantum(intensity=self.random)
    
    RANK_WEIGHTINGS = ('uniform', 'linear', 'nes')
    PRUNE_CRITERIA = ('pmf', 'lift', 'empirical', 'surrogate')
    # Pseudo-count pulling an op's empirical score toward the neutral 0.5 percentile, so
    # an op seen in only a couple of individuals can't win (or lose) the prune on luck.
    EMPIRICAL_PRIOR_COUNT = 5

    def _set_update_and_prune_options(self, quantum_lr, rank_weighting, top_k, negative_lr,
                                      prob_floor, max_prob, prune_criterion,
                                      progressive_mode):
        """ Validate and store the rank-based update / prune-criterion options. """
        if quantum_lr is None:
            if (rank_weighting, top_k, negative_lr, prob_floor) != ('uniform', 1, 0.0, 0.0):
                raise ValueError(
                    "quantum_rank_weighting, quantum_top_k, quantum_negative_lr and "
                    "quantum_prob_floor only apply to the rank-based update - set "
                    "quantum_lr to enable it."
                )
        elif not 0.0 < quantum_lr <= 1.0:
            raise ValueError("quantum_lr must be in (0, 1] (or None for the original rules).")
        if rank_weighting not in self.RANK_WEIGHTINGS:
            raise ValueError(f"quantum_rank_weighting must be one of {self.RANK_WEIGHTINGS}, "
                             f"got {rank_weighting!r}")
        if top_k < 1:
            raise ValueError("quantum_top_k must be >= 1.")
        if not 0.0 <= negative_lr < 1.0:
            raise ValueError("quantum_negative_lr must be in [0, 1).")
        if not 0.0 <= prob_floor < 1.0:
            raise ValueError("quantum_prob_floor must be in [0, 1).")
        if not 0.0 < max_prob <= 1.0:
            raise ValueError("quantum_max_prob must be in (0, 1].")
        if prune_criterion not in self.PRUNE_CRITERIA:
            raise ValueError(f"prune_criterion must be one of {self.PRUNE_CRITERIA}, "
                             f"got {prune_criterion!r}")
        if prune_criterion != 'pmf' and progressive_mode == 'dynamic':
            raise ValueError(
                "prune_criterion 'lift'/'empirical'/'surrogate' is only supported in deterministic "
                "progressive mode - dynamic nucleus pruning thresholds cumulative "
                "probability mass, which these scores are not."
            )

        self.quantum_lr = quantum_lr
        self.quantum_rank_weighting = rank_weighting
        self.quantum_top_k = top_k
        self.quantum_negative_lr = negative_lr
        self.quantum_prob_floor = prob_floor
        self.prune_criterion = prune_criterion

    def _rank_weights(self, k):
        """ Weights (summing to 1) of a quantum individual's k best targets, best first. """
        j = np.arange(k)
        if self.quantum_rank_weighting == 'linear':
            weights = (k - j).astype(np.float64)
        elif self.quantum_rank_weighting == 'nes':
            weights = np.maximum(0.0, np.log(k / 2.0 + 1.0) - np.log(j + 1.0))
        else:
            weights = np.ones(k, dtype=np.float64)
        return weights / weights.sum()

    def _ranked_update_targets(self):
        """ Targets of each quantum individual for QPopulationNetwork.update_quantum_ranked,
            from current_pop (sorted best first):
              - 'ancestor_decay': the quantum_top_k best rows of its own lineage, weighted
                by rank and by exp(-quantum_update_age_decay * age); pushed away from the
                lineage's worst row when that row is not a target;
              - 'default': ranks q, q + num_ind, ... (q's rank, as in the original rule,
                plus further ranks when quantum_top_k > 1); pushed away from rank n-1-q.
        """
        pop_size = self.qpop_net.current_pop.shape[0]
        num_ind = self.qpop_net.num_ind
        targets = [None] * num_ind

        for q in range(num_ind):
            if self.quantum_update_engine == 'ancestor_decay':
                lineage = np.where(self.classic_ancestor == q)[0]
                if lineage.size == 0:
                    continue
                rows = lineage[:self.quantum_top_k]
                weights = self._rank_weights(rows.size) * np.exp(
                    -self.quantum_update_age_decay * self.classic_age[rows])
                worst = lineage[-1] if lineage.size > rows.size else None
            else:
                ranks = np.arange(q, pop_size, num_ind)
                if ranks.size == 0:
                    continue
                rows = ranks[:self.quantum_top_k]
                weights = self._rank_weights(rows.size)
                worst = pop_size - 1 - q
                if worst < 0 or worst in rows:
                    worst = None
            targets[q] = (rows, weights, worst)

        return targets

    def _snapshot_stage_start_probs(self):
        """ Record the per-node mean PMF at the start of a stage (after initialization or
            an op-list change) and restart the stage's evaluation record. """
        self._stage_start_probs = [p.mean(axis=0).copy() for p in self.qpop_net.probabilities]
        self._stage_eval_net = []
        self._stage_eval_fit = []
        self._stage_start_gen = self.current_gen

    def _reconstruct_stage_start_probs(self):
        """ Best guess of the stage-start PMF for a checkpoint that predates its recording:
            the configured prior in stage 0, uniform over each node's menu afterwards (exact
            under reset_probs_on_stage_change, an approximation under carry-over). """
        if self.current_stage_idx == 0 and not self.progressive_mode == 'dynamic':
            prior = np.asarray(self.qpop_net.initial_probs, dtype=np.float64)
            if all(p.shape[1] == prior.shape[0] for p in self.qpop_net.probabilities):
                return [prior.copy() for _ in self.qpop_net.probabilities]
        if not self.reset_probs_on_stage_change and self.prune_criterion == 'lift':
            self.logger.warning(
                "Checkpoint has no stage-start PMF: 'lift' pruning uses a uniform reference "
                "for this stage (exact only with reset_probs_on_stage_change)."
            )
        return [np.full(p.shape[1], 1.0 / p.shape[1]) for p in self.qpop_net.probabilities]

    def _pmf_stats(self):
        """ (mean normalized entropy, mean KL to the stage-start PMF) of the network PMF. """
        entropies, kls = [], []
        refs = self._stage_start_probs or [None] * len(self.qpop_net.probabilities)
        for probs, ref in zip(self.qpop_net.probabilities, refs):
            safe = np.clip(probs, 1e-12, None)
            n_ops = probs.shape[1]
            entropies.append(0.0 if n_ops < 2 else
                             float(np.mean(-(probs * np.log(safe)).sum(axis=1) / np.log(n_ops))))
            if ref is not None and ref.shape[0] == n_ops:
                kl = (probs * (np.log(safe) - np.log(np.clip(ref, 1e-12, None)))).sum(axis=1)
                kls.append(float(np.mean(kl)))
        return float(np.mean(entropies)), (float(np.mean(kls)) if kls else float('nan'))

    def _op_scores(self):
        """ Per-node op scores for progressive pruning, aligned with each node's fn_list,
            or None for the original 'pmf' criterion (the caller keeps its own mean). """
        if self.prune_criterion == 'pmf':
            return None
        if self.prune_criterion == 'surrogate':
            scores = self._surrogate_op_scores()
            if scores is not None:
                return scores
            self.logger.warning(
                "Op pruning: too few evaluations of this stage for the surrogate - ranking by 'lift'."
            )
        if self.prune_criterion == 'empirical':
            scores = self._empirical_op_scores()
            if scores is not None:
                return scores
            self.logger.warning(
                "Op pruning: no evaluation recorded for this stage - ranking by 'lift'."
            )
        refs = self._stage_start_probs
        if refs is None or len(refs) != len(self.qpop_net.probabilities):
            refs = self._reconstruct_stage_start_probs()
        return [probs.mean(axis=0) / np.clip(ref, 1e-12, None)
                for probs, ref in zip(self.qpop_net.probabilities, refs)]

    def _empirical_op_scores(self):
        """ Score of op k at node i = mean fitness percentile (0 worst .. 1 best) of the
            individuals evaluated in this stage that used k at i, shrunk toward 0.5 by
            EMPIRICAL_PRIOR_COUNT pseudo-observations. Uses the in-memory record of the
            stage; after a resume (record lost) falls back to the run's cache.json. """
        if self._stage_eval_net:
            net = np.concatenate(self._stage_eval_net)
            fit = np.concatenate(self._stage_eval_fit)
        else:
            net, fit = self._stage_evaluations_from_cache()
        if net is None or net.shape[0] == 0:
            return None

        # A diverged individual's fitness is NaN (in-memory) or None, normalized to NaN
        # by _stage_evaluations_from_cache's np.float64 cast (cache fallback) - argsort
        # puts NaN last, i.e. ranked as the single BEST evaluation, so it must be dropped
        # rather than scored.
        finite = np.isfinite(fit)
        if not finite.all():
            self.logger.warning(
                "Op pruning: dropping %d diverged (non-finite fitness) evaluation(s) out "
                "of %d from the empirical op-score calculation.",
                int((~finite).sum()), fit.shape[0],
            )
            net, fit = net[finite], fit[finite]
        if fit.shape[0] == 0:
            return None

        order = np.argsort(fit, kind='stable')
        ranks = np.empty(fit.shape[0], dtype=np.float64)
        ranks[order] = np.arange(fit.shape[0])
        _, tie_group = np.unique(fit, return_inverse=True)
        ranks = (np.bincount(tie_group, weights=ranks) / np.bincount(tie_group))[tie_group]
        pct = ranks / (fit.shape[0] - 1) if fit.shape[0] > 1 else np.full(1, 0.5)

        prior = self.EMPIRICAL_PRIOR_COUNT
        scores = []
        for node, names in enumerate(self.qpop_net.chromosome.fn_list):
            genes = net[:, node]
            valid = genes >= 0
            sums = np.bincount(genes[valid], weights=pct[valid], minlength=len(names))
            counts = np.bincount(genes[valid], minlength=len(names))
            scores.append((sums + 0.5 * prior) / (counts + prior))
        return scores

    def _surrogate_op_scores(self):
        """ Score of op k at node i = its coefficient in a ridge-ensemble fit of the stage's
            evaluations (plan v2, D1/D3): the op's marginal effect on fitness with the other
            nodes accounted for, unlike 'empirical', which averages over whatever those other
            nodes happened to be. None with fewer than 2 finite evaluations. """
        net, fit = self._stage_training_data()
        if net is None or net.shape[0] < 2:
            return None
        num_functions = self.qpop_net.chromosome.num_functions
        model = surrogate.RidgeEnsembleSurrogate(
            num_models=self.surrogate_ensemble if self.surrogate_enabled else 5,
            ridge_lambda=self.surrogate_ridge_lambda if self.surrogate_enabled else 3.0,
            rng=np.random.default_rng(0))
        model.fit(surrogate.one_hot(net, num_functions), fit)
        if not model.fitted:
            return None
        return model.op_effects(num_functions)

    def _stage_evaluations_from_cache(self):
        """ (net, fitness) of every cached architecture that fits the current depth and
            per-node op menus - the current stage's evaluations when resuming. Cached
            fitness is the raw (unpenalized) one. Returns (None, None) without a cache. """
        cache = getattr(self.eval_func, 'architecture_cache', None)
        if cache is None:
            return None, None
        index = {}
        cache._locked_read_modify_write(index.update)

        fn_list = self.qpop_net.chromosome.fn_list
        positions = [{name: k for k, name in enumerate(names)} for names in fn_list]
        rows, fits = [], []
        for key, entry in index.items():
            ops = key.split('|')
            if len(ops) != len(fn_list) or not all(op in pos for op, pos in zip(ops, positions)):
                continue
            rows.append([pos[op] for op, pos in zip(ops, positions)])
            fits.append(entry['fitness'])
        if not rows:
            return None, None
        self.logger.info(f"Op pruning: using {len(rows)} cached evaluations of this stage.")
        return np.array(rows, dtype=np.int64), np.array(fits, dtype=np.float64)

    def go_next_gen(self):
        """ Go to the next generation --> update quantum genes, log data, delete unnecessary
            training files and update generation counter.
        """

        self.update_quantum()
        self._maybe_restart_niches()

        self.save_data()
        self.log_data()
        #self.save_train_data()

        # Remove Tensorflow models files
        delete_old_dirs(self.experiment_path, keep_best=True,
                        best_id=f'{self.best_so_far_id[0]}_{self.best_so_far_id[1]}')
        self.current_gen += 1

    def _rank_and_prune_ops(self, names, mean_weight, num_ops):
        """Rank one node's (or the whole network's, in global mode) op list by
        *mean_weight* and keep the top *num_ops*.

        *self.noop_fn_name* is never eligible for pruning: it's excluded from the
        ranking pool and unconditionally re-added (spending one of *num_ops* slots),
        so it's kept forever. This guarantees a no-op is always available to extend
        surviving classical individuals into newly grown nodes without invalidating
        their fitness (see QPopulationNetwork.filter_and_remap_classical).

        Args:
            names: (list[str]) current op-name list to prune.
            mean_weight: 1D array, mean quantum probability mass per op in *names*.
            num_ops: (int) number of ops to keep.

        Returns:
            list[str]: the pruned op-name list.
        """
        if num_ops >= len(names):
            return list(names)

        rankable = [i for i, n in enumerate(names) if n != self.noop_fn_name]
        ranked = sorted(rankable, key=lambda i: (mean_weight[i], -i), reverse=True)
        kept = set(ranked[:max(num_ops - 1, 0)])
        if self.noop_fn_name in names:
            kept.add(names.index(self.noop_fn_name))
        kept = sorted(kept)
        return [names[i] for i in kept]

    def _rank_and_prune_all_nodes(self, old_fn_list, num_ops):
        """Rank each node's ops independently by that node's own mean quantum
        probability mass (across quantum individuals only) and keep the top *num_ops*
        for that node. Different nodes can end up keeping different op subsets.
        Used when *self.global_op_pruning* is False (the default).

        Args:
            old_fn_list: list of length num_genes, each entry that node's current
                op-name list.
            num_ops: (int) number of ops to keep per node for the new stage.

        Returns:
            list of length num_genes, each entry that node's pruned op-name list.
        """
        scores = self._op_scores()
        label = 'PMF' if scores is None else self.prune_criterion
        new_fn_list = []
        for node_idx, names in enumerate(old_fn_list):
            if scores is None:
                mean_weight = self.qpop_net.probabilities[node_idx].mean(axis=0)
            else:
                mean_weight = scores[node_idx]
            node_new_fn_list = self._rank_and_prune_ops(names, mean_weight, num_ops)
            self.logger.info(
                f"Op pruning ({label}-ranked), node {node_idx}: {names} -> {node_new_fn_list}"
            )
            new_fn_list.append(node_new_fn_list)

        return new_fn_list

    def _rank_and_prune_globally(self, old_fn_list, num_ops):
        """Rank ops ONCE using the mean quantum probability mass across every node
        AND every quantum individual, and apply that single surviving op list
        uniformly to every node - i.e. the pre-per-node-pruning behavior. Used when
        *self.global_op_pruning* is True.

        Requires every node to currently share the same op list (true as long as
        global_op_pruning has been used consistently since stage 0 - per-node
        pruning can make nodes diverge, at which point a network-wide mean is no
        longer well-defined across differing op sets).

        Args:
            old_fn_list: list of length num_genes, each entry that node's current
                op-name list (all identical).
            num_ops: (int) number of ops to keep, network-wide.

        Returns:
            list of length num_genes, each entry the SAME pruned op-name list.
        """
        names = old_fn_list[0]
        if any(node_names != names for node_names in old_fn_list):
            raise ValueError(
                "global_op_pruning requires every node to share the same op list - "
                "nodes have already diverged (was global_op_pruning enabled from "
                "stage 0 onward?)."
            )

        scores = self._op_scores()
        if scores is None:
            mean_weight = np.stack(self.qpop_net.probabilities, axis=1).mean(axis=(0, 1))
        else:
            mean_weight = np.mean(scores, axis=0)
        label = 'PMF' if scores is None else self.prune_criterion
        new_fn_list = self._rank_and_prune_ops(names, mean_weight, num_ops)
        self.logger.info(f"Op pruning ({label}-ranked, global): {names} -> {new_fn_list}")

        return [list(new_fn_list) for _ in old_fn_list]

    def _apply_fn_list_change(self, new_fn_list, new_num_nodes):
        """Apply an op-list change (and, if new_num_nodes > current, depth growth) to
        the quantum PMF, then filter+remap the classical population against it instead
        of wiping it: individuals whose ops all survived keep their evaluated fitness;
        newly grown node positions are set to a no-op so the decoded network - and its
        fitness - doesn't change just because the population grew deeper. Shared by
        deterministic stage transitions (_transition_stage) and dynamic prune/growth
        steps (_dynamic_prune_step).

        Args:
            new_fn_list: list of length *current* num_genes (before this call), each
                entry that (existing) node's new op-name list.
            new_num_nodes: new depth (>= current depth).
        """
        old_fn_list = [list(names) for names in self.qpop_net.chromosome.fn_list]

        self.qpop_net.grow_and_prune_discrete(
            new_num_nodes, new_fn_list, reset_probs=self.reset_probs_on_stage_change,
        )

        if self.qpop_net.current_pop is not None:
            kept_mask, remapped_net = self.qpop_net.filter_and_remap_classical(
                self.qpop_net.current_pop, old_fn_list,
                self.qpop_net.chromosome.fn_list, self.noop_fn_name,
            )
            if kept_mask.any():
                self.qpop_net.current_pop = remapped_net
                self.qpop_params.current_pop = self.qpop_params.current_pop[kept_mask]
                self.fitnesses = self.fitnesses[kept_mask]
                self.raw_fitnesses = self.raw_fitnesses[kept_mask]
                self.logger.info(
                    f"Op-list change: kept {int(kept_mask.sum())}/{len(kept_mask)} "
                    f"classical individuals across the transition."
                )
            else:
                self.qpop_net.current_pop = None
                self.logger.warning(
                    "Op-list change: no classical individual survived the op "
                    "prune - starting the next generation from scratch."
                )

        self.eval_func.fn_list = self.qpop_net.chromosome.fn_list
        self._just_transitioned = True
        self._snapshot_stage_start_probs()

    def _transition_stage(self, new_stage_idx):
        """Grow depth / prune ops at a progressive-stage boundary (deterministic mode
        only), preserving the classical population's already-evaluated fitness
        wherever possible instead of discarding it.
        """
        stage = self.progressive_stages[new_stage_idx]
        old_num_nodes = self.qpop_net.chromosome.num_genes
        old_fn_list = [list(names) for names in self.qpop_net.chromosome.fn_list]

        if self.global_op_pruning:
            new_fn_list = self._rank_and_prune_globally(old_fn_list, stage['num_ops'])
        else:
            new_fn_list = self._rank_and_prune_all_nodes(old_fn_list, stage['num_ops'])

        self._apply_fn_list_change(new_fn_list, stage['num_nodes'])

        self.current_stage_idx = new_stage_idx
        self.logger.info(
            f"Progressive stage transition -> stage {new_stage_idx} "
            f"(gen_start={stage['gen_start']}, num_nodes={stage['num_nodes']}, "
            f"num_ops={stage['num_ops']}): "
            f"{old_num_nodes} -> {stage['num_nodes']} nodes."
        )

    def _nucleus_prune_ops(self, names, mean_weight, threshold, min_ops, flatness_epsilon):
        """Dynamic-mode counterpart to _rank_and_prune_ops: keep the smallest prefix of
        *names* (ranked descending by mean_weight, NoOp excluded from ranking and
        always force-kept, mirroring _rank_and_prune_ops) whose cumulative share of
        mean_weight is >= threshold. Floors the total kept count (NoOp included) at
        min_ops.

        Flatness guard: if the spread (max - min) across the rankable (non-NoOp) ops'
        mean_weight is <= flatness_epsilon, the distribution hasn't differentiated
        enough to prune yet - returns *names* unchanged instead of cutting to whatever
        prefix the nucleus walk happens to identify.

        Args:
            names: (list[str]) current op-name list to prune.
            mean_weight: 1D array, mean quantum probability mass per op in *names*.
            threshold: (float) cumulative probability mass to keep, in (0, 1].
            min_ops: (int) floor on the total number of ops kept (NoOp included).
            flatness_epsilon: (float) skip pruning if the rankable pool's spread is
                <= this.

        Returns:
            (new_names, changed): changed is True iff new_names != names.
        """
        has_noop = self.noop_fn_name in names
        rankable = [i for i, n in enumerate(names) if n != self.noop_fn_name]
        if not rankable:
            return list(names), False

        rankable_weights = mean_weight[rankable]
        if rankable_weights.max() - rankable_weights.min() <= flatness_epsilon:
            return list(names), False

        ranked = sorted(rankable, key=lambda i: (mean_weight[i], -i), reverse=True)
        total = rankable_weights.sum()
        min_rankable = max(min_ops - 1, 0) if has_noop else min_ops

        cumulative = 0.0
        num_kept_rankable = 0
        for i in ranked:
            cumulative += mean_weight[i]
            num_kept_rankable += 1
            if cumulative / total >= threshold:
                break
        num_kept_rankable = max(num_kept_rankable, min(min_rankable, len(ranked)))

        kept = set(ranked[:num_kept_rankable])
        if has_noop:
            kept.add(names.index(self.noop_fn_name))
        kept = sorted(kept)
        new_names = [names[i] for i in kept]
        return new_names, new_names != list(names)

    def _nucleus_prune_all_nodes(self, old_fn_list):
        """Nucleus-cut (with flatness guard) each node's ops independently, using that
        node's own mean quantum probability mass. Dynamic-mode counterpart to
        _rank_and_prune_all_nodes.

        Args:
            old_fn_list: list of length num_genes, each entry that node's current
                op-name list.

        Returns:
            (new_fn_list, changed_per_node): changed_per_node[i] is True iff node i's
            op list changed this call.
        """
        new_fn_list = []
        changed_per_node = []
        for node_idx, names in enumerate(old_fn_list):
            mean_weight = self.qpop_net.probabilities[node_idx].mean(axis=0)
            node_new_fn_list, changed = self._nucleus_prune_ops(
                names, mean_weight, self.dynamic_probability_threshold,
                self.dynamic_min_ops, self.dynamic_flatness_epsilon,
            )
            if changed:
                self.logger.info(
                    f"Dynamic op pruning (nucleus), node {node_idx}: "
                    f"{names} -> {node_new_fn_list}"
                )
            new_fn_list.append(node_new_fn_list)
            changed_per_node.append(changed)

        return new_fn_list, changed_per_node

    def _nucleus_prune_globally(self, old_fn_list):
        """Nucleus-cut (with flatness guard) ONCE using the mean quantum probability
        mass across every node and individual, applying the same surviving op list to
        every node. Dynamic-mode counterpart to _rank_and_prune_globally.

        Requires every node to currently share the same op list (see
        _rank_and_prune_globally's docstring for why).

        Args:
            old_fn_list: list of length num_genes, each entry that node's current
                op-name list (all identical).

        Returns:
            (new_fn_list, changed): new_fn_list has the same (single) op list repeated
            for every node; changed is a single network-wide flag.
        """
        names = old_fn_list[0]
        if any(node_names != names for node_names in old_fn_list):
            raise ValueError(
                "global_op_pruning requires every node to share the same op list - "
                "nodes have already diverged (was global_op_pruning enabled from the "
                "start?)."
            )

        mean_weight = np.stack(self.qpop_net.probabilities, axis=1).mean(axis=(0, 1))
        new_names, changed = self._nucleus_prune_ops(
            names, mean_weight, self.dynamic_probability_threshold,
            self.dynamic_min_ops, self.dynamic_flatness_epsilon,
        )
        if changed:
            self.logger.info(f"Dynamic op pruning (nucleus, global): {names} -> {new_names}")

        return [list(new_names) for _ in old_fn_list], changed

    def _init_stability_streaks(self):
        """(Re)initialize dynamic mode's op-pruning stability streak - zeroed per node
        (or a single zero under global_op_pruning). Called on init and after every
        depth-growth event, so streaks always count from the current depth.
        """
        if self.global_op_pruning:
            self._stable_streak = 0
        else:
            self._stable_streak = [0] * self.qpop_net.chromosome.num_genes

    def _update_stability_streaks(self, changed_flags):
        """Advance dynamic mode's stability streak(s): a node (or, under
        global_op_pruning, the whole network) whose op list did NOT change this check
        increments its streak; a change resets it to 0.
        """
        if self.global_op_pruning:
            self._stable_streak = 0 if changed_flags[0] else self._stable_streak + 1
        else:
            self._stable_streak = [
                0 if changed else streak + 1
                for changed, streak in zip(changed_flags, self._stable_streak)
            ]

    def _dynamic_prune_step(self):
        """Dynamic-mode periodic check: nucleus-cut ops (no depth change), then decide
        whether op-pruning has stabilized enough to grow depth. See
        docs/QNAS_PROGRESSIVE_DYNAMIC_PLAN.md.
        """
        old_fn_list = [list(names) for names in self.qpop_net.chromosome.fn_list]
        current_num_nodes = self.qpop_net.chromosome.num_genes

        if self.global_op_pruning:
            new_fn_list, changed = self._nucleus_prune_globally(old_fn_list)
            changed_flags = [changed] * current_num_nodes
        else:
            new_fn_list, changed_flags = self._nucleus_prune_all_nodes(old_fn_list)

        self._apply_fn_list_change(new_fn_list, current_num_nodes)
        self._update_stability_streaks(changed_flags)

        at_ceiling = current_num_nodes >= self.dynamic_max_num_nodes
        streak = self._stable_streak if self.global_op_pruning else min(self._stable_streak)
        if streak >= self.dynamic_growth_patience and not at_ceiling:
            new_num_nodes = min(
                current_num_nodes + self.dynamic_node_growth_amount,
                self.dynamic_max_num_nodes,
            )
            grown_fn_list = [list(names) for names in self.qpop_net.chromosome.fn_list]
            self._apply_fn_list_change(grown_fn_list, new_num_nodes)
            self._init_stability_streaks()
            self.logger.info(
                f"Dynamic progressive growth: {current_num_nodes} -> {new_num_nodes} "
                f"nodes (stable for >= {self.dynamic_growth_patience} checks)."
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

            if self.progressive_stages:
                target_stage_idx = max(
                    i for i, s in enumerate(self.progressive_stages)
                    if s['gen_start'] <= self.current_gen
                )
                if target_stage_idx != self.current_stage_idx:
                    self._transition_stage(target_stage_idx)
            elif (self.progressive_mode == 'dynamic'
                    and self.current_gen % self.dynamic_check_every_gen == 0
                    and self.current_gen > 0):
                self._dynamic_prune_step()

            self.generate_classical()
            self.go_next_gen()
            
            if self.early_stopping and self.check_early_stopping():
                break
        
        end_evolution = time.time()
        evolution_hours, evolution_minutes = calculate_time(start_evolution,end_evolution)
        self.logger.info(f"Total evolution time: {evolution_hours} hours and {evolution_minutes} minutes")
        self.logger.info(f"Best network found: {self.best_so_far_id} --> {self.best_so_far:.5f}")