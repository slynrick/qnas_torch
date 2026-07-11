""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Quantum population classes.
"""

import numpy as np

from chromosome import QChromosomeParams, QChromosomeNetwork


class QPopulation(object):
    """ QNAS Population to be evolved. """

    def __init__(self, num_quantum_ind, repetition, update_quantum_rate):
        """ Initialize QPopulation.

        Args:
            num_quantum_ind: (int) number of quantum individuals.
            repetition: (int) ratio between the number of classic individuals in the classic
                population and the quantum individuals in the quantum population.
            update_quantum_rate: (float) probability that a quantum gene will be updated.
        """

        self.dtype = np.float64  # Type of quantum population arrays.

        self.chromosome = None
        self.current_pop = None
        self.num_ind = num_quantum_ind

        self.repetition = repetition
        self.update_quantum_rate = update_quantum_rate

    def initialize_qpop(self):
        raise NotImplementedError('initialize_qpop() must be implemented in sub classes')

    def generate_classical(self):
        raise NotImplementedError('generate_classical() must be implemented in sub classes')

    def update_quantum(self, intensity):
        raise NotImplementedError('update_quantum() must be implemented in sub classes')


class QPopulationParams(QPopulation):
    """ QNAS Chromosomes for the hyperparameters to be evolved. """

    def __init__(self, num_quantum_ind, params_ranges, repetition, crossover_rate,
                update_quantum_rate):
        """ Initialize QPopulationParams.

        Args:
            num_quantum_ind: (int) number of quantum individuals.
            params_ranges: {'parameter_name': [parameter_lower_limit, parameter_upper_limit]}.
            repetition: (int) ratio between the number of classic individuals in the classic
                population and the quantum individuals in the quantum population.
            crossover_rate: (float) crossover rate.
            update_quantum_rate: (float) probability that a quantum gene will be updated.
        """

        super(QPopulationParams, self).__init__(num_quantum_ind, repetition,
                                                update_quantum_rate)

        self.tolerance = 1.e-15  # Tolerance to compare floating point

        self.lower = None
        self.upper = None
        self.crossover = crossover_rate

        self.chromosome = QChromosomeParams(params_ranges, self.dtype)

        self.initial_lower, self.initial_upper = self.chromosome.initialize_qgenes()

        self.initialize_qpop()

    def initialize_qpop(self):
        """ Initialize quantum population with *self.num_ind* individuals. """

        self.lower = np.tile(self.initial_lower, (self.num_ind, 1))
        self.upper = np.tile(self.initial_upper, (self.num_ind, 1))

    def classic_crossover(self, new_pop, distance):
        """ Perform arithmetic crossover of the old classic population with the new one.

        Args:
            new_pop: float numpy array representing the new classical population.
            distance: (float) random distance for arithmetic crossover (range = [0, 1]).
        """

        mask = np.random.rand(self.num_ind * self.repetition, self.chromosome.num_genes)
        idx = np.where(mask <= self.crossover)
        new_pop[idx] = new_pop[idx] + (self.current_pop[idx] - new_pop[idx]) * distance

        return new_pop

    def generate_classical(self):
        """ Generate a specific number of classical individuals from the observation of quantum
            individuals. This number is equal to (*num_ind* x *repetition*).
        """

        random_numbers = np.random.rand(self.num_ind * self.repetition,
                                        self.chromosome.num_genes).astype(self.dtype)

        new_pop = random_numbers * np.tile(self.upper - self.lower, (self.repetition, 1)) \
            + np.tile(self.lower, (self.repetition, 1))

        return new_pop

    def update_quantum(self, intensity):
        """ Update self.lower and self.upper.

        Args:
            intensity: (float) value defining the maximum intensity of the update.
        """

        random = np.random.rand(self.num_ind, self.chromosome.num_genes)
        mask = np.where(random <= self.update_quantum_rate)

        max_genes = np.max(self.current_pop, axis=0)
        min_genes = np.min(self.current_pop, axis=0)
        diff = np.tile(max_genes - min_genes, (self.num_ind, 1))

        update = self.current_pop[mask] - self.lower[mask] - (diff[mask] / 2)
        self.lower[mask] += intensity * update

        update = self.current_pop[mask] - self.upper[mask] + (diff[mask] / 2)
        self.upper[mask] += intensity * update
        # Correct limits (truncate) if they get out of the initial boundaries
        for i in range(self.num_ind):
            idx = np.where(self.lower[i] - self.initial_lower < -self.tolerance)
            self.lower[i][idx] = self.initial_lower[idx]
            idx = np.where(self.upper[i] - self.initial_upper > self.tolerance)
            self.upper[i][idx] = self.initial_upper[idx]


class QPopulationNetwork(QPopulation):
    """ QNAS Chromosomes for the networks to be evolved. """

    def __init__(self, num_quantum_ind, max_num_nodes, repetition, update_quantum_rate,
                fn_list, initial_probs,crossover_method='hux'):
        """ Initialize QPopulationNetwork.

        Args:
            num_quantum_ind: (int) number of quantum individuals.
            max_num_nodes: (int) maximum number of nodes of the network, which will be the
                number of genes in a individual.
            repetition: (int) ratio between the number of classic individuals in the classic
                population and the quantum individuals in the quantum population.
            update_quantum_rate: (float) probability that a quantum gene will be updated.
            fn_list: list of possible functions.
            initial_probs: list defining the initial probabilities for each function; if empty,
                the algorithm will give the same probability for each function.
        """

        super(QPopulationNetwork, self).__init__(num_quantum_ind, repetition,
                                                update_quantum_rate)
        self.probabilities = None

        self.max_update = 0.05
        self.max_prob = 0.99

        self.chromosome = QChromosomeNetwork(max_num_nodes, fn_list, self.dtype)

        self.initial_probs = self.chromosome.initialize_qgenes(initial_probs=initial_probs)
        self.crossover_method = crossover_method  # Crossover method selection
        self.initialize_qpop()

    def initialize_qpop(self):
        """ Initialize quantum population with *self.num_ind* individuals.

        self.probabilities is a list of length num_genes (one entry per node), each a
        [num_ind, num_functions] array - kept ragged (not a single dense 3D array) since
        nodes can later diverge to different op-set sizes/identities after a progressive-
        stage prune (see grow_and_prune_discrete). Never coerce this list into a single
        np.array() - a ragged 2nd axis would force dtype=object with no vectorized ops.
        """

        self.probabilities = [
            np.tile(self.initial_probs, (self.num_ind, 1))
            for _ in range(self.chromosome.num_genes)
        ]

    def generate_classical(self):
        """ Generate a specific number of classical individuals from the observation of quantum
            individuals. This number is equal to (*num_ind* x *repetition*).
        """

        new_pop = np.zeros(shape=(self.num_ind * self.repetition, self.chromosome.num_genes),
                            dtype=np.int32)

        for node in range(self.chromosome.num_genes):
            size = len(self.chromosome.fn_list[node])
            temp_prob = np.tile(self.probabilities[node], (self.repetition, 1))
            for ind in range(self.num_ind * self.repetition):
                new_pop[ind, node] = np.random.choice(size, p=temp_prob[ind, :])

        return new_pop
    
    def hux_crossover(self, parent1, parent2):
        """ Perform Half Uniform Crossover (HUX) between two parent chromosomes. """
        differing_indices = np.where(parent1 != parent2)[0]
        num_swaps = len(differing_indices) // 2
        swap_indices = np.random.choice(differing_indices, num_swaps, replace=False)
        offspring1, offspring2 = parent1.copy(), parent2.copy()
        offspring1[swap_indices], offspring2[swap_indices] = parent2[swap_indices], parent1[swap_indices]
        return offspring1, offspring2

    def uniform_crossover(self, parent1, parent2):
        """ Perform Uniform Crossover with a crossover mask between two parent chromosomes. """
        chromosome_length = len(parent1)
        crossover_mask = np.random.randint(0, 2, size=chromosome_length).astype(bool)
        offspring1, offspring2 = parent1.copy(), parent2.copy()
        offspring1[crossover_mask], offspring2[crossover_mask] = parent2[crossover_mask], parent1[crossover_mask]
        return offspring1, offspring2

    def apply_crossover(self, best_current_pop, new_pop):
        """ Apply the selected crossover method between best individuals of the current and new populations. 
        
        Args:
            best_current_pop: numpy array representing the best individuals from the current population.
            new_pop: numpy array representing the new population.
        
        Returns:
            A population of offspring resulting from the selected crossover method.
        """
        offspring = []
        for parent1, parent2 in zip(best_current_pop, new_pop):
            if self.crossover_method == 'hux':
                child1, child2 = self.hux_crossover(parent1, parent2)
            elif self.crossover_method == 'uniform':
                child1, child2 = self.uniform_crossover(parent1, parent2)
            else:
                raise ValueError(f"Unknown crossover method: {self.crossover_method}")
            offspring.extend([child1, child2])

        return np.array(offspring[:len(new_pop)])  # Ensure offspring size matches new_pop size

    def set_crossover_method(self, method):
        """ Set the crossover method for this population. """
        if method in ['hux', 'uniform']:
            self.crossover_method = method
        else:
            raise ValueError(f"Unknown crossover method: {method}")

    def _update(self, chromosomes, idx, update_value):
        """ Modify *chromosomes* by adding *update_value* to the genes indicated by *idx* and
            subtracting *update_value* from the other genes proportional to the size of each
            probability.

        Args:
            chromosomes: 2D float numpy array representing the chromosomes to be updated.
            idx: (int) index of the genes to have their value increased.
            update_value: (float) value that will be added to the selected functions in
                *chromosomes* by *idx*.

        Returns:
            modified chromosome
        """

        idx0 = np.arange(chromosomes.shape[0])
        update_array = np.where(chromosomes[idx0, idx] + update_value > self.max_prob,
                                0, update_value)
        sum_values = chromosomes[idx0, idx] + update_array
        chromosomes[idx0, idx] = 0
        decrease = (update_array / np.sum(chromosomes, axis=1)).reshape(-1, 1)
        decrease = decrease * chromosomes
        chromosomes = chromosomes - decrease
        chromosomes[idx0, idx] = sum_values

        return chromosomes

    def update_quantum(self, intensity):
        """ Update self.probabilities.

        Args:
            intensity: (float) value defining the intensity of the update.
        """

        update_value = intensity * self.max_update
        best_classic = self.current_pop[:self.num_ind]

        for node in range(self.chromosome.num_genes):
            random = np.random.rand(self.num_ind)
            idx = np.where(random <= self.update_quantum_rate)[0]

            self.probabilities[node][idx] = self._update(
                self.probabilities[node][idx], best_classic[idx, node], update_value,
            )

    def grow_and_prune_discrete(self, new_num_nodes: int, new_fn_list: list,
                                reset_probs: bool = False):
        """Resize *self.probabilities* (the quantum PMF) for a P-DARTS-style progressive
        stage transition. Each node prunes and grows independently:

        - Existing nodes (index < old_num_nodes): *new_fn_list[i]* is that node's own
          already-pruned op list (produced per node by the caller, e.g.
          QNAS._rank_and_prune_all_nodes). If *reset_probs* is False (default), kept ops
          carry over their probability mass, matched by name since pruning can reorder
          indices; each node's row is renormalized afterwards so it still sums to 1, as
          required by generate_classical()'s np.random.choice(..., p=...). If an
          individual's carried mass for a node is ~0 (all its mass was on ops pruned away
          at that node), that individual's row resets to uniform over the node's new op
          list. If *reset_probs* is True, every existing node's row is set uniform over
          its new op list instead of carrying anything over - i.e. the search "forgets"
          what it learned before the transition and restarts from scratch on the new
          (smaller) op menu.
        - New nodes (index >= old_num_nodes, from growing depth): always start UNIFORM
          (not mean-seeded) over the distinct UNION of ops that survived pruning across
          all existing nodes (uncapped - can exceed the stage's num_ops), regardless of
          *reset_probs* - they have no prior probability mass to carry over or reset.
          Every individual gets the same uniform row for a new node.

        self.probabilities/self.chromosome.fn_list stay ragged Python lists (see
        initialize_qpop) - never coerced into one dense 3D array.

        Args:
            new_num_nodes: new depth (>= current depth).
            new_fn_list: list of length old_num_nodes, each entry the (already pruned,
                per-node) new ordered list of operation names for that existing node.
            reset_probs: (bool) if True, reset existing nodes' probabilities to uniform
                instead of carrying over/renormalizing the pre-transition PMF.
        """
        old_fn_list = self.chromosome.fn_list
        old_num_nodes = self.chromosome.num_genes

        new_probabilities = [None] * new_num_nodes
        for i in range(old_num_nodes):
            node_new_fn_list = new_fn_list[i]

            if reset_probs:
                new_probabilities[i] = np.full(
                    (self.num_ind, len(node_new_fn_list)),
                    1.0 / len(node_new_fn_list), dtype=self.dtype,
                )
                continue

            old_idx = {name: k for k, name in enumerate(old_fn_list[i])}
            carried = np.zeros((self.num_ind, len(node_new_fn_list)), dtype=self.dtype)
            for k, name in enumerate(node_new_fn_list):
                carried[:, k] = self.probabilities[i][:, old_idx[name]]

            row_sums = carried.sum(axis=1, keepdims=True)
            zero_mask = row_sums[:, 0] <= 1e-12
            carried[zero_mask] = 1.0 / len(node_new_fn_list)
            carried[~zero_mask] /= row_sums[~zero_mask]

            new_probabilities[i] = carried

        if new_num_nodes > old_num_nodes:
            union_ops = sorted(set().union(*new_fn_list[:old_num_nodes]))
            if not union_ops:
                raise ValueError(
                    "grow_and_prune_discrete: union of surviving ops across existing "
                    "nodes is empty - cannot seed new nodes."
                )
            for i in range(old_num_nodes, new_num_nodes):
                new_fn_list.append(union_ops)
                new_probabilities[i] = np.full(
                    (self.num_ind, len(union_ops)), 1.0 / len(union_ops), dtype=self.dtype,
                )

        self.probabilities = new_probabilities
        self.chromosome.fn_list = new_fn_list
        self.chromosome.num_functions = [len(l) for l in new_fn_list]
        self.chromosome.set_num_genes(new_num_nodes)

    def filter_and_remap_classical(self, old_current_pop, old_fn_list, new_fn_list, noop_name):
        """Filter and remap a classical (gene-index) population across a progressive-stage
        transition, so individuals with real evaluated fitness survive instead of being
        thrown away.

        For each individual and each pre-existing node position, the op name it currently
        encodes is looked up against that node's new (pruned) op list: if the op didn't
        survive the prune at that node, the whole individual is dropped (its network is no
        longer decodable/valid). Surviving individuals have their gene at that node
        remapped to the op's new index (pruning can reorder indices). Any brand-new node
        positions (growth) are set to *noop_name*'s index - a no-op is a computational
        identity, so appending it doesn't change what the individual's fitness measured;
        it does not need to be re-evaluated just because the population grew deeper.

        Args:
            old_current_pop: int ndarray [n_individuals, old_num_nodes], gene indices.
            old_fn_list: list of length old_num_nodes (per-node op lists before the prune).
            new_fn_list: list of length new_num_nodes (per-node op lists after prune+growth,
                i.e. self.chromosome.fn_list right after grow_and_prune_discrete).
            noop_name: (str) name of the no-op function, used to fill newly grown node
                positions for surviving individuals.

        Returns:
            (kept_mask, remapped_pop): kept_mask is a bool ndarray of length
            n_individuals (True for individuals to keep); remapped_pop is an int ndarray
            [n_kept, new_num_nodes] with remapped/extended gene indices for kept individuals
            (empty first axis if none survive).
        """
        old_num_nodes = len(old_fn_list)
        new_num_nodes = len(new_fn_list)
        n_individuals = old_current_pop.shape[0]

        kept_mask = np.ones(n_individuals, dtype=bool)
        remapped = np.zeros((n_individuals, new_num_nodes), dtype=old_current_pop.dtype)

        for node in range(old_num_nodes):
            new_index_of = {name: k for k, name in enumerate(new_fn_list[node])}
            for ind in range(n_individuals):
                if not kept_mask[ind]:
                    continue
                gene = old_current_pop[ind, node]
                name = old_fn_list[node][gene]
                if name not in new_index_of:
                    kept_mask[ind] = False
                    continue
                remapped[ind, node] = new_index_of[name]

        for node in range(old_num_nodes, new_num_nodes):
            noop_idx = new_fn_list[node].index(noop_name)
            remapped[:, node] = noop_idx

        return kept_mask, remapped[kept_mask]
