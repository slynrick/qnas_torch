import numpy as np
import pytest

from conftest import FN_LIST, NOOP, PARAMS_RANGES
from population import QPopulationNetwork, QPopulationParams

NUM_IND, REPETITION, NUM_NODES = 3, 2, 4


def make_net_pop(update_quantum_rate=1.0, fn_list=FN_LIST, num_nodes=NUM_NODES, **kw):
    return QPopulationNetwork(num_quantum_ind=NUM_IND, max_num_nodes=num_nodes,
                              repetition=REPETITION, update_quantum_rate=update_quantum_rate,
                              fn_list=fn_list, initial_probs=[], **kw)


def make_params_pop(update_quantum_rate=1.0):
    return QPopulationParams(num_quantum_ind=NUM_IND, params_ranges=PARAMS_RANGES,
                             repetition=REPETITION, crossover_rate=0.5,
                             update_quantum_rate=update_quantum_rate)


def assert_valid_pmf(pop):
    for probs in pop.probabilities:
        assert np.all(probs >= -1e-12)
        assert np.allclose(probs.sum(axis=1), 1.0)


class TestQPopulationParams:
    def test_initial_ranges_tiled_per_quantum_individual(self):
        pop = make_params_pop()
        assert pop.lower.shape == pop.upper.shape == (NUM_IND, len(PARAMS_RANGES))
        assert np.all(pop.lower == pop.initial_lower)
        assert np.all(pop.upper == pop.initial_upper)

    def test_generate_classical_within_ranges(self):
        pop = make_params_pop()
        sample = pop.generate_classical()
        assert sample.shape == (NUM_IND * REPETITION, len(PARAMS_RANGES))
        assert np.all(sample >= pop.initial_lower)
        assert np.all(sample <= pop.initial_upper)

    def test_generate_classical_respects_narrowed_range(self):
        pop = make_params_pop()
        pop.upper = pop.lower + 1e-9
        sample = pop.generate_classical()
        assert np.allclose(sample, np.tile(pop.lower, (REPETITION, 1)), atol=1e-8)

    def test_classic_crossover_distance_zero_is_identity(self):
        pop = make_params_pop()
        pop.current_pop = pop.generate_classical()
        new = pop.generate_classical()
        assert np.allclose(pop.classic_crossover(new.copy(), distance=0.0), new)

    def test_classic_crossover_tolerates_smaller_old_population(self):
        # After a progressive transition drops individuals, current_pop has fewer rows.
        pop = make_params_pop()
        pop.crossover = 1.0
        pop.current_pop = pop.generate_classical()[:2]
        new = pop.generate_classical()
        untouched = new[2:].copy()
        out = pop.classic_crossover(new, distance=1.0)
        assert np.allclose(out[:2], pop.current_pop)   # crossed over
        assert np.allclose(out[2:], untouched)         # no old counterpart -> unchanged

    def test_classic_crossover_distance_one_copies_old_genes(self):
        pop = make_params_pop()
        pop.crossover = 1.0  # every gene is crossed over
        pop.current_pop = pop.generate_classical()
        new = pop.generate_classical()
        assert np.allclose(pop.classic_crossover(new, distance=1.0), pop.current_pop)

    def test_update_quantum_stays_within_initial_bounds(self):
        pop = make_params_pop()
        for _ in range(20):
            pop.current_pop = pop.generate_classical()
            pop.update_quantum(intensity=1.0)
            assert np.all(pop.lower >= pop.initial_lower - 1e-12)
            assert np.all(pop.upper <= pop.initial_upper + 1e-12)

    def test_update_quantum_rate_zero_changes_nothing(self):
        pop = make_params_pop(update_quantum_rate=-1.0)
        pop.current_pop = pop.generate_classical()
        lower, upper = pop.lower.copy(), pop.upper.copy()
        pop.update_quantum(intensity=1.0)
        assert np.array_equal(pop.lower, lower)
        assert np.array_equal(pop.upper, upper)

    def test_update_quantum_narrows_toward_population(self):
        pop = make_params_pop()
        width_before = (pop.upper - pop.lower).copy()
        for _ in range(30):
            pop.current_pop = pop.generate_classical()
            pop.update_quantum(intensity=0.5)
        assert np.all(pop.upper - pop.lower <= width_before + 1e-12)
        assert np.any(pop.upper - pop.lower < width_before)


class TestQPopulationNetworkInit:
    def test_probabilities_is_ragged_list_of_uniform_pmfs(self):
        pop = make_net_pop()
        assert isinstance(pop.probabilities, list)
        assert len(pop.probabilities) == NUM_NODES
        for probs in pop.probabilities:
            assert probs.shape == (NUM_IND, len(FN_LIST))
        assert_valid_pmf(pop)
        assert np.allclose(pop.probabilities[0], 1 / len(FN_LIST))

    def test_nodes_do_not_share_arrays(self):
        pop = make_net_pop()
        pop.probabilities[0][0, 0] = 0.9
        assert pop.probabilities[1][0, 0] != 0.9

    def test_generate_classical_shape_and_index_range(self):
        pop = make_net_pop()
        sample = pop.generate_classical()
        assert sample.shape == (NUM_IND * REPETITION, NUM_NODES)
        assert sample.min() >= 0 and sample.max() < len(FN_LIST)

    def test_generate_classical_follows_degenerate_pmf(self):
        pop = make_net_pop()
        for probs in pop.probabilities:
            probs[:] = 0.0
            probs[:, 2] = 1.0
        assert np.all(pop.generate_classical() == 2)

    def test_generate_classical_row_i_comes_from_quantum_individual_i_mod_n(self):
        # QNAS.generate_classical relies on this to build the ancestor lineage.
        pop = make_net_pop()
        for node_probs in pop.probabilities:
            node_probs[:] = 0.0
            for q in range(NUM_IND):
                node_probs[q, q] = 1.0
        sample = pop.generate_classical()
        expected_row = np.arange(NUM_IND * REPETITION) % NUM_IND
        assert np.all(sample == expected_row[:, None])


class TestUpdateRule:
    def test_update_preserves_pmf_and_boosts_target(self):
        pop = make_net_pop()
        chromosomes = pop.probabilities[0].copy()
        idx = np.array([1, 2, 3])
        updated = pop._update(chromosomes.copy(), idx, 0.05)
        assert np.allclose(updated.sum(axis=1), 1.0)
        rows = np.arange(NUM_IND)
        assert np.all(updated[rows, idx] > chromosomes[rows, idx])

    def test_update_never_exceeds_max_prob(self):
        pop = make_net_pop()
        chromosomes = pop.probabilities[0].copy()
        idx = np.zeros(NUM_IND, dtype=int)
        for _ in range(500):
            chromosomes = pop._update(chromosomes, idx, 0.05)
        assert chromosomes[:, 0].max() <= pop.max_prob + 1e-9
        assert np.allclose(chromosomes.sum(axis=1), 1.0)

    def test_update_quantum_default_targets_top_ranked(self):
        pop = make_net_pop()
        # Rank-i classical individual picks op i at every node.
        pop.current_pop = np.tile(np.arange(NUM_IND)[:, None], (2, NUM_NODES))
        pop.update_quantum(intensity=1.0)
        assert_valid_pmf(pop)
        for probs in pop.probabilities:
            for q in range(NUM_IND):
                assert probs[q, q] == probs[q].max()
                assert probs[q, q] > 1 / len(FN_LIST)

    def test_update_quantum_rate_zero_is_noop(self):
        pop = make_net_pop(update_quantum_rate=-1.0)
        pop.current_pop = np.zeros((NUM_IND * REPETITION, NUM_NODES), dtype=int)
        before = [p.copy() for p in pop.probabilities]
        pop.update_quantum(intensity=1.0)
        assert all(np.array_equal(a, b) for a, b in zip(before, pop.probabilities))


class TestAncestorDecayUpdate:
    def setup_pop(self, rows_per_ancestor=None):
        pop = make_net_pop()
        n = NUM_IND * REPETITION
        pop.current_pop = np.zeros((n, NUM_NODES), dtype=int)  # everyone picked op 0
        return pop, n

    def test_targets_the_producing_quantum_individual_not_the_rank(self):
        pop, n = self.setup_pop()
        # Only quantum individual 2 produced any classical individual.
        ancestors = np.full(n, 2)
        pop.update_quantum_ancestor_decay(1.0, np.zeros(n, dtype=int), ancestors, 0.5)
        for probs in pop.probabilities:
            assert probs[2, 0] > 1 / len(FN_LIST)
            assert np.allclose(probs[0], 1 / len(FN_LIST))
            assert np.allclose(probs[1], 1 / len(FN_LIST))
        assert_valid_pmf(pop)

    def test_older_individuals_push_less(self):
        young, n = self.setup_pop()
        old, _ = self.setup_pop()
        ancestors = np.zeros(n, dtype=int)
        young.update_quantum_ancestor_decay(1.0, np.zeros(n, dtype=int), ancestors, 0.5)
        old.update_quantum_ancestor_decay(1.0, np.full(n, 10), ancestors, 0.5)
        assert young.probabilities[0][0, 0] > old.probabilities[0][0, 0] > 1 / len(FN_LIST)

    def test_zero_decay_rate_ignores_age(self):
        a, n = self.setup_pop()
        b, _ = self.setup_pop()
        ancestors = np.zeros(n, dtype=int)
        np.random.seed(7)
        a.update_quantum_ancestor_decay(1.0, np.zeros(n, dtype=int), ancestors, 0.0)
        np.random.seed(7)
        b.update_quantum_ancestor_decay(1.0, np.full(n, 50), ancestors, 0.0)
        assert all(np.allclose(x, y) for x, y in zip(a.probabilities, b.probabilities))

    def test_very_old_individual_barely_moves_pmf(self):
        pop, n = self.setup_pop()
        pop.update_quantum_ancestor_decay(1.0, np.full(n, 200), np.zeros(n, dtype=int), 0.5)
        assert np.allclose(pop.probabilities[0], 1 / len(FN_LIST), atol=1e-6)


class TestNetworkCrossover:
    def test_hux_swaps_half_of_differing_genes(self):
        pop = make_net_pop()
        p1 = np.array([0, 0, 0, 0, 1, 1])
        p2 = np.array([1, 1, 1, 1, 1, 1])
        c1, c2 = pop.hux_crossover(p1, p2)
        assert (c1 != p1).sum() == 2  # 4 differing genes -> 2 swapped
        assert np.array_equal(np.sort(np.concatenate([c1, c2])), np.sort(np.concatenate([p1, p2])))

    def test_hux_identical_parents_unchanged(self):
        pop = make_net_pop()
        p = np.array([1, 2, 3])
        c1, c2 = pop.hux_crossover(p, p)
        assert np.array_equal(c1, p) and np.array_equal(c2, p)

    def test_uniform_crossover_preserves_gene_multiset(self):
        pop = make_net_pop()
        p1, p2 = np.arange(8), np.arange(8)[::-1].copy()
        c1, c2 = pop.uniform_crossover(p1, p2)
        assert np.array_equal(c1 + c2, p1 + p2)

    def test_parents_not_mutated(self):
        pop = make_net_pop()
        p1, p2 = np.array([0, 1, 2, 3]), np.array([3, 2, 1, 0])
        pop.hux_crossover(p1, p2)
        pop.uniform_crossover(p1, p2)
        assert list(p1) == [0, 1, 2, 3] and list(p2) == [3, 2, 1, 0]

    @pytest.mark.parametrize('method', ['hux', 'uniform'])
    def test_apply_crossover_matches_new_pop_shape(self, method):
        pop = make_net_pop(crossover_method=method)
        best = np.random.randint(0, 5, size=(3, NUM_NODES))
        new = np.random.randint(0, 5, size=(3, NUM_NODES))
        assert pop.apply_crossover(best, new).shape == new.shape

    def test_unknown_method_raises(self):
        pop = make_net_pop()
        with pytest.raises(ValueError):
            pop.set_crossover_method('bogus')
        pop.crossover_method = 'bogus'
        with pytest.raises(ValueError):
            pop.apply_crossover(np.zeros((1, 2), int), np.zeros((1, 2), int))

    def test_set_crossover_method(self):
        pop = make_net_pop()
        pop.set_crossover_method('uniform')
        assert pop.crossover_method == 'uniform'


class TestGrowAndPrune:
    def pruned(self):
        return [['conv_3', 'pool', NOOP] for _ in range(NUM_NODES)]

    def test_prune_carries_mass_by_name_and_renormalizes(self):
        pop = make_net_pop()
        for probs in pop.probabilities:
            probs[:] = [0.4, 0.1, 0.2, 0.1, 0.2]  # conv_3, conv_5, pool, skip, no_op
        pop.grow_and_prune_discrete(NUM_NODES, self.pruned())
        expected = np.array([0.4, 0.2, 0.2]) / 0.8
        for probs in pop.probabilities:
            assert probs.shape == (NUM_IND, 3)
            assert np.allclose(probs, expected)
        assert pop.chromosome.num_functions == [3] * NUM_NODES

    def test_reset_probs_makes_rows_uniform(self):
        pop = make_net_pop()
        pop.probabilities[0][:] = [0.9, 0.05, 0.02, 0.02, 0.01]
        pop.grow_and_prune_discrete(NUM_NODES, self.pruned(), reset_probs=True)
        assert np.allclose(pop.probabilities[0], 1 / 3)

    def test_zero_mass_row_falls_back_to_uniform(self):
        pop = make_net_pop()
        pop.probabilities[0][:] = [0.0, 0.5, 0.0, 0.5, 0.0]  # all mass on pruned ops
        pop.grow_and_prune_discrete(NUM_NODES, self.pruned())
        assert np.allclose(pop.probabilities[0], 1 / 3)

    def test_growth_adds_uniform_nodes_over_union_of_ops(self):
        pop = make_net_pop()
        pruned = [['conv_3', NOOP], ['pool', NOOP], ['conv_3', NOOP], ['conv_5', NOOP]]
        pop.grow_and_prune_discrete(NUM_NODES + 2, pruned)
        union = ['conv_3', NOOP, 'pool', 'conv_5']  # order of first appearance
        assert pop.chromosome.num_genes == NUM_NODES + 2
        assert len(pop.probabilities) == len(pop.chromosome.fn_list) == NUM_NODES + 2
        for node in (NUM_NODES, NUM_NODES + 1):
            assert pop.chromosome.fn_list[node] == union
            assert np.allclose(pop.probabilities[node], 1 / len(union))
        assert_valid_pmf(pop)

    def test_nodes_can_diverge_and_still_sample(self):
        pop = make_net_pop()
        pruned = [['conv_3', NOOP], ['pool', 'skip', NOOP], ['conv_5'], ['skip', NOOP]]
        pop.grow_and_prune_discrete(NUM_NODES, pruned)
        sample = pop.generate_classical()
        for node, names in enumerate(pruned):
            assert sample[:, node].max() < len(names)


class TestFilterAndRemap:
    def test_drops_individuals_using_pruned_ops_and_remaps_indices(self):
        pop = make_net_pop(num_nodes=2)
        old_fn_list = [list(FN_LIST), list(FN_LIST)]
        new_fn_list = [['pool', NOOP], ['pool', NOOP]]
        # conv_3=0 conv_5=1 pool=2 skip=3 no_op=4
        old_pop = np.array([[2, 4],   # pool, no_op   -> kept
                            [0, 2],   # conv_3, pool  -> dropped
                            [4, 2]])  # no_op, pool   -> kept
        mask, remapped = pop.filter_and_remap_classical(old_pop, old_fn_list, new_fn_list, NOOP)
        assert mask.tolist() == [True, False, True]
        assert remapped.tolist() == [[0, 1], [1, 0]]

    def test_new_nodes_are_filled_with_noop(self):
        pop = make_net_pop(num_nodes=1)
        old_fn_list = [list(FN_LIST)]
        new_fn_list = [['conv_3', NOOP], ['conv_3', NOOP, 'pool'], [NOOP, 'skip']]
        mask, remapped = pop.filter_and_remap_classical(
            np.array([[0]]), old_fn_list, new_fn_list, NOOP)
        assert mask.tolist() == [True]
        assert remapped.tolist() == [[0, 1, 0]]  # no_op index in each new node

    def test_no_survivors_returns_empty(self):
        pop = make_net_pop(num_nodes=1)
        mask, remapped = pop.filter_and_remap_classical(
            np.array([[0], [1]]), [list(FN_LIST)], [[NOOP]], NOOP)
        assert not mask.any()
        assert remapped.shape[0] == 0
