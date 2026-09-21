import numpy as np
import pytest

from chromosome import QChromosome, QChromosomeNetwork, QChromosomeParams
from conftest import FN_LIST, PARAMS_RANGES


class TestQChromosomeBase:
    def test_abstract_methods_raise(self):
        base = QChromosome(np.float64)
        with pytest.raises(NotImplementedError):
            base.initialize_qgenes()
        with pytest.raises(NotImplementedError):
            base.decode(np.zeros(1))


class TestQChromosomeParams:
    def test_num_genes_and_names(self):
        c = QChromosomeParams(PARAMS_RANGES)
        assert c.num_genes == len(PARAMS_RANGES)
        assert c.params_names == list(PARAMS_RANGES)

    def test_limits_follow_insertion_order(self):
        lower, upper = QChromosomeParams(PARAMS_RANGES).get_limits()
        assert lower == [r[0] for r in PARAMS_RANGES.values()]
        assert upper == [r[1] for r in PARAMS_RANGES.values()]

    def test_initialize_qgenes_dtype(self):
        lower, upper = QChromosomeParams(PARAMS_RANGES, dtype=np.float32).initialize_qgenes()
        assert lower.dtype == upper.dtype == np.float32
        assert np.all(lower < upper)

    def test_decode_returns_python_floats(self):
        # Regression: decode used np.asscalar, which was removed in NumPy 1.23.
        c = QChromosomeParams(PARAMS_RANGES)
        decoded = c.decode(np.array([0.9, 1e-3, 0.5, 1e-4]))
        assert decoded == {'decay': 0.9, 'learning_rate': 1e-3, 'momentum': 0.5,
                           'weight_decay': 1e-4}
        assert all(type(v) is float for v in decoded.values())


class TestQChromosomeNetwork:
    def test_flat_fn_list_is_expanded_per_node(self):
        c = QChromosomeNetwork(max_num_nodes=4, fn_list=FN_LIST)
        assert len(c.fn_list) == 4
        assert all(node == FN_LIST for node in c.fn_list)
        assert c.num_functions == [len(FN_LIST)] * 4
        assert c.num_genes == 4

    def test_expanded_lists_are_independent_copies(self):
        c = QChromosomeNetwork(max_num_nodes=2, fn_list=FN_LIST)
        c.fn_list[0].append('extra')
        assert 'extra' not in c.fn_list[1]
        assert 'extra' not in FN_LIST

    def test_ragged_fn_list_is_kept(self):
        ragged = [['a', 'b', 'c'], ['a'], ['b', 'c']]
        c = QChromosomeNetwork(max_num_nodes=3, fn_list=ragged)
        assert c.fn_list == ragged
        assert c.num_functions == [3, 1, 2]

    def test_uniform_initial_probs(self):
        probs = QChromosomeNetwork(3, FN_LIST).initialize_qgenes()
        assert probs.shape == (len(FN_LIST),)
        assert probs.sum() == pytest.approx(1.0)
        assert np.allclose(probs, 1 / len(FN_LIST))

    def test_explicit_initial_probs(self):
        given = [0.4, 0.3, 0.1, 0.1, 0.1]
        probs = QChromosomeNetwork(3, FN_LIST).initialize_qgenes(initial_probs=given)
        assert np.array_equal(probs, np.array(given))

    def test_decode_maps_indices_per_node(self):
        ragged = [['a', 'b'], ['c', 'd'], ['e', 'f']]
        c = QChromosomeNetwork(3, ragged)
        assert c.decode(np.array([1, 0, 1])) == ['b', 'c', 'f']

    def test_decode_negative_gene_is_none(self):
        c = QChromosomeNetwork(3, FN_LIST)
        assert c.decode(np.array([0, -1, 2])) == [FN_LIST[0], None, FN_LIST[2]]
