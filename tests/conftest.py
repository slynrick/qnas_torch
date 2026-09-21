"""Shared fixtures. src/ is put on sys.path by [tool.pytest.ini_options].pythonpath."""

import os
from collections import OrderedDict

import numpy as np
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FN_LIST = ['conv_3', 'conv_5', 'pool', 'skip', 'no_op']
NOOP = 'no_op'
REDUCING = ['pool']

PARAMS_RANGES = OrderedDict([
    ('decay', [0.8, 0.99]),
    ('learning_rate', [1e-4, 1e-2]),
    ('momentum', [0.1, 0.9]),
    ('weight_decay', [1e-5, 1e-3]),
])


@pytest.fixture(autouse=True)
def _seed():
    """Every test starts from the same numpy RNG state."""
    np.random.seed(1234)


class FakeEval:
    """Stand-in for evaluation.EvalPopulation: no GPU, no training.

    Fitness is a deterministic function of the architecture (its non-no-op count plus a
    tiny per-op bias), so identical architectures score identically, like the real cache.
    """

    def __init__(self, fn_list):
        self.fn_list = fn_list
        self.calls = []

    def __call__(self, decoded_params, decoded_nets, generation):
        self.calls.append((len(decoded_nets), generation))
        fitness = np.array([
            sum(1.0 + 0.1 * (op == 'conv_3') for op in net if op is not None and op != NOOP)
            for net in decoded_nets
        ], dtype=np.float64)
        return fitness, np.zeros(len(decoded_nets), dtype=bool)


def qnas_kwargs(**overrides):
    """Minimal valid kwargs for QNAS.initialize_qnas()."""
    kwargs = dict(
        num_quantum_ind=3,
        params_ranges=PARAMS_RANGES,
        repetition=2,
        max_generations=6,
        crossover_rate=0.5,
        update_quantum_gen=2,
        replace_method='best',
        fn_list=FN_LIST,
        initial_probs=[],
        update_quantum_rate=1.0,
        max_num_nodes=4,
        reducing_fns_list=REDUCING,
        patience=100,
        early_stopping=False,
        save_data_freq=0,
        penalize_number=0,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def make_qnas(tmp_path):
    """Factory: build an initialized QNAS wired to a FakeEval, writing into tmp_path."""
    import qnas

    def _make(**overrides):
        fake = FakeEval(list(overrides.get('fn_list', FN_LIST)))
        q = qnas.QNAS(fake, str(tmp_path), log_file=str(tmp_path / 'log_QNAS.txt'),
                      log_level='NONE', data_file=str(tmp_path / 'data_QNAS.pkl'))
        q.initialize_qnas(**qnas_kwargs(**overrides))
        return q

    return _make


def concentrated_probs(fn_list):
    """Initial PMF with all mass on conv_3/pool/no_op, so any prune to those three ops
    keeps every sampled individual alive (no partial-survivor edge cases)."""
    mass = {'conv_3': 0.4, 'pool': 0.3, NOOP: 0.3}
    return [mass.get(name, 0.0) for name in fn_list]
