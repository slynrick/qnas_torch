"""Golden test: with every optional search-engine knob at its default, evolve() must replay
the exact same populations, PMFs and fitnesses as the code that produced the existing
experiments (reference recorded in tests/data/golden_default_path.pkl).

New options must branch off the default path without touching it - including the order of
numpy RNG draws - so configs, checkpoints and run logs written before them keep their
meaning. Regenerate the reference only for a deliberate behavior change:

    QNAS_REGENERATE_GOLDEN=1 make test ARGS="tests/test_default_path_golden.py"
"""

import os
import pickle

import numpy as np
import pytest

import util
from conftest import FN_LIST, NOOP, ROOT_DIR, concentrated_probs

GOLDEN_FILE = os.path.join(ROOT_DIR, 'tests', 'data', 'golden_default_path.pkl')

STAGES = [{'gen_start': 0, 'num_nodes': 3, 'num_ops': 5},
          {'gen_start': 3, 'num_nodes': 4, 'num_ops': 4},
          {'gen_start': 6, 'num_nodes': 5, 'num_ops': 3}]

SCENARIOS = {
    'plain_default': dict(max_generations=8, update_quantum_rate=0.5),
    'plain_ancestor_elitism': dict(max_generations=8, update_quantum_rate=0.5,
                                   replace_method='elitism',
                                   quantum_update_engine='ancestor_decay'),
    'plain_crossover': dict(max_generations=8, update_quantum_rate=0.5, en_pop_crossover=True,
                            crossover_frequency=2, pop_crossover_rate=0.5),
    'deterministic_carry_over': dict(max_generations=9, update_quantum_rate=0.5,
                                     progressive_stages=STAGES, noop_fn_name=NOOP,
                                     progressive_mode='deterministic'),
    'deterministic_reset_ancestor': dict(max_generations=9, update_quantum_rate=0.5,
                                         progressive_stages=STAGES, noop_fn_name=NOOP,
                                         progressive_mode='deterministic',
                                         reset_probs_on_stage_change=True,
                                         quantum_update_engine='ancestor_decay'),
    'deterministic_global': dict(max_generations=9, update_quantum_rate=0.5,
                                 progressive_stages=STAGES, noop_fn_name=NOOP,
                                 progressive_mode='deterministic', global_op_pruning=True),
    'dynamic_ancestor': dict(max_generations=14, progressive_mode='dynamic', noop_fn_name=NOOP,
                             initial_probs=concentrated_probs(FN_LIST),
                             dynamic_initial_num_nodes=2, dynamic_max_num_nodes=4,
                             dynamic_probability_threshold=0.6, dynamic_min_ops=2,
                             dynamic_growth_patience=1, dynamic_node_growth_amount=1,
                             update_quantum_gen=2, dynamic_check_every_gen=2,
                             quantum_update_engine='ancestor_decay'),
}


def trace(q):
    """Everything evolve() recorded per generation that the search engine decides."""
    history = util.load_pkl(q.data_file)
    return {gen: {'net_probs': [np.array(p) for p in entry['net_probs']],
                  'net_pop': np.array(entry['net_pop']),
                  'fitnesses': np.array(entry['fitnesses']),
                  'fn_list': entry.get('fn_list')}
            for gen, entry in history.items()}


def run_scenario(make_qnas, name):
    np.random.seed(2026)
    q = make_qnas(**SCENARIOS[name])
    q.evolve()
    return trace(q)


def load_golden():
    with open(GOLDEN_FILE, 'rb') as f:
        return pickle.load(f)


@pytest.mark.e2e
@pytest.mark.parametrize('name', sorted(SCENARIOS))
def test_default_path_matches_recorded_reference(make_qnas, name):
    got = run_scenario(make_qnas, name)

    if os.environ.get('QNAS_REGENERATE_GOLDEN'):
        golden = load_golden() if os.path.exists(GOLDEN_FILE) else {}
        golden[name] = got
        os.makedirs(os.path.dirname(GOLDEN_FILE), exist_ok=True)
        with open(GOLDEN_FILE, 'wb') as f:
            pickle.dump(golden, f)
        pytest.skip('golden reference regenerated')

    expected = load_golden()[name]
    assert sorted(got) == sorted(expected)
    for gen in expected:
        assert got[gen]['fn_list'] == expected[gen]['fn_list'], f'gen {gen}'
        np.testing.assert_array_equal(got[gen]['net_pop'], expected[gen]['net_pop'],
                                      err_msg=f'gen {gen}')
        np.testing.assert_array_equal(got[gen]['fitnesses'], expected[gen]['fitnesses'],
                                      err_msg=f'gen {gen}')
        assert len(got[gen]['net_probs']) == len(expected[gen]['net_probs'])
        for node, (a, b) in enumerate(zip(got[gen]['net_probs'], expected[gen]['net_probs'])):
            np.testing.assert_array_equal(a, b, err_msg=f'gen {gen} node {node}')
