""" Defaults of search-engine options added after experiments had already been run.

Each default is the behavior of the code before the option existed, so a config or a
recorded log_params_evolution.txt without the key means exactly the default. Kept free
of heavy imports: diff_runs.py uses it to compare old run logs (which lack these keys)
with new ones without reporting false differences.
"""

# QNAS_spec key -> default. 'prune_criterion' is read from QNAS.progressive in the YAML
# (it only matters for progressive pruning); the others from the QNAS block itself.
SEARCH_ENGINE_DEFAULTS = {
    'quantum_lr': None,
    'quantum_rank_weighting': 'uniform',
    'quantum_top_k': 1,
    'quantum_negative_lr': 0.0,
    'quantum_prob_floor': 0.0,
    'quantum_max_update': 0.05,
    'quantum_max_prob': 0.99,
    'prune_criterion': 'pmf',
}

# The same defaults as flattened log_params keys (see diff_runs.flatten).
LOGGED_DEFAULTS = {f'QNAS.{key}': value for key, value in SEARCH_ENGINE_DEFAULTS.items()}
