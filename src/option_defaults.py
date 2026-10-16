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

# Options of docs/QNAS_SEARCH_ENGINE_IMPROVEMENT_PLAN_V2.md. Each YAML sub-block maps to
# flat QNAS_spec keys '<prefix><name>' (e.g. QNAS.surrogate.enabled -> surrogate_enabled),
# so QNAS.initialize_qnas() takes them as plain keyword arguments.
#   D1 - surrogate-assisted screening (QNAS.surrogate.*)
SURROGATE_DEFAULTS = {
    'enabled': False,
    'oversample': 1,          # candidates generated per evaluated individual
    'n_eval': None,           # individuals actually trained per generation (None: num_ind * repetition)
    'warmup_evals': 200,      # evaluations of a stage before screening starts
    'tau0': 1.0,              # Boltzmann temperature at the start of a stage
    'tau_min': 0.1,
    'tau_decay_gens': 20,     # tau = max(tau_min, tau0 / (1 + gens_in_stage / tau_decay_gens))
    'beta': 0.5,              # exploration bonus on the ensemble spread
    'random_frac': 0.2,       # share of the evaluated slots picked uniformly at random
    'ensemble': 5,
    'ridge_lambda': 3.0,
    'seed': 0,                # seed of the surrogate's own RNG (never the global numpy RNG)
}
#   D3 - temperature on the PMF sampling (QNAS.sampling_temperature.*)
SAMPLING_TEMPERATURE_DEFAULTS = {
    'tau0': None,             # None: sample the PMF as is (the original behavior)
    'tau_min': 0.3,
    'tau_decay_gens': 30,     # tau = max(tau_min, tau0 / (1 + gens_in_stage / tau_decay_gens))
}
#   D4 - niches per quantum individual (QNAS.niche.*, used with replace_method: niche)
NICHE_DEFAULTS = {
    'max_age': None,          # survivors older than this leave the pool (the best-ever one stays)
    'restart_patience': None, # reset a niche's PMF after this many generations without improving
    'restart_kl': None,       # ... or when its PMF gets this close (KL) to the best niche's
}
#   D2 - two-rung fidelity inside a generation (QNAS.fidelity.*)
FIDELITY_DEFAULTS = {
    'enabled': False,
    'low_epochs': 10,         # epochs of the cheap rung
    'oversample': 2,          # individuals in the cheap rung per individual kept
}

# YAML block under QNAS -> (flat QNAS_spec key prefix, {name: default}).
NESTED_BLOCKS = {
    'surrogate': ('surrogate_', SURROGATE_DEFAULTS),
    'sampling_temperature': ('sampling_', SAMPLING_TEMPERATURE_DEFAULTS),
    'niche': ('niche_', NICHE_DEFAULTS),
    'fidelity': ('fidelity_', FIDELITY_DEFAULTS),
}


def flat_key(block, name):
    """ QNAS.<block>.<name> in the YAML -> its QNAS_spec / initialize_qnas() keyword. """
    return NESTED_BLOCKS[block][0] + name


V2_QNAS_DEFAULTS = {
    **{flat_key(block, name): value
       for block, (_, defaults) in NESTED_BLOCKS.items() for name, value in defaults.items()},
    'selection_capacity_lambda': 0.0,  # D2: + lambda * log(params_m) on the selection fitness
}

# train-block options (D2 - fitness proxy). Defaults reproduce the original proxy.
TRAIN_DEFAULTS = {
    'fitness_aggregation': 'max',      # 'max' over the evaluated epochs, or 'mean_last_k'
    'fitness_last_k': None,            # None: every evaluated epoch (epochs_to_eval)
    'limit_val_value': None,           # None: limit_data_value - train samples (1000)
    'search_data_augmentation': False, # augmentation during the search's fitness training
}

# The same defaults as flattened log_params keys (see diff_runs.flatten).
LOGGED_DEFAULTS = {f'QNAS.{key}': value for key, value in SEARCH_ENGINE_DEFAULTS.items()}
LOGGED_DEFAULTS.update({f'QNAS.{key}': value for key, value in V2_QNAS_DEFAULTS.items()})
LOGGED_DEFAULTS.update({f'train.{key}': value for key, value in TRAIN_DEFAULTS.items()})
