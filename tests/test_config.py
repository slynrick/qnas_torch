"""ConfigParameters: every shipped YAML must load, and the option validation must bite."""

import copy
import glob
import os

import pytest
import yaml

import qnas
import qnas_config as cfg
from conftest import ROOT_DIR

CONFIGS = sorted(glob.glob(os.path.join(ROOT_DIR, 'configs', '*', '*.yml')))

# Configs from before the current schema (they lack keys such as train.network_gap,
# QNAS.crossover_frequency or QNAS.patience, or are not valid YAML). They still parse
# nowhere in the current code, so they are tracked here rather than silently skipped;
# non-strict xfail means fixing one is welcome, not an error. Every config NOT listed
# here must keep loading.
STALE_CONFIGS = {
    *(f'configs/config_files_atleta/config{i}.yml' for i in (0, 1, 2, 3, 4, 6, 7, 10)),
    *(f'configs/config_files_med/config{i}.yml' for i in (1, 4, 6, 7, 10)),
    *(f'configs/config_files_medmnist/config{i}.yml' for i in (1, 2, 3, 4)),
}


def _config_param(path):
    rel = os.path.relpath(path, ROOT_DIR)
    marks = []
    if rel in STALE_CONFIGS:
        marks.append(pytest.mark.xfail(
            reason='legacy config predating the current schema',
            raises=(KeyError, yaml.YAMLError), strict=False))
    return pytest.param(path, id=rel, marks=marks)


def make_args(config_file, experiment_path, **overrides):
    args = dict(
        config_file=config_file, experiment_path=str(experiment_path), data_path='unused',
        dataset='cifar10', fitness_metric='best_accuracy', optimizer='AdamW',
        data_augmentation=False, early_stopping=False, en_pop_crossover=False,
        save_checkpoints_epochs=5, limit_data_value=10000, network_gap=False,
        network_config='default', log_level='NONE',
    )
    args.update(overrides)
    return args


def load(config_file, tmp_path, phase='evolution', **overrides):
    config = cfg.ConfigParameters(make_args(config_file, tmp_path, **overrides), phase=phase)
    config.get_parameters()
    return config


@pytest.mark.parametrize('path', [_config_param(p) for p in CONFIGS])
def test_shipped_config_loads_and_builds_qnas(path, tmp_path):
    config = load(path, tmp_path)
    spec = config.QNAS_spec

    assert spec['fn_list'] and set(spec['fn_list']) == set(config.fn_dict)
    assert all('prob' not in d for d in config.fn_dict.values())  # popped by _get_fn_spec
    assert spec['quantum_update_engine'] in ('default', 'ancestor_decay')
    assert config.files_spec['data_file'].endswith('data_QNAS.pkl')

    # The parsed spec must be accepted verbatim by QNAS.initialize_qnas().
    q = qnas.QNAS(lambda *a, **k: None, str(tmp_path), log_file=str(tmp_path / 'log.txt'),
                  log_level='NONE', data_file=str(tmp_path / 'd.pkl'))
    q.initialize_qnas(**spec)
    assert q.qpop_net.chromosome.num_genes >= 1
    assert len(q.qpop_net.chromosome.fn_list) == q.qpop_net.chromosome.num_genes


def _progressive_config():
    return os.path.join(ROOT_DIR, 'configs', 'config_files_cifar', '01_deterministic_13-8-4.yml')


def _base_yaml():
    with open(_progressive_config()) as f:
        return yaml.safe_load(f)


def write_variant(tmp_path, mutate):
    data = _base_yaml()
    mutate(data)
    path = tmp_path / 'variant.yml'
    path.write_text(yaml.safe_dump(data))
    return str(path)


class TestProgressiveSettings:
    def test_deterministic_stages_are_sorted_and_activated(self, tmp_path):
        spec = load(_progressive_config(), tmp_path).QNAS_spec
        assert spec['progressive_mode'] == 'deterministic'
        gens = [s['gen_start'] for s in spec['progressive_stages']]
        assert gens == sorted(gens) and gens[0] == 0
        assert spec['noop_fn_name']  # required by progressive mode

    def test_enabled_false_turns_progressive_off(self, tmp_path):
        path = write_variant(tmp_path, lambda d: d['QNAS']['progressive'].update(enabled=False))
        spec = load(path, tmp_path).QNAS_spec
        assert spec['progressive_stages'] is None and spec['progressive_mode'] is None
        assert 'noop_fn_name' not in spec

    def test_invalid_mode(self, tmp_path):
        path = write_variant(tmp_path, lambda d: d['QNAS']['progressive'].update(mode='weird'))
        with pytest.raises(ValueError, match="'deterministic' or 'dynamic'"):
            load(path, tmp_path)

    def test_enabled_true_requires_stages(self, tmp_path):
        def mutate(d):
            d['QNAS']['progressive']['deterministic'] = {}
        with pytest.raises(ValueError, match='requires'):
            load(write_variant(tmp_path, mutate), tmp_path)

    def test_first_stage_must_start_at_zero(self, tmp_path):
        def mutate(d):
            d['QNAS']['progressive']['deterministic']['stages'][0]['gen_start'] = 5
        with pytest.raises(ValueError, match='gen_start=0'):
            load(write_variant(tmp_path, mutate), tmp_path)

    def test_dynamic_mode_needs_initial_num_nodes(self, tmp_path):
        def mutate(d):
            d['QNAS']['progressive'].update(mode='dynamic', dynamic={})
        with pytest.raises(ValueError, match='initial_num_nodes'):
            load(write_variant(tmp_path, mutate), tmp_path)

    @pytest.mark.parametrize('field, value, message', [
        ('check_every_gen', 3, 'multiple'),           # update_quantum_gen is 5
        ('probability_threshold', 0.0, 'probability_threshold'),
        ('probability_threshold', 1.5, 'probability_threshold'),
        ('min_ops', 0, 'min_ops'),
        ('flatness_epsilon', -1.0, 'flatness_epsilon'),
        ('growth_patience', 0, 'growth_patience'),
        ('node_growth_amount', 0, 'node_growth_amount'),
        ('max_num_nodes', 1, 'max_num_nodes'),        # below initial_num_nodes
    ])
    def test_dynamic_validation(self, tmp_path, field, value, message):
        def mutate(d):
            d['QNAS']['progressive'].update(mode='dynamic', dynamic={
                'initial_num_nodes': 4, 'max_num_nodes': 8, field: value})
        with pytest.raises(ValueError, match=message):
            load(write_variant(tmp_path, mutate), tmp_path)

    def test_dynamic_mode_populates_dynamic_spec(self, tmp_path):
        def mutate(d):
            d['QNAS']['progressive'].update(mode='dynamic', dynamic={
                'initial_num_nodes': 4, 'max_num_nodes': 9, 'min_ops': 3,
                'growth_patience': 2})
        spec = load(write_variant(tmp_path, mutate), tmp_path).QNAS_spec
        assert spec['progressive_mode'] == 'dynamic'
        assert spec['dynamic_initial_num_nodes'] == 4
        assert spec['dynamic_max_num_nodes'] == 9
        assert spec['dynamic_min_ops'] == 3
        assert spec['dynamic_growth_patience'] == 2
        assert spec['dynamic_check_every_gen'] == 5  # defaults to update_quantum_gen


class TestQuantumUpdateEngineSetting:
    def test_default_engine(self, tmp_path):
        spec = load(_progressive_config(), tmp_path).QNAS_spec
        assert spec['quantum_update_engine'] == 'default'
        assert spec['quantum_update_age_decay'] == 0.5

    def test_ancestor_decay_engine_and_decay_rate(self, tmp_path):
        def mutate(d):
            d['QNAS'].update(quantum_update_engine='ancestor_decay', quantum_update_age_decay=1.5)
        spec = load(write_variant(tmp_path, mutate), tmp_path).QNAS_spec
        assert spec['quantum_update_engine'] == 'ancestor_decay'
        assert spec['quantum_update_age_decay'] == 1.5

    def test_unknown_engine_rejected(self, tmp_path):
        path = write_variant(tmp_path, lambda d: d['QNAS'].update(quantum_update_engine='nope'))
        with pytest.raises(ValueError, match="'default' or 'ancestor_decay'"):
            load(path, tmp_path)

    # decay config -> its non-decay base, per each file's own header comment
    # (config_files_cifar/*.yml). 03's base predates the 01-08 renumbering and
    # is no longer in this directory (see 03's header) - mapped to None, so
    # there is nothing to diff it against, but it's still accounted for below.
    _ANCESTOR_DECAY_TWINS = {
        '02_deterministic_13-8-4_ancestor-decay.yml': '01_deterministic_13-8-4.yml',
        '03_deterministic_13-8-4_reset_ancestor-decay.yml': None,
        '05_deterministic_13-10-8_reset_ancestor-decay.yml': '04_deterministic_13-10-8_reset.yml',
        '08_dynamic_v2_ancestor-decay.yml': '07_dynamic_v2.yml',
    }

    def test_ancestor_decay_configs_only_differ_in_engine(self):
        base_dir = os.path.join(ROOT_DIR, 'configs', 'config_files_cifar')
        decay_files = {os.path.basename(p)
                        for p in glob.glob(os.path.join(base_dir, '*ancestor-decay.yml'))}
        assert decay_files == set(self._ANCESTOR_DECAY_TWINS), (
            'a config was added/renamed without updating _ANCESTOR_DECAY_TWINS')
        for decay_name, base_name in self._ANCESTOR_DECAY_TWINS.items():
            if base_name is None:
                continue
            path = os.path.join(base_dir, decay_name)
            twin = os.path.join(base_dir, base_name)
            with open(path) as f, open(twin) as g:
                a, b = yaml.safe_load(f), yaml.safe_load(g)
            a_q, b_q = copy.deepcopy(a['QNAS']), copy.deepcopy(b['QNAS'])
            assert a_q.pop('quantum_update_engine') == 'ancestor_decay'
            a_q.pop('quantum_update_age_decay', None)
            b_q.pop('quantum_update_engine', None)
            b_q.pop('quantum_update_age_decay', None)
            assert a_q == b_q, f'{os.path.basename(path)} drifted from {os.path.basename(twin)}'


class TestEnPopCrossoverSetting:
    def test_absent_from_config_falls_back_to_cli_arg(self, tmp_path):
        # _progressive_config() (01) itself sets QNAS.en_pop_crossover: True - use a
        # variant without the key, so the CLI-arg fallback is actually exercised.
        path = write_variant(tmp_path, lambda d: d['QNAS'].pop('en_pop_crossover', None))
        spec_default = load(path, tmp_path).QNAS_spec
        assert spec_default['en_pop_crossover'] is False
        spec_cli_on = load(path, tmp_path, en_pop_crossover=True).QNAS_spec
        assert spec_cli_on['en_pop_crossover'] is True

    def test_config_true_wins_over_cli_arg_false(self, tmp_path):
        path = write_variant(tmp_path, lambda d: d['QNAS'].update(en_pop_crossover=True))
        spec = load(path, tmp_path, en_pop_crossover=False).QNAS_spec
        assert spec['en_pop_crossover'] is True

    def test_config_false_wins_over_cli_arg_true(self, tmp_path):
        path = write_variant(tmp_path, lambda d: d['QNAS'].update(en_pop_crossover=False))
        spec = load(path, tmp_path, en_pop_crossover=True).QNAS_spec
        assert spec['en_pop_crossover'] is False


class TestValidation:
    def test_missing_required_variable(self, tmp_path):
        path = write_variant(tmp_path, lambda d: d['QNAS'].pop('repetition'))
        with pytest.raises(KeyError, match='repetition'):
            load(path, tmp_path)

    def test_wrong_type(self, tmp_path):
        path = write_variant(tmp_path, lambda d: d['QNAS'].update(repetition='four'))
        with pytest.raises(TypeError, match='repetition'):
            load(path, tmp_path)

    def test_epochs_to_eval_must_be_below_max_epochs(self, tmp_path):
        def mutate(d):
            d['train']['epochs_to_eval'] = d['train']['max_epochs']
        with pytest.raises(ValueError, match='epochs_to_eval'):
            load(write_variant(tmp_path, mutate), tmp_path)

    def test_params_range_out_of_bounds(self, tmp_path):
        path = write_variant(
            tmp_path, lambda d: d['QNAS']['params_ranges'].update(learning_rate=[1e-9, 0.5]))
        with pytest.raises(ValueError, match='out of bound'):
            load(path, tmp_path)

    def test_unknown_function_rejected(self, tmp_path):
        def mutate(d):
            name = next(iter(d['QNAS']['function_dict']))
            d['QNAS']['function_dict'][name]['function'] = 'DoesNotExist'
        with pytest.raises(ValueError, match='not a valid function'):
            load(write_variant(tmp_path, mutate), tmp_path)

    def test_fixed_params_are_not_evolved(self, tmp_path):
        spec_config = load(_progressive_config(), tmp_path)
        ranges = spec_config.QNAS_spec['params_ranges']
        # 01_deterministic_13-8-4.yml pins every hyperparameter to a scalar
        assert all(isinstance(v, list) for v in ranges.values())
        for name in ('learning_rate', 'weight_decay'):
            if name not in ranges:
                assert name in spec_config.train_spec


class TestParamsLogRoundtrip:
    def test_saved_log_can_be_reloaded_for_continue_and_retrain(self, tmp_path):
        first = load(_progressive_config(), tmp_path)
        first.save_params_logfile()
        log = tmp_path / 'log_params_evolution.txt'
        assert log.is_file() and log.stat().st_size > 0

        resumed = load(_progressive_config(), tmp_path, phase='continue_evolution',
                       continue_path=str(tmp_path))
        assert resumed.QNAS_spec['num_quantum_ind'] == first.QNAS_spec['num_quantum_ind']
        assert resumed.QNAS_spec['max_generations'] == first.QNAS_spec['max_generations']
        assert set(resumed.fn_dict) == set(first.fn_dict)
        assert list(resumed.QNAS_spec['params_ranges']) == list(first.QNAS_spec['params_ranges'])


def test_stale_config_list_only_names_existing_files():
    existing = {os.path.relpath(p, ROOT_DIR) for p in CONFIGS}
    assert STALE_CONFIGS <= existing, sorted(STALE_CONFIGS - existing)
