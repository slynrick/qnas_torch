"""diff_runs: compare the effective configuration of runs and configs."""

import os

import pytest
import yaml

import diff_runs
import qnas_config as cfg
from conftest import ROOT_DIR
from test_config import make_args

CONFIG = os.path.join(ROOT_DIR, 'configs', 'config_files_cifar', '01_deterministic_13-8-4.yml')


def write_log(directory, tree):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'log_params_evolution.txt').write_text(yaml.safe_dump(tree))
    return directory


class TestDiff:
    def test_flatten_expands_nested_dicts_only(self):
        tree = {'QNAS': {'a': 1, 'stages': [{'x': 1}], 'empty': {}}, 'fn_dict': {'op': {'k': 3}}}
        assert diff_runs.flatten(tree) == {
            'QNAS.a': 1, 'QNAS.stages': [{'x': 1}], 'QNAS.empty': {}, 'fn_dict.op.k': 3}

    def test_paths_and_provenance_are_ignored(self):
        a = {'QNAS': {'r': True}, 'files': {'config_file': 'a.yml'},
             'train': {'data_path': '/x', 'experiment_path': 'e1', 'phase': 'evolution',
                       'log_level': 'INFO', 'seed': 42}}
        b = {'QNAS': {'r': True}, 'files': {'config_file': 'b.yml', 'continue_path': 'e2'},
             'train': {'data_path': '/y', 'experiment_path': 'e2', 'phase': 'continue_evolution',
                       'log_level': 'NONE', 'seed': 42}}
        assert diff_runs.diff_params(a, b) == ({}, {}, {})

    def test_changed_and_one_sided_keys(self):
        a = {'QNAS': {'reset': True, 'old': 1}}
        b = {'QNAS': {'reset': False, 'engine': 'ancestor_decay'}}
        changed, only_a, only_b = diff_runs.diff_params(a, b)
        assert changed == {'QNAS.reset': (True, False)}
        assert only_a == {'QNAS.old': 1}
        assert only_b == {'QNAS.engine': 'ancestor_decay'}

    def test_main_exit_status_and_report(self, tmp_path, capsys):
        same_a = write_log(tmp_path / 'a', {'QNAS': {'reset': True}})
        same_b = write_log(tmp_path / 'b', {'QNAS': {'reset': True}})
        other = write_log(tmp_path / 'c', {'QNAS': {'reset': False}})
        assert diff_runs.main([str(same_a), str(same_b)]) == 0
        assert 'Same configuration' in capsys.readouterr().out
        assert diff_runs.main([str(same_a), str(other / 'log_params_evolution.txt')]) == 1
        assert 'QNAS.reset: True -> False' in capsys.readouterr().out


class TestRender:
    def test_render_matches_what_a_run_records(self, tmp_path):
        """A rendered .yml must equal the log_params_evolution.txt an evolve run writes,
        otherwise diffing a config against a run would report false differences."""
        config = cfg.ConfigParameters(make_args(CONFIG, tmp_path, en_pop_crossover=True),
                                      phase='evolution')
        config.get_parameters()
        config.save_params_logfile()
        rendered = diff_runs.render_config(CONFIG, '--en_pop_crossover')
        assert diff_runs.diff_params(rendered, diff_runs.load_params(str(tmp_path))) == ({}, {}, {})

    @pytest.mark.parametrize('flag, expected', [('', False), ('--en_pop_crossover', True)])
    def test_cli_flags_reach_the_rendered_qnas_block(self, tmp_path, flag, expected):
        # CONFIG itself sets QNAS.en_pop_crossover: True, which wins over the CLI flag
        # (see qnas_config.py) - render a variant without it, so the flag's effect on
        # the rendered block is what's actually under test here.
        with open(CONFIG) as f:
            data = yaml.safe_load(f)
        data['QNAS'].pop('en_pop_crossover', None)
        variant = tmp_path / 'variant.yml'
        variant.write_text(yaml.safe_dump(data))
        assert diff_runs.render_config(str(variant), flag)['QNAS']['en_pop_crossover'] is expected
