import json
import logging
import os
import pickle

import pytest
import yaml

import util


class TestNaturalKey:
    def test_sorts_numbers_numerically(self):
        names = ['1_10', '1_2', '1_1', '1_13', '1_5']
        assert sorted(names, key=util.natural_key) == ['1_1', '1_2', '1_5', '1_10', '1_13']

    def test_mixed_text(self):
        names = ['conv_3_1_128', 'conv_3_1_64', 'conv_1_1_32']
        assert sorted(names, key=util.natural_key) == \
            ['conv_1_1_32', 'conv_3_1_64', 'conv_3_1_128']


class TestFileHelpers:
    def test_load_yaml(self, tmp_path):
        path = tmp_path / 'a.yml'
        path.write_text('a: 1\nb: [1, 2]\n')
        assert util.load_yaml(str(path)) == {'a': 1, 'b': [1, 2]}

    def test_load_pkl(self, tmp_path):
        path = tmp_path / 'a.pkl'
        path.write_bytes(pickle.dumps({0: [1, 2]}))
        assert util.load_pkl(str(path)) == {0: [1, 2]}

    def test_load_pkl_merges_several_appended_records(self, tmp_path):
        path = tmp_path / 'a.pkl'
        with open(path, 'ab') as f:
            pickle.dump({0: 'first'}, f)
            pickle.dump({1: 'second'}, f)
        assert util.load_pkl(str(path)) == {0: 'first', 1: 'second'}

    def test_load_pkl_later_record_wins_for_the_same_key(self, tmp_path):
        path = tmp_path / 'a.pkl'
        with open(path, 'ab') as f:
            pickle.dump({0: 'stale'}, f)
            pickle.dump({0: 'fresh'}, f)
        assert util.load_pkl(str(path)) == {0: 'fresh'}

    def test_create_info_file_roundtrip(self, tmp_path):
        util.create_info_file(str(tmp_path), {'x': 1, 'y': 'z'})
        assert yaml.safe_load((tmp_path / 'data_info.txt').read_text()) == {'x': 1, 'y': 'z'}

    def test_save_results_file_writes_json(self, tmp_path):
        util.save_results_file(str(tmp_path), {'run1': {'test_accuracy': 0.9}})
        with open(tmp_path / 'retrain_results.txt') as f:
            assert json.load(f) == {'run1': {'test_accuracy': 0.9}}

    def test_check_file_exists(self, tmp_path):
        (tmp_path / 'f').write_text('x')
        assert util.check_file_exists(str(tmp_path / 'f')) is True
        assert util.check_file_exists(str(tmp_path / 'g')) is False


def make_experiment(root, with_training=True, with_params=True, gen_ind='3_1'):
    folder = root / gen_ind
    folder.mkdir(parents=True)
    if with_training:
        (folder / 'training_params.txt').write_text(yaml.safe_dump(
            {'net_list': ['conv_3', 'no_op'], 'generation': 3, 'individual': 1,
             'best_accuracy': 0.81}))
    if with_params:
        (root / 'log_params_evolution.txt').write_text('QNAS: {}\n')
    return root


class TestCheckFiles:
    def test_valid_experiment_passes(self, tmp_path):
        util.check_files(str(make_experiment(tmp_path)))

    def test_missing_directory(self, tmp_path):
        with pytest.raises(OSError, match='valid'):
            util.check_files(str(tmp_path / 'nope'))

    def test_missing_training_params(self, tmp_path):
        with pytest.raises(OSError, match='training_params.txt not found'):
            util.check_files(str(make_experiment(tmp_path, with_training=False)))

    def test_empty_training_params(self, tmp_path):
        make_experiment(tmp_path)
        (tmp_path / '3_1' / 'training_params.txt').write_text('')
        with pytest.raises(OSError, match='valid data file'):
            util.check_files(str(tmp_path))

    def test_missing_log_params(self, tmp_path):
        with pytest.raises(OSError, match='log_params_evolution.txt not found'):
            util.check_files(str(make_experiment(tmp_path, with_params=False)))

    def test_empty_log_params(self, tmp_path):
        make_experiment(tmp_path)
        (tmp_path / 'log_params_evolution.txt').write_text('')
        with pytest.raises(OSError, match='valid config_file'):
            util.check_files(str(tmp_path))


class TestLoadEvolvedData:
    def test_reads_best_individual(self, tmp_path):
        make_experiment(tmp_path)
        assert util.load_evolved_data(str(tmp_path)) == {
            'net': ['conv_3', 'no_op'], 'generation': 3, 'individual': 1,
            'best_accuracy': 0.81}

    def test_old_format_recovers_ids_from_folder_name(self, tmp_path):
        folder = tmp_path / '12_4'
        folder.mkdir()
        (folder / 'training_params.txt').write_text(yaml.safe_dump({'net_list': ['a']}))
        data = util.load_evolved_data(str(tmp_path))
        assert (data['generation'], data['individual']) == (12, 4)

    def test_log_params_evolution(self, tmp_path):
        (tmp_path / 'log_params_evolution.txt').write_text(yaml.safe_dump(
            {'train': {'a': 1}, 'QNAS': {'b': 2}, 'fn_dict': {'c': 3}}))
        assert util.load_log_params_evolution(str(tmp_path)) == {
            'train_spec': {'a': 1}, 'QNAS_spec': {'b': 2}, 'fn_dict': {'c': 3}}

    def test_log_params_evolution_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            util.load_log_params_evolution(str(tmp_path))


class TestDeleteOldDirs:
    def make(self, tmp_path):
        for name in ('0_0', '0_1', '1_0', 'retrain_run1_1'):
            (tmp_path / name).mkdir()
            (tmp_path / name / 'model.pth').write_text('w')
        (tmp_path / 'cache.json').write_text('{}')

    def test_deletes_only_digit_prefixed_dirs(self, tmp_path):
        self.make(tmp_path)
        util.delete_old_dirs(str(tmp_path))
        assert sorted(os.listdir(tmp_path)) == ['cache.json', 'retrain_run1_1']

    def test_keep_best(self, tmp_path):
        self.make(tmp_path)
        util.delete_old_dirs(str(tmp_path), keep_best=True, best_id='0_1')
        assert sorted(os.listdir(tmp_path)) == ['0_1', 'cache.json', 'retrain_run1_1']


class TestCalculateTime:
    def test_elapsed_only(self):
        assert util.calculate_time(0, 3 * 3600 + 25 * 60 + 10) == (3, 25)

    def test_estimates_remaining_from_average_generation_time(self):
        # 10 generations in 1h -> 6 min/gen -> 90 remaining gens = 9h
        assert util.calculate_time(0, 3600, current_gen=10, max_generations=100,
                                   end_evol=False) == (1, 0, 9, 0)

    def test_generation_zero_has_no_estimate(self):
        assert util.calculate_time(0, 60, current_gen=0, end_evol=False) == (0, 1, 0, 0)


class TestInitLog:
    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        for name in ('t_info', 't_debug', 't_none', 't_file'):
            logging.getLogger(name).handlers.clear()

    def test_levels(self):
        assert util.init_log('INFO', 't_info').level == logging.INFO
        assert util.init_log('DEBUG', 't_debug').level == logging.DEBUG
        assert util.init_log('NONE', 't_none').level == logging.NOTSET

    def test_writes_to_file(self, tmp_path):
        path = tmp_path / 'run.log'
        logger = util.init_log('INFO', 't_file', file_path=str(path))
        logger.info('hello world')
        for handler in logger.handlers:
            handler.flush()
        assert 'hello world' in path.read_text()
