"""Options of docs/QNAS_SEARCH_ENGINE_IMPROVEMENT_PLAN_V2.md: surrogate screening (D1),
fitness proxy and two-rung fidelity (D2), sampling temperature and 'surrogate' pruning (D3),
niches (D4). The default path is covered by test_default_path_golden.py."""

import json

import numpy as np
import pytest

import diff_runs
import evaluation
import option_defaults
import surrogate
from architecture_cache import ArchitectureCache
from cnn.train import aggregate_fitness
from conftest import NOOP, FakeEval
from population import QPopulationNetwork
from test_config import load, write_variant
from test_default_path_golden import SCENARIOS, run_scenario, trace
from test_diff_runs import CONFIG

STAGES = [{'gen_start': 0, 'num_nodes': 3, 'num_ops': 5},
          {'gen_start': 5, 'num_nodes': 4, 'num_ops': 3}]


class FidelityEval(FakeEval):
    """FakeEval that accepts the two-rung keywords and exposes last_params_m."""

    def __init__(self, fn_list):
        super().__init__(fn_list)
        self.rungs = []
        self.last_params_m = None

    def __call__(self, decoded_params, decoded_nets, generation, fidelity_rung=None,
                 fidelity_epochs=None, id_offset=0, resume_ids=None):
        fitness, hits = super().__call__(decoded_params, decoded_nets, generation)
        self.rungs.append((fidelity_rung, len(decoded_nets), id_offset, resume_ids))
        self.last_params_m = np.array([0.5 + len(net) for net in decoded_nets], dtype=float)
        if fidelity_rung == 'low':
            fitness = fitness - 1.0  # a cheap rung reads lower, on the same ranking
        return fitness, hits


def with_eval(q, cls=FidelityEval):
    q.eval_func = cls(list(q.qpop_net.chromosome.fn_list[0]))
    return q


# ----------------------------------------------------------------------------- surrogate.py
class TestSurrogateModule:
    def test_one_hot_marks_one_op_per_node(self):
        x = surrogate.one_hot(np.array([[0, 2], [1, -1]]), [2, 3])
        np.testing.assert_array_equal(x, [[1, 0, 0, 0, 1], [0, 1, 0, 0, 0]])

    def test_ensemble_recovers_an_additive_signal(self):
        rng = np.random.default_rng(1)
        effects = rng.normal(size=(4, 3))
        net = rng.integers(0, 3, size=(300, 4))
        fit = effects[np.arange(4), net].sum(axis=1) + rng.normal(scale=0.05, size=300)
        model = surrogate.RidgeEnsembleSurrogate(rng=rng).fit(
            surrogate.one_hot(net[:200], [3] * 4), fit[:200])
        mu, sigma = model.predict(surrogate.one_hot(net[200:], [3] * 4))
        assert np.corrcoef(mu, fit[200:])[0, 1] > 0.95
        assert (sigma >= 0).all()
        coef = model.op_effects([3] * 4)
        assert [int(np.argmax(c)) for c in coef] == [int(np.argmax(e)) for e in effects]

    def test_screen_prefers_predicted_best_and_keeps_random_share(self):
        rng = np.random.default_rng(2)
        train = rng.integers(0, 3, size=(200, 4))
        train_fit = (train == 0).sum(axis=1).astype(float)
        cands = rng.integers(0, 3, size=(60, 4))
        idx, mu = surrogate.screen(cands, [3] * 4, train, train_fit, 10, tau=0.01, beta=0.0,
                                   random_frac=0.2, rng=rng)
        assert len(set(idx.tolist())) == 10
        guided = idx[:8]
        true = (cands == 0).sum(axis=1)
        assert true[guided].mean() > true.mean() + 0.5

    def test_screen_without_training_data_picks_at_random(self):
        rng = np.random.default_rng(3)
        idx, mu = surrogate.screen(np.zeros((5, 2), int), [2, 2], np.zeros((0, 2), int),
                                   np.zeros(0), 3, 1.0, 0.5, 0.2, rng)
        assert len(set(idx.tolist())) == 3 and np.isnan(mu).all()


# ----------------------------------------------------------------------------- D1 in QNAS
class TestSurrogateScreening:
    def test_unscreened_generations_replay_the_default_path(self, make_qnas):
        """Enabled but never past warm-up, with oversample 1: same RNG draws, same search."""
        expected = run_scenario(make_qnas, 'plain_default')
        np.random.seed(2026)
        q = make_qnas(**SCENARIOS['plain_default'], surrogate_enabled=True,
                      surrogate_warmup_evals=10 ** 9)
        q.evolve()
        got = trace(q)
        for gen in expected:
            np.testing.assert_array_equal(got[gen]['net_pop'], expected[gen]['net_pop'])
            np.testing.assert_array_equal(got[gen]['fitnesses'], expected[gen]['fitnesses'])

    def test_same_cost_mode_screens_a_bigger_pool(self, make_qnas):
        q = make_qnas(max_generations=8, surrogate_enabled=True, surrogate_oversample=4,
                      surrogate_warmup_evals=12, max_num_nodes=3)
        q.evolve()
        sizes = [n for n, _ in q.eval_func.calls]
        assert sizes == [6] * 8  # num_quantum_ind 3 x repetition 2, every generation
        pool, kept, tau = q.last_screening
        assert kept == 6 and pool > kept and 0 < tau <= 1.0

    def test_half_cost_mode_trains_fewer_and_keeps_the_population(self, make_qnas):
        q = make_qnas(max_generations=8, surrogate_enabled=True, surrogate_oversample=4,
                      surrogate_n_eval=3, surrogate_warmup_evals=6, max_num_nodes=3)
        q.evolve()
        assert [n for n, _ in q.eval_func.calls] == [3] * 8
        assert q.qpop_net.current_pop.shape[0] == 6

    def test_screening_does_not_touch_the_global_rng_beyond_sampling(self, make_qnas):
        """The surrogate draws from its own generator: two runs whose only difference is
        the surrogate seed consume the global RNG identically up to the first screening."""
        np.random.seed(5)
        a = make_qnas(max_generations=3, surrogate_enabled=True, surrogate_warmup_evals=6,
                      surrogate_oversample=2, surrogate_seed=1)
        a.evolve()
        state_a = np.random.get_state()[1].copy()
        np.random.seed(5)
        b = make_qnas(max_generations=3, surrogate_enabled=True, surrogate_warmup_evals=6,
                      surrogate_oversample=2, surrogate_seed=99)
        b.evolve()
        np.testing.assert_array_equal(state_a, np.random.get_state()[1])

    def test_progressive_screening_restarts_each_stage(self, make_qnas):
        q = make_qnas(max_generations=8, progressive_stages=STAGES, noop_fn_name=NOOP,
                      progressive_mode='deterministic', surrogate_enabled=True,
                      surrogate_oversample=3, surrogate_warmup_evals=12)
        q.evolve()
        assert q._stage_start_gen == 5
        assert all(net.shape[1] == 4 for net in q._stage_eval_net)

    @pytest.mark.parametrize('bad', [dict(surrogate_oversample=0), dict(surrogate_n_eval=1),
                                     dict(surrogate_n_eval=99), dict(surrogate_tau0=0),
                                     dict(surrogate_random_frac=1.5)])
    def test_invalid_options(self, make_qnas, bad):
        with pytest.raises(ValueError):
            make_qnas(surrogate_enabled=True, **bad)

    def test_rejected_with_dynamic_mode(self, make_qnas):
        with pytest.raises(ValueError, match='dynamic'):
            make_qnas(progressive_mode='dynamic', noop_fn_name=NOOP, surrogate_enabled=True,
                      dynamic_initial_num_nodes=2, dynamic_max_num_nodes=3)


# ----------------------------------------------------------------------------- D2
class TestFidelityAndProxy:
    def test_two_rungs_keep_the_best_of_the_cheap_rung(self, make_qnas):
        q = with_eval(make_qnas(max_generations=3, fidelity_enabled=True, fidelity_oversample=2,
                                fidelity_low_epochs=4))
        q.evolve()
        low = [r for r in q.eval_func.rungs if r[0] == 'low']
        resume = [r for r in q.eval_func.rungs if r[0] == 'resume']
        assert len(low) == len(resume) == 3
        assert all(n == 12 and off == q.FIDELITY_ID_OFFSET for _, n, off, _ in low)
        assert all(n == 6 and len(ids) == 6 and min(ids) >= q.FIDELITY_ID_OFFSET
                   for _, n, _, ids in resume)
        assert q.qpop_net.current_pop.shape[0] == 6

    def test_capacity_bonus_changes_selection_fitness_only(self, make_qnas):
        q = with_eval(make_qnas(selection_capacity_lambda=0.5))
        net = q.qpop_net.generate_classical()
        params = q.qpop_params.generate_classical()
        fit, raw = q.eval_pop(params, net)
        np.testing.assert_allclose(fit - raw, 0.5 * np.log(0.5 + net.shape[1]))

    def test_capacity_bonus_needs_parameter_counts(self, make_qnas):
        q = make_qnas(selection_capacity_lambda=0.5)
        with pytest.raises(RuntimeError, match='last_params_m'):
            q.eval_pop(q.qpop_params.generate_classical(), q.qpop_net.generate_classical())

    def test_aggregate_fitness(self):
        accs, losses = [70.0, 72.0, 71.0, 74.0], [1.0, 0.9, 0.95, 0.8]
        assert aggregate_fitness(accs, losses, 74.0, 0.8) == (74.0, 0.8)
        acc, loss = aggregate_fitness(accs, losses, 74.0, 0.8, 'mean_last_k', 2)
        assert acc == pytest.approx(72.5) and loss == pytest.approx(0.875)
        assert aggregate_fitness(accs, losses, 74.0, 0.8, 'mean_last_k')[0] == pytest.approx(71.75)

    def test_cache_protocol_isolates_fitness_definitions(self, tmp_path):
        cache = ArchitectureCache(str(tmp_path / 'cache.json'))
        cache.register(['a', 'b'], 70.0, 1.0, 1.0)  # original proxy: no protocol stored
        assert 'protocol' not in json.loads((tmp_path / 'cache.json').read_text())['a|b']
        assert cache.find_cached_result(['a', 'b'])['fitness'] == 70.0
        assert cache.find_cached_result(['a', 'b'], protocol='p1') is None
        cache.register(['c'], 60.0, 1.0, 1.0, protocol='p1')
        assert cache.find_cached_result(['c'], protocol='p1')['fitness'] == 60.0
        assert cache.find_cached_result(['c']) is None

    def test_cache_protocol_signature(self):
        base = dict(max_epochs=50, epochs_to_eval=5, limit_data_value=10000,
                    data_augmentation=False)
        assert evaluation.cache_protocol(base) is None
        assert evaluation.cache_protocol({**base, **option_defaults.TRAIN_DEFAULTS}) is None
        sig = evaluation.cache_protocol({**base, 'limit_val_value': 5000})
        assert 'limit_val_value=5000' in sig
        assert sig != evaluation.cache_protocol({**base, 'fitness_aggregation': 'mean_last_k'})


# ----------------------------------------------------------------------------- D3
class TestSamplingTemperature:
    def test_tempered_rows(self):
        probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]])
        np.testing.assert_allclose(QPopulationNetwork.tempered(probs, 1.0), probs)
        sharp = QPopulationNetwork.tempered(probs, 0.05)
        np.testing.assert_allclose(sharp.sum(axis=1), 1.0)
        assert sharp[0, 0] > 0.99 and sharp[1, 2] > 0.99
        flat = QPopulationNetwork.tempered(probs, 100.0)
        assert flat.max() - flat.min() < 0.02

    def test_schedule_anneals_within_each_stage(self, make_qnas):
        q = make_qnas(max_generations=8, progressive_stages=STAGES, noop_fn_name=NOOP,
                      progressive_mode='deterministic', sampling_tau0=1.0, sampling_tau_min=0.2,
                      sampling_tau_decay_gens=2)
        taus = {}
        original = q.go_next_gen

        def spy():
            taus[q.current_gen] = q.qpop_net.sampling_tau
            original()
        q.go_next_gen = spy
        q.evolve()
        assert taus[0] == 1.0 and taus[1] < taus[0] and taus[4] < taus[1]
        assert taus[5] == 1.0  # new stage restarts the schedule
        assert min(taus.values()) >= 0.2

    def test_surrogate_prune_keeps_the_ops_that_raise_fitness(self, make_qnas):
        q = make_qnas(progressive_stages=STAGES, noop_fn_name=NOOP,
                      progressive_mode='deterministic', prune_criterion='surrogate')
        rng = np.random.default_rng(0)
        net = rng.integers(0, 5, size=(200, 3))
        names = q.qpop_net.chromosome.fn_list[0]
        fit = (net == names.index('conv_5')).sum(axis=1) * 2.0 + \
            (net == names.index('skip')).sum(axis=1) * 1.0
        q._stage_eval_net, q._stage_eval_fit = [net], [fit]
        new = q._rank_and_prune_all_nodes(q.qpop_net.chromosome.fn_list, 3)
        assert all(set(n) == {'conv_5', 'skip', NOOP} for n in new)

    def test_surrogate_prune_falls_back_to_lift_without_data(self, make_qnas):
        q = make_qnas(progressive_stages=STAGES, noop_fn_name=NOOP,
                      progressive_mode='deterministic', prune_criterion='surrogate')
        scores = q._op_scores()
        assert len(scores) == 3  # lift scores, one array per node


# ----------------------------------------------------------------------------- D4
class TestNiches:
    def niche_q(self, make_qnas, **kw):
        q = make_qnas(replace_method='niche', **kw)
        return q

    def test_selection_gives_each_lineage_its_quota(self, make_qnas):
        q = self.niche_q(make_qnas)  # 3 quantum individuals, 6 classical -> quota 2
        q.fitnesses = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1.0])
        q.classic_ancestor = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
        q.classic_age = np.zeros(9, dtype=np.int64)
        keep = q._niche_selection(6)
        assert sorted(keep.tolist()) == [0, 1, 4, 5, 7, 8]

    def test_short_lineage_frees_its_slots_for_the_best_others(self, make_qnas):
        q = self.niche_q(make_qnas)
        q.fitnesses = np.array([9, 8, 7, 6, 5, 4, 3.0])
        q.classic_ancestor = np.array([0, 0, 0, 0, 1, 1, 1])  # lineage 2 empty
        q.classic_age = np.zeros(7, dtype=np.int64)
        assert sorted(q._niche_selection(6).tolist()) == [0, 1, 2, 3, 4, 5]

    def test_max_age_drops_old_members_but_never_the_best(self, make_qnas):
        q = self.niche_q(make_qnas, niche_max_age=3)
        q.fitnesses = np.array([9, 8, 7, 6, 5, 4, 3, 2.0])
        q.classic_ancestor = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        q.classic_age = np.array([10, 10, 0, 0, 0, 0, 0, 0])
        keep = q._niche_selection(6).tolist()
        assert 0 in keep and 1 not in keep

    def test_evolve_keeps_every_lineage_alive(self, make_qnas):
        q = self.niche_q(make_qnas, max_generations=8)
        q.evolve()
        assert q.qpop_net.current_pop.shape[0] == 6
        assert set(q.classic_ancestor.tolist()) == {0, 1, 2}

    def test_stalled_niche_restarts_to_uniform(self, make_qnas):
        q = self.niche_q(make_qnas, niche_restart_patience=2)
        for probs in q.qpop_net.probabilities:
            probs[1] = np.array([0.9, 0.05, 0.03, 0.01, 0.01])
        q.niche_best = np.array([5.0, 1.0, 3.0])
        q.niche_last_improve = np.array([0, 0, 0])
        q.current_gen = 4
        q._maybe_restart_niches()
        for probs in q.qpop_net.probabilities:
            np.testing.assert_allclose(probs[1], 0.2)
        assert q.niche_last_improve[1] == 4 and q.niche_last_improve[0] == 0  # best niche kept

    def test_collapsed_niche_restarts(self, make_qnas):
        q = self.niche_q(make_qnas, niche_restart_kl=0.01)
        for probs in q.qpop_net.probabilities:
            probs[:] = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
        q.niche_best = np.array([1.0, 5.0, 2.0])
        q._maybe_restart_niches()
        for probs in q.qpop_net.probabilities:
            np.testing.assert_allclose(probs[0], 0.2)
            np.testing.assert_allclose(probs[1], [0.6, 0.1, 0.1, 0.1, 0.1])

    def test_niche_options_need_niche_replacement(self, make_qnas):
        with pytest.raises(ValueError, match='niche'):
            make_qnas(niche_max_age=5)

    def test_niche_state_survives_a_resume(self, make_qnas, tmp_path):
        q = self.niche_q(make_qnas, max_generations=4, niche_restart_patience=50)
        q.evolve()
        best = q.niche_best.copy()
        q2 = self.niche_q(make_qnas, max_generations=6, niche_restart_patience=50)
        q2.data_file = str(tmp_path / 'resumed.pkl')
        q2.load_qnas_data(q.data_file)
        np.testing.assert_array_equal(q2.niche_best, best)


# ----------------------------------------------------------------------------- config
class TestV2Config:
    def test_defaults_when_config_omits_the_options(self, tmp_path):
        config = load(CONFIG, tmp_path)
        for key, value in option_defaults.V2_QNAS_DEFAULTS.items():
            assert config.QNAS_spec[key] == value
        for key, value in option_defaults.TRAIN_DEFAULTS.items():
            assert config.train_spec[key] == value

    def test_nested_blocks_become_flat_keys(self, tmp_path):
        def mutate(d):
            d['QNAS']['surrogate'] = {'enabled': True, 'oversample': 5, 'n_eval': 10}
            d['QNAS']['sampling_temperature'] = {'tau0': 1.0}
            d['QNAS']['replace_method'] = 'niche'
            d['QNAS']['niche'] = {'restart_patience': 30}
            d['QNAS']['fidelity'] = {'enabled': True, 'low_epochs': 10}
            d['train'].update(fitness_aggregation='mean_last_k', limit_val_value=5000,
                              search_data_augmentation=True)
        config = load(write_variant(tmp_path, mutate), tmp_path)
        spec = config.QNAS_spec
        assert (spec['surrogate_enabled'], spec['surrogate_oversample'], spec['surrogate_n_eval'],
                spec['sampling_tau0'], spec['niche_restart_patience'], spec['fidelity_enabled']) \
            == (True, 5, 10, 1.0, 30, True)
        assert 'surrogate' not in spec and 'niche' not in spec
        assert config.train_spec['data_augmentation'] is True
        assert config.train_spec['limit_val_value'] == 5000

    @pytest.mark.parametrize('mutate, match', [
        (lambda d: d['QNAS'].update(surrogate={'enable': True}), 'Unknown key'),
        (lambda d: d['train'].update(fitness_aggregation='median'), 'fitness_aggregation'),
        (lambda d: d['train'].update(fitness_last_k=99), 'fitness_last_k'),
        (lambda d: d['QNAS'].update(fidelity={'enabled': True, 'low_epochs': 999}), 'low_epochs'),
        (lambda d: (d['QNAS'].update(surrogate={'enabled': True}),
                    d['QNAS']['progressive'].update(mode='dynamic',
                                                    dynamic={'initial_num_nodes': 3})), 'dynamic'),
    ])
    def test_invalid_yaml(self, tmp_path, mutate, match):
        with pytest.raises(ValueError, match=match):
            load(write_variant(tmp_path, mutate), tmp_path)

    def test_run_log_predating_the_options_matches_a_new_render(self):
        new = diff_runs.render_config(CONFIG, '--en_pop_crossover')
        old = diff_runs.render_config(CONFIG, '--en_pop_crossover')
        for key in option_defaults.V2_QNAS_DEFAULTS:
            old['QNAS'].pop(key)
        for key in option_defaults.TRAIN_DEFAULTS:
            old['train'].pop(key)
        assert diff_runs.diff_params(old, new) == ({}, {}, {})


# ----------------------------------------------------------------------------- D2 training
class TestFidelityTraining:
    """cnn/train.py on CPU with a tiny network and random data: the cheap rung leaves a
    checkpoint and no cache entry; the second rung resumes from it and registers."""

    def params(self, tmp_path, **kw):
        p = dict(experiment_path=str(tmp_path), device='cpu', dataset='cifar10',
                 data_path=str(tmp_path), batch_size=16, max_epochs=4, epochs_to_eval=1,
                 mixed_precision=False, fitness_metric='best_accuracy', mo_metric_base='accuracy',
                 optimizer='AdamW', network_config='default', network_gap=False,
                 max_params=1e6, max_inference_time=1e6, cache_protocol=None)
        p.update(kw)
        return p

    def loaders(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        g = torch.Generator().manual_seed(0)
        x = torch.randn(32, 3, 32, 32, generator=g)
        y = torch.randint(0, 10, (32,), generator=g)
        data = TensorDataset(x, y)
        return DataLoader(data, batch_size=16), DataLoader(data, batch_size=16)

    def test_low_rung_then_resume(self, tmp_path):
        import os
        from cnn import train
        fn_dict = {'conv': {'function': 'ConvBlock',
                            'params': {'kernel': 3, 'strides': 1, 'filters': 4}},
                   'no_op': {'function': 'NoOp', 'params': {}}}
        cache = ArchitectureCache(str(tmp_path / 'cache.json'))
        train_loader, val_loader = self.loaders()

        low = [0.0] * 5
        train.fitness_calculation('0_1000', self.params(tmp_path, fidelity_rung='low', max_epochs=2),
                                  fn_dict, ['conv', 'no_op'], train_loader, val_loader, low,
                                  architecture_cache=cache)
        assert os.path.exists(tmp_path / '0_1000' / train.FIDELITY_CHECKPOINT)
        assert not os.path.exists(tmp_path / '0_1000' / 'best_model.pth')
        assert cache.find_cached_result(['conv', 'no_op']) is None
        assert low[4] > 0  # training seconds reported

        final = [0.0] * 5
        p = self.params(tmp_path, fidelity_rung='resume', resume_from='0_1000')
        train.fitness_calculation('0_0', p, fn_dict, ['conv', 'no_op'], train_loader,
                                  val_loader, final, architecture_cache=cache)
        assert p['start_epoch'] == 3
        assert os.path.exists(tmp_path / '0_0' / 'best_model.pth')
        assert cache.find_cached_result(['conv', 'no_op'])['fitness'] == pytest.approx(final[0])


# ----------------------------------------------------------------------------- v2 configs
V2_CONFIGS = sorted(__import__('glob').glob(
    __import__('os').path.join(__import__('conftest').ROOT_DIR, 'configs', 'config_files_cifar',
                               '[0-9][0-9]_v2_pq*.yml')))


@pytest.mark.e2e
@pytest.mark.parametrize('path', V2_CONFIGS, ids=lambda p: p.rsplit('/', 1)[-1])
def test_v2_config_evolves_end_to_end(path, tmp_path):
    """Every shipped plan-v2 config, shrunk to a few generations (stages every 3, warm-up
    of one generation), runs evolve() with every direction it enables."""
    import qnas
    spec = dict(load(path, tmp_path).QNAS_spec)
    spec['progressive_stages'] = [dict(s, gen_start=3 * i)
                                  for i, s in enumerate(spec['progressive_stages'])]
    spec.update(max_generations=9, surrogate_warmup_evals=20)
    q = qnas.QNAS(None, str(tmp_path), log_file=str(tmp_path / 'log_QNAS.txt'),
                  log_level='NONE', data_file=str(tmp_path / 'data_QNAS.pkl'))
    q.initialize_qnas(**spec)
    q.eval_func = FidelityEval(spec['fn_list'])
    q.evolve()
    assert q.current_gen == 9
    assert q.qpop_net.chromosome.num_genes == spec['progressive_stages'][-1]['num_nodes']
    num_classic = spec['num_quantum_ind'] * spec['repetition']
    assert q.qpop_net.current_pop.shape[0] == num_classic
    if spec['surrogate_enabled']:
        assert q.last_screening is not None
