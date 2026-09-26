"""Improvement 1 of docs/QNAS_SEARCH_ENGINE_IMPROVEMENT_PLAN.md: rank-based quantum update
(quantum_lr and friends) and the 'lift'/'empirical' progressive prune criteria. The default
path is covered by test_default_path_golden.py."""

import numpy as np
import pytest
import yaml

import diff_runs
import option_defaults
import util
from architecture_cache import ArchitectureCache
from conftest import FN_LIST, NOOP
from population import QPopulationNetwork
from test_config import load, write_variant
from test_diff_runs import CONFIG, write_log

RANKED = dict(quantum_lr=0.2, update_quantum_rate=1.0, update_quantum_gen=1)
STAGES = [{'gen_start': 0, 'num_nodes': 3, 'num_ops': 5},
          {'gen_start': 4, 'num_nodes': 4, 'num_ops': 3}]


def net_pop(num_nodes=2, **kw):
    pop = QPopulationNetwork(num_quantum_ind=2, max_num_nodes=num_nodes, repetition=2,
                             update_quantum_rate=1.0, fn_list=FN_LIST, initial_probs=[], **kw)
    pop.current_pop = np.array([[0, 1], [2, 1], [3, 3], [4, 4]])
    return pop


class TestRankedUpdate:
    def test_moves_lr_fraction_toward_target(self):
        pop = net_pop()
        before = pop.probabilities[0][0].copy()
        pop.update_quantum_ranked(0.5, [(np.array([0]), np.array([1.0]), None), None])
        expected = 0.5 * before
        expected[0] += 0.5
        np.testing.assert_allclose(pop.probabilities[0][0], expected)
        np.testing.assert_allclose(pop.probabilities[0][1], before)  # no targets -> untouched

    def test_weights_below_one_shrink_the_step(self):
        pop = net_pop()
        before = pop.probabilities[0][0].copy()
        pop.update_quantum_ranked(0.5, [(np.array([0]), np.array([0.5]), None), None])
        assert pop.probabilities[0][0][0] == pytest.approx(0.75 * before[0] + 0.25)

    def test_negative_learning_lowers_the_poor_op(self):
        pop = net_pop()
        pop.update_quantum_ranked(0.1, [(np.array([0]), np.array([1.0]), 2), None],
                                  negative_lr=0.5)
        probs = pop.probabilities[0][0]
        assert probs[3] < probs[1]  # op 3 (worst row) pushed below an untouched op
        assert probs.sum() == pytest.approx(1.0)

    def test_floor_and_ceiling_hold_and_rows_stay_normalized(self):
        pop = net_pop(max_prob=0.7)
        for _ in range(50):
            pop.update_quantum_ranked(1.0, [(np.array([0]), np.array([1.0]), None)] * 2,
                                      prob_floor=0.5)
        for probs in pop.probabilities:
            assert np.allclose(probs.sum(axis=1), 1.0)
            assert probs.min() >= 0.5 / len(FN_LIST) - 1e-12
            assert probs.max() <= 0.7 + 1e-12

    def test_single_op_node_stays_certain(self):
        assert QPopulationNetwork._bound_probs(np.array([1.0]), 0.1, 0.9).tolist() == [1.0]


class TestOptionValidation:
    @pytest.mark.parametrize('kw', [dict(quantum_top_k=2), dict(quantum_prob_floor=0.1),
                                    dict(quantum_negative_lr=0.1),
                                    dict(quantum_rank_weighting='linear')])
    def test_rank_options_require_quantum_lr(self, make_qnas, kw):
        with pytest.raises(ValueError, match='set quantum_lr'):
            make_qnas(**kw)

    @pytest.mark.parametrize('kw', [dict(quantum_lr=0.0), dict(quantum_lr=1.5),
                                    dict(quantum_lr=0.1, quantum_rank_weighting='x'),
                                    dict(quantum_lr=0.1, quantum_top_k=0),
                                    dict(quantum_lr=0.1, quantum_prob_floor=1.0),
                                    dict(prune_criterion='x')])
    def test_out_of_range_values_rejected(self, make_qnas, kw):
        with pytest.raises(ValueError):
            make_qnas(**kw)

    def test_non_pmf_prune_rejected_in_dynamic_mode(self, make_qnas):
        with pytest.raises(ValueError, match='deterministic'):
            make_qnas(progressive_mode='dynamic', noop_fn_name=NOOP, prune_criterion='lift',
                      dynamic_initial_num_nodes=2, dynamic_max_num_nodes=4)


class TestRankedTargets:
    def prepared(self, make_qnas, **kw):
        q = make_qnas(**RANKED, **kw)
        q.generate_classical()
        return q

    @pytest.mark.parametrize('weighting', ['uniform', 'linear', 'nes'])
    def test_weights_sum_to_one_best_first(self, make_qnas, weighting):
        q = make_qnas(**RANKED, quantum_rank_weighting=weighting)
        w = q._rank_weights(3)
        assert w.sum() == pytest.approx(1.0)
        assert np.all(np.diff(w) <= 0)

    def test_default_engine_targets_strided_ranks(self, make_qnas):
        q = self.prepared(make_qnas, quantum_top_k=2)
        n, num_ind = q.qpop_net.current_pop.shape[0], q.qpop_net.num_ind
        for idx, (rows, weights, worst) in enumerate(q._ranked_update_targets()):
            assert rows.tolist() == [idx, idx + num_ind]
            assert worst == (None if n - 1 - idx in rows else n - 1 - idx)

    def test_ancestor_engine_targets_own_lineage_best_first(self, make_qnas):
        q = self.prepared(make_qnas, quantum_top_k=2, quantum_update_engine='ancestor_decay')
        for idx, target in enumerate(q._ranked_update_targets()):
            if target is None:
                continue
            rows, weights, worst = target
            assert np.all(q.classic_ancestor[rows] == idx)
            assert rows.tolist() == sorted(rows.tolist())  # current_pop is best first
            assert weights.sum() <= 1.0 + 1e-12


class TestPruneCriteria:
    def biased(self, make_qnas, criterion):
        """Stage 0 prior puts most mass on 'pool'; the quantum individuals then move a
        little toward 'conv_5' - pmf keeps the prior favorite, lift the learned one."""
        prior = [0.1, 0.1, 0.5, 0.1, 0.2]  # conv_3, conv_5, pool, skip, no_op
        q = make_qnas(progressive_stages=STAGES, noop_fn_name=NOOP,
                      progressive_mode='deterministic', initial_probs=prior,
                      prune_criterion=criterion)
        for probs in q.qpop_net.probabilities:
            probs[:] = [0.1, 0.2, 0.45, 0.05, 0.2]
        return q

    def test_pmf_keeps_prior_favorite(self, make_qnas):
        q = self.biased(make_qnas, 'pmf')
        new = q._rank_and_prune_all_nodes(q.qpop_net.chromosome.fn_list, 3)
        assert all(names == ['conv_5', 'pool', NOOP] for names in new)

    def test_lift_ranks_by_change_from_stage_start(self, make_qnas):
        q = self.biased(make_qnas, 'lift')
        new = q._rank_and_prune_all_nodes(q.qpop_net.chromosome.fn_list, 3)
        assert all(names == ['conv_3', 'conv_5', NOOP] for names in new)
        glob = q._rank_and_prune_globally(q.qpop_net.chromosome.fn_list, 3)
        assert glob[0] == ['conv_3', 'conv_5', NOOP]

    def test_empirical_ranks_by_fitness_of_evaluated_individuals(self, make_qnas):
        q = make_qnas(progressive_stages=STAGES, noop_fn_name=NOOP,
                      progressive_mode='deterministic', prune_criterion='empirical')
        # skip (3) always in the best individuals, pool (2) in the worst
        q._stage_eval_net = [np.array([[3, 3, 3], [3, 0, 3], [2, 2, 2], [2, 1, 2]])]
        q._stage_eval_fit = [np.array([9.0, 8.0, 1.0, 2.0])]
        new = q._rank_and_prune_all_nodes(q.qpop_net.chromosome.fn_list, 2)
        assert new[0] == ['skip', NOOP] and new[2] == ['skip', NOOP]

    def test_empirical_falls_back_to_cache_then_lift(self, make_qnas, tmp_path):
        q = make_qnas(progressive_stages=STAGES, noop_fn_name=NOOP,
                      progressive_mode='deterministic', prune_criterion='empirical')
        assert q._empirical_op_scores() is None  # no record, no cache
        q.eval_func.architecture_cache = ArchitectureCache(str(tmp_path / 'c' / 'cache.json'))
        q.eval_func.architecture_cache.register(['skip', 'skip', 'skip'], 9.0, 0.1, 1.0)
        q.eval_func.architecture_cache.register(['pool', 'pool', 'pool'], 1.0, 0.1, 1.0)
        q.eval_func.architecture_cache.register(['pool', 'pool'], 99.0, 0.1, 1.0)  # other depth
        scores = q._empirical_op_scores()
        names = q.qpop_net.chromosome.fn_list[0]
        assert scores[0][names.index('skip')] > scores[0][names.index('pool')]

    def test_stage_record_restarts_after_transition(self, make_qnas):
        q = make_qnas(**RANKED, progressive_stages=STAGES, noop_fn_name=NOOP,
                      progressive_mode='deterministic', prune_criterion='empirical',
                      max_generations=4)
        q.evolve()
        assert len(q._stage_eval_net) == 4
        q._transition_stage(1)
        assert q._stage_eval_net == []
        assert len(q._stage_start_probs) == 4


class TestCheckpointCompat:
    def test_stage_start_probs_saved_and_restored(self, make_qnas):
        q = make_qnas(**RANKED, max_generations=3)
        q.evolve()
        entry = util.load_pkl(q.data_file)[2]
        assert {'stage_start_probs', 'pmf_entropy', 'pmf_kl_stage_start'} <= set(entry)
        resumed = make_qnas(**RANKED, max_generations=5)
        resumed.load_qnas_data(q.data_file)
        np.testing.assert_array_equal(resumed._stage_start_probs[0], entry['stage_start_probs'][0])

    def test_old_checkpoint_without_new_keys_resumes(self, make_qnas):
        q = make_qnas(progressive_stages=STAGES, noop_fn_name=NOOP,
                      progressive_mode='deterministic', max_generations=6,
                      reset_probs_on_stage_change=True)
        q.evolve()
        data = util.load_pkl(q.data_file)
        for entry in data.values():
            for key in ('stage_start_probs', 'pmf_entropy', 'pmf_kl_stage_start'):
                entry.pop(key)
        q.dump_pkl_data(data)
        resumed = make_qnas(progressive_stages=STAGES, noop_fn_name=NOOP,
                            progressive_mode='deterministic', max_generations=8,
                            reset_probs_on_stage_change=True, prune_criterion='lift')
        resumed.load_qnas_data(q.data_file)
        # stage 1 after a reset: uniform over each node's menu
        for probs, ref in zip(resumed.qpop_net.probabilities, resumed._stage_start_probs):
            np.testing.assert_allclose(ref, 1.0 / probs.shape[1])
        resumed.evolve()
        assert resumed.current_gen == 8


@pytest.mark.e2e
class TestRankedEvolve:
    @pytest.mark.parametrize('engine', ['default', 'ancestor_decay'])
    @pytest.mark.parametrize('criterion', ['pmf', 'lift', 'empirical'])
    def test_progressive_run_completes_with_valid_pmf(self, make_qnas, engine, criterion):
        q = make_qnas(**RANKED, quantum_top_k=2, quantum_rank_weighting='linear',
                      quantum_prob_floor=0.2, quantum_negative_lr=0.05,
                      quantum_update_engine=engine, prune_criterion=criterion,
                      progressive_stages=STAGES, noop_fn_name=NOOP,
                      progressive_mode='deterministic', max_generations=8)
        q.evolve()
        assert q.qpop_net.chromosome.num_genes == 4
        for probs, names in zip(q.qpop_net.probabilities, q.qpop_net.chromosome.fn_list):
            assert probs.shape[1] == len(names)
            assert np.allclose(probs.sum(axis=1), 1.0)
            assert probs.min() >= 0.2 / len(names) - 1e-12

    def test_rank_update_moves_the_pmf_more_than_the_original_rule(self, make_qnas):
        def kl_after(**kw):
            np.random.seed(7)
            q = make_qnas(max_generations=12, **kw)
            q.evolve()
            return q._pmf_stats()[1]
        original = kl_after(update_quantum_rate=0.1, update_quantum_gen=5)
        assert kl_after(**RANKED) > 5 * original


class TestConfigAndDiff:
    def test_defaults_when_config_omits_the_options(self, tmp_path):
        spec = load(CONFIG, tmp_path).QNAS_spec
        for key, value in option_defaults.SEARCH_ENGINE_DEFAULTS.items():
            assert spec[key] == value

    def test_options_read_from_yaml(self, tmp_path):
        def mutate(d):
            d['QNAS'].update(quantum_lr=0.05, quantum_top_k=2, quantum_prob_floor=0.2)
            d['QNAS']['progressive']['prune_criterion'] = 'lift'
        spec = load(write_variant(tmp_path, mutate), tmp_path).QNAS_spec
        assert (spec['quantum_lr'], spec['quantum_top_k'], spec['prune_criterion']) == \
            (0.05, 2, 'lift')

    def test_non_pmf_prune_rejected_for_dynamic_yaml(self, tmp_path):
        def mutate(d):
            d['QNAS']['progressive'].update(mode='dynamic', prune_criterion='lift',
                                            dynamic={'initial_num_nodes': 3})
        with pytest.raises(ValueError, match='deterministic'):
            load(write_variant(tmp_path, mutate), tmp_path)

    def test_logged_defaults_roundtrip_through_the_params_printer(self, tmp_path):
        """LOGGED_DEFAULTS must equal what a run writes to log_params_evolution.txt."""
        config = load(CONFIG, tmp_path)
        config.save_params_logfile()
        logged = diff_runs.flatten(diff_runs.load_params(str(tmp_path)))
        for key, value in option_defaults.LOGGED_DEFAULTS.items():
            assert logged[key] == value

    def test_run_log_predating_the_options_matches_a_new_render(self, tmp_path):
        new = diff_runs.render_config(CONFIG, '--en_pop_crossover')
        old = yaml.safe_load(yaml.safe_dump(new))
        for key in option_defaults.SEARCH_ENGINE_DEFAULTS:
            old['QNAS'].pop(key)
        assert diff_runs.diff_params(old, new) == ({}, {}, {})
        new['QNAS']['quantum_lr'] = 0.05
        changed, only_a, only_b = diff_runs.diff_params(old, new)
        assert changed == {'QNAS.quantum_lr': (None, 0.05)} and not only_a and not only_b

    def test_main_on_logs(self, tmp_path, capsys):
        a = write_log(tmp_path / 'a', {'QNAS': {'reset': True}})
        b = write_log(tmp_path / 'b', {'QNAS': {'reset': True, 'prune_criterion': 'pmf'}})
        assert diff_runs.main([str(a), str(b)]) == 0
        assert 'Same configuration' in capsys.readouterr().out

