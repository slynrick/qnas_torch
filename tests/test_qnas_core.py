import numpy as np
import pytest

import qnas
import util
from conftest import FN_LIST, NOOP, PARAMS_RANGES, concentrated_probs


def pop_arrays(n, nodes=4):
    net = np.random.randint(0, len(FN_LIST), size=(n, nodes))
    params = np.random.rand(n, len(PARAMS_RANGES))
    return params, net


class TestOrderPop:
    def test_sorts_descending_and_keeps_rows_aligned(self):
        fit = np.array([1.0, 3.0, 2.0])
        raw = fit * 10
        params = np.array([[1.], [3.], [2.]])
        net = np.array([[1], [3], [2]])
        f, r, p, n = qnas.QNAS.order_pop(fit, raw, params, net)
        assert f.tolist() == [3.0, 2.0, 1.0]
        assert r.tolist() == [30.0, 20.0, 10.0]
        assert p.ravel().tolist() == n.ravel().tolist() == [3, 2, 1]

    def test_selection_truncates(self):
        fit = np.array([1.0, 3.0, 2.0])
        f, r, p, n = qnas.QNAS.order_pop(fit, fit, fit[:, None], fit[:, None], selection=range(2))
        assert f.tolist() == [3.0, 2.0]

    def test_extra_arrays_follow_the_same_permutation(self):
        fit = np.array([1.0, 3.0, 2.0])
        age = np.array([10, 30, 20])
        anc = np.array([0, 1, 2])
        *_, (age_o, anc_o) = qnas.QNAS.order_pop(
            fit, fit, fit[:, None], fit[:, None], selection=range(3), extra=[age, anc])
        assert age_o.tolist() == [30, 20, 10]
        assert anc_o.tolist() == [1, 2, 0]

    def test_without_extra_returns_four_values(self):
        fit = np.array([1.0, 2.0])
        assert len(qnas.QNAS.order_pop(fit, fit, fit[:, None], fit[:, None])) == 4


class TestBestId:
    def test_improvement_updates_best_so_far_id(self, make_qnas):
        q = make_qnas()
        q.current_gen = 3
        q.best_so_far = 1.0
        q.update_best_id(np.array([0.5, 2.0, 1.5]))
        assert q.current_best_id == [3, 1]
        assert q.best_so_far_id == [3, 1]

    def test_no_improvement_keeps_previous_best_id(self, make_qnas):
        q = make_qnas()
        q.current_gen = 4
        q.best_so_far = 5.0
        q.best_so_far_id = [1, 0]
        q.update_best_id(np.array([0.5, 2.0]))
        assert q.current_best_id == [4, 1]
        assert q.best_so_far_id == [1, 0]


class TestInitializeQnas:
    def test_defaults(self, make_qnas):
        q = make_qnas()
        assert q.quantum_update_engine == 'default'
        assert q.qpop_net.chromosome.num_genes == 4
        assert q.qpop_params.chromosome.num_genes == len(PARAMS_RANGES)
        assert q.penalties.shape == (6,)  # reducing_fns_list given: num_ind * repetition
        assert q.reducing_fns_names == {'pool'}

    def test_progressive_first_stage_must_use_full_op_menu(self, make_qnas):
        stages = [{'gen_start': 0, 'num_nodes': 3, 'num_ops': 2}]
        with pytest.raises(ValueError, match='num_ops must equal len'):
            make_qnas(progressive_stages=stages, noop_fn_name=NOOP)

    def test_progressive_requires_noop_in_fn_list(self, make_qnas):
        stages = [{'gen_start': 0, 'num_nodes': 3, 'num_ops': len(FN_LIST)}]
        with pytest.raises(ValueError, match='noop_fn_name'):
            make_qnas(progressive_stages=stages, noop_fn_name=None)
        with pytest.raises(ValueError, match='noop_fn_name'):
            make_qnas(progressive_stages=stages, noop_fn_name='missing')

    def test_progressive_stage_needs_positive_nodes(self, make_qnas):
        stages = [{'gen_start': 0, 'num_nodes': 0, 'num_ops': len(FN_LIST)}]
        with pytest.raises(ValueError, match='num_nodes'):
            make_qnas(progressive_stages=stages, noop_fn_name=NOOP)

    def test_progressive_stage_zero_sets_initial_depth(self, make_qnas):
        stages = [{'gen_start': 0, 'num_nodes': 3, 'num_ops': len(FN_LIST)},
                  {'gen_start': 4, 'num_nodes': 5, 'num_ops': 3}]
        q = make_qnas(progressive_stages=stages, noop_fn_name=NOOP, progressive_mode='deterministic')
        assert q.qpop_net.chromosome.num_genes == 3

    def test_dynamic_requires_noop(self, make_qnas):
        with pytest.raises(ValueError, match='noop_fn_name'):
            make_qnas(progressive_mode='dynamic', noop_fn_name=None,
                      dynamic_initial_num_nodes=2, dynamic_max_num_nodes=4)

    def test_dynamic_initializes_stability_streaks(self, make_qnas):
        q = make_qnas(progressive_mode='dynamic', noop_fn_name=NOOP,
                      dynamic_initial_num_nodes=2, dynamic_max_num_nodes=4)
        assert q.qpop_net.chromosome.num_genes == 2
        assert q._stable_streak == [0, 0]
        assert q.dynamic_check_every_gen == q.update_quantum_gen


class TestGeneration:
    def test_generate_classical_records_lineage_and_evaluates(self, make_qnas):
        q = make_qnas()
        q.generate_classical()
        n = 3 * 2
        assert q.qpop_net.current_pop.shape == (n, 4)
        assert q.qpop_params.current_pop.shape == (n, len(PARAMS_RANGES))
        assert q.fitnesses.shape == (n,)
        assert q.classic_age.tolist() == [0] * n
        # Sampled from quantum individual row % 3, then reordered by fitness in lockstep.
        assert sorted(q.classic_ancestor.tolist()) == [0, 0, 1, 1, 2, 2]
        assert q.fitnesses.tolist() == sorted(q.fitnesses.tolist(), reverse=True)
        assert q.eval_func.calls == [(n, 0)]

    def test_decode_pop(self, make_qnas):
        q = make_qnas()
        params, net = q.qpop_params.generate_classical(), q.qpop_net.generate_classical()
        dparams, dnets = q.decode_pop(params, net)
        assert len(dparams) == len(dnets) == net.shape[0]
        assert set(dparams[0]) == set(PARAMS_RANGES)
        assert all(op in FN_LIST for op in dnets[0])

    def test_total_eval_counts_evaluated_individuals(self, make_qnas):
        q = make_qnas()
        q.generate_classical()
        assert q.total_eval == 6
        assert q.new_architectures_count == 6


class TestPenalties:
    def test_only_excess_reducing_layers_are_penalized(self, make_qnas):
        q = make_qnas(penalize_number=1)
        pool = FN_LIST.index('pool')
        net = np.array([[pool, pool, pool, 0],   # 3 reducing -> excess 2
                        [pool, 0, 0, 0],         # 1 reducing -> no penalty
                        [0, 0, 0, 0]])
        assert q.get_penalties(net, penalty_factor=0.01).tolist() == pytest.approx([0.02, 0, 0])

    def test_penalty_is_per_node_name_not_raw_index(self, make_qnas):
        q = make_qnas(penalize_number=0.5)  # any single reducing layer is excess
        # Node 1 no longer has 'pool' at index 2: the same gene value means another op.
        q.qpop_net.chromosome.fn_list[1] = ['conv_3', 'conv_5', 'skip', NOOP]
        net = np.array([[0, 2, 0, 0]])
        assert q.get_penalties(net)[0] == 0.0

    def test_eval_pop_subtracts_penalties(self, make_qnas):
        q = make_qnas(penalize_number=1)
        pool = FN_LIST.index('pool')
        params = q.qpop_params.generate_classical()
        net = np.zeros((6, 4), dtype=int)
        net[0] = pool  # 4 reducing layers, 3 over the limit
        penalized, raw = q.eval_pop(params, net)
        assert raw[0] - penalized[0] == pytest.approx(0.03)
        assert np.allclose(raw[1:], penalized[1:])


class TestReplacePop:
    def prime_gen0(self, q, fitness):
        params, net = pop_arrays(len(fitness))
        q.replace_pop(params, net, np.asarray(fitness, float), np.asarray(fitness, float))
        return params, net

    def test_generation_zero_adopts_new_population(self, make_qnas):
        q = make_qnas()
        params, net = self.prime_gen0(q, [1, 2, 3, 4, 5, 6])
        # adopted as-is, then ordered by fitness (descending), lineage in lockstep
        assert np.array_equal(q.qpop_net.current_pop, net[::-1])
        assert np.array_equal(q.qpop_params.current_pop, params[::-1])
        assert q.fitnesses.tolist() == [6, 5, 4, 3, 2, 1]
        assert q.classic_age.tolist() == [0] * 6
        assert q.classic_ancestor.tolist() == [2, 1, 0, 2, 1, 0]

    def test_best_method_keeps_top_of_union_sorted(self, make_qnas):
        q = make_qnas(replace_method='best')
        self.prime_gen0(q, [1, 2, 3, 4, 5, 6])
        q.current_gen = 1
        params, net = pop_arrays(6)
        q.replace_pop(params, net, np.array([10., 0, 0, 0, 0, 0]), np.array([10., 0, 0, 0, 0, 0]))
        # union of 12, top-6 kept, ordered descending
        assert q.fitnesses.tolist() == [10, 6, 5, 4, 3, 2]
        assert q.qpop_net.current_pop.shape == (6, 4)
        assert q.qpop_params.current_pop.shape == (6, len(PARAMS_RANGES))

    def test_best_method_ages_survivors_and_resets_newcomers(self, make_qnas):
        q = make_qnas(replace_method='best')
        self.prime_gen0(q, [1, 2, 3, 4, 5, 6])
        q.current_gen = 1
        params, net = pop_arrays(6)
        q.replace_pop(params, net, np.array([10., 0, 0, 0, 0, 0]), np.array([10., 0, 0, 0, 0, 0]),
                      new_ancestor=np.array([2, 0, 0, 0, 0, 0]))
        # newcomer (fitness 10) is age 0 with ancestor 2; survivors are one generation older
        assert q.classic_age.tolist() == [0, 1, 1, 1, 1, 1]
        assert q.classic_ancestor[0] == 2
        assert len(q.classic_age) == len(q.classic_ancestor) == len(q.fitnesses)

    def test_elitism_keeps_only_previous_best(self, make_qnas):
        q = make_qnas(replace_method='elitism')
        self.prime_gen0(q, [1, 2, 3, 4, 5, 6])
        best_net_row = q.qpop_net.current_pop[np.argmax(q.fitnesses)].copy()
        q.current_gen = 1
        params, net = pop_arrays(6)
        new_fit = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        q.replace_pop(params, net, new_fit, new_fit)
        # old best (6) + the 5 best of the new population, sorted
        assert q.fitnesses.tolist() == [6, 0.6, 0.5, 0.4, 0.3, 0.2]
        assert np.array_equal(q.qpop_net.current_pop[0], best_net_row)
        assert q.classic_age.tolist() == [1, 0, 0, 0, 0, 0]

    def test_stale_lineage_arrays_self_heal(self, make_qnas):
        q = make_qnas(replace_method='best')
        self.prime_gen0(q, [1, 2, 3, 4, 5, 6])
        q.classic_age = q.classic_ancestor = None
        q.current_gen = 1
        params, net = pop_arrays(6)
        q.replace_pop(params, net, np.arange(6.0), np.arange(6.0))
        assert len(q.classic_age) == len(q.classic_ancestor) == 6

    def test_best_so_far_only_moves_on_improvement(self, make_qnas):
        q = make_qnas()
        q.best_so_far = 100.0
        self.prime_gen0(q, [1, 2, 3, 4, 5, 6])
        assert q.best_so_far_id == [0, 0]  # untouched: nothing beat 100


class TestUpdateQuantumDispatch:
    def prime(self, make_qnas, **kw):
        q = make_qnas(**kw)
        q.generate_classical()
        return q

    def test_only_fires_on_multiples_of_update_quantum_gen(self, make_qnas):
        q = self.prime(make_qnas)
        before = [p.copy() for p in q.qpop_net.probabilities]
        q.current_gen = 1  # update_quantum_gen == 2
        q.update_quantum()
        assert all(np.array_equal(a, b) for a, b in zip(before, q.qpop_net.probabilities))
        q.current_gen = 2
        q.update_quantum()
        assert not all(np.array_equal(a, b) for a, b in zip(before, q.qpop_net.probabilities))

    def test_generation_zero_never_updates(self, make_qnas):
        q = self.prime(make_qnas)
        before = [p.copy() for p in q.qpop_net.probabilities]
        q.current_gen = 0
        q.update_quantum()
        assert all(np.array_equal(a, b) for a, b in zip(before, q.qpop_net.probabilities))

    def test_engine_selects_update_method(self, make_qnas, monkeypatch):
        calls = []
        for engine, expected in (('default', 'update_quantum'),
                                 ('ancestor_decay', 'update_quantum_ancestor_decay')):
            q = self.prime(make_qnas, quantum_update_engine=engine, quantum_update_age_decay=0.25)
            monkeypatch.setattr(q.qpop_net, 'update_quantum',
                                lambda **kw: calls.append(('update_quantum', kw)))
            monkeypatch.setattr(q.qpop_net, 'update_quantum_ancestor_decay',
                                lambda **kw: calls.append(('update_quantum_ancestor_decay', kw)))
            q.current_gen = 2
            q.update_quantum()
            assert calls[-1][0] == expected
        decay_kwargs = calls[-1][1]
        assert decay_kwargs['decay_rate'] == 0.25
        assert decay_kwargs['ages'] is q.classic_age
        assert decay_kwargs['ancestors'] is q.classic_ancestor

    def test_ancestor_decay_engine_keeps_pmf_valid(self, make_qnas):
        q = make_qnas(quantum_update_engine='ancestor_decay')
        for gen in range(6):
            q.current_gen = gen
            q.generate_classical()
            q.go_next_gen()
        for probs in q.qpop_net.probabilities:
            assert np.allclose(probs.sum(axis=1), 1.0)


class TestPruning:
    @pytest.fixture
    def q(self, make_qnas):
        return make_qnas(noop_fn_name=NOOP)

    def test_rank_prune_keeps_top_ops_plus_noop(self, q):
        weights = np.array([0.4, 0.05, 0.3, 0.05, 0.2])  # conv_3 conv_5 pool skip no_op
        kept = q._rank_and_prune_ops(list(FN_LIST), weights, num_ops=3)
        assert kept == ['conv_3', 'pool', NOOP]

    def test_rank_prune_never_drops_noop_even_if_lowest(self, q):
        weights = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
        assert NOOP in q._rank_and_prune_ops(list(FN_LIST), weights, num_ops=2)

    def test_rank_prune_noop_when_num_ops_covers_everything(self, q):
        assert q._rank_and_prune_ops(list(FN_LIST), np.ones(5), num_ops=9) == FN_LIST

    def test_rank_prune_ties_prefer_lower_index(self, q):
        kept = q._rank_and_prune_ops(list(FN_LIST), np.full(5, 0.2), num_ops=3)
        assert kept == ['conv_3', 'conv_5', NOOP]

    def test_rank_prune_all_nodes_uses_each_nodes_own_pmf(self, q):
        q.noop_fn_name = NOOP
        q.qpop_net.probabilities[0][:] = [0.5, 0.1, 0.1, 0.1, 0.2]
        q.qpop_net.probabilities[1][:] = [0.1, 0.5, 0.1, 0.1, 0.2]
        new = q._rank_and_prune_all_nodes(q.qpop_net.chromosome.fn_list, num_ops=2)
        assert new[0] == ['conv_3', NOOP]
        assert new[1] == ['conv_5', NOOP]

    def test_rank_prune_globally_requires_shared_menu(self, q):
        q.noop_fn_name = NOOP
        diverged = [list(FN_LIST), ['conv_3', NOOP], list(FN_LIST), list(FN_LIST)]
        with pytest.raises(ValueError, match='share the same op list'):
            q._rank_and_prune_globally(diverged, num_ops=2)

    def test_rank_prune_globally_applies_one_menu_everywhere(self, q):
        q.noop_fn_name = NOOP
        for probs in q.qpop_net.probabilities:
            probs[:] = [0.5, 0.1, 0.1, 0.1, 0.2]
        new = q._rank_and_prune_globally(q.qpop_net.chromosome.fn_list, num_ops=2)
        assert all(node == ['conv_3', NOOP] for node in new)


class TestNucleusPrune:
    @pytest.fixture
    def q(self, make_qnas):
        return make_qnas()

    def test_keeps_smallest_prefix_reaching_threshold(self, q):
        q.noop_fn_name = NOOP
        weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        names, changed = q._nucleus_prune_ops(list(FN_LIST), weights, threshold=0.8,
                                              min_ops=1, flatness_epsilon=0.0)
        # rankable mass = 0.95; conv_3 (.5) + conv_5 (.3) = .8/.95 = 0.84 >= 0.8
        assert names == ['conv_3', 'conv_5', NOOP]
        assert changed

    def test_min_ops_floors_the_cut(self, q):
        q.noop_fn_name = NOOP
        weights = np.array([0.9, 0.05, 0.03, 0.01, 0.01])
        names, _ = q._nucleus_prune_ops(list(FN_LIST), weights, threshold=0.5,
                                        min_ops=4, flatness_epsilon=0.0)
        assert len(names) == 4 and NOOP in names

    def test_flat_distribution_is_not_pruned(self, q):
        q.noop_fn_name = NOOP
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        names, changed = q._nucleus_prune_ops(list(FN_LIST), weights, threshold=0.5,
                                              min_ops=1, flatness_epsilon=1e-6)
        assert names == FN_LIST and not changed

    def test_no_change_reports_unchanged(self, q):
        q.noop_fn_name = NOOP
        weights = np.array([0.3, 0.3, 0.3, 0.05, 0.05])
        names, changed = q._nucleus_prune_ops(list(FN_LIST), weights, threshold=1.0,
                                              min_ops=1, flatness_epsilon=0.0)
        assert names == FN_LIST and not changed

    def test_only_noop_is_a_noop(self, q):
        q.noop_fn_name = NOOP
        names, changed = q._nucleus_prune_ops([NOOP], np.array([1.0]), 0.8, 1, 0.0)
        assert names == [NOOP] and not changed


class TestStabilityStreaks:
    def dynamic(self, make_qnas, global_op_pruning=False):
        return make_qnas(progressive_mode='dynamic', noop_fn_name=NOOP,
                         dynamic_initial_num_nodes=3, dynamic_max_num_nodes=5,
                         global_op_pruning=global_op_pruning)

    def test_per_node_streaks_increment_and_reset(self, make_qnas):
        q = self.dynamic(make_qnas)
        q._update_stability_streaks([False, True, False])
        assert q._stable_streak == [1, 0, 1]
        q._update_stability_streaks([False, False, True])
        assert q._stable_streak == [2, 1, 0]

    def test_global_streak(self, make_qnas):
        q = self.dynamic(make_qnas, global_op_pruning=True)
        assert q._stable_streak == 0
        q._update_stability_streaks([False])
        q._update_stability_streaks([False])
        assert q._stable_streak == 2
        q._update_stability_streaks([True])
        assert q._stable_streak == 0


class TestCheckpoint:
    def test_save_and_load_roundtrip(self, make_qnas, tmp_path):
        q = make_qnas()
        for gen in range(3):
            q.current_gen = gen
            q.generate_classical()
            q.go_next_gen()

        fresh = make_qnas()
        fresh.load_qnas_data(q.data_file)
        assert fresh.current_gen == 2
        assert fresh.best_so_far == q.best_so_far
        assert fresh.total_eval == q.total_eval
        assert np.array_equal(fresh.qpop_net.current_pop, q.qpop_net.current_pop)
        assert all(np.array_equal(a, b) for a, b in
                   zip(fresh.qpop_net.probabilities, q.qpop_net.probabilities))

    def test_save_data_accumulates_generations(self, make_qnas):
        q = make_qnas()
        for gen in range(3):
            q.current_gen = gen
            q.generate_classical()
            q.save_data()
        assert sorted(util.load_pkl(q.data_file)) == [0, 1, 2]

    def test_flat_fn_list_checkpoint_rejected_for_progressive(self, make_qnas):
        stages = [{'gen_start': 0, 'num_nodes': 3, 'num_ops': len(FN_LIST)}]
        q = make_qnas(progressive_stages=stages, noop_fn_name=NOOP)
        q.generate_classical()
        q.save_data()
        old_format = {0: {**util.load_pkl(q.data_file)[0], 'fn_list': list(FN_LIST)}}
        q.dump_pkl_data(old_format)
        with pytest.raises(RuntimeError, match='per-node'):
            q.load_qnas_data(q.data_file)


class TestEarlyStopping:
    def stagnant(self, make_qnas, patience):
        q = make_qnas(patience=patience, early_stopping=True)
        q.best_so_far = q.last_best_so_far = 1.0
        return q

    def test_stops_after_patience_stagnant_generations(self, make_qnas):
        q = self.stagnant(make_qnas, patience=2)
        outcomes = []
        for gen in (2, 3):
            q.current_gen = gen
            outcomes.append(q.check_early_stopping())
        assert outcomes == [False, True]
        assert q.early_stopping_counter == 2

    def test_improvement_resets_counter(self, make_qnas):
        q = self.stagnant(make_qnas, patience=5)
        q.current_gen = 2
        q.check_early_stopping()
        q.check_early_stopping()
        assert q.early_stopping_counter == 2
        q.best_so_far = 1.1  # +10% > 0.5% threshold
        q.current_gen = 3
        assert q.check_early_stopping() is False
        assert q.early_stopping_counter == 0

    def test_disabled_before_generation_two(self, make_qnas):
        q = self.stagnant(make_qnas, patience=1)
        q.current_gen = 1
        assert q.check_early_stopping() is False
        assert q.early_stopping_counter == 0

    def test_first_check_after_resume_only_sets_the_baseline(self, make_qnas):
        # Regression: last_best_so_far used to be unset, so resuming at current_gen > 1
        # with early_stopping enabled raised AttributeError.
        q = make_qnas(patience=3, early_stopping=True)
        q.best_so_far = 1.0
        q.current_gen = 5  # as restored by load_qnas_data()
        assert q.check_early_stopping() is False
        assert q.early_stopping_counter == 0
        assert q.last_best_so_far == 1.0
        q.current_gen = 6
        q.check_early_stopping()  # now compared against the recorded baseline
        assert q.early_stopping_counter == 1

    def test_zero_baseline_does_not_divide_by_zero(self, make_qnas):
        q = make_qnas(patience=3, early_stopping=True)
        q.best_so_far = q.last_best_so_far = 0.0
        q.current_gen = 2
        assert q.check_early_stopping() is False
        assert q.early_stopping_counter == 1  # 0 -> 0 is no improvement
        q.best_so_far = 5.0
        q.current_gen = 3
        q.check_early_stopping()
        assert q.early_stopping_counter == 0  # 0 -> 5 is an improvement

    def test_resumed_run_with_early_stopping_completes(self, make_qnas):
        first = make_qnas(max_generations=4, early_stopping=True, patience=50)
        first.evolve()
        resumed = make_qnas(max_generations=8, early_stopping=True, patience=50)
        resumed.load_qnas_data(first.data_file)
        resumed.evolve()
        assert resumed.current_gen == 8


class TestProgressiveRegressions:
    def test_partial_survivors_do_not_break_params_crossover(self, make_qnas):
        # Regression: when a transition drops SOME individuals, params classic_crossover
        # indexed the shrunken current_pop with a full-size mask (IndexError).
        q = make_qnas(progressive_mode='dynamic', noop_fn_name=NOOP,
                      dynamic_initial_num_nodes=4, dynamic_max_num_nodes=4)
        q.generate_classical()
        q.current_gen = 1
        # Half the individuals use conv_3 at node 0 (won't survive the prune), half don't.
        pool = FN_LIST.index('pool')
        q.qpop_net.current_pop[:] = pool
        q.qpop_net.current_pop[:3, 0] = FN_LIST.index('conv_3')
        no_conv3 = [[n for n in FN_LIST if n != 'conv_3'] for _ in range(4)]
        q._apply_fn_list_change(no_conv3, 4)
        assert 0 < len(q.qpop_params.current_pop) < 6

        q.generate_classical()
        assert q.qpop_net.current_pop.shape[0] == 6
        assert q.qpop_params.current_pop.shape[0] == 6

    def test_global_pruning_after_growth_with_unsorted_menu(self, make_qnas):
        # Regression: grown nodes used to get sorted(union) while existing nodes kept
        # their own order, so the next global prune saw "diverged" nodes.
        q = make_qnas(progressive_mode='dynamic', noop_fn_name=NOOP, global_op_pruning=True,
                      dynamic_initial_num_nodes=3, dynamic_max_num_nodes=5)
        pruned = [['conv_3', 'pool', NOOP] for _ in range(3)]
        q._apply_fn_list_change(pruned, 4)  # grows by one node
        fn_list = q.qpop_net.chromosome.fn_list
        assert fn_list == [['conv_3', 'pool', NOOP]] * 4
        new_fn_list, _ = q._nucleus_prune_globally(fn_list)  # must not raise
        assert len(new_fn_list) == 4


@pytest.mark.e2e
class TestEvolveEndToEnd:
    def run(self, q):
        q.evolve()
        return q

    def test_plain_run_completes_and_best_never_regresses(self, make_qnas):
        q = self.run(make_qnas(max_generations=6, replace_method='best'))
        assert q.current_gen == 6
        assert len(q.eval_func.calls) == 6
        history = util.load_pkl(q.data_file)
        best = [history[g]['best_so_far'] for g in sorted(history)]
        assert best == sorted(best)

    def test_elitism_keeps_population_size_constant(self, make_qnas):
        q = self.run(make_qnas(max_generations=5, replace_method='elitism'))
        assert q.qpop_net.current_pop.shape[0] == 6
        assert len(q.classic_age) == len(q.classic_ancestor) == 6

    def test_population_crossover_path(self, make_qnas):
        q = make_qnas(max_generations=6, en_pop_crossover=True, crossover_frequency=2,
                      pop_crossover_rate=0.5)
        self.run(q)
        assert q.current_gen == 6

    def test_deterministic_progressive_grows_and_prunes(self, make_qnas):
        stages = [{'gen_start': 0, 'num_nodes': 3, 'num_ops': 5},
                  {'gen_start': 3, 'num_nodes': 5, 'num_ops': 3}]
        q = self.run(make_qnas(max_generations=6, progressive_stages=stages, noop_fn_name=NOOP,
                               progressive_mode='deterministic',
                               initial_probs=concentrated_probs(FN_LIST)))
        assert q.current_stage_idx == 1
        assert q.qpop_net.chromosome.num_genes == 5
        assert all(len(f) <= 3 for f in q.qpop_net.chromosome.fn_list[:3])
        assert q.eval_func.fn_list is q.qpop_net.chromosome.fn_list
        for probs in q.qpop_net.probabilities:
            assert np.allclose(probs.sum(axis=1), 1.0)

    @pytest.mark.parametrize('seed', range(5))
    def test_progressive_survives_partial_population_loss(self, make_qnas, seed):
        # Uniform PMF: a stage transition drops an arbitrary subset of the individuals.
        np.random.seed(seed)
        stages = [{'gen_start': 0, 'num_nodes': 3, 'num_ops': 5},
                  {'gen_start': 3, 'num_nodes': 4, 'num_ops': 3},
                  {'gen_start': 6, 'num_nodes': 5, 'num_ops': 2}]
        q = self.run(make_qnas(max_generations=9, progressive_stages=stages, noop_fn_name=NOOP,
                               progressive_mode='deterministic', replace_method='best'))
        assert q.current_gen == 9
        assert q.qpop_net.chromosome.num_genes == 5

    @pytest.mark.parametrize('engine', ['default', 'ancestor_decay'])
    @pytest.mark.parametrize('global_pruning', [False, True])
    def test_dynamic_progressive_grows_up_to_ceiling(self, make_qnas, engine, global_pruning):
        q = self.run(make_qnas(
            max_generations=14, progressive_mode='dynamic', noop_fn_name=NOOP,
            initial_probs=concentrated_probs(FN_LIST),
            dynamic_initial_num_nodes=2, dynamic_max_num_nodes=4, dynamic_probability_threshold=0.6,
            dynamic_min_ops=2, dynamic_growth_patience=1, dynamic_node_growth_amount=1,
            update_quantum_gen=2, dynamic_check_every_gen=2, global_op_pruning=global_pruning,
            quantum_update_engine=engine))
        assert q.qpop_net.chromosome.num_genes == 4  # grew from 2 to the ceiling
        assert len(q.qpop_net.probabilities) == q.qpop_net.chromosome.num_genes
        assert len(q.qpop_net.chromosome.fn_list) == q.qpop_net.chromosome.num_genes
        for node, names in enumerate(q.qpop_net.chromosome.fn_list):
            assert NOOP in names
            assert q.qpop_net.probabilities[node].shape[1] == len(names)

    def test_resume_continues_from_next_generation(self, make_qnas, tmp_path):
        first = make_qnas(max_generations=3)
        first.evolve()
        resumed = make_qnas(max_generations=5)
        resumed.load_qnas_data(first.data_file)
        resumed.evolve()
        assert resumed.current_gen == 5
        assert [c[1] for c in resumed.eval_func.calls] == [3, 4]
