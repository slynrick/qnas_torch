import argparse
import os
import sqlite3
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from rich.console import Console

from qnas_queue import cli, db, runner, worker


@pytest.fixture(autouse=True)
def isolated_queue(tmp_path, monkeypatch):
    """Point every queue path at tmp_path so tests never touch the real .qnas_queue/."""
    root = tmp_path / 'project'
    queue = root / '.qnas_queue'
    monkeypatch.setattr(db, 'PROJECT_ROOT', root)
    monkeypatch.setattr(db, 'QUEUE_DIR', queue)
    monkeypatch.setattr(db, 'DB_PATH', queue / 'queue.db')
    monkeypatch.setattr(db, 'LOG_DIR', queue / 'logs')
    monkeypatch.setattr(db, 'SNAPSHOT_DIR', queue / 'configs')
    monkeypatch.setattr(db, 'WORKER_LOG_PATH', queue / 'worker.log')
    root.mkdir()
    return root


def add(conn, mode='pipeline', config='c.yml', exp='exp1', extra='', priority=0, gpu_ids=None):
    return db.add_job(conn, mode, config, exp, extra, priority, gpu_ids=gpu_ids)


class TestDb:
    def test_connect_creates_schema_with_no_workers(self):
        with db.connect() as conn:
            assert db.list_workers(conn) == []
            assert db.list_jobs(conn) == []
        assert db.DB_PATH.is_file() and db.LOG_DIR.is_dir()

    def test_add_and_get_job_defaults(self):
        with db.connect() as conn:
            job_id = add(conn, extra='-M')
            job = db.get_job(conn, job_id)
        assert job['status'] == 'queued'
        assert job['extra_args'] == '-M'
        assert job['created_at'] and job['started_at'] is None

    def test_get_job_none_id(self):
        with db.connect() as conn:
            assert db.get_job(conn, None) is None
            assert db.get_job(conn, 999) is None

    def test_invalid_mode_rejected_by_schema(self):
        with db.connect() as conn, pytest.raises(sqlite3.IntegrityError):
            add(conn, mode='bogus')

    def test_list_jobs_filters_by_status(self):
        with db.connect() as conn:
            a, b = add(conn), add(conn)
            db.update_job(conn, b, status='done')
            assert [r['id'] for r in db.list_jobs(conn, status='queued')] == [a]
            assert [r['id'] for r in db.list_jobs(conn)] == [a, b]

    def test_claim_next_job_orders_by_priority_then_id(self):
        with db.connect() as conn:
            low = add(conn, priority=0)
            high = add(conn, priority=5)
            low2 = add(conn, priority=0)
            claimed = [db.claim_next_job(conn, 111)['id'] for _ in range(3)]
            assert claimed == [high, low, low2]
            assert db.claim_next_job(conn, 111) is None

    def test_claim_marks_running_with_start_time_and_worker_pid(self):
        with db.connect() as conn:
            add(conn)
            job = db.claim_next_job(conn, 111)
        assert job['status'] == 'running' and job['started_at']

    def test_claim_with_gpu_pool_assigns_lowest_free_id(self):
        with db.connect() as conn:
            add(conn)
            job = db.claim_next_job(conn, 111, gpu_pool={0, 1, 2})
        assert job['gpu_ids'] == '0'

    def test_claim_with_gpu_pool_skips_ids_already_in_use(self):
        with db.connect() as conn:
            a = add(conn)
            db.update_job(conn, a, status='running', gpu_ids='0')
            b = add(conn)
            job = db.claim_next_job(conn, 111, gpu_pool={0, 1})
        assert job['id'] == b and job['gpu_ids'] == '1'

    def test_claim_with_gpu_pool_fully_occupied_returns_none(self):
        with db.connect() as conn:
            a = add(conn)
            db.update_job(conn, a, status='running', gpu_ids='0')
            add(conn)  # b - queued, would also need the pool
            assert db.claim_next_job(conn, 111, gpu_pool={0}) is None

    def test_claim_with_gpu_pool_leaves_a_manually_pinned_job_untouched(self):
        with db.connect() as conn:
            job_id = add(conn, gpu_ids='0')
            job = db.claim_next_job(conn, 111, gpu_pool={5, 6})
        assert job['id'] == job_id and job['gpu_ids'] == '0'  # pin kept, not drawn from pool

    def test_claim_with_gpu_pool_skips_exhausted_front_job_for_a_pinned_one_behind_it(self):
        with db.connect() as conn:
            a = add(conn)  # front of queue, priority 0, needs the pool
            db.update_job(conn, a, status='running', gpu_ids='0')  # the only pool GPU is taken
            b = add(conn, gpu_ids='7')  # behind it, but pinned - doesn't need the pool
            job = db.claim_next_job(conn, 111, gpu_pool={0})
        assert job['id'] == b and job['gpu_ids'] == '7'

    def test_claim_without_gpu_pool_ignores_gpu_ids_column(self):
        with db.connect() as conn:
            add(conn)
            job = db.claim_next_job(conn, 111)  # gpu_pool=None, same as before this feature
        assert job['gpu_ids'] is None
        assert job['worker_pid'] == 111

    def test_cancel_and_delete(self):
        with db.connect() as conn:
            job_id = add(conn)
            db.cancel_job(conn, job_id)
            job = db.get_job(conn, job_id)
            assert job['status'] == 'cancelled' and job['finished_at']
            db.delete_job(conn, job_id)
            assert db.get_job(conn, job_id) is None

    def test_update_job_no_fields_is_noop(self):
        with db.connect() as conn:
            job_id = add(conn)
            db.update_job(conn, job_id)
            assert db.get_job(conn, job_id)['status'] == 'queued'

    def test_resume_stopped_jobs_requeues_and_clears_run_state(self):
        with db.connect() as conn:
            job_id = add(conn)
            db.update_job(conn, job_id, status='stopped', pgid=77, exit_code=-15,
                          started_at='x', finished_at='y')
            assert db.resume_stopped_jobs(conn) == [job_id]
            job = db.get_job(conn, job_id)
        assert job['status'] == 'queued'
        assert job['pgid'] is None and job['exit_code'] is None
        assert job['started_at'] is None and job['finished_at'] is None

    def test_list_running_jobs(self):
        with db.connect() as conn:
            a, _ = add(conn), add(conn)
            db.update_job(conn, a, status='running')
            assert [r['id'] for r in db.list_running_jobs(conn)] == [a]

    def test_register_worker_creates_row(self):
        with db.connect() as conn:
            db.register_worker(conn, 123)
            worker = db.get_worker(conn, 123)
        assert worker['pid'] == 123 and worker['status'] == 'running'
        assert worker['current_job_id'] is None and worker['started_at']

    def test_register_worker_is_idempotent_and_resets_current_job(self):
        with db.connect() as conn:
            db.register_worker(conn, 123)
            db.set_worker(conn, 123, current_job_id=7)
            db.register_worker(conn, 123)  # e.g. a restarted process reusing the pid
            worker = db.get_worker(conn, 123)
        assert worker['current_job_id'] is None

    def test_set_worker_updates_timestamp(self):
        with db.connect() as conn:
            db.register_worker(conn, 123)
            db.set_worker(conn, 123, current_job_id=5)
            worker = db.get_worker(conn, 123)
        assert worker['current_job_id'] == 5 and worker['updated_at']

    def test_list_workers_orders_by_pid(self):
        with db.connect() as conn:
            db.register_worker(conn, 200)
            db.register_worker(conn, 100)
            assert [w['pid'] for w in db.list_workers(conn)] == [100, 200]

    def test_remove_worker(self):
        with db.connect() as conn:
            db.register_worker(conn, 123)
            db.remove_worker(conn, 123)
            assert db.list_workers(conn) == []

    def test_add_job_stores_gpu_ids(self):
        with db.connect() as conn:
            job_id = add(conn, gpu_ids='0,1')
            assert db.get_job(conn, job_id)['gpu_ids'] == '0,1'

    def test_add_job_gpu_ids_defaults_to_none(self):
        with db.connect() as conn:
            job_id = add(conn)
            assert db.get_job(conn, job_id)['gpu_ids'] is None

    def test_changes_are_committed_on_context_exit(self):
        with db.connect() as conn:
            add(conn)
        with db.connect() as conn:
            assert len(db.list_jobs(conn)) == 1

    def test_pid_alive(self):
        assert db.pid_alive(os.getpid())
        assert not db.pid_alive(None)
        assert not db.pid_alive(0)
        assert not db.pid_alive(2 ** 22 + 12345)  # above the default pid_max

    def test_resolve_path_relative_and_absolute(self, isolated_queue):
        assert db.resolve_path('configs/a.yml') == isolated_queue / 'configs' / 'a.yml'
        assert db.resolve_path('/abs/a.yml') == Path('/abs/a.yml')

    def test_migration_adds_cancelled_status_and_keeps_rows(self):
        db.ensure_dirs()
        legacy = sqlite3.connect(db.DB_PATH)
        legacy.executescript(db.SCHEMA.replace(",'cancelled'", ''))
        legacy.execute(
            "INSERT INTO jobs (mode, config_path, experiment_path, created_at) "
            "VALUES ('evolve', 'c.yml', 'exp', 'now')")
        legacy.commit()
        legacy.close()
        with db.connect() as conn:
            assert 'cancelled' in conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'jobs'").fetchone()['sql']
            assert len(db.list_jobs(conn)) == 1
            db.cancel_job(conn, db.list_jobs(conn)[0]['id'])  # would violate the old CHECK


    def test_migration_adds_config_snapshot_column(self):
        db.ensure_dirs()
        legacy = sqlite3.connect(db.DB_PATH)
        legacy.executescript(db.SCHEMA.replace('    config_snapshot TEXT,\n', ''))
        legacy.execute(
            "INSERT INTO jobs (mode, config_path, experiment_path, created_at) "
            "VALUES ('evolve', 'c.yml', 'exp', 'now')")
        legacy.commit()
        legacy.close()
        with db.connect() as conn:
            job = db.list_jobs(conn)[0]
            assert job['config_snapshot'] is None
            assert db.config_for_run(job) == 'c.yml'  # pre-snapshot rows run as before

    def test_config_for_run_prefers_snapshot(self):
        with db.connect() as conn:
            job_id = add(conn, config='configs/c.yml')
            db.update_job(conn, job_id, config_snapshot='.qnas_queue/configs/job_1_c.yml')
            assert db.config_for_run(db.get_job(conn, job_id)) == '.qnas_queue/configs/job_1_c.yml'

    def test_migration_adds_gpu_ids_and_worker_pid_columns(self):
        db.ensure_dirs()
        legacy = sqlite3.connect(db.DB_PATH)
        legacy.executescript(
            db.SCHEMA.replace('    gpu_ids         TEXT,\n', '')
                     .replace('    worker_pid      INTEGER,\n', '')
        )
        legacy.execute(
            "INSERT INTO jobs (mode, config_path, experiment_path, created_at) "
            "VALUES ('evolve', 'c.yml', 'exp', 'now')")
        legacy.commit()
        legacy.close()
        with db.connect() as conn:
            job = db.list_jobs(conn)[0]
            assert job['gpu_ids'] is None and job['worker_pid'] is None

    def test_migration_drops_legacy_singleton_worker_table(self):
        db.ensure_dirs()
        legacy = sqlite3.connect(db.DB_PATH)
        legacy.executescript(
            "CREATE TABLE worker (id INTEGER PRIMARY KEY CHECK (id = 1), pid INTEGER, "
            "status TEXT, current_job_id INTEGER, started_at TEXT, updated_at TEXT);"
        )
        legacy.execute("INSERT INTO worker (id, pid, status) VALUES (1, 999, 'running')")
        legacy.commit()
        legacy.close()
        with db.connect() as conn:
            tables = {r['name'] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert 'worker' not in tables and 'workers' in tables
            assert db.list_workers(conn) == []


class TestWorkerGpuPool:
    def test_gpu_pool_from_env_parses_comma_separated_ids(self, monkeypatch):
        monkeypatch.setenv('QNAS_QUEUE_GPU_POOL', '0,1,2')
        assert worker._gpu_pool_from_env() == {0, 1, 2}

    def test_gpu_pool_from_env_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv('QNAS_QUEUE_GPU_POOL', raising=False)
        assert worker._gpu_pool_from_env() is None


class TestRunner:
    def test_evolve_argv(self):
        argv = runner.build_argv('evolve', 'cfg.yml', 'exp1', '--dataset cifar10 -x')
        assert argv[:4] == ['uv', 'run', 'python', str(runner.SRC_DIR / 'run_evolution.py')]
        assert argv[argv.index('--experiment_path') + 1] == 'exp1'
        assert argv[argv.index('--config_file') + 1] == 'cfg.yml'
        assert argv[-3:] == ['--dataset', 'cifar10', '-x']

    def test_retrain_argv_ignores_config(self):
        argv = runner.build_argv('retrain', 'cfg.yml', 'exp1', '')
        assert argv[3].endswith('retrain_model.py')
        assert 'cfg.yml' not in argv

    def test_pipeline_argv(self):
        argv = runner.build_argv('pipeline', 'cfg.yml', 'exp1', '-d cifar10 -M -T')
        assert argv[0] == 'bash' and argv[1].endswith('run_pipeline.sh')
        assert argv[2:6] == ['-e', 'exp1', '-c', 'cfg.yml']
        assert argv[6:] == ['-d', 'cifar10', '-M', '-T']

    def test_extra_args_are_shell_split(self):
        argv = runner.build_argv('pipeline', 'c', 'e', '--note "two words"')
        assert argv[-2:] == ['--note', 'two words']

    def test_none_extra_args(self):
        assert runner.build_argv('evolve', 'c', 'e', None)[-1] == 'c'

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match='Unknown mode'):
            runner.build_argv('nope', 'c', 'e', '')

    def test_scripts_referenced_by_runner_exist(self):
        assert (runner.SRC_DIR / 'run_evolution.py').is_file()
        assert (runner.SRC_DIR / 'retrain_model.py').is_file()
        assert (runner.SCRIPTS_DIR / 'run_pipeline.sh').is_file()


class TestCli:
    def ns(self, **kw):
        kw.setdefault('gpu_ids', None)
        kw.setdefault('gpu_pool', None)
        return argparse.Namespace(**kw)

    def make_config(self, root, text='QNAS: {}\n'):
        path = root / 'cfg.yml'
        path.write_text(text)
        return path

    def test_add_stores_project_relative_config(self, isolated_queue, capsys):
        cfg = self.make_config(isolated_queue)
        cli.cmd_add(self.ns(config=str(cfg), mode='pipeline', experiment_path='exp1',
                            extra='-M', priority=3))
        with db.connect() as conn:
            job = db.list_jobs(conn)[0]
        assert job['config_path'] == 'cfg.yml'
        assert (job['mode'], job['experiment_path'], job['extra_args'], job['priority']) == \
               ('pipeline', 'exp1', '-M', 3)
        assert 'Queued job' in capsys.readouterr().out

    def test_add_freezes_a_snapshot_of_the_config(self, isolated_queue):
        cfg = self.make_config(isolated_queue, 'QNAS: {reset: true}\n')
        cli.cmd_add(self.ns(config=str(cfg), mode='pipeline', experiment_path='exp1',
                            extra='', priority=0))
        cfg.write_text('QNAS: {reset: false}\n')  # edited after queueing
        with db.connect() as conn:
            job = db.list_jobs(conn)[0]
        assert job['config_path'] == 'cfg.yml'  # provenance kept
        assert job['config_snapshot'] == f".qnas_queue/configs/job_{job['id']}_cfg.yml"
        assert db.config_for_run(job) == job['config_snapshot']
        assert db.resolve_path(job['config_snapshot']).read_text() == 'QNAS: {reset: true}\n'

    def test_remove_deletes_the_snapshot(self, isolated_queue):
        cfg = self.make_config(isolated_queue)
        cli.cmd_add(self.ns(config=str(cfg), mode='evolve', experiment_path='e',
                            extra='', priority=0))
        with db.connect() as conn:
            job = db.list_jobs(conn)[0]
        snapshot = db.resolve_path(job['config_snapshot'])
        assert snapshot.is_file()
        cli.cmd_remove(self.ns(id=job['id']))
        assert not snapshot.exists()

    def test_add_outside_project_keeps_absolute_path(self, tmp_path):
        outside = tmp_path / 'elsewhere.yml'
        outside.write_text('a: 1\n')
        cli.cmd_add(self.ns(config=str(outside), mode='evolve', experiment_path='e',
                            extra=None, priority=0))
        with db.connect() as conn:
            assert db.list_jobs(conn)[0]['config_path'] == str(outside)

    def test_add_missing_config_exits(self):
        with pytest.raises(SystemExit, match='not found'):
            cli.cmd_add(self.ns(config='/no/such.yml', mode='evolve', experiment_path='e',
                                extra='', priority=0))

    def test_add_invalid_yaml_exits(self, isolated_queue):
        bad = self.make_config(isolated_queue, 'a: [unclosed\n')
        with pytest.raises(SystemExit, match='not valid YAML'):
            cli.cmd_add(self.ns(config=str(bad), mode='evolve', experiment_path='e',
                                extra='', priority=0))

    def seed(self, status):
        with db.connect() as conn:
            job_id = add(conn)
            db.update_job(conn, job_id, status=status)
        return job_id

    def test_cancel_only_queued(self):
        queued, running = self.seed('queued'), self.seed('running')
        cli.cmd_cancel(self.ns(id=queued))
        with pytest.raises(SystemExit, match='only queued'):
            cli.cmd_cancel(self.ns(id=running))
        with pytest.raises(SystemExit, match='no job'):
            cli.cmd_cancel(self.ns(id=999))

    def test_remove_refuses_running(self):
        running, done = self.seed('running'), self.seed('done')
        with pytest.raises(SystemExit, match='running'):
            cli.cmd_remove(self.ns(id=running))
        cli.cmd_remove(self.ns(id=done))
        with db.connect() as conn:
            assert db.get_job(conn, done) is None

    @pytest.mark.parametrize('status', ['failed', 'stopped', 'cancelled'])
    def test_retry_requeues_terminal_jobs(self, status):
        job_id = self.seed(status)
        cli.cmd_retry(self.ns(id=job_id))
        with db.connect() as conn:
            assert db.get_job(conn, job_id)['status'] == 'queued'

    @pytest.mark.parametrize('status', ['queued', 'running', 'done'])
    def test_retry_rejects_other_states(self, status):
        job_id = self.seed(status)
        with pytest.raises(SystemExit, match='only failed/stopped/cancelled'):
            cli.cmd_retry(self.ns(id=job_id))

    def test_list_empty_and_populated(self, capsys):
        cli.cmd_list(self.ns(status=None))
        assert 'Queue is empty' in capsys.readouterr().out
        self.seed('queued')
        cli.cmd_list(self.ns(status=None))
        out = capsys.readouterr().out
        assert 'status' in out and 'queued' in out and 'exp1' in out

    def test_stop_with_no_worker_marks_orphaned_running_jobs_stopped(self, capsys):
        job_id = self.seed('running')
        cli.cmd_stop(self.ns())
        with db.connect() as conn:
            assert db.get_job(conn, job_id)['status'] == 'stopped'
        assert 'No worker is running' in capsys.readouterr().out

    def test_add_with_valid_gpu_ids(self, isolated_queue):
        cfg = self.make_config(isolated_queue)
        cli.cmd_add(self.ns(config=str(cfg), mode='evolve', experiment_path='e',
                            extra='', priority=0, gpu_ids='0, 1'))
        with db.connect() as conn:
            assert db.list_jobs(conn)[0]['gpu_ids'] == '0,1'

    def test_add_with_invalid_gpu_ids_exits(self, isolated_queue):
        cfg = self.make_config(isolated_queue)
        with pytest.raises(SystemExit, match='gpu-ids'):
            cli.cmd_add(self.ns(config=str(cfg), mode='evolve', experiment_path='e',
                                extra='', priority=0, gpu_ids='0,x'))

    def test_list_shows_gpu_column(self, isolated_queue, capsys):
        cfg = self.make_config(isolated_queue)
        cli.cmd_add(self.ns(config=str(cfg), mode='evolve', experiment_path='e',
                            extra='', priority=0, gpu_ids='0'))
        cli.cmd_list(self.ns(status=None))
        out = capsys.readouterr().out
        assert 'gpus' in out and '0' in out

    def test_start_tops_up_to_target_worker_count(self, isolated_queue, monkeypatch):
        spawned = []

        class FakeProc:
            def __init__(self, pid):
                self.pid = pid

        def fake_popen(*args, **kwargs):
            pid = 1000 + len(spawned)
            spawned.append(pid)
            return FakeProc(pid)

        monkeypatch.setattr(cli.subprocess, 'Popen', fake_popen)
        cli.cmd_start(self.ns(workers=2))
        assert len(spawned) == 2
        with db.connect() as conn:
            # cmd_start only spawns processes; it does not register them in `workers`
            # itself - that happens inside worker.run() once each process starts.
            assert db.list_workers(conn) == []

    def test_start_forwards_gpu_pool_into_each_worker_env(self, isolated_queue, monkeypatch):
        seen_envs = []

        class FakeProc:
            def __init__(self, pid):
                self.pid = pid

        def fake_popen(*args, **kwargs):
            seen_envs.append(kwargs['env'])
            return FakeProc(1000 + len(seen_envs))

        monkeypatch.setattr(cli.subprocess, 'Popen', fake_popen)
        cli.cmd_start(self.ns(workers=2, gpu_pool='0, 1'))
        assert len(seen_envs) == 2
        assert all(env['QNAS_QUEUE_GPU_POOL'] == '0,1' for env in seen_envs)

    def test_start_without_gpu_pool_does_not_set_env_var(self, isolated_queue, monkeypatch):
        seen_envs = []

        class FakeProc:
            def __init__(self, pid):
                self.pid = pid

        monkeypatch.setattr(cli.subprocess, 'Popen',
                             lambda *a, **kw: (seen_envs.append(kw['env']), FakeProc(1))[1])
        cli.cmd_start(self.ns(workers=1))
        assert 'QNAS_QUEUE_GPU_POOL' not in seen_envs[0]

    def test_start_with_invalid_gpu_pool_exits(self):
        with pytest.raises(SystemExit, match='gpu-pool'):
            cli.cmd_start(self.ns(workers=1, gpu_pool='0,x'))

    def test_start_does_not_spawn_past_target_when_workers_already_alive(
            self, isolated_queue, monkeypatch):
        with db.connect() as conn:
            db.register_worker(conn, os.getpid())  # a real, alive pid

        spawned = []
        monkeypatch.setattr(cli.subprocess, 'Popen',
                             lambda *a, **kw: spawned.append(1) or None)
        cli.cmd_start(self.ns(workers=1))
        assert spawned == []

    def test_stop_signals_every_alive_worker(self, isolated_queue, monkeypatch):
        with db.connect() as conn:
            db.register_worker(conn, os.getpid())

        real_kill = os.kill
        killed = []

        def fake_kill(pid, sig):
            if sig == 0:  # db.pid_alive's liveness probe - let it through for real
                return real_kill(pid, sig)
            killed.append((pid, sig))

        monkeypatch.setattr(cli.os, 'kill', fake_kill)
        cli.cmd_stop(self.ns())
        assert killed == [(os.getpid(), cli.signal.SIGTERM)]

    def seed_running_with_log(self, isolated_queue, lines):
        job_id = self.seed('running')
        log_path = isolated_queue / '.qnas_queue' / 'logs' / f'job_{job_id}.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(lines)
        with db.connect() as conn:
            db.update_job(conn, job_id, log_path=f'.qnas_queue/logs/job_{job_id}.log')
        return job_id

    def test_logs_all_jobs_shows_each_running_job(self, isolated_queue, capsys):
        # Non-follow mode reuses _tail_sources, which headers each source with
        # "==> label <==" rather than prefixing every line (that per-line
        # [label] prefix is _tail_follow's follow-mode behavior instead).
        a = self.seed_running_with_log(isolated_queue, 'from a\n')
        b = self.seed_running_with_log(isolated_queue, 'from b\n')
        cli.cmd_logs(self.ns(id=None, follow=False, lines=50, all_jobs=True))
        out = capsys.readouterr().out
        assert f'job {a}' in out and 'from a' in out
        assert f'job {b}' in out and 'from b' in out

    def test_logs_all_jobs_follow_prefixes_each_line_by_job_id(
            self, isolated_queue, capsys, monkeypatch):
        a = self.seed_running_with_log(isolated_queue, 'from a\n')
        b = self.seed_running_with_log(isolated_queue, 'from b\n')
        monkeypatch.setattr(cli.time, 'sleep', lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
        cli.cmd_logs(self.ns(id=None, follow=True, lines=50, all_jobs=True))
        out = capsys.readouterr().out
        assert f'[job {a}] from a' in out
        assert f'[job {b}] from b' in out

    def test_logs_all_jobs_with_no_running_jobs(self, capsys):
        cli.cmd_logs(self.ns(id=None, follow=False, lines=50, all_jobs=True))
        assert 'No jobs currently running' in capsys.readouterr().out

    def test_logs_all_jobs_rejects_explicit_id(self):
        with pytest.raises(SystemExit, match='all-jobs'):
            cli.cmd_logs(self.ns(id=5, follow=False, lines=50, all_jobs=True))

    def seed_job_with_experiment(self, isolated_queue, mode='evolve', log_qnas_text=None,
                                 max_generations=300, gpu_ids=None):
        """A running job with a real experiment dir (for log_QNAS.txt) and a real
        frozen config snapshot (for max_generations) - what `watch`'s summary row
        needs, as opposed to seed()'s bare row with no files behind it."""
        with db.connect() as conn:
            job_id = add(conn, mode=mode, gpu_ids=gpu_ids)
            db.update_job(conn, job_id, status='running', started_at=db.now_iso())
        exp_dir = isolated_queue / f'exp_{job_id}'
        exp_dir.mkdir(parents=True, exist_ok=True)
        if log_qnas_text is not None:
            (exp_dir / 'log_QNAS.txt').write_text(log_qnas_text)
        snapshot = isolated_queue / '.qnas_queue' / 'configs' / f'job_{job_id}_cfg.yml'
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(f'QNAS:\n  max_generations: {max_generations}\n')
        with db.connect() as conn:
            db.update_job(conn, job_id, experiment_path=f'exp_{job_id}',
                          config_snapshot=f'.qnas_queue/configs/job_{job_id}_cfg.yml')
            return db.get_job(conn, job_id)

    def test_parse_generation_summary_returns_last_block(self, tmp_path):
        # best_so_far_id is logged as a Python list ("[gen, ind]", comma-SPACE
        # included) - see qnas.py's log_data() - not an underscore-joined string;
        # the fixture must match that exactly, or a regex regression that breaks
        # on the embedded space (see test_best_so_far_id_with_embedded_space_is_
        # still_parsed below) would go unnoticed.
        log = tmp_path / 'log_QNAS.txt'
        log.write_text(
            "INFO: qnas: 2026-01-01 00:00:00,000 - New generation finished running!\n"
            "- Generation: 1\n"
            "- Best so far: [0, 0] --> 0.50000\n"
            "- Fitnesses: [0.5]\n"
            "INFO: qnas: 2026-01-01 00:01:00,000 - New generation finished running!\n"
            "- Generation: 2\n"
            "- Best so far: [2, 3] --> 0.77123\n"
            "- Fitnesses: [0.6, 0.77123]\n"
        )
        assert cli._parse_generation_summary(log) == {
            'generation': 2, 'best_id': '[2, 3]', 'best_fitness': 0.77123,
            'best_gen': 2, 'best_ind': 3,
            'fitness_delta': pytest.approx(0.77123 - 0.5),
            'fitness_spread': pytest.approx(0.77123 - 0.6),
        }

    def test_best_so_far_id_with_embedded_space_is_still_parsed(self, tmp_path):
        """Regression: `\\S+` in _BEST_SO_FAR_RE could never match "[1, 13]" (the
        real on-disk format) because of the comma-space inside the brackets,
        silently leaving best_fitness as None in the watch dashboard."""
        log = tmp_path / 'log_QNAS.txt'
        log.write_text("- Generation: 1\n- Best so far: [1, 13] --> 70.00000\n")
        summary = cli._parse_generation_summary(log)
        assert summary['best_id'] == '[1, 13]'
        assert summary['best_fitness'] == 70.0
        assert (summary['best_gen'], summary['best_ind']) == (1, 13)

    def test_fitnesses_array_wrapped_across_lines_is_parsed(self, tmp_path):
        """numpy's default repr wraps a long fitness array across several lines -
        the spread calc must still see every value, not just the first line."""
        log = tmp_path / 'log_QNAS.txt'
        log.write_text(
            "- Generation: 0\n"
            "- Best so far: [0, 10] --> 66.50000\n"
            "- Fitnesses: [66.5  65.5  61.7  61.1  60.5  59.1  58.8  58.1  57.9  57.5\n"
            " 56.6  52.4  50.8  49.69 46.4  34.9  21.17  7.98]\n"
            "- Fitnesses without penalties: [66.5 65.5 61.7 61.1 60.5 59.1 58.8 58.1\n"
            " 57.9 57.5 56.6 52.4 50.8 49.7 46.4 34.9 21.2  8. ]\n"
        )
        summary = cli._parse_generation_summary(log)
        assert summary['fitness_spread'] == pytest.approx(66.5 - 7.98)

    def test_fitness_delta_between_last_two_generations(self, tmp_path):
        log = tmp_path / 'log_QNAS.txt'
        log.write_text(
            "- Generation: 0\n- Best so far: [0, 0] --> 60.00000\n- Fitnesses: [60.0]\n"
            "- Generation: 1\n- Best so far: [1, 5] --> 63.50000\n- Fitnesses: [63.5]\n"
        )
        summary = cli._parse_generation_summary(log)
        assert summary['fitness_delta'] == pytest.approx(3.5)

    def test_fitness_delta_is_none_on_the_first_generation(self, tmp_path):
        log = tmp_path / 'log_QNAS.txt'
        log.write_text("- Generation: 0\n- Best so far: [0, 0] --> 60.00000\n")
        summary = cli._parse_generation_summary(log)
        assert summary['fitness_delta'] is None

    def test_parse_generation_summary_missing_or_empty_file_returns_none(self, tmp_path):
        assert cli._parse_generation_summary(tmp_path / 'missing.txt') is None
        empty = tmp_path / 'log_QNAS.txt'
        empty.write_text('')
        assert cli._parse_generation_summary(empty) is None

    def test_job_max_generations_reads_config_snapshot(self, isolated_queue):
        cfg = isolated_queue / 'cfg.yml'
        cfg.write_text('QNAS:\n  max_generations: 150\n')
        with db.connect() as conn:
            job_id = add(conn, config='cfg.yml')
            job = db.get_job(conn, job_id)
        cache = {}
        assert cli._job_max_generations(job, cache) == 150
        assert cache[job_id] == 150  # cached, so a 2nd call would not re-read the file

    def test_job_max_generations_missing_config_returns_none(self, isolated_queue):
        with db.connect() as conn:
            job_id = add(conn, config='does-not-exist.yml')
            job = db.get_job(conn, job_id)
        assert cli._job_max_generations(job, {}) is None

    def test_job_summary_row_starting_when_no_generation_yet(self, isolated_queue):
        job = self.seed_job_with_experiment(isolated_queue, mode='evolve')
        row = cli._job_summary_row(job, {})
        assert row[3] == 'starting...' and row[4] == '-'

    def test_job_summary_row_retrain_shows_na_instead_of_starting(self, isolated_queue):
        job = self.seed_job_with_experiment(isolated_queue, mode='retrain')
        row = cli._job_summary_row(job, {})
        assert row[3] == 'n/a'

    def test_job_summary_row_formats_generation_fitness_and_gpu(self, isolated_queue):
        job = self.seed_job_with_experiment(
            isolated_queue, mode='evolve', gpu_ids='0',
            log_qnas_text='- Generation: 5\n- Best so far: [5, 1] --> 0.61234\n',
            max_generations=300,
        )
        row = cli._job_summary_row(job, {})
        assert row[:6] == (str(job['id']), 'evolve', '0', '5/300', '0.61234',
                          'gen 5/ind 1')
        assert row[8] != '-'  # elapsed, formatted since started_at is set
        assert row[9] != '-'  # ETA, extrapolated from generation 5's progress

    def test_job_summary_row_delta_and_spread(self, isolated_queue):
        job = self.seed_job_with_experiment(
            isolated_queue, mode='evolve',
            log_qnas_text=(
                '- Generation: 0\n- Best so far: [0, 0] --> 60.00000\n'
                '- Fitnesses: [60.0, 55.0]\n'
                '- Generation: 1\n- Best so far: [1, 2] --> 63.50000\n'
                '- Fitnesses: [63.5, 61.0]\n'
            ),
        )
        row = cli._job_summary_row(job, {})
        assert row[6] == '+3.50000'  # fitness delta vs the previous generation
        assert row[7] == '2.50000'  # spread within generation 1 (63.5 - 61.0)

    def test_job_summary_row_dashes_when_no_generation_yet(self, isolated_queue):
        job = self.seed_job_with_experiment(isolated_queue, mode='evolve')
        row = cli._job_summary_row(job, {})
        assert row[4:7] == ('-', '-', '-')
        assert row[9] == '-'  # ETA

    def test_format_eta_extrapolates_from_progress_so_far(self):
        started = (datetime.now(timezone.utc) - timedelta(seconds=100)).isoformat()
        # 2 generations completed (0 and 1) in 100s -> 50s/gen; 3 remain of max 5.
        eta = cli._format_eta(started, generation=1, max_gen=5)
        assert eta == '00:02:30'

    def test_format_eta_dash_without_enough_data(self):
        now = datetime.now(timezone.utc).isoformat()
        assert cli._format_eta(None, 1, 5) == '-'
        assert cli._format_eta(now, None, 5) == '-'
        assert cli._format_eta(now, 1, None) == '-'
        assert cli._format_eta(now, 0, 5) == '-'  # generation 0: one data point

    def test_watch_layout_renders_every_job_row_without_clipping(self, isolated_queue):
        # Regression: an earlier version wrapped the summary Table in its own
        # Panel and under-counted the combined border/header/title height, so
        # the Layout's fixed `size` clipped exactly the data row(s) - headers
        # rendered, but every job's actual gen/fitness numbers were cut off.
        jobs = [self.seed_job_with_experiment(isolated_queue, gpu_ids=str(i))
                for i in range(3)]
        console = Console(width=100, height=40)
        layout = cli._watch_layout(console, jobs, {}, deque())
        with console.capture() as cap:
            console.print(layout)
        out = cap.get()
        for job in jobs:
            assert f"{job['id']}" in out and "starting..." in out

    def test_drain_new_detail_lines_buffers_raw_job_id_tuples(self, isolated_queue):
        job = self.seed_running_with_log(isolated_queue, 'hello\n')
        detail_buffer = deque(maxlen=10)
        with db.connect() as conn:
            cli._drain_new_detail_lines({}, conn, detail_buffer)
        assert list(detail_buffer) == [(job, 'hello')]

    def test_watch_layout_pads_job_id_prefixes_to_a_common_width(self, isolated_queue):
        # job ids 2 and 10 have different digit counts - their "[job N]"
        # prefixes in the detail panel must still line up.
        console = Console(width=100, height=40)
        detail_buffer = deque([(2, 'from job two'), (10, 'from job ten')])
        layout = cli._watch_layout(console, [], {}, detail_buffer)
        with console.capture() as cap:
            console.print(layout)
        out = cap.get()
        assert '[job  2] from job two' in out
        assert '[job 10] from job ten' in out

    def test_watch_layout_colors_rows_by_job_id(self, isolated_queue):
        jobs = [self.seed_job_with_experiment(isolated_queue, gpu_ids=str(i))
                for i in range(2)]
        assert cli._job_style(jobs[0]['id']) != cli._job_style(jobs[1]['id'])
        console = Console(width=100, height=40, force_terminal=True, color_system='standard')
        layout = cli._watch_layout(console, jobs, {}, deque())
        with console.capture() as cap:
            console.print(layout)
        out = cap.get()
        assert '\x1b[' in out  # ANSI escapes present - rows are actually styled

    def test_cmd_watch_requires_a_tty(self, monkeypatch):
        monkeypatch.setattr(cli.sys.stdout, 'isatty', lambda: False)
        with pytest.raises(SystemExit, match='interactive terminal'):
            cli.cmd_watch(self.ns())

    def test_parser_wires_every_subcommand(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(cli, 'cmd_list', lambda args: seen.update(status=args.status))
        monkeypatch.setattr('sys.argv', ['qnas-queue', 'list', '--status', 'done'])
        cli.main()
        assert seen == {'status': 'done'}
