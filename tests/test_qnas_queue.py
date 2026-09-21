import argparse
import os
import sqlite3
from pathlib import Path

import pytest

from qnas_queue import cli, db, runner


@pytest.fixture(autouse=True)
def isolated_queue(tmp_path, monkeypatch):
    """Point every queue path at tmp_path so tests never touch the real .qnas_queue/."""
    root = tmp_path / 'project'
    queue = root / '.qnas_queue'
    monkeypatch.setattr(db, 'PROJECT_ROOT', root)
    monkeypatch.setattr(db, 'QUEUE_DIR', queue)
    monkeypatch.setattr(db, 'DB_PATH', queue / 'queue.db')
    monkeypatch.setattr(db, 'LOG_DIR', queue / 'logs')
    monkeypatch.setattr(db, 'WORKER_PID_FILE', queue / 'worker.pid')
    monkeypatch.setattr(db, 'WORKER_LOG_PATH', queue / 'worker.log')
    root.mkdir()
    return root


def add(conn, mode='pipeline', config='c.yml', exp='exp1', extra='', priority=0):
    return db.add_job(conn, mode, config, exp, extra, priority)


class TestDb:
    def test_connect_creates_schema_and_worker_row(self):
        with db.connect() as conn:
            assert db.get_worker(conn)['status'] == 'stopped'
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
            claimed = [db.claim_next_job(conn)['id'] for _ in range(3)]
            assert claimed == [high, low, low2]
            assert db.claim_next_job(conn) is None

    def test_claim_marks_running_with_start_time(self):
        with db.connect() as conn:
            add(conn)
            job = db.claim_next_job(conn)
        assert job['status'] == 'running' and job['started_at']

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

    def test_set_worker_updates_timestamp(self):
        with db.connect() as conn:
            db.set_worker(conn, pid=123, status='running')
            worker = db.get_worker(conn)
        assert worker['pid'] == 123 and worker['updated_at']

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
        assert 'not running' in capsys.readouterr().out

    def test_parser_wires_every_subcommand(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(cli, 'cmd_list', lambda args: seen.update(status=args.status))
        monkeypatch.setattr('sys.argv', ['qnas-queue', 'list', '--status', 'done'])
        cli.main()
        assert seen == {'status': 'done'}
