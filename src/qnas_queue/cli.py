import argparse
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import yaml

from qnas_queue import db

_COLUMNS = ["id", "mode", "status", "config", "experiment_path", "created_at", "started_at", "finished_at"]


def _row_values(row):
    return [
        row["id"], row["mode"], row["status"], Path(row["config_path"]).name,
        row["experiment_path"], row["created_at"] or "", row["started_at"] or "",
        row["finished_at"] or "",
    ]


def _print_table(rows):
    table = [_COLUMNS] + [[str(v) for v in _row_values(r)] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(_COLUMNS))]
    for i, row in enumerate(table):
        print("  ".join(val.ljust(widths[j]) for j, val in enumerate(row)))
        if i == 0:
            print("  ".join("-" * w for w in widths))


def cmd_add(args):
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        sys.exit(f"error: config file not found: {config_path}")
    try:
        with open(config_path) as f:
            yaml.safe_load(f)
    except yaml.YAMLError as e:
        sys.exit(f"error: config file is not valid YAML: {e}")

    try:
        stored_config_path = str(config_path.relative_to(db.PROJECT_ROOT))
    except ValueError:
        stored_config_path = str(config_path)

    with db.connect() as conn:
        job_id = db.add_job(
            conn, mode=args.mode, config_path=stored_config_path,
            experiment_path=args.experiment_path, extra_args=args.extra or "",
            priority=args.priority,
        )
    print(f"Queued job {job_id} ({args.mode}): {config_path.name} -> {args.experiment_path}")


def cmd_list(args):
    with db.connect() as conn:
        rows = db.list_jobs(conn, status=args.status)
    if not rows:
        print("Queue is empty.")
        return
    _print_table(rows)


def cmd_remove(args):
    with db.connect() as conn:
        job = db.get_job(conn, args.id)
        if job is None:
            sys.exit(f"error: no job with id {args.id}")
        if job["status"] == "running":
            sys.exit(f"error: job {args.id} is running - stop it first (qnas-queue stop)")
        db.delete_job(conn, args.id)
    print(f"Removed job {args.id}.")


def cmd_cancel(args):
    with db.connect() as conn:
        job = db.get_job(conn, args.id)
        if job is None:
            sys.exit(f"error: no job with id {args.id}")
        if job["status"] != "queued":
            sys.exit(f"error: job {args.id} is {job['status']} - only queued jobs can be cancelled")
        db.cancel_job(conn, args.id)
    print(f"Job {args.id} cancelled.")


def cmd_retry(args):
    with db.connect() as conn:
        job = db.get_job(conn, args.id)
        if job is None:
            sys.exit(f"error: no job with id {args.id}")
        if job["status"] not in ("failed", "stopped", "cancelled"):
            sys.exit(f"error: job {args.id} is {job['status']} - "
                     f"only failed/stopped/cancelled jobs can be retried")
        db.update_job(
            conn, args.id, status="queued", started_at=None, finished_at=None,
            exit_code=None, error_message=None, pgid=None,
        )
    print(f"Job {args.id} re-queued.")


def cmd_start(args):
    with db.connect() as conn:
        worker = db.get_worker(conn)
    if worker and db.pid_alive(worker["pid"]):
        print(f"Worker already running (pid {worker['pid']}).")
        return

    db.ensure_dirs()
    log_file = open(db.WORKER_LOG_PATH, "a")
    proc = subprocess.Popen(
        [sys.executable, "-m", "qnas_queue.worker"],
        cwd=str(db.PROJECT_ROOT), env=os.environ.copy(),
        stdout=log_file, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    db.WORKER_PID_FILE.write_text(str(proc.pid))
    print(f"Worker started (pid {proc.pid}). Worker log: {db.WORKER_LOG_PATH}")


def _stop_orphaned_jobs():
    """With no live worker, a job still marked 'running' is stale (the worker died
    or was killed without cleaning up). Mark it 'stopped' - the same state a
    graceful stop leaves - so the next `start` resumes it, and terminate its
    process group if the subprocess outlived the worker."""

    with db.connect() as conn:
        for job in db.list_running_jobs(conn):
            pgid = job["pgid"]
            if pgid and db.pid_alive(pgid):
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            db.update_job(conn, job["id"], status="stopped", finished_at=db.now_iso(), pgid=None)
            print(f"Job {job['id']} was marked running without a worker - "
                  f"set to stopped; it will be retried on the next start.")


def cmd_stop(args):
    with db.connect() as conn:
        worker = db.get_worker(conn)
    pid = worker["pid"] if worker else None
    if not pid or not db.pid_alive(pid):
        print("Worker is not running.")
        _stop_orphaned_jobs()
        return
    os.kill(pid, signal.SIGTERM)
    print(f"Sent stop signal to worker (pid {pid}). Current job will be terminated.")


def cmd_status(args):
    with db.connect() as conn:
        worker = db.get_worker(conn)
        counts = {
            row["status"]: row["n"]
            for row in conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
        }
        current = db.get_job(conn, worker["current_job_id"]) if worker else None

    alive = worker is not None and db.pid_alive(worker["pid"])
    print(f"Worker: {'running' if alive else 'stopped'}" + (f" (pid {worker['pid']})" if alive else ""))
    if current:
        print(f"Current job: {current['id']} [{current['mode']}] "
              f"{Path(current['config_path']).name} -> {current['experiment_path']}")
        print(f"  started: {current['started_at']}")
        print(f"  log: {current['log_path']}")
    print("Queue counts: " + ", ".join(
        f"{status}={counts.get(status, 0)}"
        for status in ("queued", "running", "done", "failed", "stopped", "cancelled")
    ))


def _tail_lines(path, n):
    with open(path) as f:
        lines = deque(f, maxlen=n)
    sys.stdout.writelines(lines)


def _resolve_job(conn, job_id):
    """Picks the job `logs` should show: *job_id* if given, else the worker's
    current job, else the most recently touched non-queued job."""

    if job_id:
        job = db.get_job(conn, job_id)
        if job is None:
            sys.exit(f"error: no job with id {job_id}")
        return job
    worker = db.get_worker(conn)
    job = db.get_job(conn, worker["current_job_id"]) if worker else None
    if job is None:
        job = conn.execute(
            "SELECT * FROM jobs WHERE status != 'queued' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return job


def _job_log_sources(job, view):
    """Returns the [(label, path)] log files *view* ('detail', 'summary' or 'both')
    shows for *job*, in display order."""

    sources = []
    if view in ("detail", "both") and job["log_path"]:
        sources.append(("detail", db.resolve_path(job["log_path"])))
    if view in ("summary", "both"):
        # log_QNAS.txt is the evolution-level summary log (one entry per generation:
        # best-so-far, fitnesses, progressive-stage transitions - see
        # qnas_config.py::files_spec['log_file']), as opposed to the detail view's raw
        # stdout/stderr capture of the underlying evolve/retrain/pipeline subprocess.
        sources.append(("summary", db.resolve_path(job["experiment_path"]) / "log_QNAS.txt"))
    return sources


def _tail_sources(sources, n):
    for label, path in sources:
        if not path.exists():
            continue
        if len(sources) > 1:
            print(f"==> {label}: {path} <==")
        _tail_lines(path, n)


def _tail_follow(sources, job_id, pinned):
    """Tails *sources* ([(label, path)]), live. With several sources every line is
    prefixed with its label and the files are read round-robin, so lines land in
    (approximately) the order they were written; a source that doesn't exist yet
    (e.g. log_QNAS.txt before the first generation) is picked up once it appears.
    When the files stop growing because job *job_id* has reached a terminal status,
    returns True if the queue has already moved on to a different job (the caller
    should switch to that job's log) or False if there is nothing else to follow
    (or the user hit Ctrl-C). *pinned* jobs (an explicit id was requested) never
    trigger an auto-switch - it's the default "current job" view that should flow
    from one queued job into the next instead of going stale once its job finishes.
    """

    multi = len(sources) > 1
    handles = {}

    def emit(label, line):
        sys.stdout.write(f"[{label}] {line}" if multi else line)

    def open_source(label, path):
        if path in handles or not path.exists():
            return
        f = open(path)
        handles[path] = f
        for line in deque(f, maxlen=20):
            emit(label, line)

    try:
        while True:
            progressed = False
            for label, path in sources:
                open_source(label, path)
                f = handles.get(path)
                line = f.readline() if f else ""
                if line:
                    emit(label, line)
                    progressed = True
            if progressed:
                continue

            sys.stdout.flush()
            if not pinned:
                with db.connect() as conn:
                    current = db.get_job(conn, job_id)
                    if current is not None and current["status"] not in (
                            "queued", "running"):
                        next_job = _resolve_job(conn, None)
                        if next_job is not None and next_job["id"] != job_id:
                            return True
            time.sleep(0.5)
    except KeyboardInterrupt:
        return False
    finally:
        for f in handles.values():
            f.close()


def _wait_for_log(log_path, job_id):
    """In follow mode, a just-started (or just-switched-to) job may not have
    created its log file yet - poll for it instead of giving up immediately, so
    `logs -f` flows straight into the next job without the caller re-running it.
    Returns True once the file exists, False if the job already ended without
    ever creating one, or the user hit Ctrl-C.
    """

    try:
        while not log_path.exists():
            with db.connect() as conn:
                current = db.get_job(conn, job_id)
            if current is not None and current["status"] not in ("queued", "running"):
                return log_path.exists()
            time.sleep(0.5)
    except KeyboardInterrupt:
        return False
    return True


def cmd_logs(args):
    with db.connect() as conn:
        job = _resolve_job(conn, args.id)

    if job is None:
        print("No job logs available yet.")
        return

    while True:
        sources = _job_log_sources(job, args.view)
        if not sources:
            print("No job logs available yet.")
            return
        if not any(path.exists() for _, path in sources):
            wait_path = sources[0][1]
            if not args.follow or not _wait_for_log(wait_path, job["id"]):
                print(f"Log file not found yet: {wait_path}")
                return

        if not args.follow:
            _tail_sources(sources, args.lines)
            return

        switched = _tail_follow(sources, job["id"], pinned=bool(args.id))
        if not switched:
            return

        with db.connect() as conn:
            next_job = _resolve_job(conn, None)
        if next_job is None or next_job["id"] == job["id"]:
            return
        print(f"\n=== job {job['id']} finished - following job {next_job['id']} "
            f"({next_job['experiment_path']}) ===\n")
        job = next_job


def main():
    parser = argparse.ArgumentParser(prog="qnas-queue", description="QNAS experiment queue.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Queue a new experiment job.")
    p_add.add_argument("--mode", required=True, choices=["evolve", "retrain", "pipeline"],
                        help="How the job must be executed.")
    p_add.add_argument("--config", required=True, help="Path to the config yaml.")
    p_add.add_argument("--experiment-path", required=True, dest="experiment_path")
    p_add.add_argument("--extra", default="",
                        help="Extra args appended verbatim to the underlying command "
                             "(e.g. \"--dataset cifar10 --network_config default\").")
    p_add.add_argument("--priority", type=int, default=0)
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="List queued/past jobs.")
    p_list.add_argument(
        "--status",
        choices=["queued", "running", "done", "failed", "stopped", "cancelled"],
    )
    p_list.set_defaults(func=cmd_list)

    p_remove = sub.add_parser("remove", help="Delete a job from the queue/history.")
    p_remove.add_argument("id", type=int)
    p_remove.set_defaults(func=cmd_remove)

    p_cancel = sub.add_parser("cancel", help="Cancel a queued job (kept in history as 'cancelled').")
    p_cancel.add_argument("id", type=int)
    p_cancel.set_defaults(func=cmd_cancel)

    p_retry = sub.add_parser("retry", help="Re-queue a failed/stopped/cancelled job.")
    p_retry.add_argument("id", type=int)
    p_retry.set_defaults(func=cmd_retry)

    p_start = sub.add_parser(
        "start", help="Start the background worker (also resumes any stopped job).")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="Stop the worker and the currently running job.")
    p_stop.set_defaults(func=cmd_stop)

    p_status = sub.add_parser("status", help="Show worker and queue status.")
    p_status.set_defaults(func=cmd_status)

    p_logs = sub.add_parser("logs", help="Show logs for a job.")
    p_logs.add_argument("id", type=int, nargs="?", default=None,
                         help="Job id (defaults to the current/most recent job).")
    p_logs.add_argument("--follow", "-f", action="store_true")
    p_logs.add_argument("--lines", "-n", type=int, default=50)
    log_view = p_logs.add_mutually_exclusive_group()
    log_view.add_argument("--detail", dest="view", action="store_const", const="detail",
                           help="Raw subprocess stdout/stderr log (default).")
    log_view.add_argument("--summary", dest="view", action="store_const", const="summary",
                           help="Evolution-level summary log (log_QNAS.txt): "
                                "best-so-far, fitnesses and progressive-stage "
                                "transitions per generation.")
    log_view.add_argument("--both", dest="view", action="store_const", const="both",
                           help="Detail and summary logs together, each line "
                                "prefixed with [detail] / [summary].")
    p_logs.set_defaults(func=cmd_logs, view="detail")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
