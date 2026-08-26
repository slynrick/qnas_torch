import os
import signal
import subprocess
import threading
import time

from qnas_queue import db
from qnas_queue.runner import PROJECT_ROOT, build_argv

POLL_INTERVAL = 3
GRACE_PERIOD = 15

_current_proc = None
_stop_requested = False


def _handle_stop_signal(signum, frame):
    global _stop_requested
    _stop_requested = True
    proc = _current_proc
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return

    def _escalate():
        if proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    threading.Timer(GRACE_PERIOD, _escalate).start()


def _run_job(job):
    global _current_proc

    log_path = db.LOG_DIR / f"job_{job['id']}.log"
    with db.connect() as conn:
        db.update_job(conn, job["id"], log_path=str(log_path.relative_to(PROJECT_ROOT)))
        db.set_worker(conn, current_job_id=job["id"])

    argv = build_argv(job["mode"], job["config_path"], job["experiment_path"], job["extra_args"])

    with open(log_path, "a") as log_file:
        log_file.write(f"\n=== job {job['id']} started {db.now_iso()} ===\n$ {' '.join(argv)}\n\n")
        log_file.flush()
        _current_proc = subprocess.Popen(
            argv, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with db.connect() as conn:
            db.update_job(conn, job["id"], pgid=os.getpgid(_current_proc.pid))
        returncode = _current_proc.wait()

    status = "stopped" if _stop_requested else ("done" if returncode == 0 else "failed")
    with db.connect() as conn:
        db.update_job(conn, job["id"], status=status, exit_code=returncode,
                       finished_at=db.now_iso(), pgid=None)
        db.set_worker(conn, current_job_id=None)
    _current_proc = None


def run():
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)

    with db.connect() as conn:
        db.set_worker(conn, pid=os.getpid(), status="running",
                       current_job_id=None, started_at=db.now_iso())
        resumed = db.resume_stopped_jobs(conn)
    for job_id in resumed:
        print(f"Resuming stopped job {job_id}.")

    while not _stop_requested:
        with db.connect() as conn:
            job = db.claim_next_job(conn)
        if job is None:
            time.sleep(POLL_INTERVAL)
            continue
        _run_job(job)

    with db.connect() as conn:
        db.set_worker(conn, status="stopped", pid=None, current_job_id=None)
    try:
        db.WORKER_PID_FILE.unlink()
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    run()
