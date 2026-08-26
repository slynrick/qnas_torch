"""Thin wrappers around qnas_queue.db for the dashboard: read helpers plus the
mutating actions (cancel/retry/remove/start/stop) that the sidebar exposes.
Reuses qnas_queue's own DB/process logic instead of re-implementing it so the
dashboard and the `qnas-queue` CLI can never disagree about queue semantics.
"""

import os
import signal
import subprocess
import sys

from qnas_queue import db


def list_jobs(status=None):
    with db.connect() as conn:
        return [dict(row) for row in db.list_jobs(conn, status=status)]


def get_job(job_id):
    with db.connect() as conn:
        row = db.get_job(conn, job_id)
        return dict(row) if row else None


def get_worker():
    with db.connect() as conn:
        row = db.get_worker(conn)
        worker = dict(row) if row else None
        current = None
        if worker and worker.get("current_job_id"):
            current_row = db.get_job(conn, worker["current_job_id"])
            current = dict(current_row) if current_row else None
    if worker is not None:
        worker["alive"] = db.pid_alive(worker.get("pid"))
    return worker, current


def status_counts():
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        ).fetchall()
    return {row["status"]: row["n"] for row in rows}


# --- mutating actions -------------------------------------------------

def cancel_job(job_id):
    with db.connect() as conn:
        job = db.get_job(conn, job_id)
        if job is None:
            return False, f"No job with id {job_id}."
        if job["status"] != "queued":
            return False, f"Job {job_id} is {job['status']} - only queued jobs can be cancelled."
        db.cancel_job(conn, job_id)
    return True, f"Job {job_id} cancelled."


def retry_job(job_id):
    with db.connect() as conn:
        job = db.get_job(conn, job_id)
        if job is None:
            return False, f"No job with id {job_id}."
        if job["status"] not in ("failed", "stopped", "cancelled"):
            return False, f"Job {job_id} is {job['status']} - only failed/stopped/cancelled jobs can be retried."
        db.update_job(
            conn, job_id, status="queued", started_at=None, finished_at=None,
            exit_code=None, error_message=None, pgid=None,
        )
    return True, f"Job {job_id} re-queued."


def remove_job(job_id):
    with db.connect() as conn:
        job = db.get_job(conn, job_id)
        if job is None:
            return False, f"No job with id {job_id}."
        if job["status"] == "running":
            return False, f"Job {job_id} is running - stop it first."
        db.delete_job(conn, job_id)
    return True, f"Job {job_id} removed."


def start_worker():
    with db.connect() as conn:
        worker = db.get_worker(conn)
    if worker and db.pid_alive(worker["pid"]):
        return False, f"Worker already running (pid {worker['pid']})."

    db.ensure_dirs()
    log_file = open(db.WORKER_LOG_PATH, "a")
    proc = subprocess.Popen(
        [sys.executable, "-m", "qnas_queue.worker"],
        cwd=str(db.PROJECT_ROOT), env=os.environ.copy(),
        stdout=log_file, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    db.WORKER_PID_FILE.write_text(str(proc.pid))
    return True, f"Worker started (pid {proc.pid})."


def stop_worker():
    with db.connect() as conn:
        worker = db.get_worker(conn)
    pid = worker["pid"] if worker else None
    if not pid or not db.pid_alive(pid):
        return False, "Worker is not running."
    os.kill(pid, signal.SIGTERM)
    return True, f"Sent stop signal to worker (pid {pid}). Current job will be terminated."
