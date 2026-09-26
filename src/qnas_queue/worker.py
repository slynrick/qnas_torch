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


def _run_job(job, pid):
    global _current_proc

    log_path = db.LOG_DIR / f"job_{job['id']}.log"
    with db.connect() as conn:
        db.update_job(conn, job["id"], log_path=str(log_path.relative_to(PROJECT_ROOT)))
        db.set_worker(conn, pid, current_job_id=job["id"])

    argv = build_argv(job["mode"], db.config_for_run(job), job["experiment_path"],
                      job["extra_args"])

    # gpu_ids (set via `qnas-queue add --gpu-ids`) restricts this job to specific
    # GPUs by narrowing CUDA_VISIBLE_DEVICES for its subprocess only - every CUDA
    # index the job's own code sees (evaluation.py's per-individual round robin,
    # cnn/train.py's train.device) is relative to this narrowed set, not the host's
    # full device list. Unset (None): the job sees every GPU, as before this option
    # existed.
    env = os.environ.copy()
    if job["gpu_ids"]:
        env["CUDA_VISIBLE_DEVICES"] = job["gpu_ids"]

    with open(log_path, "a") as log_file:
        log_file.write(f"\n=== job {job['id']} started {db.now_iso()} ===\n$ {' '.join(argv)}\n")
        if job["gpu_ids"]:
            log_file.write(f"CUDA_VISIBLE_DEVICES={job['gpu_ids']}\n")
        log_file.write("\n")
        log_file.flush()
        _current_proc = subprocess.Popen(
            argv, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,
        )
        with db.connect() as conn:
            db.update_job(conn, job["id"], pgid=os.getpgid(_current_proc.pid))
        returncode = _current_proc.wait()

    status = "stopped" if _stop_requested else ("done" if returncode == 0 else "failed")
    with db.connect() as conn:
        db.update_job(conn, job["id"], status=status, exit_code=returncode,
                       finished_at=db.now_iso(), pgid=None)
        db.set_worker(conn, pid, current_job_id=None)
    _current_proc = None


def _gpu_pool_from_env():
    """$QNAS_QUEUE_GPU_POOL (set by cli.py's cmd_start on this worker's own
    subprocess env, from `start --gpu-pool`), as a set of GPU indices, or None
    if this worker fleet isn't using a pool. Already validated by
    cli._parse_gpu_ids before cmd_start ever set the env var."""
    value = os.environ.get("QNAS_QUEUE_GPU_POOL")
    if not value:
        return None
    return {int(x) for x in value.split(",") if x.strip()}


def run():
    pid = os.getpid()
    gpu_pool = _gpu_pool_from_env()
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)

    with db.connect() as conn:
        db.register_worker(conn, pid)
        resumed = db.resume_stopped_jobs(conn)
    for job_id in resumed:
        print(f"Resuming stopped job {job_id}.")

    while not _stop_requested:
        with db.connect() as conn:
            job = db.claim_next_job(conn, pid, gpu_pool=gpu_pool)
        if job is None:
            time.sleep(POLL_INTERVAL)
            continue
        _run_job(job, pid)

    with db.connect() as conn:
        db.remove_worker(conn, pid)


if __name__ == "__main__":
    run()
