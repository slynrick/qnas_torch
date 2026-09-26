import argparse
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from qnas_queue import db

_COLUMNS = ["id", "mode", "status", "gpus", "config", "experiment_path",
            "created_at", "started_at", "finished_at"]


def _row_values(row):
    return [
        row["id"], row["mode"], row["status"], row["gpu_ids"] or "-",
        Path(row["config_path"]).name, row["experiment_path"],
        row["created_at"] or "", row["started_at"] or "", row["finished_at"] or "",
    ]


def _print_table(rows):
    table = [_COLUMNS] + [[str(v) for v in _row_values(r)] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(_COLUMNS))]
    for i, row in enumerate(table):
        print("  ".join(val.ljust(widths[j]) for j, val in enumerate(row)))
        if i == 0:
            print("  ".join("-" * w for w in widths))


def _parse_gpu_ids(value, flag="--gpu-ids"):
    """Normalizes a GPU list ("0,1", "0, 1", "" or None) to a canonical
    comma-separated string of indices, or None if not given. Exits on anything
    that isn't a list of non-negative integers, so a typo is caught at `add`/
    `start` time rather than surfacing as a CUDA error deep in a queued job's
    log. Shared by `add --gpu-ids` and `start --gpu-pool` - *flag* names
    whichever one is being parsed, for the error message."""
    if not value:
        return None
    ids = [x.strip() for x in value.split(",") if x.strip()]
    if not ids or not all(x.isdigit() for x in ids):
        sys.exit(f"error: {flag} must be a comma-separated list of GPU indices "
                  f"(e.g. \"0,1\"), got: {value!r}")
    return ",".join(ids)


def cmd_add(args):
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        sys.exit(f"error: config file not found: {config_path}")
    try:
        with open(config_path) as f:
            yaml.safe_load(f)
    except yaml.YAMLError as e:
        sys.exit(f"error: config file is not valid YAML: {e}")

    gpu_ids = _parse_gpu_ids(args.gpu_ids)

    try:
        stored_config_path = str(config_path.relative_to(db.PROJECT_ROOT))
    except ValueError:
        stored_config_path = str(config_path)

    with db.connect() as conn:
        job_id = db.add_job(
            conn, mode=args.mode, config_path=stored_config_path,
            experiment_path=args.experiment_path, extra_args=args.extra or "",
            priority=args.priority, gpu_ids=gpu_ids,
        )
        # Inside the transaction: if the copy fails, the job row is rolled back too.
        snapshot = db.snapshot_config(job_id, config_path)
        db.update_job(conn, job_id, config_snapshot=snapshot)
    gpu_note = f" on GPU(s) {gpu_ids}" if gpu_ids else ""
    print(f"Queued job {job_id} ({args.mode}): {config_path.name} -> "
          f"{args.experiment_path}{gpu_note}")
    print(f"  config frozen at {snapshot} - later edits to {config_path.name} do not affect this job")


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
        db.remove_snapshot(job)
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
    note = " (runs its config snapshot from when it was added)" if job["config_snapshot"] else ""
    print(f"Job {args.id} re-queued{note}.")


def _requeue_orphaned_jobs(conn):
    """A job still marked 'running' whose worker_pid is no longer alive is stale
    (that worker died, or was killed, without cleaning up). Mark it 'stopped' -
    the same state a graceful stop leaves - so the next `start` resumes it, and
    terminate its process group if the subprocess outlived its worker. A job
    with no worker_pid on record (pre-multi-worker row) is treated the same way
    whenever no worker at all is alive, matching the old single-worker
    behavior."""

    workers_alive = any(db.pid_alive(w["pid"]) for w in db.list_workers(conn))
    for job in db.list_running_jobs(conn):
        worker_pid = job["worker_pid"]
        orphaned = not db.pid_alive(worker_pid) if worker_pid else not workers_alive
        if not orphaned:
            continue
        pgid = job["pgid"]
        if pgid and db.pid_alive(pgid):
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        db.update_job(conn, job["id"], status="stopped", finished_at=db.now_iso(), pgid=None)
        print(f"Job {job['id']} was marked running under a worker that is no longer "
              f"alive - set to stopped; it will be retried on the next start.")


def cmd_start(args):
    gpu_pool = _parse_gpu_ids(args.gpu_pool, flag="--gpu-pool")

    with db.connect() as conn:
        for worker in db.list_workers(conn):
            if not db.pid_alive(worker["pid"]):
                db.remove_worker(conn, worker["pid"])
        _requeue_orphaned_jobs(conn)
        alive = [w["pid"] for w in db.list_workers(conn)]

    target = args.workers
    if len(alive) >= target:
        print(f"{len(alive)} worker(s) already running (pid(s) "
              f"{', '.join(map(str, alive))}); target concurrency is {target}.")
        return

    db.ensure_dirs()
    log_file = open(db.WORKER_LOG_PATH, "a")
    env = os.environ.copy()
    if gpu_pool:
        env["QNAS_QUEUE_GPU_POOL"] = gpu_pool
    started = []
    for _ in range(target - len(alive)):
        proc = subprocess.Popen(
            [sys.executable, "-m", "qnas_queue.worker"],
            cwd=str(db.PROJECT_ROOT), env=env,
            stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started.append(proc.pid)
    pool_note = f" sharing GPU pool {gpu_pool}" if gpu_pool else ""
    print(f"Started {len(started)} worker(s) (pid(s) {', '.join(map(str, started))}){pool_note}. "
          f"Worker log: {db.WORKER_LOG_PATH}")


def cmd_stop(args):
    with db.connect() as conn:
        alive = [w["pid"] for w in db.list_workers(conn) if db.pid_alive(w["pid"])]
    if not alive:
        print("No worker is running.")
        with db.connect() as conn:
            _requeue_orphaned_jobs(conn)
        return
    for pid in alive:
        os.kill(pid, signal.SIGTERM)
    print(f"Sent stop signal to {len(alive)} worker(s) (pid(s) {', '.join(map(str, alive))}). "
          f"Their running jobs will be terminated.")


def cmd_status(args):
    with db.connect() as conn:
        workers = db.list_workers(conn)
        counts = {
            row["status"]: row["n"]
            for row in conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
        }
        running_by_id = {job["id"]: job for job in db.list_running_jobs(conn)}

    alive = [w for w in workers if db.pid_alive(w["pid"])]
    print(f"Workers: {len(alive)} running" +
          (f" (pid(s) {', '.join(str(w['pid']) for w in alive)})" if alive else ""))
    for worker in alive:
        job = running_by_id.get(worker["current_job_id"])
        if job is None:
            print(f"  worker {worker['pid']}: idle")
            continue
        gpu_note = f" [GPU {job['gpu_ids']}]" if job["gpu_ids"] else ""
        print(f"  worker {worker['pid']}: job {job['id']} [{job['mode']}] "
              f"{Path(job['config_path']).name} -> {job['experiment_path']}{gpu_note}")
        print(f"    started: {job['started_at']}  log: {job['log_path']}")
    print("Queue counts: " + ", ".join(
        f"{status}={counts.get(status, 0)}"
        for status in ("queued", "running", "done", "failed", "stopped", "cancelled")
    ))


def _tail_lines(path, n):
    with open(path) as f:
        lines = deque(f, maxlen=n)
    sys.stdout.writelines(lines)


def _resolve_job(conn, job_id):
    """Picks the job `logs` should show: *job_id* if given, else the most
    recently started currently-running job (with several workers, several
    jobs may be running at once - the newest one is the most likely one the
    caller just queued and is waiting on), else the most recently touched
    non-queued job."""

    if job_id:
        job = db.get_job(conn, job_id)
        if job is None:
            sys.exit(f"error: no job with id {job_id}")
        return job
    running = db.list_running_jobs(conn)
    if running:
        return running[-1]
    return conn.execute(
        "SELECT * FROM jobs WHERE status != 'queued' ORDER BY id DESC LIMIT 1"
    ).fetchone()


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


def _running_log_sources(conn):
    """[(job_id, path)] for every job currently 'running' that has a log path
    yet - one entry per concurrently running job, as opposed to
    _job_log_sources's several views of a single job. Returns the raw job id
    (not a pre-formatted label) so each caller can format/align/style it as
    it needs - `watch`'s detail panel pads and colors it, `logs --all-jobs`
    keeps the plain unpadded "job N" text."""
    return [
        (job["id"], db.resolve_path(job["log_path"]))
        for job in db.list_running_jobs(conn) if job["log_path"]
    ]


def _tail_all_running(args):
    """`logs --all-jobs`: with several workers, several jobs can be running at
    once - _resolve_job's single "current job" pick would only show one of
    them. This instead tails/follows every running job's detail log
    simultaneously, each line prefixed by its job id, and picks up newly
    started jobs (and drops finished ones) as the queue moves on."""

    with db.connect() as conn:
        sources = _running_log_sources(conn)
    if not sources:
        print("No jobs currently running.")
        return

    if not args.follow:
        _tail_sources([(f"job {jid}", path) for jid, path in sources], args.lines)
        return

    handles = {}

    def open_new(conn):
        for job_id, path in _running_log_sources(conn):
            if path in handles or not path.exists():
                continue
            label = f"job {job_id}"
            f = open(path)
            handles[path] = (label, f)
            for line in deque(f, maxlen=20):
                print(f"[{label}] {line}", end="")

    with db.connect() as conn:
        open_new(conn)
    try:
        while True:
            progressed = False
            for label, f in handles.values():
                line = f.readline()
                if line:
                    print(f"[{label}] {line}", end="")
                    progressed = True
            with db.connect() as conn:
                open_new(conn)
            if not progressed:
                sys.stdout.flush()
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for _, f in handles.values():
            f.close()


# Matches the per-generation message qnas.py's evolve loop logs to log_QNAS.txt
# (src/qnas.py:569-575: "- Generation: N" / "- Best so far: id --> fitness" /
# "- Fitnesses: [...]"). `watch`'s summary table is coupled to this exact text
# on purpose (see the plan/PR for this feature) - update both together if that
# message ever changes.
_GENERATION_RE = re.compile(r"^- Generation: (\d+)")
# best_so_far_id is logged as a Python list, e.g. "[1, 13]" - the comma-space
# inside it means `\S+` never matches (it stops at the space, leaving " --> "
# unable to follow immediately), so this must allow whitespace in the id.
_BEST_SO_FAR_RE = re.compile(r"^- Best so far: (.+) --> ([\d.]+)")
_BEST_ID_GEN_IND_RE = re.compile(r"\[(\d+),\s*(\d+)\]")
# "- Fitnesses: " (colon right after the word) - deliberately does NOT match
# "- Fitnesses without penalties: [...]", the very next line qnas.py logs.
_FITNESSES_RE = re.compile(r"^- Fitnesses: (.*)")
_FLOAT_RE = re.compile(r"-?\d+\.?\d*")


def _parse_generation_summary(log_qnas_path):
    """The latest generation logged to *log_qnas_path*, as {'generation': int,
    'best_id': str, 'best_fitness': float, 'best_gen': int or None,
    'best_ind': int or None, 'fitness_delta': float or None,
    'fitness_spread': float or None}, or None if the file doesn't exist yet or
    has no generation block yet (job just started, or a retrain-only job that
    never writes one).

    'fitness_delta' is how much best_fitness changed versus the PREVIOUS
    logged generation (None if there isn't one yet in the tail read below).
    'fitness_spread' is max - min across the latest generation's own
    population ("- Fitnesses: [...]", which may itself wrap across several
    lines - numpy's default array repr).

    Reads only the tail of the file - a generous margin so at least the last
    two generation blocks are present even for a large population (each
    block, plus the population-matrix dump logged between them, is roughly
    30-40 lines) - and keeps the LAST two complete blocks, since a job's own
    log only ever grows.
    """
    if not log_qnas_path.exists():
        return None
    with open(log_qnas_path) as f:
        tail = deque(f, maxlen=4000)

    blocks = []
    current = None
    fitness_chunks = None
    for line in tail:
        m = _GENERATION_RE.match(line)
        if m:
            current = {"generation": int(m.group(1)), "best_id": None,
                       "best_fitness": None, "fitnesses": None}
            blocks.append(current)
            fitness_chunks = None
            continue
        if current is None:
            continue
        m = _BEST_SO_FAR_RE.match(line)
        if m:
            current["best_id"] = m.group(1)
            current["best_fitness"] = float(m.group(2))
            continue
        m = _FITNESSES_RE.match(line)
        if m:
            fitness_chunks = [m.group(1)]
            if "]" in m.group(1):
                current["fitnesses"] = [float(x) for x in _FLOAT_RE.findall(fitness_chunks[0])]
                fitness_chunks = None
            continue
        if fitness_chunks is not None:
            fitness_chunks.append(line)
            if "]" in line:
                current["fitnesses"] = [float(x) for x in
                                        _FLOAT_RE.findall("".join(fitness_chunks))]
                fitness_chunks = None

    if not blocks:
        return None
    latest = blocks[-1]
    previous = blocks[-2] if len(blocks) >= 2 else None

    best_gen = best_ind = None
    if latest["best_id"] is not None:
        m = _BEST_ID_GEN_IND_RE.search(latest["best_id"])
        if m:
            best_gen, best_ind = int(m.group(1)), int(m.group(2))

    fitness_delta = None
    if (previous is not None and latest["best_fitness"] is not None
            and previous["best_fitness"] is not None):
        fitness_delta = latest["best_fitness"] - previous["best_fitness"]

    fitness_spread = None
    if latest["fitnesses"]:
        fitness_spread = max(latest["fitnesses"]) - min(latest["fitnesses"])

    return {
        "generation": latest["generation"],
        "best_id": latest["best_id"],
        "best_fitness": latest["best_fitness"],
        "best_gen": best_gen,
        "best_ind": best_ind,
        "fitness_delta": fitness_delta,
        "fitness_spread": fitness_spread,
    }


def _job_max_generations(job, cache):
    """QNAS.max_generations from *job*'s frozen config snapshot, cached per job
    id for the life of the `watch` process - a snapshot never changes once
    queued (db.snapshot_config), so it only needs reading once."""
    if job["id"] not in cache:
        try:
            config = yaml.safe_load(db.resolve_path(db.config_for_run(job)).read_text())
            cache[job["id"]] = (config or {}).get("QNAS", {}).get("max_generations")
        except (OSError, yaml.YAMLError):
            cache[job["id"]] = None
    return cache[job["id"]]


def _format_elapsed(started_at):
    if not started_at:
        return "-"
    elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(started_at)
    hours, rem = divmod(int(elapsed.total_seconds()), 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_eta(started_at, generation, max_gen):
    """Rough ETA to *max_gen*, extrapolated from this job's own progress so far
    (elapsed wall-clock time / generations completed) * generations
    remaining - no dependency on qnas.py's own periodic (every-5-generations)
    "Estimated time to finish" log line, so this stays current every time
    `watch` refreshes rather than only every 5 generations.

    "-" if there isn't enough information yet: job not started, no generation
    logged yet, max_generations unknown (e.g. a retrain job), or the very
    first generation (generation 0) - one data point isn't a rate.
    """
    if not started_at or generation is None or not max_gen or generation < 1:
        return "-"
    completed = generation + 1
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds()
    if elapsed <= 0:
        return "-"
    remaining = max(max_gen - completed, 0)
    eta_seconds = int((elapsed / completed) * remaining)
    hours, rem = divmod(eta_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _job_summary_row(job, gen_cache):
    """One `watch` table row for *job*: (id, mode, gpu, gen, best fitness,
    best origin, fitness delta, fitness spread, elapsed, ETA). "n/a" for
    gen/fitness on a retrain job (never writes generation blocks);
    "starting..." for an evolve/pipeline job that hasn't logged its first
    generation yet.

    best origin: "gen G/ind I" - which generation/individual produced the
    current best_fitness (parsed from best_so_far_id, e.g. "[1, 13]").
    fitness delta: best_fitness's change versus the previous logged
    generation (a signed value - positive means it just improved).
    fitness spread: max - min across the latest generation's own population,
    i.e. how converged/diverse that generation currently is.
    ETA: rough time remaining to max_generations - see _format_eta.
    """
    summary = _parse_generation_summary(
        db.resolve_path(job["experiment_path"]) / "log_QNAS.txt")
    if summary is None:
        gen_cell = "n/a" if job["mode"] == "retrain" else "starting..."
        fitness_cell = origin_cell = delta_cell = spread_cell = eta_cell = "-"
    else:
        max_gen = _job_max_generations(job, gen_cache)
        gen_cell = (f"{summary['generation']}/{max_gen}" if max_gen is not None
                    else str(summary["generation"])) if summary["generation"] is not None else "-"
        fitness_cell = (f"{summary['best_fitness']:.5f}"
                         if summary["best_fitness"] is not None else "-")
        origin_cell = (f"gen {summary['best_gen']}/ind {summary['best_ind']}"
                       if summary["best_gen"] is not None else "-")
        delta_cell = (f"{summary['fitness_delta']:+.5f}"
                     if summary["fitness_delta"] is not None else "-")
        spread_cell = (f"{summary['fitness_spread']:.5f}"
                       if summary["fitness_spread"] is not None else "-")
        eta_cell = _format_eta(job["started_at"], summary["generation"], max_gen)
    return (str(job["id"]), job["mode"], job["gpu_ids"] or "-", gen_cell,
            fitness_cell, origin_cell, delta_cell, spread_cell,
            _format_elapsed(job["started_at"]), eta_cell)


def _drain_new_detail_lines(handles, conn, detail_buffer):
    """Feeds newly-available lines from every running job's detail log into
    *detail_buffer* (a bounded deque of (job_id, line) tuples - kept raw,
    rather than a pre-formatted "[job N] ..." string, so _watch_layout can
    align/color the "[job N]" prefix consistently across every buffered line
    at render time, including ones appended before a job id's digit count
    changed). Adapted from _running_log_sources/open_new in
    _tail_all_running, but also closes and drops handles for jobs that
    stopped running, so a long `watch` session doesn't accumulate open file
    descriptors for finished jobs."""
    current = _running_log_sources(conn)
    current_paths = {path for _, path in current}
    for path in [p for p in handles if p not in current_paths]:
        _, f = handles.pop(path)
        f.close()
    for job_id, path in current:
        if path not in handles and path.exists():
            handles[path] = (job_id, open(path))
    for job_id, f in handles.values():
        line = f.readline()
        while line:
            detail_buffer.append((job_id, line.rstrip()))
            line = f.readline()


# A stable color per job id, cycling through this fixed palette - used for both
# the summary table's rows and the detail panel's "[job N]" prefixes below, so
# the same job reads as the same color in both places.
_JOB_COLOR_PALETTE = ("cyan", "magenta", "yellow", "green", "bright_blue",
                     "bright_red", "bright_cyan", "bright_magenta")


def _job_style(job_id):
    return _JOB_COLOR_PALETTE[job_id % len(_JOB_COLOR_PALETTE)]


def _watch_layout(console, jobs, gen_cache, detail_buffer):
    if jobs:
        # Table's own title/box, not wrapped in a Panel - a table row wrapped in
        # its own Panel needs the height of BOTH sets of borders accounted for,
        # which is easy to under-count and clip the last data row (title line +
        # top border + header + separator + bottom border = 5 fixed lines).
        top = Table(title="Running jobs", expand=True)
        for column in ("id", "mode", "gpu", "gen", "best fitness", "best origin",
                       "Δ best", "spread", "elapsed", "ETA"):
            top.add_column(column)
        for job in jobs:
            top.add_row(*_job_summary_row(job, gen_cache), style=_job_style(job["id"]))
        top_height = len(jobs) + 5
    else:
        top = Panel("No jobs currently running - waiting...", title="Running jobs")
        top_height = 3

    available = max(console.size.height - top_height - 2, 3)
    recent = list(detail_buffer)[-available:]
    # Pad every "[job N]" prefix in this render to the widest job id currently
    # visible, so lines from different jobs (e.g. job 2 and job 10) line up
    # instead of the log text after the bracket starting at different columns
    # depending on how many digits that job's id happens to have.
    label_width = max((len(str(job_id)) for job_id, _ in recent), default=1)
    detail_text = Text()
    for job_id, line in recent:
        detail_text.append(f"[job {job_id:>{label_width}}] ", style=_job_style(job_id))
        detail_text.append(line + "\n")

    layout = Layout()
    layout.split_column(
        Layout(top, size=top_height),
        Layout(Panel(detail_text, title="Detail")),
    )
    return layout


def cmd_watch(args):
    """Live dashboard (like `docker stats`/`htop`): a summary table of every
    running job's current generation/best fitness, updating in place, with
    every job's detail log tailed underneath, tagged by job id. Needs a real
    terminal - `logs --all-jobs` is the streaming equivalent for redirected
    or non-interactive output."""
    if not sys.stdout.isatty():
        sys.exit("error: `watch` needs an interactive terminal - "
                  "use `qnas-queue logs --all-jobs` for redirected/non-tty output.")

    console = Console()
    gen_cache = {}
    handles = {}
    detail_buffer = deque(maxlen=500)

    def render():
        with db.connect() as conn:
            jobs = db.list_running_jobs(conn)
            _drain_new_detail_lines(handles, conn, detail_buffer)
        return _watch_layout(console, jobs, gen_cache, detail_buffer)

    try:
        with Live(render(), console=console, refresh_per_second=2) as live:
            while True:
                time.sleep(0.5)
                live.update(render())
    except KeyboardInterrupt:
        pass
    finally:
        for _, f in handles.values():
            f.close()


def cmd_logs(args):
    if args.all_jobs:
        if args.id:
            sys.exit("error: --all-jobs shows every running job and takes no job id")
        _tail_all_running(args)
        return

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
    p_add.add_argument("--gpu-ids", dest="gpu_ids", default=None,
                        help="Restrict this job to specific GPU(s) via CUDA_VISIBLE_DEVICES "
                             "(e.g. \"0\" or \"0,1\"). Omit to let it see every GPU (default).")
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
        "start", help="Start background worker(s) (also resumes any stopped job).")
    p_start.add_argument(
        "--workers", type=int, default=int(os.environ.get("QNAS_QUEUE_WORKERS", "1")),
        help="Target number of concurrent worker processes, each running at most one "
             "job at a time (default: 1, or $QNAS_QUEUE_WORKERS). Running this again "
             "with a higher number tops up to the new target instead of restarting "
             "existing workers.")
    p_start.add_argument(
        "--gpu-pool", default=os.environ.get("QNAS_QUEUE_GPU_POOL"),
        help="Shared pool of GPU indices (e.g. \"0,1,2,3\", or $QNAS_QUEUE_GPU_POOL) "
             "every started worker draws from: each worker picks a currently-free GPU "
             "from the pool for whatever job it claims next, instead of the job "
             "declaring one. Jobs queued with their own `add --gpu-ids` are left alone "
             "either way - the pool only fills in jobs that didn't pin a GPU.")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="Stop every worker and the jobs it is running.")
    p_stop.set_defaults(func=cmd_stop)

    p_status = sub.add_parser("status", help="Show worker and queue status.")
    p_status.set_defaults(func=cmd_status)

    p_logs = sub.add_parser("logs", help="Show logs for a job.")
    p_logs.add_argument("id", type=int, nargs="?", default=None,
                         help="Job id (defaults to the current/most recent job).")
    p_logs.add_argument("--follow", "-f", action="store_true")
    p_logs.add_argument("--lines", "-n", type=int, default=50)
    p_logs.add_argument("--all-jobs", "-a", action="store_true",
                         help="Show/follow every currently running job's detail log at "
                              "once, each line prefixed by its job id - useful with "
                              "`start --workers` > 1, where several jobs can run "
                              "concurrently and the default single-job view only shows "
                              "one of them. Takes no job id and ignores --detail/"
                              "--summary/--both (always the detail log).")
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

    p_watch = sub.add_parser(
        "watch", help="Live dashboard: every running job's generation/best "
                      "fitness updating in place, detail logs tailed below.")
    p_watch.set_defaults(func=cmd_watch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
