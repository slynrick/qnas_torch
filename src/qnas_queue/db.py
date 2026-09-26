import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUEUE_DIR = PROJECT_ROOT / ".qnas_queue"
DB_PATH = QUEUE_DIR / "queue.db"
LOG_DIR = QUEUE_DIR / "logs"
SNAPSHOT_DIR = QUEUE_DIR / "configs"
WORKER_LOG_PATH = QUEUE_DIR / "worker.log"


def resolve_path(path):
    """Resolves a stored job path (config_path/log_path) against the current
    PROJECT_ROOT if it's relative, so jobs stay runnable after the project
    directory is moved. Absolute paths (old rows, or configs outside the
    project tree) pass through unchanged.
    """
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mode            TEXT NOT NULL CHECK(mode IN ('evolve','retrain','pipeline')),
    config_path     TEXT NOT NULL,
    config_snapshot TEXT,
    experiment_path TEXT NOT NULL,
    extra_args      TEXT NOT NULL DEFAULT '',
    priority        INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'queued'
                    CHECK(status IN ('queued','running','done','failed','stopped','cancelled')),
    gpu_ids         TEXT,
    worker_pid      INTEGER,
    pgid            INTEGER,
    log_path        TEXT,
    exit_code       INTEGER,
    error_message   TEXT,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT
);

-- One row per live worker process, keyed by its own pid, so several workers
-- (configurable concurrency, see cli.py's `start --workers`) can each claim
-- and run a job at the same time without contending over a single "current
-- job" slot.
CREATE TABLE IF NOT EXISTS workers (
    pid             INTEGER PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'running' CHECK(status IN ('running','stopped')),
    current_job_id  INTEGER,
    started_at      TEXT,
    updated_at      TEXT
);
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dirs():
    QUEUE_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)


def _migrate_jobs_table(conn):
    """Older DBs have a jobs.status CHECK constraint without 'cancelled'; SQLite can't
    ALTER a CHECK constraint in place, so rebuild the table when that's detected. Copies
    whichever columns the old table happens to have that also exist in the current
    schema (e.g. an old DB might already have config_snapshot but not yet gpu_ids), so
    this rebuild never silently drops a column just because it was added later."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
    ).fetchone()
    if row is None or "cancelled" in row["sql"]:
        return
    old_columns = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
    conn.executescript("ALTER TABLE jobs RENAME TO jobs_old;")
    conn.executescript(SCHEMA)
    new_columns = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
    shared = ", ".join(sorted(old_columns & new_columns))
    conn.execute(f"INSERT INTO jobs ({shared}) SELECT {shared} FROM jobs_old")
    conn.executescript("DROP TABLE jobs_old;")


def _add_missing_columns(conn):
    """Older DBs predate some jobs columns; their rows keep the new ones NULL
    and behave as they always did (config_snapshot: run from config_path;
    gpu_ids: no CUDA_VISIBLE_DEVICES override, i.e. every visible GPU;
    worker_pid: unknown, treated as orphaned if still 'running')."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
    for column, decl in (
        ("config_snapshot", "TEXT"), ("gpu_ids", "TEXT"), ("worker_pid", "INTEGER"),
    ):
        if column not in columns:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {decl}")


def _migrate_worker_table(conn):
    """Pre-multi-worker DBs have a singleton `worker` table (a single id=1 row).
    Replaced by `workers`, one row per live worker process keyed by its pid -
    the singleton's data is a snapshot of whichever process last held it, not
    something worth carrying over, so it is just dropped."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'worker'"
    ).fetchone()
    if row is not None:
        conn.executescript("DROP TABLE worker;")


@contextmanager
def connect():
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        _migrate_jobs_table(conn)
        _add_missing_columns(conn)
        _migrate_worker_table(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def add_job(conn, mode, config_path, experiment_path, extra_args, priority, gpu_ids=None):
    cur = conn.execute(
        "INSERT INTO jobs (mode, config_path, experiment_path, extra_args, priority, "
        "gpu_ids, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (mode, config_path, experiment_path, extra_args, priority, gpu_ids, now_iso()),
    )
    return cur.lastrowid


def snapshot_config(job_id, config_path):
    """Copy *config_path* to SNAPSHOT_DIR/job_<id>_<name> and return the copy's path
    (project-relative when possible). The job runs from this frozen copy, so editing
    the original YAML after queueing - the queue runs jobs days later - cannot change
    what the job does."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    source = Path(config_path)
    target = SNAPSHOT_DIR / f"job_{job_id}_{source.name}"
    shutil.copy2(source, target)
    try:
        return str(target.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(target)


def config_for_run(job):
    """The config file a job actually runs with: its snapshot, or config_path for
    jobs queued before snapshots existed."""
    return job["config_snapshot"] or job["config_path"]


def remove_snapshot(job):
    if job["config_snapshot"]:
        resolve_path(job["config_snapshot"]).unlink(missing_ok=True)


def list_jobs(conn, status=None):
    if status:
        return conn.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY id", (status,)
        ).fetchall()
    return conn.execute("SELECT * FROM jobs ORDER BY id").fetchall()


def get_job(conn, job_id):
    if job_id is None:
        return None
    return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def delete_job(conn, job_id):
    conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def cancel_job(conn, job_id):
    update_job(conn, job_id, status="cancelled", finished_at=now_iso())


def update_job(conn, job_id, **fields):
    if not fields:
        return
    cols = ", ".join(f"{key} = ?" for key in fields)
    conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))


def resume_stopped_jobs(conn):
    rows = conn.execute("SELECT id FROM jobs WHERE status = 'stopped'").fetchall()
    for row in rows:
        update_job(
            conn, row["id"], status="queued", started_at=None, finished_at=None,
            exit_code=None, error_message=None, pgid=None,
        )
    return [row["id"] for row in rows]


def list_running_jobs(conn):
    return conn.execute("SELECT * FROM jobs WHERE status = 'running' ORDER BY id").fetchall()


def _gpu_ids_in_use(conn):
    """The set of individual GPU indices currently claimed by any 'running' job
    (pool-assigned or manually pinned via `add --gpu-ids` alike)."""
    used = set()
    for row in conn.execute(
            "SELECT gpu_ids FROM jobs WHERE status = 'running' AND gpu_ids IS NOT NULL"):
        used.update(int(x) for x in row["gpu_ids"].split(","))
    return used


def claim_next_job(conn, worker_pid, gpu_pool=None):
    """Claims the next queued job for *worker_pid*. With no *gpu_pool*, this is
    exactly the highest-priority queued job, as before. With a *gpu_pool* (a
    set of GPU indices), a job that didn't pin its own `gpu_ids` at `add` time
    only gets claimed once a GPU from the pool is free (not in use by another
    running job); if the front job needs the pool and it's fully occupied,
    later queued jobs are tried instead - so one resource-starved job never
    stalls jobs that don't need the pool, or that need a GPU still free in it.
    A job that already has `gpu_ids` (a manual pin) is claimed exactly as
    before, never blocked or reassigned by pool logic."""
    candidates = conn.execute(
        "SELECT * FROM jobs WHERE status = 'queued' ORDER BY priority DESC, id ASC"
    ).fetchall()
    used = None  # computed at most once per call, only if a pool job is seen
    for row in candidates:
        gpu_ids = row["gpu_ids"]
        if gpu_pool and not gpu_ids:
            if used is None:
                used = _gpu_ids_in_use(conn)
            free = sorted(gpu_pool - used)
            if not free:
                continue
            gpu_ids = str(free[0])
        conn.execute(
            "UPDATE jobs SET status = 'running', started_at = ?, worker_pid = ?, gpu_ids = ? "
            "WHERE id = ?",
            (now_iso(), worker_pid, gpu_ids, row["id"]),
        )
        return get_job(conn, row["id"])
    return None


def register_worker(conn, pid):
    conn.execute(
        "INSERT INTO workers (pid, status, current_job_id, started_at, updated_at) "
        "VALUES (?, 'running', NULL, ?, ?) "
        "ON CONFLICT(pid) DO UPDATE SET status = 'running', current_job_id = NULL, "
        "started_at = excluded.started_at, updated_at = excluded.updated_at",
        (pid, now_iso(), now_iso()),
    )


def remove_worker(conn, pid):
    conn.execute("DELETE FROM workers WHERE pid = ?", (pid,))


def get_worker(conn, pid):
    return conn.execute("SELECT * FROM workers WHERE pid = ?", (pid,)).fetchone()


def list_workers(conn):
    return conn.execute("SELECT * FROM workers ORDER BY pid").fetchall()


def set_worker(conn, pid, **fields):
    if not fields:
        return
    fields = {**fields, "updated_at": now_iso()}
    cols = ", ".join(f"{key} = ?" for key in fields)
    conn.execute(f"UPDATE workers SET {cols} WHERE pid = ?", (*fields.values(), pid))


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
