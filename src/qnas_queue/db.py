import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUEUE_DIR = PROJECT_ROOT / ".qnas_queue"
DB_PATH = QUEUE_DIR / "queue.db"
LOG_DIR = QUEUE_DIR / "logs"
WORKER_PID_FILE = QUEUE_DIR / "worker.pid"
WORKER_LOG_PATH = QUEUE_DIR / "worker.log"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mode            TEXT NOT NULL CHECK(mode IN ('evolve','retrain','pipeline')),
    config_path     TEXT NOT NULL,
    experiment_path TEXT NOT NULL,
    extra_args      TEXT NOT NULL DEFAULT '',
    priority        INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'queued'
                    CHECK(status IN ('queued','running','done','failed','stopped')),
    pgid            INTEGER,
    log_path        TEXT,
    exit_code       INTEGER,
    error_message   TEXT,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT
);

CREATE TABLE IF NOT EXISTS worker (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    pid             INTEGER,
    status          TEXT NOT NULL DEFAULT 'stopped' CHECK(status IN ('running','stopped')),
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


@contextmanager
def connect():
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        conn.execute("INSERT OR IGNORE INTO worker (id, status) VALUES (1, 'stopped')")
        yield conn
        conn.commit()
    finally:
        conn.close()


def add_job(conn, mode, config_path, experiment_path, extra_args, priority):
    cur = conn.execute(
        "INSERT INTO jobs (mode, config_path, experiment_path, extra_args, priority, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (mode, config_path, experiment_path, extra_args, priority, now_iso()),
    )
    return cur.lastrowid


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


def update_job(conn, job_id, **fields):
    if not fields:
        return
    cols = ", ".join(f"{key} = ?" for key in fields)
    conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))


def claim_next_job(conn):
    row = conn.execute(
        "SELECT * FROM jobs WHERE status = 'queued' ORDER BY priority DESC, id ASC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    conn.execute(
        "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
        (now_iso(), row["id"]),
    )
    return get_job(conn, row["id"])


def get_worker(conn):
    return conn.execute("SELECT * FROM worker WHERE id = 1").fetchone()


def set_worker(conn, **fields):
    if not fields:
        return
    fields = {**fields, "updated_at": now_iso()}
    cols = ", ".join(f"{key} = ?" for key in fields)
    conn.execute(f"UPDATE worker SET {cols} WHERE id = 1", tuple(fields.values()))


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
