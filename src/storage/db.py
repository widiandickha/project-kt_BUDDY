"""SQLite connection, migrations (schema_version), retention cleanup, and a batching writer thread.

Only numbers and events are stored, never images. Tables follow spec FASE 6; `app_state`
(key/value text, e.g. first-run disclaimer) is an extension (D-023).
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SEC_PER_DAY = 86400.0

MIGRATIONS: tuple[str, ...] = (
    # 1: spec schema
    """
    CREATE TABLE sessions (id INTEGER PRIMARY KEY, start_ts REAL, end_ts REAL);
    CREATE TABLE stress_log (
        ts REAL, session_id INTEGER, score REAL, level TEXT,
        perclos REAL, blink_rate REAL, yawn_count INTEGER,
        furrow_ratio REAL, hand_count INTEGER, user_present INTEGER);
    CREATE TABLE events (ts REAL, session_id INTEGER, type TEXT, severity INTEGER, detail TEXT);
    CREATE TABLE advice_log (ts REAL, message_key TEXT, expression TEXT, response TEXT);
    CREATE TABLE daily_summary (
        date TEXT PRIMARY KEY, work_minutes REAL, avg_stress REAL, peak_stress REAL,
        peak_time TEXT, yawn_total INTEGER, break_count INTEGER, longest_stretch_min REAL,
        bad_posture_minutes REAL, advice_count INTEGER, advice_followed INTEGER,
        burnout_flag INTEGER, health_score REAL);
    CREATE TABLE user_calibration (key TEXT PRIMARY KEY, value REAL, updated_ts REAL);
    CREATE INDEX idx_stress_ts ON stress_log(ts);
    CREATE INDEX idx_events_ts ON events(ts);
    CREATE INDEX idx_advice_ts ON advice_log(ts);
    """,
    # 2: app_state (extension)
    """
    CREATE TABLE app_state (key TEXT PRIMARY KEY, value TEXT, updated_ts REAL);
    """,
)
TIMESTAMPED_TABLES: tuple[str, ...] = ("stress_log", "events", "advice_log")
ALL_TABLES: tuple[str, ...] = ("sessions", "stress_log", "events", "advice_log", "daily_summary", "user_calibration", "app_state")


class Database:
    """One shared connection guarded by a lock (SQLite in WAL mode)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    def schema_version(self) -> int:
        """Current schema version (0 = empty database)."""
        with self.lock:
            self.conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
            row = self.conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            return int(row["v"] or 0)

    def migrate(self) -> int:
        """Apply pending migrations; returns the new version."""
        version = self.schema_version()
        with self.lock:
            for i, script in enumerate(MIGRATIONS[version:], start=version + 1):
                self.conn.executescript(script)
                self.conn.execute("INSERT INTO schema_version (version) VALUES (?)", (i,))
                self.conn.commit()
                logger.info("Database dimigrasi ke versi %d", i)
        return self.schema_version()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Execute and commit one statement."""
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        """Execute a statement for many rows in one transaction."""
        with self.lock:
            self.conn.executemany(sql, rows)
            self.conn.commit()

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Fetch all rows."""
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def cleanup(self, retention_days: int, now: float | None = None) -> int:
        """Delete data older than retention_days (run at startup). Returns deleted rows."""
        cutoff = (now if now is not None else time.time()) - retention_days * SEC_PER_DAY
        deleted = 0
        with self.lock:
            for table in TIMESTAMPED_TABLES:
                deleted += self.conn.execute(f"DELETE FROM {table} WHERE ts < ?", (cutoff,)).rowcount  # noqa: S608
            deleted += self.conn.execute("DELETE FROM sessions WHERE start_ts < ?", (cutoff,)).rowcount
            cutoff_date = time.strftime("%Y-%m-%d", time.localtime(cutoff))
            deleted += self.conn.execute("DELETE FROM daily_summary WHERE date < ?", (cutoff_date,)).rowcount
            self.conn.commit()
        return deleted

    def delete_all(self) -> None:
        """'Hapus semua data': empty every data table (schema stays)."""
        with self.lock:
            for table in ALL_TABLES:
                self.conn.execute(f"DELETE FROM {table}")  # noqa: S608
            self.conn.commit()
            self.conn.execute("VACUUM")

    def close(self) -> None:
        """Close the connection."""
        with self.lock:
            self.conn.close()


class DbWriter(threading.Thread):
    """Background writer: callers enqueue (sql, params); rows are flushed every flush_sec
    (or on flush()/stop()) in one transaction, so the UI and inference never wait on disk."""

    def __init__(self, db: Database, flush_sec: float) -> None:
        super().__init__(name="DbWriter", daemon=True)
        self.db = db
        self.flush_sec = flush_sec
        self._q: queue.Queue[tuple[str, tuple[Any, ...]] | None] = queue.Queue()
        self._flush_now = threading.Event()
        self._stopped = threading.Event()
        self._idle = threading.Event()
        self._idle.set()

    def put(self, sql: str, params: Sequence[Any]) -> None:
        """Queue one write."""
        self._idle.clear()
        self._q.put((sql, tuple(params)))

    def flush(self, timeout: float = 5.0) -> None:
        """Write everything queued so far and wait for it."""
        self._flush_now.set()
        self._idle.wait(timeout)

    def _drain(self) -> None:
        batch: list[tuple[str, tuple[Any, ...]]] = []
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                batch.append(item)
        if batch:
            try:
                with self.db.lock:
                    for sql, params in batch:
                        self.db.conn.execute(sql, params)
                    self.db.conn.commit()
            except sqlite3.Error:
                logger.exception("Gagal menulis %d baris ke database", len(batch))
        if self._q.empty():
            self._idle.set()

    def run(self) -> None:
        """Flush loop."""
        while not self._stopped.is_set():
            self._flush_now.wait(self.flush_sec)
            self._flush_now.clear()
            self._drain()
        self._drain()

    def stop(self) -> None:
        """Flush and end the thread (clean exit)."""
        self._stopped.set()
        self._flush_now.set()
        if self.is_alive():
            self.join(timeout=5.0)
        else:
            self._drain()
