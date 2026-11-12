"""CRUD on top of Database/DbWriter. Writes from the pipeline go through the batching writer;
reads (dashboard, summaries) query directly."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.storage.db import Database, DbWriter


@dataclass
class StressRow:
    """One stress_log row (written every storage.log_every_sec)."""

    ts: float
    session_id: int
    score: float | None
    level: str | None
    perclos: float
    blink_rate: float | None
    yawn_count: int
    furrow_ratio: float
    hand_count: int
    user_present: bool


class Repository:
    """Data access used by the app, summaries, dashboard and tests."""

    def __init__(self, db: Database, writer: DbWriter | None = None) -> None:
        self.db = db
        self.writer = writer

    def _write(self, sql: str, params: Sequence[Any]) -> None:
        if self.writer is not None:
            self.writer.put(sql, params)
        else:
            self.db.execute(sql, params)

    # -- sessions -------------------------------------------------------------------
    def start_session(self, ts: float | None = None) -> int:
        """New session id (written immediately)."""
        cur = self.db.execute("INSERT INTO sessions (start_ts) VALUES (?)", (ts if ts is not None else time.time(),))
        return int(cur.lastrowid)

    def end_session(self, session_id: int, ts: float | None = None) -> None:
        """Close a session."""
        self.db.execute("UPDATE sessions SET end_ts = ? WHERE id = ?", (ts if ts is not None else time.time(), session_id))

    # -- logs -----------------------------------------------------------------------
    def add_stress(self, r: StressRow) -> None:
        """Queue one stress_log row."""
        self._write(
            "INSERT INTO stress_log (ts, session_id, score, level, perclos, blink_rate, yawn_count, furrow_ratio,"
            " hand_count, user_present) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (r.ts, r.session_id, r.score, r.level, r.perclos, r.blink_rate, r.yawn_count, r.furrow_ratio,
             r.hand_count, int(r.user_present)),
        )

    def add_event(self, ts: float, session_id: int | None, type_: str, severity: int = 1, detail: dict[str, Any] | None = None) -> None:
        """Queue one event (yawn, hand_to_forehead, user_away, habit alerts, ...)."""
        self._write("INSERT INTO events (ts, session_id, type, severity, detail) VALUES (?, ?, ?, ?, ?)",
                    (ts, session_id, type_, severity, json.dumps(detail or {}, ensure_ascii=False)))

    def add_advice(self, ts: float, message_key: str, expression: str) -> None:
        """Queue an advice bubble (response filled in later)."""
        self._write("INSERT INTO advice_log (ts, message_key, expression, response) VALUES (?, ?, ?, NULL)",
                    (ts, message_key, expression))

    def set_advice_response(self, ts: float, message_key: str, response: str) -> None:
        """Record the user's response to the bubble shown at ts."""
        self._write("UPDATE advice_log SET response = ? WHERE ts = ? AND message_key = ?", (response, ts, message_key))

    def flush(self) -> None:
        """Make queued writes visible to readers."""
        if self.writer is not None:
            self.writer.flush()

    # -- calibration & state ------------------------------------------------------------
    def get_calibration(self) -> dict[str, float]:
        """All user_calibration rows."""
        return {r["key"]: float(r["value"]) for r in self.db.query("SELECT key, value FROM user_calibration")}

    def set_calibration(self, values: dict[str, float]) -> None:
        """Replace calibration values."""
        now = time.time()
        self.db.executemany(
            "INSERT INTO user_calibration (key, value, updated_ts) VALUES (?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_ts = excluded.updated_ts",
            [(k, float(v), now) for k, v in values.items()],
        )

    def get_state(self, key: str) -> str | None:
        """app_state value."""
        rows = self.db.query("SELECT value FROM app_state WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    def set_state(self, key: str, value: str) -> None:
        """Upsert an app_state value."""
        self.db.execute("INSERT INTO app_state (key, value, updated_ts) VALUES (?, ?, ?)"
                        " ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_ts = excluded.updated_ts",
                        (key, value, time.time()))

    # -- reads ----------------------------------------------------------------------------
    def stress_between(self, start: float, end: float) -> list[dict[str, Any]]:
        """stress_log rows in [start, end)."""
        return [dict(r) for r in self.db.query("SELECT * FROM stress_log WHERE ts >= ? AND ts < ? ORDER BY ts", (start, end))]

    def events_between(self, start: float, end: float, types: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """events in [start, end), optionally filtered by type; detail decoded from JSON."""
        sql, params = "SELECT * FROM events WHERE ts >= ? AND ts < ?", [start, end]
        if types:
            sql += f" AND type IN ({','.join('?' * len(types))})"
            params += list(types)
        rows = []
        for r in self.db.query(sql + " ORDER BY ts", params):
            d = dict(r)
            d["detail"] = json.loads(d["detail"] or "{}")
            rows.append(d)
        return rows

    def advice_between(self, start: float, end: float) -> list[dict[str, Any]]:
        """advice_log rows in [start, end)."""
        return [dict(r) for r in self.db.query("SELECT * FROM advice_log WHERE ts >= ? AND ts < ? ORDER BY ts", (start, end))]

    def minute_means(self, start: float, end: float) -> list[tuple[float, float]]:
        """Per-minute average stress score (for seeding the burnout trend after a restart)."""
        rows = self.db.query(
            "SELECT CAST(ts / 60 AS INTEGER) * 60 AS m, AVG(score) AS s FROM stress_log"
            " WHERE ts >= ? AND ts < ? AND score IS NOT NULL GROUP BY m ORDER BY m", (start, end))
        return [(float(r["m"]), float(r["s"])) for r in rows]

    def upsert_summary(self, summary: Any) -> None:
        """Insert or replace a DailySummary dataclass."""
        d = asdict(summary)
        cols = list(d)
        self.db.execute(
            f"INSERT OR REPLACE INTO daily_summary ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",  # noqa: S608
            [d[c] for c in cols],
        )

    def summaries(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        """daily_summary rows with start_date <= date <= end_date."""
        return [dict(r) for r in self.db.query(
            "SELECT * FROM daily_summary WHERE date >= ? AND date <= ? ORDER BY date", (start_date, end_date))]

    # -- export ----------------------------------------------------------------------------
    def export_csv(self, table: str, path: Path, start: float | None = None, end: float | None = None) -> int:
        """Write a table (optionally a time range) to CSV; returns the row count."""
        allowed = {"stress_log", "events", "advice_log", "daily_summary"}
        if table not in allowed:
            raise ValueError(f"Tabel tidak bisa diekspor: {table}")
        sql, params = f"SELECT * FROM {table}", []  # noqa: S608
        if start is not None and end is not None and table != "daily_summary":
            sql += " WHERE ts >= ? AND ts < ?"
            params = [start, end]
        rows = self.db.query(sql, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if rows:
                w.writerow(rows[0].keys())
                w.writerows([tuple(r) for r in rows])
        return len(rows)
