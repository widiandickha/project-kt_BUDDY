from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from src.config import ROOT_DIR

DEFAULT_DB_PATH = ROOT_DIR / "data" / "wellness_buddy.db"


class DatabaseManager:
    """Penyimpanan lokal SQLite untuk sesi, log skor, kejadian, dan riwayat Virtual Buddy.

    Tidak ada frame gambar atau video yang disimpan ke dalam database demi menjaga privasi.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # SQLite connections accessed from both UI and worker threads need a lock.
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    baseline_ear REAL,
                    baseline_brow_ratio REAL,
                    baseline_posture_ratio REAL,
                    avg_score REAL DEFAULT 0.0,
                    max_score REAL DEFAULT 0.0,
                    burnout_detected INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS score_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    ear_value REAL NOT NULL,
                    blink_rate_per_min REAL NOT NULL,
                    eye_closed_ratio REAL NOT NULL,
                    yawn_ratio REAL NOT NULL,
                    brow_furrow_ratio REAL NOT NULL,
                    slouch_ratio REAL NOT NULL,
                    hand_to_head_ratio REAL NOT NULL,
                    head_restlessness REAL NOT NULL,
                    stress_score REAL NOT NULL,
                    stress_level TEXT NOT NULL,
                    moving_avg_15m REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    details TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS buddy_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    expression TEXT NOT NULL,
                    message TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    user_action TEXT DEFAULT 'ditampilkan',
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_score_log_ts ON score_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_buddy_log_ts ON buddy_log(timestamp);
                """
            )

    def start_session(
        self,
        baseline_ear: float,
        baseline_brow_ratio: float,
        baseline_posture_ratio: float,
    ) -> int:
        now_iso = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO sessions (
                    start_time, baseline_ear, baseline_brow_ratio, baseline_posture_ratio
                ) VALUES (?, ?, ?, ?)
                """,
                (now_iso, baseline_ear, baseline_brow_ratio, baseline_posture_ratio),
            )
            return int(cur.lastrowid)

    def update_session_baseline(
        self,
        session_id: int,
        baseline_ear: float,
        baseline_brow_ratio: float,
        baseline_posture_ratio: float,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET baseline_ear = ?, baseline_brow_ratio = ?, baseline_posture_ratio = ?
                WHERE id = ?
                """,
                (baseline_ear, baseline_brow_ratio, baseline_posture_ratio, session_id),
            )

    def end_session(self, session_id: int) -> None:
        now_iso = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            stats = conn.execute(
                """
                SELECT COALESCE(AVG(stress_score), 0.0) AS avg_s,
                       COALESCE(MAX(stress_score), 0.0) AS max_s
                FROM score_log
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            avg_s = float(stats["avg_s"]) if stats else 0.0
            max_s = float(stats["max_s"]) if stats else 0.0
            conn.execute(
                """
                UPDATE sessions
                SET end_time = ?, avg_score = ?, max_score = ?
                WHERE id = ?
                """,
                (now_iso, avg_s, max_s, session_id),
            )

    def log_score(
        self,
        session_id: int,
        ear_value: float,
        blink_rate_per_min: float,
        eye_closed_ratio: float,
        yawn_ratio: float,
        brow_furrow_ratio: float,
        slouch_ratio: float,
        hand_to_head_ratio: float,
        head_restlessness: float,
        stress_score: float,
        stress_level: str,
        moving_avg_15m: float,
        burnout_detected: bool = False,
    ) -> None:
        now_iso = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO score_log (
                    session_id, timestamp, ear_value, blink_rate_per_min,
                    eye_closed_ratio, yawn_ratio, brow_furrow_ratio,
                    slouch_ratio, hand_to_head_ratio, head_restlessness,
                    stress_score, stress_level, moving_avg_15m
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    now_iso,
                    round(ear_value, 4),
                    round(blink_rate_per_min, 2),
                    round(eye_closed_ratio, 4),
                    round(yawn_ratio, 4),
                    round(brow_furrow_ratio, 4),
                    round(slouch_ratio, 4),
                    round(hand_to_head_ratio, 4),
                    round(head_restlessness, 4),
                    round(stress_score, 2),
                    stress_level,
                    round(moving_avg_15m, 2),
                ),
            )
            if burnout_detected:
                conn.execute(
                    "UPDATE sessions SET burnout_detected = 1 WHERE id = ?",
                    (session_id,),
                )

    def log_event(
        self,
        session_id: int,
        event_type: str,
        severity: str,
        details: str,
    ) -> None:
        now_iso = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (session_id, timestamp, event_type, severity, details)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, now_iso, event_type, severity, details),
            )

    def log_buddy(
        self,
        session_id: int,
        expression: str,
        message: str,
        trigger_reason: str,
        user_action: str = "ditampilkan",
    ) -> int:
        now_iso = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO buddy_log (
                    session_id, timestamp, expression, message, trigger_reason, user_action
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, now_iso, expression, message, trigger_reason, user_action),
            )
            return int(cur.lastrowid)

    def update_buddy_action(self, buddy_log_id: int, user_action: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE buddy_log SET user_action = ? WHERE id = ?",
                (user_action, buddy_log_id),
            )

    def get_daily_summary(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        date_prefix = target_date or datetime.now().strftime("%Y-%m-%d")
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS log_count,
                       COALESCE(AVG(stress_score), 0.0) AS avg_score,
                       COALESCE(MAX(stress_score), 0.0) AS max_score,
                       COALESCE(AVG(blink_rate_per_min), 0.0) AS avg_blink,
                       SUM(CASE WHEN stress_level = 'Tinggi' THEN 1 ELSE 0 END) AS high_count,
                       SUM(CASE WHEN stress_level = 'Waspada' THEN 1 ELSE 0 END) AS warn_count
                FROM score_log
                WHERE timestamp LIKE ?
                """,
                (f"{date_prefix}%",),
            ).fetchone()

            buddy_count_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM buddy_log WHERE timestamp LIKE ?",
                (f"{date_prefix}%",),
            ).fetchone()

            recent_series = conn.execute(
                """
                SELECT timestamp, stress_score, stress_level, moving_avg_15m
                FROM score_log
                WHERE timestamp LIKE ?
                ORDER BY timestamp ASC
                LIMIT 200
                """,
                (f"{date_prefix}%",),
            ).fetchall()

            recent_buddy = conn.execute(
                """
                SELECT timestamp, expression, trigger_reason, message, user_action
                FROM buddy_log
                WHERE timestamp LIKE ?
                ORDER BY timestamp DESC
                LIMIT 25
                """,
                (f"{date_prefix}%",),
            ).fetchall()

        return {
            "date": date_prefix,
            "log_count": int(row["log_count"] or 0) if row else 0,
            "avg_score": round(float(row["avg_score"] or 0.0), 1) if row else 0.0,
            "max_score": round(float(row["max_score"] or 0.0), 1) if row else 0.0,
            "avg_blink": round(float(row["avg_blink"] or 0.0), 1) if row else 0.0,
            "high_count": int(row["high_count"] or 0) if row else 0,
            "warn_count": int(row["warn_count"] or 0) if row else 0,
            "buddy_triggers": int(buddy_count_row["cnt"] or 0) if buddy_count_row else 0,
            "series": [dict(r) for r in recent_series],
            "buddy_logs": [dict(r) for r in recent_buddy],
        }

    def get_weekly_summary(self) -> List[Dict[str, Any]]:
        start_date = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT substr(timestamp, 1, 10) AS day,
                       COUNT(*) AS samples,
                       ROUND(AVG(stress_score), 1) AS avg_score,
                       ROUND(MAX(stress_score), 1) AS max_score,
                       SUM(CASE WHEN stress_level = 'Tinggi' THEN 1 ELSE 0 END) AS high_samples
                FROM score_log
                WHERE substr(timestamp, 1, 10) >= ?
                GROUP BY substr(timestamp, 1, 10)
                ORDER BY day DESC
                """,
                (start_date,),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_all_data(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                DELETE FROM buddy_log;
                DELETE FROM events;
                DELETE FROM score_log;
                DELETE FROM sessions;
                """
            )
