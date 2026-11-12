"""Deterministic dummy history (tests, dashboard demo). Never used on real user data."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np

from src.storage.repository import Repository, StressRow

LOG_EVERY_SEC = 10.0
WORK_START_HOUR = 9
WORK_HOURS = 7


def fill_demo_data(repo: Repository, end_day: date, days: int = 7, seed: int = 3) -> None:
    """Write `days` days of fake work sessions ending at end_day (inclusive)."""
    rng = np.random.default_rng(seed)
    for back in range(days - 1, -1, -1):
        day = end_day - timedelta(days=back)
        start = datetime.combine(day, datetime.min.time()).replace(hour=WORK_START_HOUR).timestamp()
        sid = repo.start_session(start)
        score = rng.uniform(10, 30)
        t = start
        end = start + WORK_HOURS * 3600
        next_break = start + rng.uniform(45, 80) * 60
        while t < end:
            if t >= next_break:  # 6-10 min away
                away = rng.uniform(6, 10) * 60
                repo.add_event(t + 15, sid, "user_away", 1, {})
                repo.add_event(t + away, sid, "user_returned", 1, {"away_sec": away})
                repo.add_event(t + away, sid, "good_break", 1, {})
                for k in range(int(away // LOG_EVERY_SEC)):
                    repo.add_stress(StressRow(t + k * LOG_EVERY_SEC, sid, None, None, 0, None, 0, 0, 0, False))
                t += away
                next_break = t + rng.uniform(45, 110) * 60
                score = max(5.0, score - 15)
                continue
            hour = datetime.fromtimestamp(t).hour
            drift = 0.08 if 14 <= hour <= 16 else -0.02
            score = float(np.clip(score + drift + rng.normal(0, 0.8), 0, 100))
            level = "tinggi" if score > 65 else "sedang" if score >= 35 else "rendah"
            repo.add_stress(StressRow(t, sid, round(score, 1), level, round(score / 300, 3), float(rng.uniform(9, 20)),
                                      int(score > 40), round(score / 250, 3), int(score > 50), True))
            if rng.random() < 0.004:
                repo.add_event(t, sid, "yawn", 1, {})
            if rng.random() < 0.0015:
                repo.add_event(t, sid, "bad_posture", 2, {})
            if rng.random() < 0.002:
                key = rng.choice(["need_break", "bad_posture", "dry_eyes", "stress_sedang"])
                repo.add_advice(t, str(key), "worried")
                repo.set_advice_response(t, str(key), str(rng.choice(["oke", "nanti", "diabaikan"])))
            t += LOG_EVERY_SEC
        repo.end_session(sid, end)
    repo.flush()
