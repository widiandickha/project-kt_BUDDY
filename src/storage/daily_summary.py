"""Daily aggregation (spec FASE 6): run when the app closes and at midnight.

health_score (0..100) = 100 - avg_stress * health.stress_weight
                        - no_break_penalty_per_hour * hours worked beyond habit.work_break_min in the longest stretch
                        - posture_penalty_per_10min * bad_posture_minutes / 10
                        + good_break_bonus * good_breaks (max health.max_bonus), clamped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

from src.storage.repository import Repository

SEC_PER_MIN = 60.0
MIN_PER_HOUR = 60.0
DAYS_COMPARE = 7
GAP_FACTOR = 3  # a gap > 3 log intervals breaks a work stretch


@dataclass
class DailySummary:
    """One daily_summary row."""

    date: str
    work_minutes: float
    avg_stress: float | None
    peak_stress: float | None
    peak_time: str | None
    yawn_total: int
    break_count: int
    longest_stretch_min: float
    bad_posture_minutes: float
    advice_count: int
    advice_followed: int
    burnout_flag: int
    health_score: float | None


def day_bounds(day: date) -> tuple[float, float]:
    """Local-time [start, end) epoch seconds of a calendar day."""
    start = datetime.combine(day, datetime.min.time())
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def longest_stretch_minutes(present_ts: list[float], log_every_sec: float) -> float:
    """Longest run of consecutive present rows (gaps <= GAP_FACTOR * interval)."""
    if not present_ts:
        return 0.0
    best, start, prev = 0.0, present_ts[0], present_ts[0]
    for ts in present_ts[1:]:
        if ts - prev > GAP_FACTOR * log_every_sec:
            best = max(best, prev - start + log_every_sec)
            start = ts
        prev = ts
    best = max(best, prev - start + log_every_sec)
    return best / SEC_PER_MIN


def health_score(avg_stress: float | None, longest_min: float, bad_posture_min: float, good_breaks: int,
                 cfg: dict[str, Any]) -> float | None:
    """Formula from the module docstring (None when there was no work that day)."""
    if avg_stress is None:
        return None
    h = cfg["health"]
    overwork_h = max(0.0, longest_min - float(cfg["habit"]["work_break_min"])) / MIN_PER_HOUR
    score = (100.0 - avg_stress * float(h["stress_weight"])
             - float(h["no_break_penalty_per_hour"]) * overwork_h
             - float(h["posture_penalty_per_10min"]) * bad_posture_min / 10.0
             + min(float(h["max_bonus"]), float(h["good_break_bonus"]) * good_breaks))
    return round(float(np.clip(score, 0.0, 100.0)), 1)


def aggregate(repo: Repository, day: date, cfg: dict[str, Any]) -> DailySummary:
    """Compute (and store) the summary of one day."""
    start, end = day_bounds(day)
    log_every = float(cfg["storage"]["log_every_sec"])
    rows = repo.stress_between(start, end)
    present = [r for r in rows if r["user_present"]]
    scored = [r for r in present if r["score"] is not None]
    work_minutes = len(present) * log_every / SEC_PER_MIN
    avg = float(np.mean([r["score"] for r in scored])) if scored else None
    peak_row = max(scored, key=lambda r: r["score"]) if scored else None

    events = repo.events_between(start, end)
    kinds = [e["type"] for e in events]
    away_min = float(cfg["habit"]["away_min_sec"])
    breaks = sum(1 for e in events if e["type"] == "user_returned" and float(e["detail"].get("away_sec", 0)) >= away_min)
    posture_min = kinds.count("bad_posture") * float(cfg["habit"]["posture_hold_min"])
    longest = longest_stretch_minutes([r["ts"] for r in present], log_every)
    advice = repo.advice_between(start, end)

    summary = DailySummary(
        date=day.isoformat(),
        work_minutes=round(work_minutes, 1),
        avg_stress=round(avg, 1) if avg is not None else None,
        peak_stress=round(peak_row["score"], 1) if peak_row else None,
        peak_time=datetime.fromtimestamp(peak_row["ts"]).strftime("%H:%M") if peak_row else None,
        yawn_total=kinds.count("yawn"),
        break_count=breaks,
        longest_stretch_min=round(longest, 1),
        bad_posture_minutes=posture_min,
        advice_count=len(advice),
        advice_followed=sum(1 for a in advice if a["response"] == "oke"),
        burnout_flag=int("burnout_risk" in kinds),
        health_score=health_score(avg, longest, posture_min, kinds.count("good_break"), cfg),
    )
    repo.upsert_summary(summary)
    return summary


def hourly_stress(repo: Repository, day: date) -> list[tuple[int, float]]:
    """(hour, mean score) for the hours with data."""
    start, end = day_bounds(day)
    buckets: dict[int, list[float]] = {}
    for r in repo.stress_between(start, end):
        if r["score"] is not None:
            buckets.setdefault(datetime.fromtimestamp(r["ts"]).hour, []).append(r["score"])
    return [(h, float(np.mean(v))) for h, v in sorted(buckets.items())]


def last_days(repo: Repository, day: date, cfg: dict[str, Any], days: int = DAYS_COMPARE) -> list[DailySummary]:
    """Summaries of `day` and the days before it (recomputed so they are up to date)."""
    return [aggregate(repo, day - timedelta(days=i), cfg) for i in range(days - 1, -1, -1)]


def insights(today: DailySummary, history: list[DailySummary]) -> list[str]:
    """Short automatic observations for the dashboard."""
    out: list[str] = []
    if today.work_minutes <= 0:
        return ["Belum ada data hari ini."]
    if today.peak_time and today.peak_stress is not None:
        out.append(f"Stres puncak pukul {today.peak_time.replace(':', '.')} (skor {today.peak_stress:.0f}).")
    if today.longest_stretch_min >= 50:
        out.append(f"Kerja terlama tanpa jeda: {today.longest_stretch_min:.0f} menit. Coba istirahat tiap 50 menit.")
    prev = [d for d in history if d.date != today.date and d.avg_stress is not None]
    if prev and today.avg_stress is not None:
        mean_prev = float(np.mean([d.avg_stress for d in prev]))
        diff = today.avg_stress - mean_prev
        if abs(diff) >= 5:
            arah = "lebih tinggi" if diff > 0 else "lebih rendah"
            out.append(f"Rata-rata stres hari ini {abs(diff):.0f} poin {arah} dari {len(prev)} hari sebelumnya.")
    if today.break_count == 0 and today.work_minutes >= 120:
        out.append("Belum ada istirahat 5 menit atau lebih hari ini.")
    if today.advice_count:
        out.append(f"{today.advice_followed} dari {today.advice_count} saran Buddy diikuti.")
    if today.burnout_flag:
        out.append("Tren stres tinggi berjam-jam terdeteksi (risiko burnout).")
    return out or ["Hari yang tenang. Pertahankan!"]
