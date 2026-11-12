"""FASE 6: database, repository, daily aggregation, dashboard (offscreen)."""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from src.core.config import load_config
from src.storage import daily_summary as ds
from src.storage.db import MIGRATIONS, Database, DbWriter
from src.storage.demo_data import fill_demo_data
from src.storage.repository import Repository, StressRow

CFG = load_config()
DAY = date(2026, 11, 10)


def ts(hour: int, minute: int = 0, second: int = 0, day: date = DAY) -> float:
    return datetime.combine(day, datetime.min.time()).replace(hour=hour, minute=minute, second=second).timestamp()


@pytest.fixture()
def repo(tmp_path: Path) -> Repository:
    return Repository(Database(tmp_path / "t.db"))


def test_migrations_are_versioned_and_idempotent(tmp_path: Path) -> None:
    db = Database(tmp_path / "m.db")
    assert db.schema_version() == len(MIGRATIONS)
    db.close()
    db2 = Database(tmp_path / "m.db")  # reopening does not re-run migrations
    assert db2.schema_version() == len(MIGRATIONS)
    tables = {r["name"] for r in db2.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sessions", "stress_log", "events", "advice_log", "daily_summary", "user_calibration", "app_state"} <= tables


def test_writer_batches_and_flushes(tmp_path: Path) -> None:
    db = Database(tmp_path / "w.db")
    writer = DbWriter(db, flush_sec=60)  # long interval: only explicit flushes write
    writer.start()
    repo = Repository(db, writer)
    for i in range(50):
        repo.add_event(float(i), 1, "yawn")
    assert db.query("SELECT COUNT(*) AS n FROM events")[0]["n"] == 0
    repo.flush()
    assert db.query("SELECT COUNT(*) AS n FROM events")[0]["n"] == 50
    repo.add_event(99.0, 1, "yawn")
    writer.stop()  # clean exit flushes the rest
    assert db.query("SELECT COUNT(*) AS n FROM events")[0]["n"] == 51


def test_repository_round_trips(repo: Repository) -> None:
    sid = repo.start_session(ts(9))
    repo.add_stress(StressRow(ts(9, 0, 10), sid, 20.0, "rendah", 0.05, 15.0, 0, 0.0, 0, True))
    repo.add_event(ts(9, 1), sid, "user_returned", 1, {"away_sec": 400})
    repo.add_advice(ts(9, 2), "need_break", "alert")
    repo.set_advice_response(ts(9, 2), "need_break", "oke")
    repo.end_session(sid, ts(10))
    s, e = ds.day_bounds(DAY)
    assert repo.stress_between(s, e)[0]["score"] == 20.0
    assert repo.events_between(s, e, ["user_returned"])[0]["detail"] == {"away_sec": 400}
    assert repo.advice_between(s, e)[0]["response"] == "oke"
    repo.set_calibration({"open_ear": 0.3, "closed_ear": 0.1})
    assert repo.get_calibration() == {"open_ear": 0.3, "closed_ear": 0.1}
    repo.set_state("disclaimer_ack", "1")
    assert repo.get_state("disclaimer_ack") == "1" and repo.get_state("nope") is None


def test_minute_means(repo: Repository) -> None:
    for i in range(12):
        repo.add_stress(StressRow(ts(9, 0, 0) + i * 10, 1, 10.0 if i < 6 else 30.0, "rendah", 0, None, 0, 0, 0, True))
    means = repo.minute_means(ts(9), ts(10))
    assert [m for _, m in means] == [10.0, 30.0]


def test_retention_and_delete_all(repo: Repository) -> None:
    now = ts(12)
    repo.add_event(now - 100 * 86400, 1, "old")
    repo.add_event(now - 10 * 86400, 1, "recent")
    assert repo.db.cleanup(90, now=now) == 1
    assert [e["type"] for e in repo.events_between(0, now)] == ["recent"]
    repo.db.delete_all()
    assert repo.events_between(0, now) == []


def test_export_csv(repo: Repository, tmp_path: Path) -> None:
    repo.add_event(ts(9), 1, "yawn")
    out = tmp_path / "e.csv"
    assert repo.export_csv("events", out, ts(0), ts(23)) == 1
    rows = list(csv.reader(out.open()))
    assert rows[0][:3] == ["ts", "session_id", "type"] and rows[1][2] == "yawn"
    with pytest.raises(ValueError):
        repo.export_csv("user_calibration", out)


# --- aggregation ------------------------------------------------------------------

def test_longest_stretch() -> None:
    assert ds.longest_stretch_minutes([], 10) == 0
    run1 = [i * 10.0 for i in range(60)]           # 10 min
    run2 = [2000 + i * 10.0 for i in range(180)]   # 30 min
    assert ds.longest_stretch_minutes(run1 + run2, 10) == pytest.approx(30.0)


def test_health_score_formula() -> None:
    assert ds.health_score(None, 0, 0, 0, CFG) is None
    # 100 - 40*0.5 - 5*(110-50)/60 - 2*6/10 + 3*2
    assert ds.health_score(40, 110, 6, 2, CFG) == pytest.approx(100 - 20 - 5 - 1.2 + 6)
    assert ds.health_score(0, 0, 0, 50, CFG) == 100.0  # clamped, bonus capped
    assert ds.health_score(100, 600, 300, 0, CFG) == 0.0


def test_aggregate_known_day(repo: Repository) -> None:
    sid = repo.start_session(ts(9))
    for i in range(360):  # 1 hour present, score 10..; peak 80 at 09:30
        score = 80.0 if i == 180 else 20.0
        repo.add_stress(StressRow(ts(9) + i * 10, sid, score, "rendah", 0, 15, 0, 0, 0, True))
    for i in range(36):  # 6 min away
        repo.add_stress(StressRow(ts(10) + i * 10, sid, None, None, 0, None, 0, 0, 0, False))
    repo.add_event(ts(10, 6), sid, "user_returned", 1, {"away_sec": 360})
    repo.add_event(ts(10, 6), sid, "good_break", 1, {})
    repo.add_event(ts(9, 10), sid, "yawn")
    repo.add_event(ts(9, 20), sid, "yawn")
    repo.add_event(ts(9, 40), sid, "bad_posture", 2)
    repo.add_advice(ts(9, 41), "bad_posture", "worried")
    repo.set_advice_response(ts(9, 41), "bad_posture", "oke")
    repo.add_advice(ts(9, 50), "dry_eyes", "eye_care")

    s = ds.aggregate(repo, DAY, CFG)
    assert s.work_minutes == 60.0 and s.peak_stress == 80.0 and s.peak_time == "09:30"
    assert s.avg_stress == pytest.approx((359 * 20 + 80) / 360, abs=0.1)
    assert (s.yawn_total, s.break_count, s.advice_count, s.advice_followed, s.burnout_flag) == (2, 1, 2, 1, 0)
    assert s.longest_stretch_min == 60.0 and s.bad_posture_minutes == 3.0
    assert repo.summaries(DAY.isoformat(), DAY.isoformat())[0]["health_score"] == s.health_score
    assert ds.hourly_stress(repo, DAY)[0][0] == 9


def test_insights() -> None:
    empty = ds.DailySummary(DAY.isoformat(), 0, None, None, None, 0, 0, 0, 0, 0, 0, 0, None)
    assert ds.insights(empty, []) == ["Belum ada data hari ini."]
    today = ds.DailySummary(DAY.isoformat(), 300, 50, 80, "15:00", 3, 0, 120, 0, 4, 1, 1, 40)
    prev = ds.DailySummary((DAY - timedelta(days=1)).isoformat(), 300, 30, 50, "11:00", 0, 3, 40, 0, 0, 0, 0, 80)
    text = " ".join(ds.insights(today, [prev, today]))
    assert "15.00" in text and "120 menit" in text and "20 poin lebih tinggi" in text
    assert "1 dari 4 saran" in text and "burnout" in text and "Belum ada istirahat" in text


def test_demo_data_and_dashboard(qapp, tmp_path: Path) -> None:
    from src.ui.dashboard import Dashboard

    repo = Repository(Database(tmp_path / "demo.db"))
    fill_demo_data(repo, DAY, days=3)
    w = Dashboard(repo, CFG)
    from PySide6.QtCore import QDate

    w.date_edit.setDate(QDate(DAY.year, DAY.month, DAY.day))
    assert w.today is not None and w.today.work_minutes > 300 and w.today.break_count > 0
    assert w.values["breaks"].text() == str(w.today.break_count)
    assert w.insights.count() >= 1
    files = w.export_csv(tmp_path / "export")
    assert len(files) == 4 and all(f.exists() for f in files)
