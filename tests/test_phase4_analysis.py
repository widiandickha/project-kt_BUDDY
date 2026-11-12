"""FASE 4: fusion, sliding window, scoring + hysteresis, burnout, habits, calibration, replay."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import pytest

from src.analysis.burnout_trend import BurnoutTrend
from src.analysis.calibration import Calibration, CalibrationSession
from src.analysis.engine import AnalysisEngine
from src.analysis.feature_extractor import FeatureExtractor
from src.analysis.habit_evaluator import HabitEvaluator, Sustained
from src.analysis.replay import make_sample, read_features, replay
from src.analysis.sliding_window import SlidingWindow, WindowStats, eye_closed
from src.analysis.stress_scorer import StressScorer, blink_abnormality, normalized_indicators
from src.core.config import load_config
from src.core.events import FrameFeatures
from src.detection.ear_calculator import LandmarkResult

CFG = load_config()
FPS = 10
DT = 1.0 / FPS
CAL = Calibration.from_values(0.31, 0.09, 0.62, 0.30, 0.30)


def frame(ts: float, ear: float = 0.31, present: bool = True, **kw) -> FrameFeatures:
    return FrameFeatures(ts=ts, user_present=present, ear_mean=ear if present else None, nose_xy=(0.5, 0.5), **kw)


def feed(win: SlidingWindow, start: float, seconds: float, ear_at=lambda t: 0.31, **kw) -> float:
    t = start
    while t < start + seconds:
        win.add(frame(t, ear_at(t - start), **{k: (v(t - start) if callable(v) else v) for k, v in kw.items()}))
        t += DT
    return t


# --- fusion -----------------------------------------------------------------

def test_fusion_rules() -> None:
    thr = 0.21
    assert eye_closed(frame(0, 0.1), thr, CFG) is True
    assert eye_closed(frame(0, 0.3), thr, CFG) is False
    both = FrameFeatures(ts=0, user_present=True, ear_mean=0.3, det_available=True, eye_closed_yolo=0.99)
    assert eye_closed(both, thr, CFG) is False  # 0.6*0 + 0.4*0.99 < 0.5: EAR dominates
    yolo_only = FrameFeatures(ts=0, user_present=True, det_available=True, eye_closed_yolo=0.8)
    assert eye_closed(yolo_only, thr, CFG) is True
    assert eye_closed(FrameFeatures(ts=0, user_present=False), thr, CFG) is None


# --- sliding window -----------------------------------------------------------

def test_perclos_ignores_normal_blinks_and_counts_blink_rate() -> None:
    win = SlidingWindow(CFG, 0.21)
    # 60 s with a 200 ms blink every 4 s -> 15 blinks/min, PERCLOS 0
    end = feed(win, 0.0, 60.0, ear_at=lambda t: 0.08 if (t % 4.0) < 0.2 else 0.31)
    st = win.stats(end)
    assert st.perclos == pytest.approx(0.0)
    assert st.blink_rate == pytest.approx(15.0, abs=1.5)


def test_perclos_counts_microsleeps() -> None:
    win = SlidingWindow(CFG, 0.21)
    end = feed(win, 0.0, 60.0, ear_at=lambda t: 0.08 if (t % 10.0) < 2.0 else 0.31)  # 2 s of every 10 s
    assert win.stats(end).perclos == pytest.approx(0.2, abs=0.02)


def test_yawn_needs_duration_and_debounce() -> None:
    win = SlidingWindow(CFG, 0.21)
    # short mouth openings (0.5 s, talking) every 5 s -> never a yawn
    end = feed(win, 0.0, 30.0, yawn=lambda t: 1.0 if t % 5 < 0.5 else 0.0)
    assert win.stats(end).yawn_count == 0
    win2 = SlidingWindow(CFG, 0.21)
    # 1.5 s yawns at t=0, 4, 20 -> the one at 4 s is inside the 10 s debounce
    end = feed(win2, 0.0, 30.0, yawn=lambda t: 1.0 if any(s <= t < s + 1.5 for s in (0, 4, 20)) else 0.0)
    assert win2.stats(end).yawn_count == 2


def test_yawn_not_double_counted_across_ticks() -> None:
    win = SlidingWindow(CFG, 0.21)
    t = feed(win, 0.0, 5.0, yawn=lambda t: 1.0 if t < 2 else 0.0)
    assert win.stats(t).yawn_count == 1
    t = feed(win, t, 5.0)
    assert win.stats(t).yawn_count == 1


def test_hand_count_furrow_ratio_and_motion() -> None:
    win = SlidingWindow(CFG, 0.21)
    end = feed(win, 0.0, 40.0, hand_to_forehead=lambda t: t % 10 < 2, brow_furrow=lambda t: 1.0 if t < 20 else 0.0)
    st = win.stats(end)
    assert st.hand_forehead_count == 4
    assert st.furrow_ratio == pytest.approx(0.5, abs=0.03)
    assert st.trajectory_motion == pytest.approx(0.0)


def test_window_drops_old_samples() -> None:
    win = SlidingWindow(CFG, 0.21)
    end = feed(win, 0.0, 30.0, ear_at=lambda t: 0.08)  # eyes closed 30 s
    end = feed(win, end, 70.0)  # then open long enough to push it out
    assert win.stats(end).perclos == 0.0


# --- scoring -------------------------------------------------------------------

def stats(**kw) -> WindowStats:
    base = dict(span_sec=60, present_ratio=1.0, perclos=0.0, blink_rate=17.0, yawn_count=0, furrow_ratio=0.0,
                hand_forehead_ratio=0.0, hand_forehead_count=0, trajectory_motion=0.0, eyes_closed_now=False)
    base.update(kw)
    return WindowStats(**base)


def test_normalization_matches_spec_table() -> None:
    n = normalized_indicators(stats(perclos=0.15, yawn_count=6, furrow_ratio=0.2, hand_forehead_count=2,
                                    blink_rate=35, trajectory_motion=0.025), CFG)
    assert n == pytest.approx({"perclos": 0.5, "yawn": 1.0, "furrow": 0.5, "hand": 0.5, "blink": 1.0, "motion": 0.5})
    assert blink_abnormality(17, (15, 20)) == 0 and blink_abnormality(5, (15, 20)) == 10 and blink_abnormality(None, (15, 20)) == 0


def test_max_score_is_100_and_weights_sum() -> None:
    sc = StressScorer(CFG)
    raw, comp = sc.raw_score(stats(perclos=1, yawn_count=9, furrow_ratio=1, hand_forehead_count=9, blink_rate=60, trajectory_motion=1))
    assert raw == pytest.approx(100.0) and sum(comp.values()) == pytest.approx(100.0)


def test_ema_smoothing() -> None:
    sc = StressScorer(CFG)
    high = stats(perclos=0.3, yawn_count=3, furrow_ratio=0.4, hand_forehead_count=4)
    s1 = sc.update(stats(), 0.0)
    s2 = sc.update(high, 1.0)
    assert s1.score == 0.0
    assert s2.score == pytest.approx(0.3 * s2.raw_score)


def test_hysteresis_up_and_down() -> None:
    sc = StressScorer(CFG)
    sc.alpha = 1.0  # no smoothing: isolate the hysteresis
    high = stats(perclos=0.3, yawn_count=3, furrow_ratio=0.4, hand_forehead_count=4)  # 85
    levels = [sc.update(high, float(t)).level for t in range(0, 25)]
    assert levels[19] == "rendah" and levels[20] == "tinggi"  # needs 20 s above the threshold
    low_ok = stats(perclos=0.21)  # 21 points: below 35 but not below 35 - 5? -> below 30 yes
    down = [sc.update(low_ok, 25.0 + t).level for t in range(0, 35)]
    assert down[29] == "tinggi" and down[30] == "rendah"  # needs 30 s below threshold - margin


def test_no_snapshot_without_data_or_presence() -> None:
    sc = StressScorer(CFG)
    assert sc.update(stats(span_sec=5), 0.0) is None
    assert sc.update(stats(present_ratio=0.0), 0.0) is None


# --- burnout -------------------------------------------------------------------

def test_burnout_slope_and_risk() -> None:
    cfg = copy.deepcopy(CFG)
    cfg["burnout"]["min_duration_min"] = 30
    bt = BurnoutTrend(cfg)
    minutes = [(i * 60.0, 50.0 + i * 0.1) for i in range(240)]  # +6 points/hour, ends ~74
    bt.seed(minutes)
    st = bt.status(240 * 60.0)
    assert st.slope_per_hour == pytest.approx(6.0, abs=0.3)
    assert st.ma_long > 55 and not st.risk  # condition just started
    assert bt.status(240 * 60.0 + 30 * 60).risk


def test_burnout_minute_aggregation() -> None:
    bt = BurnoutTrend(CFG)
    for s in range(0, 180):
        bt.add(float(s), 10.0 if s < 60 else 30.0)
    assert [m for _, m in bt.minute_means()] == [10.0, 30.0]


# --- habits --------------------------------------------------------------------

def make_habits(hour: int = 14) -> HabitEvaluator:
    return HabitEvaluator(CFG, CAL, clock=lambda ts: datetime(2026, 11, 12, hour, 0))


def run_habit(h: HabitEvaluator, seconds: int, start: float = 0.0, st: WindowStats | None = None, **kw) -> list[str]:
    kinds: list[str] = []
    for s in range(seconds):
        t = start + s
        h.observe(frame(t, **kw) if kw.get("present", True) else frame(t, present=False))
        presence, alerts = h.tick(t, st or stats(), None)
        kinds += [p.kind for p in presence] + [a.kind for a in alerts]
    return kinds


def test_need_break_after_50_minutes() -> None:
    kinds = run_habit(make_habits(), 50 * 60 + 5)
    assert kinds.count("need_break") == 1


def test_bad_posture_after_3_minutes() -> None:
    h = make_habits()
    assert "bad_posture" not in run_habit(h, 170, shoulder_slouch=0.45)
    assert "bad_posture" in run_habit(h, 20, start=170, shoulder_slouch=0.45)


def test_too_close_and_dry_eyes() -> None:
    assert "too_close" in run_habit(make_habits(), 125, face_size_ratio=0.45)
    assert "dry_eyes" in run_habit(make_habits(), 185, st=stats(blink_rate=5.0))


def test_late_night_once_per_night() -> None:
    h = HabitEvaluator(CFG, CAL, clock=lambda ts: datetime(2026, 11, 12, 23, 0) if ts < 100 else datetime(2026, 11, 13, 1, 0))
    assert run_habit(h, 200).count("late_night") == 1


def test_away_return_and_good_break() -> None:
    h = make_habits()
    kinds = run_habit(h, 31 * 60)  # long work
    kinds += run_habit(h, 6 * 60, start=31 * 60, present=False)  # 6 min away
    kinds += run_habit(h, 5, start=37 * 60)
    assert kinds.count("user_away") == 1 and kinds.count("user_returned") == 1 and "good_break" in kinds


def test_short_absence_is_not_a_break() -> None:
    h = make_habits()
    kinds = run_habit(h, 31 * 60) + run_habit(h, 60, start=31 * 60, present=False) + run_habit(h, 5, start=32 * 60)
    assert "good_break" not in kinds and "user_away" in kinds


def test_sustained_fires_once_per_episode() -> None:
    s = Sustained(2)
    assert [s.update(t, c) for t, c in enumerate([True, True, True, True, False, True, True, True])] == \
        [False, False, True, False, False, False, False, True]


# --- calibration & features ------------------------------------------------------

def test_calibration_threshold_formula_and_rows() -> None:
    cal = Calibration.from_values(0.30, 0.10, 0.6, 0.3, 0.3)
    assert cal.ear_threshold == pytest.approx(0.30 - 0.4 * 0.20)
    assert Calibration.from_rows(cal.to_rows(), CFG) == cal
    assert Calibration.from_rows({}, CFG) is None
    assert Calibration.default(CFG).ear_threshold == 0.21


def test_calibration_session_phases() -> None:
    sess = CalibrationSession(CFG, start_ts=0.0)
    t = 0.0
    while t < 20.0:
        ear = 0.30 if sess.phase(t) == "open" else 0.08
        sess.add(FrameFeatures(ts=t, user_present=True, ear_mean=ear, shoulder_slouch=0.6, face_size_ratio=0.3, brow_ratio=0.3))
        t += DT
    assert sess.phase(20.0) == "done"
    cal = sess.result()
    assert cal is not None and cal.open_ear == pytest.approx(0.30) and cal.closed_ear == pytest.approx(0.08)


def test_calibration_fails_without_closed_phase() -> None:
    sess = CalibrationSession(CFG, start_ts=0.0)
    for i in range(100):
        sess.add(FrameFeatures(ts=i * DT, user_present=True, ear_mean=0.3))
    assert sess.result() is None


def test_feature_extractor_landmark_fallbacks() -> None:
    fx = FeatureExtractor(CFG, CAL)
    lm = LandmarkResult(face_found=True, ear_left=0.3, ear_right=0.3, ear_mean=0.3, mar=0.7, brow_ratio=0.25,
                        face_size_ratio=0.3, nose_xy=(0.5, 0.5))
    f = fx.extract(1.0, lm, faces=None, pose=None)
    assert f.user_present and f.yawn == 1.0 and f.brow_furrow == 1.0 and not f.det_available
    nobody = fx.extract(2.0, LandmarkResult(face_found=False), None, None)
    assert not nobody.user_present and nobody.ear_mean is None


# --- engine & replay ----------------------------------------------------------------

def test_replay_sample_reaches_high_level_and_presence(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    n = make_sample(path)
    feats = list(read_features(path))
    assert len(feats) == n
    results = replay(feats, AnalysisEngine(CFG, CAL))
    levels = {r.snapshot.level for r in results if r.snapshot}
    events = [p.kind for r in results for p in r.presence]
    assert {"rendah", "sedang", "tinggi"} <= levels
    assert events == ["user_away", "user_returned"]
    assert max(r.stats.yawn_count for r in results) >= 3


def test_monotonic_counters_for_event_logging() -> None:
    win = SlidingWindow(CFG, 0.21)
    t = feed(win, 0.0, 30.0, yawn=lambda t: 1.0 if t < 1.5 else 0.0, hand_to_forehead=lambda t: t % 10 < 2)
    win.stats(t)
    t = feed(win, t, 70.0)  # old samples leave the window, counters stay
    win.stats(t)
    assert win.total_yawns == 1 and win.total_hand_episodes == 3
