"""FASE 7: capture thread, pipeline (fake models), settings overlay, full App (offscreen)."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.analysis.calibration import Calibration
from src.capture.camera import STATUS_ACTIVE, STATUS_PAUSED, STATUS_UNAVAILABLE, CaptureThread, Frame
from src.core.config import load_app_config, load_config, save_user_settings
from src.core.pipeline import InferenceThread, Models, Pipeline
from src.detection.ear_calculator import LandmarkResult
from src.detection.pose_estimator import PoseResult
from src.storage.db import Database
from src.storage.repository import Repository

CFG = load_config()
FAST = copy.deepcopy(CFG)  # short windows so real-time tests stay quick
FAST["score"]["min_window_sec"] = 1
FAST["habit"]["away_after_sec"] = 0.5
CAL = Calibration.from_values(0.31, 0.09, 0.62, 0.30, 0.30)


def wait_for(cond, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture()
def frames_dir(tmp_path: Path) -> Path:
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(5):
        cv2.imwrite(str(d / f"{i:03d}.jpg"), np.full((48, 64, 3), i * 40, np.uint8))
    return d


# --- capture ----------------------------------------------------------------------

def test_queue_keeps_only_newest() -> None:
    cap = CaptureThread("0", 64, 48, 30, 1, queue_size=2)
    for i in range(5):
        cap._push(Frame(float(i), np.zeros((2, 2, 3), np.uint8)))
    assert cap.dropped == 3
    assert [cap.get_latest(0.1).ts, cap.get_latest(0.1).ts] == [3.0, 4.0]


def test_folder_source_pause_releases_and_resume(frames_dir: Path) -> None:
    statuses: list[str] = []
    cap = CaptureThread(str(frames_dir), 64, 48, 50, 0.2, on_status=lambda s, m: statuses.append(s))
    cap.start()
    assert cap.get_latest(2.0) is not None
    cap.pause()
    assert wait_for(lambda: cap.status == STATUS_PAUSED) and cap._cap is None  # device released
    cap.resume()
    assert wait_for(lambda: cap.status == STATUS_ACTIVE)
    cap.stop()
    assert statuses[:3] == [STATUS_ACTIVE, STATUS_PAUSED, STATUS_ACTIVE] and not cap.is_alive()


def test_missing_source_retries(tmp_path: Path) -> None:
    cap = CaptureThread(str(tmp_path / "nope.mp4"), 64, 48, 30, 0.1)
    cap.start()
    assert wait_for(lambda: cap.status == STATUS_UNAVAILABLE)
    cap.stop()


# --- fake models ---------------------------------------------------------------------

class FakeEar:
    def __init__(self, closed_every: int = 0, fail_times: int = 0) -> None:
        self.n, self.closed_every, self.fail_times = 0, closed_every, fail_times

    def process(self, img, ts=None) -> LandmarkResult:
        self.n += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("simulated model crash")
        ear = 0.08 if self.closed_every and self.n % self.closed_every < self.closed_every // 2 else 0.31
        return LandmarkResult(True, ear, ear, ear, 0.1, 0.3, 0.3, (0, 0, 10, 10), (0.5, 0.5))


class FakePose:
    def process(self, img) -> PoseResult:
        return PoseResult(person_found=True, shoulder_slouch=0.62, nose_xy=(0.5, 0.45))


def factory(**kw):
    return lambda cfg: Models(FakeEar(**kw), FakePose(), None)


def make_pipeline(tmp_path: Path, frames_dir: Path, **kw) -> Pipeline:
    repo = Repository(Database(tmp_path / "p.db"))
    return Pipeline(FAST, repo, CAL, source=str(frames_dir), model_factory=factory(**kw), source_fps=60,
                    record_path=tmp_path / "rec.jsonl")


def test_pipeline_end_to_end_with_fake_models(tmp_path: Path, frames_dir: Path) -> None:
    p = make_pipeline(tmp_path, frames_dir, closed_every=60)  # eyes closed half of every second
    p.start()
    assert wait_for(lambda: p.inference.frames > 90)  # > 1.5 s of frames at 60 fps
    outs = []
    for _ in range(4):
        outs.append(p.tick())
        time.sleep(0.25)
    p.stop()
    snaps = [o.result.snapshot for o in outs if o.result and o.result.snapshot]
    assert snaps and snaps[-1].components["perclos"] > 0
    rows = p.repo.stress_between(0, time.time() + 1)
    assert len(rows) == 1 and rows[0]["session_id"] == p.session_id and rows[0]["user_present"] == 1
    assert (tmp_path / "rec.jsonl").read_text().count("\n") >= 60
    session = p.repo.db.query("SELECT end_ts FROM sessions")[0]
    assert session["end_ts"] is not None


def test_pause_stops_analysis_without_user_away(tmp_path: Path, frames_dir: Path) -> None:
    p = make_pipeline(tmp_path, frames_dir)
    p.start()
    assert wait_for(lambda: p.inference.frames > 10)
    p.tick()
    p.pause()
    assert p.tick().result is None and not p.camera_active
    assert wait_for(lambda: p.capture._cap is None)  # device released
    time.sleep(1.0)  # longer than away_after_sec (0.5 s in FAST)
    p.resume()
    out = p.tick()  # right after resuming: pausing is not leaving the desk
    p.stop()
    assert not any(e.kind == "user_away" for e in (out.result.presence if out.result else []))


def test_inference_restarts_models_after_errors(tmp_path: Path, frames_dir: Path) -> None:
    calls = {"n": 0}

    def flaky(cfg):
        calls["n"] += 1
        return Models(FakeEar(fail_times=6 if calls["n"] == 1 else 0), FakePose(), None)

    repo = Repository(Database(tmp_path / "f.db"))
    p = Pipeline(CFG, repo, CAL, source=str(frames_dir), model_factory=flaky, source_fps=60)
    p.start()
    assert wait_for(lambda: p.inference.restarts >= 1 and p.inference.frames > 20)
    p.stop()


def test_watchdog_restarts_dead_inference(tmp_path: Path, frames_dir: Path) -> None:
    p = make_pipeline(tmp_path, frames_dir)
    p.start()
    old = p.inference
    old._stop_event.set()
    old.join(timeout=3)
    old._stop_event.clear()  # pretend it died on its own
    p.tick()
    assert p.watchdog_restarts == 1 and p.inference is not old and p.inference.is_alive()
    p.stop()


def test_adaptive_frame_skip() -> None:
    t = InferenceThread(CFG, CaptureThread("0", 1, 1, 1, 1), None, lambda f: None)
    t.fps = 5.0
    t._adapt()
    t._adapt()
    t._adapt()
    assert (t.pose_every, t.det_every) == (3, 2)
    t.fps = 30.0
    t._adapt()
    assert (t.pose_every, t.det_every) == (2, 1)


# --- settings overlay ------------------------------------------------------------------

def test_user_settings_overlay(tmp_path: Path) -> None:
    sp = tmp_path / "settings.yaml"
    save_user_settings({"buddy": {"scale": 1.2}}, sp)
    save_user_settings({"habit": {"work_break_min": 40}}, sp)
    cfg = load_app_config(settings_path=sp)
    assert cfg["buddy"]["scale"] == 1.2 and cfg["habit"]["work_break_min"] == 40
    assert cfg["buddy"]["base_px"] == CFG["buddy"]["base_px"]  # untouched keys kept


# --- full app ---------------------------------------------------------------------------

def test_app_starts_ticks_and_shuts_down_cleanly(qapp, tmp_path: Path, frames_dir: Path) -> None:
    from src.main import DISCLAIMER_KEY, App, parse_args

    args = parse_args(["--source", str(frames_dir), "--skip-intro", "--db", str(tmp_path / "app.db"), "--source-fps", "60"])
    app = App(qapp, CFG, args, model_factory=factory())
    app.start()
    assert wait_for(lambda: app.pipeline.inference.frames > 30)
    for _ in range(3):
        app.tick()
    app.pause_for(0)
    assert app.pipeline.paused
    app.resume()
    app.shutdown()
    db = Database(tmp_path / "app.db")
    repo = Repository(db)
    assert repo.get_state(DISCLAIMER_KEY) == "1"
    assert db.query("SELECT COUNT(*) AS n FROM daily_summary")[0]["n"] == 1
    assert not app.pipeline.capture.is_alive() and not app.pipeline.inference.is_alive()
