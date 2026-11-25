"""D-032: live camera preview (memory only)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtGui import QImage

from src.analysis.calibration import Calibration
from src.core.config import load_config
from src.core.events import FrameFeatures, StressSnapshot
from src.core.pipeline import PreviewFrame
from src.detection.ear_calculator import LandmarkResult, measure
from src.detection.face_detector import Detection, FaceDetections
from src.detection.pose_estimator import PoseResult
from src.ui.preview_window import PreviewWindow, indicators, to_qimage
from tests.test_phase7_integration import FAST, factory, frames_dir, make_pipeline, wait_for  # noqa: F401

CFG = load_config()
CAL = Calibration.from_values(0.31, 0.09, 0.62, 0.30, 0.30)


def _frame(ear: float = 0.31, yawn: float = 0.0, furrow: float = 0.0, det: bool = True) -> PreviewFrame:
    img = np.full((480, 640, 3), 90, np.uint8)
    f = FrameFeatures(ts=1.0, user_present=True, ear_mean=ear, ear_left=ear, ear_right=ear, mar=0.2,
                      yawn=yawn, brow_furrow=furrow, det_available=det, eye_closed_yolo=0.0,
                      shoulder_slouch=0.40, face_size_ratio=0.45)
    lm = LandmarkResult(True, ear, ear, ear, 0.2, 0.3, 0.3, (200, 100, 440, 400), (0.5, 0.5),
                        eye_box=(250, 200, 390, 240), mouth_box=(280, 320, 360, 360))
    faces = FaceDetections()
    faces.boxes.append(Detection("mata_terbuka", 0.91, (250, 200, 390, 240)))
    kp = np.zeros((17, 3))
    kp[[0, 1, 2, 5, 6], :] = [[320, 260, 0.9], [300, 240, 0.9], [340, 240, 0.9], [220, 430, 0.9], [420, 430, 0.9]]
    return PreviewFrame(img, f, lm, faces if det else None, PoseResult(True, keypoints=kp), fps=11.8)


def test_indicators_text_and_colours() -> None:
    snap = StressSnapshot(ts=1.0, score=72.0, level="tinggi", components={}, burnout_trend=0.0)
    ok = indicators(_frame(), CAL, CFG, snap)
    assert ok["mata"][0].startswith("terbuka") and ok["menguap"][0].startswith("tidak")
    assert ok["skor"][0] == "72 · Tinggi" and ok["fps"][0] == "11.8"
    assert ok["postur"][0] == "membungkuk"  # 0.40 < 0.62 * (1 - slouch_drop)
    assert ok["jarak"][0] == "terlalu dekat"
    assert ok["kerutan"][0] == "tidak (landmark)"  # D-031: landmark even with YOLO
    bad = indicators(_frame(ear=0.05, yawn=1.0, furrow=1.0), CAL, CFG, None)
    assert bad["mata"][0].startswith("TERTUTUP") and bad["menguap"][0].startswith("YA")
    assert bad["kerutan"][0] == "YA (landmark)" and bad["skor"][0] == "menghitung…"
    uncal = indicators(_frame(), Calibration.default(CFG), CFG, None)
    assert uncal["kerutan"][0] == "perlu kalibrasi" and "perlu kalibrasi" in uncal["postur"][0]


def test_measure_gives_eye_and_mouth_boxes() -> None:
    pts = np.random.default_rng(0).uniform(100, 400, size=(478, 2))
    lm = measure(pts, 640, 480)
    for b in (lm.eye_box, lm.mouth_box):
        assert b is not None and 0 <= b[0] < b[2] < 640 and 0 <= b[1] < b[3] < 480


def test_window_renders_and_never_writes_frames(qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    written: list[str] = []
    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: written.append("cv2") or True)
    monkeypatch.setattr(QImage, "save", lambda *a, **k: written.append("qimage") or True)
    monkeypatch.chdir(tmp_path)
    w = PreviewWindow(CFG, lambda: CAL, lambda: None, lambda: True)
    closed = []
    w.closed.connect(lambda: closed.append(1))
    w.show()
    w.push(_frame())
    w.refresh()
    assert w.view.frame is not None and w.values["mata"].text().startswith("terbuka")
    assert not w.view.grab().isNull()  # paints boxes, pose and the LIVE badge
    w.cb_mirror.setChecked(False)
    w.cb_show_yolo.setChecked(False)
    assert not w.view.mirror and not w.view.show_yolo
    w.close()
    assert closed == [1] and w.view.frame is None and w.view.image is None and w._pending is None
    assert written == [] and list(tmp_path.iterdir()) == []


def test_window_shows_paused_camera(qapp) -> None:
    w = PreviewWindow(CFG, lambda: CAL, lambda: None, lambda: False)
    w.refresh()
    assert "dijeda" in w.view.message
    w.close()


def test_to_qimage_converts_bgr() -> None:
    img = np.zeros((4, 6, 3), np.uint8)
    img[..., 2] = 255  # red in BGR
    q = to_qimage(img)
    assert (q.width(), q.height()) == (6, 4) and q.pixelColor(0, 0).red() == 255


def test_pipeline_hook_only_while_listening(tmp_path: Path, frames_dir: Path) -> None:  # noqa: F811
    p = make_pipeline(tmp_path, frames_dir)
    got: list[PreviewFrame] = []
    p.start()
    assert wait_for(lambda: p.inference.frames > 10)
    assert got == []  # nothing handed over while the preview is closed
    p.set_preview_listener(got.append)
    assert wait_for(lambda: len(got) > 5)
    first = got[0]
    assert first.image.ndim == 3 and first.features.user_present and first.landmark is not None
    p.set_preview_listener(None)
    n = len(got)
    time.sleep(0.2)
    assert len(got) <= n + 1  # at most one frame in flight
    # a watchdog restart keeps the current listener
    p.set_preview_listener(got.append)
    assert p._new_inference().frame_listener is not None
    p.stop()


def test_app_opens_and_closes_preview(qapp, tmp_path: Path, frames_dir: Path) -> None:  # noqa: F811
    from src.main import App, parse_args

    args = parse_args(["--source", str(frames_dir), "--skip-intro", "--db", str(tmp_path / "a.db"),
                       "--source-fps", "60", "--preview"])
    app = App(qapp, FAST, args, model_factory=factory())
    app.start()
    assert wait_for(lambda: app.pipeline.inference.frames > 10)
    app.show_preview()
    assert app.pipeline.preview_listener is not None
    assert wait_for(lambda: app.preview._pending is not None or app.preview.view.frame is not None)
    app.preview.close()
    assert app.pipeline.preview_listener is None and app.pipeline.inference.frame_listener is None
    app.shutdown()
