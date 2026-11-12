"""FASE 4: EAR / landmark measurements, pose geometry, event contracts."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.events import EVENT_STRESS_SNAPSHOT, EventBus, FrameFeatures
from src.detection import ear_calculator as ec
from src.detection import pose_estimator as pe


def eye_points(ear: float) -> np.ndarray:
    pts = np.zeros((478, 2), dtype=np.float32)
    for idx, x0 in ((ec.RIGHT_EYE, 100.0), (ec.LEFT_EYE, 200.0)):
        half = ear * 30.0 / 2.0
        p1, p2, p3, p4, p5, p6 = idx
        pts[p1], pts[p4] = (x0, 100), (x0 + 30, 100)
        pts[p2], pts[p3] = (x0 + 10, 100 - half), (x0 + 20, 100 - half)
        pts[p6], pts[p5] = (x0 + 10, 100 + half), (x0 + 20, 100 + half)
    return pts


@pytest.mark.parametrize("ear", [0.05, 0.21, 0.32])
def test_ear_formula(ear: float) -> None:
    pts = eye_points(ear)
    assert ec.compute_ear(pts, ec.LEFT_EYE) == pytest.approx(ear, abs=1e-5)
    assert ec.compute_ear(pts, ec.RIGHT_EYE) == pytest.approx(ear, abs=1e-5)


def test_spec_eye_indices() -> None:
    assert ec.LEFT_EYE == (362, 385, 387, 263, 373, 380)
    assert ec.RIGHT_EYE == (33, 160, 158, 133, 153, 144)


def test_degenerate_eye_is_zero() -> None:
    assert ec.compute_ear(np.zeros((478, 2)), ec.LEFT_EYE) == 0.0


def test_measure_returns_face_size_and_nose() -> None:
    pts = eye_points(0.3)
    pts[:, 0] = np.clip(pts[:, 0], 100, 260)
    pts[ec.NOSE_TIP] = (320, 240)
    res = ec.measure(pts, 640, 480)
    assert res.face_found and res.ear_mean == pytest.approx(0.3, abs=1e-4)
    assert res.nose_xy == pytest.approx((0.5, 0.5))
    assert 0 < res.face_size_ratio <= 1


def kpts(wrist_xy: tuple[float, float], conf: float = 0.9) -> np.ndarray:
    k = np.zeros((17, 3), dtype=np.float32)
    k[pe.NOSE] = (320, 200, conf)
    k[pe.L_EYE], k[pe.R_EYE] = (340, 180, conf), (300, 180, conf)
    k[pe.L_SH], k[pe.R_SH] = (420, 320, conf), (220, 320, conf)
    k[pe.L_WR] = (*wrist_xy, conf)
    k[pe.R_WR] = (200, 460, conf)
    return k


def test_hand_to_forehead_rule() -> None:
    near = pe.analyze_keypoints(kpts((330, 170)), 640, 480, 0.5, 0.45)
    assert near.hand_near_forehead
    below_nose = pe.analyze_keypoints(kpts((330, 230)), 640, 480, 0.5, 0.45)
    assert not below_nose.hand_near_forehead  # wrist under the nose
    far = pe.analyze_keypoints(kpts((560, 100)), 640, 480, 0.5, 0.45)
    assert not far.hand_near_forehead


def test_low_confidence_keypoints_ignored() -> None:
    res = pe.analyze_keypoints(kpts((330, 170), conf=0.3), 640, 480, 0.5, 0.45)
    assert not res.hand_near_forehead and res.shoulder_slouch is None


def test_slouch_and_pitch() -> None:
    upright = pe.analyze_keypoints(kpts((0, 0), 0.9), 640, 480, 0.5, 0.45)
    k = kpts((0, 0))
    k[pe.NOSE, 1] = 280  # head dropped toward the shoulders
    low = pe.analyze_keypoints(k, 640, 480, 0.5, 0.45)
    assert upright.shoulder_slouch == pytest.approx(0.6)
    assert low.shoulder_slouch < upright.shoulder_slouch
    assert low.head_pitch_proxy > upright.head_pitch_proxy


def test_consecutive_frames() -> None:
    c = pe.ConsecutiveFrames(3)
    assert [c.update(v) for v in (True, True, True, True, False, True)] == [False, False, True, True, False, False]


def test_event_bus_dispatch_and_isolation() -> None:
    bus = EventBus()
    got: list[float] = []

    def bad(_: object) -> None:
        raise RuntimeError("boom")

    bus.subscribe(EVENT_STRESS_SNAPSHOT, bad)
    unsub = bus.subscribe(EVENT_STRESS_SNAPSHOT, lambda s: got.append(s))
    bus.publish(EVENT_STRESS_SNAPSHOT, 1.0)
    unsub()
    bus.publish(EVENT_STRESS_SNAPSHOT, 2.0)
    assert got == [1.0]
    with pytest.raises(ValueError):
        bus.subscribe("nope", print)


def test_frame_features_round_trip() -> None:
    f = FrameFeatures(ts=1.5, user_present=True, ear_mean=0.3, pose_keypoints=np.ones((17, 3)), nose_xy=(0.1, 0.2))
    back = FrameFeatures.from_dict(f.to_dict())
    assert back.ear_mean == 0.3 and back.nose_xy == (0.1, 0.2) and back.pose_keypoints.shape == (17, 3)
