"""Tests for scripts/preannotate.py using synthetic landmark arrays (no MediaPipe needed)."""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.classes import KERUTAN_DAHI, MATA_TERBUKA, MATA_TERTUTUP, MENGUAP
from tests.conftest import load_script

W, H = 640, 480


@pytest.fixture(scope="module")
def pre():
    return load_script("preannotate")


@pytest.fixture()
def th(pre):
    return pre.Thresholds(ear_closed=0.18, ear_open=0.24, mar_yawn=0.45, pad_eyes=0.08, pad_mouth=0.06, pad_brow=0.05)


def face(pre, ear: float, mar: float) -> np.ndarray:
    """478 landmarks around a face at the image center with the given EAR and MAR."""
    pts = np.tile(np.array([[320.0, 240.0]]), (478, 1))
    pts[pre.FACE_SIDES[0]] = (220, 240)
    pts[pre.FACE_SIDES[1]] = (420, 240)
    for idx, x0 in ((pre.EAR_A, 260.0), (pre.EAR_B, 350.0)):
        half = ear * 30.0 / 2.0  # EAR = 2*(2*half) / (2*30)
        p1, p2, p3, p4, p5, p6 = idx
        pts[p1], pts[p4] = (x0, 200), (x0 + 30, 200)
        pts[p2], pts[p3] = (x0 + 10, 200 - half), (x0 + 20, 200 - half)
        pts[p6], pts[p5] = (x0 + 10, 200 + half), (x0 + 20, 200 + half)
    pts[pre.MOUTH_CORNERS[0]], pts[pre.MOUTH_CORNERS[1]] = (290, 300), (350, 300)
    pts[pre.MOUTH_INNER_V[0]], pts[pre.MOUTH_INNER_V[1]] = (320, 300 - mar * 30), (320, 300 + mar * 30)
    pts[152] = (320, 360)  # chin
    pts[10] = (320, 120)  # forehead top
    return pts


def test_ear_and_mar_helpers(pre) -> None:
    pts = face(pre, ear=0.30, mar=0.6)
    assert pre.ear_of(pts, pre.EAR_A) == pytest.approx(0.30)
    assert pre.mar_of(pts) == pytest.approx(0.6 * 60 / 60)


def test_open_eyes_no_yawn(pre, th) -> None:
    ann = pre.annotate_points(face(pre, 0.30, 0.1), W, H, "normal", th)
    assert [b.cls for b in ann.boxes] == [MATA_TERBUKA]
    assert ann.flags == []


def test_closed_eyes_and_yawn(pre, th) -> None:
    ann = pre.annotate_points(face(pre, 0.10, 0.6), W, H, "menguap", th)
    assert [b.cls for b in ann.boxes] == [MATA_TERTUTUP, MENGUAP]
    yawn = ann.boxes[1]
    assert yawn.cy > ann.boxes[0].cy  # mouth box is below the eye box


def test_ambiguous_ear_flagged(pre, th) -> None:
    ann = pre.annotate_points(face(pre, 0.20, 0.1), W, H, "normal", th)
    assert "ragu_mata" in ann.flags
    assert ann.boxes[0].cls == MATA_TERTUTUP  # 0.20 < midpoint 0.21


def test_hint_mismatch_flags(pre, th) -> None:
    assert "hint_menguap_mulut_tertutup" in pre.annotate_points(face(pre, 0.3, 0.1), W, H, "menguap", th).flags
    assert "menguap_tak_terduga" in pre.annotate_points(face(pre, 0.3, 0.6), W, H, "normal", th).flags


def test_brow_box_only_for_kerutan_hint(pre, th) -> None:
    ann = pre.annotate_points(face(pre, 0.3, 0.1), W, H, "kerutan_dahi", th)
    assert KERUTAN_DAHI in [b.cls for b in ann.boxes]
    assert KERUTAN_DAHI not in [b.cls for b in pre.annotate_points(face(pre, 0.3, 0.1), W, H, "normal", th).boxes]
