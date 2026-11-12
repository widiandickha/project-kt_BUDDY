"""Tests for src/detection/face_detector.py (no trained weights needed)."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.config import load_config
from src.detection import face_detector as fd
from src.detection.classes import CLASS_NAMES

NAMES = dict(enumerate(CLASS_NAMES))


def test_parse_keeps_max_conf_per_class_and_sorts() -> None:
    res = fd.parse_detections(
        NAMES,
        cls_ids=np.array([1, 0, 1, 2]),
        confs=np.array([0.55, 0.9, 0.7, 0.4]),
        xyxy=np.array([[0, 0, 10, 10], [1, 1, 5, 5], [2, 2, 8, 8], [0, 5, 9, 9]], dtype=float),
    )
    assert res.conf("mata_terbuka") == pytest.approx(0.7)
    assert res.conf("mata_tertutup") == pytest.approx(0.9)
    assert res.conf("menguap") == pytest.approx(0.4)
    assert res.conf("kerutan_dahi") == 0.0
    assert [d.conf for d in res.boxes] == sorted((d.conf for d in res.boxes), reverse=True)
    assert res.boxes[0].xyxy == (1, 1, 5, 5)


def test_parse_ignores_unknown_classes() -> None:
    res = fd.parse_detections({0: "person"}, np.array([0]), np.array([0.9]), np.array([[0, 0, 1, 1]]))
    assert res.boxes == [] and all(v == 0.0 for v in res.confidences.values())


def test_empty_result_has_all_classes_zero() -> None:
    assert fd.FaceDetections().confidences == {name: 0.0 for name in CLASS_NAMES}


def test_to_gray3() -> None:
    img = np.zeros((4, 4, 3), np.uint8)
    img[..., 2] = 200
    gray = fd.to_gray3(img)
    assert gray.shape == (4, 4, 3)
    assert (gray[..., 0] == gray[..., 1]).all() and (gray[..., 1] == gray[..., 2]).all()


def test_draw_detections_marks_frame() -> None:
    frame = np.zeros((160, 360, 3), np.uint8)
    res = fd.parse_detections(NAMES, np.array([2]), np.array([0.8]), np.array([[260, 60, 320, 140]], dtype=float))
    fd.draw_detections(frame, res, fps=10.0)
    assert tuple(frame[100, 260]) == fd.CLASS_COLORS["menguap"]  # left edge, clear of the HUD panel
    assert tuple(frame[20, 20]) != (0, 0, 0)  # HUD drawn


def test_missing_weights_raise_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="training/train.py"):
        fd.FaceDetector(load_config(), weights=tmp_path / "none.pt")


def test_cli_requires_demo_flag() -> None:
    assert fd.main([]) == 2
    args = fd.parse_args(["--demo", "--source", "x.mp4", "--no-show", "--duration", "3"])
    assert args.demo and args.no_show and args.duration == 3.0
