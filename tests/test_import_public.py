"""Tests for scripts/import_public.py (synthetic data, no real dataset needed)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.conftest import load_script


@pytest.fixture(scope="module")
def imp():
    return load_script("import_public")


def test_parse_nthu(imp) -> None:
    f = imp.parse_nthu(Path("x/yawning/002_glasses_yawning_1773_drowsy.jpg"))
    assert (f.subject, f.glasses, f.scenario, f.frame, f.hint) == ("002", "glasses", "yawning", 1773, "menguap")
    assert imp.parse_nthu(Path("x/notdrowsy/005_noglasses_nonsleepyCombination_1811_notdrowsy.jpg")).hint == "normal"
    assert imp.parse_nthu(Path("x/other/002_glasses_yawning_1_drowsy.jpg")) is None
    assert imp.parse_nthu(Path("x/yawning/readme.jpg")) is None


def test_select_evenly(imp) -> None:
    assert imp.select_evenly(list(range(10)), 3) == [0, 4, 9]
    assert imp.select_evenly([1, 2], 5) == [1, 2]
    assert imp.select_evenly([1, 2], 0) == []


def test_remove_color_overlay_cleans_boxes(imp) -> None:
    img = np.full((100, 100, 3), 120, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 255), 1)
    cv2.rectangle(img, (30, 40), (60, 50), (0, 255, 0), 1)
    cleaned, n = imp.remove_color_overlay(img, color_diff=40, dilate_px=1)
    assert n > 0
    spread = cleaned.astype(int).max(axis=2) - cleaned.astype(int).min(axis=2)
    assert spread.max() == 0  # grayscale again
    assert abs(int(cleaned[20, 50, 0]) - 120) < 15  # line replaced by surrounding gray


def test_remove_color_overlay_keeps_clean_frames(imp) -> None:
    img = np.full((10, 10, 3), 90, dtype=np.uint8)
    out, n = imp.remove_color_overlay(img, 40, 1)
    assert n == 0 and out is img


def test_import_nthu_end_to_end(imp, tmp_path: Path) -> None:
    src = tmp_path / "nthu" / "Multi class" / "train" / "drowsy" / "yawning"
    src.mkdir(parents=True)
    for i in range(6):
        img = np.full((48, 64, 3), 100, dtype=np.uint8)
        cv2.rectangle(img, (5, 5), (30, 30), (0, 255, 0), 1)
        cv2.imwrite(str(src / f"001_glasses_yawning_{i}_drowsy.jpg"), img)
    raw = tmp_path / "raw"
    counts = imp.import_nthu(tmp_path / "nthu", raw, {"menguap": 3}, color_diff=40, dilate_px=1)
    assert counts["menguap"] == 3
    files = sorted((raw / "nthu-001" / "menguap").glob("*.jpg"))
    assert [f.name for f in files][0].startswith("nthu-001__menguap__glasses__yawning-")
    out = cv2.imread(str(files[0])).astype(int)
    assert (out.max(axis=2) - out.min(axis=2)).max() <= 2  # overlay gone (JPEG tolerance)


def test_uta_naming(imp) -> None:
    assert imp.uta_participant(Path("/d/Fold1_part1/07/10.mov")) == "uta-7"
    assert imp.uta_participant(Path("/d/person_x/0.mp4")) == "uta-person-x"
    assert imp.uta_hint(Path("a/0.mov")) == "alert"
    assert imp.uta_hint(Path("a/5.MOV")) == "low_vigilant"
    assert imp.uta_hint(Path("a/10.mp4")) == "drowsy"
    assert imp.uta_hint(Path("a/Extra Clip.mp4")) == "extra_clip"


def test_extract_frames_and_import_uta(imp, tmp_path: Path) -> None:
    video = tmp_path / "uta" / "03" / "10.avi"
    video.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (1280, 720))
    for i in range(50):  # 5 s at 10 fps
        writer.write(np.full((720, 1280, 3), i * 4, dtype=np.uint8))
    writer.release()

    frames = list(imp.extract_frames(video, every_sec=1.0, max_frames=3))
    assert [pos for pos, _ in frames] == [0, 10, 20]

    raw = tmp_path / "raw"
    counts = imp.import_uta(tmp_path / "uta", raw, every_sec=2.0, max_frames=10, width=640)
    assert counts["drowsy"] == 3
    out = sorted((raw / "uta-3" / "drowsy").glob("*.jpg"))
    assert len(out) == 3
    assert cv2.imread(str(out[0])).shape[:2] == (360, 640)
