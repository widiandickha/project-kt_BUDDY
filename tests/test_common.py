"""Tests for training/common.py and the fixed class list."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.detection.classes import CLASS_NAMES
from training.common import (
    YoloBox,
    label_path_for,
    make_raw_name,
    parse_raw_name,
    read_labels,
    slug,
    write_labels,
    xyxy_to_yolo,
)


def test_class_order_is_final() -> None:
    assert CLASS_NAMES == ("mata_tertutup", "mata_terbuka", "menguap", "kerutan_dahi")


def test_raw_name_round_trip() -> None:
    name = make_raw_name("Self Andi", "menguap", "Redup Kacamata", 1712)
    assert name == "self-andi__menguap__redup-kacamata__1712.jpg"
    meta = parse_raw_name(name)
    assert (meta.subject, meta.hint, meta.condition, meta.frame_id) == ("self-andi", "menguap", "redup-kacamata", "1712")


def test_parse_survives_roboflow_rename() -> None:
    meta = parse_raw_name("nthu-001__menguap__glasses__yawning-1773_jpg.rf.4f2a9c.jpg")
    assert meta.subject == "nthu-001"
    assert meta.frame_id == "yawning-1773"


def test_bad_names_rejected() -> None:
    with pytest.raises(ValueError):
        parse_raw_name("random_photo.jpg")
    with pytest.raises(ValueError):
        make_raw_name("a", "Bad Hint!", "x", 1)
    with pytest.raises(ValueError):
        slug("___")


def test_labels_round_trip_and_negative(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    boxes = [YoloBox(0, 0.5, 0.4, 0.2, 0.1), YoloBox(2, 0.5, 0.7, 0.3, 0.3)]
    write_labels(p, boxes)
    assert read_labels(p, 4) == [YoloBox(0, 0.5, 0.4, 0.2, 0.1), YoloBox(2, 0.5, 0.7, 0.3, 0.3)]
    write_labels(p, [])
    assert read_labels(p, 4) == []


@pytest.mark.parametrize("line", ["7 0.5 0.5 0.1 0.1", "0 1.5 0.5 0.1 0.1", "0 0.5 0.5", "0 0.5 0.5 0 0.1"])
def test_invalid_label_lines(tmp_path: Path, line: str) -> None:
    p = tmp_path / "bad.txt"
    p.write_text(line + "\n")
    with pytest.raises(ValueError):
        read_labels(p, 4)


def test_xyxy_to_yolo_clips_and_drops() -> None:
    box = xyxy_to_yolo(1, -10, 20, 50, 60, 100, 100)
    assert box == YoloBox(1, 0.25, 0.4, 0.5, 0.4)
    assert xyxy_to_yolo(1, 120, 10, 150, 20, 100, 100) is None


def test_label_path_beside_or_in_labels_dir(tmp_path: Path) -> None:
    img = tmp_path / "export" / "images" / "x.jpg"
    img.parent.mkdir(parents=True)
    img.touch()
    assert label_path_for(img) == tmp_path / "export" / "labels" / "x.txt"
    beside = img.with_suffix(".txt")
    beside.touch()
    assert label_path_for(img) == beside
