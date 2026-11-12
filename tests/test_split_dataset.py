"""Tests for training/split_dataset.py."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from training import split_dataset as sd
from training.common import make_raw_name

RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}


def test_every_split_gets_a_subject_when_possible() -> None:
    counts = {"a": 100, "b": 100, "c": 100, "d": 100, "e": 100, "f": 100}
    result = sd.assign_subjects(counts, RATIOS, seed=1)
    assert set(result.values()) == {"train", "val", "test"}
    assert sum(1 for s in result.values() if s == "train") >= 3


def test_assignment_is_deterministic_and_respects_fixed() -> None:
    counts = {"a": 50, "b": 40, "c": 30, "d": 20}
    first = sd.assign_subjects(counts, RATIOS, seed=7, fixed={"d": "train"})
    assert first == sd.assign_subjects(counts, RATIOS, seed=7, fixed={"d": "train"})
    assert first["d"] == "train"
    with pytest.raises(ValueError):
        sd.assign_subjects(counts, RATIOS, seed=7, fixed={"a": "holdout"})


def test_two_subjects_fill_train_first() -> None:
    result = sd.assign_subjects({"a": 10, "b": 10}, RATIOS, seed=0)
    assert sorted(result.values()) == ["train", "val"]


def _make(raw: Path, subject: str, n: int, label: str | None) -> None:
    folder = raw / subject / "x"
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = folder / make_raw_name(subject, "normal", "terang", i)
        cv2.imwrite(str(img), np.zeros((8, 8, 3), np.uint8))
        if label is not None:
            img.with_suffix(".txt").write_text(label)


def test_end_to_end_build(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for s in ("s1", "s2", "s3", "s4"):
        _make(raw, s, 5, "1 0.5 0.4 0.3 0.1\n")
    _make(raw, "s5", 2, "")  # negatives
    _make(raw, "s6", 3, None)  # not annotated yet

    items, unlabeled, errors = sd.collect_items([raw])
    assert errors == [] and len(unlabeled) == 3 and len(items) == 22

    counts = {s: sum(1 for it in items if it.subject == s) for s in {it.subject for it in items}}
    assignment = sd.assign_subjects(counts, RATIOS, 0)
    out = tmp_path / "dataset"
    stats = sd.build_dataset(items, assignment, out)
    assert sum(st.images for st in stats.values()) == 22
    assert sum(st.negatives for st in stats.values()) == 2
    for split, st in stats.items():
        assert len(list((out / "images" / split).glob("*.jpg"))) == st.images
    subjects_per_split = [st.subjects for st in stats.values()]
    assert not (subjects_per_split[0] & subjects_per_split[1]) and not (subjects_per_split[0] & subjects_per_split[2])

    data_yaml = tmp_path / "data.yaml"
    sd.write_data_yaml(data_yaml, out)
    doc = yaml.safe_load(data_yaml.read_text())
    assert doc["names"] == {0: "mata_tertutup", 1: "mata_terbuka", 2: "menguap", 3: "kerutan_dahi"}
    assert doc["train"] == "images/train"

    report = sd.build_report(items, assignment, stats, unlabeled, target_per_class=500, min_subjects=5)
    assert "3 gambar belum punya file label" in report
    assert "`menguap`: 0 instance" in report


def test_bad_label_is_reported(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make(raw, "s1", 1, "9 0.5 0.5 0.1 0.1\n")
    items, _, errors = sd.collect_items([raw])
    assert items == [] and len(errors) == 1


def test_parse_assign() -> None:
    assert sd.parse_assign(["nthu-006=train"]) == {"nthu-006": "train"}
    with pytest.raises(SystemExit):
        sd.parse_assign(["nthu-006"])


def test_place_grayscale_converts_color_and_links_gray(tmp_path: Path) -> None:
    color = tmp_path / "s__x__y__1.png"
    gray = tmp_path / "s__x__y__2.jpg"
    img = np.zeros((10, 10, 3), np.uint8)
    img[..., 2] = 220
    cv2.imwrite(str(color), img)
    cv2.imwrite(str(gray), np.full((10, 10, 3), 90, np.uint8))
    out = tmp_path / "out"
    sd.place_grayscale(color, out)
    sd.place_grayscale(gray, out)
    converted = cv2.imread(str(out / "s__x__y__1.jpg")).astype(int)
    assert (converted.max(axis=2) - converted.min(axis=2)).max() <= sd.GRAY_TOLERANCE
    assert (out / "s__x__y__2.jpg").stat().st_ino == gray.stat().st_ino  # hard link, no re-encode
