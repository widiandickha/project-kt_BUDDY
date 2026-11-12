"""Tests for training/augment.py."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from training import augment as aug
from training.common import YoloBox, read_labels, write_labels

PARAMS = aug.AugmentParams(brightness=0.25, contrast=0.25, blur_max_kernel=5, rotate_deg=10, hflip_p=0.5)


def test_hflip_mirrors_box() -> None:
    img = np.zeros((10, 20, 3), np.uint8)
    img[:, :5] = 255
    out, boxes = aug.hflip(img, [YoloBox(1, 0.2, 0.5, 0.1, 0.2)])
    assert out[:, -5:].min() == 255
    assert boxes[0].cx == pytest.approx(0.8)


def test_rotate_zero_is_identity() -> None:
    img = np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    box = YoloBox(0, 0.5, 0.5, 0.2, 0.2)
    out, boxes = aug.rotate(img, [box], 0.0)
    assert np.array_equal(out, img)
    assert boxes[0].cx == pytest.approx(0.5) and boxes[0].w == pytest.approx(0.2)


def test_rotate_grows_box_and_keeps_it_normalized() -> None:
    img = np.zeros((100, 100, 3), np.uint8)
    _, boxes = aug.rotate(img, [YoloBox(2, 0.5, 0.5, 0.4, 0.1)], 10.0)
    b = boxes[0]
    assert b.h > 0.1 and 0 < b.w <= 1 and 0 < b.h <= 1
    assert b.cx == pytest.approx(0.5, abs=1e-6)


def test_brightness_contrast_clip() -> None:
    img = np.full((2, 2, 3), 250, np.uint8)
    assert aug.adjust_brightness_contrast(img, 0.25, 0.0).max() == 255
    assert aug.adjust_brightness_contrast(img, 0.0, 0.0).max() == 250


def test_blur_kernel_rules() -> None:
    img = np.zeros((9, 9, 3), np.uint8)
    img[4, 4] = 255
    assert np.array_equal(aug.blur(img, 1), img)
    assert aug.blur(img, 4)[4, 4].max() < 255  # even kernel rounded down to 3


def test_augment_split_writes_copies_once(tmp_path: Path) -> None:
    images, labels = tmp_path / "images" / "train", tmp_path / "labels" / "train"
    images.mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(images / f"s__normal__x__{i}.jpg"), np.full((60, 80, 3), 100, np.uint8))
        write_labels(labels / f"s__normal__x__{i}.txt", [YoloBox(1, 0.5, 0.5, 0.3, 0.2)])

    assert aug.augment_split(images, labels, copies=2, p=PARAMS, seed=0) == 6
    # Running again must not augment the augmented copies.
    assert aug.augment_split(images, labels, copies=2, p=PARAMS, seed=0) == 6
    assert len(list(images.glob("*.jpg"))) == 9
    for lbl in labels.glob(f"*{aug.AUG_TAG}*.txt"):
        assert all(b.cls == 1 for b in read_labels(lbl, 4))
