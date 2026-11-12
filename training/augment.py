#!/usr/bin/env python3
"""FASE 2: offline augmentation of the TRAIN split (brightness, contrast, light blur, ±10° rotation, h-flip).

Usage:
    python training/augment.py                 # dataset.augment.copies per train image
    python training/augment.py --copies 2

Only data/dataset/{images,labels}/train is augmented, so val/test stay untouched and no person
leaks across splits. Copies are named <stem>__aug<k>; existing copies are replaced, and running
training/split_dataset.py again removes them. Horizontal flip is safe because both eyes share one
box (D-009), so there is no left/right convention to break.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.core.config import load_config, resolve_path  # noqa: E402
from src.detection.classes import CLASS_NAMES  # noqa: E402
from training.common import IMAGE_EXTS, YoloBox, read_labels, write_labels, xyxy_to_yolo  # noqa: E402

AUG_TAG = "__aug"
MIN_BOX_KEEP = 0.4  # drop a box if rotation/clipping leaves < 40% of its area


@dataclass(frozen=True)
class AugmentParams:
    """Augmentation ranges from config dataset.augment."""

    brightness: float
    contrast: float
    blur_max_kernel: int
    rotate_deg: float
    hflip_p: float


def adjust_brightness_contrast(img: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """out = (img - 128) * (1 + contrast) + 128 + brightness * 255, clipped."""
    out = (img.astype(np.float32) - 128.0) * (1.0 + contrast) + 128.0 + brightness * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def blur(img: np.ndarray, kernel: int) -> np.ndarray:
    """Gaussian blur with an odd kernel; kernel < 3 returns the image unchanged."""
    if kernel < 3:
        return img
    k = kernel if kernel % 2 == 1 else kernel - 1
    return cv2.GaussianBlur(img, (k, k), 0)


def hflip(img: np.ndarray, boxes: list[YoloBox]) -> tuple[np.ndarray, list[YoloBox]]:
    """Mirror horizontally."""
    return cv2.flip(img, 1), [YoloBox(b.cls, 1.0 - b.cx, b.cy, b.w, b.h) for b in boxes]


def rotate(img: np.ndarray, boxes: list[YoloBox], deg: float) -> tuple[np.ndarray, list[YoloBox]]:
    """Rotate around the center; boxes become the axis-aligned hull of their rotated corners."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    new_boxes: list[YoloBox] = []
    for b in boxes:
        x1, y1 = (b.cx - b.w / 2) * w, (b.cy - b.h / 2) * h
        x2, y2 = (b.cx + b.w / 2) * w, (b.cy + b.h / 2) * h
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float64)
        pts = corners @ m.T
        nb = xyxy_to_yolo(b.cls, pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max(), w, h)
        if nb is None:
            continue
        hull_area = (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
        if hull_area > 0 and (nb.w * w * nb.h * h) / hull_area >= MIN_BOX_KEEP:
            new_boxes.append(nb)
    return out, new_boxes


def augment_once(
    img: np.ndarray, boxes: list[YoloBox], p: AugmentParams, rng: np.random.Generator
) -> tuple[np.ndarray, list[YoloBox]]:
    """Apply one random combination of all augmentations."""
    out = adjust_brightness_contrast(
        img, rng.uniform(-p.brightness, p.brightness), rng.uniform(-p.contrast, p.contrast)
    )
    out = blur(out, int(rng.integers(0, p.blur_max_kernel + 1)))
    out, boxes = rotate(out, boxes, rng.uniform(-p.rotate_deg, p.rotate_deg))
    if rng.random() < p.hflip_p:
        out, boxes = hflip(out, boxes)
    return out, boxes


def augment_split(images_dir: Path, labels_dir: Path, copies: int, p: AugmentParams, seed: int) -> int:
    """Write `copies` augmented versions of each original image; returns how many were written."""
    rng = np.random.default_rng(seed)
    originals = sorted(
        x for x in images_dir.iterdir() if x.suffix.lower() in IMAGE_EXTS and AUG_TAG not in x.stem
    )
    written = 0
    for img_path in originals:
        label = labels_dir / f"{img_path.stem}.txt"
        if not label.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes = read_labels(label, len(CLASS_NAMES))
        for k in range(1, copies + 1):
            out, out_boxes = augment_once(img, boxes, p, rng)
            name = f"{img_path.stem}{AUG_TAG}{k}"
            cv2.imwrite(str(images_dir / f"{name}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_labels(labels_dir / f"{name}.txt", out_boxes)
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copies", type=int, default=None, help="jumlah salinan per gambar (default dataset.augment.copies)")
    args = ap.parse_args(argv)

    cfg = load_config()
    aug = cfg["dataset"]["augment"]
    out_dir = resolve_path(cfg["dataset"]["out_dir"])
    images_dir, labels_dir = out_dir / "images" / "train", out_dir / "labels" / "train"
    if not images_dir.exists():
        print("Split train belum ada. Jalankan training/split_dataset.py dulu.")
        return 1
    params = AugmentParams(
        brightness=float(aug["brightness"]), contrast=float(aug["contrast"]),
        blur_max_kernel=int(aug["blur_max_kernel"]), rotate_deg=float(aug["rotate_deg"]), hflip_p=float(aug["hflip_p"]),
    )
    copies = args.copies if args.copies is not None else int(aug["copies"])
    n = augment_split(images_dir, labels_dir, copies, params, int(aug["seed"]))
    print(f"{n} gambar augmentasi ditulis ke {images_dir}")
    # Ultralytics caches label scans; drop the cache so the new files are picked up.
    cache = labels_dir.parent / "train.cache"
    if cache.exists():
        cache.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
