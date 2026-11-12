#!/usr/bin/env python3
"""FASE 2 DoD check: run Ultralytics validation over every split of data/data.yaml.

Usage:
    python training/check_dataset.py

Uses an untrained 4-class YOLOv8n built from yolov8n.yaml, so class names match the dataset.
Metrics are meaningless (≈0); the point is that the dataset structure loads without errors and
that every split has images and labels. Exit code 1 on any problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import ROOT_DIR, load_config, resolve_path  # noqa: E402
from src.core.device import select_device  # noqa: E402
from src.detection.classes import CLASS_NAMES  # noqa: E402

SPLITS: tuple[str, ...] = ("train", "val", "test")


def main() -> int:
    """Entry point."""
    import os

    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    cfg = load_config()
    data_yaml = resolve_path(cfg["dataset"]["data_yaml"])
    os.chdir(ROOT_DIR)  # data.yaml uses a path relative to the project root (spec 2.1)

    data = check_det_dataset(str(data_yaml))
    names = tuple(data["names"][i] for i in range(len(data["names"])))
    if names != CLASS_NAMES:
        print(f"Urutan kelas salah: {names} != {CLASS_NAMES}")
        return 1

    device = select_device(cfg["inference"]["device"]).device
    model = YOLO("yolov8n.yaml")
    problems = 0
    for split in SPLITS:
        img_dir = Path(data[split])
        n_images = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if n_images == 0:
            print(f"[{split}] KOSONG: {img_dir}")
            problems += 1
            continue
        model.val(
            data=str(data_yaml), split=split, imgsz=int(cfg["inference"]["imgsz"]), batch=16,
            device=device, plots=False, verbose=False, project=str(ROOT_DIR / "runs" / "check"),
            name=split, exist_ok=True,
        )
        print(f"[{split}] OK: {n_images} gambar tervalidasi")
    print("DoD FASE 2: yolo val jalan tanpa error." if problems == 0 else f"{problems} split bermasalah.")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
