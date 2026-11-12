#!/usr/bin/env python3
"""FASE 3: train the YOLOv8 face detector (4 classes) and copy the best weights to models/best.pt.

Usage:
    python training/train.py                         # hyperparameters from config `train`
    python training/train.py --model yolov8s.pt      # bigger base when mAP50 is below target
    python training/train.py --epochs 1 --fraction 0.1 --name smoke   # quick smoke test
    python training/train.py --resume                # continue an interrupted run

Logs, curves and weights go to runs/detect/<name>/. Only a full run (not --no-copy) replaces
models/best.pt; the previous file is kept as models/best.prev.pt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import ROOT_DIR, load_config, resolve_path  # noqa: E402
from src.core.device import select_device  # noqa: E402

# Ultralytics augmentation knobs that only make sense for color images; zeroed because the
# detector sees grayscale input (D-013).
GRAYSCALE_OVERRIDES: dict[str, float] = {"hsv_h": 0.0, "hsv_s": 0.0}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="bobot awal (default train.base_model)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None, help="auto | cuda | mps | cpu (default inference.device)")
    ap.add_argument("--name", default=None, help="nama run di runs/detect/ (default train.name)")
    ap.add_argument("--fraction", type=float, default=1.0, help="pakai sebagian data train (smoke test)")
    ap.add_argument("--resume", action="store_true", help="lanjutkan run terakhir dengan nama yang sama")
    ap.add_argument("--no-copy", action="store_true", help="jangan salin best.pt ke models/")
    return ap.parse_args(argv)


def build_train_kwargs(cfg: dict[str, Any], args: argparse.Namespace, device: str) -> dict[str, Any]:
    """Merge config and CLI into Ultralytics train() keyword arguments."""
    tr = cfg["train"]
    kwargs: dict[str, Any] = {
        "data": str(resolve_path(cfg["dataset"]["data_yaml"])),
        "epochs": args.epochs or int(tr["epochs"]),
        "imgsz": args.imgsz or int(tr["imgsz"]),
        "batch": args.batch or int(tr["batch"]),
        "patience": int(tr["patience"]),
        "augment": bool(tr["augment"]),
        "workers": int(tr["workers"]),
        "seed": int(tr["seed"]),
        "deterministic": True,
        "device": device,
        "project": str(resolve_path(tr["project"])),
        "name": args.name or str(tr["name"]),
        "exist_ok": False,
        "fraction": args.fraction,
        "plots": True,
        "verbose": True,
    }
    if cfg["det"].get("grayscale", False):
        kwargs.update(GRAYSCALE_OVERRIDES)
    return kwargs


def copy_best(best: Path, models_dir: Path) -> Path:
    """Copy best.pt to models/best.pt, keeping the previous one as best.prev.pt."""
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / "best.pt"
    if target.exists():
        shutil.copy2(target, models_dir / "best.prev.pt")
    shutil.copy2(best, target)
    return target


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    import os

    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    args = parse_args(argv)
    cfg = load_config()
    os.chdir(ROOT_DIR)  # data.yaml paths are relative to the project root
    device = select_device(args.device or cfg["inference"]["device"]).device

    if args.resume:
        name = args.name or str(cfg["train"]["name"])
        last = resolve_path(cfg["train"]["project"]) / name / "weights" / "last.pt"
        if not last.exists():
            print(f"Tidak ada checkpoint untuk dilanjutkan: {last}")
            return 1
        model = YOLO(str(last))
        results = model.train(resume=True)
        save_dir = Path(model.trainer.save_dir)
    else:
        base = args.model or str(cfg["train"]["base_model"])
        weights = attempt_download_asset(str(resolve_path(Path("models") / base))) if "/" not in base else base
        model = YOLO(str(weights))
        kwargs = build_train_kwargs(cfg, args, device)
        print(f"Training {base} di {device}: " + ", ".join(f"{k}={kwargs[k]}" for k in ("epochs", "imgsz", "batch", "patience")))
        started = time.time()
        results = model.train(**kwargs)
        save_dir = Path(model.trainer.save_dir)
        meta = {
            "base_model": base,
            "device": device,
            "duration_min": round((time.time() - started) / 60.0, 1),
            "train_kwargs": kwargs,
            "best_epoch": getattr(model.trainer, "best_epoch", None),
        }
        (save_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"best.pt tidak ditemukan di {best}")
        return 1
    print(f"Run selesai: {save_dir}")
    if args.no_copy or args.fraction < 1.0:
        print("best.pt tidak disalin (smoke test / --no-copy).")
    else:
        print(f"Bobot terbaik disalin ke {copy_best(best, ROOT_DIR / 'models')}")
    _ = results
    return 0


if __name__ == "__main__":
    sys.exit(main())
