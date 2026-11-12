#!/usr/bin/env python3
"""FASE 3: evaluate the face detector per class on the TEST split and write docs/eval/.

Usage:
    python training/evaluate.py                          # models/best.pt, split test, label draf
    python training/evaluate.py --label-status final     # only once labels are hand-corrected
    python training/evaluate.py --weights runs/detect/face2/weights/best.pt --split val

Outputs in docs/eval/: metrics.md, metrics.json, confusion_matrix(_normalized).png,
PR/F1 curves, and train_results.png (training curves of the run that produced the weights).
The test split holds subjects never seen in training (split per person, FASE 2).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import ROOT_DIR, load_config, resolve_path  # noqa: E402
from src.core.device import select_device  # noqa: E402
from src.detection.classes import CLASS_NAMES  # noqa: E402

PLOT_FILES: tuple[str, ...] = (
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "PR_curve.png",
    "BoxF1_curve.png",
    "F1_curve.png",
)
LABEL_STATUSES: tuple[str, ...] = ("draf", "final")


@dataclass
class ClassMetrics:
    """Detection metrics of one class on the evaluated split."""

    name: str
    instances: int
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None
    meets_target: bool | None


def count_instances(labels_dir: Path) -> list[int]:
    """Instances per class from the YOLO label files of a split."""
    counts = [0] * len(CLASS_NAMES)
    for txt in labels_dir.glob("*.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return counts


def per_class_metrics(
    class_results: dict[int, tuple[float, float, float, float]], instances: list[int], target: float
) -> list[ClassMetrics]:
    """Build the per-class table; classes absent from the split get None metrics."""
    rows: list[ClassMetrics] = []
    for i, name in enumerate(CLASS_NAMES):
        if i in class_results and instances[i] > 0:
            p, r, m50, m = class_results[i]
            rows.append(ClassMetrics(name, instances[i], p, r, m50, m, m50 >= target))
        else:
            rows.append(ClassMetrics(name, instances[i], None, None, None, None, None))
    return rows


def _fmt(v: float | None) -> str:
    return "–" if v is None else f"{v:.3f}"


def render_markdown(
    rows: list[ClassMetrics],
    overall: dict[str, float],
    meta: dict[str, Any],
    target: float,
    label_status: str,
) -> str:
    """Markdown report for docs/eval/metrics.md."""
    lines = ["# Evaluasi Detektor Wajah", ""]
    if label_status == "draf":
        lines += [
            "> ⚠️ **Label draf.** Dataset masih memakai pra-anotasi otomatis yang belum dikoreksi manual "
            "(lihat `docs/dataset_report.md`). Angka di bawah hanya untuk mengecek pipeline dan **tidak boleh** "
            "dipakai di laporan Bab IV.",
            "",
        ]
    lines += [
        f"- Dievaluasi: {meta['evaluated_at']}",
        f"- Bobot: `{meta['weights']}`",
        f"- Split: `{meta['split']}` (subjek: {', '.join(meta['subjects']) or '-'}; tidak dipakai saat training)",
        f"- imgsz: {meta['imgsz']} · device: {meta['device']} · input grayscale: {meta['grayscale']}",
    ]
    if meta.get("train"):
        t = meta["train"]
        lines.append(
            f"- Training: base `{t.get('base_model')}`, epoch terbaik {t.get('best_epoch')}, "
            f"durasi {t.get('duration_min')} menit di {t.get('device')}"
        )
    lines += [
        "",
        "## Per kelas",
        "",
        f"Target spesifikasi: mAP50 ≥ {target:.2f} per kelas.",
        "",
        "| Kelas | Instance | Precision | Recall | mAP50 | mAP50-95 | Target |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        status = "–" if r.meets_target is None else ("✅" if r.meets_target else "❌")
        lines.append(
            f"| {r.name} | {r.instances} | {_fmt(r.precision)} | {_fmt(r.recall)} | {_fmt(r.map50)} | {_fmt(r.map50_95)} | {status} |"
        )
    lines.append(
        f"| **semua** | {sum(r.instances for r in rows)} | {_fmt(overall['precision'])} | {_fmt(overall['recall'])} | "
        f"{_fmt(overall['map50'])} | {_fmt(overall['map50_95'])} | |"
    )
    speed = meta.get("speed_ms", {})
    if speed:
        lines += ["", f"Kecepatan per gambar: preprocess {speed.get('preprocess', 0):.1f} ms, "
                  f"inferensi {speed.get('inference', 0):.1f} ms, postprocess {speed.get('postprocess', 0):.1f} ms."]

    missing = [r.name for r in rows if r.meets_target is None]
    below = [r.name for r in rows if r.meets_target is False]
    lines += ["", "## Keterbatasan", ""]
    if missing:
        lines.append(f"- Tidak ada instance di split ini, jadi tidak bisa diukur: {', '.join(f'`{m}`' for m in missing)}.")
    if below:
        lines.append(f"- Di bawah target: {', '.join(f'`{b}`' for b in below)}. Sesuai spesifikasi, catat sebagai keterbatasan "
                     "dan turunkan bobot indikatornya di `score.weights` bila tetap rendah setelah data diperbaiki.")
    if not missing and not below:
        lines.append("- Semua kelas memenuhi target.")
    lines += ["", "Gambar: `confusion_matrix.png`, `confusion_matrix_normalized.png`, kurva PR dan F1, `train_results.png`.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    import os

    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--imgsz", type=int, default=None, help="default train.imgsz")
    ap.add_argument("--device", default=None)
    ap.add_argument("--label-status", default="draf", choices=LABEL_STATUSES,
                    help="'final' hanya bila semua label sudah dikoreksi manual")
    args = ap.parse_args(argv)

    cfg = load_config()
    os.chdir(ROOT_DIR)
    weights = resolve_path(args.weights)
    if not weights.exists():
        print(f"Bobot tidak ditemukan: {weights}. Jalankan training/train.py dulu.")
        return 1
    device = select_device(args.device or cfg["inference"]["device"]).device
    imgsz = args.imgsz or int(cfg["train"]["imgsz"])
    target = float(cfg["train"]["target_map50"])
    eval_dir = resolve_path(cfg["train"]["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(resolve_path(cfg["dataset"]["data_yaml"])), split=args.split, imgsz=imgsz, device=device,
        conf=0.001, iou=float(cfg["det"]["iou"]), plots=True, verbose=False,
        project=str(ROOT_DIR / "runs" / "eval"), name=args.split, exist_ok=True,
    )

    class_results = {int(c): tuple(float(v) for v in metrics.box.class_result(i))
                     for i, c in enumerate(metrics.box.ap_class_index)}
    labels_dir = resolve_path(cfg["dataset"]["out_dir"]) / "labels" / args.split
    instances = count_instances(labels_dir)
    rows = per_class_metrics(class_results, instances, target)
    overall = {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
               "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)}

    subjects = sorted({p.name.split("__")[0] for p in labels_dir.glob("*.txt")})
    run_meta_path = weights.parent.parent / "run_meta.json"
    train_meta: dict[str, Any] = {}
    if run_meta_path.exists():
        train_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    else:  # models/best.pt: use the most recent training run
        runs = sorted(resolve_path(cfg["train"]["project"]).glob("*/run_meta.json"), key=lambda p: p.stat().st_mtime)
        if runs:
            run_meta_path = runs[-1]
            train_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    meta = {
        "evaluated_at": f"{datetime.now():%Y-%m-%d %H:%M}",
        "weights": str(weights.relative_to(ROOT_DIR)) if weights.is_relative_to(ROOT_DIR) else str(weights),
        "split": args.split,
        "subjects": subjects,
        "imgsz": imgsz,
        "device": device,
        "grayscale": bool(cfg["det"].get("grayscale", False)),
        "label_status": args.label_status,
        "speed_ms": {k: float(v) for k, v in (metrics.speed or {}).items()},
        "train": {k: train_meta.get(k) for k in ("base_model", "best_epoch", "duration_min", "device")} if train_meta else None,
    }

    save_dir = Path(metrics.save_dir)
    for name in PLOT_FILES:
        if (save_dir / name).exists():
            shutil.copy2(save_dir / name, eval_dir / name)
    if run_meta_path.exists() and (run_meta_path.parent / "results.png").exists():
        shutil.copy2(run_meta_path.parent / "results.png", eval_dir / "train_results.png")

    (eval_dir / "metrics.md").write_text(render_markdown(rows, overall, meta, target, args.label_status), encoding="utf-8")
    (eval_dir / "metrics.json").write_text(
        json.dumps({"meta": meta, "overall": overall, "classes": [asdict(r) for r in rows]}, indent=2), encoding="utf-8"
    )
    for r in rows:
        print(f"{r.name:14s} n={r.instances:4d}  P={_fmt(r.precision)}  R={_fmt(r.recall)}  mAP50={_fmt(r.map50)}  mAP50-95={_fmt(r.map50_95)}")
    print(f"Laporan: {eval_dir / 'metrics.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
