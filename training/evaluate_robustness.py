#!/usr/bin/env python3
"""FASE 8.1: accuracy per recording condition (light, glasses, distance ...) -> docs/eval/robustness.md

Usage:
    python training/evaluate_robustness.py --method landmark      # rules the app uses without YOLO (EAR/MAR)
    python training/evaluate_robustness.py --method yolo          # models/best.pt, mAP per condition
    python training/evaluate_robustness.py --method yolo --split val --label-status final

The condition is the third field of the raw file name (<subject>__<hint>__<condition>__<id>),
e.g. glasses/noglasses (NTHU) or the --kondisi tag of collect_data.py (terang, redup, backlight,
kacamata, jauh ...). Ground truth comes from the YOLO labels of the split.
With draft labels (made by preannotate.py from the same EAR/MAR rules) the landmark numbers
are circular and only check the pipeline: run with --label-status final after manual review.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.core.config import load_config, resolve_path  # noqa: E402
from src.detection.classes import CLASS_NAMES, MATA_TERBUKA, MATA_TERTUTUP, MENGUAP  # noqa: E402
from training.common import IMAGE_EXTS, parse_raw_name, read_labels  # noqa: E402

DRAFT_NOTE = ("> ⚠️ **Label draf.** Label masih hasil pra-anotasi otomatis; untuk metode landmark angka ini "
              "melingkar (aturan yang sama membuat labelnya). Hanya untuk cek pipeline, bukan untuk Bab IV.\n")


@dataclass
class BinaryScore:
    """Confusion counts of one binary decision."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, truth: bool, pred: bool) -> None:
        """Count one sample."""
        if truth and pred:
            self.tp += 1
        elif pred:
            self.fp += 1
        elif truth:
            self.fn += 1
        else:
            self.tn += 1

    def metrics(self) -> dict[str, float | None]:
        """precision, recall, f1, accuracy."""
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else None
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else None
        f1 = 2 * p * r / (p + r) if p and r else None
        n = self.tp + self.fp + self.fn + self.tn
        return {"precision": p, "recall": r, "f1": f1, "accuracy": (self.tp + self.tn) / n if n else None, "n": n}


def group_by_condition(images_dir: Path) -> dict[str, list[Path]]:
    """Images of a split grouped by the condition field of their name."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS or "__aug" in img.stem:
            continue
        try:
            groups[parse_raw_name(img).condition].append(img)
        except ValueError:
            groups["unknown"].append(img)
    return dict(groups)


def truth_of(label: Path) -> tuple[bool | None, bool]:
    """(eyes closed? None if no eye box, yawning?) from a YOLO label file."""
    classes = {b.cls for b in read_labels(label, len(CLASS_NAMES))} if label.exists() else set()
    eye = True if MATA_TERTUTUP in classes else False if MATA_TERBUKA in classes else None
    return eye, MENGUAP in classes


def eval_landmark(groups: dict[str, list[Path]], labels_dir: Path, cfg: dict[str, Any], ear_threshold: float) -> dict[str, Any]:
    """EAR/MAR rule decisions vs labels, per condition."""
    import cv2

    from src.detection.ear_calculator import ImageLandmarker, ensure_landmarker, measure

    det = ImageLandmarker(ensure_landmarker(resolve_path(cfg["landmark"]["model"])))
    mar_thr = float(cfg["landmark"]["mar_yawn"])
    out: dict[str, Any] = {}
    try:
        for cond, imgs in sorted(groups.items()):
            eyes, yawn, missed = BinaryScore(), BinaryScore(), 0
            for img in imgs:
                bgr = cv2.imread(str(img))
                pts = det.points(bgr) if bgr is not None else None
                eye_truth, yawn_truth = truth_of(labels_dir / f"{img.stem}.txt")
                if pts is None:
                    missed += 1
                    continue
                m = measure(pts, bgr.shape[1], bgr.shape[0])
                if eye_truth is not None:
                    eyes.add(eye_truth, (m.ear_mean or 0.0) < ear_threshold)
                yawn.add(yawn_truth, (m.mar or 0.0) > mar_thr)
            out[cond] = {"images": len(imgs), "face_missed": missed, "mata_tertutup": eyes.metrics(), "menguap": yawn.metrics()}
    finally:
        det.close()
    return out


def eval_yolo(groups: dict[str, list[Path]], labels_dir: Path, cfg: dict[str, Any], weights: Path) -> dict[str, Any]:
    """Ultralytics val per condition (temporary one-condition datasets)."""
    from ultralytics import YOLO

    from src.core.device import select_device

    model = YOLO(str(weights))
    device = select_device(cfg["inference"]["device"]).device
    out: dict[str, Any] = {}
    tmp = Path(tempfile.mkdtemp(prefix="robust_"))
    try:
        for cond, imgs in sorted(groups.items()):
            root = tmp / cond
            (root / "images" / "val").mkdir(parents=True)
            (root / "labels" / "val").mkdir(parents=True)
            for img in imgs:
                os.symlink(img, root / "images" / "val" / img.name)
                lab = labels_dir / f"{img.stem}.txt"
                if lab.exists():
                    os.symlink(lab, root / "labels" / "val" / lab.name)
            data = root / "data.yaml"
            data.write_text(yaml.safe_dump({"path": str(root), "train": "images/val", "val": "images/val",
                                            "names": dict(enumerate(CLASS_NAMES))}), encoding="utf-8")
            m = model.val(data=str(data), split="val", imgsz=int(cfg["train"]["imgsz"]), device=device,
                          plots=False, verbose=False, project=str(tmp / "runs"), name=cond)
            per_class = {CLASS_NAMES[int(c)]: float(m.box.class_result(i)[2]) for i, c in enumerate(m.box.ap_class_index)}
            out[cond] = {"images": len(imgs), "precision": float(m.box.mp), "recall": float(m.box.mr),
                         "map50": float(m.box.map50), "map50_95": float(m.box.map), "map50_per_class": per_class}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def _f(v: Any) -> str:
    return "–" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)


def render(method: str, split: str, results: dict[str, Any], label_status: str) -> str:
    """Markdown section."""
    lines = [f"## Metode `{method}` · split `{split}` · {datetime.now():%Y-%m-%d %H:%M}", ""]
    if label_status == "draf":
        lines.append(DRAFT_NOTE)
    if method == "landmark":
        lines += ["| Kondisi | Gambar | Wajah gagal | Mata tertutup P / R / F1 | Menguap P / R / F1 |", "|---|---|---|---|---|"]
        for cond, r in results.items():
            e, y = r["mata_tertutup"], r["menguap"]
            lines.append(f"| {cond} | {r['images']} | {r['face_missed']} | {_f(e['precision'])} / {_f(e['recall'])} / {_f(e['f1'])} | "
                         f"{_f(y['precision'])} / {_f(y['recall'])} / {_f(y['f1'])} |")
    else:
        lines += ["| Kondisi | Gambar | Precision | Recall | mAP50 | mAP50-95 | mAP50 per kelas |", "|---|---|---|---|---|---|---|"]
        for cond, r in results.items():
            pcs = ", ".join(f"{k} {v:.2f}" for k, v in r["map50_per_class"].items()) or "–"
            lines.append(f"| {cond} | {r['images']} | {_f(r['precision'])} | {_f(r['recall'])} | {_f(r['map50'])} | {_f(r['map50_95'])} | {pcs} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=("landmark", "yolo"), required=True)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--label-status", default="draf", choices=("draf", "final"))
    args = ap.parse_args(argv)

    cfg = load_config()
    out_dir = resolve_path(cfg["dataset"]["out_dir"])
    images_dir, labels_dir = out_dir / "images" / args.split, out_dir / "labels" / args.split
    if not images_dir.exists():
        print(f"Split {args.split} tidak ada. Jalankan training/split_dataset.py dulu.")
        return 1
    groups = group_by_condition(images_dir)
    if args.method == "yolo":
        weights = resolve_path(args.weights)
        if not weights.exists():
            print(f"{weights} belum ada. Latih model dulu (training/train.py) atau pakai --method landmark.")
            return 1
        results = eval_yolo(groups, labels_dir, cfg, weights)
    else:
        results = eval_landmark(groups, labels_dir, cfg, float(cfg["ear"]["closed_threshold"]))

    eval_dir = resolve_path(cfg["train"]["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    md = eval_dir / "robustness.md"
    if not md.exists():
        md.write_text("# Uji Ketahanan per Kondisi (FASE 8.1)\n\nKondisi diambil dari nama file (tag `--kondisi` saat perekaman).\n\n",
                      encoding="utf-8")
    with md.open("a", encoding="utf-8") as fh:
        fh.write(render(args.method, args.split, results, args.label_status) + "\n")
    (eval_dir / f"robustness_{args.method}.json").write_text(
        json.dumps({"split": args.split, "label_status": args.label_status, "results": results}, indent=2, default=str),
        encoding="utf-8")
    for cond, r in results.items():
        print(cond, json.dumps(r, default=str)[:200])
    print(f"Laporan: {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
