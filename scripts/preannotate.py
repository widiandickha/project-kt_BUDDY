#!/usr/bin/env python3
"""FASE 2: draft YOLO labels from MediaPipe face landmarks, to be corrected in an annotation tool.

Usage:
    python scripts/preannotate.py                    # every image in data/raw without a label
    python scripts/preannotate.py --src data/raw/self-andi
    python scripts/preannotate.py --overwrite        # also redo existing labels (careful!)

Box conventions (docs/DECISIONS.md D-009):
    mata_tertutup / mata_terbuka : ONE box covering both eyes (state from EAR)
    menguap                      : lower face, nose base to chin, incl. the open mouth (from MAR)
    kerutan_dahi                 : forehead + brows; only added when the hint is kerutan_dahi

Existing label files are never touched without --overwrite, so corrected labels are safe.
Writes <image>.txt beside each image, classes.txt in the raw root, and _preannotate.csv
listing EAR/MAR and review flags (sort on the "flags" column to review doubtful ones first).
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.core.config import load_config, resolve_path  # noqa: E402
from src.detection.classes import CLASS_NAMES, KERUTAN_DAHI, MATA_TERBUKA, MATA_TERTUTUP, MENGUAP  # noqa: E402
from src.detection.ear_calculator import (  # noqa: E402
    FACE_SIDES,
    LEFT_EYE,
    MOUTH_CORNERS,
    MOUTH_INNER_V,
    RIGHT_EYE,
    compute_ear,
    compute_mar,
    ImageLandmarker,
    ensure_landmarker,
)
from training.common import YoloBox, iter_images, parse_raw_name, write_labels, xyxy_to_yolo  # noqa: E402

# Same EAR/MAR definitions as the runtime (src/detection/ear_calculator.py).
EAR_A: tuple[int, ...] = RIGHT_EYE
EAR_B: tuple[int, ...] = LEFT_EYE
ear_of = compute_ear
mar_of = compute_mar
_ = (MOUTH_CORNERS, MOUTH_INNER_V, FACE_SIDES)  # re-exported for tests
EYES: tuple[int, ...] = (
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
)
LOWER_FACE: tuple[int, ...] = (2, 152, 61, 291, 0, 17, 148, 377, 176, 400, 58, 288)
BROW_FOREHEAD: tuple[int, ...] = (
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 336, 296, 334, 293, 300, 285, 295, 282, 283, 276,
    10, 109, 338, 67, 297, 151,
)


@dataclass
class Thresholds:
    """Pre-annotation thresholds from config dataset.preannotate."""

    ear_closed: float
    ear_open: float
    mar_yawn: float
    pad_eyes: float
    pad_mouth: float
    pad_brow: float


@dataclass
class Annotation:
    """Result for one image."""

    face: bool
    ear: float | None = None
    mar: float | None = None
    boxes: list[YoloBox] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def region_box(pts: np.ndarray, idx: tuple[int, ...], pad: float, face_w: float, cls: int, w: int, h: int) -> YoloBox | None:
    """YOLO box around a landmark subset, padded by `pad * face width` on each side."""
    sub = pts[list(idx)]
    p = pad * face_w
    return xyxy_to_yolo(cls, sub[:, 0].min() - p, sub[:, 1].min() - p, sub[:, 0].max() + p, sub[:, 1].max() + p, w, h)


def annotate_points(pts: np.ndarray, width: int, height: int, hint: str, th: Thresholds) -> Annotation:
    """Turn 478 pixel landmarks into draft boxes + review flags."""
    face_w = float(np.linalg.norm(pts[FACE_SIDES[0]] - pts[FACE_SIDES[1]]))
    ear = (ear_of(pts, EAR_A) + ear_of(pts, EAR_B)) / 2.0
    mar = mar_of(pts)
    ann = Annotation(face=True, ear=ear, mar=mar)

    if ear < th.ear_closed:
        eye_cls = MATA_TERTUTUP
    elif ear > th.ear_open:
        eye_cls = MATA_TERBUKA
    else:
        mid = (th.ear_closed + th.ear_open) / 2.0
        eye_cls = MATA_TERTUTUP if ear < mid else MATA_TERBUKA
        ann.flags.append("ragu_mata")
    for cls, idx, pad in ((eye_cls, EYES, th.pad_eyes),):
        box = region_box(pts, idx, pad, face_w, cls, width, height)
        if box:
            ann.boxes.append(box)

    yawn = mar > th.mar_yawn
    if yawn:
        box = region_box(pts, LOWER_FACE, th.pad_mouth, face_w, MENGUAP, width, height)
        if box:
            ann.boxes.append(box)
        if hint != "menguap":
            ann.flags.append("menguap_tak_terduga")
    elif hint == "menguap":
        ann.flags.append("hint_menguap_mulut_tertutup")

    if hint == "kerutan_dahi":
        box = region_box(pts, BROW_FOREHEAD, th.pad_brow, face_w, KERUTAN_DAHI, width, height)
        if box:
            ann.boxes.append(box)
        ann.flags.append("cek_kerutan")
    if hint == "mata_tertutup" and eye_cls != MATA_TERTUTUP:
        ann.flags.append("hint_tertutup_ear_terbuka")
    return ann


LandmarkDetector = ImageLandmarker  # kept name for readability


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=None, help="folder gambar (default dataset.raw_dir)")
    ap.add_argument("--overwrite", action="store_true", help="timpa label yang sudah ada (hati-hati: label hasil koreksi hilang)")
    args = ap.parse_args(argv)

    cfg = load_config()
    ds = cfg["dataset"]
    raw_dir = resolve_path(ds["raw_dir"])
    src = resolve_path(args.src) if args.src else raw_dir
    pre = ds["preannotate"]
    th = Thresholds(**{k: float(pre[k]) for k in ("ear_closed", "ear_open", "mar_yawn", "pad_eyes", "pad_mouth", "pad_brow")})

    detector = LandmarkDetector(ensure_landmarker(resolve_path(pre["landmarker"])))
    (raw_dir / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    csv_path = src / "_preannotate.csv"
    stats = {"total": 0, "dilabeli": 0, "dilewati_sudah_ada": 0, "tanpa_wajah": 0, "ditandai": 0}

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", "subject", "hint", "face", "ear", "mar", "classes", "flags"])
            for img_path in iter_images(src):
                stats["total"] += 1
                label_path = img_path.with_suffix(".txt")
                if label_path.exists() and not args.overwrite:
                    stats["dilewati_sudah_ada"] += 1
                    continue
                try:
                    meta = parse_raw_name(img_path)
                except ValueError:
                    print(f"  lewati (nama tidak sesuai skema): {img_path.name}")
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                pts = detector.points(img)
                if pts is None:
                    ann = Annotation(face=False, flags=["wajah_tidak_terdeteksi"])
                    stats["tanpa_wajah"] += 1
                else:
                    ann = annotate_points(pts, img.shape[1], img.shape[0], meta.hint, th)
                    write_labels(label_path, ann.boxes)
                    stats["dilabeli"] += 1
                if ann.flags:
                    stats["ditandai"] += 1
                writer.writerow([
                    img_path.relative_to(src), meta.subject, meta.hint, int(ann.face),
                    f"{ann.ear:.3f}" if ann.ear is not None else "", f"{ann.mar:.3f}" if ann.mar is not None else "",
                    " ".join(CLASS_NAMES[b.cls] for b in ann.boxes), " ".join(ann.flags),
                ])
                if stats["total"] % 500 == 0:
                    print(f"  {stats['total']} gambar diproses ...")
    finally:
        detector.close()

    print(f"Selesai: {stats}\nLaporan pra-anotasi: {csv_path}")
    print("Gambar tanpa wajah TIDAK diberi label: anotasi manual atau hapus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
