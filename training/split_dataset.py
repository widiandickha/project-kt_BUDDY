#!/usr/bin/env python3
"""FASE 2: build data/dataset (YOLO layout) from annotated raw images, split per person.

Usage:
    python training/split_dataset.py                          # from dataset.raw_dir
    python training/split_dataset.py --src data/raw data/roboflow_export
    python training/split_dataset.py --assign nthu-006=train --assign self-andi=test

Input: images whose name follows the raw scheme (training/common.py) with a YOLO label either
beside the image or in a sibling labels/ folder (Roboflow/CVAT export). Images without a label
file are skipped (not annotated yet); an empty label file is a negative sample.

Output: data/dataset/{images,labels}/{train,val,test}, data/data.yaml, docs/dataset_report.md.
The output folder is regenerated from scratch on every run.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.core.config import ROOT_DIR, load_config, resolve_path  # noqa: E402
from src.detection.classes import CLASS_NAMES  # noqa: E402
from training.common import iter_images, label_path_for, parse_raw_name, read_labels  # noqa: E402

SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass
class Item:
    """One annotated image."""

    image: Path
    label: Path
    subject: str
    condition: str
    classes: list[int]


@dataclass
class SplitStats:
    """Counts for one split."""

    subjects: set[str] = field(default_factory=set)
    images: int = 0
    negatives: int = 0
    instances: Counter[int] = field(default_factory=Counter)


def collect_items(sources: list[Path]) -> tuple[list[Item], list[Path], list[str]]:
    """Find annotated images. Returns (items, unlabeled images, errors)."""
    items: list[Item] = []
    unlabeled: list[Path] = []
    errors: list[str] = []
    seen: set[str] = set()
    for src in sources:
        for img in iter_images(src):
            try:
                meta = parse_raw_name(img)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            label = label_path_for(img)
            if not label.exists():
                unlabeled.append(img)
                continue
            key = f"{meta.subject}/{meta.hint}/{meta.condition}/{meta.frame_id}"
            if key in seen:
                errors.append(f"Duplikat frame {key}: {img}")
                continue
            seen.add(key)
            try:
                boxes = read_labels(label, len(CLASS_NAMES))
            except ValueError as exc:
                errors.append(str(exc))
                continue
            items.append(Item(img, label, meta.subject, meta.condition, [b.cls for b in boxes]))
    return items, unlabeled, errors


def assign_subjects(
    counts: dict[str, int], ratios: dict[str, float], seed: int, fixed: dict[str, str] | None = None
) -> dict[str, str]:
    """Assign whole subjects to splits so image counts approach the ratios.

    With >= 3 subjects, val and test each get at least one subject (with 2, val does).
    `fixed` pins subjects.
    """
    fixed = dict(fixed or {})
    unknown = [s for s, sp in fixed.items() if sp not in SPLITS]
    if unknown:
        raise ValueError(f"Invalid split for {unknown}; use one of {SPLITS}")
    total = sum(counts.values())
    target = {sp: ratios[sp] * total for sp in SPLITS}
    current = {sp: 0.0 for sp in SPLITS}
    result: dict[str, str] = {}
    for subj, sp in fixed.items():
        if subj in counts:
            result[subj] = sp
            current[sp] += counts[subj]

    rng = random.Random(seed)
    free = sorted(s for s in counts if s not in result)
    rng.shuffle(free)
    free.sort(key=lambda s: -counts[s])  # stable: ties keep the seeded order

    reserved = ("test", "val") if len(counts) >= 3 else ("val",) if len(counts) == 2 else ()
    if reserved:
        for sp in reserved:
            if current[sp] == 0 and free:
                pick = min(free, key=lambda s: abs(counts[s] - target[sp]))
                free.remove(pick)
                result[pick] = sp
                current[sp] += counts[pick]

    for subj in free:
        sp = max(SPLITS, key=lambda k: (target[k] - current[k], k == "train"))
        result[subj] = sp
        current[sp] += counts[subj]
    return result


def place(src: Path, dst: Path) -> None:
    """Hard-link (same disk, no extra space) or copy a file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


GRAY_TOLERANCE = 2  # JPEG chroma noise on an already-gray image


def place_grayscale(src: Path, dst_dir: Path) -> None:
    """Place a 3-channel grayscale version of src in dst_dir (detector input is grayscale, D-013).

    Already-gray images are linked as-is; color images are converted and written as JPEG.
    """
    import cv2
    import numpy as np

    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"Gambar tidak terbaca: {src}")
    signed = img.astype(np.int16)
    spread = int((signed.max(axis=2) - signed.min(axis=2)).max())
    if spread <= GRAY_TOLERANCE:
        place(src, dst_dir / f"{src.stem}{src.suffix.lower()}")
        return
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    dst_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_dir / f"{src.stem}.jpg"), gray, [cv2.IMWRITE_JPEG_QUALITY, 95])


def build_dataset(
    items: list[Item], assignment: dict[str, str], out_dir: Path, grayscale: bool = False
) -> dict[str, SplitStats]:
    """Write the YOLO folder layout and return per-split stats."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    stats = {sp: SplitStats() for sp in SPLITS}
    for it in items:
        sp = assignment[it.subject]
        stem = it.image.stem
        if grayscale:
            place_grayscale(it.image, out_dir / "images" / sp)
        else:
            place(it.image, out_dir / "images" / sp / f"{stem}{it.image.suffix.lower()}")
        place(it.label, out_dir / "labels" / sp / f"{stem}.txt")
        st = stats[sp]
        st.subjects.add(it.subject)
        st.images += 1
        st.negatives += int(not it.classes)
        st.instances.update(it.classes)
    return stats


def write_data_yaml(path: Path, out_dir: Path) -> None:
    """data.yaml in the spec 2.1 format (path relative to the project root)."""
    try:
        rel = out_dir.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        rel = out_dir.as_posix()
    doc = {
        "path": rel,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")


def build_report(
    items: list[Item],
    assignment: dict[str, str],
    stats: dict[str, SplitStats],
    unlabeled: list[Path],
    target_per_class: int,
    min_subjects: int,
) -> str:
    """Markdown dataset report (docs/dataset_report.md)."""
    total = Counter[int]()
    for st in stats.values():
        total.update(st.instances)
    lines = [
        "# Laporan Dataset",
        "",
        f"Dibuat otomatis oleh `training/split_dataset.py` pada {datetime.now():%Y-%m-%d %H:%M}.",
        "",
        "## Ringkasan per split",
        "",
        "| Split | Subjek | Gambar | Negatif | " + " | ".join(CLASS_NAMES) + " |",
        "|---|---|---|---|" + "---|" * len(CLASS_NAMES),
    ]
    for sp in SPLITS:
        st = stats[sp]
        cells = " | ".join(str(st.instances[i]) for i in range(len(CLASS_NAMES)))
        lines.append(f"| {sp} | {', '.join(sorted(st.subjects)) or '-'} | {st.images} | {st.negatives} | {cells} |")
    all_imgs = sum(st.images for st in stats.values())
    lines.append(
        f"| **total** | {len(assignment)} | {all_imgs} | {sum(st.negatives for st in stats.values())} | "
        + " | ".join(f"**{total[i]}**" for i in range(len(CLASS_NAMES)))
        + " |"
    )

    lines += ["", "## Per subjek", "", "| Subjek | Split | Gambar | Kondisi | " + " | ".join(CLASS_NAMES) + " |",
              "|---|---|---|---|" + "---|" * len(CLASS_NAMES)]
    per_subj: dict[str, Counter[int]] = defaultdict(Counter)
    per_subj_imgs: Counter[str] = Counter()
    per_subj_cond: dict[str, Counter[str]] = defaultdict(Counter)
    for it in items:
        per_subj[it.subject].update(it.classes)
        per_subj_imgs[it.subject] += 1
        per_subj_cond[it.subject][it.condition] += 1
    for subj in sorted(per_subj_imgs):
        cond = ", ".join(f"{c} {n}" for c, n in per_subj_cond[subj].most_common())
        cells = " | ".join(str(per_subj[subj][i]) for i in range(len(CLASS_NAMES)))
        lines.append(f"| {subj} | {assignment[subj]} | {per_subj_imgs[subj]} | {cond} | {cells} |")

    warnings: list[str] = []
    if len(assignment) < min_subjects:
        warnings.append(f"Baru {len(assignment)} subjek; spesifikasi meminta minimal {min_subjects} orang berbeda.")
    for i, name in enumerate(CLASS_NAMES):
        if total[i] < target_per_class:
            warnings.append(f"`{name}`: {total[i]} instance, di bawah target {target_per_class}.")
        for sp in ("val", "test"):
            if stats[sp].images and stats[sp].instances[i] == 0:
                warnings.append(f"`{name}` tidak ada di split {sp}; metrik kelas ini tidak bisa diukur di sana.")
    if unlabeled:
        warnings.append(f"{len(unlabeled)} gambar belum punya file label dan dilewati.")
    lines += ["", "## Peringatan", ""] + ([f"- {w}" for w in warnings] or ["- Tidak ada."])
    lines += ["", "Split dilakukan per subjek (orang), sehingga tidak ada orang yang muncul di dua split.", ""]
    return "\n".join(lines)


def parse_assign(values: list[str]) -> dict[str, str]:
    """Parse --assign subject=split pairs."""
    out: dict[str, str] = {}
    for v in values:
        subj, _, sp = v.partition("=")
        if not subj or sp not in SPLITS:
            raise SystemExit(f"--assign harus berbentuk subjek=train|val|test, bukan '{v}'")
        out[subj.strip()] = sp
    return out


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", type=Path, default=None, help="folder gambar beranotasi (default dataset.raw_dir)")
    ap.add_argument("--assign", action="append", default=[], help="paksa subjek ke split, mis. nthu-006=train")
    args = ap.parse_args(argv)

    cfg = load_config()
    ds = cfg["dataset"]
    sources = [resolve_path(p) for p in (args.src or [ds["raw_dir"]])]
    items, unlabeled, errors = collect_items(sources)
    if errors:
        print("Label/nama file bermasalah, perbaiki dulu:")
        for e in errors[:50]:
            print("  -", e)
        return 1
    if not items:
        print("Tidak ada gambar beranotasi.")
        return 1

    counts = Counter(it.subject for it in items)
    assignment = assign_subjects(dict(counts), ds["split"], int(ds["split"]["seed"]), parse_assign(args.assign))
    out_dir = resolve_path(ds["out_dir"])
    stats = build_dataset(items, assignment, out_dir, grayscale=bool(cfg["det"].get("grayscale", False)))
    write_data_yaml(resolve_path(ds["data_yaml"]), out_dir)
    report = build_report(items, assignment, stats, unlabeled, int(ds["target_per_class"]), int(ds["min_subjects"]))
    report_path = resolve_path(ds["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    for sp in SPLITS:
        st = stats[sp]
        print(f"{sp:5s}: {st.images:5d} gambar, subjek {sorted(st.subjects)}")
    print(f"data.yaml -> {resolve_path(ds['data_yaml'])}\nlaporan   -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
