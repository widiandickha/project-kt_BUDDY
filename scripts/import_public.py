#!/usr/bin/env python3
"""FASE 2: copy a sample of the public datasets into data/raw/ using the raw naming scheme.

Usage:
    python scripts/import_public.py nthu --src "~/Downloads/KAGGLE/3. NTHU DDD"
    python scripts/import_public.py uta  --src "/Volumes/Data/uta-reallife-drowsiness-dataset"
    python scripts/import_public.py folder                          # data/incoming/kerutan_dahi/<orang>/...
    python scripts/import_public.py folder --kelas menguap --src ~/Desktop/menguap --kondisi redup
    python scripts/import_public.py roboflow --preset driver-yawn --src ~/Downloads/driver-yawn-yolov8
    python scripts/import_public.py roboflow --preset wrinkles    --src ~/Downloads/wrinkles-yolov8

Folder (your own photos/videos, e.g. kerutan_dahi which no public dataset has): put one sub-folder
per person, ``<src>/<kode_orang>/*.jpg|png|mp4``, so the split stays per person. Images are copied
(resized to the camera width) and videos are sampled every ``dataset.import.uta_every_sec``; the
class becomes the file-name hint that scripts/preannotate.py uses for the draft box.

NTHU-DDD multi-class (Kaggle samymesbah): consecutive 640x480 video frames named
``<subject>_<glasses|noglasses>_<scenario>_<frame>_<drowsy|notdrowsy>.jpg``. Frames are taken
evenly per group (subject x glasses x folder x scenario) to avoid near-duplicates.

UTA-RLDD (Kaggle rishab260): one video per participant and state (0 = alert,
5 = low vigilant, 10 = drowsy). A frame is extracted every ``dataset.import.uta_every_sec``.

Roboflow (a YOLOv8 export: data.yaml + */images + */labels, boxes or segmentation polygons): the
source labels only choose each image's hint (e.g. any ``yawn`` box -> ``menguap``); boxes are NOT
copied, scripts/preannotate.py redraws them with our conventions (D-009). Roboflow file names carry
no person id, so the whole export becomes one subject ``rf-<name>`` (keep it in train with
``split_dataset.py --assign rf-<name>=train``) unless ``--subjek-regex`` can read one from the name.

MRL Eye is not imported: it holds single-eye infrared crops, not webcam faces (D-007).
Labels are NOT created here; run scripts/preannotate.py next.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.core.config import load_config, resolve_path  # noqa: E402
from src.detection.classes import CLASS_NAMES  # noqa: E402
from training.common import make_raw_name, slug  # noqa: E402

NTHU_FOLDER_HINT: dict[str, str] = {
    "notdrowsy": "normal",
    "sleepyCombination": "mengantuk",
    "slowBlinkWithNodding": "kedip_lambat",
    "yawning": "menguap",
}
NTHU_RE = re.compile(r"^(?P<subject>\d+)_(?P<glasses>[A-Za-z]+)_(?P<scenario>[A-Za-z]+)_(?P<frame>\d+)_(?:not)?drowsy$")

UTA_STATE_HINT: dict[str, str] = {"0": "alert", "5": "low_vigilant", "10": "drowsy"}
VIDEO_EXTS: frozenset[str] = frozenset({".mov", ".mp4", ".avi", ".m4v", ".mkv"})


@dataclass(frozen=True)
class NthuFrame:
    """One parsed NTHU-DDD frame."""

    path: Path
    subject: str
    glasses: str
    scenario: str
    frame: int
    hint: str


def parse_nthu(path: Path) -> NthuFrame | None:
    """Parse an NTHU file; None if the name or folder is not recognized."""
    m = NTHU_RE.match(path.stem)
    hint = NTHU_FOLDER_HINT.get(path.parent.name)
    if m is None or hint is None:
        return None
    return NthuFrame(path, m["subject"], m["glasses"].lower(), m["scenario"], int(m["frame"]), hint)


def select_evenly(items: list, n: int) -> list:
    """Pick up to n items spread evenly over the (sorted) list."""
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


def remove_color_overlay(img: np.ndarray, color_diff: int, dilate_px: int) -> tuple[np.ndarray, int]:
    """Inpaint colored pixels of an otherwise grayscale frame (NTHU has boxes burned in, D-010).

    Returns the cleaned image and the number of masked pixels.
    """
    signed = img.astype(np.int16)
    b, g, r = signed[..., 0], signed[..., 1], signed[..., 2]
    spread = np.maximum.reduce([np.abs(b - g), np.abs(g - r), np.abs(b - r)])
    mask = (spread > color_diff).astype(np.uint8) * 255
    n = int(np.count_nonzero(mask))
    if n == 0:
        return img, 0
    if dilate_px > 0:
        kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        mask = cv2.dilate(mask, kernel)
    cleaned = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), n


def import_nthu(
    src: Path, raw_dir: Path, per_group: dict[str, int], color_diff: int, dilate_px: int
) -> Counter[str]:
    """Copy an evenly spaced, overlay-cleaned sample of NTHU-DDD frames; returns counts per hint."""
    groups: dict[tuple[str, str, str, str], list[NthuFrame]] = defaultdict(list)
    for p in src.rglob("*.jpg"):
        f = parse_nthu(p)
        if f is not None:
            groups[(f.subject, f.glasses, f.hint, f.scenario)].append(f)
    if not groups:
        raise SystemExit(f"Tidak ada frame NTHU yang dikenali di {src} (cari 'Multi class/train/...').")

    counts: Counter[str] = Counter()
    for (subject, glasses, hint, scenario), frames in sorted(groups.items()):
        frames.sort(key=lambda fr: fr.frame)
        for fr in select_evenly(frames, int(per_group.get(hint, 0))):
            subj = f"nthu-{subject}"
            name = make_raw_name(subj, hint, glasses, f"{scenario}-{fr.frame}")
            dst = raw_dir / subj / hint / name
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                img = cv2.imread(str(fr.path))
                if img is None:
                    continue
                cleaned, _ = remove_color_overlay(img, color_diff, dilate_px)
                cv2.imwrite(str(dst), cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
            counts[hint] += 1
    return counts


def uta_participant(video: Path) -> str:
    """Participant id from the nearest all-digit parent folder (fallback: parent name)."""
    for parent in video.parents:
        if parent.name.isdigit():
            return f"uta-{int(parent.name)}"
    return f"uta-{slug(video.parent.name)}"


def uta_hint(video: Path) -> str:
    """State hint from the video file stem (0/5/10)."""
    stem = video.stem.strip()
    return UTA_STATE_HINT.get(stem, re.sub(r"[^a-z0-9_]+", "_", stem.lower()).strip("_") or "unknown")


def extract_frames(video: Path, every_sec: float, max_frames: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, frame) every `every_sec` seconds, at most `max_frames` frames."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * every_sec)))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        emitted = 0
        pos = 0
        while emitted < max_frames and (total <= 0 or pos < total):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield pos, frame
            emitted += 1
            pos += step
    finally:
        cap.release()


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    """Downscale to the runtime camera width so training matches inference."""
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def import_uta(src: Path, raw_dir: Path, every_sec: float, max_frames: int, width: int) -> Counter[str]:
    """Extract sampled frames from every UTA-RLDD video under src."""
    videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("."))
    if not videos:
        raise SystemExit(f"Tidak ada video di {src}.")
    counts: Counter[str] = Counter()
    for video in videos:
        subj, hint = uta_participant(video), uta_hint(video)
        for pos, frame in extract_frames(video, every_sec, max_frames):
            dst = raw_dir / subj / hint / make_raw_name(subj, hint, "unknown", f"{hint}-{pos:06d}")
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), resize_to_width(frame, width))
            counts[hint] += 1
        print(f"  {video.relative_to(src)} -> {subj}/{hint}")
    return counts


IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
INCOMING_DIR = "data/incoming"


def folder_subject(path: Path, src: Path, fallback: str) -> str:
    """Person code = first sub-folder under src (files directly in src use `fallback`)."""
    rel = path.relative_to(src)
    return f"ext-{slug(rel.parts[0])}" if len(rel.parts) > 1 else f"ext-{slug(fallback)}"


def import_folder(
    src: Path, raw_dir: Path, kelas: str, condition: str, fallback_subject: str,
    every_sec: float, max_frames: int, width: int,
) -> Counter[str]:
    """Copy own images/videos of one class into data/raw/; returns counts per subject."""
    files = sorted(p for p in src.rglob("*") if p.is_file() and not p.name.startswith(".")
                   and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
    if not files:
        raise SystemExit(f"Tidak ada gambar/video di {src}. Taruh file di {src}/<kode_orang>/.")
    counts: Counter[str] = Counter()
    for i, f in enumerate(files):
        subj = folder_subject(f, src, fallback_subject)
        out_dir = raw_dir / subj / kelas
        stem = slug(f.stem)[:40]
        if f.suffix.lower() in VIDEO_EXTS:
            frames = ((f"{stem}-{pos:06d}", frame) for pos, frame in extract_frames(f, every_sec, max_frames))
        else:
            img = cv2.imread(str(f))
            if img is None:
                print(f"  lewati (tidak terbaca): {f.relative_to(src)}")
                continue
            frames = iter([(f"{stem}-{i:05d}", img)])
        for frame_id, frame in frames:
            dst = out_dir / make_raw_name(subj, kelas, condition, frame_id)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), resize_to_width(frame, width), [cv2.IMWRITE_JPEG_QUALITY, 95])
            counts[subj] += 1
    return counts


ROBOFLOW_PRESETS: dict[str, dict] = {
    # Universidad Carlos III Madrid "driver-yawn" (YawDD frames): yawn / eyes per side.
    "driver-yawn": {"map": {"yawn": "menguap", "close_eyel": "mata_tertutup", "close_eyer": "mata_tertutup"},
                    "default": "normal", "max_per_hint": 400},
    # "roboflow one wrinkles detection": frown = vertical lines between the brows.
    "wrinkles": {"map": {"frown": "kerutan_dahi"}, "default": "normal", "max_per_hint": 300},
}
RF_SUFFIX = re.compile(r"(_(jpe?g|png|bmp|webp))?\.rf\.[0-9a-f]+$", re.IGNORECASE)


def roboflow_original_stem(stem: str) -> str:
    """'12-MaleGlasses_jpg.rf.3f2a...' -> '12-MaleGlasses'."""
    return RF_SUFFIX.sub("", stem)


def roboflow_class_names(src: Path) -> list[str]:
    """Class names (lowercase) from the export's data.yaml."""
    import yaml

    found = sorted(src.rglob("data.yaml"), key=lambda p: len(p.parts))
    if not found:
        raise SystemExit(f"data.yaml tidak ditemukan di {src}. Export dari Roboflow dengan format YOLOv8.")
    names = yaml.safe_load(found[0].read_text(encoding="utf-8")).get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return [str(n).strip().lower() for n in names]


def roboflow_label_classes(label: Path, names: list[str]) -> set[str]:
    """Class names present in a YOLO label file (box or polygon lines)."""
    if not label.exists():
        return set()
    out = set()
    for line in label.read_text(encoding="utf-8").splitlines():
        tok = line.split()
        if tok and tok[0].isdigit() and int(tok[0]) < len(names):
            out.add(names[int(tok[0])])
    return out


def import_roboflow(
    src: Path, raw_dir: Path, name: str, mapping: dict[str, str], default_hint: str | None,
    max_per_hint: int | None, subject_regex: str | None, width: int,
) -> Counter[str]:
    """Copy a Roboflow YOLO export into data/raw/, one hint per image from its source labels."""
    names = roboflow_class_names(src)
    unknown = sorted(set(mapping) - set(names))
    if unknown:
        raise SystemExit(f"Kelas {unknown} tidak ada di data.yaml ({names}). Cek --map.")
    pattern = re.compile(subject_regex) if subject_regex else None
    images = sorted(p for p in src.rglob("*") if p.parent.name == "images" and p.suffix.lower() in IMAGE_EXTS)
    by_hint: dict[str, list[Path]] = defaultdict(list)
    for img in images:
        present = roboflow_label_classes(img.parent.parent / "labels" / f"{img.stem}.txt", names)
        hint = next((h for c, h in mapping.items() if c in present), default_hint)
        if hint is not None:
            by_hint[hint].append(img)
    counts: Counter[str] = Counter()
    for hint, files in sorted(by_hint.items()):
        for img in select_evenly(files, max_per_hint or len(files)):
            orig = roboflow_original_stem(img.stem)
            m = pattern.search(orig) if pattern else None
            subj = f"rf-{slug(name)}" + (f"-{slug(m.group(1))}" if m else "")
            dst = raw_dir / subj / hint / make_raw_name(subj, hint, "unknown", slug(orig)[:60] or "img")
            if dst.exists():
                counts[hint] += 1
                continue
            frame = cv2.imread(str(img))
            if frame is None:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), resize_to_width(frame, width), [cv2.IMWRITE_JPEG_QUALITY, 95])
            counts[hint] += 1
    return counts


def _parse_map(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        src_cls, _, hint = item.partition("=")
        if not hint or not re.fullmatch(r"[a-z0-9_]+", hint):
            raise SystemExit(f"--map '{item}' harus berbentuk kelas_sumber=petunjuk, mis. yawn=menguap")
        out[src_cls.strip().lower()] = hint
    return out


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=("nthu", "uta", "folder", "roboflow"))
    ap.add_argument("--src", type=lambda s: Path(s).expanduser(),
                    help="folder dataset (Kaggle); untuk 'folder' default data/incoming/<kelas>")
    ap.add_argument("--kelas", default="kerutan_dahi", choices=CLASS_NAMES, help="folder: kelas isi folder")
    ap.add_argument("--kondisi", default="unknown", help="folder: tag kondisi (terang, redup, kacamata, ...)")
    ap.add_argument("--subjek", default="lain", help="folder: kode orang untuk file yang tidak di sub-folder")
    ap.add_argument("--preset", choices=sorted(ROBOFLOW_PRESETS), help="roboflow: pemetaan kelas siap pakai")
    ap.add_argument("--nama", help="roboflow: nama sumber -> subjek rf-<nama> (default nama preset)")
    ap.add_argument("--map", action="append", default=[], help="roboflow: kelas_sumber=petunjuk (boleh berulang)")
    ap.add_argument("--default-hint", help="roboflow: petunjuk untuk gambar tanpa kelas terpetakan ('-' = lewati)")
    ap.add_argument("--max-per-hint", type=int, help="roboflow: batas gambar per petunjuk (diambil merata)")
    ap.add_argument("--subjek-regex", help="roboflow: regex dengan 1 grup untuk membaca kode orang dari nama file")
    args = ap.parse_args(argv)
    if args.src is None:
        if args.dataset != "folder":
            ap.error("--src wajib untuk nthu/uta/roboflow")
        args.src = resolve_path(f"{INCOMING_DIR}/{args.kelas}")

    cfg = load_config()
    ds = cfg["dataset"]
    raw_dir = resolve_path(ds["raw_dir"])
    if args.dataset == "nthu":
        imp = ds["import"]
        counts = import_nthu(
            args.src, raw_dir, imp["nthu_per_group"], int(imp["overlay_color_diff"]), int(imp["overlay_dilate_px"])
        )
    elif args.dataset == "folder":
        counts = import_folder(
            args.src, raw_dir, args.kelas, args.kondisi, args.subjek, float(ds["import"]["uta_every_sec"]),
            int(ds["import"]["uta_max_per_video"]), int(cfg["camera"]["width"]),
        )
        print("Lanjut: python scripts/preannotate.py, lalu koreksi box di Roboflow/CVAT (docs/ANOTASI.md).")
    elif args.dataset == "roboflow":
        preset = ROBOFLOW_PRESETS.get(args.preset or "", {})
        name = args.nama or args.preset
        mapping = _parse_map(args.map) if args.map else dict(preset.get("map", {}))
        if not name or not mapping:
            ap.error("roboflow butuh --preset, atau --nama dan minimal satu --map")
        default = args.default_hint if args.default_hint is not None else preset.get("default")
        counts = import_roboflow(
            args.src, raw_dir, name, mapping, None if default in (None, "-") else default,
            args.max_per_hint or preset.get("max_per_hint"), args.subjek_regex, int(cfg["camera"]["width"]),
        )
        print(f"Lanjut: preannotate.py, lalu split_dataset.py --assign rf-{slug(name)}=train (subjek tak dikenal).")
    else:
        counts = import_uta(
            args.src, raw_dir, float(ds["import"]["uta_every_sec"]),
            int(ds["import"]["uta_max_per_video"]), int(cfg["camera"]["width"]),
        )
    print(f"Selesai ({args.dataset}) -> {raw_dir}: {dict(counts)} (total {sum(counts.values())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
