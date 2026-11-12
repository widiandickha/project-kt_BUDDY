#!/usr/bin/env python3
"""FASE 2: copy a sample of the public datasets into data/raw/ using the raw naming scheme.

Usage:
    python scripts/import_public.py nthu --src "~/Downloads/KAGGLE/3. NTHU DDD"
    python scripts/import_public.py uta  --src "/Volumes/Data/uta-reallife-drowsiness-dataset"

NTHU-DDD multi-class (Kaggle samymesbah): consecutive 640x480 video frames named
``<subject>_<glasses|noglasses>_<scenario>_<frame>_<drowsy|notdrowsy>.jpg``. Frames are taken
evenly per group (subject x glasses x folder x scenario) to avoid near-duplicates.

UTA-RLDD (Kaggle rishab260): one video per participant and state (0 = alert,
5 = low vigilant, 10 = drowsy). A frame is extracted every ``dataset.import.uta_every_sec``.

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


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=("nthu", "uta"))
    ap.add_argument("--src", required=True, type=lambda s: Path(s).expanduser(), help="folder dataset hasil unduh Kaggle")
    args = ap.parse_args(argv)

    cfg = load_config()
    ds = cfg["dataset"]
    raw_dir = resolve_path(ds["raw_dir"])
    if args.dataset == "nthu":
        imp = ds["import"]
        counts = import_nthu(
            args.src, raw_dir, imp["nthu_per_group"], int(imp["overlay_color_diff"]), int(imp["overlay_dilate_px"])
        )
    else:
        counts = import_uta(
            args.src, raw_dir, float(ds["import"]["uta_every_sec"]),
            int(ds["import"]["uta_max_per_video"]), int(cfg["camera"]["width"]),
        )
    print(f"Selesai ({args.dataset}) -> {raw_dir}: {dict(counts)} (total {sum(counts.values())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
