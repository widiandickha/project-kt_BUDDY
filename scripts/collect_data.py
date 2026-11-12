#!/usr/bin/env python3
"""FASE 2 data collection: save webcam frames of a consenting volunteer into data/raw/.

Usage:
    python scripts/collect_data.py --relawan andi --kondisi terang
    python scripts/collect_data.py --relawan andi --kondisi redup-kacamata --camera 1

Keys in the preview window:
    1  pejamkan mata        -> mata_tertutup
    2  buka mata normal     -> mata_terbuka
    3  menguap              -> menguap
    4  kerutkan dahi        -> kerutan_dahi
    SPASI  simpan 1 frame untuk instruksi aktif
    B      mode burst on/off (simpan otomatis tiap dataset.collect.burst_interval_sec)
    Q/ESC  selesai

Files: data/raw/self-<relawan>/<kelas>/<subject>__<kelas>__<kondisi>__<id>.jpg (training/common.py).
Only the clean frame is saved; the on-screen overlay is never written. This tool is the one
place where frames go to disk, and only after the volunteer's written consent.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
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

WINDOW = "collect_data"
KEY_ESC = 27
KEY_SPACE = 32


@dataclass(frozen=True)
class Instruction:
    """A pose the volunteer is asked to perform."""

    key: str
    cls: str
    text: str


INSTRUCTIONS: tuple[Instruction, ...] = (
    Instruction("1", "mata_tertutup", "Pejamkan mata"),
    Instruction("2", "mata_terbuka", "Buka mata normal, lihat layar"),
    Instruction("3", "menguap", "Menguap / buka mulut lebar"),
    Instruction("4", "kerutan_dahi", "Kerutkan dahi"),
)
assert {i.cls for i in INSTRUCTIONS} == set(CLASS_NAMES)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relawan", required=True, help="nama/kode relawan (jadi ID subjek: self-<relawan>)")
    ap.add_argument("--kondisi", default="normal", help="tag kondisi, mis. terang, redup, backlight, kacamata, jauh")
    ap.add_argument("--camera", type=int, default=None, help="indeks kamera (default camera.index di config)")
    ap.add_argument("--consent", action="store_true", help="lewati prompt: relawan sudah menandatangani persetujuan")
    return ap.parse_args(argv)


def confirm_consent(relawan: str) -> bool:
    """Ask the operator to confirm written consent before any frame is saved."""
    answer = input(f"Relawan '{relawan}' sudah menandatangani persetujuan tertulis? ketik 'ya': ")
    return answer.strip().lower() == "ya"


def subject_id(relawan: str) -> str:
    """Subject id for a self-recorded volunteer."""
    return f"self-{slug(relawan)}"


def output_path(raw_dir: Path, subject: str, cls: str, condition: str, frame_id: str) -> Path:
    """Where a captured frame is stored."""
    return raw_dir / subject / cls / make_raw_name(subject, cls, condition, frame_id)


def count_existing(raw_dir: Path, subject: str) -> Counter[str]:
    """Images already collected per class for this subject."""
    counts: Counter[str] = Counter()
    for inst in INSTRUCTIONS:
        folder = raw_dir / subject / inst.cls
        if folder.exists():
            counts[inst.cls] = sum(1 for p in folder.iterdir() if p.suffix.lower() == ".jpg")
    return counts


def draw_hud(frame: np.ndarray, active: Instruction, counts: Counter[str], burst: bool, target: int) -> np.ndarray:
    """Return a copy of the frame with instructions and counters (never saved)."""
    view = frame.copy()
    cv2.rectangle(view, (0, 0), (view.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(view, f"[{active.key}] {active.text}" + ("  | BURST" if burst else ""), (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255) if burst else (255, 255, 255), 2)
    y = view.shape[0] - 10 - 22 * (len(INSTRUCTIONS) - 1)
    for inst in INSTRUCTIONS:
        color = (0, 220, 255) if inst is active else (240, 240, 240)
        cv2.putText(view, f"{inst.key} {inst.cls}: {counts[inst.cls]}/{target}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 22
    cv2.putText(view, "SPASI simpan | B burst | Q selesai", (view.shape[1] - 300, view.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
    return view


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    if not args.consent and not confirm_consent(args.relawan):
        print("Dibatalkan: tanpa persetujuan tertulis tidak ada frame yang disimpan.")
        return 1

    cfg = load_config()
    ds = cfg["dataset"]
    raw_dir = resolve_path(ds["raw_dir"])
    burst_interval = float(ds["collect"]["burst_interval_sec"])
    jpeg_quality = int(ds["collect"]["jpeg_quality"])
    target = int(ds["target_per_class"])
    subject = subject_id(args.relawan)
    condition = slug(args.kondisi)

    cam_index = args.camera if args.camera is not None else int(cfg["camera"]["index"])
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg["camera"]["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg["camera"]["height"]))
    if not cap.isOpened():
        print("Kamera tidak bisa dibuka. Cek izin kamera di System Settings > Privacy & Security > Camera.")
        return 1

    counts = count_existing(raw_dir, subject)
    active = INSTRUCTIONS[0]
    burst = False
    last_save = 0.0
    by_key = {ord(i.key): i for i in INSTRUCTIONS}
    print(f"Subjek: {subject} | kondisi: {condition} | simpan ke {raw_dir / subject}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)  # mirror, same as the app preview

            key = cv2.waitKey(1) & 0xFF
            now = time.monotonic()
            save = key == KEY_SPACE or (burst and now - last_save >= burst_interval)
            if key in (ord("q"), KEY_ESC):
                break
            if key in by_key:
                active = by_key[key]
            if key in (ord("b"), ord("B")):
                burst = not burst

            if save:
                frame_id = f"{int(time.time() * 1000)}"
                path = output_path(raw_dir, subject, active.cls, condition, frame_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                counts[active.cls] += 1
                last_save = now

            cv2.imshow(WINDOW, draw_hud(frame, active, counts, burst, target))
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Selesai. Jumlah per kelas untuk", subject, dict(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
