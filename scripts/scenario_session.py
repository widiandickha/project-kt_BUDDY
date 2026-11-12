#!/usr/bin/env python3
"""FASE 8.2: guided test session. Shows timed instructions on the webcam preview, records
FrameFeatures (numbers only) and writes the matching label CSV from the script timing.

Usage (only with the participant's written consent):
    python scripts/scenario_session.py --peserta p01 --out docs/eval/sesi
    python scripts/eval_events.py --features docs/eval/sesi/p01_features.jsonl --labels docs/eval/sesi/p01_labels.csv --name p01

The default script (~3.5 min) contains 3 yawns, 3 long eye closures, 3 hand-to-forehead
gestures, and talking/normal-blinking segments as negatives. Participants start each action
when the instruction appears; the ±2 s tolerance of eval_events.py absorbs reaction time.
No frame is written to disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.analysis.calibration import Calibration  # noqa: E402
from src.analysis.feature_extractor import FeatureExtractor  # noqa: E402
from src.analysis.replay import compact  # noqa: E402
from src.core.config import load_config, resolve_path  # noqa: E402


@dataclass(frozen=True)
class Step:
    """One scripted segment: instruction shown for `duration` s; `event` is labeled at its start."""

    duration: float
    text: str
    event: str | None = None


DEFAULT_SCRIPT: tuple[Step, ...] = (
    Step(20, "Duduk normal, lihat layar, berkedip biasa"),
    Step(10, "MENGUAP sekarang (lebar, 2-3 detik)", "yawn"),
    Step(12, "Normal"),
    Step(8, "PEJAMKAN MATA 3 detik", "eye_closed_long"),
    Step(12, "Normal"),
    Step(8, "Letakkan TANGAN DI DAHI 3 detik", "hand_to_forehead"),
    Step(15, "BICARA: baca kalimat ini keras-keras (bukan menguap)"),
    Step(10, "MENGUAP sekarang", "yawn"),
    Step(12, "Normal"),
    Step(8, "PEJAMKAN MATA 3 detik", "eye_closed_long"),
    Step(12, "Normal, berkedip cepat beberapa kali"),
    Step(8, "Letakkan TANGAN DI DAHI 3 detik", "hand_to_forehead"),
    Step(12, "Normal"),
    Step(10, "MENGUAP sekarang", "yawn"),
    Step(12, "Normal"),
    Step(8, "PEJAMKAN MATA 3 detik", "eye_closed_long"),
    Step(12, "Normal"),
    Step(8, "Letakkan TANGAN DI DAHI 3 detik", "hand_to_forehead"),
    Step(15, "Normal, selesai sebentar lagi"),
)


def script_labels(script: tuple[Step, ...]) -> list[tuple[str, float]]:
    """(event, start_sec) for every labeled step."""
    t, out = 0.0, []
    for s in script:
        if s.event:
            out.append((s.event, t))
        t += s.duration
    return out


def step_at(script: tuple[Step, ...], t: float) -> tuple[Step | None, float]:
    """Active step and seconds left in it (None when finished)."""
    acc = 0.0
    for s in script:
        if t < acc + s.duration:
            return s, acc + s.duration - t
        acc += s.duration
    return None, 0.0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    from src.detection.ear_calculator import EarCalculator
    from src.detection.pose_estimator import PoseEstimator

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peserta", required=True, help="kode peserta (tanpa nama asli)")
    ap.add_argument("--out", type=Path, default=Path("docs/eval/sesi"))
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--consent", action="store_true", help="lewati prompt persetujuan")
    args = ap.parse_args(argv)
    if not args.consent and input(f"Peserta {args.peserta} sudah menandatangani persetujuan? ketik 'ya': ").strip().lower() != "ya":
        print("Dibatalkan.")
        return 1

    cfg = load_config()
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path, label_path = out_dir / f"{args.peserta}_features.jsonl", out_dir / f"{args.peserta}_labels.csv"
    with label_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["type", "start_sec"])
        w.writerows(script_labels(DEFAULT_SCRIPT))

    ear, pose = EarCalculator(cfg), PoseEstimator(cfg)
    extractor = FeatureExtractor(cfg, Calibration.default(cfg))
    cam = cfg["camera"]
    index = args.camera if args.camera is not None else int(cam["index"])
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
    if not cap.isOpened():
        print("Kamera tidak bisa dibuka.")
        return 1
    start = time.time()
    with feat_path.open("w", encoding="utf-8") as fh:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                now = time.time()
                step, left = step_at(DEFAULT_SCRIPT, now - start)
                if step is None:
                    break
                f = extractor.extract(now, ear.process(frame, time.monotonic()), None, pose.process(frame))
                if f.ts - start < 0.05:
                    f.ts = start  # first frame anchors t = 0 for the labels
                fh.write(json.dumps(compact(f.to_dict()), separators=(",", ":")) + "\n")
                view = frame.copy()
                cv2.rectangle(view, (0, 0), (view.shape[1], 40), (20, 20, 20), -1)
                color = (0, 220, 255) if step.event else (255, 255, 255)
                cv2.putText(view, f"{step.text} ({left:4.1f} s)", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow("Sesi uji (q = batal)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Dibatalkan oleh operator.")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            ear.close()
    print(f"Fitur: {feat_path}\nLabel: {label_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
