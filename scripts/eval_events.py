#!/usr/bin/env python3
"""FASE 8.2: event-level precision/recall (yawn, long eye closure, hand to forehead).

Usage:
    python scripts/eval_events.py --features rek.jsonl --labels label.csv [--tolerance 2] [--name sesi1]

* features: FrameFeatures recorded with `python -m src.main --record-features` or
  scripts/scenario_session.py (numbers only, no images).
* labels: CSV with columns `type,start_sec` (seconds from the first frame), types:
  yawn | eye_closed_long | hand_to_forehead. scenario_session.py writes this file for you.

System events are detected with the same rules as the app (fused eye state, PERCLOS closure
>= ear.microsleep_ms, yawn >= window.yawn_min_sec with debounce, hand-to-forehead episodes).
A detection matches a label of the same type when their start times differ <= tolerance.
Results are appended to docs/eval/events.md and docs/eval/events.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.calibration import Calibration  # noqa: E402
from src.analysis.replay import read_features  # noqa: E402
from src.analysis.sliding_window import ACTIVE, eye_closed  # noqa: E402
from src.core.config import load_config, resolve_path  # noqa: E402
from src.core.events import FrameFeatures  # noqa: E402

EVENT_TYPES: tuple[str, ...] = ("yawn", "eye_closed_long", "hand_to_forehead")
DEFAULT_TOLERANCE = 2.0


@dataclass
class TypeResult:
    """Matching result of one event type."""

    type: str
    labels: int
    detected: int
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None


def _runs(times: list[float], flags: list[bool]) -> list[tuple[float, float]]:
    """(start, duration) of consecutive True runs; a run ends at the next sample's time."""
    out, start = [], None
    for i, (t, on) in enumerate(zip(times, flags, strict=True)):
        if on and start is None:
            start = t
        if not on and start is not None:
            out.append((start, t - start))
            start = None
        if i == len(times) - 1 and start is not None:
            out.append((start, t - start))
    return out


def detect_events(features: list[FrameFeatures], cfg: dict[str, Any], ear_threshold: float) -> dict[str, list[float]]:
    """Start times (s from the first frame) of system-detected events."""
    if not features:
        return {t: [] for t in EVENT_TYPES}
    t0 = features[0].ts
    times = [f.ts - t0 for f in features]
    closed = [bool(eye_closed(f, ear_threshold, cfg)) if f.user_present else False for f in features]
    yawn = [f.user_present and f.yawn > ACTIVE for f in features]
    hand = [f.user_present and f.hand_to_forehead for f in features]
    micro = float(cfg["ear"]["microsleep_ms"]) / 1000.0
    yawn_min = float(cfg["window"]["yawn_min_sec"])
    debounce = float(cfg["window"]["yawn_debounce_sec"])

    yawns, last = [], float("-inf")
    for start, dur in _runs(times, yawn):
        if dur >= yawn_min and start - last >= debounce:
            yawns.append(start)
            last = start
    return {
        "yawn": yawns,
        "eye_closed_long": [s for s, d in _runs(times, closed) if d >= micro],
        "hand_to_forehead": [s for s, _ in _runs(times, hand)],
    }


def match(detected: list[float], labels: list[float], tolerance: float) -> tuple[int, int, int]:
    """Greedy one-to-one matching by time; returns (tp, fp, fn)."""
    used = [False] * len(labels)
    tp = 0
    for d in sorted(detected):
        best, best_dt = None, tolerance
        for i, lab in enumerate(labels):
            dt = abs(d - lab)
            if not used[i] and dt <= best_dt:
                best, best_dt = i, dt
        if best is not None:
            used[best] = True
            tp += 1
    return tp, len(detected) - tp, len(labels) - tp


def _ratio(a: int, b: int) -> float | None:
    return None if b == 0 else a / b


def evaluate(detected: dict[str, list[float]], labels: dict[str, list[float]], tolerance: float) -> list[TypeResult]:
    """Per-type precision/recall/F1."""
    out = []
    for t in EVENT_TYPES:
        tp, fp, fn = match(detected.get(t, []), labels.get(t, []), tolerance)
        p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        f1 = None if p is None or r is None or p + r == 0 else 2 * p * r / (p + r)
        out.append(TypeResult(t, len(labels.get(t, [])), len(detected.get(t, [])), tp, fp, fn, p, r, f1))
    return out


def read_labels(path: Path) -> dict[str, list[float]]:
    """Label CSV -> {type: [start_sec, ...]}."""
    out: dict[str, list[float]] = {t: [] for t in EVENT_TYPES}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            t = row["type"].strip()
            if t not in out:
                raise ValueError(f"Tipe label tidak dikenal: {t} (pakai {', '.join(EVENT_TYPES)})")
            out[t].append(float(row["start_sec"]))
    return out


def _fmt(v: float | None) -> str:
    return "–" if v is None else f"{v:.2f}"


def write_report(results: list[TypeResult], name: str, tolerance: float, eval_dir: Path) -> None:
    """Append a section to events.md and the raw numbers to events.json."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    md = eval_dir / "events.md"
    if not md.exists():
        md.write_text("# Akurasi Sistem Level Event (FASE 8.2)\n\nToleransi waktu ±2 detik kecuali disebut lain.\n", encoding="utf-8")
    lines = [f"\n## {name} ({datetime.now():%Y-%m-%d %H:%M}, toleransi ±{tolerance:g} s)\n",
             "| Event | Label | Terdeteksi | TP | FP | FN | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.type} | {r.labels} | {r.detected} | {r.tp} | {r.fp} | {r.fn} | {_fmt(r.precision)} | {_fmt(r.recall)} | {_fmt(r.f1)} |")
    with md.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    js = eval_dir / "events.json"
    data = json.loads(js.read_text(encoding="utf-8")) if js.exists() else []
    data.append({"name": name, "tolerance": tolerance, "at": datetime.now().isoformat(timespec="seconds"),
                 "results": [asdict(r) for r in results]})
    js.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    ap.add_argument("--name", default=None, help="nama sesi di laporan (default nama file fitur)")
    ap.add_argument("--ear-threshold", type=float, default=None, help="default: kalibrasi tersimpan di DB, lalu config")
    args = ap.parse_args(argv)

    cfg = load_config()
    thr = args.ear_threshold
    if thr is None:
        from src.storage.db import Database
        from src.storage.repository import Repository

        db_path = resolve_path(cfg["storage"]["db_path"])
        cal = Calibration.from_rows(Repository(Database(db_path)).get_calibration(), cfg) if db_path.exists() else None
        thr = (cal or Calibration.default(cfg)).ear_threshold
    features = list(read_features(args.features))
    results = evaluate(detect_events(features, cfg, thr), read_labels(args.labels), args.tolerance)
    for r in results:
        print(f"{r.type:17s} label={r.labels:3d} deteksi={r.detected:3d} P={_fmt(r.precision)} R={_fmt(r.recall)} F1={_fmt(r.f1)}")
    write_report(results, args.name or args.features.stem, args.tolerance, resolve_path(cfg["train"]["eval_dir"]))
    print("Laporan: docs/eval/events.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
