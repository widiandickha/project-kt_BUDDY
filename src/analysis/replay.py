"""Replay recorded FrameFeatures (JSON lines) through the analysis engine and print one row per second.

Usage:
    python -m src.analysis.replay --file data/samples/sample.jsonl
    python -m src.analysis.replay --make-sample data/samples/sample.jsonl   # synthetic scenario
    python -m src.analysis.replay --file rec.jsonl --csv out.csv            # also save the rows

Feature files hold numbers only (no images), e.g. from `python -m src.main --record-features`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src.analysis.calibration import Calibration
from src.analysis.engine import AnalysisEngine, TickResult
from src.core.config import load_config, resolve_path
from src.core.events import FrameFeatures

SAMPLE_FPS = 10.0
SAMPLE_T0 = 1_700_000_000.0  # fixed start so the sample is reproducible
REQUIRED_KEYS: frozenset[str] = frozenset({"ts", "user_present"})


def read_features(path: Path) -> Iterator[FrameFeatures]:
    """Yield FrameFeatures from a .jsonl file."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield FrameFeatures.from_dict(json.loads(line))


def _round(v, nd: int = 3):
    """Round floats, recursively inside lists."""
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, list):
        return [_round(x, nd) for x in v]
    return v


def compact(d: dict) -> dict:
    """Drop default-valued fields (FrameFeatures.from_dict restores them) and round floats
    (keypoint pixels to 1 decimal, everything else to 3)."""
    out = {}
    for k, v in d.items():
        if k not in REQUIRED_KEYS and (v is None or v is False or v == 0.0):
            continue
        out[k] = _round(v, 1 if k == "pose_keypoints" else 3)
    return out


def make_sample(path: Path, seed: int = 7) -> int:
    """Write a synthetic ~8-minute scenario: normal, drowsy (long closures), 3 yawns, hand to
    forehead, slouching, a combined fatigue peak, 20 s away, return. Returns the frame count."""
    rng = np.random.default_rng(seed)
    segments = [("normal", 90), ("drowsy", 60), ("yawn", 45), ("hand", 40), ("slouch", 60), ("lelah", 120), ("away", 20), ("return", 45)]
    t, rows, next_blink = 0.0, [], 3.0
    dt = 1.0 / SAMPLE_FPS
    for name, dur in segments:
        end = t + dur
        seg_start = t
        while t < end:
            present = name != "away"
            ear = 0.31 + rng.normal(0, 0.01)
            if t >= next_blink:  # 150 ms blink every ~3.5 s
                if t < next_blink + 0.15:
                    ear = 0.09
                else:
                    next_blink = t + rng.uniform(2.5, 4.5)
            if name in ("drowsy", "lelah") and (t - seg_start) % 8 < (1.5 if name == "drowsy" else 2.5):
                ear = 0.08  # 1.5 s closures every 8 s
            mar = 0.1
            if name in ("yawn", "lelah") and (t - seg_start) % 15 < 2.0:
                mar = 0.7  # 2 s yawn every 15 s
            hand = name in ("hand", "lelah") and (t - seg_start) % 10 < 4.0
            slouch = 0.62 if name != "slouch" else 0.42
            f = FrameFeatures(
                ts=SAMPLE_T0 + t, user_present=present,
                ear_left=ear if present else None, ear_right=ear if present else None, ear_mean=ear if present else None,
                mar=mar if present else None, yawn=1.0 if present and mar > 0.45 else 0.0,
                brow_ratio=0.30 if present else None, hand_to_forehead=present and hand,
                shoulder_slouch=slouch if present else None, face_size_ratio=0.30 if present else None,
                nose_xy=(0.5 + 0.01 * math.sin(t), 0.45) if present else None,
            )
            rows.append(json.dumps(compact(f.to_dict()), separators=(",", ":")))
            t += dt
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def replay(features: list[FrameFeatures], engine: AnalysisEngine) -> list[TickResult]:
    """Feed features in time order and tick at every whole second."""
    results: list[TickResult] = []
    if not features:
        return results
    next_tick = math.floor(features[0].ts) + 1.0
    for f in features:
        while f.ts >= next_tick:
            results.append(engine.tick(next_tick))
            next_tick += 1.0
        engine.add(f)
    results.append(engine.tick(next_tick))
    return results


def format_row(r: TickResult, t0: float) -> str:
    """One printable line."""
    snap = r.snapshot
    events = [p.kind for p in r.presence] + [a.kind for a in r.alerts]
    blink = "-" if r.stats.blink_rate is None else f"{r.stats.blink_rate:4.1f}"
    score = "  -  " if snap is None else f"{snap.score:5.1f}"
    level = "-" if snap is None else snap.level
    return (f"t={r.ts - t0:6.0f}s skor={score} level={level:7s} perclos={r.stats.perclos:.2f} "
            f"kedip/menit={blink} menguap={r.stats.yawn_count} tangan={r.stats.hand_forehead_count}"
            + (f"  >> {', '.join(events)}" if events else ""))


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, help="file .jsonl berisi FrameFeatures")
    ap.add_argument("--make-sample", type=Path, help="buat skenario sintetis ke path ini")
    ap.add_argument("--csv", type=Path, help="simpan hasil per detik ke CSV")
    args = ap.parse_args(argv)

    if args.make_sample:
        n = make_sample(resolve_path(args.make_sample))
        print(f"{n} frame sintetis ditulis ke {args.make_sample}")
        if not args.file:
            return 0
    if not args.file:
        ap.error("--file wajib diisi (atau --make-sample)")

    cfg = load_config()
    calibration = Calibration.from_values(0.31, 0.09, 0.62, 0.30, 0.30)
    features = list(read_features(resolve_path(args.file)))
    results = replay(features, AnalysisEngine(cfg, calibration))
    t0 = features[0].ts if features else 0.0
    for r in results:
        print(format_row(r, t0))
    if args.csv:
        with resolve_path(args.csv).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "score", "level", "perclos", "blink_rate", "yawn_count", "hand_count", "events"])
            for r in results:
                s = r.snapshot
                w.writerow([round(r.ts - t0), s.score if s else "", s.level if s else "", round(r.stats.perclos, 3),
                            r.stats.blink_rate, r.stats.yawn_count, r.stats.hand_forehead_count,
                            " ".join([p.kind for p in r.presence] + [a.kind for a in r.alerts])])
    return 0


if __name__ == "__main__":
    sys.exit(main())
