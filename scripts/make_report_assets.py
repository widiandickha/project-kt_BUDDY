#!/usr/bin/env python3
"""FASE 9: generate report figures and tables from docs/eval/ automatically.

Usage:
    python scripts/make_report_assets.py                 # -> docs/report_assets/
    python scripts/make_report_assets.py --out /tmp/assets

Always produced: architecture.png, workflow.png, buddy_expressions.png, score_timeline.png
(replay of data/samples/sample.jsonl). Produced when the input exists: model_metrics.png
(docs/eval/metrics.json), events.png (events.json), performance.png (performance.md),
stress_correlation.png (stress_correlation_data.csv), ux.png (ux_summary.csv), plus Markdown
tables. docs/report_assets/README.md lists what was made and what is still waiting for data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.core.config import load_config, resolve_path  # noqa: E402

ACCENT = "#9A3412"
GREEN = "#166534"
AMBER = "#92400E"
RED = "#991B1B"
SURFACE = "#F0ECE1"
TEXT = "#1C1917"
DPI = 160


def _box(ax, x: float, y: float, w: float, h: float, text: str, color: str = SURFACE) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=color, ec=TEXT, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=TEXT, wrap=True)


def _arrow(ax, a: tuple[float, float], b: tuple[float, float], label: str = "", curve: float = 0.0,
           label_xy: tuple[float, float] | None = None) -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=1, color=TEXT,
                                 connectionstyle=f"arc3,rad={curve}"))
    if label:
        x, y = label_xy or ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.02)
        ax.text(x, y, label, ha="center", fontsize=7, color=AMBER)


def architecture(out: Path) -> Path:
    """Runtime architecture (spec section 4)."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1)
    ax.axis("off")
    _box(ax, 0.01, 0.62, 0.13, 0.2, "Webcam\n(frame hanya\ndi memori)")
    _box(ax, 0.18, 0.62, 0.15, 0.2, "Thread A\nCaptureThread\nQueue(maxsize=2)")
    _box(ax, 0.37, 0.55, 0.2, 0.34, "Thread B · Inferensi\n• MediaPipe → EAR/MAR/alis\n• YOLOv8n-pose → tangan,\n  postur\n• YOLOv8 wajah (opsional)\n→ FrameFeatures")
    _box(ax, 0.62, 0.55, 0.2, 0.34, "Analisis (tick 1 s)\n• Sliding window 60 s\n• Skor stres + EMA\n  + hysteresis\n• Habit & presence\n• Tren burnout")
    _box(ax, 0.86, 0.66, 0.13, 0.16, "EventBus\nstress_snapshot\nhabit_alert ...")
    _box(ax, 0.62, 0.12, 0.2, 0.25, "Virtual Buddy\nstate machine 12 ekspresi\nsprite + bubble saran\n(cooldown, eskalasi)", "#FFF7ED")
    _box(ax, 0.86, 0.12, 0.13, 0.25, "SQLite (DbWriter)\nstress_log, events,\nadvice_log,\ndaily_summary")
    _box(ax, 0.37, 0.12, 0.2, 0.25, "UI main thread (Qt)\ntray · dashboard ·\nkalibrasi · pengaturan")
    _arrow(ax, (0.14, 0.72), (0.18, 0.72), "frame")
    _arrow(ax, (0.33, 0.72), (0.37, 0.72), "terbaru")
    _arrow(ax, (0.57, 0.72), (0.62, 0.72), "fitur")
    _arrow(ax, (0.82, 0.74), (0.86, 0.74))
    _arrow(ax, (0.72, 0.55), (0.72, 0.37), "BuddyEvent")
    _arrow(ax, (0.92, 0.66), (0.92, 0.37), "batch 10 s")
    _arrow(ax, (0.925, 0.12), (0.47, 0.12), "ringkasan harian", curve=-0.25, label_xy=(0.70, 0.015))
    ax.set_title("Arsitektur runtime Stress Buddy (semua lokal, tanpa penyimpanan video)", fontsize=11)
    path = out / "architecture.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def workflow(out: Path) -> Path:
    """Development workflow FASE 1-9."""
    steps = ["1 Setup &\nbenchmark", "2 Dataset &\nanotasi", "3 Training\nYOLOv8", "4 Analisis &\nskoring", "5 Virtual\nBuddy",
             "6 Database &\ndashboard", "7 Integrasi", "8 Pengujian", "9 Laporan &\ndemo"]
    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, s in enumerate(steps):
        _box(ax, i + 0.05, 0.25, 0.8, 0.5, s, "#FFF7ED" if i in (1, 2) else SURFACE)
        if i < len(steps) - 1:
            _arrow(ax, (i + 0.85, 0.5), (i + 1.05, 0.5))
    ax.set_title("Alur pengerjaan per fase (dataset & training menunggu data relawan dan hardware)", fontsize=10)
    path = out / "workflow.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def buddy_sheet(out: Path) -> Path:
    """12 expressions rendered by the real sprite player."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from src.buddy.demo import render_sheet

    _app = QApplication.instance() or QApplication([])
    path = out / "buddy_expressions.png"
    render_sheet(path)
    return path


def score_timeline(out: Path, cfg: dict[str, Any]) -> Path:
    """Score/level over the synthetic sample with events (methodology illustration)."""
    from src.analysis.calibration import Calibration
    from src.analysis.engine import AnalysisEngine
    from src.analysis.replay import make_sample, read_features, replay

    sample = resolve_path("data/samples/sample.jsonl")
    if not sample.exists():
        make_sample(sample)
    feats = list(read_features(sample))
    res = replay(feats, AnalysisEngine(cfg, Calibration.from_values(0.31, 0.09, 0.62, 0.30, 0.30)))
    t0 = feats[0].ts
    t = [r.ts - t0 for r in res if r.snapshot]
    score = [r.snapshot.score for r in res if r.snapshot]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, score, color=ACCENT, lw=1.6, label="skor stres (EMA)")
    low, high = cfg["score"]["thresholds"]["low"], cfg["score"]["thresholds"]["high"]
    ax.axhspan(0, low, color=GREEN, alpha=0.07)
    ax.axhspan(low, high, color=AMBER, alpha=0.07)
    ax.axhspan(high, 100, color=RED, alpha=0.07)
    for r in res:
        for name in [p.kind for p in r.presence] + [a.kind for a in r.alerts]:
            ax.axvline(r.ts - t0, color="#44403C", ls=":", lw=0.8)
            ax.text(r.ts - t0, 92, name, rotation=90, fontsize=7, va="top")
    ax.set(xlabel="detik", ylabel="skor (0-100)", ylim=(0, 100),
           title="Contoh replay skenario sintetis: normal → mengantuk → menguap → tangan ke dahi → lelah → pergi → kembali")
    ax.legend(loc="upper left", fontsize=8)
    path = out / "score_timeline.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def model_metrics(eval_dir: Path, out: Path) -> list[Path]:
    """Per-class bars from metrics.json."""
    src = eval_dir / "metrics.json"
    if not src.exists():
        return []
    data = json.loads(src.read_text(encoding="utf-8"))
    rows = [c for c in data["classes"] if c["map50"] is not None]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = range(len(rows))
    w = 0.27
    for k, (key, color) in enumerate((("precision", GREEN), ("recall", AMBER), ("map50", ACCENT))):
        ax.bar([i + (k - 1) * w for i in x], [r[key] for r in rows], w, label=key, color=color)
    ax.axhline(load_config()["train"]["target_map50"], color=RED, ls="--", lw=0.8, label="target mAP50")
    ax.set_xticks(list(x), [r["name"] for r in rows])
    status = data["meta"].get("label_status", "draf")
    ax.set(ylim=(0, 1), title=f"Detektor wajah per kelas (split {data['meta']['split']}, label {status})")
    ax.legend(fontsize=8)
    path = out / "model_metrics.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return [path]


def events_chart(eval_dir: Path, out: Path) -> list[Path]:
    """Precision/recall per event type, mean over sessions."""
    src = eval_dir / "events.json"
    if not src.exists():
        return []
    rows = [r for s in json.loads(src.read_text(encoding="utf-8")) for r in s["results"]]
    df = pd.DataFrame(rows)
    agg = df.groupby("type")[["tp", "fp", "fn"]].sum()
    agg["precision"] = agg.tp / (agg.tp + agg.fp).where(lambda v: v > 0)
    agg["recall"] = agg.tp / (agg.tp + agg.fn).where(lambda v: v > 0)
    ax = agg[["precision", "recall"]].plot.bar(color=[GREEN, AMBER], figsize=(6, 3.5), rot=0)
    ax.set(ylim=(0, 1), title="Akurasi level event (gabungan semua sesi, ±2 s)")
    path = out / "events.png"
    ax.figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(ax.figure)
    (out / "events_table.md").write_text(agg.round(2).to_markdown(), encoding="utf-8")
    return [path, out / "events_table.md"]


def performance_chart(eval_dir: Path, out: Path) -> list[Path]:
    """FPS and CPU% per benchmark row of performance.md."""
    src = eval_dir / "performance.md"
    if not src.exists():
        return []
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 12 and re.match(r"\d{4}-\d{2}-\d{2}", cells[0]):
            rows.append({"run": f"{cells[2]} · batas {cells[6]}", "fps": float(cells[7]), "cpu": float(cells[9])})
    if not rows:
        return []
    df = pd.DataFrame(rows).set_index("run")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    df["fps"].plot.bar(ax=axes[0], color=GREEN, rot=20)
    axes[0].axhline(10, color=RED, ls="--", lw=0.8)
    axes[0].set(title="FPS inferensi (target ≥ 10)", xlabel="")
    df["cpu"].plot.bar(ax=axes[1], color=ACCENT, rot=20)
    axes[1].axhline(40, color=RED, ls="--", lw=0.8)
    axes[1].set(title="CPU % proses (target < 40)", xlabel="")
    path = out / "performance.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return [path]


def correlation_chart(eval_dir: Path, out: Path) -> list[Path]:
    """Scatter of system score vs subjective rating."""
    src = eval_dir / "stress_correlation_data.csv"
    if not src.exists():
        return []
    df = pd.read_csv(src).dropna(subset=["skor_sistem", "skor_lelah"])
    if df.empty:
        return []
    fig, ax = plt.subplots(figsize=(5, 4))
    for p, g in df.groupby("peserta"):
        ax.scatter(g["skor_lelah"], g["skor_sistem"], label=str(p), s=18)
    ax.set(xlabel="rating lelah subjektif", ylabel="skor sistem (rata-rata 30 menit)", title="Skor sistem vs kuesioner")
    if df["peserta"].nunique() <= 12:
        ax.legend(fontsize=7)
    path = out / "stress_correlation.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return [path]


def ux_chart(eval_dir: Path, out: Path) -> list[Path]:
    """Mean ± SD of each Likert item."""
    src = eval_dir / "ux_summary.csv"
    if not src.exists():
        return []
    df = pd.read_csv(src)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh(df["item"], df["mean"], xerr=df["sd"].fillna(0), color=ACCENT)
    ax.set(xlim=(1, 5), title="Kuesioner pengguna (Likert 1–5)")
    path = out / "ux.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return [path]


PENDING = {
    "model_metrics.png": "metrics.json (training/evaluate.py setelah training)",
    "events.png": "events.json (scripts/eval_events.py dari sesi berskenario)",
    "performance.png": "performance.md (scripts/benchmark_fps.py --pipeline)",
    "stress_correlation.png": "stress_correlation_data.csv (analyze_questionnaire.py stress, >= 10 partisipan)",
    "ux.png": "ux_summary.csv (analyze_questionnaire.py ux)",
}


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("docs/report_assets"))
    ap.add_argument("--eval-dir", type=Path, default=None)
    ap.add_argument("--skip-buddy", action="store_true", help="tanpa render Qt (lingkungan tanpa Qt)")
    args = ap.parse_args(argv)
    cfg = load_config()
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_dir = resolve_path(args.eval_dir or cfg["train"]["eval_dir"])

    made: list[Path] = [architecture(out), workflow(out), score_timeline(out, cfg)]
    if not args.skip_buddy:
        made.append(buddy_sheet(out))
    for fn in (model_metrics, events_chart, performance_chart, correlation_chart, ux_chart):
        made += fn(eval_dir, out)
    names = {p.name for p in made}
    lines = ["# Aset Laporan", "", "Dibuat otomatis oleh `scripts/make_report_assets.py`. Jalankan ulang setelah ada data baru.", "",
             "## Tersedia", ""] + [f"- `{p.name}`" for p in made]
    pending = [f"- `{k}` ← {v}" for k, v in PENDING.items() if k not in names]
    if pending:
        lines += ["", "## Menunggu data", ""] + pending
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for p in made:
        print(p)
    if pending:
        print("Menunggu data:", ", ".join(k for k in PENDING if k not in names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
