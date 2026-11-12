#!/usr/bin/env python3
"""FASE 8.4 + 8.5: compare the stress score with subjective ratings, and summarise the UX survey.

Usage:
    python scripts/analyze_questionnaire.py template --out docs/eval/templates
    python scripts/analyze_questionnaire.py stress --ratings ratings.csv --db p01=data/p01.db --db p02=data/p02.db
    python scripts/analyze_questionnaire.py ux --likert likert.csv [--db p01=data/p01.db ...]

stress: ratings.csv columns `peserta,waktu,skor_lelah` (waktu = 'YYYY-MM-DD HH:MM', skor_lelah
1-5 or any numeric scale, e.g. NASA-TLX). For each rating, the mean system score of the
preceding --window-min minutes is taken from that participant's database. Pearson and Spearman
correlations are reported per participant and pooled - honestly, also when weak.

ux: likert.csv columns `peserta` + one column per item (1-5). Reports mean/SD/median per item.
With --db, also the share of Buddy advice the participant followed (response = 'oke').
Reports go to docs/eval/stress_correlation.md and docs/eval/ux.md.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.core.config import load_config, resolve_path  # noqa: E402

DEFAULT_WINDOW_MIN = 30
MIN_PAIRS = 3
UX_ITEMS: tuple[str, ...] = ("gangguan", "kejelasan_saran", "kelucuan", "kepercayaan_saran", "ingin_memakai_lagi")


def parse_dbs(values: list[str]) -> dict[str, Path]:
    """--db peserta=path pairs."""
    out = {}
    for v in values:
        name, _, path = v.partition("=")
        if not name or not path:
            raise SystemExit(f"--db harus berbentuk peserta=path, bukan '{v}'")
        out[name.strip()] = Path(path)
    return out


def mean_score_before(conn: sqlite3.Connection, ts: float, window_min: float) -> float | None:
    """Mean stress score in (ts - window, ts]."""
    row = conn.execute("SELECT AVG(score) FROM stress_log WHERE ts > ? AND ts <= ? AND score IS NOT NULL",
                       (ts - window_min * 60.0, ts)).fetchone()
    return None if row[0] is None else float(row[0])


def correlations(df: pd.DataFrame) -> dict[str, float | int | None]:
    """Pearson + Spearman of skor_sistem vs skor_lelah (None when < MIN_PAIRS or no variance)."""
    d = df.dropna(subset=["skor_sistem", "skor_lelah"])
    n = len(d)
    if n < MIN_PAIRS or d["skor_sistem"].nunique() < 2 or d["skor_lelah"].nunique() < 2:
        return {"n": n, "pearson": None, "spearman": None}
    return {"n": n, "pearson": float(d["skor_sistem"].corr(d["skor_lelah"], method="pearson")),
            "spearman": float(d["skor_sistem"].rank().corr(d["skor_lelah"].rank(), method="pearson"))}


def strength(r: float | None) -> str:
    """Verbal strength of a correlation coefficient."""
    if r is None:
        return "tidak bisa dihitung"
    a = abs(r)
    word = "sangat lemah" if a < 0.2 else "lemah" if a < 0.4 else "sedang" if a < 0.6 else "kuat" if a < 0.8 else "sangat kuat"
    return f"{word} ({'positif' if r >= 0 else 'negatif'})"


def stress_analysis(ratings: pd.DataFrame, dbs: dict[str, Path], window_min: float) -> tuple[pd.DataFrame, list[dict]]:
    """Attach system scores to ratings and compute correlations per participant + pooled."""
    rows = []
    for _, r in ratings.iterrows():
        db = dbs.get(str(r["peserta"]))
        score = None
        if db is not None and db.exists():
            ts = datetime.strptime(str(r["waktu"]), "%Y-%m-%d %H:%M").timestamp()
            with sqlite3.connect(db) as conn:
                score = mean_score_before(conn, ts, window_min)
        rows.append({**r.to_dict(), "skor_sistem": score})
    df = pd.DataFrame(rows)
    stats = [{"peserta": p, **correlations(g)} for p, g in df.groupby("peserta")]
    stats.append({"peserta": "SEMUA", **correlations(df)})
    return df, stats


def _f(v: float | None) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"


def write_stress_report(stats: list[dict], window_min: float, path: Path) -> None:
    """docs/eval/stress_correlation.md"""
    lines = ["# Korelasi Skor Stres Sistem vs Kuesioner (FASE 8.4)", "",
             f"Dibuat {datetime.now():%Y-%m-%d %H:%M}. Skor sistem = rata-rata {window_min:g} menit sebelum tiap pengisian.", "",
             "| Peserta | n | Pearson r | Spearman ρ | Kekuatan (Spearman) |", "|---|---|---|---|---|"]
    for s in stats:
        lines.append(f"| {s['peserta']} | {s['n']} | {_f(s['pearson'])} | {_f(s['spearman'])} | {strength(s['spearman'])} |")
    lines += ["", "Skor sistem adalah indikator visual perilaku, bukan ukuran klinis. Korelasi lemah dilaporkan apa adanya.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def ux_analysis(likert: pd.DataFrame) -> pd.DataFrame:
    """Mean/SD/median/n per Likert item."""
    items = [c for c in likert.columns if c != "peserta"]
    return pd.DataFrame([{"item": c, "n": int(likert[c].notna().sum()), "mean": float(likert[c].mean()),
                          "sd": float(likert[c].std(ddof=1)) if likert[c].notna().sum() > 1 else float("nan"),
                          "median": float(likert[c].median())} for c in items])


def advice_follow(dbs: dict[str, Path]) -> list[dict]:
    """Share of advice answered with 'oke' per participant."""
    out = []
    for p, db in dbs.items():
        with sqlite3.connect(db) as conn:
            total, ok = conn.execute("SELECT COUNT(*), SUM(response = 'oke') FROM advice_log").fetchone()
        out.append({"peserta": p, "saran": int(total or 0), "diikuti": int(ok or 0),
                    "rasio": None if not total else (ok or 0) / total})
    return out


def write_ux_report(summary: pd.DataFrame, follow: list[dict], path: Path) -> None:
    """docs/eval/ux.md"""
    lines = ["# Uji Pengguna Virtual Buddy (FASE 8.5)", "", f"Dibuat {datetime.now():%Y-%m-%d %H:%M}. Skala Likert 1–5.", "",
             "| Item | n | Rata-rata | SD | Median |", "|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['item']} | {r['n']} | {_f(r['mean'])} | {_f(r['sd'])} | {_f(r['median'])} |")
    if follow:
        lines += ["", "## Rasio saran yang diikuti (advice_log)", "", "| Peserta | Saran | Diikuti | Rasio |", "|---|---|---|---|"]
        for f in follow:
            lines.append(f"| {f['peserta']} | {f['saran']} | {f['diikuti']} | {_f(f['rasio'])} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_templates(out: Path) -> list[Path]:
    """Empty CSV templates for the participants' forms."""
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "ratings.csv": "peserta,waktu,skor_lelah\np01,2026-11-20 10:30,2\n",
        "likert.csv": "peserta," + ",".join(UX_ITEMS) + "\np01,2,4,5,4,4\n",
        "event_labels.csv": "type,start_sec\nyawn,20\neye_closed_long,42\nhand_to_forehead,62\n",
    }
    paths = []
    for name, text in files.items():
        p = out / name
        if not p.exists():
            p.write_text(text, encoding="utf-8")
        paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("--out", type=Path, default=Path("docs/eval/templates"))
    s = sub.add_parser("stress")
    s.add_argument("--ratings", type=Path, required=True)
    s.add_argument("--db", action="append", default=[], help="peserta=path ke stress_buddy.db")
    s.add_argument("--window-min", type=float, default=DEFAULT_WINDOW_MIN)
    u = sub.add_parser("ux")
    u.add_argument("--likert", type=Path, required=True)
    u.add_argument("--db", action="append", default=[])
    args = ap.parse_args(argv)

    eval_dir = resolve_path(load_config()["train"]["eval_dir"])
    if args.cmd == "template":
        for p in write_templates(resolve_path(args.out)):
            print(p)
        return 0
    if args.cmd == "stress":
        df, stats = stress_analysis(pd.read_csv(args.ratings), parse_dbs(args.db), args.window_min)
        write_stress_report(stats, args.window_min, eval_dir / "stress_correlation.md")
        df.to_csv(eval_dir / "stress_correlation_data.csv", index=False)
        for st in stats:
            print(f"{st['peserta']:8s} n={st['n']:3d} pearson={_f(st['pearson'])} spearman={_f(st['spearman'])}")
        return 0
    summary = ux_analysis(pd.read_csv(args.likert))
    follow = advice_follow(parse_dbs(args.db)) if args.db else []
    write_ux_report(summary, follow, eval_dir / "ux.md")
    summary.to_csv(eval_dir / "ux_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
