"""FASE 9: report asset generation."""

from __future__ import annotations

import json
from pathlib import Path

from tests.conftest import load_script


def test_assets_with_and_without_data(qapp, tmp_path: Path) -> None:
    mra = load_script("make_report_assets")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    out = tmp_path / "assets"
    assert mra.main(["--out", str(out), "--eval-dir", str(eval_dir)]) == 0
    made = {p.name for p in out.iterdir()}
    assert {"architecture.png", "workflow.png", "score_timeline.png", "buddy_expressions.png", "README.md"} <= made
    readme = (out / "README.md").read_text()
    assert "Menunggu data" in readme and "model_metrics.png" in readme

    (eval_dir / "metrics.json").write_text(json.dumps({
        "meta": {"split": "test", "label_status": "final"},
        "classes": [{"name": "mata_tertutup", "precision": 0.9, "recall": 0.8, "map50": 0.85},
                    {"name": "kerutan_dahi", "precision": None, "recall": None, "map50": None}]}))
    (eval_dir / "events.json").write_text(json.dumps([{"results": [
        {"type": "yawn", "tp": 2, "fp": 1, "fn": 1}, {"type": "hand_to_forehead", "tp": 3, "fp": 0, "fn": 0}]}]))
    (eval_dir / "performance.md").write_text(
        "| Waktu | Mesin | Device | imgsz | YOLO wajah | Sumber | Batas FPS | FPS | Latensi | CPU % | RAM | E2B |\n"
        "| 2026-11-12 11:03 | Darwin arm64 | cpu | 480 | tidak | x | 12 | 10.9 | 27 / 38 | 63 | 329 | 727 / 1238 |\n")
    (eval_dir / "stress_correlation_data.csv").write_text("peserta,waktu,skor_lelah,skor_sistem\np1,x,1,10\np1,y,3,40\n")
    (eval_dir / "ux_summary.csv").write_text("item,n,mean,sd,median\ngangguan,3,2.0,1.0,2\n")
    assert mra.main(["--out", str(out), "--eval-dir", str(eval_dir), "--skip-buddy"]) == 0
    made = {p.name for p in out.iterdir()}
    assert {"model_metrics.png", "events.png", "events_table.md", "performance.png", "stress_correlation.png", "ux.png"} <= made
    assert "Menunggu data" not in (out / "README.md").read_text()
