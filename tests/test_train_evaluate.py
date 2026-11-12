"""Tests for training/train.py and training/evaluate.py helpers (no training run)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.core.config import load_config
from training import evaluate as ev
from training import train as tr


def test_train_kwargs_follow_spec_and_cli() -> None:
    cfg = load_config()
    kwargs = tr.build_train_kwargs(cfg, tr.parse_args([]), device="cpu")
    assert (kwargs["epochs"], kwargs["imgsz"], kwargs["batch"], kwargs["patience"]) == (100, 640, 16, 20)
    assert kwargs["augment"] is True and kwargs["data"].endswith("data/data.yaml")
    assert kwargs["hsv_h"] == 0.0 and kwargs["hsv_s"] == 0.0  # grayscale input

    override = tr.build_train_kwargs(cfg, tr.parse_args(["--epochs", "3", "--name", "x"]), device="mps")
    assert override["epochs"] == 3 and override["name"] == "x" and override["device"] == "mps"


def test_no_hsv_override_for_color_input() -> None:
    cfg = copy.deepcopy(load_config())
    cfg["det"]["grayscale"] = False
    assert "hsv_s" not in tr.build_train_kwargs(cfg, tr.parse_args([]), device="cpu")


def test_copy_best_keeps_previous(tmp_path: Path) -> None:
    best = tmp_path / "run" / "best.pt"
    best.parent.mkdir()
    best.write_bytes(b"new")
    models = tmp_path / "models"
    models.mkdir()
    (models / "best.pt").write_bytes(b"old")
    target = tr.copy_best(best, models)
    assert target.read_bytes() == b"new"
    assert (models / "best.prev.pt").read_bytes() == b"old"


def test_count_instances(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n2 0.5 0.5 0.1 0.1\n")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "c.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert ev.count_instances(tmp_path) == [2, 0, 1, 0]


def test_per_class_metrics_and_report() -> None:
    rows = ev.per_class_metrics({0: (0.9, 0.8, 0.85, 0.5), 2: (0.6, 0.5, 0.55, 0.3)}, [10, 0, 4, 0], target=0.80)
    by_name = {r.name: r for r in rows}
    assert by_name["mata_tertutup"].meets_target is True
    assert by_name["menguap"].meets_target is False
    assert by_name["mata_terbuka"].meets_target is None  # no instances in split
    meta = {"evaluated_at": "now", "weights": "models/best.pt", "split": "test", "subjects": ["s1"],
            "imgsz": 640, "device": "cpu", "grayscale": True, "speed_ms": {"inference": 5.0}, "train": None}
    overall = {"precision": 0.7, "recall": 0.6, "map50": 0.7, "map50_95": 0.4}
    draft = ev.render_markdown(rows, overall, meta, 0.80, "draf")
    assert "Label draf" in draft and "❌" in draft and "✅" in draft
    assert "`menguap`" in draft and "`kerutan_dahi`" in draft
    assert "Label draf" not in ev.render_markdown(rows, overall, meta, 0.80, "final")


@pytest.mark.parametrize("status", ["draf", "final"])
def test_label_status_choices(status: str) -> None:
    assert status in ev.LABEL_STATUSES
