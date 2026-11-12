"""Tests for src.core.config."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.core.config import ROOT_DIR, ConfigError, load_config, resolve_path, validate_config


@pytest.fixture()
def cfg() -> dict:
    return load_config()


def test_project_config_loads_with_all_sections(cfg: dict) -> None:
    for section in ("camera", "inference", "det", "pose", "ear", "window", "score", "buddy", "storage", "privacy"):
        assert section in cfg


def test_score_weights_sum_to_one(cfg: dict) -> None:
    assert sum(cfg["score"]["weights"].values()) == pytest.approx(1.0)


def test_spec_defaults_are_kept(cfg: dict) -> None:
    assert cfg["ear"]["closed_threshold"] == 0.21
    assert cfg["score"]["thresholds"] == {"low": 35, "high": 65}
    assert cfg["inference"]["imgsz"] == 480
    assert cfg["privacy"]["save_frames"] is False


def test_bad_weights_rejected(cfg: dict) -> None:
    bad = copy.deepcopy(cfg)
    bad["score"]["weights"]["perclos"] = 0.9
    with pytest.raises(ConfigError, match="sum to 1.0"):
        validate_config(bad)


def test_missing_section_rejected(cfg: dict) -> None:
    bad = copy.deepcopy(cfg)
    del bad["buddy"]
    with pytest.raises(ConfigError, match="buddy"):
        validate_config(bad)


def test_save_frames_true_rejected(cfg: dict) -> None:
    bad = copy.deepcopy(cfg)
    bad["privacy"]["save_frames"] = True
    with pytest.raises(ConfigError, match="save_frames"):
        validate_config(bad)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_load_from_custom_path(tmp_path: Path, cfg: dict) -> None:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert load_config(p) == cfg


def test_resolve_path_relative_to_root() -> None:
    assert resolve_path("models/best.pt") == ROOT_DIR / "models" / "best.pt"
    assert resolve_path("/abs/x.pt") == Path("/abs/x.pt")
