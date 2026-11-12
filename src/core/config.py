"""Load and validate the single project configuration file (config.yaml)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = ROOT_DIR / "config.yaml"

REQUIRED_SECTIONS: tuple[str, ...] = (
    "camera",
    "inference",
    "det",
    "pose",
    "ear",
    "window",
    "score",
    "burnout",
    "habit",
    "buddy",
    "storage",
    "privacy",
    "logging",
    "dataset",
    "train",
    "landmark",
    "calibration",
    "health",
    "runtime",
)
USER_SETTINGS_PATH: Path = ROOT_DIR / "data" / "settings.yaml"

WEIGHT_SUM_TOLERANCE: float = 1e-6


class ConfigError(ValueError):
    """Raised when config.yaml is missing, malformed, or violates a project rule."""


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Read config.yaml, validate it, and return it as a nested dict."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {cfg_path}")
    validate_config(data)
    return data


def validate_config(cfg: dict[str, Any]) -> None:
    """Check required sections and invariants (score weights sum to 1.0, sane thresholds)."""
    missing = [name for name in REQUIRED_SECTIONS if name not in cfg]
    if missing:
        raise ConfigError(f"Missing config sections: {', '.join(missing)}")

    weights = cfg["score"].get("weights", {})
    total = float(sum(weights.values()))
    if not math.isclose(total, 1.0, abs_tol=WEIGHT_SUM_TOLERANCE):
        raise ConfigError(f"score.weights must sum to 1.0, got {total:.4f}")

    thresholds = cfg["score"].get("thresholds", {})
    if not thresholds.get("low", 0) < thresholds.get("high", 0):
        raise ConfigError("score.thresholds.low must be lower than score.thresholds.high")

    fusion = cfg["ear"]["fusion_w_ear"] + cfg["ear"]["fusion_w_yolo"]
    if not math.isclose(fusion, 1.0, abs_tol=WEIGHT_SUM_TOLERANCE):
        raise ConfigError(f"ear.fusion_w_ear + ear.fusion_w_yolo must be 1.0, got {fusion:.4f}")

    split = cfg["dataset"]["split"]
    split_total = split["train"] + split["val"] + split["test"]
    if not math.isclose(split_total, 1.0, abs_tol=WEIGHT_SUM_TOLERANCE):
        raise ConfigError(f"dataset.split train+val+test must be 1.0, got {split_total:.4f}")

    pre = cfg["dataset"]["preannotate"]
    if not pre["ear_closed"] < pre["ear_open"]:
        raise ConfigError("dataset.preannotate.ear_closed must be lower than ear_open")

    if cfg["privacy"].get("save_frames", False):
        raise ConfigError("privacy.save_frames must stay false: frames may never be written to disk")


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config path (e.g. models/best.pt) against the project root."""
    p = Path(relative)
    return p if p.is_absolute() else ROOT_DIR / p


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into a copy of base."""
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_app_config(path: Path | str | None = None, settings_path: Path | None = None) -> dict[str, Any]:
    """config.yaml + the user's saved settings (data/settings.yaml), validated (D-024)."""
    cfg = load_config(path)
    sp = settings_path or USER_SETTINGS_PATH
    if sp.exists():
        override = yaml.safe_load(sp.read_text(encoding="utf-8")) or {}
        cfg = deep_merge(cfg, override)
        validate_config(cfg)
    return cfg


def save_user_settings(changes: dict[str, Any], settings_path: Path | None = None) -> None:
    """Merge `changes` into data/settings.yaml (config.yaml itself is never rewritten)."""
    sp = settings_path or USER_SETTINGS_PATH
    current = (yaml.safe_load(sp.read_text(encoding="utf-8")) or {}) if sp.exists() else {}
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(yaml.safe_dump(deep_merge(current, changes), sort_keys=False, allow_unicode=True), encoding="utf-8")
