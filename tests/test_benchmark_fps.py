"""Smoke test for scripts/benchmark_fps.py helpers (no webcam, no model download)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_fps.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_fps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


def test_default_args_match_spec() -> None:
    args = _load_module().parse_args([])
    assert args.duration == 30.0
    assert args.model.endswith("yolov8n.pt")


def test_draw_overlay_draws_box_and_label() -> None:
    bench = _load_module()
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    bench.draw_overlay(frame, [(60, 50, 120, 110, 0.9)], fps=12.3, device="cpu")
    assert frame.any()
    assert tuple(frame[80, 60]) == bench.BOX_COLOR


def test_read_frame_loops_still_image() -> None:
    bench = _load_module()
    img = np.full((4, 4, 3), 7, dtype=np.uint8)
    first = bench.read_frame(img)
    first[:] = 0
    assert bench.read_frame(img).max() == 7
