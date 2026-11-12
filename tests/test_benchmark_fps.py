"""Smoke test for scripts/benchmark_fps.py helpers (no webcam, no model download)."""

from __future__ import annotations

import numpy as np

from tests.conftest import load_script


def test_default_args_match_spec() -> None:
    args = load_script("benchmark_fps").parse_args([])
    assert args.duration == 30.0
    assert args.model.endswith("yolov8n.pt")


def test_draw_overlay_draws_box_and_label() -> None:
    bench = load_script("benchmark_fps")
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    bench.draw_overlay(frame, [(60, 50, 120, 110, 0.9)], fps=12.3, device="cpu")
    assert frame.any()
    assert tuple(frame[80, 60]) == bench.BOX_COLOR


def test_read_frame_loops_still_image() -> None:
    bench = load_script("benchmark_fps")
    img = np.full((4, 4, 3), 7, dtype=np.uint8)
    first = bench.read_frame(img)
    first[:] = 0
    assert bench.read_frame(img).max() == 7
