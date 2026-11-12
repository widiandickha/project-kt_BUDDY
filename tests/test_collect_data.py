"""Tests for scripts/collect_data.py helpers (no webcam)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.detection.classes import CLASS_NAMES
from tests.conftest import load_script
from training.common import parse_raw_name


def test_instructions_cover_every_class() -> None:
    cd = load_script("collect_data")
    assert sorted(i.cls for i in cd.INSTRUCTIONS) == sorted(CLASS_NAMES)
    assert len({i.key for i in cd.INSTRUCTIONS}) == len(cd.INSTRUCTIONS)


def test_output_path_follows_raw_scheme(tmp_path: Path) -> None:
    cd = load_script("collect_data")
    subj = cd.subject_id("Andi Pratama")
    p = cd.output_path(tmp_path, subj, "kerutan_dahi", "redup", "1700000000000")
    assert p.parent == tmp_path / "self-andi-pratama" / "kerutan_dahi"
    meta = parse_raw_name(p)
    assert (meta.subject, meta.hint, meta.condition) == ("self-andi-pratama", "kerutan_dahi", "redup")


def test_hud_never_touches_saved_frame() -> None:
    cd = load_script("collect_data")
    frame = np.zeros((240, 320, 3), np.uint8)
    view = cd.draw_hud(frame, cd.INSTRUCTIONS[0], cd.Counter(), burst=True, target=500)
    assert view.any() and not frame.any()


def test_count_existing(tmp_path: Path) -> None:
    cd = load_script("collect_data")
    folder = tmp_path / "self-x" / "menguap"
    folder.mkdir(parents=True)
    (folder / "a.jpg").touch()
    (folder / "b.jpg").touch()
    assert cd.count_existing(tmp_path, "self-x")["menguap"] == 2
