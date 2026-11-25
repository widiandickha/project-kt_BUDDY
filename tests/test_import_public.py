"""Tests for scripts/import_public.py (synthetic data, no real dataset needed)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.conftest import load_script


@pytest.fixture(scope="module")
def imp():
    return load_script("import_public")


def test_parse_nthu(imp) -> None:
    f = imp.parse_nthu(Path("x/yawning/002_glasses_yawning_1773_drowsy.jpg"))
    assert (f.subject, f.glasses, f.scenario, f.frame, f.hint) == ("002", "glasses", "yawning", 1773, "menguap")
    assert imp.parse_nthu(Path("x/notdrowsy/005_noglasses_nonsleepyCombination_1811_notdrowsy.jpg")).hint == "normal"
    assert imp.parse_nthu(Path("x/other/002_glasses_yawning_1_drowsy.jpg")) is None
    assert imp.parse_nthu(Path("x/yawning/readme.jpg")) is None


def test_select_evenly(imp) -> None:
    assert imp.select_evenly(list(range(10)), 3) == [0, 4, 9]
    assert imp.select_evenly([1, 2], 5) == [1, 2]
    assert imp.select_evenly([1, 2], 0) == []


def test_remove_color_overlay_cleans_boxes(imp) -> None:
    img = np.full((100, 100, 3), 120, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 255), 1)
    cv2.rectangle(img, (30, 40), (60, 50), (0, 255, 0), 1)
    cleaned, n = imp.remove_color_overlay(img, color_diff=40, dilate_px=1)
    assert n > 0
    spread = cleaned.astype(int).max(axis=2) - cleaned.astype(int).min(axis=2)
    assert spread.max() == 0  # grayscale again
    assert abs(int(cleaned[20, 50, 0]) - 120) < 15  # line replaced by surrounding gray


def test_remove_color_overlay_keeps_clean_frames(imp) -> None:
    img = np.full((10, 10, 3), 90, dtype=np.uint8)
    out, n = imp.remove_color_overlay(img, 40, 1)
    assert n == 0 and out is img


def test_import_nthu_end_to_end(imp, tmp_path: Path) -> None:
    src = tmp_path / "nthu" / "Multi class" / "train" / "drowsy" / "yawning"
    src.mkdir(parents=True)
    for i in range(6):
        img = np.full((48, 64, 3), 100, dtype=np.uint8)
        cv2.rectangle(img, (5, 5), (30, 30), (0, 255, 0), 1)
        cv2.imwrite(str(src / f"001_glasses_yawning_{i}_drowsy.jpg"), img)
    raw = tmp_path / "raw"
    counts = imp.import_nthu(tmp_path / "nthu", raw, {"menguap": 3}, color_diff=40, dilate_px=1)
    assert counts["menguap"] == 3
    files = sorted((raw / "nthu-001" / "menguap").glob("*.jpg"))
    assert [f.name for f in files][0].startswith("nthu-001__menguap__glasses__yawning-")
    out = cv2.imread(str(files[0])).astype(int)
    assert (out.max(axis=2) - out.min(axis=2)).max() <= 2  # overlay gone (JPEG tolerance)


def test_uta_naming(imp) -> None:
    assert imp.uta_participant(Path("/d/Fold1_part1/07/10.mov")) == "uta-7"
    assert imp.uta_participant(Path("/d/person_x/0.mp4")) == "uta-person-x"
    assert imp.uta_hint(Path("a/0.mov")) == "alert"
    assert imp.uta_hint(Path("a/5.MOV")) == "low_vigilant"
    assert imp.uta_hint(Path("a/10.mp4")) == "drowsy"
    assert imp.uta_hint(Path("a/Extra Clip.mp4")) == "extra_clip"


def test_extract_frames_and_import_uta(imp, tmp_path: Path) -> None:
    video = tmp_path / "uta" / "03" / "10.avi"
    video.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (1280, 720))
    for i in range(50):  # 5 s at 10 fps
        writer.write(np.full((720, 1280, 3), i * 4, dtype=np.uint8))
    writer.release()

    frames = list(imp.extract_frames(video, every_sec=1.0, max_frames=3))
    assert [pos for pos, _ in frames] == [0, 10, 20]

    raw = tmp_path / "raw"
    counts = imp.import_uta(tmp_path / "uta", raw, every_sec=2.0, max_frames=10, width=640)
    assert counts["drowsy"] == 3
    out = sorted((raw / "uta-3" / "drowsy").glob("*.jpg"))
    assert len(out) == 3
    assert cv2.imread(str(out[0])).shape[:2] == (360, 640)


def test_import_folder_per_person(imp, tmp_path: Path) -> None:
    from training.common import parse_raw_name

    src = tmp_path / "incoming"
    (src / "Rina A").mkdir(parents=True)
    (src / "r02").mkdir()
    cv2.imwrite(str(src / "Rina A" / "IMG_1.png"), np.full((960, 1280, 3), 120, np.uint8))
    cv2.imwrite(str(src / "Rina A" / "IMG_2.jpg"), np.full((480, 640, 3), 90, np.uint8))
    (src / "Rina A" / "catatan.txt").write_text("bukan gambar")
    video = src / "r02" / "dahi.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (640, 480))
    for i in range(30):
        writer.write(np.full((480, 640, 3), i * 5, dtype=np.uint8))
    writer.release()
    cv2.imwrite(str(src / "lepas.jpg"), np.zeros((100, 100, 3), np.uint8))

    raw = tmp_path / "raw"
    counts = imp.import_folder(src, raw, "menguap", "terang", "lain", every_sec=1.0, max_frames=2, width=640)
    assert counts == {"ext-rina-a": 2, "ext-r02": 2, "ext-lain": 1}
    files = sorted((raw / "ext-rina-a" / "menguap").glob("*.jpg"))
    assert len(files) == 2 and cv2.imread(str(files[0])).shape[1] == 640
    meta = parse_raw_name(files[0])
    assert (meta.subject, meta.hint, meta.condition) == ("ext-rina-a", "menguap", "terang")
    # idempotent: running again does not duplicate files
    imp.import_folder(src, raw, "menguap", "terang", "lain", every_sec=1.0, max_frames=2, width=640)
    assert len(list(raw.rglob("*.jpg"))) == 5


def test_import_folder_empty_explains(imp, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="kode_orang"):
        imp.import_folder(tmp_path, tmp_path / "raw", "menguap", "x", "lain", 1.0, 2, 640)


def _rf_export(root: Path, names: list[str], items: dict[str, list[str]]) -> None:
    """Fake Roboflow YOLOv8 export: data.yaml + train/{images,labels}."""
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text("names: [" + ", ".join(names) + "]\nnc: %d\n" % len(names))
    for stem, lines in items.items():
        cv2.imwrite(str(root / "train" / "images" / f"{stem}.jpg"), np.full((480, 640, 3), 100, np.uint8))
        if lines is not None:
            (root / "train" / "labels" / f"{stem}.txt").write_text("\n".join(lines))


def test_roboflow_original_stem(imp) -> None:
    assert imp.roboflow_original_stem("12-MaleGlasses-Yawning_frame3_jpg.rf.3f2a9c") == "12-MaleGlasses-Yawning_frame3"
    assert imp.roboflow_original_stem("plain") == "plain"


def test_import_roboflow_driver_yawn_preset(imp, tmp_path: Path) -> None:
    from training.common import parse_raw_name

    src = tmp_path / "export"
    _rf_export(src, ["close_eyeL", "close_eyeR", "no_yawn", "open_eyeL", "open_eyeR", "yawn"], {
        "3-FemaleGlasses-Yawning_f1_jpg.rf.aa11": ["5 0.5 0.6 0.2 0.2", "3 0.4 0.4 0.1 0.05"],
        "3-FemaleGlasses-Yawning_f2_jpg.rf.bb22": ["0 0.4 0.4 0.1 0.05", "1 0.6 0.4 0.1 0.05"],
        "7-Male-Normal_f9_jpg.rf.cc33": ["2 0.5 0.6 0.2 0.2"],
        "7-Male-Normal_f10_jpg.rf.dd44": [],
    })
    preset = imp.ROBOFLOW_PRESETS["driver-yawn"]
    raw = tmp_path / "raw"
    counts = imp.import_roboflow(src, raw, "driver-yawn", preset["map"], "normal", None, r"^(\d+)-", 640)
    assert counts == {"menguap": 1, "mata_tertutup": 1, "normal": 2}
    files = sorted(raw.rglob("*.jpg"))
    metas = {(parse_raw_name(f).subject, parse_raw_name(f).hint) for f in files}
    assert metas == {("rf-driver-yawn-3", "menguap"), ("rf-driver-yawn-3", "mata_tertutup"), ("rf-driver-yawn-7", "normal")}
    assert not list(raw.rglob("*.txt"))  # boxes are redrawn by preannotate, never copied


def test_import_roboflow_polygons_limit_and_skip(imp, tmp_path: Path) -> None:
    src = tmp_path / "export"
    poly = "1 0.45 0.30 0.50 0.28 0.55 0.31 0.50 0.35"
    _rf_export(src, ["mouth", "yawn", "other"], {f"img{i}": [poly] for i in range(5)} | {"other": ["0 0.5 0.2 0.3 0.1"]})
    raw = tmp_path / "raw"
    counts = imp.import_roboflow(src, raw, "custom-yawn", {"yawn": "menguap"}, None, 3, None, 640)
    assert counts == {"menguap": 3}
    assert {p.parent.parent.name for p in raw.rglob("*.jpg")} == {"rf-custom-yawn"}
    with pytest.raises(SystemExit, match="tidak ada di data.yaml"):
        imp.import_roboflow(src, raw, "custom-yawn", {"yawns": "menguap"}, None, None, None, 640)
