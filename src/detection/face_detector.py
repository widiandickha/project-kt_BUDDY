"""YOLO face-state detector (mata_tertutup, mata_terbuka, menguap, kerutan_dahi).

Runtime use:
    detector = FaceDetector(load_config())
    result = detector.predict(frame_bgr)   # FaceDetections with per-class confidence 0..1

Live demo (FASE 3 DoD):
    python -m src.detection.face_detector --demo
    python -m src.detection.face_detector --demo --source clip.mp4 --duration 10

Frames are processed in memory only and never written to disk.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.config import load_config, resolve_path
from src.core.device import select_device
from src.detection.classes import CLASS_NAMES

logger = logging.getLogger(__name__)

# BGR colors per class for the demo overlay.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "mata_tertutup": (60, 60, 230),
    "mata_terbuka": (60, 190, 60),
    "menguap": (230, 140, 40),
    "kerutan_dahi": (200, 60, 200),
}


@dataclass(frozen=True)
class Detection:
    """One detected box in pixel coordinates."""

    cls_name: str
    conf: float
    xyxy: tuple[int, int, int, int]


@dataclass
class FaceDetections:
    """Per-frame detector output: highest confidence per class (0 if absent) and all boxes."""

    confidences: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in CLASS_NAMES})
    boxes: list[Detection] = field(default_factory=list)

    def conf(self, name: str) -> float:
        """Confidence of a class, 0.0 when not detected."""
        return self.confidences.get(name, 0.0)


def to_gray3(frame_bgr: np.ndarray) -> np.ndarray:
    """3-channel grayscale copy, matching the training input (D-013)."""
    return cv2.cvtColor(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def parse_detections(
    names: dict[int, str], cls_ids: np.ndarray, confs: np.ndarray, xyxy: np.ndarray
) -> FaceDetections:
    """Turn raw YOLO arrays into FaceDetections; unknown class names are ignored."""
    out = FaceDetections()
    for cid, conf, box in zip(cls_ids.astype(int), confs.astype(float), xyxy, strict=True):
        name = names.get(int(cid))
        if name not in out.confidences:
            continue
        out.boxes.append(Detection(name, float(conf), tuple(int(v) for v in box)))  # type: ignore[arg-type]
        out.confidences[name] = max(out.confidences[name], float(conf))
    out.boxes.sort(key=lambda d: -d.conf)
    return out


class FaceDetector:
    """Load the trained weights (det.model) and run inference with config thresholds."""

    def __init__(self, cfg: dict[str, Any], weights: str | Path | None = None, device: str | None = None) -> None:
        from ultralytics import YOLO

        det = cfg["det"]
        path = resolve_path(weights or det["model"])
        if not path.exists():
            raise FileNotFoundError(f"Bobot detektor tidak ada: {path}. Jalankan training/train.py (FASE 3).")
        self.model = YOLO(str(path))
        self.conf = float(det["conf"])
        self.iou = float(det["iou"])
        self.grayscale = bool(det.get("grayscale", False))
        self.imgsz = int(cfg["inference"]["imgsz"])
        self.device = select_device(device or cfg["inference"]["device"]).device
        names = self.model.names
        if tuple(names[i] for i in range(len(names))) != CLASS_NAMES:
            raise ValueError(f"Kelas model {names} tidak sama dengan {CLASS_NAMES}")

    def predict(self, frame_bgr: np.ndarray) -> FaceDetections:
        """Detect face states in one BGR frame."""
        src = to_gray3(frame_bgr) if self.grayscale else frame_bgr
        res = self.model.predict(src, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return FaceDetections()
        return parse_detections(
            res.names, res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.xyxy.cpu().numpy()
        )


def draw_detections(frame: np.ndarray, result: FaceDetections, fps: float) -> None:
    """Draw boxes, per-class confidence bars and FPS in place (demo only)."""
    for d in result.boxes:
        color = CLASS_COLORS.get(d.cls_name, (255, 255, 255))
        x1, y1, x2, y2 = d.xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{d.cls_name} {d.conf:.2f}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.rectangle(frame, (8, 8), (230, 30 + 20 * len(CLASS_NAMES)), (20, 20, 20), -1)
    cv2.putText(frame, f"FPS {fps:5.1f}", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    for i, name in enumerate(CLASS_NAMES):
        y = 46 + 20 * i
        conf = result.conf(name)
        cv2.rectangle(frame, (120, y - 10), (120 + int(100 * conf), y), CLASS_COLORS[name], -1)
        cv2.putText(frame, name, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def run_demo(args: argparse.Namespace) -> int:
    """Live demo: webcam (or video/image) with detected boxes and FPS."""
    cfg = load_config()
    detector = FaceDetector(cfg, weights=args.weights, device=args.device)
    cam = cfg["camera"]
    source = args.source if args.source is not None else str(cam["index"])
    still: np.ndarray | None = None
    cap: cv2.VideoCapture | None = None
    if not source.isdigit() and Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        still = cv2.imread(source)
        if still is None:
            print(f"Gagal membaca gambar {source}")
            return 1
    else:
        if source.isdigit():
            cap = cv2.VideoCapture(int(source), cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam["width"]))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam["height"]))
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Kamera/video tidak bisa dibuka. Cek izin kamera di System Settings > Privacy & Security > Camera.")
            return 1

    window = "face_detector demo (q = keluar)"
    frames, start = 0, time.perf_counter()
    seen = {name: 0 for name in CLASS_NAMES}
    try:
        while args.duration is None or time.perf_counter() - start < args.duration:
            if still is not None:
                frame = still.copy()
            else:
                ok, frame = cap.read()  # type: ignore[union-attr]
                if not ok or frame is None:
                    break
                if source.isdigit():
                    frame = cv2.flip(frame, 1)
            result = detector.predict(frame)
            frames += 1
            for name in CLASS_NAMES:
                seen[name] += int(result.conf(name) > 0)
            fps = frames / max(1e-6, time.perf_counter() - start)
            if not args.no_show:
                draw_detections(frame, result, fps)
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if cap is not None:
            cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - start
    print(f"{frames} frame dalam {elapsed:.1f} s ({frames / max(elapsed, 1e-6):.1f} FPS) di {detector.device}")
    print("frame dengan deteksi per kelas:", seen)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="tampilkan deteksi live")
    ap.add_argument("--source", default=None, help="indeks kamera atau path video/gambar")
    ap.add_argument("--weights", default=None, help="default det.model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--duration", type=float, default=None, help="detik; default sampai tekan q")
    ap.add_argument("--no-show", action="store_true", help="tanpa jendela (cek headless)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m src.detection.face_detector`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    if not args.demo:
        print("Gunakan --demo untuk menampilkan deteksi live.")
        return 2
    return run_demo(args)


if __name__ == "__main__":
    sys.exit(main())
