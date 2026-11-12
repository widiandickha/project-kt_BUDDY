#!/usr/bin/env python3
"""FASE 1 benchmark: run pretrained yolov8n on the webcam and report average FPS.

Usage:
    python scripts/benchmark_fps.py                    # webcam from config.yaml, 30 s, live window
    python scripts/benchmark_fps.py --device cpu       # force CPU
    python scripts/benchmark_fps.py --source clip.mp4  # video/image instead of webcam
    python scripts/benchmark_fps.py --no-show          # headless (no preview window)

Frames are only held in memory; nothing is written to disk (privacy rule).
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.core.config import load_config, resolve_path  # noqa: E402
from src.core.device import select_device  # noqa: E402

logger = logging.getLogger("benchmark_fps")

PERSON_CLASS_ID = 0  # COCO "person"
CPU_FALLBACK_IMGSZ = 480  # spec FASE 1: CPU fallback uses imgsz=480
WINDOW_NAME = "benchmark_fps (q = keluar)"
BOX_COLOR = (52, 101, 22)  # BGR
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (20, 20, 20)


@dataclass
class BenchmarkResult:
    """Summary of a benchmark run."""

    device: str
    imgsz: int
    frames: int
    duration_sec: float
    avg_fps: float
    infer_ms_p50: float
    infer_ms_p95: float
    persons_max: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=30.0, help="lama benchmark dalam detik (default 30)")
    ap.add_argument("--model", default="models/yolov8n.pt", help="bobot YOLO pretrained (diunduh otomatis bila belum ada)")
    ap.add_argument("--device", default=None, help="auto | cuda | mps | cpu (default: inference.device di config)")
    ap.add_argument("--imgsz", type=int, default=None, help="ukuran input (default: inference.imgsz di config)")
    ap.add_argument("--source", default=None, help="indeks kamera atau path video/gambar (default: camera.index)")
    ap.add_argument("--no-show", action="store_true", help="jangan tampilkan jendela preview")
    ap.add_argument("--config", default=None, help="path config.yaml alternatif")
    return ap.parse_args(argv)


def open_source(source: str | int, width: int, height: int) -> cv2.VideoCapture | np.ndarray:
    """Open a webcam index or video file; a still image is returned as an array to loop on."""
    if isinstance(source, str) and not source.isdigit():
        path = Path(source)
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            img = cv2.imread(str(path))
            if img is None:
                raise SystemExit(f"Gagal membaca gambar: {path}")
            return img
        cap = cv2.VideoCapture(str(path))
    else:
        index = int(source)
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(
            f"Gagal membuka sumber video '{source}'. Di macOS, izinkan akses kamera untuk "
            "Terminal/IDE di System Settings > Privacy & Security > Camera."
        )
    return cap


def read_frame(src: cv2.VideoCapture | np.ndarray) -> np.ndarray | None:
    """Read the next frame; still images and finished videos loop."""
    if isinstance(src, np.ndarray):
        return src.copy()
    ok, frame = src.read()
    if ok:
        return frame
    if src.get(cv2.CAP_PROP_FRAME_COUNT) > 0:  # video file: rewind
        src.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = src.read()
        return frame if ok else None
    return None


def draw_overlay(frame: np.ndarray, boxes: list[tuple[int, int, int, int, float]], fps: float, device: str) -> None:
    """Draw person boxes and the FPS label in place."""
    for x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(frame, f"person {conf:.2f}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 2)
    label = f"FPS {fps:5.1f} | {device}"
    cv2.rectangle(frame, (8, 8), (230, 36), TEXT_BG, -1)
    cv2.putText(frame, label, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)


def run(args: argparse.Namespace) -> BenchmarkResult:
    """Run the benchmark loop and return the summary."""
    from ultralytics import YOLO

    cfg = load_config(args.config)
    info = select_device(args.device or cfg["inference"]["device"])
    imgsz = args.imgsz or int(cfg["inference"]["imgsz"])
    if info.device == "cpu" and args.imgsz is None:
        imgsz = min(imgsz, CPU_FALLBACK_IMGSZ)
    logger.info("Device: %s (cuda=%s, mps=%s), imgsz=%d", info.device, info.cuda_available, info.mps_available, imgsz)

    model_path = resolve_path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Official Ultralytics weights are fetched into models/ on first run if absent.
    from ultralytics.utils.downloads import attempt_download_asset

    model = YOLO(str(attempt_download_asset(str(model_path))))

    cam = cfg["camera"]
    source = args.source if args.source is not None else str(cam["index"])
    src = open_source(source, int(cam["width"]), int(cam["height"]))

    infer_ms: list[float] = []
    persons_max = 0
    frames = 0
    show = not args.no_show
    fps_live = 0.0

    # Warm-up so model load / first-call overhead does not skew the average.
    warm = read_frame(src)
    if warm is None:
        raise SystemExit("Sumber video tidak menghasilkan frame.")
    model.predict(warm, imgsz=imgsz, device=info.device, classes=[PERSON_CLASS_ID], verbose=False)

    start = time.perf_counter()
    try:
        while time.perf_counter() - start < args.duration:
            frame = read_frame(src)
            if frame is None:
                logger.warning("Frame kosong, berhenti.")
                break
            t0 = time.perf_counter()
            res = model.predict(frame, imgsz=imgsz, device=info.device, classes=[PERSON_CLASS_ID], verbose=False)[0]
            dt = time.perf_counter() - t0
            infer_ms.append(dt * 1000.0)
            frames += 1

            boxes = [
                (*map(int, b.xyxy[0].tolist()), float(b.conf[0]))
                for b in (res.boxes if res.boxes is not None else [])
            ]
            persons_max = max(persons_max, len(boxes))
            elapsed = time.perf_counter() - start
            fps_live = frames / elapsed if elapsed > 0 else 0.0

            if show:
                draw_overlay(frame, boxes, fps_live, info.device)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if isinstance(src, cv2.VideoCapture):
            src.release()
        if show:
            cv2.destroyAllWindows()

    duration = time.perf_counter() - start
    ordered = sorted(infer_ms) or [0.0]
    p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return BenchmarkResult(
        device=info.device,
        imgsz=imgsz,
        frames=frames,
        duration_sec=duration,
        avg_fps=frames / duration if duration > 0 else 0.0,
        infer_ms_p50=statistics.median(ordered),
        infer_ms_p95=ordered[p95_idx],
        persons_max=persons_max,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = run(parse_args(argv))
    print(
        f"\n=== benchmark_fps ===\n"
        f"device       : {result.device}\n"
        f"imgsz        : {result.imgsz}\n"
        f"frames       : {result.frames} in {result.duration_sec:.1f} s\n"
        f"FPS rata-rata: {result.avg_fps:.1f}\n"
        f"latensi inferensi p50/p95: {result.infer_ms_p50:.1f} / {result.infer_ms_p95:.1f} ms\n"
        f"orang terdeteksi (maks/frame): {result.persons_max}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
