"""Script training YOLOv8n, evaluasi pada subjek uji (test split), benchmark FPS, dan ekspor ONNX."""

import argparse
from pathlib import Path
import shutil
import time
import numpy as np
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = ROOT_DIR / "datasets" / "face_fatigue_yolo" / "data.yaml"
DEFAULT_OUTPUT_MODEL = ROOT_DIR / "models" / "face_expression_yolov8n.pt"


def benchmark_fps(model: YOLO, imgsz: int = 640, num_frames: int = 40) -> tuple[float, float]:
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict(source=dummy, imgsz=imgsz, verbose=False)

    latencies = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        model.predict(source=dummy, imgsz=imgsz, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(latencies))
    fps = 1000.0 / max(0.1, avg_ms)
    return fps, avg_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latih dan evaluasi detektor ekspresi wajah YOLOv8n."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"File {args.data} belum ada. Jalankan scripts/prepare_yolo_dataset.py terlebih dahulu."
        )

    model = YOLO("yolov8n.pt")
    train_results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT_DIR / "runs" / "detect"),
        name="face_fatigue_yolov8n",
        exist_ok=True,
    )

    best_pt = Path(train_results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        DEFAULT_OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, DEFAULT_OUTPUT_MODEL)
        print(f"Bobot terbaik disalin ke: {DEFAULT_OUTPUT_MODEL}")
        trained_model = YOLO(str(DEFAULT_OUTPUT_MODEL))
    else:
        trained_model = model

    print("\nEvaluasi pada split TEST (subjek yang tidak digunakan saat training):")
    metrics = trained_model.val(data=str(args.data), split="test", imgsz=args.imgsz)
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")

    fps, avg_ms = benchmark_fps(trained_model, imgsz=args.imgsz)
    print(f"\nPerforma Inferensi Lokal: {fps:.1f} FPS (rata-rata {avg_ms:.1f} ms/frame)")

    if args.export_onnx:
        onnx_path = trained_model.export(format="onnx", imgsz=args.imgsz)
        print(f"Model ONNX berhasil diekspor ke: {onnx_path}")


if __name__ == "__main__":
    main()
