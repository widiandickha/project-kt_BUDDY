"""Alat perekam dan anotasi otomatis berbasis webcam untuk mengumpulkan kelas kerutan_dahi, menguap, dan mata_tertutup."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.detectors.ear_landmark import FaceLandmarkAnalyzer


CLASS_MAP = {
    ord("1"): (0, "mata_tertutup"),
    ord("2"): (1, "menguap"),
    ord("3"): (2, "kerutan_dahi"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rekam dan anotasi sampel wajah secara cepat menggunakan webcam."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT_DIR / "datasets" / "custom_kerutan_dahi",
    )
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    img_dir = args.out_dir / "images"
    lbl_dir = args.out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    analyzer = FaceLandmarkAnalyzer()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Tidak dapat membuka kamera indeks {args.camera}.")
        return

    print("Panduan Tombol Rekam:")
    print("  [1] Simpan sampel kelas 0: mata_tertutup")
    print("  [2] Simpan sampel kelas 1: menguap")
    print("  [3] Simpan sampel kelas 2: kerutan_dahi")
    print("  [q] Keluar")

    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w = frame.shape[:2]
        lm = analyzer.analyze(frame)
        display = frame.copy()

        if lm.bbox is not None:
            x1, y1, x2, y2 = lm.bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (22, 101, 52), 2)

        cv2.putText(
            display,
            f"Tersimpan: {saved_count} | [1] Mata Tertutup [2] Menguap [3] Kerutan Dahi [Q] Keluar",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Anotasi Cepat YOLOv8 - Kucing Teman Kerja", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        if key in CLASS_MAP and lm.bbox is not None:
            cls_id, cls_name = CLASS_MAP[key]
            x1, y1, x2, y2 = lm.bbox
            xc = ((x1 + x2) * 0.5) / w
            yc = ((y1 + y2) * 0.5) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            stem = f"subjlocal_{cls_name}_{stamp}"
            img_path = img_dir / f"{stem}.jpg"
            lbl_path = lbl_dir / f"{stem}.txt"

            cv2.imwrite(str(img_path), frame)
            lbl_path.write_text(
                f"{cls_id} {xc:.5f} {yc:.5f} {bw:.5f} {bh:.5f}\n",
                encoding="utf-8",
            )
            saved_count += 1
            print(f"[{saved_count}] Tersimpan: {img_path.name} ({cls_name})")

    cap.release()
    analyzer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
