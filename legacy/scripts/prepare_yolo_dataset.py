"""Konversi dataset MRL Eye, NTHU-DDD, UTA-RLDD (opsional), dan rekaman kustom ke format YOLOv8.

Pembagian train/val/test dilakukan berdasarkan ID subjek (subject-wise split)
agar evaluasi pada set test benar-benar menguji subjek yang tidak pernah dilihat saat training.
"""

import argparse
from pathlib import Path
import random
import shutil
from typing import List, Optional, Tuple
import cv2
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MRL_DIR = ROOT_DIR / "1. MRL DATASET"
DEFAULT_NTHU_DIR = ROOT_DIR / "3. NTHU DDD"
DEFAULT_CUSTOM_DIR = ROOT_DIR / "datasets" / "custom_kerutan_dahi"
DEFAULT_OUT_DIR = ROOT_DIR / "datasets" / "face_fatigue_yolo"


def detect_face_yolo_box(
    img_bgr,
    face_cascade: Optional[cv2.CascadeClassifier],
) -> Tuple[float, float, float, float]:
    """Mencari bounding box wajah terpusat dalam format normalisasi YOLO (xc, yc, w, h)."""
    h, w = img_bgr.shape[:2]
    if face_cascade is not None and not face_cascade.empty():
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4)
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            xc = (x + fw * 0.5) / w
            yc = (y + fh * 0.5) / h
            bw = min(1.0, (fw * 1.05) / w)
            bh = min(1.0, (fh * 1.05) / h)
            return (
                round(max(0.0, min(1.0, xc)), 5),
                round(max(0.0, min(1.0, yc)), 5),
                round(max(0.05, min(1.0, bw)), 5),
                round(max(0.05, min(1.0, bh)), 5),
            )
    return (0.5, 0.52, 0.58, 0.68)


def nthu_subject_split(filename: str) -> str:
    """Memetakan ID subjek NTHU-DDD (001, 002, 005, 006) ke train, val, atau test."""
    subj = filename.split("_")[0].strip()
    if subj in ("001", "002"):
        return "train"
    if subj == "005":
        return "val"
    if subj == "006":
        return "test"
    # Subjek tambahan dipetakan secara deterministik berdasarkan hash nama subjek
    bucket = sum(ord(c) for c in subj) % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "val"
    return "test"


def mrl_subject_split(filename: str) -> str:
    """Memetakan ID subjek MRL (s0001..s0037) ke train, val, atau test tanpa kebocoran subjek."""
    subj = filename.split("_")[0].strip().lstrip("sS")
    try:
        num = int(subj)
    except ValueError:
        return "train"
    if num <= 26:
        return "train"
    if num <= 32:
        return "val"
    return "test"


def save_yolo_sample(
    src_img_path: Path,
    out_root: Path,
    split: str,
    prefix: str,
    class_id: Optional[int],
    bbox: Tuple[float, float, float, float],
) -> bool:
    img_out_dir = out_root / "images" / split
    lbl_out_dir = out_root / "labels" / split
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{prefix}_{src_img_path.stem}"
    dst_img = img_out_dir / f"{stem}{src_img_path.suffix}"
    dst_lbl = lbl_out_dir / f"{stem}.txt"

    shutil.copy2(src_img_path, dst_img)
    if class_id is None:
        # Sampel negatif (wajah normal/terjaga) menggunakan file label kosong di YOLOv8.
        dst_lbl.write_text("", encoding="utf-8")
    else:
        xc, yc, bw, bh = bbox
        dst_lbl.write_text(f"{class_id} {xc:.5f} {yc:.5f} {bw:.5f} {bh:.5f}\n", encoding="utf-8")
    return True


def process_nthu(
    nthu_dir: Path,
    out_root: Path,
    max_per_folder: int,
    face_cascade: cv2.CascadeClassifier,
    rng: random.Random,
) -> dict:
    counts = {"train": 0, "val": 0, "test": 0}
    train_root = nthu_dir / "Multi class" / "train"
    if not train_root.exists():
        print(f"[INFO] Folder NTHU-DDD tidak ditemukan di: {train_root}")
        return counts

    folder_specs = [
        (train_root / "drowsy" / "slowBlinkWithNodding", 0, "nthu_blink"),
        (train_root / "drowsy" / "sleepyCombination", 0, "nthu_sleepy"),
        (train_root / "drowsy" / "yawning", 1, "nthu_yawn"),
        (train_root / "notdrowsy", None, "nthu_awake"),
    ]

    for folder, cls_id, prefix in folder_specs:
        if not folder.exists():
            continue
        files = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        if len(files) > max_per_folder:
            files = rng.sample(files, max_per_folder)

        for img_path in files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            split = nthu_subject_split(img_path.name)
            bbox = detect_face_yolo_box(img, face_cascade) if cls_id is not None else (0.5, 0.5, 0.5, 0.5)
            save_yolo_sample(img_path, out_root, split, prefix, cls_id, bbox)
            counts[split] += 1

    return counts


def process_mrl(
    mrl_dir: Path,
    out_root: Path,
    max_per_group: int,
    rng: random.Random,
) -> dict:
    counts = {"train": 0, "val": 0, "test": 0}
    data_root = mrl_dir / "data"
    if not data_root.exists():
        print(f"[INFO] Folder MRL Dataset tidak ditemukan di: {data_root}")
        return counts

    for orig_split in ("train", "val", "test"):
        for sub_name, cls_id in (("sleepy", 0), ("awake", None)):
            folder = data_root / orig_split / sub_name
            if not folder.exists():
                continue
            files = sorted(
                [p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
            )
            if len(files) > max_per_group:
                files = rng.sample(files, max_per_group)

            for img_path in files:
                split = mrl_subject_split(img_path.name)
                # Gambar MRL sudah berupa potongan area mata sehingga bbox mencakup hampir seluruh frame.
                bbox = (0.5, 0.5, 0.90, 0.90)
                save_yolo_sample(
                    img_path,
                    out_root,
                    split,
                    f"mrl_{sub_name}",
                    cls_id,
                    bbox,
                )
                counts[split] += 1

    return counts


def process_uta_videos(
    uta_dir: Optional[Path],
    out_root: Path,
    every_n_frames: int,
    max_frames_per_video: int,
    face_cascade: cv2.CascadeClassifier,
) -> dict:
    counts = {"train": 0, "val": 0, "test": 0}
    if uta_dir is None or not uta_dir.exists():
        return counts

    videos = sorted(
        [
            p
            for p in uta_dir.rglob("*")
            if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
        ]
    )
    for idx, vid_path in enumerate(videos):
        # Pembagian berdasarkan urutan video/subjek agar subjek terpisah antar split
        split = "train" if (idx % 5 < 3) else ("val" if idx % 5 == 3 else "test")
        stem_low = vid_path.stem.lower()
        if "drowsy" in stem_low or "sleep" in stem_low or stem_low.endswith("10"):
            cls_id: Optional[int] = 0
        elif "alert" in stem_low or "awake" in stem_low or stem_low.endswith("0"):
            cls_id = None
        else:
            continue

        cap = cv2.VideoCapture(str(vid_path))
        frame_idx = 0
        saved_from_vid = 0
        while cap.isOpened() and saved_from_vid < max_frames_per_video:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_idx += 1
            if frame_idx % every_n_frames != 0:
                continue

            img_out_dir = out_root / "images" / split
            lbl_out_dir = out_root / "labels" / split
            img_out_dir.mkdir(parents=True, exist_ok=True)
            lbl_out_dir.mkdir(parents=True, exist_ok=True)

            name_base = f"uta_{vid_path.parent.name}_{vid_path.stem}_f{frame_idx:05d}"
            dst_img = img_out_dir / f"{name_base}.jpg"
            dst_lbl = lbl_out_dir / f"{name_base}.txt"
            cv2.imwrite(str(dst_img), frame)

            if cls_id is None:
                dst_lbl.write_text("", encoding="utf-8")
            else:
                xc, yc, bw, bh = detect_face_yolo_box(frame, face_cascade)
                dst_lbl.write_text(
                    f"{cls_id} {xc:.5f} {yc:.5f} {bw:.5f} {bh:.5f}\n",
                    encoding="utf-8",
                )
            saved_from_vid += 1
            counts[split] += 1
        cap.release()

    return counts


def process_custom_annotations(custom_dir: Path, out_root: Path) -> dict:
    counts = {"train": 0, "val": 0, "test": 0}
    img_dir = custom_dir / "images"
    lbl_dir = custom_dir / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        return counts

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    for idx, img_path in enumerate(images):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        split = "train" if (idx % 5 < 3) else ("val" if idx % 5 == 3 else "test")
        dst_img = out_root / "images" / split / f"custom_{img_path.name}"
        dst_lbl = out_root / "labels" / split / f"custom_{img_path.stem}.txt"
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dst_img)
        shutil.copy2(lbl_path, dst_lbl)
        counts[split] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Siapkan dataset YOLOv8 dari 1. MRL DATASET, 3. NTHU DDD, dan opsional UTA-RLDD."
    )
    parser.add_argument("--mrl-dir", type=Path, default=DEFAULT_MRL_DIR)
    parser.add_argument("--nthu-dir", type=Path, default=DEFAULT_NTHU_DIR)
    parser.add_argument(
        "--uta-dir",
        type=Path,
        default=None,
        help="Path eksternal ke folder video UTA-RLDD (tidak perlu dicopy ke dalam proyek).",
    )
    parser.add_argument("--custom-dir", type=Path, default=DEFAULT_CUSTOM_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--max-nthu-per-folder",
        type=int,
        default=400,
        help="Jumlah sampel maksimum per sub-folder NTHU-DDD agar ukuran dataset seimbang.",
    )
    parser.add_argument(
        "--max-mrl-per-group",
        type=int,
        default=250,
        help="Jumlah sampel maksimum per grup MRL Eye Dataset.",
    )
    parser.add_argument(
        "--two-class",
        action="store_true",
        help="Gunakan skema 2 kelas (mata_tertutup, menguap) tanpa kerutan_dahi.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    xml_path = (
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if hasattr(cv2, "data")
        else Path("")
    )
    face_cascade = cv2.CascadeClassifier(str(xml_path)) if xml_path.exists() else None

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    print("Memproses 3. NTHU DDD (pembagian per subjek)...")
    c_nthu = process_nthu(
        args.nthu_dir,
        args.out_dir,
        args.max_nthu_per_folder,
        face_cascade,
        rng,
    )
    print(f"  Hasil NTHU-DDD: {c_nthu}")

    print("Memproses 1. MRL DATASET (pembagian per subjek)...")
    c_mrl = process_mrl(
        args.mrl_dir,
        args.out_dir,
        args.max_mrl_per_group,
        rng,
    )
    print(f"  Hasil MRL Eye: {c_mrl}")

    if args.uta_dir is not None:
        print(f"Memproses video UTA-RLDD dari path eksternal: {args.uta_dir} ...")
        c_uta = process_uta_videos(args.uta_dir, args.out_dir, 30, 40, face_cascade)
        print(f"  Hasil UTA-RLDD: {c_uta}")

    if not args.two_class:
        c_custom = process_custom_annotations(args.custom_dir, args.out_dir)
        if sum(c_custom.values()) > 0:
            print(f"  Hasil data kustom kerutan_dahi: {c_custom}")

    names_map = (
        {0: "mata_tertutup", 1: "menguap"}
        if args.two_class
        else {0: "mata_tertutup", 1: "menguap", 2: "kerutan_dahi"}
    )
    data_yaml = {
        "path": str(args.out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names_map,
    }
    yaml_path = args.out_dir / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data_yaml, fh, sort_keys=False, allow_unicode=True)

    print(f"\nDataset YOLOv8 siap digunakan di: {yaml_path}")


if __name__ == "__main__":
    main()
