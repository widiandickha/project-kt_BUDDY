# Kucing Teman Kerja: Pemantau Kelelahan & Stres Pengguna PC

Aplikasi desktop berbasis Python dan PySide6 yang memantau tanda kelelahan dan stres visual secara pasif melalui webcam lokal, menghitung skor kelelahan dalam jendela geser (*sliding window*) 60 detik, menampilkan pengingat lewat karakter kucing *Virtual Buddy*, dan menyimpan riwayat angka indikator ke database SQLite untuk evaluasi harian maupun mingguan.

Seluruh pemrosesan berjalan 100% lokal di perangkat pengguna dan **tidak ada gambar atau rekaman video yang disimpan**.

---

## 1. Struktur Folder Proyek

```text
KT/
├── 1. MRL DATASET/               # Dataset 1: MRL Eye Dataset (sudah terpasang di root)
├── 3. NTHU DDD/                  # Dataset 3: NTHU-DDD Multi-Class (sudah terpasang di root)
├── config.yaml                   # Konfigurasi ambang skor, bobot indikator, cooldown, dan model
├── DESIGN.md                     # Dokumen arah desain, palet kontras WCAG AA, dan alasan keputusan visual
├── main.py                       # Titik masuk utama aplikasi desktop
├── requirements.txt              # Daftar pustaka Python yang dibutuhkan
├── data/
│   └── wellness_buddy.db         # Database SQLite otomatis (sessions, score_log, events, buddy_log)
├── models/
│   ├── yolov8n-pose.pt           # Model pose tubuh bawaan (diunduh otomatis saat pertama jalan)
│   └── face_expression_yolov8n.pt # Model hasil training ekspresi wajah kustom
├── scripts/
│   ├── prepare_yolo_dataset.py   # Konversi dataset MRL, NTHU-DDD, dan UTA-RLDD ke format YOLOv8
│   ├── annotate_custom_webcam.py # Alat rekam & anotasi cepat webcam untuk kelas kerutan_dahi
│   └── train_and_evaluate.py     # Script training YOLOv8n, evaluasi test split per subjek, & ekspor ONNX
└── src/
    ├── config.py                 # Pembaca dan penyimpan config.yaml
    ├── database.py               # Pengelola SQLite thread-safe
    ├── detectors/                # Modul EAR/MAR/alis MediaPipe, YOLOv8 wajah, dan YOLOv8-pose
    ├── analysis/                 # Kalibrasi baseline awal sesi, sliding window 60d, & deteksi burnout
    ├── buddy/                    # 5 ekspresi kucing, pesan saran, dan decision engine (cooldown/snooze)
    └── ui/                       # Jendela utama PySide6, overlay Buddy, grafik tren, dan system tray
```

---

## 2. Cara Menjalankan Aplikasi

Semua dependensi telah diinstal di dalam virtual environment `.venv` (Python 3.12).

### Menjalankan dengan Webcam Asli
```bash
.venv/bin/python main.py
```

### Menjalankan dalam Mode Simulasi (Tanpa Kamera Fisik)
Gunakan opsi `--simulate` untuk menguji siklus perubahan skor (Normal, Mengantuk/Menguap, Dahi Tegang, hingga Kelelahan Tinggi), pemicu *Virtual Buddy*, dan grafik evaluasi tanpa perlu menyalakan webcam:
```bash
.venv/bin/python main.py --simulate
```

---

## 3. Tutorial Menautkan Dataset & Menyiapkan Format YOLOv8

Folder `1. MRL DATASET` dan `3. NTHU DDD` yang sudah kamu tempatkan di root proyek langsung dikenali secara otomatis oleh script `scripts/prepare_yolo_dataset.py`.

### Pemetaan Kelas & Pembagian Subjek (Tanpa Kebocoran Data)
Sesuai kebutuhan evaluasi pada Bab IV (pengujian wajib dilakukan pada subjek yang tidak dipakai saat training), script melakukan pembagian *subject-wise split* berdasarkan prefiks nama file:
1. **Dataset 3 (`3. NTHU DDD/Multi class/train/`)**:
   - Subjek `001` dan `002` masuk ke `train`, subjek `005` masuk ke `val`, dan subjek `006` masuk ke `test`.
   - Subfolder `drowsy/slowBlinkWithNodding` dan `drowsy/sleepyCombination` dipetakan ke kelas `0` (`mata_tertutup`).
   - Subfolder `drowsy/yawning` dipetakan ke kelas `1` (`menguap`).
   - Subfolder `notdrowsy` dijadikan sampel negatif (file label `.txt` kosong) agar model mengenali wajah normal dan menekan *false alarm* per jam.
2. **Dataset 1 (`1. MRL DATASET/data/`)**:
   - Subjek `s0001` sampai `s0026` masuk ke `train`, `s0027` sampai `s0032` ke `val`, dan `s0033` sampai `s0037` ke `test`.
   - Subfolder `sleepy` dipetakan ke kelas `0` (`mata_tertutup`), sedangkan `awake` menjadi sampel negatif.

### Menjalankan Konversi Dataset 1 & Dataset 3 (Langsung Jalan)
Jalankan perintah berikut dari root proyek untuk membuat dataset siap latih di `datasets/face_fatigue_yolo/`:
```bash
.venv/bin/python scripts/prepare_yolo_dataset.py --max-nthu-per-folder 400 --max-mrl-per-group 250
```
Setelah selesai, file konfigurasi `datasets/face_fatigue_yolo/data.yaml` akan terbentuk otomatis.

### Cara Menautkan Dataset 2 (`UTA-RLDD`) Tanpa Menyalin ke Folder Proyek
Karena ukuran `UTA-RLDD` sangat besar, kamu **tidak perlu menyalinnya** ke dalam folder proyek `KT`. Cukup simpan di *hard disk* eksternal atau folder `Downloads`, lalu tautkan dengan salah satu dari dua cara berikut:

1. **Cara 1: Berikan argumen `--uta-dir` saat menjalankan script konversi**
   Script hanya akan mengekstrak sebagian frame penting (misalnya setiap 30 frame) ke ukuran kecil di `datasets/face_fatigue_yolo/`:
   ```bash
   .venv/bin/python scripts/prepare_yolo_dataset.py \
     --uta-dir "/Volumes/DriveEksternal/uta-reallife-drowsiness-dataset"
   ```
2. **Cara 2: Buat Symbolic Link (Tautan Simbolik) di macOS**
   Jika ingin meletakkan pintasan folder di dalam proyek tanpa memakai ruang penyimpanan ganda:
   ```bash
   ln -s "/Volumes/DriveEksternal/uta-reallife-drowsiness-dataset" "2. UTA RLDD"
   ```

---

## 4. Menangani Kelas `kerutan_dahi` & Rencana Cadangan

Proyek ini menyediakan dua jalur yang berjalan berdampingan untuk indikator `kerutan_dahi`:

### Jalur A: Merekam Data Kustom `kerutan_dahi` via Webcam
Jalankan alat bantu perekam berikut untuk mengambil sampel wajah sendiri beserta kotak pembatas (*bounding box*) otomatis dari MediaPipe:
```bash
.venv/bin/python scripts/annotate_custom_webcam.py
```
- Tekan tombol `1` untuk menyimpan sampel `mata_tertutup`.
- Tekan tombol `2` untuk menyimpan sampel `menguap`.
- Tekan tombol `3` saat mengerutkan dahi untuk menyimpan sampel `kerutan_dahi`.
- Tekan `q` untuk keluar, lalu jalankan ulang `scripts/prepare_yolo_dataset.py` agar rekaman kustom digabungkan ke `datasets/face_fatigue_yolo/`.

### Jalur B: Mode 2 Kelas + Pengukuran Jarak Alis Landmark (Rencana Cadangan)
Jika waktu pengumpulan data `kerutan_dahi` terbatas:
1. Buat dataset 2 kelas (`mata_tertutup` dan `menguap`):
   ```bash
   .venv/bin/python scripts/prepare_yolo_dataset.py --two-class
   ```
2. Di aplikasi utama, buka tab **Pengaturan & Privasi** lalu centang **Gunakan mode 2 kelas**. Aplikasi akan mendeteksi ketegangan dahi menggunakan rasio penyempitan jarak alis ke mata dari *landmark* MediaPipe terhadap *baseline* awal sesi.

---

## 5. Training Model YOLOv8n, Evaluasi, & Ekspor ONNX

Setelah `datasets/face_fatigue_yolo/data.yaml` terbentuk, jalankan training dan evaluasi otomatis:
```bash
.venv/bin/python scripts/train_and_evaluate.py --epochs 25 --imgsz 640 --batch 16 --device mps --export-onnx
```
Script ini akan:
1. Melatih `yolov8n.pt` dan menyalin bobot terbaik (`best.pt`) ke `models/face_expression_yolov8n.pt`.
2. Mengevaluasi model pada split `test` (subjek yang tidak dipakai saat training) serta mencetak nilai `mAP50`, `mAP50-95`, `Precision`, dan `Recall`. Confusion matrix tersimpan otomatis di folder `runs/detect/face_fatigue_yolov8n/`.
3. Mengukur rata-rata FPS dan latensi inferensi per frame.
4. Mengekspor model ke `models/face_expression_yolov8n.onnx` untuk akselerasi inferensi CPU.

Begitu file `models/face_expression_yolov8n.pt` atau `.onnx` tersedia, aplikasi desktop `main.py` otomatis memuatnya pada sesi berikutnya.

---

## 6. Batasan Sistem (Untuk Dokumentasi Laporan Bab I-V)

- **Indikator Visual, Bukan Diagnosis Medis**: Skor stres (0 sampai 100) dihitung dari frekuensi mata tertutup, menguap, penyempitan jarak alis, postur membungkuk, serta gerakan tangan ke kepala. Angka ini adalah pengingat ergonomi dan istirahat kerja, bukan alat diagnosis klinis.
- **Privasi Data Lokal**: Kamera hanya membaca matriks piksel di memori RAM sementara untuk diekstrak menjadi angka rasio geometri. Database SQLite (`data/wellness_buddy.db`) hanya mencatat stempel waktu dan angka indikator.
