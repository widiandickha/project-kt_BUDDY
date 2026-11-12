# Stress Buddy: Deteksi Dini Stres & Evaluasi Habit Pengguna PC

Aplikasi desktop yang memantau pengguna PC lewat webcam, mendeteksi tanda lelah atau stres (mata tertutup, menguap, kerutan dahi, tangan ke dahi, postur), lalu memberi umpan balik lewat kucing *Virtual Buddy* di bawah layar. Semua pemrosesan lokal dan frame video tidak pernah disimpan.

> Sistem ini adalah alat bantu kesadaran diri, **bukan alat diagnosis medis**.

Proyek dibangun mengikuti dokumen *Spesifikasi Build* per fase. Prototipe lama tersimpan di `legacy/` sebagai referensi (lihat `docs/AUDIT_SPEC.md` dan `docs/DECISIONS.md`).

## Status fase

| Fase | Isi | Status |
|---|---|---|
| 1 | Setup environment, config, benchmark FPS | ✅ selesai |
| 2 | Dataset & anotasi (tool, pra-anotasi, split, augmentasi) | ✅ tool selesai · ⏳ anotasi manual & rekaman relawan |
| 3 | Training YOLOv8, evaluasi, demo detektor | ✅ pipeline selesai · ⏳ training ulang setelah label final |
| 4 | Analisis & skoring (EAR, PERCLOS, skor, hysteresis, burnout, habit, replay) | ✅ selesai |
| 5 | Virtual Buddy (12 ekspresi, sprite player, state machine, saran, debug panel) | ✅ selesai |
| 6 | Database, ringkasan harian, dashboard, ekspor CSV | ✅ selesai |
| 7 | Integrasi (thread kamera/inferensi, tray, kalibrasi, pengaturan, `src.main`) | ✅ selesai |
| 8 | Pengujian | – |
| 9 | Laporan & demo | – |

## Instalasi

Butuh Python 3.11. Cara termudah di macOS adalah dengan [uv](https://docs.astral.sh/uv/):

```bash
brew install uv
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Tanpa uv (Python 3.10/3.11 sudah terpasang):

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Menjalankan

### Aplikasi lengkap

```bash
.venv/bin/python -m src.main
```

Saat pertama kali dijalankan, muncul disclaimer lalu wizard kalibrasi 20 detik. Kucing duduk di bawah layar, dan ikon tray menunjukkan status kamera (titik merah = kamera aktif). Kamera bisa dijeda dari tray atau lewat klik kanan pada kucing → "Jeda 30 menit".

Opsi:

| Opsi | Fungsi |
|---|---|
| `--source video.mp4` atau `--source folder/` | pakai video atau folder gambar, bukan webcam |
| `--record-features rek.jsonl` | rekam fitur angka (tanpa gambar) untuk replay dan uji event |
| `--seconds 60 --skip-intro` | uji otomatis tanpa dialog |

### Benchmark & tool lain

Benchmark FPS (webcam + `yolov8n.pt` pretrained, 30 detik, jendela live dengan kotak orang dan angka FPS; tekan `q` untuk keluar):

```bash
.venv/bin/python scripts/benchmark_fps.py
```

Opsi yang berguna:

| Opsi | Fungsi |
|---|---|
| `--device cpu` | paksa CPU (imgsz otomatis 480) |
| `--duration 10` | ubah lama benchmark |
| `--source video.mp4` | pakai video atau gambar, bukan webcam |
| `--no-show` | tanpa jendela preview |

Bobot `models/yolov8n.pt` diunduh otomatis pada run pertama. Di macOS, izinkan akses kamera untuk Terminal/IDE di **System Settings → Privacy & Security → Camera**.

Test:

```bash
.venv/bin/python -m pytest
```

### Dataset (FASE 2)

Alur lengkap dan konvensi box ada di [docs/ANOTASI.md](docs/ANOTASI.md). Ringkasnya:

```bash
.venv/bin/python scripts/collect_data.py --relawan andi --kondisi terang     # rekam relawan (dengan persetujuan)
.venv/bin/python scripts/import_public.py nthu --src "~/Downloads/KAGGLE/3. NTHU DDD"
.venv/bin/python scripts/preannotate.py                                      # draf label dari landmark
# koreksi label di Roboflow/CVAT, export YOLO ke data/annotated/
.venv/bin/python training/split_dataset.py --src data/annotated              # split per orang + laporan
.venv/bin/python training/augment.py                                         # augmentasi train
.venv/bin/python training/check_dataset.py                                   # DoD: yolo val
```

Status dataset terbaru ada di [docs/dataset_report.md](docs/dataset_report.md).

### Training & evaluasi (FASE 3)

```bash
.venv/bin/python training/train.py                        # 100 epoch, imgsz 640, batch 16, patience 20 -> models/best.pt
.venv/bin/python training/train.py --model yolov8s.pt     # bila mAP50 < 0.80
.venv/bin/python training/evaluate.py                     # metrik per kelas di split test -> docs/eval/
.venv/bin/python -m src.detection.face_detector --demo    # deteksi live dari webcam (q = keluar)
```

Hasil evaluasi terbaru ada di [docs/eval/metrics.md](docs/eval/metrics.md). Selama label masih draf, angka di sana bukan untuk laporan.

### Analisis (FASE 4)

```bash
.venv/bin/python -m src.analysis.replay --file data/samples/sample.jsonl   # skor per detik dari fitur rekaman
.venv/bin/python -m src.analysis.replay --make-sample data/samples/sample.jsonl
```

Tanpa `models/best.pt`, mata/menguap/kerutan diambil dari landmark MediaPipe (D-016).

### Virtual Buddy (FASE 5)

```bash
.venv/bin/python -m src.buddy.demo                       # kucing di bawah layar + debug panel
.venv/bin/python -m src.buddy.demo --sheet buddy.png     # render 12 ekspresi ke satu gambar
.venv/bin/python scripts/generate_placeholder_sprites.py # buat ulang sprite placeholder (tidak menimpa)
```

Ganti sprite placeholder dengan menimpa `src/buddy/assets/cat/<ekspresi>.png` (PNG RGBA 512×512, kaki di bawah-tengah). Tidak perlu mengubah kode. Saran Buddy ada di `src/buddy/assets/advice_id.yaml`.

### Dashboard (FASE 6)

```bash
.venv/bin/python -m src.ui.dashboard               # data asli (data/stress_buddy.db)
.venv/bin/python -m src.ui.dashboard --demo-data   # database sementara berisi 7 hari data dummy
```

## Struktur

```text
config.yaml            # satu-satunya tempat ambang, bobot, dan parameter
requirements.txt       # versi dipin
docs/                  # DECISIONS.md, AUDIT_SPEC.md, ANOTASI.md, dataset_report.md, eval/
data/raw/              # gambar mentah + draf label (tidak di-commit)
data/annotated/        # hasil export tool anotasi (tidak di-commit)
data/dataset/          # dataset format YOLO (tidak di-commit)
data/data.yaml         # konfigurasi dataset YOLO (4 kelas)
models/                # bobot model (tidak di-commit)
src/
  core/                # config.py, device.py, events.py, (FASE 7) pipeline.py
  detection/           # classes.py, face_detector.py (+ --demo), ear_calculator.py, pose_estimator.py
  analysis/            # feature_extractor, sliding_window, stress_scorer, burnout_trend, habit_evaluator, calibration, engine, replay
  buddy/               # buddy_window, sprite_player, transforms, overlays, buddy_state, advice_engine, speech_bubble, controller, demo, assets/
  storage/             # db (migrasi, retensi, DbWriter), repository, daily_summary, demo_data
  ui/                  # dashboard, theme
  capture/
training/              # common.py, split_dataset.py, augment.py, check_dataset.py, train.py, evaluate.py
scripts/               # benchmark_fps.py, collect_data.py, import_public.py, preannotate.py
tests/
legacy/                # prototipe lama (referensi, tidak dijalankan)
```

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `Gagal membuka sumber video '0'` | Izin kamera belum diberikan, atau kamera sedang dipakai aplikasi lain. Coba `--source 1` bila ada kamera eksternal. |
| `ensurepip` gagal dengan `pyexpat ... Symbol not found` | `python@3.11` dari Homebrew tidak cocok dengan macOS 26. Pakai Python dari `uv` (lihat Instalasi). |
| FPS rendah di CPU | Coba `--imgsz 320`. Target spesifikasi ≥ 10 FPS di CPU kelas menengah. |
