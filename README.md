# Stress Buddy: Deteksi Dini Stres & Evaluasi Habit Pengguna PC

Aplikasi desktop yang memantau pengguna PC lewat webcam, mendeteksi tanda lelah atau stres (mata tertutup, menguap, kerutan dahi, tangan ke dahi, postur), lalu memberi umpan balik lewat kucing *Virtual Buddy* di bawah layar. Semua pemrosesan lokal dan frame video tidak pernah disimpan.

> Sistem ini adalah alat bantu kesadaran diri, **bukan alat diagnosis medis**.

Proyek dibangun mengikuti dokumen *Spesifikasi Build* per fase. Prototipe lama tersimpan di `legacy/` sebagai referensi (lihat `docs/AUDIT_SPEC.md` dan `docs/DECISIONS.md`).

## Status fase

| Fase | Isi | Status |
|---|---|---|
| 1 | Setup environment, config, benchmark FPS | ✅ selesai |
| 2 | Dataset & anotasi (tool, pra-anotasi, split, augmentasi) | ✅ tool selesai · ⏳ anotasi manual & rekaman relawan |
| 3 | Training YOLOv8 | ⏳ berikutnya |
| 4 | Analisis & skoring | – |
| 5 | Virtual Buddy | – |
| 6 | Database & ringkasan harian | – |
| 7 | Integrasi | – |
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

## Struktur

```text
config.yaml            # satu-satunya tempat ambang, bobot, dan parameter
requirements.txt       # versi dipin
docs/                  # DECISIONS.md, AUDIT_SPEC.md, ANOTASI.md, dataset_report.md, (nanti) eval/
data/raw/              # gambar mentah + draf label (tidak di-commit)
data/annotated/        # hasil export tool anotasi (tidak di-commit)
data/dataset/          # dataset format YOLO (tidak di-commit)
data/data.yaml         # konfigurasi dataset YOLO (4 kelas)
models/                # bobot model (tidak di-commit)
src/
  core/                # config.py, device.py, (FASE 4/7) events.py, pipeline.py
  detection/classes.py # urutan kelas final
  capture/  analysis/  buddy/  storage/  ui/
training/              # common.py, split_dataset.py, augment.py, check_dataset.py, (FASE 3) train.py, evaluate.py
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
