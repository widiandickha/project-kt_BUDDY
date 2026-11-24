# Panduan Dataset & Training (untuk yang membantu)

Panduan ini untuk teman yang mengurus **dataset sekaligus training** detektor wajah Stress Buddy, mulai dari mengunduh data mentah sampai mengirim balik `best.pt`. Semua perintah dijalankan dari folder repo.

Detektor mengenali 4 kelas. Box-nya digambar menurut konvensi di [ANOTASI.md](ANOTASI.md) bagian 3.

| ID | Kelas | Sekarang | Target |
|---|---|---|---|
| 0 | `mata_tertutup` | cukup | ≥ 500 instance |
| 1 | `mata_terbuka` | cukup | ≥ 500 instance |
| 2 | `menguap` | 328 | ≥ 500 instance |
| 3 | `kerutan_dahi` | **0** | ≥ 500 instance, muncul juga di split val & test |
| – | jumlah orang | 4 | ≥ 5 (makin banyak makin bagus) |

## Pembagian tugas

| Pemilik proyek | Teman (kamu) |
|---|---|
| Mencari relawan, meminta tanda tangan persetujuan, merekam `kerutan_dahi` | Mengunduh dataset publik |
| Mengirim rekaman relawan lewat folder Drive yang dibagikan **hanya ke emailmu** | Import, pra-anotasi, koreksi label |
| Menyimpan formulir persetujuan | Split, augmentasi, training, evaluasi |
| Mengecek hasil akhir | Mengirim balik model dan laporan |

## Aturan data (wajib)

- Semua dataset ini berisi **wajah orang**. Jangan diunggah ke tempat publik, jangan di-commit ke git (folder `data/raw`, `data/incoming`, `data/annotated`, dan `data/dataset` sudah di-ignore), dan jangan dibagikan ke orang lain.
- NTHU-DDD, UTA-RLDD, dan rekaman relawan **tidak boleh** masuk ke proyek Roboflow publik. Paket gratis Roboflow biasanya membuat proyek bersifat publik, jadi cek setelan workspace dulu atau pakai CVAT (lihat langkah 4).
- Setelah proyek selesai, hapus semua data wajah dari laptopmu.

## 0. Persiapan

1. Ambil versi terbaru repo:
   ```bash
   git clone https://github.com/widiandickha/project-kt_BUDDY.git   # atau: git pull
   ```
2. Pasang Python 3.11, venv, dan PyTorch CUDA seperti di panduan training sebelumnya. Kali ini **MediaPipe wajib terpasang**, karena dipakai untuk pra-anotasi:
   ```bash
   pip install -r requirements.txt
   ```
3. Siapkan ruang disk kosong sekitar **40 GB** (video UTA cukup besar).

Di Windows, ganti `.venv/bin/python` dengan `.venv\Scripts\python` (atau aktifkan venv lalu pakai `python`).

## 1. Unduh dataset

Simpan semua unduhan **di luar** folder repo, misalnya di `D:\dataset-kt\`.

| # | Dataset | Untuk | Cara unduh | Lisensi |
|---|---|---|---|---|
| 1 | [NTHU-DDD multi-class](https://www.kaggle.com/datasets/samymesbah/nthu-dataset-ddd-multi-class) (±2,3 GB) | mata, menguap | Kaggle → Download (perlu login) | riset |
| 2 | [UTA-RLDD](https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset) (total 96,6 GB) | tambahan orang | Di *Data Explorer*, unduh **satu atau dua folder saja**, misalnya `Fold1_part1` dan `Fold1_part2` (masing-masing ±10 GB, ±6 orang) | CC0 |
| 3 | [Driver yawn](https://universe.roboflow.com/universidad-carlos-3-madrid/driver-yawn) (1.448 gambar dari YawDD) | menguap | Pilih **Dataset** → pilih versi **tanpa augmentasi** (lihat bagian *Augmentations* di halaman versi), lalu **Download → YOLOv8 → zip**. Perlu akun Roboflow gratis. | CC BY 4.0 |
| 4 | [Roboflow One wrinkles](https://universe.roboflow.com/roboflow-one/roboflow-one-wrinkles-detection) (915 foto) | kerutan_dahi (pelengkap) | **Download → YOLOv8** (polygon segmentasi tidak masalah) | CC BY 4.0 |
| 5 | Rekaman relawan | **kerutan_dahi** (utama) | Dari pemilik proyek lewat Drive | persetujuan relawan |

Catatan untuk nomor 4: kebanyakan foto berisi orang tua dengan kerutan permanen, bukan dahi yang sedang dikerutkan. Pakai seperlunya (preset membatasi 300 gambar). Sumber utama `kerutan_dahi` tetap rekaman relawan.

## 2. Import ke `data/raw/`

```bash
# 1. NTHU (frame dipilih merata + overlay warna dibersihkan otomatis)
.venv/bin/python scripts/import_public.py nthu --src "D:/dataset-kt/nthu"

# 2. UTA (1 frame tiap 2 detik, maks 150 per video)
.venv/bin/python scripts/import_public.py uta --src "D:/dataset-kt/uta"

# 3. Driver yawn. Lihat dulu nama file di folder images/: kalau diawali nomor orang
#    seperti "12-MaleGlasses-Yawning...", tambahkan --subjek-regex "^(\d+)-"
.venv/bin/python scripts/import_public.py roboflow --preset driver-yawn --src "D:/dataset-kt/driver-yawn"

# 4. Wrinkles
.venv/bin/python scripts/import_public.py roboflow --preset wrinkles --src "D:/dataset-kt/wrinkles"

# 5. Relawan: taruh di data/incoming/kerutan_dahi/<kode_orang>/ (satu folder per orang)
.venv/bin/python scripts/import_public.py folder --kondisi terang
```

- Kalau rekaman relawan berbentuk folder `self-<kode>/` hasil `collect_data.py`, salin langsung ke `data/raw/`. Nama filenya sudah sesuai skema.
- Untuk importer Roboflow, label sumber **hanya dipakai untuk menentukan petunjuk** per gambar (misalnya ada box `yawn` → `menguap`). Box-nya digambar ulang oleh pra-anotasi supaya mengikuti konvensi kita.
- Semua perintah aman diulang, karena file yang sudah ada tidak diduplikasi.

## 3. Pra-anotasi otomatis

```bash
.venv/bin/python scripts/preannotate.py
```

Perintah ini membuat draf label `.txt` di samping setiap gambar dan file `data/raw/_preannotate.csv` berisi *flag* untuk diperiksa. Arti setiap flag ada di [ANOTASI.md](ANOTASI.md) bagian 2.

## 4. Koreksi label (bagian terpenting)

Draf otomatis **wajib dikoreksi manual**. Tanpa koreksi, metriknya tidak sah untuk laporan.

- **Tool yang disarankan: CVAT** (cvat.ai, atau dipasang sendiri di laptop). Proyek di sana tidak publik. Cara import dan export format *YOLO 1.1* ada di [ANOTASI.md](ANOTASI.md) bagian 4.
- Roboflow hanya boleh dipakai untuk data yang memang sudah publik (subjek `rf-*`).
- Urutan kerja:
  1. Semua gambar `kerutan_dahi` (flag `cek_kerutan`): pastikan dahinya memang berkerut, dan hapus box kalau tidak.
  2. Gambar dengan flag `ragu_mata`, `menguap_tak_terduga`, dan `wajah_tidak_terdeteksi`.
  3. Sisanya dicek sekilas.
- Hasil export dikumpulkan ke `data/annotated/`. Nama file harus tetap diawali `<subjek>__...`.
- Pekerjaan ini bisa dibagi dua dengan pemilik proyek lewat satu proyek CVAT bersama.

## 5. Susun dataset

```bash
.venv/bin/python training/split_dataset.py --src data/annotated \
    --assign rf-driver-yawn=train --assign rf-wrinkles=train
.venv/bin/python training/augment.py
.venv/bin/python training/check_dataset.py
```

- Split dibagi **per orang** (70/20/10), jadi satu orang tidak pernah muncul di dua split.
- Sumber Roboflow tidak punya identitas orang (kecuali kalau `--subjek-regex` dipakai), jadi semuanya ditaruh di train.
- Buka `docs/dataset_report.md` dan pastikan:
  - Tiap kelas ≥ 500 instance.
  - Ada ≥ 5 orang.
  - **Tidak ada peringatan** "`kerutan_dahi` tidak ada di split val/test". Kalau masih ada, arahkan satu relawan ke val dan satu ke test, misalnya `--assign self-r02=val --assign ext-r03=test`, lalu jalankan ulang ketiga perintah di atas.

## 6. Training & evaluasi

```bash
.venv/bin/python training/train.py                          # 100 epoch, berhenti otomatis kalau tidak membaik
.venv/bin/python training/evaluate.py --label-status final   # hanya kalau SEMUA label sudah dikoreksi
```

- Kalau masih ada label yang belum dikoreksi, jalankan `evaluate.py` tanpa `--label-status final`.
- Kalau mAP50 di bawah 0,80: `train.py --model yolov8s.pt --name face_s`, lalu evaluasi lagi.
- Masalah GPU (`CUDA out of memory`, `cuda` tidak terbaca, error *workers* di Windows) dan solusinya ada di tabel "Masalah umum" pada panduan training sebelumnya.

## 7. Kirim balik (zip, tanpa gambar wajah)

| File | Isi |
|---|---|
| `models/best.pt` | bobot model |
| `runs/detect/<nama run>/` | kurva training, confusion matrix, `run_meta.json` |
| `docs/eval/` | metrik per kelas di split test |
| `docs/dataset_report.md` | jumlah per kelas, per orang, per split |
| `data/raw/_preannotate.csv` | ringkasan flag pra-anotasi (berisi angka, bukan gambar) |

Jangan kirim gambar. Pemilik proyek cukup menerima model dan laporannya.
