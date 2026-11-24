# Panduan Anotasi (FASE 2)

Alur lengkap dari data mentah sampai dataset siap latih.

```text
collect_data.py / import_public.py  ->  data/raw/<subjek>/<petunjuk>/*.jpg
preannotate.py                      ->  draf label .txt di samping tiap gambar
Roboflow / CVAT (koreksi manual)    ->  label final format YOLO
split_dataset.py                    ->  data/dataset + data/data.yaml + docs/dataset_report.md
augment.py (opsional)               ->  salinan augmentasi khusus train
check_dataset.py                    ->  DoD: yolo val jalan di semua split
```

## 1. Kumpulkan data

**Rekaman relawan.** Wajib ada persetujuan tertulis. Rekam per kondisi:

```bash
.venv/bin/python scripts/collect_data.py --relawan andi --kondisi terang
.venv/bin/python scripts/collect_data.py --relawan andi --kondisi redup-kacamata
```

- Tombol `1` sampai `4` memilih instruksi. `SPASI` menyimpan satu frame, `B` menyalakan mode burst.
- Variasi wajib: cahaya terang, redup, dan backlight; dengan dan tanpa kacamata; jarak 40–90 cm; sudut kepala ringan; latar yang berbeda.
- Target: 500 instance per kelas dari minimal 5 orang.

**Foto/video sendiri, terutama `kerutan_dahi`.** Tidak ada dataset publik yang punya kelas ini, jadi datanya harus dikumpulkan sendiri. Selain `collect_data.py`, kamu bisa memakai foto atau video dari HP:

1. Taruh file di `data/incoming/kerutan_dahi/<kode_orang>/`, misalnya `data/incoming/kerutan_dahi/r01/*.jpg` atau `.../r02/dahi.mp4`. Satu folder untuk satu orang, karena split dibagi per orang. Gunakan kode, bukan nama, dan pastikan setiap orang sudah menandatangani persetujuan.
2. Jalankan:

   ```bash
   .venv/bin/python scripts/import_public.py folder                       # default: kelas kerutan_dahi
   .venv/bin/python scripts/import_public.py folder --kondisi redup      # tag kondisi untuk uji ketahanan
   .venv/bin/python scripts/import_public.py folder --kelas menguap --src ~/Desktop/menguap
   ```

   Gambar disalin ke `data/raw/ext-<kode>/kerutan_dahi/` dan diperkecil ke lebar kamera. Untuk video, diambil 1 frame tiap `dataset.import.uta_every_sec` detik. Folder `data/incoming/` tidak ikut ke git.
3. Isi yang bagus: dahi berkerut atau alis menegang yang **jelas terlihat**, wajah menghadap kamera dari jarak 40–90 cm, dari minimal 3–5 orang, sekitar 100–150 frame per orang per kondisi. Selipkan juga foto wajah netral dari orang yang sama, supaya model belajar membedakan.

**Dataset publik:**

```bash
.venv/bin/python scripts/import_public.py nthu --src "~/Downloads/KAGGLE/3. NTHU DDD"
.venv/bin/python scripts/import_public.py uta  --src "/path/ke/uta-reallife-drowsiness-dataset"
.venv/bin/python scripts/import_public.py roboflow --preset driver-yawn --src "/path/ke/export-yolov8"
.venv/bin/python scripts/import_public.py roboflow --preset wrinkles    --src "/path/ke/export-yolov8"
```

Untuk sumber Roboflow, label aslinya hanya dipakai sebagai petunjuk per gambar, lalu box-nya digambar ulang oleh pra-anotasi. Seluruh export menjadi satu subjek `rf-<nama>`, jadi tempatkan di train dengan `split_dataset.py --assign rf-<nama>=train`. Daftar sumber, lisensi, dan alur lengkap untuk yang mengurus dataset ada di [PANDUAN_DATASET.md](PANDUAN_DATASET.md).

## 2. Pra-anotasi otomatis

```bash
.venv/bin/python scripts/preannotate.py
```

- Membuat draf label dari landmark MediaPipe.
- Label yang sudah ada **tidak ditimpa**, jadi aman dijalankan ulang setelah menambah data.
- File `data/raw/_preannotate.csv` (hanya berisi hasil run terakhir) mencantumkan EAR, MAR, dan *flag*. Urutkan berdasarkan kolom `flags`, lalu periksa yang ditandai lebih dulu:

| Flag | Artinya |
|---|---|
| `ragu_mata` | EAR di antara ambang tertutup dan terbuka. Tentukan sendiri kelas matanya. |
| `hint_menguap_mulut_tertutup` | Frame berasal dari skenario menguap, tapi mulut tertutup. Biasanya benar (sebelum atau sesudah menguap). |
| `menguap_tak_terduga` | Mulut terbuka lebar di luar skenario menguap. Bisa jadi sedang bicara, jadi hapus box `menguap` bila bukan menguap. |
| `hint_tertutup_ear_terbuka` | Relawan diminta memejam, tapi EAR menunjukkan mata terbuka. |
| `cek_kerutan` | Box `kerutan_dahi` hanya diberikan dari instruksi. Pastikan dahinya memang berkerut. |
| `wajah_tidak_terdeteksi` | Tidak ada label. Anotasi manual atau hapus gambarnya. |

## 3. Konvensi box (wajib konsisten)

| Kelas | Box |
|---|---|
| `mata_tertutup` / `mata_terbuka` | **Satu box untuk kedua mata**, dari ujung luar mata kiri sampai kanan, termasuk kelopak. Pilih satu kelas saja per wajah. Mata setengah terpejam (mengantuk) = `mata_tertutup` bila pupil hampir tidak terlihat. |
| `menguap` | Wajah bagian bawah: dari pangkal hidung sampai dagu, selebar pipi, termasuk mulut terbuka lebar. Hanya untuk menguap, **bukan bicara atau tersenyum**. |
| `kerutan_dahi` | Area dahi dan alis, dari bawah alis sampai batas rambut. Hanya bila kerutan atau alis menegang terlihat jelas. |

Frame tanpa wajah boleh dipakai sebagai sampel negatif dengan file `.txt` kosong.

## 4. Koreksi di tool anotasi

> ⚠️ **Privasi:** paket gratis Roboflow biasanya membuat proyek bersifat publik. NTHU-DDD, UTA-RLDD, dan rekaman relawan **tidak boleh** diunggah ke proyek publik. Untuk data seperti itu, pakai CVAT. Roboflow hanya aman untuk sumber yang memang sudah publik (subjek `rf-*`).

**Roboflow (paling mudah):**
1. Buat proyek *Object Detection*.
2. Upload folder `data/raw/` beserta `data/raw/classes.txt`. Roboflow membaca file `.txt` YOLO di samping gambar sebagai label awal.
3. Koreksi box, lalu **Export → YOLOv8**. Matikan semua *preprocessing* dan augmentasi Roboflow (resize, auto-orient) agar ukuran gambar tetap.
4. Ekstrak hasil export ke `data/annotated/` (folder ini di-ignore oleh git).

**CVAT:** import format *YOLO 1.1*, koreksi, lalu export *YOLO 1.1* ke `data/annotated/`.

Nama file harus tetap diawali `<subjek>__...`. Roboflow menambahkan akhiran `_jpg.rf.xxx`, dan itu tidak masalah karena subjek tetap terbaca.

## 5. Bangun dataset

```bash
.venv/bin/python training/split_dataset.py --src data/annotated
.venv/bin/python training/augment.py
.venv/bin/python training/check_dataset.py
```

- `split_dataset.py` membagi dataset **per orang** (70/20/10) dan menulis `docs/dataset_report.md`.
- Untuk memaksa subjek tertentu ke split tertentu, gunakan `--assign nthu-006=train`.
- `augment.py` hanya menambah data ke split train.
