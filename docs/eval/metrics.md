# Evaluasi Detektor Wajah

> ⚠️ **Label draf.** Dataset masih memakai pra-anotasi otomatis yang belum dikoreksi manual (lihat `docs/dataset_report.md`). Angka di bawah hanya untuk mengecek pipeline dan **tidak boleh** dipakai di laporan Bab IV tanpa catatan ini. Perbandingan dengan v1: [perbandingan_model.md](perbandingan_model.md).

- Dievaluasi: 2026-11-26 06:02
- Bobot: `models/best.pt`
- Split: `test` (subjek: nthu-002; tidak dipakai saat training)
- imgsz: 640 · device: cuda · input grayscale: True
- Training: base `yolov8n.pt`, epoch terbaik 35, durasi 156.5 menit di cuda

## Per kelas

Target spesifikasi: mAP50 ≥ 0.80 per kelas.

| Kelas | Instance | Precision | Recall | mAP50 | mAP50-95 | Target |
|---|---|---|---|---|---|---|
| mata_tertutup | 459 | 0.868 | 0.842 | 0.929 | 0.790 | ✅ |
| mata_terbuka | 773 | 0.864 | 0.978 | 0.953 | 0.853 | ✅ |
| menguap | 164 | 0.957 | 0.933 | 0.987 | 0.863 | ✅ |
| kerutan_dahi | 0 | – | – | – | – | – |
| **semua** | 1396 | 0.896 | 0.918 | 0.956 | 0.835 | |

Kecepatan per gambar: preprocess 1.1 ms, inferensi 3.8 ms, postprocess 1.0 ms.

## Keterbatasan

- Tidak ada instance di split ini, jadi tidak bisa diukur: `kerutan_dahi`.

Gambar: `confusion_matrix.png`, `confusion_matrix_normalized.png`, kurva PR dan F1, `train_results.png`.
