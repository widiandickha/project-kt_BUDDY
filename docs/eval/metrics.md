# Evaluasi Detektor Wajah

> ⚠️ **Label draf.** Dataset masih memakai pra-anotasi otomatis yang belum dikoreksi manual (lihat `docs/dataset_report.md`). Angka di bawah hanya untuk mengecek pipeline dan **tidak boleh** dipakai di laporan Bab IV.

- Dievaluasi: 2026-11-19 09:57
- Bobot: `runs/detect/face_rtx3050/weights/best.pt`
- Split: `test` (subjek: nthu-006; tidak dipakai saat training)
- imgsz: 640 · device: mps · input grayscale: True
- Training: base `yolov8n.pt`, epoch terbaik 10, durasi 53.2 menit di cuda

## Per kelas

Target spesifikasi: mAP50 ≥ 0.80 per kelas.

| Kelas | Instance | Precision | Recall | mAP50 | mAP50-95 | Target |
|---|---|---|---|---|---|---|
| mata_tertutup | 86 | 0.426 | 1.000 | 0.760 | 0.566 | ❌ |
| mata_terbuka | 192 | 0.914 | 0.740 | 0.950 | 0.718 | ✅ |
| menguap | 9 | 1.000 | 0.738 | 0.848 | 0.567 | ✅ |
| kerutan_dahi | 0 | – | – | – | – | – |
| **semua** | 287 | 0.780 | 0.826 | 0.853 | 0.617 | |

Kecepatan per gambar: preprocess 0.2 ms, inferensi 1.8 ms, postprocess 5.9 ms.

## Keterbatasan

- Tidak ada instance di split ini, jadi tidak bisa diukur: `kerutan_dahi`.
- Di bawah target: `mata_tertutup`. Sesuai spesifikasi, catat sebagai keterbatasan dan turunkan bobot indikatornya di `score.weights` bila tetap rendah setelah data diperbaiki.

Gambar: `confusion_matrix.png`, `confusion_matrix_normalized.png`, kurva PR dan F1, `train_results.png`.
