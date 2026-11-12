# Performa (FASE 8.3)

Target spesifikasi: >= 10 FPS di laptop CPU kelas menengah, latensi < 1 detik, CPU < 40%.
CPU% dihitung psutil untuk proses ini (100% = satu core penuh). Inferensi dibatasi `inference.target_fps` seperti di aplikasi, kecuali ditandai 'tanpa batas'. Event→bubble: rata-rata = inferensi p50 + rata-rata tunggu tick 0,5 s + animasi 200 ms; maks = inferensi p95 + tick 1 s + 200 ms.

| Waktu | Mesin | Device | imgsz | YOLO wajah | Sumber | Batas FPS | FPS | Latensi p50/p95 (ms) | CPU % | RAM (MB) | Event→bubble rata-rata / maks (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-12 11:03 | Darwin arm64 | cpu | 480 | tidak | nthu_seq | 12 | 10.9 | 27 / 38 | 63 | 329 | 727 / 1238 |
| 2026-11-12 11:04 | Darwin arm64 | mps | 480 | tidak | nthu_seq | 12 | 10.7 | 38 / 47 | 24 | 441 | 738 / 1247 |
| 2026-11-12 11:13 | Darwin arm64 | cpu | 480 | tidak | nthu_seq | tanpa batas | 15.2 | 62 / 108 | 182 | 525 | 762 / 1308 |

Catatan run 12 Nov 2026 (Mac Apple M4, tanpa model YOLO wajah, sumber 736 frame NTHU 640×480):
- Target ≥ 10 FPS terpenuhi (10,7–10,9 FPS dengan batas 12).
- CPU < 40% terpenuhi di MPS (24%), tidak di CPU saja (63%). Di laptop tanpa GPU, aktifkan `inference.performance_mode`.
- Latensi event→bubble rata-rata ±0,73 s (< 1 s). Kasus terburuk ±1,24 s karena tick analisis 1 detik.
- Baris "tanpa batas" diukur bersamaan dengan uji ketahanan, jadi angkanya kurang mewakili.
- Uji ketahanan 10 menit (aplikasi penuh, offscreen): RSS stabil ±600 MB (tidak ada kebocoran), CPU 15–26%, 0 error.
  Uji 1 jam dengan webcam asli dilakukan oleh pengguna (lihat failure_tests.md).
