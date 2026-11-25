# Perbandingan Model v1 vs Final

Kedua model dievaluasi ulang di Mac yang sama (MPS), memakai **gambar dan label yang sama**. Dua subjek NTHU yang tidak pernah masuk ke bobot training kedua model:

- `nthu-002`: dipakai v1 sebagai val (untuk memilih epoch), dipakai final sebagai test.
- `nthu-006`: dipakai v1 sebagai test, dipakai final sebagai val.

> ⚠️ Label kedua subjek masih **draf** (pra-anotasi MediaPipe yang belum dikoreksi manual). Angka-angka ini mengukur seberapa konsisten model dengan aturan landmark, belum terhadap kebenaran manusia.

## Data training

| | v1 | Final |
|---|---|---|
| Subjek di train | 2 (nthu-001, nthu-005) | 13 (nthu-001, 11 orang UTA, Roboflow driver-yawn) |
| Gambar total | 4.067 | 10.671 |
| Instance `menguap` | 328 | 853 |
| Epoch terbaik / durasi | 10 / 53 menit | 35 / 156,5 menit (RTX 3050) |

## mAP (conf 0,001, IoU 0,5, imgsz 640)

| Subjek | Model | P | R | mAP50 | mAP50-95 | mata_tertutup | mata_terbuka | menguap |
|---|---|---|---|---|---|---|---|---|
| nthu-002 | v1 | 0,873 | 0,858 | 0,923 | 0,703 | 0,882 | 0,932 | 0,954 |
| nthu-002 | **final** | 0,896 | 0,917 | **0,956** | **0,835** | 0,928 | 0,952 | 0,987 |
| nthu-006 | v1 | 0,780 | 0,826 | 0,853 | 0,617 | 0,760 | 0,950 | 0,848 |
| nthu-006 | **final** | 0,753 | 0,985 | **0,951** | **0,781** | 0,905 | 0,976 | 0,973 |

Kenaikan mAP50-95 yang besar (+0,13 dan +0,16) menunjukkan posisi box jauh lebih tepat.

## Keputusan mata per frame (ambang aplikasi `conf_eye` 0,5)

Satu frame dihitung sekali: kelas mata dengan confidence tertinggi dibandingkan dengan label.

| Model | Terbuka → terbaca tertutup | Tertutup → terbaca terbuka | Benar |
|---|---|---|---|
| v1 | 186 dari 957 (19,4%) | 59 dari 545 (10,8%) | 1.246 dari 1.521 (81,9%) |
| **final** | **86 dari 957 (9,0%)** | 92 dari 545 (16,9%) | **1.322 dari 1.521 (86,9%)** |

- Masalah utama v1 (mata terbuka sering dianggap terpejam, terutama pada orang berkacamata) turun lebih dari setengahnya.
- Gantinya, model final lebih sering melewatkan mata yang terpejam (kebanyakan di `nthu-002`). Di aplikasi, efek ini diredam fusi 0,6 EAR + 0,4 YOLO, jadi kedipan panjang tetap terbaca dari EAR.

## Kesimpulan

Model final dipakai sebagai `models/best.pt`, sedangkan v1 disimpan sebagai `models/best.prev.pt` (keduanya tidak masuk git). Angka mAP tetap memakai label draf. Untuk Bab IV, tulis catatan ini, atau koreksi label test (`nthu-002`) di CVAT lalu jalankan `training/evaluate.py --label-status final`.
