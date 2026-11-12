# Laporan Dataset

Dibuat otomatis oleh `training/split_dataset.py` pada 2026-11-12 10:10.

## Ringkasan per split

| Split | Subjek | Gambar | Negatif | mata_tertutup | mata_terbuka | menguap | kerutan_dahi |
|---|---|---|---|---|---|---|---|
| train | nthu-001, nthu-005 | 2557 | 0 | 1196 | 1361 | 155 | 0 |
| val | nthu-002 | 1232 | 0 | 458 | 774 | 164 | 0 |
| test | nthu-006 | 278 | 0 | 86 | 192 | 9 | 0 |
| **total** | 4 | 4067 | 0 | **1740** | **2327** | **328** | **0** |

## Per subjek

| Subjek | Split | Gambar | Kondisi | mata_tertutup | mata_terbuka | menguap | kerutan_dahi |
|---|---|---|---|---|---|---|---|
| nthu-001 | train | 1238 | noglasses 620, glasses 618 | 254 | 984 | 11 | 0 |
| nthu-002 | val | 1232 | glasses 618, noglasses 614 | 458 | 774 | 164 | 0 |
| nthu-005 | train | 1319 | glasses 699, noglasses 620 | 942 | 377 | 144 | 0 |
| nthu-006 | test | 278 | glasses 278 | 86 | 192 | 9 | 0 |

## Peringatan

- Baru 4 subjek; spesifikasi meminta minimal 5 orang berbeda.
- `menguap`: 328 instance, di bawah target 500.
- `kerutan_dahi`: 0 instance, di bawah target 500.
- `kerutan_dahi` tidak ada di split val; metrik kelas ini tidak bisa diukur di sana.
- `kerutan_dahi` tidak ada di split test; metrik kelas ini tidak bisa diukur di sana.
- 13 gambar belum punya file label dan dilewati.

Split dilakukan per subjek (orang), sehingga tidak ada orang yang muncul di dua split.
