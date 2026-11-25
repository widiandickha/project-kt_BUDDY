# Laporan Dataset

Dibuat otomatis oleh `training/split_dataset.py` pada 2026-11-26 00:29.

## Ringkasan per split

| Split | Subjek | Gambar | Negatif | mata_tertutup | mata_terbuka | menguap | kerutan_dahi |
|---|---|---|---|---|---|---|---|
| train | nthu-001, rf-driver-yawn, uta-1, uta-10, uta-11, uta-12, uta-2, uta-3, uta-4, uta-5, uta-6, uta-7, uta-9 | 7394 | 0 | 765 | 6629 | 534 | 0 |
| val | nthu-005, nthu-006, uta-8 | 2045 | 0 | 1060 | 985 | 155 | 0 |
| test | nthu-002 | 1232 | 0 | 459 | 773 | 164 | 0 |
| **total** | 17 | 10671 | 0 | **2284** | **8387** | **853** | **0** |

## Per subjek

| Subjek | Split | Gambar | Kondisi | mata_tertutup | mata_terbuka | menguap | kerutan_dahi |
|---|---|---|---|---|---|---|---|
| nthu-001 | train | 1238 | noglasses 620, glasses 618 | 254 | 984 | 11 | 0 |
| nthu-002 | test | 1232 | glasses 618, noglasses 614 | 459 | 773 | 164 | 0 |
| nthu-005 | val | 1319 | glasses 699, noglasses 620 | 943 | 376 | 144 | 0 |
| nthu-006 | val | 278 | glasses 278 | 85 | 193 | 9 | 0 |
| rf-driver-yawn | train | 1211 | unknown 1211 | 233 | 978 | 519 | 0 |
| uta-1 | train | 450 | unknown 450 | 19 | 431 | 2 | 0 |
| uta-10 | train | 450 | unknown 450 | 35 | 415 | 0 | 0 |
| uta-11 | train | 447 | unknown 447 | 19 | 428 | 0 | 0 |
| uta-12 | train | 449 | unknown 449 | 48 | 401 | 0 | 0 |
| uta-2 | train | 450 | unknown 450 | 10 | 440 | 0 | 0 |
| uta-3 | train | 450 | unknown 450 | 59 | 391 | 0 | 0 |
| uta-4 | train | 449 | unknown 449 | 16 | 433 | 0 | 0 |
| uta-5 | train | 450 | unknown 450 | 14 | 436 | 0 | 0 |
| uta-6 | train | 450 | unknown 450 | 15 | 435 | 2 | 0 |
| uta-7 | train | 450 | unknown 450 | 29 | 421 | 0 | 0 |
| uta-8 | val | 448 | unknown 448 | 32 | 416 | 2 | 0 |
| uta-9 | train | 450 | unknown 450 | 14 | 436 | 0 | 0 |

## Peringatan

- `kerutan_dahi`: 0 instance, di bawah target 500.
- `kerutan_dahi` tidak ada di split val; metrik kelas ini tidak bisa diukur di sana.
- `kerutan_dahi` tidak ada di split test; metrik kelas ini tidak bisa diukur di sana.
- 29 gambar belum punya file label dan dilewati.

Split dilakukan per subjek (orang), sehingga tidak ada orang yang muncul di dua split.
