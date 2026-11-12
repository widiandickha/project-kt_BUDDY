# Audit Proyek Lama terhadap Spesifikasi Build

Tanggal audit: 12 November 2026
Acuan: *Spesifikasi Build: Deteksi Dini Stres & Evaluasi Habit Pengguna PC dengan Virtual Buddy Kucing*
Kode yang diaudit: commit `997925d` (branch `main`) + perubahan `config.yaml` yang belum di-commit.

Legenda: ✅ sesuai · 🟡 sebagian / beda pendekatan · ❌ belum ada

## Ringkasan

Kode lama adalah prototipe yang **berjalan** (16 sesi tercatat di database) dan logika intinya searah dengan spesifikasi. Namun **tidak ada satu pun Definition of Done (DoD) fase yang terpenuhi**. Penyebab utamanya:

1. Tidak ada folder `tests/`, jadi tidak ada unit test sama sekali (aturan 0 dan DoD setiap fase).
2. Struktur folder dan arsitektur berbeda. Semua pekerjaan (capture, inferensi, skoring, DB, Buddy) berjalan di satu `QThread`, tidak ada `EventBus` dan dataclass kontrak (`FrameFeatures`, `StressSnapshot`, `BuddyEvent`).
3. Model wajah buatan sendiri belum di-training, dan definisi kelasnya beda dengan spesifikasi (3 kelas, bukan 4).
4. Buddy masih berupa overlay notifikasi dengan 5 ekspresi. Spesifikasi meminta widget kucing desktop dengan 12 ekspresi, state machine, sprite player, dan click-through.

**Rekomendasi:** bangun ulang mengikuti struktur dan fase spesifikasi, lalu **pindahkan (port) bagian yang sudah benar** dari kode lama. Jangan menambal kode lama, karena perbedaan arsitekturnya (thread, event bus, kontrak data) menyentuh hampir semua file.

## Bagian 0: Aturan main

| Aturan | Status | Temuan |
|---|---|---|
| Kode Inggris, teks UI/kelas Indonesia | 🟡 | Nama variabel sudah Inggris, tapi docstring dan sebagian nilai internal berbahasa Indonesia (`"Tinggi"`, `"kantuk"`, `"tegang_dahi"`). |
| Semua angka di `config.yaml`, tanpa angka ajaib | ❌ | Banyak konstanta di dalam modul: pengali `2.2`, `3.0`, `2.0`, `1.6` dan `28.0` di `scoring.py`; conf pose `0.35`/`0.25`/`0.30` dan offset dahi `0.18` di `pose_yolo.py`; batas kedip `1..8` frame; ambang `dominant_reason` (`0.25`, `0.12`, `0.20`, `0.30`, `0.35`). |
| Type hints, docstring, ≥1 test per modul | ❌ | Type hints umumnya ada. Docstring hanya di level kelas. **Folder `tests/` tidak ada.** |
| Lokal, frame tidak ditulis ke disk | ✅ | Aplikasi tidak menyimpan frame. `cv2.imwrite` hanya dipakai di script dataset, dan itu memang tujuannya. |
| Loosely coupled lewat dataclass + event queue | ❌ | `CameraWorker` memanggil detektor, scorer, decision engine, dan DB secara langsung. |
| `docs/DECISIONS.md` | ❌ | Folder `docs/` tidak ada sebelum audit ini. |
| Commit `phase-N: ...` per fase | ❌ | Riwayat commit memakai `feat:`/`fix:`, tidak per fase. |

## Bagian 2: Tech stack

| Item | Status | Temuan |
|---|---|---|
| Python 3.10/3.11 | ❌ | `.venv` memakai Python 3.12. |
| Versi dipin | ❌ | `requirements.txt` memakai `>=` (tidak dipin). |
| pandas, pytest, matplotlib/pyqtgraph, pillow | ❌ | Tidak ada di `requirements.txt`. |
| MediaPipe untuk EAR | ✅ | FaceLandmarker Tasks API (`models/face_landmarker.task`). |
| yolov8n-pose | ✅ | `models/yolov8n-pose.pt`. |

## Bagian 3 & 4: Struktur dan arsitektur runtime

| Item | Status | Temuan |
|---|---|---|
| Struktur `src/core`, `capture`, `detection`, `analysis`, `buddy`, `storage`, `ui`, `training/` | ❌ | Struktur berbeda (`src/detectors`, `src/ui/camera_worker.py`, dan semuanya ada di `scripts/`). |
| Thread A: CaptureThread + `Queue(maxsize=2)` buang frame lama | ❌ | Capture dan inferensi ada di thread yang sama, tanpa queue. |
| Thread B: Inference 10–15 FPS + frame skipping | 🟡 | Ada target FPS di config, tapi tidak ada frame skipping. Tiga model (MediaPipe, pose, YOLO wajah) jalan berurutan di setiap frame. |
| Thread C / timer: skor setiap 1 detik | ❌ | Skor dihitung **per frame**, bukan per detik. |
| UI menerima lewat Qt Signal | ✅ | `frame_processed`, `buddy_triggered`, dan lain-lain. |
| Dataclass `FrameFeatures` / `StressSnapshot` / `BuddyEvent` | ❌ | Ada `FrameSample`, `WindowSummary`, dan `BuddyAdvice`, tapi isinya berbeda. Contohnya, `ear_left/right`, `face_size_ratio`, `pose_keypoints`, `components`, dan `burnout_trend` tidak ada. |
| Event `user_away` / `user_returned` | ❌ | Tidak ada. Saat wajah tidak terdeteksi, EAR diisi default `0.28` dan frame tetap masuk ke window. |

## FASE 1: Setup

| DoD | Status |
|---|---|
| `scripts/benchmark_fps.py` (webcam + yolov8n, FPS 30 detik) | ❌ Tidak ada. FPS hanya diukur di dalam `train_and_evaluate.py` (offline). |
| Log `torch.cuda.is_available()` + fallback CPU `imgsz=480` | ❌ |
| `pytest` jalan | ❌ |

## FASE 2: Dataset & anotasi

| Item | Status | Temuan |
|---|---|---|
| Kelas `0 mata_tertutup, 1 mata_terbuka, 2 menguap, 3 kerutan_dahi` | ❌ | Dataset lama: `0 mata_tertutup, 1 menguap, 2 kerutan_dahi`. Kelas `mata_terbuka` tidak ada dan urutannya berbeda, jadi **tidak kompatibel**. |
| ≥500 gambar/kelas, ≥5 orang | ❌ | Total 340 gambar. Instance: 170 `mata_tertutup`, 40 `menguap`, 0 `kerutan_dahi`. |
| Konvensi box (mata per mata, menguap = wajah bawah, kerutan = dahi + alis) | ❌ | Box MRL berukuran 90% gambar, padahal gambarnya **potongan satu mata 150×150**, sehingga domainnya beda dengan webcam wajah penuh. Box NTHU yang dicek memakai koordinat tetap (`0.5 0.52 0.58 0.68`), bukan posisi mulut sebenarnya. |
| Split per orang 70/20/10 | 🟡 | Split per subjek sudah ada (bagus), tapi rasionya bukan 70/20/10. NTHU hanya punya 4 subjek. |
| `scripts/collect_data.py` → `data/raw/<relawan>/<kelas>/` | 🟡 | `annotate_custom_webcam.py` sudah ada (tombol 1/2/3), tapi tanpa nama relawan dan struktur folder spesifikasi. |
| `training/augment.py` | ❌ | |
| `docs/dataset_report.md` | ❌ | |
| Folder sumber NTHU | ❌ | `3. NTHU DDD/` sudah tidak ada di root, jadi dataset tidak bisa dibuat ulang. |

## FASE 3: Training

| Item | Status | Temuan |
|---|---|---|
| `training/train.py` (epochs 100, patience 20) | 🟡 | `scripts/train_and_evaluate.py` sudah ada, tapi default di README 25 epoch. |
| `training/evaluate.py` → `docs/eval/` | 🟡 | Evaluasi split test ada, tapi hasilnya tidak disimpan ke `docs/eval/`. |
| `models/best.pt` | ❌ | Belum pernah di-training. Nama yang dipakai juga beda (`face_expression_yolov8n.pt`). |
| mAP50 ≥ 0.80 per kelas | ❌ | Belum ada angka. |
| Demo `python -m src.detection.face_detector --demo` | ❌ | |

## FASE 4: Analisis & skoring

| Item | Status | Temuan |
|---|---|---|
| EAR 6 titik, indeks Face Mesh | ✅ | Rumus dan indeks benar. Hanya penamaan kiri/kanan yang tertukar (`LEFT_EYE_IDX` memakai indeks mata kanan versi spesifikasi), tapi rata-ratanya tetap sama. |
| `ear_left/right/mean` terpisah | ❌ | Hanya ada nilai rata-rata. |
| Ambang tertutup `0.21` + kalibrasi `open − 0.4·(open − closed)` | 🟡 | Rumusnya berbeda: `baseline × 0.75`, dan kalibrasi hanya mengukur mata terbuka (tidak meminta pengguna memejam). |
| Fusi `0.6·EAR + 0.4·YOLO`, fallback YOLO | ❌ | Fusinya pakai `OR` boolean. |
| ⚠️ Fallback Haar | ❌ | Bila MediaPipe gagal tapi Haar menemukan wajah tanpa mata, EAR diisi `0.18`, sehingga dihitung **mata tertutup**. Ini rawan false alarm saat kepala menoleh. |
| Hand-to-forehead: jarak ke titik tengah mata, `wrist_y ≤ nose_y`, tahan ≥5 frame | 🟡 | Jaraknya diukur ke target dahi buatan (hidung − 0.18 × lebar bahu). Tidak ada syarat `wrist_y ≤ nose_y` dan tidak ada syarat frame berturut-turut. |
| `min_kp_conf 0.5` | 🟡 | Kode memakai 0.25/0.30, hardcode. |
| `shoulder_slouch` | ✅ | Pendekatannya setara (`posture_ratio`). |
| Window berbasis waktu 60 detik | ✅ | Window pendek sudah berbasis waktu. |
| Window panjang 30 menit | ❌ | Hanya ada riwayat burnout 15 menit. |
| PERCLOS (abaikan kedipan < 400 ms) | ❌ | `eye_closed_ratio` ikut menghitung kedipan normal. |
| Blink rate berbasis waktu | 🟡 | Ada, tapi berbasis jumlah frame (1–8 frame), sehingga bergantung pada FPS. |
| `yawn_count` (≥1 detik + debounce 10 detik) | ❌ | Yang ada `yawn_ratio` per frame, tanpa durasi minimum. Jadi bicara bisa terhitung menguap. |
| `furrow_ratio`, `hand_forehead_count` | 🟡 | Ada versi ratio, tapi tidak ada count. |
| Bobot skor (0.30/0.20/0.20/0.15/0.10/0.05) dan normalisasi spesifikasi | ❌ | Bobot berbeda dan ada komponen `slouch` (0.15) di skor stres. Spesifikasi menaruh postur di habit, bukan di skor. |
| Level `rendah/sedang/tinggi` (35/65) | ❌ | Level yang dipakai `Normal/Waspada/Tinggi` (40/70). |
| EMA α = 0.3 | ❌ | |
| Hysteresis (naik 20 detik, turun 30 detik) | ❌ | |
| ⚠️ `sensitivity` mengalikan skor | — | Tidak ada di spesifikasi. Nilai `1.5` di config yang belum di-commit menaikkan semua skor 50%. |
| `components` untuk transparansi | ❌ | |
| Kalibrasi first-run 20 detik + tabel `user_calibration` | 🟡 | Kalibrasi 10 detik dijalankan **setiap sesi** (bukan sekali) dan disimpan ke `sessions`. Tidak ada `face_size_ratio` dan EAR tertutup. |
| Burnout: MA 30 menit + 4 jam, slope regresi, ≥55 selama ≥2 jam | ❌ | ⚠️ Burnout sudah terpicu bila rata-rata 15 menit ≥ 60 **setelah data baru 2 menit**. Ini terlalu cepat untuk disebut "burnout". |
| `habit_evaluator.py` (need_break, bad_posture, too_close, dry_eyes, late_night, good_break, doing_great) | ❌ | Tidak ada modul habit. Sebagian (postur, jarang berkedip) digabung ke `dominant_reason`. |
| `user_away` > 10 detik | ❌ | |
| Test sintetis + `python -m src.analysis.replay` | ❌ | |

## FASE 5: Virtual Buddy

| Item | Status | Temuan |
|---|---|---|
| Widget desktop Frameless + StaysOnTop + Tool | 🟡 | `overlay_window.py` memakai flag ini, tapi bentuknya kartu notifikasi yang muncul lalu hilang, bukan kucing yang selalu ada di bawah layar. |
| `WindowDoesNotAcceptFocus`, `WA_TranslucentBackground` | ❌ | |
| `availableGeometry()` | ✅ | |
| Multi-monitor, skala DPI | ❌ | |
| Click-through `setMask`, drag, klik kiri/kanan | ❌ | |
| Deteksi fullscreen / DND, "Diam hari ini" | ❌ | |
| Tray (pause, buka, keluar) | ✅ | Ada pause, snooze, preview, buka, dan keluar. |
| 12 ekspresi (idle … celebrate) | ❌ | Baru ada 5 ekspresi lain (`menguap`, `minum`, `istirahat`, `olahraga`, `jalan`). |
| `sprite_player.py`, `animations.yaml`, transform, overlay QPainter | ❌ | `cat_canvas.py` menggambar kucing statis dengan QPainter (bisa dijadikan referensi fallback prosedural). |
| Script placeholder Lampiran B | ❌ | Belum dijalankan. |
| State machine enum + tabel transisi | ❌ | `decision_engine.py` memakai if-else. |
| Speech bubble 8 detik, ≤90 karakter, tombol Oke/Nanti/Jangan hari ini | 🟡 | Overlay punya durasi dan tombol "Mengerti". Pesannya **lebih dari 90 karakter** (contoh: pesan burnout ±110 karakter). |
| `assets/advice_id.yaml`, ≥4 variasi per kunci | ❌ | Saran di-hardcode di `messages.py`, dan hanya 1–2 variasi per kunci. |
| Cooldown per kunci 15 menit + global 3 menit + maks 8/jam + eskalasi | 🟡 | Hanya ada cooldown global 180 detik (dipotong setengah untuk level tinggi) dan snooze. |
| Catat saran + respons ke DB | ✅ | `buddy_log` + `update_buddy_action`. |
| `python -m src.buddy.demo` + debug panel 12 tombol | ❌ | Ada tombol preview ekspresi di UI utama. |

## FASE 6: Database & ringkasan harian

| Item | Status | Temuan |
|---|---|---|
| `data/stress_buddy.db` + `schema_version` | ❌ | Nama file `wellness_buddy.db`, tanpa versi skema. |
| Tabel `sessions, stress_log, events, advice_log, daily_summary, user_calibration` | 🟡 | Yang ada `sessions, score_log, events, buddy_log`. Tabel `daily_summary` dan `user_calibration` tidak ada, dan nama serta kolomnya berbeda. |
| Tulis batch (flush 10 detik) lewat thread khusus | ❌ | Tulis langsung dari thread kamera setiap 5 detik. |
| Retensi 90 hari | ❌ | |
| Tombol hapus semua data | ✅ | `clear_all_data()`. |
| `health_score` | ❌ | |
| Dashboard: kartu, grafik per jam, istirahat, 7 hari, insight, ekspor CSV | 🟡 | Ada ringkasan harian dan mingguan, tapi tidak ada ekspor CSV, insight otomatis, atau hitungan istirahat. |

## FASE 7: Integrasi

| Item | Status | Temuan |
|---|---|---|
| `pipeline.py` + EventBus | ❌ | |
| Wizard kalibrasi saat first-run | 🟡 | Kalibrasi otomatis di setiap sesi. |
| Disclaimer first-run | 🟡 | Kalimat batasan ada di dashboard, tapi tidak ditampilkan sebagai dialog saat pertama kali dijalankan. |
| Indikator kamera di tray | ✅ | |
| **Jeda melepas kamera** | ❌ | `set_paused` hanya menghentikan loop, sementara `VideoCapture` tetap terbuka. Akibatnya lampu kamera tetap menyala. Ini juga **penting untuk privasi**. |
| Retry webcam tiap 5 detik | ❌ | Setelah 15 kali gagal, aplikasi pindah ke **mode simulasi** dengan data palsu. Data palsu itu ikut tercatat ke DB seolah-olah data nyata. |
| Restart thread inferensi saat crash | ❌ | |
| Adaptive frame skip, `performance_mode` | ❌ | |
| `logs/app.log` | ❌ | Memakai `print`. |
| `--self-test` | ➕ | Fitur tambahan yang berguna dan bisa dipertahankan. |

## FASE 8 & 9: Pengujian dan laporan

Semuanya ❌: robustness per kondisi, event-level precision/recall, benchmark CPU/RAM/latensi, korelasi dengan kuesioner, uji UX, uji kegagalan, dan `scripts/make_report_assets.py`.

## Yang layak dipindahkan ke struktur baru

| Dari kode lama | Ke modul spesifikasi | Catatan |
|---|---|---|
| `compute_ear`, perhitungan MAR dan jarak alis di `ear_landmark.py` | `detection/ear_calculator.py` | Pisahkan EAR kiri/kanan dan hapus fallback Haar yang memalsukan EAR. |
| Rumus postur di `pose_yolo.py` | `detection/pose_estimator.py` | Ganti logika hand-to-forehead sesuai 4.1 dan pindahkan semua angka ke config. |
| Pemuatan YOLO `.pt`/`.onnx` di `face_yolo.py` | `detection/face_detector.py` | Keluarkan confidence 0..1 per kelas, jangan boolean. |
| Split per subjek di `prepare_yolo_dataset.py` | `training/` + `scripts/collect_data.py` | Hapus MRL dan template box tetap. Box dibuat dari landmark atau anotasi manual. |
| `database.py` (thread-safe, hapus data) | `storage/db.py`, `repository.py` | Sesuaikan skema bagian 6. |
| `theme.py`, `DESIGN.md` (palet WCAG) | `ui/` | Bisa dipakai apa adanya. |
| Mode `--simulate` dan `--self-test` | `analysis/replay.py`, tests | Ubah menjadi generator `sample.jsonl` untuk replay dan test. |

## Langkah berikutnya

Ikuti urutan bagian 12 spesifikasi, mulai dari **FASE 1** di branch baru:
- Python 3.11 dan versi library yang dipin.
- Struktur folder sesuai spesifikasi.
- `config.yaml` sesuai bagian 10.
- `scripts/benchmark_fps.py`.
- `pytest` yang bisa jalan.
- `docs/DECISIONS.md` yang mencatat keputusan port dari kode lama.
