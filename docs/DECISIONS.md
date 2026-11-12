# Catatan Keputusan

Keputusan yang diambil saat spesifikasi ambigu atau saat kode menyimpang dari spesifikasi. Format: ID, konteks, keputusan, alasan.

## FASE 1

### D-001 · Kode lama dipindah ke `legacy/`
- **Konteks:** repo sudah berisi prototipe dari Antigravity dengan struktur yang berbeda dari bagian 3. Hasil auditnya ada di `docs/AUDIT_SPEC.md`.
- **Keputusan:** semua kode lama (`main.py`, `src/`, `scripts/`, `config.yaml`, `README.md`, `DESIGN.md`, `requirements.txt`) dipindah apa adanya ke `legacy/` dengan `git mv`, sehingga riwayatnya tetap terlacak. Venv lama (Python 3.12) diganti nama menjadi `.venv-legacy/`.
- **Alasan:** struktur baru bisa dibangun bersih tanpa bentrok nama, sementara kode lama tetap jadi referensi untuk di-port per fase. `legacy/` **tidak dimaksudkan untuk dijalankan** karena path `models/` di dalamnya kini relatif ke `legacy/`.

### D-002 · Python 3.11 via `uv`
- **Konteks:** spesifikasi meminta Python 3.10 atau 3.11. Di mesin ini, `python@3.11` dari Homebrew rusak di macOS 26: `pyexpat` membutuhkan simbol yang tidak ada di `/usr/lib/libexpat`, sehingga `ensurepip` gagal.
- **Keputusan:** Python 3.11.18 dipasang lewat `uv python install 3.11` (build mandiri yang membawa `libexpat` sendiri). Venv dibuat dengan `uv venv --python 3.11 .venv`.
- **Alasan:** cara paling sederhana yang lolos DoD. `uv` juga yang dulu dipakai untuk venv lama.

### D-003 · Pemilihan device `auto` = CUDA → MPS → CPU
- **Konteks:** spesifikasi hanya menyebut "log `torch.cuda.is_available()`, fallback CPU dengan imgsz=480". Laptop pengembangan adalah Mac Apple Silicon (tanpa CUDA).
- **Keputusan:** `src/core/device.py` memilih CUDA bila ada, lalu Apple MPS, lalu CPU. Accelerator yang diminta tapi tidak tersedia jatuh ke CPU dengan peringatan. Di CPU, `imgsz` dibatasi maksimal 480. `device.py` adalah modul tambahan di luar daftar bagian 3.
- **Alasan:** MPS jauh lebih cepat daripada CPU di Mac. Target performa bagian 8.3 tetap bisa diuji dengan `--device cpu`.

### D-004 · Bagian `logging` di config
- **Konteks:** bagian 7 meminta "logging ke `logs/app.log`, level dari config", tetapi bagian 10 tidak punya kunci untuk itu.
- **Keputusan:** ditambahkan `logging: {level: INFO, file: logs/app.log}`.

### D-005 · Validasi config saat dimuat
- **Keputusan:** `load_config()` menolak config bila: ada bagian yang hilang, `score.weights` tidak berjumlah 1.0, `low ≥ high`, `fusion_w_ear + fusion_w_yolo ≠ 1.0`, atau `privacy.save_frames = true`.
- **Alasan:** aturan "total bobot = 1.0" dan "frame tidak boleh ditulis ke disk" ditegakkan di satu tempat.

### D-006 · Benchmark menerima sumber selain webcam
- **Keputusan:** `scripts/benchmark_fps.py` punya opsi `--source <video/gambar>` dan `--no-show`. Bobot `yolov8n.pt` resmi diunduh otomatis ke `models/`.
- **Alasan:** benchmark bisa diuji ulang tanpa kamera (CI atau mesin tanpa webcam). Mode default tetap webcam + jendela live sesuai DoD.

## FASE 2

### D-007 · MRL Eye tidak dipakai untuk detektor
- **Konteks:** MRL berisi potongan satu mata (inframerah, ±80–150 px), bukan frame webcam wajah penuh. Kode lama memberinya box 90% gambar.
- **Keputusan:** MRL tidak diimpor ke dataset detektor.
- **Alasan:** domainnya beda dan tidak cocok dengan konvensi box D-009.

### D-008 · Skema nama file mentah `<subjek>__<petunjuk>__<kondisi>__<id>`
- **Keputusan:** semua gambar di `data/raw/` diberi nama sesuai skema ini (`training/common.py`). Contoh subjek: `self-andi`, `nthu-001`, `uta-12`.
- **Alasan:** split per orang dan uji per kondisi (FASE 8.1) butuh identitas subjek dan kondisi. Pemisah `__` tetap utuh walaupun Roboflow menambahkan akhiran `_jpg.rf.<hash>`.

### D-009 · Konvensi box
- **Keputusan:**
  - `mata_*`: **satu box untuk kedua mata**.
  - `menguap`: wajah bawah, dari pangkal hidung sampai dagu, termasuk mulut.
  - `kerutan_dahi`: dahi dan alis.
  - Rincian lengkap ada di `docs/ANOTASI.md`.
- **Alasan:** box satu mata di jarak 90 cm hanya sekitar 20 px, sulit untuk YOLOv8n di imgsz 480. Box gabungan lebih besar dan stabil. Kondisi satu mata terpejam jarang terjadi dan EAR per mata tetap tersedia dari MediaPipe. Dengan konvensi ini tidak ada aturan kiri/kanan, jadi horizontal flip aman dipakai.

### D-010 · Overlay tercetak di NTHU-DDD dihapus
- **Konteks:** sekitar 97% frame NTHU versi Kaggle (samymesbah) memuat kotak berwarna tercetak (wajah hijau, mata biru, mulut merah) yang sering tidak pas. Model bisa belajar mengenali garis itu, bukan wajahnya.
- **Keputusan:** saat import, piksel berwarna (selisih kanal > `overlay_color_diff`) di-inpaint, karena frame aslinya grayscale. Pengecekan acak menunjukkan 0 frame yang masih berwarna.
- **Catatan FASE 3:** frame NTHU grayscale, sedangkan webcam berwarna. Kesenjangan domain ini perlu ditutup dengan rekaman relawan (berwarna) dan/atau augmentasi grayscale saat training.

### D-011 · Pra-anotasi dari landmark MediaPipe
- **Keputusan:**
  - `scripts/preannotate.py` membuat draf box: EAR < 0.18 → `mata_tertutup`, EAR > 0.24 → `mata_terbuka`, di antaranya ditandai `ragu_mata`; MAR > 0.45 → `menguap`.
  - `kerutan_dahi` hanya diberikan dari petunjuk instruksi, karena landmark tidak bisa menilai kerutan dengan andal.
  - Label yang sudah ada tidak pernah ditimpa tanpa `--overwrite`.
- **Alasan:** anotasi ribuan frame dari nol terlalu lambat. Draf ini mempercepat kerja, tapi **tetap harus dikoreksi manual** sesuai spesifikasi.

### D-012 · Algoritma split dan pengecekan DoD
- **Split:** subjek utuh dibagikan secara greedy agar mendekati 70/20/10 jumlah gambar. Dengan ≥ 3 subjek, val dan test masing-masing dijamin dapat satu subjek; dengan 2 subjek, val yang dijamin. Ada `--assign` untuk memaksa subjek ke split tertentu.
- **Augmentasi:** offline dan hanya di train (`__aug<k>`). Ultralytics tetap menjalankan augmentasi online saat training.
- **DoD `yolo val`:** dijalankan lewat `training/check_dataset.py` dengan YOLOv8n 4 kelas tanpa bobot, agar nama kelas cocok. Metriknya ≈ 0 dan memang tidak bermakna di fase ini.
- **Sampling NTHU:** frame-nya berurutan dari video, sehingga diambil merata per grup (subjek × kacamata × skenario). Kuota per petunjuk ada di config (`menguap` 300 agar instance menguap cukup).

## FASE 3

### D-013 · Input detektor grayscale
- **Konteks:** seluruh data NTHU grayscale, sedangkan webcam dan rekaman relawan berwarna (D-010).
- **Keputusan:** `det.grayscale: true`.
  - `split_dataset.py` mengubah gambar berwarna menjadi grayscale 3 kanal. Gambar yang sudah abu-abu cukup di-link.
  - `FaceDetector.predict()` mengubah frame webcam dengan cara yang sama.
  - Augmentasi warna Ultralytics (`hsv_h`, `hsv_s`) dimatikan.
- **Alasan:** cara paling sederhana agar training dan inferensi melihat domain yang sama. Warna tidak dibutuhkan untuk membedakan mata tertutup, menguap, atau kerutan dahi.

### D-014 · Training di Apple MPS
- **Keputusan:** training memakai device `auto`, yang di Mac berarti MPS. Hyperparameter mengikuti spesifikasi: `epochs=100, imgsz=640, batch=16, patience=20`, plus `deterministic=True, seed=42`.
  - Ultralytics memaksa `workers=0` di MPS.
  - Kecepatannya sekitar 4 menit per epoch untuk 5.114 gambar train (2.557 asli + 2.557 augmentasi).
- **Status (12 Nov 2026):** run pertama dengan label draf dihentikan atas permintaan setelah 2 epoch (val mAP50 0,90 → 0,86). Angka tinggi ini wajar karena label draf berasal dari aturan MediaPipe yang mudah ditiru. Checkpoint ada di `runs/detect/face/` (bisa dilanjutkan dengan `--resume`). `models/best.pt` belum dibuat, sehingga DoD FASE 3 belum terpenuhi.
- **Catatan:** beberapa operasi MPS tidak punya implementasi deterministik (muncul peringatan), sehingga hasil bisa sedikit berbeda antar-run. Untuk run final, GPU CUDA atau Google Colab lebih cepat; caranya cukup `--device cuda`.

### D-015 · Evaluasi dan status label
- **Keputusan:**
  - `training/evaluate.py` mengevaluasi split **test** (subjek yang tidak dipakai saat training) di imgsz training, lalu menulis `docs/eval/metrics.md`, `metrics.json`, confusion matrix, kurva PR/F1, dan kurva training.
  - Flag `--label-status` defaultnya `draf`, dan laporannya diberi peringatan "tidak untuk Bab IV". Pakai `--label-status final` hanya setelah label dikoreksi manual.
- **Alasan:** mencegah angka dari label otomatis tidak sengaja masuk laporan.
- **Catatan:** inferensi runtime memakai `inference.imgsz=480` agar lebih cepat. Selisih akurasi 640 vs 480 diukur di FASE 8 (`evaluate.py --imgsz 480`).

## FASE 4

### D-016 · Detektor YOLO wajah opsional
- **Konteks:** training ditunda karena belum ada hardware yang memadai, sehingga `models/best.pt` belum ada.
- **Keputusan:** seluruh pipeline jalan tanpa YOLO wajah.
  - EAR (sumber utama mata) tetap dari MediaPipe.
  - `yawn` diambil dari MAR > `landmark.mar_yawn`.
  - `brow_furrow` diambil dari rasio alis di bawah baseline kalibrasi × (1 − `landmark.brow_drop`). Ini rencana cadangan dari risiko `kerutan_dahi`.
  - Begitu `models/best.pt` ada, nilai confidence YOLO otomatis dipakai dan `det_available = true`.

### D-017 · Tambahan pada kontrak data dan modul `engine.py`
- **`FrameFeatures`** mendapat field tambahan `det_available`, `mar`, `brow_ratio`, dan `nose_xy`. Gunanya: fusi hanya dipakai saat YOLO tersedia, ada data mentah untuk cadangan landmark dan kalibrasi, dan posisi hidung untuk `trajectory_motion`.
- **`StressSnapshot`** mendapat `raw_score` (sebelum EMA) dan `burnout_risk`.
- **Dataclass baru:** `HabitAlert` dan `PresenceEvent`.
- **`src/analysis/engine.py`** menggabungkan window, scorer, burnout, dan habit di balik `add()`/`tick()`. Modul ini dipakai oleh `replay.py` dan pipeline FASE 7, supaya logikanya hanya ada di satu tempat.

### D-018 · Rumus postur dan jendela panjang
- **`shoulder_slouch`:** spesifikasi menulis `(nose_y − mean(shoulder_y)) / lebar_bahu` dan "makin kecil = kepala makin turun". Karena sumbu y gambar mengarah ke bawah, rumus itu justru membesar saat kepala turun. Yang dipakai: `(mean(shoulder_y) − nose_y) / lebar_bahu`, yang memang mengecil saat kepala turun.
- **`head_pitch_proxy`:** `(nose_y − mean(shoulder_y)) / tinggi_frame`.
- **Jendela panjang 30 menit:** tidak disimpan per frame. Kebutuhannya dipenuhi oleh timer di `HabitEvaluator` (durasi 2–3 menit dan 50 menit) dan rata-rata per menit di `BurnoutTrend`, sehingga memorinya kecil.

### D-019 · Detail perhitungan yang ambigu di spesifikasi
- **Fusi mata:** sinyal EAR biner, sinyal YOLO berupa confidence, lalu dibandingkan dengan `ear.fusion_threshold` 0,5. Dengan bobot 0,6/0,4, EAR praktis menentukan saat keduanya ada, sedangkan YOLO menjadi cadangan saat EAR tidak ada.
- **Blink abnormal:** jarak blink rate di luar rentang normal `score.blink_normal` [15, 20] per menit, dibagi 15.
- **Motion:** rata-rata standar deviasi posisi hidung ternormalisasi, dibagi `score.norm.motion` (0,05).
- **Menguap:** episode ≥ 1 detik dihitung sekali. Episode berikutnya yang mulai < 10 detik kemudian tidak dihitung, dan hitungan disimpan antar-tick agar tidak ganda.
- **Skor:** dihitung bila jendela sudah berisi ≥ `score.min_window_sec` (20 detik) dan pengguna terlihat. Tidak dihitung selama `user_away`.
- **Burnout slope:** satuannya poin per jam, dari regresi linier rata-rata bergerak 30 menit selama 4 jam terakhir. `slope_min` 0,5 diartikan 0,5 poin per jam.
- **late_night:** jam ≥ 23 sampai < 04 dihitung sebagai satu malam, jadi alert hanya muncul sekali.
- **good_break:** away 5–15 menit setelah kerja ≥ `habit.long_work_min` (30 menit).
- **doing_great:** 2 jam berturut-turut di level rendah.

## Catatan untuk FASE 2: dataset yang tersedia

Tiga dataset Kaggle yang dimiliki tim:

| Dataset | Isi | Cocok untuk | Batasan |
|---|---|---|---|
| [MRL Eye Dataset](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset) | Potongan **satu mata** (inframerah, grayscale) dengan label awake/sleepy, banyak subjek | Kontras `mata_terbuka`/`mata_tertutup` level crop; pengujian EAR tidak bisa | Bukan wajah penuh, jadi tidak sama dengan frame webcam. Tidak ada bounding box. |
| [UTA-RLDD](https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset) | **Video webcam** banyak partisipan (alert / low vigilant / drowsy), variasi kacamata, cahaya, dan latar | Sumber frame utama untuk anotasi mata dan menguap; juga untuk uji event (FASE 8.2) | Label hanya per video, tanpa box. Frame harus diekstrak lalu dianotasi. |
| [NTHU-DDD multi-class](https://www.kaggle.com/datasets/samymesbah/nthu-dataset-ddd-multi-class) | Frame wajah dengan label drowsy (yawning, slowBlinkWithNodding, sleepyCombination) / notdrowsy | Kelas `menguap` dan `mata_tertutup` | Jumlah subjek sedikit dan tanpa box (kode lama memakai box posisi tetap, sehingga tidak valid). |

Konsekuensinya:
- **Tidak satu pun dataset punya bounding box.** Anotasi (Roboflow/CVAT/Label Studio, atau pra-anotasi otomatis dari landmark MediaPipe lalu dikoreksi manual) wajib dilakukan.
- **`kerutan_dahi` tidak ada di ketiga dataset**, jadi harus direkam sendiri dari relawan (`scripts/collect_data.py`).
- Keputusan konvensi box mata (per mata atau kedua mata) akan dicatat di FASE 2.
