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
