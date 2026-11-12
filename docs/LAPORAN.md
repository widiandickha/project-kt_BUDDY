# Kerangka Laporan & Skenario Demo (FASE 9)

Peta dari setiap bab laporan ke file atau hasil di repo ini. Gambar dibuat ulang dengan:

```bash
.venv/bin/python scripts/make_report_assets.py   # -> docs/report_assets/
```

## Struktur laporan

| Bab | Isi | Sumber di repo |
|---|---|---|
| I. Pendahuluan | Latar belakang (jarang berkedip, kantuk, tegang, burnout), rumusan masalah, tujuan, batasan, manfaat | Ringkasan proyek; batasan di bagian "Keterbatasan" di bawah |
| II. Tinjauan pustaka | EAR (Soukupová & Čech 2016), PERCLOS, YOLOv8 (deteksi + pose), MediaPipe Face Mesh, indikator visual kelelahan, ergonomi kerja (aturan 20-20-20, jeda tiap 50 menit) | – |
| III. Metodologi | Arsitektur dan alur thread; kontrak data; sliding window; skor rule-based + EMA + hysteresis; tren burnout; evaluasi habit; kalibrasi; state machine Buddy; pengendali spam saran; skema database | `report_assets/architecture.png`, `workflow.png`, `score_timeline.png`, `buddy_expressions.png`; `config.yaml`; `docs/DECISIONS.md` |
| IV-a. Dataset | Sumber (NTHU-DDD, UTA-RLDD, rekaman relawan), pembersihan overlay NTHU, konvensi box, pra-anotasi + koreksi manual, split per orang 70/20/10 | `docs/dataset_report.md`, `docs/ANOTASI.md`, D-007 s.d. D-012 |
| IV-b. Hasil training | P/R/mAP per kelas di split test (subjek tidak dilatih), confusion matrix, kurva PR | `docs/eval/metrics.md`, `report_assets/model_metrics.png` (setelah training dengan `--label-status final`) |
| IV-c. Hasil pengujian | Ketahanan per kondisi, akurasi level event, performa (FPS/CPU/RAM/latensi), korelasi skor vs kuesioner, UX, uji kegagalan | `docs/eval/robustness.md`, `events.md`, `performance.md`, `stress_correlation.md`, `ux.md`, `failure_tests.md` + grafik di `report_assets/` |
| V. Diskusi | Keterbatasan, perbandingan landmark vs YOLO, dampak cahaya/kacamata, bias dataset (NTHU grayscale, sedikit subjek), etika dan privasi | Bagian di bawah |
| VI. Kesimpulan & saran | Capaian terhadap target; pekerjaan lanjutan | – |

## Keterbatasan (wajib ditulis)

- **Bukan alat diagnosis.** Skor stres adalah indikator visual perilaku di depan layar (mata, mulut, alis, tangan, postur), bukan ukuran klinis stres.
- **Data latih terbatas.** NTHU-DDD hanya 4 subjek, frame-nya grayscale/IR, dan versi Kaggle-nya memuat overlay tercetak yang harus dihapus. Kelas `kerutan_dahi` bergantung pada rekaman relawan.
- **Kondisi kamera.** Cahaya redup atau backlight, kacamata, sudut kepala, dan masker menurunkan kualitas landmark. Seberapa besar penurunannya diukur di `robustness.md`.
- **Latensi keputusan.** Skor dihitung tiap 1 detik dan memakai hysteresis 20/30 detik. Ini disengaja agar Buddy tidak mengganggu, tapi reaksinya memang tidak seketika.
- **Validasi skor.** Bobot dan ambang di `config.yaml` adalah nilai awal dari spesifikasi. Korelasinya dengan rasa lelah subjektif harus dilaporkan apa adanya, termasuk bila lemah.
- **Deteksi fullscreen** bersifat best-effort (macOS dan Windows).

## Etika & privasi

- Semua pemrosesan dilakukan di perangkat. Tidak ada panggilan jaringan saat aplikasi berjalan (unduhan hanya untuk bobot model resmi saat pertama kali).
- Frame video tidak pernah ditulis ke disk oleh aplikasi; `privacy.save_frames` dikunci `false` oleh validasi config. Database hanya berisi angka dan event.
- Kamera bisa dijeda kapan saja (perangkat benar-benar dilepas), dan ikon tray menunjukkan saat kamera aktif.
- Tombol "Hapus semua data" tersedia, dan retensi default 90 hari.
- Pengumpulan data dan sesi uji memakai persetujuan tertulis (`collect_data.py` dan `scenario_session.py` memintanya). Peserta dicatat dengan kode, bukan nama.
- Video demo tidak menampilkan wajah orang tanpa izin.

## Skenario demo 5 menit

| Menit | Langkah | Yang ditunjukkan |
|---|---|---|
| 0:00–1:00 | `python -m src.main` → disclaimer → wizard kalibrasi (duduk tegak, lalu pejam) | Privasi, kalibrasi personal, ikon tray dengan titik merah |
| 1:00–2:30 | Pejam mata 3 detik beberapa kali, menguap 2–3 kali, tangan ke dahi | Kucing berubah WORRIED/SLEEPY lalu ALERT; skor naik (bisa ditunjukkan juga lewat `python -m src.buddy.demo` bila waktu sempit) |
| 2:30–3:15 | Tunggu bubble saran, klik **Oke** / **Nanti** | Gelembung saran, cooldown, respons tercatat |
| 3:15–4:15 | Tinggalkan kursi (atau tutup kamera) > 2 menit; percepat dengan `buddy.away_sleep_sec` kecil di `data/settings.yaml` | Kucing tidur (zZz), lalu WAKE_GREET saat kembali |
| 4:15–5:00 | Tray → Buka dashboard | Kartu harian, grafik stres per jam, 7 hari, insight, ekspor CSV |

Cadangan bila webcam bermasalah saat demo:
- `python -m src.main --source <video/folder> --skip-intro`
- `python -m src.ui.dashboard --demo-data`
