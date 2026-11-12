# Uji Kegagalan & Kondisi Nyata (FASE 8.6 + DoD FASE 7)

Isi kolom **Hasil** dan **Catatan** saat menguji dengan aplikasi asli (`python -m src.main`).
Kolom "Otomatis" menunjukkan perilaku yang sudah dicek oleh test (`pytest`).

| # | Skenario | Langkah | Hasil yang diharapkan | Otomatis | Hasil | Catatan |
|---|---|---|---|---|---|---|
| 1 | Jalan 1 jam | Jalankan aplikasi 60 menit sambil bekerja biasa | Tidak crash; RAM stabil (cek Activity Monitor tiap 15 menit) | Sebagian (uji 10 menit offscreen: RAM stabil) | | |
| 2 | Pejam mata lama | Pejam 3–5 detik, beberapa kali dalam 1 menit | PERCLOS naik; Buddy SLEEPY atau WORRIED; saran "sleepy" | Ya (replay sintetis) | | |
| 3 | Menguap 3× | Menguap lebar 3 kali dalam 1 menit | 3 event `yawn` di DB; Buddy SLEEPY | Ya (sliding window) | | |
| 4 | Bicara | Bicara/membaca keras 30 detik | Tidak terhitung menguap (durasi < 1 detik) | Ya | | |
| 5 | Tangan ke dahi | Tangan di dahi 3 detik, 4 kali | Event `hand_to_forehead`; skor naik | Ya | | |
| 6 | Membungkuk | Duduk membungkuk 3 menit (setelah kalibrasi) | `bad_posture`; Buddy WORRIED + saran postur | Ya (habit) | | |
| 7 | Terlalu dekat | Wajah ±30 cm dari layar 2 menit | `too_close`; Buddy CURIOUS | Ya (habit) | | |
| 8 | Pergi dari kursi | Tinggalkan kursi > 2 menit | `user_away` setelah 10 s; Buddy tidur (zZz) setelah 2 menit | Ya | | |
| 9 | Kembali | Duduk lagi setelah 5–15 menit (setelah kerja ≥ 30 menit) | `user_returned` + `good_break`; Buddy WAKE_GREET lalu HAPPY | Ya | | |
| 10 | Masker | Pakai masker 1 menit | Mata tetap terbaca bila landmark terdeteksi; tidak crash | – | | |
| 11 | Banyak orang | Dua orang di depan kamera | Pose memakai orang terbesar; tidak crash | Sebagian | | |
| 12 | Kamera mati mendadak | Cabut webcam USB / buka aplikasi lain yang memakai kamera | Tray "Kamera tidak tersedia", Buddy tidur, dicoba ulang tiap 5 s, lanjut sendiri | Ya (sumber hilang) | | |
| 13 | Jeda | Tray → Jeda kamera | Lampu kamera mati; tidak ada `user_away` palsu saat lanjut | Ya | | |
| 14 | Fullscreen game/presentasi | Buka video/presentasi fullscreen | Kucing sembunyi, saran ditahan, inferensi berhenti | Sebagian (deteksi butuh macOS Quartz/Windows) | | |
| 15 | Multi-monitor | Pengaturan → Monitor 1, buka ulang | Kucing pindah ke bawah monitor kedua, di atas taskbar/dock | – | | |
| 16 | DPI 125% / 150% | Ubah skala tampilan OS | Kucing tetap 128–192 px logis, tidak buram | – | | |
| 17 | Klik tembus | Klik area transparan di sekitar kucing | Klik sampai ke aplikasi di bawahnya | – | | |
| 18 | Tidak mencuri fokus | Mengetik di editor saat bubble muncul | Ketikan tidak terputus | – | | |
| 19 | Cahaya redup / backlight | Ulangi #2–#3 dalam kondisi redup dan membelakangi jendela | Catat penurunan deteksi (bandingkan robustness.md) | – | | |
| 20 | Kacamata | Ulangi #2 dengan kacamata | Catat penurunan deteksi | – | | |
| 21 | Hapus semua data | Pengaturan → Hapus semua data | Dashboard kosong; disclaimer tidak muncul lagi | Ya | | |
