# Design Direction: Kucing Teman Kerja (Virtual Buddy Pemantau Kelelahan)

> Status: **draft without direction** (menggunakan pengaturan jujur `Dial: ENERGY 1 / RHYTHM 1 / MOTION 1` sesuai aturan R-37 karena belum ada panduan merek khusus dari pengguna).

## Design Read
Reading this as: aplikasi utilitas kesehatan kerja desktop untuk pengguna PC harian, dengan bahasa visual tenang berfokus pada keterbacaan indikator dan privasi lokal, dial `ENERGY 1 / RHYTHM 1 / MOTION 1`.

## Tiga Dial (Part 3)
- **ENERGY 1 (Calm)**: Antarmuka pendamping kerja tidak boleh memicu stres visual tambahan. Warna berbasis kertas hangat (terang) dan batu tulis gelap (gelap).
- **RHYTHM 1 (Uniform & Predictable)**: Tata letak dibuat konsisten antar tab agar pengguna yang sedang lelah dapat memindai metrik tanpa beban kognitif.
- **MOTION 1 (Static / State-Driven)**: Tidak ada animasi berulang tanpa henti (tanpa *pulsing dot*). Overlay Buddy hanya muncul ketika dipicu ambang skor dan menutup otomatis setelah durasi selesai.

## Palet Warna & Kontras (WCAG AA Terverifikasi)

### Mode Terang (Default)
- **Latar Utama (`#FAF8F5`)**: Warna kertas hangat untuk mengurangi silau layar dibandingkan putih murni `#FFFFFF`.
- **Permukaan Panel (`#F0ECE1`)**: Memisahkan kelompok informasi dari latar utama tanpa bayangan mengambang.
- **Teks Utama (`#1C1917`)**: Kontras 16.58:1 terhadap `#FAF8F5` dan 14.77:1 terhadap `#F0ECE1` (Lulus WCAG AAA).
- **Teks Sekunder (`#44403C`)**: Kontras 9.29:1 terhadap `#FAF8F5` dan 8.27:1 terhadap `#F0ECE1` (Lulus WCAG AAA).
- **Garis Tepi & Kontrol (`#78716C`)**: Kontras 4.42:1 terhadap `#FAF8F5` dan 3.94:1 terhadap `#F0ECE1` (Lulus batas non-teks 3:1).
- **Satu Aksen Sengaja (`#9A3412`)**: Terakota hangat untuk aksi utama dan indikator fokus; kontras 7.45:1 terhadap `#FAF8F5` dan teks `#FFFFFF` di atasnya memiliki kontras 7.98:1.
- **Warna Status (Selalu Berpasangan dengan Label Teks)**:
  - Normal (`#166534`): Kontras 6.72:1 terhadap `#FAF8F5`.
  - Waspada (`#92400E`): Kontras 6.46:1 terhadap `#FAF8F5`.
  - Tinggi (`#991B1B`): Kontras 6.91:1 terhadap `#FAF8F5`.

### Mode Gelap (Toggle Aktif)
- **Latar Utama (`#18181B`)**: Abu batu gelap untuk pemakaian malam hari.
- **Permukaan Panel (`#27272A`)**: Membedakan area kartu data dari latar utama.
- **Teks Utama (`#F4F4F5`)**: Kontras 15.65:1 terhadap `#18181B` dan 12.69:1 terhadap `#27272A`.
- **Teks Sekunder (`#D4D4D8`)**: Kontras 11.45:1 terhadap `#18181B` dan 9.28:1 terhadap `#27272A`.
- **Garis Tepi & Kontrol (`#A1A1AA`)**: Kontras 6.32:1 terhadap `#18181B` dan 5.12:1 terhadap `#27272A` (Lulus batas non-teks 3:1).
- **Satu Aksen Sengaja (`#FDBA74`)**: Amber hangat untuk fokus dan tombol utama; kontras 10.42:1 terhadap `#18181B` dan teks `#18181B` di atasnya memiliki kontras 10.42:1.

## Alasan Satu Baris untuk Setiap Keputusan Visual (R-31)
- **Mengapa warna ini**: Palet kertas hangat (`#FAF8F5`) dan aksen terakota (`#9A3412`) mengurangi ketegangan mata pada aplikasi yang justru memantau kelelahan visual.
- **Mengapa tata letak ini**: Bilah status kamera ditempatkan permanen di bagian paling atas agar pengguna selalu tahu status privasi kamera dan bisa menjeda pemantauan dalam satu klik.
- **Mengapa tipografi ini**: Font sistem (`SF Pro Text` / `Segoe UI` / `Sans-Serif` bawaan OS) dipilih agar ukuran huruf dapat diperbesar secara proporsional dan nyaman dibaca di layar desktop.
- **Mengapa jarak ruang (spacing) ini**: Jarak antar kelompok 16px dan padding dalam 14px memisahkan indikator waktu nyata dari riwayat sesi tanpa memadatkan layar.
- **Mengapa menggunakan panel datar**: Panel berbingkai tegas (`1px solid`) menggantikan bayangan lembut agar batas komponen melampaui rasio kontras 3:1 untuk pengguna dengan penglihatan rendah.
- **Mengapa ilustrasi kucing vektor**: Kucing digambar langsung dengan bangun geometri yang menampilkan ekspresi fungsional (`menguap`, `minum`, `istirahat`, `olahraga`, `jalan`) sesuai pemicu kondisi pengguna.
