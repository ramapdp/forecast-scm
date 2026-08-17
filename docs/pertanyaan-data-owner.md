# Pertanyaan untuk Pemilik Data — Kebuli Yaman

**Tanggal:** 17 Agustus 2026 *(revisi keempat — semua pertanyaan terjawab)*
**Untuk:** Tim SCM / pemilik data Kebuli Yaman
**Dari:** Tim penelitian peramalan permintaan

## Sudah terjawab — terima kasih ✅

| Pertanyaan | Jawaban |
|---|---|
| Nama kota 8 cabang (Kutabumi, Ciampea, Cibinong, Cikeas, Jatimakmur, Harapan Indah, Sepatan, Rawalumbu) | Koreksi kami benar, termasuk KY001 Kutabumi = Kabupaten Tangerang |
| Barang aqiqah (Kambing Aqiqah Betina/Jantan, Lunch Box Aqiqah) | Benar, memang berjalan lewat pesanan |
| Jadwal kirim Bintara, Citayam, Grand Wisata Bekasi | Benar, Kawasan 2 — Selasa & Jumat |
| Tingkat keamanan stok (service level) | 9 dari 10, sama untuk semua barang — pengiriman dari pusat mencakup semua item sekaligus |
| **A. Arsip jadwal kirim lokasi lama** | **`outlet_mapping.csv` sudah menjadi arsipnya; jadwal dicocokkan lewat nama outlet baru dan itu yang berlaku (2026-08-17)** |
| **C. KY068 Kramatwatu, 28 Juni – 10 Juli 2025** | **Tutup sementara — bukan celah pencatatan (2026-08-17)** |
| **D. Cikarang Pusat** | **Belum ada tanggal buka; menunggu data baru (2026-08-17)** |
| **B. Kambing Oven (`FGS.00048`)** | **Masih dijual — memang barang slow mover; buku menu memuatnya sebagai paket tersendiri (2026-08-17)** |

**Semua pertanyaan sudah terjawab.** Tidak ada yang menahan pekerjaan.

---

## Tindak lanjut dari jawaban 17 Agustus — sudah kami kerjakan

**A. Jadwal kirim lokasi lama.** Kami cek ulang `dataset/outlet_mapping.csv`:
ke-9 cabang yang pernah relokasi **semuanya sudah punya `kawasan` dan
`hari_pengiriman` di bawah nama outlet barunya**, tidak ada satu pun yang
kosong. Jadi tidak ada yang perlu dikirim lagi — dan karena jadwal outlet baru
memang yang berlaku, 205.513 baris pra-relokasi (13,7% dataset) berhenti menjadi
asumsi. Batasan B-8 di `docs/batasan-penelitian.md` kami tutup.

*Satu catatan pendukung:* `KY029 - Kebuli Yaman Cinere` (lokasi lama Bintara)
masih punya barisnya sendiri di `outlet_mapping.csv`, dan jadwalnya **sama
persis** dengan Bintara — Kawasan 2, Selasa & Jumat. Satu-satunya lokasi lama
yang bisa kami bandingkan langsung, dan hasilnya cocok.

**C. KY068 Kramatwatu.** Kami cek data mentahnya: di `dataset/csv/jan-des-25.csv`
transaksi terakhir sebelum jeda adalah **27 Juni 2025** dan baru muncul lagi
**11 Juli 2025** — persis kosong 28 Juni – 10 Juli, tidak ada satu baris pun
yang tercecer. Cocok dengan keterangan tutup sementara. Sudah kami catat di
`dataset/outlet_closures.csv`, sehingga 13 hari itu **tidak lagi dihitung
sebagai permintaan nol**, melainkan sebagai hari cabang tidak beroperasi.

**D. Cikarang Pusat.** Tetap tercatat tutup sejak 1 Desember 2025 tanpa tanggal
buka. Akan kami isi begitu data periode baru masuk.

---

## B. Kambing Oven (`FGS.00048`) — terjawab 17 Agustus 2026 ✅

**Jawaban: masih dijual, dan memang barang slow mover.** Bukti pendukungnya
buku menu: item ini muncul sebagai **paket tersendiri — "Kambing Muda Rempah
Oven (Satu Ekor)", termasuk 50 porsi Nasi Kebuli Rempah polos, acar & sambal —
di kotak terpisah dari Paket Aqiqah**.

Konsekuensinya untuk data: `FGS.00048` **tidak** kami keluarkan (batal jadi
kandidat `EXCLUDED_ITEMS`), dan `is_event_driven` tetap `false` di
`dataset/event_driven_items.csv`. Ini juga cocok dengan pemeriksaan kami
sebelumnya: barang ini tidak pernah bergerak di hari yang sama dengan SKU
aqiqah (0 dari 8 hari), jadi memang bukan bagian dari paket aqiqah.

Angka lengkapnya, untuk arsip (koreksi dari revisi sebelumnya, yang menulis
"4 unit di 1 cabang" — itu angka setelah penyaringan, bukan angka mentahnya):

| | |
|---|---|
| Total di data mentah | **10 ekor, 8 transaksi, 6 cabang** |
| Rentang | 8 Maret 2024 – 20 September 2025 |
| Cabang | KY002 Cilegon (3 transaksi/4 ekor), KY008 Depok Kelapa Dua (2 ekor), KY048 Banjar Wijaya, KY045 Graha Raya, KY003 Serang, KY052 Bantarjati Bogor (masing-masing 1 ekor) |
| Yang masuk model | **hanya KY002 Cilegon (4 ekor)** — cabang lain cuma bergerak 1 hari, jadi terbuang oleh syarat minimal 60 hari riwayat |

Ringkasnya: sekitar **1 ekor per 2 bulan untuk seluruh jaringan** — laju yang
sekarang sudah kami pahami sebagai memang segitu, bukan tanda barang berhenti.

*Satu pengamatan tambahan, tidak perlu ditanggapi:* di ke-8 hari paket ini
keluar, **Nasi Kebuli (`FGS-00004`) di cabang tersebut selalu masuk 8% hari
tertinggi cabangnya sendiri** (persentil 92–99%). Kelebihannya (105–255 porsi
di atas median) jauh di atas 50 porsi yang menempel di paket, jadi tampaknya
hari-hari itu memang hari acara besar di cabang, bukan sekadar bawaan paket.
Dengan 8 kejadian dalam 2 tahun, ini terlalu jarang untuk dijadikan fitur —
kami catat saja supaya lonjakan itu tidak salah dibaca sebagai outlier.

*(Catatan teknis, tidak perlu ditanggapi: kode `FGS.00048` memakai titik,
sedangkan `FGS-00048` dengan strip adalah Kentang Mustofa Mie Goreng. Dua barang
berbeda dengan nomor yang sama; pipeline sudah membedakannya dengan benar.)*

---

## Yang terjawab dari data — tidak perlu dijawab, hanya untuk diketahui

Kami sempat menyiapkan pertanyaan soal 9 barang kelompok Loyang dan 2 kemasan
(Mika Bento, Cup 60 ml). Setelah konfirmasi aqiqah kemarin, pertanyaan itu bisa
kami jawab sendiri: barang aqiqah sekarang jadi **contoh pembanding** — kami
tinggal melihat barang mana yang bergerak di hari dan cabang yang sama.

| Barang | Temuan | Kesimpulan |
|---|---|---|
| Cup 60 ml | Keluar bersama barang aqiqah di **100%** hari geraknya | Bagian dari paket aqiqah |
| Mika Bento | **93% jumlahnya** keluar di hari aqiqah (sekali keluar ±60 pcs; di hari biasa cuma 1 pcs) | Terutama untuk aqiqah |
| 9 barang Loyang | Hampir tidak pernah bareng aqiqah (0,9–1,4%, sama dengan barang harian biasa) | Kemasan operasional harian |

Dua hal yang ikut ketahuan dan cukup membantu:

- **Loyang, Box Loyang, dan Cup Sambal selalu keluar bersama** dalam takaran
  tetap: 1 loyang = 1 box = 2 cup sambal, konsisten di hampir 100% hari. Jadi
  sebenarnya ada 3 barang (Mini, Sedang, Besar), bukan 9.
- **Loyang Besar bukan barang acara**, hanya ukuran yang lebih jarang laku — ia
  keluar di 80,7% hari yang sama dengan Loyang Sedang.

Kalau ada satu saja dari kesimpulan di atas yang terasa keliru menurut
pengalaman di lapangan, mohon beri tahu — itu lebih berharga daripada angka
kami.

---

Sekali lagi terima kasih. Semua jawaban kami catat beserta tanggalnya di
dokumentasi penelitian, supaya jelas mana yang keputusan Kebuli Yaman, mana yang
kesimpulan dari data, dan mana yang masih asumsi kami.
