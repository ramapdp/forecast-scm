# Batasan Penelitian

Daftar batasan yang melekat pada data dan perumusan masalah, bukan pada pilihan
model. Semuanya dikonfirmasi pemilik data pada **2026-08-15** kecuali disebut lain.
Ditulis untuk dikutip langsung di bab batasan/limitasi laporan.

Dokumen ini menampung batasan yang **tidak bisa dihilangkan dengan menulis kode
lebih baik**. Untuk pekerjaan yang masih bisa diselesaikan, lihat
`docs/todolist-data-preprocessing.md`; untuk state pipeline, `docs/pipeline-overview.md`.

---

## B-1. Kolom `Tanggal` adalah tanggal barang diambil, bukan tanggal pesanan masuk

**Fakta.** Setiap baris `dataset/*.csv` adalah catatan barang keluar pada saat
pelanggan mengambil barang. Ketika pelanggan memesan hari Senin untuk diambil hari
Kamis, manajer outlet mengabari pusat hari Senin juga, tetapi baris transaksinya
baru muncul di data bertanggal **Kamis**.

**Konsekuensi.**

- Seluruh deret waktu dalam proyek ini berjalan pada sumbu **waktu pengambilan**,
  bukan waktu permintaan muncul. Semua fitur (lag, rolling, kalender, event) dan
  target konsisten pada sumbu yang sama, jadi tidak ada pencampuran sumbu — tetapi
  interpretasinya harus selalu "permintaan yang terealisasi", bukan "permintaan
  yang timbul".
- Pola hari-dalam-minggu yang terlihat di data (Minggu indeks 143, Senin 77 dengan
  rata-rata 100) adalah pola **preferensi pengambilan**, bukan pola pemesanan.
- Autokorelasi tertinggi ada di lag 7 hari (0,441) dan lag 14 (0,388), keduanya di
  atas lag 1 hari (0,349). Ritme yang mengatur data ini mingguan, bukan harian.

## B-2. Buku pesanan tidak tersedia dan tidak akan tersedia

**Fakta.** Sistem POS **tidak menyimpan tanggal pesanan** di samping tanggal
pengambilan. Alasannya operasional: pesanan bisa dibatalkan, sehingga hanya
transaksi yang benar-benar terealisasi yang dicatat.

**Konsekuensi.**

- Pada hari Senin, manajer outlet sudah mengetahui sebagian permintaan hari Kamis
  dengan pasti. Informasi itu **nyata, dipakai dalam operasi, dan tidak terekam di
  data mana pun** yang tersedia untuk penelitian ini.
- Ini adalah **plafon akurasi** bagi ketiga model. Model bekerja dengan informasi
  yang lebih sedikit daripada yang dimiliki manajer outlet pada saat yang sama.
  Selisih akurasi yang tipis terhadap baseline naif harus dibaca sebagai
  keterbatasan informasi, bukan kegagalan arsitektur model.
- Batasan ini **tidak bisa diperbaiki dengan data refresh**, karena kolomnya memang
  tidak pernah direkam. Menambahkannya berarti mengubah sistem POS terlebih dahulu.

## B-3. Model diperlukan di luar pesanan; pesanan ditangani terpisah

**Fakta.** Logika bisnis Kebuli Yaman saat ini: ramalan dibutuhkan untuk permintaan
**di luar pesanan**. Pesanan itu sendiri dianggap tidak dapat diprediksi, dan tim
sudah punya cara sendiri menanganinya.

**Konsekuensi.** Ada ketidaksesuaian antara lingkup model dan isi target:

- Kolom `Kuantitas` adalah **total** barang keluar — pesanan dan non-pesanan
  bercampur di satu angka, tanpa penanda yang memisahkannya.
- `target_lead_time_cumulative` dibangun dari `Kuantitas` mentah, sehingga model
  saat ini dilatih dan dinilai atas komponen yang secara eksplisit **bukan
  tanggung jawabnya**.
- Besarannya, diukur pada 2026-08-15 dengan `is_spike` (≥5× median pair) sebagai
  proksi terbaik yang tersedia untuk pesanan besar:

  | Ukuran | Nilai |
  |---|---|
  | Baris bertanda lonjakan | 13.568 (0,90%) |
  | Volume di baris lonjakan | 3,48% dari total |
  | Kelebihan di atas batas cap | 1,00% dari total volume |
  | Selisih target mentah vs di-cap | 0,99% dari total target |
  | **Kontribusi baris lonjakan terhadap absolute error (test Des 2025)** | **11,5% dari error, hanya 2,41% baris** |

  Baseline naif `roll_mean_7 × lead_time` yang sama menghasilkan MAE 12,99 terhadap
  target mentah, 12,16 terhadap target di-cap, dan 11,78 bila baris lonjakan
  dikeluarkan sama sekali.

- **`is_spike` adalah proksi, bukan label kebenaran.** Ia mendeteksi lonjakan
  relatif terhadap median pair, yang bisa juga disebabkan tren naik atau perubahan
  level — bukan hanya pesanan. Tidak ada cara memvalidasinya tanpa data pesanan
  (lihat B-2).
- **Pengecualian sempit (dikonfirmasi 2026-08-16):** untuk tiga SKU aqiqah —
  `FGS-00018`, `FGS-00034`, `PCG-00002` — pemilik data memastikan permintaannya
  memang berjalan lewat pesanan. Di ketiga SKU itu `is_event_driven` bukan lagi
  proksi melainkan label yang benar, dan seluruh permintaannya berada di luar
  lingkup yang menjadi tanggung jawab model. Ini satu-satunya irisan data yang
  punya penanda pesanan terkonfirmasi; untuk 67 SKU lainnya batasan di atas tetap
  berlaku utuh.
- Diuji pada 2026-08-15: lonjakan **tidak acak**. Peluang dasar suatu hari menjadi
  lonjakan adalah 0,90%; bila hari sebelumnya lonjakan, peluangnya naik menjadi
  12,24%; bila 7 hari sebelumnya lonjakan, 11,77%. Artinya kecenderungan sebuah
  pair menerima pesanan besar bersifat persisten, meski pesanan individualnya tidak
  dapat diprediksi.

**Implikasi untuk perumusan.** Metrik sebaiknya dilaporkan tiga kali berdampingan —
terhadap target mentah (akuntabilitas penuh), terhadap target di-cap (lingkup
tanggung jawab model), dan pada baris non-lonjakan saja (lingkup paling murni) —
dengan pilihan mana yang jadi angka utama dinyatakan eksplisit di awal.

## B-4. Sepertiga pasangan item-cabang tidak pernah dievaluasi

Dari 2.979 pasangan (item, cabang) di periode latih, hanya **1.920** yang punya
baris di Desember 2025. Sebanyak **1.059 pasangan (35,6%)** berhenti muncul sebelum
periode uji sehingga tidak pernah masuk penilaian. Satu cabang penuh, Kebuli Yaman
Cikarang Pusat, tidak punya baris uji karena masih tutup sejak 2025-12-01 (29.007
baris latih, nol baris uji).

Klaim hasil penelitian berlaku untuk **1.920 pasangan aktif**, bukan seluruh katalog.

## B-5. Periode uji efektif lebih pendek dari satu bulan penuh

`target_lead_time_cumulative` bernilai null pada 7,9% baris uji, terkonsentrasi di
ujung bulan: 100% pada 30–31 Desember, 27,7% pada 29 Desember, dan 10–15% pada
26–28 Desember. Ini perilaku yang benar — permintaan Januari 2026 belum ada di data
— tetapi berarti periode evaluasi efektif adalah **1–29 Desember 2025** dengan
cakupan yang menipis di ujungnya.

## B-6. Hanya 29% baris uji merupakan momen keputusan sungguhan

Pusat hanya mengirim dua kali seminggu (kawasan 1: Senin & Kamis; kawasan 2: Selasa
& Jumat). Dari 55.046 baris uji, hanya **16.031 (29,1%)** jatuh pada hari kirim, dan
14.117 di antaranya bertarget non-null. Pada hari kirim `lead_time_days` hanya
pernah bernilai 3 atau 4; baris dengan lead time 1–2 hari (58% data uji) tidak pernah
menjadi momen pusat memutuskan mengirim berapa.

Melatih pada semua hari tetap sah, tetapi metrik utama sebaiknya dilaporkan juga
khusus untuk baris hari-kirim.

## B-7. Empat tanggal relokasi hanya batas bawah

Lima cabang (`Mayor Oking`, `Teluk Pucung`, `Bukit Gading Balaraja`,
`Grand Wisata Bekasi`, dan sebagian `Cikarang Pusat`) direlokasi setelah cakupan
data berakhir, sehingga tanggal di `RELOCATION_DATES` merupakan proksi batas bawah,
bukan tanggal sebenarnya. Untuk cabang-cabang ini `days_since_relocation` selalu
negatif dan tidak membawa informasi selain "cabang ini akan pindah nanti".

Hanya empat relokasi yang tanggalnya persis dan teramati di dalam data — Tigaraksa,
Cadas, Citayam, Bintara — dan hanya tiga di antaranya punya data pasca-relokasi
yang cukup untuk dinilai.

## B-8. ~~Kawasan lokasi lama tidak tercatat~~ — ditutup (2026-08-17)

**Sebelumnya:** `dataset/outlet_mapping.csv` hanya menyimpan `kawasan` dan
`hari_pengiriman` lokasi **saat ini**, sedangkan untuk cabang yang direlokasi kolom
`old_name` diisi sama dengan `new_name` — sehingga jadwal kirim lokasi lama dianggap
tidak dapat direkonstruksi. 205.513 baris pra-relokasi (13,7% dataset) memakai jadwal
lokasi baru untuk menghitung `lead_time_days`, dan itu dicatat sebagai asumsi.

**Sekarang:** pemilik data mengonfirmasi (2026-08-17) bahwa `outlet_mapping.csv`
memang **sudah menjadi arsip jadwal kirimnya**, dan pencocokan lewat nama outlet
**baru** adalah cara yang berlaku — bukan penambalan. Verifikasi: ke-9 cabang yang
pernah relokasi punya `kawasan` dan `hari_pengiriman` terisi di bawah nama barunya,
tanpa satu pun nilai kosong di seluruh file. Jadi 205.513 baris itu berhenti menjadi
asumsi yang dinyatakan.

Satu pemeriksaan silang yang tersedia mendukungnya: `KY029 - Kebuli Yaman Cinere`
(lokasi lama Bintara) masih punya baris sendiri di `outlet_mapping.csv`, dengan
`kawasan = 2` / `Selasa dan Jumat` — sama persis dengan Bintara. Ini satu-satunya
lokasi lama yang bisa dibandingkan langsung; delapan sisanya bersandar pada
konfirmasi pemilik data, bukan pada pembandingan baris.

## B-9. Beberapa keputusan pembersihan belum di-sign-off

- `dataset/event_driven_items.csv` adalah draf yang diturunkan dari bentuk
  permintaan. **3 dari 14 SKU prioritas-1 dikonfirmasi pemilik data
  (2026-08-16)**: `FGS-00018`/`FGS-00034` (Kambing Kebuli Aqiqah Betina/Jantan)
  dan `PCG-00002` (Lunch Box Aqiqah) memang berjalan lewat pesanan — draf sudah
  `true` untuk ketiganya, jadi tidak ada perubahan data. **11 SKU masih terbuka**:
  9 barang kelompok Loyang (`PCG-00003`–`00008`, `PCG-00011`–`00013`, draf
  `false` melawan intuisi nama karena datanya rutin harian) dan 2 barang berpola
  borongan yang namanya tidak menyebut acara (`PCG-00027` Mika Bento,
  `PCG-00028` Cup 60 ml, draf `true`). **11 SKU sisanya diputuskan dari bukti
  data (2026-08-16), bukan dari putaran pertanyaan kedua** — memakai
  ko-okurensi dengan SKU aqiqah yang sudah dikonfirmasi sebagai acuan (baseline:
  0,84% branch-day aktif). Cup 60 ml ko-okur di **100%** hari geraknya, Mika
  Bento membawa **93% volumenya** di hari aqiqah, sedangkan 9 SKU Loyang hanya
  0,9%–1,4% — tidak ada kaitan. Tidak ada flag yang berubah; draf sudah benar
  untuk ke-70 SKU. Batas bukti ini perlu dinyatakan di laporan: yang bisa
  dibuktikan adalah "bukan barang acara/aqiqah", **bukan** "tidak pernah
  dipesan lebih dulu" — tanggal pesan tidak tercatat di mana pun (B-1/B-2),
  jadi untuk kelompok Loyang kemungkinan sebagian dipesan sehari sebelumnya
  tetap tidak terobservasi.
- Tingkat layanan target **dikonfirmasi pemilik data (2026-08-16): kuantil 0,9,
  seragam untuk semua item**. Tidak ada pembedaan per kategori — pengiriman dari
  pusat mencakup semua item dalam satu kali kirim, sehingga satu tingkat layanan
  berlaku untuk seluruh pengiriman.

Sudah tertutup: 8 nilai `Kota Override` di `dataset/outlet_name_overrides.csv`
dikonfirmasi pemilik data (2026-08-16), termasuk `KY001` Kutabumi sebagai
`Kabupaten Tangerang` meski kolom `Kecamatan` di `outlets.csv` menyebut
Jatiuwung. `kawasan = 2` / `hari_pengiriman = Selasa dan Jumat` untuk Bintara,
Citayam, dan Grand Wisata Bekasi juga dikonfirmasi pada tanggal yang sama —
nilai yang sebelumnya diinferensi dari cabang Kota Bekasi/Kota Depok lain
ternyata benar, sehingga 82.068 baris (5,5% dataset) tidak lagi bergantung pada
asumsi. Jadwal kirim lokasi **lama** menyusul ditutup sehari kemudian — lihat B-8,
yang sudah tidak lagi berlaku sejak 2026-08-17.

Ditutup juga: jeda 13 hari `KY068 - Kebuli Yaman Kramatwatu` (28 Juni – 10 Juli
2025) **dikonfirmasi tutup sementara (2026-08-17)**, bukan celah pencatatan —
data mentah `jan-des-25.csv` juga kosong pada rentang itu (transaksi terakhir
2025-06-27, kembali 2025-07-11). Sudah dicatat di `dataset/outlet_closures.csv`,
sehingga ke-13 hari itu tidak lagi ikut sebagai permintaan nol.

---

## Ringkasan untuk bab batasan

Tiga batasan terpenting, berurutan:

1. **Variabel paling prediktif dalam bisnis ini tidak terekam** (B-1, B-2). Buku
   pesanan menentukan sebagian besar permintaan beberapa hari ke depan, dan tidak
   ada di data. Ini membatasi akurasi maksimum ketiga model sekaligus, dan tidak
   dapat diperbaiki tanpa mengubah sistem POS.
2. **Lingkup model dan isi target tidak sepenuhnya berimpit** (B-3). Model
   diperlukan untuk permintaan di luar pesanan, tetapi target berisi keduanya
   bercampur tanpa penanda pemisah.
3. **Evaluasi berlaku untuk irisan yang lebih sempit dari yang terlihat** (B-4,
   B-5, B-6): 1.920 dari 2.979 pasangan, 1–29 Desember 2025, dan hanya 29% baris
   yang merupakan momen keputusan sungguhan.

Ketiganya adalah batasan perumusan masalah dan ketersediaan data. Tidak satu pun
disebabkan oleh pilihan arsitektur model, dan tidak satu pun dapat diselesaikan
dengan menyetel ulang hyperparameter.
