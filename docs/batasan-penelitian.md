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

  **Klarifikasi lanjutan (2026-08-22).** Konfirmasi ulang pemilik data
  menjelaskan bahwa "seragam untuk semua item" dimaksudkan sebagai **komitmen
  agregat di level pengiriman**, bukan larangan variasi teknis per item —
  karena setiap item punya tren pasar berbeda, bahkan item yang sama bisa
  punya tren berbeda antar cabang. Pengiriman dari pusat tetap mencakup semua
  item dalam satu kali kirim dengan satu janji tingkat layanan 0,9 ke outlet;
  yang boleh bervariasi adalah kuantil input per segmen (kategori barang ×
  `demand_segment`) yang dipakai mencapai janji itu, selama rata-rata
  tertimbangnya kembali ke 0,9 secara agregat.

  **Konsekuensi.** Model produksi yang dibekukan lewat protokol §19 tetap
  dilatih dan dipilih pada kuantil 0,9 seragam — klarifikasi ini tidak
  mengubah proses pemilihan model. Perluasan ke alokasi kuantil tersegmentasi
  adalah pekerjaan lanjutan terpisah, dijalankan setelah pemenang ditetapkan,
  bukan bagian dari perbandingan tiga arsitektur. Lihat
  `docs/superpowers/specs/2026-08-22-segmented-quantile-allocation-design.md`.

  **Koreksi atas paragraf "Konsekuensi" di atas (2026-08-24).** Kalimat
  "klarifikasi ini tidak mengubah proses pemilihan model" **sudah tidak berlaku**.
  Proses pemilihan model memang kemudian berubah, lewat keputusan terpisah yang
  dirancang di
  `docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`:
  kriteria utama K1 tidak lagi pinball loss pada satu titik kuantil (0,9),
  melainkan **rata-rata pinball loss lintas 19 titik kuantil** (0,05–0,95), dan
  K2 memeriksa coverage per kuantil, bukan hanya di 0,9. Ketiga arsitektur
  diperluas ke multi-kuantil **sebelum** pemenang ditetapkan, bukan sesudahnya.

  Teks 2026-08-16 dan klarifikasi 2026-08-22 di atas **tidak dicabut** dan tidak
  perlu dicabut — keduanya berbicara soal kuantil 0,9 sebagai **komitmen bisnis**
  yang mengatur apa yang benar-benar dikirim ke outlet, dan komitmen itu tetap
  utuh apa adanya. Yang keliru hanya prediksi di paragraf "Konsekuensi" tentang
  bagaimana pemilihan model akan dijalankan. Pembedaan itu ditulis di sini supaya
  pembaca tidak menyimpulkan bahwa janji 0,9 ikut berubah: **komitmen bisnis 0,9
  tetap; kriteria perbandingan model yang berpindah ke multi-kuantil.**

  Konsekuensi lanjutan yang layak dicatat: karena perluasan multi-kuantil kini
  selesai untuk ketiga arsitektur sebelum pemenang ditetapkan, alokasi kuantil
  tersegmentasi tidak lagi perlu menunggu model pemenang diperluas lebih dulu —
  jalurnya menjadi lebih pendek, bukan lebih panjang.

Sudah tertutup: 8 nilai `Kota Override` di `dataset/outlet_name_overrides.csv`
dikonfirmasi pemilik data (2026-08-16), termasuk `KY001` Kutabumi sebagai
`Kabupaten Tangerang` meski kolom `Kecamatan` di `outlets.csv` menyebut
Jatiuwung. `kawasan = 2` / `hari_pengiriman = Selasa dan Jumat` untuk Bintara,
Citayam, dan Grand Wisata Bekasi juga dikonfirmasi pada tanggal yang sama —
nilai yang sebelumnya diinferensi dari cabang Kota Bekasi/Kota Depok lain
ternyata benar, sehingga 82.068 baris (5,5% dataset) tidak lagi bergantung pada
asumsi. Jadwal kirim lokasi **lama** menyusul ditutup sehari kemudian — lihat B-8,
yang sudah tidak lagi berlaku sejak 2026-08-17.

Ditutup juga: `FGS.00048` (Kambing Oven) **dikonfirmasi masih dijual dan memang
slow mover (2026-08-17)** — buku menu memuatnya sebagai paket tersendiri
("Kambing Muda Rempah Oven", satu ekor + 50 porsi nasi kebuli) di luar Paket
Aqiqah. Batal menjadi kandidat `EXCLUDED_ITEMS`, `is_event_driven` tetap
`false`, jadi tidak ada flag yang berubah. Perlu dicatat cakupan konfirmasinya:
yang dipastikan adalah **status jual dan sifat lambat lakunya**, bukan bahwa
paket seharga jutaan rupiah ini dibeli tanpa pesan lebih dulu — seperti seluruh
dataset, tanggal pesan tidak terekam (B-1/B-2), sehingga sebagian dari 10 ekor
itu mungkin tetap merupakan pesanan yang secara lingkup masuk B-3.

Ditutup juga: jeda 13 hari `KY068 - Kebuli Yaman Kramatwatu` (28 Juni – 10 Juli
2025) **dikonfirmasi tutup sementara (2026-08-17)**, bukan celah pencatatan —
data mentah `jan-des-25.csv` juga kosong pada rentang itu (transaksi terakhir
2025-06-27, kembali 2025-07-11). Sudah dicatat di `dataset/outlet_closures.csv`,
sehingga ke-13 hari itu tidak lagi ikut sebagai permintaan nol.

Ditutup juga: kategori `Barang Semi FG (WIP-2)` **dikonfirmasi sebagai label
administratif lama (2026-08-22)** untuk sepuluh SKU — `FGS-00001`, `FGS-00002`,
`FGS-00003`, `FGS-00004`, `FGS-00005`, `FGS-00012`, `FGS-00013`, `FGS-00018`,
`FGS-00049`, dan `FGS-00053`. Cara penanganan barangnya tidak pernah berubah;
hanya penamaan kategorinya yang diperbarui, sehingga seluruh riwayat kesepuluh
SKU itu harus dibaca sebagai `Barang Jadi (FG)`.

**Konfirmasi ini menggantikan konfirmasi 2026-08-10 untuk kesepuluh SKU
tersebut, dan hanya untuk mereka.** Konfirmasi 2026-08-10 menetapkan bahwa
WIP-2 dan `Barang Jadi (FG)` adalah kategori yang benar-benar berbeda sehingga
tidak boleh disatukan otomatis seperti pasangan sinonim `Minuman`/`Minuman - FG`.
Aturan umum itu **tetap berlaku** dan tetap dikunci di
`canonicalize_item_categories()`; yang berubah hanyalah bahwa kesepuluh SKU di
atas kini menjadi pengecualian bernama. `FGS-00014` (Club Mineral 600ml) **bukan**
bagian dari kelompok ini — konfirmasi 2026-08-10 untuknya tetap utuh, dan
kategorinya `Minuman - FG`, bukan `Barang Jadi (FG)`.

Reklasifikasi diterapkan di lapisan normalisasi lewat
`normalize_items.EXPLICIT_CATEGORY_OVERRIDES`, bukan dengan mengekspor ulang
data sumber. Konsekuensinya `dataset/dataset.csv` tetap memuat 14.828 baris
berlabel WIP-2 selamanya, dan itu normal — bukan tanda reklasifikasi gagal.
Gerbang `utils/verify_category_consistency.py` memeriksa keduanya secara
terpisah: lapis sumber bersifat informasional, lapis ternormalisasi yang
menjadi syarat lulus (nol SKU dengan kategori bervariasi).
Perlu dicatat bahwa 14.828 itu mencakup **11 SKU**, bukan sepuluh: 2.192 baris
di antaranya milik `FGS-00014`, yang memang tidak ikut pindah ke
`Barang Jadi (FG)`. Jadi angka lapis sumber ini tidak sebanding langsung
dengan angka lapis panel di bawah — lihat rantai rekonsiliasinya.

**Dampak terukur pada artefak (diukur 2026-08-22).** `model_input.parquet` kini
memuat **7 kategori**, turun dari 8; 19.987 baris berpindah dari WIP-2 ke
`Barang Jadi (FG)` (265.181 → 285.168). Jumlah baris **tidak berubah**
(1.502.522), begitu pula 70 SKU, 59 cabang, dan 2.979 pasangan item-cabang —
relabeling hanya menulis ulang satu kolom. Jumlah SKU dengan kategori
bervariasi turun dari 10 menjadi **0**. Sesuai kebijakan stabilitas indeks
(§4.12(e) `metodologi-preprocessing.md`), indeks 4 milik WIP-2 **dipertahankan
sebagai indeks yatim** di `category_mapping.json` dan tidak ada kategori lain
yang dinomori ulang, sehingga `Kategori Barang_idx` sekarang memakai
{1, 2, 3, 5, 6, 7, 8}.

**Rekonsiliasi 14.828 (lapis sumber) dengan 19.987 (lapis panel).** Kedua angka
mengukur artefak yang berbeda pada satuan baris yang berbeda: `dataset.csv`
berisi satu baris per line item transaksi, sedangkan `model_input.parquet`
berisi satu baris per `(Kode Barang, Nama Cabang, Tanggal)` termasuk hari
nol-permintaan. Rantai penuhnya, diukur ulang 2026-08-23 dengan menjalankan
kembali pipeline memakai `EXPLICIT_CATEGORY_OVERRIDES` versi lama (hanya
`FGS-00014`), yang mereproduksi baseline 265.181/19.987 persis:

| Langkah | Baris | Keterangan |
|---|---:|---|
| WIP-2 di `dataset.csv` | 14.828 | 11 SKU |
| − `FGS-00014` | −2.192 | di-override ke `Minuman - FG` sejak konfirmasi 2026-08-10, tidak pernah menjadi FG |
| = hari-transaksi WIP-2, 10 SKU | 12.636 | agregasi harian tidak menyusutkan apa pun — line item-nya sudah unik per SKU/hari/cabang |
| − gugur di `filter_min_history` | −24 | pasangan dengan <60 hari riwayat pra-cutoff |
| = baris panel WIP-2, `Kuantitas > 0` | 12.612 | |
| + hari nol-permintaan hasil `ffill()` | +7.375 | `build_panel.py`, pengisian kolom `CARRY_COLS` pada panel padat |
| = **baris yang berpindah ke FG** | **19.987** | |

Sumber pertambahan 7.375 baris itu: di data sumber pergantian label terjadi
serentak pada **2024-03-01** (WIP-2 hanya ada di `jan-24.csv` dan `feb-24.csv`),
tetapi `build_dense_panel()` mem-`ffill` kategori per pasangan item-cabang.
Pasangan yang transaksi terakhirnya Januari–Februari 2024 lalu lama menganggur
menyeret label lama itu jauh ke depan — `FGS-00018` paling ekstrem: 20
hari-transaksi menjadi 2.279 baris panel (2.259 di antaranya nol, label
terbawa sampai 2025-03-12), dan `FGS-00013` terbawa sampai 2025-07-20.
Sebaliknya `FGS-00001` hanya menambah 5 baris karena hampir setiap harinya
ada transaksi.

Angka 19.987 juga jauh lebih kecil daripada 244.745 — total baris panel
kesepuluh SKU tersebut — karena mayoritas riwayat panel mereka memang sudah
berlabel `Barang Jadi (FG)` sejak 2024-03-01; yang berpindah hanya jendela
Januari–Februari 2024 beserta ekor hari nol yang mewarisinya.

**Verifikasi akhir pada artefak hasil refresh (diukur 2026-08-23).** Dijalankan
langsung atas `dataset/model_ready/model_input.parquet` (1.502.522 baris, 82
kolom) setelah pipeline dijalankan ulang, dua angka penutup rantai di atas
terkonfirmasi harfiah:

1. Jumlah SKU yang punya lebih dari satu nilai `Kategori Barang` sepanjang
   riwayatnya: **0** — sesuai syarat lulus gerbang
   `utils/verify_category_consistency.py` pada lapis ternormalisasi.
2. Nilai unik `Kategori Barang`: **7 nama**, tanpa WIP-2 — `Bahan Baku (RM)`,
   `Barang Dalam Process (WIP-1)`, `Barang Jadi (FG)`, `Barang Umum`,
   `Minuman - FG`, `Packaging`, `Snack (FG)`.

Dua catatan penamaan yang bukan anomali refresh, dan yang penyebabnya
**berbeda** meski sering disebut bersamaan:

- `Snack` muncul sebagai `Snack (FG)` — ini memang hasil kanonikalisasi di
  `normalize_items.CATEGORY_SYNONYMS`.
- `Tambahan` (dan `Perlengkapan Resto`) tidak muncul sebagai kategori
  tersendiri di `model_input.parquet` — ini **bukan** kanonikalisasi.
  Keduanya masih utuh setelah `load_and_normalize()`, dan itulah sebabnya
  `verify_category_consistency` melaporkan 9 kategori sementara
  `model_input.parquet` hanya memuat 7. Keduanya gugur lebih jauh di hilir
  lewat `build_panel.filter_min_history` (`MIN_HISTORY_DAYS = 60`), karena
  seluruh SKU-nya berumur pendek: `Tambahan` hanya `FGS-00072` dan
  `FGS-00073` (82 baris, 2025-01-24 s.d. 2025-03-16), `Perlengkapan Resto`
  hanya `SPR-00004` (6 baris, 2024-01-21 s.d. 2024-02-20).

Selisih 9 lawan 7 itu karena itu bukan tanda gerbang dan pipeline tidak
sepakat — keduanya memang mengukur lapisan yang berbeda. Tidak satu pun dari
keduanya efek reklasifikasi WIP-2.

## B-10. Data biaya/margin per item tidak tersedia pada granularitas dataset saat ini — batasan sementara, menunggu pemetaan

**Status: TERBUKA** sejak 2026-08-22. **Fleksibel** — batasan ini dirancang
untuk hilang begitu data tersedia, tanpa perubahan kode.

**Fakta.** Harga yang tersedia berbentuk harga menu utuh, bukan per komponen
item seperti granularitas `(Kode Barang, Nama Cabang)` yang dipakai seluruh
pipeline ini. Tidak ada satu pun berkas sumber yang memuat biaya produksi
atau margin per item pada tanggal dokumen ini ditulis.

**Konsekuensi saat ini.** Perhitungan critical ratio (Cu/Co) untuk alokasi
kuantil per segmen
(`docs/superpowers/specs/2026-08-22-segmented-quantile-allocation-design.md`)
tidak presisi untuk SKU yang belum punya entri biaya. SKU tersebut jatuh ke
jalur proksi (peringkat ordinal masa simpan per kategori dan elisitasi
kualitatif tim SCM, lihat `dataset/shelf_life_rank_by_category.csv`), yang
secara metodologis lebih lemah daripada perhitungan biaya langsung.

**Mekanisme penanganan.** `dataset/item_cost_margin.csv` dibaca per baris;
SKU dengan entri `cost_confidence` bukan `rendah` otomatis memakai jalur
presisi, sisanya memakai proksi. Keputusan diambil **per SKU**, bukan per
keseluruhan berkas — pengisian bertahap tidak memerlukan perubahan kode.

**Tabel pelacakan cakupan** (diperbarui setiap kali berkas
`item_cost_margin.csv` berubah signifikan):

| Tanggal cek | SKU dengan entri presisi | % dari total SKU (70) | % dari total volume | Sumber pengisian |
|---|---|---|---|---|
| 2026-08-22 | 0 | 0% | 0% | — (belum dimulai) |

**Kriteria penutupan butir ini.** Ditutup (dipindah ke bagian "Sudah
tertutup") ketika SKU dengan entri presisi mencakup **≥80% dari total
volume** — sejalan dengan ambang yang diusulkan di spec segmentasi kuantil
untuk beralih dari days of supply ke rupiah sebagai metrik utama. Sampai
ambang ini tercapai, butir tetap berstatus terbuka meskipun sebagian data
sudah masuk.

## B-11. Trade off overstock/understock dilaporkan dalam days of supply, bukan rupiah, sampai B-10 mencapai ambang cakupan

**Status: mengikuti status B-10.** Butir ini otomatis berubah begitu B-10
ditutup — lihat kriteria penutupan di B-10 untuk ambang yang sama.

**Fakta.** Satuan fisik pada dataset ini campur (Kg, Porsi, Botol, PCS)
sebagaimana sudah dicatat di seluruh dokumen hasil model
(`docs/hasil-modeling-{rf,xgb,lstm}.md` §1). Tanpa data biaya (B-10), tidak
ada cara mengonversi unit-unit tersebut ke satuan yang sebanding secara
ekonomi.

**Konsekuensi saat ini.** Metrik evaluasi utama untuk simulasi alokasi
kuantil tersegmentasi memakai days of supply (unit overstock/shortfall
dibagi rata-rata demand harian pasangan, `roll_mean_28`) sebagai proksi
sementara, bukan nilai rupiah. Ini memberi perbandingan yang adil secara
proporsional antar kategori, tapi tidak menjawab pertanyaan "berapa rupiah
yang dihemat", yang baru bisa dijawab setelah B-10 terselesaikan sebagian
besar.

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

B-10 dan B-11 adalah batasan yang berbeda sifatnya dari kesembilan butir di
atas — keduanya bukan keterbatasan data historis yang permanen, melainkan
kekosongan sementara pada pekerjaan lanjutan (alokasi kuantil tersegmentasi)
yang secara eksplisit dirancang untuk hilang begitu data biaya/margin per
item tersedia, tanpa mengubah perumusan masalah inti proyek ini.
