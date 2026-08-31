# Detail Tahap Exploratory Data Analysis (EDA)

Dokumen ini adalah elaborasi mendetail dari `notebook/eda.ipynb`, pass EDA yang berdiri
sendiri atas `dataset/csv/dataset.csv` (keluaran tahap merge dan agregasi — lihat
`docs/detail-tahap-merge-agregasi-data.md`). Sebagaimana dokumen itu, dokumen ini ditulis
dengan gaya yang sama: dua bagian, Akademis lalu Teknis, dengan detail data, kode, dan angka
yang diverifikasi langsung dari notebook dan dari kode `utils/data_preprocessing/` terkait.

Dokumen ini **menggantikan** `docs/temuan-eda.md` — seluruh narasi, konteks bisnis, dan
checklist pra-modeling yang sebelumnya ada di sana sudah dipindah dan diperluas ke Bagian 1
di bawah (lihat khususnya bagian 1.1 dan bagian 1.14). `docs/temuan-eda.md` sendiri dihapus supaya tidak
ada dua sumber kebenaran yang bisa saling bergeser.

Seluruh angka pada dokumen ini berasal dari hasil eksekusi `notebook/eda.ipynb` per
2026-08-26 (commit terakhir yang mengubah notebook ini), dijalankan atas
`dataset/csv/dataset.csv` yang sama dengan yang ada di disk saat dokumen ini ditulis
(2026-08-28, tidak berubah sejak 2026-08-25) — bukan disalin dari dokumentasi lama. Beberapa
angka yang tumpang tindih dengan hasil `docs/detail-tahap-merge-agregasi-data.md` (mis. 693.563
baris) dijadikan pemeriksaan silang antar dua dokumen, bukan diasumsikan sama.

---

# Bagian 1 — Akademis (Bahan Laporan)

## 1.1 Pendahuluan dan Konteks Bisnis

Tujuan proyek ini adalah peramalan permintaan (_demand forecasting_) untuk tiap kombinasi
item-cabang, dengan horizon ≤4 hari. Tim SCM pusat saat ini mengirim ke outlet
**Region 1** setiap **Senin & Kamis**, dan ke outlet **Region 2** setiap **Selasa & Jumat** —
sehingga model yang dibangun perlu memprediksi permintaan _kumulatif_ selama jendela lead
time sampai pengiriman berikutnya (jendela 3 atau 4 hari, tergantung region dan hari kirim).

**Keterbatasan penting yang berlaku di seluruh EDA ini:** pada saat notebook ini dijalankan,
tidak ada pemetaan Region 1 / Region 2 (atau jadwal pengiriman apa pun) di data manapun —
`dataset/outlets.csv` hanya memuat `Kecamatan`/`Kota` (dikonfirmasi ulang di bagian 1.13). Analisis
hari-dalam-minggu (bagian 1.9) dan jendela lead time (bagian 1.12) di bawah ini karena itu sengaja bersifat
generik/lintas-cabang, bukan tersegmentasi per region. Kesenjangan ini dibahas eksplisit di
bagian 1.13 dan menjadi salah satu dari tiga penghambat prioritas tertinggi sebelum pemodelan
(bagian 1.14).

Pass EDA ini murni deskriptif: ia tidak mengubah, mem-_filter_, atau menuliskan ulang
`dataset/csv/dataset.csv` — seluruhnya baca-saja, dan berdiri terpisah dari
`notebook/data_processing.ipynb` (lihat bagian 1.2).

## 1.2 Posisi Notebook dalam Pipeline dan Independensinya

`notebook/eda.ipynb` adalah pass EDA yang **sepenuhnya independen** atas tabel transaksi
bersih dan teragregasi yang dihasilkan `notebook/merge_and_aggregate.ipynb`
(`merge_dataset.py` + `aggregate_dataset.py` → `dataset/csv/dataset.csv`). Ia tidak
mengubah, menggantikan, atau bergantung pada `notebook/data_processing.ipynb` — beberapa
pemeriksaan dasar (nilai hilang, jumlah unik, dsb.) sengaja tumpang tindih antara kedua
notebook, dan itu diterima sebagai desain, bukan duplikasi yang perlu dihapus.

Sejak refactor notebook 2026-08-25/26 (lihat "Notebook convention" di `CLAUDE.md`), sel
markdown notebook ini dibatasi pada judul bagian saja — narasi tingkat-dokumen yang dulunya
hidup di sel markdown sekarang seluruhnya berada di sini. Notebook berisi 15 bagian bernomor
(judul persis seperti sel markdown-nya), yang menjadi kerangka bagian 1.3–1.13 di bawah:

| #   | Judul di notebook                                | Dibahas di          |
| --- | ------------------------------------------------ | ------------------- |
| 1   | Konfigurasi                                      | bagian 2.1 (Teknis) |
| 2   | Fungsi & konstanta pipeline yang dipakai di sini | bagian 2.2 (Teknis) |
| 3   | Muat dataset                                     | bagian 1.3          |
| 4   | Data health check                                | bagian 1.3          |
| 5   | Konsistensi Satuan & Kategori Barang per SKU     | bagian 1.4          |
| 6   | Struktur cabang & SKU teratas                    | bagian 1.5          |
| 7   | Distribusi Kuantitas & outlier                   | bagian 1.6          |
| 8   | Deteksi spike relatif terhadap baseline          | bagian 1.7          |
| 9   | Cakupan & kontinuitas waktu                      | bagian 1.8          |
| 10  | Deret waktu & musiman                            | bagian 1.9          |
| 11  | Konsistensi pola mingguan 2024 vs 2025           | bagian 1.10         |
| 12  | Struktur item x outlet                           | bagian 1.11         |
| 13  | Analisis proxy lead time                         | bagian 1.12         |
| 14  | Open questions — cek data                        | bagian 1.13         |
| 15  | _(Opsional)_ Cek sinkron dengan `utils/`         | bagian 2.3 (Teknis) |

## 1.3 Muat Dataset dan Pemeriksaan Kesehatan Data

**Tujuan.** Memastikan `dataset/csv/dataset.csv` sungguh bersih dan siap dianalisis lebih
lanjut sebelum menafsirkan pola apa pun di dalamnya — bukan mengasumsikan tahap merge/agregasi
sudah benar begitu saja.

**Prosedur.** Berkas dimuat dengan tipe eksplisit per kolom (`Kategori Barang`, `Nama Cabang`,
`Satuan` sebagai `category`; `Kuantitas` sebagai `float64`; `Tanggal` diuraikan dengan format
`%d %b %Y`), lalu diperiksa: bentuk tabel, nilai hilang per kolom, baris duplikat (baik
duplikat persis maupun duplikat menurut kunci bisnis enam-kolom), dan kardinalitas tiap kolom
kategorikal.

**Temuan.**

| Pemeriksaan                                                                    | Hasil                   |
| ------------------------------------------------------------------------------ | ----------------------- |
| Bentuk tabel                                                                   | 693.563 baris x 7 kolom |
| Nilai hilang                                                                   | 0 di seluruh kolom      |
| Baris duplikat persis                                                          | 0                       |
| Baris duplikat menurut kunci bisnis (Tanggal+Kategori+Kode+Nama+Cabang+Satuan) | 0                       |
| `Kategori Barang` unik                                                         | 12                      |
| `Kode Barang` unik                                                             | 109                     |
| `Nama Barang` unik                                                             | 112                     |
| `Nama Cabang` unik                                                             | 67                      |
| `Satuan` unik                                                                  | 12                      |

**Justifikasi.** Angka 693.563 baris ini bersesuaian persis dengan keluaran tahap agregasi
pada `docs/detail-tahap-merge-agregasi-data.md` bagian 1.4 — pemeriksaan silang yang mengonfirmasi
tidak ada baris yang hilang atau berubah antara tahap agregasi dan pemuatan di EDA ini. Nol
duplikat menurut kunci bisnis mengonfirmasi jaminan ketunggalan baris yang dibangun tahap
agregasi (lihat dokumen tersebut bagian 1.4) masih berlaku pada berkas final di disk. `Nama Barang`
(112) sedikit lebih besar dari `Kode Barang` (109) karena sebagian kode memiliki lebih dari
satu varian nama tercatat sepanjang waktu (lihat bagian 1.4 di bawah untuk kategori, pola serupa).

## 1.4 Konsistensi Satuan dan Kategori Barang per SKU

**Tujuan.** Tahap agregasi hilir (`utils/data_preprocessing/normalize_items.py` dan
seterusnya) mengasumsikan setiap `Kode Barang` punya satu `Satuan` tetap. Asumsi ini belum
pernah diverifikasi eksplisit sebelum EDA ini — jika dilanggar, penjumlahan `Kuantitas` lintas
baris untuk satu SKU akan mencampur satuan yang berbeda tanpa terdeteksi.

**Prosedur.** Untuk tiap `Kode Barang`, dihitung jumlah nilai `Satuan` unik dan jumlah nilai
`Kategori Barang` unik yang pernah tercatat.

**Temuan.**

- **Satuan: 0 dari 109 SKU** punya lebih dari satu `Satuan` — asumsi hilir aman, tidak
  ditemukan pelanggaran.
- **Kategori: 27 dari 109 SKU (24,8%)** punya lebih dari satu `Kategori Barang` tercatat
  sepanjang waktu — mis. `Minuman` → `Minuman - FG`, `Barang Semi FG (WIP-2)` →
  `Barang Jadi (FG)`, `Snack` → `Snack (FG)`. Pola pasangannya konsisten (satu label lama
  ke satu label baru per SKU), lebih menyerupai satu peristiwa _rename_ taksonomi kategori di
  pertengahan 2024 daripada derau input data acak.

**Status penyelesaian (diverifikasi terhadap kode saat ini).** Temuan kategori ini sudah
ditindaklanjuti pemilik data dan diimplementasikan di
`utils/data_preprocessing/normalize_items.py`: `CATEGORY_SYNONYMS` menyatukan varian penamaan
yang identik secara bisnis (`Minuman`→`Minuman - FG`, `Snack`→`Snack (FG)`), dan
`EXPLICIT_CATEGORY_OVERRIDES` menulis ulang kategori sepuluh SKU yang dikonfirmasi pemilik
data pada 2026-08-22 sebagai relabel administratif WIP-2→Barang Jadi (FG), plus
FGS-00014 (Club Mineral 600ml) yang dikonfirmasi 2026-08-10 sebagai minuman, bukan WIP-2.
`utils/eda/verify_category_consistency.py` berfungsi sebagai _refresh gate_: dijalankan
setelah setiap refresh data untuk memastikan tidak ada SKU yang masih punya kategori
bervariasi pasca-normalisasi. Baris terkait pada checklist bagian 1.14 (dulu baris 10 di
`docs/temuan-eda.md`) karena itu dianggap **selesai**, bukan lagi _open question_.

## 1.5 Struktur Cabang dan SKU Teratas

**Tujuan.** Memahami sebaran volume transaksi antar 67 cabang dan mengidentifikasi SKU
berdampak terbesar, baik dari sisi frekuensi transaksi maupun dari sisi volume kumulatif.

**Prosedur.** Jumlah baris per `Nama Cabang` divisualisasikan sebagai bar chart terurut.
20 SKU teratas dihitung dua cara: berdasarkan jumlah baris transaksi (`value_counts`), dan
berdasarkan total `Kuantitas` terjumlah (`groupby(...).sum()`) — dua pengurutan yang bisa
berbeda karena satu SKU bisa sering ditransaksikan dalam jumlah kecil, atau jarang tapi dalam
jumlah besar.

**Temuan.** Kedua pengurutan didominasi SKU yang sama — `Nasi Kebuli`, `Sambal - FG`,
`Cup Sambal Take Away`, `Ayam Kebuli (0.9)` — konsisten dengan posisinya sebagai menu inti.
`Nasi Kebuli` dan `Sambal - FG` memimpin dari sisi volume total (masing-masing >4,1 juta
unit kumulatif), sementara urutan berdasarkan jumlah baris hampir identik untuk keempat SKU
teratas (36.598–36.730 baris masing-masing), mengindikasikan keduanya memang dipesan hampir
setiap hari operasional di hampir semua cabang.

## 1.6 Distribusi Kuantitas dan Deteksi Outlier

**Tujuan.** Memahami bentuk distribusi `Kuantitas` (untuk memutuskan transformasi/parameterisasi
model yang sesuai) dan mengidentifikasi baris dengan nilai ekstrem sebagai kandidat pemeriksaan
kesalahan input.

**Prosedur.** Statistik deskriptif dasar, histogram pada skala linear dan `log1p`, boxplot
per `Kategori Barang` (skala log, outlier ekstrem disembunyikan lewat `showfliers=False` supaya
kotak/whisker inti tetap terbaca), dan 20 baris dengan `Kuantitas` terbesar secara mentah.

**Temuan.**

| Statistik    | Nilai   |
| ------------ | ------- |
| count        | 693.563 |
| mean         | 30,39   |
| std          | 60,997  |
| min          | 1,0     |
| 25%          | 2,0     |
| median (50%) | 5,0     |
| 75%          | 34,0    |
| max          | 5.250,0 |

`Kuantitas` sangat _right-skewed_ (median 5, mean ~30,4, maksimum 5.250) dan bertipe kontinu
(float64) — nilai pecahan seperti `5,8` atau `92,5` genuinely muncul di data, bukan artefak
ekspor, dikonfirmasi pemilik data. Boxplot per kategori memperlihatkan dua kelompok: kategori
`Tambahan`, `Barang Jadi (FG)`, `Barang Semi FG (WIP-2)`, `Packaging` punya median dan IQR jauh
lebih tinggi dan lebar (item yang memang dikirim/dikonsumsi banyak per transaksi), sementara
`Bahan Baku (RM)`, `Barang Dalam Process (WIP-1)`, `Barang Umum`, `Minuman`,
`Perlengkapan Resto`, `Snack`, `Snack (FG)` konsisten tercatat median 1 unit per baris.

**Implikasi.** Praproses/model hilir wajib memperlakukan `Kuantitas` sebagai kontinu, bukan
count integer. 20 baris `Kuantitas` terbesar (nilai puncak 5.250 dkk.) perlu ditinjau manual
untuk kemungkinan kesalahan pengali satuan sebelum diperlakukan sebagai pesanan besar yang sah
— ini menjadi salah satu poin checklist bagian 1.14.

## 1.7 Deteksi Spike Relatif terhadap Baseline

**Tujuan.** `nlargest(20, "Kuantitas")` pada bagian 1.6 bias terhadap satuan dengan hitungan
per-transaksi besar (Porsi, PCS) dan hanya memunculkan satu item terbesar per hari peristiwa —
gagal menangkap lonjakan pada item bervolume kecil yang tetap signifikan _relatif terhadap
ukuran pesanan normalnya sendiri_ (mis. pesanan katering/aqiqah untuk item yang jarang
dipesan).

**Prosedur.** Setiap baris dibandingkan terhadap median historis pasangan (`Kode Barang`,
`Nama Cabang`)-nya sendiri (`baseline_ratio = Kuantitas / median historis pasangan`), dengan
pasangan yang punya <30 observasi dikecualikan supaya baseline tidak berisik, dan item
`xxx.`-berprefiks (kandidat discontinued) juga dikecualikan. Item dengan `baseline_ratio ≥ 5x`
dihitung per (`Tanggal`, `Nama Cabang`) untuk menghitung berapa item _berbeda_ yang melonjak
bersamaan pada hari yang sama — indikator peristiwa multi-item (pesanan besar/katering)
dibanding satu item melonjak sendirian.

**Temuan.** Hari dengan jumlah item melonjak bersamaan terbanyak: 22 Maret 2025 di KY001 dan
27 Juni 2025 di KY067 (masing-masing 17 item berbeda melonjak ≥5x baseline-nya). Studi kasus
mendetail pada 23 Februari 2025 di KY001 mengonfirmasi bahwa `Ayam Kebuli (0,9)` (~3,8x
baseline) dan `Rice Bowl 600 ml` (~7,1x baseline) ikut melonjak hari itu, meski keduanya absen
dari tabel `nlargest(20, "Kuantitas")` mentah semata karena nilai absolutnya (873 potong, 974
pcs) berada di bawah ambang top-20 mentah — bukan karena permintaannya sebenarnya diam. Item
bervolume kecil seperti `Loyang Besar`/`Cup Sambal Loyang Besar` melonjak jauh lebih tinggi
secara rasio (10–14x) namun nyaris tidak pernah muncul di ranking absolut, karena baseline-nya
memang kecil — pola khas paket aqiqah/katering yang jarang dipesan.

**Justifikasi.** Perbandingan ini membuktikan pentingnya menormalkan skala per pasangan
item-cabang sebelum menilai "seberapa besar" sebuah lonjakan — sebuah pertimbangan yang relevan
untuk desain fitur/loss yang sadar-skala di tahap pemodelan, dan untuk validasi manual
outlier di bagian 1.6.

## 1.8 Cakupan dan Kontinuitas Waktu

**Tujuan.** Memverifikasi tidak ada celah waktu yang tidak terduga pada level cabang, sebagai
prasyarat sebelum membangun panel harian per pasangan item-cabang.

**Prosedur.** Volume transaksi harian (jumlah baris per tanggal) divisualisasikan sebagai
deret waktu. Kelengkapan tanggal per cabang dihitung sebagai
`(jumlah tanggal unik tercatat) / (rentang hari dari tanggal-min sampai tanggal-max cabang itu) x 100%`,
dengan ambang 95% sebagai penanda cabang yang perlu ditinjau.

**Temuan.** Kelengkapan tanggal antar-cabang kuat: **hanya 1 dari 67 cabang berada di bawah
95%** — KY056 (Kebuli Yaman Tigaraksa), pada 92,25% (619 dari 671 hari yang diharapkan dalam
rentang 2024-03-01 s.d. 2025-12-31). 66 cabang lainnya seluruhnya di atas ambang.

**Implikasi.** Perlu dikonfirmasi ke pemilik data apakah celah KY056 adalah celah pelaporan,
penutupan sementara, atau outlet yang baru dibuka pertengahan periode — poin ini tetap terbuka
di checklist bagian 1.14.

## 1.9 Pola Musiman dan Deret Waktu

**Tujuan.** Mengidentifikasi pola musiman terstruktur (mingguan, bulanan, hari besar
keagamaan) pada total permintaan, sebagai dasar rekayasa fitur kalender.

**Prosedur.** Total `Kuantitas` harian diplot dengan _rolling mean_ 7 hari, dengan periode
Ramadan (`RAMADAN_PERIODS`) diarsir dan tanggal Idul Fitri/Idul Adha ditandai garis vertikal.
Rata-rata total harian dihitung per hari-dalam-minggu (generik lintas cabang — lihat catatan
di bagian 1.1) dan per kategori barang, per bulan, dan per kombinasi cabang x hari-dalam-minggu
(heatmap deviasi persen terhadap rata-rata harian cabang itu sendiri).

**Temuan.** Musiman mingguan dan musiman terkait Ramadan/Idul Fitri terlihat jelas baik secara
keseluruhan maupun per kategori barang. Heatmap cabang x hari-dalam-minggu memperlihatkan
variasi substansial antar cabang dalam pola mingguannya — beberapa cabang punya hari puncak
yang jelas berbeda dari cabang lain, mengindikasikan pola hari-dalam-minggu tidak seragam
lintas cabang dan sebaiknya dimodelkan per-cabang (atau per-region, begitu mapping-nya
tersedia — lihat bagian 1.13), bukan sebagai satu efek global.

**Justifikasi.** Karena penemuan ini heatmap-nya menggabungkan dua tahun data, kekuatan
pola per-cabang ini perlu diverifikasi terlebih dahulu sebagai pola _struktural_ (berulang
tiap tahun) dan bukan sekadar artefak satu tahun tertentu — itulah tujuan bagian 1.10 berikutnya.

## 1.10 Konsistensi Pola Mingguan 2024 vs 2025

**Tujuan.** Sebelum mempercayai deviasi hari-dalam-minggu per cabang (bagian 1.9) sebagai fitur
forecasting, memverifikasi apakah polanya struktural (berulang tiap tahun) atau sekadar
artefak satu tahun tertentu (cabang baru, promo, peristiwa sekali jalan).

**Prosedur.** Deviasi hari-dalam-minggu per cabang dihitung ulang terpisah untuk 2024 dan
2025 (hanya cabang dengan cakupan penuh tujuh hari di kedua tahun), lalu bentuk kurva
Senin–Minggu antar dua tahun itu dikorelasikan (Pearson) per cabang.

**Temuan.**

| Metrik                                                   | Nilai                |
| -------------------------------------------------------- | -------------------- |
| Cabang dengan cakupan 7-hari penuh di kedua tahun        | 48                   |
| Korelasi rata-rata per-cabang (2024 vs 2025)             | 0,915 (median 0,939) |
| Cabang dengan korelasi <0,5                              | 0                    |
| Korelasi gabungan seluruh sel cabang x hari-dalam-minggu | 0,929                |

Cabang paling konsisten: KY042 (Batavia, 0,998), KY012 (Cibinong, 0,995), KY001 (Kutabumi/
Pusat, 0,995), KY050 (TangCity Mall, 0,993), KY031 (Perum, 0,991). Cabang paling tidak
konsisten (tetap positif dan cukup tinggi): KY002 (Cilegon, 0,800), KY036 (Kota Harapan
Indah, 0,794), KY044 (Pahlawan Bogor, 0,787), KY005 (Pandeglang, 0,787), KY057 (Rawalumbu
Bekasi, 0,639) — masih di atas ambang 0,5, jadi tidak ada satu cabang pun yang gagal
memenuhi kriteria konsistensi.

**Justifikasi.** Korelasi rata-rata 0,915 dan tidak ada cabang di bawah ambang 0,5
mengonfirmasi pola hari-dalam-minggu per cabang bersifat struktural — aman dijadikan basis
fitur musiman-mingguan per cabang, bukan sekadar artefak satu tahun.

## 1.11 Struktur Item x Outlet

**Tujuan.** Mengukur tiga karakteristik struktural yang berdampak langsung pada desain model
dan evaluasi: keragaman SKU per cabang, intermitensi permintaan pada level pasangan
item-cabang, dan konsentrasi volume — plus memvalidasi ambang riwayat minimum yang sudah
dipakai di pipeline hilir.

**Prosedur.** Jumlah SKU distinct per cabang dan jumlah cabang distinct per SKU dihitung dan
divisualisasikan. Panel harian padat (`build_dense_panel`, lihat bagian 2.2) dibangun per pasangan
(`Kode Barang`, `Nama Cabang`), lalu persentase hari-nol-permintaan dihitung per pasangan.
Konsentrasi volume dihitung sebagai kurva Lorenz (persentase kumulatif volume vs persentase
kumulatif pasangan, terurut volume menurun). Panjang riwayat pra-cutoff (sebelum
`TEST_START = 2025-12-01`) per pasangan dibandingkan terhadap `MIN_HISTORY_DAYS = 60` yang
sudah dipakai `filter_min_history()` di pipeline pemodelan.

**Temuan.**

| Metrik                                                              | Nilai                     |
| ------------------------------------------------------------------- | ------------------------- |
| Bentuk panel harian padat                                           | 1.516.114 baris x 8 kolom |
| Median persentase hari-nol-permintaan per pasangan                  | 64,0%                     |
| Pasangan dengan >50% hari-nol-permintaan                            | 57,1%                     |
| Persentase pasangan (terurut volume) yang mencakup 80% volume total | 6,6%                      |
| Pasangan yang memenuhi ambang riwayat minimum 60 hari               | 3.040 / 3.882 (78,3%)     |

Permintaan sangat intermiten pada level pasangan item-cabang (median 64% hari nol; 57,1%
pasangan di atas 50% hari nol). Volume sangat terkonsentrasi: 6,6% pasangan teratas
menyumbang 80% volume total. Di bawah aturan `MIN_HISTORY_DAYS` saat ini, 78,3% pasangan
(3.040/3.882) lolos ambang; ~22% akan tersingkir.

**Justifikasi.** Intermitensi tinggi berarti pendekatan model bergaya ARIMA murni yang
mengasumsikan permintaan kontinu tidak cocok — ini keputusan metode pemodelan, bukan
pertanyaan data. Konsentrasi volume menjadi pertimbangan pembobotan evaluasi. ~22% pasangan
yang tersingkir ambang riwayat perlu ditinjau pemilik data: SKU baru, cabang baru, atau
celah pelaporan? — poin ini tetap terbuka di checklist bagian 1.14.

## 1.12 Analisis Proxy Lead Time

**Tujuan.** Memberi sinyal awal tentang jendela agregasi mana (3 hari vs 4 hari) yang lebih
mudah diprediksi, sebagai proxy generik untuk pertanyaan "permintaan kumulatif sampai
pengiriman berikutnya" — tanpa menambatkan ke tebakan pembagian region/hari mana pun (lihat
keterbatasan di bagian 1.1 dan bagian 1.13).

**Prosedur.** Jumlah bergulir (_rolling sum_) 3-hari dan 4-hari dihitung baik pada level total
harian per cabang maupun pada level pasangan item-cabang (dari panel harian padat bagian 1.11).
Koefisien variasi (`std/mean`) dari tiap jendela dihitung sebagai proksi prediktabilitas —
CV lebih rendah berarti jendela itu relatif lebih stabil/mudah diprediksi.

**Temuan.**

| Statistik CV (level pasangan item-cabang) | Jendela 3-hari | Jendela 4-hari |
| ----------------------------------------- | -------------: | -------------: |
| count                                     |          3.707 |          3.696 |
| mean                                      |          1,729 |          1,560 |
| median (50%)                              |          1,251 |          1,118 |
| 75%                                       |          2,087 |          1,850 |

Jendela 4-hari secara konsisten sedikit lebih prediktabel (CV median ~1,12) dibanding jendela
3-hari (CV median ~1,25), baik pada level cabang maupun level pasangan item-cabang.

**Justifikasi.** Ini proxy generik non-tersegmentasi-region — jangan difinalisasi sebagai
keputusan desain jendela lead time sebelum pemetaan Region 1/2 di bagian 1.13 tersedia, karena
jendela riil per cabang bergantung pada hari pengiriman cabang tersebut (Senin/Kamis untuk
Region 1, Selasa/Jumat untuk Region 2), bukan jendela tetap 3 atau 4 hari yang sama untuk
semua cabang.

## 1.13 Pertanyaan Terbuka dan Kesenjangan Data

**Tujuan.** Bagian ini menjalankan dua pemeriksaan ringan pendukung checklist bagian 1.14: kolom
apa saja yang tersedia di `dataset/outlets.csv` (menunjukkan mapping region ada atau tidak),
dan apakah masih ada `Kuantitas` negatif pada `dataset.csv` saat ini.

**Temuan.**

- **Kolom `dataset/outlets.csv`:** `Nama Outlet`, `Alamat`, `Kecamatan`, `Kota`,
  `has_shopee`, `has_gofood`, `has_grabfood`. **Tidak ada kolom region atau jadwal
  pengiriman.** Ini mengonfirmasi ulang batasan yang disebut di bagian 1.1 — pemetaan Region 1
  (Senin/Kamis) / Region 2 (Selasa/Jumat) memang belum ada di data manapun pada saat EDA ini
  dijalankan.
- **Kuantitas negatif:** 0 baris ditemukan pada `dataset/csv/dataset.csv` saat ini. Anomali
  historis di cabang KY011 tanggal 29 Februari 2024 (lihat `CLAUDE.md`) tidak lagi teramati
  pada berkas yang tersedia — namun ini pemeriksaan lunak (soft check), bukan assertion yang
  menggagalkan proses, sehingga perlu dijalankan ulang setiap kali `dataset.csv`
  diregenerasi.

## 1.14 Kesimpulan dan Checklist Verifikasi Pra-Modeling

Satu baris per temuan: kesimpulan dari pass EDA ini, dan — jika relevan — apa yang masih
perlu keputusan pemilik data/tim SCM sebelum dataset ini masuk ke
`normalize_items.py` / `build_panel.py` / `prepare_forecast_data.py`. Tabel ini adalah versi
gabungan dan diperbarui dari checklist yang sebelumnya ada di `docs/temuan-eda.md`
(sekarang dihapus); status "Selesai" ditambahkan untuk baris yang sudah ditindaklanjuti di
kode sejak checklist itu pertama ditulis (2026-08-06).

| #   | Temuan / Kesimpulan                                                                                                                                                                                                                                                                                  | Rujukan                 | Verifikasi sebelum pemodelan                                                                                                                                                                                                                                                                                                                       |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Kualitas data bersih: 693.563 baris, 2024-01-01 s.d. 2025-12-31, 0 nilai hilang, 0 duplikat.                                                                                                                                                                                                         | bagian 1.3              | Tidak ada — siap dipakai apa adanya.                                                                                                                                                                                                                                                                                                               |
| 2   | `Kuantitas` sangat _right-skewed_ (median 5, mean ~30,4, maks 5.250) dan genuinely fraksional (float64) — dikonfirmasi disengaja oleh pemilik data, bukan artefak ekspor.                                                                                                                            | bagian 1.6              | Tinjau manual 20 baris outlier teratas (bagian 1.6) untuk kemungkinan kesalahan pengali satuan/input sebelum diperlakukan sebagai pesanan besar yang sah. Pastikan kode praproses/model hilir memperlakukan `Kuantitas` sebagai kontinu, bukan count integer.                                                                                      |
| 3   | Kelengkapan tanggal per cabang kuat: 66/67 cabang ≥95%; hanya KY056 (Kebuli Yaman Tigaraksa) di bawah, pada 92,3%.                                                                                                                                                                                   | bagian 1.8              | **Terbuka.** Tanyakan ke pemilik data apakah celah KY056 adalah celah pelaporan, penutupan sementara, atau outlet yang lebih baru.                                                                                                                                                                                                                 |
| 4   | Permintaan sangat intermiten pada level pasangan item-cabang (median 64% hari-nol-permintaan; 57,1% pasangan >50% nol).                                                                                                                                                                              | bagian 1.11             | Tidak ada — ini keputusan metode pemodelan (hindari asumsi permintaan kontinu bergaya ARIMA murni), bukan pertanyaan data.                                                                                                                                                                                                                         |
| 5   | Volume terkonsentrasi: 6,6% pasangan item-cabang teratas menyumbang 80% volume total.                                                                                                                                                                                                                | bagian 1.11             | Tidak ada — menjadi pertimbangan pembobotan evaluasi, bukan pertanyaan data.                                                                                                                                                                                                                                                                       |
| 6   | 78,3% pasangan item-cabang (3.040/3.882) lolos ambang `MIN_HISTORY_DAYS` = 60 hari yang ada saat ini; ~22% akan tersingkir aturan saat ini.                                                                                                                                                          | bagian 1.11             | **Terbuka.** Pemilik data meninjau pasangan yang gagal — SKU baru, cabang baru, atau celah pelaporan? — sebelum memutuskan membuang atau melonggarkan ambang.                                                                                                                                                                                      |
| 7   | Item berprefiks `xxx.` tidak selalu bisa otomatis digabung ke padanan non-prefiksnya — mis. `xxx.FGS.00069` "Cendol Pandan" vs `FGS.00069` "Cendol" beda nama, sementara `xxx.FGS.00070`/`00071` (Santan Cendol, Gula Cendol) justru identik namanya dengan padanannya.                              | bagian 1.7              | **Selesai.** `utils/data_preprocessing/normalize_items.py` sudah menangani ini secara eksplisit per kode: `xxx.FGS.00069` masuk `EXCLUDED_ITEMS` (dibuang, bukan digabung paksa), sementara `xxx.FGS.00070`/`00071` digabung lewat `NAME_OVERRIDES`/normalisasi kode karena namanya memang identik.                                                |
| 8   | Tidak ditemukan baris `Kuantitas` negatif pada `dataset/csv/dataset.csv` saat ini — anomali historis KY011 2024-02-29 tidak lagi teramati.                                                                                                                                                           | bagian 1.13             | **Belum permanen dikonfirmasi pemilik data.** Jalankan ulang pemeriksaan ini setiap kali `dataset.csv` diregenerasi — ini soft check, bukan assertion yang menggagalkan proses.                                                                                                                                                                    |
| 9   | Setiap `Kode Barang` memetakan ke tepat satu `Satuan` — 0/109 SKU punya lebih dari satu satuan.                                                                                                                                                                                                      | bagian 1.4              | Tidak ada — dikonfirmasi konsisten, aman untuk mengagregasi `Kuantitas` per SKU.                                                                                                                                                                                                                                                                   |
| 10  | 27/109 SKU (24,8%) tercatat di bawah lebih dari satu `Kategori Barang` sepanjang waktu — pola pasangan konsisten yang menyerupai _rename_ taksonomi kategori pertengahan 2024.                                                                                                                       | bagian 1.4              | **Selesai.** `normalize_items.py` menangani lewat `CATEGORY_SYNONYMS` (penyatuan varian penamaan) dan `EXPLICIT_CATEGORY_OVERRIDES` (relabel 10 SKU yang dikonfirmasi pemilik data 2026-08-22, plus FGS-00014 2026-08-10). `utils/eda/verify_category_consistency.py` adalah gerbang refresh yang memverifikasi ulang setiap kali data diperbarui. |
| 11  | Tidak ada pemetaan jadwal pengiriman Region 1 (Sen/Kam) / Region 2 (Sel/Jum) di `dataset/outlets.csv` — hanya ada `Kecamatan`/`Kota`. Ini menghambat versi tersegmentasi-region dari analisis hari-dalam-minggu (bagian 1.9) dan jendela lead time (bagian 1.12), yang saat ini hanya proxy generik. | bagian 1.1, bagian 1.13 | **Terbuka — prioritas tertinggi.** Minta tim SCM menambahkan kolom cabang → region (atau cabang → hari-pengiriman) di `dataset/outlets.csv`, digabungkan dengan cara yang sama seperti `outlet_features.py` menggabungkan `has_shopee`/`has_gofood`/`has_grabfood`.                                                                                |
| 12  | Proxy lead time: jendela bergulir 4-hari sedikit lebih prediktabel dari jendela 3-hari (CV median ~1,12 vs ~1,25).                                                                                                                                                                                   | bagian 1.12             | Proxy generik non-tersegmentasi-region — jangan difinalisasi sebagai keputusan desain sebelum mapping region baris #11 tersedia.                                                                                                                                                                                                                   |
| 13  | Musiman mingguan dan musiman terkait Ramadan/Idul Fitri terlihat jelas, baik keseluruhan maupun per kategori; pola hari-dalam-minggu per cabang terbukti struktural (korelasi 2024 vs 2025 mean 0,915, tidak ada cabang <0,5).                                                                       | bagian 1.9, bagian 1.10 | `check_year_coverage()` di `calendar_features.py` sudah menjadi assertion runtime yang menggagalkan pipeline bila tahun data tidak tercakup `RAMADAN_PERIODS`/`EID_AL_FITR_DATES`/`EID_AL_ADHA_DATES` — cakupan 2024–2025 karena itu terjamin selama pipeline utama berjalan sukses, tidak perlu pengecekan manual terpisah.                       |

**Urutan prioritas:** baris 3, 6, dan 11 adalah penghambat terbesar yang masih terbuka — masing-
masing berpotensi mendistorsi keluaran `normalize_items.py`/`build_panel.py` secara diam-diam
jika dilewatkan, dan tidak satu pun bisa diselesaikan dari data saja. Baris 7 dan 10, yang
dulunya juga masuk prioritas tertinggi, sudah selesai ditindaklanjuti di kode sejak checklist
ini pertama ditulis.

## 1.15 Ringkasan dan Kaitan dengan Tahap Berikutnya

Pass EDA ini mengonfirmasi `dataset/csv/dataset.csv` bersih secara struktural (bagian 1.3),
memvalidasi asumsi kunci yang dipakai tahap normalisasi hilir (bagian 1.4), dan mengkarakterisasi
tiga sifat data yang menentukan pendekatan pemodelan: intermitensi tinggi, konsentrasi volume,
dan musiman mingguan yang struktural per cabang (bagian 1.9–1.11). Ia juga secara eksplisit
menandai satu kesenjangan data yang paling berdampak — pemetaan Region 1/2 yang belum ada
(bagian 1.13) — sebagai prasyarat sebelum jendela lead time dan fitur hari-dalam-minggu bisa
difinalisasi secara tersegmentasi-region. Checklist bagian 1.14 menjadi gerbang yang harus ditinjau
pemilik data sebelum tahap normalisasi dan rekayasa fitur (`utils/data_preprocessing/`)
dijalankan atas asumsi bahwa seluruh temuan EDA sudah diperhitungkan.

---

# Bagian 2 — Teknis (Mendetail)

## 2.1 Struktur Notebook dan Entry Point

`notebook/eda.ipynb` terdiri atas 53 sel (15 sel markdown berjudul + 38 sel kode), mengikuti
konvensi notebook mandiri (lihat "Notebook convention" di `CLAUDE.md`): tidak ada
`sys.path.append`/`from utils... import` untuk logika inti — setiap fungsi yang dipakai
didefinisikan langsung di **Bagian 1. Konfigurasi** dan **Bagian 2. Fungsi & konstanta
pipeline yang dipakai di sini** (sel 2 dan 4), lalu dipakai sepanjang sisa notebook.
Satu-satunya impor eksternal `utils` terjadi di sel opsional terakhir (bagian 2.3 di bawah), khusus
untuk memverifikasi sinkronisasi, bukan untuk menjalankan analisis.

Menjalankan seluruh notebook dari awal:

```bash
jupyter nbconvert --to notebook --execute --inplace --allow-errors notebook/eda.ipynb
```

`find_base_dir()` (sel 2) mencari root repo dengan berjalan ke atas dari direktori kerja
mencari folder `dataset/csv/`, sehingga notebook berjalan sama baik dibuka dari `notebook/`
(Jupyter) maupun dieksekusi dari root repo (`nbconvert --execute`) — tidak pernah
mengandalkan path relatif `..` yang di-hardcode.

`DATA_PATH` = `dataset/csv/dataset.csv`; `OUTLETS_FILE` = `dataset/outlets.csv`. Keduanya
hanya **dibaca**, tidak pernah ditulis ulang oleh notebook ini.

## 2.2 Fungsi dan Konstanta yang Didefinisikan di Notebook

Seluruhnya didefinisikan di sel 2 dan sel 4 (Bagian 1–2 notebook), dan dipakai berkali-kali
di sel-sel analisis berikutnya:

- **`find_base_dir(start=None) -> Path`** — lihat bagian 2.1.
- **Palet warna** (`BLUE`, `BLUE_DARK`, `CATEGORICAL`, `DIVERGING_CMAP`) — satu biru
  sekuensial untuk grafik magnitudo/ranking satu seri, palet kategorikal kecil (≤8 warna)
  untuk multi-seri yang identitasnya memang berbeda, dan colormap diverging khusus untuk
  satu-satunya grafik berpolaritas (heatmap deviasi cabang x hari-dalam-minggu, bagian 1.9).
- **`RAMADAN_PERIODS`, `EID_AL_FITR_DATES`, `EID_AL_ADHA_DATES`** — salinan verbatim dari
  `utils/data_preprocessing/calendar_features.py`, dipakai bagian 1.9 (arsiran/garis penanda pada
  plot deret waktu) dan bagian 1.13 (via `check_year_coverage`, lihat bagian 2.3).
- **`TEST_START = pd.Timestamp("2025-12-01")`** — batas awal set test, salinan dari
  `modeling_prep.py`/desain split; dipakai bagian 1.11 untuk memisahkan riwayat pra-cutoff saat
  menghitung kelayakan `MIN_HISTORY_DAYS`.
- **`MIN_HISTORY_DAYS = 60`, `PAIR_COLS = ["Kode Barang", "Nama Cabang"]`,
  `CARRY_COLS = ["Kategori Barang", "Nama Barang", "Satuan"]`, `SEGMENT_COL = "segment_id"`** —
  salinan dari `utils/data_preprocessing/build_panel.py`, dipakai bagian 1.11.
- **`load_outlets(path) -> pd.DataFrame`** — baca `dataset/outlets.csv` (`;`, `utf-8-sig`).
  Dipakai bagian 1.13 untuk memeriksa kolom yang tersedia (cek mapping region).
- **`filter_min_history(df, cutoff, min_days, pair_cols, date_col) -> pd.DataFrame`** —
  membuang pasangan (item, cabang) yang belum punya `min_days` hari riwayat sebelum `cutoff`.
  Salinan verbatim dari `utils/data_preprocessing/build_panel.py`. Dipakai bagian 1.11.
- **`_drop_closed_dates`, `_segment_ids`, `build_dense_panel`** — reindex tiap pasangan
  item-cabang ke panel harian padat atas rentang aktifnya sendiri, dengan hari di dalam
  interval penutupan cabang tidak menghasilkan baris sama sekali (bukan di-nol-kan), dan tiap
  jalur tanggal berurutan diberi nomor `segment_id` supaya lag/rolling window/target tidak
  pernah menjembatani penutupan atau relokasi cabang. Salinan verbatim dari
  `utils/data_preprocessing/build_panel.py`. Dipakai bagian 1.11 (panel dasar untuk metrik
  intermitensi) dan bagian 1.12 (panel dasar untuk rolling sum level-pasangan).
- **`MIN_PAIR_HISTORY = 30`** (sel bagian 1.7) — pasangan item-cabang dengan observasi kurang dari
  ini dikecualikan dari perhitungan `baseline_ratio` supaya baseline-nya tidak berisik.
- **`SPIKE_RATIO_THRESHOLD = 5.0`, `DETAIL_RATIO_THRESHOLD = 2.0`** (sel bagian 1.7) — ambang
  "melonjak" untuk menghitung hari-peristiwa multi-item, dan ambang lebih rendah untuk
  tampilan detail keranjang satu-hari supaya lonjakan menengah tidak tersembunyi.

## 2.3 Mekanisme Sinkronisasi dengan `utils/data_preprocessing`

Sel terakhir notebook (bagian 15, _"Cek sinkron dengan `utils/`"_) adalah satu-satunya sel yang
mengimpor dari `utils` (`utils.data_preprocessing.build_panel`, `.calendar_features`,
`.outlet_features`). Ia membandingkan `inspect.getsource()` tiap fungsi/konstanta yang punya
nama sama di notebook dan di modul `utils` tersebut, setelah menghapus qualifier modul
(`build_panel.`, `calendar_features.`, `outlet_features.`) dari kode `utils` supaya
perbandingannya adil (di notebook semua fungsi berbagi satu namespace tanpa qualifier).

Hasil verifikasi saat notebook ini terakhir dijalankan (2026-08-26): **5 fungsi + 8 konstanta
identik** dengan `utils/data_preprocessing/`. Sel ini bersifat opsional dan bukan bagian dari
alur analisis — ia hanya alarm dini bila salinan notebook mulai menyimpang dari sumber
`utils/` yang sebenarnya diuji (lihat "Notebook convention" di `CLAUDE.md`).

Berkaitan dengan bagian 1.13: `check_year_coverage()` di `utils/data_preprocessing/calendar_features.py`
(dipanggil dari `add_calendar_features()`, bukan dari notebook ini) adalah assertion runtime
yang menggagalkan pipeline utama bila tahun yang muncul di data tidak tercakup
`RAMADAN_PERIODS`. Notebook EDA ini sendiri **tidak memanggil** fungsi tersebut — ia hanya
memakai konstanta kalender untuk visualisasi (bagian 1.9) — sehingga jaminan cakupan tahun berasal
dari pipeline utama saat dijalankan, bukan dari EDA ini.

## 2.4 Ringkasan Angka Terverifikasi (per Bagian)

Tabel ini mengumpulkan seluruh angka kunci dari Bagian 1 dalam satu tempat, untuk memudahkan
pemeriksaan silang cepat tanpa membaca ulang tiap subbagian.

| Bagian notebook                        | Angka kunci                                                                                                                                                    |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3–4 (Muat & health check)              | 693.563 baris x 7 kolom; 0 nilai hilang; 0 duplikat (persis maupun kunci bisnis)                                                                               |
| 5 (Konsistensi satuan/kategori)        | 0/109 SKU multi-satuan; 27/109 (24,8%) SKU multi-kategori                                                                                                      |
| 6 (Struktur cabang/SKU)                | 67 cabang; SKU volume tertinggi: Nasi Kebuli, Sambal - FG (>4,1 juta unit masing-masing)                                                                       |
| 7 (Distribusi Kuantitas)               | median 5,0; mean 30,39; std 60,997; maks 5.250,0                                                                                                               |
| 8 (Deteksi spike)                      | hari peristiwa multi-item terbanyak: 22 Mar 2025 KY001 dan 27 Jun 2025 KY067 (17 item melonjak ≥5x)                                                            |
| 9 (Cakupan waktu)                      | 1/67 cabang <95% kelengkapan tanggal (KY056, 92,25%)                                                                                                           |
| 11 (Konsistensi mingguan 2024 vs 2025) | korelasi mean 0,915 (median 0,939); 0 cabang <0,5; korelasi gabungan 0,929                                                                                     |
| 12 (Struktur item x outlet)            | panel padat 1.516.114 baris; median hari-nol 64,0%; 57,1% pasangan >50% nol; 6,6% pasangan = 80% volume; 78,3% (3.040/3.882) pasangan lolos `MIN_HISTORY_DAYS` |
| 13 (Proxy lead time)                   | CV median: 3-hari 1,251, 4-hari 1,118 (level pasangan item-cabang)                                                                                             |
| 14 (Open questions)                    | `outlets.csv`: 7 kolom, tanpa mapping region; 0 baris `Kuantitas` negatif                                                                                      |
| 15 (Sinkron `utils/`)                  | 5 fungsi + 8 konstanta identik dengan `utils/data_preprocessing/`                                                                                              |

## 2.5 Cakupan dan Batasan Dokumen Ini

Secara eksplisit di luar cakupan dokumen ini, karena merupakan alat/modul terpisah yang tidak
dijalankan sebagai bagian dari `notebook/eda.ipynb`:

- **`utils/eda/verify_category_consistency.py`** — gerbang refresh command-line
  (`python3 -m utils.eda.verify_category_consistency`) yang memvalidasi ulang temuan bagian 1.4
  setelah normalisasi; dirujuk di bagian 1.4 sebagai status penyelesaian, tetapi bukan bagian dari
  alur notebook ini.
- **`utils/eda/analyze_spike_recovery.py`** dan **`utils/eda/analyze_spike_comovement.py`** —
  analisis lanjutan atas fenomena spike yang diperkenalkan di bagian 1.7, dengan cakupan dan metodologi
  sendiri di luar dokumen ini.
- **`utils/eda/generate_item_cost_margin_template.py`** — alat pembangkit template terpisah,
  tidak berkaitan dengan analisis eksploratif yang dibahas di sini.
- **Tahap normalisasi dan rekayasa fitur** (`normalize_items.py`, `build_panel.py`,
  `calendar_features.py`, `outlet_features.py`, `outlier_handling.py`,
  `prepare_forecast_data.py`) — dirujuk di beberapa tempat (bagian 1.4, bagian 2.2, bagian 2.3) sebagai konsumen
  temuan EDA ini, tetapi implementasi detailnya didokumentasikan di
  `docs/superpowers/specs/2026-07-21-forecast-data-prep-design.md` dan dokumen desain terkait,
  bukan di sini.
