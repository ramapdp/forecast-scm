# Metodologi Pemodelan dan Strategi Pemilihan Model

Dokumen tunggal yang menjelaskan alur penelitian ini ujung ke ujung: dari
penggabungan lima berkas ekspor mentah, melewati prapemrosesan sampai data siap
model, lalu pembangunan tiga model, dan berakhir pada **strategi pemilihan model
terbaik yang direncanakan** beserta protokol pembukaan test set.

| Atribut | Keterangan |
|---|---|
| Objek studi | Jaringan gerai Kebuli Yaman — 59 cabang aktif di 16 kota |
| Periode data | 1 Januari 2024 – 31 Desember 2025 (731 hari kalender) |
| Unit analisis | Pasangan (kode barang × cabang) per hari kalender |
| Berkas siap model | `dataset/model_ready/model_input.parquet` — 1.502.522 baris × 82 kolom |
| Fitur | 56 kolom (`modeling_prep.FEATURE_COLS`) |
| Target | `target_lead_time_cumulative` |
| Kandidat model | Random Forest kuantil, XGBoost kuantil, LSTM kuantil |
| Kriteria pemilihan | `pinball@0.9` |
| Status | Ketiga model selesai divalidasi; **pemenang belum ditetapkan**; test set Desember 2025 masih terkunci |
| Artefak model | `models/{random_forest,xgboost,lstm}_q90.joblib` **basi per 2026-08-23** — dilatih sebelum refresh kategori WIP-2, akan dilatih ulang (§19a) |
| Verifikasi | 667 unit test lulus (`.venv/bin/python3 -m unittest discover -p "test_*.py"`) |
| Tanggal dokumen | 22 Agustus 2026 |

**Hubungan dengan dokumen lain.** Dokumen ini adalah lanjutan dari
`docs/metodologi-preprocessing.md`. Untuk detail per-fungsi tiap tahap
prapemrosesan, rujuk dokumen itu (formal, untuk laporan) atau
`docs/dokumentasi-preprocessing-id.md` (naratif, beserta trade-off tiap
keputusan). Bagian I di sini menulis ulang alur itu **pada tingkat strategi** —
apa yang dilakukan tiap tahap, keputusan apa yang diambil, dan kenapa — supaya
dokumen ini bisa dibaca berdiri sendiri, bukan supaya menggantikan keduanya.
Angka hasil terukur tiap model ada di `docs/hasil-modeling-{rf,xgb,lstm}.md`.
Batasan yang tidak bisa dihilangkan dengan kode ada di
`docs/batasan-penelitian.md`.

---

## Daftar Isi

- [0. Kerangka: tiga tingkat keputusan](#0-kerangka-tiga-tingkat-keputusan)
- **Bagian I — Strategi prapemrosesan data**
  - [1. Titik berangkat: data mentah dan perumusan target](#1-titik-berangkat-data-mentah-dan-perumusan-target)
  - [2. Tahap 1–4 — Menyatukan dan membersihkan sumber](#2-tahap-14--menyatukan-dan-membersihkan-sumber)
  - [3. Tahap 5 — Panel harian padat dan segmentasi](#3-tahap-5--panel-harian-padat-dan-segmentasi)
  - [4. Tahap 6–7 — Kalender dan penanganan lonjakan](#4-tahap-67--kalender-dan-penanganan-lonjakan)
  - [5. Tahap 8 — Rekayasa fitur dan target](#5-tahap-8--rekayasa-fitur-dan-target)
  - [6. Tahap 9–12 — Ekspor, pemisahan, purging, QA](#6-tahap-912--ekspor-pemisahan-purging-qa)
  - [7. Tahap 13–14 — Prapemrosesan pemodelan dan adapter](#7-tahap-1314--prapemrosesan-pemodelan-dan-adapter)
  - [8. Enam aturan anti-kebocoran yang berulang](#8-enam-aturan-anti-kebocoran-yang-berulang)
- **Bagian II — Pembangunan tiga model**
  - [9. Kontrak bersama: satu mesin evaluasi, tiga model disuntikkan](#9-kontrak-bersama-satu-mesin-evaluasi-tiga-model-disuntikkan)
  - [10. Lantai: tiga baseline naif](#10-lantai-tiga-baseline-naif)
  - [11. Model 1 — Random Forest kuantil](#11-model-1--random-forest-kuantil)
  - [12. Model 2 — XGBoost kuantil](#12-model-2--xgboost-kuantil)
  - [13. Model 3 — LSTM kuantil](#13-model-3--lstm-kuantil)
  - [14. Ringkasan banding konstruksi](#14-ringkasan-banding-konstruksi)
- **Bagian III — Strategi komparasi dan pemilihan model**
  - [15. Metrik dan justifikasinya](#15-metrik-dan-justifikasinya)
  - [16. Posisi hasil saat ini](#16-posisi-hasil-saat-ini)
  - [17. Strategi pemilihan: tangga kriteria bertingkat](#17-strategi-pemilihan-tangga-kriteria-bertingkat)
  - [18. Penerapan tangga pada angka yang ada](#18-penerapan-tangga-pada-angka-yang-ada)
  - [19. Protokol pembukaan test set Desember](#19-protokol-pembukaan-test-set-desember)
  - [20. Apa yang akan dan tidak akan disimpulkan](#20-apa-yang-akan-dan-tidak-akan-disimpulkan)
  - [21. Rencana kerja tersisa](#21-rencana-kerja-tersisa)

---

## 0. Kerangka: tiga tingkat keputusan

Satu prinsip mengatur seluruh desain penelitian ini:

> **Setiap keputusan dinilai pada data yang tidak ikut membuat keputusan itu.**

Prinsip itu diterapkan bertingkat, karena di proyek ini ada tiga jenis keputusan
yang sering tertukar namanya menjadi "menguji model":

| Tingkat | Keputusan yang diambil | Dinilai pada | Status |
|---|---|---|---|
| **A** | Hyperparameter terbaik **di dalam satu arsitektur** | Fold 3 & 5 (September, November 2025) | ✅ selesai untuk ketiganya |
| **B** | Arsitektur pemenang **antar tiga model** | Fold 1, 2, 4 (Juli, Agustus, Oktober 2025) — potongan yang tidak menyentuh tingkat A | ⬜ inti dokumen ini |
| **C** | Seberapa baik pemenang bekerja | Desember 2025 | 🔒 belum dibuka |

Tingkat C **bukan pemilihan**. Ia pengukuran. Membalik urutan B dan C — menguji
ketiganya di Desember lalu memilih yang tertinggi — akan membuat angka final
menjadi maksimum dari tiga undian, yaitu estimasi yang bias ke atas secara
sistematis dan tidak bisa dipertahankan sebagai ukuran kinerja.

Kekeliruan yang sering muncul di sini adalah anggapan bahwa metrik hanya lahir
dari test set. Tidak: metrik lahir dari **data mana pun yang tidak dipakai
melatih**. Validasi walk-forward proyek ini mencakup 345.547 baris di lima bulan
terpisah, sementara Desember hanya menyumbang 49.717 baris yang dinilai di satu
bulan yang paling tidak mewakili operasi normal (Natal dan Tahun Baru). Untuk
urusan **memilih**, bukti validasi jauh lebih kuat.

---

# Bagian I — Strategi Prapemrosesan Data

## 1. Titik berangkat: data mentah dan perumusan target

### 1.1 Sumber

Lima berkas CSV hasil ekspor sistem POS, membagi waktu tanpa tumpang tindih:

| Berkas | Cakupan |
|---|---|
| `jan-24.csv` | Januari 2024 |
| `feb-24.csv` | Februari 2024 |
| `mar-24.csv` | Maret 2024 |
| `apr-des-24.csv` | April–Desember 2024 |
| `jan-des-25.csv` | Januari–Desember 2025 |

Setiap baris adalah satu item barang keluar ("Barang Keluar") dengan tujuh
kolom: `Tanggal`, `Kategori Barang`, `Kode Barang`, `Nama Barang`,
`Nama Cabang`, `Satuan`, `Kuantitas`. Total 693.563 baris transaksi mentah.

Empat berkas konfigurasi yang dipelihara manual mendampinginya:
`outlets.csv` (master cabang), `outlet_name_overrides.csv` (koreksi nama),
`outlet_mapping.csv` (kawasan & hari pengiriman), `event_driven_items.csv`
(SKU yang permintaannya digerakkan acara), dan `outlet_closures.csv` (interval
cabang tidak beroperasi).

### 1.2 Perumusan target — keputusan paling menentukan di seluruh proyek

Tim SCM pusat mengirim barang dengan jadwal tetap:

| Kawasan | Hari pengiriman |
|---|---|
| Kawasan 1 | Senin dan Kamis |
| Kawasan 2 | Selasa dan Jumat |

Konsekuensinya, pertanyaan bisnis yang sebenarnya **bukan** "berapa permintaan
besok", melainkan **"berapa total permintaan sampai pengiriman berikutnya"** —
karena itulah jumlah yang harus dinaikkan ke truk hari ini. Maka target utama
penelitian ini adalah:

```
target_lead_time_cumulative(H) = Σ Kuantitas(H+1 .. H+lead_time_days)
```

dengan `lead_time_days` dihitung per baris dari `(hari transaksi, kawasan)`,
selalu ketat ke depan — tidak pernah 0, bahkan ketika hari transaksi itu sendiri
hari pengiriman. Transaksi hari Senin di Kawasan 1 punya lead time 3 hari
(ke Kamis); transaksi hari Kamis punya lead time 4 hari (ke Senin depan).

Target ini dibangun dari **`Kuantitas` mentah, bukan yang di-cap** (lihat §4.2):
lonjakan adalah permintaan nyata yang model harus dinilai terhadapnya, bukan
sesuatu yang disembunyikan dari label.

44,35% baris target bernilai nol. Angka ini penting dan akan berulang di seluruh
dokumen — ia alasan kenapa satu angka metrik global menyesatkan.

### 1.3 Batasan yang melekat pada perumusan ini

`Tanggal` adalah **tanggal pengambilan, bukan tanggal pesanan** (dikonfirmasi
pemilik data 2026-08-15). Pelanggan yang memesan Senin untuk diambil Kamis
menghasilkan baris bertanggal Kamis; manajer outlet mengabari pusat hari Senin,
tetapi tidak ada yang tercatat sampai barang benar-benar keluar. Sistem POS
sengaja tidak menyimpan tanggal pesanan karena pesanan bisa dibatalkan.

Akibatnya seluruh deret waktu proyek ini berjalan pada sumbu waktu pengambilan,
dan model bekerja dengan informasi **lebih sedikit** daripada yang sudah dimiliki
manajer outlet pada saat yang sama. Ini plafon akurasi yang tidak bisa dinaikkan
dengan kode. Rinciannya di `docs/batasan-penelitian.md` (B-1, B-2, B-3).

---

## 2. Tahap 1–4 — Menyatukan dan membersihkan sumber

**Tahap 1 — Penggabungan periode** (`merge_dataset.py`). Lima CSV menjadi satu
`dataset/dataset.csv` dengan skema 7 kolom yang seragam. Dua kuirk sumber
ditangani di sini: BOM UTF-8 di awal setiap berkas, dan dua kolom kosong
tambahan di `jan-des-25.csv` (9 field vs 7).

**Tahap 2 — Agregasi baris duplikat** (`aggregate_dataset.py`). Baris yang
berbagi (tanggal, kategori, kode, nama, cabang, satuan) dijumlahkan
`Kuantitas`-nya. Satu transaksi bisa muncul sebagai beberapa baris item di
ekspor mentah.

**Tahap 3 — Normalisasi kode barang** (`normalize_items.py`). Empat pekerjaan:

1. Membuang prefiks `xxx.` dan menyeragamkan tanda pemisah pada `Kode Barang`.
2. Menggabungkan kode yang hanya berbeda kosmetik — **hanya bila nama barangnya
   juga sepakat**. Syarat kedua ini yang mencegah dua produk berbeda menyatu.
3. Mengonversi segelintir item yang `Kuantitas`-nya tercatat dalam gram alih-alih
   `Porsi` (Santan Cendol faktor 40, Gula Cendol faktor 30 — diturunkan dari
   fakta bahwa **setiap** nilai mentahnya kelipatan bulat faktor itu), supaya
   menyatu bersih dengan deret Porsi-nya di periode berikutnya.
4. Membuang item yang dikonfirmasi tidak lagi dijual (Nasi Putih, Cendol Pandan,
   Ayam Crispy Original/Spicy).

Catatan metodologis yang layak dikutip: dua item terakhir sebelumnya
*digabung paksa* ke SKU lain lewat tabel `EXPLICIT_ITEM_RENAMES`. Setelah
ditelusuri, penggabungan itu ternyata menyatukan produk yang benar-benar
berbeda. Keduanya sekarang dibuang, dan tabel rename dikosongkan — contoh
keputusan pembersihan yang dibatalkan karena tidak lolos pemeriksaan, bukan
karena hasilnya kurang bagus.

**Tahap 4 — Penyaringan dan kanonikalisasi cabang** (`outlet_features.py`).
`filter_matched_branches` membuang baris cabang yang tidak punya entri di
`outlets.csv` (cabang yang sudah tidak beroperasi). `canonicalize_branch_names`
menulis ulang setiap `Nama Cabang` ke nama kanonik outletnya, sehingga cabang
yang tercatat dengan dua string berbeda di sumber (mis. nama pendek warisan yang
hanya dipakai di satu periode ekspor) menyatu jadi satu riwayat yang bersambung,
bukan terbelah dua. Setelah itu `reaggregate_daily` dijalankan lagi untuk
menjumlahkan baris yang baru saja bertabrakan akibat penggantian nama.

---

## 3. Tahap 5 — Panel harian padat dan segmentasi

`build_panel.py` mengubah log transaksi menjadi **panel harian padat**: setiap
pasangan (barang × cabang) direindeks ke satu baris per hari kalender sepanjang
rentang tanggal yang teramati untuk pasangan itu. Hari tanpa transaksi diisi
`Kuantitas = 0`, kolom deskriptif di-forward-fill.

Kepadatan ini bukan kosmetik — ia **prasyarat**. Seluruh fitur lag dan rolling
di tahap 8 dibangun dengan `shift()`, yang mengasumsikan baris ke-*n* adalah
hari ke-*n*. Satu tanggal bolong di tengah deret akan membuat "lag 7 hari"
diam-diam berarti sesuatu yang lain.

### 3.1 Nol yang benar vs. nol yang dikarang: `segment_id`

Masalahnya: mengisi nol untuk hari saat cabangnya **tidak beroperasi** berarti
mengarang riwayat permintaan. `dataset/outlet_closures.csv` mencatat interval
`[tanggal_tutup, tanggal_buka)` per cabang kanonik, dan tanggal di dalam interval
itu **tidak menghasilkan baris sama sekali**. Setiap rentetan tanggal aktif yang
bersambung diberi nomor `segment_id`.

Semua fitur berbasis `shift` kemudian dikelompokkan per (pasangan, segmen),
sehingga **tidak ada lag, rolling window, target, atau jendela LSTM yang
menyeberangi masa tutup**.

Skala kesalahan yang dicegah: 19.304 baris nol-karangan ada di data, dan
membuat `branch_avg_daily_qty` cabang KY011 Bekasi Galaxy meleset 3,6×
(104,0 vs 371,3) — menempatkannya sebagai cabang terkecil dari 59, padahal
peringkat sebenarnya #46.

`detect_unrecorded_gaps()` memperingatkan jeda transaksi ≥14 hari yang tidak
dijelaskan konfigurasi, **tetapi tidak pernah bertindak sendiri**. Ambang 14 hari
menangkap kandidat, tidak mendefinisikannya — setiap kandidat dikonfirmasi ke
pemilik data sebelum masuk `outlet_closures.csv`. Lewat mekanisme inilah jeda
13 hari KY068 Kramatwatu (2025-06-28 s/d 2025-07-10) ditemukan dan dikonfirmasi.

### 3.2 Relokasi: segmen baru, baris tetap

Relokasi diperlakukan berbeda dari penutupan. `OBSERVED_RELOCATION_DATES`
memulai segmen baru **tanpa membuang baris**: outlet tidak pernah berhenti
berdagang, ia pindah pasar. Tingkat permintaannya bergeser 2,18×–2,64× di tiga
kepindahan yang teramati, dengan data pasca-pindah yang cukup untuk mengukurnya.

Hanya relokasi yang **teramati di dalam data** yang memenuhi syarat. Lima
relokasi bertanggal batas-bawah menggambarkan kepindahan setelah cakupan data
berakhir; memutus segmen di sana akan mengiris segmen sepanjang satu hari di
dalam jendela test.

### 3.3 Penyaringan riwayat minimum

`filter_min_history` membuang pasangan dengan riwayat kurang dari 60 hari sebelum
batas Desember 2025 — tidak cukup untuk jendela lag/rolling terpanjang.
Penyaringan tetap di tingkat pasangan, bukan segmen, karena riwayat di kedua sisi
sebuah penutupan tetap riwayat yang sah.

---

## 4. Tahap 6–7 — Kalender dan penanganan lonjakan

### 4.1 Fitur kalender (`calendar_features.py`)

Hari-dalam-minggu, hari-dalam-bulan, bulan, penanda akhir pekan, hari libur
nasional Indonesia, serta penanda **plus fitur jarak** (hari-menuju dan
hari-sejak) untuk empat musim tinggi: Ramadan/Idulfitri, Iduladha, HUT RI
(17 Agustus), dan Tahun Baru (1 Januari).

Tahap ini sengaja berjalan **sebelum** penanganan lonjakan, karena penanda
acaranya dipakai untuk memutuskan lonjakan mana yang dikecualikan dari
pemangkasan.

### 4.2 Penanganan lonjakan permintaan (`outlier_handling.py`)

`compute_pair_baseline` menghitung median historis `Kuantitas` tiap pasangan
**hanya dari transaksi nyata (bukan nol hasil pengisian) di periode training**.
Pasangan dengan kurang dari 30 hari semacam itu ditandai tidak memenuhi syarat
dan tidak pernah dipangkas.

`apply_outlier_capping` menandai baris ≥5× median itu sebagai `is_spike` dan
menghasilkan `Kuantitas_capped` — nilai dipangkas ke `median × 5`, **kecuali**
bila baris jatuh di jendela musim tinggi yang dikenal, yang dibiarkan utuh
karena diperlakukan sebagai pola berulang yang nyata, bukan derau.

Tiga keputusan halus di sini:

1. **Pembulatan ke atas untuk pasangan bilangan bulat.** Pasangan yang seluruh
   riwayat training-nya bilangan bulat ditandai `pair_integer_only`, dan
   pemangkasannya dibulatkan **ke atas** ke unit bulat berikutnya (lalu dijepit
   kembali ke kuantitas mentah, sehingga `Kuantitas_capped ≤ Kuantitas` tetap
   berlaku). Median berakhiran ,5 akan menempatkan cap pada setengah PCS;
   membulatkan ke atas — bukan ke terdekat — adalah arah yang melayani
   "outlet tidak kehabisan". Flag diturunkan dari riwayat pasangan itu sendiri,
   bukan dari `Satuan`, karena satuannya tidak terbelah bersih: `Potong` membawa
   6.510 baris pecahan sementara `PCS` dan `Botol` tidak sama sekali.
2. **Efek terukur:** 7.552 baris dipangkas, dan nilai cap pecahan turun dari 238
   menjadi 31 — semuanya pada pasangan yang memang berdagang dalam pecahan
   `Potong`.
3. **`baseline_ratio` dan `is_spike` disimpan sebagai kolom tetapi dikeluarkan
   dari `FEATURE_COLS`.** Keduanya diturunkan dari `Kuantitas` hari itu sendiri,
   sementara setiap fitur lag dan rolling berhenti di H-1. Memasukkannya akan
   membuat "diketahui pada saat prediksi" berarti dua hal berbeda dalam satu
   baris.

`Kuantitas` mentah dipertahankan utuh untuk perhitungan target.

---

## 5. Tahap 8 — Rekayasa fitur dan target

`prepare_forecast_data.py::build_featured_dataset` menjalankan urutan berikut.
Perhatikan pembagian sumbernya — ini inti strategi anti-kebocoran:

| Komponen | Dihitung dari | Alasan |
|---|---|---|
| Target (`target_h1..h7`, `target_lead_time_cumulative`) | `Kuantitas` **mentah** | Lonjakan adalah permintaan nyata yang model harus dinilai terhadapnya |
| Fitur lag & rolling | `Kuantitas_capped` | Satu hari ekstrem tidak boleh mendominasi input |
| Statistik cabang | `Kuantitas_capped`, **hanya periode training** | Dibekukan lalu diterapkan ke kedua split |

Rinciannya:

- **`add_targets`** — `target_h1`…`target_h7`, yaitu `Kuantitas` mentah digeser
  1–7 hari ke depan, dikelompokkan per (pasangan, segmen). Karena
  `lead_time_days` tidak pernah melebihi 4, `target_h5`–`target_h7` dibuang
  pipeline ini. Target harian ini tidak dilatih di penelitian ini; ia
  dipertahankan dan divalidasi untuk dekomposisi penjelas (§21).
- **`apply_region_features`** — menggabungkan `kawasan`/`hari_pengiriman` dari
  `outlet_mapping.csv`, lalu menghitung `lead_time_days` per baris.
- **`apply_outlet_features`** — fitur statis per cabang: `kota`, `has_shopee`,
  `has_gofood`, `has_grabfood`, dan turunannya `can_order_online`.
- **`add_relocation_feature`** — `days_since_relocation` (negatif sebelum tanggal
  relokasi, 0 pada hari itu, positif sesudahnya; `NaN` untuk cabang yang tidak
  pindah). Fitur ini ada karena kanonikalisasi nama berjalan lebih dulu,
  sehingga `kota`/`kawasan` cabang yang pindah mencerminkan lokasi
  **sekarang** untuk **seluruh** riwayatnya — termasuk baris pra-relokasi yang
  tercatat di lokasi lama yang sering beda kota. Flag ini yang memberi tahap
  pemodelan cara memperhitungkan pergeseran rezim itu.
- **`add_lead_time_target`** — target utama (§1.2).
- **`add_lag_features`** — lag 1, 2, 3, 7, 14, 21, 28 hari.
- **`add_rolling_features`** — rata-rata & simpangan baku bergulir 7/14/28 hari,
  **digeser satu hari sebelum jendela dihitung**, sehingga nilai hari ini tidak
  pernah bocor ke statistik bergulirnya sendiri.
- **`compute_branch_stats` / `apply_branch_stats`** — rata-rata kuantitas harian
  cabang, koefisien variasi permintaan, tingkat volume, dan umur cabang dalam
  hari.

---

## 6. Tahap 9–12 — Ekspor, pemisahan, purging, QA

**Tahap 9 — Ekspor.** `dataset/model_ready/featured.parquet`, tabel penuh yang
sudah bersih dan berfitur: 1.502.522 baris × 68 kolom.

**Tahap 10 — Pemisahan train/test.** Train = sebelum 2025-12-01; test =
Desember 2025. Target yang tanggal sasarannya jatuh setelah 2025-12-31
dibiarkan `NaN` alih-alih mempersempit jendela test.

**Purging** (`utils/purging.py`) — baris training yang jendela lead-time-nya
menjangkau ke Desember **dibuang**, bukan dipertahankan: labelnya sebagian
dijumlahkan dari permintaan periode test. 6.188 baris sebelum perbaikan, 0
sesudahnya. `fold_train_mask()` menerapkan purge yang sama di setiap batas fold
walk-forward.

**Tahap 11 — Ekspor split.** `train.parquet` dan `test.parquet`.

**Tahap 12 — Pemeriksaan QA.** `run_qa_checks()` berjalan dari skrip maupun
notebook, memeriksa sembilan invarian: tidak ada `Kuantitas` negatif; tidak ada
baris (barang, cabang, tanggal) duplikat; `Kuantitas_capped` tidak pernah melebihi
mentahnya; tidak ada `kota == "Unknown"`; tidak ada cabang tanpa `kawasan`;
tidak ada cabang yang memetakan ke lebih dari satu kota; tidak ada baris di
dalam interval penutupan; `segment_id` mulai dari 1 dan bersambung per pasangan;
dan tidak ada jeda tanggal **di dalam** sebuah segmen — invarian kepadatan yang
menjadi sandaran `shift`.

> **Pelajaran metodologis yang layak dicatat di laporan.** Tahap 1–12 didefinisikan
> **satu kali saja**, di dalam fungsi yang dipanggil baik oleh notebook maupun
> skrip. Sebelumnya notebook menyalin urutan langkah dengan tangan, dan salinan
> itu melewatkan `add_relocation_feature` — diam-diam mengekspor
> `featured.parquet` berkolom 62 alih-alih 68. Kesalahan itu tidak terdeteksi
> oleh test mana pun, karena keduanya sama-sama "berhasil". Perbaikannya struktural:
> notebook sekarang memanggil fungsi, bukan mengulang daftar langkahnya.

---

## 7. Tahap 13–14 — Prapemrosesan pemodelan dan adapter

`utils/modeling_prep.py` menambahkan lima hal terakhir sebelum data menyentuh
model, menghasilkan `model_input.parquet` (1.502.522 baris × 82 kolom).

### 7.1 `is_event_driven`

Penanda per-SKU dari `dataset/event_driven_items.csv` — barang yang
permintaannya digerakkan acara.

### 7.2 `demand_segment` — klasifikasi Syntetos-Boylan

Setiap pasangan diklasifikasikan dari dua besaran yang dihitung
**hanya dari periode training**:

- **ADI** — rata-rata interval antar hari berpermintaan (ambang 1,32)
- **CV²** — kuadrat koefisien variasi kuantitas bukan-nol (ambang 0,49)

| | CV² < 0,49 | CV² ≥ 0,49 |
|---|---|---|
| **ADI < 1,32** | `smooth` | `erratic` |
| **ADI ≥ 1,32** | `intermittent` | `lumpy` |

Segmentasi ini bukan fitur hiasan — ia **sumbu pelaporan wajib**. Dengan 44%
target bernilai nol, satu angka metrik global bisa menobatkan model yang hanya
unggul di tempat menebak nol itu mudah.

### 7.3 `fold_id` — lima fold walk-forward jendela mengembang

```
FOLD_STARTS = [2025-07-01, 2025-08-01, 2025-09-01, 2025-10-01, 2025-11-01]
```

Training untuk fold *k* adalah setiap baris bertanggal sebelum
`FOLD_STARTS[k-1]` (setelah purge); validasinya bulan itu sendiri. Desember 2025
sengaja tidak diberi label — ia test set final yang terkunci.

```
fold 1  train ██████████████████████ │ valid ▓ Jul 2025
fold 2  train ████████████████████████ │ valid ▓ Agu 2025
fold 3  train ██████████████████████████ │ valid ▓ Sep 2025
fold 4  train ████████████████████████████ │ valid ▓ Okt 2025
fold 5  train ██████████████████████████████ │ valid ▓ Nov 2025
FINAL   train ████████████████████████████████ │ 🔒 Des 2025
```

### 7.4 Imputasi yang mempertahankan makna

Nilai kosong diisi dengan **kolom indikator pendamping**: `was_relocated`,
`has_baseline`, `has_full_history`, `missing_history_count`. Nilai kosong pada
lag dan rolling ikut diisi — jendela LSTM menjangkau mundur melewati baris
warm-up, dan membiarkannya kosong membuat 5,43% jendela sekuens tidak terjangkau
padahal matriks tabularnya bersih.

> Catatan implementasi penting: `model_input.parquet` **sudah** terimputasi.
> Memanggil `impute_features()` untuk kedua kalinya akan menghitung ulang
> `was_relocated` dari kolom yang kini sudah terisi 0,0 — menyalakan indikator itu
> di **setiap** baris dan menghapus perbedaan yang justru menjadi alasan
> keberadaannya.

### 7.5 Pengindeksan kategorikal yang diperluas, tidak pernah dibangun ulang

Nilai kategorikal dipetakan ke indeks bilangan bulat, disimpan ke
`category_mapping.json`. Peta itu **diperluas, tidak pernah dibangun ulang**:
nilai baru ditambahkan setelah indeks tertinggi yang sudah dibagikan, dan nilai
yang sudah pensiun tetap di tempatnya.

Alasannya terukur: mengurutkan ulang seluruh himpunan saat data disegarkan
membuat enam SKU baru menggeser indeks 32 dari 70 nilai yang sudah ada —
diam-diam membatalkan model apa pun yang sudah dilatih.

### 7.6 `FEATURE_COLS` — 56 kolom

Kolom yang dilatih ketiga model, dengan `baseline_ratio` dan `is_spike`
dikecualikan secara sengaja (§4.2).

### 7.7 Dua adapter, satu kontrak

| Adapter | Untuk | Bentuk keluaran |
|---|---|---|
| `to_tabular()` | XGBoost, Random Forest | Matriks baris × 56 fitur |
| `to_sequences()` | LSTM | Jendela 28 hari yang berakhir **di baris prediksi, inklusif** |

Keduanya membuang 28 baris warm-up pertama tiap segmen (5,93% baris; 996 baris
test) dan baris **tanpa target** — 7.434 baris (0,53%), 4.333 di antaranya di
Desember (8,02% periode test). Dua pertiga dari yang di Desember (2.790) sekadar
tepi dataset dan akan terselesaikan begitu data 2026 masuk; sisanya milik 1.854
pasangan yang berhenti selamanya.

Konsekuensinya untuk pelaporan: **Desember dinilai pada 49.717 dari 55.046
baris panelnya.** Baris yang dikeluarkan bukan irisan yang bias — rata-rata
`Kuantitas` 19,99 vs 19,41 dan komposisi `demand_segment` nyaris identik
terhadap baris yang dipertahankan — tetapi penyebutnya wajib disebut di laporan.

`validate_contract()` menegaskan keduanya memaparkan himpunan (pasangan, tanggal),
target, dan penugasan fold yang **identik**; bahwa tidak ada target NaN; dan
bahwa tidak ada NaN di blok fitur mana pun. Inilah yang membuat perbandingan
model tabular vs. sekuens sah dilakukan.

---

## 8. Enam aturan anti-kebocoran yang berulang

Seluruh Bagian I sebenarnya penerapan berulang dari enam aturan yang sama.
Tabel ini ringkasan yang bisa dikutip langsung:

| # | Aturan | Diterapkan di | Yang dicegah |
|---|---|---|---|
| 1 | Setiap fitur historis berhenti di **H-1** | lag, rolling (digeser 1 hari) | Nilai hari ini bocor ke prediktornya sendiri |
| 2 | Setiap target ketat **ke depan** | `add_targets`, `add_lead_time_target` | Target memuat hari yang sudah teramati |
| 3 | Statistik agregat hanya dari **periode training**, lalu dibekukan | statistik cabang, baseline pencilan, `demand_segment` | Informasi masa depan mengalir ke fitur |
| 4 | **Purge** di setiap batas | batas Desember, batas tiap fold, ekor early stopping | Label yang sebagian tersusun dari periode penilaian |
| 5 | Tidak ada `shift` yang **menyeberangi segmen** | `segment_id` dioper ke semua fungsi berbasis shift | Lag menjembatani masa cabang tutup |
| 6 | Kolom turunan hari-ini **dikeluarkan dari fitur** | `baseline_ratio`, `is_spike` | Dua arti berbeda untuk "diketahui saat prediksi" dalam satu baris |

---

# Bagian II — Pembangunan Tiga Model

## 9. Kontrak bersama: satu mesin evaluasi, tiga model disuntikkan

Ini keputusan arsitektur paling penting di seluruh fase pemodelan, dan
alasannya metodologis, bukan teknis.

Sebuah perbandingan hanya layak dilaporkan bila ketiga model melihat **baris yang
sama persis**. Itu bukan sesuatu yang bisa dijamin oleh kedisiplinan penulisan
tiga skrip training terpisah. Maka jaminan itu dipindahkan ke struktur:

> `utils/walk_forward.py` **memiliki** definisi kelayakan baris, batas fold, dan
> penilaian. Ia tidak tahu apa pun tentang model. Seluruh antarmuka model adalah
> satu callable:
>
> ```python
> fit_predict(train_df, valid_df) -> np.ndarray
> ```

Apa pun yang dibutuhkan sebuah model di luar itu — pemilihan fitur, imputasi,
transformasi target, penskalaan — berada di dalam pembungkusnya sendiri, karena
justru itulah pilihan-pilihan yang ingin **dipaparkan** oleh perbandingan ini,
bukan disembunyikan.

Bukti bahwa mekanisme ini bekerja: ketiga baseline naif mencetak angka yang
**sama persis** di ketiga run model. `walk_forward.py` tidak disentuh sama sekali
saat LSTM ditambahkan.

### 9.1 Kelayakan baris — tiga potongan

`eligible_rows()` menerapkan, berurutan:

1. **Desember 2025 dan sesudahnya dibuang.** Redundan dengan definisi fold, dan
   sengaja dipertahankan: ongkos satu kebocoran tak sengaja adalah kredibilitas
   angka final, dan penjaga rangkap lebih murah dari itu.
2. **28 hari pertama tiap segmen dibuang.** Dihitung pada deret utuh, tidak
   pernah di dalam fold — potongan per-fold akan menghapus 28 hari pertama setiap
   bulan untuk setiap pasangan.
3. **Baris tanpa target dibuang.**

Hasilnya: **345.547 baris validasi** di lima fold.

### 9.2 Mesin bersama: `utils/model_common.py`

Bagian yang tidak dimiliki model mana pun secara khusus dikumpulkan di sini —
karena membiarkannya di dalam modul Random Forest berarti memperbaiki bug
checkpoint yang sama dua kali, lalu tiga kali saat LSTM datang:

| Komponen | Fungsi |
|---|---|
| `sample_search_space()` | Penarikan acak kandidat, dengan penyaring keterjangkauan yang **disuntikkan** (batas memori daun RF tidak punya padanan di XGBoost) |
| `run_search()` | Menilai tiap kandidat di fold pencarian, menulis **checkpoint tiap kandidat selesai**, melanjutkan dari sana bila dijalankan ulang |
| `select_best()` | **Kandidat dengan pinball gabungan terendah.** Satu baris yang menentukan segalanya |
| `expand_one_hot()` | Ekspansi kategorikal bersama |
| `split_early_stopping()` | Ekor 30 hari terakhir jendela training, **dengan purge yang sama** seperti di batas fold |
| `save_bundle()` / `load_bundle()` | Format bundel yang mendeskripsikan dirinya sendiri |

Kriteria seleksi tidak berpindah-pindah:

```python
best_id = int(scored.loc[scored["pinball"].idxmin(), "candidate_id"])
```

### 9.3 Protokol pencarian yang sama untuk ketiganya

| Aspek | Nilai |
|---|---|
| Fold pencarian | **Fold 3 (September) dan 5 (November) saja** |
| Kriteria | pinball@0.9 gabungan, **dibobot jumlah baris**, bukan dirata-rata polos |
| Subsampling | **Tidak ada** — seluruh baris training tiap fold dipakai |
| Seed | 42 |

Dua fold, bukan lima, karena pencarian harus murah: lima fold × puluhan kandidat
tidak muat di plafon waktu mana pun. Konsekuensinya — skor fold 3 dan 5 bukan
out-of-sample terhadap seleksi — ditangani secara eksplisit di §16.

---

## 10. Lantai: tiga baseline naif

Skor model tidak berarti apa-apa berdiri sendiri. Sebelum model apa pun dilatih,
lantainya ditetapkan lewat `evaluation.NAIVE_BASELINES`:

| Baseline | Prediksi |
|---|---|
| `naive_zero` | 0 |
| `naive_lag_1` | `lag_1 × lead_time_days` |
| `naive_roll_mean_7` | `roll_mean_7 × lead_time_days` |

Setiap baseline menskalakan estimasi permintaan yang melihat ke belakang dengan
jumlah hari yang harus ditanggung pengiriman. Keduanya sudah ada sebagai fitur,
jadi tidak berongkos — dan **persis itulah yang akan dilakukan manajer outlet
dengan tangan**.

`naive_roll_mean_7` adalah lantai yang sebenarnya. Di Desember ia mencapai
MAE 13,05 dan pinball@0.9 6,61 tanpa model sama sekali. Model yang tidak
melewatinya tidak layak diterapkan, seberapa pun canggih arsitekturnya.

> Angka baseline Desember di atas boleh dihitung tanpa melanggar kunci test set,
> karena tidak ada model yang terlibat dan tidak ada keputusan yang diambil
> darinya. Yang dikunci adalah **keputusan**, bukan aritmetika.

Coverage baseline itu **0,61** terhadap target service level 0,90. Itulah
argumen paling gamblang untuk melatih pada pinball alih-alih pada rata-rata:
peramal titik-tengah, secara konstruksi, kehabisan stok jauh lebih sering
daripada yang dijanjikan.

---

## 11. Model 1 — Random Forest kuantil

| | |
|---|---|
| Implementasi | `quantile_forest.RandomForestQuantileRegressor` |
| Modul | `utils/model_random_forest.py` |
| Ruang pencarian | 1.152 kombinasi |
| Kandidat ditarik | **18** |

### 11.1 Kenapa `quantile-forest`, bukan `RandomForestRegressor`

`RandomForestRegressor` milik sklearn hanya meminimalkan galat kuadrat atau
absolut — ia **tidak punya** loss kuantil. Quantile regression forest
(Meinshausen 2006) membaca kuantil 0,9 dari distribusi empiris yang disimpan di
tiap daun, dan itulah estimator yang benar untuk service level yang sudah dikunci.

### 11.2 Benchmark sebelum membakar jam

Satu fit pada training set penuh fold 5 dijalankan lebih dulu, untuk memastikan
batas penyimpanan daun yang dipakai menyaring kandidat memang berlaku
**sebelum** 18 fit dijalankan:

| | |
|---|---|
| Baris training fold 5 | 1.292.778 |
| Estimasi penyimpanan daun | 1,54 GB (budget 3 GB) |
| Wall time satu fit + predict | 6,6 menit (395 detik) |

Setiap kandidat disaring lebih dulu lewat `estimate_leaf_memory_bytes()`
terhadap budget 3 GB — penyaring yang disuntikkan ke `sample_search_space()`,
dan yang tidak punya padanan di dua model lain.

### 11.3 Ruang pencarian dan pemenangnya

```python
SEARCH_SPACE = {
    "max_depth":        [12, 16, 20],
    "min_samples_leaf": [20, 50, 100, 200],
    "max_samples_leaf": [1, 20, 50],
    "max_features":     ["sqrt", 0.3, 0.5, 1.0],
    "max_samples":      [None, 0.5],
    "log_target":       [False, True],
    "one_hot":          [False, True],
}
```

Parameter terpilih: `max_depth=12`, `max_features=0.5`, `min_samples_leaf=20`,
`max_samples_leaf=50`, `one_hot=True`, `log_target=False`, `n_estimators=200`.

Dua bacaan dari sebarannya:

1. **Ruangnya datar.** Dari 2,4221 ke 2,7476 hanya rentang 13%, dan lima kandidat
   teratas berjarak 0,6% satu sama lain. Pemenang mengalahkan runner-up dengan
   selisih 0,0004 — praktis seri.
2. **`max_features="sqrt"` satu-satunya pilihan yang benar-benar merugikan.**
   Dua kandidat terburuk keduanya memakainya. Dengan 56 fitur, `sqrt` menyisakan
   ~7 fitur per split — terlalu sedikit.

### 11.4 Model final

`fit_final()` melatih ulang konfigurasi pemenang dengan **`n_estimators`
dinaikkan 200 → 400** pada 1.349.011 baris layak sebelum Desember. Bundelnya
menyimpan urutan kolom training beserta flag `one_hot`/`log_target` — forest yang
dimuat ulang dengan urutan kolom berbeda tidak gagal, ia memprediksi dengan
percaya diri dari fitur yang salah.

`models/random_forest_q90.joblib` — **821 MB**.

---

## 12. Model 2 — XGBoost kuantil

| | |
|---|---|
| Implementasi | `xgboost==2.1.4`, `XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.9, tree_method="hist")` |
| Modul | `utils/model_xgboost.py` |
| Ruang pencarian | 2.592 kombinasi |
| Kandidat ditarik | **30** |

### 12.1 Objektif yang identik dengan kriteria

Fungsi objektifnya adalah pinball loss yang sama dengan kriteria seleksi. Apa
yang dioptimalkan saat training dan apa yang dinilai saat evaluasi bukan dua hal
berbeda — properti yang tidak berlaku untuk model galat-kuadrat yang dimintai
kuantil tinggi setelahnya.

### 12.2 Protokol dua fit

Jumlah ronde boosting itu sendiri adalah **keputusan regularisasi**, dan tempat
paling wajar untuk mengambilnya — fold validasi — justru tempat yang bocor. Maka:

1. Early stopping berjalan di **ekor 30 hari terakhir jendela training**
   (`split_early_stopping`, dengan purge yang sama seperti di batas fold),
   `EARLY_STOPPING_ROUNDS = 50`, `MAX_ROUNDS = 2000`.
2. Model dibuang, lalu **difit ulang pada seluruh baris training** pada jumlah
   ronde itu.

Fit kedua itu yang membuat XGBoost akhirnya dilatih pada populasi baris yang
sama persis dengan yang dilihat Random Forest.

`n_estimators` sengaja **tidak ada** di ruang pencarian: early stopping sudah
memutuskannya per kandidat per fold, jadi mencarinya akan menghabiskan anggaran
untuk pertanyaan yang sudah punya mekanisme.

Benchmark: 2,4 menit untuk dua fit penuh + predict — dua fit tanpa menambah waktu
jam dinding yang berarti.

### 12.3 Ruang pencarian dan pemenangnya

```python
SEARCH_SPACE = {
    "max_depth":        [4, 6, 8, 10],
    "learning_rate":    [0.03, 0.05, 0.1],
    "min_child_weight": [1, 10, 50],
    "subsample":        [0.7, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "reg_lambda":       [1.0, 10.0],
    "encoding":         ["ordinal", "native", "one_hot"],
    "log_target":       [False, True],
}
```

Parameter terpilih: `encoding="native"`, `max_depth=6`, `learning_rate=0.05`,
`min_child_weight=50`, `subsample=0.7`, `colsample_bytree=0.7`,
`reg_lambda=10.0`, `log_target=False`.

Tiga bacaan dari sebarannya:

1. **Ruangnya datar, lebih rapat dari RF.** Rentang 6,8%; pemenang unggul
   0,0001 dari runner-up.
2. **`encoding` memang memisah, dan `native` yang menang.** Empat kandidat teratas
   semuanya `native`, dan urutan mediannya sejalan dengan urutan minimumnya —
   penanganan kategori bawaan XGBoost (partisi kategori yang benar, bukan indeks
   yang diperlakukan sebagai besaran terurut) memang membeli sesuatu di sini.
   Perlu dicatat pembagian 9/7/14 kandidat itu hasil undian acak, bukan desain
   berimbang, jadi ini indikatif dan bukan eksperimen terkontrol.
3. **`log_target=True` merugikan** — melatih di ruang log lalu mentransformasi
   balik menggeser kuantil yang dioptimalkan.

Ronde per fold berkisar 313–682, tidak ada yang mendekati plafon 2.000 — bukti
early stopping bekerja, dan bahwa jumlah ronde memang bergerak menurut foldnya.

### 12.4 Model final

1.349.011 baris training, `best_iteration = 607`.
`models/xgboost_q90.joblib` — **4,7 MB**.

Bundelnya menyimpan urutan kolom training, flag `encoding`/`log_target`, dan
**level kategori** yang dipakai saat training. Yang terakhir wajib untuk mode
`native`: booster yang dimuat ulang bulan depan terhadap kategori yang diurutkan
berbeda tidak gagal — ia memprediksi dengan percaya diri dari fitur yang salah.

---

## 13. Model 3 — LSTM kuantil

| | |
|---|---|
| Implementasi | `torch==2.8.0`, `QuantileLSTM` |
| Modul | `utils/model_lstm.py`, `utils/sequence_windows.py` |
| Ruang pencarian | 48 kombinasi |
| Kandidat ditarik | **12** |

### 13.1 Arsitektur

```
49 kanal dinamis  ─→ LSTM 1 lapis (hidden 128) ─┐
                                                ├─→ concat ─→ Linear ─→ ReLU
7 kategorikal     ─→ Embedding (dim = min(16, (n+1)//2)) ─┘        ─→ Dropout
                     dibaca di baris prediksi saja                 ─→ Linear ─→ ŷ
```

Jendela 28 hari (`LOOKBACK`), berakhir **di baris prediksi, inklusif**.
Fungsi lossnya pinball — sama dengan kriteria seleksi, properti yang sama seperti
`reg:quantileerror` pada XGBoost.

### 13.2 Empat detail konstruksi yang menentukan sah-tidaknya hasilnya

1. **Kategorikal dibaca di baris prediksi, tidak diulang sepanjang jendela.**
   `Kategori Barang_idx` berubah di dalam 301 segmen nyata, jadi "kategori milik
   segmen ini" bukan hal yang terdefinisi untuk diulang 28 kali.
2. **Ukuran embedding datang dari `category_mapping.json`**, bukan dari nilai yang
   kebetulan muncul di baris training satu fold. Cabang yang baru buka setelah
   model dilatih memetakan ke slot UNKNOWN 0 dan tetap dalam rentang;
   alternatifnya gagal berbulan-bulan kemudian, di produksi, dengan index error.
3. **Scaler dipasang per fold, hanya dari baris training**, lalu dipakai kedua fit.
   Berbagi satu scaler itulah yang membuat `best_epoch` bermakna sama di kedua fit.
4. **Jendela dipotong dari panel penuh, bukan dari `eligible_rows()`.** Jendela
   milik baris validasi 1 Juli menjangkau mundur ke Juni, melewati baris yang
   dihapus warm-up dan purge fold. Membaca *fitur* baris itu aman: setiap jendela
   berakhir di baris prediksinya sendiri dan setiap lag berhenti di H-1, jadi
   tidak ada nilai target yang bisa masuk jendela. Yang dicegah purging adalah
   training atas *label* baris tersebut, dan itu tetap tidak pernah terjadi.

Protokol dua fitnya identik dengan XGBoost, karena masalahnya identik: jumlah
epoch adalah keputusan kapasitas. `MAX_EPOCHS = 100`,
`EARLY_STOPPING_EPOCHS = 5`.

### 13.3 Benchmark yang membentuk — dan memotong — ruang pencarian

Ini bagian paling instruktif dari ketiga model, dan layak ditulis apa adanya
di laporan.

| | |
|---|---|
| Device terpilih | **CPU** |
| `best_epoch` benchmark | 3 |
| `sec_per_epoch` | 102,0 |
| Wall time dua fit + predict | 18,7 menit |
| Peak RSS | 4,60 GB |
| `N_CANDIDATES` diturunkan | **12** |

1. **CPU menang atas MPS, 2×.** Probe 15 batch di fold 5: 0,392 s/batch di MPS
   lawan 0,193 s/batch di CPU — MPS tidak punya kernel LSTM ter-fusi di ukuran
   hidden ini.
2. **Ruang pencarian dikecilkan sebelum undian pertama ditarik.** Ongkos per epoch
   terukur:

   | hidden | layers | batch | s/epoch |
   |---:|---:|---:|---:|
   | 64 | 1 | 1024 | 75 |
   | 64 | 1 | 2048 | 47 |
   | 128 | 1 | 1024 | 104 |
   | 128 | 1 | 2048 | 83 |
   | 128 | **2** | 1024 | **259** |

   `SEARCH_SPACE` karena itu membuang `num_layers=2` dan `hidden_size=256`.
   Kedalaman yang dikorbankan, bukan lebar, karena membuangnya membeli detik
   paling banyak per dimensi yang dihapus. **Konsekuensinya: pencarian ini tidak
   pernah menanyakan apakah lapisan kedua akan menolong.**
3. **Jendela padat tidak pernah dimaterialisasi.** Tensor 1.502.522 × 28 × 56
   tidak muat di memori; `sequence_windows` membangun indeks dan mengambil jendela
   saat dibutuhkan.

### 13.4 Ruang pencarian dan pemenangnya

```python
SEARCH_SPACE = {
    "hidden_size":   [64, 128],
    "num_layers":    [1],
    "dropout":       [0.0, 0.2, 0.3],
    "learning_rate": [3e-4, 1e-3],
    "batch_size":    [1024, 2048],
    "log_target":    [False, True],
}
```

Parameter terpilih: `hidden_size=128`, `dropout=0.3`, `learning_rate=3e-4`,
`batch_size=1024`, `num_layers=1`, `log_target=False`.

Tiga bacaan:

1. **`log_target=True` bukan cuma merugikan — ia menghancurkan.** Empat kandidat
   `True` menempati empat posisi terbawah tanpa kecuali, median 3,190 lawan
   2,455: **30% lebih buruk**, jurang yang tidak ada bandingannya di pencarian RF
   maupun XGBoost (di XGBoost selisih mediannya hanya 1%). Pada LSTM, yang
   gradiennya mengalir lewat normalisasi dan bukan lewat partisi, distorsi
   transformasi balik itu tidak punya tempat sembunyi. MAE-nya ikut membengkak
   (23–29 lawan 15–17), jadi ini bukan pertukaran kalibrasi — ini murni lebih
   buruk.
2. **Sisanya datar**, rentang 2,7% di antara delapan kandidat `False`.
3. **`batch_size=2048` selalu kalah** dari kembarannya di 1024 — batch dua kali
   lebih besar berarti setengah langkah gradien per epoch.

### 13.5 Model final

1.349.011 baris training, `best_epoch = 8` (di tengah rentang 6–10 yang dipilih
kelima fold), wall time 43 menit.
`models/lstm_q90.joblib` — **466 KB**.

Bundelnya menyimpan `state_dict`, urutan kolom (dinamis dan kategorikal
terpisah), ukuran embedding, **scaler**, flag `log_target`, `lookback`, dan
`best_epoch`. Scaler wajib ikut: jaringan yang dimuat ulang lalu diberi fitur
berskala mentah tidak gagal — ia memprediksi dengan percaya diri dari input yang
salah skala.

Satu perbedaan pemakaian dibanding dua model lain: `predict_bundle()`
**mewajibkan panel**, bukan menerimanya sebagai opsi. LSTM tidak bisa memprediksi
dari satu baris sendirian — ia butuh 28 hari di belakangnya.

### 13.6 Plafon waktu yang terlampaui

Anggaran diturunkan dari `best_epoch = 3` yang terukur di benchmark, sementara
kandidat sebenarnya berhenti di epoch 3–13, sehingga ongkos riilnya ~11,4 jam
terhadap plafon 8 jam. `candidate_budget()` bekerja sesuai rumusnya; yang salah
adalah asumsi bahwa `best_epoch` benchmark mewakili ruang pencarian. Bila
mekanisme ini dipakai lagi, angka epoch yang disuntikkan sebaiknya yang paling
pesimistis, bukan yang terukur di satu konfigurasi.

---

## 14. Ringkasan banding konstruksi

| | Random Forest | XGBoost | LSTM |
|---|---|---|---|
| Mekanisme kuantil | Kuantil empiris dibaca dari daun | `reg:quantileerror` | Pinball loss langsung |
| Objektif = kriteria? | Tidak (dibaca, bukan dioptimalkan) | **Ya** | **Ya** |
| Kapasitas ditentukan | `n_estimators` dipatok | Early stopping per fold (313–682 ronde) | Early stopping per fold (6–10 epoch) |
| Protokol fit | Satu fit | **Dua fit** | **Dua fit** |
| Ruang pencarian | 1.152 | 2.592 | 48 (dua dimensi sudah dipotong) |
| Kandidat | 18 | **30** | **12** |
| Penyaring keterjangkauan | Batas memori daun 3 GB | Tidak perlu | Anggaran waktu |
| Bergantung seed acak | Tidak | Tidak | **Ya** |
| Benchmark satu putaran | 6,6 mnt (1 fit) | 2,4 mnt (2 fit) | 18,7 mnt (2 fit) |
| Wall time walk-forward | Berjam-jam | Berjam-jam | 3 jam 6 mnt |
| Wall time model final | tidak dicatat | tidak dicatat | 43 mnt |
| Ukuran artefak | **821 MB** | **4,7 MB** | **466 KB** |
| Butuh panel saat prediksi | Tidak | Tidak | **Ya** (28 hari) |
| Dependensi tambahan | `quantile-forest` | `xgboost` + `libomp` | `torch` |

Baris **kandidat** dan **ruang pencarian** adalah asimetri anggaran yang harus
disebut di setiap pembahasan hasil: perbandingannya belum dinetralkan.

---

# Bagian III — Strategi Komparasi dan Pemilihan Model

## 15. Metrik dan justifikasinya

### 15.1 Kriteria tunggal: `pinball@0.9`

```python
delta = actual - predicted
loss  = np.where(delta >= 0, alpha * delta, (alpha - 1.0) * delta).mean()
```

Kekurangan dikali **0,9**; kelebihan dikali **0,1**.

**Empat alasan berlapis:**

**(a) Karena service level-nya 0,9, dan itu keputusan bisnis yang sudah dikunci.**
Dikonfirmasi pemilik data 2026-08-16, **seragam untuk setiap SKU** — pemisahan
per-kategori (FG vs Packaging) secara eksplisit ditolak, karena kantor pusat
mengirim semua barang dalam satu konsinyasi sehingga satu service level mengatur
seluruh pengiriman. Angka 0,9 adalah pernyataan bisnis: *kehabisan stok sembilan
kali lebih mahal daripada kelebihan stok dalam jumlah yang sama.* Pinball@0.9
adalah terjemahan matematis langsung dari kalimat itu.

**(b) Karena meramal rata-rata berarti kehabisan stok separuh waktu.** Model yang
meminimalkan MSE meramal *mean*; yang meminimalkan MAE meramal *median*.
Keduanya, secara definisi, terlampaui permintaan aktual sekitar separuh hari.
Coverage baseline titik-tengah yang terukur — **0,61** — adalah bukti empirisnya.

**(c) Karena yang dilatih dan yang dinilai harus fungsi yang sama.** Dua dari tiga
model mengoptimalkan persis metrik yang menilai mereka (§14).

**(d) Karena kriterianya tidak berpindah.** `select_best()` memakainya sejak
tahap pencarian sampai pemilihan akhir.

### 15.2 Metrik pendamping dan perannya

`evaluation.score()` mengembalikan tujuh angka, dengan pembagian peran yang tegas:

| Metrik | Peran | Alasan keberadaannya |
|---|---|---|
| **`pinball`** | **Kriteria pemilihan** | Satu-satunya yang memutuskan |
| `coverage` | Cek kalibrasi | Dilatih di 0,9 → harus kembali ~0,9. Jauh di atas = overstock sistematis; jauh di bawah = janji service level tidak ditepati |
| `fill_rate` | Kriteria sukses pemilik data | "Outlet tidak kehabisan", dinyatakan dalam unit. Kekurangan dijumlahkan **sebelum** dibagi, sehingga surplus di satu outlet-hari tidak bisa menutupi kehabisan di outlet lain — barangnya sudah berada di cabang yang salah pada hari yang salah |
| `shortfall_units` / `overstock_units` | Penerjemah ke bahasa bisnis | "pinball 2,390" tidak bisa didiskusikan di rapat; "kekurangan turun 73%, kelebihan naik 2,5×" bisa |
| `mae` | **Konteks saja** | Dilaporkan, tidak memutuskan |
| `n` | Bukti baris identik | Menjamin perbandingan sah |

> **Catatan satuan.** `shortfall_units` dan `overstock_units` menjumlahkan unit
> lintas SKU yang satuannya campur (Kg, Porsi, Botol, PCS, …), jadi angkanya sah
> untuk membandingkan antar model **pada baris yang sama**, tetapi tidak punya
> makna fisik sebagai satu besaran tunggal.

### 15.3 Kenapa MSE / RMSE tidak dipakai — dan tidak dihitung

`utils/evaluation.py` tidak memiliki fungsi RMSE maupun MSE. Itu bukan
kelalaian. Empat alasan, dua di antaranya spesifik pada data ini:

**1. Simetris — bertentangan langsung dengan service level.** Menghukum
kelebihan stok sekeras kehabisan stok, padahal seluruh perumusan masalah
menyatakan keduanya tidak setara.

**2. Menobatkan baseline naif sebagai juara.** Ini bukan argumen hipotetis; ia
terukur:

| model | MAE | shortfall | overstock |
|---|---:|---:|---:|
| `naive_roll_mean_7` | **9,65** ← juara MAE | 1.528.393 | 1.804.789 |
| XGBoost | 14,31 | **414.172** | 4.529.674 |

Kriteria yang menobatkan rata-rata bergerak tujuh hari — model tanpa training
sama sekali, yang membiarkan kekurangan stok 3,7× lipat lebih besar — bukan
kriteria yang mengukur tujuan penelitian ini.

**3. RMSE akan memilih model yang paling jago meramal hal yang bukan tugasnya.**
Ini alasan terkuat, dan khusus proyek ini. RMSE mengkuadratkan galat, sehingga
didominasi baris bergalat terbesar — yaitu **lonjakan**. Tetapi
`docs/batasan-penelitian.md` B-3 menyatakan lonjakan justru **di luar lingkup
model**: logika bisnis saat ini adalah ramalan dibutuhkan untuk permintaan
*di luar pesanan*, sementara pesanan besar ditangani tim secara terpisah. Dan
besarannya terukur: **baris lonjakan menyumbang 11,5% absolute error padahal
hanya 2,41% baris.** Dengan RMSE, porsi itu dikuadratkan dan membengkak jauh
lebih besar lagi.

**4. 44,35% target bernilai nol.** Metrik yang didominasi tengah distribusi bisa
menobatkan model yang unggul hanya di tempat menebak nol itu mudah — alasan
`GROUP_COLS` ada.

> **Jika template laporan mewajibkan RMSE**, ia mudah ditambahkan ke
> `evaluation.score()` dan dilaporkan sebagai **metrik deskriptif pendamping**.
> Yang tidak boleh adalah menjadikannya dasar pemilihan. Tabel di alasan (2)
> adalah pembelaan yang siap dikutip saat pertanyaan itu muncul.

### 15.4 Tidak pernah satu angka global

Setiap hasil dilaporkan dalam **tiga potongan**: gabungan, per `demand_segment`,
dan per `is_delivery_day` (`walk_forward.GROUP_COLS`). Delivery day adalah baris
yang benar-benar menaikkan barang ke truk; segmen permintaan adalah sumbu tempat
satu angka global paling mudah menyesatkan.

---

## 16. Posisi hasil saat ini

Ketiga model sudah dijalankan penuh. Angka berikut dari **345.547 baris validasi
yang identik** — dijamin `walk_forward.eligible_rows()`, dikonfirmasi oleh fakta
bahwa ketiga baseline mencetak angka yang sama persis di ketiga run.

### 16.1 Gabungan

| potongan | model | pinball | mae | coverage |
|---|---|---:|---:|---:|
| semua fold | **xgboost** | **2,3896** | 14,307 | 0,909 |
| | random_forest | 2,4104 | 15,640 | 0,932 |
| | lstm | 2,4268 | 14,930 | 0,907 |
| **fold 1/2/4 (bersih)** | **xgboost** | **2,3998** | 14,351 | 0,910 |
| | random_forest | 2,4033 | 15,593 | 0,934 |
| | lstm | 2,4210 | **14,278** | 0,902 |
| | `naive_roll_mean_7` | 4,527 | 9,721 | 0,696 |

**Potongan fold 1/2/4 adalah angka yang sah untuk memilih**, karena fold 3 dan 5
ikut menentukan hyperparameter tiap model (§9.3).

### 16.2 Per fold

| fold | random_forest | xgboost | lstm |
|---|---:|---:|---:|
| 1 (Jul) | 2,1904 | 2,1897 | **2,1281** |
| 2 (Agu) | **2,3516** | 2,3699 | 2,4710 |
| 3 (Sep) | 2,3690 | **2,3104** | 2,4267 |
| 4 (Okt) | 2,6891 | **2,6584** | 2,6801 |
| 5 (Nov) | 2,4848 | **2,4464** | 2,4478 |

Tabel ini adalah bukti kunci untuk seluruh Bagian III: **varians antar-fold satu
model lebih besar daripada selisih antar-model.** LSTM mencetak skor terbaik yang
pernah dicapai model mana pun di fold mana pun (2,128 di Juli, 2,8% di bawah dua
tetangganya), lalu menjadi yang terburuk di Agustus (5,1% di atas RF). Dua fold
berurutan, satu model, arah berlawanan.

### 16.3 Per segmen permintaan

| segmen | n (validasi) | random_forest | xgboost | lstm |
|---|---:|---:|---:|---:|
| smooth | 45.485 | 8,570 | **8,373** | 8,482 |
| erratic | 54.511 | 4,810 | **4,785** | 4,909 |
| lumpy | 123.545 | **1,043** | 1,060 | 1,079 |
| intermittent | 122.006 | 0,426 | 0,435 | **0,425** |

Ketiganya membagi kemenangan. Tidak ada model yang mendominasi semua segmen.

### 16.4 Sisi bisnis

| | kekurangan (shortfall) | kelebihan (overstock) |
|---|---:|---:|
| `xgboost` | 414.172 | **4.529.674** |
| `lstm` | 403.337 | 4.755.695 |
| `random_forest` | **365.576** | 5.038.816 |
| `naive_roll_mean_7` | 1.528.393 | 1.804.789 |

Ketiganya memangkas kekurangan stok 73–76% dari baseline dengan ongkos kelebihan
stok 2,5–2,8× lipat. Apakah itu pertukaran yang benar adalah keputusan bisnis,
bukan keputusan model — tetapi itu **persis** pertukaran yang diminta ketika
service level dipatok di 0,9.

---

## 17. Strategi pemilihan: tangga kriteria bertingkat

Inilah prosedur yang direncanakan, **ditetapkan sebelum test set dibuka**.
Tangga dituruni berurutan; berhenti di anak tangga pertama yang benar-benar
memisahkan.

### Gerbang G0 — Kelayakan

> Model harus mengalahkan baseline terbaik (`naive_roll_mean_7`) pada
> pinball@0.9 **di kelima fold**, bukan hanya di gabungan.

Model yang menang secara gabungan tetapi kalah di satu bulan bukan model yang
bisa diterapkan; SCM mengirim setiap minggu, bukan setiap tahun.

### K1 — Kriteria utama: pinball@0.9 di potongan fold bersih (1/2/4)

> Pemenang adalah pinball terendah, **asalkan selisihnya ≥ 2%** terhadap
> pesaing terdekat. Di bawah ambang itu, hasilnya dinyatakan **seri** dan tangga
> diteruskan ke K2.

**Justifikasi ambang 2%.** Ambang ini tidak dipilih dari kebiasaan, melainkan
dikalibrasi dari kebisingan yang terukur di run ini sendiri:

- Selisih antar-model pada fold yang sama mencapai **5,1%** (LSTM vs RF di fold 2).
- Satu model yang sama bergerak **16%** antar fold berurutan (LSTM: 2,128 → 2,471).
- Setiap konfigurasi dilatih **satu kali dengan satu seed**, dan LSTM satu-satunya
  yang hasilnya bergantung pada inisialisasi acak dan urutan batch.

Dengan tingkat kebisingan sebesar itu, ambang 2% justru **konservatif**. Selisih
di bawahnya tidak bisa dipisahkan dari varians seed dan varians bulan.

### K2 — Kalibrasi terhadap service level yang dijanjikan

> `|coverage − 0,90|` pada potongan fold bersih, dibaca bersama stabilitasnya
> antar fold.

Ini bukan kriteria cadangan sembarangan — ia menguji **janji yang sama** dengan
kriteria utama, dari sudut yang berbeda. Model yang coverage-nya jauh di atas 0,90
menepati janji dengan cara yang mahal (overstock sistematis); yang jauh di bawah
tidak menepatinya sama sekali.

### K3 — Ongkos operasional dan reprodusibilitas

> Wall time training, ukuran artefak, ketergantungan pada seed acak, dan bobot
> dependensi.

Anak tangga ini sah menjadi penentu justru **karena** K1 menyatakan seri. Bila
tiga model sama akuratnya dalam batas yang bisa diukur, memilih yang paling murah
dan paling bisa direproduksi adalah keputusan rekayasa yang benar — bukan
kompromi.

### K4 — Risiko integrasi produksi

> Apa yang harus tersedia saat model dipanggil untuk meramal, dan apa yang gagal
> secara diam-diam bila salah dipasang.

---

## 18. Penerapan tangga pada angka yang ada

Menuruni tangga dengan angka §16 — ini **usulan**, ditulis sebelum Desember dibuka
dan menunggu persetujuan sebelum dibekukan:

### G0 — Kelayakan: ketiganya lolos

Semua mengalahkan `naive_roll_mean_7` di **kelima** fold dengan margin serupa
(~2,4 lawan ~4,5). Fold 4 (Oktober) konsisten paling berat bagi ketiganya —
properti bulannya, bukan properti modelnya.

### K1 — Pinball fold bersih: **SERI**

| model | pinball (fold 1/2/4) | selisih ke terbaik |
|---|---:|---:|
| xgboost | 2,3998 | — |
| random_forest | 2,4033 | +0,15% |
| lstm | 2,4210 | +0,88% |

Rentang juara ke juru kunci **0,88%** — di bawah ambang 2%. **Tangga diteruskan.**

### K2 — Kalibrasi: Random Forest tersisih

| model | coverage (bersih) | simpangan dari 0,90 | pola antar fold |
|---|---:|---:|---|
| lstm | 0,902 | **+0,002** | bergoyang 0,890–0,925, dua fold di bawah target |
| xgboost | 0,910 | +0,010 | menurun rapi 0,924 → 0,894 |
| random_forest | 0,934 | +0,034 | konservatif di **seluruh** fold |

**Random Forest tersisih di sini.** Simpangan +0,034 bukan derau — ia konsisten
searah di kelima fold, dan ongkosnya terukur: 5,04 juta unit kelebihan stok
lawan 4,53 juta milik XGBoost, **11% lebih besar**, untuk service level yang
dijanjikan sama. RF menepati janjinya dengan cara yang lebih mahal dari yang
diminta.

LSTM dan XGBoost keduanya lolos. LSTM sedikit lebih baik pada gabungan, XGBoost
lebih terduga polanya. Perbedaan keduanya tidak cukup tegas untuk memutuskan —
**tangga diteruskan.**

### K3 — Ongkos dan reprodusibilitas: **XGBoost menang telak**

| | xgboost | lstm |
|---|---|---|
| Satu putaran fit | **2,4 menit** | 18,7 menit |
| Walk-forward penuh | berjam-jam | 3 jam 6 menit |
| Pencarian | dalam plafon | **11,4 jam** (melampaui plafon 8 jam) |
| Kandidat teruji | 30 dari 2.592 | 12 dari 48 (2 dimensi sudah dipotong) |
| Ukuran artefak | 4,7 MB | 466 KB |
| Bergantung seed | **tidak** | **ya**, dan hanya satu seed dijalankan |
| Dependensi | `xgboost` + `libomp` | `torch` |

XGBoost hampir delapan kali lebih murah per putaran, deterministik, dan sudah menguji
ruang parameter yang jauh lebih luas. Reprodusibilitas bukan kenyamanan di sini:
hasil LSTM tidak dapat direproduksi persis tanpa mengunci seed, dan tidak ada
pengulangan seed yang pernah dijalankan untuk mengukur variansnya.

### K4 — Risiko integrasi: memperkuat arah yang sama

- **XGBoost** — bundel mendeskripsikan diri sendiri (urutan kolom, encoding, level
  kategori); memprediksi dari satu baris.
- **LSTM** — `predict_bundle()` **mewajibkan panel 28 hari**. Ia tidak bisa
  meramal satu baris sendirian. Untuk penerapan produksi, ini berarti pipeline
  panel harus hidup di sisi inferensi, bukan hanya di sisi training.
- **Random Forest** (sudah tersisih) — artefak 821 MB, sekitar 170× lebih besar
  dari XGBoost.

### Usulan keputusan

> **Model produksi yang diusulkan: XGBoost kuantil 0.9.**
>
> Dasarnya harus dinyatakan apa adanya: **bukan** karena XGBoost lebih akurat.
> Pada kriteria utama ketiganya seri dalam batas kebisingan yang bisa diukur run
> ini. XGBoost terpilih karena setelah Random Forest tersisih pada kalibrasi, ia
> **hampir delapan kali lebih murah, deterministik, sudah menguji ruang parameter
> terluas, dan paling ringan diintegrasikan** — perbedaan yang, tidak seperti
> selisih 0,88% pada pinball, besar dan nyata.

Menyatakan dasar itu terus terang lebih kuat, bukan lebih lemah, daripada
mengklaim kemenangan akurasi yang datanya tidak menopang.

---

## 19. Protokol pembukaan test set Desember

Enam langkah, dijalankan **sekali**, setelah §18 dibekukan dalam sebuah commit.

**1. Bekukan keputusan lebih dulu.** Dokumen ini di-commit dengan pemenang
tertulis, **sebelum** satu baris Desember disentuh model mana pun. Commit itulah
bukti bahwa pemilihan tidak dipengaruhi hasil test.

**2. Satu run, ketiga model sekaligus.** Ketiga model final di `models/*.joblib`
plus ketiga baseline dinilai pada baris Desember yang identik, dalam satu
eksekusi. Tidak dicicil — agar tidak ada kesempatan mengintip lalu mengulang.

> **Prasyarat yang belum terpenuhi (2026-08-23).** Ketiga berkas `models/*.joblib`
> yang ada sekarang dilatih 19–20 Agu 2026, sebelum refresh kategori WIP-2
> 2026-08-22 memindahkan 19.987 baris dari WIP-2 ke `Barang Jadi (FG)`.
> Berkas-berkas itu **tidak boleh dipakai** untuk langkah ini: ketiganya masih
> memuat WIP-2 sebagai kategori hidup (kolom one-hot pada RF, level 4 pada XGB,
> baris embedding pada LSTM) padahal data sekarang tidak pernah menghasilkannya
> lagi. Tidak ada satu pun yang akan gagal saat dimuat — itulah risikonya.
> Ketiganya dilatih ulang lebih dulu sebagai bagian dari migrasi multi-kuantil
> (`docs/superpowers/specs/2026-08-22-model-comparison-refactor-migration.md`),
> dan §19 baru dapat dijalankan di atas hasil pelatihan ulang itu. Rinciannya di
> §0 `docs/pipeline-overview.md` dan B-9 `docs/batasan-penelitian.md`.

**3. Laporkan penyebutnya.** Desember dinilai pada **49.717 dari 55.046 baris
panel** (996 baris warm-up, 4.333 baris tanpa target). Baris yang dikeluarkan
bukan irisan yang bias, tetapi angkanya wajib disebut.

**4. Angka utama = skor model terpilih.** pinball@0.9, coverage, dan fill_rate
XGBoost adalah angka final penelitian. Dua model lain dilaporkan sebagai
**pembanding deskriptif**, dengan label eksplisit bahwa pemenang ditetapkan
sebelum Desember dibuka.

**5. Tiga potongan, bukan satu angka.** Gabungan, per `demand_segment`, per
`is_delivery_day` — sama seperti seluruh pelaporan validasi.

**6. Jangan menukar pemenang.** Bila model lain mencetak angka lebih tinggi di
Desember, itu **temuan** yang ditulis di pembahasan — dan temuan yang justru
mengonfirmasi klaim K1 bahwa urutannya tidak stabil di bawah tingkat kebisingan
ini. Ia bukan alasan mengganti pilihan.

### 19.1 Yang akan membatalkan rencana ini

Satu-satunya hasil yang menuntut lebih dari sekadar pelaporan adalah **kegagalan
kalibrasi yang besar**: coverage XGBoost di Desember jatuh jauh di bawah 0,90
(indikatif: < 0,85). Itu bukan sinyal untuk menukar model — ketiganya menargetkan
kuantil yang sama dan akan bergerak searah — melainkan sinyal bahwa Desember
berperilaku berbeda dari lima bulan validasi, yang harus dibahas sebagai
keterbatasan generalisasi musiman, bukan disembunyikan.

---

## 20. Apa yang akan dan tidak akan disimpulkan

Bagian ini ada supaya klaim di laporan tidak melampaui apa yang ditopang data.

### Yang bisa dinyatakan dengan yakin

- Ketiga model **mengalahkan baseline operasional secara telak dan konsisten**:
  ~46–47% lebih baik pada pinball@0.9, di kelima fold, tanpa satu bulan pun yang
  menggendong hasilnya.
- Model kuantil **memangkas kekurangan stok 73–76%** dibanding praktik
  rata-rata-bergerak, dengan ongkos kelebihan stok 2,5–2,8× — pertukaran yang
  memang diminta service level 0,9.
- **Kalibrasi ketiganya mendarat di sasaran** (0,902–0,934 terhadap 0,90), yang
  berarti mekanisme kuantilnya bekerja sebagaimana dirancang.
- Perbedaan antar ketiga arsitektur **berada di bawah tingkat kebisingan yang
  bisa diukur run ini**.

### Yang tidak boleh dinyatakan

- ❌ "XGBoost adalah arsitektur terbaik untuk peramalan permintaan rantai pasok."
  Anggarannya tidak setara (30 vs 18 vs 12 kandidat), ruang LSTM sudah kehilangan
  dua dimensi kapasitas sebelum undian pertama, satu seed per konfigurasi, satu
  dataset.
- ❌ "LSTM lebih buruk dari XGBoost." Tanpa pengulangan seed, "sedikit lebih
  buruk" tidak bisa dipisahkan dari "dapat seed yang kurang beruntung".
- ❌ Klaim apa pun berbasis MAE terhadap baseline (§15.3).
- ❌ Klaim yang mengandaikan sumbu waktu pemesanan (`docs/batasan-penelitian.md`
  B-1, B-2, B-3).

### Batasan yang wajib disebut di bab batasan

1. **Desember belum dibuka** saat dokumen ini ditulis; seluruh angka Bagian III
   adalah validasi walk-forward.
2. **Sumbu waktunya waktu pengambilan, bukan waktu pemesanan** — plafon akurasi
   yang tidak bisa dinaikkan dengan kode.
3. **Target mencampur pesanan dan non-pesanan**, sementara model secara bisnis
   hanya bertanggung jawab atas yang kedua.
4. **Anggaran pencarian tidak setara** antar ketiga model.
5. **Satu seed, tanpa pengulangan** — khusus merugikan LSTM.
6. **Kedalaman LSTM tidak pernah diuji** (`num_layers=2`, `hidden_size=256`
   dibuang karena ongkos, bukan karena kalah).
7. **Fold 3 dan 5 ikut memilih hyperparameter**, jadi skor di sana bukan
   out-of-sample terhadap seleksi. Potongan fold 1/2/4 adalah angka yang bersih.
8. **Desember dinilai pada 49.717 dari 55.046 baris**, dan Desember adalah bulan
   atipikal (Natal/Tahun Baru).

---

## 21. Rencana kerja tersisa

| # | Pekerjaan | Status |
|---|---|---|
| 1 | Membekukan usulan §18 dalam sebuah commit | ⬜ menunggu persetujuan |
| 2 | Menjalankan protokol §19 — buka Desember sekali | ⬜ setelah #1 |
| 3 | Menulis `docs/hasil-test-desember.md` | ⬜ setelah #2 |
| 4 | **Dekomposisi harian** (`target_h1`…`target_h4`) untuk ketiga model — menjawab "kapan permintaan terkonsentrasi" | ⬜ direncanakan di spec pemodelan |
| 5 | **SHAP untuk pemenang saja** — menjawab "kenapa model meyakini ini" | ⬜ direncanakan di spec pemodelan |
| 6 | Mengisi `tanggal_buka` Cikarang Pusat di `outlet_closures.csv` + memperbarui `RELOCATION_DATES` | ⬜ menunggu pemilik data |
| 7 | Memperluas `calendar_features.py` ke 2026 sebelum data periode baru masuk | ⬜ |

Butir 4 dan 5 bukan tambahan opsional — keduanya sudah tertulis sebagai rencana
penjelasan (*explainability*) di
`docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md`, dengan
pembagian peran yang tegas: dekomposisi menjawab **kapan**, SHAP menjawab
**kenapa**. SHAP hanya dijalankan untuk pemenang, karena menjalankannya untuk
ketiganya berarti membayar ongkos penjelasan untuk model yang tidak akan dipakai.

---

## Rujukan

| Topik | Berkas |
|---|---|
| Prapemrosesan, formal untuk laporan | `docs/metodologi-preprocessing.md` |
| Prapemrosesan, naratif + trade-off | `docs/dokumentasi-preprocessing-id.md` |
| Pipeline ujung ke ujung (Inggris) | `docs/pipeline-overview.md` |
| Batasan yang tidak bisa dikode | `docs/batasan-penelitian.md` |
| Hasil terukur per model | `docs/hasil-modeling-{rf,xgb,lstm}.md` |
| Desain prapemrosesan pemodelan | `docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md` |
| Desain tiap model | `docs/superpowers/specs/2026-08-{18-random-forest,19-xgboost,19-lstm}-modeling-design.md` |
| Mesin evaluasi bersama | `utils/walk_forward.py`, `utils/model_common.py`, `utils/evaluation.py` |
