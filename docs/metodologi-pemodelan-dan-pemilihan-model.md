# Metodologi Pemodelan dan Strategi Pemilihan Model

Dokumen tunggal yang menjelaskan alur penelitian ini ujung ke ujung: dari
penggabungan lima berkas ekspor mentah, melewati prapemrosesan sampai data siap
model, lalu pembangunan tiga model, dan berakhir pada **strategi pemilihan model
terbaik yang direncanakan** beserta protokol pembukaan test set.

| Atribut            | Keterangan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Objek studi        | Jaringan gerai Kebuli Yaman — 59 cabang aktif di 16 kota                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Periode data       | 1 Januari 2024 – 31 Desember 2025 (731 hari kalender)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Unit analisis      | Pasangan (kode barang × cabang) per hari kalender                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Berkas siap model  | `dataset/model_ready/model_input.parquet` — 1.502.522 baris × 82 kolom                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Fitur              | 56 kolom (`modeling_prep.FEATURE_COLS`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Target             | `target_lead_time_cumulative`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Kandidat model     | Random Forest kuantil, XGBoost kuantil, LSTM kuantil                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Kriteria pemilihan | Rata-rata `pinball` lintas `QUANTILE_SET` (19 titik, 0,05–0,95) — direvisi 2026-08-24, sebelumnya `pinball@0.9` tunggal                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Status             | **Migrasi multi-kuantil: ketiga model dilatih ulang dan dievaluasi (RF 2026-08-25, XGBoost 2026-08-27, LSTM 2026-08-28).** Bagian 16 dan 18 ditulis ulang 2026-08-29 dengan angka K1; `crossing_rate` diuji hari yang sama — **XGBoost defek sungguhan** (butuh rearrangement kuantil), **LSTM derau numerik** (bukan keraguan). **Derau seed LSTM diverifikasi 2026-08-30**: walk-forward 5-fold seed 43 memberi K1 LSTM kalah 7,80% dari RF (bukan seri 1,09% seperti seed 42 yang dipakai sebelumnya) — **"K1 seri" ditarik**. **Rearrangement kuantil XGBoost dikerjakan 2026-08-30**: `crossing_rate` → 0, K1 membaik 1,08% tapi gap ke RF (2,13%) tetap di atas ambang 2%. **Prasyarat wajib terpenuhi, RF adalah kandidat terdepan di semua perbandingan** — bagian 18. **Pemenang belum resmi dibekukan** (menunggu persetujuan eksplisit pemilik proyek dalam sebuah commit, `docs/todolist-proyek.md` Fase E). Test set Desember 2025 masih terkunci |
| Artefak model      | `models/{random_forest,xgboost,lstm}_q90.joblib` — semuanya berlaku, dilatih pasca-reklasifikasi WIP-2 dan pasca-migrasi K1 (RF 2026-08-25, XGBoost 2026-08-27, LSTM 2026-08-28, `docs/todolist-proyek.md` butir 0c)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Verifikasi         | Lihat `docs/todolist-proyek.md` untuk jumlah test terkini — angka di sini tidak dijaga sinkron otomatis                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Tanggal dokumen    | 22 Agustus 2026 (dibuat) — bagian 16/17/18 terakhir ditulis ulang 2026-08-29                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

**Hubungan dengan dokumen lain.** Dokumen ini adalah lanjutan dari
`docs/preprocessing.md`. Untuk detail per-fungsi tiap tahap
prapemrosesan, rujuk Bagian 1 dokumen itu (formal, untuk laporan) atau
Bagian 2 (naratif, beserta trade-off tiap
keputusan). Bagian I di sini menulis ulang alur itu **pada tingkat strategi** —
apa yang dilakukan tiap tahap, keputusan apa yang diambil, dan kenapa — supaya
dokumen ini bisa dibaca berdiri sendiri, bukan supaya menggantikan keduanya.
Angka hasil terukur tiap model ada di `docs/hasil-modeling-{rf,xgb,lstm}.md`.
Batasan yang tidak bisa dihilangkan dengan kode ada di
`docs/batasan-penelitian.md`.

> **Migrasi evaluasi multi-kuantil (2026-08-24).** Kriteria perbandingan model
> di Bagian III berubah dari pinball loss pada satu titik kuantil (0,9) menjadi
> **rata-rata pinball loss lintas banyak titik kuantil sekaligus**. Metodologinya
> ada di
> `docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`;
> checklist penerapannya di
> `docs/superpowers/specs/2026-08-22-model-comparison-refactor-migration.md`.
>
> Bagian yang **sudah** direvisi di dokumen ini: bagian 15 (definisi metrik), bagian 17
> (definisi K1 dan K2), bagian 19 (angka utama protokol Desember), bagian 21 (rencana
> kerja). Bagian yang **belum**, karena menunggu ketiga notebook dijalankan
> ulang: bagian 16 (posisi hasil) dan bagian 18 (penerapan tangga) — keduanya diberi
> penanda eksplisit di tempatnya.
>
> Migrasi ini tidak membuang hasil out-of-sample apa pun: test set Desember
> belum pernah dibuka, jadi yang diulang hanya pencarian hyperparameter dan
> walk-forward di kelima fold latih.

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

| Tingkat | Keputusan yang diambil                              | Dinilai pada                                                                         | Status                     |
| ------- | --------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------- |
| **A**   | Hyperparameter terbaik **di dalam satu arsitektur** | Fold 3 & 5 (September, November 2025)                                                | ✅ selesai untuk ketiganya |
| **B**   | Arsitektur pemenang **antar tiga model**            | Fold 1, 2, 4 (Juli, Agustus, Oktober 2025) — potongan yang tidak menyentuh tingkat A | ⬜ inti dokumen ini        |
| **C**   | Seberapa baik pemenang bekerja                      | Desember 2025                                                                        | 🔒 belum dibuka            |

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

| Berkas           | Cakupan               |
| ---------------- | --------------------- |
| `jan-24.csv`     | Januari 2024          |
| `feb-24.csv`     | Februari 2024         |
| `mar-24.csv`     | Maret 2024            |
| `apr-des-24.csv` | April–Desember 2024   |
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

| Kawasan   | Hari pengiriman  |
| --------- | ---------------- |
| Kawasan 1 | Senin dan Kamis  |
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

Target ini dibangun dari **`Kuantitas` mentah, bukan yang di-cap** (lihat bagian 4.2):
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
_digabung paksa_ ke SKU lain lewat tabel `EXPLICIT_ITEM_RENAMES`. Setelah
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
di tahap 8 dibangun dengan `shift()`, yang mengasumsikan baris ke-_n_ adalah
hari ke-_n_. Satu tanggal bolong di tengah deret akan membuat "lag 7 hari"
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

| Komponen                                                | Dihitung dari                                  | Alasan                                                                          |
| ------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------- |
| Target **penilaian** (`target_lead_time_cumulative`)    | `Kuantitas` **mentah**                         | Lonjakan adalah permintaan nyata yang model harus dinilai terhadapnya           |
| Target **latih** (`target_lead_time_cumulative_capped`) | `Kuantitas_capped`                             | Porsi yang dipangkas adalah proksi pre-order, yang ditangani jalur manual (B-3) |
| Target harian (`target_h1..h7`)                         | `Kuantitas` **mentah**                         | Dipertahankan untuk dekomposisi penjelas, tidak dilatih di penelitian ini       |
| Fitur lag & rolling                                     | `Kuantitas_capped`                             | Satu hari ekstrem tidak boleh mendominasi input                                 |
| Statistik cabang                                        | `Kuantitas_capped`, **hanya periode training** | Dibekukan lalu diterapkan ke kedua split                                        |

**Dua target, dan itu disengaja (keputusan pemilik proyek, 2026-08-24).** Model
**dilatih** pada target capped dan **dinilai** pada target mentah. Baris ini
sebelumnya menyebut satu target saja; perumusan lama itu lebih tua daripada
konfirmasi pemilik data 2026-08-17 dan sudah tidak berlaku.

- **Kenapa latih di capped.** Kantor pusat sudah menangani pesanan lewat jalur
  manual, sehingga model dibutuhkan untuk permintaan _di luar_ pesanan (B-3).
  Komponen pre-order itu sendiri tidak dapat diprediksi dari data mana pun di
  proyek ini — buku pesanan tidak pernah terekam (B-1, B-2) — jadi melatihnya
  hanya menambah derau yang sebabnya tidak terlihat model. Baris yang dipangkas
  adalah proksi terdekat yang tersedia untuk komponen itu.
- **Kenapa nilai di mentah.** Permintaan yang dihadapi outlet adalah permintaan
  mentah. Kriteria yang dihitung pada deret yang sudah dipangkas punya sifat
  yang tidak boleh dimiliki kriteria pemilihan model: ia bisa diperbaiki dengan
  memangkas lebih banyak.
- **Ongkosnya dinyatakan, bukan disembunyikan.** Selisih kedua target di jendela
  Desember 2025 adalah 1.223 dari 50.692 baris (2,41%) dan 44.470 unit (2,03%
  massa permintaan). Ketiga model membayar ongkos itu sama besar, jadi peringkat
  antar model tidak terpengaruh — yang terpengaruh adalah level absolut K1, dan
  itu harus dibaca sebagai jarak terhadap permintaan nyata, bukan sebagai
  kegagalan model.

Pemisahannya dijaga di kode, bukan di niat: `modeling_prep.TRAIN_TARGET_COL` dan
`modeling_prep.EVAL_TARGET_COL` (tidak ada lagi nama `TARGET_COL` yang ambigu),
satu seam label latih di `model_common.train_target()` yang dipakai ketiga model,
dan `walk_forward.eligible_rows()` menolak panel yang kedua targetnya tidak
sepakat soal baris kosong.

Rinciannya:

- **`add_targets`** — `target_h1`…`target_h7`, yaitu `Kuantitas` mentah digeser
  1–7 hari ke depan, dikelompokkan per (pasangan, segmen). Karena
  `lead_time_days` tidak pernah melebihi 4, `target_h5`–`target_h7` dibuang
  pipeline ini. Target harian ini tidak dilatih di penelitian ini; ia
  dipertahankan dan divalidasi untuk dekomposisi penjelas (bagian 21).
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
- **`add_lead_time_target`** — target utama (bagian 1.2).
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

**Purging** (`utils/modelling/purging.py`) — baris training yang jendela lead-time-nya
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

`utils/modelling/modeling_prep.py` menambahkan lima hal terakhir sebelum data menyentuh
model, menghasilkan `model_input.parquet` (1.502.522 baris × 82 kolom).

### 7.1 `is_event_driven`

Penanda per-SKU dari `dataset/event_driven_items.csv` — barang yang
permintaannya digerakkan acara.

### 7.2 `demand_segment` — klasifikasi Syntetos-Boylan

Setiap pasangan diklasifikasikan dari dua besaran yang dihitung
**hanya dari periode training**:

- **ADI** — rata-rata interval antar hari berpermintaan (ambang 1,32)
- **CV²** — kuadrat koefisien variasi kuantitas bukan-nol (ambang 0,49)

|                | CV² < 0,49     | CV² ≥ 0,49 |
| -------------- | -------------- | ---------- |
| **ADI < 1,32** | `smooth`       | `erratic`  |
| **ADI ≥ 1,32** | `intermittent` | `lumpy`    |

Segmentasi ini bukan fitur hiasan — ia **sumbu pelaporan wajib**. Dengan 44%
target bernilai nol, satu angka metrik global bisa menobatkan model yang hanya
unggul di tempat menebak nol itu mudah.

### 7.3 `fold_id` — lima fold walk-forward jendela mengembang

```
FOLD_STARTS = [2025-07-01, 2025-08-01, 2025-09-01, 2025-10-01, 2025-11-01]
```

Training untuk fold _k_ adalah setiap baris bertanggal sebelum
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
dikecualikan secara sengaja (bagian 4.2).

### 7.7 Dua adapter, satu kontrak

| Adapter          | Untuk                  | Bentuk keluaran                                               |
| ---------------- | ---------------------- | ------------------------------------------------------------- |
| `to_tabular()`   | XGBoost, Random Forest | Matriks baris × 56 fitur                                      |
| `to_sequences()` | LSTM                   | Jendela 28 hari yang berakhir **di baris prediksi, inklusif** |

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

| #   | Aturan                                                            | Diterapkan di                                         | Yang dicegah                                                      |
| --- | ----------------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------- |
| 1   | Setiap fitur historis berhenti di **H-1**                         | lag, rolling (digeser 1 hari)                         | Nilai hari ini bocor ke prediktornya sendiri                      |
| 2   | Setiap target ketat **ke depan**                                  | `add_targets`, `add_lead_time_target`                 | Target memuat hari yang sudah teramati                            |
| 3   | Statistik agregat hanya dari **periode training**, lalu dibekukan | statistik cabang, baseline pencilan, `demand_segment` | Informasi masa depan mengalir ke fitur                            |
| 4   | **Purge** di setiap batas                                         | batas Desember, batas tiap fold, ekor early stopping  | Label yang sebagian tersusun dari periode penilaian               |
| 5   | Tidak ada `shift` yang **menyeberangi segmen**                    | `segment_id` dioper ke semua fungsi berbasis shift    | Lag menjembatani masa cabang tutup                                |
| 6   | Kolom turunan hari-ini **dikeluarkan dari fitur**                 | `baseline_ratio`, `is_spike`                          | Dua arti berbeda untuk "diketahui saat prediksi" dalam satu baris |

---

# Bagian II — Pembangunan Tiga Model

## 9. Kontrak bersama: satu mesin evaluasi, tiga model disuntikkan

Ini keputusan arsitektur paling penting di seluruh fase pemodelan, dan
alasannya metodologis, bukan teknis.

Sebuah perbandingan hanya layak dilaporkan bila ketiga model melihat **baris yang
sama persis**. Itu bukan sesuatu yang bisa dijamin oleh kedisiplinan penulisan
tiga skrip training terpisah. Maka jaminan itu dipindahkan ke struktur:

> `utils/modelling/walk_forward.py` **memiliki** definisi kelayakan baris, batas fold, dan
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

### 9.2 Mesin bersama: `utils/modelling/model_common.py`

Bagian yang tidak dimiliki model mana pun secara khusus dikumpulkan di sini —
karena membiarkannya di dalam modul Random Forest berarti memperbaiki bug
checkpoint yang sama dua kali, lalu tiga kali saat LSTM datang:

| Komponen                          | Fungsi                                                                                                                              |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `sample_search_space()`           | Penarikan acak kandidat, dengan penyaring keterjangkauan yang **disuntikkan** (batas memori daun RF tidak punya padanan di XGBoost) |
| `run_search()`                    | Menilai tiap kandidat di fold pencarian, menulis **checkpoint tiap kandidat selesai**, melanjutkan dari sana bila dijalankan ulang  |
| `select_best()`                   | **Kandidat dengan pinball gabungan terendah.** Satu baris yang menentukan segalanya                                                 |
| `expand_one_hot()`                | Ekspansi kategorikal bersama                                                                                                        |
| `split_early_stopping()`          | Ekor 30 hari terakhir jendela training, **dengan purge yang sama** seperti di batas fold                                            |
| `save_bundle()` / `load_bundle()` | Format bundel yang mendeskripsikan dirinya sendiri                                                                                  |

Kriteria seleksi tidak berpindah-pindah:

```python
best_id = int(scored.loc[scored["pinball"].idxmin(), "candidate_id"])
```

### 9.3 Protokol pencarian yang sama untuk ketiganya

| Aspek          | Nilai                                                                   |
| -------------- | ----------------------------------------------------------------------- |
| Fold pencarian | **Fold 3 (September) dan 5 (November) saja**                            |
| Kriteria       | pinball@0.9 gabungan, **dibobot jumlah baris**, bukan dirata-rata polos |
| Subsampling    | **Tidak ada** — seluruh baris training tiap fold dipakai                |
| Seed           | 42                                                                      |

Dua fold, bukan lima, karena pencarian harus murah: lima fold × puluhan kandidat
tidak muat di plafon waktu mana pun. Konsekuensinya — skor fold 3 dan 5 bukan
out-of-sample terhadap seleksi — ditangani secara eksplisit di bagian 16.

---

## 10. Lantai: tiga baseline naif

Skor model tidak berarti apa-apa berdiri sendiri. Sebelum model apa pun dilatih,
lantainya ditetapkan lewat `evaluation.NAIVE_BASELINES`:

| Baseline            | Prediksi                       |
| ------------------- | ------------------------------ |
| `naive_zero`        | 0                              |
| `naive_lag_1`       | `lag_1 × lead_time_days`       |
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

|                  |                                                 |
| ---------------- | ----------------------------------------------- |
| Implementasi     | `quantile_forest.RandomForestQuantileRegressor` |
| Modul            | `utils/modelling/model_random_forest.py`        |
| Ruang pencarian  | 1.152 kombinasi                                 |
| Kandidat ditarik | **18**                                          |

### 11.1 Kenapa `quantile-forest`, bukan `RandomForestRegressor`

`RandomForestRegressor` milik sklearn hanya meminimalkan galat kuadrat atau
absolut — ia **tidak punya** loss kuantil. Quantile regression forest
(Meinshausen 2006) membaca kuantil 0,9 dari distribusi empiris yang disimpan di
tiap daun, dan itulah estimator yang benar untuk service level yang sudah dikunci.

### 11.2 Benchmark sebelum membakar jam

**Angka run K1 (2026-08-25),** `docs/hasil-modeling-rf.md` bagian 3. Satu fit
pada training set penuh fold 5 dijalankan lebih dulu dengan `DEFAULT_PARAMS`,
untuk memastikan batas penyimpanan daun yang dipakai menyaring kandidat
memang berlaku **sebelum** 18 fit dijalankan:

|                                       |                       |
| ------------------------------------- | --------------------- |
| Baris training fold 5                 | 1.292.778             |
| Estimasi penyimpanan daun             | 1,54 GB (budget 3 GB) |
| Wall time satu fit + predict 19 titik | 9,7 menit             |

19 titik kuantil memakan **×1,47** dibanding satu titik (6,6 → 9,7 menit) —
pengganda paling murah dari ketiga model, karena seluruh titik dibaca dari
daun yang sama; membangun pohonnya yang mahal, bukan membaca persentilnya
(bandingkan XGBoost, ×15,2, bagian 12.2).

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

Ruang dan seed identik dengan run kuantil-tunggal, tapi **pemenangnya
berubah**: kandidat 1 (`max_depth=20`, `max_features=1.0`,
`min_samples_leaf=20`, `max_samples_leaf=1`, `one_hot=False`,
`log_target=False`, `n_estimators=200`) — bukan lagi kandidat 17
(`max_depth=12`, `max_features=0.5`, `max_samples_leaf=50`, `one_hot=True`)
seperti di run lama. Kedua kandidat itu terpisah **0,0004** di kriteria lama
(praktis seri) tapi **0,0177** di K1 — merata-ratakan 19 titik meredam derau
yang tadinya menyembunyikan selisih ini (`docs/hasil-modeling-rf.md`
bagian 4.3, termasuk Spearman ρ = 0,975 terhadap peringkat lama).

Dua bacaan dari sebarannya (K1, `docs/hasil-modeling-rf.md` bagian 4.2):

1. **Ruangnya datar.** Dari 2,8808 ke 3,1969 hanya rentang 11%, dan lima
   kandidat teratas berjarak 1,46% satu sama lain.
2. **`max_features="sqrt"` satu-satunya pilihan yang benar-benar merugikan.**
   Dua kandidat terburuk keduanya memakainya. Dengan 56 fitur, `sqrt` menyisakan
   ~7 fitur per split — terlalu sedikit.

### 11.4 Model final

`fit_final()` melatih ulang konfigurasi pemenang dengan **`n_estimators`
dinaikkan 200 → 400** pada 1.349.011 baris layak sebelum Desember. Bundelnya
menyimpan urutan kolom training beserta flag `one_hot`/`log_target` — forest yang
dimuat ulang dengan urutan kolom berbeda tidak gagal, ia memprediksi dengan
percaya diri dari fitur yang salah.

`models/random_forest_q90.joblib` — **826 MB** (25 Agu 2026, 19 titik
kuantil; naik dari 821 MB di run kuantil-tunggal — tiap titik tambahan
menyimpan array nilai daun tambahan).

---

## 12. Model 2 — XGBoost kuantil

|                  |                                                                                                                                                    |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Implementasi     | `xgboost==2.1.4`, `XGBRegressor(objective="reg:quantileerror", multi_strategy="multi_output_tree", tree_method="hist")`, 19 titik `QUANTILE_SET_A` |
| Modul            | `utils/modelling/model_xgboost.py`                                                                                                                 |
| Ruang pencarian  | 2.592 kombinasi                                                                                                                                    |
| Kandidat ditarik | **30**                                                                                                                                             |

`quantile_alpha=0.9` tunggal (run lama) diganti `multi_strategy` dengan 19
titik keluaran sejak migrasi K1 — konsekuensinya di bagian 12.2.

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

**Benchmark K1 (CPU Mac): 265,2 menit untuk dua fit penuh + predict 19
titik** (`docs/hasil-modeling-xgb.md` bagian 3) — jauh dari 2,4 menit di run
kuantil-tunggal. `multi_strategy` membangun **satu pohon per titik kuantil
per ronde boosting**, jadi 19 titik bukan ongkos tipis seperti di RF:
pengganda terukur **×15,2** (bandingkan RF ×1,47, bagian 11.2). Inilah
satu-satunya alasan tahap pencarian (bukan walk-forward/fit final) dipindah
ke GPU Windows sejak 2026-08-26 — lihat bagian mesin-terpisah di
`docs/hasil-modeling-xgb.md`.

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

Ruang dan seed identik dengan run kuantil-tunggal (30 kandidat, `commit`
berbeda karena dijalankan di GPU — lihat bagian mesin-terpisah). **Pemenangnya
berubah**: kandidat 17 (`encoding="native"`, `max_depth=10`,
`learning_rate=0.05`, `min_child_weight=1`, `subsample=0.7`,
`colsample_bytree=0.5`, `reg_lambda=10.0`, `log_target=False`) — bukan lagi
kandidat 11 (`max_depth=6`, `min_child_weight=50`, `colsample_bytree=0.7`)
seperti run lama.

Tiga bacaan dari sebarannya (K1, `docs/hasil-modeling-xgb.md` bagian 4.2):

1. **`max_depth` penentu utama, dan monoton** — rata-rata K1 per kedalaman:
   4 → 3,0344, 6 → 2,9483, 8 → 2,9153, 10 → 2,8945. Rentang keseluruhan 6,8%,
   tapi seluruh 9 kandidat `max_depth=4` ada di separuh bawah tabel. **Beda
   dari run lama**, di mana kedalaman tidak sejelas ini sebagai penentu —
   dengan 19 pohon dibangun per ronde, kapasitas per titik kuantil ternyata
   lebih menuntut kedalaman.
2. **`encoding="native"` tetap menang**, tapi bedanya menyempit —
   rata-rata `native` 2,9557 vs `one_hot` 2,9448 vs `ordinal` 2,9743 — encoding
   jauh lebih kecil pengaruhnya daripada `max_depth`.
3. **Peringkat kandidat sendiri jauh kurang stabil terhadap K1** daripada RF:
   Spearman ρ = 0,73 (RF: 0,975) terhadap peringkat pinball@0,9 lama,
   walau seed dan ruang identik — konsisten dengan hipotesis lama di bagian
   ini bahwa `multi_strategy` berinteraksi dengan jumlah titik kuantil.

`best_iteration` per fold di walk-forward final: 242–390 — jauh di bawah
plafon 2.000 yang nyaris tersentuh `DEFAULT_PARAMS` di benchmark (bagian
12.2), bukti early stopping bekerja pada konfigurasi pemenang.

### 12.4 Model final

1.349.011 baris training, `best_iteration = 201`.
`models/xgboost_q90.joblib` — **292 MB** (27 Agu 2026; naik dari 4,7 MB di
run kuantil-tunggal — 19 pohon per ronde boosting, bukan satu).

Bundelnya menyimpan urutan kolom training, flag `encoding`/`log_target`, dan
**level kategori** yang dipakai saat training. Yang terakhir wajib untuk mode
`native`: booster yang dimuat ulang bulan depan terhadap kategori yang diurutkan
berbeda tidak gagal — ia memprediksi dengan percaya diri dari fitur yang salah.

**Temuan konstruksi yang belum dijelaskan:** `crossing_rate` di walk-forward
final = **97,7%** — nyaris seluruh baris punya sedikitnya satu pasang titik
kuantil yang urutannya terbalik. `multi_strategy` tidak punya jaminan
monotonicity struktural antar titik seperti daun RF; apakah ini defek
arsitektural atau artefak definisi crossing pada target yang 99,55% bilangan
bulat masih terbuka — dibahas penuh di `docs/hasil-modeling-xgb.md`
bagian 5.2 dan menggerakkan status "keputusan ditahan" di bagian 18.

---

## 13. Model 3 — LSTM kuantil

|                  |                                                                                                                                                            |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Implementasi     | `torch==2.8.0`, `QuantileLSTM`, kepala 19-titik `QUANTILE_SET_A`                                                                                           |
| Modul            | `utils/modelling/model_lstm.py`, `utils/modelling/sequence_windows.py`                                                                                     |
| Ruang pencarian  | **144 kombinasi** (dipulihkan dari 48 — dua dimensi kapasitas, `num_layers` dan `hidden_size`, dikembalikan ke `SEARCH_SPACE` sejak migrasi K1, bagian 21) |
| Kandidat ditarik | **30** (dipatok setara XGBoost, dari 12)                                                                                                                   |

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
   dihapus warm-up dan purge fold. Membaca _fitur_ baris itu aman: setiap jendela
   berakhir di baris prediksinya sendiri dan setiap lag berhenti di H-1, jadi
   tidak ada nilai target yang bisa masuk jendela. Yang dicegah purging adalah
   training atas _label_ baris tersebut, dan itu tetap tidak pernah terjadi.

Protokol dua fitnya identik dengan XGBoost, karena masalahnya identik: jumlah
epoch adalah keputusan kapasitas. `MAX_EPOCHS = 100`,
`EARLY_STOPPING_EPOCHS = 5`.

### 13.3 Benchmark, dan ruang pencarian yang sempat dipotong lalu dipulihkan

Ini bagian paling instruktif dari ketiga model, dan layak ditulis apa adanya
di laporan — termasuk bagian di mana keputusan awal ternyata terbalik.

**Angka run K1 (CPU Mac, 2026-08-25),** `docs/hasil-modeling-lstm.md` bagian 3:

|                                           |                                                               |
| ----------------------------------------- | ------------------------------------------------------------- |
| Device terpilih                           | **CPU**                                                       |
| `best_epoch` benchmark                    | 2                                                             |
| `sec_per_epoch`                           | 106,8                                                         |
| Wall time satu putaran + predict 19 titik | 16,0 menit                                                    |
| Peak RSS                                  | 5,93 GB                                                       |
| `N` menurut formula plafon 8 jam          | 14 (informatif saja)                                          |
| `N` yang benar-benar dipakai              | **30** — dipatok setara XGBoost, bukan diturunkan dari plafon |

1. **CPU menang atas MPS, 2×.** Probe 15 batch di fold 5 (2026-08-19, tidak
   diulang di run ini karena kesimpulannya tidak berubah): 0,392 s/batch di
   MPS lawan 0,193 s/batch di CPU — MPS tidak punya kernel LSTM ter-fusi di
   ukuran hidden ini.
2. **Ruang pencarian sempat dikecilkan (2026-08-19), lalu dipulihkan penuh
   untuk migrasi K1 (2026-08-24) — dan kombinasi yang tadinya dibuang justru
   jadi pemenang.** Ongkos per epoch yang dulu mendasari pemotongan:

   | hidden | layers | batch | s/epoch |
   | -----: | -----: | ----: | ------: |
   |     64 |      1 |  1024 |      75 |
   |     64 |      1 |  2048 |      47 |
   |    128 |      1 |  1024 |     104 |
   |    128 |      1 |  2048 |      83 |
   |    128 |  **2** |  1024 | **259** |

   Waktu itu `num_layers=2` dan `hidden_size=256` dibuang dari `SEARCH_SPACE`
   karena mahal, dan konsekuensinya dicatat eksplisit: _"pencarian ini tidak
   pernah menanyakan apakah lapisan kedua akan menolong."_ Keputusan pemilik
   proyek 2026-08-24 (bagian 21) membalikkan ini demi validitas atribusi —
   kalau LSTM kalah di K1, ketimpangan anggaran tidak boleh jadi alasan yang
   tidak bisa disingkirkan. Hasilnya (bagian 13.4): **pemenang pencarian K1
   justru memakai `hidden_size=256, num_layers=2`** — kombinasi paling mahal
   yang dulu dipotong. Pertanyaan yang "tidak pernah ditanyakan" itu sekarang
   punya jawaban, dan jawabannya "ya, lapisan kedua menolong."

3. **Jendela padat tidak pernah dimaterialisasi.** Tensor 1.502.522 × 28 × 56
   tidak muat di memori; `sequence_windows` membangun indeks dan mengambil jendela
   saat dibutuhkan.

### 13.4 Ruang pencarian dan pemenangnya

```python
SEARCH_SPACE = {
    "hidden_size":   [64, 128, 256],
    "num_layers":    [1, 2],
    "dropout":       [0.0, 0.2, 0.3],
    "learning_rate": [3e-4, 1e-3],
    "batch_size":    [1024, 2048],
    "log_target":    [False, True],
}
```

3×2×3×2×2×2 = 144 kombinasi. Pencarian **30 kandidat, di GPU Windows** sejak
2026-08-26 (bagian mesin-terpisah, `docs/hasil-modeling-lstm.md`).

Parameter terpilih: `hidden_size=256`, `num_layers=2`, `dropout=0.0`,
`learning_rate=3e-4`, `batch_size=2048`, `log_target=True`, `grad_clip=1.0`.
**Tidak ada satu pun nilai yang sama** dengan pemenang run kuantil-tunggal
lama (`hidden_size=128`, `num_layers=1`, `dropout=0.3`, `batch_size=1024`,
`log_target=False`) — perbandingan id-per-id seperti RF/XGBoost tidak berlaku
di sini karena ruang pencariannya sendiri berbeda, bukan cuma kriterianya
(`docs/hasil-modeling-lstm.md` bagian 4.3).

Lima bacaan dari sebarannya (K1, `docs/hasil-modeling-lstm.md` bagian 4.2):

1. **Ruang parameternya jauh lebih curam daripada RF/XGBoost.** Rentang K1
   20,8% — lebih dari 3× rentang XGBoost (6,8%) dan hampir 2× RF (11%). Lima
   kandidat teratas berjarak 2,25% — juga terlebar dari ketiganya.
2. **`dropout=0` konsisten terbaik** (rata-rata K1 2,9607 lawan 3,0796 di
   `dropout=0,2` dan 3,1668 di `0,3`) — regularisasi tambahan merugikan di
   sini, beda arah dari run lama yang justru memenangkan `dropout=0,3`.
3. **`hidden_size` monoton**: 64 → 3,0785, 128 → 3,0467, 256 → 2,9964 — lebih
   besar lebih baik pada rentang yang diuji, alasan langsung kenapa memulihkan
   dimensi ini penting (poin 2, bagian 13.3).
4. **`log_target=False` menang secara agregat** (2,9876 lawan 3,1199 untuk
   `True`) — tapi pemenangnya sendiri memakai `log_target=True`. Bukan
   kontradiksi: interaksi antar parameter lebih kental daripada efek utama
   tunggal di LSTM, konsisten dengan rentang yang jauh lebih lebar di poin 1.
5. **`batch_size=2048`** — di run lama selalu kalah dari 1024; di run K1
   justru dipakai pemenang. Sinyal tambahan bahwa interaksi antar
   hyperparameter, bukan efek satu-per-satu, yang mendominasi di LSTM.

### 13.5 Model final

1.349.011 baris training, `best_epoch = 5`, wall time **~78 menit** (CPU Mac
— tahap tunggal termahal dari ketiga model gabungan dengan walk-forward-nya,
lihat bagian 14).
`models/lstm_q90.joblib` — **3,7 MB** (28 Agu 2026, 19 titik kuantil; naik
dari 466 KB di run kuantil-tunggal, tapi tetap jauh lebih kecil dari RF
826 MB dan XGBoost 292 MB — bobot jaringan jauh lebih ringkas daripada
struktur pohon tersimpan).

Bundelnya menyimpan `state_dict`, urutan kolom (dinamis dan kategorikal
terpisah), ukuran embedding, **scaler**, flag `log_target`, `lookback`, dan
`best_epoch`. Scaler wajib ikut: jaringan yang dimuat ulang lalu diberi fitur
berskala mentah tidak gagal — ia memprediksi dengan percaya diri dari input yang
salah skala.

Satu perbedaan pemakaian dibanding dua model lain: `predict_bundle()`
**mewajibkan panel**, bukan menerimanya sebagai opsi. LSTM tidak bisa memprediksi
dari satu baris sendirian — ia butuh 28 hari di belakangnya.

**Temuan konstruksi yang belum dijelaskan:** `crossing_rate` di walk-forward
final = **43,4%** — di antara RF (0%, struktural) dan XGBoost (97,7%, bagian
12.4). Kepala keluaran 19-titik LSTM juga tidak punya jaminan monotonicity
struktural, sama seperti `multi_strategy` XGBoost. Dibahas penuh di
`docs/hasil-modeling-lstm.md` bagian 5.2.

### 13.6 Anggaran waktu: dari plafon yang terlampaui ke anggaran yang dipatok

Di run kuantil-tunggal, anggaran diturunkan dari `best_epoch = 3` yang
terukur di benchmark, sementara kandidat sebenarnya berhenti di epoch 3–13,
sehingga ongkos riilnya ~11,4 jam terhadap plafon 8 jam. `candidate_budget()`
bekerja sesuai rumusnya; yang salah adalah asumsi bahwa `best_epoch`
benchmark mewakili ruang pencarian.

**Migrasi K1 mengganti mekanismenya**, bukan memperbaiki rumusnya: alih-alih
menurunkan `N` dari plafon waktu, `N` **dipatok 30** — setara XGBoost, terlepas
dari berapa pun hasil formula (`N` menurut plafon = 14, hanya dicetak
informatif, bagian 13.3). Ini konsisten dengan alasan pemulihan `SEARCH_SPACE`
di atas: anggaran yang setara adalah syarat supaya kekalahan/kemenangan LSTM
bisa diatribusikan ke arsitektur, bukan ke seberapa dalam ruang parameternya
sempat digali. Wall-clock pencarian sebenarnya (di GPU Windows) tidak
sebanding dengan plafon 8 jam yang dihitung untuk CPU — lihat bagian
mesin-terpisah di `docs/hasil-modeling-lstm.md` bagian 7.

### 13.7 Pengulangan 3 seed pada pemenang

LSTM satu-satunya dari ketiga model yang inisialisasi bobotnya **acak**
(forest dan boosting-nya deterministik pada `random_state` tetap). Konfigurasi
pemenang (bagian 13.4) diulang pada seed 42/43/44, dinilai di fold pencarian
yang sama (3 dan 5) — bukan walk-forward 5-fold penuh, karena mengulang 5-fold
untuk 3 seed terlalu mahal untuk anggaran saat ini.

| seed | K1 (fold 3&5) |
| ---: | ------------: |
|   44 |        2,8399 |
|   42 |        2,8617 |
|   43 |        3,0915 |

Rentang **0,2517** — jauh lebih lebar daripada jarak K1 LSTM ke Random Forest
di potongan bersih (0,0310, bagian 16.5/18). Seluruh angka K1 LSTM di bagian
16/18 berasal dari **seed 42 saja**; rincian dan konsekuensinya untuk
keputusan pemenang ada di bagian 18 K1 dan `docs/hasil-modeling-lstm.md`
bagian 5.1b.

---

## 14. Ringkasan banding konstruksi

**Diperbarui 2026-08-29** dengan angka dari run multi-kuantil selesai
(`docs/hasil-modeling-{rf,xgb,lstm}.md`) — menggantikan tabel kuantil-0,9
tunggal yang berlaku sampai butir 0d selesai.

|                                             | Random Forest                                | XGBoost                                 | LSTM                                                      |
| ------------------------------------------- | -------------------------------------------- | --------------------------------------- | --------------------------------------------------------- |
| Mekanisme kuantil                           | Kuantil empiris dibaca dari daun             | `reg:quantileerror`, `multi_strategy`   | Pinball loss langsung, kepala 19-titik                    |
| Objektif = kriteria?                        | Tidak (dibaca, bukan dioptimalkan)           | **Ya**                                  | **Ya**                                                    |
| Kapasitas ditentukan                        | `n_estimators` dipatok (200 cari, 400 final) | Early stopping per fold (242–390 ronde) | Early stopping per fold (5–11 epoch)                      |
| Protokol fit                                | Satu fit                                     | **Dua fit** (early stop + refit)        | Satu putaran per fold, early stopping internal            |
| Ruang pencarian                             | 1.152                                        | 2.592                                   | **144** (dua dimensi kapasitas dikembalikan)              |
| Kandidat                                    | 18                                           | 30                                      | **30** (setara XGBoost)                                   |
| Penyaring keterjangkauan                    | Batas memori daun 3 GB                       | Tidak perlu                             | Anggaran waktu                                            |
| Bergantung seed acak                        | Tidak                                        | Tidak diverifikasi                      | **Ya** — rentang K1 **0,2517** antar 3 seed (bagian 16.5) |
| Benchmark satu putaran                      | 9,7 menit (1 fit)                            | 265,2 menit (2 fit)                     | 16,0 menit                                                |
| Device pencarian                            | CPU Mac                                      | **GPU Windows** (sejak 2026-08-26)      | **GPU Windows**                                           |
| Wall time walk-forward (CPU Mac, sebanding) | ~45 menit                                    | ~3 jam 1 menit                          | **~8 jam 28 menit**                                       |
| Wall time model final (CPU Mac)             | ~48 menit                                    | ~23 menit                               | ~1 jam 18 menit                                           |
| Ukuran artefak                              | **826 MB**                                   | **292 MB**                              | **3,7 MB**                                                |
| Butuh panel saat prediksi                   | Tidak                                        | Tidak                                   | **Ya** (jendela sekuens)                                  |
| Dependensi tambahan                         | `quantile-forest`                            | `xgboost` + `libomp`                    | `torch`                                                   |

Baris **kandidat** dan **ruang pencarian** LSTM sudah setara XGBoost sejak
migrasi K1 (12→30 kandidat, 48→144 ruang, keputusan 2026-08-24 — alasan dan
konsekuensi ongkosnya di bagian 21). Yang **belum** setara: pengulangan seed
(hanya LSTM yang diulang, 3×, karena hanya LSTM yang inisialisasinya acak) dan
device pencarian (XGBoost/LSTM pindah ke GPU Windows 2026-08-26, RF tetap CPU
Mac — bagian mesin-terpisah di `docs/hasil-modeling-xgb.md` bagian 7). Baris
wall-time walk-forward/model final di atas **sebanding lintas ketiganya**
karena ketiganya dijalankan di CPU Mac yang sama, terlepas dari device
pencariannya masing-masing.

---

# Bagian III — Strategi Komparasi dan Pemilihan Model

## 15. Metrik dan justifikasinya

### 15.1 Kriteria tunggal: rata-rata `pinball` lintas `QUANTILE_SET`

**Direvisi 2026-08-24.** Sampai tanggal itu kriterianya adalah `pinball@0.9`
tunggal. Perumusan lamanya dipertahankan di 15.1.1 di bawah, karena empat
alasan yang menopangnya tidak dibatalkan — hanya diperluas.

```python
delta = actual - predicted
def pinball(alpha):
    return np.where(delta >= 0, alpha * delta, (alpha - 1.0) * delta).mean()

K1 = np.mean([pinball(tau) for tau in QUANTILE_SET])
```

Pada setiap titik τ, kekurangan dikali **τ** dan kelebihan dikali **1 − τ**.
Rata-ratanya **tak berbobot**: setiap titik kuantil menyumbang sama besar.

`actual` di rumus di atas adalah **`target_lead_time_cumulative` mentah**, bukan
varian capped yang dipakai untuk melatih (bagian 5). K1 mengukur jarak terhadap
permintaan yang benar-benar dihadapi outlet.

`QUANTILE_SET` saat ini di **Tahap A**: 19 titik merata
`[0,05, 0,10, …, 0,90, 0,95]`. **Kerapatan 19 titik dipertahankan** (keputusan
pemilik proyek, 2026-08-24) meskipun memperkecilnya ke 9 titik akan menghemat
~38 jam komputasi di XGBoost: kerapatan grid adalah alasan rata-rata pinball
boleh dibaca sebagai hampiran CRPS, sehingga memangkasnya melemahkan justifikasi
kriteria utama itu sendiri — bukan sekadar menurunkan presisinya. Ia berpindah otomatis ke Tahap B — grid yang
diturunkan dari sebaran critical ratio aktual — begitu B-10 mencapai ambang ≥80%
volume dengan data biaya presisi. Definisi lengkap kedua tahap dan mekanisme
peralihannya ada di
`docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md` Bagian 1.
K1 ditulis generik terhadap ukuran grid, jadi peralihan itu tidak menuntut
perubahan rumus di atas.

**Kenapa berpindah dari satu titik ke banyak titik — tiga alasan:**

**(a) Peringkat model terbukti berpindah tergantung titik evaluasinya.** Model
yang unggul di pinball@0,9 tidak dijamin unggul di kuantil lain (Serafin et al.,
2024). Pada data ini alasannya bukan teoretis: bagian 18 mencatat ketiga model **seri
dalam 0,88%** pada satu titik kuantil. Peringkat yang tidak terpisahkan di satu
titik adalah justru situasi di mana memperluas titik evaluasi paling mungkin
memisahkan — dan kalau ternyata tetap seri lintas 19 titik, itu temuan yang jauh
lebih kuat daripada seri di satu titik.

**(b) Kalibrasi baik di satu titik bisa kebetulan.** Menilai model hanya di 0,9
berisiko menangkap kalibrasi yang benar di titik itu saja tanpa menjamin
kalibrasi di seluruh distribusi (Gneiting & Resin, 2022). bagian 16 sudah memberi
petunjuknya: coverage ketiga model di 0,9 berjarak 0,002–0,034 dari target, tapi
tidak ada satu angka pun yang memberitahu apa yang terjadi di ekor bawah.

**(c) Karena standar bidangnya memang begitu.** Kompetisi M5 Uncertainty —
patokan forecasting demand ritel skala besar, 42.840 deret Walmart — menilai
model pada **sembilan** titik kuantil sekaligus lewat pinball loss terskala yang
dirata-ratakan, bukan satu titik (Makridakis et al., 2021). Rata-rata pinball
pada grid yang cukup padat juga mendekati CRPS (Bröcker, 2012), sehingga K1
berperilaku seperti proper scoring rule untuk seluruh distribusi ramalan, bukan
untuk satu persentilnya.

**Ongkos komputasinya sengaja dikesampingkan.** Tujuan proyek ini menemukan model
terbaik, bukan model termurah pada tahap penelitian. Ongkos tetap dilaporkan,
tapi tempatnya di K3 (bagian 17) sebagai tie-breaker, bukan sebagai alasan mempersempit
kriteria utama.

#### 15.1.1 Kenapa 0,9 tetap penting — dan di mana tempatnya sekarang

Kuantil 0,9 **tidak dicabut**; ia berpindah peran. Ia tetap **komitmen bisnis**
yang mengatur apa yang dikirim ke outlet, dan berhenti menjadi **satu-satunya
titik tempat model dibandingkan**. Empat alasan yang menopang pilihan 0,9 masih
berlaku persis apa adanya:

**(a) Karena service level-nya 0,9, dan itu keputusan bisnis yang sudah dikunci.**
Dikonfirmasi pemilik data 2026-08-16, **seragam untuk setiap SKU** — pemisahan
per-kategori (FG vs Packaging) secara eksplisit ditolak, karena kantor pusat
mengirim semua barang dalam satu konsinyasi sehingga satu service level mengatur
seluruh pengiriman. Angka 0,9 adalah pernyataan bisnis: _kehabisan stok sembilan
kali lebih mahal daripada kelebihan stok dalam jumlah yang sama._ Pinball@0.9
adalah terjemahan matematis langsung dari kalimat itu.

> **Klarifikasi pemilik data 2026-08-22** (B-9 `docs/batasan-penelitian.md`):
> "seragam untuk semua item" dimaksudkan sebagai **komitmen agregat di level
> pengiriman**, bukan larangan variasi teknis per item. Janji 0,9 ke outlet tetap
> utuh; yang boleh bervariasi adalah kuantil input per segmen yang dipakai
> mencapainya, selama rata-rata tertimbangnya kembali ke 0,9. Itulah yang membuat
> menilai model di luar titik 0,9 bukan hanya sah secara metodologis, tapi
> **diperlukan**: alokasi tersegmentasi
> (`docs/superpowers/specs/2026-08-22-segmented-quantile-allocation-design.md`)
> akan benar-benar membaca kuantil selain 0,9 dari model produksi, jadi model itu
> harus terbukti kalibrasinya di sana juga.

**(b) Karena meramal rata-rata berarti kehabisan stok separuh waktu.** Model yang
meminimalkan MSE meramal _mean_; yang meminimalkan MAE meramal _median_.
Keduanya, secara definisi, terlampaui permintaan aktual sekitar separuh hari.
Coverage baseline titik-tengah yang terukur — **0,61** — adalah bukti empirisnya.

**(c) Karena yang dilatih dan yang dinilai harus fungsi yang sama.** Ini tetap
terpenuhi setelah migrasi, dan untuk ketiga model sekaligus, bukan dua dari tiga
seperti sebelumnya: XGBoost melatih `quantile_alpha=QUANTILE_SET`, LSTM
menjumlahkan pinball lintas kuantil di head-nya, dan Random Forest membaca
seluruh titik dari distribusi empiris daun yang sama. Objektif latih dan
kriteria pemilihan tetap fungsi yang sama.

**(d) Karena kriterianya tidak berpindah.** `select_best()` memakai K1 yang sama
sejak tahap pencarian sampai pemilihan akhir. Yang berubah sekali, dan dicatat
di sini, adalah definisi K1 itu sendiri — dan itu terjadi **sebelum** test set
Desember dibuka, sehingga tidak ada hasil out-of-sample yang dipilih ulang
setelah melihat angkanya.

### 15.2 Metrik pendamping dan perannya

`evaluation.score()` mengembalikan tujuh angka, dengan pembagian peran yang tegas:

| Metrik                                | Peran                                              | Alasan keberadaannya                                                                                                                                                                                                                             |
| ------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`pinball` per τ**                   | **Kriteria pemilihan** (dirata-ratakan menjadi K1) | Satu-satunya yang memutuskan. Dilaporkan **per titik kuantil berdampingan**, bukan hanya sebagai satu rata-rata — model yang menang di rata-rata tapi kalah telak di satu ujung grid harus terlihat, bukan tertelan                              |
| `coverage` per τ                      | Cek kalibrasi                                      | Untuk setiap τ, proporsi baris dengan `actual ≤ prediksi` harus mendekati τ. Diperiksa **per kuantil**, bukan hanya di 0,9 (bagian 17 K2). Jauh di atas = overstock sistematis; jauh di bawah = janji service level tidak ditepati               |
| `fill_rate`                           | Kriteria sukses pemilik data                       | "Outlet tidak kehabisan", dinyatakan dalam unit. Kekurangan dijumlahkan **sebelum** dibagi, sehingga surplus di satu outlet-hari tidak bisa menutupi kehabisan di outlet lain — barangnya sudah berada di cabang yang salah pada hari yang salah |
| `shortfall_units` / `overstock_units` | Penerjemah ke bahasa bisnis                        | "pinball 2,390" tidak bisa didiskusikan di rapat; "kekurangan turun 73%, kelebihan naik 2,5×" bisa                                                                                                                                               |
| `mae`                                 | **Konteks saja**                                   | Dilaporkan, tidak memutuskan                                                                                                                                                                                                                     |
| `n`                                   | Bukti baris identik                                | Menjamin perbandingan sah                                                                                                                                                                                                                        |

> **Catatan satuan.** `shortfall_units` dan `overstock_units` menjumlahkan unit
> lintas SKU yang satuannya campur (Kg, Porsi, Botol, PCS, …), jadi angkanya sah
> untuk membandingkan antar model **pada baris yang sama**, tetapi tidak punya
> makna fisik sebagai satu besaran tunggal.

### 15.3 Kenapa MSE / RMSE tidak dipakai — dan tidak dihitung

`utils/modelling/evaluation.py` tidak memiliki fungsi RMSE maupun MSE. Itu bukan
kelalaian. Empat alasan, dua di antaranya spesifik pada data ini:

**1. Simetris — bertentangan langsung dengan service level.** Menghukum
kelebihan stok sekeras kehabisan stok, padahal seluruh perumusan masalah
menyatakan keduanya tidak setara.

**2. Menobatkan baseline naif sebagai juara.** Ini bukan argumen hipotetis; ia
terukur:

| model               |                  MAE |   shortfall | overstock |
| ------------------- | -------------------: | ----------: | --------: |
| `naive_roll_mean_7` | **9,65** ← juara MAE |   1.528.393 | 1.804.789 |
| XGBoost             |                14,31 | **414.172** | 4.529.674 |

Kriteria yang menobatkan rata-rata bergerak tujuh hari — model tanpa training
sama sekali, yang membiarkan kekurangan stok 3,7× lipat lebih besar — bukan
kriteria yang mengukur tujuan penelitian ini.

**3. RMSE akan memilih model yang paling jago meramal hal yang bukan tugasnya.**
Ini alasan terkuat, dan khusus proyek ini. RMSE mengkuadratkan galat, sehingga
didominasi baris bergalat terbesar — yaitu **lonjakan**. Tetapi
`docs/batasan-penelitian.md` B-3 menyatakan lonjakan justru **di luar lingkup
model**: logika bisnis saat ini adalah ramalan dibutuhkan untuk permintaan
_di luar pesanan_, sementara pesanan besar ditangani tim secara terpisah. Dan
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

Setiap hasil dilaporkan dalam **empat potongan**: gabungan, per `demand_segment`,
per `is_delivery_day` (`walk_forward.GROUP_COLS`), dan — sejak 2026-08-24 — **per
titik kuantil**. Delivery day adalah baris yang benar-benar menaikkan barang ke
truk; segmen permintaan adalah sumbu tempat satu angka global paling mudah
menyesatkan.

Potongan keempat lahir dari alasan yang sama persis dengan tiga yang pertama.
K1 adalah rata-rata 19 angka, dan sebuah rata-rata bisa menyembunyikan model yang
kalibrasinya rusak di ekor bawah tapi tertolong di tengah grid. Aturan yang
sudah berlaku di dokumen ini — jangan pernah melaporkan satu angka yang bisa
menyembunyikan kegagalan di sub-populasi — sekarang berlaku juga pada sumbu
kuantil, bukan hanya pada sumbu baris.

---

## 16. Posisi hasil saat ini

**Ditulis ulang 2026-08-29** setelah `docs/hasil-modeling-{rf,xgb,lstm}.md`
selesai diregenerasi di bawah K1 (rata-rata pinball lintas 19 titik
`QUANTILE_SET_A`) dan artefak pasca-reklasifikasi kategori WIP-2. Angka di
bagian ini **menggantikan** versi lama yang dihitung pada pinball@0,9 tunggal
— versi lama tidak diarsipkan terpisah di sini (bandingkan dengan angka model
dan dokumen hasil, yang diarsipkan sebagai `*.single-quantile.bak.*`), tapi
tetap terbaca di riwayat git berkas ini.

Ketiga model sudah dijalankan penuh: RF 2026-08-25, XGBoost 2026-08-27, LSTM
2026-08-28 (`docs/todolist-proyek.md` butir 0c). Angka berikut dari **345.547
baris validasi yang identik** — dijamin `walk_forward.eligible_rows()`,
dikonfirmasi oleh fakta bahwa ketiga baseline mencetak angka yang sama persis
di ketiga run. **Pencarian hyperparameter XGBoost dan LSTM berjalan di GPU
Windows (keputusan 2026-08-26); walk-forward dan fit final ketiganya tetap
satu mesin CPU Mac**, jadi angka K1/K2/K3 di bagian ini tetap sebanding lintas
model (`docs/hasil-modeling-xgb.md` bagian 7 bagian mesin-terpisah).

### 16.1 Gabungan

| potongan                | model               |         K1 | mae@0,9 | coverage@0,9 | `crossing_rate` |
| ----------------------- | ------------------- | ---------: | ------: | -----------: | --------------: |
| semua fold              | random_forest       |     2,8621 |  15,081 |        0,928 |          0,0000 |
|                         | lstm                |     2,8828 |  13,929 |        0,906 |          0,4345 |
|                         | xgboost             |     2,9197 |  13,408 |        0,902 |          0,9767 |
| **fold 1/2/4 (bersih)** | **random_forest**   | **2,8508** |  15,055 |        0,930 |          0,0000 |
|                         | lstm                |     2,8818 |  14,065 |        0,915 |               — |
|                         | xgboost             |     2,9433 |  13,467 |        0,905 |               — |
|                         | `naive_roll_mean_7` |     4,8603 |   9,721 |        0,696 |          0,0000 |

**Potongan fold 1/2/4 adalah angka yang sah untuk memilih**, karena fold 3 dan
5 ikut menentukan hyperparameter tiap model (bagian 9.3). Random Forest
unggul K1 di potongan bersih, tapi jaraknya ke LSTM (0,0310, 1,1%) jauh lebih
sempit daripada ke XGBoost (0,0925, 3,2%) — lihat bagian 18 untuk apakah
selisih itu melewati ambang 2% K1.

`crossing_rate` (kolom terakhir, gabungan 5 fold — tidak dipecah per potongan
karena crossing adalah properti per-baris-prediksi, bukan per-fold) adalah
**temuan paling mencolok di tabel ini**: RF 0% (struktural, dijamin bentuk
forest), LSTM 43,4%, XGBoost 97,7%. Dibaca sendirian, K1 menunjukkan
XGBoost/LSTM hampir menyaingi RF; dibaca bersama `crossing_rate`, hampir
seluruh baris prediksi XGBoost punya sedikitnya satu pasang titik kuantil
yang urutannya terbalik. Ini **belum dijelaskan** (`docs/hasil-modeling-xgb.md`
bagian 5.2, `docs/hasil-modeling-lstm.md` bagian 5.2) dan langsung
memengaruhi bagian 18 K2.

### 16.2 Per fold

K1 per fold:

| fold    | random_forest |  xgboost |     lstm |
| ------- | ------------: | -------: | -------: |
| 1 (Jul) |    **2,6819** |   2,8640 |   2,7058 |
| 2 (Agu) |    **2,8441** |   2,9178 |   2,8743 |
| 3 (Sep) |    **2,7541** | 2,7366\* | 2,7206\* |
| 4 (Okt) |    **3,0396** |   3,0568 |   3,0791 |
| 5 (Nov) |    **3,0305** | 3,0510\* | 3,0780\* |

\* Fold 3 dan 5 ikut memilih hyperparameter XGBoost dan LSTM (bagian 9.3),
jadi kedua kolom itu di kedua fold ini **bukan** murni out-of-sample untuk
XGBoost/LSTM — RF juga dipilih dari fold 3&5, jadi ini bukan keuntungan yang
timpang, tapi tetap dicatat.

**Random Forest menang K1 di kelima fold** — beda dari run kuantil-tunggal
lama, di mana kemenangan berpindah-pindah antar model per fold (bagian 16.2
lama, diarsipkan di riwayat git: LSTM terbaik di fold 1, RF di fold 2, dst.).
Di bawah K1, RF **konsisten** unggul, bukan berganti-ganti — argumen "varians
antar-fold lebih besar dari selisih antar-model" yang dulu berlaku untuk
pinball@0,9 tunggal **tidak lagi didukung data** di K1: rentang K1 antar fold
untuk satu model (RF: 2,682–3,040, 13%) memang tetap lebih lebar daripada
selisih K1 antar model pada fold yang sama (mis. fold 1: RF 2,682 vs LSTM
2,706, hanya 0,9%), tapi arah kemenangannya sekarang **stabil**, bukan acak.

### 16.3 Per segmen permintaan

K1 per segmen, gabungan 5 fold:

| segmen       | n (validasi) | random_forest | xgboost |    lstm |
| ------------ | -----------: | ------------: | ------: | ------: |
| smooth       |       45.485 |   **10,9478** | 11,0466 | 11,0092 |
| erratic      |       54.511 |    **5,4788** |  5,4969 |  5,4961 |
| lumpy        |      123.545 |    **1,1430** |  1,1823 |  1,1664 |
| intermittent |      122.006 |    **0,4194** |  0,4978 |  0,4236 |

**Random Forest menang K1 di keempat segmen** — berbeda dari kesimpulan run
lama ("ketiganya membagi kemenangan, tidak ada yang mendominasi"). Marginnya
tidak seragam: di `intermittent`, LSTM (0,4236) sangat dekat ke RF (0,4194,
+1,0%) sementara XGBoost jauh di belakang (0,4978, +18,7%); di segmen lain
ketiganya lebih rapat (0,3–3,4%). LSTM tidak pernah menjadi yang terbaik
mutlak di segmen mana pun, tapi juga tidak pernah menjadi yang terburuk.

### 16.4 Sisi bisnis

Kekurangan (shortfall) dan kelebihan (overstock) di τ=0,9, dijumlahkan lintas
5 fold, unit lintas SKU bersatuan campur (sah membandingkan model pada baris
sama, tidak punya makna fisik sebagai satu besaran tunggal):

|                     | kekurangan (shortfall) | kelebihan (overstock) |
| ------------------- | ---------------------: | --------------------: |
| `random_forest`     |            **418.250** |             4.793.038 |
| `lstm`              |                461.320 |             4.351.815 |
| `xgboost`           |                500.579 |         **4.132.651** |
| `naive_roll_mean_7` |              1.528.393 |             1.804.789 |

Ketiganya memangkas kekurangan stok 67–73% dari baseline dengan ongkos
kelebihan stok 2,3–2,7× lipat. Polanya monoton dengan coverage@0,9 (16.1): RF
paling tinggi coverage-nya → shortfall terendah, overstock tertinggi; XGBoost
paling rendah coverage-nya → sebaliknya; LSTM di tengah pada kedua sisi. Mana
yang disukai bisnis tergantung ongkos relatif shortfall vs overstock —
keputusan bisnis, bukan keputusan model, tapi ini persis pertukaran yang
diminta saat service level dipatok di 0,9.

### 16.5 Derau seed (LSTM) — dikonfirmasi 2026-08-30, bukan lagi dugaan

LSTM satu-satunya model dengan inisialisasi bobot acak. Diulang 3 seed pada
konfigurasi pemenang, dinilai di fold 3&5 saja (`docs/hasil-modeling-lstm.md`
bagian 5.1b):

| seed | K1 (fold 3&5) |
| ---: | ------------: |
|   44 |        2,8399 |
|   42 |        2,8617 |
|   43 |        3,0915 |

Rentang **0,2517** di fold 3&5 — delapan kali lebih lebar daripada jarak K1
LSTM ke RF di potongan bersih (0,0310, bagian 16.1). Waktu ditulis pertama
kali, ini masih dugaan tak-langsung (fold 3&5 bukan fold 1/2/4 yang dipakai
kriteria resmi).

**Diverifikasi langsung**: walk-forward **5-fold penuh** diulang dengan
`random_state=43` (bukan cuma fold 3&5), ~9,8 jam CPU Mac:

|                         | seed 42 (resmi, dipakai di seluruh dokumen) |    seed 43 |
| ----------------------- | ------------------------------------------: | ---------: |
| K1 (fold 1/2/4, bersih) |                                  **2,8818** | **3,0732** |

Selisih **0,1914 (6,6%)** — pada fold yang **sama persis** dipakai untuk
klaim K1, bukan proksi. Ini jauh melebihi ambang keputusan 2%. Dengan seed
43, LSTM kalah dari RF **7,80%**, lebih buruk bahkan dari XGBoost (3,0732 vs
2,9433) — bukan lagi "hampir seri", tapi model terburuk dari ketiganya pada
titik data ini. Rata-rata dua seed (2,9775) juga tetap kalah dari RF melebihi
ambang 2%.

**Konsekuensi untuk bagian 18 K1**: angka K1 LSTM yang dipakai di seluruh
dokumen ini (seed 42, 2,8818) **terbukti bukan representasi stabil** —
mendarat di ujung yang menguntungkan LSTM dari sebaran yang lebar, bukan
titik tengah. Kesimpulan "K1 seri, RF vs LSTM" tidak lagi bisa
dipertahankan sebagaimana ditulis semula.

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

### K1 — Kriteria utama: rata-rata pinball lintas `QUANTILE_SET` di potongan fold bersih (1/2/4)

**Direvisi 2026-08-24** (bagian 15.1). Sebelumnya: pinball@0,9 tunggal.

> Pemenang adalah **rata-rata pinball lintas `QUANTILE_SET` terendah**,
> **asalkan selisihnya ≥ 2%** terhadap pesaing terdekat. Di bawah ambang itu,
> hasilnya dinyatakan **seri** dan tangga diteruskan ke K2.

Yang berubah **hanya targetnya**, dari satu titik menjadi rata-rata seluruh titik
di `QUANTILE_SET`. Disiplin walk-forward-nya identik: potongan fold bersih tetap
1/2/4, ambang 2% tetap dikalibrasi dari varians terukur run ini sendiri
(justifikasi di bawah tidak berubah), dan pemenang tetap harus lolos G0 lebih
dulu.

Dua hal yang perlu dinyatakan supaya kriteria ini tidak disalahbaca:

- **Rata-ratanya tak berbobot.** Setiap titik kuantil menyumbang sama besar,
  termasuk 0,9. Pembobotan yang lebih berat ke arah 0,9 — karena itu komitmen
  bisnis di B-9 — sudah dipertimbangkan dan **belum diputuskan**; ia tercatat
  sebagai pertanyaan terbuka nomor 1 di
  `docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`.
  Sampai itu diputuskan, tak berbobot adalah yang berlaku, dan pilihan itu
  dinyatakan di sini supaya tidak terbaca sebagai kelalaian.
- **Skor per titik tetap dilaporkan berdampingan.** K1 memutuskan lewat satu
  angka, tapi angka itu tidak pernah berdiri sendiri di laporan (bagian 15.4).

**Kalibrasi ulang dilakukan (2026-08-29)**, seperti yang dijanjikan paragraf
ini sebelum ketiga run K1 selesai — bukan lagi angka dari metrik lama. Sumber:
`docs/hasil-modeling-{rf,xgb,lstm}.md` bagian 5.1 dan 5.1b, dan bagian 16 di
atas.

**Justifikasi ambang 2%, dihitung ulang dari K1.** Tiga sumber kebisingan
yang benar-benar terukur di run 2026-08-25/27/28:

- **Selisih antar-model pada fold yang sama** bergerak 1,2%–6,8% tergantung
  fold (fold 3, yang ikut memilih ketiganya: 1,2%; fold 1, yang tidak: 6,8%).
- **Satu model yang sama bergerak 12–14% antar fold berurutan** (RF 13,3%,
  XGBoost 11,7%, LSTM 13,8% — bagian 16.2).
- **Satu model, satu konfigurasi, tiga seed berbeda (LSTM saja) bergerak
  8,8%** (bagian 16.5) — ini sumber kebisingan yang jauh lebih ketat dari dua
  sebelumnya, karena satu-satunya variabel yang berubah adalah inisialisasi
  bobot, bukan bulan atau model.

**Kesimpulan yang berubah dari versi lama:** ambang 2% **tidak lagi jelas
konservatif**. Derau antar-seed LSTM saja (8,8%) jauh di atas ambang, jadi
selisih K1 sebesar 2% — persis di ambang — bisa jadi masih derau seed, bukan
sinyal model. Ambang 2% dipertahankan **apa adanya** untuk sekarang (bukan
dinaikkan secara spekulatif), tapi kepercayaan padanya berkurang: bagian 18
K1 membaca RF vs LSTM sebagai "seri" pada gap 1,09%, dan angka derau seed di
atas memperkuat bacaan itu, bukan melemahkannya — gap yang jauh di bawah
kebisingan yang terukur sendiri.

### K2 — Kalibrasi terhadap service level yang dijanjikan

**Direvisi 2026-08-24.** Sebelumnya diperiksa hanya di 0,9.

> Untuk **setiap** τ di `QUANTILE_SET`: `|coverage(τ) − τ|` pada potongan fold
> bersih, dibaca bersama stabilitasnya antar fold. `|coverage(0,9) − 0,90|`
> tetap dilaporkan terpisah dan diberi bobot khusus, karena 0,9 adalah titik yang
> benar-benar dijanjikan ke bisnis (B-9).

Ini bukan kriteria cadangan sembarangan — ia menguji **janji yang sama** dengan
kriteria utama, dari sudut yang berbeda. Model yang coverage-nya jauh di atas
targetnya menepati janji dengan cara yang mahal (overstock sistematis); yang jauh
di bawah tidak menepatinya sama sekali.

**Kenapa memeriksa seluruh τ memperkuat anak tangga ini, bukan sekadar
memperbanyak angkanya.** Simpangan yang **konsisten searah di seluruh** titik
kuantil adalah sinyal jauh lebih kuat daripada simpangan yang hanya muncul di
satu titik: yang pertama berarti seluruh distribusi ramalan model itu bergeser,
yang kedua bisa kebetulan. Dua pola yang harus dibedakan eksplisit saat menuruni
anak tangga ini:

| Pola                                             | Bacaan                                                               |
| ------------------------------------------------ | -------------------------------------------------------------------- |
| Simpangan searah di hampir seluruh τ             | Bias sistematis pada seluruh distribusi — alasan kuat untuk tersisih |
| Simpangan besar hanya di beberapa τ, arah campur | Derau atau kelemahan lokal — dicatat, bukan alasan menyisihkan       |

Contoh dari data lama: simpangan Random Forest +0,034 di 0,9 dinilai bukan derau
justru karena **konsisten searah di kelima fold**. Sumbu kuantil menambah
dimensi kedua untuk uji konsistensi yang sama.

**Quantile crossing.** Prediksi yang tidak monoton terhadap τ pada baris yang sama
(mis. prediksi τ=0,7 melampaui τ=0,8) adalah kegagalan kalibrasi yang hanya
terlihat setelah multi-kuantil dijalankan. Ia dilaporkan sebagai laju (proporsi
baris yang punya minimal satu inversi) untuk XGBoost dan LSTM; Random Forest
kebal secara struktural karena setiap titiknya adalah persentil dari satu
distribusi empiris yang sama. Laju crossing yang material dibaca di K2, bukan
diperbaiki diam-diam dengan mengurutkan hasil.

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

**Ditulis ulang 2026-08-29** dengan angka K1 bagian 16. Kesimpulan bagian ini
**berbeda secara struktural** dari versi lama: tangga tidak sampai ke K3/K4
sebagai penentu, karena K2 membuka temuan yang tidak bisa dijawab dari angka
yang ada sekarang. Ini bukan kegagalan menuruni tangga — ini tangga bekerja
sebagaimana dirancang: berhenti begitu ditemukan sesuatu yang harus dijawab
lebih dulu, bukan dilangkahi.

### G0 — Kelayakan: ketiganya lolos

Semua mengalahkan `naive_roll_mean_7` pada pinball@0,9 di **kelima** fold.
Margin serupa: RF 40,5–48,7%, XGBoost 42–45%, LSTM 43–48%
(`docs/hasil-modeling-{rf,xgb,lstm}.md` bagian 5.0). Fold 4 (Oktober)
konsisten paling berat bagi ketiganya — properti bulannya, bukan properti
modelnya.

### K1 — Rata-rata pinball 19 titik, fold bersih: **"SERI" DITARIK — seed 42 terbukti bukan titik representatif**

> **Direvisi 2026-08-30.** Verdict "seri" di bawah ini adalah bacaan literal
> terhadap K1 seed 42 saja — dan seed 42 sudah dikonfirmasi (bagian 16.5)
> mendarat di ujung sebaran yang menguntungkan LSTM, bukan titik tengah.
> Bacaan yang menggantikannya ada di penutup subbagian ini.

| model                              | K1 (fold 1/2/4) | selisih ke RF |
| ----------------------------------- | --------------: | ------------: |
| **random_forest**                  |      **2,8508** |             — |
| lstm (seed 42, resmi)              |          2,8818 |        +1,09% |
| xgboost (rearranged, 2026-08-30)   |          2,9115 |        +2,13% |
| xgboost (mentah, sebelum rearrange)|          2,9433 |        +3,24% |
| **lstm (seed 43, verifikasi)**     |      **3,0732** |    **+7,80%** |
| lstm (rata-rata 2 seed)            |          2,9775 |        +4,44% |

Ambang keputusan K1 adalah **≥2% terhadap pesaing terdekat** (bagian 17).
Dibaca lewat seed 42 saja, RF vs LSTM di bawah ambang → "seri", tangga
diteruskan ke K2 — itu yang tertulis semula. **Tapi seed 43, diukur dengan
protokol identik (walk-forward 5-fold penuh, fold 1/2/4 yang sama), memberi
RF kemenangan 7,80% — jauh di atas ambang.** Rata-rata dua seed pun tetap di
atas ambang (4,44%). Tidak ada bacaan yang konsisten menghasilkan "seri"
kecuali kalau seed 42 secara khusus dipilih sebagai yang mewakili.

**XGBoost, setelah rearrangement kuantil (2026-08-30, `docs/hasil-modeling-xgb.md`
bagian 5.5)**, membaik dari +3,24% menjadi **+2,13%** terhadap RF — tetap di
atas ambang 2%, jadi tetap kalah, dengan margin yang lebih tipis sekarang
crossing tidak lagi mengaburkan angkanya. Ini angka XGBoost yang **sah**
dipakai untuk perbandingan lintas model mulai sekarang (K2-nya juga sudah
bisa dipercaya, lihat K2 di bawah). Dibandingkan rata-rata dua-seed LSTM
(2,9775), XGBoost-rearranged sedikit lebih baik. Urutan tiga model **tetap
tidak stabil** terhadap pilihan seed LSTM: bisa RF < LSTM-seed42 <
XGBoost-rearranged atau RF < XGBoost-rearranged < LSTM-seed43 — tapi RF
unggul di **kedua** urutan itu.

**Bacaan yang menggantikan "seri"**: dengan n=2 pada seed LSTM, ini bukan
interval kepercayaan yang ketat — tapi **kedua** titik data yang ada (bukan
cuma satu) menunjuk arah yang sama: seed 42 adalah pengecualian yang
menguntungkan, bukan representasi tengah. Berat bukti saat ini condong ke
**RF unggul atas LSTM**, bukan seri — tapi menyatakannya sebagai kesimpulan
final butuh ≥1 seed lagi (masing-masing ~9,8 jam CPU Mac) untuk membedakan
"RF memang unggul" dari "kebetulan dua seed pertama sama-sama tidak mewakili
tengah sebaran". Tangga **tidak diteruskan ke K2 atas dasar "seri"** —
status K1 sekarang **tidak tuntas**, bukan seri, sampai bukti tambahan
tersedia atau keputusan diambil untuk berhenti di sini dengan RF sebagai
kandidat terdepan berdasarkan berat bukti yang ada.

### K2 — Kalibrasi dan `crossing_rate`: **XGBoost tidak lagi tertahan (rearranged, 2026-08-30)**

> **Catatan urutan (2026-08-30):** K1 di atas tidak lagi menghasilkan "seri"
> yang formal memicu langkah ini — statusnya sekarang "belum tuntas" (n=2
> seed, berat bukti condong ke RF). K2 di bawah tetap dikerjakan dan tetap
> berguna untuk dua alasan yang berdiri sendiri dari isu K1: (1) temuan
> `crossing_rate` semula menahan XGBoost, sekarang terjawab lewat
> rearrangement; (2) baris LSTM di tabel ini masih dari seed 42 yang sama
> yang sudah terbukti tidak representatif di K1 — jadi angka kalibrasi LSTM
> di sini **mewarisi keraguan yang sama**, belum tentu representatif untuk
> "LSTM" secara umum.

Di τ=0,90 (janji ke bisnis, B-9), kelebihan coverage di atas lantai
`share_nol` (metodologi lantai di `docs/hasil-modeling-rf.md` bagian 5.2).
Baris `xgboost` di bawah adalah angka **sesudah rearrangement**
(`docs/hasil-modeling-xgb.md` bagian 5.5) — yang sah dipakai sejak
2026-08-30:

| model         | coverage@0,9 | gap dari 0,90 | kelebihan di atas lantai | `crossing_rate` |
| ------------- | -----------: | ------------: | -----------------------: | --------------: |
| xgboost (rearranged) | 0,908 |        +0,008 |              **+0,0077** |      **0,0000** |
| lstm          |        0,906 |        +0,006 |                  +0,0062 |          0,4345 |
| random_forest |        0,928 |        +0,028 |                  +0,0281 |      **0,0000** |

Sesudah rearrangement, ketiga model kini punya `crossing_rate` yang bisa
dipercaya (0 untuk RF dan XGBoost, derau numerik yang diabaikan untuk LSTM),
jadi ketiga baris di tabel ini sekarang **sebanding secara adil**.

**`crossing_rate` diuji 2026-08-29** (`docs/hasil-modeling-{xgb,lstm}.md`
bagian 5.2, metodologi: hitung ulang crossing dengan toleransi jarak minimum
di atas prediksi dari bundle tersimpan) — jawabannya **berbeda arah untuk
tiap model**, bukan satu penjelasan untuk keduanya seperti diduga semula:

- **XGBoost: sebagian besar defek sungguhan.** Rate bertahan **~20–25% di
  toleransi 0,5–1,0 unit**, ekor distribusi sampai 139 unit. Angka kalibrasi
  XGBoost di tabel di atas **tidak bisa dipercaya begitu saja** — model yang
  kuantilnya saling silang secara material tidak benar-benar tahu di mana
  τ=0,9-nya berada pada seperlima-seperempat barisnya.
- **LSTM: hampir seluruhnya derau numerik.** Rate ambruk ke 1,1% di toleransi
  0,1, median inversi 0,0027 unit. Angka kalibrasi LSTM di tabel di atas
  **bisa dipercaya** — crossing bukan alasan valid untuk meragukannya.

**Konsekuensi (semula): XGBoost tertahan di anak tangga ini** — butuh
post-hoc rearrangement kuantil (Chernozhukov et al., 2010) dan angka
K1/K2-nya dihitung ulang di atas prediksi yang sudah diperbaiki sebelum ia
sah disandingkan lagi.

**Dikerjakan 2026-08-30** (`docs/hasil-modeling-xgb.md` bagian 5.5,
`xgb_rearrangement_walkforward.py`) — kelima model fold di-fit ulang dengan
hyperparameter pemenang yang sama, prediksi diurutkan (sort) per baris
sebelum dinilai. Cek reproduksibilitas cocok persis dengan run resmi
(`best_iteration` per fold identik, K1 sebelum sort = 2,9433 persis sama),
jadi angka sesudahnya sah dipercaya: `crossing_rate` → **0,0000**, K1 fold
bersih membaik 1,08% (2,9433 → 2,9115). **XGBoost tidak lagi tertahan** —
tapi perbaikannya tidak cukup membalik urutan: gap ke RF menyempit dari
3,24% menjadi **2,13%, masih di atas ambang 2%**. XGBoost tetap kalah dari
RF, sekarang dengan angka yang sudah adil dibandingkan.

**RF vs LSTM di K2 (τ=0,90): LSTM (+0,0062) lebih dekat ke target daripada
RF (+0,0281)** — tapi baris LSTM ini dari seed 42, seed yang sama yang
sudah terbukti bukan representasi K1 yang stabil (bagian 16.5, K1 di atas).
Belum diverifikasi apakah kalibrasi LSTM di τ=0,90 ikut goyang sebesar K1-nya
kalau diukur dengan seed 43 — kemungkinan besar iya, mengingat keduanya
dihitung dari model dan fold yang sama. Selisihnya di sini **tipis dan
satu-titik** (kedua model punya bentuk over-coverage serupa di seluruh grid
19 titik, memuncak +0,16–0,18 di τ≈0,40–0,45 — `docs/hasil-modeling-rf.md`
bagian 5.2, `docs/hasil-modeling-lstm.md` bagian 5.2), jadi ini **tidak**
cukup untuk menyisihkan RF secara mandiri, dan dengan K1 sekarang condong ke
RF (bukan lagi seri), K2 di titik ini **tidak mengubah arah kesimpulan K1**
— hanya konteks tambahan yang sama-sama perlu diverifikasi ulang dengan
seed LSTM yang lebih dari satu.

### K3 — Ongkos dan reprodusibilitas (dicatat, belum jadi penentu)

Karena K2 belum tuntas, K3 belum sah dipakai memutuskan (bagian 17: K3 "sah
menjadi penentu justru karena K1 menyatakan seri" — bukan pengganti K2 yang
belum tuntas). Dicatat di sini supaya tersedia begitu K2 selesai:

|                                                                |           random_forest |              xgboost |                                                  lstm |
| -------------------------------------------------------------- | ----------------------: | -------------------: | ----------------------------------------------------: |
| Walk-forward + fit final (CPU Mac, sebanding lintas ketiganya) |               ~93 menit |           ~204 menit |                                        **~586 menit** |
| Pencarian (device beda, tidak sebanding)                       |          3,85 jam (CPU) |      ~14,7 jam (GPU) |               ~4,2 jam (GPU) + ~50 menit 3-seed (GPU) |
| Ukuran artefak                                                 |                  826 MB |               292 MB |                                            **3,7 MB** |
| Bergantung seed                                                | tidak (bagging meredam) |   tidak diverifikasi | **ya** — rentang K1 0,2517 antar 3 seed (bagian 16.5) |
| Dependensi                                                     |       `quantile-forest` | `xgboost` + `libomp` |                                               `torch` |

Pada satu-satunya tahap yang device-nya sebanding lintas ketiganya
(walk-forward + fit final, CPU Mac), **LSTM ~6,3× lebih lambat dari RF dan
~2,9× lebih lambat dari XGBoost** — dan satu-satunya model yang hasilnya
terbukti bergantung seed pada besaran yang jauh melebihi selisih K1 antar
model (bagian 16.5). RF tetap yang paling murah di tahap yang sebanding, dan
XGBoost di tengah.

### K4 — Risiko integrasi (dicatat, belum jadi penentu)

- **XGBoost** — bundel `models/xgboost_q90.joblib` mendeskripsikan diri
  sendiri (urutan kolom, encoding, level kategori); memprediksi dari satu
  baris.
- **LSTM** — `predict_bundle()` mewajibkan panel berjendela (sequence
  window), tidak bisa meramal satu baris sendirian — pipeline panel harus
  hidup di sisi inferensi, bukan hanya training.
- **Random Forest** — artefak 826 MB, jauh lebih besar dari XGBoost (292 MB)
  dan LSTM (3,7 MB), tapi memprediksi dari satu baris seperti XGBoost.

### Status keputusan

> **Prasyarat wajib terjawab (2026-08-30) — RF adalah kandidat terdepan
> berdasarkan berat bukti, tapi pemenang belum resmi dibekukan** (itu
> langkah terpisah, lihat penutup subbagian ini). Tiga pertanyaan yang
> menahan keputusan sejak 2026-08-29 sekarang semuanya terjawab dengan data:
>
> - **`crossing_rate` diuji (2026-08-29)**: XGBoost defek sungguhan (~20–25%
>   baris crossing material) — tertahan sendiri. LSTM hampir seluruhnya
>   derau numerik — tidak jadi keraguan untuk LSTM.
> - **Derau seed LSTM diverifikasi langsung (2026-08-30)**: walk-forward
>   5-fold penuh dengan seed 43 memberi K1 LSTM = 3,0732 di fold 1/2/4 —
>   **kalah 7,80% dari RF**, jauh di atas ambang 2%. Seed 42 (2,8818, dipakai
>   di seluruh dokumen sampai sebelumnya) terbukti bukan titik tengah
>   sebaran. "K1 seri, RF vs LSTM" **ditarik**.
> - **Rearrangement kuantil XGBoost dikerjakan (2026-08-30,
>   `docs/hasil-modeling-xgb.md` bagian 5.5)**: `crossing_rate` → 0,0000, K1
>   fold bersih membaik 1,08% (2,9433 → **2,9115**) — tapi gap ke RF cuma
>   menyempit dari 3,24% menjadi **2,13%, masih di atas ambang 2%**. XGBoost
>   kini bisa dibandingkan adil, dan hasilnya **tetap kalah**.
>
> **Ketiga model sekarang punya angka K1/K2 yang sah dibandingkan
> langsung** — tidak ada lagi crossing yang mengaburkan XGBoost, dan LSTM
> sudah diuji dengan seed kedua. Pada kedua titik data LSTM yang ada (seed
> 42 dan 43) dan pada angka XGBoost yang sudah di-rearrange, **RF unggul di
> setiap perbandingan**: RF < LSTM-seed42 < XGBoost-rearranged < LSTM-seed43.
>
> **Ini bukan "RF resmi menang" secara administratif** — n=2 pada seed LSTM
> tidak memberi interval kepercayaan yang ketat secara statistik, dan
> membekukan pemenang dalam sebuah commit tetap memerlukan **persetujuan
> eksplisit pemilik proyek** (`docs/todolist-proyek.md` Fase E, butir 1).
> Tapi tidak ada lagi prasyarat teknis yang menahan langkah itu — semua
> yang tercatat sebagai "belum dikerjakan" di bagian ini sampai 2026-08-29
> sekarang sudah punya jawaban.
>
> **Langkah berikutnya** (`docs/todolist-proyek.md` butir 10, Fase E):
> pemilik proyek meninjau angka di atas dan menyetujui RF sebagai pemenang
> dalam sebuah commit, baru test set Desember 2025 dibuka sekali lewat
> protokol bagian 19.

---

## 19. Protokol pembukaan test set Desember

Enam langkah, dijalankan **sekali**, setelah bagian 18 dibekukan dalam sebuah commit.

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
> dan bagian 19 baru dapat dijalankan di atas hasil pelatihan ulang itu. Rinciannya di
> bagian 0 `docs/pipeline-overview.md` dan B-9 `docs/batasan-penelitian.md`.

**3. Laporkan penyebutnya.** Desember dinilai pada **49.717 dari 55.046 baris
panel** (996 baris warm-up, 4.333 baris tanpa target). Baris yang dikeluarkan
bukan irisan yang bias, tetapi angkanya wajib disebut.

**4. Angka utama = skor model terpilih.** Rata-rata pinball lintas
`QUANTILE_SET` (K1), coverage **per titik kuantil** (K2), dan fill_rate model
terpilih adalah angka final penelitian. Skor pinball dan coverage di **τ = 0,9**
dilaporkan terpisah dan diberi tempat khusus, karena itulah titik yang dijanjikan
ke bisnis (B-9) — tetapi ia adalah salah satu angka yang dilaporkan, bukan lagi
satu-satunya angka utama. Dua model lain dilaporkan sebagai **pembanding
deskriptif**, dengan label eksplisit bahwa pemenang ditetapkan sebelum Desember
dibuka.

_Direvisi 2026-08-24; sebelumnya berbunyi "pinball@0.9, coverage, dan fill_rate
XGBoost", yang mengasumsikan pemenang sudah pasti XGBoost — asumsi yang ditahan
sampai run ulang selesai (bagian 18)._

**5. Tiga potongan, bukan satu angka.** Gabungan, per `demand_segment`, per
`is_delivery_day` — sama seperti seluruh pelaporan validasi.

**6. Jangan menukar pemenang.** Bila model lain mencetak angka lebih tinggi di
Desember, itu **temuan** yang ditulis di pembahasan — dan temuan yang justru
mengonfirmasi klaim K1 bahwa urutannya tidak stabil di bawah tingkat kebisingan
ini. Ia bukan alasan mengganti pilihan.

### 19.1 Yang akan membatalkan rencana ini

Satu-satunya hasil yang menuntut lebih dari sekadar pelaporan adalah **kegagalan
kalibrasi yang besar**: coverage model terpilih di Desember jatuh jauh di bawah
targetnya (indikatif: coverage di τ=0,9 turun di bawah 0,85, atau simpangan
searah yang besar di seluruh `QUANTILE_SET`). Itu bukan sinyal untuk menukar
model — ketiganya menargetkan grid kuantil yang sama dan akan bergerak searah —
melainkan sinyal bahwa Desember berperilaku berbeda dari lima bulan validasi,
yang harus dibahas sebagai keterbatasan generalisasi musiman, bukan
disembunyikan.

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

> **Dua butir di atas berubah pada run multi-kuantil (2026-08-24).** Keduanya
> ditulis untuk run kuantil-0,9 tunggal dan tetap berlaku untuk angka-angka run
> itu. Untuk run multi-kuantil, penyetaraan anggaran (LSTM 30 kandidat, ruang
> 144, 3 seed pada pemenang — bagian 21) mencabut tiga dari empat alasan di butir
> pertama dan seluruh alasan di butir kedua. Yang **tetap** berlaku dan tidak
> boleh hilang saat bagian 20 ditulis ulang: satu dataset, satu domain, satu periode.
> Penyetaraan anggaran membuat "LSTM kalah karena arsitekturnya" menjadi klaim
> yang bisa dipertahankan pada dataset ini — bukan menjadi klaim tentang
> arsitektur LSTM secara umum.

- ❌ Klaim apa pun berbasis MAE terhadap baseline (bagian 15.3).
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

**Diperbarui 2026-08-24** — migrasi multi-kuantil menyisipkan butir 0a–0d di
depan, dan butir 1 tidak boleh dijalankan sebelum keempatnya selesai.

| #      | Pekerjaan                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Status                                                                                                                                                                                                                                              |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0a** | **Revisi spec RF/XGB/LSTM + bagian 15/17/19/21 ke kriteria multi-kuantil**                                                                                                                                                                                                                                                                                                                                                                                    | ✅ selesai 2026-08-24                                                                                                                                                                                                                               |
| **0b** | **Implementasi multi-kuantil di `evaluation.py`, `walk_forward.py`, `model_common.py`, `model_xgboost.py`, `model_lstm.py`, `model_random_forest.py` + ketiga notebook**                                                                                                                                                                                                                                                                                      | ✅ selesai 2026-08-24 — kontrak `fit_predict` kini `(n, len(QUANTILE_SET))`, 739 tes lolos, notebook diubah tetapi **belum dijalankan**                                                                                                             |
| **0c** | **Menjalankan ulang ketiga notebook** — ketiganya dengan pencarian hyperparameter penuh pada data pasca-reclass: Random Forest (18 kandidat), XGBoost (30 kandidat), dan LSTM (**30 kandidat, ruang 144, + 3 seed pada pemenang**). Pencarian RF semula hendak dipakai ulang; keputusan itu dibalik 2026-08-24 karena `rf_best_params.json` dipilih di atas data pra-reclass WIP-2 — kebasian yang sama yang sudah dipakai sebagai alasan membuang bundle-nya | ✅ **selesai** — RF 2026-08-25, XGBoost 2026-08-27, LSTM 2026-08-28. Pencarian XGB/LSTM pindah ke GPU Windows 2026-08-26 (walk-forward/fit final tetap CPU Mac untuk ketiganya)                                                                     |
| **0d** | **Menulis ulang `docs/hasil-modeling-{rf,xgb,lstm}.md` dari nol, lalu bagian 16 dan bagian 18**                                                                                                                                                                                                                                                                                                                                                               | ✅ **dokumen ditulis (2026-08-29)** — tapi bagian 18 berstatus "keputusan ditahan", bukan pemenang yang dibekukan (lihat 2 prasyarat baru di bawah butir 1)                                                                                         |
| 1      | Membekukan usulan bagian 18 dalam sebuah commit                                                                                                                                                                                                                                                                                                                                                                                                               | ⬜ **diblokir** — bagian 18 sendiri belum sampai ke usulan pemenang. Dua prasyarat: (a) jelaskan `crossing_rate` XGBoost 97,7%/LSTM 43,4%; (b) walk-forward LSTM ≥1 seed tambahan (rentang K1 antar-seed 8,8% jauh di atas ambang K1 2%, bagian 17) |
| 2      | Menjalankan protokol bagian 19 — buka Desember sekali                                                                                                                                                                                                                                                                                                                                                                                                         | ⬜ setelah #1                                                                                                                                                                                                                                       |
| 3      | Menulis `docs/hasil-test-desember.md`                                                                                                                                                                                                                                                                                                                                                                                                                         | ⬜ setelah #2                                                                                                                                                                                                                                       |
| **3a** | **Alokasi kuantil tersegmentasi pada model pemenang** — Bagian 4 spec-nya (perluasan multi-kuantil) sudah diwarisi dari #0c, jadi tinggal simulasi λ dan seterusnya                                                                                                                                                                                                                                                                                           | ⬜ setelah #3, `2026-08-22-segmented-quantile-allocation-design.md`                                                                                                                                                                                 |
| 4      | **Dekomposisi harian** (`target_h1`…`target_h4`) untuk ketiga model — menjawab "kapan permintaan terkonsentrasi"                                                                                                                                                                                                                                                                                                                                              | ⬜ direncanakan di spec pemodelan                                                                                                                                                                                                                   |
| 5      | **SHAP untuk pemenang saja** — menjawab "kenapa model meyakini ini"                                                                                                                                                                                                                                                                                                                                                                                           | ⬜ direncanakan di spec pemodelan                                                                                                                                                                                                                   |
| 6      | Mengisi `tanggal_buka` Cikarang Pusat di `outlet_closures.csv` + memperbarui `RELOCATION_DATES`                                                                                                                                                                                                                                                                                                                                                               | ⬜ menunggu pemilik data                                                                                                                                                                                                                            |
| 7      | Memperluas `calendar_features.py` ke 2026 sebelum data periode baru masuk                                                                                                                                                                                                                                                                                                                                                                                     | ⬜                                                                                                                                                                                                                                                  |

**Penyetaraan anggaran pencarian (keputusan pemilik proyek, 2026-08-24,
sesudah T-7).** Butir 0b dan 0c dijalankan dengan anggaran pencarian LSTM yang
dinaikkan: **30 kandidat** (dari 12, setara XGBoost), **ruang 144** (dua dimensi
kapasitas — `num_layers` dan `hidden_size` — dikembalikan ke `SEARCH_SPACE`),
dan **3 seed** pada konfigurasi terbaik. Anggaran RF (18) dan XGBoost (30) tidak
berubah.

Alasannya adalah validitas atribusi, bukan ongkos. Ketimpangan anggaran selama
ini tercatat sebagai keterbatasan yang dibaca bersama hasil (bagian 14, bagian 20) —
posisi yang bisa diterima ketika anggaran itu warisan run sebelumnya. Begitu
**ketiga model dicari ulang penuh dari nol** di butir 0c, ekonominya berubah:
ketimpangan itu tidak lagi diwarisi, melainkan dipilih ulang, dan mempertahankan
LSTM di 12 draw berarti secara sadar memilih menghasilkan angka yang tidak dapat
diatribusikan. Tanpa penyetaraan ini, kalau LSTM kalah di K1 kita tidak bisa
membedakan apakah **arsitekturnya memang kurang cocok** atau **pencariannya yang
paling dangkal** — dan itu persis pertanyaan inti penelitian ini. Dua dimensi
kapasitas itu dipotong 2026-08-19 karena ongkos per epoch, bukan karena terbukti
tidak menolong; bagian 18 dan `hasil-modeling-lstm.md` mencatatnya sebagai pertanyaan
yang tidak pernah ditanyakan, bukan pertanyaan yang sudah dijawab. Tiga seed
menjawab keberatan yang berdiri sendiri: LSTM satu-satunya model yang
inisialisasinya acak, sehingga variansnya selama ini hanya bisa **diduga** dari
selisih antar fold — yang mencampur varians seed dengan varians data.

Konsekuensi ongkos dinyatakan terbuka: ini menaikkan ongkos butir 0c secara
signifikan, karena LSTM model termahal per fit dan ketiga perubahan mengalikan
ongkosnya sekaligus — 2,5x kandidat, head 19 keluaran, dua fit tambahan untuk
seed kedua dan ketiga, dan sebagian draw kini boleh mengambil `num_layers=2`
atau `hidden_size=256` yang per epoch-nya jauh lebih mahal (259 s vs 104 s pada
pengukuran 2026-08-19). Plafon 8 jam LSTM sudah ditinggalkan sebelum ini;
keputusan ini memperbesar kelampauannya, dan wall clock sebenarnya dicatat di
`docs/hasil-modeling-lstm.md` sebagai ongkos terukur.

Butir 0a–0d adalah migrasi evaluasi multi-kuantil
(`docs/superpowers/specs/2026-08-22-model-comparison-refactor-migration.md`).
Urutannya tidak bebas: butir 1 membekukan pemenang, dan membekukan pemenang di
atas angka kriteria lama akan membekukan keputusan yang kriterianya sendiri sudah
diganti. Migrasi ini tidak membuang hasil out-of-sample apa pun karena Desember
belum pernah dibuka — yang diulang hanya pencarian dan walk-forward di kelima
fold latih.

Butir 4 dan 5 bukan tambahan opsional — keduanya sudah tertulis sebagai rencana
penjelasan (_explainability_) di
`docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md`, dengan
pembagian peran yang tegas: dekomposisi menjawab **kapan**, SHAP menjawab
**kenapa**. SHAP hanya dijalankan untuk pemenang, karena menjalankannya untuk
ketiganya berarti membayar ongkos penjelasan untuk model yang tidak akan dipakai.

---

## Rujukan

| Topik                               | Berkas                                                                                                |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Prapemrosesan, formal untuk laporan | `docs/preprocessing.md` (Bagian 1)                                                                    |
| Prapemrosesan, naratif + trade-off  | `docs/preprocessing.md` (Bagian 2)                                                                    |
| Pipeline ujung ke ujung (Inggris)   | `docs/pipeline-overview.md`                                                                           |
| Batasan yang tidak bisa dikode      | `docs/batasan-penelitian.md`                                                                          |
| Hasil terukur per model             | `docs/hasil-modeling-{rf,xgb,lstm}.md`                                                                |
| Desain prapemrosesan pemodelan      | `docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md`                                  |
| Desain tiap model                   | `docs/superpowers/specs/2026-08-{18-random-forest,19-xgboost,19-lstm}-modeling-design.md`             |
| Mesin evaluasi bersama              | `utils/modelling/walk_forward.py`, `utils/modelling/model_common.py`, `utils/modelling/evaluation.py` |
| Metodologi evaluasi multi-kuantil   | `docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`                               |
| Checklist migrasi multi-kuantil     | `docs/superpowers/specs/2026-08-22-model-comparison-refactor-migration.md`                            |
| Alokasi kuantil tersegmentasi       | `docs/superpowers/specs/2026-08-22-segmented-quantile-allocation-design.md`                           |
