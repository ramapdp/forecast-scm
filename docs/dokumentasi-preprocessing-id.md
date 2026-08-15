# Dokumentasi Pipeline Data Preprocessing (Bahasa Indonesia)

Dokumen ini menjelaskan secara rinci **seluruh alur preprocessing data** pada
proyek peramalan permintaan (demand forecasting) rantai pasok Kebuli Yaman —
mulai dari file ekspor mentah sampai tabel siap-latih untuk XGBoost, Random
Forest, dan LSTM — beserta **keputusan desain yang diambil saat brainstorming**
dan **trade-off** dari setiap keputusan tersebut.

Ditujukan sebagai bahan penyusunan laporan. Rujukan teknis berbahasa Inggris
ada di `docs/pipeline-overview.md` dan `docs/superpowers/specs/*.md`.

- Status kode: 14 tahap terimplementasi, **306 unit test lulus** (`.venv/bin/python3 -m unittest discover -p "test_*.py"`).
- Tanggal dokumen: 2026-08-15.

---

## 1. Konteks bisnis dan tujuan peramalan

Tim SCM pusat mengirim barang ke outlet dengan jadwal tetap:

| Kawasan | Hari pengiriman |
|---|---|
| Kawasan 1 | Senin dan Kamis |
| Kawasan 2 | Selasa dan Jumat |

Angka yang benar-benar dibutuhkan tim SCM bukan "permintaan besok", melainkan
**total permintaan sejak besok sampai kiriman berikutnya tiba**, per pasangan
(item, cabang). Angka itulah yang ditulis di surat jalan. Karena itu target
utama pipeline adalah kolom **`target_lead_time_cumulative`**.

Dua konsekuensi penting dari fakta bahwa model akan dipakai langsung oleh tim
SCM (bukan sekadar dilaporkan):

1. Preprocessing harus **dapat dijalankan ulang** setiap minggu atas data baru.
2. Prediksi harus **dapat dijelaskan** kepada orang yang bertindak atasnya.

Kedua hal ini mempengaruhi banyak keputusan di bawah (mapping kategori yang
dipersistensi, target harian sebagai dekomposisi, dan sebagainya).

---

## 2. Sumber data

| Sumber | Isi | Catatan |
|---|---|---|
| `dataset/excel/*.xlsx` → `dataset/*.csv` | Log "Barang Keluar", 1 baris per line item, Jan 2024–Des 2025 | 5 file memecah **waktu**, bukan kategori; delimiter `;`, UTF-8 BOM |
| `outlets.json` / `dataset/outlets.csv` | Master outlet: kota, kecamatan, kanal online (Shopee/GoFood/GrabFood) | Disinkronkan lewat `sync_outlets.py` |
| `dataset/outlet_name_overrides.csv` | Koreksi manual nama cabang & kota ambigu | 10 baris |
| `dataset/outlet_mapping.csv` | `kawasan` + `hari_pengiriman` per cabang | Dasar perhitungan lead time |
| `dataset/event_driven_items.csv` | Penanda SKU yang permintaannya digerakkan pesanan acara | Draf, 5 dari 70 SKU ditandai `true` |
| `dataset/outlet_closures.csv` | Interval outlet tidak beroperasi (tutup sementara, relokasi berjalan) | 3 baris, dikonfirmasi pemilik data 2026-08-15 |

Kolom mentah: `Tanggal`, `Kategori Barang`, `Kode Barang`, `Nama Barang`,
`Nama Cabang`, `Satuan`, `Kuantitas`.

**Keanehan data yang harus ditangani sejak awal** (semuanya sudah ditangani
kode):

- `jan-des-25.csv` punya 2 kolom kosong tambahan (9 field vs 7).
- Sebagian kode/nama barang berawalan `xxx.` (penanda internal, bukan SKU baru).
- Tiga file punya baris kosong `;;;;;;` di akhir (artefak ekspor Excel).
- File berawalan BOM UTF-8 sehingga harus dibaca dengan `encoding="utf-8-sig"`.

---

## 3. Peta alur keseluruhan

```
File mentah .xlsx / .csv
  │
  ├─ 1  merge_dataset.py            → dataset/dataset.csv
  ├─ 2  aggregate_dataset.py        (gabung baris duplikat)
  ├─ 3  normalize_items.py          (bersihkan kode barang & satuan)
  ├─ 4  outlet_features             (filter + kanonikalisasi cabang)
  ├─ 5  build_panel.py              (panel harian padat per segmen aktif +
  │                                  filter riwayat min.)
  ├─ 6  calendar_features.py        (fitur kalender & musim tinggi)
  ├─ 7  outlier_handling.py         (deteksi & capping lonjakan)
  ├─ 8  prepare_forecast_data.py    (target, lead time, outlet, lag, rolling,
  │                                  statistik cabang)
  ├─ 9  export_featured             → dataset/model_ready/featured.parquet
  ├─ 10 split_train_test            (train < 2025-12-01, test = Des 2025)
  ├─ 11 export_splits               → train.parquet / test.parquet
  ├─ 12 run_qa_checks()             (10 asersi kualitas data)
  │
  ├─ 13 modeling_prep.py            (event flag → segmen permintaan → fold →
  │                                  imputasi → encoding kategori)
  │                                 → dataset/model_ready/model_input.parquet
  └─ 14 adapter                     to_tabular()   → XGBoost, Random Forest
                                    to_sequences() → LSTM
                                    validate_contract() mengikat keduanya
```

Ukuran data setelah pipeline (terverifikasi dari file parquet):

| Artefak | Baris | Kolom |
|---|---|---|
| `featured.parquet` | 1.503.564 | 64 |
| `train.parquet` (sebelum 2025-12-01) | 1.448.518 | 64 |
| `test.parquet` (Desember 2025) | 55.046 | 64 |
| `model_input.parquet` | 1.503.564 | 76 |

Cakupan: **70 SKU × 59 cabang = 2.979 pasangan aktif**, 731 hari kalender
(2024-01-01 → 2025-12-31), 16 kota.

---

## 4. Tahap 1–4 — Penggabungan dan pembersihan sumber

### Tahap 1 — Menggabungkan lima periode (`merge_dataset.py`)

Membaca 5 CSV berurutan kronologis, memotong skema jadi 7 kolom, mem-parsing
`Tanggal` (`%d %b %Y`) hanya untuk keperluan pengurutan, lalu menulis
`dataset/dataset.csv`.

**Keputusan & trade-off**

| Keputusan | Alasan | Trade-off |
|---|---|---|
| Script berbasis pustaka standar (`csv`, `datetime`), bukan pandas | Saat itu repo belum punya manifest dependensi | Kode sedikit lebih panjang; dibayar dengan tidak ada dependensi untuk tahap paling awal |
| **Gagal keras** (raise) bila kolom ke-8/9 tidak kosong atau tanggal gagal di-parse | Data rusak harus terlihat, bukan hilang diam-diam | Pipeline bisa berhenti karena satu baris rusak — disengaja |
| Baris kosong `;;;;;;` dibuang, bukan dianggap error | Terbukti artefak ekspor Excel, bukan transaksi | Jika suatu saat baris kosong berarti sesuatu, informasi itu hilang (risiko sangat kecil) |
| Format tanggal & delimiter asli dipertahankan di output | File tetap bisa dibuka pemilik data seperti aslinya | Parsing ulang diperlukan di tahap berikutnya |

### Tahap 2 — Agregasi baris duplikat (`aggregate_dataset.py`)

Menjumlahkan `Kuantitas` untuk baris yang identik pada (tanggal, kategori, kode
barang, nama barang, cabang, satuan). Satu transaksi kadang tercatat sebagai
beberapa line item.

### Tahap 3 — Normalisasi kode barang (`normalize_items.py`)

Dua bentuk normalisasi kode: **membuang awalan `xxx.`** dan **menyeragamkan
pemisah** (`FGS.00047` → `FGS-00047`).

**Aturan penggabungan bersyarat (keputusan terpenting di tahap ini):** dua kode
hanya digabung jika **nama barangnya juga cocok** setelah normalisasi ringan
(buang awalan `xxx.`, rapikan spasi, buang anotasi dalam kurung).

Alasannya diperoleh dari pemeriksaan data nyata, bukan asumsi:

- Penyeragaman pemisah saja akan menabrakkan **5 pasang produk yang sama sekali
  berbeda** yang kebetulan berbagi digit — contoh `FGS-00047` = *Kentang Mustofa
  Rumput Laut* (Pack) vs `FGS.00047` = *Air Isi Ulang* (Galon).
- Pembuangan awalan `xxx.` menghasilkan 4 grup dengan nama tidak cocok; 3 bersifat
  kosmetik (`250ml` vs `250 ml`), 1 (`Cendol Pandan - FG` vs `Cendol - FG`) tidak
  digabung karena perbedaan rasa tak bisa disingkirkan dari data saja.

Selain itu tahap ini:

- **Konversi satuan**: beberapa item berawalan `xxx.` (Santan Cendol, Gula Cendol)
  tercatat dalam gram, bukan porsi. Faktor 40 dan 30 gram/porsi diturunkan dari
  fakta bahwa **setiap nilai mentah merupakan kelipatan bulat** dari faktor itu —
  bukti empiris, bukan tebakan — sehingga deretnya menyambung mulus dengan
  periode setelahnya yang sudah berdenominasi porsi.
- **Eksklusi item discontinued**: Nasi Putih, Cendol Pandan, Ayam Crispy
  Original/Spicy. Dua yang terakhir sebelumnya dipaksa-gabung lewat
  `EXPLICIT_ITEM_RENAMES`; ternyata itu menggabungkan produk yang benar-benar
  berbeda, sehingga keduanya kini dikeluarkan dan tabel rename dikosongkan.
- **Re-agregasi** pada level kode ternormalisasi, karena normalisasi dapat
  membuat baris yang tadinya berbeda menjadi kunci yang sama.

| Keputusan | Trade-off |
|---|---|
| Gabung hanya bila nama cocok | Beberapa deret tetap terpecah (mis. Cendol) meski mungkin sebenarnya satu produk — dipilih karena salah-gabung merusak label, salah-pisah hanya mengurangi riwayat |
| Konversi gram→porsi berbasis bukti kelipatan | Jika asumsi faktor salah, seluruh deret item itu bias skala; risiko ditekan lewat uji kelipatan bulat |
| `Satuan` tidak dijadikan fitur | Konstan per item sehingga tidak informatif | Dipertahankan selama QA untuk membuktikan konstansinya |

### Tahap 4 — Filter dan kanonikalisasi cabang (`outlet_features.py`)

1. `filter_matched_branches` **membuang baris cabang yang tidak ada di
   `outlets.csv`** — outlet yang sudah tidak beroperasi (10 cabang, ±10% baris).
2. `canonicalize_branch_names` menulis ulang setiap `Nama Cabang` ke nama
   kanonik outletnya, sehingga cabang yang tercatat dengan dua string berbeda
   (mis. nama lama di periode ekspor awal) menyatu menjadi satu riwayat kontinu.
3. `reaggregate_daily` dijalankan lagi untuk menjumlahkan baris yang bertabrakan
   akibat penggantian nama tersebut.

**Keputusan & trade-off**

| Keputusan | Alasan | Trade-off |
|---|---|---|
| `outlets.csv` dijadikan **sumber kebenaran keberadaan cabang** | Tidak ada gunanya meramal cabang yang sudah tutup | Tidak bisa membedakan "cabang tutup" dari "cabang baru yang belum didaftarkan". Mitigasi: daftar cabang yang dibuang dicetak setiap run agar anomali langsung terlihat |
| Cabang tak cocok **dibuang**, bukan diberi `kota="Unknown"` | Keputusan revisi dari desain awal — data cabang mati mengotori statistik | Volume data latih berkurang ±10% |
| Duplikat cabang (`KY069`→`KY011` Bekasi Galaxy; `TOD M1 Bandara`→`KY051`) diselesaikan lewat file override, bukan fuzzy matching | Fuzzy matching bisa salah diam-diam | Butuh pemeliharaan manual file override |
| Prefiks `Kota `/`Kabupaten ` **dipertahankan**, tidak dipangkas | Kota dan kabupaten punya pola permintaan berbeda | 8 nilai kota masih tebakan terbaik dari `Kecamatan`, belum dikonfirmasi pemilik data |
| `Kecamatan` **tidak** dipakai sebagai fitur | 56 nilai unik untuk 62 outlet — hampir identik dengan identitas outlet, terlalu jarang untuk digeneralisasi model global | Kehilangan granularitas lokasi yang mungkin berguna |

---

## 5. Tahap 5 — Panel harian padat (`build_panel.py`)

Setiap pasangan (item, cabang) di-*reindex* menjadi satu baris per hari
kalender, gap diisi `Kuantitas = 0`, kolom deskriptif di-*forward fill*.

**Keputusan kunci: rentang tanggal yang dipakai adalah rentang aktif masing-masing
pasangan** (tanggal transaksi pertama → terakhir miliknya sendiri), bukan rentang
penuh 2024–2025.

> Trade-off: memakai rentang penuh akan menciptakan bertahun-tahun "permintaan
> nol" palsu untuk cabang yang baru buka atau item yang baru diluncurkan —
> model akan belajar dari data yang tidak pernah ada. Konsekuensinya, pasangan
> yang berhenti aktif sebelum Desember 2025 otomatis tidak punya baris di
> periode uji; ini perilaku yang benar, bukan bug.

Kemudian `filter_min_history` **membuang pasangan dengan riwayat < 60 hari**
sebelum 2025-12-01 (`MIN_HISTORY_DAYS = 60`).

| Keputusan | Alasan | Trade-off |
|---|---|---|
| Ambang 60 hari | Fitur lag/rolling terpanjang butuh 28 hari; di bawah 60 hari rasio NaN terlalu tinggi | **842 pasangan terbuang dan sama sekali tidak mendapat ramalan** — masalah *cold start* yang secara sadar ditunda ke fase pemodelan |

### 5.1 Segmentasi pada periode outlet tutup

Reindex per pasangan di atas punya satu titik buta: kalau sebuah cabang berhenti
beroperasi selama beberapa bulan lalu buka lagi, `canonicalize_branch_names`
menyatukan kode lama dan barunya jadi satu rentang kontinu, dan **seluruh masa
tutup terisi `Kuantitas = 0`**. Hari-hari itu bukan permintaan nol; outletnya
tidak ada.

Solusinya: `dataset/outlet_closures.csv` mencatat interval
`[tanggal_tutup, tanggal_buka)` per cabang. `build_dense_panel` membuang tanggal
di dalam interval itu — **tidak menghasilkan baris sama sekali** — lalu memberi
nomor `segment_id` (1, 2, …) pada setiap blok tanggal aktif yang kontinu. Semua
fitur berbasis geser (`target_*`, `lag_*`, `roll_*`) dikelompokkan per
`(pasangan, segmen)`, sehingga tidak ada satu pun yang melintasi masa tutup.

**Contoh nyata — KY011 Bekasi Galaxy.** Kode lama bertransaksi 2024-01-01 s/d
2024-02-29, kode baru `KY069` mulai 2025-07-18; di antaranya outlet tutup total
16,5 bulan (dikonfirmasi pemilik data 2026-08-15, buka kembali di lokasi yang
sama). Sebelum diperbaiki, 505 hari itu terisi nol palsu:

| | Sebelum | Sesudah |
|---|---|---|
| Baris nol palsu | **17.640** (68,6% baris cabang ini) | 0 |
| Hari yang masuk rata-rata | 700 | 196 |
| `branch_avg_daily_qty` | 104,0 | **371,3** |
| `branch_demand_cv` | 1,863 | **0,502** |
| Peringkat volume | **#59 dari 59** (terkecil) | **#46** |

Model sebelumnya diberi tahu bahwa cabang ini yang terkecil dan paling tidak
stabil di seluruh jaringan. Keduanya keliru. `KY056 Tigaraksa` mengalami hal
serupa dalam skala lebih kecil (tutup sementara 2024-10-01 s/d 2024-11-21,
1.664 baris). Total **19.304 baris fabrikasi (1,27% dataset)** hilang dari data
latih — ini koreksi kualitas data, bukan pengurangan sampel tanpa sebab.

Sebuah **detektor** memindai gap transaksi ≥ 14 hari yang belum tercatat di
`outlet_closures.csv` dan mencetak peringatan setiap run. Detektor tidak pernah
menyegmentasi sendiri — file konfigurasi tetap satu-satunya otoritas, keputusan
tetap di tangan manusia.

| Keputusan | Alasan | Trade-off |
|---|---|---|
| Baris masa tutup **dihapus**, bukan diberi flag lalu dibuang belakangan | Kalau barisnya tetap ada, `lag_*` dan `roll_mean_*` sesudah buka tetap membaca nol-nol masa tutup | Perlu `segment_id` dan pengelompokan baru di 7 fungsi |
| Identitas cabang **tidak** dipecah | Memecahnya membuang kontinuitas relokasi dan menjadikan setiap cabang yang buka lagi kasus cold start | Segmen setelah buka mengalami pemanasan lag/rolling lagi (28 hari pertama NaN) — memang benar, lokasi/rezim barunya tidak diprediksi oleh lag sebelum tutup |
| Interval dari file manual, bukan deteksi otomatis | Heuristik "gap panjang = tutup" bisa salah membaca celah pelaporan | Butuh pemeliharaan manual; ditutup oleh peringatan detektor |

**Fakta penting yang lahir dari tahap ini:** karena `Kuantitas` mentah tidak
pernah bernilai 0 (`min = 1`, diverifikasi atas 692.993 baris), maka **setiap
`Kuantitas == 0` di panel padat pasti hari isian-gap, bukan transaksi nyata**.
Fakta ini dipakai tahap outlier untuk menghitung baseline hanya dari transaksi
riil tanpa perlu stage tambahan.

---

## 6. Tahap 6 — Fitur kalender (`calendar_features.py`)

Menghasilkan: hari-dalam-minggu, tanggal, bulan, penanda akhir pekan, penanda
hari libur nasional Indonesia, serta penanda + fitur kedekatan (*days until* /
*days since*) untuk **4 pendorong musim tinggi**: Ramadan/Idulfitri, Iduladha,
HUT RI (17 Agustus), dan Tahun Baru (1 Januari).

| Keputusan | Alasan | Trade-off |
|---|---|---|
| Fitur kedekatan disimpan sebagai **jarak hari mentah** (jendela ±14 hari, ±30 untuk Ramadan), bukan satu penanda "hari puncak" pilihan tangan | Biarkan model belajar sendiri di mana puncak permintaan setiap event berada | 10 kolom ini **84,6%–96,7% null** karena hanya terdefinisi di dalam jendela — menuntut strategi imputasi khusus di tahap 13 |
| Tanggal Ramadan/Idulfitri dari paket `holidays` (kalender ID), **dicek silang manual** dengan tanggal 2024/2025 yang diketahui | Bisnis makanan Timur Tengah sangat sensitif terhadap tanggal ini | Ketergantungan pada paket eksternal |
| Dijalankan **sebelum** penanganan outlier | Penanda event dibutuhkan untuk memutuskan lonjakan mana yang dikecualikan dari capping | Urutan pipeline jadi lebih terikat; aman karena fungsi ini murni bergantung pada `Tanggal` |
| Cakupan hanya 2024–2025 dengan `check_year_coverage` yang gagal keras | Lebih baik pipeline berhenti daripada diam-diam mengeluarkan fitur libur yang salah | **Harus diperluas sebelum data 2026 masuk** |

---

## 7. Tahap 7 — Penanganan lonjakan permintaan (`outlier_handling.py`)

Masalah yang ditemukan di EDA: mengurutkan baris berdasarkan `Kuantitas` mentah
bias terhadap satuan yang angkanya besar (`Porsi`, `PCS`) dan **melewatkan
lonjakan nyata pada item bersatuan kecil** (`Botol`, `Gr`). Contoh: 23 Feb 2025
di KY001, *Ayam Kebuli* dan *Rice Bowl 600 ml* melonjak bersama *Nasi Kebuli*
tetapi tidak pernah muncul di 20 besar absolut.

Solusinya ukuran **relatif**:

```
baseline_ratio = Kuantitas / median historis pasangan (item, cabang) itu sendiri
is_spike       = pair_eligible & (baseline_ratio >= 5.0)
should_cap     = is_spike & bukan di dalam jendela event musim tinggi
Kuantitas_capped = median * 5   bila should_cap, selain itu Kuantitas apa adanya
```

Parameter: `MIN_PAIR_HISTORY = 30` hari transaksi riil (di bawah itu pasangan
dinyatakan tidak layak dan tidak pernah di-cap), `SPIKE_RATIO_THRESHOLD = 5.0`.

### Tiga keputusan penting beserta alasannya

**(a) Target tetap mentah, input yang di-cap.**
`target_h1`…`target_h7` dan `target_lead_time_cumulative` dihitung dari
`Kuantitas` **mentah**; `lag_*`, `roll_*`, dan statistik cabang dihitung dari
`Kuantitas_capped`.

> Alasan: capping ditujukan pada kolom yang mendeskripsikan **perilaku masa
> lalu** — satu hari ekstrem tidak boleh mendominasi jendela rolling 28 hari.
> Namun label adalah tolok ukur evaluasi; meng-cap label akan membuat evaluasi
> **berbohong** tentang kemampuan model memprediksi lonjakan permintaan nyata.
> Trade-off: model dilatih dengan input yang "diperhalus" tetapi dievaluasi atas
> kenyataan penuh — memang lebih sulit, dan itu memang yang diinginkan.

**(b) Lonjakan di dalam jendela event dikecualikan dari capping.**

> Alasan: `baseline_ratio` dibandingkan terhadap median **sepanjang masa**, yang
> tertarik ke bawah oleh hari-hari sepi. Untuk item yang sangat musiman, lonjakan
> Ramadan yang nyata dan berulang akan terbaca sebagai rasio sangat tinggi;
> meng-cap-nya setiap tahun justru meratakan pola musiman yang seluruh fitur
> kalender dibangun untuk dipelajari. Trade-off: kesalahan input data yang
> kebetulan jatuh di jendela event lolos tanpa di-cap.

**(c) Pengecualian dibatasi pada 4 event musim tinggi, bukan `is_national_holiday`.**

> Lonjakan pada hari libur nasional sembarang yang tidak terkait 4 event itu
> tetap dianggap kandidat anomali.

**Penjagaan kebocoran:** `pair_median` dan `pair_eligible` dihitung **hanya dari
periode latih** (`Tanggal < 2025-12-01`) lalu dibekukan dan di-*merge* ke train
maupun test.

**Di luar cakupan (sadar):** baseline musiman yang lebih halus (median per
hari-dalam-minggu) dinilai belum perlu; `is_spike`/`baseline_ratio` hanya
diproduksi sebagai kolom, cara pemakaiannya (mis. sebagai bobot sampel)
diserahkan ke tahap pemodelan.

---

## 8. Tahap 8 — Rekayasa fitur (`prepare_forecast_data.py`)

Urutan di dalam `engineer_features()`:

| Langkah | Output | Catatan penting |
|---|---|---|
| `add_targets` | `target_h1`…`target_h7` | Dari `Kuantitas` **mentah**, digeser 1–7 hari ke depan, dikelompokkan per (pasangan, segmen) |
| `apply_region_features` | `kawasan`, `hari_pengiriman`, `lead_time_days` | Lead time **bervariasi per baris** |
| `apply_outlet_features` | `kota`, `has_shopee`, `has_gofood`, `has_grabfood`, `can_order_online` | Master data statis, tanpa risiko kebocoran |
| `add_relocation_feature` | `days_since_relocation` | Negatif sebelum pindah, 0 di hari pindah, positif sesudah |
| `add_lead_time_target` | **`target_lead_time_cumulative`** | Target bisnis utama |
| `add_lag_features` | `lag_{1,2,3,7,14,21,28}` | Dari `Kuantitas_capped` |
| `add_rolling_features` | rata-rata & std rolling 7/14/28 hari | Dari `Kuantitas_capped`, **digeser 1 hari** sebelum jendela dihitung |
| `compute_branch_stats` / `apply_branch_stats` / `add_branch_age_days` | `branch_avg_daily_qty`, `branch_demand_cv`, `branch_volume_tier`, `branch_age_days` | Tiga yang pertama dibekukan dari periode latih |

### 8.1 Lead time yang bervariasi

Sebelumnya `lead_time_days` adalah konstanta datar `4`. Aturan bisnis
sebenarnya bergantung pada hari transaksi **dan** kawasan:

```
compute_lead_time_days(hari, {hari_kirim}) = d terkecil dalam 1..7
                                             sehingga (hari + d) % 7 ∈ hari_kirim
```

Selalu **maju ketat**: jika hari transaksi kebetulan hari kirim, hasilnya adalah
jarak ke kejadian **berikutnya**, bukan 0. Contoh: Kawasan 1, transaksi Senin →
3 (ke Kamis); transaksi Kamis → 4 (ke Senin berikutnya).

| Keputusan | Alasan | Trade-off |
|---|---|---|
| `parse_delivery_days` melempar `ValueError` untuk token tak dikenal | Perubahan format atau salah ketik di `outlet_mapping.csv` harus langsung terlihat, bukan diam-diam merusak `lead_time_days` | Pipeline gagal keras karena satu typo — disengaja |
| Dihitung per kombinasi unik `(hari, hari_pengiriman)` lalu di-*merge*, bukan `.apply()` per baris | Hanya ada segelintir kombinasi vs 1,5 juta baris | Kode sedikit lebih rumit, jauh lebih cepat |
| Cabang tak cocok → `NaN`, bukan nilai default | Lebih baik kosong daripada salah | Baris tersebut kehilangan target utamanya |

### 8.2 Target kumulatif lead time

`target_lead_time_cumulative` = jumlah `Kuantitas` mentah pada jendela
**maju ketat** `(H+1 .. H+lead_time_days)` dalam pasangan yang sama.

Implementasi: untuk setiap nilai `w` yang muncul pada `lead_time_days`, hitung
kolom jumlah-maju via *reverse* → `rolling(w).sum()` → *reverse* kembali per
grup, lalu pilih kolom yang tepat per baris dengan `np.select` — menghindari
loop Python atas 1,5 juta baris sambil tetap mendukung jendela berukuran
variabel. Baris yang jendelanya melewati tanggal terakhir data diberi `NaN`.

### 8.3 Fitur relokasi

Sembilan pemetaan relokasi tercatat di `docs/outlet_relocation_notes.md` (cabang
lama → cabang baru); **4 di antaranya punya tanggal pasti**, 5 sisanya masih
perkiraan batas-bawah. Karena `canonicalize_branch_names` berjalan
**sebelum** join outlet, `kota`/`kawasan` sebuah cabang yang pindah
merefleksikan lokasi **saat ini** untuk **seluruh** riwayatnya — termasuk baris
sebelum pindah yang sebenarnya tercatat di kota lain. `days_since_relocation`
ada supaya tahap pemodelan bisa memperhitungkan pergeseran rezim ini.

> Alternatif yang ditolak: membuang riwayat pra-relokasi. Ditolak karena itu
> membuang justru satu-satunya contoh transisi yang dimiliki model. Catatan:
> 5 tanggal relokasi masih berupa perkiraan batas-bawah.

### 8.4 Aturan anti-kebocoran (leakage) — ringkas

Bagian ini paling layak dikutip di laporan karena menyangkut validitas metodologi.

| Mekanisme | Cara kerja |
|---|---|
| Rolling digeser 1 hari | Nilai hari ini tidak pernah masuk ke statistik rolling-nya sendiri |
| Statistik cabang dibekukan dari periode latih | Permintaan Desember sebuah cabang tidak pernah mempengaruhi fitur cabang itu sendiri |
| Baseline outlier dari periode latih | Baris uji dibandingkan terhadap baseline yang tidak pernah melihat periode uji |
| Segmentasi permintaan (tahap 13) dari periode latih | Perilaku masa depan tidak bocor menjadi fitur |
| Mapping kategori & scaler di-*fit* hanya di data latih (per fold) | Statistik Desember tidak bocor ke fold Juli |
| Target selalu jendela **maju ketat** | Tidak pernah menyertakan hari berjalan |
| Fitur bergeser dikelompokkan per (pasangan, segmen) | Lag, rolling, dan target tidak pernah melintasi periode outlet tutup — dua sisi masa tutup tidak diperlakukan sebagai hari berurutan |
| `branch_age_days` | Aman secara inheren — hanya membaca masa lalu cabang itu sendiri |

---

## 9. Tahap 9–12 — Ekspor, split, dan QA

**Split waktu tunggal:** train = seluruh baris sebelum 2025-12-01; test =
Desember 2025.

| Keputusan | Alasan | Trade-off |
|---|---|---|
| Split berbasis waktu, bukan acak | Data deret waktu; split acak akan bocor secara masif | Hanya satu bulan uji |
| Target yang jendelanya melewati 2025-12-31 dibiarkan `NaN`, bukan mempersempit jendela uji | Mempertahankan seluruh 55.046 baris uji untuk horizon pendek | Horizon panjang punya cakupan lebih sedikit di ujung bulan |
| Format **Parquet**, bukan CSV | 1,5 juta baris; parquet mempertahankan tipe data dan jauh lebih ringkas | Tidak bisa dibuka langsung di Excel |
| `featured.parquet` disimpan sebagai artefak antara | Memisahkan "bersih + berfitur" dari "sudah displit" | Satu file tambahan yang harus dijaga kesinkronannya |

**QA (`run_qa_checks()`)** — 10 asersi, dipanggil **baik dari notebook maupun dari
script**: tidak ada `Kuantitas` negatif; tidak ada duplikat (item, cabang,
tanggal); `Kuantitas_capped` tidak pernah melebihi nilai mentah; tidak ada
`kota == "Unknown"`; tidak ada cabang tanpa `kawasan`; tidak ada cabang yang
memetakan ke lebih dari satu kota; **tidak ada baris yang jatuh di dalam interval
tutup yang tercatat**; **`segment_id` mulai dari 1 dan kontinu per pasangan**;
**tidak ada lubang tanggal di dalam satu segmen** (invarian kepadatan yang
diandalkan `shift`); dan `main()` memastikan seluruh 64 kolom
`FEATURED_COLUMNS` hadir.

### Pelajaran penting: *drift* antara notebook dan script

Ditemukan saat EDA 2026-08-12: `featured.parquet` hanya punya 62 kolom
sementara `train`/`test` punya 63. Penyebabnya bukan satu kolom yang terlupa,
melainkan **dua jalur kode sama-sama meng-encode urutan langkah pipeline** —
notebook menuliskan ulang sembilan pemanggilan fungsi secara manual dan tidak
pernah diperbarui ketika `add_relocation_feature` ditambahkan ke script.
Notebook berjalan terakhir dan menimpa output script yang benar.

**Aturan yang ditetapkan:** hanya ada **satu tempat** yang mendefinisikan urutan
langkah, yaitu `build_featured_dataset()`. Notebook tetap menjadi penggerak
proses utama dan tetap boleh menulis parquet, tetapi **memanggil fungsi
komposit** alih-alih menuliskan ulang langkah-langkahnya. Asersi QA yang tadinya
hanya ada di notebook dipindahkan ke `run_qa_checks()` agar jalur script pun
terverifikasi.

> Trade-off: notebook kehilangan sebagian "keterlihatan langkah demi langkah"
> yang berguna untuk eksplorasi. Dinilai sepadan — *drift* semacam ini
> menghasilkan hasil eksperimen yang salah tanpa satu pun pesan error.

---

## 10. Tahap 13 — Modeling preprocessing (`utils/modeling_prep.py`)

Tahap ini mengubah `featured.parquet` (64 kolom) menjadi
`model_input.parquet` (76 kolom) — satu sumber kebenaran yang dikonsumsi
ketiga keluarga model lewat adapter tipis.

**Mengapa file terpisah, bukan menambah kolom di `featured.parquet`?**
Karena keduanya punya siklus hidup berbeda: `featured.parquet` menyatakan
**fakta tentang data** dan berguna untuk analisis apa pun, sedangkan
`model_input.parquet` meng-encode **keputusan eksperimen** (batas fold, skema
encoding, definisi segmen) yang akan berubah berkali-kali selama eksperimen.
Dengan memisahkannya, pipeline data-prep yang stabil tidak perlu dijalankan
ulang setiap kali strategi fold berubah.

Kelima fungsi bersifat murni (DataFrame masuk → DataFrame keluar, tanpa I/O
tersembunyi) sehingga bisa diuji unit dan dipanggil dari notebook.

### 13.1 `add_event_flag()` → `is_event_driven`

Membaca `dataset/event_driven_items.csv` (70 baris, satu per SKU), diisi pemilik
data. Saat ini **5 SKU** ditandai `true`.

**Draf diturunkan dari bentuk permintaan, bukan dari nama item**, karena nama
terbukti tidak andal di kedua arah. Tanda tangan pesanan acara adalah
**jarang tetapi borongan**: `ADI ≥ 50` (bergerak sekali per 50+ hari) **dan**
rata-rata ≥ 30 unit saat bergerak. Item lambat biasa punya sifat pertama tetapi
tidak yang kedua — mereka bergerak satu-dua unit saja.

Dua koreksi yang dihasilkan data terhadap tebakan berbasis nama:

- **Box Loyang (`PCG-00006/07/08`) hampir pasti *bukan* event-driven** meski
  namanya mengandung "Box": statistiknya identik dengan Loyang biasa dan Cup
  Sambal Loyang (28,4% hari nol, ADI 1,6, 59 cabang) — ketiganya bergerak
  sebagai bundel.
- **`PCG-00027` (Mika Bento) dan `PCG-00028` (Cup 60 ml) kemungkinan *memang*
  event-driven** meski namanya terdengar seperti kemasan rutin: ADI 58,6 / 88,3
  dengan rata-rata 38,5 / 84,3 unit saat bergerak — bentuk yang sama dengan item
  aqiqah yang sudah terkonfirmasi.

Aturan berbasis nama juga akan salah menangani `Lunch Box` (`PCG-00001`) yang
merupakan kemasan harian biasa, versus `Lunch Box Aqiqah` (`PCG-00002`) yang
event-driven.

> **Batas informasi, bukan kegagalan model.** Data pemesanan aqiqah tidak ada /
> tidak dapat diakses. Permintaan SKU event-driven digerakkan booking pelanggan,
> bukan pola historis — **tidak ada fitur lag atau rolling yang bisa
> memprediksinya**. Ini harus dinyatakan eksplisit di laporan sebagai langit-langit
> informasi.

### 13.2 `classify_pairs()` → `demand_segment`

Klasifikasi **Syntetos-Boylan** per pasangan dari dua statistik: **ADI**
(rata-rata jarak antar hari permintaan non-nol) dan **CV²** (kuadrat koefisien
variasi kuantitas non-nol).

| | CV² < 0,49 | CV² ≥ 0,49 |
|---|---|---|
| **ADI < 1,32** | `smooth` | `erratic` |
| **ADI ≥ 1,32** | `intermittent` | `lumpy` |

Distribusi aktual (per pasangan): intermittent 43,9%, lumpy 31,7%, erratic
13,6%, smooth 10,7% — **75,7% pasangan bersifat intermittent atau lumpy**.

Dihitung **hanya dari periode latih**. Kolom ini punya dua fungsi: sebagai input
model, **dan** sebagai sumbu pelaporan metrik.

> Alasan pelaporan per segmen: MAE global didominasi pasangan yang mayoritas
> nol, sehingga sebuah model bisa tampak menang padahal ia hanya unggul di
> tempat di mana menebak nol itu mudah.

**Mengapa `is_event_driven` dan `demand_segment` keduanya diperlukan?** Karena
kejarangan (*sparsity*) dan sifat digerakkan-acara adalah **dua sumbu berbeda
yang kebetulan beririsan**. `Mika Bento` jarang di sebagian cabang karena cabang
itu memang jarang memakainya, bukan karena menunggu acara. Menyatukan keduanya
akan mengajarkan sinyal yang salah kepada model.

Konteks intermitensi dari EDA: **54,8% baris `Kuantitas` bernilai nol** (44,9%
untuk target). 776 pasangan (23,6%) punya >90% hari nol tetapi hanya menyumbang
**0,47% volume total**. Volume sangat terkonsentrasi: 50% volume berasal dari
112 pasangan (3,4%), 80% dari 276 pasangan (8,4%).

> Keputusan: **semua pasangan tetap dilatih** (tidak ada yang dibuang karena
> jarang), tetapi diberi penanda dan segmen. Trade-off: model harus menangani
> distribusi target yang sangat timpang; imbalannya, pasangan jarang bervolume
> tinggi seperti *Lunch Box Aqiqah* (97% hari nol tetapi rata-rata 87 pcs saat
> bergerak) tidak hilang dari sistem.

### 13.3 `assign_folds()` → `fold_id` (validasi walk-forward)

```
                 TRAIN (2024-01 → 2025-11)               TEST
fold 1  ████████████████████░░ Jul
fold 2  ██████████████████████░░ Agu
fold 3  ████████████████████████░░ Sep
fold 4  ██████████████████████████░░ Okt
fold 5  ████████████████████████████░░ Nov
FINAL   ██████████████████████████████  🔒 Des 2025

█ latih   ░ validasi   🔒 dibuka sekali, di akhir
```

`fold_id` menandai fold di mana suatu baris menjadi **himpunan validasi**
(1 = Juli 2025 … 5 = November 2025). Bernilai `NaN` untuk semua baris di luar
lima bulan itu — baik baris sebelum Juli 2025 (selalu data latih) maupun baris
Desember 2025 (himpunan uji terkunci). Data latih untuk fold *k* adalah seluruh
baris bertanggal sebelum awal bulan fold *k*. Jumlah baris validasi per fold:
77.318 / 76.266 / 71.066 / 69.510 / 61.407.

| Keputusan | Alasan | Trade-off |
|---|---|---|
| Berpindah dari holdout tunggal (desain awal) ke **walk-forward 5 fold** | Tuning dan pemilihan pemenang pada bulan yang sama dengan angka final akan membuat angka final optimistis. Desember juga atipikal (Natal/Tahun Baru) | Biaya komputasi 5× untuk setiap kandidat model |
| Desember 2025 **dibuka tepat satu kali** | Menjaga integritas angka final | Tidak bisa mengiterasi berdasarkan hasil Desember |
| Satu kolom `fold_id`, bukan duplikasi baris per fold | Merepresentasikan skema *expanding window* penuh tanpa menggandakan 1,5 juta baris | Logika masking harus benar (diuji unit) |

### 13.4 `impute_features()` — imputasi yang mempertahankan makna

Ini adalah bagian dengan potensi bug paling berbahaya di seluruh pipeline.

| Kolom | % null | Arti null | Imputasi |
|---|---|---|---|
| 10 kolom kedekatan event | 84,6–96,7% | di luar jendela kedekatan | **`99`** |
| `days_since_relocation` | 84,4% | cabang tidak pernah pindah | `0` **+ indikator `was_relocated`** |
| `baseline_ratio` | 14,4% | pasangan tak layak di-cap (<30 hari riil) | `1.0` **+ indikator `has_baseline`** |
| `lag_*`, `roll_*` | 1,4–5,5% | pemanasan deret | dihapus oleh pemotongan L=28 |

> **Mengapa bukan 0?** Mengimputasi `days_until_eid_al_fitr` dengan `0` berarti
> menyatakan *"hari ini Idulfitri"* pada 96% baris — mengubah fitur berguna
> menjadi fitur yang aktif merusak. Sentinel harus berada **di luar** jendela dan
> mempertahankan makna ordinal; `99` aman melampaui semua jendela termasuk
> Ramadan ±30.
>
> **Mengapa perlu kolom indikator?** Karena `0` adalah nilai sah bagi
> `days_since_relocation` (artinya "hari relokasi"). Tanpa indikator, "tidak
> pernah pindah" dan "pindah hari ini" menjadi tak terbedakan.

Terdapat unit test khusus yang memastikan null `days_until_*` menjadi sentinel,
**bukan** `0`.

### 13.5 `encode_categoricals()` → indeks integer + mapping tersimpan

Semua kolom kategorikal menjadi indeks integer; mapping ditulis ke
`dataset/model_ready/category_mapping.json`. Kardinalitas semuanya kecil:
`Kode Barang` 70, `Nama Cabang` 59, `kota` 16, `Kategori Barang` 8,
`demand_segment` 4, `branch_volume_tier` 4, `hari_pengiriman` 2.

Indeks integer melayani ketiga keluarga model: XGBoost lewat
`enable_categorical`, Random Forest lewat ekspansi one-hot dari indeks, LSTM
lewat lapisan *embedding*.

| Keputusan | Alasan | Trade-off |
|---|---|---|
| Mapping di-*fit* **hanya di data latih** dan **dipersistensi** | Ini syarat kebenaran, bukan kerapian: SCM menjalankan ini mingguan atas data baru; tanpa mapping tersimpan, cabang ke-60 yang buka bulan depan akan menggeser **seluruh** indeks dan membatalkan model **tanpa satu pun error** | File tambahan yang wajib ikut dikirim bersama model |
| Kategori tak dikenal → indeks `<UNKNOWN>` yang direservasi (indeks 0) | Inferensi tidak gagal saat ada SKU/cabang baru | Prediksi untuk kategori baru bersifat generik |
| Encoding ditunda sampai tahap ini (bukan di data-prep) | Setiap algoritma menginginkan encoding berbeda; menahannya menjaga `featured.parquet` tetap netral | Satu tahap tambahan |

### 13.6 Catatan transformasi target

`target_lead_time_cumulative` sangat menceng ke kanan (persentil 99 = 488,
maksimum 3.067). Biasanya `log1p` bermasalah untuk regresi karena rata-rata dari
log bukan log dari rata-rata.

**Regresi kuantil dikecualikan dari masalah ini.** Kuantil bersifat *equivariant*
terhadap transformasi monoton: kuantil 0,9 dari `log1p(y)`, dilewatkan `expm1`,
**persis** sama dengan kuantil 0,9 dari `y`. Pelatihan karenanya bisa dilakukan
pada skala log — jauh lebih stabil, terutama untuk LSTM — dan dibalik tanpa
bias. Ini konsekuensi menguntungkan dari keputusan memakai *pinball loss*.

---

## 11. Tahap 14 — Adapter dan kontrak lintas-adapter

### 14.1 Panjang lookback: 28 hari

| L | Baris latih hilang | **Baris uji hilang** |
|---|---|---|
| 7 | 1,37% | 0 |
| 14 | 2,74% | 0 |
| **28** | **5,48% (83.412)** | **0** |
| 56 | 10,95% | 0 |

Tidak ada baris uji yang hilang pada lookback mana pun — setiap pasangan yang
hidup di Desember 2025 punya riwayat jauh lebih dari 28 hari. Menegakkan kontrak
"baris identik" karenanya **praktis gratis**.

28 dipilih di atas 14 karena `lag_28` bernilai null pada persis 5,48% baris yang
sama: memotong pemanasan L=28 sekaligus menghapus **seluruh** NaN lag/rolling.
Satu pemotongan menyelesaikan dua masalah.

### 14.2 `to_tabular()` — XGBoost & Random Forest

Membuang baris pemanasan (28 hari pertama setiap pasangan), lalu mengembalikan
`X`, `y`, `groups` (pasangan), dan `fold_id`. **Tanpa penskalaan.** NaN
dibiarkan — XGBoost menanganinya secara native; Random Forest mendapat imputasi
ringan di dalam adapter.

### 14.3 `to_sequences()` — LSTM

Jendela geser per pasangan menghasilkan tensor `(n_samples, 28, n_features)`.

- **Penskalaan**: standardisasi per fitur, **di-*fit* hanya pada data latih tiap
  fold** dan diterapkan ke validasi fold itu. Di-*fit* ulang per fold, bukan
  sekali secara global, agar statistik Desember tidak bocor ke fold Juli.
  Parameter scaler dipersistensi ke `scaler_params.json` untuk inferensi mingguan.
- **Jendela yang melintasi tanggal relokasi**: **tidak perlu penanganan khusus.**
  `days_since_relocation` sudah menjadi fitur per baris di dalam sekuens,
  sehingga LSTM mengamatinya menyeberang dari negatif ke positif di tengah
  jendela dan dapat mempelajari pergeseran rezimnya sendiri. Membuang jendela
  tersebut justru menghapus satu-satunya contoh transisi yang ada.

### 14.4 Kontrak — mengapa ini yang menjaga validitas perbandingan

`validate_contract()` menegakkan tiga properti dengan asersi keras:

1. `to_tabular()` dan `to_sequences()` mengembalikan himpunan `(pasangan, tanggal)`
   yang **identik** — bukan sekadar jumlah baris yang sama.
2. Vektor `y` keduanya identik nilainya.
3. Penetapan `fold_id` keduanya identik.

> Tanpa kontrak ini, kesimpulan *"LSTM 8% lebih baik"* bisa sebenarnya berarti
> *"LSTM dievaluasi atas 5% baris yang berbeda"*. Keadilan perbandingan
> ditegakkan **secara struktural**, bukan lewat disiplin manusia.

---

## 12. Ringkasan keputusan besar dan trade-off-nya

| # | Keputusan | Pilihan yang diambil | Alternatif yang ditolak | Trade-off |
|---|---|---|---|---|
| 1 | Target utama | `target_lead_time_cumulative` | `target_h1` (besok saja) | Jendela variabel per baris membuat implementasi lebih rumit; sepadan karena inilah angka yang dipakai SCM |
| 2 | Target bantu | `target_h1`…`target_h4` dipertahankan | Membuang semua horizon harian | `target_h5`–`h7` menjadi mubazir (`lead_time_days` tidak pernah > 4); dipertahankan demi dekomposisi harian untuk explainability |
| 3 | Fungsi kerugian | **Kuantil (pinball)**, default 0,9 | MSE / MAE | Regresi rata-rata *stockout* ~50% waktu secara konstruksi. Trade-off: overstock naik; disengaja karena stockout lebih mahal |
| 4 | Model | Satu model global per algoritma | Satu model per deret (2.979 model) | Deret saling berbagi kekuatan statistik; model global tidak bisa menghafal keunikan tiap deret |
| 5 | Granularitas | Harian, per (item, cabang) | Mingguan / per cabang | Data jauh lebih jarang di level harian; sesuai kebutuhan operasional |
| 6 | Validasi | Walk-forward 5 fold + Desember terkunci | Holdout tunggal | Biaya komputasi 5×; menghilangkan bias optimistis |
| 7 | Arsitektur | Tabel fitur bersama + adapter tipis | Tiga pipeline terpisah per model | Adapter menambah lapisan; menjamin perbandingan yang sah |
| 8 | Pasangan jarang | Semua dilatih + penanda + segmen | Membuang pasangan >90% nol | Distribusi target sangat timpang; tidak kehilangan item bervolume tinggi yang jarang |
| 9 | Skala target | `log1p` diperbolehkan | Skala asli | Aman **hanya** karena memakai loss kuantil |
| 10 | Peran notebook | Tetap penggerak utama, tetapi memanggil fungsi komposit | Notebook menuliskan ulang langkah | Kehilangan sebagian keterlihatan langkah; menghilangkan risiko *drift* |
| 11 | Imputasi | Sentinel yang mempertahankan makna + indikator | `fillna(0)` | Dua kolom tambahan; mencegah fitur berubah menjadi racun |
| 12 | Encoding | Indeks integer, mapping dipersistensi | One-hot langsung / encoding on-the-fly | Butuh manajemen artefak; wajib untuk inferensi mingguan yang stabil |
| 13 | Periode outlet tutup | Panel disegmentasi, baris masa tutup dihapus, `segment_id` mengelompokkan semua fitur bergeser | Beri flag lalu buang belakangan / pecah identitas cabang | Satu kolom tambahan dan pemanasan lag ulang setelah buka; imbalannya 19.304 baris fabrikasi hilang dan statistik cabang jadi benar |

---

## 13. Artefak keluaran

| File | Isi |
|---|---|
| `dataset/dataset.csv` | Gabungan 5 periode mentah |
| `dataset/model_ready/featured.parquet` | 1.503.564 × 64 — bersih + berfitur, belum displit |
| `dataset/model_ready/train.parquet` | 1.448.518 × 64 |
| `dataset/model_ready/test.parquet` | 55.046 × 64 |
| `dataset/model_ready/model_input.parquet` | 1.503.564 × 76 — sumber kebenaran untuk pemodelan |
| `dataset/model_ready/category_mapping.json` | Mapping kategori → indeks (dari data latih) |
| `dataset/model_ready/scaler_params.json` | Parameter standardisasi per fold |

**Cara menjalankan**

```bash
# Pipeline data-prep penuh (tahap 1–12)
.venv/bin/python3 -m utils.prepare_forecast_data

# Modeling preprocessing (tahap 13–14)
.venv/bin/python3 -m utils.modeling_prep

# Lewat notebook
jupyter nbconvert --to notebook --execute --inplace \
  notebook/data-processing.ipynb notebook/train_test_split.ipynb \
  notebook/modeling_prep.ipynb

# Seluruh unit test (274 test)
.venv/bin/python3 -m unittest discover -p "test_*.py" -v
```

---

## 14. Strategi pengujian

Ditulis dengan pendekatan **TDD** (test gagal lebih dulu), satu berkas test per
modul pipeline. Cakupan yang secara khusus dirancang untuk melindungi keputusan
di atas:

- **Anti-kebocoran** — `classify_pairs()` menghasilkan segmen identik antara
  input train-only dan input penuh; `assign_folds()` tidak pernah menempatkan
  tanggal validasi di dalam rentang latih fold yang sama; scaler benar-benar
  di-*fit* ulang per fold.
- **Imputasi** — null `days_until_*` menjadi sentinel, **bukan** `0`.
- **Kontrak adapter** — kedua adapter mengembalikan himpunan `(pasangan,
  tanggal)` identik; merusaknya dengan sengaja **harus** menggagalkan asersi.
- **Stabilitas mapping** — kategori tak dikenal memetakan ke `<UNKNOWN>` tanpa
  menggeser indeks yang sudah ada.
- **Target tidak pernah melihat kolom yang di-cap** — `target_h*` pada baris yang
  di-cap tetap sama dengan nilai mentah yang digeser.

---

## 15. Keterbatasan dan hal yang masih terbuka

Bagian ini penting untuk bab "keterbatasan penelitian" di laporan.

**Menghambat penyelesaian (perlu konfirmasi pemilik data)**

1. `dataset/event_driven_items.csv` masih **draf** yang diturunkan dari bentuk
   permintaan; 14 dari 70 SKU butuh keputusan sungguhan.
2. **Target service level** (kuantil mana yang dilatih — default 0,9, mungkin
   lebih tinggi untuk FG daripada Packaging) belum dikonfirmasi.

**Ada default kerja, tetapi asumsi yang salah berbiaya mahal**

3. 393 pasangan berhenti muncul di 2025Q4 — discontinued asli atau celah
   pelaporan? Saat ini diperlakukan sebagai discontinued asli.
4. 1.059 pasangan mati sebelum Desember — dilatih tetapi tidak pernah dievaluasi
   (tidak punya baris Desember). Himpunan uji karenanya **tidak berisi kasus
   cold start** sama sekali (0 pasangan baru di Desember).
5. 842 pasangan terbuang oleh `MIN_HISTORY_DAYS = 60` — **tidak mendapat ramalan
   sama sekali**; strategi fallback belum diputuskan.
6. `kawasan = 2` untuk Bintara, Citayam, dan Grand Wisata Bekasi masih
   **disimpulkan** dari pola cabang Kota Depok/Bekasi lain, belum dikonfirmasi.
7. 8 nilai `Kota Override` masih tebakan terbaik dari `Kecamatan`.

**Ditunda, tidak menghambat**

7b. `KY073 - Kebuli Yaman Cilebut` buka 2025-12-19 dan masih beroperasi, tetapi
    **tidak punya satu hari pun sebelum cutoff 2025-12-01**, sehingga terbuang
    `filter_min_history`. Menurunkan `MIN_HISTORY_DAYS` tidak menolong — ambang
    berapa pun tetap mengecualikannya. Ia masuk sendirinya begitu punya ≥60 hari
    sebelum cutoff berikutnya, tanpa perubahan kode.
7c. `KY068 - Kebuli Yaman Kramatwatu` punya gap 13 hari (2025-06-28 s/d
    2025-07-10) — persis di bawah ambang peringatan 14 hari. Perlu dikonfirmasi
    apakah tutup sementara atau celah pelaporan.
7d. `Kebuli Yaman Cikarang Pusat` masih tutup; begitu `tanggal_buka` diketahui,
    isi ke `outlet_closures.csv` **dan** perbarui `RELOCATION_DATES` secara
    manual (tidak diturunkan otomatis — lihat spec).
8. 5 tanggal relokasi masih berupa perkiraan batas-bawah.
9. `calendar_features.py` hanya mencakup 2024–2025; **harus diperluas sebelum
   data 2026 masuk**, atau pipeline gagal keras (disengaja).
10. `FGS.00048` (Kambing Oven) hanya 4 unit dalam 18 bulan di 1 cabang —
    kandidat untuk dimasukkan `EXCLUDED_ITEMS`.

**Keterbatasan metodologis yang tidak bisa dihilangkan oleh preprocessing**

- SKU event-driven punya **langit-langit informasi**: tanpa data booking, tidak
  ada fitur historis yang bisa memprediksinya. Ini harus dinyatakan sebagai
  batas data, bukan kegagalan model.
- Periode uji hanya satu bulan (Desember 2025) yang secara musiman atipikal;
  itulah sebabnya walk-forward 5 fold dipakai untuk pemilihan model dan Desember
  hanya untuk angka final.

---

## 16. Glosarium

| Istilah | Arti |
|---|---|
| **Pasangan / pair** | Kombinasi (Kode Barang, Nama Cabang) — satu deret waktu |
| **Panel padat** | Tabel dengan satu baris per pasangan per hari kalender, gap diisi 0 |
| **Segmen** | Blok tanggal aktif kontinu milik satu pasangan; dipisahkan oleh periode outlet tutup. Ditandai `segment_id` |
| **Lead time** | Jumlah hari dari tanggal transaksi sampai hari pengiriman berikutnya |
| **ADI** (*Average Demand Interval*) | Rata-rata jarak hari antar permintaan non-nol |
| **CV²** | Kuadrat koefisien variasi kuantitas non-nol |
| **Intermittent / lumpy / erratic / smooth** | Empat kelas Syntetos-Boylan berdasarkan ADI dan CV² |
| **Leakage** | Informasi masa depan bocor ke fitur/proses latih, membuat evaluasi terlalu optimistis |
| **Walk-forward** | Validasi deret waktu dengan jendela latih yang membesar dan validasi selalu di masa depan |
| **Pinball loss** | Fungsi kerugian regresi kuantil; asimetris antara prediksi kurang dan lebih |
| **Capping** | Memotong nilai lonjakan ekstrem ke ambang tertentu |
| **Sentinel** | Nilai khusus yang menandai "tidak berlaku" tanpa menyamar sebagai nilai sah |

---

## 17. Rujukan

- `docs/pipeline-overview.md` — ikhtisar 14 tahap (Inggris)
- `docs/superpowers/specs/2026-07-18-merge-dataset-design.md`
- `docs/superpowers/specs/2026-07-21-forecast-data-prep-design.md`
- `docs/superpowers/specs/2026-07-23-outlet-location-features-design.md`
- `docs/superpowers/specs/2026-08-08-outlier-handling-design.md`
- `docs/superpowers/specs/2026-08-08-lead-time-integration-design.md`
- `docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md`
- `docs/superpowers/specs/2026-08-15-outlet-lifecycle-handling-design.md`
- `docs/todolist-data-preprocessing.md` — konfirmasi pemilik data
- `docs/outlet_relocation_notes.md` — catatan relokasi cabang
