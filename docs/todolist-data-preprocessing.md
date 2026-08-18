# To-Do List — Data Preprocessing

Disusun dari tiga sumber: `notebook/eda.ipynb` (EDA atas `dataset/dataset.csv`, 693.563 baris,
2024-01-01 s/d 2025-12-31, dijalankan ulang 2026-08-07 — hasil identik dengan run sebelumnya,
lihat catatan di bagian Status), `docs/pipeline-overview.md` (state implementasi pipeline), dan
pembacaan langsung source code `utils/*.py` + `test/*.py` (2026-08-07) untuk memverifikasi mana
temuan EDA yang **sudah** ditangani di kode vs yang **belum**.

## Status pipeline saat ini

Pipeline end-to-end **sudah terimplementasi penuh**, dari CSV mentah sampai
`dataset/model_ready/{train,test}.parquet` — lihat `docs/pipeline-overview.md` §2 untuk urutan
9 stage-nya dan `utils/prepare_forecast_data.py::main()` untuk orkestrasinya. Ini lebih maju
dari yang disebut `CLAUDE.md` ("will also include normalize_items.py, build_panel.py, ...") —
keempat modul itu sudah ada dan sudah di-wire jadi satu pipeline yang jalan. Jadi to-do di
bawah ini **bukan** "bangun pipeline-nya", tapi tiga kategori pekerjaan yang tersisa: integrasi
fitur yang belum selesai, konfirmasi keputusan yang sudah di-hardcode di kode, dan gap
engineering yang belum tergarap.

## ✅ Penanganan siklus hidup outlet (selesai, 2026-08-15)

Lihat `docs/superpowers/specs/2026-08-15-outlet-lifecycle-handling-design.md`. Hari-hari saat
sebuah cabang tidak beroperasi dulunya difabrikasi jadi baris `Kuantitas = 0` oleh reindex
per-pasangan di `build_dense_panel` — 19.304 baris semacam itu ada di dataset, dan membuat
`branch_avg_daily_qty` KY011 Bekasi Galaxy salah 3,6× (104,0 vs 371,3), menempatkannya sebagai
cabang terkecil dari 59 padahal peringkat sebenarnya #46.

- [x] `dataset/outlet_closures.csv` + `outlet_features.load_closures()` — interval
  `[tanggal_tutup, tanggal_buka)` per cabang kanonik, gagal keras untuk tanggal tak valid,
  terbalik, atau tumpang tindih.
- [x] `build_dense_panel(closures=...)` membuang tanggal masa tutup dan memberi `segment_id`;
  `closures=None` mereproduksi perilaku lama persis.
- [x] `SEGMENT_COLS` dioper ke `add_targets`, `add_lag_features`, `add_rolling_features`,
  `add_lead_time_target`, `drop_warmup_rows`, `to_tabular`, `to_sequences`.
- [x] `detect_unrecorded_gaps()` memperingatkan gap ≥14 hari hilang yang belum tercatat.
- [x] 3 asersi QA baru + 32 unit test baru (274 → 306).

**Masih terbuka:**

- [x] **`KY068 - Kebuli Yaman Kramatwatu` gap 13 hari** (2025-06-28 s/d 2025-07-10) —
  **tutup sementara, dikonfirmasi pemilik data 2026-08-17**; data mentah `jan-des-25.csv` juga
  kosong (terakhir 2025-06-27, kembali 2025-07-11). Sudah masuk `outlet_closures.csv`; panel
  kehilangan 598 baris dan KY068 terbelah jadi `segment_id` 1 (s/d 2025-06-27) dan 2 (mulai
  2025-07-11). Catatan: ambang 14 hari menangkap kandidat, tidak mendefinisikannya.
- [ ] **`Kebuli Yaman Cikarang Pusat` masih tutup** — pemilik data (2026-08-17): belum ada
  tanggal buka, menunggu data periode baru. Begitu buka, isi `tanggal_buka` di
  `outlet_closures.csv` **dan** perbarui `RELOCATION_DATES` secara manual. Jangan diturunkan
  otomatis: `KY056 Tigaraksa` ada di kedua tabel dengan tanggal yang tidak berhubungan
  (relokasi 2024-03-01 vs tutup sementara Oktober 2024).
- [ ] **`KY073 - Kebuli Yaman Cilebut`** (buka 2025-12-19, masih beroperasi) tidak dapat ramalan
  karena nol hari sebelum cutoff. Bukan soal ambang — masuk sendiri di refresh berikutnya.
- [ ] Empat relokasi bertanggal batas-bawah (`Mayor Oking`, `Teluk Pucung`,
  `Bukit Gading Balaraja`, `Grand Wisata Bekasi`) kemungkinan menghasilkan pola tutup-buka yang
  sama di refresh berikutnya; detektor akan menangkapnya setelah datanya masuk.

## 🔴 Prioritas tertinggi — integrasi region/lead-time (selesai, 2026-08-08)

Lihat `docs/superpowers/specs/2026-08-08-lead-time-integration-design.md` untuk desain
lengkapnya. Ringkasan: `dataset/outlet_mapping.csv` (kolom `kawasan` — 1 = kirim Senin & Kamis,
2 = kirim Selasa & Jumat — dan `hari_pengiriman`) sekarang di-wire penuh ke pipeline via
`outlet_features.apply_region_features`, `lead_time_days` dihitung variabel per baris (bukan
konstanta flat lagi), dan target cumulative-demand-sampai-pengiriman-berikutnya
(`target_lead_time_cumulative`) sudah dibangun.

- [x] **`apply_region_features` dipanggil di `prepare_forecast_data.py::build_featured_dataset`**
  (dipanggil dari `main()`), tepat setelah `add_targets` dan berdekatan dengan
  `apply_outlet_features`.
- [x] **`lead_time_days` bervariasi per `kawasan` & hari transaksi** — dihitung oleh
  `outlet_features.compute_lead_time_days` dari `(day_of_week, hari_pengiriman)`, selalu
  strictly ke depan (kawasan 1/Senin-Kamis: transaksi hari Senin → 3 hari ke Kamis, transaksi
  hari Kamis → 4 hari ke Senin depan — sesuai kebutuhan bisnis di `eda.ipynb` cell `cell-000`).
- [x] **Target cumulative lead-time dibangun** — `prepare_forecast_data.add_lead_time_target`
  menjumlahkan `Kuantitas` mentah pada window strictly-ke-depan `(H+1..H+lead_time_days)` jadi
  `target_lead_time_cumulative`, dengan test leakage-safety (window tidak pernah termasuk hari
  ini/ke belakang).
- [x] Dokumentasi diupdate: `docs/superpowers/specs/2026-07-21-forecast-data-prep-design.md`
  dan `docs/pipeline-overview.md` sudah menyebut region/lead-time features & target barunya.
  (`eda.ipynb` §5/§7 belum dijalankan ulang untuk segmentasi per-region — dicatat sebagai
  follow-up terpisah, bukan blocker untuk pipeline data-prep.)

## 🟠 Perlu konfirmasi data owner (keputusan sudah di-hardcode di kode, belum di-sign-off)

Beda dengan bagian 🔴, poin-poin ini **sudah** punya perilaku default di kode — resikonya bukan
"pipeline belum bisa jalan" tapi "pipeline jalan dengan asumsi yang belum dikonfirmasi benar
oleh yang paham datanya".

- [x] **Santan Cendol / Gula Cendol (`xxx.FGS.00070` / `xxx.FGS.00071`) periode Gr dikonversi ke
  Porsi dan disambung; Cendol Pandan (`xxx.FGS.00069`) tetap di-drop** (2026-08-10) — root cause
  outlier `Kuantitas` ekstrem (30–5250) di ketiga item `xxx.`-prefixed ternyata cuma beda satuan
  pencatatan: periode awal (2025-01-24 → 03-24, 6 cabang) dicatat dalam **Gr**, periode setelahnya
  (mulai Jun 2025, 18 cabang) dicatat dalam **Porsi**. Setiap nilai `Kuantitas` mentah pada ketiga
  item itu adalah kelipatan bulat presisi dari satu angka tetap (dicek: 100% baris habis dibagi
  tanpa sisa) — kemungkinan besar itu berat-per-porsi resep asli: 125 gram/porsi (Cendol Pandan),
  **40 gram/porsi (Santan Cendol), 30 gram/porsi (Gula Cendol)**. `normalize_items.py` sekarang
  punya `GRAM_TO_PORSI_FACTORS = {"xxx.FGS.00070": 40, "xxx.FGS.00071": 30}` +
  `convert_gram_items_to_porsi()` (dipanggil di `load_and_normalize`, sebelum `exclude_items`)
  yang mengonversi `Kuantitas`/`Satuan` untuk Santan/Gula Cendol sebelum masuk tahap merge —
  nama kedua item ini identik di kedua periode, jadi lewat conditional-merge rule yang sudah ada
  dan jadi satu seri waktu utuh 2025-01-24 → 12-28 dalam Porsi (1147 baris masing-masing, tidak
  ada overlap tanggal+cabang). **Cendol Pandan diputuskan tetap di-drop** (bukan dikonversi+
  disambung) — beda dari Santan/Gula Cendol, nama di dua sisi tidak identik (`Cendol Pandan - FG`
  vs `Cendol - FG`) dan datanya jarang bahkan selama periode aktifnya (cuma 32 dari 60 hari
  kalender terisi, 6 cabang, trailing off sebelum berhenti total 2025-03-24) — pola lebih mirip
  pilot kecil yang sepi peminat, jadi diputuskan cukup di-exclude tanpa perlu tunggu konfirmasi
  rename dari data owner. `EXCLUDED_ITEMS` (`normalize_items.py:135`) sekarang
  `{"xxx.FGS.00066", "xxx.FGS.00069"}`. Test terkait: `TestConvertGramItemsToPorsi` (4 test),
  `test_merges_santan_and_gula_cendol_across_xxx_prefix_gap`,
  `test_drops_discontinued_cendol_pandan_by_default` di `test_normalize_items.py`.
  **Catatan verifikasi penting:** `eda.ipynb` membaca `dataset/dataset.csv` mentah langsung
  (tidak lewat `normalize_items.load_and_normalize()`), jadi cell apa pun di notebook itu —
  termasuk `extreme_rows` §3 — akan **tetap** menampilkan item yang sudah di-exclude/Kuantitas
  dalam gram apa adanya selamanya; itu bukan tanda exclude/konversinya gagal/terlewat. Jangan
  verifikasi exclude/konversi/merge ini dari notebook EDA — cek langsung lewat
  `normalize_items.load_and_normalize()` (atau `test_normalize_items.py`) atau output akhir
  `dataset/model_ready/*.parquet` setiap kali pipeline di-generate ulang.
- [x] **`xxx.FGS.00067`/`00068` (Ayam Crispy Original / Spicy) di-exclude, bukan digabung**
  (2026-08-10) — `EXPLICIT_ITEM_RENAMES` sebelumnya memaksa `xxx.FGS.00067` ("Ayam Crispy
  Original") direlabel jadi kode `FGS-00068` dengan nama "Ayam Crispy Spicy", menggabungkan
  keduanya jadi satu series. Investigasi data menemukan ini kemungkinan besar **salah**: ada 190
  kombinasi tanggal+cabang di mana Original dan Spicy sama-sama tercatat di hari yang sama dengan
  kuantitas berbeda (indikasi dua varian rasa yang dijual paralel selama Jan–Sep 2025, bukan satu
  produk yang di-rename), dan keputusan itu ternyata tidak terdokumentasi di
  `docs/superpowers/specs/2026-07-21-forecast-data-prep-design.md` §1 — bertentangan dengan
  prinsip umum desain di dokumen itu sendiri (nama tidak cocok = jangan digabung, persis alasan
  Cendol Pandan dipisah). **Dikonfirmasi data owner (2026-08-10): kedua menu ini sudah tidak ada
  saat ini** (discontinued) — karena itu, alih-alih memperbaiki merge-nya, keduanya langsung
  di-exclude (`EXCLUDED_ITEMS` di `normalize_items.py:135` sekarang mencakup
  `xxx.FGS.00067`/`00068`; `EXPLICIT_ITEM_RENAMES` dikosongkan jadi `{}`), konsisten dengan
  perlakuan Cendol Pandan. Test terkait:
  `test_drops_discontinued_ayam_crispy_original_and_spicy_by_default`,
  `test_default_renames_table_is_empty` di `test_normalize_items.py`.
- [x] **`EXCLUDED_BRANCHES = {"Kebab Saudagar - Kutabumi"}`** (`normalize_items.py:104`) — cabang
  ini juga tampak di outlier Kuantitas ekstrem `eda.ipynb` §3 (WIP `Ayam Shawarma`, 1340).
  Namanya tidak mengikuti pola `KY0NN - ...` cabang lain. **Dikonfirmasi data owner (2026-08-09):
  cabang ini sudah tidak beroperasi/tidak digunakan lagi** — exclude di kode sudah benar sesuai
  keputusan bisnis, tidak perlu tindakan lebih lanjut.
- [x] **Mapping `KY069`→`KY011` dan `TOD M1 Bandara`→`KY051` di `dataset/outlet_name_overrides.csv`
  dikonfirmasi data owner (2026-08-10)** — `KY069 - Kebuli Yaman Bekasi Galaxy` adalah nama dan
  kode lama dari `KY011 - Kebuli Yaman Bekasi Galaxy` (kota: Bekasi), dan `TOD M1 Bandara` adalah
  nama lama dari `KY051 - kebuli Yaman TOD M1 Bandara` (kota: Tangerang) — bukan dua cabang
  berbeda. Penggabungan histori `KY069`→`KY011` dan `TOD M1 Bandara`→`KY051` di pipeline sudah
  tervalidasi sebagai keputusan yang benar.
- [x] **`Kota Override` untuk semua 16 baris di `dataset/outlet_name_overrides.csv` sudah
  dilengkapi & diverifikasi (2026-08-11)** — data owner mengisi langsung kolom `Kota Override`
  untuk seluruh baris (termasuk `KY069`/`TOD M1 Bandara` dan 6 baris relokasi baru di bawah, yang
  sebelumnya dibiarkan kosong). Satu koreksi dari asumsi awal saya: **`KY001` (Kutabumi Pusat)
  final-nya `Kabupaten Tangerang`**, bukan `Kota Tangerang` seperti tebakan awal berdasarkan
  kolom `Kecamatan` (`Jatiuwung`) di `outlets.csv` — alamatnya sendiri menyebut "Pasar Kemis"
  (kecamatan di Kabupaten Tangerang), jadi kolom `Kecamatan` di `outlets.csv` kemungkinan yang
  keliru, bukan override-nya. Diverifikasi lewat `match_branch_to_outlet`/`normalize_kota`: 16/16
  baris resolve ke kota yang benar, 0 mismatch (untuk 8 baris relokasi/mapping, nilai override
  konsisten dengan `Kota` asli outlet tujuan di `outlets.csv`, jadi murni pelengkap eksplisit
  tanpa mengubah hasil akhir). 195 test suite tetap lolos (data-only change).
  **Sign-off ulang 2026-08-16:** sempat ada kontradiksi — `docs/batasan-penelitian.md` B-9 dan
  `docs/pipeline-overview.md` §3 masih mencatat 8 nilai ini sebagai "dugaan terbaik" setelah
  entri di atas ditulis. Pemilik data mengonfirmasi kedelapannya benar (termasuk `KY001`
  Kutabumi = `Kabupaten Tangerang`, meski kolom `Kecamatan` di `outlets.csv` menyebut
  Jatiuwung), dan kedua dokumen itu sudah diperbaiki. Delapan baris tersebut adalah satu-satunya
  yang benar-benar mengubah nilai `kota`; 11 baris sisanya identik dengan `outlets.csv`.
- [x] **6 relokasi "verified" di-wire ke pipeline, dikonfirmasi data owner (2026-08-11)** —
  fisik outlet memang pindah lokasi untuk keenamnya (Tambun/KY020→Mayor Oking,
  Antapani/KY035→Tigaraksa/KY056, Aryana Karawaci/KY046→Cadas, Ciomas/KY047→Cikarang Pusat,
  Bantarjati Bogor/KY052→Teluk Pucung, Dukuh Zamrud/KY059→Bukit Gading Balaraja). 6 baris baru
  ditambahkan ke `dataset/outlet_name_overrides.csv` (pola sama seperti `KY069`→`KY011`), sudah
  diverifikasi `match_branch_to_outlet` me-resolve keenamnya dengan benar, 195 test suite tetap
  lolos (data-only change, tidak mengubah logika kode), dan
  `dataset/model_ready/{train,test}.parquet` sudah di-regenerate (1.417.311 train + 52.067 test
  rows, naik dari 1.291.740+48.340 sebelumnya). Nama cabang lama sudah tidak muncul lagi di
  output, semuanya tergabung ke kode baru — termasuk **KY056 Tigaraksa yang histori aslinya kini
  mundur sampai 2024-01-01** (lewat Antapani) alih-alih 2024-03-01 seperti sebelumnya, berkat
  penyambungan ini.
- [x] **3 relokasi "pending" di-wire ke pipeline, dikonfirmasi data owner (2026-08-11)** — Condet
  (KY028)→Grand Wisata Bekasi (Kota Bekasi), Ciputat Timur (KY055)→Citayam (Kota Depok), Cinere
  (KY029)→Bintara (Kota Bekasi); ketiganya outlet baru pakai histori outlet lama. Ketiga outlet
  baru didaftarkan ke `outlets.csv` (alamat/kecamatan/kota/channel dari data owner), dan 3 baris
  mapping ditambahkan ke `outlet_name_overrides.csv`. Diverifikasi `match_branch_to_outlet`
  me-resolve ketiganya dengan benar, 195 test suite tetap lolos, parquet di-regenerate (1.467.822
  train + 55.046 test rows). Menarik: raw `dataset.csv` ternyata **sudah** punya kode native untuk
  ketiga outlet baru ini (`KY070 - Cadas`, `KY071 - Citayam`, `KY072 - Bintara`) yang mulai
  tepat setelah kode lama berhenti (Aryana Karawaci→Cadas: 2025-09-28→2025-10-03; Ciputat
  Timur→Citayam: 2025-10-30→2025-11-07; Cinere→Bintara: 2025-11-26→2025-11-28) — mengonfirmasi
  independen bahwa relokasi-relokasi ini nyata dan waktunya presisi.
- [x] **Ditemukan & ditangani: fitur lokasi (`kota`/`kawasan`) ikut ter-relabel mundur untuk
  seluruh histori cabang yang direlokasi** — karena `canonicalize_branch_names` jalan sebelum
  `apply_region_features`/`apply_outlet_features` di `prepare_forecast_data.py`, histori
  pra-relokasi (mis. Bintara periode 2024, saat itu masih Cinere di Kota Depok) ikut diberi label
  `kota` **pasca**-relokasi (Kota Bekasi) untuk seluruh baris, bukan cuma baris pasca-relokasi.
  Ini relevan karena tiap kota/daerah punya behavior demand berbeda. **Mitigasi**: fitur baru
  `days_since_relocation` ditambahkan (`outlet_features.add_relocation_feature`, dipanggil di
  `prepare_forecast_data.py::build_featured_dataset` setelah `apply_outlet_features`) — nilai
  negatif untuk baris pra-relokasi, 0 di hari relokasi, positif setelahnya; `NaN` untuk cabang
  yang tidak direlokasi. Dihitung dari `RELOCATION_DATES` (`outlet_features.py`), keyed by nama
  outlet kanonik. Test baru: `TestAddRelocationFeature` (6 test, TDD merah→hijau) di
  `test_outlet_features.py`. Diverifikasi di output akhir: **9/9 cabang relokasi** sekarang punya
  nilai (sebelumnya sempat kelewat satu — Condet→Grand Wisata Bekasi — ketahuan pas verifikasi
  jumlah cabang di parquet, sudah ditambahkan).

  Dua jenis tanggal, beda presisi:
  - **4 cabang dengan tanggal exact** (Antapani→Tigaraksa, Aryana Karawaci→Cadas, Ciputat
    Timur→Citayam, Cinere→Bintara) — raw `dataset.csv` sudah punya kode native untuk outlet
    barunya yang mulai tepat setelah kode lama berhenti, jadi tanggal transisinya langsung
    terbaca dari data (lihat detail di poin relokasi "pending" di atas).
  - **5 cabang dengan tanggal lower-bound proxy** (Tambun→Mayor Oking, Ciomas→Cikarang Pusat,
    Bantarjati Bogor→Teluk Pucung, Dukuh Zamrud→Bukit Gading Balaraja, Condet→Grand Wisata
    Bekasi) — **dikonfirmasi data owner (2026-08-11): relokasi fisiknya sudah terjadi saat ini,
    tapi terjadi setelah cakupan dataset berakhir** (kode lama tidak pernah berhenti muncul
    sampai baris terakhir data), jadi tanggal exact tidak bisa diturunkan. Dipakai tanggal
    **terakhir kode lama muncul di data** sebagai lower bound (mis. Tambun: 2025-12-31) — semua
    baris dapat `days_since_relocation` negatif yang benar arahnya, tapi magnitude-nya
    under-estimate (jarak asli ke relokasi pasti lebih jauh). Didokumentasikan jelas di komentar
    `RELOCATION_DATES`; perlu di-re-derive begitu ada refresh data yang menunjukkan kode lama
    berhenti / kode baru muncul.

  Ini murni fitur data-prep (fakta tanggal) — keputusan cara pakainya di modeling (down-weight
  histori lama, dsb.) didelegasikan ke tahap modeling. Dibahas juga implikasinya untuk LSTM: NaN
  bermasalah untuk neural net (perlu imputasi eksplisit, beda dari tree-based model yang native
  handle NaN), dan sliding window yang mencakup tanggal transisi berpotensi mencampur pola
  demand kota lama+baru dalam satu sequence — dicatat sebagai pertimbangan modeling, bukan
  blocker data-prep.
- [x] **Provenance `kawasan`/`hari_pengiriman` di `dataset/outlet_mapping.csv` dikonfirmasi data
  owner (2026-08-10)** — data ini resmi dari tim SCM (jadwal pengiriman per kawasan), bukan
  asumsi/tebakan manual. Target `target_lead_time_cumulative` yang dibangun darinya (lihat 🔴 di
  atas) aman dipakai untuk semua cabang.
- [x] **1 cabang di bawah threshold completeness 95%** — KY056 (Kebuli Yaman Tigaraksa), 92,3%
  (`eda.ipynb` §4, konsisten di run 2026-08-07). **Dikonfirmasi data owner (2026-08-10): restoran
  tutup sementara** selama periode kosong tersebut, bukan gap pelaporan atau outlet baru — gap-nya
  satu blok kontinu 52 hari (2024-10-01 s/d 2024-11-21), data lengkap sebelum dan sesudahnya
  (2024-03-01 s/d 2024-09-30 dan 2024-11-22 s/d 2025-12-31). Tidak perlu tindakan lebih lanjut di
  data-prep — 52 hari kosong itu representasi valid dari toko tutup, bukan data hilang yang perlu
  diisi/diperbaiki. Kalau tahap modeling nanti sensitif ke gap operasional seperti ini (mis. lag/
  rolling features yang melewati periode tutup), pertimbangkan sebagai catatan desain modeling,
  bukan lagi isu data-prep.
- [x] **27 dari 109 SKU (24,8%) tercatat di lebih dari satu `Kategori Barang`** sepanjang waktu
  (`eda.ipynb` §2, mis. `Minuman`→`Minuman - FG`, `Barang Semi FG (WIP-2)`→`Barang Jadi (FG)`,
  `Snack`→`Snack (FG)`). **Dikonfirmasi data owner (2026-08-10) dengan detail per-pasangan**,
  ternyata bukan satu aturan seragam:
  - `Minuman`↔`Minuman - FG` (16 SKU) dan `Snack`↔`Snack (FG)` (2 SKU): **kategori yang sama**,
    cuma beda label lama/baru — pakai kategori terbaru (varian `(FG)`) untuk seluruh histori.
  - `Barang Semi FG (WIP-2)`↔`Barang Jadi (FG)` (8 SKU, mis. `FGS-00001` Ayam Kebuli, `FGS-00002`
    Kambing Kebuli): **kategori yang memang berbeda** (bukan rename) — TIDAK boleh disamakan,
    tetap time-varying sesuai kategori tercatat aslinya per baris.
  - `FGS-00014` (Club Mineral 600ml): sempat tercatat WIP-2 di awal tapi **seharusnya selalu
    Minuman - FG** — override eksplisit per-SKU, bukan mengikuti pola pasangan mana pun.
  - Agregasi juga harus menghormati kategori final ini (dikonfirmasi user).

  Kode diupdate: `normalize_items.py::canonicalize_item_categories` sekarang pakai
  `CATEGORY_SYNONYMS` (`{"Minuman": "Minuman - FG", "Snack": "Snack (FG)"}`) untuk collapse
  pasangan yang benar-benar sinonim, dan `EXPLICIT_CATEGORY_OVERRIDES`
  (`{"FGS-00014": "Minuman - FG"}`) untuk override satu SKU — SKU dengan kategori yang genuinely
  berbeda (WIP-2/FG) dibiarkan time-varying, tidak lagi dipaksa ke kategori terbaru seperti
  perilaku lama. Verifikasi di data penuh: 8 SKU WIP-2/FG tetap 2 kategori berbeda per baris,
  semua SKU Minuman/Snack + `FGS-00014` collapse jadi 1 kategori. Test baru:
  `test_leaves_wip_to_fg_transition_time_varying_not_a_rename`,
  `test_applies_explicit_override_for_club_mineral_600ml`
  (`TestCanonicalizeItemCategories`, `test_normalize_items.py`) — ditulis TDD (merah dulu, lalu
  hijau), plus 195 test suite penuh lolos tanpa regresi. `dataset/model_ready/{train,test}.parquet`
  sudah di-regenerate ulang (`python3 -m utils.prepare_forecast_data`) dengan logika baru ini.
- [x] **Negative Kuantitas dikonfirmasi data owner (2026-08-10)** — root cause anomali KY011
  2024-02-29 adalah **salah input dari sistem sumbernya**, dan `dataset/excel/feb-24_No_Minus.xlsx`
  (dipakai via `dataset/csv/feb-24.csv`) dikonfirmasi sebagai versi final yang benar. Verifikasi:
  0 baris negatif di `dataset/csv/feb-24.csv` (45.581 baris) maupun `dataset/dataset.csv`
  (693.563 baris); 26 baris KY011 tanggal 29 Feb 2024 semuanya positif. **Tetap jalankan ulang
  soft-check ini (`eda.ipynb` §8) setiap kali `dataset.csv` di-refresh bulanan** — konfirmasi ini
  soal insiden yang sudah terjadi, bukan jaminan sistem sumber tidak akan salah input lagi ke
  depannya.

## 🟡 Gap engineering yang belum tergarap (tidak butuh keputusan data owner, murni kerjaan kode)

- [x] **Stage outlier-handling di-wire ke `notebook/data-processing.ipynb`** (2026-08-08) —
  cell 0 sekarang import `outlier_handling`. Section di-reorder: **"4. Calendar features"**
  (dulu §5) sekarang jalan lebih dulu, baru **"5. Outlier handling"** (baru,
  `compute_pair_baseline` + `apply_outlier_capping`), baru **"6. Targets, lag & rolling
  features"** (dulu §4) — urutan ini beda dari draf awal todo ("sisip antara section 3 dan 4")
  karena `apply_outlier_capping` butuh kolom event kalender (`is_ramadan`/`is_eid_al_fitr`/dst)
  dari `add_calendar_features` untuk pengecualian event-window, jadi harus jalan setelahnya —
  persis urutan di `prepare_forecast_data.py::main()`. `add_targets` tetap pakai `Kuantitas`
  mentah; `add_lag_features`/`add_rolling_features`/`compute_branch_stats` (§7, dulu §6) pakai
  `qty_col="Kuantitas_capped"`. Section 8 (QA, dulu §7) dapat tambahan assert baru: capped tidak
  pernah melebihi raw, dan baris yang tidak di-cap identik dengan raw. Notebook sudah dieksekusi
  ulang penuh (`jupyter nbconvert --execute`), semua assert lolos: 1.340.034 baris konsisten di
  semua tahap, 8.507/11.718 baris spike benar-benar di-cap (sisanya dikecualikan event window).
- [x] **7 QA assertion cuma ada di notebook, tidak di script** — **selesai**:
  `prepare_forecast_data.run_qa_checks()` (`prepare_forecast_data.py:335`) sekarang memuat
  cek no-negative-Kuantitas, no-duplicate (item, cabang, tanggal), `Kuantitas_capped` ≤ raw,
  target capped ≤ target mentah, tidak ada `kota == "Unknown"`, tidak ada cabang tanpa
  `kawasan`, satu cabang → satu kota, tidak ada baris di dalam interval tutup, `segment_id`
  mulai dari 1 & kontinu, dan tidak ada lubang tanggal di dalam satu segmen. Dipanggil dari
  `main()` **dan** dari notebook, jadi kedua jalur terverifikasi. Yang masih notebook-only:
  spot-check leakage lag/rolling, rentang tanggal per outlet, dan section visual QA.
- [ ] **Re-run `python3 -m utils.prepare_forecast_data` setiap kali ada perubahan di script
  pipeline manapun** — parquet `dataset/model_ready/*.parquet` gampang jadi stale relatif ke
  kode kalau lupa di-generate ulang; tidak ada mekanisme guard otomatis untuk ini saat ini.
- [ ] **22% item-branch pair (842 dari 3.882) gagal `MIN_HISTORY_DAYS` (60 hari)** di
  `build_panel.filter_min_history` (`eda.ipynb` §6). Bukan bug, tapi konsekuensi desain yang
  didokumentasikan sebagai "out of scope" di spec §Out of scope ("Cold-start / fallback handling
  ... belum diputuskan"). Kalau prioritas bisnis butuh forecast untuk pair yang di-drop ini
  (SKU/cabang baru dengan histori pendek), perlu desain fallback terpisah (mis. rata-rata level
  kategori) — belum ada rencana konkret untuk ini di spec manapun.
  **Diukur 2026-08-17 — `MIN_HISTORY_DAYS` tetap 60.** Menurunkannya ke 28 (pipeline dijalankan
  penuh dua kali lewat parameter `min_history_days`, keduanya lolos `run_qa_checks`) menambah 167
  pair (2.979 → 3.146) tetapi hanya **0,023% volume permintaan** (4.773 dari 20,9 juta unit).
  Setelah `drop_warmup_rows(28)` kohort itu cuma menyumbang **2.428 baris latih (0,18%)** — karena
  `LOOKBACK` + `lag_28`/`roll_28` memakan 28 hari pertama tiap segmen, jadi yang membatasi bukan
  ambangnya. Efek pada metrik justru menyesatkan: MAE baseline `roll_mean_7` turun 12,90 → 12,54
  murni karena dilusi (kohort baru 73,7% target-nya nol, rata-rata target 0,77 vs 30,8), sehingga
  angka tidak lagi sebanding dengan run sebelumnya. Config 28 juga menabrak guard
  `add_event_flag` (6 SKU tanpa entri di `event_driven_items.csv`). Cakupan untuk 621 pair sisanya
  diselesaikan lewat fallback cold-start, bukan lewat ambang latih — dan fallback itu harus
  mengalahkan pinball 0,384 (`roll_mean_7` pada kohort 28–59 hari di Des 2025) supaya layak pakai.
- [ ] **Kriteria keberhasilan & target: dikonfirmasi data owner 2026-08-17.** Ukuran keberhasilan
  adalah **"outlet tidak kehabisan barang"**, dan kehabisan stok untuk *pesanan* sudah ditangani
  manual oleh head office. Konsekuensinya **target latih & metrik utama = `..._capped`**, karena
  porsi yang di-cap adalah proxy pre-order yang sudah ditangani jalur manual. Sisa risiko
  terukur: model sempurna pada target capped, dinilai terhadap permintaan raw, memberi cycle
  service 0,977 / fill rate 0,981 — **40.281 unit kurang sepanjang Des 2025 (1,9% massa
  permintaan)**, dan itulah yang harus ditutup proses manual. Skor terhadap target raw tetap
  dilaporkan sebagai kolom kedua (selisihnya besar: MAE −6,0%, pinball −11,0% pada prediksi yang
  sama persis), supaya perbandingan antar model tidak dimenangkan oleh pilihan target.
- [ ] **Tanya data owner: apakah spike yang berdiri sendiri itu pre-order juga?** Dari 7.552 baris
  yang di-cap, **49,8%** berada di kombinasi cabang-hari dengan ≥3 item melonjak serentak (tanda
  khas satu pesanan besar), tetapi **38,6% melonjak sendirian** — lebih mirip permintaan organik
  atau restock. Condong ke akhir pekan (Minggu 1.808, Sabtu 1.547 vs Senin 532) dan didominasi
  Packaging (60%). Kalau yang sendirian itu bukan pre-order, capping memotong permintaan yang
  **tidak** ditutup jalur manual, dan bagian dari 40.281 unit di atas jadi stockout nyata.

## 🟢 Sanity check rutin (jalankan tiap kali dataset di-refresh atau sebelum training)

> **Data 2026 masuk?** Baca `docs/checklist-refresh-data-2026.md` lebih dulu — di sana
> tercatat apa yang gagal keras, apa yang salah diam-diam (terutama `ID_HOLIDAYS` yang tidak
> dijaga `check_year_coverage`), dan apa yang harus di-derive ulang.

- [ ] **Verifikasi `EXCLUDED_ITEMS` (`xxx.FGS.00066/67/68/69`) benar-benar tidak muncul
  di `dataset/model_ready/*.parquet`, dan Santan/Gula Cendol (`xxx.FGS.00070/71`) sudah dalam
  Satuan Porsi (bukan Gr)** — penting untuk diingat karena `eda.ipynb` **tidak bisa** dipakai
  untuk verifikasi ini (baca `dataset.csv` mentah, lihat catatan di poin 🟠 "Santan Cendol / Gula
  Cendol ... dikonversi" dan "Ayam Crispy Original / Spicy" di atas). Cek lewat kode langsung, mis.
  `normalize_items.load_and_normalize()["Kode Barang"].isin(normalize_items.EXCLUDED_ITEMS).sum() == 0`
  dan `(normalize_items.load_and_normalize()["Satuan"] == "Gr").sum() == 0` untuk kode Cendol,
  supaya item trial/discontinued tidak diam-diam kembali masuk dan konversi satuan tidak diam-diam
  terlewat kalau raw CSV bulanan berubah format.
- [ ] Soft-check negative `Kuantitas` (`eda.ipynb` §8) — lihat poin 🟠 di atas.
- [ ] `calendar_features.check_year_coverage` (`calendar_features.py:168`) — akan raise
  `ValueError` otomatis kalau data punya tahun yang `RAMADAN_PERIODS`/`EID_AL_FITR_DATES`/
  `EID_AL_ADHA_DATES`/`ID_HOLIDAYS` belum cover (saat ini cuma 2024 & 2025) — **penting untuk
  diingat begitu ada data 2026 masuk dari update bulanan**, karena tanpa update
  `calendar_features.py` pipeline akan gagal keras, bukan diam-diam salah.
- [ ] Distribusi `Kuantitas` tetap right-skewed & kontinu (bukan integer) — pastikan tahap
  modeling nanti treat sebagai kontinu, bukan count.
- [ ] Intermittency & konsentrasi volume (median 64% zero-demand, top 6,6% pair = 80% volume) —
  bukan hal yang perlu "diperbaiki" di preprocessing, tapi jadi konteks wajib buat keputusan
  desain modeling (hindari asumsi demand kontinu murni; pertimbangkan bobot evaluasi per pair).

## Prioritas & urutan pengerjaan yang disarankan

1. ~~**🔴 Selesaikan integrasi region/lead-time**~~ — selesai 2026-08-08 (lihat §🔴 di atas).
2. ~~**🟠 Kumpulkan konfirmasi data owner**~~ — semua item sudah dikonfirmasi & di-wire, termasuk
   seluruh 9 relokasi outlet (6 verified + 3 pending, lihat §🟠 di atas) dan fitur mitigasi
   `days_since_relocation` (9/9 cabang terisi — 4 tanggal exact, 5 lower-bound proxy pending
   re-derivation begitu data owner punya tanggal exact atau data refresh menunjukkan transisinya).
3. **🟡 Kerjakan gap engineering** yang tersisa (QA assertion ke script) — tidak ada lagi yang
   menunggu jawaban data owner.
4. **🟢 Jalankan sanity check rutin** setiap refresh dataset bulanan berikutnya — terutama
   `check_year_coverage` begitu data 2026 mulai masuk.

---
*Sumber: `notebook/eda.ipynb` (run 2026-08-07), `docs/pipeline-overview.md`,
`docs/superpowers/specs/2026-07-21-forecast-data-prep-design.md`, `CLAUDE.md`, dan pembacaan
langsung `utils/normalize_items.py`, `utils/build_panel.py`, `utils/calendar_features.py`,
`utils/prepare_forecast_data.py`, `utils/outlet_features.py`, `test/test_outlet_features.py`
(2026-08-07).*
