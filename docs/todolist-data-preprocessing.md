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

## 🔴 Prioritas tertinggi — integrasi region/lead-time (sedang berjalan, belum selesai)

Ini beda dari temuan `eda.ipynb` §8/§11 ("belum ada mapping Region 1/2") — **temuan itu sudah
usang**. Sejak commit `1b02bf2`, ada file baru `dataset/outlet_mapping.csv` berisi kolom
`kawasan` (1 = kirim Senin & Kamis, 2 = kirim Selasa & Jumat, sesuai konteks bisnis di
`eda.ipynb` cell `cell-000`) dan `hari_pengiriman`, plus fungsi
`outlet_features.apply_region_features` (dengan test-nya di `test_outlet_features.py`, kelas
`TestApplyRegionFeatures`) yang sudah bisa join `kawasan`/`hari_pengiriman` ke `Nama Cabang`
ter-kanonik. **Perubahan ini masih uncommitted** (`git diff utils/outlet_features.py` /
`test/test_outlet_features.py`) — bukan pekerjaan yang belum dimulai, tapi pekerjaan
setengah-jalan yang perlu diselesaikan:

- [ ] **`apply_region_features` belum dipanggil di `prepare_forecast_data.py::main()`** —
  bandingkan dengan `apply_outlet_features` yang sudah di-wire (baris
  `df = apply_outlet_features(...)`). Perlu ditambahkan langkah serupa, idealnya setelah
  `canonicalize_branch_names` supaya `Nama Cabang` sudah dalam bentuk kanonik saat di-join ke
  `outlet_mapping.csv`.
- [ ] **`lead_time_days` masih konstanta flat (`DEFAULT_LEAD_TIME_DAYS = 4`), tidak bervariasi
  per `kawasan`** — nama test-nya sendiri eksplisit: `test_lead_time_days_flat_default_regardless_of_kawasan`
  (`test/test_outlet_features.py:256`). Ini gap paling penting: kebutuhan bisnis di `eda.ipynb`
  cell `cell-000` adalah window 3 hari **atau** 4 hari tergantung *hari pengiriman berikutnya*
  relatif ke tanggal transaksi (mis. kawasan 1/Senin-Kamis: transaksi hari Senin → window
  3 hari ke Kamis, transaksi hari Kamis → window 4 hari ke Senin depan). Ini belum dihitung
  di mana pun.
- [ ] **Target cumulative lead-time belum dibangun sama sekali** — `add_targets` di
  `prepare_forecast_data.py:20` cuma bikin `target_h1`…`target_h7` (shift harian satu-satu),
  bukan target "total demand sampai pengiriman berikutnya" yang jadi tujuan bisnis utama
  notebook ini. Perlu fungsi baru (mis. `add_lead_time_target`) yang, per baris, menjumlahkan
  `Kuantitas` dari tanggal tersebut sampai `lead_time_days` berikutnya (variabel, hasil dari
  poin di atas), lalu tulis sebagai kolom target baru — dan tambahkan test leakage-safety-nya
  (window harus strictly ke depan, bukan termasuk hari ini/ke belakang, mengikuti pola shift
  yang sudah dipakai `add_rolling_features`).
- [ ] Setelah 3 poin di atas selesai: update `docs/superpowers/specs/2026-07-21-forecast-data-prep-design.md`
  dan `docs/pipeline-overview.md` (keduanya masih menyebut region mapping sebagai "belum ada"
  atau tidak menyebutnya sama sekali) dan jalankan ulang `eda.ipynb` §5/§7 supaya analisis
  day-of-week & lead-time window bisa disegmentasi per region, bukan proxy generik lagi.

## 🟠 Perlu konfirmasi data owner (keputusan sudah di-hardcode di kode, belum di-sign-off)

Beda dengan bagian 🔴, poin-poin ini **sudah** punya perilaku default di kode — resikonya bukan
"pipeline belum bisa jalan" tapi "pipeline jalan dengan asumsi yang belum dikonfirmasi benar
oleh yang paham datanya".

- [ ] **Item ber-prefix `xxx.`** — `normalize_items.py` sudah punya keputusan eksplisit:
  `EXCLUDED_ITEMS = {"xxx.FGS.00066", "xxx.FGS.00069", "xxx.FGS.00070", "xxx.FGS.00071"}`
  (`normalize_items.py:113`, exclude "Nasi Putih" dan trio "Cendol Pandan" / "Santan Cendol" /
  "Gula Cendol" — ketiganya cuma tercatat Jan–Mar 2025 lalu berhenti total secara serentak,
  konsisten dengan produk trial/discontinued; juga muncul sebagai outlier Kuantitas tertinggi di
  `eda.ipynb` §3, nilai 5250/2625/.../1260) dan `EXPLICIT_ITEM_RENAMES = {"xxx.FGS.00067":
  ("FGS-00068", "Ayam Crispy Spicy - FG")}` (`normalize_items.py:122`). Ini keputusan best-guess
  dari desain (lihat `docs/superpowers/specs/2026-07-21-forecast-data-prep-design.md` §1 soal
  "flavor difference can't be ruled out from the data alone"), **belum dikonfirmasi ke data
  owner**. Minta konfirmasi: apakah ke-4 item itu memang harus di-drop total, dan apakah rename
  `xxx.FGS.00067` → `FGS-00068` itu benar secara bisnis.
  **Catatan verifikasi penting:** `eda.ipynb` membaca `dataset/dataset.csv` mentah langsung
  (tidak lewat `normalize_items.load_and_normalize()`), jadi cell apa pun di notebook itu —
  termasuk `extreme_rows` §3 — akan **tetap** menampilkan ke-4 item ini apa adanya selamanya;
  itu bukan tanda exclude-nya gagal/terlewat. Jangan verifikasi exclude ini dari notebook EDA —
  cek langsung lewat `normalize_items.load_and_normalize()` (atau `test_normalize_items.py`)
  atau output akhir `dataset/model_ready/*.parquet` setiap kali pipeline di-generate ulang.
- [ ] **`EXCLUDED_BRANCHES = {"Kebab Saudagar - Kutabumi"}`** (`normalize_items.py:91`) — cabang
  ini juga tampak di outlier Kuantitas ekstrem `eda.ipynb` §3 (WIP `Ayam Shawarma`, 1340).
  Namanya tidak mengikuti pola `KY0NN - ...` cabang lain — konfirmasi apakah ini memang brand/
  lini bisnis terpisah yang sengaja dikecualikan dari forecast Kebuli Yaman, bukan cabang yang
  datanya hilang.
- [ ] **`Kota Override` & mapping `KY069`→`KY011`** di `dataset/outlet_name_overrides.csv` —
  8 baris override kota (Tangerang, Bogor, Bekasi, dll — kemungkinan nama kota di `outlets.csv`
  salah/beda format) dan satu baris `KY069 - Kebuli Yaman Bekasi Galaxy` di-map ke outlet
  `KY011 - Kebuli Yaman Bekasi Galaxy` — sudah ditandai di `pipeline-overview.md` §3 sebagai
  "best-guess corrections, not yet confirmed by the data owner". Perlu sign-off eksplisit,
  terutama mapping KY069→KY011 karena efeknya menggabungkan histori dua kode cabang jadi satu
  seri waktu.
- [ ] **Provenance `kawasan`/`hari_pengiriman` di `dataset/outlet_mapping.csv`** — file ini baru
  (belum ter-commit) dan belum ada dokumentasi dari mana sumbernya (tim SCM langsung? asumsi
  manual?). Sebelum dipakai untuk target lead-time (lihat 🔴 di atas), pastikan sumbernya resmi
  dari tim SCM, bukan tebakan sementara — kalau salah, target cumulative demand yang dihasilkan
  ikut salah untuk semua cabang.
- [ ] **1 cabang di bawah threshold completeness 95%** — KY056 (Kebuli Yaman Tigaraksa), 92,3%
  (`eda.ipynb` §4, konsisten di run 2026-08-07). Konfirmasi: gap pelaporan, outlet tutup
  sementara, atau outlet baru?
- [ ] **27 dari 109 SKU (24,8%) tercatat di lebih dari satu `Kategori Barang`** sepanjang waktu
  (`eda.ipynb` §2, mis. `Minuman`→`Minuman - FG`, `Barang Semi FG (WIP-2)`→`Barang Jadi (FG)`,
  `Snack`→`Snack (FG)`). Pola pasangannya konsisten, terlihat seperti rename taksonomi kategori
  pertengahan 2024, bukan noise input. **Sudah ditangani di kode** — `normalize_items.py` kini
  punya `canonicalize_item_categories` (dipanggil di `load_and_normalize`, sebelum
  `reaggregate_daily`), yang membekukan tiap `Kode Barang` ke `Kategori Barang` dari baris
  ber-`Tanggal` terbaru, dengan test di `TestCanonicalizeItemCategories` +
  `TestLoadAndNormalize.test_canonicalizes_category_relabel_across_time_end_to_end`
  (`test_normalize_items.py`). Yang masih kurang cuma **sign-off data owner**: keputusan "pakai
  kategori terbaru" ini masih asumsi dari pola tanggal, belum dikonfirmasi eksplisit bahwa ini
  memang rename taksonomi disengaja (bukan, misalnya, dua produk berbeda yang kebetulan pakai
  kode SKU yang sama).
- [ ] **Negative Kuantitas** — 0 baris di run terakhir (anomali KY011 2024-02-29 sudah resolved,
  `eda.ipynb` §8 soft-check), tapi `CLAUDE.md` menegaskan ini belum dikonfirmasi permanen ke
  data owner. **Jalankan ulang soft-check ini setiap kali `dataset.csv` di-refresh bulanan** —
  jangan asumsikan aman selamanya hanya karena sudah bersih bulan ini.

## 🟡 Gap engineering yang belum tergarap (tidak butuh keputusan data owner, murni kerjaan kode)

- [ ] **7 QA assertion cuma ada di notebook, tidak di script** — `pipeline-overview.md` §3 & §9
  menyebutkan cek row-count, no-duplicate, no-negative-Kuantitas, spot-check Ramadan/Eid,
  leakage lag/rolling, dan outlet-join sanity itu semua cuma jalan lewat
  `notebook/data-processing.ipynb`. `python3 prepare_forecast_data.py` langsung tidak
  memverifikasi apa pun setelah export. Pertimbangkan pindahkan minimal subset-nya (no-negative,
  no-duplicate, leakage spot-check) jadi assertion di dalam `main()` atau fungsi terpisah yang
  dipanggil dari situ, supaya CI/automation yang tidak lewat notebook tetap ke-cover.
- [ ] **Re-run `prepare_forecast_data.py` setiap kali ada perubahan di script pipeline manapun**
  — parquet `dataset/model_ready/*.parquet` gampang jadi stale relatif ke kode kalau lupa
  di-generate ulang; tidak ada mekanisme guard otomatis untuk ini saat ini.
- [ ] **22% item-branch pair (842 dari 3.882) gagal `MIN_HISTORY_DAYS` (60 hari)** di
  `build_panel.filter_min_history` (`eda.ipynb` §6). Bukan bug, tapi konsekuensi desain yang
  didokumentasikan sebagai "out of scope" di spec §Out of scope ("Cold-start / fallback handling
  ... belum diputuskan"). Kalau prioritas bisnis butuh forecast untuk pair yang di-drop ini
  (SKU/cabang baru dengan histori pendek), perlu desain fallback terpisah (mis. rata-rata level
  kategori) — belum ada rencana konkret untuk ini di spec manapun.

## 🟢 Sanity check rutin (jalankan tiap kali dataset di-refresh atau sebelum training)

- [ ] **Verifikasi `EXCLUDED_ITEMS` (`xxx.FGS.00066/69/70/71`) benar-benar tidak muncul di
  `dataset/model_ready/*.parquet`** — penting untuk diingat karena `eda.ipynb` **tidak bisa**
  dipakai untuk verifikasi ini (baca `dataset.csv` mentah, lihat catatan di poin 🟠 "Item
  ber-prefix `xxx.`" di atas). Cek lewat kode langsung, mis.
  `normalize_items.load_and_normalize()["Kode Barang"].isin(normalize_items.EXCLUDED_ITEMS).sum() == 0`,
  supaya item trial/discontinued ini tidak diam-diam kembali masuk kalau raw CSV bulanan
  ternyata memakai kode SKU baru untuk produk serupa.
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

1. **🔴 Selesaikan integrasi region/lead-time** — ini blocker langsung untuk tujuan bisnis inti
   (forecast cumulative demand sampai pengiriman berikutnya). Konfirmasi provenance
   `outlet_mapping.csv` dulu (🟠) sebelum wiring, supaya tidak membangun target di atas data
   yang belum tentu benar.
2. **🟠 Kumpulkan semua konfirmasi data owner** dalam satu putaran (item `xxx.`, cabang
   dikecualikan, kota override/KY069, kategori 27-SKU, KY056) — banyak yang bisa ditanyakan
   sekaligus ke orang yang sama.
3. **🟡 Kerjakan gap engineering** yang tidak butuh menunggu jawaban (QA assertion ke script,
   re-run parquet) sambil menunggu 2; canonicalize kategori setelah dapat jawabannya.
4. **🟢 Jalankan sanity check rutin** setiap refresh dataset bulanan berikutnya — terutama
   `check_year_coverage` begitu data 2026 mulai masuk.

---
*Sumber: `notebook/eda.ipynb` (run 2026-08-07), `docs/pipeline-overview.md`,
`docs/superpowers/specs/2026-07-21-forecast-data-prep-design.md`, `CLAUDE.md`, dan pembacaan
langsung `utils/normalize_items.py`, `utils/build_panel.py`, `utils/calendar_features.py`,
`utils/prepare_forecast_data.py`, `utils/outlet_features.py`, `test/test_outlet_features.py`
(2026-08-07).*
