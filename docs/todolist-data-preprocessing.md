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
- [ ] **`Kota Override` & mapping `KY069`→`KY011`** di `dataset/outlet_name_overrides.csv` —
  8 baris override kota (Tangerang, Bogor, Bekasi, dll — kemungkinan nama kota di `outlets.csv`
  salah/beda format) dan satu baris `KY069 - Kebuli Yaman Bekasi Galaxy` di-map ke outlet
  `KY011 - Kebuli Yaman Bekasi Galaxy` — sudah ditandai di `pipeline-overview.md` §3 sebagai
  "best-guess corrections, not yet confirmed by the data owner". Perlu sign-off eksplisit,
  terutama mapping KY069→KY011 karena efeknya menggabungkan histori dua kode cabang jadi satu
  seri waktu.
- [ ] **Outlet relocation belum di-wire ke pipeline (`docs/outlet_relocation_notes.md`)** —
  dokumen ini murni referensi manual, tidak direferensikan di `utils/*.py` manapun (cek: grep
  "reloc" di seluruh `utils/` nol hasil). Cross-check ke data aktual (2026-08-09): 8 dari 9
  "old outlet" di catatan (Tambun/KY020, Antapani/KY035, Aryana Karawaci/KY046, Ciomas/KY047,
  Bantarjati Bogor/KY052, Dukuh Zamrud/KY059, Condet/KY028, Ciputat Timur/KY055) ada di
  `dataset.csv` mentah tapi **tidak ada** di `outlets.csv`, sehingga
  `outlet_features.filter_matched_branches` membuang seluruh historinya (67.020 baris, 9,66%
  dari seluruh dataset) alih-alih menyambungkannya ke outlet baru hasil relokasi — pola yang
  seharusnya sama seperti `KY069→KY011` di `outlet_name_overrides.csv`, tapi belum diterapkan
  ke ke-6 relokasi "verified" (Tambun→Mayor Oking, Antapani→Tigaraksa, Aryana Karawaci→Cadas,
  Ciomas→Cikarang Pusat, Bantarjati Bogor→Teluk Pucung, Dukuh Zamrud→Bukit Gading Balaraja) di
  catatan itu. Anomali tambahan: `Cinere` (KY029) dicatat sebagai outlet lama yang relokasi ke
  Bintara (pending), tapi KY029 masih **aktif** di `outlets.csv` saat ini — catatan relokasi
  kemungkinan belum sinkron dengan status outlet terkini. Konfirmasi ke data owner: (1) apakah
  6 relokasi verified perlu digabung historinya via `outlet_name_overrides.csv` seperti
  KY069→KY011, (2) status Cinere/Bintara — sudah terjadi atau masih rencana, (3) 3 outlet
  target yang belum terdaftar (Grand Wisata Bekasi, Citayem/Citayam, Bintara) perlu ditambahkan
  ke `outlets.json`/`outlets.csv` dulu sebelum bisa masuk pipeline forecast.
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
- [ ] **7 QA assertion cuma ada di notebook, tidak di script** — `pipeline-overview.md` §3 & §9
  menyebutkan cek row-count, no-duplicate, no-negative-Kuantitas, spot-check Ramadan/Eid,
  leakage lag/rolling, dan outlet-join sanity itu semua cuma jalan lewat
  `notebook/data-processing.ipynb`. `python3 -m utils.prepare_forecast_data` langsung tidak
  memverifikasi apa pun setelah export. Pertimbangkan pindahkan minimal subset-nya (no-negative,
  no-duplicate, leakage spot-check) jadi assertion di dalam `main()` atau fungsi terpisah yang
  dipanggil dari situ, supaya CI/automation yang tidak lewat notebook tetap ke-cover.
- [ ] **Re-run `python3 -m utils.prepare_forecast_data` setiap kali ada perubahan di script
  pipeline manapun** — parquet `dataset/model_ready/*.parquet` gampang jadi stale relatif ke
  kode kalau lupa di-generate ulang; tidak ada mekanisme guard otomatis untuk ini saat ini.
- [ ] **22% item-branch pair (842 dari 3.882) gagal `MIN_HISTORY_DAYS` (60 hari)** di
  `build_panel.filter_min_history` (`eda.ipynb` §6). Bukan bug, tapi konsekuensi desain yang
  didokumentasikan sebagai "out of scope" di spec §Out of scope ("Cold-start / fallback handling
  ... belum diputuskan"). Kalau prioritas bisnis butuh forecast untuk pair yang di-drop ini
  (SKU/cabang baru dengan histori pendek), perlu desain fallback terpisah (mis. rata-rata level
  kategori) — belum ada rencana konkret untuk ini di spec manapun.

## 🟢 Sanity check rutin (jalankan tiap kali dataset di-refresh atau sebelum training)

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
   Provenance `outlet_mapping.csv` masih belum dikonfirmasi data owner (🟠 di bawah) — pipeline
   berjalan di atas data as-is sesuai arahan eksplisit, bukan menunggu konfirmasi.
2. **🟠 Kumpulkan semua konfirmasi data owner** dalam satu putaran (item `xxx.`, cabang
   dikecualikan, kota override/KY069, outlet relocation (67.020 baris/9,66% berpotensi hilang),
   kategori 27-SKU, KY056, provenance `kawasan`/`hari_pengiriman`) — banyak yang bisa ditanyakan
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
