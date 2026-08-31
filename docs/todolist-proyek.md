# To-Do List — Proyek Demand Forecasting SCM Kebuli Yaman

Daftar kerja untuk **keseluruhan proyek**, bukan hanya tahap data preprocessing.
Berkas ini menggantikan `docs/todolist-data-preprocessing.md` (diganti nama
2026-08-24) yang cakupannya berhenti di `train/test.parquet` — sementara
pekerjaan proyek sudah lama melewati titik itu (pemodelan tiga algoritma,
mesin evaluasi bersama, dan migrasi evaluasi multi-kuantil yang sedang berjalan).

Peta besar proyeknya ada di `docs/overview.md`; berkas ini adalah versi
_actionable_-nya: apa yang sudah selesai, apa yang sedang dikerjakan, dan apa
yang menunggu apa. Riwayat panjang tiap keputusan preprocessing yang sudah
ditutup tidak diulang di sini — sudah tercatat di
`docs/preprocessing.md` (Bagian 1 dan Bagian 2), dan
riwayat git berkas ini.

**Terakhir diverifikasi: 2026-08-30** — Ketiga model (RF, XGBoost, LSTM)
selesai training penuh, ketiga dokumen hasil ditulis, `crossing_rate` diuji,
dan **derau seed LSTM diverifikasi langsung** — lihat ringkasan di bawah.
XGBoost selesai **2026-08-27** (`models/xgboost_q90.joblib`), LSTM selesai
**2026-08-28** (`models/lstm_q90.joblib`) — training di atas keputusan
mesin-terpisah 2026-08-26 (pencarian di GPU Windows RTX 4060 Ti,
walk-forward + fit final di CPU Mac untuk ketiganya, lihat
`docs/runbook-pencarian-gpu-windows.md`).

**"K1 seri, RF vs LSTM" ditarik (2026-08-30).** Angka kepala seed 42 (resmi
di semua dokumen sampai kemarin): RF 2,8508 < LSTM 2,8818 < XGBoost 2,9433 —
LSTM tampak hampir seri dengan RF (gap 1,09%, di bawah ambang 2%). Walk
-forward 5-fold penuh diulang dengan seed 43: **K1 LSTM di fold 1/2/4
melompat ke 3,0732 — kalah 7,80% dari RF, lebih buruk bahkan dari XGBoost.**
Seed 42 terbukti bukan titik tengah sebaran, melainkan pengecualian yang
menguntungkan LSTM. `crossing_rate` juga sudah diuji (2026-08-29), hasilnya
**tidak simetris**: **XGBoost defek sungguhan** (~20–25% baris crossing
material — butuh rearrangement kuantil, tertahan sendiri) sementara **LSTM
hampir seluruhnya derau numerik** (tidak lagi jadi keraguan).

**Rearrangement kuantil XGBoost dikerjakan (2026-08-30, ~2,89 jam) — prasyarat
wajib terpenuhi.** `xgb_rearrangement_walkforward.py` mem-fit ulang kelima
model fold dengan hyperparameter pemenang yang sama, mengurutkan (sort) 19
prediksi kuantil tiap baris sebelum dinilai (definisi rearrangement
Chernozhukov et al. 2010). Cek reproduksibilitas cocok persis dengan run
resmi (`best_iteration` per fold identik, K1 sebelum sort = 2,9433 persis
sama), jadi angkanya sah. Hasil: `crossing_rate` → **0,0000**, K1 fold bersih
membaik **1,08%** (2,9433 → **2,9115**) — tapi gap ke RF cuma menyempit dari
3,24% ke **2,13%, masih di atas ambang 2%**. **XGBoost tetap kalah**, dengan
angka yang sekarang adil dibandingkan. Detail: `docs/hasil-modeling-xgb.md`
bagian 5.5, `dataset/model_ready/xgb_walk_forward_results_rearranged.csv`.

**Kesimpulan: RF adalah kandidat terdepan di semua perbandingan yang ada**
(RF < LSTM-seed42 < XGBoost-rearranged < LSTM-seed43), tidak ada lagi
prasyarat teknis yang menahan keputusan (n=2 pada seed LSTM bukan interval
kepercayaan ketat, tapi kedua titik menunjuk arah yang sama dan sudah
dianggap cukup — bagian 18 metodologi). **Pemenang belum resmi dibekukan**
— itu langkah terpisah yang menunggu persetujuan eksplisit Anda dalam
sebuah commit (Fase E, butir 10 di bawah).
2026-08-25 sebelumnya — Random Forest butir 0c selesai dijalankan penuh
(benchmark → pencarian 18 kandidat → walk-forward 5 fold → fit final, ~5,6 jam
wall clock) dan `docs/hasil-modeling-rf.md` ditulis ulang; run lama diarsipkan
sebagai `docs/hasil-modeling-rf.single-quantile.bak.md`.
Verifikasi 2026-08-24 sebelumnya — 816 tes lolos
(`.venv/bin/python3 -m unittest discover -p "test_*.py"` → `Ran 816 tests … OK`),
`QUANTILE_SET_A` (19 titik) sudah terpasang di `utils/modelling/evaluation.py`,
`walk_forward.py`, `model_common.py`, dan ketiga adapter model.

Legenda: ✅ selesai · 🔄 sedang berjalan · ⬜ belum mulai · ⚠️ selesai tapi
hasilnya usang / perlu dibaca dengan syarat · 🔒 diblokir oleh pihak lain
(pemilik data / izin komputasi).

---

## 0. Peta fase — status ringkas

| Fase  | Isi                                                                                                              | Status                                                                                                                                                                                                                                                                                                                                |
| ----- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A** | Data prep & preprocessing (CSV mentah → `train/test.parquet`)                                                    | ✅ selesai, artefak terverifikasi                                                                                                                                                                                                                                                                                                     |
| **B** | Prapemrosesan pemodelan + mesin evaluasi bersama (`modeling_prep`, `walk_forward`, `model_common`, `evaluation`) | ✅ selesai                                                                                                                                                                                                                                                                                                                            |
| **C** | Tiga model kandidat: Random Forest, XGBoost, LSTM                                                                | ✅ terimplementasi · ✅ ketiganya dilatih ulang di bawah K1 (RF 2026-08-25, XGB 2026-08-27, LSTM 2026-08-28) · ✅ dokumen hasil ketiganya ditulis (2026-08-29)                                                                                                                                                                        |
| **D** | Migrasi evaluasi multi-kuantil (butir 0a–0d)                                                                     | ✅ 0a ✅, 0b ✅, 0c ✅, 0d ✅ dokumen ditulis, `crossing_rate` ✅ diuji, seed LSTM ✅ diverifikasi, rearrangement XGBoost ✅ dikerjakan (2026-08-30) — **prasyarat wajib terpenuhi, RF kandidat terdepan di semua perbandingan**, bagian 18 sudah menyatakan ini tapi **belum resmi dibekukan** (menunggu persetujuan pemilik proyek) |
| **E** | Pembekuan pemenang + pembukaan test set Desember 2025                                                            | ⬜ siap dimulai, menunggu persetujuan Anda untuk membekukan RF                                                                                                                                                                                                                                                                        |
| **F** | Pekerjaan lanjutan: alokasi kuantil tersegmentasi, dekomposisi harian, SHAP                                      | ⬜ menunggu E                                                                                                                                                                                                                                                                                                                         |
| **G** | Hygiene rutin + item yang masih menunggu pemilik data                                                            | 🔄 berjalan terus                                                                                                                                                                                                                                                                                                                     |

Urutan D → E → F **tidak bebas**: membekukan pemenang di atas kriteria lama
berarti membekukan keputusan yang kriterianya sendiri sudah diganti
(bagian 21 `metodologi-pemodelan-dan-pemilihan-model.md`).

---

## Fase A — Data prep & preprocessing ✅

Pipeline 14 tahap dari `dataset/csv/*.csv` sampai
`dataset/model_ready/{train,test}.parquet` sudah terimplementasi penuh dan
terverifikasi terhadap output parquet-nya (`docs/pipeline-overview.md` bagian 2–3).
Orkestrasi: `utils/data_preprocessing/prepare_forecast_data.py::main()`.

### A1. Fitur & tahap pipeline (selesai)

- [x] **Integrasi region / lead-time** (2026-08-08) — `outlet_mapping.csv`
      (`kawasan`, `hari_pengiriman`) di-wire lewat
      `outlet_features.apply_region_features`; `lead_time_days` dihitung variabel
      per baris dari `(day_of_week, hari_pengiriman)`; target
      `target_lead_time_cumulative` dibangun `add_lead_time_target` pada jendela
      strictly-ke-depan `(H+1..H+lead_time_days)`, dengan tes leakage-safety.
      Spec: `2026-08-08-lead-time-integration-design.md`.
- [x] **Penanganan siklus hidup outlet** (2026-08-15) — `outlet_closures.csv` +
      `load_closures()`; `build_dense_panel(closures=...)` membuang hari tutup dan
      memberi `segment_id`; `SEGMENT_COLS` dioper ke seluruh fungsi lag/rolling/
      target/warmup/adapter, sehingga tidak ada jendela yang menyeberangi masa
      tutup; `detect_unrecorded_gaps()` memperingatkan gap ≥14 hari yang belum
      tercatat. Sebelum ini, 19.304 baris `Kuantitas = 0` difabrikasi dan membuat
      `branch_avg_daily_qty` KY011 salah 3,6×.
      Spec: `2026-08-15-outlet-lifecycle-handling-design.md`.
- [x] **Penanganan lonjakan (outlier)** di-wire ke `data_processing.ipynb`
      (2026-08-08) — urutan final: kalender → capping → target/lag/rolling, karena
      `apply_outlier_capping` butuh kolom event kalender untuk pengecualian
      event-window. `add_targets` tetap memakai `Kuantitas` mentah; lag/rolling/
      statistik cabang memakai `Kuantitas_capped`.
- [x] **`days_since_relocation`** — mitigasi untuk fitur lokasi yang ter-relabel
      mundur (kanonikalisasi nama berjalan sebelum fitur outlet, sehingga histori
      pra-relokasi ikut memakai `kota` pasca-relokasi). Terisi untuk 9/9 cabang
      relokasi: 4 tanggal exact (terbaca dari kode outlet baru di data mentah),
      5 tanggal _lower-bound proxy_ yang **perlu di-derive ulang** saat data baru
      masuk (lihat G3).
- [x] **QA dipindahkan dari notebook ke script** — `run_qa_checks()`
      (`prepare_forecast_data.py`) memuat 10+ cek: no-negative `Kuantitas`, no-
      duplicate (item, cabang, tanggal), `Kuantitas_capped ≤ Kuantitas`, target
      capped ≤ target mentah, tidak ada `kota == "Unknown"`, tidak ada cabang tanpa
      `kawasan`, satu cabang → satu kota, tidak ada baris di dalam interval tutup,
      `segment_id` mulai 1 & kontinu, tidak ada lubang tanggal dalam satu segmen.
      Dipanggil dari `main()` **dan** dari notebook.
      Masih notebook-only: spot-check leakage lag/rolling, rentang tanggal per
      outlet, dan section visual QA.
- [x] **Reklasifikasi kategori WIP-2 → FG** (2026-08-22) — 10 SKU
      (`FGS-00001/2/3/4/5/12/13/18/49/53`), 19.987 baris pindah kategori, lewat
      `EXPLICIT_CATEGORY_OVERRIDES`. Indeks WIP-2 (4) sengaja **tidak** dibebaskan
      (kebijakan stabilitas indeks, `preprocessing.md` bagian 4.12(e), Bagian 1).
- [x] **Gerbang konsistensi kategori** — `utils/eda/verify_category_consistency.py`,
      supaya kategori tidak bisa diam-diam bergeser lagi di refresh berikutnya.

### A2. Konfirmasi pemilik data (semuanya tertutup) ✅

Semua poin di bawah dulu berstatus "keputusan sudah di-hardcode, belum
di-sign-off". Semuanya sudah dikonfirmasi dan di-wire; rinciannya ada di
`preprocessing.md` dan riwayat git berkas ini.

- [x] **Santan/Gula Cendol (`xxx.FGS.00070/71`)** — periode awal tercatat dalam
      Gr, sesudahnya dalam Porsi; dikonversi (40 g dan 30 g per porsi) lalu
      disambung jadi satu deret. **Cendol Pandan (`xxx.FGS.00069`) tetap di-drop**
      (nama tidak identik di dua sisi + data jarang). 2026-08-10.
- [x] **Ayam Crispy Original/Spicy (`xxx.FGS.00067/68`) di-exclude, bukan
      digabung** — dua varian yang dijual paralel (190 hari-cabang keduanya muncul
      bersamaan), dan keduanya sudah discontinued. `EXPLICIT_ITEM_RENAMES`
      dikosongkan. 2026-08-10.
- [x] **`EXCLUDED_BRANCHES = {"Kebab Saudagar - Kutabumi"}`** — cabang sudah
      tidak beroperasi. 2026-08-09.
- [x] **Mapping `KY069`→`KY011` dan `TOD M1 Bandara`→`KY051`** — kode/nama lama
      dari cabang yang sama, bukan cabang berbeda. 2026-08-10.
- [x] **16 baris `Kota Override`** di `outlet_name_overrides.csv` lengkap dan
      diverifikasi (16/16 resolve benar, 0 mismatch); 8 di antaranya benar-benar
      mengubah nilai `kota`. Sign-off ulang 2026-08-16, termasuk `KY001` Kutabumi =
      `Kabupaten Tangerang` meski kolom `Kecamatan` di `outlets.csv` menyebut
      Jatiuwung.
- [x] **9 relokasi outlet di-wire** (6 "verified" + 3 "pending"), ketiga outlet
      baru (`KY070` Cadas, `KY071` Citayam, `KY072` Bintara) dikonfirmasi memakai
      histori outlet lama. 2026-08-11.
- [x] **Provenance `kawasan`/`hari_pengiriman`** — resmi dari tim SCM, bukan
      asumsi. 2026-08-10. Tiga cabang yang sempat kosong diisi 2026-08-11 dan
      dikonfirmasi 2026-08-16; aturan pencocokan pra-relokasi (B-8) ditutup
      2026-08-17.
- [x] **KY056 Tigaraksa completeness 92,3%** — tutup sementara 52 hari
      (2024-10-01 s/d 2024-11-21), bukan gap pelaporan. 2026-08-10.
- [x] **KY068 Kramatwatu gap 13 hari** (2025-06-28 s/d 2025-07-10) — tutup
      sementara, sudah masuk `outlet_closures.csv`. 2026-08-17.
- [x] **27 SKU multi-kategori** diselesaikan per pasangan: `Minuman`↔`Minuman -
FG` dan `Snack`↔`Snack (FG)` = sinonim (collapse); `WIP-2`↔`FG` = kategori
      berbeda (dibiarkan time-varying — lalu sebagian direklasifikasi 2026-08-22,
      lihat A1); `FGS-00014` override eksplisit ke `Minuman - FG`. 2026-08-10.
- [x] **Negative `Kuantitas`** — salah input sistem sumber; `feb-24_No_Minus`
      adalah versi final yang benar. 0 baris negatif di data saat ini. 2026-08-10.
      (Soft-check-nya tetap wajib diulang tiap refresh — lihat G1.)

### A3. Yang masih terbuka di Fase A

- [ ] **Fallback cold-start untuk pasangan yang gagal `MIN_HISTORY_DAYS` (60
      hari)** — 842 dari 3.882 pasangan (22%) tidak dapat ramalan sama sekali.
      Diukur 2026-08-17: **menurunkan ambang bukan jawabannya.** Ambang 28
      menambah 167 pasangan tetapi hanya 0,023% volume permintaan (4.773 dari 20,9
      juta unit) dan 2.428 baris latih (0,18%), karena `LOOKBACK` + `lag_28`/
      `roll_28` yang membatasi, bukan ambangnya; MAE baseline malah turun karena
      dilusi (73,7% target kohort baru nol), sehingga angka tidak lagi sebanding
      antar-run. Config 28 juga menabrak guard `add_event_flag`. **Cakupan untuk
      621 pasangan sisanya harus lewat fallback terpisah**, dan fallback itu harus
      mengalahkan pinball 0,384 (`roll_mean_7` pada kohort 28–59 hari, Des 2025)
      supaya layak pakai. `MIN_HISTORY_DAYS` tetap 60.
- [x] **Kontradiksi target latih `capped` vs mentah — DITUTUP 2026-08-24
      (pemilik proyek): latih di `capped`, nilai (K1) di mentah.** Sebelumnya kode
      memakai satu target mentah untuk latih _dan_ nilai, sementara konfirmasi
      pemilik data 2026-08-17 menyimpulkan capped — kontradiksi yang harus
      diselesaikan sebelum butir 0c, karena mengganti target sesudah run berarti
      mengulang ~157–172 jam. Keputusannya membelah keduanya: dilatih di capped
      (porsi yang dipangkas adalah proksi pre-order yang ditangani jalur manual,
      B-3; komponen itu tidak dapat diprediksi karena buku pesanan tidak terekam,
      B-1/B-2), dinilai di mentah (kriteria yang dihitung pada deret yang sudah
      dipangkas bisa diperbaiki dengan memangkas lebih banyak — sifat yang tidak
      boleh dimiliki kriteria pemilihan model). Terpasang di kode:
      `modeling_prep.TRAIN_TARGET_COL`/`EVAL_TARGET_COL` (nama `TARGET_COL` yang
      ambigu dihapus supaya tiap pemanggil harus memilih), satu seam label latih
      `model_common.train_target()` untuk ketiga model, `walk_forward.run_fold()`
      menilai di target mentah, guard baru di `eligible_rows()` dan
      `prepare_forecast_data.assert_targets_agree_on_nulls()` untuk pola nilai
      kosong yang tidak sepakat, penjaga G2 `sequence_windows` diperluas ke kedua
      target, dan ketiga bundle mencatat `train_target`/`eval_target`. bagian 5
      `metodologi-pemodelan-dan-pemilihan-model.md` sudah diperbaiki. 752 tes lolos
      (naik dari 739; 13 tes baru, ditulis TDD merah lebih dulu).
      **Ongkos yang dinyatakan terbuka:** selisih kedua target di Desember 2025
      adalah 1.223 dari 50.692 baris (2,41%) dan 44.470 unit (2,03% massa
      permintaan). Ketiga model membayarnya sama besar, jadi peringkat tidak
      terpengaruh — level absolut K1-lah yang bergeser, dan itu harus dibaca
      sebagai jarak terhadap permintaan nyata.
- [x] **Spike yang berdiri sendiri: pre-order atau bukan? — DITUTUP 2026-08-24
      (konfirmasi pemilik data). Jawabannya bukan salah satu dari dua opsi yang
      ditanyakan: bercampur, dan tidak dapat dipisahkan dari data ini.** Pertanyaan
      aslinya: dari 7.552 baris yang di-cap, 49,8% berada di cabang-hari dengan ≥3
      item melonjak serentak (tanda pesanan besar), tetapi 38,6% melonjak sendirian
      — condong ke akhir pekan dan didominasi Packaging (60,1%). Tiga langkah
      menyempitkannya sampai bisa dijawab:
  1. **Hipotesis restock gugur** (`utils/eda/analyze_spike_recovery.py`) — lonjakan
     bersifat aditif: permintaan 7 hari sesudah tidak turun dibanding 7 hari
     sebelum (lonjakan sendirian +1,4%, p = 0,24; dengan lonjakan tetangga
     dikeluarkan dari jendela +0,2%, p = 1,0), dan kadensinya juga tidak
     berubah (4,95 → 4,97 hari bergerak per 7 hari). Ia berdiri **di atas**
     garis dasar, bukan meminjam permintaan hari berikutnya.
  2. **Yang "sendirian" ternyata tidak sendirian**
     (`utils/eda/analyze_spike_comovement.py`) — pada 1.157 cabang-hari lonjakan
     Packaging berdiri sendiri, Nasi Kebuli justru naik ke 1,92× median-nya
     (hari biasa 0,91×), di persentil **0,821** dari sebaran cabangnya sendiri
     setelah dicocokkan per hari-dalam-minggu (H0 = 0,500; p = 2,9e−137).
     Sambal dan Ayam Kebuli searah. Jadi itu **peristiwa permintaan nyata**,
     bukan pergerakan kemasan saja; ia terbaca seolah hanya menyentuh kemasan
     karena ambang 5× bersifat relatif terhadap median pasangan.
  3. **Konfirmasi pemilik data (2026-08-24)** — di akhir pekan memang sering
     terjadi lonjakan, dan datangnya lewat **kedua jalur**: sebagian pesanan,
     sebagian pelanggan langsung.

  Ditulis di `docs/analisis-lonjakan-permintaan.md` (nama berkasnya sengaja
  bukan `analisis-lonjakan-packaging.md` — isinya ternyata bukan tentang
  kemasan) dan sebagai catatan bertanggal di B-3 `batasan-penelitian.md`.
  **Konsekuensi yang belum masuk pertimbangan sebelumnya:** justifikasi capping
  ("komponen ini tidak dapat diprediksi") berlaku untuk pesanan (B-1/B-2) tapi
  **tidak** untuk lonjakan pelanggan langsung akhir pekan — `day_of_week` dan
  `is_weekend` ada di 56 `FEATURE_COLS`. Untuk baris yang dipangkas, capping
  menghapus sebagian sinyal yang dapat dipelajari. Dampaknya terbatas dan
  terarah: item bervolume besar tidak terkena ambang (Nasi Kebuli 2,04×, jauh
  di bawah 5×), yang terpangkas item bervolume kecil (83,6% baris di-cap punya
  median pasangan ≤10 unit/hari; Rice Bowl, Loyang). Prediksi yang bisa diuji →
  butir 0d. **Keputusan target tidak dibuka ulang** — latih di `capped`, K1 di
  mentah; jawaban ini justru menguatkannya karena K1 di target mentah tidak
  bergantung pada mekanisme capping.

---

## Fase B — Prapemrosesan pemodelan & mesin evaluasi bersama ✅

- [x] `utils/modelling/modeling_prep.py` — `featured.parquet` → `model_input.parquet`:
      `is_event_driven`, `demand_segment` (Syntetos-Boylan), `fold_id` (5 fold
      jendela mengembang), imputasi yang mempertahankan makna, pengindeksan
      kategorikal yang tidak pernah dibangun ulang, `FEATURE_COLS` 56 kolom, dua
      adapter (tabular & sekuens) di bawah satu kontrak.
      Spec: `2026-08-12-modeling-preprocessing-design.md`.
- [x] `utils/modelling/walk_forward.py` — runner 5 fold dengan `validate_contract()`,
      memastikan ketiga model dinilai pada baris, kunci, target, dan fold yang
      identik.
- [x] `utils/modelling/purging.py` + `sequence_windows.py` — purging horizon dan
      jendela sekuens LSTM.
- [x] `utils/modelling/model_common.py` — random search dengan checkpoint/resume,
      ekspansi one-hot, format bundle.
- [x] `utils/modelling/evaluation.py` — pinball per τ, MAE, coverage, fill rate,
      crossing rate; tiga baseline naif sebagai lantai.
- [x] Kelayakan baris (tiga potongan) dan enam aturan anti-kebocoran
      terdokumentasi di bagian 8–9 `metodologi-pemodelan-dan-pemilihan-model.md`.

---

## Fase C — Tiga model kandidat ✅ training · ✅ dokumen

Ketiganya **terimplementasi, dijalankan ulang penuh, dan dokumen hasilnya
ditulis** di atas data pasca-reklasifikasi kategori dan kriteria K1
(multi-kuantil) yang berlaku: Random Forest 2026-08-25, XGBoost 2026-08-27,
LSTM 2026-08-28; `hasil-modeling-{rf,xgb,lstm}.md` ditulis ulang 2026-08-29.
Yang masih tersisa **bukan** di fase ini lagi — pemilihan pemenang lintas
model (bagian 16/18 metodologi) sudah ditulis tapi status keputusannya
ditahan, itu ada di 0d.

- [x] `utils/modelling/model_random_forest.py` — quantile forest (`quantile-forest`).
      Spec: `2026-08-18-random-forest-modeling-design.md`.
- [x] `utils/modelling/model_xgboost.py` — `reg:quantileerror`, protokol dua fit (early
      stopping pada tail 30 hari yang di-purge, lalu refit penuh).
      Spec: `2026-08-19-xgboost-modeling-design.md`.
- [x] `utils/modelling/model_lstm.py` — LSTM dengan pinball loss.
      Spec: `2026-08-19-lstm-modeling-design.md`.
- [x] Tiga notebook modeling (`modeling_{rf,xgb,lstm}.ipynb`) dan tiga dokumen
      hasil (`hasil-modeling-{rf,xgb,lstm}.md`).
- [x] ✅ **`models/*.joblib` (RF/XGB/LSTM) ketiganya berlaku kembali —
      tidak ada lagi yang usang.** Ketiganya sempat dilatih sebelum reklasifikasi
      WIP-2 (2026-08-22), sehingga gagal secara diam-diam: bundle tetap ter-load,
      tetap menemukan seluruh kolom, dan tetap mengembalikan angka yang
      meyakinkan — yang rusak adalah rute 10 SKU itu di dalam model (kolom
      one-hot mati / level kategori tak terjangkau / baris embedding mati).
      Rinciannya di bagian 0 `pipeline-overview.md`.
      **Bahaya "ter-load diam-diam" itu sudah ditutup 2026-08-24:** ketiganya
      diganti nama menjadi `*_q90.single-quantile.bak.joblib`, sehingga
      `MODEL_FILE` ketiga model tidak lagi resolve ke apa pun dan sebuah
      `load_bundle()` yang keliru gagal keras alih-alih mengembalikan angka yang
      meyakinkan. Berkasnya sendiri disimpan, sejalan dengan keputusan yang sama
      untuk artefak pencarian di butir 0c.
  - **`models/random_forest_q90.joblib` (2026-08-25):** 1.349.011 baris, 56
    kolom, 19 titik kuantil, 826 MB, device cpu.
  - **`models/xgboost_q90.joblib` (2026-08-27):** 1.349.011 baris, 56 kolom,
    encoding native, 201 ronde boosting, 19 titik kuantil, device cpu (fit
    final) — pencarian hyperparameter-nya sendiri jalan di GPU, lihat catatan
    mesin-terpisah di 0c.
  - **`models/lstm_q90.joblib` (2026-08-28):** 1.349.011 baris, best_epoch 5,
    19 titik kuantil, device cpu.

  Catatan nama: ketiganya masih bernama `..._q90` meski isinya 19 titik —
  sengaja tidak diganti di tengah migrasi supaya jalur `MODEL_FILE` tidak
  bergeser; penggantian namanya masuk hygiene Fase G, bukan blocker.

- [x] ✅ **`hasil-modeling-{xgb,lstm}.md` ditulis ulang dari nol (2026-08-29),
      ketiga dokumen hasil sekarang berlaku.** Struktur sama dengan
      `hasil-modeling-rf.md` (9 bagian), semua angka dari
      `xgb_walk_forward_results.csv`/`lstm_walk_forward_results.csv`,
      `{xgb,lstm}_search_results.csv`, dan output notebook. Temuan baru yang
      masuk kedua dokumen: `crossing_rate` XGBoost 0,9767 dan LSTM 0,4345 (RF 0%
      struktural); untuk XGBoost, peringkat kandidat pencarian jauh kurang stabil
      terhadap K1 daripada RF (Spearman 0,73 vs 0,975); untuk LSTM, ruang
      pencarian berubah dari run kuantil-tunggal (tidak bisa dibandingkan
      peringkatnya) dan rentang K1 antar 3 seed (0,2517) delapan kali lebih lebar
      daripada jarak K1 LSTM ke RF (0,0310). `hasil-modeling-rf.md` tetap berlaku
      sejak 2026-08-25.

---

## Fase D — Migrasi evaluasi multi-kuantil 🔄

Kriteria utama (K1) berubah dari pinball@0,9 tunggal menjadi **rata-rata pinball
di seluruh `QUANTILE_SET`** (Tahap A: 19 titik, 0,05–0,95).
Spec: `2026-08-22-multi-quantile-evaluation-design.md`;
checklist eksekusi: `2026-08-22-model-comparison-refactor-migration.md`.

### 0a — Revisi dokumentasi & spec ✅ (2026-08-24)

- [x] Spec XGBoost: `quantile_alpha` → daftar `QUANTILE_SET`.
- [x] Spec LSTM: head 1 neuron → `len(QUANTILE_SET)` neuron, loss = jumlah
      pinball lintas kuantil; anggaran & ruang pencarian dipulihkan.
- [x] Spec Random Forest: walk-forward membaca seluruh titik `QUANTILE_SET`
      dari forest yang sama.
- [x] `metodologi-pemodelan-dan-pemilihan-model.md` bagian 15 (definisi K1), bagian 17
      (tangga K1/K2), bagian 19 (protokol pembukaan test set), bagian 21 (rencana kerja).
- [x] Spec segmentasi kuantil: Bagian 4 ditandai _inherited_ (kapabilitas
      multi-kuantil sudah dimiliki ketiga model sebelum pemenang dipilih).
- [x] Koreksi bertanggal di `batasan-penelitian.md` B-9 (paragraf
      "Konsekuensi" yang menyatakan proses pemilihan model tidak berubah).
- [x] bagian 16 dan bagian 18 `metodologi-…` diberi penanda "menunggu penggantian" —
      ditulis ulang di butir 0d, bukan sekarang.

### 0b — Implementasi kode multi-kuantil ✅ (2026-08-24)

- [x] `utils/modelling/evaluation.py` — `QUANTILE_SET_A` (19 titik), `QUANTILE_SET_B`,
      `resolve_quantile_set()`, pinball per τ + K1.
- [x] `utils/modelling/walk_forward.py`, `utils/modelling/model_common.py` — `alpha: float` diganti
      `quantiles: tuple` di seluruh jalur; kontrak `fit_predict` kini
      `(n, len(QUANTILE_SET))`.
- [x] `utils/model_{random_forest,xgboost,lstm}.py` — ketiganya memprediksi
      seluruh `QUANTILE_SET`; bundle memakai kunci `quantiles`, dengan
      `QUANTILE = 0.9` dipertahankan terpisah sebagai konstanta service level
      (B-9).
- [x] Ruang pencarian LSTM dipulihkan ke 144 + protokol 3 seed
      (`lstm.run_seed_repeats()` → `lstm_seed_repeats.csv`).
- [x] Guard checkpoint: CSV pencarian tanpa kolom `headline_quantile` ditolak
      dengan pesan eksplisit — mencegah run kuantil-tunggal terbaca sebagai K1.
- [x] Ketiga notebook diubah (**belum dijalankan** — itu butir 0c).
- [x] 739 tes lolos (diverifikasi ulang 2026-08-24).

### 0c — Menjalankan ulang ketiga notebook ✅ selesai (2026-08-29)

Anggaran berlaku: **RF 18, XGBoost 30, LSTM 30 kandidat** (ruang 144 + 3 seed
pada pemenang). Perkiraan ongkos: **~213–228 jam ≈ 8,9–9,5 hari komputasi
nonstop berurutan** sesudah koreksi 2026-08-25 (angka lama ~157–172 jam
meleset pada bagian XGBoost-nya). Rincian lama per model per tahap di
bagian "Perkiraan ongkos Fase 3" `2026-08-22-model-comparison-refactor-migration.md`;
angka XGBoost di sana **sudah tidak berlaku** — lihat bagian 3bis dan bagian 0
`2026-08-24-distributed-gpu-training-design.md`.

**Mekanisme eksekusi terdistribusi sudah terpasang (2026-08-24).** Pencarian
dapat dipecah antar mesin lewat `model_common.run_search(..., only=[...],
provenance={...})` dan disatukan kembali oleh `model_common.merge_shards()`,
yang memakai ulang guard `_assert_checkpoint_matches()` sehingga shard yang
tertukar, berlubang, atau berasal dari run kuantil tunggal tertolak. Rencana
mesin per tahap, probe paritas device yang mengesahkan pemecahan itu, dan
alasan walk-forward tetap di Mac ada di
`docs/superpowers/specs/2026-08-24-distributed-gpu-training-design.md`.

Ketiga notebook sudah ikut di-wire lewat `utils/modelling/run_config.py`:
`FORECAST_SHARD`, `FORECAST_DEVICE`, `FORECAST_MODEL_INPUT`, dan
`FORECAST_CHECKPOINT_DIR` dibaca di sel pertama, sehingga satu notebook yang
sama jalan di Mac, Kaggle, dan Colab tanpa diedit per mesin. **Tanpa satu pun
env var, ketiganya berperilaku persis seperti sebelumnya** — diverifikasi
dengan menjalankan sel pertama masing-masing: nama berkas keluarannya identik.
Sel yang memanggil `select_best()` diberi guard `assert SHARD is None`, karena
memilih pemenang dari sebagian kandidat menghasilkan angka yang tampak
sepenuhnya wajar.

Ini **bukan** izin menjalankan 0c; ia hanya menghapus kode sebagai penghalang.

- [x] **Langkah 0 — biarkan guard checkpoint berbunyi.** Jalankan sel pencarian
      XGBoost dengan ketiga CSV lama **masih di tempatnya**; ia harus berhenti
      dalam hitungan detik dengan `ValueError` yang menyebut "berasal dari run
      kuantil tunggal". Kalau **tidak** berbunyi, hentikan Fase 3 — berarti
      guard-nya tidak bekerja dan setiap angka sesudahnya tidak dapat dipercaya.
      (Diverifikasi 2026-08-24: ketiga CSV masih ada dan memang belum punya kolom
      `headline_quantile`, jadi kondisi ujinya masih utuh.)
      **DIJALANKAN 2026-08-24 — guard berbunyi.** XGBoost berhenti setelah **3,2
      detik** dan Random Forest setelah **2,8 detik**, keduanya dengan
      `ValueError: … berasal dari run kuantil tunggal (tidak ada kolom
'headline_quantile')`, sebelum satu pohon pun dibangun; hash
      `xgb_search_results.csv` sebelum dan sesudah identik, jadi guard menolak
      tanpa menulis apa pun. Jalur LSTM tidak diuji dengan cara yang sama karena
      `run_search`-nya memanggil `bind_panel()` di dalam daftar argumennya,
      sehingga window index 1,5 juta baris dibangun sebelum guard terbaca —
      ongkosnya menit, bukan detik; guard-nya kode yang sama dan
      `lstm_search_results.csv` sama-sama tidak punya kolom itu.
- [x] ~~Baru **hapus**~~ **Diganti nama, bukan dihapus (2026-08-24, pemilik
      proyek).** Sembilan artefak run kuantil-tunggal diberi akhiran
      `.single-quantile.bak.<ext>` di `dataset/model_ready/`: ketiga
      `*_search_results.csv`, ketiga `*_best_params.json`, dan ketiga
      `*_walk_forward_results.csv`. Cakupannya diperluas dari yang semula tertulis
      (tiga CSV + `rf_best_params.json`) karena keenam berkas lain sama-sama lahir
      di bawah kriteria lama, dan yang tertinggal di folder itu akan terbaca
      sebagai "parameter terbaik" oleh siapa pun yang membukanya sebelum run baru
      selesai.

  Efeknya ke pipeline identik dengan penghapusan — `SEARCH_FILE`,
  `BEST_PARAMS_FILE`, dan `RESULTS_FILE` ketiga model diverifikasi tidak lagi
  ada, jadi pencarian mulai dari nol. Yang dibeli dengan 25 KB itu adalah
  satu-satunya catatan pinball@0,9 **per kandidat** yang masih tersisa:
  ketiga `hasil-modeling-*.md` yang meringkasnya akan ditulis ulang dari nol
  di butir 0d, sehingga tanpa berkas ini pertanyaan "apakah peringkat kandidat
  berubah setelah pindah ke K1?" — yang wajar muncul saat menulis bab hasil —
  tidak akan punya jawaban lagi. Angkanya tetap **tidak boleh** disandingkan
  dengan K1 baru sebagai perbandingan langsung (T-10).

- [x] **Tahap 0 probe device untuk XGBoost — 4 dari 5 tertutup (2026-08-25).**
      `candidate_id 0` dijalankan penuh (`SEARCH_FOLDS = (3, 5)`, 19 kuantil) di Mac
      `cpu` dan Kaggle `cuda:0`, commit `ce84707` di keduanya. **Paritas device:
      selisih K1 0,124%** (2,960221 lokal lawan 2,963888 GPU) terhadap ambang 2% —
      jadi peringkat kandidat yang lahir di GPU berlaku di CPU tempat pemenangnya
      di-refit. **Pengganda: ×7,96** (5,54 jam → 0,70 jam), di atas estimasi ~6×.
      Versi paket cocok persis (`xgboost 2.1.4`, `numpy 2.0.2`, `pandas 2.3.3`,
      `scikit-learn 1.6.1`); Python 3.12.13 di Kaggle lawan 3.9.6 lokal dicatat
      sebagai perbedaan lingkungan yang diketahui. Rinciannya, termasuk mengapa
      selisih 0,124% itu early stopping dan bukan noise, ada di bagian 3bis
      `2026-08-24-distributed-gpu-training-design.md`.
      **Masih terbuka:** akuntansi kuota T4×2 (sesi ini hanya memakai `cuda:0`).
      **Angka paritas ini sudah dibawa ke `docs/hasil-modeling-xgb.md`** (paragraf
      pembuka, "selisih K1 0,124%, jauh di bawah ambang 2%") saat 0d menulis
      ulang dokumen itu 2026-08-29 — versi ringkas, tanpa dua angka mentah
      (2,960221 vs 2,963888), pengganda ×7,96, atau detail versi paket di atas.
      Kalau detail itu perlu ada di dokumen hasil juga (bukan cuma di spec
      ini), belum dikerjakan.
- [x] **Koreksi estimasi ongkos CPU XGBoost: 64,6 jam → ~120 jam.** Candidate 0
      sendirian memakan 5,54 jam di Mac; dibobot ke 30 kandidat, angka lama meleset
      ~2×. Jadwal GPU tidak bergeser (~15 GPU-jam, ~7,6 jam per GPU di T4×2), tetapi
      **risiko 7.1 jadi lebih mahal**: mengulang satu kandidat `one_hot` +
      `max_depth=10` yang OOM di GPU berongkos belasan jam di CPU, bukan beberapa
      jam. Tujuh kandidat `one_hot` (id 1, 3, 7, 13, 19, 22, 24), tiga di antaranya
      berkedalaman 10.
- [x] **Keputusan 2026-08-25 (pemilik proyek): seluruh Fase 3 dijalankan di CPU
      Mac lokal — XGBoost dan LSTM tidak jadi dikirim ke GPU sewaan.** RF memang
      sudah lokal di rencana mana pun. Rencana GPU terdistribusi karenanya
      **superseded pada bagian alokasi mesinnya**; hasil probe Tahap 0, aturan dua
      lapis, dan seam kode (`only=`, `provenance=`, `merge_shards()`,
      `run_config`) tetap berlaku dan tetap tidak-aktif secara default. Alasan
      lengkap dan tabel ongkosnya di bagian 0 `2026-08-24-distributed-gpu-training-design.md`.

  Yang dibeli: tidak ada penyerahan device sama sekali (bandingkan rencana GPU
  yang memperingkat kandidat di GPU lalu me-refit pemenangnya di CPU), K3 dalam
  bacaan paling ketat karena ketiga model diukur di CPU yang sama dan langsung
  sebanding dengan run 2026-08-18/19/20, dan komentar `SATU MODEL = SATU DEVICE`
  di sel 1 `modeling_xgb.ipynb` tidak perlu diperlonggar. Untuk LSTM tidak ada
  kecepatan lokal yang ditinggalkan: **MPS sudah diukur kalah 2× dari CPU** di
  mesin ini (0,392 lawan 0,193 s/batch, bagian 3 `docs/hasil-modeling-lstm.md`).

  Yang dibayar — perkiraan sisa Fase 3, **berurutan, tidak boleh paralel**
  (dua model yang berebut core menghasilkan wall time yang mengukur kontensi,
  bukan model — lubang K3 yang sama, hanya di dalam satu mesin):

  | Model                                   |                       Perkiraan | Dasar                                    |
  | --------------------------------------- | ------------------------------: | ---------------------------------------- |
  | Random Forest                           |                        ~4,8 jam | estimasi migrasi, belum dikoreksi        |
  | XGBoost (29 kandidat sisa + WF + final) |                    **~125 jam** | candidate 0 terukur 5,54 jam             |
  | LSTM                                    |                      ~83–98 jam | estimasi migrasi, **belum diverifikasi** |
  | **Total**                               | **~213–228 jam ≈ 8,9–9,5 hari** |                                          |

- [x] **Candidate 0 XGBoost tidak perlu diulang.** Baris CPU-nya dari probe
      Tahap 0 diganti nama menjadi `dataset/model_ready/xgb_search_results.csv`
      (2026-08-25) dan diverifikasi lolos `_assert_checkpoint_matches()` terhadap
      ruang pencarian saat ini, sehingga `resume=True` melewatinya — 5,54 jam
      hemat. Baris **GPU** candidate 0 dari Kaggle sengaja **tidak** ikut:
      mencampur dua device dalam satu pencarian persis yang dihindari keputusan di
      atas. Ia tinggal sebagai bukti probe (3d).

- [x] **Keputusan "seluruh Fase 3 di CPU Mac lokal" (2026-08-25) DISUSUL
      sehari kemudian (2026-08-26): pencarian XGBoost & LSTM dipindah ke GPU
      Windows, walk-forward + fit final tetap di CPU Mac.** Bukan pembatalan diam
      -diam — ada di komentar kode (`modeling_xgb.ipynb` sel 10: _"cuda dipakai
      tahap pencarian sejak 2026-08-26"_) dan di `docs/runbook-pencarian-gpu-windows.md`
      serta `CLAUDE.md`, yang belum diikuti pembaruan di berkas ini sampai
      2026-08-29. Verifikasi dari data: **seluruh 30 kandidat XGBoost dan 30
      kandidat LSTM di `{xgb,lstm}_search_results.csv` bercatat `device=cuda`,
      `commit=e074421`** — konsisten dengan RTX 4060 Ti 8 GB. Pagar yang menjaga
      K3 tetap sah: walk-forward dan fit final ketiga model **tetap satu mesin
      (Mac, cpu)** — dikonfirmasi dari output notebook (`device cpu` di bundle
      XGBoost & LSTM final, RF sudah cpu dari awal) — jadi K1/K3 dari walk-forward
      tetap sebanding lintas model persis seperti klaim RF di atas ("diukur di CPU
      yang sama"). **Yang tidak lagi sebanding:** wall-clock tahap **pencarian**
      saja — estimasi ~125 jam (XGBoost) dan ~83–98 jam (LSTM) dihitung untuk CPU
      Mac, sementara yang benar-benar terjadi adalah GPU Windows. Jangan baca
      `elapsed_seconds` total pencarian (XGBoost ~14,7 jam, LSTM ~4,2 jam,
      dijumlah dari `elapsed_seconds` per kandidat) sebagai "jauh lebih cepat dari
      perkiraan" — itu device yang beda, bukan pipeline yang lebih efisien.
      Paritas GPU↔CPU untuk peringkat kandidat sudah divalidasi lebih dulu (selisih
      K1 0,124% pada probe Tahap 0, jauh di bawah ambang 2%), jadi pemenang yang
      dipilih di GPU tetap dipercaya.
- [x] ~~**Verifikasi estimasi LSTM sebelum mengomit 83–98 jam.**~~ —
      **superseded** oleh butir di atas: begitu pencarian pindah ke GPU, jepitan
      `elapsed_seconds` kandidat pertama vs 3.412 s (estimasi CPU) tidak lagi
      relevan sebagai gerbang keputusan. Kandidat 0 LSTM di GPU tercatat 351,3 s —
      jauh di bawah 3.412 s, tapi itu murni karena device, bukan bukti estimasi
      CPU-nya salah.
- [x] **Random Forest — SELESAI 2026-08-25.** Benchmark → pencarian 18 kandidat
      (`SEARCH_FOLDS = (3, 5)`) → walk-forward 5 fold → fit final, keempatnya
      tuntas, 0 kandidat gagal. **Wall clock ~5,6 jam** (benchmark 9,7 mnt →
      pencarian 3,85 jam → WF ~45 mnt → fit final ~48 mnt) lawan estimasi ~4,8 jam:
      meleset **+17%**, jauh lebih jinak daripada XGBoost yang meleset ~1,87×.
      `device=cpu`, commit `5325b55`.

  **K1 = 2,8508** di potongan fold bersih 1/2/4 (kriteria resmi), lawan 4,8603
  milik `naive_roll_mean_7` — 41% lebih baik. Gabungan 5 fold 2,8621. **G0
  lolos**: RF menang pinball@0,9 di kelima fold, margin 40,5%–48,7%. Pemenang
  pencarian kandidat 1 (`max_depth=20`, `min_samples_leaf=20`,
  `max_features=1.0`, `one_hot=False`, `log_target=False`). Angka lengkapnya di
  `docs/hasil-modeling-rf.md`.

  Dua hal yang lahir dari run ini dan berlaku lintas model, jadi dicatat di sini
  dan bukan hanya di dokumen hasil RF:
  1. **Peringkat kandidat nyaris tidak berubah setelah pindah ke K1** —
     Spearman ρ = 0,975, Kendall τ = 0,895 terhadap peringkat pinball@0,9 di
     `rf_search_results.single-quantile.bak.csv` (ke-18 kandidat identik id per
     id, diverifikasi kolom demi kolom). Pemenangnya **tetap berubah** (17 → 1)
     karena di kriteria lama keduanya terpisah 0,0004; di K1 jaraknya 0,0177.
     Inilah imbalan dari keputusan mengganti nama artefak lama alih-alih
     menghapusnya. **Jangan digeneralkan ke XGB/LSTM** — keduanya punya
     mekanisme yang berinteraksi dengan jumlah titik kuantil (`multi_strategy`,
     kepala keluaran multi-titik), RF tidak.
  2. **Ongkos 19 titik kuantil di RF hanya ×1,47** (benchmark konfigurasi sama:
     6,6 → 9,7 menit), karena seluruh titik dibaca dari daun yang sama.
     Bandingkan XGBoost ×15,2. Ini menegaskan estimasi ongkos ketiga model
     memang tidak boleh diturunkan dari satu pengganda bersama.

- [x] **XGBoost — SELESAI 2026-08-27.** Benchmark → pencarian 30 kandidat
      (GPU Windows) → walk-forward 5 fold (CPU Mac) → fit final (CPU Mac),
      keempatnya tuntas. Search 30/30 kandidat lolos (0 gagal), `device=cuda`,
      `commit=e074421` di seluruh baris. Fit final: 1.349.011 baris, 56 kolom,
      encoding `native`, 201 ronde boosting, `device=cpu`, 19 titik kuantil.

  **K1 = 2,9433** di potongan fold bersih 1/2/4 (2,9197 gabungan 5 fold),
  lawan **4,8603** milik `naive_roll_mean_7` — menang, tapi **kalah** dari RF
  (2,8508) dan LSTM (2,8818) di K1. `best_iteration` per fold: [375, 242,
  255, 297, 390]. `mae@0,9` 13,408 (lebih rendah dari RF 15,081 — XGBoost
  lebih presisi di median tapi lebih longgar K1-nya), `coverage@0,9` 0,902.
  Pemenang pencarian: `max_depth=10`, `learning_rate=0,05`,
  `min_child_weight=1`, `subsample=0,7`, `colsample_bytree=0,5`,
  `reg_lambda=10,0`, `encoding=native`, `log_target=False`.

  **🆕 `crossing_rate` = 0,9767 (97,7% baris)** — jauh di atas RF (0%
  struktural). Lihat catatan 🆕 di 0d: ini gerbang K2 yang perlu diperiksa
  sebelum XGBoost bisa disandingkan adil dengan RF/LSTM di bagian 16/18.
  Sinkron kode: 18 fungsi + 15 konstanta identik dengan
  `utils/modelling/model_xgboost.py` (sel 10 notebook).

- [x] **LSTM — SELESAI 2026-08-28.** Benchmark (CPU, 961,1 s, `best_epoch`
      2, `sec_per_epoch` 106,8) → pencarian 30 kandidat pada ruang 144 (GPU
      Windows) → pengulangan 3 seed pada pemenang (seed 42/43/44, fold 3&5) →
      walk-forward 5 fold (CPU Mac) → fit final (CPU Mac), semuanya tuntas.
      Fit final: `best_epoch=5`, 1.349.011 baris, 19 titik kuantil.

  **K1 = 2,8818** di fold bersih 1/2/4 (2,8828 gabungan 5 fold) — **kedua
  terbaik**, di antara RF (2,8508) dan XGBoost (2,9433), jarak ke RF hanya
  0,0310. `mae@0,9` 13,929, `coverage@0,9` 0,906. Pemenang pencarian:
  `hidden_size=256`, `num_layers=2`, `dropout=0`, `learning_rate=0,0003`,
  `batch_size=2048`, `log_target=True`, `grad_clip=1,0`.

  **Pengulangan 3 seed**: K1 min 2,8399 (seed 44) — mean 2,9310 — max 3,0915
  (seed 43), rentang **0,2517** — lebih lebar dari jarak LSTM↔RF (0,0310),
  jadi keunggulan/kekalahan LSTM terhadap RF **tidak bisa dipisahkan dari
  derau seed** dengan run tunggal ini. Selisih baris seed-42 terhadap baris
  pemenang di search: +0,000000 (konsisten, cek nondeterminisme lolos).

  **🆕 `crossing_rate` = 0,4345 (43,4% baris)** — di antara RF (0%) dan
  XGBoost (97,7%). Lihat catatan 🆕 di 0d.

- [x] Catat selama run: wall clock per tahap per model; `crossing_rate` XGB &
      LSTM (RF harus 0 secara struktural — kalau tidak, ada bug); selisih baris
      seed 42 di `lstm_seed_repeats.csv` terhadap baris pemenang di
      `lstm_search_results.csv` (harus nol, kalau tidak yang terukur bukan varians
      seed melainkan nondeterminisme); rentang K1 antar seed dibaca bersama jarak
      K1 antar model.
      **RF tercatat 2026-08-25:** wall clock per tahap ada di butir RF di atas dan
      bagian 7 `hasil-modeling-rf.md`; `crossing_rate = 0,0000` di **seluruh** baris
      pencarian, benchmark, dan walk-forward — cek strukturalnya lolos.
      **XGBoost & LSTM tercatat 2026-08-29** (angka di dua butir di atas):
      `crossing_rate` **97,7%** (XGBoost) dan **43,4%** (LSTM) — **jauh** di atas
      0%, dan ini **belum dijelaskan**. Wall clock tahap pencarian tidak
      sebanding lintas device (lihat catatan mesin-terpisah di atas); wall clock
      walk-forward/fit final per tahap **belum tercatat eksplisit** di output
      notebook (tidak ada printout durasi seperti RF) — kalau butuh angka pasti
      untuk bagian 7 dokumen hasil, baca ulang dari log run atau timestamp berkas
      checkpoint. Selisih baris seed 42 LSTM: 0,000000 (lolos). Rentang K1 antar
      seed LSTM (0,2517) sudah dicatat di butir LSTM di atas.
- [ ] **Jangan**: menyandingkan K1 baru dengan pinball@0,9 = 6,56 dari dokumen
      hasil lama (bukan besaran yang sama, T-10); menjalankan
      `resolve_quantile_set()` dengan cakupan biaya di atas ambang tanpa critical
      ratio (sengaja melempar `NotImplementedError`, T-8).
- [x] **Keputusan terbuka sebelum run**: memperkecil `QUANTILE_SET` dari 19 ke 9
      titik menghemat ~38 jam (XGBoost ×7,0 alih-alih ×15,2), tetapi mengubah
      metodologi — kerapatan grid adalah alasan rata-rata pinball mendekati CRPS.
      **Diputuskan secara implisit dengan berjalannya run**: ketiga model selesai
      di 19 titik, tidak dipotong.

### 0d — Menulis ulang dokumen hasil ✅ dokumen selesai (2026-08-29) · ✅ prasyarat wajib terpenuhi (2026-08-30) · 🔒 pembekuan resmi menunggu persetujuan pemilik proyek

Dokumen hasil ditulis **per model begitu model itu selesai**, bukan ditahan
sampai ketiganya rampung (keputusan pemilik proyek 2026-08-25). Yang tetap
ditahan sampai ketiganya selesai adalah bagian 16/18 metodologi, karena keduanya
memeringkat model satu sama lain.

- [x] `docs/hasil-modeling-rf.md` **ditulis ulang dari nol (2026-08-25)** — 9
      bagian, seluruh angkanya dari `rf_search_results.csv`,
      `rf_walk_forward_results.csv`, dan output notebook. Versi lama diarsipkan ke
      `docs/hasil-modeling-rf.single-quantile.bak.md`. Yang baru di dokumen ini
      dibanding kerangka lama: bagian 4.3 perbandingan peringkat lama vs K1, bagian 5.0 gerbang
      G0 terpisah, bagian 5.2 K2 di seluruh 19 titik, dan bagian 7 ongkos sebagai bahan K3.
- [x] `docs/hasil-modeling-xgb.md` **ditulis ulang dari nol (2026-08-29)** —
      9 bagian mengikuti kerangka RF, plus bagian mesin-terpisah (pencarian GPU
      Windows vs walk-forward/fit-final CPU Mac) dan temuan `crossing_rate`
      0,9767.
- [x] `docs/hasil-modeling-lstm.md` **ditulis ulang dari nol (2026-08-29)** —
      9 bagian, termasuk wall clock walk-forward sebenarnya (~8,5 jam, tahap
      tunggal termahal dari ketiga model, bagian 7) sebagai ongkos terukur,
      bukan sebagai kegagalan plafon 8 jam; bagian 5.1b khusus derau antar-seed
      (rentang K1 0,2517); bagian 4.3 mencatat eksplisit kenapa peringkat
      kandidat **tidak bisa** dibandingkan dengan run kuantil-tunggal lama (ruang
      pencarian berubah, bukan cuma kriterianya — beda dari RF/XGBoost).
- [x] bagian 16 (posisi hasil) dan bagian 18 (penerapan tangga keputusan) di
      `metodologi-pemodelan-dan-pemilihan-model.md` **ditulis ulang (2026-08-29)**
      sekarang ketiga model selesai. **Belum menghasilkan pemenang yang
      dibekukan** — K1 seri (RF vs LSTM, gap 1,09% di bawah ambang 2%). K2
      sudah diuji lebih lanjut (`crossing_rate`, sama tanggal): **XGBoost
      tertahan sendiri** (defek sungguhan, ~20-25% baris crossing material,
      butuh rearrangement kuantil), **LSTM bersih** (crossing-nya hampir
      seluruhnya derau numerik, tidak lagi jadi keraguan) — RF vs LSTM
      sekarang bisa dibandingkan langsung di K2, tapi bedanya tipis dan
      belum cukup tegas memutuskan. Status keputusan di bagian 18 secara
      eksplisit **ditahan** sampai dua prasyarat terpenuhi: (1) rearrangement
      kuantil XGBoost + hitung ulang K1/K2, (2) tambah ≥1 seed walk-forward
      LSTM. Kedua item ini ada di bawah sebagai butir 🆕.
- [x] 🆕 **Nyatakan ulang K2 terhadap lantai `share_nol` sebelum ia dipakai
      memutuskan apa pun** — **diterapkan (2026-08-29)** di bagian 18
      `metodologi-pemodelan-dan-pemilihan-model.md` (kolom "kelebihan di atas
      lantai", dihitung `max(τ, share_nol)` untuk ketiga model, bukan τ telanjang).
      Latar belakang (temuan run RF 2026-08-25, bagian 5.2
      `hasil-modeling-rf.md`). Target tak-negatif dan prediksi tak-negatif membuat
      setiap baris ber-target nol otomatis tercakup (`0 ≤ 0`), dan **41,95% baris
      validasi targetnya nol**, jadi tidak ada model tak-negatif apa pun yang bisa
      mencetak `coverage(τ) < 0,4195` — berapa pun τ-nya. Akibatnya tabel pola K2
      yang membaca "simpangan searah di hampir seluruh τ" sebagai alasan kuat untuk
      tersisih akan **menandai setiap model di dataset ini**, termasuk yang
      kalibrasinya sempurna, semata karena ada 17 titik τ di bawah 0,42.

  RF menyimpang +0,3806 di τ=0,05, tetapi simpangan **minimum yang mungkin**
  dicapai siapa pun di sana adalah 0,3695 — RF hanya 0,011 di atas lantai.
  Usulan perbaikan: bandingkan `coverage(τ)` dengan `max(τ, share_nol)`, bukan
  dengan τ telanjang. Setelah dikoreksi begitu, yang tersisa pada RF adalah
  over-coverage **+0,18 di sekitar τ=0,40–0,45** — bias nyata yang tidak
  dijelaskan massa nol, dan itulah yang semestinya dinilai K2.

  **Kerjakan sebelum XGB/LSTM selesai, bukan sesudah**: menulis aturan
  penyisihan setelah melihat angka ketiga model persis jenis keputusan yang
  ingin dihindari bagian 21.

- [x] 🆕 **Uji hipotesis efek ikatan (ties) pada coverage — DIKERJAKAN
      2026-08-29.** Prediksi dari `models/random_forest_q90.joblib` (bundle
      tersimpan, tanpa retrain) pada 345.547 baris validasi (5 fold gabungan),
      coverage dihitung dua cara (`≤` resmi vs `<` tegas).

      **Hasil: efek ikatan besar (`tie_rate` 9% di τ=0,95 sampai 43% di
      τ=0,30) — lebih besar dari over-coverage +0,18 yang jadi pertanyaan —
      tapi hipotesis "ties menjelaskan bias" hanya sebagian benar.** Mengganti
      `≤` dengan `<` tegas **tidak** membersihkan bias jadi nol — ia
      membalikkannya ke arah lain: di τ=0,40, kelebihan-atas-lantai `≤` =
      +0,175, tapi dengan `<` jadi **−0,255** (RF tampak jauh **kurang**
      meramal). Ini karena `<` menghukum setiap prediksi yang **tepat
      sasaran** (lazim terjadi karena 99,55% target bilangan bulat dan forest
      membaca kuantil dari sampel training bertipe sama) sebagai "tidak
      tercakup" — bukan koreksi metrik, melainkan definisi lain yang sama
      bias-nya ke arah berlawanan.

      **Kesimpulan**: over-coverage +0,18 di median tetap dibaca sebagai bias
      nyata (bukan bisa "dihapus" dengan ganti operator `≤`→`<`), tapi harus
      disertai konteks bahwa ~43% baris di sekitar situ memang bernilai
      identik prediksi-aktual — properti kediskretan target, bukan murni
      artefak pembulatan metrik. `≤` (standar definisi kuantil) tetap yang
      paling defensif untuk dipakai sebagai coverage resmi.

- [x] 🆕 **Jelaskan kenapa `crossing_rate` XGBoost (97,7%) dan LSTM (43,4%)
      begitu tinggi dibanding RF (0% struktural) — DIKERJAKAN 2026-08-29,
      jawabannya berbeda arah untuk tiap model.** Diuji dengan menghitung ulang
      `crossing_rate` dari bundle tersimpan (tanpa retrain) memakai toleransi
      jarak minimum (`prediksi(τ_tinggi) < prediksi(τ_rendah) - gap`) — kalau
      rate ambruk dengan toleransi kecil, itu derau numerik; kalau bertahan,
      itu defek sungguhan.

      **XGBoost: sebagian besar defek sungguhan.** Rate turun dari 0,916 (gap
      0) ke 0,479 (gap 0,1) — sudah menyusut, tapi **bertahan ~20–25% di
      toleransi 0,5–1,0 unit**, dengan distribusi ekor sangat panjang (median
      inversi 0,043, mean 1,06, maksimum 139 unit). `multi_strategy`
      `reg:quantileerror` memang tidak punya jaminan monotonicity struktural,
      dan ini terbukti berdampak material, bukan cuma pembulatan. Post-hoc
      rearrangement kuantil (Chernozhukov et al. 2010) **disarankan** sebelum
      dipakai untuk keputusan stok.

      **LSTM: hampir seluruhnya derau numerik, bukan defek.** Rate ambruk dari
      0,459 (gap 0) ke **0,088 (gap 0,01)** ke **0,011 (gap 0,1)** — median
      besar inversi cuma 0,0027 unit, nyaris nol. Kepala keluaran 19-neuron
      LSTM juga tidak dijamin monoton secara arsitektur, tapi secara empiris
      dampaknya dapat diabaikan untuk keputusan stok.

      Rincian lengkap tabel toleransi di `docs/hasil-modeling-{xgb,lstm}.md`
      bagian 5.2. Bagian 16/18 `metodologi-pemodelan-dan-pemilihan-model.md`
      diperbarui mengikuti temuan ini — status K2 tidak lagi simetris antara
      XGBoost (masih tertahan) dan LSTM (crossing tidak lagi jadi pemblokir).
      Sisa pemblokir XGBoost (rearrangement kuantil belum dijalankan) dan
      butir seed LSTM di bawah tetap menahan Fase E.

- [x] 🆕 **Verifikasi jarak K1 RF↔LSTM (0,0310) dengan seed tambahan —
      DIKERJAKAN 2026-08-30. Hasilnya membalik kesimpulan "seri".**
      Walk-forward 5-fold penuh LSTM diulang dengan `random_state=43`
      (`lstm_seed_walkforward.py`, dijalankan mandiri ~9,8 jam di CPU Mac,
      tersimpan di `dataset/model_ready/lstm_walk_forward_results_seed43.csv`
      — **belum masuk git**, seperti artefak model lain).

      | model | K1 (fold 1/2/4) | gap ke RF |
      |---|---:|---:|
      | random_forest | 2,8508 | — |
      | lstm (seed 42, **resmi di semua dokumen**) | 2,8818 | +1,09% (di bawah ambang 2%) |
      | xgboost | 2,9433 | +3,24% |
      | **lstm (seed 43, baru)** | **3,0732** | **+7,80%** (jauh di atas ambang) |

      Cuma ganti seed, K1 LSTM di fold **yang sama persis** (1,2,4) melompat
      **6,6%** — jauh melebihi ambang keputusan 2% yang dipakai bagian 18 K1.
      Dengan seed 43, LSTM bahkan **lebih buruk dari XGBoost** (3,0732 >
      2,9433), bukan lagi model tengah. Rata-rata dua seed (2,9775) juga tetap
      di atas ambang 2% terhadap RF. **Kesimpulan "K1 seri, RF vs LSTM" di
      bagian 18 tidak lagi bisa dipertahankan** sebagaimana tertulis — direvisi
      di bagian 16.5/18 `metodologi-pemodelan-dan-pemilihan-model.md`
      (2026-08-30). Ini juga bukti **langsung** (bukan proksi fold 3&5 lagi)
      bahwa peringatan bagian 17 soal ambang 2% yang mungkin terlalu longgar
      terhadap derau seed LSTM adalah benar.

      **Catatan proses**: percobaan pertama (background task via harness) mati
      setelah ~70 menit tanpa jejak error, kemungkinan proses ikut terhenti
      saat siklus sesi berubah — bukan bug skrip. Percobaan kedua
      (`nohup`+`disown`+stdin `/dev/null`, di luar pelacakan task harness)
      berhasil sampai selesai.

- [x] 🆕 **Rearrangement kuantil XGBoost — DIKERJAKAN 2026-08-30, prasyarat
      wajib terakhir terpenuhi.** `xgb_walk_forward_results.csv` cuma
      menyimpan skor teragregasi, bukan prediksi mentah per baris, jadi
      rearrangement yang jujur (leakage-safe, sebanding dengan K1 RF/LSTM)
      berarti mem-fit ulang kelima model fold XGBoost dengan hyperparameter
      pemenang yang sama, lalu mengurutkan (sort) 19 prediksi kuantil tiap
      baris naik sebelum dinilai — definisi rearrangement Chernozhukov et al.
      (2010). Skrip: `xgb_rearrangement_walkforward.py` (repo root), ~2,89
      jam di CPU Mac (`nohup`+`disown`, pola yang sama dengan seed LSTM di
      atas), tersimpan di
      `dataset/model_ready/xgb_walk_forward_results_rearranged.csv`
      (belum masuk git, seperti artefak model lain).

      **Cek reproduksibilitas cocok persis dengan run resmi 27 Agu**:
      `best_iteration` per fold identik (375, 242, 255, 297, 390), K1
      sebelum sort di fold 1/2/4 = 2,9433 persis sama dengan angka resmi —
      jadi angka sesudah rearrangement sah dipercaya sebagai perbandingan
      adil, bukan run yang berbeda.

      **Hasil**: `crossing_rate` → **0,0000** (dijamin struktural oleh
      sort), K1 fold bersih membaik **1,08%** (2,9433 → **2,9115**). Tapi
      gap ke RF (2,8508) cuma menyempit dari 3,24% menjadi **2,13%** — masih
      di atas ambang keputusan 2% (bagian 17 metodologi). **XGBoost tetap
      kalah dari RF**, sekarang dengan angka yang sudah adil dibandingkan
      (crossing tidak lagi mengaburkan hasilnya). Rincian lengkap termasuk
      tabel K2: `docs/hasil-modeling-xgb.md` bagian 5.5.

      **Konsekuensi**: prasyarat wajib bagian 18 metodologi terpenuhi. RF
      unggul di semua perbandingan yang ada sekarang (RF < LSTM-seed42 <
      XGBoost-rearranged < LSTM-seed43). Tidak ada lagi prasyarat teknis
      yang menahan pembekuan pemenang — lihat butir 10 di bawah dan Fase E.

- [x] **Periksa bias akhir pekan pada segmen bervolume kecil — DIKERJAKAN
      2026-08-29, hipotesis TIDAK terkonfirmasi.** Prediksi RF dari bundle
      tersimpan (tanpa retrain) pada baris validasi, `pair_median` diambil
      persis dari `outlier_handling.compute_pair_baseline()` (bukan proksi)
      untuk pasangan bermedian ≤10 unit/hari — ambang yang sama dengan
      analisis capping asli — disilangkan dengan `is_weekend`:

      | | n | shortfall/baris | coverage@0,9 |
      |---|---:|---:|---:|
      | hari biasa & bervolume kecil | 151.077 | 0,357 | 0,9400 |
      | **akhir pekan** & bervolume kecil | 60.818 | **0,309** | 0,9406 |

      Kalau capping membuang sinyal permintaan akhir pekan, RF seharusnya
      meramal **lebih rendah** dari kenyataan di akhir pekan (shortfall lebih
      tinggi, coverage lebih rendah). Yang terjadi **sebaliknya, tipis**:
      shortfall per baris di akhir pekan 13% **lebih rendah**, coverage
      sedikit lebih tinggi — arah yang berlawanan dengan hipotesis, meski
      selisihnya kecil dan belum diuji signifikansi statistiknya.

      **Kesimpulan**: `day_of_week`/`is_weekend` di `FEATURE_COLS` tampaknya
      berhasil menyerap pola akhir pekan meski sebagian barisnya di-cap saat
      training — capping tidak terbukti semahal yang diperkirakan untuk
      dimensi ini. Rujukan: `docs/analisis-lonjakan-permintaan.md` bagian 5,
      B-3 — pertanyaan ini boleh dibaca sebagai terjawab (negatif), bukan
      lagi dugaan terbuka.

---

## Fase E — Pembekuan pemenang & test set Desember 2025 ⬜

- [ ] Membekukan usulan bagian 18 dalam sebuah commit (menunggu 0d + persetujuan
      pemilik proyek).
- [ ] Menjalankan protokol bagian 19 — **membuka test set Desember 2025 sekali saja**.
      Test set ini belum pernah dibuka; itulah sebabnya migrasi multi-kuantil tidak
      membuang hasil out-of-sample apa pun.
- [ ] Menulis `docs/hasil-test-desember.md`.
- [ ] Menuliskan batasan wajib di bab batasan (bagian 20): evaluasi hanya berlaku
      untuk 1.920 dari 2.979 pasangan, 1–29 Desember 2025, dan hanya 29% baris uji
      yang merupakan momen keputusan sungguhan (B-4, B-5, B-6).

---

## Fase F — Pekerjaan lanjutan ⬜

- [ ] **Optimasi model yang butuh retraining penuh** (mis. validasi statistik
      `SPIKE_RATIO_THRESHOLD`) — dipindah ke berkas terpisah
      `docs/todolist-optimasi-model.md` (2026-08-29) karena tiap butirnya
      mengubah target/fitur dan berarti mengulang seluruh 0c. Sengaja **tidak**
      dikerjakan sebelum Fase E supaya tidak menggeser target di tengah proses
      pemilihan pemenang yang sedang berjalan.
- [ ] **Alokasi kuantil tersegmentasi** pada model pemenang — kuantil bervariasi
      per segmen (kategori × `demand_segment`), rata-rata tertimbang tetap 0,9
      secara agregat. Bagian 4 spec-nya (perluasan multi-kuantil) sudah diwarisi
      dari butir 0c, jadi tinggal simulasi kalibrasi λ dan seterusnya.
      Spec: `2026-08-22-segmented-quantile-allocation-design.md`.
- [ ] 🔒 **Mengisi `dataset/item_cost_margin.csv`** — kolom biaya/margin masih
      kosong 100% (B-10). Ini prasyarat data untuk alokasi tersegmentasi berbasis
      critical ratio; tanpa itu `resolve_quantile_set()` sengaja gagal keras.
      `shelf_life_rank_by_category.csv` sudah terisi sebagai proksi, tetapi masih
      estimasi umum yang menunggu tinjauan tim SCM.
- [ ] **Dekomposisi harian** (`target_h1`…`target_h4`) untuk ketiga model —
      menjawab "kapan permintaan terkonsentrasi". Sudah direncanakan di
      `2026-08-12-modeling-preprocessing-design.md`, bukan tambahan opsional.
- [ ] **SHAP untuk pemenang saja** — menjawab "kenapa model meyakini ini".
      Hanya pemenang, karena menjalankannya untuk ketiganya berarti membayar ongkos
      penjelasan untuk model yang tidak akan dipakai.

---

## Fase G — Hygiene rutin & item yang menunggu pihak lain 🔄

### G1. Sanity check tiap refresh dataset / sebelum training

> **Data 2026 masuk?** Baca `docs/checklist-refresh-data-2026.md` lebih dulu —
> di sana tercatat apa yang gagal keras, apa yang salah diam-diam (terutama
> `ID_HOLIDAYS` yang tidak dijaga `check_year_coverage`), dan apa yang harus
> di-derive ulang.

- [ ] **Re-run `.venv/bin/python3 -m utils.data_preprocessing.prepare_forecast_data` setiap kali
      ada perubahan di script pipeline mana pun** — parquet gampang jadi stale
      relatif ke kode, dan tidak ada guard otomatis untuk itu.
- [ ] **Verifikasi `EXCLUDED_ITEMS` (`xxx.FGS.00066/67/68/69`) tidak muncul di
      `dataset/model_ready/*.parquet`, dan Santan/Gula Cendol (`xxx.FGS.00070/71`)
      sudah dalam Satuan Porsi (bukan Gr).** **`eda.ipynb` tidak bisa dipakai untuk
      verifikasi ini** — notebook itu membaca `dataset.csv` mentah langsung, jadi
      akan selamanya menampilkan item yang sudah di-exclude apa adanya; itu bukan
      tanda exclude-nya gagal. Cek lewat `normalize_items.load_and_normalize()`
      atau output parquet.
- [ ] Soft-check negative `Kuantitas` (`eda.ipynb` bagian 8) — konfirmasi 2026-08-10
      menyangkut insiden yang sudah terjadi, bukan jaminan sistem sumber tidak akan
      salah input lagi.
- [ ] `verify_category_consistency.py` — gerbang kategori sebelum artefak dipakai
      untuk training.
- [ ] `calendar_features.check_year_coverage` akan raise `ValueError` otomatis
      untuk tahun yang belum ter-cover (saat ini hanya 2024 & 2025).
- [ ] Ingat konteks distribusi saat menafsirkan metrik: `Kuantitas`
      right-skewed & kontinu (bukan count), median 64% zero-demand, top 6,6%
      pasangan = 80% volume.

### G2. Menunggu pemilik data 🔒

- [ ] **`tanggal_buka` Cikarang Pusat** di `outlet_closures.csv` — belum ada
      tanggal buka per 2026-08-17. Begitu buka, isi **dan** perbarui
      `RELOCATION_DATES` secara manual (jangan diturunkan otomatis: `KY056`
      Tigaraksa ada di kedua tabel dengan tanggal yang tidak berhubungan).
- [x] ~~**Spike yang berdiri sendiri: pre-order atau bukan?**~~ — dijawab
      2026-08-24: bercampur (sebagian pesanan, sebagian pelanggan langsung),
      tidak dapat dipisahkan dari data ini. Lihat A3 dan
      `docs/analisis-lonjakan-permintaan.md`.
- [ ] **Biaya/margin per SKU** (`item_cost_margin.csv`) — lihat Fase F.
- [ ] Tinjauan tim SCM atas `shelf_life_rank_by_category.csv`.

### G3. Menunggu data periode baru

- [ ] **5 tanggal relokasi _lower-bound proxy_** (Tambun→Mayor Oking,
      Ciomas→Cikarang Pusat, Bantarjati→Teluk Pucung, Dukuh Zamrud→Bukit Gading
      Balaraja, Condet→Grand Wisata Bekasi) — relokasinya sudah terjadi tetapi
      setelah cakupan data berakhir, jadi `days_since_relocation`-nya benar arahnya
      tapi magnitude-nya under-estimate. Di-derive ulang begitu data baru
      menunjukkan kode lama berhenti / kode baru muncul.
- [ ] **`KY073` Cilebut** (buka 2025-12-19) belum dapat ramalan karena nol hari
      sebelum cutoff — masuk sendiri di refresh berikutnya.
- [ ] **Memperluas `calendar_features.py` ke 2026** sebelum data periode baru
      masuk, atau pipeline gagal keras di `check_year_coverage`.

---

## Prioritas & urutan yang disarankan

1. ~~**Tutup kontradiksi target `capped` vs mentah (A3)**~~ — selesai
   2026-08-24: latih di capped, K1 di mentah.
2. ~~**Putuskan kerapatan `QUANTILE_SET`**~~ — selesai 2026-08-24: tetap 19
   titik, hemat ~38 jam ditolak karena yang dipotong adalah dasar kesimpulan,
   bukan parameter eksperimen.
3. ~~**Jalankan butir 0c**~~ — **SELESAI 2026-08-29.** Random Forest
   (2026-08-25, K1 = 2,8508 fold bersih), XGBoost (2026-08-27, K1 = 2,9433),
   dan LSTM (2026-08-28, K1 = 2,8818) semuanya sudah training penuh
   (pencarian → walk-forward 5 fold → fit final). Pencarian XGB/LSTM jalan di
   GPU Windows sejak 2026-08-26 (bukan CPU Mac seperti rencana semula), tapi
   walk-forward + fit final ketiganya tetap satu mesin CPU Mac, jadi K1/K3
   tetap sebanding. RF masih unggul K1, tapi jarak ke LSTM tipis (0,0310) dan
   `crossing_rate` XGB/LSTM tinggi (97,7%/43,4%) — lihat butir 4 baru di
   bawah sebelum membaca ini sebagai "RF menang, selesai".
4. ~~**Nyatakan ulang K2 terhadap lantai `share_nol`**~~ — **diterapkan
   2026-08-29** di bagian 18 `metodologi-pemodelan-dan-pemilihan-model.md`
   untuk ketiga model sekaligus.
5. ~~**Jepit estimasi LSTM sebelum mengomit 83–98 jam**~~ — **superseded**:
   pencarian pindah ke GPU sebelum estimasi CPU ini jadi relevan (lihat catatan
   mesin-terpisah di 0c).
6. ~~**Tulis `hasil-modeling-{xgb,lstm}.md` dan bagian 16/18**~~ — **selesai
   2026-08-29**, tapi **bukan berarti Fase E siap dibuka**: bagian 18 secara
   eksplisit menyatakan status "keputusan ditahan", bukan pemenang yang
   dibekukan. Dua langkah berikut sekarang **menggantikan** langkah lama
   "sisa 0d → Fase E":
7. ~~**Jelaskan `crossing_rate` XGBoost (97,7%) dan LSTM (43,4%)**~~ —
   **selesai 2026-08-29.** XGBoost: defek sungguhan (~20-25% baris crossing
   material) — tertahan sendiri, butuh rearrangement kuantil. LSTM: hampir
   seluruhnya derau numerik — bersih.
8. ~~**Verifikasi jarak K1 RF↔LSTM dengan ≥1 seed tambahan walk-forward
   penuh LSTM**~~ — **selesai 2026-08-30, hasilnya membalik kesimpulan.**
   Seed 43: K1 LSTM fold 1/2/4 = 3,0732, **kalah 7,80% dari RF** (bukan seri
   1,09% seperti seed 42). "K1 seri, RF vs LSTM" **ditarik** — berat bukti
   sekarang condong ke RF. Rincian: `docs/hasil-modeling-lstm.md` bagian
   5.1b, bagian 16.5/18 metodologi.
9. ~~**Satu-satunya prasyarat wajib tersisa: rearrangement kuantil pada
   prediksi XGBoost + hitung ulang K1/K2 di atasnya**~~ — **selesai
   2026-08-30, hasilnya tidak membalikkan urutan.** `crossing_rate` → 0,
   K1 fold bersih membaik 1,08% (2,9433 → 2,9115), tapi gap ke RF cuma
   menyempit dari 3,24% ke 2,13% — **masih di atas ambang 2%**. XGBoost
   tetap kalah, sekarang dengan angka yang adil dibandingkan. Rincian:
   `docs/hasil-modeling-xgb.md` bagian 5.5, bagian 18 metodologi.
10. **Prasyarat teknis sudah terpenuhi — tinggal menutup tangga di bagian
    18 secara resmi.** RF unggul di semua perbandingan yang ada (RF <
    LSTM-seed42 < XGBoost-rearranged < LSTM-seed43). Yang tersisa **bukan
    lagi teknis**: pemilik proyek meninjau angka dan menyetujui RF sebagai
    pemenang dalam sebuah commit, baru test set Desember dibuka lewat
    protokol bagian 19. Sekali saja.
11. **Fase F** setelah pemenang ditetapkan; alokasi tersegmentasi menunggu data
    biaya/margin, jadi kejar G2 secara paralel sejak sekarang.
12. **Fase G** berjalan terus, tidak menunggu apa pun.

---

_Sumber: `docs/overview.md`, `docs/pipeline-overview.md`,
`docs/metodologi-pemodelan-dan-pemilihan-model.md` bagian 21,
`docs/superpowers/specs/2026-08-22-model-comparison-refactor-migration.md`,
riwayat berkas ini sebagai `todolist-data-preprocessing.md`, dan pembacaan
langsung `utils/_.py` + eksekusi test suite (2026-08-24).\*
