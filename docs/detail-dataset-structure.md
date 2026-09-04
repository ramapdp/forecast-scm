# Detail Struktur Dataset untuk Pelatihan Model — RF, XGBoost, LSTM

| Atribut          | Keterangan                                                                                                                                          |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cakupan dokumen  | Tabel fitur **`model_input.parquet`** yang sama persis dikonsumsi ketiga model, dan bagaimana masing-masing membentuknya menjadi masukan siap-latih |
| Sumber kebenaran | `utils/modelling/modeling_prep.py` (`FEATURE_COLS`, dua adapter) dan `utils/modelling/model_{random_forest,xgboost,lstm}.py`                        |
| Baris × kolom    | 1.502.522 baris × 82 kolom (`model_input.parquet`, diverifikasi 18 Agustus 2026)                                                                    |
| Kolom fitur      | **56** — identik untuk ketiga model (`modeling_prep.FEATURE_COLS`)                                                                                  |
| Tanggal ditulis  | 2026-08-31                                                                                                                                          |

**Hubungan dengan dokumen lain.** Dokumen ini adalah **peta cepat**: satu
tabel berisi ke-56 kolom fitur, dan satu bagian per model yang menjelaskan
transformasi tambahan yang diterapkan sebelum kolom-kolom itu sampai ke
`fit()`. Untuk definisi formal tiap kolom (tipe data lengkap, rumus, alasan
desain, persentase nilai kosong) rujuk `docs/detail-tahap-preprocessing.md`
Bagian 1 Bab 6 — dokumen itu tetap otoritas kamus kolom. Untuk ruang
pencarian, hasil, dan artefak per model, rujuk
`docs/detail-tahap-modeling-{rf,xgb,lstm}.md`. Untuk perbandingan ketiganya,
rujuk `docs/detail-tahap-perbandingan-model.md`.

---

## Daftar Isi

- [Detail Struktur Dataset untuk Pelatihan Model — RF, XGBoost, LSTM](#detail-struktur-dataset-untuk-pelatihan-model--rf-xgboost-lstm)
  - [Daftar Isi](#daftar-isi)
  - [1. Dari `model_input.parquet` ke `fit_predict`](#1-dari-model_inputparquet-ke-fit_predict)
  - [2. Dua target: dilatih vs dinilai](#2-dua-target-dilatih-vs-dinilai)
  - [3. Kamus 56 kolom fitur](#3-kamus-56-kolom-fitur)
    - [3.1 Riwayat permintaan — 15 kolom](#31-riwayat-permintaan--15-kolom)
    - [3.2 Kalender — 20 kolom](#32-kalender--20-kolom)
    - [3.3 Siklus pengiriman — 4 kolom](#33-siklus-pengiriman--4-kolom)
    - [3.4 Outlet — 9 kolom](#34-outlet--9-kolom)
    - [3.5 Item — 1 kolom](#35-item--1-kolom)
    - [3.6 Kategorikal terkode — 7 kolom (`*_idx`)](#36-kategorikal-terkode--7-kolom-_idx)
    - [3.7 Kolom yang sengaja dikeluarkan](#37-kolom-yang-sengaja-dikeluarkan)
  - [4. Dua adapter: tabular vs sequence](#4-dua-adapter-tabular-vs-sequence)
  - [5. Bagaimana tiap model membentuk tabel yang sama](#5-bagaimana-tiap-model-membentuk-tabel-yang-sama)
    - [5.1 Random Forest](#51-random-forest)
    - [5.2 XGBoost](#52-xgboost)
    - [5.3 LSTM](#53-lstm)
  - [6. Kontrak kesetaraan antaradapter](#6-kontrak-kesetaraan-antaradapter)
  - [7. Ringkasan banding](#7-ringkasan-banding)

---

## 1. Dari `model_input.parquet` ke `fit_predict`

```
featured.parquet (68 kolom)
  │  modeling_prep.build_model_input()
  │    ├─ add_event_flag()       → is_event_driven
  │    ├─ classify_pairs()       → demand_segment  (Syntetos-Boylan, dari periode latih)
  │    ├─ assign_folds()         → fold_id         (5 fold walk-forward, Des 2025 dikunci)
  │    ├─ impute_features()      → has_full_history, missing_history_count, was_relocated,
  │    │                           has_baseline, + null history/event/relocation diisi
  │    └─ encode_categoricals()  → 7 kolom `*_idx`, dari category_mapping.json
  ▼
model_input.parquet (82 kolom, 1.502.522 baris)
  │
  ├─ modeling_prep.to_tabular(df, FEATURE_COLS)    ──► X flat (RF, XGBoost)
  └─ modeling_prep.to_sequences(df, FEATURE_COLS)  ──► tensor (lookback=28, LSTM)
        │
        └─ validate_contract() menjamin kedua adapter melihat baris,
           target, dan fold yang identik (Bagian 6)
```

Ketiga model membaca **kolom yang sama** — `modeling_prep.FEATURE_COLS`, satu
daftar tunggal, ditulis satu kali supaya perbandingan tidak bisa berdiri di
atas ketiga model diam-diam memilih kolom berbeda (lihat
`docs/detail-tahap-perbandingan-model.md` Bagian 1.2). Perbedaan tiga model
dimulai **setelah** titik ini: bagaimana kolom kategorikal dikodekan,
bagaimana bentuk tensornya, dan skala apa yang diterapkan (Bagian 5).

Dua baris pertama pipa (`drop_warmup_rows`, lalu baris tanpa target dibuang)
dijalankan identik oleh kedua adapter, sehingga baris yang bisa diprediksi
oleh RF/XGBoost persis sama dengan baris yang punya jendela 28-hari penuh
bagi LSTM.

## 2. Dua target: dilatih vs dinilai

| Konstanta          | Nilai                                | Dipakai untuk                                                                       |
| ------------------ | ------------------------------------ | ----------------------------------------------------------------------------------- |
| `TRAIN_TARGET_COL` | `target_lead_time_cumulative_capped` | Semua yang di dalam `fit_predict`: fit, early stopping, target scaling, `log1p`     |
| `EVAL_TARGET_COL`  | `target_lead_time_cumulative`        | Semua yang di luar `fit_predict`: K1, K2, lantai naif, setiap angka yang dilaporkan |

Ketiga model **dilatih** pada target yang lonjakan ekstremnya sudah
dipangkas (proksi terdekat bagi komponen non-pesanan, B-3), tapi **dinilai**
pada target mentah — permintaan yang benar-benar dihadapi outlet. Ini bukan
inkonsistensi: kriteria seleksi model sengaja tidak dihitung di atas deret
yang sama yang dipakai melatihnya, supaya kriteria itu tidak bisa
"diperbaiki" hanya dengan memangkas lebih agresif. Alasan lengkap ada di
komentar `modeling_prep.py` baris 29–44 dan
`docs/detail-tahap-preprocessing.md` Subbab 3.3–3.4.

`log_target` (opsional per model, Bagian 5) menerapkan `np.log1p` di atas
`TRAIN_TARGET_COL` sebelum fit; prediksi dikembalikan lewat
`modeling_prep.inverse_log_target()` (`np.expm1`), yang eksak untuk model
kuantil karena kuantil ekuivarian terhadap transformasi monoton.

## 3. Kamus 56 kolom fitur

`modeling_prep.FEATURE_COLS` — urutan persis seperti didefinisikan di kode,
dikelompokkan per keluarga. Definisi formal tiap kolom ada di
`docs/detail-tahap-preprocessing.md` Bab 6 (nomor subbab dicantumkan di
kolom terakhir).

### 3.1 Riwayat permintaan — 15 kolom

| Kolom                                 | Tipe       | Ringkasan                                                 | Bab 6 |
| ------------------------------------- | ---------- | --------------------------------------------------------- | ----- |
| `lag_1`, `lag_2`, `lag_3`             | pecahan    | Kuantitas terpangkas 1–3 hari sebelumnya                  | 6.7   |
| `lag_7`, `lag_14`, `lag_21`, `lag_28` | pecahan    | Kuantitas terpangkas pada hari sama, 1–4 pekan sebelumnya | 6.7   |
| `roll_mean_7`, `roll_std_7`           | pecahan    | Rerata & simpangan baku jendela 7 hari, berakhir $t{-}1$  | 6.7   |
| `roll_mean_14`, `roll_std_14`         | pecahan    | Idem, jendela 14 hari                                     | 6.7   |
| `roll_mean_28`, `roll_std_28`         | pecahan    | Idem, jendela 28 hari                                     | 6.7   |
| `has_full_history`                    | boolean    | Benar bila tak satu pun dari 13 kolom di atas kosong      | 6.9   |
| `missing_history_count`               | int (0–13) | Cacah kolom riwayat yang kosong pada baris ini            | 6.9   |

Seluruhnya dihitung dari `Kuantitas_capped`, dikelompokkan per (pasangan,
`segment_id`) sehingga tak satu pun menjangkau melintasi periode gerai tutup.
`lag_28` adalah yang terpanjang — panjangnya inilah yang menetapkan
`LOOKBACK = 28` dipakai `drop_warmup_rows()` dan jendela sequence LSTM.

### 3.2 Kalender — 20 kolom

| Kolom                                                                               | Tipe            | Ringkasan                                                           | Bab 6 |
| ----------------------------------------------------------------------------------- | --------------- | ------------------------------------------------------------------- | ----- |
| `day_of_week`, `day_of_month`, `month`                                              | int             | Posisi kalender deterministik                                       | 6.3   |
| `is_weekend`                                                                        | boolean         | Turunan `day_of_week`, disediakan tersendiri untuk model pohon      | 6.3   |
| `is_national_holiday`                                                               | boolean         | Libur nasional Indonesia (pustaka `holidays`)                       | 6.3   |
| `is_ramadan`, `days_into_ramadan`, `days_until_ramadan`                             | boolean/pecahan | Penanda & jarak hari Ramadan (satu-satunya tanpa batas jendela ±14) | 6.3   |
| `is_eid_al_fitr`, `days_since_eid_al_fitr`, `days_until_eid_al_fitr`                | boolean/pecahan | Idem, Idulfitri, jendela ±14 hari                                   | 6.3   |
| `is_eid_al_adha`, `days_since_eid_al_adha`, `days_until_eid_al_adha`                | boolean/pecahan | Idem, Iduladha                                                      | 6.3   |
| `is_independence_day`, `days_since_independence_day`, `days_until_independence_day` | boolean/pecahan | Idem, 17 Agustus                                                    | 6.3   |
| `is_new_year`, `days_since_new_year`, `days_until_new_year`                         | boolean/pecahan | Idem, 1 Januari                                                     | 6.3   |

Bebas dari risiko kebocoran — dibangkitkan dari `Tanggal` semata, sehingga
selalu bisa dihitung untuk tanggal mana pun di masa depan. Kolom
`days_*` peristiwa (kecuali `days_until_ramadan`) bernilai kosong di luar
jendela ±14/±15 hari; imputasinya dijelaskan di Bagian 3.5.

### 3.3 Siklus pengiriman — 4 kolom

| Kolom                        | Tipe          | Ringkasan                                                                                             | Bab 6 |
| ---------------------------- | ------------- | ----------------------------------------------------------------------------------------------------- | ----- |
| `kawasan`                    | int (1/2)     | Kawasan logistik cabang → jadwal kirim                                                                | 6.5   |
| `lead_time_days`             | int (1–4)     | Jarak hari maju-ketat menuju kiriman berikutnya; menentukan panjang jendela target                    | 6.5   |
| `is_delivery_day`            | boolean       | Benar bila tanggal baris = hari kirim cabang sendiri (28,57% baris)                                   | 6.5   |
| `target_window_weekend_days` | pecahan (0–2) | Cacah Sabtu/Minggu di dalam jendela target — komposisi hari lebih menentukan daripada panjang jendela | 6.5   |

### 3.4 Outlet — 9 kolom

| Kolom                                      | Tipe        | Ringkasan                                                                            | Bab 6 |
| ------------------------------------------ | ----------- | ------------------------------------------------------------------------------------ | ----- |
| `has_shopee`, `has_gofood`, `has_grabfood` | boolean     | Ketersediaan tiap kanal daring, dipisah karena basis pelanggan berbeda               | 6.6   |
| `can_order_online`                         | boolean     | Disjungsi ketiganya                                                                  | 6.6   |
| `branch_avg_daily_qty`                     | pecahan     | Rerata kuantitas terpangkas harian cabang, dibekukan dari periode latih              | 6.8   |
| `branch_demand_cv`                         | pecahan     | Koefisien variasi permintaan harian cabang, dibekukan dari periode latih             | 6.8   |
| `branch_age_days`                          | int (0–730) | Umur cabang di data, aman inheren (hanya membaca masa lalu cabang sendiri)           | 6.8   |
| `days_since_relocation`                    | pecahan     | Jarak hari ke tanggal relokasi; kosong pada 84,3% baris (cabang tak pernah relokasi) | 6.6   |
| `was_relocated`                            | boolean     | Penanda apakah `days_since_relocation` semula terisi (sebelum diimputasi 0,0)        | 6.9   |

### 3.5 Item — 1 kolom

| Kolom             | Tipe    | Ringkasan                                                                                                    | Bab 6 |
| ----------------- | ------- | ------------------------------------------------------------------------------------------------------------ | ----- |
| `is_event_driven` | boolean | Permintaan SKU digerakkan pemesanan acara (5 SKU, 4,37% baris) — tidak dapat diprediksi lag/rolling mana pun | 6.9   |

### 3.6 Kategorikal terkode — 7 kolom (`*_idx`)

| Kolom                    | Kardinalitas                                        | Sumber                                               |
| ------------------------ | --------------------------------------------------- | ---------------------------------------------------- |
| `Kode Barang_idx`        | 70 nilai + `<UNKNOWN>`=0                            | `Kode Barang`                                        |
| `Nama Cabang_idx`        | 59 nilai + `<UNKNOWN>`=0                            | `Nama Cabang`                                        |
| `Kategori Barang_idx`    | 8 nilai + `<UNKNOWN>`=0                             | `Kategori Barang`                                    |
| `kota_idx`               | 16 nilai + `<UNKNOWN>`=0                            | `kota`                                               |
| `hari_pengiriman_idx`    | —                                                   | `hari_pengiriman`                                    |
| `branch_volume_tier_idx` | 4 nilai (`small`…`flagship`)                        | `branch_volume_tier`, kuartil `branch_avg_daily_qty` |
| `demand_segment_idx`     | 4 nilai (`smooth`/`erratic`/`intermittent`/`lumpy`) | `demand_segment`, klasifikasi Syntetos-Boylan        |

Ketujuh kolom sumbernya (untai teks) sendiri **bukan** anggota
`FEATURE_COLS` — hanya bentuk `_idx`-nya. Pemetaan dibentuk sekali dari
periode latih (`build_category_mapping()`), dipersistensi ke
`dataset/model_ready/category_mapping.json`, dan **diperluas** (bukan
ditulis ulang) pada tiap refresh data supaya indeks yang sudah dipakai model
tersimpan tidak pernah berubah makna (`modeling_prep.py` baris 234–271).
Nilai yang tak dikenal (cabang baru, dsb.) jatuh ke `UNKNOWN_INDEX = 0`.

**Total: 15 + 20 + 4 + 9 + 1 + 7 = 56 kolom** — persis
`len(modeling_prep.FEATURE_COLS)`.

### 3.7 Kolom yang sengaja dikeluarkan

Dua kolom yang tampak seperti fitur wajar **tidak** masuk `FEATURE_COLS`,
sengaja: `baseline_ratio` dan `is_spike` — keduanya turunan `Kuantitas` hari
$t$ itu sendiri, sedangkan setiap lag/rolling berhenti di $t{-}1$.
Menyertakannya membiarkan model "mengintip" jawaban hari $H$. Biaya
terukur dari pembuangan ini kecil: baseline `roll_mean_7` bergeser dari MAE
12,99 ke 13,19 saat hari $H$ diizinkan masuk (`modeling_prep.py` baris
386–393).

## 4. Dua adapter: tabular vs sequence

|                       | `to_tabular()` (RF, XGBoost)                                 | `to_sequences()` (LSTM)                                               |
| --------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------- |
| Bentuk `X`            | `(n_baris, 56)` — satu baris = satu prediksi                 | `(n_baris, 28, 56)` — satu jendela 28-hari berakhir di baris prediksi |
| NaN pada fitur        | Dibiarkan apa adanya (RF/XGBoost boleh menanganinya sendiri) | Harus nol — `impute_features()` wajib dijalankan lebih dulu           |
| Baris yang dihasilkan | `drop_warmup_rows()` lalu buang baris tanpa target           | Sama persis (dijamin `validate_contract()`)                           |
| `keys` / `fold_id`    | Ikut dikembalikan, untuk penilaian per fold                  | Sama                                                                  |

`to_sequences()` membentuk jendela **per (pasangan, segmen)**, tidak pernah
menjembatani batas segmen — jendela pertama yang mungkin dibentuk untuk
segmen baru pascarelokasi/buka-kembali adalah 28 hari setelah segmen itu
mulai, persis titik yang sama dengan `drop_warmup_rows()` pada adapter
tabular.

## 5. Bagaimana tiap model membentuk tabel yang sama

Titik awal ketiganya identik: `X` dari `to_tabular()` (RF, XGBoost) atau
`to_sequences()` (LSTM), atas 56 kolom yang sama. Perbedaan mulai dari sini.

### 5.1 Random Forest

Ketujuh kolom `*_idx` dikonsumsi **langsung sebagai integer ordinal** —
`quantile-forest` membaca indeks kategori seperti nilai numerik biasa, yang
valid bagi model berbasis pohon karena pemisahan pohon tidak mengasumsikan
urutan bermakna.

Hyperparameter `one_hot` (dicari `[False, True]`) adalah satu-satunya
saklar tambahan: bila `True`, `expand_one_hot()` (`pd.get_dummies` atas
ketujuh kolom `_idx`) dijalankan sebelum fit, kolom validasi disejajarkan ke
kolom hasil training via `reindex`. **Konfigurasi final memenangkan
`one_hot=False`** (kandidat 1: `max_depth=20`, `max_features=1.0`,
`max_samples_leaf=1`, `min_samples_leaf=20`, `log_target=False`) — 56
kolom, tanpa ekspansi. `n_estimators` dinaikkan dari 200 (pencarian) ke 400
(`FINAL_N_ESTIMATORS`) untuk fit final.

Tidak ada penskalaan numerik — pohon tidak butuh fitur berskala sama.
Kuantil dibaca langsung dari daun (`quantile-forest`), bukan dioptimalkan
sebagai objektif.

### 5.2 XGBoost

Tiga mode `encoding` dicari (`model_common`/`model_xgboost.encode()`):

| Mode      | Perlakuan `*_idx`                                                                                                                                                                                    |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ordinal` | Dipakai apa adanya sebagai integer                                                                                                                                                                   |
| `native`  | Dikonversi ke `pandas.CategoricalDtype` per kolom (level dibekukan dari training); level tak dikenal di validasi/produksi menjadi NaN, dibaca `xgboost` sebagai _missing_, bukan meminjam level lain |
| `one_hot` | `expand_one_hot()`, sama seperti RF                                                                                                                                                                  |

**Konfigurasi final memenangkan `encoding="native"`** (kandidat 17:
`max_depth=10`, `learning_rate=0,05`, `min_child_weight=1`,
`subsample=0,7`, `colsample_bytree=0,5`, `reg_lambda=10,0`,
`log_target=False`) — 56 kolom, tujuh di antaranya bertipe kategorik native.
Objektif `reg:quantileerror` dengan `multi_strategy` mengoptimalkan
langsung kriteria K1 (bukan dibaca dari struktur pohon seperti RF). Ronde
boosting ditentukan _early stopping_ pada ekor 30-hari terakhir jendela
training tiap fold (`split_early_stopping()`, sama fungsi dipakai LSTM),
lalu direfit penuh — protokol dua-fit yang menjamin fit final tetap dilatih
atas populasi baris yang sama dilihat forest.

### 5.3 LSTM

Ke-56 kolom terpecah menjadi dua jalur berbeda di dalam `QuantileLSTM`:

- **49 kanal dinamis** (56 dikurangi 7 `*_idx`) mengalir melalui LSTM
  sebagai jendela 28-hari penuh — `sequence_windows.gather()` memisahkannya
  dari kolom `_idx` (`[col for col in feature_cols if not col.endswith("_idx")]`).
- **7 kolom `*_idx`** dibaca **hanya pada baris prediksi**, tidak diulang di
  sepanjang jendela — `Kategori Barang_idx` bisa berubah di dalam satu
  jendela 28-hari yang menjembatani segmen berbeda, sehingga "kategori
  segmen" bukan hal yang didefinisikan untuk diulang. Ketujuhnya lewat
  lapisan `nn.Embedding` masing-masing, `num_embeddings` diambil dari
  `max(mapping[kolom].values()) + 1` di `category_mapping.json` (bukan dari
  nilai yang kebetulan muncul di training fold), `embedding_dim = min(16,
(num_embeddings+1)//2)`. Output LSTM (hidden state langkah terakhir) dan
  ketujuh embedding digabung sebelum masuk kepala 19-titik kuantil.

Ke-49 kanal dinamis **distandardisasi** (`fit_scaler`/`scale_values`,
z-score) dengan scaler yang **difit khusus di training fold tersebut** —
menerapkannya ke seluruh baris (termasuk baris konteks di luar training)
aman, yang bocor hanya bila proses _fit_-nya melampaui batas fold. Kolom
`_idx` tidak diskalakan — mereka masuk sebagai indeks integer ke embedding,
bukan nilai kontinu.

**Konfigurasi final** (`lstm_best_params.json`): `hidden_size=256`,
`num_layers=2`, `dropout=0,0`, `learning_rate=0,0003`, `batch_size=2048`,
`log_target=True`, `random_state=42`. `log_target=True` di sini berarti
`TRAIN_TARGET_COL` di-`log1p` sebelum pinball loss dihitung dan prediksi
dikembalikan lewat `inverse_log_target()`. Objektif (pinball loss 19-titik)
= kriteria seleksi K1, sama seperti XGBoost.

## 6. Kontrak kesetaraan antaradapter

`modeling_prep.validate_contract(tabular, sequences)` menjalankan empat
pemeriksaan sebelum kedua adapter dipercaya untuk perbandingan:

1. Tidak ada NaN pada `X` kedua adapter (bila `require_finite=True`).
2. Jumlah baris sama.
3. Himpunan kunci `(Kode Barang, Nama Cabang, segment_id, Tanggal)` identik.
4. Nilai target dan `fold_id` identik lintas adapter (`np.allclose`).

Tanpa ini, "LSTM lebih baik 8%" bisa jadi sesungguhnya berarti "LSTM
dinilai atas 5% baris yang berbeda" — kontrak inilah yang membuat klaim
perbandingan di `docs/detail-tahap-perbandingan-model.md` sah diajukan.

## 7. Ringkasan banding

|                       | Random Forest                     | XGBoost                            | LSTM                                                    |
| --------------------- | --------------------------------- | ---------------------------------- | ------------------------------------------------------- |
| Bentuk masukan        | Tabel flat `(n, 56)`              | Tabel flat `(n, 56)`               | Tensor `(n, 28, 56)` dipecah 49 dinamis + 7 indeks      |
| Kategorikal (`*_idx`) | Ordinal integer (`one_hot=False`) | `CategoricalDtype` native          | `nn.Embedding` per kolom, dibaca di baris prediksi saja |
| Penskalaan numerik    | Tidak ada                         | Tidak ada                          | Z-score per fold, kanal dinamis saja                    |
| Target dilatih        | `..._capped`, `log_target=False`  | `..._capped`, `log_target=False`   | `..._capped`, `log_target=True`                         |
| Mekanisme kuantil     | Dibaca dari daun pohon            | `reg:quantileerror` (dioptimalkan) | Pinball loss langsung (dioptimalkan)                    |
| Kolom fitur efektif   | 56                                | 56                                 | 56 (49 + 7)                                             |

Detail konstruksi lengkap (ruang pencarian, kurva hasil, artefak) ada di
`docs/detail-tahap-modeling-{rf,xgb,lstm}.md`; perbandingan skor lintas
ketiganya ada di `docs/detail-tahap-perbandingan-model.md`.
