# Detail Tahap Pemodelan — XGBoost Kuantil

| Atribut | Keterangan |
|---|---|
| Implementasi | `xgboost==2.1.4`, `XGBRegressor(objective="reg:quantileerror", multi_strategy="multi_output_tree", tree_method="hist")` |
| Modul produksi | `utils/modelling/model_xgboost.py` |
| Modul bersama yang dipakai | `utils/modelling/model_common.py`, `walk_forward.py`, `purging.py`, `evaluation.py`, `modeling_prep.py` |
| Notebook | `notebook/modeling_xgb.ipynb` |
| Uji unit | `test/test_model_xgboost.py`, `test/test_model_common.py` |
| Ruang pencarian | 2.592 kombinasi hyperparameter |
| Kandidat ditarik | 30 |
| Run hasil terakhir | 27 Agustus 2026 — run pertama di bawah kriteria multi-kuantil K1; rearrangement kuantil 30 Agustus 2026 |
| Status | Kalah dari Random Forest di K1 (gap 2,13% setelah rearrangement, di atas ambang keputusan 2%) — lihat `docs/detail-tahap-perbandingan-model.md` |

**Hubungan dengan dokumen lain.** Sama seperti dokumen Random Forest,
dokumen ini fokus pada satu model. Prapemrosesan data ada di
`docs/detail-tahap-preprocessing.md`; kontrak evaluasi bersama, metrik, dan
**perbandingan** ketiga model ada di `docs/detail-tahap-perbandingan-model.md`;
angka yang bisa berubah setiap run diulang ada di `docs/hasil-modeling-xgb.md`,
rujukan utama Bagian 1.7–1.11 di bawah.

Desain aslinya ada di
`docs/superpowers/specs/2026-08-19-xgboost-modeling-design.md` dan
`docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`.

---

# Bagian 1 — Akademis (Bahan Laporan)

## 1.1 Pendahuluan dan Posisi dalam Alur Penelitian

XGBoost kuantil adalah kandidat kedua di antara tiga model yang
dibandingkan (bersama Random Forest kuantil dan LSTM kuantil). Ia
meramalkan target dan memakai fitur yang identik dengan kedua model
lainnya, dinilai lewat mesin evaluasi bersama yang sama
(`utils/modelling/walk_forward.py`).

Berbeda dari Random Forest, yang membaca kuantil sebagai persentil dari
distribusi empiris di daun, XGBoost mengoptimalkan **langsung** fungsi
objektif `reg:quantileerror` — pinball loss — lewat *gradient boosting*.
Ini menempatkannya di titik tengah spektrum kompleksitas: lebih ekspresif
daripada forest untuk menangkap interaksi fitur non-linear yang tajam,
tapi tanpa jaminan struktural apa pun soal bagaimana titik-titik kuantil
yang berbeda saling berhubungan satu sama lain (Bagian 1.2 dan 1.9).

## 1.2 Objektif yang Identik dengan Kriteria Seleksi

`reg:quantileerror` mengoptimalkan pinball loss yang **sama persis** dengan
kriteria K1 yang dipakai memilih model (rata-rata pinball lintas 19 titik
kuantil). Ini properti yang tidak dimiliki model galat-kuadrat yang dimintai
kuantil tinggi setelahnya: apa yang dioptimalkan saat training dan apa yang
dinilai saat evaluasi adalah fungsi yang sama, bukan dua hal yang kebetulan
berkorelasi.

`quantile_alpha` diberi **seluruh grid 19 titik sekaligus** (bukan satu
skalar), sehingga XGBoost membangun struktur boosting yang menjawab
seluruh grid dalam satu proses fit — mekanisme `multi_strategy="multi_output_tree"`
milik XGBoost sendiri, bukan sembilan belas model terpisah yang dilatih
berurutan.

## 1.3 Protokol Dua Fit

Jumlah ronde boosting adalah keputusan regularisasi tersendiri — boosting
yang berjalan terlalu lama overfit, dan tempat paling wajar untuk memilih
jumlah ronde (fold validasi) justru tempat yang bocor bila dipakai langsung.
Maka XGBoost di proyek ini memakai **dua fit** per fold:

1. **Early stopping** berjalan pada ekor 30 hari terakhir jendela training
   (`ES_TAIL_DAYS = 30`), dengan purge yang sama seperti di batas fold —
   baris yang targetnya menjangkau ke dalam ekor early-stopping dibuang
   dari himpunan fit (`split_early_stopping()`, dipakai bersama LSTM lewat
   `model_common`). `EARLY_STOPPING_ROUNDS = 50`, `MAX_ROUNDS = 2000`.
2. Model probe itu **dibuang**, lalu difit ulang pada **seluruh baris
   training** pada jumlah ronde yang ditemukan langkah 1.

Fit kedua inilah yang membuat XGBoost akhirnya dilatih pada populasi baris
yang sama persis dengan yang dilihat Random Forest — tanpa protokol ini,
XGBoost akan kehilangan 30 hari terakhir setiap jendela training hanya
untuk keperluan internalnya sendiri.

`n_estimators` sengaja tidak ada di ruang pencarian: early stopping sudah
memutuskannya per kandidat per fold, jadi mencarinya menghabiskan anggaran
untuk pertanyaan yang sudah punya mekanisme.

## 1.4 Kontrak Evaluasi Bersama

Sama seperti Random Forest, XGBoost tidak memiliki logika fold atau metrik
sendiri — seluruhnya dimiliki `walk_forward.py`, yang memperlakukan model
sebagai satu fungsi `fit_predict(train_df, valid_df) -> np.ndarray`. Detail
penuh kontrak ada di `docs/detail-tahap-perbandingan-model.md` Bagian 1.2.

## 1.5 Ongkos Multi-Kuantil: Mahal, dan Alasan Kepindahan ke GPU

Boosting membangun **satu pohon per titik kuantil per ronde**
(`multi_strategy` bawaan XGBoost untuk `reg:quantileerror`), sehingga 19
titik kuantil bukan ongkos tambahan tipis seperti pada Random Forest.
Benchmark satu putaran dua-fit di fold 5 dengan `DEFAULT_PARAMS`, di CPU
Mac: **265,2 menit (~4,4 jam)** — pengganda terukur **×15,2** dibanding 2,4
menit di run kuantil-tunggal lama (bandingkan RF, ×1,47, `docs/detail-tahap-modeling-rf.md`
Bagian 1.6).

Inilah satu-satunya alasan tahap **pencarian** (bukan walk-forward atau fit
final) dipindahkan ke GPU Windows (RTX 4060 Ti 8 GB, `device=cuda`) sejak
26 Agustus 2026 (`docs/runbook-pencarian-gpu-windows.md`) — keputusan yang
menggantikan rencana "seluruh Fase 3 di CPU Mac lokal" tertanggal sehari
sebelumnya. Walk-forward final dan fit final tetap di CPU Mac, satu mesin
yang sama dengan Random Forest dan LSTM, supaya ongkos (K3) tetap
sebanding lintas ketiga model. Paritas GPU↔CPU untuk peringkat kandidat
divalidasi lebih dulu (selisih K1 0,124%, jauh di bawah ambang 2%), jadi
pemenang yang dipilih di GPU tetap dipercaya.

`best_iteration` benchmark `DEFAULT_PARAMS` mencapai **1.999 dari plafon
2.000** — nyaris tersentuh, sinyal bahwa parameter default (bukan pemenang
pencarian) mendekati batas anggaran boosting yang disediakan.

## 1.6 Ruang Pencarian Hyperparameter

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

2.592 kombinasi, 30 kandidat ditarik acak, dinilai di **fold 3 (September)
dan fold 5 (November)** — protokol identik dengan Random Forest, tapi di
GPU Windows. Semua 30 kandidat berhasil dinilai. Parameter terpilih:

```json
{
  "colsample_bytree": 0.5,
  "encoding": "native",
  "learning_rate": 0.05,
  "log_target": false,
  "max_depth": 10,
  "min_child_weight": 1,
  "random_state": 42,
  "reg_lambda": 10.0,
  "subsample": 0.7
}
```

Empat bacaan dari sebarannya (tabel lengkap di `docs/hasil-modeling-xgb.md`
Bagian 4.1–4.2):

1. **`max_depth` adalah penentu utama, dan monoton** — rata-rata K1 per
   kedalaman: 4 → 3,0344, 6 → 2,9483, 8 → 2,9153, 10 → 2,8945. Seluruh
   sembilan kandidat `max_depth=4` ada di separuh bawah tabel peringkat.
   Berbeda dari Random Forest, di mana kedalaman pohon sudah jenuh
   manfaatnya di 56 fitur — di XGBoost, yang membangun satu pohon per
   titik kuantil, kedalaman lebih dalam masih terus membantu pada rentang
   yang diuji.
2. **`log_target=False` konsisten lebih baik**, sejalan dengan Random
   Forest.
3. **Encoding kategorikal berpengaruh kecil** dibanding `max_depth`.
   Pemenangnya `native` (XGBoost menangani kategori langsung tanpa
   one-hot atau ordinal), tapi selisihnya ke `one_hot` dan `ordinal` jauh
   lebih sempit daripada selisih antar `max_depth`.
4. **Peringkat kandidat jauh kurang stabil terhadap kriteria K1**
   dibanding Random Forest: Spearman ρ = 0,73 (RF: 0,975), Kendall τ =
   0,54 (RF: 0,895) terhadap peringkat pinball@0,9 lama, walau seed dan
   ruang pencariannya identik. Ini mengonfirmasi hipotesis bahwa
   `multi_strategy` — satu pohon per titik kuantil per ronde — memang
   berinteraksi dengan jumlah titik kuantil yang diminta, tidak seperti
   Random Forest yang membaca seluruh titik dari daun yang sudah jadi.
   Konsekuensinya: peringkat kandidat run kuantil-tunggal lama **tidak
   bisa dipakai sebagai proksi** peringkat K1 untuk XGBoost.

## 1.7 Gerbang Kelayakan (G0) dan Hasil Walk-Forward Lima Fold

**G0.** XGBoost menang di kelima fold dengan margin 42%–45% — lolos.

**K1.** Pada potongan fold 1/2/4:

| model | K1 (fold 1/2/4) | K1 (5 fold) |
|---|---:|---:|
| **xgboost** | **2,9433** | **2,9197** |
| `naive_roll_mean_7` | 4,8603 | 4,8231 |
| `naive_lag_1` | 8,1612 | 8,1755 |
| `naive_zero` | 14,8102 | 14,7469 |

**39% lebih baik** daripada baseline terbaik, tapi **kalah** dari Random
Forest (K1 2,8508) sebesar 0,0925 (3,2%, sebelum rearrangement — Bagian
1.9). XGBoost **menang MAE@0,9** melawan RF (13,467 lawan 15,055) —
prediksinya lebih dekat ke aktual di titik tengah, meski rata-rata pinball
19-titiknya sedikit lebih longgar.

K1 bergerak 2,737–3,057 antar fold (rentang 12%), stabilitas yang sebanding
dengan RF (13%). Berbeda dari RF (yang justru sedikit lebih baik di fold
bersih dibanding fold seleksi), XGBoost tampak **sedikit lebih lemah** di
fold bersih (2,9433) dibanding fold seleksi 3&5 (~2,88) — selisih kecil
(~0,06) yang jauh lebih kecil daripada jarak ke RF, jadi bukan tanda
overfitting seleksi yang besar, tapi berlawanan arah dari RF dan layak
dicatat.

## 1.8 Kalibrasi (K2) dan `crossing_rate`

Di τ=0,9, coverage XGBoost 0,902 terhadap target 0,90 (RF: 0,928) — secara
mentah **lebih dekat** ke target daripada RF, dengan fill rate 0,951.
Bacaan lantai `share_nol` sama seperti Random Forest
(`docs/detail-tahap-modeling-rf.md` Bagian 1.9): over-coverage besar di τ
rendah adalah efek 41,95% baris bertarget nol, bukan murni kalibrasi.

**Temuan yang membedakan XGBoost secara mendasar:** `crossing_rate =
0,9767` (97,7% baris) di walk-forward final — jauh di atas Random Forest
(0% struktural). Berbeda dari LSTM (Bagian 1.8, `docs/detail-tahap-modeling-lstm.md`),
yang crossing-nya hampir seluruhnya derau numerik, XGBoost punya **inti
keras defek sungguhan**: dihitung ulang dengan toleransi jarak minimum,
rate ambruk dari 0,916 (toleransi 0) ke 0,479 (toleransi 0,1) — sebagian
besar memang derau kecil, median inversi 0,043 unit — tapi **~20–25% baris
tetap crossing bahkan di toleransi 0,5–1,0 unit**, dengan ekor distribusi
sampai 139 unit. Ini bukan artefak pembulatan: `multi_strategy` XGBoost
memang tidak punya jaminan monotonicity struktural antar titik kuantil
(berbeda dari Random Forest, Bagian 1.2, dan berlawanan arah dari LSTM,
yang hasilnya ternyata hampir seluruhnya derau).

**Konsekuensi metodologis: rearrangement kuantil.** Karena inti keras
defek itu nyata, angka K1/K2 mentah XGBoost tidak bisa dipercaya begitu
saja untuk keputusan stok. Post-hoc rearrangement (Chernozhukov, Fernández-Val
& Galichon, 2010) — mengurutkan 19 prediksi kuantil tiap baris naik sebelum
dinilai, yang pada grid `QUANTILE_SET_A` yang sudah berurutan naik
menjamin `crossing_rate = 0` secara struktural — dijalankan pada 30 Agustus
2026 dengan hyperparameter pemenang yang sama (skrip
`xgb_rearrangement_walkforward.py`, ~2,89 jam CPU Mac, karena rearrangement
yang jujur lintas fold berarti fit ulang kelima model fold, bukan operasi
post-hoc murah pada prediksi yang sudah tersimpan). Cek reproduksibilitas
cocok persis dengan run resmi (`best_iteration` per fold identik, K1
sebelum sort = 2,9433 persis sama), jadi angka sesudahnya sah dipercaya:

| | sebelum | sesudah |
|---|---:|---:|
| K1 (fold 1/2/4 bersih) | 2,9433 | **2,9115** |
| `crossing_rate` | 0,9767 | **0,0000** |

K1 membaik **1,08%** dan crossing hilang seluruhnya, tapi **ini tidak
membalikkan peringkat**: gap ke RF (K1 2,8508) menyempit dari 3,24% menjadi
**2,13%** — masih di atas ambang keputusan K1 (2%,
`docs/detail-tahap-perbandingan-model.md` Bagian 1.7). Angka XGBoost
sesudah rearrangement inilah yang sah dipakai untuk perbandingan lintas
model.

## 1.9 Hasil per Segmen Permintaan dan per Hari Pengiriman

| segmen | n | K1 XGBoost | K1 RF | K1 `naive_roll_mean_7` |
|---|---:|---:|---:|---:|
| smooth | 45.485 | 11,0466 | **10,9478** | 18,6402 |
| erratic | 54.511 | 5,4969 | **5,4788** | 9,4628 |
| lumpy | 123.545 | 1,1823 | **1,1430** | 1,7686 |
| intermittent | 122.006 | 0,4978 | **0,4194** | 0,6919 |

XGBoost menang K1 di keempat segmen melawan baseline, tapi kalah dari RF di
ketiga segmen selain `lumpy` (di mana keduanya nyaris seri). Coverage@0,9
XGBoost konsisten **di bawah** RF di keempat segmen (0,88–0,92 lawan
0,90–0,95) — pola yang sama dengan Bagian 1.8: XGBoost lebih dekat ke
garis target 0,90, RF lebih longgar di atasnya.

Di hari kirim (`is_delivery_day=True`), shortfall XGBoost (40.237 unit)
jauh lebih rendah daripada RF (161.063), konsisten dengan MAE XGBoost yang
lebih ketat di seluruh dokumen ini.

## 1.10 Model Final dan Interpretasi Bisnis

Model final: 1.349.011 baris training, `best_iteration = 201` — jauh di
bawah plafon 2.000 yang nyaris tersentuh `DEFAULT_PARAMS` di benchmark
(Bagian 1.5), bukti konfigurasi pemenang jauh lebih hemat ronde untuk
konvergensi yang sama atau lebih baik.

| | kekurangan (shortfall) | kelebihan (overstock) |
|---|---:|---:|
| `xgboost` | 500.579 | 4.132.651 |
| `random_forest` | 418.250 | 4.793.038 |
| `naive_roll_mean_7` | 1.528.393 | 1.804.789 |

XGBoost punya shortfall **lebih tinggi** dan overstock **lebih rendah**
daripada RF — arahnya konsisten dengan coverage@0,9 XGBoost yang lebih
dekat ke target nominal 0,90: RF over-covers lebih jauh, jadi lebih jarang
kekurangan tapi menumpuk kelebihan lebih banyak; XGBoost lebih dekat ke
garis target, jadi kekurangannya lebih sering muncul tapi kelebihannya
lebih sedikit. Mana yang lebih disukai bisnis tergantung ongkos relatif
kedua sisi — bukan sesuatu yang bisa diputuskan dari angka model saja.

## 1.11 Batasan dan Hal yang Belum Bisa Disimpulkan

- **Desember 2025 belum dibuka.**
- **`crossing_rate` mentah (97,7%) sebagian besar defek sungguhan**,
  dikonfirmasi lewat pengujian toleransi jarak (Bagian 1.8) — bukan
  "97,7% prediksi tidak berguna" (sebagian besar derau kecil), tapi
  ~20–25% baris punya crossing material yang tidak hilang dengan
  toleransi wajar. Rearrangement (2026-08-30) menjawab prasyarat ini,
  bukan lagi batasan terbuka, tapi juga bukan alasan membalik pemenang.
- **Fold 3 dan 5 ikut memilih hyperparameter** — potongan fold 1/2/4
  adalah angka bersih.
- **Satu seed, satu kali latih.** Tidak ada pengulangan seed untuk
  XGBoost — beda dari LSTM, yang inisialisasi bobotnya acak; XGBoost
  dengan `random_state` tetap secara umum lebih deterministik, tapi
  klaim itu belum diuji langsung.
- **Peringkat kandidat pencarian jauh kurang stabil terhadap K1**
  dibanding RF (Bagian 1.6) — pencarian dengan anggaran lebih besar
  kemungkinan masih akan menggeser pemenang lebih jauh di XGBoost
  daripada di RF.

---

# Bagian 2 — Teknis (Mendetail)

## 2.1 Struktur Modul dan Fungsi

`utils/modelling/model_xgboost.py`:

| Fungsi | Peran |
|---|---|
| `make_fit_predict(params, feature_cols, quantiles, tail_days, early_stopping_rounds, max_rounds, idx_cols, device)` | Fungsi latih-dan-prediksi dua-fit yang disuntikkan ke `walk_forward.run_fold()` |
| `encode(train_X, valid_X, encoding, idx_cols)` | Menyiapkan kedua matriks di bawah salah satu dari tiga encoding yang dicari |
| `training_categories(train_X, idx_cols)` | Level kategori tiap kolom `_idx` saat training, disimpan di bundle |
| `apply_encoding(X, encoding, columns, categories, idx_cols)` | Encode frame untuk prediksi terhadap layout yang tercatat saat fit |
| `sample_search_space(n_candidates, seed, space)` | Menarik kandidat acak — **tanpa penyaring keterjangkauan**, berbeda dari RF |
| `run_search(df, candidates, ..., device)` | Menilai kandidat, mendelegasikan ke `model_common.run_search()` dengan `XGBoostError` ditambahkan ke tuple yang ditangkap |
| `fit_final(df, params, ...)` | Dua-fit pada seluruh baris layak sebelum Desember |
| `predict_bundle(bundle, frame)` | Inferensi, memaksa urutan kolom dan level kategori yang tercatat |

## 2.2 Tiga Mode Encoding Kategorikal

```python
ENCODINGS = ("ordinal", "native", "one_hot")
```

| Mode | Perlakuan |
|---|---|
| `ordinal` | Kolom `_idx` dipakai apa adanya (integer) |
| `native` | Dikonversi ke `pd.CategoricalDtype`, dipakai bersama `enable_categorical=True` — XGBoost menangani split kategorikal langsung. Level unseen di validasi menjadi `NaN`, dikonsumsi natural oleh XGBoost sebagai nilai hilang |
| `one_hot` | Ekspansi via `model_common.expand_one_hot()` |

Di ketiga mode, kolom validasi **dipaksa mengikuti kolom training**
(`reindex`) — kategori yang hanya muncul di validasi akan menggeser setiap
kolom sesudahnya, membuat booster membaca fitur yang salah di posisi yang
sama, secara diam-diam karena bentuknya tetap cocok.

## 2.3 `make_fit_predict()` — Alur Dua Fit

```python
def fit_predict(train, valid):
    model_common.assert_no_nan(train, feature_cols)
    model_common.assert_no_nan(valid, feature_cols)

    fit_rows, es_rows = split_early_stopping(train, tail_days=tail_days)
    fit_X, es_X, enable = encode(fit_rows[feature_cols], es_rows[feature_cols],
                                 params["encoding"], idx_cols=idx_cols)
    probe = build_estimator(params, max_rounds, enable_categorical=enable,
                            early_stopping_rounds=early_stopping_rounds,
                            quantiles=quantiles, device=device)
    probe.fit(fit_X, target(fit_rows), eval_set=[(es_X, target(es_rows))])
    best_iteration = int(probe.best_iteration) + 1
    fit_predict.best_iterations.append(best_iteration)

    train_X, valid_X, enable = encode(train[feature_cols], valid[feature_cols],
                                      params["encoding"], idx_cols=idx_cols)
    model = build_estimator(params, best_iteration, enable_categorical=enable,
                            quantiles=quantiles, device=device)
    model.fit(train_X, target(train))

    prediction = model.predict(valid_X)  # (n_valid, 19)
    if params["log_target"]:
        prediction = modeling_prep.inverse_log_target(prediction)
    return np.clip(prediction, 0.0, None)
```

`best_iterations` dicatat sebagai atribut pada fungsi yang dikembalikan
(bukan nilai balik), karena `walk_forward` hanya menerima prediksi —
sebaran jumlah ronde antar fold tetap layak dilaporkan
(`model_common.reported_capacity()` membacanya lewat introspeksi atribut).

Tidak ada sorting hasil prediksi di sini: `crossing_rate` diukur oleh
`evaluation.crossing_rate()`, dan sorting di titik ini akan memaksa metrik
itu ke nol tanpa memperbaiki apa pun — persis kebalikan dari tujuan
mengukurnya (bandingkan Bagian 2.6, rearrangement dilakukan eksplisit dan
dilaporkan sebagai langkah terpisah, bukan disembunyikan di jalur
prediksi normal).

## 2.4 Pencarian Hyperparameter — Implementasi

```python
SEARCH_FOLDS = (3, 5)
N_CANDIDATES = 30
```

Berbeda dari Random Forest, `sample_search_space()` XGBoost **tidak punya
penyaring keterjangkauan** — `tree_method="hist"` menyimpan matriks fitur
terkuantisasi (puluhan MB pada skala data ini), tanpa analogi batas memori
daun milik quantile forest.

`run_search()` mendelegasikan ke `model_common.run_search()` dengan
`catch=(MemoryError, ValueError, XGBoostError)` — `XGBoostError` ditambahkan
supaya kandidat yang parameter kombinasinya ditolak library dicatat sebagai
gagal, bukan menghentikan pencarian berjam-jam. Device diteruskan lewat
`functools.partial(make_fit_predict, device=device)`, karena
`model_common.run_search()` memanggil factory-nya dengan tiga keyword tetap
tanpa slot untuk device.

## 2.5 Fit Final dan Format Bundle

Bundle yang dikembalikan `fit_final()`:

```python
{
    "model": model,
    "params": params,
    "feature_cols": feature_cols,
    "columns": list(train_X.columns),
    "categories": training_categories(frame[feature_cols], idx_cols=idx_cols),
    "idx_cols": idx_cols,
    "encoding": params["encoding"],
    "log_target": params["log_target"],
    "best_iteration": best_iteration,
    "quantiles": quantiles,
    "device": device,          # provenance, bukan konfigurasi
    "n_train": int(len(frame)),
    "train_target": "target_lead_time_cumulative_capped",
    "eval_target": "target_lead_time_cumulative",
}
```

`categories` wajib untuk mode `native`: booster yang dimuat ulang bulan
depan terhadap kategori yang diurutkan berbeda tidak gagal — ia memprediksi
dengan percaya diri dari fitur yang salah. `device` dicatat sebagai
provenance: bundle mana pun bisa dilacak device yang menghasilkannya,
sehingga pemenang yang dipilih di GPU tidak pernah diam-diam difit ulang
seolah dari CPU.

## 2.6 Rearrangement Kuantil — Implementasi

Skrip `xgb_rearrangement_walkforward.py` (repo root) mem-fit ulang kelima
model fold dengan hyperparameter pemenang (`xgb_best_params.json`), lalu
mengurutkan (`np.sort`, naik) 19 prediksi kuantil tiap baris sebelum
dinilai — bukan operasi post-hoc murah pada prediksi yang sudah tersimpan,
karena `xgb_walk_forward_results.csv` hanya menyimpan skor teragregasi per
(model, fold, kuantil), tidak menyimpan prediksi mentah per baris. Fit
ulang jujur ini berongkos ~2,89 jam CPU Mac — sebanding dengan walk-forward
asli.

Pada grid `QUANTILE_SET_A` yang sudah berurutan naik, sort per baris
menempatkan statistik urutan ke-k pada posisi τ ke-k — inilah definisi
rearrangement Chernozhukov, Fernández-Val & Galichon (2010), dan ia
menjamin `crossing_rate = 0` secara struktural apa pun input yang masuk.

Artefak: `dataset/model_ready/xgb_walk_forward_results_rearranged.csv`
(model_name `xgboost_rearranged`, tidak masuk git).

## 2.7 Ongkos Komputasi

Pencarian (GPU Windows) dan walk-forward/fit-final (CPU Mac) **tidak
sebanding satu sama lain** — beda device, jadi wall-clock keduanya
dilaporkan terpisah:

| tahap | device | wall clock |
|---|---|---:|
| Benchmark | cpu (Mac) | 265,2 menit |
| Pencarian 30 kandidat | cuda (Windows) | ~14,7 jam |
| Walk-forward 5 fold | cpu (Mac) | ~3 jam 1 menit (estimasi timestamp) |
| Fit final | cpu (Mac) | ~23 menit (estimasi timestamp) |
| Rearrangement (fit ulang 5 fold) | cpu (Mac) | ~2,89 jam |

Pada tahap yang device-nya benar-benar identik dengan RF (walk-forward +
fit final, CPU Mac): RF ~93 menit lawan XGBoost **~204 menit** — XGBoost
sekitar **2,2×** lebih lambat, konsisten dengan boosting yang membangun
pohon per ronde per titik kuantil versus RF yang membaca semua titik dari
daun yang sama.

## 2.8 Reproduksi

```bash
# Pencarian — jalankan di mesin GPU, lihat docs/runbook-pencarian-gpu-windows.md
$env:FORECAST_DEVICE = "cuda"
python run_cells.py notebook\modeling_xgb.ipynb 2-10,14

# Walk-forward + fit final — jalankan di Mac (CPU), satu mesin yang sama
# dengan RF dan LSTM untuk K3
.venv/bin/python3 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebook/modeling_xgb.ipynb
```

| Artefak | Lokasi | Masuk git |
|---|---|---|
| Hasil pencarian | `dataset/model_ready/xgb_search_results.csv` | tidak |
| Parameter terpilih | `dataset/model_ready/xgb_best_params.json` | tidak |
| Tabel hasil lengkap | `dataset/model_ready/xgb_walk_forward_results.csv` | tidak |
| Hasil rearrangement | `dataset/model_ready/xgb_walk_forward_results_rearranged.csv` | tidak |
| Model terlatih | `models/xgboost_q90.joblib` (292 MB) | tidak |
| Ringkasan hasil | `docs/hasil-modeling-xgb.md` | ya |
| Arsip run kuantil-tunggal | `docs/bak/hasil-modeling-xgb.single-quantile.bak.md` | ya |

## 2.9 Strategi Pengujian

```bash
.venv/bin/python3 -m unittest test.test_model_xgboost -v
.venv/bin/python3 -m unittest test.test_model_common -v
```

`test/test_model_xgboost.py` menguji ketiga mode encoding, protokol
dua-fit, dan format bundle secara terisolasi (fixture kecil, tanpa
`model_input.parquet` penuh).

## 2.10 Rujukan

| Topik | Berkas |
|---|---|
| Desain model | `docs/superpowers/specs/2026-08-19-xgboost-modeling-design.md` |
| Desain evaluasi multi-kuantil | `docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md` |
| Angka hasil terukur | `docs/hasil-modeling-xgb.md` |
| Runbook GPU Windows | `docs/runbook-pencarian-gpu-windows.md` |
| Prapemrosesan sampai `model_input.parquet` | `docs/detail-tahap-preprocessing.md` |
| Mesin evaluasi bersama & perbandingan lintas model | `docs/detail-tahap-perbandingan-model.md` |
| Rearrangement kuantil | Chernozhukov, V., Fernández-Val, I., & Galichon, A. (2010). "Quantile and Probability Curves Without Crossing." *Econometrica* 78(3):1093–1125. |
