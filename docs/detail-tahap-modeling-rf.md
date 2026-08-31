# Detail Tahap Pemodelan — Random Forest Kuantil

| Atribut | Keterangan |
|---|---|
| Implementasi | `quantile_forest.RandomForestQuantileRegressor` (paket `quantile-forest`) |
| Modul produksi | `utils/modelling/model_random_forest.py` |
| Modul bersama yang dipakai | `utils/modelling/model_common.py`, `walk_forward.py`, `purging.py`, `evaluation.py`, `modeling_prep.py` |
| Notebook | `notebook/modeling_rf.ipynb` |
| Uji unit | `test/test_model_random_forest.py`, `test/test_model_common.py` |
| Ruang pencarian | 1.152 kombinasi hyperparameter |
| Kandidat ditarik | 18 |
| Run hasil terakhir | 25 Agustus 2026 — run pertama di bawah kriteria multi-kuantil K1 |
| Status | Kandidat terdepan di ketiga anak tangga keputusan (K1, K2, K3) — lihat `docs/detail-tahap-perbandingan-model.md` |

**Hubungan dengan dokumen lain.** Dokumen ini menjelaskan **satu model** secara
mendalam: dasar teorinya, cara ia dibangun dan dicari hyperparameternya, dan
angka hasil yang diukur untuknya sendiri. Tiga hal yang **tidak** diulang di
sini karena sudah punya rumah:

- Prapemrosesan data sampai `model_input.parquet` — `docs/detail-tahap-preprocessing.md`.
- Metodologi evaluasi bersama (kontrak `fit_predict`, definisi fold, metrik,
  tangga keputusan, dan **perbandingan** ketiga model) — `docs/detail-tahap-perbandingan-model.md`.
- Angka yang sudah diverifikasi dan bisa berubah setiap run diulang —
  `docs/hasil-modeling-rf.md`, yang menjadi rujukan utama seluruh angka di
  Bagian 1.7–1.11 dokumen ini.

Desain aslinya ada di
`docs/superpowers/specs/2026-08-18-random-forest-modeling-design.md` dan
`docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`.

---

# Bagian 1 — Akademis (Bahan Laporan)

## 1.1 Pendahuluan dan Posisi dalam Alur Penelitian

Random Forest kuantil adalah kandidat pertama dari tiga model yang
dibandingkan pada penelitian ini (bersama XGBoost kuantil dan LSTM kuantil,
lihat `docs/detail-tahap-perbandingan-model.md`). Ketiganya meramalkan
besaran yang sama — `target_lead_time_cumulative`, total permintaan yang
harus ditanggung satu pengiriman ke depan (`docs/detail-tahap-preprocessing.md`
Subbab 3.2) — pada 56 kolom fitur yang identik (`modeling_prep.FEATURE_COLS`),
dinilai lewat mesin evaluasi bersama yang sama (`utils/modelling/walk_forward.py`).

Random Forest dipilih sebagai kandidat pertama karena dua alasan: ia baseline
yang kuat untuk data tabular heterogen (campuran fitur numerik, kategorikal,
dan kalender), dan variannya kuantil (*quantile regression forest*) menjawab
kebutuhan proyek secara langsung — meramalkan **distribusi** permintaan, bukan
titik tunggalnya, karena kebijakan bisnis mensyaratkan service level 0,9
(`docs/batasan-penelitian.md` B-9).

## 1.2 Dasar Teori: Quantile Regression Forest

`RandomForestRegressor` milik scikit-learn meminimalkan galat kuadrat atau
galat absolut. Keduanya menaksir **pusat** distribusi bersyarat (mean atau
median) dan tidak punya loss kuantil sama sekali — meminta kuantil 0,9
darinya berarti mengambil titik tengah lalu berharap ia mendekati ekor atas
distribusi, yang secara struktural tidak benar.

*Quantile regression forest* (Meinshausen, 2006) mengganti pendekatan itu:
alih-alih menyimpan hanya rata-rata nilai target di tiap daun (seperti forest
biasa), ia menyimpan **seluruh nilai target training** yang jatuh di daun itu.
Kuantil berapa pun kemudian dibaca sebagai persentil dari distribusi empiris
gabungan seluruh daun tempat suatu baris uji jatuh. Konsekuensi pentingnya:

1. **Satu model menjawab banyak titik kuantil sekaligus**, tanpa dilatih
   ulang — konsekuensi langsung dari fakta bahwa struktur pohonnya (pemisahan
   di tiap simpul) tidak bergantung pada kuantil mana yang nanti dibaca dari
   daun. Ini yang membuat migrasi dari evaluasi satu-titik (pinball@0,9) ke
   evaluasi 19-titik (`QUANTILE_SET_A`) nyaris tidak menambah ongkos untuk RF
   (Bagian 1.6), berbeda dari XGBoost dan LSTM yang harus membangun struktur
   baru per titik kuantil.
2. **Monotonicity antar-kuantil terjamin secara struktural.** Kuantil 0,3 dan
   kuantil 0,7 dari distribusi empiris yang sama tidak mungkin saling
   menyilang — keduanya dibaca dari sebaran nilai yang identik. Ini terbukti
   di data (`crossing_rate = 0,0000` pada seluruh baris hasil, Bagian 1.9),
   berbeda dari XGBoost (`multi_strategy`, satu pohon per titik kuantil per
   ronde, tanpa jaminan monotonicity) dan LSTM (kepala keluaran 19-neuron,
   juga tanpa jaminan).
3. **Konsekuensi biaya: kebutuhan memori bisa dihitung sebelum pohon
   dibangun.** Penyimpanan nilai per-daun berbentuk array padat berukuran
   `n_estimators x max_node_count x n_outputs x max_samples_leaf`, sehingga
   pohon dalam dengan daun sangat kecil (`min_samples_leaf` rendah,
   `max_depth` tinggi) menuntut memori yang bisa dihitung di muka. Pembatasan
   ini kebetulan sejalan dengan alasan statistiknya sendiri: daun berisi satu
   sampel tidak bisa menaksir kuantil sama sekali — sebuah persentil dari
   satu titik data adalah titik data itu sendiri.

## 1.3 Kontrak Evaluasi Bersama

Random Forest tidak memiliki logika fold, pemotongan baris, atau metrik
sendiri. Seluruh itu dimiliki `utils/modelling/walk_forward.py`, yang
memperlakukan setiap model — RF, XGBoost, LSTM — sebagai satu fungsi
`fit_predict(train_df, valid_df) -> np.ndarray` berukuran `(n_valid, 19)`.
Detail penuh kontrak ini (kelayakan baris, definisi lima fold, purging batas
Desember) ada di `docs/detail-tahap-perbandingan-model.md` Bagian 1.2 — di
sini cukup dicatat bahwa **RF dinilai pada baris yang identik dengan dua
model lainnya**, dijamin secara struktural, bukan disiplin penulisan kode.

Yang menjadi milik RF sendiri, dan karenanya dijelaskan di dokumen ini: cara
`make_fit_predict()` (Bagian 2.3) menyusun estimator, memilih fitur,
melakukan (opsional) ekspansi one-hot dan transformasi log target, lalu
membalikkan transformasi itu setelah prediksi keluar.

## 1.4 Baseline Naif sebagai Lantai

Sebelum RF dilatih, tiga baseline tanpa model sama sekali diukur pada baris
validasi yang identik (`evaluation.NAIVE_BASELINES` — detail di
`docs/detail-tahap-perbandingan-model.md` Bagian 1.3): `naive_zero`,
`naive_lag_1`, dan `naive_roll_mean_7`. Baseline terakhir adalah lantai
operasional — persis yang akan dilakukan manajer outlet dengan tangan bila
tidak ada model sama sekali — dan menjadi pembanding utama di seluruh Bagian
1.7–1.10 di bawah.

## 1.5 Ruang Pencarian Hyperparameter

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

1.152 kombinasi (3 × 4 × 3 × 4 × 2 × 2 × 2), disaring lewat
`estimate_leaf_memory_bytes()` terhadap budget memori 3 GB **sebelum** data
dimuat (Bagian 2.2), lalu 18 kandidat ditarik acak dengan seed 42.

Dua batasan ruang yang layak dicatat sebagai keputusan desain, bukan
kealpaan:

- **`n_estimators` tidak ikut dicari.** Mutu forest monoton terhadap jumlah
  pohon — menambah pohon tidak pernah memperburuk hasil, hanya memperlambat
  fit — sehingga mencarinya membelanjakan anggaran pencarian untuk
  pertanyaan yang jawabannya sudah diketahui. Ia dipatok 200 selama
  pencarian dan dinaikkan ke 400 untuk model final (Bagian 1.9), yang
  terjangkau karena hanya satu fit yang perlu menanggung ongkosnya.
- **`max_depth=None` dan `min_samples_leaf` di bawah 20 tidak masuk ruang.**
  Keduanya melanggar budget memori (Bagian 2.2) sekaligus merusak taksiran
  kuantil — daun yang terlalu kecil tidak punya cukup sampel untuk
  menaksir kuantil ekstrem (0,05 atau 0,95) secara andal.

Penilaian kandidat berjalan di **fold 3 (September) dan fold 5 (November)
saja** — dua dari lima fold, karena tahap pencarian harus murah, dengan
kriteria K1 (rata-rata pinball tak berbobot lintas 19 titik kuantil,
dibobot jumlah baris antar fold). Konsekuensi memakai hanya dua fold untuk
memilih hyperparameter dibahas di `docs/detail-tahap-perbandingan-model.md`
Bagian 1.2.

## 1.6 Ongkos Multi-Kuantil: Nyaris Gratis

Benchmark satu putaran fit+predict pada `DEFAULT_PARAMS` (Bagian 2.1) di
fold 5 mengukur wall time 9,7 menit untuk memprediksi seluruh 19 titik
kuantil sekaligus, dibanding 6,6 menit di run kuantil-tunggal lama untuk satu
titik — pengganda **×1,47** untuk 19× lebih banyak titik keluaran. Ini
mengonfirmasi klaim teoretis di Bagian 1.2: yang mahal pada RF adalah
membangun struktur pohon, bukan membaca persentil tambahan dari daun yang
sudah jadi. Bandingkan XGBoost, yang membangun satu pohon boosting baru per
titik kuantil per ronde dan membayar pengganda **×15,2** untuk migrasi yang
sama (`docs/detail-tahap-modeling-xgb.md` Bagian 1.5).

## 1.7 Hasil Pencarian dan Pemenang

18 kandidat, seluruhnya berhasil dinilai (tidak ada yang gagal budget
memori atau numerik). Parameter terpilih:

```json
{
  "log_target": false,
  "max_depth": 20,
  "max_features": 1.0,
  "max_samples": null,
  "max_samples_leaf": 1,
  "min_samples_leaf": 20,
  "n_estimators": 200,
  "one_hot": false,
  "random_state": 42
}
```

Dua bacaan penting dari sebaran 18 kandidat (tabel lengkap di
`docs/hasil-modeling-rf.md` Bagian 4.1):

1. **Ruang parameternya datar.** K1 bergerak dari 2,8808 (terbaik) sampai
   3,1969 (terburuk) — rentang hanya 11% — dan lima kandidat teratas
   berjarak 1,46% satu sama lain. Menambah anggaran pencarian di ruang ini
   kemungkinan besar tidak banyak mengubah hasil.
2. **`max_features="sqrt"` satu-satunya pilihan yang benar-benar
   merugikan.** Kedua kandidat terburuk memakainya. Dengan 56 fitur,
   `sqrt` menyisakan hanya ~7 fitur per pemisahan simpul — terlalu sedikit
   untuk struktur fitur proyek ini, yang mencampur riwayat permintaan,
   kalender, dan atribut cabang dalam proporsi yang relatif seimbang.

Pemenangnya **berbeda** dari pemenang run kuantil-tunggal lama (kandidat 17:
`max_depth=12`, `one_hot=True` — terpisah hanya 0,0004 di kriteria lama,
praktis seri, tapi 0,0177 di K1). Peringkat kandidat sendiri nyaris tidak
berubah antara kedua kriteria (Spearman ρ = 0,975): merata-ratakan 19 titik
meredam derau yang tadinya menyembunyikan selisih kecil ini, bukan mengubah
lanskap pencarian secara mendasar.

## 1.8 Gerbang Kelayakan (G0) dan Hasil Walk-Forward Lima Fold

**G0 — kelayakan.** Model harus mengalahkan `naive_roll_mean_7` pada
pinball@0,9 di **kelima** fold, bukan hanya gabungan. RF lolos dengan margin
40,5%–48,7% di setiap bulan — tidak ada satu bulan pun yang menggendong
kemenangan gabungannya.

**K1 — kriteria utama.** Pada potongan fold 1/2/4 (tiga fold yang tidak ikut
memilih hyperparameter — lihat Bagian 1.5), RF mencetak:

| model | K1 (fold 1/2/4) | K1 (5 fold) |
|---|---:|---:|
| **random_forest** | **2,8508** | **2,8621** |
| `naive_roll_mean_7` | 4,8603 | 4,8231 |
| `naive_lag_1` | 8,1612 | 8,1755 |
| `naive_zero` | 14,8102 | 14,7469 |

Sekitar **41% lebih baik** daripada baseline terbaik. Performanya stabil
lintas fold (K1 bergerak 2,682–3,040, rentang 13%), dan margin kemenangan
serupa di kelima bulan — fold 4 (Oktober) paling berat, tapi baseline juga
memuncak di sana, jadi itu properti bulannya, bukan properti RF.

RF **kalah MAE** dari `naive_roll_mean_7` (15,055 lawan 9,721) — bukan
kegagalan, melainkan konsekuensi yang diminta: prediksi di τ=0,9 sengaja
bias ke atas, dan MAE menghukum bias ke atas persis seperti ia menghukum
kekurangan stok. MAE dilaporkan sebagai konteks, bukan kriteria kemenangan
(justifikasi lengkap di `docs/detail-tahap-perbandingan-model.md` Bagian 1.5).

## 1.9 Kalibrasi (K2) dan Fenomena "Lantai" `share_nol`

Di τ=0,9 — titik yang benar-benar dijanjikan ke bisnis (B-9) — coverage RF
0,928 terhadap target 0,90, dengan fill rate 0,959. `crossing_rate = 0,0000`
di **seluruh** baris hasil: bukan kabar baik yang mengejutkan, melainkan cek
struktural yang lolos sebagaimana dijelaskan di Bagian 1.2.

Yang lebih instruktif adalah pola kalibrasi lintas seluruh 19 titik kuantil.
Karena target tidak pernah negatif dan **41,95%** baris validasi bertarget
nol, setiap model tak-negatif otomatis mencetak coverage minimal `share_nol`
di titik kuantil berapa pun — sebuah baris ber-target nol selalu terhitung
"tercakup" (`0 ≤ prediksi`) apa pun nilai prediksinya. Lantai ini
menyebabkan simpangan `coverage(τ) − τ` besar di τ rendah pada model apa
pun di dataset ini, dan **tidak boleh dibaca langsung sebagai bias
kalibrasi** tanpa dikurangi lantainya lebih dulu.

Setelah lantai dikurangkan, RF menyisakan over-coverage yang nyata: memuncak
**+0,181 pada τ=0,40–0,45** dan tetap +0,16 di median (τ=0,50, RF mencakup
66% baris — median ramalannya bergeser terlalu tinggi). Ini bukan artefak
lantai — ia bias sistematis yang berlaku sama untuk model apa pun yang
kalibrasinya diperiksa dengan cara ini, dan menjadi salah satu titik
pembanding utama terhadap XGBoost dan LSTM di
`docs/detail-tahap-perbandingan-model.md`.

Bagian dari over-coverage ini turut dijelaskan oleh **efek ikatan**: 99,55%
target bilangan bulat dan 70,3% bernilai ≤5, sehingga prediksi dan aktual
sering bernilai sama persis, dan definisi coverage dengan `≤` menghitung
ikatan itu sebagai tercakup. Diukur langsung: mengganti `≤` dengan `<` tegas
justru **membalik arah** simpangan (dari +0,175 menjadi −0,255 di τ=0,40) —
bukan karena ikatan itu artefak, tapi karena `<` menghukum prediksi yang
tepat sasaran sebagai "tidak tercakup". `≤` (definisi standar kuantil) tetap
yang paling defensif dipakai; kesimpulannya, over-coverage +0,18 di median
tetap bias nyata, dibaca dengan konteks bahwa sekitar 43% baris di sekitarnya
memang bernilai identik prediksi-aktual (properti kediskretan target,
bukan artefak metrik semata).

## 1.10 Hasil per Segmen Permintaan dan per Hari Pengiriman

Karena 44% target bernilai nol, satu angka global bisa menyesatkan — model
bisa menang hanya karena unggul di tempat menebak nol itu mudah. Dipecah per
`demand_segment` (klasifikasi Syntetos-Boylan, `docs/detail-tahap-preprocessing.md`
Subbab 7.2):

| segmen | n | K1 RF | K1 `naive_roll_mean_7` | margin |
|---|---:|---:|---:|---:|
| smooth | 45.485 | **10,9478** | 18,6402 | 41% |
| erratic | 54.511 | **5,4788** | 9,4628 | 42% |
| lumpy | 123.545 | **1,1430** | 1,7686 | 35% |
| intermittent | 122.006 | **0,4194** | 0,6919 | 39% |

RF menang K1 di **keempat** segmen — kemenangan globalnya bukan hasil
menang di pasangan yang mayoritas nol. Menariknya, di `intermittent` dan
`lumpy`, MAE RF **lebih buruk** daripada `naive_zero` — bukan anomali,
karena coverage `naive_zero` di kedua segmen itu sama dengan share target
nolnya, jadi menebak nol terus menghasilkan MAE kecil dengan konsekuensi
fill rate nol. Ini justru alasan `demand_segment` dijadikan sumbu pelaporan
wajib.

Dipecah per `is_delivery_day` (baris yang benar-benar menaikkan barang ke
truk), margin RF praktis sama di hari kirim (39%) dan non-kirim (42%) —
berbeda dari run kuantil-tunggal lama, di mana keunggulan tampak
terkonsentrasi di hari kirim. Bacaan lama itu **tidak lagi didukung data**
di bawah kriteria K1.

## 1.11 Model Final dan Interpretasi Bisnis

Model final dilatih ulang pada seluruh 1.349.011 baris layak sebelum
Desember (`n_estimators` dinaikkan 200 → 400), dipotong oleh
`purging.lookahead_safe_mask()` di batas Desember. Di 345.547 baris validasi
gabungan lima fold:

| | kekurangan (shortfall) | kelebihan (overstock) |
|---|---:|---:|
| `random_forest` | 418.250 | 4.793.038 |
| `naive_roll_mean_7` | 1.528.393 | 1.804.789 |

RF memangkas kekurangan stok **73%** dengan ongkos kelebihan stok **2,7×**
lipat. Apakah pertukaran ini benar adalah keputusan bisnis, bukan keputusan
model — tapi itu persis pertukaran yang diminta ketika service level
dipatok di 0,9 (unit dijumlahkan lintas SKU bersatuan campur, sah untuk
membandingkan model pada baris yang sama, tidak punya makna fisik sebagai
satu besaran tunggal).

## 1.12 Batasan dan Hal yang Belum Bisa Disimpulkan

- **Desember 2025 belum dibuka.** Seluruh angka di dokumen ini adalah
  validasi walk-forward Juli–November 2025, bukan skor test set final.
- **Fold 3 dan 5 ikut memilih hyperparameter**, jadi skor di kedua fold itu
  bukan out-of-sample terhadap seleksi. Potongan fold 1/2/4 adalah angka
  bersih dan menjadi K1 resmi.
- **Satu seed, satu kali latih.** RF kurang mengkhawatirkan di sini
  dibanding LSTM — bagging 200+ pohon sudah merata-ratakan sebagian besar
  varians inisialisasi — tapi tetap berarti selisih di bawah ambang 2%
  tidak bisa dipisahkan dari derau tanpa pengulangan.
- **K2 di τ rendah** belum bisa dibaca murni sebagai kalibrasi sampai
  aturan penyisihan tangga keputusan dinyatakan ulang terhadap lantai
  `share_nol` — lihat `docs/detail-tahap-perbandingan-model.md` Bagian 1.7.
- **Satu model saja belum berarti model terbaik.** Apakah RF unggul atas
  XGBoost dan LSTM adalah pertanyaan yang dijawab di
  `docs/detail-tahap-perbandingan-model.md`, bukan di dokumen ini.

---

# Bagian 2 — Teknis (Mendetail)

## 2.1 Struktur Modul dan Fungsi

`utils/modelling/model_random_forest.py` menyediakan satu model kandidat
lengkap dengan empat tahap pemakaian:

| Fungsi | Peran |
|---|---|
| `make_fit_predict(params, feature_cols, quantiles, memory_budget)` | Membentuk fungsi latih-dan-prediksi yang disuntikkan ke `walk_forward.run_fold()` |
| `sample_search_space(n_candidates, n_train, seed, memory_budget, space)` | Menarik kandidat hyperparameter acak yang lolos penyaringan memori |
| `run_search(df, candidates, folds, ...)` | Menilai tiap kandidat di fold pencarian, mendelegasikan protokol checkpoint ke `model_common.run_search()` |
| `fit_final(df, params, feature_cols, n_estimators, ...)` | Melatih model akhir pada seluruh baris layak sebelum Desember |
| `predict_bundle(bundle, frame)` | Inferensi dari bundle tersimpan, memaksa urutan kolom yang tercatat |

Konstanta modul: `QUANTILE = 0.9` (janji bisnis, B-9 — berbeda sengaja dari
`QUANTILES`, grid evaluasi 19 titik yang diimpor dari `evaluation.QUANTILE_SET_A`
supaya migrasi Tahap A → Tahap B tidak menyisakan modul ini menilai di grid
lama), `MEMORY_BUDGET_BYTES = 3 * 1024**3` (3 GB), `DEFAULT_PARAMS`,
`SEARCH_SPACE`, `ESTIMATOR_KEYS` (kunci yang benar-benar diteruskan ke
`RandomForestQuantileRegressor`; `log_target` dan `one_hot` ditangani modul
ini sendiri, bukan oleh `quantile-forest`).

`IDX_COLS`, `assert_no_nan`, `expand_one_hot`, `select_best`, `load_bundle`
diekspor ulang dari `model_common` supaya pemanggil modul ini (test suite,
`modeling_rf.ipynb`) tetap jalan tanpa perubahan setelah ekstraksi mesin
bersama ke `model_common.py`.

## 2.2 Estimasi Memori Leaf Storage

```python
def estimate_leaf_memory_bytes(params: dict, n_train: int) -> int:
    fraction = params.get("max_samples") or 1.0
    n_bootstrap = n_train * fraction
    depth_bound = 2.0 ** (params["max_depth"] + 1)
    leaf_bound = 2.0 * n_bootstrap / params["min_samples_leaf"]
    node_count = min(depth_bound, leaf_bound)
    return int(params["n_estimators"] * node_count * params["max_samples_leaf"] * 8)
```

Jumlah simpul dibatasi dua kali — oleh kedalaman maksimum (pohon kedalaman
`d` memuat paling banyak `2^(d+1)` simpul) dan oleh ukuran daun minimum
(`n` baris yang terbagi ke daun berisi minimal `L` baris menghasilkan
paling banyak `2n/L` simpul termasuk simpul internal). Batas yang lebih
ketat di antara keduanya yang dipakai. Byte dihitung sebagai
`n_estimators × jumlah_simpul × max_samples_leaf × 8` (float64 per nilai).

Fungsi ini dipakai menyaring kandidat **sebelum data dimuat** — disuntikkan
sebagai `screen` ke `model_common.sample_search_space()` — sehingga
konfigurasi yang tidak terjangkau ditolak tanpa membangun satu pohon pun,
bukan ditemukan lewat OOM killer dua puluh menit setelah fit berjalan.

Diverifikasi terhadap benchmark nyata (fold 5, 1.292.778 baris training):
estimasi 1,54 GB terhadap budget 3 GB — meleset hanya 1% dari
`TYPICAL_N_TRAIN = 1_280_000` yang dipakai default menyaring kandidat
sebelum jumlah baris sebenarnya diketahui.

## 2.3 `make_fit_predict()` — Alur Fit dan Prediksi

Fungsi yang dikembalikan menerima `(train, valid)` dan mengembalikan matriks
prediksi `(len(valid), len(quantiles))`. Urutan langkah di dalamnya:

1. `assert_no_nan(train, feature_cols)` dan pada `valid` — model ini tidak
   melakukan imputasi sendiri; `model_input.parquet` sudah terimputasi
   (`docs/detail-tahap-preprocessing.md` Subbab 7.4).
2. Penyaringan budget memori via `estimate_leaf_memory_bytes()` — melempar
   `MemoryError` bila kandidat melebihi budget, yang ditangkap oleh
   `model_common.run_search()` sebagai kandidat gagal (metrik NaN), bukan
   menggagalkan seluruh pencarian.
3. Pemilihan fitur (`train[feature_cols]`), lalu ekspansi one-hot opsional
   via `model_common.expand_one_hot()` bila `params["one_hot"]`.
4. Pengambilan target latih via `model_common.train_target()` — target
   capped (`TRAIN_TARGET_COL`), dengan log1p opsional.
5. `build_estimator(params)` menyusun `RandomForestQuantileRegressor(n_jobs=-1, **kwargs)`
   dari `ESTIMATOR_KEYS` yang relevan, lalu `.fit()`.
6. **Satu panggilan `predict(quantiles=list(quantiles))`** untuk seluruh
   grid — bukan 19 panggilan terpisah, karena leaf sudah memuat distribusi
   training dan setiap titik tambahan hanya berongkos satu pembacaan lagi.
7. Pembalikan log1p (`modeling_prep.inverse_log_target()`) bila dipakai,
   lalu `np.clip(prediction, 0.0, None)` — kuantitas kirim negatif tidak
   punya makna fisik.

Pemilihan fitur, ekspansi one-hot, dan transformasi target sengaja tetap di
sini, bukan di `walk_forward`, karena ketiganya adalah **pilihan model** —
persis hal yang seharusnya diperbandingkan antar tiga kandidat, bukan
disembunyikan di lapisan bersama.

## 2.4 Pencarian Hyperparameter — Implementasi

`sample_search_space()` modul ini membungkus
`model_common.sample_search_space()` dengan `screen` yang memeriksa budget
memori terhadap `n_train` representatif (`TYPICAL_N_TRAIN`). `run_search()`
mendelegasikan seluruhnya ke `model_common.run_search()`: checkpoint ditulis
atomik (`os.replace`) setiap kandidat selesai ke path yang diberikan
(default `dataset/model_ready/rf_search_results.csv`), dengan guard
`_assert_checkpoint_matches()` yang menolak melanjutkan dari checkpoint yang
lahir dari ruang pencarian atau grid kuantil berbeda — pemeriksaan skema
lewat kolom `headline_quantile`, yang tidak ada di checkpoint pra-migrasi
K1.

`SEARCH_FOLDS = (3, 5)` (September dan November). `select_best()` memilih
kandidat dengan `pinball` (K1) gabungan terendah — satu baris kode
(`model_common.select_best`) yang sama dipakai ketiga model, sehingga
kriteria seleksi tidak pernah berpindah-pindah antar model.

## 2.5 Fit Final dan Format Bundle

```python
FINAL_N_ESTIMATORS = 400  # naik dari 200 yang dipakai saat pencarian
```

`fit_final()` mengambil baris layak dari `walk_forward.eligible_rows()` —
bukan filter tanggal yang ditulis ulang di modul ini — supaya populasi
baris yang melatih model final identik dengan populasi yang menilainya di
walk-forward. Dipotong lagi oleh `purging.lookahead_safe_mask()` di batas
Desember, sehingga tidak ada baris training yang targetnya menjangkau ke
dalam test set.

Bundle yang dikembalikan:

```python
{
    "model": model,                 # RandomForestQuantileRegressor terlatih
    "params": params,               # hyperparameter lengkap termasuk n_estimators=400
    "feature_cols": feature_cols,   # 56 kolom
    "columns": list(train_X.columns),  # urutan kolom SETELAH one-hot (bila dipakai)
    "quantiles": tuple(quantiles),  # grid 19 titik
    "n_train": int(len(frame)),
    "train_target": "target_lead_time_cumulative_capped",
    "eval_target": "target_lead_time_cumulative",
}
```

Urutan kolom training disimpan bersama modelnya karena forest yang dimuat
ulang dengan urutan kolom berbeda **tidak gagal** — ia meramal dengan
percaya diri dari fitur yang salah, dan itu lebih berbahaya daripada
kegagalan eksplisit. `predict_bundle()` membaca grid kuantil dari bundle,
bukan dari konstanta modul, supaya bundle lama tetap terbaca setelah
`QUANTILE_SET` berpindah dari Tahap A ke Tahap B.

## 2.6 Tabel Lengkap Hasil Pencarian (18 Kandidat)

Sumber: `dataset/model_ready/rf_search_results.csv` (tidak masuk git).
Kolom lengkap dan bacaan sebarannya ada di `docs/hasil-modeling-rf.md`
Bagian 4.1–4.3, termasuk analisis perbandingan peringkat kandidat terhadap
run kuantil-tunggal lama (Spearman ρ = 0,975, Kendall τ = 0,895 — peringkat
nyaris tidak berubah meski nilainya berubah dan pemenangnya berpindah).

## 2.7 Ongkos Komputasi

Seluruh run RF dijalankan di CPU Mac lokal (keputusan pemilik proyek
2026-08-25 bahwa seluruh Fase 3 dijalankan di satu device untuk K3 yang
sebanding — belakangan XGBoost dan LSTM memindahkan tahap pencarian mereka
ke GPU Windows, tapi RF tetap di CPU Mac sepanjang run):

| tahap | wall clock |
|---|---:|
| Benchmark (1 fit + predict 19 titik) | 9,7 menit |
| Pencarian 18 kandidat | 3,85 jam |
| Walk-forward 5 fold | ~45 menit |
| Fit final | ~48 menit |
| **Total** | **~5,6 jam** |

Peak RSS proses saat benchmark 4,14 GB — lebih besar dari budget 3 GB, dan
itu **bukan** pelanggaran: budget membatasi penyimpanan daun saja, sementara
RSS ikut memuat panel 1,5 juta baris, matriks fitur, dan struktur pohon itu
sendiri.

## 2.8 Reproduksi

```bash
.venv/bin/python3 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebook/modeling_rf.ipynb
```

Butuh berjam-jam. Pencarian menulis checkpoint tiap kandidat selesai ke
`rf_search_results.csv` dan melanjutkan dari sana bila dijalankan ulang,
dijaga `_assert_checkpoint_matches()`.

| Artefak | Lokasi | Masuk git |
|---|---|---|
| Hasil pencarian | `dataset/model_ready/rf_search_results.csv` | tidak |
| Parameter terpilih | `dataset/model_ready/rf_best_params.json` | tidak |
| Tabel hasil lengkap | `dataset/model_ready/rf_walk_forward_results.csv` | tidak |
| Forest terlatih | `models/random_forest_q90.joblib` (826 MB) | tidak |
| Ringkasan hasil | `docs/hasil-modeling-rf.md` | ya |
| Arsip run kuantil-tunggal | `docs/bak/hasil-modeling-rf.single-quantile.bak.md` | ya |

## 2.9 Strategi Pengujian

`test/test_model_random_forest.py` menguji modul ini secara terisolasi
(bentuk keluaran `make_fit_predict()`, penyaringan memori, format bundle,
pembalikan log target); `test/test_model_common.py` menguji mesin bersama
yang dipakainya (checkpoint, `select_best()`, ekspansi one-hot). Jalankan:

```bash
.venv/bin/python3 -m unittest test.test_model_random_forest -v
.venv/bin/python3 -m unittest test.test_model_common -v
```

## 2.10 Rujukan

| Topik | Berkas |
|---|---|
| Desain model | `docs/superpowers/specs/2026-08-18-random-forest-modeling-design.md` |
| Desain evaluasi multi-kuantil | `docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md` |
| Angka hasil terukur | `docs/hasil-modeling-rf.md` |
| Prapemrosesan sampai `model_input.parquet` | `docs/detail-tahap-preprocessing.md` |
| Mesin evaluasi bersama & perbandingan lintas model | `docs/detail-tahap-perbandingan-model.md` |
| Quantile regression forest | Meinshausen, N. (2006). "Quantile Regression Forests." *JMLR* 7:983–999. |
