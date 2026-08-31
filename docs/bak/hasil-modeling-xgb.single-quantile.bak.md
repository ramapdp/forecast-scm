# Hasil Modeling — XGBoost kuantil 0.9

Angka terukur dari jalannya `notebook/modeling_xgb.ipynb`. Desainnya ada di
`docs/superpowers/specs/2026-08-19-xgboost-modeling-design.md`; dokumen ini
hanya memuat hasilnya, supaya bukti di balik tulisan hidup di git dan bukan
cuma di output cell notebook yang bisa hilang saat di-clear.

**Artefak `models/xgboost_q90.joblib` sudah basi (per 2026-08-23) — jangan
dipakai untuk prediksi.** Ia dilatih 19 Agu 2026 di atas `model_input.parquet`
sebelum refresh kategori WIP-2 2026-08-22, jadi level 4 pada `Kategori Barang_idx` kini tak pernah muncul.
Model yang dimuat ulang tidak akan gagal — ia tetap memberi angka, dari fitur
yang salah. Angka di dokumen ini tetap sah sebagai catatan run tersebut, bukan
sebagai gambaran `model_input.parquet` yang sekarang. Model ini akan **dilatih
ulang** dalam migrasi multi-kuantil berikutnya; latar lengkapnya di bagian 0
`docs/pipeline-overview.md` dan B-9 `docs/batasan-penelitian.md`.

**Desember 2025 tidak dibuka.** Semua angka di bawah datang dari walk-forward
lima fold di Juli–November 2025. Test set final masih terkunci.

## 1. Ringkasan

XGBoost kuantil 0.9 (`reg:quantileerror`) mencetak pinball@0.9 gabungan
**2.390** lawan **4.503** milik baseline terbaik `naive_roll_mean_7` — sekitar
47% lebih baik. Kalibrasinya nyaris tepat di sasaran: coverage **0.909**
terhadap target service level 0.90.

Terhadap Random Forest (2.410) selisihnya **0,8%**, dan di potongan fold yang
tidak ikut memilih pemenang selisih itu menyusut jadi **0,1%** (2.400 lawan
2.403) — praktis seri. bagian 6 membahas kenapa angka setipis itu belum cukup untuk
menobatkan pemenang.

XGBoost **kalah MAE** dari `naive_roll_mean_7` (14.31 lawan 9.65), persis
seperti RF dan karena alasan yang sama: model kuantil 0.9 sengaja bias ke atas,
dan MAE menghukum bias ke atas sama kerasnya seperti ia menghukum kekurangan
stok. MAE dilaporkan untuk konteks, bukan sebagai kriteria kemenangan.

Sisi bisnisnya, di 345.547 baris validasi:

| | kekurangan (shortfall) | kelebihan (overstock) |
|---|---:|---:|
| `xgboost` | 414.172 | 4.529.674 |
| `random_forest` | 365.576 | 5.038.816 |
| `naive_roll_mean_7` | 1.528.393 | 1.804.789 |

XGBoost memangkas kekurangan stok **73%** dari baseline dengan ongkos kelebihan
stok **2,5×** lipat. Dibanding RF, ia menahan lebih sedikit kelebihan stok
(4,53 juta lawan 5,04 juta unit, −10%) dengan ongkos kekurangan stok 13% lebih
besar — konsekuensi langsung dari coverage-nya yang lebih dekat ke 0.90
ketimbang RF yang duduk di 0.932.

> Catatan satuan: `shortfall_units` dan `overstock_units` menjumlahkan unit
> lintas SKU yang satuannya campur (Kg, Porsi, Botol, PCS, …), jadi angkanya
> sah untuk membandingkan antar model pada baris yang sama, tapi tidak punya
> makna fisik sebagai satu besaran tunggal.

## 2. Setup evaluasi

| | |
|---|---|
| Data | `dataset/model_ready/model_input.parquet` — 1.502.522 baris panel, 82 kolom, 1 Jan 2024 – 31 Des 2025 |
| Fitur | 56 kolom (`modeling_prep.FEATURE_COLS`) |
| Target | `target_lead_time_cumulative` — 44,35% bernilai nol |
| Fold | 5 expanding window, validasi Juli, Agustus, September, Oktober, November 2025 |
| Test terkunci | Desember 2025 (`TEST_START = 2025-12-01`) |
| Baris tereliminasi | 28 hari awal tiap `segment_id` (jendela lag belum penuh) dan baris tanpa target |
| Baris validasi | 345.547 total |
| Kuantil | 0.9 |
| Implementasi | `xgboost==2.1.4`, `XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.9, tree_method="hist")` |
| Jumlah ronde | early stopping (`EARLY_STOPPING_ROUNDS = 50`) di ekor 30 hari terakhir jendela training, lalu **refit** di seluruh baris training pada ronde itu |
| Plafon ronde | `MAX_ROUNDS = 2000` |

Fungsi objektifnya adalah pinball loss yang sama dengan kriteria seleksi, jadi
apa yang dioptimalkan saat training dan apa yang dinilai saat evaluasi tidak
berbeda — yang tidak berlaku untuk model squared-error yang dimintai kuantil
tinggi setelahnya.

**Protokol dua fit** ada karena jumlah ronde boosting itu sendiri keputusan
regularisasi, dan tempat paling wajar untuk mengambilnya — fold validasi —
justru tempat yang bocor. Jadi early stopping berjalan di ekor 30 hari terakhir
jendela training (`split_early_stopping`, dipotong purging yang sama), lalu
model dilatih ulang di **seluruh** baris training pada ronde yang dipilih ekor
itu. Hasilnya XGBoost akhirnya dilatih di populasi baris yang persis sama
dengan yang dilihat Random Forest, jadi perbandingannya setara.

Ketiga baseline naive dinilai pada **baris yang identik** dengan XGBoost —
dijamin oleh `utils/modelling/walk_forward.py` yang memiliki definisi fold dan kelayakan
baris, dan menerima model sebagai callable yang disuntikkan. Baris yang sama
itu juga yang dipakai run Random Forest, yang membuat bagian 6 sah dilakukan.

## 3. Benchmark

Satu putaran dua-fit di fold 5 dengan `DEFAULT_PARAMS` (`max_depth=6,
learning_rate=0.05, min_child_weight=10, subsample=1.0, colsample_bytree=1.0,
reg_lambda=1.0, encoding="ordinal"`), untuk mengukur ongkos sebelum 60 fit
pencarian dijalankan dan melihat di ronde berapa early stopping mendarat.

| | |
|---|---|
| Baris training fold 5 | 1.292.778 → 1.224.830 fit + 65.140 ekor early stopping |
| Baris validasi fold 5 | 59.629 |
| `best_iteration` | **399** dari plafon 2.000 |
| Wall time dua fit + predict | 2,4 menit |
| Peak RSS proses | 3,65 GB |
| Prediksi | rata-rata 43,31, maksimum 1.789,56 |

Dua hal yang dikonfirmasi angka ini:

1. **Plafon ronde tidak mengikat.** Early stopping mendarat di 399 dari 2.000,
   jadi yang menghentikan boosting adalah datanya, bukan batas yang dipasang
   di kode. Kalau ia mendarat di 2.000, angka-angka pencarian akan mencerminkan
   anggaran ronde, bukan kualitas konfigurasi — itu tidak terjadi.
2. **Ongkosnya jauh lebih murah dari RF.** Satu putaran dua-fit memakan 2,4
   menit, lawan 6,6 menit untuk satu fit Random Forest di fold yang sama —
   2,7× lebih murah per putaran, 5,5× per fit — dengan peak RSS 3,65 GB lawan
   4,92 GB. Itulah sebabnya anggaran pencarian bisa dinaikkan ke 30 kandidat
   tanpa menambah waktu jam dinding, dan itu juga yang bikin perbandingan di
   bagian 6 punya asimetri yang harus disebutkan.

Konfigurasi benchmark ini adalah `DEFAULT_PARAMS`, bukan pemenang pencarian —
tujuannya mengukur ongkos dan menguji plafon ronde, bukan mencetak skor.

Ongkos jalan sisanya, dibaca dari timestamp artefak run terakhir: empat
kandidat terakhir pencarian 8,5 menit per kandidat (dua fold, jadi ~4,3 menit
per fold), walk-forward lima fold 16 menit, fit final 4 menit.

## 4. Pencarian hyperparameter

30 kandidat ditarik acak dengan seed 42 dari ruang **2.592** kombinasi.
Penilaian di **fold 3 (September) dan fold 5 (November)** saja, kriteria
**pinball@0.9 gabungan** (dibobot jumlah baris, bukan dirata-rata polos).
**Tidak ada subsampling** — seluruh baris training tiap fold dipakai.

`n_estimators` sengaja tidak ada di ruang pencarian: early stopping sudah
memutuskannya per kandidat per fold, jadi mencarinya akan menghabiskan anggaran
untuk pertanyaan yang sudah punya mekanisme.

Ke-30 kandidat selesai dinilai; tidak ada yang gagal (kolom `error` kosong
semua).

| # | encoding | max_depth | lr | min_child_weight | subsample | colsample | reg_lambda | log_target | pinball | mae | coverage |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| **11** | **native** | 6 | 0.05 | 50 | 0.7 | 0.7 | 10.0 | False | **2.3727** | 14.235 | 0.907 |
| 10 | native | 8 | 0.05 | 1 | 0.7 | 0.7 | 1.0 | False | 2.3728 | 14.231 | 0.911 |
| 9 | native | 6 | 0.10 | 50 | 0.7 | 1.0 | 1.0 | False | 2.3752 | 14.432 | 0.904 |
| 23 | native | 6 | 0.03 | 1 | 0.7 | 1.0 | 10.0 | False | 2.3861 | 14.886 | 0.916 |
| 22 | one_hot | 8 | 0.10 | 10 | 1.0 | 1.0 | 10.0 | False | 2.3929 | 14.382 | 0.908 |
| … | | | | | | | | | | | |
| 28 | ordinal | 4 | 0.10 | 10 | 0.7 | 1.0 | 10.0 | False | 2.4923 | 15.091 | 0.907 |
| 26 | ordinal | 6 | 0.10 | 10 | 0.7 | 1.0 | 10.0 | True | 2.4961 | 13.995 | 0.899 |
| 24 | one_hot | 4 | 0.03 | 1 | 0.7 | 1.0 | 1.0 | False | 2.5351 | 15.511 | 0.904 |

Tabel lengkap: `dataset/model_ready/xgb_search_results.csv`.

Tiga hal yang terbaca dari sebaran ini:

1. **Ruang parameternya datar, seperti pada RF.** Dari 2.3727 ke 2.5351 hanya
   6,8% rentang — bahkan lebih rapat dari rentang 13% milik Random Forest.
   Pemenang mengalahkan runner-up dengan selisih 0,0001, praktis seri. Menambah
   anggaran pencarian di ruang ini kecil sekali hasilnya.

2. **`encoding` memang memisah, dan `native` yang menang.** Inilah pertanyaan
   yang jadi alasan flag itu ada, dan jawabannya bukan "tidak ada bedanya":

   | encoding | kandidat | pinball terbaik | median |
   |---|---:|---:|---:|
   | `native` | 9 | **2.3727** | **2.3983** |
   | `one_hot` | 7 | 2.3929 | 2.4055 |
   | `ordinal` | 14 | 2.4010 | 2.4286 |

   Empat kandidat teratas semuanya `native`, dan urutan median-nya sejalan
   dengan urutan minimumnya. Artinya penanganan kategori bawaan XGBoost —
   partisi kategori yang benar, bukan angka indeks yang diperlakukan sebagai
   besaran terurut — memang membeli sesuatu di sini.

   Perlu dicatat: pembagian 9/7/14 itu hasil undian acak, bukan desain
   berimbang, jadi perbandingan median di atas indikatif dan bukan eksperimen
   terkontrol. Yang kuat adalah gabungan kedua sinyalnya (minimum *dan* median
   searah), bukan salah satunya sendirian.

3. **`log_target=True` merugikan.** 14 kandidat `False` bermedian 2.4046 lawan
   16 kandidat `True` di 2.4285, dan ke-empat kandidat teratas semuanya
   `False`. Melatih di ruang log lalu mentransformasi balik menggeser kuantil
   yang dioptimalkan — dengan objektif yang sudah pinball, transformasi itu
   justru melawan tujuannya.

Parameter terpilih (`dataset/model_ready/xgb_best_params.json`):

```json
{
  "colsample_bytree": 0.7,
  "encoding": "native",
  "learning_rate": 0.05,
  "log_target": false,
  "max_depth": 6,
  "min_child_weight": 50,
  "random_state": 42,
  "reg_lambda": 10.0,
  "subsample": 0.7
}
```

## 5. Hasil walk-forward

Pemenang dijalankan ulang di kelima fold. Tiga potongan, masing-masing melawan
ketiga baseline pada baris identik — satu angka global menyesatkan di data yang
44% targetnya nol.

### 5.1 Per fold

pinball@0.9 per fold:

| model | 1 (Jul) | 2 (Agu) | 3 (Sep) | 4 (Okt) | 5 (Nov) |
|---|---:|---:|---:|---:|---:|
| **xgboost** | **2.190** | **2.370** | **2.310** | **2.658** | **2.446** |
| naive_roll_mean_7 | 4.249 | 4.566 | 4.034 | 4.783 | 4.970 |
| naive_lag_1 | 8.355 | 8.469 | 8.045 | 8.526 | 8.372 |
| naive_zero | 26.453 | 27.254 | 23.849 | 26.219 | 29.320 |

Detail XGBoost per fold:

| fold | bulan | n | mae | pinball | coverage | fill_rate | shortfall | overstock | ronde |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Jul 2025 | 72.684 | 14.866 | 2.190 | 0.924 | 0.970 | 63.878 | 1.016.646 | 661 |
| 2 | Agu 2025 | 75.015 | 13.998 | 2.370 | 0.906 | 0.960 | 90.964 | 959.111 | 632 |
| 3 | Sep 2025 | 70.503 | 15.020 | 2.310 | 0.918 | 0.962 | 71.245 | 987.711 | 313 |
| 4 | Okt 2025 | 67.716 | 14.190 | 2.658 | 0.898 | 0.947 | 104.916 | 855.945 | 682 |
| 5 | Nov 2025 | 59.629 | 13.306 | 2.446 | 0.894 | 0.957 | 83.169 | 710.261 | 667 |

Performanya stabil, tidak digendong satu bulan: pinball bergerak 2.190–2.658
dan XGBoost menang di kelima fold dengan margin yang mirip. Fold 4 (Oktober)
konsisten paling berat untuk semua model — properti bulannya, bukan properti
modelnya, sama seperti yang terlihat di run RF.

Kolom ronde memperlihatkan early stopping bekerja: 313–682 ronde, tidak ada
yang mendekati plafon 2.000. Fold 3 mendarat jauh lebih cepat (313) dari
tetangganya, jadi jumlah ronde memang bergerak menurut foldnya — persis alasan
kenapa ia tidak dipatok satu angka untuk semua fold.

Coverage turun rapi dari 0.924 di fold 1 ke 0.894 di fold 5. Dua fold terakhir
duduk **sedikit di bawah** target 0.90 (0.898 dan 0.894) — kecil, tapi arahnya
konsisten, dan berlawanan dengan RF yang justru konservatif di seluruh fold.

**Fold 3 dan 5 adalah fold yang memilih pemenang**, jadi skornya di sana bukan
out-of-sample terhadap seleksi (pinball gabungan fold 3+5 = 2.3727, persis sama
dengan skor pencarian — konfigurasi dan seed-nya identik). Dipotong ke fold 1,
2, dan 4 saja — tiga fold yang tidak menyentuh seleksi:

| model | n | mae | pinball | coverage | fill_rate |
|---|---:|---:|---:|---:|---:|
| **xgboost** | 215.415 | 14.351 | **2.400** | 0.910 | 0.959 |
| naive_roll_mean_7 | 215.415 | 9.721 | 4.527 | 0.696 | 0.850 |
| naive_lag_1 | 215.415 | 16.322 | 8.448 | 0.653 | 0.712 |
| naive_zero | 215.415 | 29.620 | 26.658 | 0.423 | 0.000 |

2.400 lawan 2.390 gabungan kelima fold — optimisme seleksinya 0,4%, kecil, yang
masuk akal mengingat ruang parameternya sedatar itu.

### 5.2 Per `demand_segment`

Gabungan kelima fold, dibobot jumlah baris:

| segmen | model | n | mae | pinball | coverage | fill_rate |
|---|---|---:|---:|---:|---:|---:|
| **smooth** | **xgboost** | 45.485 | 54.371 | **8.373** | 0.897 | 0.975 |
| | naive_roll_mean_7 | 45.485 | 37.280 | 16.540 | 0.602 | 0.890 |
| | naive_lag_1 | 45.485 | 65.029 | 32.602 | 0.492 | 0.777 |
| | naive_zero | 45.485 | 146.363 | 131.726 | 0.013 | 0.000 |
| **erratic** | **xgboost** | 54.511 | 25.846 | **4.785** | 0.893 | 0.944 |
| | naive_roll_mean_7 | 54.511 | 18.926 | 9.028 | 0.595 | 0.820 |
| | naive_lag_1 | 54.511 | 32.943 | 16.945 | 0.504 | 0.656 |
| | naive_zero | 54.511 | 49.599 | 44.639 | 0.048 | 0.000 |
| **lumpy** | **xgboost** | 123.545 | 5.907 | **1.060** | 0.907 | 0.884 |
| | naive_roll_mean_7 | 123.545 | 3.537 | 1.873 | 0.676 | 0.625 |
| | naive_lag_1 | 123.545 | 5.316 | 2.914 | 0.640 | 0.412 |
| | naive_zero | 123.545 | 5.070 | 4.563 | 0.434 | 0.000 |
| **intermittent** | **xgboost** | 122.006 | 2.721 | **0.435** | 0.922 | 0.879 |
| | naive_roll_mean_7 | 122.006 | 1.384 | 0.657 | 0.790 | 0.613 |
| | naive_lag_1 | 122.006 | 1.965 | 0.981 | 0.791 | 0.413 |
| | naive_zero | 122.006 | 1.673 | 1.506 | 0.723 | 0.000 |

XGBoost menang pinball di **keempat** segmen, jadi kemenangan globalnya bukan
hasil menang di pasangan yang mayoritas nol — margin terbesarnya justru ada di
`smooth` (8.373 vs 16.540) dan `erratic` (4.785 vs 9.028), dua segmen yang
benar-benar bergerak.

Sama seperti RF, di `intermittent` dan `lumpy` MAE XGBoost **lebih buruk
daripada `naive_zero`** (2.721 vs 1.673; 5.907 vs 5.070). Itu bukan anomali: di
baris validasi, `intermittent` 72% targetnya nol dan `lumpy` 43%, jadi menebak
nol terus memang menghasilkan MAE kecil di sana — dengan konsekuensi fill rate
0 dan coverage 0.723/0.434.

Coverage lintas segmen 0.893–0.922, jauh lebih ketat mengelilingi target 0.90
daripada RF (0.911–0.946). Dua segmen duduk sedikit di bawah target: `erratic`
0.893 dan `smooth` 0.897 — dua segmen bervolume besar, jadi kekurangan stok
yang muncul di sana tidak sekecil selisih coverage-nya.

### 5.3 Per `is_delivery_day`

| hari kirim | model | n | mae | pinball | coverage | fill_rate | shortfall | overstock |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **True** | **xgboost** | 98.701 | 19.589 | **3.277** | 0.908 | 0.965 | 162.595 | 1.770.881 |
| | naive_roll_mean_7 | 98.701 | 13.286 | 6.869 | 0.634 | 0.853 | 683.526 | 627.820 |
| | naive_lag_1 | 98.701 | 23.757 | 14.529 | 0.548 | 0.675 | 1.499.452 | 845.412 |
| | naive_zero | 98.701 | 46.917 | 42.225 | 0.330 | 0.000 | 4.630.772 | 0 |
| **False** | **xgboost** | 246.846 | 12.195 | **2.035** | 0.909 | 0.955 | 251.577 | 2.758.793 |
| | naive_roll_mean_7 | 246.846 | 8.191 | 3.557 | 0.717 | 0.848 | 844.867 | 1.176.969 |
| | naive_lag_1 | 246.846 | 13.390 | 5.883 | 0.694 | 0.747 | 1.402.090 | 1.903.062 |
| | naive_zero | 246.846 | 22.527 | 20.274 | 0.455 | 0.000 | 5.560.708 | 0 |

Di hari kirim — baris yang benar-benar menaikkan barang ke truk — XGBoost
menjaga coverage 0.908 dan fill rate 0.965, dengan shortfall 162.595 unit lawan
683.526 milik baseline terbaik. Margin pinball di sini (3.277 vs 6.869, 52%
lebih baik) lebih lebar daripada di hari non-kirim (2.035 vs 3.557, 43%), jadi
keunggulannya terkonsentrasi persis di baris yang paling penting — pola yang
sama dengan RF.

Bedanya dengan RF ada di kalibrasi: coverage XGBoost praktis sama di kedua sisi
(0.908 dan 0.909), sementara RF 0.928 dan 0.934. XGBoost menukar sekitar 23.000
unit kekurangan stok tambahan di hari kirim dengan 214.000 unit kelebihan stok
yang tidak perlu ditahan.

## 6. Head-to-head lawan Random Forest

Sah dilakukan karena kedua model dinilai di **baris yang identik** — dijamin
`walk_forward.eligible_rows()`, bukan oleh disiplin. Ketiga baseline mencetak
angka yang sama persis di kedua run, yang mengonfirmasi itu.

| potongan | model | pinball | mae | coverage |
|---|---|---:|---:|---:|
| semua fold | **xgboost** | **2.390** | 14.307 | 0.909 |
| | random_forest | 2.410 | 15.640 | 0.932 |
| fold 1/2/4 (bersih) | **xgboost** | **2.400** | 14.351 | 0.910 |
| | random_forest | 2.403 | 15.593 | 0.934 |

Per fold: XGBoost menang di fold 3, 4, 5; RF menang di fold 2; fold 1 praktis
seri (2.1897 lawan 2.1904). Dua fold yang dimenangi XGBoost dengan margin
terlebar — 3 dan 5 — adalah persis fold yang memilih pemenang untuk **kedua**
model.

**Di potongan bersih fold 1/2/4, selisihnya 2.400 lawan 2.403: 0,1%.** Itu di
bawah level yang bisa dibedakan dari kebisingan, dan dua asimetri di bawah ini
adalah penjelasan tandingan yang sama hidupnya dengan "XGBoost memang lebih
baik":

1. **Anggaran pencarian 30 kandidat lawan 18.** XGBoost dapat 67% lebih banyak
   undian dari ruangnya karena fitnya 2,7× lebih murah (bagian 3). Sebagian dari
   selisih 0,1% itu bisa saja cuma anggaran pencarian yang lebih besar.
2. **Early stopping lawan jumlah pohon yang dipatok.** Jumlah ronde XGBoost
   dipilih per fold dari data (313–682), sementara RF memakai `n_estimators`
   tetap 200 saat pencarian dan 400 saat fit final. XGBoost menyetel satu
   dimensi regularisasi yang tidak disetel RF.

Yang **tidak** setipis itu adalah dua perbedaan lain, dan keduanya berpihak
pada XGBoost:

- **Kalibrasi.** Coverage 0.909 lawan 0.932 terhadap target 0.90. RF menembak
  3,2 poin di atas sasaran; XGBoost 0,9 poin. Kalau service level 0.90 memang
  yang diminta, XGBoost mengantar apa yang diminta, bukan lebih.
- **Ukuran model.** 4,7 MB lawan 821 MB — 175× lebih kecil, karena
  `quantile-forest` menyimpan nilai target di tiap daun sementara booster hanya
  menyimpan struktur pohon. Ini penting untuk deployment, bukan untuk akurasi.

Di level segmen keduanya membagi kemenangan: XGBoost unggul di `smooth` (8.373
vs 8.570) dan `erratic` (4.785 vs 4.810), RF unggul di `intermittent` (0.435 vs
0.426) dan `lumpy` (1.060 vs 1.043). Polanya masuk akal — XGBoost menang di
segmen bervolume yang bergerak, RF menang di segmen yang didominasi nol, tempat
kuantil empiris dari daun memang alat yang tepat.

Kesimpulan yang bisa ditopang angka ini: **XGBoost setidaknya setara dengan
Random Forest pada kriteria pinball@0.9, dan lebih baik pada kalibrasi serta
ukuran model.** Menyatakan ia lebih akurat, dengan selisih 0,1% di potongan
bersih dan dua asimetri anggaran yang belum dinetralkan, belum bisa ditopang.

## 7. Model final

`xgb.fit_final()` menjalankan protokol dua fit yang sama pada seluruh baris
layak sebelum Desember, dipotong di batas Desember oleh
`purging.lookahead_safe_mask()` — populasi baris yang sama persis dengan yang
dinilai di atas.

| | |
|---|---|
| Baris training | 1.349.011 |
| Kolom | 56 (`encoding="native"`, tanpa ekspansi one-hot) |
| `best_iteration` | 607 |
| Kuantil | 0.9 |
| File | `models/xgboost_q90.joblib` — 4,7 MB, dibuat 19 Agu 2026 |

Bundle-nya menyimpan urutan kolom training, flag `encoding`/`log_target`, dan
**level kategori** yang dipakai saat training. Yang terakhir itu wajib untuk
mode `native`: booster yang dimuat ulang bulan depan terhadap kategori yang
diurutkan berbeda tidak gagal — ia memprediksi dengan percaya diri dari fitur
yang salah.

## 8. Reproduksi

```bash
.venv/bin/python3 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebook/modeling_xgb.ipynb
```

Butuh berjam-jam. Pencarian menulis checkpoint tiap kandidat selesai ke
`xgb_search_results.csv` dan melanjutkan dari sana kalau dijalankan ulang —
run yang menghasilkan dokumen ini memang dilanjutkan dari checkpoint 20
kandidat.

XGBoost butuh runtime OpenMP yang tidak bisa dipasang pip: `brew install
libomp` sekali di macOS (Linux: `libgomp`, biasanya sudah ada).

| Artefak | Lokasi | Masuk git |
|---|---|---|
| Hasil pencarian | `dataset/model_ready/xgb_search_results.csv` | tidak |
| Parameter terpilih | `dataset/model_ready/xgb_best_params.json` | tidak |
| Tabel hasil lengkap | `dataset/model_ready/xgb_walk_forward_results.csv` | tidak |
| Booster terlatih | `models/xgboost_q90.joblib` | tidak |
| Ringkasan ini | `docs/hasil-modeling-xgb.md` | **ya** |

## 9. Batasan

- **Desember 2025 belum dibuka.** Semua angka di sini adalah validasi
  walk-forward, bukan skor test set final.
- **Sumbu waktunya waktu pengambilan, bukan waktu pemesanan.** Model ini
  meramal permintaan terealisasi pada tanggal pickup; sebagian permintaan hari
  depan sudah diketahui kantor pusat lewat pre-order yang tidak tercatat di
  dataset mana pun. Lihat `docs/batasan-penelitian.md` (B-1, B-2, B-3).
- **MAE tidak sebanding lintas model di sini.** Membandingkan MAE model kuantil
  0.9 dengan baseline titik-tengah menghukum yang pertama karena melakukan
  persis apa yang diminta. Pinball@0.9 adalah kriterianya.
- **Fold 3 dan 5 ikut memilih pemenang**, jadi skornya di potongan per-fold
  bukan out-of-sample terhadap seleksi. Potongan fold 1/2/4 di bagian 5.1 adalah
  angka yang bersih.
- **Perbandingan lawan RF belum dinetralkan anggarannya.** 30 kandidat lawan
  18, dan early stopping lawan jumlah pohon yang dipatok (bagian 6). Menyamakan
  keduanya butuh run ulang, bukan pembacaan ulang tabel yang sama.
- **Coverage sedikit di bawah target di ekor periode.** Fold 4 dan 5 mencetak
  0.898 dan 0.894 terhadap target 0.90, begitu juga segmen `erratic` (0.893)
  dan `smooth` (0.897). Selisihnya kecil, tapi arahnya konsisten — layak
  diperiksa lagi saat test set Desember dibuka.
- **Dua dari tiga model.** LSTM yang direncanakan di `docs/pipeline-overview.md`
  belum dijalankan, jadi rekomendasi model final belum bisa diambil.
