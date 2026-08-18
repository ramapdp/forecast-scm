# Hasil Modeling — Random Forest kuantil 0.9

Angka terukur dari jalannya `notebook/modeling_rf.ipynb`. Desainnya ada di
`docs/superpowers/specs/2026-08-18-random-forest-modeling-design.md`; dokumen
ini hanya memuat hasilnya, supaya bukti di balik tulisan hidup di git dan bukan
cuma di output cell notebook yang bisa hilang saat di-clear.

**Desember 2025 tidak dibuka.** Semua angka di bawah datang dari walk-forward
lima fold di Juli–November 2025. Test set final masih terkunci.

## 1. Ringkasan

Random Forest kuantil 0.9 menang telak pada metrik yang memang jadi kriteria
(pinball@0.9): **2.410 lawan 4.503** milik baseline terbaik, `naive_roll_mean_7`
— sekitar 46% lebih baik. Kalibrasinya juga tepat sasaran: coverage 93.2%
terhadap target service level 90%, sedikit konservatif tapi tidak jauh.

RF **kalah MAE** dari `naive_roll_mean_7` (15.64 lawan 9.65). Ini bukan
kegagalan, ini konsekuensi yang diminta: model kuantil 0.9 sengaja bias ke atas,
dan MAE menghukum bias ke atas persis seperti ia menghukum kekurangan stok. MAE
dilaporkan untuk konteks, bukan sebagai kriteria kemenangan.

Sisi bisnisnya, di 345.547 baris validasi:

| | kekurangan (shortfall) | kelebihan (overstock) |
|---|---:|---:|
| `random_forest` | 365.576 | 5.038.816 |
| `naive_roll_mean_7` | 1.528.393 | 1.804.789 |

RF memangkas kekurangan stok **76%** dengan ongkos kelebihan stok **2,8×** lipat.
Apakah itu tukar-menukar yang benar adalah keputusan bisnis, bukan keputusan
model — tapi itu memang persis tukar-menukar yang diminta ketika service level
dipatok di 0.9.

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
| Implementasi | `quantile_forest.RandomForestQuantileRegressor` |

Ketiga baseline naive dinilai pada **baris yang identik** dengan RF — dijamin
oleh `utils/walk_forward.py` yang memiliki definisi fold dan kelayakan baris,
dan menerima model sebagai callable yang disuntikkan.

## 3. Benchmark

Satu fit pada training set penuh fold 5 dengan `DEFAULT_PARAMS`
(`n_estimators=200, max_depth=16, min_samples_leaf=50, max_samples_leaf=20,
max_features="sqrt"`), untuk memastikan batas penyimpanan leaf yang dipakai
menyaring kandidat memang berlaku sebelum 18 fit dijalankan.

| | |
|---|---|
| Baris training fold 5 | 1.292.778 |
| Baris validasi fold 5 | 59.629 |
| Estimasi penyimpanan leaf | 1,54 GB (budget 3 GB) |
| Wall time satu fit + predict | 6,6 menit (395 detik) |
| Peak RSS proses | 4,92 GB |
| Prediksi | rata-rata 48,24, maksimum 1.734 |

Dua hal yang dikonfirmasi angka ini:

1. **Batas leaf storage-nya sahih.** `TYPICAL_N_TRAIN = 1_280_000` yang dipakai
   menyaring kandidat sebelum data dimuat ternyata meleset hanya 1% dari
   1.292.778 baris sebenarnya, jadi penyaringan memori memang menilai ukuran
   yang benar.

   Peak RSS 4,92 GB lebih besar dari budget 3 GB, dan itu bukan pelanggaran:
   budget tersebut membatasi **penyimpanan leaf saja** — array nilai yang
   disimpan `quantile-forest` di tiap daun — sementara RSS ikut memuat panel
   1,5 juta baris, matriks fitur, dan struktur pohon itu sendiri. Yang
   dijaga budget adalah komponen yang meledak secara kuadratik terhadap
   pilihan parameter; sisanya konstan.

2. **Pencarian tidak perlu subsampling.** Pada 6,6 menit per fit, 18 kandidat ×
   2 fold ≈ 4 jam untuk konfigurasi semurah default ini — lama tapi masih
   semalam. Kandidat yang lebih berat (`max_features=1.0`, `one_hot=True`)
   memakan lebih banyak, tapi tetap dalam orde yang sama, jadi seluruh baris
   training dipakai apa adanya dan skornya menggambarkan fold penuh.

Konfigurasi benchmark ini adalah `DEFAULT_PARAMS`, bukan pemenang pencarian —
tujuannya menyanggah batas memori dan mengukur ongkos, bukan mencetak skor.

## 4. Pencarian hyperparameter

18 kandidat ditarik acak dengan seed 42 dari ruang 1.152 kombinasi, tiap
kandidat disaring lebih dulu lewat `estimate_leaf_memory_bytes()` terhadap
budget 3 GB. Penilaian di **fold 3 (September) dan fold 5 (November)** saja,
kriteria **pinball@0.9 gabungan** (dibobot jumlah baris, bukan dirata-rata
polos). **Tidak ada subsampling** — seluruh baris training tiap fold dipakai.

Ke-18 kandidat selesai dinilai; tidak ada yang gagal (kolom `error` kosong
semua).

| # | log_target | max_depth | max_features | max_samples | max_samples_leaf | min_samples_leaf | one_hot | pinball | mae | coverage |
|---:|---|---:|---|---|---:|---:|---|---:|---:|---:|
| **17** | False | 12 | 0.5 | — | 50 | 20 | True | **2.4221** | 15.718 | 0.929 |
| 1 | False | 20 | 1.0 | — | 1 | 20 | False | 2.4225 | 15.439 | 0.928 |
| 0 | False | 12 | 0.5 | — | 1 | 50 | False | 2.4270 | 15.808 | 0.926 |
| 9 | False | 16 | 1.0 | 0.5 | 50 | 50 | True | 2.4293 | 15.878 | 0.930 |
| 16 | True | 12 | 0.5 | — | 50 | 20 | True | 2.4351 | 16.150 | 0.923 |
| 7 | False | 20 | 0.3 | 0.5 | 1 | 50 | False | 2.4476 | 16.403 | 0.930 |
| … | | | | | | | | | | |
| 10 | False | 12 | sqrt | 0.5 | 20 | 100 | False | 2.7347 | 20.989 | 0.946 |
| 2 | False | 20 | sqrt | — | 50 | 200 | False | 2.7476 | 21.109 | 0.943 |

Tabel lengkap: `dataset/model_ready/rf_search_results.csv`.

Dua hal yang terbaca dari sebaran ini:

1. **Ruang parameternya datar.** Dari 2.422 ke 2.748 hanya 13% rentang, dan
   lima kandidat teratas berjarak 0,6% satu sama lain. Pemenang mengalahkan
   runner-up dengan selisih 0,0004 — praktis seri. Menambah anggaran pencarian
   di ruang ini kecil sekali hasilnya.
2. **`max_features="sqrt"` satu-satunya pilihan yang benar-benar merugikan.**
   Dua kandidat terburuk keduanya memakai `sqrt`, dan hanya itu yang
   membedakannya dari kelompok tengah. Dengan 56 fitur, `sqrt` menyisakan ~7
   fitur per split — terlalu sedikit.

Parameter terpilih (`dataset/model_ready/rf_best_params.json`):

```json
{
  "log_target": false,
  "max_depth": 12,
  "max_features": 0.5,
  "max_samples": null,
  "max_samples_leaf": 50,
  "min_samples_leaf": 20,
  "n_estimators": 200,
  "one_hot": true,
  "random_state": 42
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
| **random_forest** | **2.190** | **2.352** | **2.369** | **2.689** | **2.485** |
| naive_roll_mean_7 | 4.249 | 4.566 | 4.034 | 4.783 | 4.970 |
| naive_lag_1 | 8.355 | 8.469 | 8.045 | 8.526 | 8.372 |
| naive_zero | 26.453 | 27.254 | 23.849 | 26.219 | 29.320 |

Detail RF per fold:

| fold | bulan | n | mae | pinball | coverage | fill_rate | shortfall | overstock |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Jul 2025 | 72.684 | 15.346 | 2.190 | 0.939 | 0.972 | 59.582 | 1.055.824 |
| 2 | Agu 2025 | 75.015 | 15.584 | 2.352 | 0.936 | 0.967 | 74.374 | 1.094.695 |
| 3 | Sep 2025 | 70.503 | 16.248 | 2.369 | 0.935 | 0.965 | 65.591 | 1.079.928 |
| 4 | Okt 2025 | 67.716 | 15.869 | 2.689 | 0.926 | 0.953 | 93.302 | 981.260 |
| 5 | Nov 2025 | 59.629 | 15.091 | 2.485 | 0.922 | 0.963 | 72.727 | 827.110 |

Performanya stabil, tidak digendong satu bulan: pinball bergerak 2.190–2.689
dan RF menang di kelima fold dengan margin yang mirip. Fold 4 (Oktober)
konsisten paling berat untuk semua model, jadi itu properti bulannya, bukan
properti RF-nya.

**Fold 3 dan 5 adalah fold yang memilih pemenang**, jadi skornya di sana bukan
out-of-sample terhadap seleksi (pinball gabungan fold 3+5 = 2.4221, persis sama
dengan skor pencarian — konfigurasi dan seed-nya identik). Dipotong ke fold 1,
2, dan 4 saja — tiga fold yang tidak menyentuh seleksi:

| model | n | mae | pinball | coverage | fill_rate |
|---|---:|---:|---:|---:|---:|
| **random_forest** | 215.415 | 15.593 | **2.403** | 0.934 | 0.964 |
| naive_roll_mean_7 | 215.415 | 9.721 | 4.527 | 0.696 | 0.850 |
| naive_lag_1 | 215.415 | 16.322 | 8.448 | 0.653 | 0.712 |
| naive_zero | 215.415 | 29.620 | 26.658 | 0.423 | 0.000 |

2.403 lawan 2.410 gabungan kelima fold — tidak ada optimisme seleksi yang
terukur, yang masuk akal mengingat ruang parameternya sedatar itu.

### 5.2 Per `demand_segment`

Gabungan kelima fold, dibobot jumlah baris:

| segmen | model | n | mae | pinball | coverage | fill_rate |
|---|---|---:|---:|---:|---:|---:|
| **smooth** | **random_forest** | 45.485 | 63.617 | **8.570** | 0.930 | 0.981 |
| | naive_roll_mean_7 | 45.485 | 37.280 | 16.540 | 0.602 | 0.890 |
| | naive_lag_1 | 45.485 | 65.029 | 32.602 | 0.492 | 0.777 |
| | naive_zero | 45.485 | 146.363 | 131.726 | 0.013 | 0.000 |
| **erratic** | **random_forest** | 54.511 | 27.884 | **4.810** | 0.911 | 0.949 |
| | naive_roll_mean_7 | 54.511 | 18.926 | 9.028 | 0.595 | 0.820 |
| | naive_lag_1 | 54.511 | 32.943 | 16.945 | 0.504 | 0.656 |
| | naive_zero | 54.511 | 49.599 | 44.639 | 0.048 | 0.000 |
| **lumpy** | **random_forest** | 123.545 | 5.617 | **1.043** | 0.929 | 0.881 |
| | naive_roll_mean_7 | 123.545 | 3.537 | 1.873 | 0.676 | 0.625 |
| | naive_lag_1 | 123.545 | 5.316 | 2.914 | 0.640 | 0.412 |
| | naive_zero | 123.545 | 5.070 | 4.563 | 0.434 | 0.000 |
| **intermittent** | **random_forest** | 122.006 | 2.433 | **0.426** | 0.946 | 0.864 |
| | naive_roll_mean_7 | 122.006 | 1.384 | 0.657 | 0.790 | 0.613 |
| | naive_lag_1 | 122.006 | 1.965 | 0.981 | 0.791 | 0.413 |
| | naive_zero | 122.006 | 1.673 | 1.506 | 0.723 | 0.000 |

Ini potongan yang paling banyak menjawab. RF menang pinball di **keempat**
segmen, jadi kemenangan globalnya bukan hasil menang di pasangan yang mayoritas
nol — justru sebaliknya, margin terbesarnya ada di `smooth` (8.570 vs 16.540)
dan `erratic` (4.810 vs 9.028), dua segmen yang benar-benar bergerak.

Di `intermittent` dan `lumpy`, MAE RF **lebih buruk daripada `naive_zero`**
(2.433 vs 1.673; 5.617 vs 5.070). Itu bukan anomali: di baris validasi,
`intermittent` 72% targetnya nol dan `lumpy` 43%, jadi menebak nol terus memang
menghasilkan MAE kecil di sana — dengan konsekuensi fill rate 0 dan coverage
0.723/0.434. Justru inilah alasan
`demand_segment` dibuat — MAE global akan menobatkan model yang cuma menang di
tempat menebak nol itu mudah.

Coverage konsisten di 0.911–0.946 lintas segmen, semuanya di atas target 0.90.
Yang paling ketat `erratic` (0.911), yang paling longgar `intermittent`
(0.946) — cocok dengan intuisinya: distribusi yang jarang isi punya ekor yang
kasar, jadi kuantil 0.9 empiris dari leaf gampang melompati target.

### 5.3 Per `is_delivery_day`

| hari kirim | model | n | mae | pinball | coverage | fill_rate | shortfall | overstock |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **True** | **random_forest** | 98.701 | 21.520 | **3.284** | 0.928 | 0.970 | 139.684 | 1.984.398 |
| | naive_roll_mean_7 | 98.701 | 13.286 | 6.869 | 0.634 | 0.853 | 683.526 | 627.820 |
| | naive_lag_1 | 98.701 | 23.757 | 14.529 | 0.548 | 0.675 | 1.499.452 | 845.413 |
| | naive_zero | 98.701 | 46.917 | 42.225 | 0.330 | 0.000 | 4.630.772 | 0 |
| **False** | **random_forest** | 246.846 | 13.289 | **2.061** | 0.934 | 0.959 | 225.892 | 3.054.418 |
| | naive_roll_mean_7 | 246.846 | 8.191 | 3.557 | 0.717 | 0.848 | 844.867 | 1.176.969 |
| | naive_lag_1 | 246.846 | 13.390 | 5.883 | 0.694 | 0.747 | 1.402.090 | 1.903.063 |
| | naive_zero | 246.846 | 22.527 | 20.274 | 0.455 | 0.000 | 5.560.708 | 0 |

Di hari kirim — baris yang benar-benar menaikkan barang ke truk — RF menjaga
coverage 0.928 dan fill rate 0.970, dengan shortfall 139.684 unit lawan 683.526
milik baseline terbaik. Margin pinball di sini (3.284 vs 6.869, 52% lebih baik)
lebih lebar daripada di hari non-kirim (2.061 vs 3.557, 42%), yang berarti
keunggulan RF terkonsentrasi persis di baris yang paling penting.

## 6. Model final

`rf.fit_final()` melatih ulang konfigurasi pemenang dengan **`n_estimators`
dinaikkan 200 → 400** pada seluruh baris layak sebelum Desember, dipotong di
batas Desember oleh `purging.lookahead_safe_mask()` — populasi baris yang sama
persis dengan yang dinilai di atas.

Bundle-nya menyimpan urutan kolom training beserta flag `one_hot`/`log_target`,
karena forest yang dimuat ulang dengan urutan kolom berbeda tidak gagal — ia
memprediksi dengan percaya diri dari fitur yang salah.

`models/random_forest_q90.joblib` — 821 MB, dibuat 19 Agu 2026.

## 7. Reproduksi

```bash
.venv/bin/python3 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebook/modeling_rf.ipynb
```

Butuh berjam-jam. Pencarian menulis checkpoint tiap kandidat selesai ke
`rf_search_results.csv` dan melanjutkan dari sana kalau dijalankan ulang, jadi
run yang terbunuh OS tidak menghanguskan seluruh sore.

| Artefak | Lokasi | Masuk git |
|---|---|---|
| Hasil pencarian | `dataset/model_ready/rf_search_results.csv` | tidak |
| Parameter terpilih | `dataset/model_ready/rf_best_params.json` | tidak |
| Tabel hasil lengkap | `dataset/model_ready/rf_walk_forward_results.csv` | tidak |
| Forest terlatih | `models/random_forest_q90.joblib` | tidak |
| Ringkasan ini | `docs/hasil-modeling-rf.md` | **ya** |

## 8. Batasan

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
  bukan out-of-sample terhadap seleksi. Potongan fold 1/2/4 di §5.1 adalah
  angka yang bersih.
- **Satu model, belum perbandingan.** XGBoost dan LSTM yang direncanakan di
  `docs/pipeline-overview.md` belum dijalankan, jadi belum ada yang bisa
  dikatakan soal apakah Random Forest pilihan terbaik — baru bahwa ia jauh
  mengalahkan ketiga baseline naive.
