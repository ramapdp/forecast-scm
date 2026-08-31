# Hasil Modeling — Random Forest multi-kuantil

Angka terukur dari jalannya `notebook/modeling_rf.ipynb` pada **25 Agustus
2026**, run pertama di bawah kriteria multi-kuantil (butir 0c
`docs/todolist-proyek.md`). Desainnya ada di
`docs/superpowers/specs/2026-08-18-random-forest-modeling-design.md` dan
`docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`; dokumen
ini hanya memuat hasilnya, supaya bukti di balik tulisan hidup di git dan bukan
cuma di output cell notebook yang bisa hilang saat di-clear.

**Dokumen ini menggantikan versi kuantil-tunggal**, yang diarsipkan sebagai
`docs/hasil-modeling-rf.single-quantile.bak.md`. Angka di kedua dokumen
**bukan besaran yang sama** dan tidak boleh disandingkan sebagai perbandingan
langsung: yang lama pinball@0,9 pada satu titik kuantil, yang ini K1 —
rata-rata pinball lintas 19 titik `QUANTILE_SET_A`. Larangan itu tercatat
sebagai T-10 di `docs/todolist-proyek.md`. Satu hal yang **sah** dibawa lintas
dokumen adalah *peringkat* kandidat pencarian; itu dibahas di bagian 4.3.

**Desember 2025 tidak dibuka.** Semua angka di bawah datang dari walk-forward
lima fold di Juli–November 2025. Test set final masih terkunci.

## 1. Ringkasan

Random Forest melewati gerbang G0 dan mencetak **K1 = 2,8508** pada potongan
fold bersih (1/2/4), lawan **4,8603** milik baseline terbaik
`naive_roll_mean_7` — sekitar **41% lebih baik**. Pada gabungan kelima fold
angkanya 2,8621 lawan 4,8231.

| model | K1 (fold 1/2/4) | K1 (5 fold) |
|---|---:|---:|
| **random_forest** | **2,8508** | **2,8621** |
| `naive_roll_mean_7` | 4,8603 | 4,8231 |
| `naive_lag_1` | 8,1612 | 8,1755 |
| `naive_zero` | 14,8102 | 14,7469 |

`crossing_rate = 0,0` di **seluruh** baris hasil — pencarian, benchmark, dan
walk-forward. Ini bukan kabar baik yang mengejutkan melainkan cek struktural
yang lolos: setiap titik kuantil forest adalah persentil dari satu distribusi
empiris daun yang sama, jadi inversi mustahil secara konstruksi. Nilai bukan-nol
di kolom ini akan berarti ada bug, bukan ada kelemahan model
(kriteria K2 `docs/detail-tahap-perbandingan-model.md` bagian 1.7).

Di τ=0,9 — titik yang benar-benar dijanjikan ke bisnis (B-9) — coverage 0,928
terhadap target 0,90, dengan fill rate 0,959. Sisi bisnisnya, di 345.547 baris
validasi:

| | kekurangan (shortfall) | kelebihan (overstock) |
|---|---:|---:|
| `random_forest` | 418.250 | 4.793.038 |
| `naive_roll_mean_7` | 1.528.393 | 1.804.789 |

RF memangkas kekurangan stok **73%** dengan ongkos kelebihan stok **2,7×**
lipat. Apakah itu tukar-menukar yang benar adalah keputusan bisnis, bukan
keputusan model — tapi itu memang persis tukar-menukar yang diminta ketika
service level dipatok di 0,9.

> Catatan satuan: `shortfall_units` dan `overstock_units` menjumlahkan unit
> lintas SKU yang satuannya campur (Kg, Porsi, Botol, PCS, …), jadi angkanya
> sah untuk membandingkan antar model pada baris yang sama, tapi tidak punya
> makna fisik sebagai satu besaran tunggal.

**Satu temuan yang tidak boleh dilewatkan** dan dibahas penuh di bagian 5.2: coverage
RF berada **di atas** targetnya di seluruh 19 titik kuantil, dengan simpangan
+0,381 di τ=0,05 yang mengecil monoton ke +0,011 di τ=0,95. Dibaca mentah lewat
tabel pola K2, "simpangan searah di hampir seluruh τ" adalah alasan kuat untuk
tersisih. Dibaca dengan datanya, sebagian besar simpangan di ujung bawah
**dipaksa oleh bentuk target** dan akan muncul pada model apa pun di dataset ini.
Bagian mana yang dipaksa dan bagian mana yang benar-benar bias model dipisahkan
secara kuantitatif di bagian 5.2.

## 2. Setup evaluasi

| | |
|---|---|
| Data | `dataset/model_ready/model_input.parquet` — 1.502.522 baris panel, 82 kolom, 1 Jan 2024 – 31 Des 2025 |
| Fitur | 56 kolom (`modeling_prep.FEATURE_COLS`) |
| Target | `target_lead_time_cumulative` — 44,35% bernilai nol, 99,55% bilangan bulat, 70,3% bernilai ≤ 5 |
| Fold | 5 expanding window, validasi Juli, Agustus, September, Oktober, November 2025 |
| Test terkunci | Desember 2025 (`TEST_START = 2025-12-01`) |
| Baris tereliminasi | 28 hari awal tiap `segment_id` (jendela lag belum penuh) dan baris tanpa target |
| Baris validasi | 345.547 total; 41,95% targetnya nol |
| Kuantil | `QUANTILE_SET_A` — 19 titik, 0,05 sampai 0,95 langkah 0,05 |
| Kriteria | K1 = rata-rata tak berbobot pinball lintas 19 titik, pada potongan fold 1/2/4 |
| Implementasi | `quantile_forest.RandomForestQuantileRegressor`, `device=cpu`, commit `5325b55` |

Ketiga baseline naive dinilai pada **baris yang identik** dengan RF — dijamin
oleh `utils/modelling/walk_forward.py` yang memiliki definisi fold dan kelayakan
baris, dan menerima model sebagai callable yang disuntikkan.

Perbedaan tunggal terhadap run sebelumnya adalah **jumlah titik kuantil yang
diminta dari forest yang sama**. Hyperparameter membentuk *daun*, dan seluruh
19 titik dibaca dari daun yang sama, jadi ongkos multi-kuantil di RF praktis
hanya ongkos memanggil `predict()` dengan 19 nilai τ — bukan 19 kali latih.
Itu yang membuat RF selesai dalam hitungan jam sementara XGBoost, yang
membangun satu pohon per τ per ronde boosting, diperkirakan makan ~125 jam.

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
| Wall time satu fit + predict 19 titik | 9,7 menit |
| Peak RSS proses | 4,14 GB |
| Bentuk prediksi | (59.629, 19) — baris × titik kuantil |
| Prediksi di τ=0,9 | rata-rata 47,66, maksimum 1.772,10 |
| `crossing_rate` | 0,0000 |

Tiga hal yang dikonfirmasi angka ini:

1. **Batas leaf storage-nya sahih.** `TYPICAL_N_TRAIN = 1_280_000` yang dipakai
   menyaring kandidat sebelum data dimuat meleset hanya 1% dari 1.292.778 baris
   sebenarnya, jadi penyaringan memori memang menilai ukuran yang benar.

   Peak RSS 4,14 GB lebih besar dari budget 3 GB, dan itu bukan pelanggaran:
   budget tersebut membatasi **penyimpanan leaf saja** — array nilai yang
   disimpan `quantile-forest` di tiap daun — sementara RSS ikut memuat panel
   1,5 juta baris, matriks fitur, dan struktur pohon itu sendiri.

2. **Ongkos 19 titik kuantil terukur, bukan diperkirakan.** Konfigurasi yang
   sama memakan 6,6 menit di run kuantil-tunggal dan 9,7 menit di sini —
   pengganda **×1,47** untuk 19× lebih banyak titik keluaran. Ini menegaskan
   bahwa yang mahal adalah membangun pohon, bukan membaca persentil dari daun
   yang sudah jadi. (Bandingkan XGBoost: ×15,2.)

3. **Pencarian tidak perlu subsampling.** Pada ongkos ini, 18 kandidat × 2 fold
   selesai dalam satu sore, jadi seluruh baris training dipakai apa adanya dan
   skornya menggambarkan fold penuh.

Konfigurasi benchmark ini adalah `DEFAULT_PARAMS`, bukan pemenang pencarian —
tujuannya menyanggah batas memori dan mengukur ongkos, bukan mencetak skor.

## 4. Pencarian hyperparameter

18 kandidat ditarik acak dengan seed 42 dari ruang 1.152 kombinasi
(`3 × 4 × 3 × 4 × 2 × 2 × 2`), tiap kandidat disaring lebih dulu lewat
`estimate_leaf_memory_bytes()` terhadap budget 3 GB. Penilaian di **fold 3
(September) dan fold 5 (November)** saja, kriteria **K1 gabungan** — pinball
tiap titik kuantil dibobot jumlah baris, lalu ke-19 angka itu dirata-ratakan
tak berbobot. **Tidak ada subsampling.**

`n_estimators` sengaja tidak ikut dicari: kualitas forest monoton terhadap
jumlah pohon, jadi mencarinya membelanjakan anggaran untuk pertanyaan yang
jawabannya sudah diketahui. Ia dipatok 200 selama pencarian dan dinaikkan untuk
fit final (bagian 6).

Ke-18 kandidat selesai dinilai; tidak ada yang gagal (kolom `error` kosong
semua).

### 4.1 Tabel lengkap

| # | log_target | max_depth | max_features | max_samples | max_samples_leaf | min_samples_leaf | one_hot | K1 | mae@0,9 | cov@0,9 | detik |
|---:|---|---:|---|---|---:|---:|---|---:|---:|---:|---:|
| **1** | False | 20 | 1.0 | — | 1 | 20 | False | **2,8808** | 15,124 | 0,926 | 1.234 |
| 17 | False | 12 | 0.5 | — | 50 | 20 | True | 2,8984 | 15,440 | 0,926 | 1.029 |
| 9 | False | 16 | 1.0 | 0,5 | 50 | 50 | True | 2,9108 | 15,538 | 0,927 | 1.118 |
| 16 | True | 12 | 0.5 | — | 50 | 20 | True | 2,9165 | 15,973 | 0,921 | 1.014 |
| 0 | False | 12 | 0.5 | — | 1 | 50 | False | 2,9228 | 15,608 | 0,924 | 602 |
| 7 | False | 20 | 0.3 | 0,5 | 1 | 50 | False | 2,9232 | 16,094 | 0,928 | 276 |
| 6 | True | 12 | 0.5 | 0,5 | 50 | 50 | False | 2,9320 | 16,258 | 0,922 | 559 |
| 4 | True | 12 | 0.3 | 0,5 | 1 | 20 | True | 2,9404 | 16,493 | 0,922 | 324 |
| 12 | True | 12 | 0.5 | — | 1 | 100 | True | 2,9560 | 16,310 | 0,920 | 745 |
| 8 | True | 16 | 1.0 | 0,5 | 1 | 100 | True | 2,9764 | 16,149 | 0,919 | 791 |
| 15 | True | 20 | 0.5 | — | 20 | 200 | False | 2,9785 | 16,674 | 0,923 | 889 |
| 11 | False | 20 | 0.5 | — | 50 | 200 | True | 2,9832 | 16,525 | 0,930 | 1.540 |
| 3 | True | 20 | 0.5 | — | 1 | 200 | True | 2,9940 | 16,564 | 0,920 | 902 |
| 13 | True | 16 | 0.3 | — | 50 | 200 | False | 3,0003 | 17,377 | 0,926 | 724 |
| 14 | False | 12 | 0.3 | — | 50 | 200 | False | 3,0160 | 17,474 | 0,933 | 701 |
| 5 | False | 16 | 0.5 | 0,5 | 1 | 200 | False | 3,0509 | 16,855 | 0,927 | 324 |
| 10 | False | 12 | sqrt | 0,5 | 20 | 100 | False | 3,1838 | 20,682 | 0,944 | 444 |
| 2 | False | 20 | sqrt | — | 50 | 200 | False | 3,1969 | 20,757 | 0,941 | 628 |

Sumber: `dataset/model_ready/rf_search_results.csv`.

### 4.2 Yang terbaca dari sebarannya

1. **Ruang parameternya datar.** Dari 2,8808 ke 3,1969 hanya rentang 11%, dan
   lima kandidat teratas berjarak 1,46% satu sama lain. Menambah anggaran
   pencarian di ruang ini kecil sekali hasilnya — kesimpulan yang **tidak
   berubah** dari run sebelumnya.
2. **`max_features="sqrt"` satu-satunya pilihan yang benar-benar merugikan.**
   Dua kandidat terburuk keduanya memakai `sqrt`, dan hanya itu yang
   membedakannya dari kelompok tengah. Dengan 56 fitur, `sqrt` menyisakan ~7
   fitur per split — terlalu sedikit. Juga tidak berubah dari run sebelumnya.
3. **Ongkos tidak berkorelasi dengan mutu.** Kandidat 7 (276 detik) mencetak
   2,9232, praktis seri dengan kandidat 0 yang makan 602 detik dan hanya 1,5%
   di belakang pemenang yang makan 1.234 detik. Wall time keseluruhan pencarian
   3,85 jam, median 734 detik per kandidat, rentang 276–1.540 detik.

Parameter terpilih (`dataset/model_ready/rf_best_params.json`):

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

### 4.3 Apakah peringkat kandidat berubah setelah pindah ke K1?

Inilah pertanyaan yang membuat sembilan artefak run lama diberi akhiran `.bak`
alih-alih dihapus. Karena seed dan ruang pencariannya identik, ke-18 kandidat
di kedua run adalah **kombinasi parameter yang persis sama, id per id** —
diverifikasi kolom demi kolom — jadi peringkatnya sah dibandingkan meskipun
nilainya tidak.

| | |
|---|---|
| Spearman ρ (K1 baru vs pinball@0,9 lama) | **0,975** |
| Kendall τ | **0,895** |
| Pemenang | **berubah**: kandidat 17 → kandidat 1 |
| Perpindahan peringkat terbesar | kandidat 11: #9 → #12 |

Jawabannya: **peringkatnya nyaris tidak berubah, tapi pemenangnya berubah.**
Kedua hal itu konsisten, bukan bertentangan. Di run lama kandidat 17 dan 1
terpisah **0,0004** — praktis seri, dan urutan di antara keduanya ditentukan
derau. Di K1 jaraknya melebar jadi 0,0177, cukup untuk memisahkan mereka secara
stabil. Merata-ratakan 19 titik meredam derau; itulah yang terlihat di sini.

Dua konsekuensi yang perlu dicatat untuk Fase E:

- Pemenang lama (kandidat 17: `max_depth=12`, `one_hot=True`) dan pemenang baru
  (kandidat 1: `max_depth=20`, `one_hot=False`) adalah konfigurasi yang **cukup
  berbeda**, meski skornya berdekatan. Forest final karenanya bukan forest yang
  sama dengan yang ada di dokumen lama, dan tidak boleh diperlakukan sebagai
  "model yang sama, angka baru".
- ρ = 0,975 adalah bukti langsung bahwa migrasi kriteria **tidak** membalik
  lanskap pencarian di RF. Apakah itu berlaku juga di XGBoost dan LSTM belum
  diketahui, dan tidak boleh diasumsikan — keduanya punya mekanisme yang
  memang berinteraksi dengan jumlah titik kuantil (`multi_strategy` pada
  XGBoost, kepala keluaran multi-titik pada LSTM), sementara RF tidak.

## 5. Hasil walk-forward

Pemenang dijalankan ulang di kelima fold. Empat potongan, masing-masing melawan
ketiga baseline pada baris identik — satu angka global menyesatkan di data yang
44% targetnya nol.

### 5.0 Gerbang G0

> Model harus mengalahkan `naive_roll_mean_7` pada pinball@0,9 **di kelima
> fold**, bukan hanya di gabungan.

pinball@0,9 per fold:

| model | 1 (Jul) | 2 (Agu) | 3 (Sep) | 4 (Okt) | 5 (Nov) |
|---|---:|---:|---:|---:|---:|
| **random_forest** | **2,2631** | **2,4403** | **2,4018** | **2,7571** | **2,5515** |
| `naive_roll_mean_7` | 4,2489 | 4,5665 | 4,0341 | 4,7826 | 4,9703 |
| `naive_lag_1` | 8,3545 | 8,4686 | 8,0448 | 8,5260 | 8,3717 |
| `naive_zero` | 26,4535 | 27,2539 | 23,8486 | 26,2187 | 29,3199 |

**G0 lolos.** RF menang di kelima fold dengan margin 40,5%–48,7%, jadi tidak
ada satu bulan pun yang menggendong kemenangannya.

### 5.1 Per fold

K1 per fold:

| model | 1 (Jul) | 2 (Agu) | 3 (Sep) | 4 (Okt) | 5 (Nov) |
|---|---:|---:|---:|---:|---:|
| **random_forest** | **2,6819** | **2,8441** | **2,7541** | **3,0396** | **3,0305** |
| `naive_roll_mean_7` | 4,6636 | 4,9814 | 4,6680 | 4,9372 | 4,8719 |
| `naive_lag_1` | 7,9549 | 8,5327 | 8,1080 | 7,9712 | 8,3067 |
| `naive_zero` | 14,6964 | 15,1410 | 13,2492 | 14,5660 | 16,2888 |

Detail RF per fold (kolom @0,9 kecuali K1):

| fold | bulan | n | K1 | mae | pinball | coverage | fill_rate | shortfall | overstock |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Jul 2025 | 72.684 | 2,6819 | 14,899 | 2,2631 | 0,9363 | 0,9671 | 70.250 | 1.012.673 |
| 2 | Agu 2025 | 75.015 | 2,8441 | 15,200 | 2,4403 | 0,9314 | 0,9620 | 86.294 | 1.053.931 |
| 3 | Sep 2025 | 70.503 | 2,7541 | 15,653 | 2,4018 | 0,9320 | 0,9605 | 73.719 | 1.029.883 |
| 4 | Okt 2025 | 67.716 | 3,0396 | 15,063 | 2,7571 | 0,9205 | 0,9463 | 105.879 | 914.119 |
| 5 | Nov 2025 | 59.629 | 3,0305 | 14,499 | 2,5515 | 0,9181 | 0,9577 | 82.108 | 782.432 |

Performanya stabil: K1 bergerak 2,682–3,040 (rentang 13%) dan RF menang di
kelima fold dengan margin yang mirip. Fold 4 (Oktober) paling berat untuk RF,
tapi `naive_roll_mean_7` juga memuncak di sana — properti bulannya, bukan
properti RF-nya.

Coverage@0,9 cenderung menurun sepanjang fold, dari 0,9363 (Juli) ke 0,9181
(November) — bukan monoton (fold 3 sedikit di atas fold 2), tapi arahnya
konsisten dan patut dicatat. Kelima nilainya tetap di atas 0,90 dan rentangnya
1,8 poin persentase, jadi belum cukup untuk disebut tren yang mengkhawatirkan
tanpa fold tambahan.

**Fold 3 dan 5 adalah fold yang memilih pemenang.** K1 RF di gabungan kedua
fold itu = **2,8808**, persis sama dengan skor pencarian di bagian 4.1 — konfigurasi
dan seed-nya identik, jadi kesamaan ini adalah cek reprodusibilitas yang lolos,
bukan kebetulan. Dipotong ke fold 1, 2, dan 4 saja — tiga fold yang tidak
menyentuh seleksi, dan potongan yang menjadi kriteria K1 resmi:

| model | n | K1 | mae@0,9 | cov@0,9 | fill@0,9 |
|---|---:|---:|---:|---:|---:|
| **random_forest** | 215.415 | **2,8508** | 15,055 | 0,930 | 0,959 |
| `naive_roll_mean_7` | 215.415 | 4,8603 | 9,721 | 0,696 | 0,850 |
| `naive_lag_1` | 215.415 | 8,1612 | 16,322 | 0,653 | 0,712 |
| `naive_zero` | 215.415 | 14,8102 | 29,620 | 0,423 | 0,000 |

2,8508 di fold bersih lawan 2,8808 di fold seleksi — RF justru **sedikit lebih
baik** di fold yang tidak ikut memilihnya, jadi tidak ada optimisme seleksi yang
terukur. Masuk akal mengingat ruang parameternya sedatar bagian 4.2.

RF **kalah MAE** dari `naive_roll_mean_7` (15,055 lawan 9,721). Ini bukan
kegagalan, ini konsekuensi yang diminta: prediksi di τ=0,9 sengaja bias ke atas,
dan MAE menghukum bias ke atas persis seperti ia menghukum kekurangan stok. MAE
dilaporkan untuk konteks, bukan sebagai kriteria kemenangan.

### 5.2 K2 — kalibrasi di seluruh 19 titik kuantil

Gabungan kelima fold, dibobot jumlah baris. Kolom `lantai` dijelaskan di bawah
tabel.

| τ | pinball | coverage | gap (cov − τ) | lantai | kelebihan di atas lantai |
|---:|---:|---:|---:|---:|---:|
| 0,05 | 0,7390 | 0,4306 | +0,3806 | 0,4195 | +0,0111 |
| 0,10 | 1,3013 | 0,4472 | +0,3472 | 0,4195 | +0,0277 |
| 0,15 | 1,7841 | 0,4678 | +0,3178 | 0,4195 | +0,0483 |
| 0,20 | 2,2066 | 0,4911 | +0,2911 | 0,4195 | +0,0716 |
| 0,25 | 2,5740 | 0,5171 | +0,2671 | 0,4195 | +0,0976 |
| 0,30 | 2,8925 | 0,5439 | +0,2439 | 0,4195 | +0,1244 |
| 0,35 | 3,1611 | 0,5722 | +0,2222 | 0,4195 | +0,1527 |
| 0,40 | 3,3832 | 0,6004 | +0,2004 | 0,4195 | +0,1809 |
| 0,45 | 3,5614 | 0,6307 | +0,1807 | 0,45 | +0,1807 |
| 0,50 | 3,6933 | 0,6619 | +0,1619 | 0,50 | +0,1619 |
| 0,55 | 3,7743 | 0,6929 | +0,1429 | 0,55 | +0,1429 |
| 0,60 | 3,8031 | 0,7254 | +0,1254 | 0,60 | +0,1254 |
| 0,65 | 3,7800 | 0,7584 | +0,1084 | 0,65 | +0,1084 |
| 0,70 | 3,6949 | 0,7923 | +0,0923 | 0,70 | +0,0923 |
| 0,75 | 3,5386 | 0,8255 | +0,0755 | 0,75 | +0,0755 |
| 0,80 | 3,2947 | 0,8591 | +0,0591 | 0,80 | +0,0591 |
| 0,85 | 2,9563 | 0,8937 | +0,0437 | 0,85 | +0,0437 |
| **0,90** | **2,4764** | **0,9281** | **+0,0281** | 0,90 | +0,0281 |
| 0,95 | 1,7648 | 0,9608 | +0,0108 | 0,95 | +0,0108 |

**Kenapa ada lantai.** Target tidak pernah negatif dan prediksi forest juga
tidak (setiap titiknya persentil dari nilai training yang tak-negatif), jadi
setiap baris ber-target nol otomatis terhitung tercakup — definisi coverage
adalah `actual ≤ prediksi`, dan `0 ≤ 0` benar. Di baris validasi **41,95%**
targetnya nol, jadi tidak ada model tak-negatif apa pun yang bisa mencetak
coverage di bawah 0,4195, berapa pun τ-nya. Angka 0,4195 itu terbaca langsung
dari coverage `naive_zero`, yang memang memprediksi nol di setiap baris.

Konsekuensinya, `|coverage(τ) − τ|` di τ rendah **tidak mengukur kalibrasi
model**: di τ=0,05 simpangan minimum yang mungkin dicapai siapa pun adalah
0,3695, dan RF mencetak 0,3806 — hanya **0,011 di atas lantai**. Menyisihkan
model karena angka itu berarti menyisihkannya karena bentuk target.

**Yang tersisa setelah lantai dikurangkan adalah bias yang nyata.** Kolom
terakhir memuncak di **+0,181 pada τ=0,40–0,45** dan tetap +0,16 di median.
Ini tidak dijelaskan oleh massa nol dan harus dibaca sebagai over-coverage
sungguhan: distribusi ramalan RF secara sistematis bergeser ke atas di paruh
bawah grid. Di τ=0,50, RF mencakup 66% baris — median ramalannya terlalu tinggi.

**Mekanisme ikatan — diukur 2026-08-29, bukan lagi hipotesis.** Target 99,55%
bilangan bulat dan 70,3% bernilai ≤ 5, jadi prediksi dan aktual sering
**bernilai sama persis**, dan coverage yang memakai `≤` menghitung setiap
ikatan itu sebagai tercakup. Diuji dengan memprediksi ulang (bundle
tersimpan, tanpa retrain) pada 345.547 baris validasi dan menghitung coverage
dua cara: `tie_rate` (selisih `≤` vs `<` tegas) bergerak **9% di τ=0,95
sampai 43% di τ=0,30** — jauh lebih besar dari over-coverage +0,18 di atas.

**Tapi mengganti `≤` dengan `<` bukan koreksi — ia berbalik arah.** Di
τ=0,40, kelebihan-di-atas-lantai dengan `≤` = +0,175; dengan `<` tegas jadi
**−0,255** (RF tampak jauh *kurang* meramal). `<` menghukum setiap prediksi
yang **tepat sasaran** — lazim di sini karena target integer dan forest
membaca kuantil dari sampel training bertipe sama — sebagai "tidak
tercakup", padahal itu prediksi paling akurat yang bisa dihasilkan.
Kesimpulannya: over-coverage +0,18 di median **tetap** bias nyata (bukan bisa
dihapus dengan ganti operator), tapi harus dibaca dengan konteks bahwa ~43%
baris di sekitarnya memang bernilai identik prediksi-aktual — properti
kediskretan target, bukan artefak pembulatan metrik semata. `≤` (standar
definisi kuantil) tetap definisi yang paling defensif untuk dipakai.
Rinciannya di `docs/todolist-proyek.md` (butir 🆕 "Uji hipotesis efek ikatan").

**Yang harus diputuskan sebelum Fase E.** Tabel pola K2 di
`docs/detail-tahap-perbandingan-model.md` bagian 1.7 membaca "simpangan searah di
hampir seluruh τ" sebagai alasan kuat untuk tersisih. Sebagaimana tertulis,
aturan itu akan menandai **setiap** model di dataset ini, termasuk yang
kalibrasinya sempurna, karena lantai 0,4195 memaksa simpangan searah di semua
τ < 0,42. Aturannya perlu dinyatakan ulang terhadap lantai — mis. membandingkan
`coverage(τ)` dengan `max(τ, share_nol)` seperti kolom terakhir tabel di atas —
sebelum ia dipakai memutuskan apa pun. Ini pekerjaan metodologi, bukan pekerjaan
RF, dan berlaku sama untuk XGBoost dan LSTM.

**Quantile crossing: 0,0000** di seluruh baris, sesuai harapan struktural.

### 5.3 Per `demand_segment`

Gabungan kelima fold, dibobot jumlah baris. Kolom selain K1 dibaca di τ=0,9.

| segmen | model | n | K1 | mae | coverage | fill_rate |
|---|---|---:|---:|---:|---:|---:|
| **smooth** | **random_forest** | 45.485 | **10,9478** | 60,354 | 0,9199 | 0,9789 |
| | `naive_roll_mean_7` | 45.485 | 18,6402 | 37,280 | 0,6016 | 0,8903 |
| | `naive_lag_1` | 45.485 | 32,5143 | 65,029 | 0,4920 | 0,7766 |
| | `naive_zero` | 45.485 | 73,1813 | 146,363 | 0,0129 | 0,0000 |
| **erratic** | **random_forest** | 54.511 | **5,4788** | 26,358 | 0,9029 | 0,9446 |
| | `naive_roll_mean_7` | 54.511 | 9,4628 | 18,926 | 0,5949 | 0,8199 |
| | `naive_lag_1` | 54.511 | 16,4714 | 32,943 | 0,5035 | 0,6556 |
| | `naive_zero` | 54.511 | 24,7993 | 49,599 | 0,0476 | 0,0000 |
| **lumpy** | **random_forest** | 123.545 | **1,1430** | 5,589 | 0,9244 | 0,8355 |
| | `naive_roll_mean_7` | 123.545 | 1,7686 | 3,537 | 0,6763 | 0,6251 |
| | `naive_zero` | 123.545 | 2,5351 | 5,070 | 0,4339 | 0,0000 |
| | `naive_lag_1` | 123.545 | 2,6580 | 5,316 | 0,6400 | 0,4117 |
| **intermittent** | **random_forest** | 122.006 | **0,4194** | 2,777 | 0,9462 | 0,8660 |
| | `naive_roll_mean_7` | 122.006 | 0,6919 | 1,384 | 0,7897 | 0,6131 |
| | `naive_zero` | 122.006 | 0,8365 | 1,673 | 0,7226 | 0,0000 |
| | `naive_lag_1` | 122.006 | 0,9823 | 1,965 | 0,7905 | 0,4128 |

Ini potongan yang paling banyak menjawab. RF menang K1 di **keempat** segmen,
jadi kemenangan globalnya bukan hasil menang di pasangan yang mayoritas nol.
Margin relatif terbesarnya justru di `erratic` (42% lebih baik dari baseline)
dan `smooth` (41%), dua segmen yang benar-benar bergerak; di `lumpy` 35% dan
`intermittent` 39%.

Di `intermittent` dan `lumpy`, MAE RF **lebih buruk daripada `naive_zero`**
(2,777 vs 1,673; 5,589 vs 5,070). Itu bukan anomali. Coverage `naive_zero` di
kedua segmen — 0,7226 dan 0,4339 — **adalah** share target nolnya, jadi menebak
nol terus memang menghasilkan MAE kecil di sana, dengan konsekuensi fill rate 0.
Justru inilah alasan `demand_segment` dibuat: MAE global akan menobatkan model
yang cuma menang di tempat menebak nol itu mudah.

Coverage@0,9 konsisten 0,903–0,946 lintas segmen, semuanya di atas target 0,90.
Yang paling ketat `erratic` (0,9029), yang paling longgar `intermittent`
(0,9462) — dan urutan itu sejalan dengan bagian 5.2: segmen dengan share nol
tertinggi punya lantai coverage tertinggi.

### 5.4 Per `is_delivery_day`

| hari kirim | model | n | K1 | mae@0,9 | cov@0,9 | fill@0,9 | shortfall | overstock |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **True** | **random_forest** | 98.701 | **4,0354** | 20,731 | 0,9231 | 0,9653 | 161.063 | 1.885.137 |
| | `naive_roll_mean_7` | 98.701 | 6,6430 | 13,286 | 0,6341 | 0,8528 | 683.526 | 627.820 |
| | `naive_lag_1` | 98.701 | 11,8786 | 23,757 | 0,5483 | 0,6755 | 1.499.452 | 845.412 |
| | `naive_zero` | 98.701 | 23,4586 | 46,917 | 0,3298 | 0,0000 | 4.630.772 | 0 |
| **False** | **random_forest** | 246.846 | **2,3929** | 12,822 | 0,9302 | 0,9537 | 257.187 | 2.907.901 |
| | `naive_roll_mean_7` | 246.846 | 4,0953 | 8,191 | 0,7175 | 0,8479 | 844.867 | 1.176.969 |
| | `naive_lag_1` | 246.846 | 6,6948 | 13,390 | 0,6936 | 0,7466 | 1.402.090 | 1.903.062 |
| | `naive_zero` | 246.846 | 11,2635 | 22,527 | 0,4554 | 0,0000 | 5.560.708 | 0 |

Di hari kirim — baris yang benar-benar menaikkan barang ke truk — RF menjaga
coverage 0,923 dan fill rate 0,965, dengan shortfall 161.063 unit lawan 683.526
milik baseline terbaik. Margin K1 di sini (4,0354 vs 6,6430, 39% lebih baik)
praktis sama dengan di hari non-kirim (2,3929 vs 4,0953, 42%). Di run
kuantil-tunggal margin hari kirim terbaca **lebih lebar** daripada hari
non-kirim (52% vs 42%); di bawah K1 keunggulan itu merata. Bacaan "keunggulan
RF terkonsentrasi di baris yang paling penting" karenanya **tidak lagi didukung
data** dan tidak diulang di sini.

## 6. Model final

`rf.fit_final()` melatih ulang konfigurasi pemenang dengan **`n_estimators`
dinaikkan 200 → 400** (`FINAL_N_ESTIMATORS`) pada seluruh baris layak sebelum
Desember, dipotong di batas Desember oleh `purging.lookahead_safe_mask()` —
populasi baris yang sama persis dengan yang dinilai di atas.

| | |
|---|---|
| Baris training | 1.349.011 |
| Kolom fitur | 56 (`one_hot=False`, jadi tanpa ekspansi) |
| Titik kuantil tersimpan | 19 (0,05..0,95) |
| Artefak | `models/random_forest_q90.joblib` — 826 MB, 25 Agu 2026 18:25 |

Bundle-nya menyimpan urutan kolom training beserta flag `one_hot`/`log_target`
dan grid kuantilnya, karena forest yang dimuat ulang dengan urutan kolom berbeda
tidak gagal — ia memprediksi dengan percaya diri dari fitur yang salah.
`predict_bundle()` membaca grid dari bundle, bukan dari konstanta modul, supaya
bundle lama tetap terbaca setelah `QUANTILE_SET` berubah lagi.

Nama berkasnya masih `random_forest_q90` meski isinya kini 19 titik. Nama itu
sengaja tidak diubah dalam run ini supaya jalur `MODEL_FILE` tidak berubah di
tengah migrasi; penggantian namanya masuk daftar hygiene, bukan blocker.

## 7. Ongkos (bahan K3)

Seluruh run di CPU Mac lokal — keputusan pemilik proyek 2026-08-25 bahwa
seluruh Fase 3 dijalankan di satu device, supaya K3 terbaca dalam bacaan paling
ketat dan tidak ada penyerahan device antar model
(bagian 0 `2026-08-24-distributed-gpu-training-design.md`).

| tahap | wall clock |
|---|---:|
| Benchmark | 9,7 menit |
| Pencarian 18 kandidat | 3,85 jam |
| Walk-forward 5 fold | ~45 menit |
| Fit final | ~48 menit |
| **Total** | **~5,6 jam** |

Estimasi di todolist adalah ~4,8 jam, jadi meleset **+17%** — jauh lebih jinak
daripada XGBoost, yang estimasinya meleset ~1,87×.

## 8. Reproduksi

```bash
.venv/bin/python3 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebook/modeling_rf.ipynb
```

Butuh berjam-jam. Pencarian menulis checkpoint tiap kandidat selesai ke
`rf_search_results.csv` dan melanjutkan dari sana kalau dijalankan ulang, jadi
run yang terbunuh OS tidak menghanguskan seluruh sore. Checkpoint itu dijaga
`_assert_checkpoint_matches()`, yang menolak melanjutkan dari berkas yang lahir
di ruang pencarian atau grid kuantil berbeda.

| Artefak | Lokasi | Masuk git |
|---|---|---|
| Hasil pencarian | `dataset/model_ready/rf_search_results.csv` | tidak |
| Parameter terpilih | `dataset/model_ready/rf_best_params.json` | tidak |
| Tabel hasil lengkap | `dataset/model_ready/rf_walk_forward_results.csv` | tidak |
| Forest terlatih | `models/random_forest_q90.joblib` | tidak |
| Ringkasan ini | `docs/hasil-modeling-rf.md` | **ya** |
| Arsip run kuantil-tunggal | `docs/hasil-modeling-rf.single-quantile.bak.md` | **ya** |

## 9. Batasan

- **Desember 2025 belum dibuka.** Semua angka di sini adalah validasi
  walk-forward, bukan skor test set final.
- **Sumbu waktunya waktu pengambilan, bukan waktu pemesanan.** Model ini
  meramal permintaan terealisasi pada tanggal pickup; sebagian permintaan hari
  depan sudah diketahui kantor pusat lewat pre-order yang tidak tercatat di
  dataset mana pun. Lihat `docs/batasan-penelitian.md` (B-1, B-2, B-3).
- **MAE tidak sebanding lintas model di sini.** Membandingkan MAE model kuantil
  dengan baseline titik-tengah menghukum yang pertama karena melakukan persis
  apa yang diminta. K1 adalah kriterianya.
- **K2 di τ rendah belum bisa dibaca sebagai kalibrasi** sampai aturannya
  dinyatakan ulang terhadap lantai `share_nol` (bagian 5.2). Bagian yang sudah bisa
  dibaca sekarang adalah over-coverage +0,18 di sekitar median, yang nyata.
- **Fold 3 dan 5 ikut memilih pemenang**, jadi skornya di potongan per-fold
  bukan out-of-sample terhadap seleksi. Potongan fold 1/2/4 di bagian 5.1 adalah
  angka yang bersih dan yang menjadi K1 resmi.
- **Satu seed, satu kali latih.** Setiap konfigurasi dilatih sekali. Untuk RF
  ini kurang mengkhawatirkan daripada untuk LSTM — bagging 200 pohon sudah
  merata-ratakan sebagian besar varians inisialisasi — tapi tetap berarti
  selisih di bawah ambang 2% tidak bisa dipisahkan dari derau.
- **Satu model, belum perbandingan.** XGBoost dan LSTM belum dijalankan ulang di
  bawah kriteria multi-kuantil, jadi belum ada yang bisa dikatakan soal apakah
  Random Forest pilihan terbaik — baru bahwa ia jauh mengalahkan ketiga baseline
  naive. Angka di `docs/hasil-modeling-xgb.md` dan `docs/hasil-modeling-lstm.md`
  masih pinball@0,9 dari run lama dan **tidak boleh** disandingkan dengan K1 di
  dokumen ini (T-10).
