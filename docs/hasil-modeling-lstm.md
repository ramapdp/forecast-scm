# Hasil Modeling — LSTM kuantil 0.9

Angka terukur dari jalannya `notebook/modeling_lstm.ipynb`. Desainnya ada di
`docs/superpowers/specs/2026-08-19-lstm-modeling-design.md`; dokumen ini hanya
memuat hasilnya, supaya bukti di balik tulisan hidup di git dan bukan cuma di
output cell notebook yang bisa hilang saat di-clear.

**Artefak `models/lstm_q90.joblib` sudah basi (per 2026-08-23) — jangan
dipakai untuk prediksi.** Ia dilatih 20 Agu 2026 di atas `model_input.parquet`
sebelum refresh kategori WIP-2 2026-08-22, jadi baris embedding untuk WIP-2 kini mati.
Model yang dimuat ulang tidak akan gagal — ia tetap memberi angka, dari fitur
yang salah. Angka di dokumen ini tetap sah sebagai catatan run tersebut, bukan
sebagai gambaran `model_input.parquet` yang sekarang. Model ini akan **dilatih
ulang** dalam migrasi multi-kuantil berikutnya; latar lengkapnya di §0
`docs/pipeline-overview.md` dan B-9 `docs/batasan-penelitian.md`.

**Desember 2025 tidak dibuka.** Semua angka di bawah datang dari walk-forward
lima fold di Juli–November 2025. Test set final masih terkunci.

## 1. Ringkasan

LSTM kuantil 0.9 mencetak pinball@0.9 gabungan **2.427** lawan **4.503** milik
baseline terbaik `naive_roll_mean_7` — sekitar 46% lebih baik. Kalibrasinya
paling dekat ke sasaran dari ketiga model: coverage **0.907** terhadap target
service level 0.90, lawan 0.909 milik XGBoost dan 0.932 milik Random Forest.

Pada kriteria yang dipakai memilih, LSTM **juara ketiga**: 2.427 lawan 2.390
(XGBoost) dan 2.410 (Random Forest) — tertinggal 1,6% dan 0,7%. Di potongan
fold yang tidak ikut memilih pemenang, jaraknya menyusut jadi 0,9% dan 0,7%.
§6 membahas kenapa urutan setipis itu belum menobatkan siapa pun, dan §9
mencatat satu asimetri yang khusus merugikan LSTM: ia dapat 12 undian
pencarian, XGBoost 30, RF 18.

LSTM **kalah MAE** dari `naive_roll_mean_7` (14.93 lawan 9.65), persis seperti
dua model lain dan karena alasan yang sama: model kuantil 0.9 sengaja bias ke
atas, dan MAE menghukum bias ke atas sama kerasnya seperti ia menghukum
kekurangan stok. MAE dilaporkan untuk konteks, bukan sebagai kriteria
kemenangan. Menariknya, di potongan bersih fold 1/2/4 justru MAE LSTM yang
**terbaik dari ketiga model** (14.28 lawan 14.35 dan 15.59) — ia meleset paling
kecil di tengah distribusi sambil kalah tipis di ekor atas yang dihargai
pinball.

Sisi bisnisnya, di 345.547 baris validasi:

| | kekurangan (shortfall) | kelebihan (overstock) |
|---|---:|---:|
| `lstm` | 403.337 | 4.755.695 |
| `xgboost` | 414.172 | 4.529.674 |
| `random_forest` | 365.576 | 5.038.816 |
| `naive_roll_mean_7` | 1.528.393 | 1.804.789 |

LSTM memangkas kekurangan stok **74%** dari baseline dengan ongkos kelebihan
stok **2,6×** lipat. Terhadap XGBoost ia menahan kekurangan stok 2,6% lebih
kecil dengan kelebihan stok 5% lebih besar — ketiganya duduk di tukar-menukar
yang praktis sama, yang memang diharapkan dari tiga model yang menargetkan
kuantil yang sama.

> Catatan satuan: `shortfall_units` dan `overstock_units` menjumlahkan unit
> lintas SKU yang satuannya campur (Kg, Porsi, Botol, PCS, …), jadi angkanya
> sah untuk membandingkan antar model pada baris yang sama, tapi tidak punya
> makna fisik sebagai satu besaran tunggal.

## 2. Setup evaluasi

| | |
|---|---|
| Data | `dataset/model_ready/model_input.parquet` — 1.502.522 baris panel, 82 kolom, 1 Jan 2024 – 31 Des 2025 |
| Fitur | 56 kolom (`modeling_prep.FEATURE_COLS`) — 49 dinamis + 7 kategorikal `_idx` |
| Target | `target_lead_time_cumulative` — 44,35% bernilai nol |
| Fold | 5 expanding window, validasi Juli, Agustus, September, Oktober, November 2025 |
| Test terkunci | Desember 2025 (`TEST_START = 2025-12-01`) |
| Baris tereliminasi | 28 hari awal tiap `segment_id` (jendela lag belum penuh) dan baris tanpa target |
| Baris validasi | 345.547 total |
| Kuantil | 0.9 |
| Implementasi | `torch==2.8.0`, `QuantileLSTM` — LSTM 1 lapis + embedding kategorikal + head MLP |
| Jendela | 28 hari (`modeling_prep.LOOKBACK`), berakhir di baris prediksi |
| Jumlah epoch | early stopping (`EARLY_STOPPING_EPOCHS = 5`) di ekor 30 hari terakhir jendela training, lalu **refit** di seluruh baris training pada epoch itu |
| Plafon epoch | `MAX_EPOCHS = 100` |
| Device | CPU (§3) |

**Yang membedakan model ini dari dua lainnya**: ia membaca 28 harinya sendiri,
bukan ringkasan hasil rekayasa dari 28 hari itu yang disediakan `lag_*` dan
`roll_*`. Himpunan fiturnya identik — 56 kolom yang sama, dilarang ditambah
atau diurutkan ulang — tapi jumlah informasi yang sampai ke model tidak sama:
RF dan XGBoost melihat satu baris berisi ringkasan, LSTM melihat 28 baris
mentah yang meringkasnya. Itu perbedaan yang paling layak diingat saat membaca
§6, karena artinya perbandingan ketiganya bukan cuma soal keluarga model.

Empat detail konstruksi yang menentukan sah-tidaknya angka di atas:

- **Kategorikal dibaca di baris prediksi, tidak diulang sepanjang jendela.**
  `Kategori Barang_idx` berubah di dalam 301 segmen nyata, jadi "kategori milik
  segmen ini" bukan hal yang terdefinisi untuk diulang 28 kali.
- **Ukuran embedding datang dari `category_mapping.json`**, bukan dari nilai
  yang kebetulan muncul di baris training satu fold. Cabang yang baru buka
  setelah model dilatih memetakan ke slot UNKNOWN 0 dan tetap dalam rentang;
  alternatifnya gagal berbulan-bulan kemudian, di produksi, dengan index error.
- **Scaler dipasang per fold, hanya dari baris training**, lalu dipakai kedua
  fit. Ekor early stopping ada di dalam jendela training, jadi statistiknya
  tidak bocor ke validasi — dan berbagi satu scaler itulah yang membuat
  `best_epoch` bermakna sama di kedua fit.
- **Jendela dipotong dari panel penuh, bukan dari `eligible_rows()`.** Jendela
  milik baris validasi 1 Juli menjangkau mundur ke Juni, melewati baris yang
  dihapus warm-up 28 hari dan purge fold. Membaca *fitur* baris itu aman:
  setiap jendela berakhir di baris prediksinya sendiri dan setiap lag/rolling
  berhenti di H-1, jadi tak ada nilai target yang bisa masuk jendela. Yang
  dicegah purging adalah training atas *label* baris tersebut, dan itu tetap
  tidak pernah terjadi.

**Protokol dua fit** identik dengan XGBoost, karena masalahnya identik: jumlah
epoch adalah keputusan kapasitas, dan tempat paling wajar mengambilnya — fold
validasi — justru tempat yang bocor. Jadi early stopping berjalan di ekor 30
hari terakhir jendela training (`model_common.split_early_stopping`, dengan
purge yang sama), lalu model dibuang, diinisialisasi ulang dari seed yang sama,
dan dilatih di **seluruh** baris training selama persis sebanyak epoch itu.
Hasilnya LSTM akhirnya dilatih di populasi baris yang persis sama dengan yang
dilihat RF dan XGBoost.

Ketiga baseline naive dinilai pada **baris yang identik** dengan LSTM —
dijamin oleh `utils/walk_forward.py` yang memiliki definisi fold dan kelayakan
baris, dan menerima model sebagai callable yang disuntikkan. `walk_forward.py`
tidak disentuh sama sekali untuk model ini. Baris yang sama itu juga yang
dipakai run RF dan XGBoost, yang membuat §6 sah dilakukan.

## 3. Benchmark

Diukur 2026-08-19 di fold 5 dengan `DEFAULT_PARAMS`, satu putaran dua-fit
penuh, untuk memilih device dan menurunkan anggaran pencarian dari ongkos yang
terukur alih-alih dari tebakan.

| | |
|---|---|
| Device terpilih | **CPU** |
| Baris fold 5 | 1.292.778 training / 59.629 validasi |
| `best_epoch` | **3** (dari `MAX_EPOCHS = 100`) |
| `sec_per_epoch` | **102,0** |
| Wall time dua fit + predict | 1.121,6 s (18,7 menit) |
| Peak RSS | 4,60 GB |
| Prediksi | rata-rata 44,84, maksimum 1.721,11 |
| **`N_CANDIDATES`** | **12** |

`candidate_budget(102.0, 3)` = `28800 // (2 * 102.0 * (2*3 + 5))` = 12, ditahan
oleh plafon 8 jam dan bukan oleh `MAX_CANDIDATES = 20`.

Tiga hal yang ditentukan angka ini:

1. **CPU menang atas MPS, 2×.** MPS tidak diukur ujung ke ujung: probe 15 batch
   di fold 5 mencatat **0,392 s/batch di MPS lawan 0,193 s/batch di CPU** — MPS
   tidak punya kernel LSTM ter-fusi di ukuran hidden ini — jadi satu putaran
   MPS penuh akan menghabiskan sekitar empat jam untuk mengonfirmasi device
   yang sudah kalah 2×.

2. **Ruang pencarian dikecilkan sebelum benchmark ini jalan.** Di
   `DEFAULT_PARAMS` yang lama (`hidden_size=128, num_layers=2`) satu epoch
   berongkos 259 s, yang membuat `candidate_budget` melempar error untuk
   `best_epoch >= 3` — sinyal yang memang dirancang untuk memaksa ruangnya
   dikecilkan, bukan plafonnya dinaikkan diam-diam. Ongkos per epoch terukur di
   fold 5, CPU:

   | hidden | layers | batch | s/epoch |
   |---:|---:|---:|---:|
   | 64 | 1 | 1024 | 75 |
   | 64 | 1 | 2048 | 47 |
   | 128 | 1 | 1024 | 104 |
   | 128 | 1 | 2048 | 83 |
   | 128 | 2 | 1024 | 259 |

   `SEARCH_SPACE` karena itu membuang `num_layers=2` dan `hidden_size=256`, dan
   `DEFAULT_PARAMS` pindah ke `num_layers=1`. Kedalaman yang dikorbankan, bukan
   lebar, karena membuangnya membeli detik paling banyak per dimensi yang
   dihapus. Konsekuensinya dicatat di §4 dan §9: pencarian ini **tidak pernah
   menanyakan** apakah lapisan kedua akan menolong.

3. **Jendela padat tidak pernah dimaterialisasi.** Tensor 1.502.522 × 28 × 56
   float32 berukuran **9,42 GB** di mesin 16 GB. `utils/sequence_windows.py`
   menyimpan panel sebagai satu matriks kontigu **294 MB** plus array posisi
   akhir jendela, dan tiap jendela adalah irisan `sliding_window_view` dari
   situ — tidak ada yang disalin sampai satu batch dibentuk. Ini yang membuat
   modelnya muat sama sekali, dan syaratnya (nol celah tanggal di dalam satu
   segmen) diperiksa ulang tiap `build_index`, bukan dipercaya.

**Benchmark ini meleset di satu hal, dan melesetnya mahal.** `best_epoch = 3`
yang diukur di `DEFAULT_PARAMS` ternyata tidak mewakili kandidat pencarian:
epoch yang sebenarnya dipilih early stopping bergerak **3–13** (§4). Karena
`candidate_budget` menskalakan linear terhadap angka itu, plafon 8 jam yang
dihitungnya jadi terlalu optimistis — tujuh kandidat yang ongkosnya tercatat
menghabiskan 6,63 jam, yang kalau diekstrapolasi ke 12 kandidat berarti
**sekitar 11,4 jam**, bukan 8. Plafonnya terlampaui; yang berfungsi adalah
checkpoint-nya, bukan anggarannya.

## 4. Pencarian hyperparameter

12 kandidat ditarik acak dengan seed 42 dari ruang **48** kombinasi. Penilaian
di **fold 3 (September) dan fold 5 (November)** saja, kriteria **pinball@0.9
gabungan** (dibobot jumlah baris, bukan dirata-rata polos). **Tidak ada
subsampling** — seluruh baris training tiap fold dipakai.

Jumlah epoch sengaja tidak ada di ruang pencarian: early stopping sudah
memutuskannya per kandidat per fold, sama seperti jumlah ronde pada XGBoost.

Ke-12 kandidat selesai dinilai; tidak ada yang gagal (kolom `error` kosong
semua).

| # | hidden | dropout | lr | batch | log_target | pinball | mae | coverage | epoch (fold 3,5) |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| **10** | 128 | 0.3 | 0.0003 | 1024 | False | **2.4364** | 16.010 | 0.916 | 8,6 |
| 11 | 128 | 0.2 | 0.0010 | 1024 | False | 2.4384 | 15.809 | 0.913 | 3,3 |
| 1 | 64 | 0.3 | 0.0010 | 1024 | False | 2.4388 | 15.265 | 0.907 | — |
| 8 | 64 | 0.2 | 0.0003 | 1024 | False | 2.4497 | 16.073 | 0.911 | 8,13 |
| 3 | 128 | 0.2 | 0.0003 | 1024 | False | 2.4599 | 15.364 | 0.908 | — |
| 2 | 64 | 0.0 | 0.0003 | 1024 | False | 2.4807 | 16.081 | 0.912 | — |
| 0 | 128 | 0.0 | 0.0003 | 1024 | False | 2.4875 | 15.887 | 0.900 | — |
| 7 | 128 | 0.3 | 0.0003 | 2048 | False | 2.5019 | 16.606 | 0.908 | 10,12 |
| 9 | 128 | 0.2 | 0.0003 | 2048 | **True** | 2.9590 | 23.550 | 0.933 | 4,9 |
| 6 | 128 | 0.3 | 0.0003 | 1024 | **True** | 3.1530 | 26.046 | 0.922 | 6,10 |
| 5 | 64 | 0.2 | 0.0010 | 1024 | **True** | 3.2265 | 26.913 | 0.917 | 6,5 |
| 4 | 64 | 0.2 | 0.0003 | 2048 | **True** | 3.4494 | 29.277 | 0.924 | — |

Tabel lengkap: `dataset/model_ready/lstm_search_results.csv`.

Tiga hal yang terbaca dari sebaran ini:

1. **`log_target=True` bukan cuma merugikan di sini — ia menghancurkan.**
   Empat kandidat `True` menempati empat posisi terbawah tanpa kecuali, dengan
   median 3.190 lawan 2.455 milik delapan kandidat `False`: **30% lebih
   buruk**, jurang yang tidak ada bandingannya di pencarian RF maupun XGBoost
   (di XGBoost selisih median-nya cuma 1%). Alasannya sama tapi efeknya lebih
   keras: melatih di ruang log lalu mentransformasi balik menggeser kuantil
   yang dioptimalkan, dan dengan objektif yang sudah pinball transformasi itu
   melawan tujuannya sendiri. Pada LSTM, yang gradiennya mengalir lewat
   normalisasi dan bukan lewat partisi, distorsi itu tidak punya tempat
   sembunyi. MAE-nya ikut membengkak (23–29 lawan 15–17), jadi ini bukan
   pertukaran kalibrasi — ini murni lebih buruk.

2. **Sisanya datar, seperti dua model lain.** Di antara delapan kandidat
   `log_target=False`, rentangnya cuma 2.4364–2.5019 (2,7%), dan pemenang
   mengalahkan runner-up dengan selisih 0.0020 — praktis seri. Tiga teratas
   memakai tiga kombinasi hidden/lr/dropout yang berbeda-beda, jadi tidak ada
   satu dimensi pun yang tampak menentukan.

3. **`batch_size=2048` selalu kalah dari kembarannya di 1024**, dan itu masuk
   akal: batch dua kali lebih besar berarti setengah langkah gradien per epoch,
   sementara jumlah epoch dipilih oleh early stopping yang sama. Perlu dicatat
   pembagiannya cuma 9 lawan 3 — hasil undian acak, bukan desain berimbang —
   jadi ini indikatif, bukan eksperimen terkontrol.

**Yang tidak ditanyakan pencarian ini**: apakah lapisan LSTM kedua menolong,
dan apakah `hidden_size=256` menolong. Keduanya dibuang dari ruang pencarian
karena ongkos (§3), bukan karena bukti. Jadi kesimpulan yang jujur dari tabel
di atas bukan "arsitektur ini yang terbaik", melainkan "ini yang terbaik di
antara yang sempat diuji dalam plafon 8 jam".

**Catatan pembacaan tabel**: kolom `epoch` kosong untuk lima kandidat (0–4)
karena kandidat-kandidat itu selesai sebelum kolom `best_epoch` dan
`elapsed_seconds` ditambahkan ke `model_common.run_search()` — perubahan yang
sampai dokumen ini ditulis masih belum di-commit. Run yang menghasilkan angka
ini dilanjutkan dari checkpoint lama, dan checkpoint tidak menulis ulang baris
yang sudah ada. Skor pinball-nya tidak terpengaruh; yang hilang hanya catatan
ongkosnya.

Parameter terpilih (`dataset/model_ready/lstm_best_params.json`):

```json
{
  "batch_size": 1024,
  "dropout": 0.3,
  "grad_clip": 1.0,
  "hidden_size": 128,
  "learning_rate": 0.0003,
  "log_target": false,
  "num_layers": 1,
  "random_state": 42
}
```

## 5. Hasil walk-forward

Pemenang dijalankan ulang di kelima fold. Tiga potongan, masing-masing melawan
ketiga baseline pada baris identik — satu angka global menyesatkan di data yang
44% targetnya nol.

Wall time: **11.156 s (3 jam 6 menit)** untuk kelima fold.

### 5.1 Per fold

pinball@0.9 per fold:

| model | 1 (Jul) | 2 (Agu) | 3 (Sep) | 4 (Okt) | 5 (Nov) |
|---|---:|---:|---:|---:|---:|
| **lstm** | **2.128** | **2.471** | **2.427** | **2.680** | **2.448** |
| naive_roll_mean_7 | 4.249 | 4.566 | 4.034 | 4.783 | 4.970 |
| naive_lag_1 | 8.355 | 8.469 | 8.045 | 8.526 | 8.372 |
| naive_zero | 26.453 | 27.254 | 23.849 | 26.219 | 29.320 |

Detail LSTM per fold:

| fold | bulan | n | mae | pinball | coverage | fill_rate | shortfall | overstock | epoch |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Jul 2025 | 72.684 | 14.649 | 2.128 | 0.922 | 0.972 | 60.256 | 1.004.460 | 7 |
| 2 | Agu 2025 | 75.015 | 13.929 | 2.471 | 0.890 | 0.955 | 101.092 | 943.794 | 10 |
| 3 | Sep 2025 | 70.503 | 17.170 | 2.427 | 0.925 | 0.967 | 62.546 | 1.147.967 | 8 |
| 4 | Okt 2025 | 67.716 | 14.267 | 2.680 | 0.893 | 0.946 | 106.096 | 859.979 | 6 |
| 5 | Nov 2025 | 59.629 | 14.638 | 2.448 | 0.905 | 0.962 | 73.348 | 799.495 | 6 |

LSTM menang atas ketiga baseline di kelima fold, tanpa satu bulan pun yang
menggendong hasilnya: pinball bergerak 2.128–2.680. Fold 4 (Oktober) paling
berat, persis seperti pada RF dan XGBoost — properti bulannya, bukan properti
modelnya.

Kolom epoch memperlihatkan early stopping bekerja: 6–10 epoch, tidak ada yang
mendekati plafon 100, dan tidak ada fold yang berhenti di epoch 1. Rentangnya
juga cukup lebar (6 lawan 10) untuk membenarkan keputusan tidak mematoknya satu
angka untuk semua fold.

Coverage-nya **bergoyang di sekitar target, tidak menetap di satu sisi**:
0.922, 0.890, 0.925, 0.893, 0.905. Dua fold di bawah 0.90 dan dua fold jauh di
atasnya. Pola ini berbeda dari RF (konservatif di semua fold) maupun XGBoost
(menurun rapi dari 0.924 ke 0.894) — pada LSTM sumbernya bukan tren, melainkan
sensitivitas terhadap epoch yang dipilih tiap fold, karena kuantil di sini
adalah hasil optimisasi bertahap, bukan kuantil empiris yang dibaca dari daun.

**Fold 3 dan 5 adalah fold yang memilih pemenang**, jadi skornya di sana bukan
out-of-sample terhadap seleksi (pinball gabungan fold 3+5 = 2.4364, persis sama
dengan skor pencarian — konfigurasi dan seed-nya identik). Dipotong ke fold 1,
2, dan 4 saja — tiga fold yang tidak menyentuh seleksi:

| model | n | mae | pinball | coverage | fill_rate |
|---|---:|---:|---:|---:|---:|
| **lstm** | 215.415 | 14.278 | **2.421** | 0.902 | 0.958 |
| naive_roll_mean_7 | 215.415 | 9.721 | 4.527 | 0.696 | 0.850 |
| naive_lag_1 | 215.415 | 16.322 | 8.448 | 0.653 | 0.712 |
| naive_zero | 215.415 | 29.620 | 26.658 | 0.423 | 0.000 |

2.421 di potongan bersih lawan 2.427 gabungan kelima fold — **lebih baik, bukan
lebih buruk**. Optimisme seleksinya nol yang terukur; kebetulan dua fold yang
memilih pemenang justru dua fold yang lebih sulit bagi model ini. Ini pola yang
sama dengan RF dan konsisten dengan ruang parameter yang memang datar (§4).

### 5.2 Per `demand_segment`

Gabungan kelima fold, dibobot jumlah baris:

| segmen | model | n | mae | pinball | coverage | fill_rate |
|---|---|---:|---:|---:|---:|---:|
| **smooth** | **lstm** | 45.485 | 57.850 | **8.482** | 0.900 | 0.977 |
| | naive_roll_mean_7 | 45.485 | 37.280 | 16.540 | 0.602 | 0.890 |
| | naive_lag_1 | 45.485 | 65.029 | 32.602 | 0.492 | 0.777 |
| | naive_zero | 45.485 | 146.363 | 131.726 | 0.013 | 0.000 |
| **erratic** | **lstm** | 54.511 | 27.346 | **4.909** | 0.893 | 0.945 |
| | naive_roll_mean_7 | 54.511 | 18.926 | 9.028 | 0.595 | 0.820 |
| | naive_lag_1 | 54.511 | 32.943 | 16.945 | 0.504 | 0.656 |
| | naive_zero | 54.511 | 49.599 | 44.639 | 0.048 | 0.000 |
| **lumpy** | **lstm** | 123.545 | 5.978 | **1.079** | 0.906 | 0.882 |
| | naive_roll_mean_7 | 123.545 | 3.537 | 1.873 | 0.676 | 0.625 |
| | naive_lag_1 | 123.545 | 5.316 | 2.914 | 0.640 | 0.412 |
| | naive_zero | 123.545 | 5.070 | 4.563 | 0.434 | 0.000 |
| **intermittent** | **lstm** | 122.006 | 2.447 | **0.425** | 0.917 | 0.867 |
| | naive_roll_mean_7 | 122.006 | 1.384 | 0.657 | 0.790 | 0.613 |
| | naive_lag_1 | 122.006 | 1.965 | 0.981 | 0.791 | 0.413 |
| | naive_zero | 122.006 | 1.673 | 1.506 | 0.723 | 0.000 |

LSTM menang pinball di **keempat** segmen, jadi kemenangan globalnya bukan
hasil menang di pasangan yang mayoritas nol — margin terbesarnya justru ada di
`smooth` (8.482 vs 16.540) dan `erratic` (4.909 vs 9.028), dua segmen yang
benar-benar bergerak.

Sama seperti dua model lain, di `intermittent` dan `lumpy` MAE LSTM **lebih
buruk daripada `naive_zero`** (2.447 vs 1.673; 5.978 vs 5.070). Itu bukan
anomali: di baris validasi, `intermittent` 72% targetnya nol dan `lumpy` 43%,
jadi menebak nol terus memang menghasilkan MAE kecil di sana — dengan
konsekuensi fill rate 0 dan coverage 0.723/0.434.

Coverage lintas segmen 0.893–0.917, rentang tersempit dari ketiga model (RF
0.911–0.946, XGBoost 0.893–0.922). Yang menarik, `smooth` mendarat di
0.8997 — praktis tepat di sasaran, dan itu segmen dengan volume per baris
terbesar.

### 5.3 Per `is_delivery_day`

| hari kirim | model | n | mae | pinball | coverage | fill_rate | shortfall | overstock |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **True** | **lstm** | 98.701 | 20.819 | **3.339** | 0.910 | 0.967 | 155.051 | 1.899.793 |
| | naive_roll_mean_7 | 98.701 | 13.286 | 6.869 | 0.634 | 0.853 | 683.526 | 627.820 |
| | naive_lag_1 | 98.701 | 23.757 | 14.529 | 0.548 | 0.675 | 1.499.452 | 845.413 |
| | naive_zero | 98.701 | 46.917 | 42.225 | 0.330 | 0.000 | 4.630.772 | 0 |
| **False** | **lstm** | 246.846 | 12.575 | **2.062** | 0.906 | 0.955 | 248.285 | 2.855.901 |
| | naive_roll_mean_7 | 246.846 | 8.191 | 3.557 | 0.717 | 0.848 | 844.867 | 1.176.969 |
| | naive_lag_1 | 246.846 | 13.390 | 5.883 | 0.694 | 0.747 | 1.402.090 | 1.903.063 |
| | naive_zero | 246.846 | 22.527 | 20.274 | 0.455 | 0.000 | 5.560.708 | 0 |

Di hari kirim — baris yang benar-benar menaikkan barang ke truk — LSTM menjaga
coverage 0.910 dan fill rate 0.967, dengan shortfall 155.051 unit lawan 683.526
milik baseline terbaik. Margin pinball di sini (3.339 vs 6.869, 51% lebih baik)
lebih lebar daripada di hari non-kirim (2.062 vs 3.557, 42%), jadi
keunggulannya terkonsentrasi persis di baris yang paling penting — pola yang
sama dengan RF dan XGBoost.

Kalibrasinya praktis sama di kedua sisi (0.910 dan 0.906), seperti XGBoost
(0.908/0.909) dan tidak seperti RF yang duduk 3 poin di atas target di
keduanya.

## 6. Head-to-head tiga model

Sah dilakukan karena ketiganya dinilai di **baris yang identik** — dijamin
`walk_forward.eligible_rows()`, bukan oleh disiplin. Ketiga baseline mencetak
angka yang sama persis di ketiga run, yang mengonfirmasi itu.

| potongan | model | pinball | mae | coverage |
|---|---|---:|---:|---:|
| semua fold | **xgboost** | **2.3896** | 14.307 | 0.909 |
| | random_forest | 2.4104 | 15.640 | 0.932 |
| | lstm | 2.4268 | 14.930 | 0.907 |
| fold 1/2/4 (bersih) | **xgboost** | **2.3998** | 14.351 | 0.910 |
| | random_forest | 2.4033 | 15.593 | 0.934 |
| | lstm | 2.4210 | **14.278** | 0.902 |

Per fold, pinball@0.9:

| fold | random_forest | xgboost | lstm |
|---|---:|---:|---:|
| 1 (Jul) | 2.1904 | 2.1897 | **2.1281** |
| 2 (Agu) | **2.3516** | 2.3699 | 2.4710 |
| 3 (Sep) | 2.3690 | **2.3104** | 2.4267 |
| 4 (Okt) | 2.6891 | **2.6584** | 2.6801 |
| 5 (Nov) | 2.4848 | **2.4464** | 2.4478 |

Empat hal yang terbaca dari perbandingan ini:

1. **LSTM menang telak di fold 1 dan kalah telak di fold 2.** 2.128 di Juli
   adalah skor terbaik yang dicetak model mana pun di fold mana pun — 2,8%
   di bawah dua tetangganya, margin terlebar yang muncul di tabel ini. Lalu di
   Agustus ia jadi yang terburuk, 5,1% di atas RF. Dua fold berurutan,
   satu model, arah berlawanan: variansnya antar-fold lebih besar daripada
   selisih antar-model, yang seharusnya membuat siapa pun berhati-hati
   menyimpulkan urutan dari selisih 1%.

2. **Selisihnya di bawah tingkat kebisingan yang bisa diukur run ini.** Di
   potongan bersih, 2.3998 / 2.4033 / 2.4210 — jarak juara ke juru kunci 0,9%.
   Setiap konfigurasi hanya dilatih **satu kali dengan satu seed**, dan LSTM
   adalah satu-satunya dari ketiganya yang hasilnya bergantung pada
   inisialisasi bobot acak dan urutan batch. Tanpa pengulangan seed, tidak ada
   yang bisa memisahkan "LSTM sedikit lebih buruk" dari "LSTM dapat seed yang
   sedikit kurang beruntung".

3. **Anggaran pencariannya paling kecil, dan ruangnya sudah dipotong lebih
   dulu.** 12 kandidat dari ruang 48 kombinasi, lawan 30 dari 2.592 (XGBoost)
   dan 18 dari 1.152 (RF). Lebih dari itu, ruang LSTM sudah kehilangan dua
   dimensi kapasitas (`num_layers=2`, `hidden_size=256`) sebelum undian pertama
   ditarik, karena ongkos wall-clock — bukan karena keduanya diuji dan kalah.
   Ini asimetri yang berpihak melawan LSTM, kebalikan dari asimetri yang
   berpihak pada XGBoost di perbandingannya lawan RF.

4. **Di level segmen ketiganya membagi kemenangan**, dan LSTM memegang satu:

   | segmen | random_forest | xgboost | lstm |
   |---|---:|---:|---:|
   | smooth | 8.570 | **8.373** | 8.482 |
   | erratic | 4.810 | **4.785** | 4.909 |
   | lumpy | **1.043** | 1.060 | 1.079 |
   | intermittent | 0.426 | 0.435 | **0.425** |

   `intermittent` — 122.006 baris, 72% targetnya nol — dimenangi LSTM dengan
   selisih tipis atas RF. Itu segmen tempat "membaca 28 hari mentah" paling
   masuk akal membeli sesuatu: ringkasan `roll_mean_7` dari deret yang
   sebagian besar nol membuang persis informasi yang penting, yaitu *kapan*
   nol-nol itu terputus.

Yang jelas berpihak pada LSTM dan tidak setipis skor pinball:

- **Ukuran model. 466 KB.** Sepuluh kali lebih kecil dari booster XGBoost (4,7
  MB) dan **1.720× lebih kecil** dari forest (821 MB), karena yang disimpan
  hanya bobot jaringan, scaler, dan tabel embedding — bukan struktur pohon,
  apalagi nilai target di tiap daun.
- **MAE terbaik di potongan bersih** (14.278), sambil kalibrasinya paling dekat
  ke 0.90 (0.902).

Kesimpulan yang bisa ditopang angka ini: **ketiga model praktis setara pada
pinball@0.9, dengan XGBoost unggul konsisten-tapi-tipis, dan LSTM menempati
posisi ketiga dengan jarak yang lebih kecil daripada ketidakpastian
pengukurannya sendiri.** Menyatakan LSTM lebih buruk — dengan 12 undian lawan
30, dua dimensi kapasitas yang tidak pernah diuji, dan satu seed tanpa
pengulangan — belum bisa ditopang. Yang bisa dikatakan dengan yakin: LSTM tidak
membeli lompatan akurasi yang membenarkan ongkos trainingnya (3 jam 6 menit
walk-forward lawan 16 menit XGBoost), tapi ia membeli model yang 1.720× lebih
kecil dengan akurasi yang tidak bisa dibedakan.

## 7. Model final

`lstm.fit_final()` menjalankan protokol dua fit yang sama pada seluruh baris
layak sebelum Desember, dipotong di batas Desember oleh
`purging.lookahead_safe_mask()` — populasi baris yang sama persis dengan yang
dinilai di atas. Jendelanya tetap dipotong dari panel penuh, karena baris
konteks di luar himpunan layak tetap riwayat yang sah (§2).

| | |
|---|---|
| Baris training | 1.349.011 |
| Kolom | 56 — 49 dinamis lewat LSTM, 7 kategorikal lewat embedding |
| `best_epoch` | 8 |
| Kuantil | 0.9 |
| Wall time | 2.603 s (43 menit) |
| File | `models/lstm_q90.joblib` — 466 KB, dibuat 20 Agu 2026 |

`best_epoch = 8` duduk di tengah rentang yang dipilih kelima fold (6–10),
yang berarti mekanisme early stopping berperilaku konsisten saat data training
bertambah — bukan kebetulan per fold.

Bundle-nya menyimpan `state_dict`, urutan kolom training (dinamis dan
kategorikal terpisah), ukuran embedding, **scaler**, flag `log_target`,
`lookback`, dan `best_epoch`. Scaler wajib ikut: jaringan yang dimuat ulang lalu
diberi fitur berskala mentah tidak gagal — ia memprediksi dengan percaya diri
dari input yang salah skala.

Satu perbedaan pemakaian dibanding dua model lain: `predict_bundle(bundle,
panel, frame)` **mewajibkan panel**, bukan menerimanya sebagai opsi. LSTM tidak
bisa memprediksi dari satu baris sendirian — ia butuh 28 hari di belakangnya.

## 8. Reproduksi

```bash
.venv/bin/python3 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebook/modeling_lstm.ipynb
```

Butuh belasan jam (§3). Pencarian menulis checkpoint tiap kandidat selesai ke
`lstm_search_results.csv` dan melanjutkan dari sana kalau dijalankan ulang —
run yang menghasilkan dokumen ini memang dijalankan berkali-kali dari
checkpoint, dan itu satu-satunya alasan ia selesai sama sekali.

Dependensi barunya satu baris di `requirements.txt`: `torch==2.8.0`.

**Bagaimana angka §5–§7 sebenarnya dijalankan**: cell 9 dan 10 dieksekusi
headless lewat skrip terpisah, bukan lewat nbconvert atas seluruh notebook.
Cell benchmark sengaja tidak dijalankan ulang — nilainya sudah terukur dan
tercatat (§3), dan mengukur ulang akan menarik `N_CANDIDATES` dari `best_epoch`
yang baru, sehingga N yang berbeda akan membatalkan checkpoint 12 kandidat yang
sudah selesai atau menambah kandidat yang anggarannya tidak pernah dibayar.
Menjalankan notebook itu utuh dari nol akan menghasilkan angka yang sama selama
benchmark-nya mendarat di `best_epoch = 3` lagi.

| Artefak | Lokasi | Masuk git |
|---|---|---|
| Hasil pencarian | `dataset/model_ready/lstm_search_results.csv` | tidak |
| Parameter terpilih | `dataset/model_ready/lstm_best_params.json` | tidak |
| Tabel hasil lengkap | `dataset/model_ready/lstm_walk_forward_results.csv` | tidak |
| Jaringan terlatih | `models/lstm_q90.joblib` | tidak |
| Ringkasan ini | `docs/hasil-modeling-lstm.md` | **ya** |

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
  bukan out-of-sample terhadap seleksi. Potongan fold 1/2/4 di §5.1 adalah
  angka yang bersih.
- **Kedalaman tidak pernah diuji.** `num_layers=2` dan `hidden_size=256`
  dibuang dari ruang pencarian karena ongkos wall-clock sebelum kandidat
  pertama ditarik (§3, §4). Pertanyaan "apakah LSTM yang lebih dalam menang"
  tidak dijawab run ini, dan menjawabnya butuh plafon waktu baru — bukan
  pembacaan ulang tabel yang sama.
- **Satu seed, tanpa pengulangan.** Setiap kandidat dilatih sekali dengan
  `random_state=42`. LSTM satu-satunya dari ketiga model yang hasilnya
  bergantung pada inisialisasi acak dan urutan batch, jadi selisih 0,7–0,9%
  terhadap dua model lain (§6) tidak bisa dipisahkan dari varians seed. Mengukur
  itu butuh 3–5 pengulangan per konfigurasi, yang tidak muat di plafon yang
  sama.
- **Anggaran pencariannya paling kecil dari ketiga model** — 12 kandidat lawan
  30 dan 18 — dan itu berpihak melawan LSTM di §6. Menyamakannya butuh run
  ulang.
- **Plafon 8 jam terlampaui.** Anggaran diturunkan dari `best_epoch = 3` yang
  terukur di benchmark, sementara kandidat sebenarnya berhenti di epoch 3–13,
  sehingga ongkos riilnya sekitar 11,4 jam (§3). `candidate_budget()` bekerja
  sesuai rumusnya; yang salah adalah asumsi bahwa `best_epoch` benchmark
  mewakili ruang pencarian. Kalau mekanisme ini dipakai lagi, angka epoch yang
  disuntikkan sebaiknya yang paling pesimistis, bukan yang terukur di satu
  konfigurasi.
- **Coverage bergoyang antar fold** (0.890–0.925) tanpa arah yang konsisten,
  berbeda dari RF yang konservatif merata dan XGBoost yang menurun rapi. Dua
  fold mendarat di bawah target 0.90. Layak diperiksa lagi saat test set
  Desember dibuka.
- **Ketiga model sudah dijalankan, rekomendasi final belum diambil.** Angka di
  §6 menunjukkan ketiganya praktis setara pada kriteria, jadi pemilihan model
  produksi adalah keputusan yang harus menimbang hal di luar pinball — ongkos
  training, ukuran artefak, kemudahan pemeliharaan — dan itu belum dituliskan
  di mana pun.
