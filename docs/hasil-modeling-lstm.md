# Hasil Modeling — LSTM multi-kuantil

Angka terukur dari jalannya `notebook/modeling_lstm.ipynb`, selesai **28
Agustus 2026**, run pertama di bawah kriteria multi-kuantil (butir 0c
`docs/todolist-proyek.md`). Desainnya ada di
`docs/superpowers/specs/2026-08-19-lstm-modeling-design.md` dan
`docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`; dokumen
ini hanya memuat hasilnya.

**Dokumen ini menggantikan versi kuantil-tunggal**, diarsipkan sebagai
`docs/bak/hasil-modeling-lstm.single-quantile.bak.md`. Angka di kedua dokumen
**bukan besaran yang sama** (T-10, `docs/todolist-proyek.md`): yang lama
pinball@0,9 pada satu titik kuantil, yang ini K1 — rata-rata pinball lintas 19
titik `QUANTILE_SET_A`. **Peringkat kandidat tidak bisa dibandingkan lintas
kedua run** untuk LSTM secara khusus — beda dari RF dan XGBoost — karena ruang
pencariannya sendiri berubah (dipulihkan ke 144 titik, butir 0b
`docs/todolist-proyek.md`), bukan cuma kriterianya. Lihat bagian 4.3.

**Mesin berbeda untuk pencarian vs walk-forward**, sama seperti XGBoost:
pencarian 30 kandidat + pengulangan 3 seed berjalan di **GPU Windows**
(`device=cuda`, keputusan 2026-08-26), walk-forward final dan fit final di
**CPU Mac** — satu mesin yang sama dengan RF dan XGBoost, supaya K3 tetap
sebanding. MPS (GPU Apple Silicon/Metal) **tidak dipakai untuk tahap manapun**:
diukur kalah 2× dari CPU di mesin ini (0,392 vs 0,193 detik/batch), keputusan
yang sudah berlaku sejak sebelum migrasi K1.

**Desember 2025 tidak dibuka.** Semua angka di bawah datang dari walk-forward
lima fold di Juli–November 2025. Test set final masih terkunci.

## 1. Ringkasan

LSTM melewati gerbang G0 dan mencetak **K1 = 2,8818** pada potongan fold
bersih (1/2/4), lawan **4,8603** milik baseline terbaik `naive_roll_mean_7` —
**41% lebih baik**. Pada gabungan kelima fold angkanya 2,8828 lawan 4,8231.

| model | K1 (fold 1/2/4) | K1 (5 fold) |
|---|---:|---:|
| **lstm** | **2,8818** | **2,8828** |
| `naive_roll_mean_7` | 4,8603 | 4,8231 |
| `naive_lag_1` | 8,1612 | 8,1755 |
| `naive_zero` | 14,8102 | 14,7469 |

Dibandingkan Random Forest (`docs/hasil-modeling-rf.md`, K1 fold bersih
2,8508), LSTM **seed 42** kalah tipis — 0,0310 (1,1%). Tapi **jarak ini
terbukti tidak stabil** (bagian 5.1b, diuji 2026-08-30): diulang dengan seed
43 pada walk-forward 5-fold penuh yang sama, K1 LSTM di fold 1/2/4 melompat
ke **3,0732** — kalah **7,80%** dari RF, lebih buruk bahkan dari XGBoost
(2,9433). Cuma ganti seed, hasilnya berubah dari "hampir seri" jadi "kalah
tegas dari kedua model lain". LSTM tetap **menang MAE@0,9** di seed 42
(14,065 lawan 15,055 milik RF), tapi metrik itu pun ikut goyang oleh seed
yang sama.

**Jarak ke RF (seed 42) lebih kecil daripada derau antar-seed LSTM sendiri —
dan sekarang dikonfirmasi langsung, bukan cuma diproyeksikan.** Tiga seed
(42/43/44) di fold 3&5 saja mencetak K1 = 2,8617 / 3,0915 / 2,8399 — rentang
0,2517. Walk-forward 5-fold penuh dengan seed 43 mengonfirmasi pola yang
sama pada fold yang benar-benar dipakai untuk klaim K1: 2,8818 → 3,0732,
selisih 0,1914 (6,6%) — jauh melebihi ambang keputusan 2%. **Keunggulan atau
kekalahan LSTM terhadap RF pada run seed tunggal (42) yang selama ini dipakai
di seluruh dokumen tidak bisa dipercaya sebagai representasi LSTM secara
umum** — itu satu titik dari sebaran yang lebar, dan titik itu kebetulan
berada di ujung yang menguntungkan LSTM. Rinciannya bagian 5.1b.

**`crossing_rate` = 0,4345 (43,4% baris)** di walk-forward final — di antara
RF (0% struktural) dan XGBoost (97,7%,
`docs/hasil-modeling-xgb.md` bagian 5.2). Sama seperti XGBoost, ini belum
dijelaskan dan belum boleh diabaikan; lihat bagian 5.2 dan catatan 🆕 di
`docs/todolist-proyek.md` bagian 0d.

Di τ=0,9 — titik yang dijanjikan ke bisnis (B-9) — coverage 0,906 terhadap
target 0,90 (RF: 0,928, XGBoost: 0,902), dengan fill rate 0,955. Sisi
bisnisnya, di 345.547 baris validasi:

| | kekurangan (shortfall) | kelebihan (overstock) |
|---|---:|---:|
| `lstm` | 461.320 | 4.351.815 |
| `random_forest` | 418.250 | 4.793.038 |
| `xgboost` | 500.579 | 4.132.651 |
| `naive_roll_mean_7` | 1.528.393 | 1.804.789 |

> Baris `shortfall`/`overstock` dijumlahkan dari tabel per-fold bagian 5.1.
> Catatan satuan sama seperti dokumen RF/XGBoost: unit dijumlahkan lintas SKU
> bersatuan campur, sah untuk membandingkan model pada baris yang sama, tapi
> tidak punya makna fisik sebagai satu besaran tunggal.

LSTM ada **di tengah** RF dan XGBoost di kedua sisi tukar-menukar shortfall
vs overstock, konsisten dengan coverage-nya (0,906) yang juga di tengah
(RF 0,928, XGBoost 0,902).

## 2. Setup evaluasi

| | |
|---|---|
| Data | `dataset/model_ready/model_input.parquet` — 1.502.522 baris panel, 82 kolom, 1 Jan 2024 – 31 Des 2025 |
| Fitur | 56 kolom (`modeling_prep.FEATURE_COLS`), dibentuk jadi jendela sekuens lewat `sequence_windows.py` |
| Target | `target_lead_time_cumulative` — sama populasi baris dengan RF/XGBoost |
| Fold | 5 expanding window, validasi Juli, Agustus, September, Oktober, November 2025 |
| Test terkunci | Desember 2025 (`TEST_START = 2025-12-01`) |
| Baris validasi | 345.547 total — identik dengan RF/XGBoost |
| Kuantil | `QUANTILE_SET_A` — 19 titik, 0,05 sampai 0,95 langkah 0,05, satu kepala keluaran per titik |
| Kriteria | K1 = rata-rata tak berbobot pinball lintas 19 titik, pada potongan fold 1/2/4 |
| Implementasi | LSTM PyTorch, loss = jumlah pinball lintas 19 kuantil, `torch 2.8.0` |
| Device | pencarian & 3-seed repeat: `cuda` (GPU Windows); walk-forward & fit final: `cpu` (Mac, MPS diukur kalah 2×) |
| Commit | pencarian: `e074421` |

Ketiga baseline naive dinilai pada **baris yang identik** dengan LSTM — sama
seperti RF/XGBoost.

LSTM satu-satunya dari ketiga model yang inisialisasi bobotnya **acak**
(forest dan boosting-nya deterministik pada `random_state` tetap), jadi
selisih K1 kecil antar model tidak bisa dipisahkan dari derau seed tanpa
pengulangan — itulah alasan bagian 5.1b ada khusus untuk LSTM dan tidak ada
padanannya di dokumen RF/XGBoost.

## 3. Benchmark

Satu putaran latih-prediksi di fold 5 dengan `DEFAULT_PARAMS`, **di CPU Mac**.

| | |
|---|---|
| Baris training fold 5 | 1.292.778 |
| Baris validasi fold 5 | 59.629 |
| `best_epoch` | 2 |
| `sec_per_epoch` | 106,8 detik |
| Wall time | 961,1 detik (~16,0 menit) |
| Peak RSS proses | 5,93 GB |
| Prediksi di τ=0,9 | rata-rata 43,75, maksimum 1.622,45 |
| `crossing_rate` | 0,1785 |

MPS diprobe terpisah (bukan di run ini) dan sudah diukur kalah 2× dari CPU
(0,392 vs 0,193 detik/batch) — probe itu tidak diulang di sini karena
kesimpulannya tidak berubah.

`sec_per_epoch` = 106,8 detik dipakai `candidate_budget()` untuk menghitung
anggaran pencarian — bukan tebakan (komentar notebook: *"N datang dari angka
benchmark, bukan dari tebakan"*). Plafon 8 jam menghasilkan N=14 kandidat
kalau dipatok ke plafon; anggaran yang benar-benar dipakai tetap **30**,
dipatok setara XGBoost.

## 4. Pencarian hyperparameter

30 kandidat pada ruang **144 kombinasi**
(`hidden_size × num_layers × dropout × learning_rate × batch_size ×
log_target`), dinilai di **fold 3 (September) dan fold 5 (November)** dengan
kriteria K1 gabungan, seed 42 — di **GPU Windows**. Semua 30 kandidat selesai
(0 gagal); `device=cuda`, `commit=e074421` di seluruh baris.

### 4.1 Tabel lengkap

| # | hidden | layers | dropout | lr | batch | log_target | K1 | mae@0,9 | cov@0,9 | crossing | detik |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| **21** | 256 | 2 | 0,0 | 0,0003 | 2048 | True | **2,8617** | 14,443 | 0,894 | 0,442 | 940 |
| 12 | 64 | 2 | 0,0 | 0,0003 | 2048 | True | 2,8963 | 14,747 | 0,898 | 0,475 | 692 |
| 5 | 128 | 2 | 0,0 | 0,0003 | 1024 | True | 2,9151 | 15,527 | 0,912 | 0,422 | 590 |
| 13 | 64 | 2 | 0,0 | 0,0010 | 2048 | False | 2,9242 | 13,563 | 0,887 | 0,380 | 405 |
| 23 | 256 | 2 | 0,0 | 0,0010 | 1024 | False | 2,9261 | 13,540 | 0,889 | 0,332 | 500 |
| 29 | 128 | 1 | 0,2 | 0,0003 | 2048 | False | 2,9359 | 14,722 | 0,890 | 0,218 | 641 |
| 6 | 256 | 2 | 0,0 | 0,0010 | 2048 | False | 2,9372 | 14,178 | 0,919 | 0,251 | 566 |
| 17 | 256 | 1 | 0,0 | 0,0003 | 1024 | True | 2,9394 | 16,499 | 0,912 | 0,443 | 637 |
| 24 | 256 | 1 | 0,2 | 0,0010 | 2048 | False | 2,9406 | 14,203 | 0,901 | 0,188 | 250 |
| 25 | 128 | 1 | 0,0 | 0,0003 | 1024 | False | 2,9417 | 14,610 | 0,910 | 0,250 | 412 |
| 0 | 256 | 1 | 0,0 | 0,0010 | 1024 | False | 2,9497 | 14,437 | 0,910 | 0,318 | 351 |
| 7 | 128 | 1 | 0,3 | 0,0010 | 1024 | False | 2,9506 | 15,346 | 0,897 | 0,183 | 272 |
| 3 | 128 | 1 | 0,0 | 0,0010 | 2048 | False | 2,9581 | 14,660 | 0,910 | 0,195 | 269 |
| 26 | 256 | 1 | 0,0 | 0,0003 | 1024 | False | 2,9661 | 13,879 | 0,906 | 0,213 | 548 |
| 22 | 128 | 1 | 0,0 | 0,0010 | 1024 | False | 2,9677 | 14,636 | 0,905 | 0,271 | 247 |
| 8 | 64 | 1 | 0,3 | 0,0010 | 1024 | False | 2,9765 | 15,106 | 0,900 | 0,220 | 293 |
| 28 | 64 | 2 | 0,3 | 0,0010 | 1024 | False | 3,0126 | 15,577 | 0,896 | 0,138 | 213 |
| 2 | 64 | 1 | 0,0 | 0,0003 | 1024 | False | 3,0287 | 15,244 | 0,891 | 0,353 | 363 |
| 10 | 64 | 1 | 0,2 | 0,0010 | 2048 | False | 3,0459 | 14,831 | 0,883 | 0,252 | 333 |
| 18 | 256 | 1 | 0,2 | 0,0010 | 2048 | True | 3,0495 | 19,932 | 0,915 | 0,258 | 405 |
| 20 | 64 | 2 | 0,2 | 0,0010 | 2048 | False | 3,0810 | 15,259 | 0,897 | 0,339 | 446 |
| 11 | 128 | 1 | 0,0 | 0,0010 | 1024 | True | 3,0883 | 16,828 | 0,906 | 0,483 | 469 |
| 16 | 64 | 1 | 0,0 | 0,0010 | 2048 | False | 3,1107 | 14,904 | 0,895 | 0,369 | 322 |
| 4 | 128 | 2 | 0,2 | 0,0003 | 2048 | False | 3,1241 | 15,005 | 0,904 | 0,192 | 1.050 |
| 19 | 256 | 1 | 0,3 | 0,0010 | 1024 | True | 3,1431 | 20,790 | 0,912 | 0,303 | 446 |
| 9 | 128 | 1 | 0,2 | 0,0010 | 1024 | True | 3,2085 | 20,865 | 0,909 | 0,274 | 487 |
| 15 | 256 | 2 | 0,3 | 0,0010 | 1024 | True | 3,2504 | 22,473 | 0,909 | 0,374 | 840 |
| 27 | 64 | 2 | 0,2 | 0,0003 | 1024 | True | 3,2515 | 24,919 | 0,925 | 0,282 | 502 |
| 14 | 128 | 1 | 0,3 | 0,0003 | 2048 | True | 3,3769 | 24,687 | 0,924 | 0,125 | 661 |
| 1 | 64 | 1 | 0,3 | 0,0003 | 1024 | True | 3,4575 | 28,466 | 0,907 | 0,263 | 909 |

Sumber: `dataset/model_ready/lstm_search_results.csv`. Kolom "detik" adalah
`elapsed_seconds` di GPU Windows.

### 4.2 Yang terbaca dari sebarannya

1. **Ruang parameternya jauh lebih curam daripada RF/XGBoost.** Rentang K1
   keseluruhan 2,8617–3,4575 (**20,8%**) — lebih dari 3× rentang XGBoost
   (6,8%) dan hampir 2× rentang RF (11%). Bahkan lima kandidat teratas
   berjarak **2,25%** satu sama lain — jauh lebih lebar daripada RF (1,46%)
   dan XGBoost (0,50%). Hyperparameter LSTM jauh lebih menentukan hasil
   akhirnya di dataset ini.
2. **`dropout=0` konsisten terbaik** (rata-rata K1 0,0: 2,9607; 0,2: 3,0796;
   0,3: 3,1668) — regularisasi tambahan justru merugikan, masuk akal untuk
   model yang sudah dibatasi anggaran epoch oleh early stopping ketat.
3. **`hidden_size` monoton**: 64 → 3,0785, 128 → 3,0467, 256 → 2,9964 —
   lebih besar lebih baik pada rentang yang diuji, sama arahnya dengan
   `max_depth` di XGBoost.
4. **`log_target=False` menang secara agregat** (2,9876 lawan 3,1199 untuk
   `True`) — **berlawanan** dengan pemenangnya sendiri (kandidat 21 memakai
   `log_target=True`). Bukan kontradiksi: kandidat 21 unggul karena kombinasi
   spesifiknya (`hidden_size=256, dropout=0, lr=0,0003`), bukan karena
   `log_target=True` secara umum menguntungkan — interaksi antar parameter di
   LSTM lebih kental daripada efek utama tunggal mana pun, konsisten dengan
   rentang yang jauh lebih lebar di poin 1.
5. **`crossing_rate` sudah bervariasi lebar di tahap pencarian** (0,125–0,483)
   — jauh dari nol di seluruh 30 kandidat, tapi juga jauh dari tinggi
   seragam seperti XGBoost. Tidak ada korelasi jelas dengan K1 di tabel di
   atas (kandidat terbaik #21 crossing 0,442; kandidat gagal #1 crossing
   0,263 — tidak monoton).

Parameter terpilih (`dataset/model_ready/lstm_best_params.json`):

```json
{
  "batch_size": 2048,
  "dropout": 0.0,
  "grad_clip": 1.0,
  "hidden_size": 256,
  "learning_rate": 0.0003,
  "log_target": true,
  "num_layers": 2,
  "random_state": 42
}
```

### 4.3 Apakah peringkat kandidat berubah setelah pindah ke K1?

**Tidak bisa dijawab dengan cara yang sama seperti RF/XGBoost.** Kedua model
itu punya ruang pencarian dan seed identik terhadap run kuantil-tunggal lama,
jadi `candidate_id` yang sama berarti konfigurasi yang sama. Untuk LSTM,
**ruang pencariannya sendiri berubah** — butir 0b `docs/todolist-proyek.md`
mencatat eksplisit: *"Ruang pencarian LSTM dipulihkan ke 144 + protokol 3
seed"*. Verifikasi langsung dari data:

| | |
|---|---|
| Baris di `lstm_search_results.single-quantile.bak.csv` | 12 |
| Baris di `lstm_search_results.csv` (run ini) | 30 |
| `candidate_id` yang parameternya sama di kedua berkas | 0 dari 12 yang tumpang tindih |

Menghitung Spearman/Kendall pada `candidate_id` yang tumpang tindih di sini
**akan menyesatkan** — angkanya akan membandingkan konfigurasi hyperparameter
yang berbeda seolah itu perbandingan kriteria pada objek yang sama, persis
kesalahan yang dihindari pemeriksaan kolom-demi-kolom di RF dan XGBoost.
**Tidak dilaporkan angkanya di sini** karena tidak valid, bukan karena belum
dihitung — sudah dicoba dan hasilnya dibuang.

Konsekuensinya: **tidak ada bukti langsung** untuk atau melawan hipotesis
bahwa migrasi K1 menggeser peringkat kandidat LSTM. Yang **bisa** dikatakan
tidak langsung: rentang K1 antar kandidat LSTM (20,8%, poin 1 di atas) jauh
lebih lebar daripada RF (11%) dan XGBoost (6,8%), dan LSTM juga satu-satunya
model dengan kepala keluaran multi-titik yang bersaing langsung untuk
kapasitas model yang sama — dua alasan struktural (dicatat sejak
`docs/hasil-modeling-rf.md` bagian 4.3) untuk menduga interaksi dengan jumlah
titik kuantil di LSTM setidaknya sebesar XGBoost, kemungkinan lebih besar.

## 5. Hasil walk-forward

Pemenang (kandidat 21) dijalankan ulang di kelima fold, di CPU Mac.

### 5.0 Gerbang G0

> Model harus mengalahkan `naive_roll_mean_7` pada pinball@0,9 **di kelima
> fold**, bukan hanya di gabungan.

pinball@0,9 per fold:

| model | 1 (Jul) | 2 (Agu) | 3 (Sep) | 4 (Okt) | 5 (Nov) |
|---|---:|---:|---:|---:|---:|
| **lstm** | **2,2869** | **2,4201** | **2,3158** | **2,7449** | **2,5735** |
| `naive_roll_mean_7` | 4,2489 | 4,5665 | 4,0341 | 4,7826 | 4,9703 |
| `naive_lag_1` | 8,3545 | 8,4686 | 8,0448 | 8,5260 | 8,3717 |
| `naive_zero` | 26,4535 | 27,2539 | 23,8486 | 26,2187 | 29,3199 |

**G0 lolos.** LSTM menang di kelima fold dengan margin 43–48%.

### 5.1 Per fold

K1 per fold:

| model | 1 (Jul) | 2 (Agu) | 3 (Sep) | 4 (Okt) | 5 (Nov) |
|---|---:|---:|---:|---:|---:|
| **lstm** | **2,7058** | **2,8743** | **2,7206** | **3,0791** | **3,0780** |
| `naive_roll_mean_7` | 4,6636 | 4,9814 | 4,6680 | 4,9372 | 4,8719 |
| `naive_lag_1` | 7,9549 | 8,5327 | 8,1080 | 7,9712 | 8,3067 |
| `naive_zero` | 14,6964 | 15,1410 | 13,2492 | 14,5660 | 16,2888 |

Detail LSTM per fold (kolom @0,9 kecuali K1):

| fold | bulan | n | K1 | mae | pinball | coverage | fill_rate | shortfall | overstock | `best_epoch` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Jul 2025 | 72.684 | 2,7058 | 15,644 | 2,2869 | 0,9127 | 0,9693 | 65.644 | 1.071.393 | 6 |
| 2 | Agu 2025 | 75.015 | 2,8743 | 12,434 | 2,4201 | 0,9184 | 0,9514 | 110.334 | 822.438 | 11 |
| 3 | Sep 2025 | 70.503 | 2,7206 | 14,708 | 2,3158 | 0,8962 | 0,9601 | 74.471 | 962.493 | 5 |
| 4 | Okt 2025 | 67.716 | 3,0791 | 14,178 | 2,7449 | 0,9121 | 0,9431 | 112.337 | 847.731 | 8 |
| 5 | Nov 2025 | 59.629 | 3,0780 | 12,516 | 2,5735 | 0,8878 | 0,9493 | 98.535 | 647.760 | 9 |

K1 bergerak 2,721–3,079 (rentang 13,2%), sebanding stabilnya dengan RF (13%)
dan XGBoost (12%). `best_epoch` bervariasi 5–11 antar fold — early stopping
memang menghentikan pelatihan jauh sebelum plafon, konsisten dengan
`best_epoch=2` yang sudah terlihat di benchmark `DEFAULT_PARAMS` (bagian 3).

Dipotong ke fold 1, 2, dan 4 saja — tiga fold yang tidak menyentuh seleksi,
dan potongan yang menjadi kriteria K1 resmi:

| model | n | K1 | mae@0,9 | cov@0,9 | fill@0,9 |
|---|---:|---:|---:|---:|---:|
| **lstm** | 215.415 | **2,8818** | 14,065 | 0,915 | 0,955 |
| `naive_roll_mean_7` | 215.415 | 4,8603 | 9,721 | 0,696 | 0,850 |
| `naive_lag_1` | 215.415 | 8,1612 | 16,322 | 0,653 | 0,712 |
| `naive_zero` | 215.415 | 14,8102 | 29,620 | 0,423 | 0,000 |

2,8818 di fold bersih hampir sama dengan 2,8828 di gabungan 5 fold — beda
arah dari RF (fold bersih lebih baik) dan XGBoost (fold bersih lebih buruk),
LSTM di sini nyaris tidak bergerak. Konsisten dengan seleksinya berasal dari
fold 3&5 yang levelnya (2,7206 dan 3,0780) berada di kedua ujung rentang
5-fold, bukan di salah satu ekstrem.

### 5.1b Derau antar-seed

Konfigurasi pemenang (kandidat 21) diulang pada seed 42/43/44, dinilai di
fold pencarian yang sama (3 dan 5) — **bukan** walk-forward 5-fold penuh
(pengulangan 5-fold untuk 3 seed terlalu mahal untuk anggaran saat ini).

| seed | K1 (fold 3&5) | mae@0,9 | cov@0,9 | crossing_rate | `best_epoch` per fold | detik |
|---:|---:|---:|---:|---:|---|---:|
| 42 | 2,8617 | 14,443 | 0,894 | 0,442 | 5, 9 | 917 |
| 43 | 3,0915 | 16,488 | 0,909 | 0,462 | 6, 12 | 1.113 |
| 44 | 2,8399 | 13,873 | 0,896 | 0,479 | 7, 10 | 1.062 |

| | |
|---|---|
| K1 minimum | 2,8399 (seed 44) |
| K1 rata-rata | 2,9310 |
| K1 maksimum | 3,0915 (seed 43) |
| **Rentang** | **0,2517** |
| Selisih seed-42 vs baris pemenang di search | +0,000000 (konsisten — cek nondeterminisme lolos) |

Rentang 0,2517 ini **delapan kali** lebih lebar daripada jarak K1 LSTM ke RF
(0,0310, bagian 1). Baca literalnya: kalau LSTM dilatih ulang dengan seed
berbeda tanpa mengubah apa pun lagi, hasilnya bisa saja mendarat di mana saja
dari "mengalahkan RF" (2,8399 < 2,8508) sampai "kalah jauh dari XGBoost"
(3,0915 > 2,9433). **Run tunggal (seed 42, dipakai di seluruh dokumen ini)
tidak punya kekuatan statistik untuk menyatakan LSTM lebih baik atau lebih
buruk dari RF** — ini bukan detail teknis, ini batasan utama yang harus
dibawa ke bagian 16/18 metodologi sebelum peringkat lintas model ditetapkan.

Catatan yang **sudah tidak berlaku, ditulis semula di sini**: seed 43 sempat
dibaca sebagai "anomali fold 3&5, bukan representatif dari performa LSTM
secara umum". **Itu keliru** — lihat konfirmasi walk-forward 5-fold penuh di
bawah, yang menunjukkan pola seed 43 bertahan persis di fold yang dipakai
sebagai kriteria resmi.

**Konfirmasi walk-forward 5-fold penuh, seed 43 — dikerjakan 2026-08-30.**
Fold 3&5 di atas hanya potongan pencarian; untuk tahu apakah goyangannya juga
muncul di fold 1/2/4 (fold yang benar-benar dipakai K1 resmi), walk-forward
5-fold penuh diulang dengan `random_state=43` (`lstm_seed_walkforward.py`,
~9,8 jam CPU Mac):

| | seed 42 (resmi) | seed 43 |
|---|---:|---:|
| K1 (5 fold) | 2,8828 | 3,0805 |
| K1 (fold 1/2/4, bersih) | **2,8818** | **3,0732** |
| `best_epoch` per fold | 6, 11, 5, 8, 9 | 8, 13, 6, 10, 12 |

Selisih K1 fold-bersih: **0,1914 (6,6%)** — pada fold yang sama persis,
konfigurasi yang sama persis, hanya `random_state` yang beda. Ini **jauh**
melebihi ambang keputusan K1 (2%, bagian 17 metodologi). Dengan seed 43,
LSTM kalah dari RF **7,80%** dan bahkan lebih buruk dari XGBoost (3,0732 vs
2,9433) — bukan lagi "hampir seri", tapi model terburuk dari ketiganya di
titik data ini. `best_epoch` juga naik di seluruh fold dibanding seed 42,
konsisten dengan konvergensi yang kurang baik pada inisialisasi ini.

**Bacaan yang benar sekarang**: dua data point (n=2) tidak cukup untuk
interval kepercayaan yang ketat, tapi cukup untuk menunjukkan bahwa K1 seed
42 (2,8818, dipakai di seluruh dokumen ini dan di bagian 1.6/1.8
`docs/detail-tahap-perbandingan-model.md`)
**bukan** representasi LSTM yang stabil — ia mendarat di ujung yang
menguntungkan dari sebaran yang lebar. Rata-rata dua seed (2,9775) juga tetap
kalah dari RF melebihi ambang 2%. Konsekuensinya untuk bagian 1.6/1.8 ada di
`docs/detail-tahap-perbandingan-model.md`.

### 5.2 K2 — kalibrasi di seluruh 19 titik kuantil, dan `crossing_rate`

Gabungan kelima fold, dibobot jumlah baris. Lantai `share_nol` = **0,4195**
(sama seperti RF/XGBoost — populasi baris identik). Metodologi lantai
dibahas penuh di `docs/hasil-modeling-rf.md` bagian 5.2.

| τ | pinball | coverage | gap (cov − τ) | lantai | kelebihan di atas lantai |
|---:|---:|---:|---:|---:|---:|
| 0,05 | 0,7432 | 0,4325 | +0,3825 | 0,4195 | +0,0130 |
| 0,10 | 1,3202 | 0,4502 | +0,3502 | 0,4195 | +0,0307 |
| 0,15 | 1,8074 | 0,4689 | +0,3189 | 0,4195 | +0,0494 |
| 0,20 | 2,2401 | 0,4888 | +0,2888 | 0,4195 | +0,0693 |
| 0,25 | 2,6212 | 0,5119 | +0,2619 | 0,4195 | +0,0924 |
| 0,30 | 2,9281 | 0,5312 | +0,2312 | 0,4195 | +0,1117 |
| 0,35 | 3,2235 | 0,5561 | +0,2061 | 0,4195 | +0,1366 |
| 0,40 | 3,4301 | 0,5799 | +0,1799 | 0,4195 | +0,1604 |
| 0,45 | 3,6209 | 0,6070 | +0,1570 | 0,45 | +0,1570 |
| 0,50 | 3,7487 | 0,6329 | +0,1329 | 0,50 | +0,1329 |
| 0,55 | 3,8128 | 0,6629 | +0,1129 | 0,55 | +0,1129 |
| 0,60 | 3,8198 | 0,6947 | +0,0947 | 0,60 | +0,0947 |
| 0,65 | 3,7941 | 0,7246 | +0,0746 | 0,65 | +0,0746 |
| 0,70 | 3,6841 | 0,7589 | +0,0589 | 0,70 | +0,0589 |
| 0,75 | 3,5206 | 0,7918 | +0,0418 | 0,75 | +0,0418 |
| 0,80 | 3,2813 | 0,8271 | +0,0271 | 0,80 | +0,0271 |
| 0,85 | 2,9624 | 0,8644 | +0,0144 | 0,85 | +0,0144 |
| **0,90** | **2,4609** | **0,9062** | **+0,0062** | 0,90 | +0,0062 |
| 0,95 | 1,7535 | 0,9498 | -0,0002 | 0,95 | -0,0002 |

**Bacaan sama bentuknya dengan RF dan XGBoost** — over-coverage memuncak di
paruh bawah grid, mengecil ke τ tinggi. **Levelnya di antara keduanya di
τ=0,90–0,95**: kelebihan di atas lantai τ=0,90 (+0,0062) berada di antara
XGBoost (+0,0022) dan RF (+0,0281); di τ=0,95 LSTM praktis pas di lantai
(-0,0002), sedikit lebih dekat ke nol daripada RF (+0,0108) dan XGBoost
(-0,0075).

**`crossing_rate` = 0,4345** di seluruh baris walk-forward. Per fold:
0,332–0,483 — bervariasi lebih lebar antar fold daripada XGBoost (0,966–0,987,
sempit karena sudah dekat batas atas), konsisten dengan pola pencarian bagian
4.2 poin 5 (crossing_rate tidak stabil di 30 kandidat LSTM).

**Diuji 2026-08-29 — hampir seluruhnya derau numerik, bukan defek
struktural.** Crossing dihitung ulang dari bundle tersimpan (tanpa retrain)
dengan toleransi jarak minimum, sama metodologi dengan
`docs/hasil-modeling-xgb.md` bagian 5.2:

| toleransi gap | crossing_rate |
|---:|---:|
| 0 (definisi resmi) | 0,459* |
| 0,01 | **0,088** |
| 0,1 | **0,011** |
| 0,5 | 0,007 |
| 1,0 | 0,005 |
| 5,0 | 0,001 |

\*sedikit di bawah 0,4345 karena ini prediksi dari model final gabungan,
bukan 5 model per-fold walk-forward.

Berbeda dari XGBoost: rate-nya **ambruk** begitu diberi toleransi sekecil
0,01 (46% → 8,8%), dan di toleransi 0,1 tinggal 1,1%. Median besar inversinya
**0,0027 unit** — nyaris nol, jauh lebih kecil dari median XGBoost (0,043).
Kepala keluaran 19-neuron LSTM memang tidak dipaksa monoton secara arsitektur
(hipotesis 2 sama-sama berlaku secara teori), tapi **secara empiris hasilnya
hampir seluruhnya derau angka mengambang, bukan kesalahan urutan yang
berarti** — hipotesis 1 (efek ikatan/near-tie) yang terbukti di sini, arah
yang berlawanan dengan XGBoost. **`crossing_rate` LSTM boleh dibaca sebagai
tidak bermasalah secara praktis** untuk keputusan stok; rearrangement
post-hoc tidak mendesak diperlukan seperti pada XGBoost. Detail keputusan di
bagian 1.8 `docs/detail-tahap-perbandingan-model.md`.

### 5.3 Per `demand_segment`

Gabungan kelima fold, dibobot jumlah baris. Kolom selain K1 dibaca di τ=0,9.

| segmen | model | n | K1 | mae | coverage | fill_rate |
|---|---|---:|---:|---:|---:|---:|
| **smooth** | **lstm** | 45.485 | **11,0092** | 53,448 | 0,8905 | 0,9739 |
| | `naive_roll_mean_7` | 45.485 | 18,6402 | 37,280 | 0,6016 | 0,8903 |
| | `naive_lag_1` | 45.485 | 32,5143 | 65,029 | 0,4920 | 0,7766 |
| | `naive_zero` | 45.485 | 73,1813 | 146,363 | 0,0129 | 0,0000 |
| **erratic** | **lstm** | 54.511 | **5,4961** | 25,586 | 0,8906 | 0,9412 |
| | `naive_roll_mean_7` | 54.511 | 9,4628 | 18,926 | 0,5949 | 0,8199 |
| | `naive_lag_1` | 54.511 | 16,4714 | 32,943 | 0,5035 | 0,6556 |
| | `naive_zero` | 54.511 | 24,7993 | 49,599 | 0,0476 | 0,0000 |
| **lumpy** | **lstm** | 123.545 | **1,1664** | 5,643 | 0,9050 | 0,8380 |
| | `naive_roll_mean_7` | 123.545 | 1,7686 | 3,537 | 0,6763 | 0,6251 |
| | `naive_zero` | 123.545 | 2,5351 | 5,070 | 0,4339 | 0,0000 |
| | `naive_lag_1` | 123.545 | 2,6580 | 5,316 | 0,6400 | 0,4117 |
| **intermittent** | **lstm** | 122.006 | **0,4236** | 2,378 | 0,9201 | 0,8605 |
| | `naive_roll_mean_7` | 122.006 | 0,6919 | 1,384 | 0,7897 | 0,6131 |
| | `naive_zero` | 122.006 | 0,8365 | 1,673 | 0,7226 | 0,0000 |
| | `naive_lag_1` | 122.006 | 0,9823 | 1,965 | 0,7905 | 0,4128 |

LSTM menang K1 di **keempat** segmen melawan baseline. Dibandingkan langsung
ke RF dan XGBoost: di `lumpy` LSTM ada tepat di tengah (1,1664, antara RF
1,1430 dan XGBoost 1,1823), dan di `intermittent` juga di tengah (0,4236,
antara RF 0,4194 dan XGBoost 0,4978) — RF tetap yang terendah di kedua
segmen ini, LSTM tidak pernah menjadi yang terbaik mutlak di antara ketiga
model, tapi juga tidak pernah menjadi yang terburuk. Coverage@0,9 LSTM
(0,89–0,92) juga konsisten di antara RF (0,90–0,95) dan XGBoost (0,88–0,92),
mengulang pola "di tengah" dari bagian 1 dan 5.2.

### 5.4 Per `is_delivery_day`

| hari kirim | model | n | K1 | mae@0,9 | cov@0,9 | fill@0,9 | shortfall | overstock |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **True** | **lstm** | 98.701 | **4,1169** | 19,299 | 0,9012 | 0,9609 | 36.449 | 350.508 |
| | `naive_roll_mean_7` | 98.701 | 6,6430 | 13,286 | 0,6341 | 0,8528 | 683.526 | 627.820 |
| | `naive_lag_1` | 98.701 | 11,8786 | 23,757 | 0,5483 | 0,6755 | 1.499.452 | 845.412 |
| | `naive_zero` | 98.701 | 23,4586 | 46,917 | 0,3298 | 0,0000 | 4.630.772 | 0 |
| **False** | **lstm** | 246.846 | **2,3893** | 11,782 | 0,9082 | 0,9500 | 55.450 | 529.251 |
| | `naive_roll_mean_7` | 246.846 | 4,0953 | 8,191 | 0,7175 | 0,8479 | 844.867 | 1.176.969 |
| | `naive_lag_1` | 246.846 | 6,6948 | 13,390 | 0,6936 | 0,7466 | 1.402.090 | 1.903.062 |
| | `naive_zero` | 246.846 | 11,2635 | 22,527 | 0,4554 | 0,0000 | 5.560.708 | 0 |

Margin K1 di hari kirim (4,1169 vs 6,6430, 38%) dan non-kirim (2,3893 vs
4,0953, 42%) merata, sama pola dengan RF dan XGBoost. Di hari kirim, LSTM
punya shortfall terendah dari ketiga model (36.449, lawan RF 161.063 dan
XGBoost 40.237) — baris paling penting secara bisnis (barang yang benar-benar
naik truk) adalah tempat LSTM tampil paling kuat relatif, meski itu satu
angka dari run seed tunggal yang belum diverifikasi terhadap derau (bagian
5.1b).

## 6. Model final

`fit_final()` melatih ulang konfigurasi pemenang pada seluruh baris layak
sebelum Desember, dipotong `purging.lookahead_safe_mask()` — populasi baris
sama persis dengan yang dinilai di atas. Dijalankan **di CPU Mac**.

| | |
|---|---|
| Baris training | 1.349.011 |
| `best_epoch` | 5 |
| Titik kuantil tersimpan | 19 (0,05..0,95) |
| Device | cpu |
| Artefak | `models/lstm_q90.joblib` — 3,7 MB, 28 Agu 2026 08:43 |

Artefak LSTM (3,7 MB) jauh lebih kecil dari RF (826 MB) dan XGBoost (292 MB)
— bobot jaringan jauh lebih ringkas daripada struktur pohon tersimpan,
meski wall-clock pelatihannya paling mahal dari ketiganya (bagian 7).

## 7. Ongkos (bahan K3)

**Pencarian + pengulangan 3 seed (GPU Windows) dan walk-forward/fit-final
(CPU Mac) tidak sebanding satu sama lain** — beda device, sama seperti
XGBoost (`docs/hasil-modeling-xgb.md` bagian 7).

| tahap | device | wall clock | sumber |
|---|---|---:|---|
| Benchmark | cpu (Mac) | 16,0 menit | dicetak notebook |
| Pencarian 30 kandidat | cuda (Windows) | ~4,2 jam | jumlah `elapsed_seconds` per kandidat |
| Pengulangan 3 seed | cuda (Windows) | ~50 menit | jumlah `elapsed_seconds` 3 seed (917+1.113+1.062 detik) |
| Walk-forward 5 fold | cpu (Mac) | **~8 jam 28 menit** | **estimasi** dari selisih timestamp `lstm_best_params.json` (22:57, 27 Agu) ke `lstm_walk_forward_results.csv` (07:25, 28 Agu) |
| Fit final | cpu (Mac) | ~1 jam 18 menit | **estimasi** dari selisih timestamp `lstm_walk_forward_results.csv` (07:25) ke `models/lstm_q90.joblib` (08:43), 28 Agu |

Dua baris terakhir **tidak dicetak eksplisit oleh notebook**, sama
keterbatasan dengan XGBoost — diperkirakan dari timestamp berkas.

**Walk-forward LSTM (~8,5 jam) adalah tahap tunggal termahal yang terukur di
seluruh tiga model** — lebih lama dari walk-forward RF (~45 menit) dan
XGBoost (~3 jam) digabung. Ini konsisten dengan `best_epoch` per fold
(5–11, bagian 5.1) dikali `sec_per_epoch` ~107 detik: kelima fold dilatih
dari nol (bukan warm-start), dan awal pelatihan sebelum early stopping
berhenti tetap makan waktu penuh per epoch.

Estimasi todolist untuk keseluruhan LSTM (search+3-seed+WF+final) adalah
**~83–98 jam**, dihitung untuk skenario seluruhnya di CPU Mac — sama seperti
XGBoost, angka itu tidak bisa diverifikasi/dibantah dari run ini karena
sebagian tahap pindah device. Walk-forward + fit final di CPU Mac yang sah
dibandingkan lurus: RF ~93 menit, XGBoost ~204 menit, **LSTM ~586 menit** —
LSTM sekitar **6,3×** lebih lambat dari RF dan **2,9×** lebih lambat dari
XGBoost di tahap yang device-nya benar-benar identik di ketiganya.

## 8. Reproduksi

```bash
# Pencarian + 3-seed repeat — jalankan di mesin GPU (mis. PC Windows)
$env:FORECAST_DEVICE = "cuda"
python run_cells.py notebook\modeling_lstm.ipynb 2-10,14,16,18,20,22

# Walk-forward + fit final — jalankan di Mac (CPU), satu mesin yang sama
# dengan RF dan XGBoost untuk K3
.venv/bin/python3 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebook/modeling_lstm.ipynb
```

Pencarian menulis checkpoint tiap kandidat selesai ke `lstm_search_results.csv`
(`resume=True`); walk-forward **tidak** punya checkpoint — sesi yang terpotong
di tengah kehilangan seluruhnya (komentar notebook, sel "Walk-forward final").

| Artefak | Lokasi | Masuk git |
|---|---|---|
| Hasil pencarian | `dataset/model_ready/lstm_search_results.csv` | tidak |
| Pengulangan seed | `dataset/model_ready/lstm_seed_repeats.csv` | tidak |
| Parameter terpilih | `dataset/model_ready/lstm_best_params.json` | tidak |
| Tabel hasil lengkap | `dataset/model_ready/lstm_walk_forward_results.csv` | tidak |
| Model terlatih | `models/lstm_q90.joblib` | tidak |
| Ringkasan ini | `docs/hasil-modeling-lstm.md` | **ya** |
| Arsip run kuantil-tunggal | `docs/bak/hasil-modeling-lstm.single-quantile.bak.md` | **ya** |

## 9. Batasan

- **Desember 2025 belum dibuka.** Semua angka di sini adalah validasi
  walk-forward, bukan skor test set final.
- **Sumbu waktunya waktu pengambilan, bukan waktu pemesanan** — sama batasan
  dengan RF/XGBoost. Lihat `docs/batasan-penelitian.md` (B-1, B-2, B-3).
- **MAE tidak sebanding lintas model dengan baseline titik-tengah** — sama
  catatan dengan RF/XGBoost. K1 adalah kriterianya.
- **Satu seed (42) dipakai sebagai angka kepala di seluruh dokumen ini, dan
  ini terbukti bukan representasi stabil — dikonfirmasi 2026-08-30, bukan
  lagi dugaan.** Walk-forward 5-fold penuh diulang dengan seed 43: K1 fold
  bersih melompat dari 2,8818 ke **3,0732** (+6,6%), mengubah LSTM dari
  "hampir seri dengan RF" menjadi "kalah dari RF dan XGBoost sekaligus"
  (bagian 5.1b). Ini batasan paling penting di dokumen ini — angka kepala di
  bagian 1 harus dibaca sebagai satu titik dari sebaran lebar, bukan
  performa "LSTM" yang stabil, dan bagian 1.6/1.8
  `docs/detail-tahap-perbandingan-model.md` sudah direvisi mengikuti ini.
- ~~`crossing_rate` = 0,4345 belum dijelaskan~~ — **diuji 2026-08-29** (bagian
  5.2): hampir seluruhnya derau numerik (rate ambruk ke 1,1% di toleransi gap
  0,1), bukan defek struktural seperti XGBoost. Tidak lagi batasan aktif.
- **K2 di τ rendah** punya keterbatasan yang sama dengan RF/XGBoost — belum
  bisa dibaca sebagai kalibrasi murni sampai aturan penyisihan dinyatakan
  ulang terhadap lantai `share_nol`.
- **Fold 3 dan 5 ikut memilih pemenang** — potongan fold 1/2/4 di bagian 5.1
  adalah angka bersih dan menjadi K1 resmi.
- **Peringkat kandidat pencarian tidak bisa dibandingkan dengan run
  kuantil-tunggal lama** (bagian 4.3) — beda dari RF/XGBoost, ruang
  pencarian LSTM sendiri berubah, bukan cuma kriterianya.
- **Perbandingan lintas model (RF/XGBoost/LSTM) belum final** — angka mentah
  ketiganya sudah ada, tapi bagian 1.6/1.8
  `docs/detail-tahap-perbandingan-model.md` masih menunggu jawaban atas
  `crossing_rate` (XGBoost & LSTM) dan derau seed (LSTM) di atas.
