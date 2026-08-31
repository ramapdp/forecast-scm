# Hasil Modeling — XGBoost multi-kuantil

Angka terukur dari jalannya `notebook/modeling_xgb.ipynb`, selesai **27 Agustus
2026**, run pertama di bawah kriteria multi-kuantil (butir 0c
`docs/todolist-proyek.md`). Desainnya ada di
`docs/superpowers/specs/2026-08-19-xgboost-modeling-design.md` dan
`docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`; dokumen
ini hanya memuat hasilnya.

**Dokumen ini menggantikan versi kuantil-tunggal**, diarsipkan sebagai
`docs/bak/hasil-modeling-xgb.single-quantile.bak.md`. Angka di kedua dokumen
**bukan besaran yang sama**: yang lama pinball@0,9 pada satu titik kuantil,
yang ini K1 — rata-rata pinball lintas 19 titik `QUANTILE_SET_A` (T-10,
`docs/todolist-proyek.md`). Yang **sah** dibawa lintas dokumen adalah
*peringkat* kandidat pencarian, dan di XGBoost peringkat itu **berubah lebih
banyak** daripada di RF — lihat bagian 4.3.

**Mesin berbeda untuk pencarian vs walk-forward.** Pencarian 30 kandidat
berjalan di **GPU Windows** (RTX 4060 Ti 8 GB, `device=cuda`), keputusan
2026-08-26 yang menggantikan rencana "seluruh Fase 3 di CPU Mac lokal"
2026-08-25 — lihat `docs/runbook-pencarian-gpu-windows.md`. Walk-forward final
dan fit final tetap di **CPU Mac**, satu mesin yang sama dengan RF dan LSTM,
supaya K3 (ongkos) tetap sebanding lintas ketiga model. Paritas GPU↔CPU untuk
peringkat kandidat sudah divalidasi lebih dulu di Tahap 0 (selisih K1 0,124%,
jauh di bawah ambang 2% — `docs/superpowers/specs/2026-08-24-distributed-gpu-training-design.md`
bagian 3bis), jadi pemenang yang dipilih di GPU tetap dipercaya.

**Desember 2025 tidak dibuka.** Semua angka di bawah datang dari walk-forward
lima fold di Juli–November 2025. Test set final masih terkunci.

## 1. Ringkasan

XGBoost melewati gerbang G0 dan mencetak **K1 = 2,9433** pada potongan fold
bersih (1/2/4), lawan **4,8603** milik baseline terbaik `naive_roll_mean_7` —
**39% lebih baik**. Pada gabungan kelima fold angkanya 2,9197 lawan 4,8231.

| model | K1 (fold 1/2/4) | K1 (5 fold) |
|---|---:|---:|
| **xgboost** | **2,9433** | **2,9197** |
| `naive_roll_mean_7` | 4,8603 | 4,8231 |
| `naive_lag_1` | 8,1612 | 8,1755 |
| `naive_zero` | 14,8102 | 14,7469 |

Dibandingkan Random Forest (`docs/hasil-modeling-rf.md`, K1 fold bersih
2,8508), XGBoost **kalah** 0,0925 (3,2% lebih buruk) — kalah, tapi tidak jauh.
XGBoost **menang MAE@0,9** (13,467 lawan 15,055 milik RF): prediksinya lebih
dekat ke aktual di titik tengah, meski rata-rata pinball 19 titiknya sedikit
lebih longgar.

**`crossing_rate` = 0,9767 (97,7% baris)** di walk-forward final — jauh di
atas RF (0% struktural). Diuji 2026-08-29: sebagian besar defek sungguhan
(~20–25% baris crossing material), bukan cuma derau numerik (bagian 5.2).
**Rearrangement kuantil dijalankan 2026-08-30 (bagian 5.5)**: K1 fold bersih
membaik dari 2,9433 ke **2,9115** (−1,08%) dan `crossing_rate` turun ke
**0,0000**, tapi gap ke RF (2,8508) hanya menyempit dari 3,24% ke **2,13%** —
masih di atas ambang keputusan 2%, jadi urutan RF > XGBoost **tidak
berbalik**. Lihat bagian 16/18 `metodologi-pemodelan-dan-pemilihan-model.md`
untuk bagaimana ini masuk ke keputusan pemenang.

Di τ=0,9 — titik yang dijanjikan ke bisnis (B-9) — coverage 0,902 terhadap
target 0,90 (RF: 0,928), dengan fill rate 0,951. Sisi bisnisnya, di 345.547
baris validasi:

| | kekurangan (shortfall) | kelebihan (overstock) |
|---|---:|---:|
| `xgboost` | 500.579 | 4.132.651 |
| `random_forest` | 418.250 | 4.793.038 |
| `naive_roll_mean_7` | 1.528.393 | 1.804.789 |

> Baris `shortfall`/`overstock` di atas dijumlahkan dari tabel per-fold bagian
> 5.1. Catatan satuan sama seperti dokumen RF: unit dijumlahkan lintas SKU
> bersatuan campur, jadi sah untuk membandingkan model pada baris yang sama,
> tapi tidak punya makna fisik sebagai satu besaran tunggal.

XGBoost punya shortfall **lebih tinggi** dan overstock **lebih rendah**
daripada RF — arahnya konsisten dengan coverage@0,9 XGBoost (0,902) yang
lebih dekat ke target nominal 0,90 daripada RF (0,928): RF over-covers lebih
jauh di atas 0,90, jadi lebih jarang kekurangan tapi menumpuk kelebihan stok
lebih banyak; XGBoost lebih dekat ke garis target, jadi kekurangannya lebih
sering muncul (+20% dibanding RF) tapi kelebihannya lebih sedikit (-14%
dibanding RF). Mana yang lebih disukai bisnis tergantung ongkos relatif
shortfall vs overstock — bukan sesuatu yang bisa diputuskan dari angka model
saja.

## 2. Setup evaluasi

| | |
|---|---|
| Data | `dataset/model_ready/model_input.parquet` — 1.502.522 baris panel, 82 kolom, 1 Jan 2024 – 31 Des 2025 |
| Fitur | 56 kolom (`modeling_prep.FEATURE_COLS`) |
| Target | `target_lead_time_cumulative` — sama populasi baris dengan RF (dijamin `walk_forward.eligible_rows()`) |
| Fold | 5 expanding window, validasi Juli, Agustus, September, Oktober, November 2025 |
| Test terkunci | Desember 2025 (`TEST_START = 2025-12-01`) |
| Baris validasi | 345.547 total — identik dengan RF |
| Kuantil | `QUANTILE_SET_A` — 19 titik, 0,05 sampai 0,95 langkah 0,05 |
| Kriteria | K1 = rata-rata tak berbobot pinball lintas 19 titik, pada potongan fold 1/2/4 |
| Implementasi | `xgboost` `reg:quantileerror`, `multi_strategy` (19 pohon per ronde boosting), protokol dua-fit: early stopping pada tail 30 hari purged tiap fold, lalu refit penuh |
| Device | pencarian: `cuda` (GPU Windows); walk-forward & fit final: `cpu` (Mac) |
| Commit | pencarian: `e074421`; walk-forward & fit final: lihat `models/xgboost_q90.joblib` |

Ketiga baseline naive dinilai pada **baris yang identik** dengan XGBoost —
sama seperti RF, dijamin `utils/modelling/walk_forward.py`.

Boosting membangun **satu pohon per titik kuantil per ronde** (`multi_strategy`
bawaan XGBoost untuk `reg:quantileerror`), jadi 19 titik kuantil bukan ongkos
tambahan tipis seperti di RF — pengganda terukur **×15,2** (bagian 3). Inilah
yang membuat pencarian dipindah ke GPU: satu-satunya tahap yang layak
dipindahkan, karena walk-forward dan fit final tetap harus di mesin yang sama
dengan RF/LSTM untuk K3.

## 3. Benchmark

Satu putaran dua-fit (early stopping pada tail 30 hari purged, lalu refit
penuh) di fold 5 dengan `DEFAULT_PARAMS`, **di CPU Mac** — tahap ini tidak
dipindah ke GPU karena tujuannya mengukur ongkos di mesin yang sama dengan
walk-forward/fit final.

| | |
|---|---|
| Baris training fold 5 | 1.292.778 (fit 1.224.830 + tail purged 65.140) |
| Baris validasi fold 5 | 59.629 |
| `best_iteration` | 1.999 dari 2.000 (plafon early stopping nyaris tersentuh) |
| Wall time kedua fit | 265,2 menit (~4,4 jam) |
| Bentuk prediksi | (59.629, 19) — baris × titik kuantil |
| Prediksi di τ=0,9 | rata-rata 42,04, maksimum 1.855,92 |
| `crossing_rate` | 0,8298 |

Dua hal yang dikonfirmasi angka ini:

1. **`best_iteration` 1.999 dari plafon 2.000 nyaris habis** — beda jauh dari
   RF, yang tidak punya konsep ronde/iterasi. Ini sinyal bahwa `DEFAULT_PARAMS`
   (bukan pemenang pencarian) mendekati batas anggaran boosting yang
   disediakan; pencarian sesudahnya (bagian 4) menemukan konfigurasi yang jauh
   lebih hemat ronde (lihat `best_iteration` pemenang di bagian 5).
2. **`crossing_rate` sudah tinggi bahkan di `DEFAULT_PARAMS`** (0,8298) —
   bukan artefak hyperparameter tertentu dari pencarian, melainkan properti
   yang tampaknya melekat pada cara XGBoost `reg:quantileerror` memprediksi 19
   titik di dataset ini. Konsisten dengan rentang 0,64–0,97 yang terlihat di
   seluruh 30 kandidat pencarian (bagian 4.1).

## 4. Pencarian hyperparameter

30 kandidat ditarik dari ruang **2.592 kombinasi**
(`max_depth × learning_rate × min_child_weight × subsample × colsample_bytree
× reg_lambda × encoding × log_target`), dinilai di **fold 3 (September) dan
fold 5 (November)** dengan kriteria K1 gabungan — sama protokol dengan RF,
tapi di **GPU Windows** (bagian mesin-terpisah di atas). Semua 30 kandidat
selesai (0 gagal); `device=cuda`, `commit=e074421` di seluruh baris.

### 4.1 Tabel lengkap

| # | max_depth | lr | min_child_weight | subsample | colsample_bytree | reg_lambda | encoding | log_target | K1 | mae@0,9 | cov@0,9 | detik |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| **17** | 10 | 0,05 | 1 | 0,7 | 0,5 | 10,0 | native | False | **2,8803** | 13,280 | 0,893 | 1.165 |
| 10 | 8 | 0,05 | 1 | 0,7 | 0,7 | 1,0 | native | False | 2,8816 | 13,631 | 0,900 | 1.464 |
| 22 | 8 | 0,10 | 10 | 1,0 | 1,0 | 10,0 | one_hot | False | 2,8937 | 14,021 | 0,895 | 2.031 |
| 11 | 6 | 0,05 | 50 | 0,7 | 0,7 | 10,0 | native | False | 2,8944 | 13,762 | 0,901 | 2.688 |
| 19 | 10 | 0,03 | 1 | 1,0 | 1,0 | 1,0 | one_hot | True | 2,8948 | 13,342 | 0,897 | 2.429 |
| 23 | 6 | 0,03 | 1 | 0,7 | 1,0 | 10,0 | native | False | 2,8981 | 13,780 | 0,899 | 2.417 |
| 13 | 10 | 0,10 | 50 | 1,0 | 1,0 | 10,0 | one_hot | True | 2,9004 | 13,797 | 0,898 | 922 |
| 1 | 10 | 0,10 | 1 | 0,7 | 1,0 | 1,0 | one_hot | False | 2,9024 | 13,747 | 0,899 | 1.041 |
| 3 | 6 | 0,05 | 50 | 1,0 | 0,7 | 10,0 | one_hot | False | 2,9096 | 13,881 | 0,898 | 3.985 |
| 25 | 6 | 0,05 | 10 | 0,7 | 0,5 | 10,0 | ordinal | False | 2,9100 | 13,750 | 0,895 | 1.934 |
| 9 | 6 | 0,10 | 50 | 0,7 | 1,0 | 1,0 | native | False | 2,9138 | 14,178 | 0,902 | 1.535 |
| 2 | 6 | 0,10 | 50 | 0,7 | 0,5 | 10,0 | ordinal | False | 2,9284 | 14,219 | 0,900 | 1.098 |
| 6 | 8 | 0,10 | 50 | 0,7 | 0,7 | 1,0 | ordinal | True | 2,9384 | 13,780 | 0,894 | 583 |
| 29 | 6 | 0,03 | 10 | 1,0 | 0,5 | 1,0 | ordinal | True | 2,9465 | 13,422 | 0,897 | 1.671 |
| 27 | 8 | 0,10 | 10 | 1,0 | 0,5 | 10,0 | ordinal | True | 2,9474 | 13,234 | 0,888 | 511 |
| 21 | 6 | 0,03 | 10 | 0,7 | 1,0 | 10,0 | ordinal | True | 2,9637 | 13,383 | 0,895 | 1.866 |
| 0 | 6 | 0,03 | 1 | 0,7 | 1,0 | 1,0 | ordinal | True | 2,9639 | 13,531 | 0,896 | 1.897 |
| 18 | 6 | 0,10 | 1 | 0,7 | 1,0 | 10,0 | ordinal | True | 2,9777 | 13,464 | 0,894 | 597 |
| 26 | 6 | 0,10 | 10 | 0,7 | 1,0 | 10,0 | ordinal | True | 2,9792 | 13,844 | 0,895 | 590 |
| 28 | 4 | 0,10 | 10 | 0,7 | 1,0 | 10,0 | ordinal | False | 2,9900 | 14,428 | 0,900 | 1.998 |
| 14 | 4 | 0,10 | 1 | 0,7 | 0,5 | 1,0 | ordinal | True | 3,0114 | 13,874 | 0,896 | 1.819 |
| 15 | 4 | 0,10 | 10 | 1,0 | 1,0 | 10,0 | ordinal | True | 3,0161 | 13,579 | 0,893 | 1.036 |
| 12 | 6 | 0,05 | 10 | 0,7 | 1,0 | 1,0 | native | True | 3,0171 | 13,600 | 0,899 | 946 |
| 4 | 4 | 0,03 | 1 | 0,7 | 0,7 | 10,0 | ordinal | True | 3,0217 | 13,475 | 0,895 | 1.891 |
| 8 | 6 | 0,10 | 10 | 0,7 | 0,7 | 10,0 | native | True | 3,0251 | 13,686 | 0,895 | 794 |
| 16 | 4 | 0,10 | 50 | 1,0 | 1,0 | 10,0 | native | False | 3,0379 | 14,611 | 0,901 | 2.249 |
| 7 | 4 | 0,03 | 1 | 1,0 | 0,5 | 1,0 | one_hot | True | 3,0407 | 14,405 | 0,902 | 4.274 |
| 5 | 4 | 0,10 | 50 | 0,7 | 0,7 | 10,0 | native | True | 3,0527 | 13,622 | 0,892 | 1.747 |
| 20 | 4 | 0,03 | 1 | 0,7 | 0,7 | 10,0 | ordinal | False | 3,0620 | 14,320 | 0,900 | 1.923 |
| 24 | 4 | 0,03 | 1 | 1,0 | 1,0 | 1,0 | one_hot | False | 3,0772 | 14,897 | 0,901 | 3.793 |

Sumber: `dataset/model_ready/xgb_search_results.csv`. Kolom "detik" adalah
`elapsed_seconds` di GPU Windows — **tidak sebanding** dengan detik RF/XGBoost
lama yang diukur di CPU Mac (lihat bagian 7).

### 4.2 Yang terbaca dari sebarannya

1. **`max_depth` adalah penentu utama, dan monoton.** Rata-rata K1 per
   `max_depth`: 4 → 3,0344, 6 → 2,9483, 8 → 2,9153, 10 → 2,8945. Rentang
   keseluruhan 2,8803–3,0772 (6,8%) — lebih sempit dari RF (11%), tapi ke-9
   kandidat `max_depth=4` **seluruhnya** ada di separuh bawah tabel. Beda dari
   RF, di mana kedalaman pohon sudah jenuh manfaatnya di 56 fitur; di XGBoost
   yang membangun satu pohon boosting per titik kuantil, kedalaman lebih
   dalam masih terus membantu pada rentang yang diuji.
2. **`log_target=False` konsisten lebih baik** (rata-rata 2,9414 lawan
   3,0038 untuk `log_target=True`) — sejalan dengan RF, di mana kandidat
   terbaiknya juga `log_target=False`.
3. **Encoding tidak banyak membedakan** (`native` 2,9557, `one_hot` 2,9448,
   `ordinal` 2,9743 rata-rata) — beda antar encoding jauh lebih kecil daripada
   beda antar `max_depth`. Pemenangnya `native`, sejalan dengan intuisi:
   XGBoost menangani kategori langsung tanpa perlu one-hot atau ordinal.
4. **Ongkos hampir tidak berkorelasi dengan mutu** (korelasi Pearson
   `elapsed_seconds` vs `pinball` = 0,157) — kandidat 27 (511 detik) mencetak
   2,9474, hampir seri dengan kandidat 3 yang makan 3.985 detik (2,9096).
   Lima kandidat teratas berjarak hanya **0,50%** satu sama lain — lebih
   ketat malah daripada RF (1,46%).

Parameter terpilih (`dataset/model_ready/xgb_best_params.json`):

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

### 4.3 Apakah peringkat kandidat berubah setelah pindah ke K1?

Sama seperti RF, seed dan ruang pencariannya identik terhadap run
kuantil-tunggal, jadi ke-30 kandidat di kedua run adalah **kombinasi parameter
yang persis sama, id per id** (diverifikasi kolom demi kolom, 0 selisih
parameter).

| | |
|---|---|
| Spearman ρ (K1 baru vs pinball@0,9 lama) | **0,7348** |
| Kendall τ | **0,5402** |
| Pemenang | **berubah**: kandidat 11 → kandidat 17 |

**Jauh lebih lemah daripada RF** (ρ = 0,975, τ = 0,895). Ini **mengkonfirmasi**
peringatan eksplisit di `docs/hasil-modeling-rf.md` bagian 4.3: XGBoost punya
mekanisme (`multi_strategy`, satu pohon per titik kuantil per ronde) yang
memang berinteraksi dengan jumlah titik kuantil, tidak seperti RF yang membaca
seluruh titik dari daun yang sudah jadi. Konsekuensinya: **peringkat kandidat
di run kuantil-tunggal lama (`docs/bak/hasil-modeling-xgb.single-quantile.bak.md`)
tidak bisa dipakai sebagai proksi peringkat K1** — kandidat yang optimal untuk
satu titik τ=0,9 tidak otomatis optimal untuk rata-rata 19 titik, jauh lebih
sering berpindah dibanding RF.

## 5. Hasil walk-forward

Pemenang (kandidat 17) dijalankan ulang di kelima fold, di CPU Mac.

### 5.0 Gerbang G0

> Model harus mengalahkan `naive_roll_mean_7` pada pinball@0,9 **di kelima
> fold**, bukan hanya di gabungan.

pinball@0,9 per fold:

| model | 1 (Jul) | 2 (Agu) | 3 (Sep) | 4 (Okt) | 5 (Nov) |
|---|---:|---:|---:|---:|---:|
| **xgboost** | **2,3318** | **2,4812** | **2,3251** | **2,7805** | **2,6156** |
| `naive_roll_mean_7` | 4,2489 | 4,5665 | 4,0341 | 4,7826 | 4,9703 |
| `naive_lag_1` | 8,3545 | 8,4686 | 8,0448 | 8,5260 | 8,3717 |
| `naive_zero` | 26,4535 | 27,2539 | 23,8486 | 26,2187 | 29,3199 |

**G0 lolos.** XGBoost menang di kelima fold dengan margin 42–45%.

### 5.1 Per fold

K1 per fold:

| model | 1 (Jul) | 2 (Agu) | 3 (Sep) | 4 (Okt) | 5 (Nov) |
|---|---:|---:|---:|---:|---:|
| **xgboost** | **2,8640** | **2,9178** | **2,7366** | **3,0568** | **3,0510** |
| `naive_roll_mean_7` | 4,6636 | 4,9814 | 4,6680 | 4,9372 | 4,8719 |
| `naive_lag_1` | 7,9549 | 8,5327 | 8,1080 | 7,9712 | 8,3067 |
| `naive_zero` | 14,6964 | 15,1410 | 13,2492 | 14,5660 | 16,2888 |

Detail XGBoost per fold (kolom @0,9 kecuali K1):

| fold | bulan | n | K1 | mae | pinball | coverage | fill_rate | shortfall | overstock |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Jul 2025 | 72.684 | 2,8640 | 14,227 | 2,3318 | 0,9156 | 0,9613 | 82.592 | 951.509 |
| 2 | Agu 2025 | 75.015 | 2,9178 | 12,964 | 2,4812 | 0,9055 | 0,9511 | 111.093 | 861.407 |
| 3 | Sep 2025 | 70.503 | 2,7366 | 13,925 | 2,3251 | 0,9082 | 0,9560 | 82.187 | 899.596 |
| 4 | Okt 2025 | 67.716 | 3,0568 | 13,207 | 2,7805 | 0,8918 | 0,9374 | 123.570 | 770.741 |
| 5 | Nov 2025 | 59.629 | 3,0510 | 12,587 | 2,6156 | 0,8864 | 0,9479 | 101.138 | 649.399 |

K1 bergerak 2,737–3,057 (rentang 12%), mirip stabilnya dengan RF (13%). Fold 4
(Oktober) paling berat, sama seperti RF — properti bulannya, bukan properti
modelnya. Coverage@0,9 menurun dari 0,916 (Juli) ke 0,886 (November), pola
arah yang sama dengan RF tapi levelnya sedikit lebih rendah di setiap fold
(RF: 0,936→0,918).

**Fold 3 dan 5 adalah fold yang memilih pemenang.** K1 XGBoost di gabungan
kedua fold itu (dihitung dari `xgb_walk_forward_results.csv`) = **2,8806**,
hampir persis sama dengan skor pencarian kandidat 17 di bagian 4.1 (2,8803,
selisih 0,0003) — konfigurasi dan seed identik, jadi ini cek reprodusibilitas
yang lolos, sama seperti RF. Selisih kecil yang tersisa berasal dari fold
walk-forward yang dijalankan di CPU sedangkan kandidat pencarian dinilai di
GPU (bagian mesin-terpisah di atas) — konsisten dengan paritas GPU↔CPU 0,124%
yang sudah diukur di Tahap 0. Dipotong ke fold 1, 2, dan 4 saja — tiga
fold yang tidak menyentuh seleksi, dan potongan yang menjadi kriteria K1
resmi:

| model | n | K1 | mae@0,9 | cov@0,9 | fill@0,9 |
|---|---:|---:|---:|---:|---:|
| **xgboost** | 215.415 | **2,9433** | 13,467 | 0,905 | 0,950 |
| `naive_roll_mean_7` | 215.415 | 4,8603 | 9,721 | 0,696 | 0,850 |
| `naive_lag_1` | 215.415 | 8,1612 | 16,322 | 0,653 | 0,712 |
| `naive_zero` | 215.415 | 14,8102 | 29,620 | 0,423 | 0,000 |

2,9433 di fold bersih lawan ~2,88 di fold seleksi — XGBoost tampak **sedikit
lebih lemah** di fold yang tidak ikut memilihnya, kebalikan dari RF (yang
justru sedikit lebih baik di fold bersih). Selisihnya (~0,06) jauh lebih kecil
daripada jarak ke RF (0,0925), jadi ini bukan tanda overfitting seleksi yang
besar — konsisten dengan lanskap pencarian yang cukup datar di bagian 4.2 (top
5 hanya 0,50% terpisah) — tapi berlawanan arah dari RF, layak dicatat untuk
XGBoost secara spesifik.

XGBoost **mengalahkan MAE `naive_roll_mean_7`... tidak** — sama seperti RF,
XGBoost kalah MAE (13,467 lawan 9,721). Prediksi τ=0,9 sengaja bias ke atas;
MAE dilaporkan untuk konteks, bukan kriteria kemenangan.

### 5.2 K2 — kalibrasi di seluruh 19 titik kuantil, dan `crossing_rate`

Gabungan kelima fold, dibobot jumlah baris. Lantai `share_nol` = **0,4195**
(sama seperti RF — populasi baris identik). Metodologi lantai ini dibahas
penuh di `docs/hasil-modeling-rf.md` bagian 5.2; di sini langsung dipakai.

| τ | pinball | coverage | gap (cov − τ) | lantai | kelebihan di atas lantai |
|---:|---:|---:|---:|---:|---:|
| 0,05 | 0,8421 | 0,4376 | +0,3876 | 0,4195 | +0,0181 |
| 0,10 | 1,4110 | 0,4567 | +0,3567 | 0,4195 | +0,0372 |
| 0,15 | 1,8942 | 0,4760 | +0,3260 | 0,4195 | +0,0565 |
| 0,20 | 2,2974 | 0,4979 | +0,2979 | 0,4195 | +0,0784 |
| 0,25 | 2,6553 | 0,5201 | +0,2701 | 0,4195 | +0,1006 |
| 0,30 | 2,9727 | 0,5416 | +0,2416 | 0,4195 | +0,1221 |
| 0,35 | 3,2354 | 0,5642 | +0,2142 | 0,4195 | +0,1447 |
| 0,40 | 3,4467 | 0,5893 | +0,1893 | 0,4195 | +0,1698 |
| 0,45 | 3,6180 | 0,6148 | +0,1648 | 0,45 | +0,1648 |
| 0,50 | 3,7599 | 0,6433 | +0,1433 | 0,50 | +0,1433 |
| 0,55 | 3,8297 | 0,6700 | +0,1200 | 0,55 | +0,1200 |
| 0,60 | 3,8376 | 0,6994 | +0,0994 | 0,60 | +0,0994 |
| 0,65 | 3,8028 | 0,7307 | +0,0807 | 0,65 | +0,0807 |
| 0,70 | 3,7184 | 0,7625 | +0,0625 | 0,70 | +0,0625 |
| 0,75 | 3,5767 | 0,7937 | +0,0437 | 0,75 | +0,0437 |
| 0,80 | 3,3231 | 0,8278 | +0,0278 | 0,80 | +0,0278 |
| 0,85 | 2,9676 | 0,8647 | +0,0147 | 0,85 | +0,0147 |
| **0,90** | **2,4998** | **0,9022** | **+0,0022** | 0,90 | +0,0022 |
| 0,95 | 1,7860 | 0,9425 | -0,0075 | 0,95 | -0,0075 |

**Bacaan K2 (lantai) mirip RF dalam bentuk** — over-coverage memuncak di paruh
bawah grid, mengecil ke arah τ tinggi. Tapi **levelnya lebih rendah di
τ=0,90–0,95**: XGBoost sedikit **under-coverage** di τ=0,95 (-0,0075) sementara
RF over-coverage +0,0108 di titik yang sama, dan di τ=0,90 kelebihan di atas
lantai XGBoost (+0,0022) jauh lebih kecil daripada RF (+0,0281) — kalibrasi
XGBoost di titik yang **benar-benar dijanjikan ke bisnis** (τ=0,90) lebih
dekat ke targetnya secara mentah, meski coverage absolutnya (0,9022) sedikit
lebih rendah dari 0,90 dibanding RF yang di atas (0,928).

**`crossing_rate` = 0,9767** di seluruh baris walk-forward (bukan per-τ —
kolomnya konstan lintas kuantil per fold, karena crossing adalah properti satu
baris prediksi, bukan satu titik kuantil). Rentangnya per fold: 0,966–0,987 —
tinggi dan stabil di seluruh fold, bukan anomali satu bulan.

**Diuji 2026-08-29 — sebagian besar defek sungguhan, bukan cuma artefak.**
Crossing dihitung ulang dari bundle tersimpan (tanpa retrain) dengan
toleransi jarak minimum (`prediksi(τ_tinggi) < prediksi(τ_rendah) - gap`):

| toleransi gap | crossing_rate |
|---:|---:|
| 0 (definisi resmi) | 0,916* |
| 0,01 | 0,794 |
| 0,1 | 0,479 |
| 0,5 | **0,248** |
| 1,0 | **0,202** |
| 5,0 | **0,106** |

\*sedikit di bawah 0,9767 karena ini prediksi dari model final gabungan,
bukan 5 model per-fold walk-forward.

Sebagian besar (0,916 → 0,479 di toleransi 0,1) memang derau numerik kecil —
median besar inversi cuma 0,043 unit. **Tapi ada inti keras ~20–25% baris
yang tetap crossing bahkan di toleransi 0,5–1,0 unit**, dengan distribusi
berekor sangat panjang (mean 1,06, maksimum 139 unit). Ini **bukan** artefak
pembulatan — `multi_strategy` XGBoost memang menghasilkan urutan kuantil yang
salah secara material pada seperlima sampai seperempat barisnya. Hipotesis 2
(tidak ada jaminan monotonicity struktural pada `reg:quantileerror`) yang
terbukti, bukan hipotesis 1 (efek ikatan) — bandingkan LSTM
(`docs/hasil-modeling-lstm.md` bagian 5.2), yang hasilnya berlawanan.

**Konsekuensi**: opsi baku post-hoc rearrangement kuantil (Chernozhukov et
al., 2010) **relevan dan disarankan** sebelum baris XGBoost dipakai untuk
keputusan stok — bukan cuma opsi teoretis. Detail keputusan di bagian 18
`metodologi-pemodelan-dan-pemilihan-model.md`.

### 5.3 Per `demand_segment`

Gabungan kelima fold, dibobot jumlah baris. Kolom selain K1 dibaca di τ=0,9.

| segmen | model | n | K1 | mae | coverage | fill_rate |
|---|---|---:|---:|---:|---:|---:|
| **smooth** | **xgboost** | 45.485 | **11,0466** | 50,887 | 0,8798 | 0,9696 |
| | `naive_roll_mean_7` | 45.485 | 18,6402 | 37,280 | 0,6016 | 0,8903 |
| | `naive_lag_1` | 45.485 | 32,5143 | 65,029 | 0,4920 | 0,7766 |
| | `naive_zero` | 45.485 | 73,1813 | 146,363 | 0,0129 | 0,0000 |
| **erratic** | **xgboost** | 54.511 | **5,4969** | 23,706 | 0,8796 | 0,9376 |
| | `naive_roll_mean_7` | 54.511 | 9,4628 | 18,926 | 0,5949 | 0,8199 |
| | `naive_lag_1` | 54.511 | 16,4714 | 32,943 | 0,5035 | 0,6556 |
| | `naive_zero` | 54.511 | 24,7993 | 49,599 | 0,0476 | 0,0000 |
| **lumpy** | **xgboost** | 123.545 | **1,1823** | 5,368 | 0,9016 | 0,8304 |
| | `naive_roll_mean_7` | 123.545 | 1,7686 | 3,537 | 0,6763 | 0,6251 |
| | `naive_zero` | 123.545 | 2,5351 | 5,070 | 0,4339 | 0,0000 |
| | `naive_lag_1` | 123.545 | 2,6580 | 5,316 | 0,6400 | 0,4117 |
| **intermittent** | **xgboost** | 122.006 | **0,4978** | 2,977 | 0,9213 | 0,8780 |
| | `naive_roll_mean_7` | 122.006 | 0,6919 | 1,384 | 0,7897 | 0,6131 |
| | `naive_zero` | 122.006 | 0,8365 | 1,673 | 0,7226 | 0,0000 |
| | `naive_lag_1` | 122.006 | 0,9823 | 1,965 | 0,7905 | 0,4128 |

XGBoost menang K1 di **keempat** segmen. Margin relatifnya (vs baseline
terbaik) berpola sama dengan RF: terbesar di `erratic`/`smooth` (~42%/41%),
lebih kecil di `lumpy`/`intermittent` (~33%/28%). Dibandingkan langsung ke
RF: XGBoost K1 lebih buruk di **ketiga** segmen selain `lumpy` (di mana
keduanya nyaris seri — 1,1823 vs 1,1430), dan coverage@0,9 XGBoost secara
konsisten **di bawah** RF di keempat segmen (0,88–0,92 vs 0,90–0,95 milik
RF) — pola yang sama dengan bagian 5.2: XGBoost lebih dekat ke garis target
0,90, RF lebih longgar di atasnya.

### 5.4 Per `is_delivery_day`

| hari kirim | model | n | K1 | mae@0,9 | cov@0,9 | fill@0,9 | shortfall | overstock |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **True** | **xgboost** | 98.701 | **4,1178** | 18,165 | 0,8976 | 0,9569 | 40.237 | 322.925 |
| | `naive_roll_mean_7` | 98.701 | 6,6430 | 13,286 | 0,6341 | 0,8528 | 683.526 | 627.820 |
| | `naive_lag_1` | 98.701 | 11,8786 | 23,757 | 0,5483 | 0,6755 | 1.499.452 | 845.412 |
| | `naive_zero` | 98.701 | 23,4586 | 46,917 | 0,3298 | 0,0000 | 4.630.772 | 0 |
| **False** | **xgboost** | 246.846 | **2,4407** | 11,507 | 0,9040 | 0,9461 | 59.727 | 511.843 |
| | `naive_roll_mean_7` | 246.846 | 4,0953 | 8,191 | 0,7175 | 0,8479 | 844.867 | 1.176.969 |
| | `naive_lag_1` | 246.846 | 6,6948 | 13,390 | 0,6936 | 0,7466 | 1.402.090 | 1.903.062 |
| | `naive_zero` | 246.846 | 11,2635 | 22,527 | 0,4554 | 0,0000 | 5.560.708 | 0 |

Margin K1 di hari kirim (4,1178 vs 6,6430, 38%) dan non-kirim (2,4407 vs
4,0953, 40%) hampir sama, sejalan dengan pola RF yang juga merata di kedua
kelompok. Di hari kirim, shortfall XGBoost (40.237) jauh lebih rendah
daripada RF (161.063 — lihat `hasil-modeling-rf.md` bagian 5.4), konsisten
dengan MAE XGBoost yang lebih ketat di seluruh dokumen ini.

### 5.5 Rearrangement kuantil (Chernozhukov et al., 2010) — DIKERJAKAN 2026-08-30

`xgb_walk_forward_results.csv` (bagian 5.1) hanya menyimpan skor teragregasi
per (model, fold, kuantil) — prediksi mentah per baris tidak pernah disimpan.
Jadi tidak ada jalan pintas post-hoc murni: rearrangement yang jujur lintas
fold (leakage-safe, sebanding dengan K1 RF/LSTM) berarti mem-fit ulang kelima
model fold dengan hyperparameter pemenang yang sama
(`xgb_best_params.json`), lalu mengurutkan (sort) 19 prediksi kuantil tiap
baris naik sebelum dinilai — itulah definisi rearrangement Chernozhukov: pada
grid `QUANTILE_SET_A` yang sudah berurutan naik, sort per baris menempatkan
statistik urutan ke-k pada τ ke-k, menjamin `crossing_rate = 0` secara
struktural. Skrip: `xgb_rearrangement_walkforward.py` (repo root), ~2,89 jam
di CPU Mac — ongkos yang sama dengan walk-forward asli, karena butuh fit
ulang, bukan operasi post-hoc murah.

**Cek reproduksibilitas — cocok persis dengan run resmi 27 Agu:**

| | run resmi (27 Agu) | run rearrangement (30 Agu), sebelum sort |
|---|---:|---:|
| `best_iteration` per fold | 375, 242, 255, 297, 390 | 375, 242, 255, 297, 390 |
| K1 (fold 1/2/4 bersih) | 2,9433 | 2,9433 |
| `crossing_rate` per fold | 0,966–0,987 | 0,966–0,987 |

Kecocokan persis ini mengonfirmasi run kedua benar-benar mereproduksi walk-
forward asli (bukan variasi acak lain), jadi angka sesudah rearrangement sah
dibandingkan langsung ke angka resmi.

**Sesudah rearrangement:**

| | sebelum | sesudah |
|---|---:|---:|
| K1 (fold 1/2/4 bersih) | 2,9433 | **2,9115** |
| K1 (5 fold) | 2,9197 | 2,8910 |
| `crossing_rate` | 0,9767 | **0,0000** |

K1 membaik **1,08%** (2,9433 → 2,9115) dan crossing hilang seluruhnya, sesuai
jaminan sort. **Tapi ini tidak membalikkan peringkat**: gap ke RF (K1
2,8508) menyempit dari 3,24% menjadi **2,13%** — masih di atas ambang
keputusan 2% (bagian 17 metodologi). Dibandingkan LSTM, XGBoost-rearranged
(2,9115) kini di antara seed 42 (2,8818, resmi tapi terbukti bukan
representatif — `docs/hasil-modeling-lstm.md` bagian 5.1b) dan seed 43
(3,0732) — lebih baik dari seed 43, sedikit lebih buruk dari seed 42.

**K2 (fold 1/2/4 bersih) sesudah rearrangement:**

| τ | coverage | gap terhadap τ |
|---:|---:|---:|
| 0,05 | 0,4408 | +0,3908 |
| 0,10 | 0,4604 | +0,3604 |
| 0,15 | 0,4808 | +0,3308 |
| 0,20 | 0,5027 | +0,3027 |
| 0,25 | 0,5251 | +0,2751 |
| 0,30 | 0,5470 | +0,2470 |
| 0,35 | 0,5706 | +0,2206 |
| 0,40 | 0,5952 | +0,1952 |
| 0,45 | 0,6218 | +0,1718 |
| 0,50 | 0,6496 | +0,1496 |
| 0,55 | 0,6775 | +0,1275 |
| 0,60 | 0,7068 | +0,1068 |
| 0,65 | 0,7375 | +0,0875 |
| 0,70 | 0,7684 | +0,0684 |
| 0,75 | 0,8007 | +0,0507 |
| 0,80 | 0,8352 | +0,0352 |
| 0,85 | 0,8704 | +0,0204 |
| **0,90** | **0,9077** | **+0,0077** |
| 0,95 | 0,9464 | −0,0036 |

Bentuknya sama seperti sebelum rearrangement dan sama seperti RF — over-
coverage besar di τ rendah adalah efek lantai `share_nol` (41,95% baris
target nol, bagian 18 metodologi), bukan sesuatu yang diubah rearrangement.
Di τ=0,90 kalibrasinya tetap dekat target (gap +0,0077).

**Kesimpulan**: rearrangement menjawab prasyarat wajib (`docs/todolist-proyek.md`
butir 9) — XGBoost kini bisa dibandingkan adil dengan crossing yang sudah
dihilangkan. Hasilnya memperkuat, bukan membalikkan, RF sebagai kandidat
terdepan: XGBoost tetap kalah K1, dengan margin yang lebih kecil tapi masih
di atas ambang keputusan.

Artefak: `dataset/model_ready/xgb_walk_forward_results_rearranged.csv`
(model_name `xgboost_rearranged`, tidak masuk git seperti artefak model
lain).

## 6. Model final

`fit_final()` melatih ulang konfigurasi pemenang pada seluruh baris layak
sebelum Desember, dipotong di batas Desember oleh
`purging.lookahead_safe_mask()` — populasi baris yang sama persis dengan yang
dinilai di atas. Dijalankan **di CPU Mac**, mesin yang sama dengan RF.

| | |
|---|---|
| Baris training | 1.349.011 |
| Kolom fitur | 56, encoding `native` |
| Ronde boosting (`best_iteration`) | 201 |
| Titik kuantil tersimpan | 19 (0,05..0,95) |
| Device | cpu |
| Artefak | `models/xgboost_q90.joblib` — 292 MB, 27 Agu 2026 22:38 |

201 ronde jauh di bawah plafon 2.000 yang nyaris tersentuh `DEFAULT_PARAMS` di
benchmark (bagian 3) — konfigurasi pemenang pencarian jauh lebih hemat ronde
untuk konvergensi yang sama atau lebih baik.

## 7. Ongkos (bahan K3)

**Pencarian (GPU Windows) dan walk-forward/fit-final (CPU Mac) tidak
sebanding satu sama lain** — beda device, jadi wall-clock keduanya dilaporkan
terpisah dan tidak dijumlahkan menjadi satu "total run".

| tahap | device | wall clock | sumber |
|---|---|---:|---|
| Benchmark | cpu (Mac) | 265,2 menit (~4,4 jam) | dicetak notebook |
| Pencarian 30 kandidat | cuda (Windows) | ~14,7 jam | jumlah `elapsed_seconds` per kandidat |
| Walk-forward 5 fold | cpu (Mac) | ~3 jam 1 menit | **estimasi** dari selisih timestamp `xgb_search_results.csv` (19:14) ke `xgb_walk_forward_results.csv` (22:15), 27 Agu |
| Fit final | cpu (Mac) | ~23 menit | **estimasi** dari selisih timestamp `xgb_walk_forward_results.csv` (22:15) ke `models/xgboost_q90.joblib` (22:38), 27 Agu |

Dua baris terakhir **tidak dicetak eksplisit oleh notebook** (tidak ada
`print` durasi seperti RF) — diperkirakan dari timestamp berkas, jadi
presisinya di level menit, bukan detik, dan bisa mengandung waktu setup/muat
data yang tidak murni komputasi.

Estimasi di todolist untuk keseluruhan XGBoost (search+WF+final) adalah
**~125 jam**, dihitung untuk skenario seluruhnya di CPU Mac. Karena pencarian
ternyata pindah ke GPU, angka itu **tidak bisa diverifikasi/dibantah** dari
run ini — device-nya beda. Yang **bisa** dibandingkan lurus dengan RF adalah
walk-forward + fit final di CPU Mac yang sama: RF ~93 menit (45+48) lawan
XGBoost **~204 menit (181+23)** — XGBoost sekitar **2,2×** lebih lambat dari
RF di tahap yang device-nya benar-benar identik, konsisten dengan boosting
yang membangun pohon per ronde per titik kuantil vs RF yang membaca semua
titik dari daun yang sama (bagian 2).

## 8. Reproduksi

```bash
# Pencarian — jalankan di mesin GPU (mis. PC Windows), lihat
# docs/runbook-pencarian-gpu-windows.md untuk setelan FORECAST_DEVICE
$env:FORECAST_DEVICE = "cuda"
python run_cells.py notebook\modeling_xgb.ipynb 2-10,14

# Walk-forward + fit final — jalankan di Mac (CPU), satu mesin yang sama
# dengan RF dan LSTM untuk K3
.venv/bin/python3 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebook/modeling_xgb.ipynb
```

Pencarian menulis checkpoint tiap kandidat selesai ke `xgb_search_results.csv`
dan melanjutkan dari sana kalau dijalankan ulang (`resume=True`), dijaga
`_assert_checkpoint_matches()`.

| Artefak | Lokasi | Masuk git |
|---|---|---|
| Hasil pencarian | `dataset/model_ready/xgb_search_results.csv` | tidak |
| Parameter terpilih | `dataset/model_ready/xgb_best_params.json` | tidak |
| Tabel hasil lengkap | `dataset/model_ready/xgb_walk_forward_results.csv` | tidak |
| Model terlatih | `models/xgboost_q90.joblib` | tidak |
| Ringkasan ini | `docs/hasil-modeling-xgb.md` | **ya** |
| Arsip run kuantil-tunggal | `docs/bak/hasil-modeling-xgb.single-quantile.bak.md` | **ya** |

## 9. Batasan

- **Desember 2025 belum dibuka.** Semua angka di sini adalah validasi
  walk-forward, bukan skor test set final.
- **Sumbu waktunya waktu pengambilan, bukan waktu pemesanan** — sama batasan
  dengan RF. Lihat `docs/batasan-penelitian.md` (B-1, B-2, B-3).
- **MAE tidak sebanding lintas model dengan baseline titik-tengah** — sama
  catatan dengan RF. K1 adalah kriterianya.
- **`crossing_rate` = 0,9767 sebagian besar defek sungguhan, dikonfirmasi
  2026-08-29** (bagian 5.2) — bukan "97,7% prediksi XGBoost tidak berguna"
  (sebagian besar derau kecil di bawah toleransi 0,1), tapi **~20–25% baris
  punya crossing material** (≥0,5–1,0 unit) yang tidak hilang dengan toleransi
  wajar. **Rearrangement dijalankan 2026-08-30 (bagian 5.5)**: menghapus
  crossing sepenuhnya dan memperbaiki K1 1,08%, tapi gap ke RF tetap di atas
  ambang 2% (2,13%) — bukan lagi batasan terbuka, tapi juga bukan alasan
  untuk membalik pemenang.
- **K2 di τ rendah** punya keterbatasan yang sama dengan RF — belum bisa
  dibaca sebagai kalibrasi murni sampai aturan penyisihan dinyatakan ulang
  terhadap lantai `share_nol` (dibahas di `hasil-modeling-rf.md` bagian 5.2,
  berlaku sama di sini).
- **Fold 3 dan 5 ikut memilih pemenang** — potongan fold 1/2/4 di bagian 5.1
  adalah angka bersih dan menjadi K1 resmi.
- **Satu seed, satu kali latih.** Tidak ada pengulangan seed untuk XGBoost
  (beda dari LSTM, yang diulang 3 seed karena inisialisasi bobotnya acak;
  XGBoost dengan `random_state` tetap secara umum lebih deterministik, tapi
  klaim itu belum diuji langsung di sini).
- **Peringkat kandidat pencarian jauh kurang stabil terhadap kriteria K1**
  dibanding RF (Spearman 0,73 vs 0,975, bagian 4.3) — pencarian dengan
  anggaran lebih besar kemungkinan masih akan menggeser pemenang lebih jauh
  di XGBoost daripada di RF.
- **Perbandingan lintas model (RF/XGBoost/LSTM) sudah adil, tapi pemenang
  belum resmi dibekukan** — prasyarat wajib (rearrangement XGBoost) sudah
  terjawab 2026-08-30, RF tetap unggul. Bagian 16/18
  `metodologi-pemodelan-dan-pemilihan-model.md` masih menunggu persetujuan
  eksplisit pemilik proyek sebelum tangga keputusan ditutup dalam sebuah
  commit (Fase E, `docs/todolist-proyek.md`).
