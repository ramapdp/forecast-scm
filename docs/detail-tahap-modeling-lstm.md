# Detail Tahap Pemodelan — LSTM Kuantil

| Atribut                    | Keterangan                                                                                                                                                                                    |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Implementasi               | `torch==2.8.0`, `QuantileLSTM` (kustom), kepala keluaran 19-titik                                                                                                                             |
| Modul produksi             | `utils/modelling/model_lstm.py`, `utils/modelling/sequence_windows.py`                                                                                                                        |
| Modul bersama yang dipakai | `utils/modelling/model_common.py`, `walk_forward.py`, `purging.py`, `evaluation.py`, `modeling_prep.py`                                                                                       |
| Notebook                   | `notebook/modeling_lstm.ipynb`                                                                                                                                                                |
| Uji unit                   | `test/test_model_lstm.py`, `test/test_model_common.py`                                                                                                                                        |
| Ruang pencarian            | 144 kombinasi hyperparameter                                                                                                                                                                  |
| Kandidat ditarik           | 30 (dipatok setara XGBoost)                                                                                                                                                                   |
| Run hasil terakhir         | 28 Agustus 2026 — run pertama di bawah kriteria multi-kuantil K1; konfirmasi seed 43 pada 30 Agustus 2026                                                                                     |
| Status                     | Satu-satunya model dengan inisialisasi bobot acak; K1 seed 42 (2,8818) terbukti **bukan** representasi stabil — seed 43 kalah 7,80% dari RF — lihat `docs/detail-tahap-perbandingan-model.md` |

**Hubungan dengan dokumen lain.** Sama seperti dokumen Random Forest dan
XGBoost, dokumen ini fokus pada satu model. Prapemrosesan data ada di
`docs/detail-tahap-preprocessing.md`; kontrak evaluasi bersama, metrik, dan
**perbandingan** ketiga model ada di `docs/detail-tahap-perbandingan-model.md`;
angka yang bisa berubah setiap run diulang ada di `docs/hasil-modeling-lstm.md`,
rujukan utama Bagian 1.7–1.11 di bawah.

Desain aslinya ada di
`docs/superpowers/specs/2026-08-19-lstm-modeling-design.md` dan
`docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`.

---

# Bagian 1 — Akademis (Bahan Laporan)

## 1.1 Pendahuluan dan Posisi dalam Alur Penelitian

LSTM kuantil adalah kandidat ketiga dan satu-satunya model deep learning
di antara tiga model yang dibandingkan. Bedanya paling mendasar dengan
Random Forest dan XGBoost bukan pada fungsi objektifnya — ketiganya
meminimalkan pinball loss (Bagian 1.2) — melainkan pada **bentuk input**
yang mereka konsumsi: RF dan XGBoost menerima ringkasan riwayat permintaan
yang sudah direkayasa tangan (`lag_*`, `roll_mean_*`, `roll_std_*`),
sementara LSTM membaca **28 hari mentah** sebagai jendela sekuens dan
membiarkan jaringannya sendiri menemukan ringkasan yang berguna. Fitur
yang dilihat kedua pendekatan itu identik secara isi (56 kolom yang sama);
yang berbeda adalah jumlah informasi yang benar-benar sampai ke model —
`docs/detail-tahap-perbandingan-model.md` membahas apakah perbedaan ini
terbukti menguntungkan LSTM secara empiris di dataset ini.

## 1.2 Arsitektur

```
49 kanal dinamis  ─→ LSTM (hidden 256, 2 lapis) ─┐
                                                  ├─→ concat ─→ Linear ─→ ReLU
7 kategorikal     ─→ Embedding (dim = min(16, (n+1)//2)) ─┘        ─→ Dropout
                     dibaca di baris prediksi saja                 ─→ Linear ─→ ŷ (19 titik)
```

Jendela 28 hari (`LOOKBACK`), berakhir **di baris prediksi, inklusif**.
Loss-nya jumlah pinball lintas 19 titik kuantil (bukan rata-rata) — sengaja
dijumlah, bukan dirata-rata, supaya gradien tiap titik ekstrem tidak
tenggelam oleh kepadatan titik di tengah grid; K1 sendiri (rata-rata) hanya
berbeda dari loss ini oleh konstanta `len(quantiles)`, sehingga training
objective dan kriteria seleksi tetap fungsi yang sama — properti yang sama
yang dimiliki `reg:quantileerror` pada XGBoost.

Kategorikal dibaca **di baris prediksi saja**, tidak diulang sepanjang
jendela: `Kategori Barang_idx` berubah di dalam 301 segmen nyata, sehingga
"kategori milik segmen ini" bukan hal yang terdefinisi untuk diulang 28
kali. Ukuran embedding datang dari `category_mapping.json` (jumlah level
tertinggi + 1), bukan dari nilai yang kebetulan muncul di baris training
satu fold — cabang yang baru buka setelah model dilatih memetakan ke slot
UNKNOWN 0 dan tetap dalam rentang; alternatifnya gagal berbulan-bulan
kemudian, di produksi, dengan index error.

## 1.3 Empat Detail Konstruksi yang Menentukan Sah-Tidaknya Hasil

1. **Kategorikal dibaca di baris prediksi, tidak diulang sepanjang
   jendela** (Bagian 1.2).
2. **Ukuran embedding dari `category_mapping.json`**, bukan dari fold —
   menjaga stabilitas indeks lintas refresh data.
3. **Scaler dipasang per fold, hanya dari baris training**, lalu dipakai
   kedua fit dalam protokol dua-fit (Bagian 1.4). Berbagi satu scaler
   itulah yang membuat `best_epoch` bermakna sama di kedua fit.
4. **Jendela dipotong dari panel penuh, bukan dari baris layak
   (`eligible_rows()`).** Jendela milik baris validasi 1 Juli menjangkau
   mundur ke Juni, melewati baris yang dihapus warm-up dan purge fold.
   Membaca _fitur_ baris itu aman: setiap jendela berakhir di baris
   prediksinya sendiri dan setiap lag berhenti di H-1, jadi tidak ada
   nilai target yang bisa masuk jendela. Yang dicegah purging adalah
   training atas _label_ baris tersebut, dan itu tetap tidak pernah
   terjadi.

## 1.4 Protokol Dua Fit

Identik alasannya dengan XGBoost: jumlah epoch adalah keputusan kapasitas,
dan validasi fold adalah tempat yang bocor bila dipakai langsung memilih
epoch. `MAX_EPOCHS = 100`, `EARLY_STOPPING_EPOCHS = 5`, ekor purged 30 hari
yang sama (`ES_TAIL_DAYS = 30`, dibagi dengan XGBoost lewat
`model_common.split_early_stopping()`). Model probe dibuang, lalu
diinisialisasi ulang dari seed yang sama dan dilatih pada seluruh baris
training untuk epoch yang ditemukan langkah pertama.

## 1.5 Kontrak Evaluasi Bersama

Sama seperti dua model lain, LSTM tidak memiliki logika fold sendiri.
Yang berbeda secara mekanis: `walk_forward.run_search()`/`run_fold()`
memanggil `fit_predict(train_df, valid_df)` yang sama untuk ketiga model,
tapi LSTM butuh akses ke **panel penuh** untuk membangun jendela sekuens
(Bagian 1.3 poin 4) — sesuatu yang tidak dibutuhkan RF/XGBoost. Ini
diselesaikan dengan `bind_panel()` (Bagian 2.1), bukan dengan mengubah
signature `fit_predict` yang sudah disepakati ketiga model.

## 1.6 Benchmark, dan Ruang Pencarian yang Sempat Dipotong Lalu Dipulihkan

Bagian ini paling instruktif dari ketiga model — termasuk satu keputusan
awal yang ternyata terbalik saat diverifikasi ulang.

Benchmark satu putaran di CPU Mac: `best_epoch = 2`, `sec_per_epoch =
106,8`, wall time 16,0 menit. **CPU mengalahkan MPS (GPU Apple Silicon)
2×** (0,193 s/batch lawan 0,392 s/batch) — MPS tidak punya kernel LSTM
ter-fusi di ukuran hidden yang diuji, keputusan yang sudah berlaku sejak
sebelum migrasi K1 dan tidak diulang pengukurannya di run ini.

**Ruang pencarian sempat dikecilkan pada 19 Agustus 2026**, membuang
`num_layers=2` dan `hidden_size=256` dari `SEARCH_SPACE` karena ongkos per
epoch — dua lapis hidden-128 berongkos 259 detik lawan 104 detik untuk
varian satu lapisnya. Konsekuensinya dicatat eksplisit saat itu: "pencarian
ini tidak pernah menanyakan apakah lapisan kedua akan menolong."

**Keputusan itu dibalik pada 24 Agustus 2026** demi validitas atribusi:
kalau LSTM kalah di K1, ketimpangan anggaran tidak boleh menjadi alasan
yang tidak bisa disingkirkan. Ruang pencarian dipulihkan penuh ke 144
kombinasi, dan hasilnya membuktikan keputusan itu benar — **pemenang
pencarian K1 justru memakai `hidden_size=256, num_layers=2`**, kombinasi
paling mahal yang dulu dipotong. Pertanyaan yang "tidak pernah ditanyakan"
sekarang punya jawaban: ya, lapisan kedua menolong.

Anggaran kandidat juga dipatok **30** (setara XGBoost), bukan diturunkan
dari plafon waktu 8 jam (yang, bila dipakai, hanya menghasilkan N=14
kandidat) — penyetaraan yang sama-sama demi validitas atribusi: LSTM
satu-satunya model yang inisialisasi bobotnya acak, dan bila ia kalah,
harus bisa dibedakan apakah karena arsitekturnya kurang cocok atau karena
pencariannya paling dangkal.

## 1.7 Ruang Pencarian Hyperparameter dan Pemenangnya

```python
SEARCH_SPACE = {
    "hidden_size":   [64, 128, 256],
    "num_layers":    [1, 2],
    "dropout":       [0.0, 0.2, 0.3],
    "learning_rate": [3e-4, 1e-3],
    "batch_size":    [1024, 2048],
    "log_target":    [False, True],
}
```

144 kombinasi, 30 kandidat, dinilai di **fold 3 dan 5**, di GPU Windows.
Parameter terpilih:

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

**Tidak ada satu pun nilai** yang sama dengan pemenang run kuantil-tunggal
lama (`hidden_size=128`, `num_layers=1`, `dropout=0,3`, `batch_size=1024`,
`log_target=False`) — perbandingan peringkat kandidat id-per-id seperti RF
dan XGBoost **tidak berlaku** di sini karena ruang pencariannya sendiri
berbeda, bukan cuma kriterianya (Bagian 1.6).

Lima bacaan dari sebarannya (tabel lengkap di `docs/hasil-modeling-lstm.md`
Bagian 4.1–4.2):

1. **Ruang parameternya jauh lebih curam** daripada RF/XGBoost: rentang K1
   20,8% — lebih dari 3× rentang XGBoost (6,8%) dan hampir 2× RF (11%).
   Lima kandidat teratas berjarak 2,25% — juga terlebar dari ketiganya.
   Hyperparameter LSTM jauh lebih menentukan hasil akhir di dataset ini.
2. **`dropout=0` konsisten terbaik** — regularisasi tambahan justru
   merugikan, masuk akal untuk model yang sudah dibatasi anggaran epoch
   oleh early stopping ketat.
3. **`hidden_size` monoton**: lebih besar lebih baik pada rentang yang
   diuji, alasan langsung kenapa memulihkan dimensi ini (Bagian 1.6)
   penting.
4. **`log_target=False` menang secara agregat, tapi pemenangnya sendiri
   memakai `log_target=True`** — bukan kontradiksi: interaksi antar
   parameter lebih kental daripada efek utama tunggal mana pun di LSTM,
   konsisten dengan rentang lebar di poin 1.
5. **`crossing_rate` sudah bervariasi lebar di tahap pencarian**
   (0,125–0,483 di antara 30 kandidat) — jauh dari nol, tapi juga jauh
   dari tinggi seragam seperti XGBoost, dan tidak berkorelasi jelas
   dengan K1.

## 1.8 Gerbang Kelayakan (G0) dan Hasil Walk-Forward Lima Fold

**G0.** LSTM menang di kelima fold dengan margin 43%–48% — lolos.

**K1 (seed 42, angka resmi di seluruh dokumen ini sampai Bagian 1.9):**

| model               | K1 (fold 1/2/4) | K1 (5 fold) |
| ------------------- | --------------: | ----------: |
| **lstm (seed 42)**  |      **2,8818** |  **2,8828** |
| `naive_roll_mean_7` |          4,8603 |      4,8231 |
| `naive_lag_1`       |          8,1612 |      8,1755 |
| `naive_zero`        |         14,8102 |     14,7469 |

**41% lebih baik** daripada baseline terbaik, dan **kalah tipis** dari
Random Forest (2,8508) sebesar 0,0310 (1,1%). LSTM **menang MAE@0,9**
melawan RF (14,065 lawan 15,055). K1 bergerak 2,721–3,079 antar fold
(rentang 13,2%), stabilitas yang sebanding dengan RF (13%) dan XGBoost
(12%).

**Angka di atas tidak boleh dibaca berdiri sendiri** — lihat Bagian 1.9.

## 1.9 Derau Antar-Seed: Temuan Paling Penting di Dokumen Ini

LSTM satu-satunya dari ketiga model dengan inisialisasi bobot **acak**
(forest dan boosting-nya deterministik pada `random_state` tetap), sehingga
selisih K1 kecil ke model lain tidak bisa dipisahkan dari derau seed tanpa
pengulangan.

**Langkah 1 — tiga seed pada fold pencarian saja (3&5):**

| seed | K1 (fold 3&5) |
| ---: | ------------: |
|   44 |        2,8399 |
|   42 |        2,8617 |
|   43 |        3,0915 |

Rentang **0,2517** — delapan kali lebih lebar daripada jarak K1 LSTM ke RF
(0,0310). Ini indikasi tak-langsung: fold 3&5 bukan fold 1/2/4 yang dipakai
kriteria resmi.

**Langkah 2 — konfirmasi langsung, walk-forward 5-fold penuh dengan seed
43** (dikerjakan 30 Agustus 2026, ~9,8 jam CPU Mac, pada fold yang **sama
persis** dipakai untuk klaim K1 resmi):

|                         | seed 42 (resmi) |    seed 43 |
| ----------------------- | --------------: | ---------: |
| K1 (fold 1/2/4, bersih) |      **2,8818** | **3,0732** |

Selisih **0,1914 (6,6%)** — konfigurasi identik, fold identik, hanya
`random_state` yang berbeda. Ini jauh melebihi ambang keputusan K1 (2%).
Dengan seed 43, LSTM kalah dari RF **7,80%**, dan bahkan lebih buruk dari
XGBoost (3,0732 lawan 2,9433) — bukan lagi "hampir seri", tapi model
terburuk dari ketiganya pada titik data ini. Rata-rata dua seed (2,9775)
juga tetap kalah dari RF melebihi ambang 2%.

**Bacaan yang benar:** dengan n=2, ini bukan interval kepercayaan yang
ketat secara statistik, tapi cukup untuk menunjukkan bahwa **K1 seed 42
bukan representasi LSTM yang stabil** — ia mendarat di ujung yang
menguntungkan dari sebaran yang lebar, bukan titik tengah. Klaim "LSTM
hampir seri dengan RF" yang sebelumnya bersandar pada seed 42 saja
**ditarik**. Konsekuensinya untuk keputusan pemenang lintas model ada di
`docs/detail-tahap-perbandingan-model.md`.

## 1.10 Kalibrasi (K2) dan `crossing_rate`

Di τ=0,9 (angka seed 42), coverage LSTM 0,906 — di antara RF (0,928) dan
XGBoost (0,902) — dengan fill rate 0,955. Bentuk kalibrasi lintas 19 titik
sama dengan RF/XGBoost (over-coverage terbesar di paruh bawah grid, efek
lantai `share_nol`), levelnya juga di tengah kedua model lain di
τ=0,90–0,95.

**`crossing_rate = 0,4345`** (43,4% baris) — di antara RF (0%, struktural)
dan XGBoost (97,7%). Berbeda arah dari XGBoost saat diuji dengan toleransi
jarak minimum: rate LSTM **ambruk** ke 0,088 di toleransi 0,01 dan 0,011
di toleransi 0,1 — median inversi hanya 0,0027 unit (bandingkan XGBoost,
0,043 unit). Kepala keluaran 19-neuron LSTM juga tidak dipaksa monoton
secara arsitektur, tapi **secara empiris hasilnya hampir seluruhnya derau
angka mengambang**, bukan kesalahan urutan yang berarti. `crossing_rate`
LSTM boleh dibaca sebagai tidak bermasalah secara praktis untuk keputusan
stok — rearrangement post-hoc tidak mendesak diperlukan seperti pada
XGBoost.

Perlu dicatat: angka kalibrasi ini, seperti K1, berasal dari seed 42 —
belum diverifikasi apakah kalibrasi LSTM ikut goyang sebesar K1-nya bila
diukur dengan seed 43.

## 1.11 Hasil per Segmen Permintaan dan per Hari Pengiriman

| segmen       |       n | K1 LSTM (seed 42) |       K1 RF | K1 XGBoost |
| ------------ | ------: | ----------------: | ----------: | ---------: |
| smooth       |  45.485 |           11,0092 | **10,9478** |    11,0466 |
| erratic      |  54.511 |            5,4961 |  **5,4788** |     5,4969 |
| lumpy        | 123.545 |            1,1664 |  **1,1430** |     1,1823 |
| intermittent | 122.006 |            0,4236 |  **0,4194** |     0,4978 |

LSTM menang K1 di keempat segmen melawan baseline, dan **selalu ada di
tengah** RF dan XGBoost di setiap segmen — tidak pernah menjadi yang
terbaik mutlak, tapi juga tidak pernah menjadi yang terburuk. Di hari
kirim, LSTM punya shortfall terendah dari ketiga model (36.449 unit,
lawan RF 161.063 dan XGBoost 40.237) — baris paling penting secara bisnis
adalah tempat LSTM tampil paling kuat relatif, meski itu satu angka dari
run seed tunggal yang belum diverifikasi terhadap derau (Bagian 1.9).

## 1.12 Model Final dan Interpretasi Bisnis

Model final (seed 42): 1.349.011 baris training, `best_epoch = 5`.
Artefak `models/lstm_q90.joblib` hanya **3,7 MB** — jauh lebih kecil dari
RF (826 MB) dan XGBoost (292 MB), karena bobot jaringan jauh lebih ringkas
daripada struktur pohon tersimpan, meski wall-clock pelatihannya paling
mahal dari ketiganya (Bagian 2.6).

|                 | kekurangan (shortfall) | kelebihan (overstock) |
| --------------- | ---------------------: | --------------------: |
| `lstm`          |                461.320 |             4.351.815 |
| `random_forest` |                418.250 |             4.793.038 |
| `xgboost`       |                500.579 |             4.132.651 |

LSTM ada **di tengah** RF dan XGBoost di kedua sisi pertukaran shortfall
vs overstock, konsisten dengan coverage-nya yang juga di tengah.

## 1.13 Batasan dan Hal yang Belum Bisa Disimpulkan

- **Desember 2025 belum dibuka.**
- **Satu seed (42) dipakai sebagai angka kepala di Bagian 1.8, 1.10, 1.11,
  1.12 — dan ini terbukti bukan representasi stabil (Bagian 1.9).** Ini
  batasan paling penting di dokumen ini: setiap angka LSTM di atas selain
  bagian derau seed harus dibaca sebagai satu titik dari sebaran lebar,
  bukan performa "LSTM" yang stabil.
- **Peringkat kandidat pencarian tidak bisa dibandingkan dengan run
  kuantil-tunggal lama** (Bagian 1.7) — beda dari RF/XGBoost, ruang
  pencarian LSTM sendiri berubah, bukan cuma kriterianya.
- **Fold 3 dan 5 ikut memilih hyperparameter** — potongan fold 1/2/4
  adalah angka bersih.
- **Pengulangan seed hanya di fold 3&5 untuk langkah eksploratif; hanya
  seed 43 yang diverifikasi lewat walk-forward 5-fold penuh.** Seed 44
  belum diuji pada fold 1/2/4.

---

# Bagian 2 — Teknis (Mendetail)

## 2.1 Struktur Modul dan Fungsi

`utils/modelling/model_lstm.py`:

| Fungsi                                                     | Peran                                                                                                                 |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `QuantileLSTM` (kelas `nn.Module`)                         | Arsitektur: embedding per kategorikal + LSTM atas kanal dinamis + kepala `Linear-ReLU-Dropout-Linear`                 |
| `pinball_loss(prediction, target, quantiles)`              | Fungsi loss training — jumlah pinball lintas grid, dijelaskan Bagian 1.2                                              |
| `embedding_sizes(mapping, idx_cols)`                       | `(num_embeddings, embedding_dim)` per kolom `_idx`, dari `category_mapping.json`                                      |
| `build_model(params, n_dynamic, sizes, seed, n_quantiles)` | Konstruksi model dengan seed eksplisit — memastikan dua fit protokol dua-fit mulai dari bobot identik                 |
| `resolve_device(name)`                                     | Validasi eksplisit ketersediaan MPS/CUDA sebelum training dimulai, bukan gagal di tengah loop                         |
| `fit_with_early_stopping(...)`                             | Fit pertama: berhenti pada epoch terbaik di ekor purged                                                               |
| `fit_epochs(...)`                                          | Fit kedua: seluruh baris training, epoch tetap dari fit pertama                                                       |
| `bind_panel(panel, ...)`                                   | Mengikat panel penuh ke closure `make_fit_predict`, memberi `model_common.run_search()` callable bersignature standar |
| `make_fit_predict(params, index, ...)`                     | Callable `(train, valid) -> ndarray` yang sebenarnya dipanggil `walk_forward.run_fold()`                              |
| `run_seed_repeats(df, params, seeds, folds, ...)`          | Menjalankan satu konfigurasi pada beberapa seed, dipakai Bagian 1.9 langkah 1                                         |
| `fit_final(df, params, ...)`                               | Dua-fit pada seluruh baris layak sebelum Desember                                                                     |
| `predict_bundle(bundle, panel, frame)`                     | Inferensi — **mewajibkan panel**, bukan opsi, karena LSTM tidak bisa memprediksi dari satu baris sendirian            |

`utils/modelling/sequence_windows.py` — modul terpisah yang **tidak tahu
apa pun tentang LSTM**, hanya tentang memori dan indeks:

| Fungsi                                            | Peran                                                                                                                                            |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `build_index(panel, feature_cols, lookback, ...)` | Satu view kontigu dari seluruh panel (`float32`, `(n_baris, n_fitur_dinamis)`) plus array posisi/segmen/tanggal untuk pengalamatan jendela       |
| `window_ends(index, frame)`                       | Posisi baris di `index["values"]` untuk frame baris prediksi, dalam urutan baris frame sendiri                                                   |
| `gather(values, ends, lookback)`                  | `(len(ends), lookback, n_fitur)` — jendela yang berakhir di tiap posisi, via `sliding_window_view` (tanpa copy sampai batch benar-benar diambil) |

## 2.2 Jendela Sekuens Tanpa Materialisasi Tensor Padat

Tensor padat `1.502.522 × 28 × 56` float32 berukuran **9,42 GB** — tidak
muat di memori mesin 16 GB yang dipakai. `sequence_windows.build_index()`
menyimpan bentuk kontigu **294 MB** (`(n_baris, n_fitur_dinamis)`), dan
setiap jendela adalah `sliding_window_view` — slice bertingkat (_strided
view_) atas array itu, bukan salinan. Alokasi baru hanya terjadi saat
sebuah batch benar-benar diambil lewat fancy indexing (`gather()`).

Ini bergantung pada satu invarian yang diverifikasi ulang setiap kali
indeks dibangun, bukan diasumsikan: **panel tidak punya celah tanggal di
dalam satu segmen** (`_assert_dense()`), sehingga aritmetika posisi
(`positions[i] - positions[i-1] == 1`) setara dengan aritmetika tanggal.
Bila properti ini pernah berhenti berlaku, setiap jendela akan diam-diam
menjangkau hari yang salah — karenanya diperiksa lewat assertion eksplisit,
bukan dipercaya begitu saja dari dokumentasi tahap panel (`docs/detail-tahap-preprocessing.md`
Subbab 4.5).

Dua guard tambahan yang dijalankan sekali per `build_index()`:

- `_assert_windows_fit()` — setiap jendela yang layak dipakai (posisi ≥
  lookback) harus seluruhnya berada di satu segmen yang sama dan mencakup
  tepat 28 hari berurutan.
- Target (`EVAL_TARGET_COL`, `TRAIN_TARGET_COL`) tidak boleh menjadi
  kolom dinamis — sebuah kanal target di dalam jendela akan menjadi
  prediktor sempurna bagi dirinya sendiri.

## 2.3 `make_fit_predict()` dan `bind_panel()` — Alur Dua Fit

`bind_panel()` membangun `sequence_windows.build_index()` **satu kali**
(biaya sort ~1,5 juta baris), lalu mengembalikan factory
`make(params, feature_cols, quantiles)` bersignature yang sama dengan
`model_random_forest.make_fit_predict` dan `model_xgboost.make_fit_predict`
— inilah yang memungkinkan `model_common.run_search()` memanggil ketiga
model lewat satu baris kode yang sama, meski hanya LSTM yang butuh akses
ke panel penuh.

Di dalam `fit_predict(train, valid)` yang dikembalikan:

1. `assert_no_nan` pada kedua frame.
2. `window_ends()` menerjemahkan baris `train`/`valid` menjadi posisi di
   indeks, lalu dua guard tambahan dijalankan: `_assert_train_precedes_valid()`
   (memastikan posisi window training benar-benar mendahului validasi —
   bukan mempercayai `fold_train_mask` begitu saja, karena yang bisa
   salah di sini adalah `window_ends()` sendiri) dan `_assert_no_december()`
   (redundan dengan definisi fold, kembali dipertahankan sebagai penjaga
   rangkap).
3. Scaler dipasang dari `train` saja (`modeling_prep.fit_scaler()`), lalu
   diterapkan ke **seluruh** array `values` sekali — aman karena yang
   bocor adalah _memasang_ scaler di luar jendela training, bukan
   _menerapkannya_.
4. `model_common.split_early_stopping()` memecah `train` menjadi
   `fit_rows`/`es_rows` (ekor purged 30 hari) — fungsi yang sama dipakai
   XGBoost.
5. `fit_with_early_stopping()` — fit pertama, mengembalikan `best_epoch`.
   Dicatat ke `fit_predict.best_epochs` (atribut pada closure, dibaca oleh
   `model_common.reported_capacity()`).
6. `fit_epochs()` — fit kedua, seluruh baris `train`, epoch tetap.
7. `predict()` pada `valid_ends`, pembalikan log1p bila dipakai, `clip`
   ke non-negatif.

## 2.4 Konstruksi `QuantileLSTM`

```python
class QuantileLSTM(nn.Module):
    def __init__(self, n_dynamic, sizes, hidden_size, num_layers, dropout, n_quantiles):
        self.embeddings = nn.ModuleList([nn.Embedding(c, d) for c, d in sizes])
        self.lstm = nn.LSTM(n_dynamic, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        width = hidden_size + sum(d for _, d in sizes)
        self.head = nn.Sequential(
            nn.Linear(width, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, n_quantiles),
        )

    def forward(self, x_dynamic, x_cats):
        output, _ = self.lstm(x_dynamic)
        last = output[:, -1, :]                     # state akhir jendela
        embedded = [layer(x_cats[:, i]) for i, layer in enumerate(self.embeddings)]
        return self.head(torch.cat([last, *embedded], dim=1))
```

`nn.LSTM` mengabaikan `dropout` saat `num_layers == 1` — karenanya head di
atas selalu menerapkan dropout-nya sendiri, supaya flag yang dicari tidak
kehilangan makna pada separuh ruang pencarian (`num_layers=1`).

Satu kepala keluaran 19-neuron di atas satu trunk bersama — setiap titik
kuantil membaca state LSTM dan embedding yang sama — lebih murah daripada
19 jaringan terpisah dan memungkinkan titik kuantil saling menginformasikan
satu sama lain lewat bobot bersama. Tidak ada mekanisme yang memaksa
keluarannya monoton terhadap τ; crossing diukur (Bagian 1.10), bukan
dirancang hilang.

## 2.5 Format Bundle

```python
{
    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    "params": params,
    "feature_cols": index["feature_cols"],
    "dynamic_cols": index["dynamic_cols"],
    "idx_cols": index["idx_cols"],
    "embedding_sizes": sizes,
    "scaler": scaler,
    "log_target": params["log_target"],
    "best_epoch": int(best_epoch),
    "quantiles": quantiles,       # urutan head, tidak terekstrak dari state_dict saja
    "lookback": lookback,
    "n_train": int(len(frame)),
    "train_target": "target_lead_time_cumulative_capped",
    "eval_target": "target_lead_time_cumulative",
}
```

Scaler wajib ikut: jaringan yang dimuat ulang lalu diberi fitur berskala
mentah tidak gagal — ia memprediksi dengan percaya diri dari input yang
salah skala. `quantiles` disimpan eksplisit karena urutan kepala keluaran
tidak bisa direkonstruksi dari `state_dict` saja — tanpanya, model yang
dimuat ulang mengembalikan sembilan belas angka tanpa label.

`predict_bundle(bundle, panel, frame)` **mewajibkan** `panel`, bukan
menerimanya sebagai opsi — satu-satunya perbedaan pemakaian dibanding
`predict_bundle()` RF/XGBoost. LSTM tidak bisa memprediksi dari satu baris
sendirian; ia butuh 28 hari di belakangnya.

## 2.6 Ongkos Komputasi

**Pencarian + pengulangan 3 seed (GPU Windows) dan walk-forward/fit-final
(CPU Mac) tidak sebanding satu sama lain** — beda device, sama seperti
XGBoost:

| tahap                                     | device         |      wall clock |
| ----------------------------------------- | -------------- | --------------: |
| Benchmark                                 | cpu (Mac)      |      16,0 menit |
| Pencarian 30 kandidat                     | cuda (Windows) |        ~4,2 jam |
| Pengulangan 3 seed (fold 3&5)             | cuda (Windows) |       ~50 menit |
| Walk-forward 5 fold (seed 42)             | cpu (Mac)      | ~8 jam 28 menit |
| Fit final                                 | cpu (Mac)      | ~1 jam 18 menit |
| Walk-forward 5 fold (seed 43, verifikasi) | cpu (Mac)      |        ~9,8 jam |

**Walk-forward LSTM (~8,5 jam) adalah tahap tunggal termahal yang terukur
di seluruh tiga model** — lebih lama dari walk-forward RF (~45 menit) dan
XGBoost (~3 jam) digabung. Pada tahap yang device-nya sebanding lintas
ketiganya (walk-forward + fit final, CPU Mac): RF ~93 menit, XGBoost ~204
menit, **LSTM ~586 menit** — LSTM sekitar **6,3×** lebih lambat dari RF
dan **2,9×** lebih lambat dari XGBoost.

## 2.7 Reproduksi

```bash
# Pencarian + 3-seed repeat — jalankan di mesin GPU
$env:FORECAST_DEVICE = "cuda"
python run_cells.py notebook\modeling_lstm.ipynb 2-10,14,16,18,20,22

# Walk-forward + fit final — jalankan di Mac (CPU), satu mesin yang sama
# dengan RF dan XGBoost untuk K3
.venv/bin/python3 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebook/modeling_lstm.ipynb
```

Pencarian menulis checkpoint tiap kandidat selesai (`resume=True`);
walk-forward **tidak** punya checkpoint — sesi yang terpotong di tengah
kehilangan seluruhnya.

| Artefak                   | Lokasi                                                | Masuk git |
| ------------------------- | ----------------------------------------------------- | --------- |
| Hasil pencarian           | `dataset/model_ready/lstm_search_results.csv`         | tidak     |
| Pengulangan seed          | `dataset/model_ready/lstm_seed_repeats.csv`           | tidak     |
| Parameter terpilih        | `dataset/model_ready/lstm_best_params.json`           | tidak     |
| Tabel hasil lengkap       | `dataset/model_ready/lstm_walk_forward_results.csv`   | tidak     |
| Model terlatih            | `models/lstm_q90.joblib` (3,7 MB)                     | tidak     |
| Ringkasan hasil           | `docs/hasil-modeling-lstm.md`                         | ya        |
| Arsip run kuantil-tunggal | `docs/bak/hasil-modeling-lstm.single-quantile.bak.md` | ya        |

## 2.8 Strategi Pengujian

```bash
.venv/bin/python3 -m unittest test.test_model_lstm -v
.venv/bin/python3 -m unittest test.test_model_common -v
```

`test/test_model_lstm.py` menguji `sequence_windows` (kepadatan, batas
segmen, `window_ends`) dan `model_lstm` (bentuk keluaran, guard G2/G3/G5,
format bundle) dengan fixture kecil yang tidak menuntut `model_input.parquet`
penuh maupun GPU.

## 2.9 Rujukan

| Topik                                              | Berkas                                                                  |
| -------------------------------------------------- | ----------------------------------------------------------------------- |
| Desain model                                       | `docs/superpowers/specs/2026-08-19-lstm-modeling-design.md`             |
| Desain evaluasi multi-kuantil                      | `docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md` |
| Angka hasil terukur                                | `docs/hasil-modeling-lstm.md`                                           |
| Runbook GPU Windows                                | `docs/runbook-pencarian-gpu-windows.md`                                 |
| Prapemrosesan sampai `model_input.parquet`         | `docs/detail-tahap-preprocessing.md`                                    |
| Mesin evaluasi bersama & perbandingan lintas model | `docs/detail-tahap-perbandingan-model.md`                               |
