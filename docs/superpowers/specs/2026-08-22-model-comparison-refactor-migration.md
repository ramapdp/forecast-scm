# Migrasi perbandingan model ke evaluasi multi-kuantil — refactor plan

## Status

**Menggantikan pendekatan evaluasi di `metodologi-pemodelan-dan-pemilihan-model.md`
bagian 15–18 sebelum test set Desember 2025 dibuka.** Test set belum pernah
dibuka (dikonfirmasi bagian 1 `hasil-modeling-rf.md`: "Desember 2025 tidak
dibuka"), sehingga migrasi ini **tidak membuang hasil evaluasi out-of-sample
apa pun** — yang perlu diulang hanya pencarian hyperparameter dan
walk-forward di kelima fold latih (Juli–November 2025), bukan pengujian
final.

**Dokumen ini murni checklist eksekusi.** Untuk penjelasan apa yang
berubah dan kenapa (definisi K1/K2/K3 baru, mekanisme dua tahap penentuan
kuantil, dasar bukti dari jurnal), lihat
`2026-08-22-multi-quantile-evaluation-design.md` — dokumen ini tidak
mengulang isinya, hanya merujuknya per poin.

## Catatan eksekusi (2026-08-24)

Checklist ini mulai dijalankan 2026-08-24. Status per butir:

| Butir | Status |
|---|---|
| 1 — spec XGBoost | ✅ selesai |
| 2 — spec LSTM | ✅ selesai |
| 3 — spec Random Forest | ✅ selesai |
| 0b — implementasi kode multi-kuantil | ✅ selesai 2026-08-24 (lihat di bawah) |
| 4 — `metodologi` bagian 15–18 | ⚠️ sebagian: definisi K1/K2 di bagian 15 dan bagian 17 selesai, plus bagian 19 dan bagian 21 (perluasan cakupan, disetujui pemilik proyek 2026-08-24). bagian 16 dan bagian 18 diberi penanda "menunggu penggantian" dan ditulis ulang setelah butir 5 |
| 5 — `hasil-modeling-{rf,xgb,lstm}.md` | ⬜ menunggu notebook dijalankan ulang |
| 6 — spec segmentasi kuantil | ✅ selesai |
| 7 — `batasan-penelitian.md` / `pipeline-overview.md` | ⚠️ direvisi, lihat "Koreksi butir 7" di bawah |

**Prasyarat yang belum tercatat di checklist ini.** "Urutan eksekusi yang
disarankan" langkah 2 meminta ketiga notebook dijalankan ulang "sesuai spec yang
sudah diubah", tetapi tidak ada butir yang mencakup **perubahan kodenya**. Kode
saat ini masih kuantil tunggal dari ujung ke ujung: `evaluation.py`
`DEFAULT_ALPHA = 0.9` dengan `score()` mengembalikan satu angka pinball,
`walk_forward.py` dan `model_common.py` menerima `alpha: float` skalar, dan tidak
ada `QUANTILE_SET` di mana pun. Langkah 2 tidak dapat dijalankan sebelum
`evaluation.py`, `walk_forward.py`, `model_common.py`, `model_xgboost.py`,
`model_lstm.py`, `model_random_forest.py`, dan ketiga notebook diubah. Pekerjaan
itu dicatat sebagai butir 0b di bagian 21 `metodologi-pemodelan-dan-pemilihan-model.md`.

**Koreksi angka kandidat.** Butir 1 di bawah menyebut "18 kandidat" untuk
XGBoost, dan tabel "Dampak teknis per model" di
`2026-08-22-multi-quantile-evaluation-design.md` menyebut angka yang sama. Itu
keliru — 18 adalah anggaran **Random Forest**. Anggaran XGBoost yang benar-benar
dijalankan adalah **30** (`dataset/model_ready/xgb_search_results.csv` berisi 30
baris; `docs/hasil-modeling-xgb.md` bagian "Pencarian hyperparameter"). LSTM = 12 sudah
benar. Angka yang berlaku: **XGBoost 30, LSTM 12.**

**Keputusan anggaran pencarian (2026-08-24).** Pertanyaan terbuka nomor 2 di spec
multi-kuantil, bagian anggaran, **ditutup**: anggaran dipertahankan pada 30
(XGBoost) dan 12 (LSTM), tidak dikurangi meskipun tiap kandidat kini memprediksi
19 kuantil sekaligus. Dasarnya konsisten dengan posisi proyek yang sudah
tertulis: ongkos komputasi sengaja dikesampingkan karena tujuannya menemukan
model terbaik. Untuk LSTM ini berarti N **dipatok** 12, bukan diturunkan ulang
dari formula anggaran — konsekuensinya plafon 8 jam kemungkinan terlampaui, dan
itu dicatat sebagai ongkos terukur, bukan kegagalan (lihat bagian 2.2 spec LSTM).
Bagian *warm start* dari pertanyaan terbuka nomor 2 (peralihan Tahap A → Tahap B)
**tetap terbuka**.

**Revisi keputusan anggaran (2026-08-24, sesudah T-7).** Paragraf di atas
dipertahankan sebagai jejak keputusan, tetapi angka LSTM-nya **diganti**:
anggaran LSTM naik dari 12 menjadi **30 kandidat**, ruang pencariannya
dipulihkan dari 48 ke **144** di `utils/model_lstm.py` (`num_layers` dan
`hidden_size` dikembalikan), dan konfigurasi terbaiknya diulang pada **3 seed**.
Angka yang berlaku sejak revisi ini: **RF 18, XGBoost 30, LSTM 30.** Dasarnya
bukan perubahan posisi soal ongkos melainkan validitas atribusi ketika ketiga
model dicari ulang dari nol — uraian lengkap di bagian 21
`docs/metodologi-pemodelan-dan-pemilihan-model.md` dan bagian 2.2
`2026-08-19-lstm-modeling-design.md`. Konsekuensinya butir 0c menjadi jauh lebih
mahal, dan perkiraan ongkosnya dilaporkan sebelum butir itu dijalankan.

**Butir 0b selesai (2026-08-24).** Implementasi multi-kuantil mendarat di
`utils/evaluation.py`, `utils/walk_forward.py`, `utils/model_common.py`,
`utils/model_random_forest.py`, `utils/model_xgboost.py`, `utils/model_lstm.py`
dan ketiga notebook. Kontrak `fit_predict` kini `(n, len(QUANTILE_SET))`;
`alpha: float` diganti `quantiles: tuple` di seluruh jalur; `pinball` di CSV
pencarian adalah K1; bundle ketiga model memakai kunci `quantiles`, dengan
`QUANTILE = 0.9` dipertahankan terpisah sebagai konstanta service level B-9.
Ruang pencarian LSTM dipulihkan ke 144 dan protokol tiga seed ditambahkan
(`lstm.run_seed_repeats()` -> `dataset/model_ready/lstm_seed_repeats.csv`).
739 tes lolos. **Notebook diubah tetapi tidak dijalankan** — itu butir 0c.

**Dua keputusan metodologis yang menutup prasyarat Fase 3 (2026-08-24,
pemilik proyek).** Keduanya harus diambil sebelum butir 0c dijalankan, karena
mengubahnya sesudah run berarti mengulang ~157–172 jam:

1. **Target: latih di `..._capped`, nilai (K1) di target mentah.** Kode
   sebelumnya memakai target mentah untuk keduanya, sementara konfirmasi pemilik
   data 2026-08-17 menyimpulkan capped — kontradiksi yang tidak pernah
   diselesaikan. Sekarang dibelah: `modeling_prep.TRAIN_TARGET_COL` vs
   `EVAL_TARGET_COL` (nama `TARGET_COL` yang ambigu dihapus), satu seam label
   latih `model_common.train_target()`, dan guard baru di
   `walk_forward.eligible_rows()` + `prepare_forecast_data` untuk pola nilai
   kosong yang tidak sepakat. Ketiga bundle kini mencatat
   `train_target`/`eval_target`. 752 tes lolos. Ongkos yang dinyatakan terbuka:
   selisih kedua target di Desember 2025 = 1.223/50.692 baris (2,41%) dan 44.470
   unit (2,03% massa) — dibayar sama besar oleh ketiga model, jadi peringkat
   tidak terpengaruh, hanya level absolut K1.
2. **`QUANTILE_SET` tetap 19 titik.** Hemat ~38 jam dari grid 9 titik (T-14)
   **ditolak**: kerapatan grid adalah dasar klaim hampiran CRPS dan grid 9 titik
   membuang kedua ekor, termasuk 0,95 yang berpotensi dipakai alokasi
   tersegmentasi. Lihat pertanyaan terbuka nomor 0 di
   `2026-08-22-multi-quantile-evaluation-design.md`.

Konsekuensinya: **perkiraan ongkos di bawah berlaku apa adanya** (~157–172 jam),
dan tidak ada lagi keputusan tertunda yang bisa membatalkan hasil run.

## Prosedur Fase 3 (butir 0c) — menunggu izin terpisah

Fase 3 adalah menjalankan ketiga notebook. Urutannya tidak bebas dan langkah
nol-nya sengaja dibuat gagal.

### Langkah 0 — biarkan guard checkpoint berbunyi. Jangan hapus CSV di muka.

Tiga berkas checkpoint dari run kuantil-tunggal masih ada di disk:

```
dataset/model_ready/rf_search_results.csv
dataset/model_ready/xgb_search_results.csv
dataset/model_ready/lstm_search_results.csv
```

Ketiganya ditulis dengan ruang pencarian dan seed yang **sama** dengan run
yang akan dijalankan, jadi pemeriksaan parameter di
`model_common._assert_checkpoint_matches()` menerima semuanya. Tanpa guard
skema, pencarian XGBoost akan melihat 30 kandidat "sudah selesai", melewati
seluruhnya, dan mengembalikan angka pinball@0,9 dari 2026-08-19 — yang lalu
ditulis sebagai K1 (T-13).

Guard-nya sudah ada: checkpoint tanpa kolom `headline_quantile` ditolak dengan
pesan eksplisit. **Ketiga berkas itu tidak dihapus sebelum run**, dan itu
keputusan yang diambil sadar (pemilik proyek, 2026-08-24), bukan kelalaian:

> Kalau dihapus sebelum run, guard yang baru dibuat tidak pernah teruji di
> kondisi nyata, dan kita tidak akan tahu apakah ia benar-benar bekerja
> sampai suatu saat dibutuhkan dan ternyata tidak. Membiarkannya berbunyi
> memberi konfirmasi langsung bahwa perlindungannya nyata.

Jadi urutan yang benar:

1. Jalankan sel pencarian XGBoost **dengan ketiga CSV masih di tempatnya**.
   Ia harus berhenti dalam hitungan detik dengan `ValueError` yang menyebut
   "berasal dari run kuantil tunggal".
2. Catat bahwa ia berbunyi. Kalau **tidak** berbunyi, hentikan Fase 3 — itu
   berarti guard-nya tidak bekerja dan setiap angka sesudahnya tidak dapat
   dipercaya.
3. Baru hapus ketiga berkas, lalu jalankan ulang dari awal.

Sejak pencarian RF ikut diulang (2026-08-24), **`rf_best_params.json` juga
dihapus** bersama ketiga CSV itu. Berkas itu berasal dari run 2026-08-18 di
atas data pra-reclass; membiarkannya berarti pencarian RF yang baru menimpanya
tanpa ada yang salah, tetapi kalau selnya suatu saat dikembalikan ke jalur
"pakai ulang kalau ada", berkas basi itu yang akan terpilih diam-diam.

Sel notebook yang bersangkutan sudah memuat catatan ini, supaya berhentinya
tidak terbaca sebagai kecelakaan.

### Langkah 1–3 — urutan per model

| # | Model | Yang dijalankan | Yang tidak |
|---|---|---|---|
| 1 | Random Forest | benchmark, **pencarian 18 kandidat**, walk-forward 5 fold, fit final | — |
| 2 | XGBoost | benchmark, pencarian 30 kandidat, walk-forward 5 fold, fit final | — |
| 3 | LSTM | benchmark, pencarian 30 kandidat pada ruang 144, **pengulangan 3 seed pada pemenang**, walk-forward 5 fold, fit final | — |

**Pencarian RF ikut diulang — keputusan dibalik 2026-08-24 (pemilik proyek).**
Revisi sebelumnya memakai ulang `rf_best_params.json` apa adanya, dengan alasan
hyperparameter forest membentuk daun dan seluruh `QUANTILE_SET` dibaca dari daun
yang sama. Alasan itu masih berlaku dan tetap tercatat di Part 2
`2026-08-18-random-forest-modeling-design.md`, tetapi tidak menyentuh
keberatan yang membalikkannya: params itu dipilih 2026-08-18, **sebelum**
reclass WIP-2 masuk ke artefak (dibangun ulang 2026-08-23 22:52). Kebasian itu
sudah diterima sebagai alasan membuang bundle RF; memakai ulang hyperparameter
yang dipilih di atas data yang sama adalah posisi yang tidak konsisten dengan
keputusan itu. Anggarannya tidak berubah (18 kandidat, `SEARCH_FOLDS = (3, 5)`).

Efek sampingnya: premis T-7 — "ketiga model dicari ulang penuh dari nol"
(bagian 21 `metodologi-pemodelan-dan-pemilihan-model.md`) — kini benar apa adanya,
dan asimetri kriteria yang sebelumnya diserahkan ke bagian keterbatasan hilang.

### Perkiraan ongkos Fase 3 (2026-08-24)

**Angka terukur, bukan tebakan — kecuali dua asumsi yang ditandai.** Dasarnya
adalah wall time run 2026-08-18/19/20 yang tercatat di ketiga
`hasil-modeling-*.md`, dikalikan pengganda multi-kuantil yang **diukur hari
ini** pada data sintetis 200.000 baris x 56 kolom (rasio 1 kuantil vs 19
kuantil; yang diperlukan hanya rasionya, dan rasio itu stabil terhadap n).

| Model | Pengganda 19 kuantil | Diukur pada |
|---|---:|---|
| Random Forest | **x1,05** | fit tidak terpengaruh sama sekali; hanya `predict` naik x1,08 |
| XGBoost | **x15,2** | fit x15,2, predict x16,7 |
| LSTM | **x1,00** | head 19 keluaran praktis gratis di sebelah trunk LSTM |

#### Rincian per model per tahap

**Random Forest** — asumsi: satu fit fold 5 = 395 s (terukur, `DEFAULT_PARAMS`);
walk-forward lima fold = 2.220 s (selisih timestamp `rf_best_params.json` ->
`rf_walk_forward_results.csv`, 2026-08-18); fit final = 2x satu fit fold
(400 pohon lawan 200, tanpa predict) — *asumsi, wall time-nya tidak pernah
dicatat*.

| Tahap | Jumlah fit | Perkiraan |
|---|---:|---:|
| Pencarian (18 kandidat x 2 fold) | 36 | **3,9 jam** |
| Walk-forward | 5 | 39 menit |
| Fit final | 1 | 16 menit |
| **Total** | **42** | **~4,8 jam** |

Pencarian dihitung dari `SEARCH_FOLDS = (3, 5)`: fold 5 = 395 s (terukur),
fold 3 diskalakan dari jumlah baris trainingnya (1.149.345 / 1.292.778 = 0,889)
-> 351 s, jadi 746 s per kandidat x 18 x pengganda 1,05 = 3,9 jam. Ini
satu-satunya pencarian dari ketiga model yang bisa diulang tanpa mengubah
skala Fase 3 — bandingkan 64,6 jam (XGBoost) dan 71,5 jam (LSTM).

**XGBoost** — asumsi: 510 s per kandidat untuk dua fold (terukur, 8,5
menit/kandidat pada empat kandidat terakhir run 2026-08-19); walk-forward 960 s;
fit final 240 s. Semuanya dikali 15,2.

| Tahap | Jumlah fit | Perkiraan |
|---|---:|---:|
| Pencarian (30 kandidat x 2 fold x 2 fit) | 120 | **64,6 jam** |
| Walk-forward (5 fold x 2 fit) | 10 | 4,1 jam |
| Fit final (2 fit) | 2 | 1,0 jam |
| **Total** | **132** | **~70 jam (2,9 hari)** |

**LSTM** — asumsi: 3.412 s per kandidat untuk dua fold (terukur, rerata tujuh
kandidat yang sempat mencatat `elapsed_seconds` di
`lstm_search_results.csv` = 6,63 jam); walk-forward 11.156 s; fit final 2.603 s.
Pengganda ruang 144/48 = **x2,52**, diturunkan dari tabel s/epoch terukur
(64/1 = 75 s, 128/1 = 104 s, 128/2 = 259 s) plus dua *asumsi*: hidden 256 =
2x hidden 128 pada kedalaman sama, dan batch 2048 = 0,8x batch 1024.

| Tahap | Jumlah fit | Perkiraan |
|---|---:|---:|
| Pencarian (30 kandidat x 2 fold x 2 fit) | 120 | **71,5 jam** |
| Pengulangan 3 seed (3 x 2 fold x 2 fit) | 12 | 7,2 jam |
| Walk-forward (5 fold x 2 fit) | 10 | 3,1–15,5 jam |
| Fit final (2 fit) | 2 | 0,7–3,6 jam |
| **Total** | **144** | **~83–98 jam (3,4–4,1 hari)** |

Rentang pada dua tahap terakhir bukan ketidaktahuan tentang ongkosnya,
melainkan tentang **siapa yang menang**: ruang 144 kini boleh menghasilkan
pemenang `num_layers=2, hidden_size=256`, yang per epoch sekitar 5x pemenang
lama (`hidden_size=128, num_layers=1`).

#### Total

**~157–172 jam ≈ 6,5–7,2 hari komputasi nonstop** di mesin ini. (Naik 3,9 jam
dari ~153–168 sejak pencarian RF ikut diulang, 2026-08-24.)

#### Ongkos yang dibeli penyetaraan anggaran LSTM

Dipisahkan supaya keputusan 2026-08-24 dapat dinilai harganya:

| | Ongkos LSTM (pencarian + seed) |
|---|---:|
| Tanpa penyetaraan (12 kandidat, ruang 48, tanpa seed) | 11,4 jam |
| Dengan penyetaraan (30 kandidat, ruang 144, 3 seed) | 78,7 jam |
| — dari 12 -> 30 kandidat | +17,1 jam |
| — dari ruang 48 -> 144 (konfigurasi mahal masuk undian) | +43,1 jam |
| — dari 3 seed pada pemenang | +7,2 jam |
| **Selisih** | **+67,3 jam (2,8 hari)** |

Bagian termahalnya bukan jumlah kandidat melainkan **pemulihan ruang**: dua
dimensi kapasitas yang dikembalikan membuat rerata ongkos per kandidat naik
2,5x, dan itulah yang dibayar untuk dapat menjawab "apakah lapisan kedua
menolong" — pertanyaan yang selama ini tercatat sebagai tidak pernah
ditanyakan.

#### T-14 — pengganda XGBoost x15,2 tidak pernah diperhitungkan di spec mana pun

Ini temuan baru dan ia mengubah peringkat ongkos ketiga model. `quantile_alpha`
berisi daftar membuat XGBoost menjadi regresi multi-keluaran, dan
`multi_strategy` bawaannya adalah `one_output_per_tree` — jadi **setiap ronde
boosting membangun 19 pohon, bukan satu**. Spec XGBoost menulis "passing a list
rather than a scalar fits every point in one call", yang benar secara API tetapi
terbaca seolah ongkosnya tidak berubah. Akibatnya XGBoost berbalik dari model
termurah (2,4 menit dua fit lawan 6,6 menit satu fit RF) menjadi salah satu
dari dua yang termahal.

Dua jalan keluar dinilai; keduanya keputusan Anda, bukan keputusan kode:

1. **`multi_strategy="multi_output_tree"`** — satu pohon dengan daun bervektor
   19 per ronde. **Tidak tersedia**: XGBoost 2.1.4 menolaknya untuk
   `reg:quantileerror` (`Update tree leaf support for multi-target tree is not
   yet implemented`). Diuji hari ini, bukan diasumsikan.
2. **Memperkecil `QUANTILE_SET` dari 19 titik ke 9** (0,1–0,9 langkah 0,1) —
   terukur x7,0 alih-alih x15,2, jadi XGBoost turun dari ~70 jam ke ~32 jam
   (hemat ~38 jam). Tetapi ini perubahan metodologi, bukan setelan: kerapatan
   grid adalah alasan rata-rata pinball mendekati CRPS, dan itu ada di
   pertanyaan terbuka nomor 1 spec multi-kuantil. Tidak diubah tanpa keputusan.

Kalau tidak ada yang diubah, ongkos di atas berlaku apa adanya.

### Yang wajib dicatat selama run

- **Wall clock per tahap per model.** Plafon 8 jam LSTM sudah ditinggalkan
  secara sadar; angka sebenarnya masuk ke `docs/hasil-modeling-lstm.md`
  sebagai ongkos terukur, bukan sebagai kegagalan.
- **`crossing_rate`** untuk XGBoost dan LSTM. Di atas beberapa persen, ia
  sinyal bahwa arctan pinball loss (Sluijterman dkk. 2024) sepadan dengan
  kerumitannya. RF harus 0 secara struktural — kalau tidak, ada bug.
- **Selisih baris seed 42** di `lstm_seed_repeats.csv` terhadap baris pemenang
  di `lstm_search_results.csv`. Harus nol. Kalau tidak, yang terukur bukan
  varians seed melainkan nondeterminisme — dan itu temuan tersendiri.
- **Rentang K1 antar seed** dibaca bersama jarak K1 antar model. Kalau
  rentangnya lebih besar, jarak antar model tidak boleh dibaca sebagai
  perbedaan antar arsitektur.

### Yang tidak boleh dilakukan di Fase 3

- Menyandingkan K1 dengan angka pinball@0,9 6,56 dari dokumen hasil lama.
  Lantai naif kini dinilai di seluruh 19 titik; keduanya bukan besaran yang
  sama (T-10).
- Menjalankan `resolve_quantile_set()` dengan cakupan biaya di atas ambang
  tetapi tanpa critical ratio — ia sengaja melempar `NotImplementedError`
  (T-8), dan jalurnya baru ada di butir 3a.

**Koreksi butir 7.** Butir 7 menyatakan tidak ada perubahan pada
`batasan-penelitian.md` dengan alasan "B-9 berbicara soal kuantil 0,9 sebagai
komitmen bisnis, bukan kriteria pemilihan model". Alasan itu benar untuk isi
utama B-9, tetapi tidak untuk seluruh isinya: paragraf **"Konsekuensi"** di bawah
"Klarifikasi lanjutan (2026-08-22)" berbicara eksplisit tentang proses pemilihan
model — "klarifikasi ini tidak mengubah proses pemilihan model" — dan pernyataan
itu menjadi tidak benar setelah migrasi ini. Sebuah catatan koreksi bertanggal
2026-08-24 ditambahkan di bawah paragraf tersebut (disetujui pemilik proyek).
Teks 2026-08-16 dan 2026-08-22 tidak dihapus atau diubah. `pipeline-overview.md`
memang tidak berubah, sesuai butir 7.

**Prasyarat Random Forest yang perlu dibaca bersama butir 3.** "RF tidak perlu
retrain" berlaku untuk **pencarian hyperparameter**, bukan untuk artefak
terlatihnya. `models/random_forest_q90.joblib` basi sejak reclass kategori WIP-2
2026-08-22 (bagian 0 `docs/pipeline-overview.md`, prasyarat bagian 19
`docs/metodologi-pemodelan-dan-pemilihan-model.md`), jadi walk-forward RF dan fit
final-nya tetap harus dijalankan ulang — hanya `rf_best_params.json` yang dipakai
ulang apa adanya.

## Purpose

Menerapkan desain di `2026-08-22-multi-quantile-evaluation-design.md` ke
seluruh spec dan dokumen proyek yang sudah ada, dengan urutan dan rincian
per file yang eksplisit, supaya bisa dieksekusi langsung tanpa perlu
menafsirkan ulang desain metodologinya.

## Dampak pada spec segmentasi kuantil (`2026-08-22-segmented-quantile-allocation-design.md`)

Bagian "Urutan pengerjaan relatif terhadap rencana kerja yang sudah ada"
di spec tersebut perlu diperbarui:

- **Sebelum migrasi ini**: perluasan multi-kuantil dikerjakan *setelah*
  pemenang ditetapkan (Bagian 4 spec tersebut, sebagai pekerjaan lanjutan
  khusus model pemenang).
- **Setelah migrasi ini**: perluasan multi-kuantil sudah selesai dikerjakan
  untuk **ketiga model** sebagai bagian dari K1 yang baru, sebelum pemenang
  ditetapkan. Begitu pemenang dipilih di K1–K3 yang sudah direvisi, ia
  **sudah otomatis punya kapabilitas multi-kuantil** — Bagian 4 spec
  segmentasi kuantil menjadi pekerjaan yang sudah selesai (inherited),
  bukan pekerjaan yang masih perlu dilakukan. Simulasi kalibrasi λ (Bagian
  5 spec tersebut) bisa langsung dimulai begitu pemenang ditetapkan, tanpa
  menunggu perluasan model tambahan.

Ini mempercepat, bukan menambah, jalur menuju segmentasi kuantil —
konsekuensi baik dari migrasi ini yang layak dicatat eksplisit di spec
segmentasi kuantil supaya tidak terlihat seolah menambah beban kerja
berganda.

## Dampak per dokumen (ringkasan cepat, rincian eksekusi di "Documentation updates")

| Dokumen | Dampak |
|---|---|
| `metodologi-pemodelan-dan-pemilihan-model.md` | bagian 15–18 direvisi: definisi K1/K2 (lihat `2026-08-22-multi-quantile-evaluation-design.md` Bagian 2–3), tabel hasil, kesimpulan tangga keputusan — semuanya perlu ditulis ulang dengan angka baru |
| `2026-08-18-random-forest-modeling-design.md` | Bagian evaluasi diperluas ke `QUANTILE_SET`; bagian pencarian hyperparameter **tidak berubah** (RF tidak perlu retrain) |
| `2026-08-19-xgboost-modeling-design.md` | `quantile_alpha` diubah ke daftar; pencarian hyperparameter diulang |
| `2026-08-19-lstm-modeling-design.md` | Arsitektur head diubah; pencarian hyperparameter diulang |
| `hasil-modeling-{rf,xgb,lstm}.md` | **Seluruh angka perlu digenerate ulang** — dokumen-dokumen ini adalah bukti hasil, bukan spec, sehingga tidak "direvisi" tapi dijalankan ulang lalu ditulis ulang dari nol mengikuti template yang sama |
| `2026-08-22-segmented-quantile-allocation-design.md` | Bagian urutan pengerjaan diperbarui (lihat "Dampak pada spec segmentasi kuantil" di atas); Bagian 4 ditandai selesai lebih awal |
| `batasan-penelitian.md` | Tidak ada perubahan isi — B-9 berbicara soal kuantil 0,9 sebagai *komitmen bisnis*, bukan kriteria pemilihan model, jadi tetap valid apa adanya |
| `pipeline-overview.md` | Tidak ada perubahan — migrasi ini di tahap pemodelan, bukan preprocessing |

## Documentation updates (in scope for this work)

1. **`docs/superpowers/specs/2026-08-19-xgboost-modeling-design.md`**: ganti
   `quantile_alpha=0.9` menjadi `quantile_alpha=QUANTILE_SET` (definisi di
   `2026-08-22-multi-quantile-evaluation-design.md` Bagian 1); catat bahwa
   18 kandidat pencarian perlu dijalankan ulang dengan objective baru.
2. **`docs/superpowers/specs/2026-08-19-lstm-modeling-design.md`**: ubah
   spesifikasi arsitektur head dari 1 neuron menjadi `len(QUANTILE_SET)`
   neuron, loss total = jumlah pinball loss lintas kuantil; catat bahwa
   pencarian hyperparameter (12 kandidat) perlu diulang.
3. **`docs/superpowers/specs/2026-08-18-random-forest-modeling-design.md`**:
   tambahkan catatan bahwa evaluasi walk-forward sekarang membaca seluruh
   titik `QUANTILE_SET` dari forest yang sama; **tidak ada perubahan pada
   bagian pencarian hyperparameter**.
4. **`docs/metodologi-pemodelan-dan-pemilihan-model.md`** bagian 15–18: revisi
   definisi K1 (rata-rata pinball di `QUANTILE_SET`, bukan pinball@0,9
   tunggal — definisi lengkap di `2026-08-22-multi-quantile-evaluation-design.md`
   Bagian 2), revisi K2 (coverage dicek per kuantil, Bagian 3), tabel hasil
   tangga keputusan ditulis ulang setelah ketiga model dijalankan ulang.
5. **`docs/hasil-modeling-rf.md`, `docs/hasil-modeling-xgb.md`,
   `docs/hasil-modeling-lstm.md`**: dijalankan ulang penuh dari notebook
   masing-masing setelah perubahan 1–3 diterapkan, ditulis ulang mengikuti
   struktur yang sama (ringkasan, setup evaluasi, benchmark, pencarian
   hyperparameter, hasil walk-forward per fold/segmen/hari-kirim, model
   final, batasan) — bukan diedit sebagian, karena seluruh angka di
   dalamnya berubah.
6. **`docs/superpowers/specs/2026-08-22-segmented-quantile-allocation-design.md`**:
   perbarui bagian "Urutan pengerjaan relatif" sesuai "Dampak pada spec
   segmentasi kuantil" di atas; tandai Bagian 4 (perluasan multi-kuantil)
   sebagai *inherited/selesai lebih awal* begitu migrasi ini dijalankan,
   bukan dihapus (supaya jejak keputusan asli tetap terbaca).
7. Tidak ada perubahan pada `batasan-penelitian.md` atau
   `pipeline-overview.md` — dikonfirmasi di atas.

## Urutan eksekusi yang disarankan

Bukan bebas urutan — beberapa langkah bergantung pada langkah sebelumnya:

1. Terapkan poin 1–3 (ubah spec XGBoost, LSTM, RF) lebih dulu, karena
   poin 4–5 butuh spec model sudah final sebagai acuan implementasi.
2. Jalankan ulang notebook ketiga model sesuai spec yang sudah diubah.
3. Tulis ulang poin 5 (`hasil-modeling-*.md`) dari hasil run tersebut.
4. Baru revisi poin 4 (`metodologi-pemodelan-dan-pemilihan-model.md`
   bagian 15–18), karena tabel hasil di situ mengutip angka dari poin 5.
5. Terakhir, poin 6 (spec segmentasi kuantil) — independen dari 1–5,
   bisa dikerjakan kapan saja, tapi logis ditutup terakhir karena ia
   mengonsumsi hasil dari langkah 1–4 (pemenang model + kapabilitas
   multi-kuantilnya).

## Out of scope

- Perubahan isi metodologi itu sendiri — sepenuhnya mengikuti
  `2026-08-22-multi-quantile-evaluation-design.md`, dokumen ini tidak
  mendefinisikan ulang apa pun.
- Alokasi kuantil tersegmentasi — tetap sepenuhnya di
  `2026-08-22-segmented-quantile-allocation-design.md`.
- Mengubah `target_lead_time_cumulative`, mekanisme purging, atau split
  train/test — seluruhnya dipakai apa adanya.

## References

- `docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`
  — sumber kebenaran metodologi untuk seluruh perubahan di dokumen ini.
- `docs/superpowers/specs/2026-08-22-segmented-quantile-allocation-design.md`
  — Bagian 4 dan bagian urutan pengerjaan yang terdampak migrasi ini.
- `docs/batasan-penelitian.md` B-9 — komitmen kuantil 0,9 yang tidak
  berubah isinya akibat migrasi ini.
- `docs/metodologi-pemodelan-dan-pemilihan-model.md` bagian 15–18 — tangga
  keputusan yang menjadi target revisi dokumen ini.
