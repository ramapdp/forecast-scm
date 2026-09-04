# Detail Tahap Perbandingan Model — Random Forest, XGBoost, LSTM

| Atribut                | Keterangan                                                                                                                                                                             |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Kandidat dibandingkan  | Random Forest kuantil, XGBoost kuantil, LSTM kuantil                                                                                                                                   |
| Mesin evaluasi bersama | `utils/modelling/walk_forward.py`, `model_common.py`, `evaluation.py`                                                                                                                  |
| Kriteria utama (K1)    | Rata-rata `pinball` lintas `QUANTILE_SET` (19 titik, 0,05–0,95), pada potongan fold 1/2/4                                                                                              |
| Ambang keputusan K1    | ≥2% terhadap pesaing terdekat, di bawah itu dibaca sebagai tidak terpisahkan                                                                                                           |
| Status keputusan       | **Random Forest kandidat terdepan berdasarkan berat bukti** (K1, K2) — belum resmi dibekukan, menunggu persetujuan eksplisit pemilik proyek. Ongkos komputasi dicatat sebagai catatan pendamping, bukan kriteria pembanding (Bagian 1.5, 2.6). Test set Desember 2025 masih terkunci |
| Tanggal dokumen        | Ditulis ulang dari `docs/metodologi-pemodelan-dan-pemilihan-model.md` — status keputusan terakhir diperbarui 2026-08-30                                                                |

**Hubungan dengan dokumen lain.** Dokumen ini fokus pada **perbandingan**
tiga model yang sudah dibangun masing-masing — bukan pada cara membangunnya.
Untuk detail konstruksi, ruang pencarian, dan hasil satu model secara
mandiri, rujuk:

- `docs/detail-tahap-modeling-rf.md` — Random Forest kuantil
- `docs/detail-tahap-modeling-xgb.md` — XGBoost kuantil
- `docs/detail-tahap-modeling-lstm.md` — LSTM kuantil

Untuk prapemrosesan data sampai `model_input.parquet` (termasuk perumusan
target, `demand_segment`, `fold_id`, dan kontrak dua adapter), rujuk
`docs/detail-tahap-preprocessing.md`. Batasan yang tidak bisa dihilangkan
dengan kode ada di `docs/batasan-penelitian.md`. Angka mentah yang bisa
berubah tiap run diulang ada di `docs/hasil-modeling-{rf,xgb,lstm}.md` —
dokumen ini merangkum dan membandingkannya, bukan menggantikannya.

> **Riwayat dokumen.** Dokumen ini menggantikan
> `docs/metodologi-pemodelan-dan-pemilihan-model.md`, yang sebelumnya
> memuat strategi prapemrosesan (kini di `docs/detail-tahap-preprocessing.md`),
> konstruksi tiap model (kini di tiga dokumen `docs/detail-tahap-modeling-*.md`
> di atas), dan strategi perbandingan. Bagian terakhir itulah yang bertahan
> dan menjadi isi dokumen ini, ditulis ulang lebih fokus. Riwayat lengkap
> revisi metrik dan hasil tetap terbaca di riwayat git berkas lama.

---

## Daftar Isi

- **Bagian 1 — Akademis (Bahan Laporan)**
  - [1.1 Kerangka Tiga Tingkat Keputusan](#11-kerangka-tiga-tingkat-keputusan)
  - [1.2 Kontrak Evaluasi Bersama](#12-kontrak-evaluasi-bersama)
  - [1.3 Lantai: Tiga Baseline Naif](#13-lantai-tiga-baseline-naif)
  - [1.4 Ringkasan Banding Konstruksi Ketiga Model](#14-ringkasan-banding-konstruksi-ketiga-model)
  - [1.5 Metrik dan Justifikasinya](#15-metrik-dan-justifikasinya)
  - [1.6 Posisi Hasil Saat Ini](#16-posisi-hasil-saat-ini)
  - [1.7 Strategi Pemilihan: Tangga Kriteria Bertingkat](#17-strategi-pemilihan-tangga-kriteria-bertingkat)
  - [1.8 Penerapan Tangga pada Angka yang Ada](#18-penerapan-tangga-pada-angka-yang-ada)
  - [1.9 Protokol Pembukaan Test Set Desember](#19-protokol-pembukaan-test-set-desember)
  - [1.10 Apa yang Akan dan Tidak Akan Disimpulkan](#110-apa-yang-akan-dan-tidak-akan-disimpulkan)
- **Bagian 2 — Teknis (Mendetail)**
  - [2.1 Arsitektur Mesin Evaluasi Bersama](#21-arsitektur-mesin-evaluasi-bersama)
  - [2.2 Definisi Metrik — Implementasi](#22-definisi-metrik--implementasi)
  - [2.3 Tabel Lengkap Hasil Lintas Model](#23-tabel-lengkap-hasil-lintas-model)
  - [2.4 Derau Seed LSTM — Detail Pengujian](#24-derau-seed-lstm--detail-pengujian)
  - [2.5 Rearrangement Kuantil XGBoost — Detail Pengujian](#25-rearrangement-kuantil-xgboost--detail-pengujian)
  - [2.6 Catatan: Ongkos Komputasi Lintas Model](#26-catatan-ongkos-komputasi-lintas-model)
  - [2.7 Risiko Integrasi Produksi (K3)](#27-risiko-integrasi-produksi-k3)
  - [2.8 Status Keputusan dan Rencana Kerja Tersisa](#28-status-keputusan-dan-rencana-kerja-tersisa)
  - [2.9 Rujukan](#29-rujukan)

---

# Bagian 1 — Akademis (Bahan Laporan)

## 1.1 Kerangka Tiga Tingkat Keputusan

Satu prinsip mengatur seluruh desain perbandingan ini:

> **Setiap keputusan dinilai pada data yang tidak ikut membuat keputusan
> itu.**

Prinsip itu diterapkan bertingkat, karena ada tiga jenis keputusan yang
sering tertukar namanya menjadi "menguji model":

| Tingkat | Keputusan yang diambil                          | Dinilai pada                                                                         | Status                  |
| ------- | ----------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------- |
| **A**   | Hyperparameter terbaik di dalam satu arsitektur | Fold 3 & 5 (September, November 2025)                                                | selesai untuk ketiganya |
| **B**   | Arsitektur pemenang antar tiga model            | Fold 1, 2, 4 (Juli, Agustus, Oktober 2025) — potongan yang tidak menyentuh tingkat A | inti dokumen ini        |
| **C**   | Seberapa baik pemenang bekerja                  | Desember 2025                                                                        | belum dibuka            |

Tingkat C **bukan pemilihan** — ia pengukuran. Membalik urutan B dan C
(menguji ketiganya di Desember lalu memilih yang tertinggi) akan membuat
angka final menjadi maksimum dari tiga undian: estimasi yang bias ke atas
secara sistematis dan tidak bisa dipertahankan sebagai ukuran kinerja.

Kekeliruan yang sering muncul di sini adalah anggapan bahwa metrik hanya
lahir dari test set. Tidak: metrik lahir dari **data mana pun yang tidak
dipakai melatih**. Validasi walk-forward proyek ini mencakup 345.547 baris
di lima bulan terpisah, sementara Desember hanya menyumbang 49.717 baris
yang dinilai di satu bulan yang paling tidak mewakili operasi normal (Natal
dan Tahun Baru). Untuk urusan **memilih**, bukti validasi jauh lebih kuat.

## 1.2 Kontrak Evaluasi Bersama

Sebuah perbandingan hanya layak dilaporkan bila ketiga model melihat baris
yang sama persis. Itu bukan sesuatu yang bisa dijamin oleh kedisiplinan
penulisan tiga skrip training terpisah — maka jaminan itu dipindahkan ke
struktur:

> `utils/modelling/walk_forward.py` **memiliki** definisi kelayakan baris,
> batas fold, dan penilaian. Ia tidak tahu apa pun tentang model. Seluruh
> antarmuka model adalah satu callable:
>
> ```python
> fit_predict(train_df, valid_df) -> np.ndarray  # (len(valid), 19)
> ```

Apa pun yang dibutuhkan sebuah model di luar itu — pemilihan fitur,
imputasi, transformasi target, penskalaan — berada di dalam pembungkusnya
sendiri (dijelaskan masing-masing di `docs/detail-tahap-modeling-{rf,xgb,lstm}.md`
Bagian 2), karena justru itulah pilihan-pilihan yang ingin **dipaparkan**
oleh perbandingan ini, bukan disembunyikan.

Bukti bahwa mekanisme ini bekerja: ketiga baseline naif (Bagian 1.3)
mencetak angka yang **sama persis** di ketiga run model, dan
`walk_forward.py` tidak pernah disentuh saat LSTM ditambahkan sebagai
kandidat ketiga.

**Kelayakan baris** (`eligible_rows()`) diterapkan berurutan: (1) Desember
2025 dan sesudahnya dibuang — redundan dengan definisi fold, sengaja
dipertahankan karena ongkos satu kebocoran tak sengaja adalah kredibilitas
angka final; (2) 28 hari pertama tiap segmen dibuang, dihitung pada deret
utuh, tidak pernah di dalam fold; (3) baris tanpa target dibuang. Hasilnya:
**345.547 baris validasi** di lima fold, identik untuk ketiga model.

**Protokol pencarian** juga sama untuk ketiganya: fold pencarian 3
(September) dan 5 (November) saja, kriteria K1 gabungan dibobot jumlah
baris, tanpa subsampling, seed 42. Dua fold, bukan lima, karena pencarian
harus murah — lima fold × puluhan kandidat tidak muat di plafon waktu mana
pun. Konsekuensinya (skor fold 3 dan 5 bukan out-of-sample terhadap
seleksi) ditangani secara eksplisit di Bagian 1.6.

## 1.3 Lantai: Tiga Baseline Naif

Skor model tidak berarti apa-apa berdiri sendiri. Sebelum model apa pun
dilatih, lantainya ditetapkan lewat `evaluation.NAIVE_BASELINES`:

| Baseline            | Prediksi                       |
| ------------------- | ------------------------------ |
| `naive_zero`        | 0                              |
| `naive_lag_1`       | `lag_1 × lead_time_days`       |
| `naive_roll_mean_7` | `roll_mean_7 × lead_time_days` |

Setiap baseline menskalakan estimasi permintaan yang melihat ke belakang
dengan jumlah hari yang harus ditanggung pengiriman. Keduanya sudah ada
sebagai fitur, jadi tidak berongkos — dan persis itulah yang akan dilakukan
manajer outlet dengan tangan.

`naive_roll_mean_7` adalah lantai yang sebenarnya. Coverage-nya hanya
**0,61** terhadap target service level 0,90 — argumen paling gamblang untuk
melatih pada pinball alih-alih pada rata-rata: peramal titik-tengah, secara
konstruksi, kehabisan stok jauh lebih sering daripada yang dijanjikan.

## 1.4 Ringkasan Banding Konstruksi Ketiga Model

|                                             | Random Forest                                | XGBoost                                 | LSTM                                                     |
| ------------------------------------------- | -------------------------------------------- | --------------------------------------- | -------------------------------------------------------- |
| Mekanisme kuantil                           | Kuantil empiris dibaca dari daun             | `reg:quantileerror`, `multi_strategy`   | Pinball loss langsung, kepala 19-titik                   |
| Objektif = kriteria?                        | Tidak (dibaca, bukan dioptimalkan)           | Ya                                      | Ya                                                       |
| Kapasitas ditentukan                        | `n_estimators` dipatok (200 cari, 400 final) | Early stopping per fold (201–390 ronde) | Early stopping per fold (5–11 epoch)                     |
| Protokol fit                                | Satu fit                                     | Dua fit (early stop + refit)            | Satu putaran per fold, early stopping internal (dua fit) |
| Ruang pencarian                             | 1.152                                        | 2.592                                   | 144                                                      |
| Kandidat                                    | 18                                           | 30                                      | 30 (setara XGBoost)                                      |
| Penyaring keterjangkauan                    | Batas memori daun 3 GB                       | Tidak perlu                             | Anggaran waktu                                           |
| Bergantung seed acak                        | Tidak                                        | Tidak diverifikasi                      | **Ya** — rentang K1 0,2517 antar 3 seed                  |
| Benchmark satu putaran                      | 9,7 menit                                    | 265,2 menit                             | 16,0 menit                                               |
| Device pencarian                            | CPU Mac                                      | GPU Windows                             | GPU Windows                                              |
| Wall time walk-forward (CPU Mac, sebanding) | ~45 menit                                    | ~3 jam 1 menit                          | ~8 jam 28 menit                                          |
| Ukuran artefak                              | 826 MB                                       | 292 MB                                  | 3,7 MB                                                   |
| Butuh panel saat prediksi                   | Tidak                                        | Tidak                                   | Ya (jendela sekuens)                                     |
| Dependensi tambahan                         | `quantile-forest`                            | `xgboost` + `libomp`                    | `torch`                                                  |

Rincian tiap baris ada di dokumen per model masing-masing
(`docs/detail-tahap-modeling-{rf,xgb,lstm}.md` Bagian 1.4–1.6). Tabel ini
adalah pandangan sisi-berdampingan yang khusus berguna untuk membandingkan.
Baris ongkos di tabel ini (benchmark, wall time, ukuran artefak) bersifat
deskriptif saja — bukan kriteria pembanding; rujuk Bagian 1.5 dan 2.6.

## 1.5 Metrik dan Justifikasinya

Perbandingan ini mencari model yang paling baik **membaca fluktuasi tren
permintaan** tiap pasangan item–outlet — naik-turunnya kebutuhan dari hari
ke hari dan dari satu titik sebaran ke titik lain — karena kemampuan itulah
yang menentukan tercapai-tidaknya tujuan utama proyek: memenuhi kebutuhan
stok setiap outlet tanpa kehabisan maupun menumpuk berlebihan. Ongkos
komputasi bukan bagian dari pertanyaan itu, sehingga **tidak pernah menjadi
metrik pembanding** antarmodel di dokumen ini, sekecil apa pun perannya —
ia dicatat sebagai informasi pendamping saja (Bagian 2.6), bukan sesuatu
yang bisa memenangkan atau mengalahkan sebuah model.

### Kriteria tunggal: rata-rata `pinball` lintas `QUANTILE_SET`

```python
delta = actual - predicted
def pinball(alpha):
    return np.where(delta >= 0, alpha * delta, (alpha - 1.0) * delta).mean()

K1 = np.mean([pinball(tau) for tau in QUANTILE_SET])
```

Pada setiap titik τ, kekurangan dikali **τ** dan kelebihan dikali **1−τ**.
Rata-ratanya **tak berbobot**: setiap titik kuantil menyumbang sama besar,
termasuk 0,9. `actual` adalah `target_lead_time_cumulative` **mentah**,
bukan varian capped yang dipakai untuk melatih — K1 mengukur jarak
terhadap permintaan yang benar-benar dihadapi outlet.

`QUANTILE_SET` saat ini di **Tahap A**: 19 titik merata `[0,05, 0,10, …,
0,95]`. Kerapatan ini dipertahankan (keputusan pemilik proyek, 2026-08-24)
meski memperkecilnya akan menghemat komputasi signifikan di XGBoost —
kerapatan grid adalah alasan rata-rata pinball boleh dibaca sebagai
hampiran CRPS (Bröcker, 2012), sehingga memangkasnya melemahkan justifikasi
kriteria itu sendiri. Ia berpindah otomatis ke **Tahap B** — grid yang
diturunkan dari sebaran critical ratio aktual — begitu cakupan data biaya
mencapai ambang ≥80% volume (B-10). Definisi lengkap dan mekanisme
peralihan ada di `docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`.

**Kenapa berpindah dari satu titik ke banyak titik — tiga alasan:**

**(a) Peringkat model terbukti berpindah tergantung titik evaluasinya.**
Model yang unggul di pinball@0,9 tidak dijamin unggul di kuantil lain
(Serafin et al., 2024). Pada data ini ketiga model sempat seri dalam 0,88%
pada satu titik kuantil (run kuantil-tunggal lama) — situasi di mana
memperluas titik evaluasi paling mungkin memisahkan.

**(b) Kalibrasi baik di satu titik bisa kebetulan.** Menilai model hanya
di 0,9 berisiko menangkap kalibrasi yang benar di titik itu saja tanpa
menjamin kalibrasi di seluruh distribusi (Gneiting & Resin, 2022).

**(c) Karena standar bidangnya memang begitu.** Kompetisi M5 Uncertainty —
patokan forecasting demand ritel skala besar, 42.840 deret Walmart —
menilai model pada sembilan titik kuantil sekaligus lewat pinball loss
terskala yang dirata-ratakan (Makridakis et al., 2021).

**Ongkos komputasi sengaja dikesampingkan sepenuhnya** dari tangga
keputusan — bukan hanya dari kriteria utama, tapi dari setiap anak tangga,
termasuk sebagai penentu tambahan saat kriteria lain seri. Tujuan
penelitian ini menemukan model yang paling baik membaca fluktuasi tren
permintaan, bukan model termurah dijalankan; model yang murah tapi kurang
akurat tetap gagal memenuhi tujuan utama proyek. Ongkos tetap dilaporkan
sebagai catatan pendamping (Bagian 2.6), tapi tidak satu angka pun di sana
pernah dipakai memutuskan pemenang.

**Kuantil 0,9 tidak dicabut; ia berpindah peran.** Ia tetap komitmen bisnis
yang mengatur apa yang dikirim ke outlet (B-9, seragam untuk setiap SKU),
dan berhenti menjadi satu-satunya titik tempat model dibandingkan. Skor
pinball dan coverage di τ=0,9 tetap dilaporkan terpisah dan diberi tempat
khusus di seluruh dokumen ini.

### Metrik pendamping dan perannya

| Metrik                                | Peran                                       | Alasan keberadaannya                                                                                                                                 |
| ------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pinball` per τ                       | Kriteria pemilihan (dirata-ratakan jadi K1) | Dilaporkan per titik kuantil berdampingan, bukan hanya sebagai satu rata-rata                                                                        |
| `coverage` per τ                      | Cek kalibrasi                               | Proporsi baris dengan `actual ≤ prediksi` harus mendekati τ. Jauh di atas = overstock sistematis; jauh di bawah = janji service level tidak ditepati |
| `fill_rate`                           | Kriteria sukses pemilik data                | "Outlet tidak kehabisan", dalam unit. Kekurangan dijumlahkan sebelum dibagi                                                                          |
| `shortfall_units` / `overstock_units` | Penerjemah ke bahasa bisnis                 | "pinball 2,390" tidak bisa didiskusikan di rapat; "kekurangan turun 73%" bisa                                                                        |
| `mae`                                 | Konteks saja                                | Dilaporkan, tidak memutuskan                                                                                                                         |
| `n`                                   | Bukti baris identik                         | Menjamin perbandingan sah                                                                                                                            |

> **Catatan satuan.** `shortfall_units`/`overstock_units` menjumlahkan unit
> lintas SKU yang satuannya campur (Kg, Porsi, Botol, PCS, …) — sah untuk
> membandingkan model pada baris yang sama, tidak punya makna fisik sebagai
> satu besaran tunggal.

### Kenapa MSE/RMSE tidak dipakai

`utils/modelling/evaluation.py` tidak memiliki fungsi RMSE maupun MSE —
bukan kelalaian. Empat alasan:

1. **Simetris — bertentangan langsung dengan service level.** Menghukum
   kelebihan stok sekeras kehabisan stok, padahal keduanya tidak setara.
2. **Menobatkan baseline naif sebagai juara**, terukur: `naive_roll_mean_7`
   mencetak MAE 9,65 (juara) lawan XGBoost 14,31, padahal shortfall
   `naive_roll_mean_7` 3,7× lipat lebih besar.
3. **RMSE akan memilih model yang paling jago meramal hal yang bukan
   tugasnya.** RMSE mengkuadratkan galat, didominasi lonjakan — tapi
   `docs/batasan-penelitian.md` B-3 menyatakan lonjakan justru di luar
   lingkup model. Baris lonjakan menyumbang 11,5% absolute error padahal
   hanya 2,41% baris.
4. **44,35% target bernilai nol.** Metrik yang didominasi tengah distribusi
   bisa menobatkan model yang unggul hanya di tempat menebak nol itu mudah.

### Tidak pernah satu angka global

Setiap hasil dilaporkan dalam empat potongan: gabungan, per
`demand_segment`, per `is_delivery_day`, dan per titik kuantil. K1 adalah
rata-rata 19 angka, dan sebuah rata-rata bisa menyembunyikan model yang
kalibrasinya rusak di ekor bawah tapi tertolong di tengah grid.

## 1.6 Posisi Hasil Saat Ini

Ketiga model sudah dijalankan penuh di bawah kriteria K1: RF 25 Agustus
2026, XGBoost 27 Agustus 2026 (rearrangement 30 Agustus), LSTM 28 Agustus
2026 (verifikasi seed 43, 30 Agustus). Angka berikut dari 345.547 baris
validasi yang identik.

### Gabungan

| potongan                | model                               |         K1 | mae@0,9 | coverage@0,9 | `crossing_rate` |
| ----------------------- | ----------------------------------- | ---------: | ------: | -----------: | --------------: |
| semua fold              | random_forest                       |     2,8621 |  15,081 |        0,928 |          0,0000 |
|                         | lstm (seed 42)                      |     2,8828 |  13,929 |        0,906 |          0,4345 |
|                         | xgboost (rearranged)                |     2,8910 |       — |            — |          0,0000 |
| **fold 1/2/4 (bersih)** | **random_forest**                   | **2,8508** |  15,055 |        0,930 |          0,0000 |
|                         | lstm (seed 42)                      |     2,8818 |  14,065 |        0,915 |               — |
|                         | xgboost (rearranged)                |     2,9115 |       — |            — |          0,0000 |
|                         | xgboost (mentah, sebelum rearrange) |     2,9433 |  13,467 |        0,905 |          0,9767 |
|                         | **lstm (seed 43, verifikasi)**      | **3,0732** |       — |            — |               — |
|                         | `naive_roll_mean_7`                 |     4,8603 |   9,721 |        0,696 |          0,0000 |

`crossing_rate` (gabungan 5 fold, karena crossing adalah properti
per-baris-prediksi, bukan per-fold) adalah temuan paling mencolok: RF 0%
(struktural), LSTM 43,4% (sebagian besar derau numerik, Bagian 2.4), dan
XGBoost 97,7% sebelum rearrangement (sebagian besar defek sungguhan,
Bagian 2.5). Dibaca sendirian, K1 mentah menunjukkan XGBoost/LSTM hampir
menyaingi RF; dibaca bersama `crossing_rate`, angka XGBoost mentah tidak
bisa dipercaya sampai rearrangement dijalankan.

### Per fold

K1 per fold (RF vs XGBoost mentah vs LSTM seed 42):

| fold      | random_forest | xgboost |   lstm |
| --------- | ------------: | ------: | -----: |
| 1 (Jul)   |    **2,6819** |  2,8640 | 2,7058 |
| 2 (Agu)   |    **2,8441** |  2,9178 | 2,8743 |
| 3 (Sep)\* |    **2,7541** |  2,7366 | 2,7206 |
| 4 (Okt)   |    **3,0396** |  3,0568 | 3,0791 |
| 5 (Nov)\* |    **3,0305** |  3,0510 | 3,0780 |

\* Fold 3 dan 5 ikut memilih hyperparameter ketiga model, jadi kedua kolom
di kedua fold ini bukan murni out-of-sample — RF juga dipilih dari fold
3&5, jadi ini bukan keuntungan yang timpang, tapi tetap dicatat.

**Random Forest menang K1 di kelima fold** — berbeda dari run
kuantil-tunggal lama, di mana kemenangan berpindah-pindah antar model per
fold. Di bawah K1, RF konsisten unggul, bukan berganti-ganti — argumen
"varians antar-fold lebih besar dari selisih antar-model" yang dulu
berlaku tidak lagi didukung data secara penuh di K1: rentang K1 antar fold
untuk satu model (RF: 13%) memang tetap lebih lebar daripada selisih K1
antar model pada fold yang sama (fold 1: RF vs LSTM hanya 0,9%), tapi arah
kemenangannya sekarang stabil.

### Per segmen permintaan

K1 per segmen, gabungan 5 fold:

| segmen       |       n | random_forest | xgboost |    lstm |
| ------------ | ------: | ------------: | ------: | ------: |
| smooth       |  45.485 |   **10,9478** | 11,0466 | 11,0092 |
| erratic      |  54.511 |    **5,4788** |  5,4969 |  5,4961 |
| lumpy        | 123.545 |    **1,1430** |  1,1823 |  1,1664 |
| intermittent | 122.006 |    **0,4194** |  0,4978 |  0,4236 |

Random Forest menang K1 di keempat segmen. Marginnya tidak seragam: di
`intermittent`, LSTM (0,4236) sangat dekat ke RF (+1,0%) sementara XGBoost
jauh di belakang (+18,7%); di segmen lain ketiganya lebih rapat (0,3–3,4%).
LSTM tidak pernah menjadi yang terbaik mutlak, tapi juga tidak pernah
menjadi yang terburuk.

### Sisi bisnis

Shortfall dan overstock di τ=0,9, gabungan 5 fold:

|                     | kekurangan (shortfall) | kelebihan (overstock) |
| ------------------- | ---------------------: | --------------------: |
| `random_forest`     |            **418.250** |             4.793.038 |
| `lstm`              |                461.320 |             4.351.815 |
| `xgboost`           |                500.579 |         **4.132.651** |
| `naive_roll_mean_7` |              1.528.393 |             1.804.789 |

Ketiganya memangkas kekurangan stok 67–73% dari baseline dengan ongkos
kelebihan stok 2,3–2,7×. Polanya monoton dengan coverage@0,9: RF paling
tinggi coverage-nya → shortfall terendah, overstock tertinggi; XGBoost
paling rendah coverage-nya → sebaliknya; LSTM di tengah pada kedua sisi.
Mana yang disukai bisnis tergantung ongkos relatif kedua sisi — keputusan
bisnis, bukan keputusan model.

### Derau seed (LSTM)

LSTM satu-satunya model dengan inisialisasi bobot acak. Diverifikasi dua
tingkat (rincian di Bagian 2.4 dan `docs/detail-tahap-modeling-lstm.md`
Bagian 1.9): tiga seed pada fold 3&5 menunjukkan rentang K1 0,2517 — delapan
kali lebih lebar daripada jarak K1 LSTM ke RF (0,0310). Konfirmasi langsung
lewat walk-forward 5-fold penuh dengan seed 43 memberi K1 = 3,0732 di fold
1/2/4 — kalah **7,80%** dari RF, jauh melebihi ambang keputusan 2%. **K1
seed 42 (2,8818) terbukti bukan representasi LSTM yang stabil.**

## 1.7 Strategi Pemilihan: Tangga Kriteria Bertingkat

Prosedur berikut ditetapkan **sebelum** test set dibuka. Tangga dituruni
berurutan; berhenti di anak tangga pertama yang benar-benar memisahkan.

### Gerbang G0 — Kelayakan

> Model harus mengalahkan baseline terbaik (`naive_roll_mean_7`) pada
> pinball@0,9 **di kelima fold**, bukan hanya gabungan.

Model yang menang secara gabungan tetapi kalah di satu bulan bukan model
yang bisa diterapkan; SCM mengirim setiap minggu, bukan setiap tahun.

### K1 — Kriteria utama: rata-rata pinball 19 titik, potongan fold bersih (1/2/4)

> Pemenang adalah rata-rata pinball terendah, **asalkan selisihnya ≥2%**
> terhadap pesaing terdekat. Di bawah ambang itu, hasilnya dinyatakan
> **seri** dan tangga diteruskan ke K2.

Rata-ratanya tak berbobot — setiap titik kuantil menyumbang sama besar,
termasuk 0,9. Pembobotan lebih berat ke 0,9 sudah dipertimbangkan dan
**belum diputuskan** (pertanyaan terbuka nomor 1,
`docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`).

**Justifikasi ambang 2%**, dihitung dari tiga sumber kebisingan yang
benar-benar terukur:

- Selisih antar-model pada fold yang sama bergerak 1,2%–6,8% tergantung
  fold.
- Satu model yang sama bergerak 12–14% antar fold berurutan.
- **Satu model, satu konfigurasi, tiga seed berbeda (LSTM saja) bergerak
  8,8%** — sumber kebisingan yang jauh lebih ketat dari dua sebelumnya,
  karena satu-satunya variabel yang berubah adalah inisialisasi bobot.

Ambang 2% dipertahankan apa adanya, tapi kepercayaan padanya berkurang:
derau antar-seed LSTM (8,8%) jauh di atas ambang, jadi selisih K1 sebesar
2% — persis di ambang — bisa jadi masih derau seed, bukan sinyal model.

### K2 — Kalibrasi terhadap service level yang dijanjikan

> Untuk setiap τ di `QUANTILE_SET`: `|coverage(τ) − τ|` pada potongan fold
> bersih, dibaca bersama stabilitasnya antar fold. `|coverage(0,9) − 0,90|`
> tetap dilaporkan terpisah dan diberi bobot khusus.

Ini bukan kriteria cadangan — ia menguji **janji yang sama** dengan
kriteria utama, dari sudut berbeda. Dua pola yang harus dibedakan eksplisit:

| Pola                                             | Bacaan                                                               |
| ------------------------------------------------ | -------------------------------------------------------------------- |
| Simpangan searah di hampir seluruh τ             | Bias sistematis pada seluruh distribusi — alasan kuat untuk tersisih |
| Simpangan besar hanya di beberapa τ, arah campur | Derau atau kelemahan lokal — dicatat, bukan alasan menyisihkan       |

**Peringatan yang harus dibaca sebelum menerapkan aturan ini**: karena
41,95% baris validasi bertarget nol, setiap model tak-negatif otomatis
mencetak simpangan searah di seluruh τ < ~0,42 (lantai `share_nol`). Dibaca
mentah, aturan di atas akan menandai **setiap** model di dataset ini,
termasuk yang kalibrasinya sempurna. Aturan harus dinyatakan ulang
terhadap lantai — membandingkan `coverage(τ)` dengan `max(τ, share_nol)` —
sebelum dipakai memutuskan apa pun (dibahas penuh di
`docs/detail-tahap-modeling-rf.md` Bagian 1.9).

**Quantile crossing** adalah kegagalan kalibrasi yang hanya terlihat
setelah multi-kuantil dijalankan: prediksi yang tidak monoton terhadap τ
pada baris yang sama. Dilaporkan sebagai laju untuk XGBoost dan LSTM;
Random Forest kebal secara struktural. Laju yang material dibaca di K2,
bukan diperbaiki diam-diam dengan mengurutkan hasil (Bagian 2.5).

### Catatan — Ongkos Operasional dan Reprodusibilitas (bukan anak tangga)

> Wall time training, ukuran artefak, ketergantungan pada seed acak, dan
> bobot dependensi — dicatat sebagai informasi pendamping, **tidak pernah**
> dipakai memutuskan pemenang, termasuk saat K1/K2 menyatakan seri.

Ini bukan sekadar diletakkan di anak tangga terakhir — ia sengaja
dikeluarkan dari tangga sama sekali. Alasannya tidak berubah dari Bagian
1.5: tujuan perbandingan ini adalah menemukan model yang paling baik
membaca fluktuasi tren permintaan, sehingga kebutuhan stok tiap outlet
benar-benar terpenuhi, bukan menemukan model yang paling murah dijalankan.
Model yang murah tapi kalah akurat tetap kalah, seri atau tidak. Angka
lengkapnya tetap dilaporkan (Bagian 2.6) supaya tim yang akan
mengoperasikan model tahu konsekuensi rekayasanya, tapi tidak satu baris
pun di sana mengubah urutan kemenangan.

### K3 — Risiko integrasi produksi

> Apa yang harus tersedia saat model dipanggil untuk meramal, dan apa yang
> gagal secara diam-diam bila salah dipasang.

## 1.8 Penerapan Tangga pada Angka yang Ada

### G0 — ketiganya lolos

Semua mengalahkan `naive_roll_mean_7` pada pinball@0,9 di kelima fold.
Margin serupa: RF 40,5–48,7%, XGBoost 42–45%, LSTM 43–48%. Fold 4 (Oktober)
konsisten paling berat bagi ketiganya — properti bulannya, bukan properti
modelnya.

### K1 — "seri" ditarik: seed 42 terbukti bukan titik representatif

| model                          | K1 (fold 1/2/4) | selisih ke RF |
| ------------------------------ | --------------: | ------------: |
| **random_forest**              |      **2,8508** |             — |
| lstm (seed 42, resmi)          |          2,8818 |        +1,09% |
| xgboost (rearranged)           |          2,9115 |        +2,13% |
| xgboost (mentah)               |          2,9433 |        +3,24% |
| **lstm (seed 43, verifikasi)** |      **3,0732** |    **+7,80%** |
| lstm (rata-rata 2 seed)        |          2,9775 |        +4,44% |

Dibaca lewat seed 42 saja, RF vs LSTM di bawah ambang 2% → "seri". Tapi
seed 43, diukur dengan protokol identik, memberi RF kemenangan 7,80% — jauh
di atas ambang. Rata-rata dua seed pun tetap di atas ambang (4,44%). Tidak
ada bacaan yang konsisten menghasilkan "seri" kecuali seed 42 secara khusus
dipilih sebagai yang mewakili.

XGBoost, setelah rearrangement kuantil, membaik dari +3,24% menjadi
**+2,13%** terhadap RF — tetap di atas ambang 2%, jadi tetap kalah, dengan
margin lebih tipis sekarang crossing tidak lagi mengaburkan angkanya.

**Bacaan yang menggantikan "seri"**: dengan n=2 pada seed LSTM, ini bukan
interval kepercayaan yang ketat — tapi kedua titik data yang ada menunjuk
arah yang sama: seed 42 adalah pengecualian yang menguntungkan, bukan
representasi tengah. Berat bukti condong ke **RF unggul atas LSTM**, bukan
seri — tapi menyatakannya sebagai kesimpulan final butuh ≥1 seed lagi
untuk membedakan "RF memang unggul" dari "kebetulan dua seed pertama
sama-sama tidak mewakili tengah sebaran". Tangga **tidak diteruskan ke K2
atas dasar "seri"** — status K1 sekarang **tidak tuntas**, bukan seri.

### K2 — XGBoost tidak lagi tertahan (rearranged)

| model                | coverage@0,9 | gap dari 0,90 | kelebihan di atas lantai | `crossing_rate` |
| -------------------- | -----------: | ------------: | -----------------------: | --------------: |
| xgboost (rearranged) |        0,908 |        +0,008 |              **+0,0077** |      **0,0000** |
| lstm                 |        0,906 |        +0,006 |                  +0,0062 |          0,4345 |
| random_forest        |        0,928 |        +0,028 |                  +0,0281 |      **0,0000** |

`crossing_rate` diuji terpisah untuk tiap model (Bagian 2.4, 2.5) — jawaban
**berbeda arah untuk tiap model**: XGBoost sebagian besar defek sungguhan
(rate bertahan ~20–25% di toleransi 0,5–1,0 unit), LSTM hampir seluruhnya
derau numerik (rate ambruk ke 1,1% di toleransi 0,1). Sesudah rearrangement,
ketiga model kini punya `crossing_rate` yang bisa dipercaya, jadi ketiga
baris di tabel ini sekarang sebanding secara adil.

RF vs LSTM di K2 (τ=0,9): LSTM (+0,0062) lebih dekat ke target daripada RF
(+0,0281) — tapi baris LSTM ini dari seed 42, seed yang sama yang sudah
terbukti bukan representasi K1 yang stabil. Selisihnya di sini tipis dan
satu-titik, jadi tidak cukup untuk menyisihkan RF secara mandiri, dan
dengan K1 sekarang condong ke RF, K2 di titik ini tidak mengubah arah
kesimpulan K1.

### Catatan — Ongkos operasional (bukan anak tangga, tidak memengaruhi keputusan)

Dicatat sebagai informasi pendamping saja (tabel lengkap Bagian 2.6): pada
satu-satunya tahap yang device-nya sebanding lintas ketiganya (walk-forward
+ fit final, CPU Mac), LSTM ~6,3× lebih lambat dari RF dan ~2,9× lebih
lambat dari XGBoost — dan satu-satunya model yang hasilnya terbukti
bergantung seed pada besaran yang jauh melebihi selisih K1 antar model. RF
kebetulan juga yang paling murah pada tahap yang sebanding, dan XGBoost di
tengah — tapi fakta ini tidak pernah dipakai untuk memutuskan siapa yang
menang, di sini maupun di Bagian 1.10.

### K3 — dicatat, belum jadi penentu

Rincian di Bagian 2.7.

### Status keputusan

> **Prasyarat wajib terjawab (2026-08-30) — RF adalah kandidat terdepan
> berdasarkan berat bukti, tapi pemenang belum resmi dibekukan.** Tiga
> pertanyaan yang menahan keputusan sebelumnya sekarang semuanya terjawab:
> `crossing_rate` diuji untuk kedua model (XGBoost defek sungguhan, LSTM
> derau numerik); derau seed LSTM diverifikasi langsung (seed 43 kalah
> 7,80% dari RF); rearrangement kuantil XGBoost dikerjakan (gap ke RF
> menyempit tapi tetap di atas ambang).
>
> Pada kedua titik data LSTM yang ada (seed 42 dan 43) dan pada angka
> XGBoost yang sudah di-rearrange, **RF unggul di setiap perbandingan**:
> RF < LSTM-seed42 < XGBoost-rearranged < LSTM-seed43.
>
> **Ini bukan "RF resmi menang" secara administratif** — n=2 pada seed
> LSTM tidak memberi interval kepercayaan yang ketat secara statistik, dan
> membekukan pemenang dalam sebuah commit tetap memerlukan **persetujuan
> eksplisit pemilik proyek**. Tapi tidak ada lagi prasyarat teknis yang
> menahan langkah itu.
>
> **Langkah berikutnya**: pemilik proyek meninjau angka di atas dan
> menyetujui RF sebagai pemenang dalam sebuah commit, baru test set
> Desember 2025 dibuka sekali lewat protokol Bagian 1.9.

## 1.9 Protokol Pembukaan Test Set Desember

Enam langkah, dijalankan **sekali**, setelah Bagian 1.8 dibekukan dalam
sebuah commit.

**1. Bekukan keputusan lebih dulu.** Dokumen ini di-commit dengan pemenang
tertulis, sebelum satu baris Desember disentuh model mana pun.

**2. Satu run, ketiga model sekaligus.** Ketiga model final di
`models/*.joblib` plus ketiga baseline dinilai pada baris Desember yang
identik, dalam satu eksekusi.

**3. Laporkan penyebutnya.** Desember dinilai pada **49.717 dari 55.046
baris panel** (996 baris warm-up, 4.333 baris tanpa target). Baris yang
dikeluarkan bukan irisan yang bias, tetapi angkanya wajib disebut.

**4. Angka utama = skor model terpilih.** K1, coverage per titik kuantil
(K2), dan fill_rate model terpilih adalah angka final penelitian. Skor
pinball dan coverage di τ=0,9 dilaporkan terpisah dan diberi tempat khusus.
Dua model lain dilaporkan sebagai pembanding deskriptif, dengan label
eksplisit bahwa pemenang ditetapkan sebelum Desember dibuka.

**5. Tiga potongan, bukan satu angka.** Gabungan, per `demand_segment`,
per `is_delivery_day`.

**6. Jangan menukar pemenang.** Bila model lain mencetak angka lebih
tinggi di Desember, itu temuan yang ditulis di pembahasan — bukan alasan
mengganti pilihan.

**Yang akan membatalkan rencana ini**: kegagalan kalibrasi yang besar
(indikatif: coverage di τ=0,9 turun di bawah 0,85, atau simpangan searah
yang besar di seluruh grid). Itu bukan sinyal untuk menukar model —
ketiganya menargetkan grid kuantil yang sama dan akan bergerak searah —
melainkan sinyal bahwa Desember berperilaku berbeda dari lima bulan
validasi, yang harus dibahas sebagai keterbatasan generalisasi musiman.

## 1.10 Apa yang Akan dan Tidak Akan Disimpulkan

### Yang bisa dinyatakan dengan yakin

- Ketiga model mengalahkan baseline operasional secara telak dan
  konsisten: ~46–47% lebih baik pada pinball@0,9, di kelima fold, tanpa
  satu bulan pun yang menggendong hasilnya.
- Model kuantil memangkas kekurangan stok 73–76% dibanding praktik
  rata-rata-bergerak, dengan ongkos kelebihan stok 2,5–2,8× — pertukaran
  yang memang diminta service level 0,9.
- Kalibrasi ketiganya mendarat di sasaran (0,902–0,934 terhadap 0,90).
- Dengan anggaran pencarian yang sudah disetarakan (RF 18, XGBoost dan
  LSTM 30 kandidat pada ruang yang proporsional, 3 seed pada LSTM), Random
  Forest unggul secara konsisten atas keduanya di K1 dan K2 — dua kriteria
  yang benar-benar dipakai memutuskan. Sebagai catatan tambahan yang tidak
  memengaruhi keputusan ini, RF juga yang paling murah dijalankan pada
  tahap yang sebanding (Bagian 2.6).

### Yang tidak boleh dinyatakan

- ❌ "XGBoost/LSTM adalah arsitektur yang buruk untuk peramalan permintaan
  rantai pasok secara umum." Satu dataset, satu domain, satu periode —
  penyetaraan anggaran membuat "kalah karena arsitekturnya" jadi klaim
  yang bisa dipertahankan **pada dataset ini**, bukan klaim tentang
  arsitektur secara umum.
- ❌ Klaim apa pun berbasis MAE terhadap baseline (Bagian 1.5).
- ❌ Klaim yang mengandaikan sumbu waktu pemesanan
  (`docs/batasan-penelitian.md` B-1, B-2, B-3).
- ❌ "RF resmi menang" sebelum commit persetujuan pemilik proyek — status
  saat ini adalah kandidat terdepan berdasarkan berat bukti, bukan
  keputusan final (Bagian 1.8).

### Batasan yang wajib disebut di bab batasan laporan

1. Desember belum dibuka saat dokumen ini ditulis.
2. Sumbu waktunya waktu pengambilan, bukan waktu pemesanan.
3. Target mencampur pesanan dan non-pesanan, sementara model secara bisnis
   hanya bertanggung jawab atas yang kedua.
4. Anggaran pencarian tidak identik antar ketiga model (18 vs 30 vs 30
   kandidat, ruang berbeda-beda) — disetarakan sejauh validitas atribusi
   membutuhkannya, tidak identik penuh.
5. LSTM: hanya dua seed yang diverifikasi lewat walk-forward 5-fold penuh
   (42 dan 43); seed 44 baru diuji di fold 3&5.
6. Fold 3 dan 5 ikut memilih hyperparameter, jadi skor di sana bukan
   out-of-sample terhadap seleksi. Potongan fold 1/2/4 adalah angka bersih.
7. Desember dinilai pada 49.717 dari 55.046 baris, dan Desember adalah
   bulan atipikal (Natal/Tahun Baru).

---

# Bagian 2 — Teknis (Mendetail)

## 2.1 Arsitektur Mesin Evaluasi Bersama

Tiga modul memiliki bagian yang tidak dimiliki model mana pun secara
khusus — mengumpulkannya di sini (bukan di modul Random Forest) berarti
tidak perlu memperbaiki bug checkpoint yang sama dua kali, lalu tiga kali
saat LSTM datang.

**`utils/modelling/walk_forward.py`** — kelayakan baris, batas fold,
penilaian:

| Fungsi                                                        | Peran                                                                                                     |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `eligible_rows(df, ...)`                                      | Tiga potongan berurutan (Bagian 1.2), dijamin identik untuk ketiga model                                  |
| `prepare_fold(df, fold_id, ...)`                              | Frame train/valid satu fold, purge di batas fold via `modeling_prep.fold_train_mask()`                    |
| `run_fold(df, fold_id, fit_predict, ...)`                     | Fit pada training fold, skor pada validasi fold, termasuk ketiga baseline naif dinilai di baris yang sama |
| `run_walk_forward(df, fit_predict, folds, ...)`               | Seluruh fold, satu frame hasil panjang                                                                    |
| `pooled_metric(results, model_name, metric, folds, quantile)` | Satu angka lintas fold, dibobot jumlah baris `n`                                                          |
| `pooled_k1(results, model_name, folds)`                       | K1 — pembungkus `pooled_metric` di `metric="pinball"`                                                     |
| `coverage_by_quantile(results, model_name, folds)`            | Tabel K2: coverage teramati vs target di tiap τ, dengan `gap` bertanda                                    |

**`utils/modelling/model_common.py`** — bagian yang dibagikan tiga model:

| Komponen                          | Fungsi                                                                                                                                                   |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sample_search_space()`           | Penarikan acak kandidat, dengan penyaring keterjangkauan yang **disuntikkan** (batas memori daun RF tidak punya padanan di XGBoost/LSTM)                 |
| `run_search()`                    | Menilai tiap kandidat di fold pencarian, menulis checkpoint tiap kandidat selesai (atomik via `os.replace`), melanjutkan dari sana bila dijalankan ulang |
| `select_best()`                   | Kandidat dengan `pinball` gabungan terendah — satu baris kode yang sama dipakai ketiga model                                                             |
| `expand_one_hot()`                | Ekspansi kategorikal bersama (RF dan XGBoost mode `one_hot`)                                                                                             |
| `split_early_stopping()`          | Ekor 30 hari terakhir jendela training, dengan purge yang sama seperti di batas fold — dipakai XGBoost dan LSTM                                          |
| `merge_shards()`                  | Menggabungkan beberapa CSV hasil pencarian dari mesin berbeda, dengan empat pemeriksaan integritas (Bagian 2.6 GPU/CPU)                                  |
| `save_bundle()` / `load_bundle()` | Format bundel yang mendeskripsikan dirinya sendiri                                                                                                       |

**`utils/modelling/evaluation.py`** — metrik dan baseline (detail Bagian
2.2).

Kriteria seleksi tidak pernah berpindah-pindah antar model:

```python
best_id = int(scored.loc[scored["pinball"].idxmin(), "candidate_id"])
```

## 2.2 Definisi Metrik — Implementasi

```python
def pinball_loss(y_true, y_pred, alpha=DEFAULT_ALPHA):
    delta = y_true - y_pred
    return np.where(delta >= 0, alpha * delta, (alpha - 1.0) * delta).mean()

def k1_score(per_quantile, metric="pinball"):
    return per_quantile[metric].dropna().mean()

def crossing_rate(predictions, quantiles=QUANTILE_SET_A):
    values = as_quantile_frame(predictions, quantiles).to_numpy()
    inverted = np.diff(values, axis=1) < 0
    return float(inverted.any(axis=1).mean())
```

`QUANTILE_SET_A = tuple(round(0.05 * step, 2) for step in range(1, 20))` —
19 titik berjarak sama 0,05 sampai 0,95. `resolve_quantile_set()` memilih
Tahap A atau Tahap B berdasarkan status data biaya (`item_cost_margin.csv`),
bukan lewat pengubahan kode — mekanisme peralihan otomatis yang dijelaskan
di `docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`.

`crossing_rate` adalah properti **satu baris prediksi utuh**, bukan satu
titik kuantil — nilainya diulang di setiap baris kuantil pada cell yang
sama saat disimpan long-form (`walk_forward.run_fold()`), dan konsumen
yang mengagregasinya **wajib** memfilter ke satu τ terlebih dulu; meratakan
kolom itu tanpa filter kebetulan mengembalikan angka yang benar (karena
konstan per cell) tapi menjumlahkannya akan mengembalikan sembilan belas
kali lipat nilai sebenarnya.

`resolve_quantile_set()`, `naive_predictions()`, dan `score()` (pembungkus
tujuh metrik: `n`, `mae`, `pinball`, `coverage`, `fill_rate`,
`shortfall_units`, `overstock_units`) semuanya didefinisikan di
`evaluation.py` dan dipakai identik oleh ketiga model — tidak ada modul
model yang mendefinisikan metriknya sendiri.

## 2.3 Tabel Lengkap Hasil Lintas Model

Tabel gabungan, per fold, per segmen, dan sisi bisnis di Bagian 1.6 di atas
adalah ringkasan yang dipakai keputusan. Tabel per-baris lengkap (per
kandidat pencarian, per titik kuantil K2 penuh 19 baris, per
`is_delivery_day`) ada di masing-masing dokumen hasil:

- `docs/hasil-modeling-rf.md` — Bagian 4 (18 kandidat), Bagian 5.1–5.4
  (per fold, K2 19 titik, per segmen, per hari kirim)
- `docs/hasil-modeling-xgb.md` — Bagian 4 (30 kandidat), Bagian 5.1–5.5
  (termasuk rearrangement)
- `docs/hasil-modeling-lstm.md` — Bagian 4 (30 kandidat), Bagian 5.1–5.4,
  Bagian 5.1b (derau seed)

Dokumen ini sengaja tidak menyalin ulang tabel-tabel itu secara lengkap —
angkanya berasal dari CSV yang tidak masuk git (`dataset/model_ready/*_search_results.csv`,
`*_walk_forward_results.csv`) dan bisa berubah setiap run diulang. Menyalin
ulang berarti dua sumber kebenaran yang bisa saling menyimpang saat salah
satu diperbarui — terutama begitu test set Desember dibuka (Bagian 1.9) dan
dokumen hasil baru ditulis.

## 2.4 Derau Seed LSTM — Detail Pengujian

Dijalankan dua tingkat, keduanya lewat `model_lstm.run_seed_repeats()`
(fold 3&5) dan skrip khusus `lstm_seed_walkforward.py` (5-fold penuh):

**Tingkat 1 — tiga seed di fold pencarian (3&5), ~50 menit GPU Windows.**
Konfigurasi pemenang diulang pada `random_state` 42/43/44:

| seed | K1 (fold 3&5) |
| ---: | ------------: |
|   44 |        2,8399 |
|   42 |        2,8617 |
|   43 |        3,0915 |

`seed_spread()` melaporkan `min`/`mean`/`max`/`range` — rentang 0,2517
adalah delapan kali jarak K1 LSTM-RF di potongan bersih.

**Tingkat 2 — konfirmasi seed 43 di walk-forward 5-fold penuh**, ~9,8 jam
CPU Mac, pada fold yang sama persis dipakai klaim K1 resmi (fold 1/2/4):

|                         | seed 42 (resmi) |    seed 43 |
| ----------------------- | --------------: | ---------: |
| K1 (fold 1/2/4, bersih) |      **2,8818** | **3,0732** |

Selisih 0,1914 (6,6%) — hanya `random_state` yang berbeda. Baris pemenang
di search CSV dengan seed 42 (via `run_seed_repeats`) dicek identik dengan
baris kandidat pemenang di `lstm_search_results.csv` — perbandingan lolos
(selisih 0,000000), memastikan tidak ada nondeterminisme tersembunyi di
luar seed yang mencemari perbandingan ini.

Seed 44 belum diverifikasi lewat walk-forward 5-fold penuh — hanya di
fold 3&5, sehingga Bagian 1.8 tidak memasukkannya ke tabel keputusan K1.

## 2.5 Rearrangement Kuantil XGBoost — Detail Pengujian

**Uji toleransi jarak (2026-08-29)**, dijalankan identik untuk XGBoost dan
LSTM: crossing dihitung ulang dari bundle tersimpan (tanpa retrain) dengan
toleransi jarak minimum `prediksi(τ_tinggi) < prediksi(τ_rendah) − gap`:

|      toleransi gap | XGBoost |    LSTM |
| -----------------: | ------: | ------: |
| 0 (definisi resmi) | 0,916\* | 0,459\* |
|               0,01 |   0,794 |   0,088 |
|                0,1 |   0,479 |   0,011 |
|                0,5 |   0,248 |   0,007 |
|                1,0 |   0,202 |   0,005 |
|                5,0 |   0,106 |   0,001 |

\*sedikit di bawah `crossing_rate` walk-forward resmi karena ini prediksi
dari model final gabungan, bukan lima model per-fold walk-forward.

XGBoost bertahan di ~20–25% pada toleransi 0,5–1,0 unit (median inversi
0,043 unit, ekor sampai 139 unit) — inti keras defek sungguhan. LSTM ambruk
ke ~1% pada toleransi 0,1 (median inversi 0,0027 unit) — hampir seluruhnya
derau angka mengambang. Kesimpulan berlawanan arah untuk kedua model:
hipotesis 2 (`multi_strategy` XGBoost tanpa jaminan monotonicity struktural)
terbukti untuk XGBoost; hipotesis 1 (efek ikatan/near-tie) terbukti untuk
LSTM.

**Rearrangement (2026-08-30)**, skrip `xgb_rearrangement_walkforward.py`
(repo root), ~2,89 jam CPU Mac: kelima model fold di-fit ulang dengan
hyperparameter pemenang yang sama, 19 prediksi kuantil tiap baris diurutkan
naik sebelum dinilai — pada `QUANTILE_SET_A` yang sudah berurutan naik,
sort per baris menempatkan statistik urutan ke-k pada τ ke-k, menjamin
`crossing_rate = 0` secara struktural (Chernozhukov, Fernández-Val &
Galichon, 2010).

Cek reproduksibilitas — `best_iteration` per fold dan K1 sebelum sort
cocok persis dengan run resmi 27 Agustus — mengonfirmasi run rearrangement
benar-benar mereproduksi walk-forward asli, sehingga angka sesudahnya sah
dibandingkan langsung:

|                        | sebelum |    sesudah |
| ---------------------- | ------: | ---------: |
| K1 (fold 1/2/4 bersih) |  2,9433 | **2,9115** |
| `crossing_rate`        |  0,9767 | **0,0000** |

Artefak: `dataset/model_ready/xgb_walk_forward_results_rearranged.csv`
(model_name `xgboost_rearranged`, tidak masuk git).

## 2.6 Catatan: Ongkos Komputasi Lintas Model

**Bukan kriteria pembanding** (Bagian 1.5, 1.7) — dicatat semata sebagai
informasi pendamping bagi tim yang akan mengoperasikan model terpilih.
Tidak satu angka pun di tabel ini pernah dipakai memutuskan pemenang.

|                                               |           random_forest |              xgboost |                                    lstm |
| --------------------------------------------- | ----------------------: | -------------------: | --------------------------------------: |
| Walk-forward + fit final (CPU Mac, sebanding) |               ~93 menit |           ~204 menit |                          **~586 menit** |
| Pencarian (device beda, tidak sebanding)      |          3,85 jam (CPU) |      ~14,7 jam (GPU) | ~4,2 jam (GPU) + ~50 menit 3-seed (GPU) |
| Ukuran artefak                                |                  826 MB |               292 MB |                              **3,7 MB** |
| Bergantung seed                               | tidak (bagging meredam) |   tidak diverifikasi | **ya** — rentang K1 0,2517 antar 3 seed |
| Dependensi                                    |       `quantile-forest` | `xgboost` + `libomp` |                                 `torch` |

**Pemisahan device bukan kealpaan.** RF dijalankan penuh di CPU Mac sejak
awal (keputusan pemilik proyek 2026-08-25). XGBoost dan LSTM memindahkan
**hanya tahap pencarian** ke GPU Windows (RTX 4060 Ti 8 GB) sejak
26 Agustus 2026 — walk-forward final dan fit final ketiganya tetap di satu
mesin CPU Mac yang sama, supaya baris "walk-forward + fit final" di atas
tetap sebanding lintas ketiga model apa pun device pencariannya. Paritas
GPU↔CPU untuk peringkat kandidat divalidasi (selisih K1 0,124%, jauh di
bawah ambang 2%) sebelum keputusan pemindahan ini diambil — lihat
`docs/superpowers/specs/2026-08-24-distributed-gpu-training-design.md`
Bagian 3bis dan `docs/runbook-pencarian-gpu-windows.md`.

Pada tahap yang sebanding, LSTM ~6,3× lebih lambat dari RF dan ~2,9× lebih
lambat dari XGBoost — dan satu-satunya model yang hasilnya terbukti
bergantung seed pada besaran yang jauh melebihi selisih K1 antar model.

## 2.7 Risiko Integrasi Produksi (K3)

| Model             | Yang wajib tersedia saat inferensi                                                                                                                                              | Kegagalan diam-diam bila salah dipasang                                                                                                                                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Random Forest** | Bundle `models/random_forest_q90.joblib` (826 MB) mendeskripsikan diri sendiri (urutan kolom, flag `one_hot`/`log_target`, grid kuantil)                                        | Forest yang dimuat dengan urutan kolom berbeda tidak gagal — ia meramal dengan percaya diri dari fitur yang salah                                                                                                                                |
| **XGBoost**       | Bundle `models/xgboost_q90.joblib` (292 MB) — urutan kolom, encoding, level kategori                                                                                            | Booster mode `native` yang dimuat terhadap kategori yang diurutkan berbeda tidak gagal — sama seperti RF                                                                                                                                         |
| **LSTM**          | Bundle `models/lstm_q90.joblib` (3,7 MB) — `state_dict`, scaler, ukuran embedding — **dan panel berjendela** saat inferensi (`predict_bundle()` mewajibkan `panel`, bukan opsi) | Tidak bisa meramal satu baris sendirian; pipeline panel harus hidup di sisi inferensi, bukan hanya training. Jaringan yang diberi fitur berskala mentah (scaler hilang) tidak gagal — ia meramal dengan percaya diri dari input yang salah skala |

Random Forest dan XGBoost memprediksi dari satu baris; LSTM satu-satunya
yang butuh 28 hari riwayat di belakang setiap baris yang diramalkan — beban
operasional tambahan yang tidak muncul di skor K1 atau K2 mana pun, dan
justru inilah yang ditangkap K3 di bagian ini.

## 2.8 Status Keputusan dan Rencana Kerja Tersisa

Status keputusan lengkap ada di Bagian 1.8. Rencana kerja setelahnya:

| #   | Pekerjaan                                                                                                    | Status                                                              |
| --- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| 1   | Membekukan usulan Bagian 1.8 dalam sebuah commit (persetujuan eksplisit pemilik proyek)                      | ⬜ menunggu peninjauan pemilik proyek                               |
| 2   | Menjalankan protokol Bagian 1.9 — buka Desember sekali                                                       | ⬜ setelah #1                                                       |
| 3   | Menulis `docs/hasil-test-desember.md`                                                                        | ⬜ setelah #2                                                       |
| 3a  | Alokasi kuantil tersegmentasi pada model pemenang                                                            | ⬜ setelah #3, `2026-08-22-segmented-quantile-allocation-design.md` |
| 4   | Dekomposisi harian (`target_h1`…`target_h4`) untuk ketiga model — menjawab "kapan permintaan terkonsentrasi" | ⬜ direncanakan                                                     |
| 5   | SHAP untuk pemenang saja — menjawab "kenapa model meyakini ini"                                              | ⬜ direncanakan                                                     |

Butir 4 dan 5 bukan tambahan opsional — keduanya sudah tertulis sebagai
rencana penjelasan (_explainability_) di
`docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md`,
dengan pembagian peran tegas: dekomposisi menjawab **kapan**, SHAP menjawab
**kenapa**. SHAP hanya dijalankan untuk pemenang, karena menjalankannya
untuk ketiganya berarti membayar ongkos penjelasan untuk model yang tidak
akan dipakai.

## 2.9 Rujukan

| Topik                                               | Berkas                                                                                                                                          |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Prapemrosesan sampai `model_input.parquet`          | `docs/detail-tahap-preprocessing.md`                                                                                                            |
| Pipeline ujung ke ujung (ringkas)                   | `docs/pipeline-overview.md`                                                                                                                     |
| Batasan yang tidak bisa dikode                      | `docs/batasan-penelitian.md`                                                                                                                    |
| Konstruksi & hasil Random Forest                    | `docs/detail-tahap-modeling-rf.md`, `docs/hasil-modeling-rf.md`                                                                                 |
| Konstruksi & hasil XGBoost                          | `docs/detail-tahap-modeling-xgb.md`, `docs/hasil-modeling-xgb.md`                                                                               |
| Konstruksi & hasil LSTM                             | `docs/detail-tahap-modeling-lstm.md`, `docs/hasil-modeling-lstm.md`                                                                             |
| Desain prapemrosesan pemodelan                      | `docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md`                                                                            |
| Desain tiap model                                   | `docs/superpowers/specs/2026-08-{18-random-forest,19-xgboost,19-lstm}-modeling-design.md`                                                       |
| Mesin evaluasi bersama                              | `utils/modelling/walk_forward.py`, `model_common.py`, `evaluation.py`                                                                           |
| Metodologi evaluasi multi-kuantil                   | `docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`                                                                         |
| Checklist migrasi multi-kuantil                     | `docs/superpowers/specs/2026-08-22-model-comparison-refactor-migration.md`                                                                      |
| Alokasi kuantil tersegmentasi                       | `docs/superpowers/specs/2026-08-22-segmented-quantile-allocation-design.md`                                                                     |
| Runbook GPU Windows                                 | `docs/runbook-pencarian-gpu-windows.md`                                                                                                         |
| CRPS dan grid kuantil rapat                         | Bröcker, J. (2012). "Evaluating raw ensembles with the continuous ranked probability score." _QJRMS_ 138:1611–1617.                             |
| Peringkat model bergantung titik evaluasi           | Serafin et al. (2024). Perbandingan model kuantil lintas titik evaluasi.                                                                        |
| Kalibrasi seluruh distribusi                        | Gneiting, T. & Resin, J. (2022). "Regression Diagnostics meets Forecast Evaluation."                                                            |
| Standar evaluasi multi-kuantil di forecasting ritel | Makridakis, S. et al. (2021). "M5 accuracy competition." _IJF_ 38(4).                                                                           |
| Rearrangement kuantil                               | Chernozhukov, V., Fernández-Val, I., & Galichon, A. (2010). "Quantile and Probability Curves Without Crossing." _Econometrica_ 78(3):1093–1125. |
| Quantile regression forest                          | Meinshausen, N. (2006). "Quantile Regression Forests." _JMLR_ 7:983–999.                                                                        |
