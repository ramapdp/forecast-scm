# Menjalankan Fase 3 multi-kuantil terdistribusi di GPU gratis (Kaggle + Colab) — design

## Status

Spec eksekusi. Ia **tidak** mengubah metodologi apa pun — set kuantil, target,
purging, split, anggaran kandidat, dan kriteria K1–K4 seluruhnya dipakai apa
adanya dari `2026-08-22-multi-quantile-evaluation-design.md` dan
`docs/metodologi-pemodelan-dan-pemilihan-model.md`. Yang diatur di sini hanya
**di mesin mana** tiap potongan Fase 3 dijalankan, bagaimana potongan-potongan
itu disatukan kembali, dan apa yang harus diukur supaya pemecahan itu sah.

Prasyarat: butir 0c `docs/todolist-proyek.md` (menjalankan ulang ketiga
notebook) masih 🔒 menunggu izin pemilik proyek. Spec ini menyiapkan
mekanismenya; ia bukan izin untuk mulai.

## Purpose

Fase 3 diperkirakan **~157–172 jam ≈ 6,5–7,2 hari komputasi nonstop** di CPU
Mac (§"Perkiraan ongkos Fase 3",
`2026-08-22-model-comparison-refactor-migration.md`). Spec ini memangkasnya
menjadi **~3–4 hari** (atau ~2–3 hari dengan Colab Pro) dengan menjalankan
pencarian ketiga model serentak di GPU gratis Kaggle dan Colab, lalu
mengembalikan walk-forward dan fit final ke Mac — **tanpa** menukar satu pun
properti metodologis untuk kecepatan itu.

Aturan yang mengikat seluruh dokumen: setiap kali kecepatan bertabrakan dengan
keterbandingan angka, yang mengalah adalah kecepatan. Itu posisi yang sudah
diambil proyek ini ketika menolak hemat 38 jam dari pemerkecilan grid (butir 0,
"Pertanyaan terbuka", spec multi-kuantil).

## Latar belakang

### Apa yang sudah terpasang di kode

- `model_xgboost.DEFAULT_DEVICE` dan parameter `device` yang mengalir sampai
  `build_estimator()`, `make_fit_predict()`, `fit_final()`, dan `run_search()`.
  Bundle mencatat `device` sebagai **provenance**, sehingga pemenang yang
  dipilih di satu device tidak pernah diam-diam di-refit di device lain.
- `model_lstm.resolve_device()` yang menolak `cuda`/`mps` yang tidak ada di
  mesin ini, dipanggil di `bind_panel()` **sebelum** window index dibangun —
  device yang tidak tersedia berongkos nol, bukan satu sort 1,5 juta baris.
- `model_common.run_search()` dengan checkpoint per kandidat, `resume=True`,
  dan `_assert_checkpoint_matches()` yang menolak checkpoint dari ruang
  pencarian atau seed berbeda.
- `sample_search_space(seed=42)` yang deterministik: daftar kandidat yang
  sama lahir di mesin mana pun, sehingga `candidate_id` adalah alamat yang
  stabil lintas platform.

Empat hal ini yang membuat spec ini mungkin ditulis; tanpa `device` dan
checkpoint, pemecahan pekerjaan hanyalah harapan.

### Angka ongkos yang jadi dasar alokasi

| Model | Pengganda 19 kuantil | Pencarian | Total Fase 3 | Bisa di-GPU? |
|---|---:|---:|---:|---|
| Random Forest | ×1,05 | 3,9 jam | ~4,8 jam | **Tidak.** `quantile-forest` murni CPU. |
| XGBoost | **×15,2** | **64,6 jam** | ~70 jam | Ya (`tree_method="hist"`, `device="cuda"`) |
| LSTM | ×1,00 | **71,5 jam** (+7,2 jam 3 seed) | ~83–98 jam | Ya (PyTorch) |

## Keputusan desain

### 1. Aturan dua lapis: apa yang boleh dipecah, apa yang tidak

**Pencarian hyperparameter boleh dipecah antar mesin.** Keluarannya hanya
*peringkat kandidat di dalam satu model*, dan probe paritas (Bagian 3d) yang
membuktikan peringkat itu tidak digerakkan oleh hardware.

**Walk-forward 5 fold dan fit final ketiga model wajib di satu mesin yang
sama.** Alasannya bukan kerapian, melainkan K3. §17
`metodologi-pemodelan-dan-pemilihan-model.md` mendefinisikan K3 sebagai *"wall
time training, ukuran artefak, ketergantungan pada seed acak, dan bobot
dependensi"* — dan pada run sebelumnya **K3-lah yang menentukan pemenang**,
karena K1 ketiga model seri dalam 0,88% (§18). Kalau wall time XGBoost lahir
di P100 Kaggle sementara wall time LSTM lahir di T4 Colab, dan K1 seri lagi,
maka tie-breaker itu membandingkan **hardware, bukan model**. Lubang itu tidak
bisa ditambal sesudah run tanpa mengulang seluruhnya.

Ongkos aturan ini kecil: walk-forward + fit final ketiga model berjumlah
~9,8–25,1 jam, lawan ~136 jam pencarian. Ia murah **asal direncanakan sejak
awal**.

**Mesin itu adalah Mac lokal** (keputusan 2026-08-24, pemilik proyek), bukan
sesi cloud, karena tiga hal yang berdiri sendiri-sendiri:

1. **`run_walk_forward()` tidak punya checkpoint.** Ia satu list comprehension
   atas lima fold (`walk_forward.py:216-222`) — tidak seperti pencarian, yang
   menulis checkpoint setiap kandidat selesai. Sesi Kaggle yang terpotong di
   jam ke-12 di tengah walk-forward kehilangan seluruhnya, tanpa resume. Di
   ujung pesimistis, LSTM (WF 15,5 jam + final 3,6 jam = **19,1 jam**) memang
   tidak muat dalam satu sesi 12 jam.
2. **Kuota GPU adalah sumber daya paling langka di rencana ini**, dan
   walk-forward tidak menuntut GPU untuk alasan metodologis apa pun.
   Menjalankannya lokal mengembalikan 8–20 jam ke anggaran pencarian Kaggle —
   satu-satunya tahap yang benar-benar dibatasi kuota.
3. **Kontinuitas dengan angka yang sudah tercatat.** Seluruh wall time di
   ketiga `hasil-modeling-*.md` (run 2026-08-18/19/20) diukur di Mac ini, jadi
   angka K3 yang baru sebanding dengan yang lama alih-alih memulai basis
   pengukuran ketiga.

### 2. Alokasi per mesin

| Mesin | Beban | Estimasi wall-clock | Kuota terpakai |
|---|---|---|---|
| **Mac lokal** (tahap 1) | Pencarian RF 18 kandidat; lalu jadi CPU-fallback untuk kandidat XGBoost yang OOM di GPU (Bagian 7.1) | ~3,9 jam | — |
| **Kaggle T4×2**, headless *Save & Run All (commit)* | XGBoost: 30 kandidat → 2 proses × 15, `cuda:0` dan `cuda:1` | 64,6 j ÷ ~6× ÷ 2 GPU ≈ **5–6 jam**, muat dalam satu commit 12 jam | ~6 jam |
| **Colab T4** (ditunggui) | LSTM: 30 kandidat + pengulangan 3 seed | 71,5 j ÷ ~5× ≈ **14 jam** → 3–4 sesi | — |
| **Mac lokal** (tahap 2, sesudah semua pencarian selesai) | Walk-forward 5 fold + fit final **ketiga model**, di satu mesin (Bagian 1) | RF 0,9 j + XGB 5,1 j + LSTM 3,8–19,1 j = **~9,8–25,1 jam**, 1–2 run semalam | — |

Pengganda `~6×` dan `~5×` adalah **estimasi, bukan ukuran**. Probe (3c)
menggantikannya dengan angka nyata, dan seluruh tabel ini dihitung ulang dari
situ sebelum satu jam pun dikomitkan.

Total kuota Kaggle: **~6 jam dari 30 per minggu** — longgar, karena
walk-forward dan RF dua-duanya lokal. Kelonggaran itu bukan sisa yang
menganggur: ia bantalan untuk kandidat yang harus diulang (Bagian 7.1), untuk
probe yang gagal, dan untuk pengganda GPU yang ternyata lebih kecil dari
estimasi.

RF sengaja **tidak** dikirim ke cloud: GPU tidak mempercepatnya sama sekali,
jadi memindahkannya hanya membakar kuota yang langka untuk pekerjaan yang
tidak diuntungkan. Efek sampingnya bagus — Mac tetap bebas sebagai jalur CPU
untuk kandidat yang gagal di GPU, dan tetap menjadi mesin yang menjalankan
tahap 2.

### 3. Tahap 0 — lima probe sebelum satu jam pun dikomitkan (~1–2 jam)

| # | Probe | Kalau gagal |
|---|---|---|
| a | **Guard checkpoint kuantil-tunggal.** Langkah 0 yang sudah tertulis di todolist butir 0c: jalankan sel pencarian XGBoost dengan ketiga CSV lama **masih di tempatnya**; harus berhenti dalam hitungan detik dengan `ValueError` yang menyebut "berasal dari run kuantil tunggal". | **Hentikan Fase 3.** Guard-nya tidak bekerja dan setiap angka sesudahnya tidak dapat dipercaya. |
| b | **Smoke test CUDA XGBoost.** `reg:quantileerror` dengan 19 `quantile_alpha` di `device="cuda"`, xgboost 2.1.4, satu fold kecil. | Kalau library menolak multi-kuantil di GPU, seluruh rencana XGBoost berubah. Harus ketahuan di menit ke-10, bukan jam ke-8. |
| c | **Pengganda kecepatan terukur.** 1 kandidat × 1 fold di tiap mesin, dibandingkan angka lokal. | Mengganti tebakan "GPU ~6×" dengan angka, dan menentukan pembagian shard yang proporsional. |
| d | **Probe paritas device.** `candidate_id 0` dijalankan penuh (2 fold, `SEARCH_FOLDS = (3, 5)`) di **setiap** mesin yang dipakai — **CPU Mac lokal termasuk** (lihat di bawah); selisih K1-nya masuk `docs/hasil-modeling-*.md` sebagai angka. | Selisih ≥ ambang seleksi **2%** → pemecahan dalam satu model dibatalkan; jatuh ke satu-model-satu-platform. |
| e | **Versi paket dipin dan dicatat per mesin**: `xgboost==2.1.4`, `torch==2.8.0`, `numpy==2.0.2`, `pandas==2.3.3`, `quantile-forest==1.4.2`, `scikit-learn==1.6.1`. | Probe (d) tidak sah kalau versinya berbeda: yang terukur jadi selisih library, bukan selisih hardware. |

**CPU lokal wajib ikut probe (d), dan itu bukan formalitas.** Karena
walk-forward dan fit final berjalan di Mac (Bagian 1), rencana ini menciptakan
satu penyerahan device: kandidat **diperingkat di GPU**, pemenangnya
**di-refit di CPU**. Komentar konstanta `DEFAULT_DEVICE` di `model_xgboost.py`
ditulis persis untuk situasi ini — *"a winner chosen on one device is never
silently refitted on another"*. Ia tidak melarang penyerahan itu; ia menuntut
penyerahan itu terlihat dan terukur. Yang harus dibuktikan: peringkat kandidat
yang lahir di GPU masih berlaku di CPU tempat pemenangnya dilatih. Kalau
selisih K1 GPU↔CPU jauh di bawah 2% **dan** jauh di bawah jarak antar
kandidat, peringkat itu stabil dan penyerahannya sah — tercatat sebagai angka,
bukan sebagai asumsi.

Probe (c) juga yang menjawab apakah plafon **30 GPU-jam/minggu** Kaggle cukup.
Kalau pengganda XGBoost ternyata hanya ~2×, 64,6 jam turun ke ~32 jam dan
**tidak muat** — itu harus diketahui sebelum mulai, bukan di tengah jalan.
Yang boleh dipotong pada situasi itu adalah **jumlah kandidat**, yang oleh
proyek ini sudah dinyatakan dapat dilaporkan sebagai batas anggaran — tidak
seperti kerapatan grid 19 titik, yang tidak boleh disentuh.

### 4. Pengungkit T4×2: dua GPU dengan harga satu jam kuota

Kuota GPU Kaggle dihitung dari **wall time sesi**, bukan jumlah akselerator:
sesi T4×2 memakan kuota yang sama dengan sesi P100. Karena
`model_xgboost.DEFAULT_DEVICE` menerima string apa adanya (`"cuda:1"` dan
kerabatnya — komentar konstanta itu menyebutnya eksplisit), dua shard dapat
jalan sebagai dua proses dalam satu sesi, masing-masing dipin ke satu GPU.

Hasilnya **2× throughput per jam kuota**. Ini pengungkit tunggal terbesar di
seluruh rencana, dan gratis. Kalau sesi menawarkan P100 atau T4×2, ambil
**T4×2** meski P100 lebih cepat per-GPU.

Akuntansi kuota ini **dikonfirmasi sekali di sesi Kaggle pertama** dan
hasilnya dicatat; kebijakan platform berubah dari waktu ke waktu dan asumsi
ini tidak boleh diwarisi tanpa bukti.

### 5. Seam kode: `only=` dan penggabungan shard yang terverifikasi

Satu perubahan kecil di `model_common.run_search()`:

```
run_search(..., only: Optional[Iterable[int]] = None)
```

Menjalankan hanya `candidate_id` yang disebut, **tanpa menggeser
penomorannya**. Ini bukan kenyamanan: `random_search` menomori kandidat lewat
`enumerate(candidates)`, jadi memotong daftar kandidat di sisi pemanggil akan
menomori ulang dan membuat shard tidak dapat disatukan. `only` menjaga
`candidate_id` tetap absolut terhadap `sample_search_space(seed=42)`.

Penggabungan shard **diverifikasi, bukan dipercaya**:

1. concat seluruh CSV shard;
2. tolak `candidate_id` ganda;
3. pastikan gabungannya menutup `0…N-1` tanpa lubang;
4. jalankan `_assert_checkpoint_matches()` terhadap daftar kandidat penuh — ia
   sudah mencocokkan nilai parameter tiap baris dengan `candidate_id` yang
   diklaimnya, jadi shard yang tertukar atau lahir dari ruang berbeda tertolak
   di sini;
5. baru `select_best()` dipanggil di atas CSV gabungan.

Kolom `device` dan hash commit git ditulis di tiap baris shard, sejajar dengan
`device` yang sudah dicatat bundle. Sebuah angka harus selalu bisa ditelusuri
ke mesin dan versi kode yang melahirkannya.

### 6. Persistensi, resume, dan pengiriman kode/data

| Platform | Checkpoint ke | Resume |
|---|---|---|
| Kaggle | `/kaggle/working/<model>_shard.csv`, lalu *Save Version* → output jadi dataset yang di-attach sesi berikutnya | `resume=True` (default) membacanya |
| Colab | Google Drive ter-mount, `checkpoint_path` langsung ke Drive | Otomatis saat sesi diputus |

Checkpoint ditulis **setiap kandidat selesai**, jadi sesi yang putus paling
banyak kehilangan satu kandidat.

Pengiriman: `model_input.parquet` (42 MB) sebagai Kaggle Dataset privat / file
Drive; `utils/` lewat `git clone` dari repo privat, supaya **commit hash-nya
tercatat** di tiap shard. `modeling_prep` tidak perlu dijalankan di cloud —
`model_input.parquet` sudah jadi, dan mengimpor ketiga modul model tidak
membaca berkas apa pun (terverifikasi 2026-08-24).

## 7. Risiko yang harus diukur, bukan diasumsikan

### 7.1. Kandidat `encoding="one_hot"` bisa OOM di VRAM padahal lolos di CPU

1,3 juta baris × ~230 kolom setelah ekspansi one-hot, ditambah 19 pohon per
ronde boosting. Bahayanya halus: `run_search` menangkap `XGBoostError` dan
mencatatnya sebagai **baris NaN** — perilaku yang benar untuk run panjang,
tetapi artinya pencarian **diam-diam menyusut** dari 30 kandidat.

Aturannya: **setiap baris NaN dari shard GPU wajib diulang di CPU sebelum
`select_best()`**. Kalau tidak, yang dilaporkan bukan pencarian 30 kandidat
yang dijanjikan desain, dan ketimpangan anggaran yang sengaja dihindari di
spec multi-kuantil kembali lewat pintu belakang.

### 7.2. LSTM bisa kelaparan data di GPU

`_to_tensors()` membangun window per batch **di CPU**, dan Colab gratis hanya
2 vCPU — jauh di bawah Mac. Mungkin saja GPU menganggur menunggu host. Probe
(3c) menunjukkannya; kalau terbukti, LSTM lebih baik pindah ke Kaggle (4 vCPU)
atau tetap di MPS/CPU lokal.

### 7.3. Plafon 30 GPU-jam/minggu Kaggle

Lihat Bagian 3. Sesudah walk-forward dipindahkan ke Mac (Bagian 1), alokasi
Bagian 2 hanya memakai **~6 jam dari 30** — longgar. Risiko ini karenanya
turun dari pengikat jadwal menjadi bantalan: ia baru menggigit kalau probe
(3c) menunjukkan pengganda GPU jauh di bawah estimasi ~6×, mis. ~2×, yang
membuat pencarian XGBoost sendiri menjadi ~32 jam dan melewati plafon.

## Colab Pro: layak, tetapi dibeli **sesudah** Tahap 0

Jalur kritis rencana ini adalah **LSTM di Colab gratis** — bukan karena
komputasinya paling berat, melainkan karena sesinya paling sering putus dan
paling butuh ditunggui. Pro menyerang tepat di situ, di dua sumbu:

| Yang Pro beri | Dampak di rencana ini |
|---|---|
| **Background execution** | Sel LSTM jalan saat tab ditutup. Ini nilai terbesarnya — menghapus "ditunggui" dari jalur kritis, bukan kecepatan. |
| **GPU L4** | ~2–3× untuk beban ini: 14 jam → ~5–7 jam. |
| **Runtime high-RAM, ~8 vCPU** | Menyerang risiko 6.2 langsung. Kalau LSTM ternyata *CPU-bound*, inilah yang menolong, bukan GPU-nya. |
| ~100 compute unit/bulan | ≈ 30–50 jam GPU; praktis melipatgandakan anggaran cloud di luar kuota Kaggle. |

Dengan Pro, estimasi ujung-ke-ujung turun dari ~3–4 hari ke **~2–3 hari** dan
babysitting hampir hilang. Perlu jujur soal batasnya: sesudah walk-forward
pindah ke Mac, tahap 2 (~9,8–25,1 jam) berdiri di jalur kritis dan **tidak
tersentuh Pro sama sekali** — Pro hanya memendekkan tahap pencarian.

**Keputusan: tunggu Tahap 0.** Probe (3c) berongkos ~1 jam dan menjawab
pertanyaan yang menentukan nilai Pro: LSTM terbatas GPU atau terbatas CPU
host? Kalau terbatas GPU, L4 memberi 2–3×. Kalau terbatas CPU host, T4 gratis
dan L4 berbayar hampir sama cepat, dan yang dibeli sebenarnya hanya background
execution — masih berguna, tapi dengan harga yang harus dinilai ulang. Angka
itu gratis dan datang lebih dulu.

Pro+ (~$50) menambah sesi background sampai 24 jam. Untuk beban ini — yang
potongan terpanjangnya ~14 jam dan sudah ter-checkpoint per kandidat — itu
berlebihan.

Konsekuensi kalau Pro jadi dibeli: **L4 adalah hardware ketiga**, jadi probe
paritas (3d) dijalankan sekali lagi di sana. Ongkosnya satu kandidat.

## Testing

Ditulis TDD, merah lebih dulu, sejalan dengan 752 tes yang sudah ada.

- `only=` tidak menggeser `candidate_id`: `run_search(only={5, 7})`
  menghasilkan baris dengan id 5 dan 7, dengan nilai parameter yang identik
  dengan run utuh.
- Shard + merge = run utuh: dua shard disjoint yang digabung menghasilkan
  frame yang identik kolom-per-kolom dengan satu run atas kandidat yang sama.
- Merge menolak `candidate_id` ganda.
- Merge menolak gabungan yang bolong (tidak menutup `0…N-1`).
- Merge menolak shard dari ruang pencarian atau seed berbeda — lewat
  `_assert_checkpoint_matches()` yang sudah ada, bukan lewat pengecekan baru.
- `only=` berinteraksi benar dengan `resume=True`: kandidat yang sudah ada di
  checkpoint tetap dilewati, kandidat di luar `only` tidak pernah dijalankan
  dan tidak pernah ditulis sebagai baris kosong.
- Baris shard membawa kolom `device` dan hash commit.

## Out of scope

- **Izin menjalankan Fase 3 itu sendiri** (butir 0c todolist) — spec ini
  menyiapkan mekanismenya, bukan memberi izinnya.
- **Perubahan metodologi apa pun**: set kuantil, target latih/nilai, purging,
  split, anggaran kandidat (RF 18, XGB 30, LSTM 30 + 3 seed), dan definisi
  K1–K4 dipakai apa adanya.
- **Peralihan Tahap A → Tahap B** (`resolve_quantile_set`) — tetap Tahap A;
  cakupan `item_cost_margin.csv` masih jauh di bawah ambang 80% (68 dari 70
  SKU `cost_confidence: rendah`, diverifikasi 2026-08-24).
- **Fallback cold-start** untuk 842 pasangan yang gagal `MIN_HISTORY_DAYS`
  (A3 todolist) — tidak tersentuh oleh spec ini.

## Pertanyaan terbuka

1. ~~**Mesin mana yang menjalankan walk-forward + fit final ketiga model.**~~
   **DITUTUP 2026-08-24 (pemilik proyek): Mac lokal.** Alasannya di Bagian 1.
   Yang perlu dicatat: pilihan ini **membubarkan** dilema yang tertulis di
   sini semula, bukan memilih salah satu sisinya. Dua pembacaan K3 yang
   bersaing — (a) satu mesin untuk semua, dengan RF ikut menanggung 4 vCPU
   Kaggle, versus (b) K3 sebagai profil ongkos per kelas hardware yang wajar
   untuk tiap model — hanya bertabrakan selama "satu mesin" diartikan kotak
   cloud. Di Mac, ketiga model diukur di satu mesin **dan** RF mendapat CPU
   yang wajar untuknya. Yang dibayar sebagai gantinya adalah penyerahan device
   GPU→CPU, dan itu ditutup oleh perluasan probe (3d).
2. **Apakah pemecahan XGBoost tetap dijalankan kalau probe (3c) menunjukkan
   satu sesi Kaggle sudah cukup.** Kalau 30 kandidat muat dalam satu commit 12
   jam di T4×2, pemecahan lintas-platform tidak lagi diperlukan untuk XGBoost
   dan probe paritas menjadi lebih sedikit. Diputuskan dari angka (3c), bukan
   sekarang.
3. **Akuntansi kuota T4×2** (Bagian 4) — dikonfirmasi di sesi pertama.

## References

- `2026-08-22-multi-quantile-evaluation-design.md` — sumber kebenaran
  metodologi multi-kuantil; spec ini tidak mengubah satu pun keputusannya.
- `2026-08-22-model-comparison-refactor-migration.md` §"Perkiraan ongkos Fase
  3" — angka 157–172 jam dan rinciannya per model per tahap.
- `docs/metodologi-pemodelan-dan-pemilihan-model.md` §17 (definisi K3), §18
  (K1 seri dalam 0,88%, K3 jadi penentu).
- `docs/todolist-proyek.md` butir 0c — Langkah 0 (guard checkpoint) dan
  anggaran kandidat yang berlaku.
- `docs/checklist-refresh-data-2026.md` — doktrin "timestamp bukan bukti
  kesegaran", yang diikuti Bagian 5 dengan mencatat hash commit per shard.
