# Runbook — menjalankan Tahap B (pencarian) di PC Windows + RTX 3060

Dokumen ini adalah prosedur, bukan desain. Alasan di balik pembagian mesinnya
ada di Bagian 1 dan §0bis
`superpowers/specs/2026-08-24-distributed-gpu-training-design.md`; di sini
hanya langkah-langkahnya, berurutan, untuk dijalankan sekali dari awal sampai
selesai.

Keputusan pemilik proyek **2026-08-26**: pencarian hyperparameter XGBoost dan
LSTM dijalankan di PC Windows ber-RTX 3060; walk-forward dan fit final tetap
di Mac. Random Forest tidak ikut sama sekali — ketiga tahapnya sudah selesai di
Mac (run 2026-08-25), dan `quantile-forest` murni CPU.

Perkiraan total di PC: **~30 jam** (XGB ~15 jam, LSTM ~14 jam, pengulangan seed
~1,5 jam), berurutan.

---

## Ringkasan: sel mana di mesin mana

| | Sel di PC (`cuda`) | Sel di Mac (`cpu`) |
|---|---|---|
| `modeling_xgb.ipynb` | `2-10,14` | `2-10,14,16-24` |
| `modeling_lstm.ipynb` | `2-21,23,26` | `2-34` |

Sel markdown yang kebetulan masuk rentang dilewati diam-diam, jadi rentangnya
boleh ditulis apa adanya. `.venv\Scripts\python run_cells.py
notebook\modeling_xgb.ipynb --list` mencetak peta sel kalau perlu dicocokkan.

Yang perlu dipahami sebelum mulai: sel pencarian di Mac (`14` dan `21`) **tidak
menghitung ulang apa pun** selama checkpoint dari PC sudah lengkap — ia hanya
membaca CSV-nya dan melewati ke-30 kandidat. Itulah yang membuat pembagian ini
mungkin tanpa mengubah satu baris kode model.

---

## Langkah 1 — Siapkan berkas di Mac

Dari root repo di Mac:

```bash
cd ~/Project/Personal/forecast-scm
mkdir -p /tmp/kirim-ke-pc
cp dataset/model_ready/model_input.parquet /tmp/kirim-ke-pc/
git log -1 --format=%h            # catat hash ini, PC harus di commit yang sama
```

`model_input.parquet` berukuran ~42 MB. Ia **tidak** ada di git (`dataset/`
gitignored), jadi harus disalin manual — flashdisk, jaringan lokal, atau cloud
drive, terserah.

Yang **tidak** disalin: `xgb_search_results.csv`. Kedua kandidat yang sudah
dinilai di CPU Mac sengaja dibuang supaya seluruh peringkat lahir di satu
device; berkasnya tinggal di Mac sebagai `xgb_search_results.cpu-partial.bak.csv`.

---

## Langkah 2 — Siapkan repo di PC

Di PowerShell:

```powershell
cd C:\Users\<nama>\Projects
git clone <url-repo-atau-path-jaringan> forecast-scm
cd forecast-scm
git checkout <hash-dari-langkah-1>
```

Kalau repo-nya tidak punya remote, salin saja seluruh foldernya dari Mac —
tapi **buang `.venv/`** dari salinan itu (venv macOS tidak berjalan di
Windows).

Lalu bangun tata letak `dataset/` yang dibutuhkan:

```powershell
mkdir dataset\csv
mkdir dataset\model_ready
copy <lokasi>\model_input.parquet dataset\model_ready\
```

`dataset\csv` sengaja dibuat **kosong**. Sel pertama tiap notebook memanggil
`find_base_dir()`, yang menemukan root repo dengan mencari folder berisi
`dataset/csv/`. Tanpa folder itu, notebook berhenti di sel pertama dengan
`RuntimeError: Root repo tidak ditemukan`.

Jangan set `FORECAST_CHECKPOINT_DIR` maupun `FORECAST_MODEL_INPUT`. Dengan tata
letak di atas, nilai bawaannya sudah menunjuk tempat yang benar — dan
`SEED_REPEATS_FILE` memang tidak lewat `run_config`, jadi menyetel
`FORECAST_CHECKPOINT_DIR` justru memisahkan keluaran LSTM ke dua folder yang
berbeda.

---

## Langkah 3 — Pasang Python dan dependensi

Pakai **Python 3.12**. Ini plafon, bukan preferensi: `numpy==2.0.2` hanya
menerbitkan wheel Windows sampai cp312 — tidak ada cp313, apalagi cp314.
`scikit-learn==1.6.1` dan `torch==2.8.0` berhenti di cp313.

| Pin di `requirements.txt` | Wheel Windows yang tersedia |
|---|---|
| `numpy==2.0.2` | cp39, cp310, cp311, **cp312** |
| `scikit-learn==1.6.1` | cp39 … cp313 |
| `torch==2.8.0` | cp39 … cp313 |
| `pandas==2.3.3` | cp39 … cp314 |
| `xgboost==2.1.4` | `py3-none-win_amd64` (versi apa pun) |

Di Python 3.13/3.14, pip tidak menemukan wheel numpy, jatuh ke sdist, dan
build-nya gagal — numpy 2.0.2 terbit jauh sebelum perubahan C API 3.14. Godaan
berikutnya, melepas pin versinya, justru yang harus ditolak: probe 2026-08-25
mencatat versi paket Kaggle identik dengan pin lokal, **dan itulah** yang
membuat selisih K1 0,124% terbaca sebagai selisih hardware alih-alih selisih
library. Versi Python boleh berbeda dari Mac (Kaggle 3.12.13 lawan Mac 3.9.6
menghasilkan daftar kandidat yang identik — `random.Random(42)` stabil lintas
versi); versi library tidak boleh.

Python 3.14 yang sudah terpasang tidak perlu dicopot. Pasang 3.12
berdampingan dan tunjuk ia lewat py launcher:

```powershell
py -3.12 --version          # harus mencetak Python 3.12.x
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
```

Setelah venv dibuat, `.venv\Scripts\python` sudah menunjuk 3.12 — sisa
dokumen ini memakai itu dan tidak pernah memanggil `python` polos, supaya
interpreter sistem tidak pernah ikut campur.

Lalu **ganti torch**, dan ini bukan opsional:

```powershell
.venv\Scripts\pip uninstall -y torch
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Baris `torch==2.8.0` di `requirements.txt` memasang build **CPU-only** di
Windows, karena wheel CUDA PyTorch tidak muat di batas ukuran PyPI dan hanya
diterbitkan di indeks PyTorch sendiri. Ia gagal tanpa satu pun pesan error:
`torch.cuda.is_available()` menjadi `False`, `resolve_device("cuda")` melempar
ValueError, sel benchmark LSTM mencetak `cuda dilewati`, dan pencarian berjalan
berhari-hari di CPU PC. Ganti `cu126` dengan versi CUDA yang cocok menurut
[pytorch.org](https://pytorch.org/get-started/locally/) untuk driver PC-nya.

Dua hal lain:

- **Visual C++ Redistributable** harus terpasang — XGBoost butuh DLL-nya.
  Sudah ada kalau Visual Studio terpasang; kalau tidak, unduh dari Microsoft.
- **CUDA Toolkit tidak perlu dipasang.** Wheel xgboost dan torch sudah membawa
  runtime CUDA-nya sendiri. Yang perlu hanyalah driver NVIDIA yang cukup baru.

Kalau `quantile-forest` gagal dipasang di Windows, abaikan saja — ia hanya
dipakai Random Forest, yang tidak dijalankan di PC. Pasang sisanya dan
lanjutkan.

---

## Langkah 4 — Verifikasi, sebelum satu jam pun dikomitkan

Empat pemeriksaan. Jangan lanjut sebelum keempatnya benar; masing-masing
menutup satu cara run 30 jam bisa terbuang.

**4a. GPU terlihat oleh torch**

```powershell
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Harus mencetak `True NVIDIA GeForce RTX 3060`. Kalau `False`, Langkah 3 belum
tuntas — torch-nya masih CPU-only.

**4b. XGBoost multi-kuantil benar-benar berjalan di CUDA**

```powershell
.venv\Scripts\python -c "import numpy as np, xgboost as xgb; X=np.random.rand(5000,10); y=np.random.rand(5000); m=xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=np.linspace(0.05,0.95,19), tree_method='hist', device='cuda', n_estimators=10).fit(X,y); print(m.predict(X).shape)"
```

Harus mencetak `(5000, 19)`. Bentuk itu — 19 kolom, bukan 1 — yang membuktikan
seluruh grid kuantil dilatih di device cuda dan bukan diam-diam jatuh ke satu
kuantil. Ini probe (b) Bagian 3 spec, dijalankan ulang di mesin ini.

**4c. Tes repo lolos, dan tes CUDA-nya benar-benar berjalan**

```powershell
.venv\Scripts\python -m unittest discover -p "test_*.py"
```

Harus `OK`. Lalu:

```powershell
.venv\Scripts\python -m unittest test.test_model_lstm.TestResolveDevice -v
```

`test_cuda_is_returned_when_the_machine_has_one` harus **berjalan**, bukan
`skipped`. Di Mac ia selalu di-skip; kalau di PC ia masih di-skip, torch-nya
masih CPU-only.

**4d. Notebook menemukan datanya**

```powershell
.venv\Scripts\python run_cells.py notebook\modeling_xgb.ipynb 2-10
```

Sepuluh detik, bukan sepuluh menit. Harus mencetak:

```
1,502,522 rows x 82 columns
device: cuda | seluruh kandidat | input: ...\dataset\model_ready\model_input.parquet | checkpoint: ...\dataset\model_ready
```

Perhatikan `device: cuda`. Kalau tertulis `cpu`, variabel environment di
Langkah 5 belum diset — set dulu, ulangi 4d.

---

## Langkah 5 — Matikan sleep, set device

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

(Kalau ditolak, jalankan PowerShell sebagai Administrator.) Sleep di jam ke-9
menghentikan proses; checkpoint memang menyelamatkan kandidat yang sudah
selesai, tapi kandidat yang sedang berjalan hilang.

```powershell
$env:FORECAST_DEVICE = "cuda"
```

Variabel ini **hanya hidup selama jendela PowerShell itu**. Kalau nanti membuka
jendela baru, set lagi — dan verifikasi lewat 4d sebelum menjalankan apa pun
yang panjang. `FORECAST_DEVICE` wajib untuk XGBoost karena
`DEFAULT_DEVICE = "cpu"`; untuk LSTM ia sebenarnya opsional (sel 19 memilih
device tercepat dari benchmarknya sendiri), tetapi membiarkannya terset tidak
merugikan.

---

## Langkah 6 — Jalankan pencarian XGBoost (~15 jam)

Dari root repo, di jendela PowerShell yang sama:

```powershell
.venv\Scripts\python run_cells.py notebook\modeling_xgb.ipynb 2-10,14 *> dataset\model_ready\xgb_run_B_search.log
```

Biarkan jendela itu terbuka. Untuk memantau, buka jendela PowerShell **kedua**:

```powershell
cd C:\Users\<nama>\Projects\forecast-scm
Get-Content dataset\model_ready\xgb_run_B_search.log -Wait -Tail 20
```

### Periksa kandidat pertama — ini gerbangnya

Kandidat 0 adalah kandidat yang sama yang diukur **19.958 detik (5,54 jam)** di
CPU Mac. Di RTX 3060 ia seharusnya selesai dalam **~40 menit**.

- ~40 menit → pengganda ×8 berlaku, ~15 jam untuk 30 kandidat. Lanjutkan.
- 2 jam → pengganda hanya ×3; totalnya ~40 jam. Masih jauh lebih baik daripada
  120 jam CPU, tapi rencanakan ulang jadwalnya.
- ~5 jam → **ia berjalan di CPU.** Hentikan, kembali ke Langkah 4d.

Baris log per kandidat berbentuk:

```
[1/30] pinball=2.9602 epoch=1244,1388 2510s
```

Untuk melihat tabelnya kapan saja:

```powershell
.venv\Scripts\python -c "import pandas as pd; d=pd.read_csv('dataset/model_ready/xgb_search_results.csv'); print(len(d),'kandidat selesai'); print(d[['candidate_id','pinball','elapsed_seconds','device','error']])"
```

Kolom `device` harus berisi `cuda` di **setiap** baris.

---

## Langkah 7 — Jalankan pencarian LSTM (~15,5 jam)

Setelah XGBoost tuntas (log berakhir `=== SELESAI`):

```powershell
.venv\Scripts\python run_cells.py notebook\modeling_lstm.ipynb 2-21,23,26 *> dataset\model_ready\lstm_run_B_search.log
```

Jangan dijalankan berbarengan dengan XGBoost. Dua proses yang berebut GPU yang
sama menghasilkan wall time yang mengukur kontensi, bukan model — dan
`elapsed_seconds` itu ikut dibaca sebagai ongkos terukur.

Rentang `2-21,23,26` mencakup tiga hal: sel 17 mem-benchmark `cpu` dan `cuda`
lalu sel 19 memilih yang tercepat (~1 jam, dan angka CPU-nya memang ikut
diukur — itu disengaja), sel 21 menjalankan 30 kandidat, sel 23 memilih
pemenang, sel 26 mengulang pemenang di seed 42/43/44.

Sel 26 ikut dijalankan di PC — bukan di Mac — karena ia memeriksa bahwa baris
seed 42 identik dengan baris pemenang di `lstm_search_results.csv`. Dua device
berselisih setingkat paritas (0,124%), sehingga menjalankannya di Mac di atas
pencarian GPU membuat pemeriksaan itu **pasti** mencetak
`PERIKSA - ada nondeterminisme` — alarm palsu yang menghapus nilai penjagaannya.

### Periksa kandidat pertama, sekali lagi

Dasar CPU LSTM adalah **3.412 detik per kandidat**. Di 3060 harapkan
~700 detik. Kalau ia mendekati 3.400 detik, sel benchmark memilih `cpu` —
periksa keluaran sel 19 di log:

```
device tercepat  : cuda (diukur)
device dipakai   : cuda
```

---

## Langkah 8 — Kalau proses mati di tengah

Jalankan ulang perintah yang sama persis. Tidak ada langkah pemulihan lain.

`run_search()` menulis checkpoint secara atomik setiap satu kandidat selesai
(tulis ke `.tmp`, lalu `os.replace`), dan `resume=True` adalah bawaannya. Yang
hilang hanya kandidat yang sedang berjalan saat proses mati. Sel-sel definisi
di depan rentang berjalan ulang dalam hitungan detik.

Yang **tidak** boleh dilakukan: menghapus `xgb_search_results.csv` atau
`lstm_search_results.csv` untuk "mulai bersih". Itu membuang setiap jam yang
sudah dibayar.

---

## Langkah 9 — Bawa pulang ke Mac

Salin tiga berkas ini ke `dataset/model_ready/` di Mac:

- `xgb_search_results.csv`
- `lstm_search_results.csv`
- `lstm_seed_repeats.csv`

Ketiga log (`*_run_B_search.log`) berguna dibawa juga sebagai catatan, tapi
tidak dibaca oleh kode apa pun.

**Jangan bawa pulang berkas `.ipynb`.** `run_cells.py` menyimpan notebook
setiap sel selesai, jadi salinan di PC berisi output PC. Menyalinnya kembali
akan menimpa output Mac — termasuk angka benchmark XGBoost 4,42 jam yang sudah
tercatat dan tidak akan diukur ulang.

Sebelum lanjut, periksa kelengkapannya di Mac:

```bash
.venv/bin/python3 -c "
import pandas as pd
for nama in ('xgb', 'lstm'):
    d = pd.read_csv(f'dataset/model_ready/{nama}_search_results.csv')
    ids = sorted(int(v) for v in d['candidate_id'])
    print(nama, len(ids), 'kandidat, hilang:', sorted(set(range(30)) - set(ids)),
          ', device:', sorted(d['device'].unique()),
          ', gagal:', sorted(d.loc[d['pinball'].isna(),'candidate_id']))
"
```

Harus 30 kandidat, tidak ada yang hilang, `device` seluruhnya `cuda`. Kandidat
yang gagal (pinball NaN) boleh ada — kolom `error` menyebutkan sebabnya — tapi
kalau ada, baca dulu sebabnya sebelum memilih pemenang dari 29 kandidat.

---

## Langkah 10 — Tahap C di Mac

Tanpa `FORECAST_DEVICE` sama sekali, supaya keduanya kembali ke `cpu`:

```bash
cd ~/Project/Personal/forecast-scm
.venv/bin/python3 run_cells.py notebook/modeling_xgb.ipynb 2-10,14,16-24
.venv/bin/python3 run_cells.py notebook/modeling_lstm.ipynb 2-34
```

Berurutan, tidak paralel — wall time keduanya masuk K3, dan dua model yang
berebut core yang sama mengukur kontensi, bukan model.

Yang dihitung ulang di Mac: hanya walk-forward dan fit final. Sel pencarian
membaca checkpoint penuh, sel 26 LSTM membaca kembali `lstm_seed_repeats.csv`.
Yang tetap dibayar: sel 17 LSTM menjalankan benchmark lagi (~30 menit), karena
sel 19 membutuhkan `sec_per_epoch` dan pilihan device dari mesin ini.

Perkiraan: XGB ~3,5–15,5 jam, LSTM ~3,8–19,1 jam. Angka XGB-nya menjadi pasti
begitu pencarian selesai — **Tahap C ≈ 2,8 × `elapsed_seconds` kandidat
pemenang**, rasio yang dihitung dari jumlah baris latih tiap fold dikali
jumlah ronde yang dipilih early stopping.

---

## Lampiran — pesan yang mungkin muncul, dan artinya

| Pesan | Artinya | Tindakan |
|---|---|---|
| `RuntimeError: Root repo tidak ditemukan dari ...` | folder `dataset\csv` belum dibuat | Langkah 2 |
| `cuda dilewati: CUDA tidak tersedia di mesin ini` (LSTM sel 17) | torch CPU-only | Langkah 3 |
| `device: cpu` di keluaran sel 10 | `FORECAST_DEVICE` belum diset di jendela ini | Langkah 5 |
| `checkpoint ... berasal dari run kuantil tunggal` | CSV pra-2026-08-24 ikut tersalin | hapus CSV itu, ia bukan K1 |
| `checkpoint ... tidak cocok dengan ruang pencarian` | PC di commit yang berbeda dari yang melahirkan checkpoint | samakan commit-nya |
| `Falling back to prediction using DMatrix due to mismatched devices` | prediksi melintasi host↔device | **normal, bukan error** — ia sebabnya ×7,96 disebut lantai, bukan plafon |
| `!!! sel N GAGAL setelah X mnt` | satu sel melempar exception | baca traceback di log; kandidat yang sudah selesai aman di checkpoint |
| baris dengan `pinball` kosong dan `error` terisi | satu kandidat ditolak XGBoost/kehabisan memori | dicatat dan dilewati dengan sengaja; baca `error` sebelum memilih pemenang |
