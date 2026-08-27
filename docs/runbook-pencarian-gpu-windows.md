# Runbook — menjalankan Tahap B (pencarian) di PC Windows + RTX 4060 Ti 8 GB

Dokumen ini adalah prosedur, bukan desain. Alasan di balik pembagian mesinnya
ada di Bagian 1 dan §0bis
`superpowers/specs/2026-08-24-distributed-gpu-training-design.md`; di sini
hanya langkah-langkahnya, berurutan, untuk dijalankan sekali dari awal sampai
selesai.

Keputusan pemilik proyek **2026-08-26**: pencarian hyperparameter XGBoost dan
LSTM dijalankan di PC Windows ber-RTX 4060 Ti 8 GB; walk-forward dan fit
final tetap di Mac. Random Forest tidak ikut sama sekali — ketiga tahapnya
sudah selesai di Mac (run 2026-08-25), dan `quantile-forest` murni CPU.

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
cp dataset/model_ready/category_mapping.json /tmp/kirim-ke-pc/
git log -1 --format=%h            # catat hash ini, PC harus di commit yang sama
```

`model_input.parquet` berukuran ~42 MB. Ia **tidak** ada di git (`dataset/`
gitignored), jadi harus disalin manual — flashdisk, jaringan lokal, atau cloud
drive, terserah.

`category_mapping.json` (~5 KB) **wajib ikut** meski jauh lebih kecil — ia
keluaran kedua `modeling_prep.build_model_input()`, ditulis bersamaan dengan
`model_input.parquet` tapi sebagai berkas terpisah. LSTM membacanya lewat
`modeling_prep.load_category_mapping()` untuk ukuran embedding tiap kategori;
tanpanya sel benchmark LSTM gagal `FileNotFoundError` di tengah jalan.

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
copy <lokasi>\category_mapping.json dataset\model_ready\
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

### Dua prasyarat sistem, di luar pip

Keduanya punya cek versi (menjelaskan **kenapa** rusak) dan cek fungsional
(menjawab **apakah** rusak). Cek fungsionalnya ada di Langkah 4 dan lebih kuat
— kalau 4a dan 4b lolos, kedua prasyarat ini sudah terpenuhi menurut definisi,
berapa pun angka versinya. Yang di bawah ini dipakai saat 4a atau 4b gagal.

**A. Visual C++ Redistributable (x64, 2015–2022)**

Yang harus ada tiga DLL: `vcruntime140.dll`, `vcruntime140_1.dll`, dan
`msvcp140.dll`. Yang biasanya hilang adalah **`vcruntime140_1.dll`** — ia baru
ikut sejak redistributable 2019, jadi PC yang hanya pernah memasang versi 2015
punya dua DLL pertama dan tidak punya yang ini. Baik `xgboost.dll` maupun
`torch` menautnya.

```powershell
"vcruntime140.dll","vcruntime140_1.dll","msvcp140.dll" | ForEach-Object {
    "{0,-22} {1}" -f $_, (Test-Path "$env:SystemRoot\System32\$_")
}
Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" `
    -ErrorAction SilentlyContinue | Select-Object Installed, Version
```

Ketiganya harus `True`, dan registrinya `Installed : 1`. Kalau ada yang
`False`, pasang `vc_redist.x64.exe` dari
<https://aka.ms/vs/17/release/vc_redist.x64.exe>, lalu ulangi.

Cek fungsionalnya satu baris:

```powershell
.venv\Scripts\python -c "import xgboost; print(xgboost.__version__)"
```

Berhasil mencetak `2.1.4` berarti DLL-nya sudah teresolusi seluruhnya.
`OSError: [WinError 126] The specified module could not be found` atau
`XGBoostError` saat memuat pustaka berarti belum.

**B. Driver NVIDIA**

CUDA Toolkit **tidak perlu dipasang** — wheel xgboost dan torch sudah membawa
runtime CUDA-nya sendiri. Yang harus ada hanyalah driver, dan cukup baru:
lantai untuk CUDA 12.x di Windows adalah **527.41**. Arsitektur GPU-nya tidak
pernah jadi soal — RTX 4060 Ti itu Ada Lovelace (SM 8.9), didukung penuh
sejak CUDA 11.8 dan oleh seluruh CUDA 12.x; yang bisa salah hanya umur
drivernya.

```powershell
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvidia-smi
```

Baris pertama mencetak mis. `NVIDIA GeForce RTX 4060 Ti, 581.29` — angka kedua
harus ≥ 527.41. Kalau perintahnya sendiri tidak dikenali, drivernya belum
terpasang sama sekali.

Satu hal yang sering disalahbaca: kolom `CUDA Version: 12.x` di pojok kanan
atas keluaran `nvidia-smi` **bukan** CUDA yang terpasang, melainkan CUDA
tertinggi yang sanggup dilayani driver ini. Ia tidak perlu sama dengan CUDA
wheel torch (`cu126`) — minor version compatibility membuat runtime 12.6
berjalan di atas driver yang mendukung 12.0. Yang perlu dicocokkan hanya
lantainya.

Driver di PC ini melaporkan `CUDA Version: 13.1` — jauh di atas lantai, dan
itu tidak menuntut apa pun. Driver NVIDIA kompatibel mundur terhadap runtime
CUDA yang lebih lama, jadi wheel `cu126` berjalan apa adanya. Jangan tergoda
mengejar wheel CUDA 13: torch 2.8.0 tidak menerbitkannya, dan menaikkan
versi torch berarti melepas pin yang membuat hasil PC sebanding dengan hasil
Mac.

Cek fungsionalnya:

```powershell
.venv\Scripts\python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

Harus mencetak dua nilai, mis. `12.6 True`. **`None` di nilai pertama berarti
wheel yang terpasang CPU-only** — itu soal Langkah 3, bukan soal driver, dan
tidak akan tertolong dengan memperbarui driver. `12.6 False` barulah soal
driver: wheel-nya benar, tetapi driver menolak atau terlalu tua.

Kalau `quantile-forest` gagal dipasang di Windows, abaikan saja — ia hanya
dipakai Random Forest, yang tidak dijalankan di PC. Pasang sisanya dan
lanjutkan.

`ipykernel` ada di `requirements.txt` sejak 2026-08-26 dan tidak boleh
dilewatkan: `run_cells.py` menjalankan notebook lewat kernel bernama `python3`,
dan kernelspec itu dipasang oleh ipykernel — bukan oleh nbconvert maupun
nbclient, yang tidak bergantung padanya sama sekali. Tanpa ia, perintah apa pun
di Langkah 4d dan seterusnya berhenti dengan
`jupyter_client.kernelspec.NoSuchKernel: No such kernel named python3`.

---

## Langkah 4 — Verifikasi, sebelum satu jam pun dikomitkan

Empat pemeriksaan. Jangan lanjut sebelum keempatnya benar; masing-masing
menutup satu cara run 30 jam bisa terbuang.

**4a. GPU terlihat oleh torch**

```powershell
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Harus mencetak `True NVIDIA GeForce RTX 4060 Ti`. Kalau `False`, Langkah 3 belum
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

Set device-nya **permanen di level user**, bukan per jendela:

```powershell
[Environment]::SetEnvironmentVariable("FORECAST_DEVICE", "cuda", "User")
```

Lalu **tutup jendela PowerShell itu dan buka yang baru** — jendela yang sudah
terbuka tidak pernah memungut nilai baru. Cara ini dipilih karena pekerjaannya
berjalan berhari-hari dan hampir pasti melintasi beberapa jendela; `$env:` yang
hanya hidup satu sesi adalah cara paling mudah menjalankan 15 jam pencarian di
CPU tanpa sadar.

Kalau lebih suka per sesi, gabungkan set dan jalankan dalam **satu baris**,
supaya keduanya tidak mungkin terpisah jendela:

```powershell
$env:FORECAST_DEVICE = "cuda"; .venv\Scripts\python run_cells.py notebook\modeling_xgb.ipynb 2-10
```

### Verifikasi — yang dilihat Python, bukan yang diketik shell

```powershell
.venv\Scripts\python -c "import os; print(repr(os.environ.get('FORECAST_DEVICE')))"
```

Harus mencetak `'cuda'`. `None` berarti belum sampai, apa pun yang tampak sudah
diketik. Lalu ulangi Langkah 4d — barisnya harus berbunyi `device: cuda`.

Tiga cara ia gagal diam-diam, semuanya tanpa pesan error:

| Yang diketik | Di PowerShell | Di cmd.exe |
|---|---|---|
| `$env:FORECAST_DEVICE = "cuda"` | **benar** | error sintaks (kelihatan) |
| `set FORECAST_DEVICE=cuda` | **gagal diam-diam** | benar |
| `export FORECAST_DEVICE=cuda` | error (kelihatan) | error (kelihatan) |

Baris tengah itu jebakannya: di PowerShell, `set` adalah alias `Set-Variable`,
jadi `set FORECAST_DEVICE=cuda` membuat variabel **PowerShell** bernama
`FORECAST_DEVICE=cuda` — bukan variabel environment. Ia tidak mengeluh, dan
Python tidak pernah melihatnya.

Penyebab keempat yang sama seringnya: variabelnya diset di satu jendela dan
perintahnya dijalankan di jendela lain.

`FORECAST_DEVICE` wajib untuk XGBoost karena `DEFAULT_DEVICE = "cpu"`. Untuk
LSTM ia opsional — sel 19 memilih device tercepat dari benchmarknya sendiri —
tetapi membiarkannya terset tidak merugikan.

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
CPU Mac. Di RTX 4060 Ti ia seharusnya selesai dalam **~40 menit**.

Angka itu diturunkan dari pengganda ×7,96 yang diukur di T4 Kaggle, bukan di
kartu ini. 4060 Ti unggul jauh di compute (22 lawan 8,1 TFLOPS FP32) tetapi
busnya lebih sempit (288 lawan 320 GB/s), dan pemindaian matriks terkuantisasi
XGBoost sensitif terhadap bandwidth — L2 32 MB-nya yang menutup selisih itu.
Jadi ~40 menit adalah patokan yang wajar, bukan prediksi.

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

Dasar CPU LSTM adalah **3.412 detik per kandidat**. Di 4060 Ti harapkan
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
| `NoSuchKernel: No such kernel named python3` | `ipykernel` belum terpasang; ia yang memasang kernelspec `python3`, bukan nbconvert | `.venv\Scripts\pip install ipykernel` |
| `RuntimeError: Root repo tidak ditemukan dari ...` | folder `dataset\csv` belum dibuat | Langkah 2 |
| `cuda dilewati: CUDA tidak tersedia di mesin ini` (LSTM sel 17) | torch CPU-only | Langkah 3 |
| `device: cpu` di keluaran sel 10 | `FORECAST_DEVICE` tidak sampai ke Python — jendela lain, atau `set` dipakai di PowerShell | Langkah 5, mulai dari perintah verifikasi |
| `checkpoint ... berasal dari run kuantil tunggal` | CSV pra-2026-08-24 ikut tersalin | hapus CSV itu, ia bukan K1 |
| `checkpoint ... tidak cocok dengan ruang pencarian` | PC di commit yang berbeda dari yang melahirkan checkpoint | samakan commit-nya |
| `FileNotFoundError: ... category_mapping.json` (LSTM sel 17) | hanya `model_input.parquet` yang disalin di Langkah 1/2, `category_mapping.json` (keluaran kedua `build_model_input()`, dibaca `load_category_mapping()` untuk ukuran embedding) ikut tertinggal | salin `dataset/model_ready/category_mapping.json` dari Mac ke lokasi sama di PC, ulangi Langkah 7 dari sel 17 |
| `Falling back to prediction using DMatrix due to mismatched devices` | prediksi melintasi host↔device | **normal, bukan error** — ia sebabnya ×7,96 disebut lantai, bukan plafon |
| `!!! sel N GAGAL setelah X mnt` | satu sel melempar exception | baca traceback di log; kandidat yang sudah selesai aman di checkpoint |
| baris dengan `pinball` kosong dan `error` terisi | satu kandidat ditolak XGBoost/kehabisan memori | dicatat dan dilewati dengan sengaja; baca `error` sebelum memilih pemenang |
