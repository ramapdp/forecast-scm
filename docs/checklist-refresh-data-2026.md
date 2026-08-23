# Checklist Refresh Data 2026

**Dibuat:** 18 Agustus 2026
**Berlaku saat:** file CSV/Excel periode 2026 pertama masuk ke `dataset/`
**Tujuan:** mencatat apa yang **pasti rusak**, apa yang **diam-diam salah**, dan apa yang
**harus diukur ulang** ketika cakupan data melewati 31 Desember 2025.

Semua nomor baris di bawah diverifikasi pada commit `ac286be` (18 Agustus 2026). Kalau kode
sudah bergeser, cari nama konstantanya, jangan percaya nomor barisnya.

---

## 0. Urutan menjalankan refresh

Urutannya wajib, dan yang paling mudah terlewat: `prepare_forecast_data` **tidak** memanggil
`modeling_prep`, jadi keduanya harus dijalankan sendiri-sendiri:

```bash
.venv/bin/python3 -m utils.merge_dataset               # kalau ada file CSV baru
.venv/bin/python3 -m utils.verify_category_consistency # gerbang: nol SKU multi-kategori
.venv/bin/python3 -m utils.prepare_forecast_data       # featured.parquet + train/test.parquet
.venv/bin/python3 -m utils.modeling_prep               # model_input.parquet + category_mapping.json
.venv/bin/python3 -m unittest discover -p "test_*.py"
```

Kalau hanya perintah ketiga yang dijalankan, `model_input.parquet` tetap berisi data lama tanpa
peringatan apa pun. Tidak ada guard otomatis untuk ini.

`verify_category_consistency` keluar dengan status 1 kalau setelah normalisasi masih ada SKU
yang memikul lebih dari satu `Kategori Barang` — tanda reklasifikasi kategori belum sepenuhnya
tertangani, entah karena ada SKU baru yang perlu masuk
`normalize_items.EXPLICIT_CATEGORY_OVERRIDES` atau karena data sumber berubah di luar dugaan.
Jalankan sebelum pipeline, supaya tidak membangun ulang 1,5 juta baris di atas kategori yang
salah. Baris berlabel `Barang Semi FG (WIP-2)` di `dataset.csv` **bukan** kegagalan: sejak
konfirmasi 2026-08-22 reklasifikasi ditangani di lapisan normalisasi, bukan di data sumber
(lihat B-9 di `batasan-penelitian.md`).

---

## A. Akan gagal keras atau salah diam-diam — perbaiki **sebelum** pipeline dijalankan

### A-1. Kalender 2026 — satu gagal keras, satu salah diam-diam ⚠️

`utils/calendar_features.py` hanya mengenal 2024 dan 2025. Ada **dua** konstanta yang harus
diisi, dan hanya **satu** yang dijaga:

| Konstanta | Baris | Yang terjadi kalau lupa |
|---|---|---|
| `RAMADAN_PERIODS` | `calendar_features.py:13` | **Gagal keras** — `check_year_coverage` (`:168`) raise `ValueError` |
| `EID_AL_FITR_DATES` | `calendar_features.py:17` | Tidak dijaga; `is_eid_al_fitr` jadi `False` sepanjang 2026 |
| `EID_AL_ADHA_DATES` | `calendar_features.py:21` | Tidak dijaga; sama seperti di atas |
| `ID_HOLIDAYS` | `calendar_features.py:6` | **Tidak dijaga, dan ini yang paling berbahaya** |

**Jebakan `ID_HOLIDAYS` (sudah saya uji, bukan dugaan).** Objeknya dibuat dengan
`holidays.country_holidays("ID", years=[2024, 2025])`. `is_national_holiday` memakai
`.isin(ID_HOLIDAYS)`, yang membaca *kunci yang sudah terisi* — bukan lookup keanggotaan yang
biasanya auto-populate. Hasil uji: `is_national_holiday` untuk `2026-01-01`, `2026-08-17`, dan
`2026-03-20` mengembalikan **`False` semua**, sementara `2025-08-17` benar `True`.

Konsekuensinya: kalau `RAMADAN_PERIODS` diisi 2026 tetapi `years=[...]` di baris 6 lupa
ditambah, `check_year_coverage` **lolos** dan seluruh hari libur nasional 2026 masuk model
sebagai hari biasa — tanpa error, tanpa warning. Tambahkan `2026` di baris 6 bersamaan dengan
tiga konstanta lainnya, lalu buktikan:

```bash
.venv/bin/python3 -c "
import pandas as pd; from utils import calendar_features as cf
print(cf.is_national_holiday(pd.Series(pd.to_datetime(['2026-01-01','2026-08-17']))).tolist())"
# harus [True, True]
```

Tanggal Ramadan/Idulfitri/Iduladha 2026 **wajib dicocokkan ke kalender resmi pemerintah**, bukan
diturunkan sendiri — komentar di `calendar_features.py:8-10` mencatat prosedur verifikasi yang
dipakai untuk 2024–2025; ikuti prosedur yang sama.

### A-2. SKU baru tanpa entri di `event_driven_items.csv` — gagal keras

`modeling_prep.add_event_flag` (`modeling_prep.py:36`) sengaja raise, bukan default ke `false`:

```
ValueError: SKU tanpa entri di .../event_driven_items.csv: [...]
```

File saat ini memuat 70 SKU. Setiap SKU baru di data 2026 **harus diklasifikasi pemilik data**
(`is_event_driven` true/false), tidak boleh ditebak sendiri. Ini disengaja — lihat `B-9` di
`docs/batasan-penelitian.md` soal batas bukti klasifikasi ini.

### A-3. Cabang baru tanpa baris di `outlets.csv` / `outlet_mapping.csv` — gagal di QA

`prepare_forecast_data.run_qa_checks` (`prepare_forecast_data.py:335`) menolak `kota == "Unknown"`
dan cabang tanpa `kawasan`. Untuk tiap cabang baru, siapkan:

- `dataset/outlets.csv` — alamat, kecamatan, kota, channel online
- `dataset/outlet_mapping.csv` — `kawasan` (1 = Senin & Kamis, 2 = Selasa & Jumat) dan
  `hari_pengiriman`, **dari tim SCM**, bukan diinferensi dari cabang tetangga
- `dataset/outlet_name_overrides.csv` — hanya kalau cabang itu sebenarnya kelanjutan cabang lama

### A-4. `Kebuli Yaman Cikarang Pusat` — buang seluruh 2026-nya diam-diam kalau lupa

`dataset/outlet_closures.csv` mencatatnya tutup sejak `2025-12-01` dengan `tanggal_buka`
**kosong** = tutup tak berbatas. `build_dense_panel` membuang setiap tanggal di dalam interval
tutup, jadi selama kolom itu kosong, **semua baris 2026 cabang ini hilang dari panel tanpa
error**. Begitu pemilik data memberi tanggal buka, isi `tanggal_buka` **dan** perbarui
`RELOCATION_DATES` (lihat B-1) — keduanya manual, jangan diturunkan otomatis satu dari yang lain.

---

## B. Harus di-derive ulang — nilainya jadi salah kalau dibiarkan

### B-1. Lima tanggal relokasi batas-bawah akan **berbalik tanda** 🔴

`outlet_features.RELOCATION_DATES` (`outlet_features.py:323`) memuat lima cabang dengan tanggal
proksi `2025-12-31` (`Mayor Oking`, `Teluk Pucung`, `Bukit Gading Balaraja`,
`Grand Wisata Bekasi`) dan `2025-11-30` (`Cikarang Pusat`) — dipakai karena relokasi fisiknya
terjadi setelah data berakhir, jadi kode lama tidak pernah berhenti muncul.

Selama data berhenti di 2025-12-31, `days_since_relocation` selalu negatif dan artinya masih
benar ("cabang ini akan pindah nanti"). Begitu baris 2026 masuk, tanggal-tanggal itu jadi masa
lalu dan fiturnya berubah **positif** — artinya berubah total jadi "sudah sekian hari sejak
pindah", untuk relokasi yang tanggal aslinya tidak diketahui. Salah, dan tidak ada assert yang
menangkapnya.

**Yang harus dilakukan:** untuk tiap dari lima cabang itu, cari di data baru kapan kode lama
berhenti muncul / kode baru mulai muncul (pola yang sama seperti Cadas, Citayam, Bintara), lalu
ganti tanggalnya dengan tanggal exact dan hapus komentar `# lower bound`. Kalau kode lama
**masih** muncul di 2026, geser proksinya ke tanggal terakhir data yang baru — jangan biarkan di
2025-12-31. Perbarui juga `B-7` di `docs/batasan-penelitian.md`.

### B-2. `TEST_START` dan `FOLD_STARTS` harus digeser bersamaan

- `build_panel.TEST_START` (`build_panel.py:5`) = `2025-12-01`. Satu konstanta ini mengendalikan
  split train/test, guard kebocoran baseline outlier (`outlier_handling.py:15`), statistik cabang,
  `classify_pairs`, dan pembentukan `category_mapping` — `modeling_prep.py:71` sengaja mengimpor
  ulang dari `build_panel` supaya tidak ada literal kedua.
- `modeling_prep.FOLD_STARTS` (`modeling_prep.py:135`) = Jul–Nov 2025, **konstanta terpisah**.

Kalau `TEST_START` maju ke bulan 2026 tetapi `FOLD_STARTS` dibiarkan, lima fold validasi tetap
menempel di 2025 dan semakin jauh dari periode uji — tidak error, tapi seleksi model dilakukan di
periode yang tidak lagi mewakili. Geser keduanya: fold = lima bulan terakhir **sebelum** bulan uji
yang baru.

### B-3. Jalankan detektor gap, lalu tanyakan hasilnya ke pemilik data

`outlet_features.detect_unrecorded_gaps` (`outlet_features.py:115`, ambang `MIN_GAP_WARN_DAYS = 14`)
melaporkan jeda panjang yang tidak dijelaskan `outlet_closures.csv`. Ambang 14 hari **menangkap
kandidat, tidak mendefinisikannya** — KY068 Kramatwatu hanya 13 hari dan tetap tutup sementara.
Tiap temuan harus dikonfirmasi pemilik data (tutup sementara vs celah pencatatan) sebelum masuk
`outlet_closures.csv`; jangan diisi berdasarkan tebakan, karena file itu satu-satunya otoritas
atas apa yang dianggap "tutup" oleh pipeline.

Empat relokasi batas-bawah di B-1 kemungkinan besar akan muncul di sini sebagai pola tutup-buka.

### B-4. Kandidat pair baru masuk sendiri

`KY073 - Kebuli Yaman Cilebut` (buka 2025-12-19) belum pernah dapat ramalan karena nol hari
sebelum cutoff. Dengan data 2026 ia lolos `MIN_HISTORY_DAYS` (`build_panel.py:6`, 60 hari) dengan
sendirinya. Cek jumlah pair naik dari **2.979** dan jumlah cabang dari **59**.

---

## C. Sudah aman — jangan "diperbaiki"

- **`dataset/model_ready/category_mapping.json` jangan dihapus.** `build_category_mapping`
  (`modeling_prep.py:196`) menerima `existing=` dan `build_model_input` (`:592`) sudah mengoper
  `load_existing_mapping()`. Ini yang menjaga indeks kategori stabil lintas refresh — tanpa file
  itu, penomoran disusun ulang dari nol dan model yang sudah dilatih menunjuk kategori yang salah
  (terukur: 6 SKU baru menggeser indeks 32 dari 70 SKU lama).
- **Baseline outlier tidak bocor.** `compute_pair_baseline` (`outlier_handling.py:15`) sudah
  memfilter ke `< cutoff` sebelum agregasi. Angka capping akan berubah setelah cutoff bergeser —
  itu perilaku yang benar, bukan regresi.
- **`normalize_items` sudah menangani kuirk sumber**: `EXCLUDED_BRANCHES` (`:133`),
  `EXCLUDED_ITEMS` (`:164`), `GRAM_TO_PORSI_FACTORS` (`:142`), `CATEGORY_SYNONYMS` (`:89`),
  `EXPLICIT_CATEGORY_OVERRIDES` (`:99`). `EXPLICIT_ITEM_RENAMES` (`:173`) sengaja kosong — jangan
  diisi tanpa konfirmasi rename dari pemilik data.

---

## D. Angka di dokumen yang harus diukur ulang

Nilai-nilai ini di-hardcode dalam narasi dan akan basi begitu cakupan data berubah:

| Lokasi | Angka yang basi |
|---|---|
| `docs/batasan-penelitian.md` B-4 | 2.979 pair, 1.920 punya baris uji, 1.059 (35,6%) tidak pernah dinilai |
| `docs/batasan-penelitian.md` B-5 | target null 7,9% baris uji; periode efektif 1–29 Des 2025 |
| `docs/batasan-penelitian.md` B-6 | 55.046 baris uji, 16.031 (29,1%) hari kirim |
| `docs/batasan-penelitian.md` B-7 | daftar lima cabang batas-bawah (lihat B-1 di atas) |
| `docs/todolist-data-preprocessing.md` 🟡 | 842 pair gagal ambang 60 hari; pinball 0,384 `roll_mean_7`; 40.281 unit shortfall; 7.552 baris di-cap |
| `docs/pipeline-overview.md` | jumlah baris train/test, 205.513 baris pra-relokasi |

Ukur ulang setelah pipeline selesai, jangan disalin dari edisi sebelumnya.

---

## E. Sanity check rutin tiap refresh

- [ ] `Kuantitas` negatif — anomali KY011 2024-02-29 dulu berasal dari salah input sistem sumber.
      Konfirmasi 2026-08-10 menutup insidennya, **bukan** menjamin sistemnya tidak salah lagi.
- [ ] `EXCLUDED_ITEMS` (`xxx.FGS.00066/67/68/69`) tidak muncul di `model_ready/*.parquet`, dan
      Santan/Gula Cendol (`xxx.FGS.00070/71`) tetap dalam Satuan `Porsi`, bukan `Gr`.
      **Jangan verifikasi lewat `notebook/eda.ipynb`** — notebook itu membaca `dataset.csv` mentah,
      jadi item yang sudah di-exclude akan selamanya tetap tampil di sana. Cek lewat
      `normalize_items.load_and_normalize()` atau parquet akhir.
- [ ] File CSV baru ditambahkan ke `merge_dataset.SOURCE_FILES` (`merge_dataset.py:58`) —
      daftarnya hardcoded, bukan glob. Cek juga kuirk sumber: BOM UTF-8, pemisah `;`, dan kolom
      kosong berlebih di ujung (`jan-des-25.csv` punya 9 field, bukan 7).
- [ ] `.venv/bin/python3 -m unittest discover -p "test_*.py"` — 442 test per 18 Agustus 2026.
- [ ] Distribusi `Kuantitas` tetap right-skewed & kontinu (bukan integer), dan intermittency
      masih di kisaran yang sama (per Des 2025: 75,7% pair intermittent/lumpy).

---

## F. Belum jadi masalah, tapi akan

- **Fitur proksimitas Ramadan tidak melintasi tahun.** `days_until_ramadan`
  (`calendar_features.py:62-70`) mencari `RAMADAN_PERIODS[d.year]` — tahun baris itu sendiri. Baris
  Desember 2025 karena itu tidak pernah "melihat" Ramadan 2026 dan bernilai NaN. Untuk 2026 efeknya
  kecil (Ramadan jatuh pertengahan Februari, >45 hari dari Desember), tapi Ramadan mundur ~11 hari
  tiap tahun — begitu ia jatuh di Januari, baris Desember tahun sebelumnya akan kehilangan sinyal
  yang justru paling kuat. Perbaiki dengan lookahead lintas tahun sebelum itu terjadi.
- **Cold-start belum punya jalan keluar.** 842 pair gagal ambang 60 hari. Sudah diukur (17 Agustus)
  bahwa menurunkan ambang ke 28 **bukan** jawabannya: hanya menambah 0,023% volume permintaan dan
  0,18% baris latih, sambil membuat metrik tidak sebanding antar-run karena dilusi. Solusinya
  fallback terpisah yang harus mengalahkan pinball **0,384**. Data 2026 mengecilkan kohort ini,
  tapi tidak menghapusnya — pair baru terus lahir tiap kali ada cabang atau SKU baru.
- **Satu pertanyaan pemilik data masih layak diajukan:** dari baris yang di-cap, 38,6% melonjak
  sendirian (bukan pola pesanan besar serentak). Kalau lonjakan tunggal itu ternyata permintaan
  organik, capping memotong permintaan yang tidak ditutup jalur manual — dan sebagian dari 40.281
  unit shortfall jadi stockout nyata, bukan pesanan yang sudah ditangani head office.
