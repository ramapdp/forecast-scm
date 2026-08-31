# Detail Tahap Merge dan Agregasi Data

Dokumen ini adalah elaborasi mendetail dari Tahap 1 (Penggabungan periode) dan
Tahap 2 (Agregasi baris duplikat) yang sudah dijelaskan secara ringkas di
`docs/preprocessing.md` Bagian 1 bagian 4.1–4.2 dan Bagian 2 bagian 14.
Kedua bagian itu sengaja tetap ringkas karena mencakup 13–14 tahap
pipeline sekaligus; dokumen ini fokus hanya pada dua tahap paling awal —
penggabungan lima berkas periode menjadi satu tabel transaksi, dan agregasi
baris duplikat di dalamnya — dengan detail data, kode, dan pengujian yang
lebih dalam.

Dokumen dibagi menjadi dua bagian:

- **Bagian 1 — Akademis**, ditulis untuk menjadi bahan laporan/skripsi:
  penjelasan konseptual dengan struktur Tujuan/Prosedur/Keluaran/Justifikasi,
  tanpa detail implementasi.
- **Bagian 2 — Teknis**, ditulis untuk pembaca yang perlu memahami atau
  mengubah kode: nama fungsi, nomor baris, skema kolom eksak, dan hasil
  pengujian.

Seluruh angka pada dokumen ini diverifikasi langsung dari berkas yang ada di
`dataset/csv/` dan dari isi kode `utils/merge_split_data/merge_dataset.py` /
`aggregate_dataset.py` saat dokumen ini ditulis (2026-08-28), bukan disalin
dari dokumentasi lama.

---

# Bagian 1 — Akademis (Bahan Laporan)

## 1.1 Pendahuluan dan Posisi Tahap dalam Pipeline

Prapemrosesan data pada proyek peramalan permintaan rantai pasok ini terdiri
atas 14 tahap, dimulai dari penggabungan data mentah hingga penyusunan
kontrak masukan model. Dua tahap paling awal — penggabungan periode dan
agregasi baris duplikat — membentuk fondasi bagi seluruh tahap berikutnya:
keduanya menghasilkan satu tabel transaksi tunggal yang setiap barisnya unik
menurut kombinasi tanggal, item, cabang, dan satuan. Ketunggalan ini adalah
prasyarat yang dibutuhkan tahap konstruksi panel harian, karena tanpanya
akan muncul lebih dari satu baris untuk satu kombinasi tanggal-item-cabang,
sesuatu yang tidak dapat direpresentasikan oleh sebuah deret waktu tunggal.

## 1.2 Deskripsi Data Mentah Sebelum Penggabungan

Data transaksi mentah tersimpan dalam lima berkas CSV yang mempartisi sumbu
waktu tanpa tumpang tindih, mencakup periode Januari 2024 hingga Desember
2025: `jan-24.csv`, `feb-24.csv`, `mar-24.csv`, `apr-des-24.csv`, dan
`jan-des-25.csv`. Setiap berkas merupakan hasil ekspor dari berkas Excel
sumber pada periode yang sama.

Kelima berkas berbagi skema logis yang identik, yaitu tujuh kolom bertajuk
bahasa Indonesia: `Tanggal`, `Kategori Barang`, `Kode Barang`, `Nama Barang`,
`Nama Cabang`, `Satuan`, dan `Kuantitas`. Seluruh berkas menggunakan tanda
titik koma sebagai pemisah kolom dan encoding UTF-8 dengan penanda BOM (byte
order mark) di awal berkas. Kolom `Tanggal` berbentuk teks tanggal-bulan-tahun
(misalnya "01 Jan 2024"), sedangkan kolom `Kuantitas` berbentuk teks numerik
dengan tanda koma sebagai pemisah desimal (misalnya "220,0"), mengikuti
konvensi lokal Indonesia.

Volume baris data (tidak termasuk baris tajuk) pada tiap berkas, setelah
memisahkan baris data sebenarnya dari baris kosong yang dijelaskan pada bagian 1.6,
adalah sebagai berikut:

| Berkas         |    Baris data |
| -------------- | ------------: |
| jan-24.csv     |        48.233 |
| feb-24.csv     |        45.581 |
| mar-24.csv     |        61.693 |
| apr-des-24.csv |       509.493 |
| jan-des-25.csv |       883.269 |
| **Total**      | **1.548.269** |

Ekspor mentah dari sistem sumber juga mengandung artefak non-data — baris
yang seluruh kolomnya kosong — yang merupakan sisa proses ekspor ke Excel,
bukan bagian dari fenomena transaksi yang diteliti. Baris semacam ini
disingkirkan sebelum data dianggap sebagai data analisis; mekanisme
penyaringannya dijelaskan pada Bagian 2.

## 1.3 Tahap Penggabungan (Merge)

**Tujuan.** Menyatukan lima berkas yang mempartisi sumbu waktu menjadi satu
tabel tunggal yang berurutan secara kronologis, sebagai fondasi bagi seluruh
tahap berikutnya.

**Prosedur.** Kelima berkas dibaca satu per satu sesuai urutan periodenya.
Setiap baris divalidasi terhadap skema tujuh kolom yang telah ditetapkan.
Kolom `Tanggal` diuraikan (di-_parse_) semata-mata untuk keperluan
pengurutan — representasi teks aslinya tetap dipertahankan pada data
keluaran, tidak diubah formatnya. Seluruh baris dari kelima berkas kemudian
digabungkan dan diurutkan secara stabil berdasarkan tanggal yang telah
diuraikan tersebut.

**Keluaran.** Sebuah tabel gabungan berisi 1.548.269 baris — jumlah ini
adalah penjumlahan langsung dari kelima berkas sumber, karena tahap
penggabungan belum melakukan deduplikasi apa pun. Tabel ini ditulis dengan
skema dan format yang identik dengan berkas-berkas sumber (tujuh kolom yang
sama, pemisah titik koma, encoding yang sama).

**Justifikasi.** Tahap ini menganut prinsip kegagalan eksplisit: baris yang
tidak sesuai skema yang diharapkan menyebabkan proses berhenti dengan galat,
bukan diloloskan atau dihilangkan secara diam-diam. Prinsip ini penting
karena kesalahan skema pada data transaksi keuangan/inventori berisiko
tinggi jika tidak terdeteksi. Pengurutan yang stabil menjamin bahwa baris-
baris dengan tanggal yang sama mempertahankan urutan asalnya (urutan berkas
sumber, kemudian urutan dalam berkas), sehingga proses ini bersifat
deterministik dan dapat direproduksi dari data mentah yang sama.

## 1.4 Tahap Agregasi (Aggregate)

**Tujuan.** Menyatukan baris-baris yang identik pada seluruh atribut
kuncinya — tanggal, kategori barang, kode barang, nama barang, cabang, dan
satuan — menjadi satu baris per kombinasi unik. Kebutuhan ini muncul karena
satu peristiwa penyerahan barang di dunia nyata terkadang tercatat sebagai
beberapa baris (_line item_) terpisah di sistem sumber, misalnya karena
barang yang sama diserahkan dalam beberapa kali penyerahan pada hari dan
cabang yang sama.

**Prosedur.** Baris-baris hasil penggabungan dikelompokkan berdasarkan enam
kolom pertamanya (semua kolom kecuali `Kuantitas`). Untuk setiap kelompok,
nilai `Kuantitas` dari seluruh baris anggotanya dijumlahkan — bukan dipilih
salah satu — lalu hasilnya dibulatkan hingga satu angka desimal. Setiap
kelompok menghasilkan tepat satu baris pada keluaran.

**Keluaran.** Tabel akhir berisi 693.563 baris, masing-masing merepresentasikan
satu kombinasi unik tanggal-kategori-kode-nama-cabang-satuan, menggantikan
berkas hasil penggabungan pada lokasi yang sama.

**Justifikasi.** Penjumlahan (bukan pemilihan salah satu nilai) adalah
keputusan metodologis yang disengaja: setiap baris dalam satu kelompok
merepresentasikan kuantitas barang yang benar-benar diserahkan, sehingga
menjumlahkannya menghasilkan total penyerahan yang sebenarnya terjadi pada
kombinasi tanggal-item-cabang-satuan tersebut. Tahap ini juga memenuhi
prasyarat teknis bagi tahap konstruksi panel harian berikutnya, yang
mengharuskan tepat satu baris untuk setiap kombinasi kunci pada setiap
tanggal.

## 1.5 Karakteristik dan Struktur Data Keluaran

Skema kolom data keluaran identik secara struktural dengan skema data
masukan — tetap tujuh kolom yang sama — namun berbeda secara isi: data
keluaran sudah terdeduplikasi, dan nilai `Kuantitas` pada setiap barisnya
adalah hasil penjumlahan, bukan nilai transaksi tunggal.

Sebagai ilustrasi konkret, item dengan kode FGS-00001 di cabang KY001 pada
tanggal 1 Januari 2024 tercatat sebagai tujuh baris terpisah pada berkas
`jan-24.csv`, dengan kuantitas berturut-turut 220,0; 1,0; 2,0; 2,0; 3,0; 6,0;
dan 1,0. Ketujuh baris ini merepresentasikan penyerahan barang yang sama-
sama terjadi pada tanggal dan cabang yang sama, sehingga tahap agregasi
menyatukannya menjadi satu baris dengan kuantitas 235,0 — jumlah dari
ketujuh nilai tersebut.

Secara keseluruhan, jumlah baris berkurang dari 1.548.269 (hasil
penggabungan, sebelum agregasi) menjadi 693.563 (hasil akhir setelah
agregasi), sebuah reduksi sebesar 55,2%. Besarnya reduksi ini menjadi
indikator kuantitatif tingkat duplikasi baris pada data transaksi mentah:
lebih dari separuh baris pada data mentah adalah bagian dari kelompok
duplikat yang merepresentasikan peristiwa penyerahan yang sama.

## 1.6 Anomali dan Keterbatasan Data yang Relevan dengan Tahap Ini

Sebagaimana disinggung pada bagian 1.2, ekspor data mentah mengandung baris-baris
yang seluruh kolomnya kosong — artefak dari proses ekspor spreadsheet, bukan
transaksi sungguhan. Fenomena ini paling ekstrem terjadi pada `mar-24.csv`:
dari 1.048.575 baris data mentah pada berkas tersebut, sebanyak 986.882
baris (94,1%) adalah baris kosong, dan hanya 61.693 baris yang merupakan
data transaksi sesungguhnya. Baris-baris kosong ini disingkirkan sebelum
data dianggap sebagai bagian dari korpus analisis.

Perlu dicatat pula bahwa desain awal proses penggabungan turut
mengantisipasi kemungkinan salah satu berkas sumber memiliki dua kolom
tambahan yang kosong (sembilan kolom, bukan tujuh) — sebuah kondisi yang
pernah teramati pada data di masa lalu. Mekanisme penanganannya tetap
dipertahankan dalam kode sebagai langkah antisipatif dan telah diuji secara
formal, meskipun berkas yang tersedia saat dokumen ini ditulis tidak lagi
menunjukkan kondisi tersebut. Rincian teknis mengenai hal ini dijelaskan
pada Bagian 2 (bagian 2.7).

Dua anomali data lain yang relevan dengan data transaksi mentah — penanda
`xxx.` pada sebagian kode/nama barang di `apr-des-24.csv`, dan anomali
kuantitas negatif yang pernah teramati pada cabang KY011 tanggal 29 Februari
2024 — **berada di luar cakupan tahap penggabungan dan agregasi**. Keduanya
ditangani pada tahap-tahap berikutnya dalam pipeline (lihat
`docs/preprocessing.md` Bagian 1 bagian 4.3 untuk normalisasi kode barang, dan
bagian penjaminan mutu pada tahap prapemrosesan akhir untuk pemeriksaan
kuantitas negatif).

## 1.7 Ringkasan dan Kaitan dengan Tahap Berikutnya

Tahap penggabungan dan agregasi mengubah lima berkas transaksi periodik yang
terpisah, dengan potensi baris duplikat, menjadi satu tabel transaksi
tunggal (`dataset/csv/dataset.csv`, 693.563 baris) yang setiap barisnya unik
menurut kombinasi tanggal-item-cabang-satuan. Tabel inilah yang menjadi
satu-satunya sumber bagi seluruh tahap normalisasi dan rekayasa fitur
berikutnya dalam pipeline. Validitas seluruh hasil pemodelan pada akhirnya
bergantung pada ketunggalan baris yang dijamin oleh kedua tahap ini.

---

# Bagian 2 — Teknis (Mendetail)

## 2.1 Ringkasan Modul dan Entry Point

Tahap penggabungan dan agregasi diimplementasikan pada dua modul terpisah di
`utils/merge_split_data/`:

- `merge_dataset.py` — penggabungan lima berkas sumber
- `aggregate_dataset.py` — agregasi baris duplikat pada hasil penggabungan

Keduanya ditulis murni dengan pustaka standar Python (`csv`, `datetime`,
`pathlib`) tanpa dependensi eksternal. `aggregate_dataset.py` mengimpor
`merge_dataset` sebagai modul (`from . import merge_dataset`) untuk memakai
ulang fungsi I/O-nya — bukan memanggil `merge_dataset.main()`.

Entry point dijalankan sebagai modul dari root repo, **dalam urutan ini**
(agregasi membaca berkas hasil tulis penggabungan, jadi urutannya wajib):

```bash
.venv/bin/python3 -m utils.merge_split_data.merge_dataset
.venv/bin/python3 -m utils.merge_split_data.aggregate_dataset
```

Kedua modul memakai konvensi resolusi path yang sama:
`BASE_DIR = Path(__file__).resolve().parents[2]`, yaitu root repo, tiga
tingkat di atas `utils/merge_split_data/<modul>.py`.

## 2.2 Skema dan Format Berkas Sumber

Kelima berkas sumber, path-nya di-hardcode pada konstanta `SOURCE_FILES`
(`merge_dataset.py:82-88`) — **bukan** hasil glob pattern, sehingga
penambahan berkas periode baru mengharuskan penyuntingan manual konstanta
ini (lihat juga `docs/checklist-refresh-data-2026.md`).

Tajuk kolom yang identik di kelima berkas:

```
Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas
```

Format bersama: pemisah `;`, encoding `utf-8-sig` (BOM), `Kuantitas`
berdesimal koma (misalnya `"220,0"`).

Rincian volume baris per berkas, diverifikasi langsung dengan mereplikasi
logika `read_rows()` (lihat bagian 2.3) terhadap berkas di `dataset/csv/`:

| Berkas               | Baris data (non-kosong) | Baris kosong (filler) | Total field per baris |
| -------------------- | ----------------------: | --------------------: | --------------------- |
| jan-24.csv           |                  48.233 |                     0 | 7 (konsisten)         |
| feb-24.csv           |                  45.581 |                     0 | 7 (konsisten)         |
| mar-24.csv           |                  61.693 |               986.882 | 7 (konsisten)         |
| apr-des-24.csv       |                 509.493 |                     0 | 7 (konsisten)         |
| jan-des-25.csv       |                 883.269 |                     0 | 7 (konsisten)         |
| **Total baris data** |           **1.548.269** |                     — | —                     |

## 2.3 Fungsi-fungsi pada `merge_dataset.py`

- **`parse_tanggal(value: str) -> datetime.date`** (baris 12-14) — mengurai
  string bertipe `"01 Jan 2024"` menggunakan `DATE_FORMAT = "%d %b %Y"`.
  Hanya dipakai sebagai kunci pengurutan; teks tanggal asli pada kolom
  `Tanggal` tidak diubah.

- **`normalize_row(row) -> list[str]`** (baris 17-31) — memaksa setiap baris
  memiliki tepat `EXPECTED_FIELD_COUNT = 7` kolom. Jika baris punya kurang
  dari 7 kolom, memunculkan `ValueError`. Jika lebih dari 7 kolom, kolom ke-8
  dan seterusnya dipotong (_truncate_) — tapi **hanya jika seluruhnya kosong
  setelah di-`strip()`**; jika ada kolom tambahan yang tidak kosong, fungsi
  ini memunculkan `ValueError` ("Unexpected non-empty trailing field(s)").
  Inilah mekanisme eksak yang dirancang untuk menangani berkas sumber dengan
  kolom tambahan kosong (lihat bagian 2.7).

- **`read_rows(path) -> list[list[str]]`** (baris 36-49) — membuka berkas
  dengan `encoding="utf-8-sig"`, membaca dengan `csv.reader(f, delimiter=";")`,
  melewati baris tajuk (`next(reader)`), lalu menyaring baris yang seluruh
  kolomnya kosong (`if any(field.strip() for field in row)`) sebelum
  memvalidasi tiap baris tersisa lewat `normalize_row`. Filter baris-kosong
  inilah yang menyingkirkan 986.882 baris kosong pada `mar-24.csv`.

- **`write_rows(rows, path)`** (baris 52-57) — menulis dengan
  `encoding="utf-8-sig"` dan `delimiter=";"`, tajuk = `FIELDNAMES`, diikuti
  seluruh baris data.

- **`merge_and_sort(paths) -> list[list[str]]`** (baris 62-73) — membaca
  seluruh path lewat `read_rows()` sesuai urutan `SOURCE_FILES`, menggabungkan
  hasilnya, lalu mengurutkan dengan `list.sort(key=lambda row: parse_tanggal(row[0]))`.
  Karena `sort()` bawaan Python bersifat stabil dan berkas dibaca dalam
  urutan kronologis tetap, baris-baris dengan tanggal yang sama
  mempertahankan urutan berkas-asal kemudian urutan-dalam-berkas.

- **`FIELDNAMES`** (baris 76-79) — skema tujuh kolom kanonis:
  `["Tanggal", "Kategori Barang", "Kode Barang", "Nama Barang", "Nama Cabang", "Satuan", "Kuantitas"]`.
  Seluruh kolom, termasuk `Kuantitas`, tetap bertipe string di modul ini —
  tidak ada konversi ke tipe numerik.

- **`main(source_paths=SOURCE_FILES, output_path=OUTPUT_FILE)`** (baris
  95-99) — memanggil `merge_and_sort()` lalu `write_rows()`, mencetak jumlah
  baris yang ditulis. Tidak membaca argumen dari CLI (`sys.argv`); parameter
  `main()` hanya untuk keperluan pengujian.

**Path**: `SOURCE_FILES` = lima path `dataset/csv/{jan-24,feb-24,mar-24,apr-des-24,jan-des-25}.csv`;
`OUTPUT_FILE` = `dataset/csv/dataset.csv` (baris 82-90).

## 2.4 Fungsi-fungsi pada `aggregate_dataset.py`

- **`GROUP_FIELD_COUNT = 6`** (baris 3) — enam kolom pertama (semua kecuali
  `Kuantitas`) membentuk kunci pengelompokan.

- **`parse_kuantitas(value: str) -> float`** (baris 8-14) —
  `float(value.replace(",", "."))`, mengonversi format desimal-koma pada
  data sumber menjadi float Python.

- **`aggregate_rows(rows) -> list[list[str]]`** (baris 19-33) — inti tahap
  agregasi:

  ```python
  totals: dict[tuple[str, ...], float] = {}
  for row in rows:
      key = tuple(row[:GROUP_FIELD_COUNT])
      totals[key] = totals.get(key, 0.0) + parse_kuantitas(row[GROUP_FIELD_COUNT])
  return [list(key) + [str(round(total, 1))] for key, total in totals.items()]
  ```

  Dedup dilakukan secara implisit lewat `dict` Python biasa yang di-_key_
  dengan tuple 6-kolom — dua baris dengan 6-tuple yang identik otomatis
  menyatu. Tidak ada `pandas.groupby`, tidak ada penyaringan atau penanganan
  pencilan di fungsi ini — murni penjumlahan dan pembulatan satu desimal.
  Urutan baris pada keluaran mengikuti urutan kemunculan pertama tiap kunci
  pada data masukan (jaminan `dict` Python 3.7+ mempertahankan urutan
  penyisipan).

- **`INPUT_FILE = merge_dataset.OUTPUT_FILE`** (baris 36) — agregasi membaca
  dan menulis ke **berkas yang sama** dengan hasil penggabungan; tidak ada
  path keluaran terpisah.

- **`main(path=INPUT_FILE)`** (baris 41-46) — membaca lewat
  `merge_dataset.read_rows(path)`, menjalankan `aggregate_rows()`, lalu
  menulis kembali lewat `merge_dataset.write_rows(aggregated, path)` —
  **menimpa** berkas yang sama yang baru saja ditulis oleh `merge_dataset`.

## 2.5 Alur Eksekusi End-to-End

Langkah konkret menjalankan kedua tahap secara berurutan, beserta state
berkas `dataset/csv/dataset.csv` pada tiap langkah:

1. `python3 -m utils.merge_split_data.merge_dataset` dijalankan →
   `dataset/csv/dataset.csv` ditulis dengan **1.548.269 baris**, terurut
   kronologis, belum terdeduplikasi.
2. `python3 -m utils.merge_split_data.aggregate_dataset` dijalankan → skrip
   membaca 1.548.269 baris tersebut, mengagregasinya, lalu **menimpa** path
   yang sama dengan **693.563 baris** hasil agregasi.

Tidak ada artefak atau berkas antara yang disimpan di antara kedua langkah
ini — berkas ditimpa langsung. Untuk memeriksa state 1.548.269-baris
(sebelum agregasi), satu-satunya cara adalah menjalankan ulang
`merge_dataset` saja tanpa melanjutkan ke `aggregate_dataset`.

Contoh nyata end-to-end (diverifikasi langsung dari berkas di disk): tujuh
baris pada `jan-24.csv` untuk kombinasi (01 Jan 2024, Barang Semi FG (WIP-2),
FGS-00001, Ayam Kebuli (0.9), KY001 - Kebuli Yaman Kutabumi (Pusat), Potong)
dengan `Kuantitas` `220,0`, `1,0`, `2,0`, `2,0`, `3,0`, `6,0`, `1,0` —
masing-masing diuraikan lewat `parse_kuantitas()` menjadi float, dijumlahkan
menjadi `235.0`, lalu ditulis sebagai baris pertama pada
`dataset/csv/dataset.csv`:

```
01 Jan 2024;Barang Semi FG (WIP-2);FGS-00001;Ayam Kebuli (0.9);KY001 - Kebuli Yaman Kutabumi (Pusat);Potong;235.0
```

## 2.6 Skema Kolom Keluaran (Kamus Kolom)

| Kolom             | Setelah merge                               | Setelah aggregate                                                    | Format             | Catatan                                            |
| ----------------- | ------------------------------------------- | -------------------------------------------------------------------- | ------------------ | -------------------------------------------------- |
| `Tanggal`         | ada, teks asli                              | ada, tidak berubah                                                   | teks `DD Mon YYYY` | dipakai sebagai bagian kunci grup                  |
| `Kategori Barang` | ada                                         | ada, tidak berubah                                                   | teks               | bagian kunci grup                                  |
| `Kode Barang`     | ada                                         | ada, tidak berubah                                                   | teks               | bagian kunci grup                                  |
| `Nama Barang`     | ada                                         | ada, tidak berubah                                                   | teks               | bagian kunci grup                                  |
| `Nama Cabang`     | ada                                         | ada, tidak berubah                                                   | teks               | bagian kunci grup                                  |
| `Satuan`          | ada                                         | ada, tidak berubah                                                   | teks               | bagian kunci grup                                  |
| `Kuantitas`       | teks desimal-koma (nilai transaksi tunggal) | teks desimal-**titik** (hasil `sum` per kunci, dibulatkan 1 desimal) | teks               | satu-satunya kolom yang isinya berubah antar tahap |

Tajuk kolom, pemisah `;`, dan encoding `utf-8-sig` tidak berubah di kedua
tahap — keduanya memakai `FIELDNAMES` dan `write_rows()` yang sama dari
`merge_dataset.py`.

## 2.7 Diskrepansi dan Catatan Implementasi (Caveat)

Dua detail yang didokumentasikan pada desain awal (`docs/superpowers/specs/2026-07-18-merge-dataset-design.md`)
dan `CLAUDE.md` tidak lagi cocok dengan berkas yang tersedia di disk pada
saat dokumen ini ditulis (2026-08-28). Kedua mekanisme kode tetap nyata,
teruji lewat unit test, dan sengaja dipertahankan — catatan di bawah ini
menjelaskan **kondisi pemicunya yang sudah bergeser**, bukan bug pada kode:

- **Skema 9-kolom vs 7-kolom.** Logika `normalize_row()`/`EXPECTED_FIELD_COUNT = 7`
  dirancang dan diuji untuk menangani `jan-des-25.csv` versi lama yang punya
  sembilan kolom (dua kolom kosong tambahan di akhir setiap baris), dengan
  perilaku memotong kolom kosong tersebut dan memunculkan `ValueError` jika
  kolom tambahan itu ternyata tidak kosong. Verifikasi langsung terhadap
  berkas `dataset/csv/jan-des-25.csv` saat ini menunjukkan **setiap baris
  sudah memiliki tepat 7 kolom** — tidak ada kolom tambahan sama sekali.
  Mekanisme ini tetap berguna untuk dipertahankan sebagai pengaman: jika
  ekspor mendatang kembali menyertakan kolom tambahan yang tidak kosong,
  proses akan gagal secara eksplisit alih-alih diam-diam memuat data salah.

- **Jumlah baris kosong.** Dokumentasi/desain lama menyebut tiga berkas
  (`jan-24.csv`, `feb-24.csv`, `apr-des-24.csv`) masing-masing memiliki "3
  baris kosong trailing" (`;;;;;;`) sebagai artefak ekspor Excel. Verifikasi
  langsung saat ini menunjukkan **ketiga berkas tersebut tidak punya baris
  kosong sama sekali** (0 baris), sedangkan `mar-24.csv` — yang pada desain
  awal tidak disebut memiliki baris kosong — justru memilikinya dalam jumlah
  sangat besar: **986.882 dari 1.048.575 baris data** (94,1%). Filter baris-
  kosong pada `read_rows()` tetap aktif dan menjadi krusial khusus untuk
  `mar-24.csv` saat ini.

Kesimpulan: kedua mekanisme adalah perilaku defensif yang diimplementasikan
dan diuji dengan sengaja, dan kondisi nyata yang memicunya bergeser mengikuti
refresh data dari waktu ke waktu — bukan indikasi bahwa kode ketinggalan
zaman atau perlu diubah.

## 2.8 Pengujian

Perilaku pada bagian 2.3–2.7 divalidasi oleh test berikut:

**`test/test_merge_dataset.py`**

- `test_parses_day_month_year`, `test_parses_end_of_year_date` — `parse_tanggal`
- `test_row_with_exact_field_count_is_unchanged`, `test_row_with_empty_trailing_fields_is_trimmed`,
  `test_row_with_non_empty_trailing_field_raises`, `test_row_with_too_few_fields_raises` — `normalize_row`, termasuk kasus 9-vs-7-kolom
- `test_reads_and_normalizes_rows`, `test_skips_fully_blank_trailing_row`,
  `test_skips_blank_row_but_raises_on_bad_trailing_field_in_same_file` — `read_rows`, termasuk filter baris kosong
- `test_merges_and_sorts_chronologically_with_stable_ties`, `test_unparseable_tanggal_raises` — `merge_and_sort`
- `test_writes_bom_semicolon_header_and_rows` — `write_rows`
- `test_main_writes_merged_sorted_output` — `main`

**`test/test_aggregate_dataset.py`**

- `test_parses_plain_integer_string`, `test_parses_comma_decimal_whole_number`,
  `test_parses_comma_decimal_fractional_value` — `parse_kuantitas`
- `test_sums_kuantitas_for_rows_with_matching_key`, `test_keeps_rows_with_different_keys_separate`,
  `test_sums_rows_with_fractional_kuantitas`, `test_output_order_follows_first_occurrence_across_dates` — `aggregate_rows`
- `test_main_aggregates_file_in_place` — `main`

Perintah menjalankan:

```bash
.venv/bin/python3 -m unittest test.test_merge_dataset -v
.venv/bin/python3 -m unittest test.test_aggregate_dataset -v
```

## 2.9 Cakupan dan Batasan Dokumen Ini

Secara eksplisit di luar cakupan dokumen ini, karena ditangani pada tahap
lain dalam pipeline:

- **Normalisasi kode barang** (termasuk pembersihan penanda `xxx.` pada
  sebagian `Kode Barang`/`Nama Barang`) → `utils/data_preprocessing/normalize_items.py`,
  tahap berikutnya setelah agregasi.
- **Pemeriksaan kuantitas negatif** (anomali historis di cabang KY011,
  29 Februari 2024) → asersi QA pada `utils/data_preprocessing/prepare_forecast_data.py`
  (`run_qa_checks()`), bukan bagian dari tahap penggabungan/agregasi.
- **`sync_outlets.py`** — modul terpisah dan tidak berkaitan yang mengonversi
  `dataset/outlets.json` menjadi `dataset/outlets.csv` untuk keperluan tahap
  filter/kanonikalisasi cabang; bukan bagian dari alur data transaksional
  yang dibahas di dokumen ini.
