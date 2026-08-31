# Analisis Lonjakan Permintaan — Apa yang Sebenarnya Dipangkas Capping

**Status: DITUTUP 2026-08-24** dengan konfirmasi pemilik data. Dokumen ini
merangkum rantai temuan untuk butir A3 no. 3 dan G2 "spike yang berdiri
sendiri" di `docs/todolist-proyek.md`, sekaligus menjadi catatan permanennya.

## Catatan nama berkas

Dokumen ini semula direncanakan bernama `analisis-lonjakan-packaging.md`, karena
pertanyaannya berangkat dari fakta bahwa 60,1% baris yang di-cap berkategori
`Packaging`. Namanya diganti menjadi `analisis-lonjakan-permintaan.md` setelah
analisisnya selesai, dan pergantian nama itu **adalah salah satu temuannya**:
isinya ternyata bukan tentang kemasan.

Yang sebenarnya terjadi adalah ambang deteksi yang bersifat **relatif** (≥5×
median pasangan) membuat peristiwa permintaan yang nyata dan menyeluruh terbaca
seolah hanya menyentuh kemasan. Pada hari ramai, Nasi Kebuli naik ~2× median-nya
— besar secara volume, tetapi jauh di bawah ambang 5×, jadi tidak pernah
ditandai. Rice Bowl 600 ml, yang median hariannya hanya 8 unit, naik melewati 5×
pada hari yang sama dan ditandai. Kategori yang muncul di daftar lonjakan karena
itu ditentukan oleh **kecilnya median pasangan**, bukan oleh apa yang bergerak.

---

## 1. Pertanyaan yang dijawab

Dari 7.552 baris panel yang dipangkas `apply_outlier_capping`
(`utils/data_preprocessing/outlier_handling.py`, ambang `SPIKE_RATIO_THRESHOLD = 5.0`), 38,6%
melonjak **sendirian** — hanya satu item yang melonjak di cabang itu pada hari
itu, tanpa item lain menyertainya. Sisanya melonjak serentak dengan ≥2 item
lain, pola yang khas untuk pesanan besar yang dilayani sekaligus.

Lonjakan serentak mudah dibaca sebagai pesanan. Yang sendirian tidak. Kalau
lonjakan sendirian ternyata **bukan** pesanan, capping memangkas permintaan yang
tidak ditutup jalur manual head office (B-3), dan sebagian shortfall yang
terukur adalah stockout sungguhan, bukan pesanan yang sudah ditangani di luar
model.

Pertanyaan aslinya diajukan sebagai pilihan biner: pre-order, atau permintaan
organik/restock? Jawabannya bukan salah satu dari keduanya — lihat bagian 4.

## 2. Langkah 1 — hipotesis restock gugur

**Skrip:** `utils/eda/analyze_spike_recovery.py` (`.venv/bin/python3 -m
utils.eda.analyze_spike_recovery`).

Hipotesis restock: lonjakan bukan permintaan tambahan, melainkan pembelian
borongan yang "meminjam" permintaan hari-hari berikutnya — outlet menarik stok
ke gudangnya sendiri, lalu beberapa hari sesudahnya tidak perlu mengambil apa
pun. Kalau benar, permintaan sesudah lonjakan turun di bawah level sebelumnya.

Diukur pada jendela 7 hari sebelum vs 7 hari sesudah, pada pasangan (item,
cabang) yang sama, dengan `segment_id` ikut menjadi kunci sehingga tidak ada
jendela yang menyeberangi masa tutup cabang:

| Kelompok (Packaging di-cap)           |      n | rata2 sebelum | rata2 sesudah |   selisih | rasio median | p (Wilcoxon) |
| ------------------------------------- | -----: | ------------: | ------------: | --------: | -----------: | -----------: |
| semua baris di-cap                    |  4.269 |         17,59 |         17,71 |     +0,7% |        1,000 |        0,676 |
| lonjakan **sendirian**                |  1.122 |         25,35 |         25,70 | **+1,4%** |        1,000 |        0,237 |
| lonjakan serentak (≥3 item)           |  2.784 |         10,87 |         10,64 |     −2,0% |        1,000 |        0,919 |
| kontrol: hari biasa (rasio 0,8–1,25×) | 93.004 |         24,33 |         24,50 |     +0,7% |        1,000 |        0,003 |

Dijalankan ulang dengan hari lonjakan lain **dikeluarkan** dari kedua jendela
(supaya lonjakan tetangga tidak menutupi penurunan yang sedang dicari), hasilnya
tetap: lonjakan sendirian +0,2%, p = 0,999.

Sebagai pemeriksaan ketiga, **kadensi** pengambilan juga tidak berubah — jumlah
hari bergerak (`Kuantitas > 0`) dalam 7 hari sesudah dibandingkan 7 hari
sebelum:

| Kelompok               | hari bergerak/7 sebelum | sesudah |   selisih |
| ---------------------- | ----------------------: | ------: | --------: |
| semua Packaging di-cap |                    4,93 |    5,02 |     +1,7% |
| lonjakan sendirian     |                    4,95 |    4,97 | **+0,3%** |
| kontrol hari biasa     |                    5,24 |    5,24 |     +0,0% |

**Kesimpulan.** Tidak ada penurunan sesudah lonjakan, baik pada volume maupun
pada frekuensi pengambilan. Lonjakan bersifat **aditif**: ia berdiri di atas
garis dasar, bukan menggantikan permintaan hari-hari berikutnya. Hipotesis
restock/borongan gugur.

Perlu dicatat satu hal yang justru menguatkan pembacaan ini: kelompok kontrol
punya p = 0,003 — hari biasa pun menunjukkan kenaikan kecil sesudah, kemungkinan
tren umum — sementara ketiga kelompok lonjakan tidak signifikan sama sekali.
Jadi bukan hanya "tidak turun", melainkan tidak berbeda dari perilaku hari biasa
pada arah mana pun.

## 3. Langkah 2 — yang melonjak sendirian ternyata bukan sendirian

**Skrip:** `utils/eda/analyze_spike_comovement.py` (`.venv/bin/python3 -m
utils.eda.analyze_spike_comovement`).

Setelah restock gugur, tersisa dua kemungkinan: hari ramai sungguhan (permintaan
menyeluruh, kemasan hanya bagian yang tertangkap ambang) atau gerakan kemasan
saja (penerimaan/penataan gudang, makanan tetap di level biasa).

Keduanya dipisahkan dengan memeriksa apakah makanan bervolume tinggi ikut naik
di cabang yang sama pada hari yang sama. Pembandingnya selalu cabang itu
sendiri: rasio hari lonjakan diletakkan sebagai persentil di dalam sebaran
rasio pasangan itu pada hari biasa, sehingga hipotesis nolnya adalah 0,500 —
angka yang tidak dikarang, melainkan konsekuensi dari "hari lonjakan kemasan
adalah hari acak di cabang itu".

1.157 cabang-hari punya lonjakan Packaging berdiri sendiri; 29.905 cabang-hari
tanpa lonjakan apa pun menjadi pembanding.

| Item                    | rasio median hari lonjakan | rasio median hari biasa |    pangsa ≥2× | persentil (dicocokkan per hari-dalam-minggu) |        p |
| ----------------------- | -------------------------: | ----------------------: | ------------: | -------------------------------------------: | -------: |
| Nasi Kebuli (FGS-00004) |                  **1,915** |                   0,912 | 46,8% vs 6,5% |                                    **0,821** | 2,9e−137 |
| Sambal - FG (FGS-00005) |                      1,885 |                   0,917 | 45,4% vs 6,3% |                                        0,818 | 3,2e−136 |
| Ayam Kebuli (FGS-00001) |                      1,659 |                   0,921 | 32,9% vs 5,7% |                                        0,756 | 1,7e−112 |

Pencocokan per hari-dalam-minggu itu perlu, bukan hiasan: 65,1% lonjakan
Packaging sendirian jatuh di akhir pekan sementara hari biasa hanya 24,7%. Tanpa
pencocokan, yang terukur bisa jadi hanya "akhir pekan memang lebih ramai".
Persentil tetap 0,82 sesudah dicocokkan (turun dari 0,88 tanpa pencocokan), jadi
efeknya bertahan di atas efek hari.

Dipecah per kemasan yang melonjak, persentil Nasi Kebuli pada hari itu:

| Kemasan yang melonjak |   n | rasio median Nasi Kebuli | persentil |       p |
| --------------------- | --: | -----------------------: | --------: | ------: |
| Rice Bowl 600 ml      | 815 |                 **2,04** |     0,848 | 4,3e−97 |
| Lunch Box             |  35 |                     2,76 |     0,856 | 9,6e−08 |
| Gelas 16 Oz           |  87 |                     1,31 |     0,729 | 7,1e−11 |
| Gelas 22 Oz           | 135 |                     1,28 |     0,695 | 2,8e−12 |

**Kesimpulan.** Lonjakan Packaging "sendirian" adalah **peristiwa permintaan
nyata**, bukan pergerakan kemasan saja. Makanan ikut naik ~2× median-nya, di
persentil ~0,82 dari sebaran cabangnya sendiri. Ia terbaca sendirian hanya
karena ambang 5× tidak pernah tersentuh oleh item bervolume besar.

## 4. Langkah 3 — konfirmasi pemilik data (2026-08-24)

Pertanyaan diajukan sebagai pilihan biner: lonjakan yang berdiri sendiri itu
pesanan, atau permintaan pelanggan langsung?

**Jawaban pemilik data: keduanya.** Di akhir pekan memang sering terjadi
lonjakan, dan lonjakan itu datang lewat **kedua jalur sekaligus** — sebagian
pesanan, sebagian pelanggan yang datang langsung.

Jadi resolusinya bukan salah satu dari dua opsi yang ditanyakan, melainkan opsi
ketiga yang tidak ada dalam pertanyaan: **bercampur, dan tidak dapat dipisahkan
dari data yang ada.**

Ini konsisten dengan kedua langkah sebelumnya. Aditif (bagian 2) menjelaskan kenapa
tidak ada penurunan sesudahnya — permintaan itu memang benar-benar terjadi,
lewat jalur mana pun ia datang. Ko-gerakan makanan (bagian 3) menjelaskan kenapa
lonjakan itu terlihat menyeluruh di cabang. Konsentrasi akhir pekan yang terukur
(bagian 5) adalah bentuk yang disebutkan pemilik data.

## 5. Konsekuensi yang belum masuk pertimbangan sebelumnya

Justifikasi capping selama ini: komponen yang dipangkas **tidak dapat
diprediksi**, karena ia adalah proksi pre-order yang buku pesanannya tidak
terekam (B-1/B-2) dan ditangani jalur manual (B-3).

Alasan itu tetap berlaku untuk **komponen pesanan**. Ia **tidak** berlaku untuk
komponen pelanggan langsung akhir pekan. `day_of_week` dan `is_weekend` ada di
56 kolom `FEATURE_COLS` (`utils/modelling/modeling_prep.py`) — pola akhir pekan justru
dapat dan seharusnya dipelajari model. Untuk baris yang dipangkas, capping
karena itu menghapus **sebagian sinyal yang sebenarnya dapat dipelajari**.

Konsentrasi akhir pekannya tegas dan bergradasi rapi:

| Hari       | Pangsa baris di-cap | Pangsa seluruh panel |
| ---------- | ------------------: | -------------------: |
| Senin      |                7,0% |                14,3% |
| Selasa     |                8,7% |                14,3% |
| Rabu       |               10,3% |                14,3% |
| Kamis      |               12,2% |                14,2% |
| Jumat      |               17,4% |                14,3% |
| **Sabtu**  |           **20,4%** |                14,3% |
| **Minggu** |           **23,9%** |                14,3% |

### Dampaknya terbatas, dan arahnya bisa diperkirakan

Karena ambang 5× bersifat **relatif terhadap median pasangan**, item bervolume
besar praktis tidak terkena. Nasi Kebuli naik 2,04× pada hari-hari itu — jauh di
bawah ambang — sehingga pola akhir pekannya **tetap utuh di target latih**.
Median `pair_median` untuk Nasi Kebuli adalah 86 unit/hari, dengan hanya 94
baris di-cap sepanjang dua tahun data.

Yang terpangkas adalah item bervolume kecil:

| Median pasangan (unit/hari) | Baris di-cap | % baris di-cap |
| --------------------------- | -----------: | -------------: |
| ≤2                          |        3.819 |          50,6% |
| 3–5                         |        1.562 |          20,7% |
| 6–10                        |          935 |          12,4% |
| 11–25                       |          565 |           7,5% |
| 26–100                      |          632 |           8,4% |
| >100                        |           39 |           0,5% |

83,6% baris yang dipangkas berasal dari pasangan dengan median ≤10 unit/hari;
median dari `pair_median` pada baris yang di-cap adalah **2 unit/hari**. Item
terbanyak: Rice Bowl 600 ml (1.186 baris, median pasangan 8) dan kelompok Loyang
(`PCG-00003`–`00013`, 2.538 baris, median pasangan 2–4).

### Prediksi yang bisa diuji

Model kemungkinan **sistematis meramal terlalu rendah di akhir pekan untuk item
bervolume kecil**, karena justru di segmen itu target latihnya dipangkas paling
sering, dan justru pada hari yang polanya bisa dipelajari.

Pemeriksaannya ada di `docs/todolist-proyek.md` Fase D butir 0d. Kalau
terkonfirmasi, itu ongkos capping yang **terukur** dan masuk bab batasan sebagai
angka, bukan dugaan. Kalau tidak terkonfirmasi, itu juga temuan yang layak
ditulis — berarti capping tidak semahal yang diperkirakan di sini.

## 6. Yang tidak berubah

**Keputusan target tetap: latih di `capped`, nilai (K1) di mentah**
(A3, ditutup 2026-08-24; `modeling_prep.TRAIN_TARGET_COL`/`EVAL_TARGET_COL`).

Jawaban pemilik data justru **menguatkan** keputusan itu, bukan membukanya
kembali. Menilai di target mentah tidak bergantung pada mekanisme capping sama
sekali — dan mekanisme itulah yang lewat analisis ini terbukti cakupannya tidak
sebersih asumsi awal. Kriteria pemilihan model karena itu tetap kebal terhadap
temuan di sini; yang bergeser hanya pemahaman kita tentang apa yang hilang di
sisi latih, dan itu ongkos yang diukur di butir 0d, bukan alasan mengganti
target.

## 7. Rujukan

- `docs/batasan-penelitian.md` B-3 (catatan bertanggal 2026-08-24), B-1, B-2
- `docs/todolist-proyek.md` A3 no. 3, G2, Fase D butir 0d
- `utils/eda/analyze_spike_recovery.py`, `utils/eda/analyze_spike_comovement.py` —
  keduanya hanya membaca `featured.parquet` dan mencetak tabel; bukan bagian
  pipeline, tidak menulis artefak
- `utils/data_preprocessing/outlier_handling.py` — `SPIKE_RATIO_THRESHOLD = 5.0`, pengecualian
  jendela event
