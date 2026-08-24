# Metodologi Prapemrosesan Data Peramalan Permintaan Rantai Pasok

**Dokumen metodologis untuk penulisan laporan/artikel ilmiah**

| Atribut | Keterangan |
|---|---|
| Domain kajian | Peramalan permintaan (*demand forecasting*) rantai pasok kuliner multi-cabang |
| Objek studi | Jaringan gerai Kebuli Yaman, 59 cabang aktif di 16 kota |
| Periode data | 1 Januari 2024 – 31 Desember 2025 (731 hari kalender) |
| Unit analisis | Pasangan (kode barang × cabang) per hari kalender |
| Berkas keluaran akhir | `dataset/model_ready/model_input.parquet` (1.502.522 baris × 82 kolom) |
| Status verifikasi | 442 pengujian unit lulus; seluruh angka pada dokumen ini dibaca langsung dari berkas keluaran |
| Tanggal dokumen | 18 Agustus 2026 |

---

## Daftar Isi

1. [Pendahuluan](#1-pendahuluan)
2. [Deskripsi Data Awal](#2-deskripsi-data-awal)
3. [Perumusan Variabel Target](#3-perumusan-variabel-target)
4. [Tahapan Prapemrosesan](#4-tahapan-prapemrosesan)
5. [Hasil Prapemrosesan](#5-hasil-prapemrosesan)
6. [Kamus Kolom Keluaran](#6-kamus-kolom-keluaran)
7. [Pengendalian Kebocoran Informasi](#7-pengendalian-kebocoran-informasi)
8. [Verifikasi dan Pengujian](#8-verifikasi-dan-pengujian)
9. [Keterbatasan Metodologis](#9-keterbatasan-metodologis)
10. [Glosarium dan Rujukan](#10-glosarium-dan-rujukan)

---

## 1. Pendahuluan

### 1.1 Ruang lingkup dokumen

Dokumen ini memaparkan secara sistematis seluruh prosedur prapemrosesan data (*data preprocessing*) yang diterapkan pada penelitian peramalan permintaan rantai pasok Kebuli Yaman, terhitung sejak berkas ekspor mentah sistem *point of sale* hingga tersusunnya matriks fitur dan vektor target yang siap dikonsumsi algoritma pembelajaran mesin.

Pemaparan disusun mengikuti alur baku pelaporan ilmiah: deskripsi data awal (Bab 2), perumusan variabel terikat (Bab 3), prosedur transformasi (Bab 4), karakteristik data hasil transformasi (Bab 5), dan kamus lengkap variabel keluaran (Bab 6). Tiga bab penutup memuat prosedur pengendalian validitas (Bab 7), skema verifikasi (Bab 8), serta keterbatasan yang melekat pada rancangan ini (Bab 9).

Dokumen ini bersifat metodologis dan ditujukan untuk dikutip pada bagian *Materials and Methods* naskah publikasi. Catatan rekayasa yang lebih rinci — termasuk riwayat pengambilan keputusan desain beserta alternatif yang ditolak — tersimpan terpisah pada `docs/dokumentasi-preprocessing-id.md` dan berkas spesifikasi di `docs/superpowers/specs/`.

### 1.2 Posisi prapemrosesan dalam alur penelitian

Penelitian ini membandingkan tiga keluarga algoritma peramalan — *Extreme Gradient Boosting* (XGBoost), *Random Forest*, dan *Long Short-Term Memory* (LSTM) — atas persoalan peramalan permintaan yang sama. Perbandingan antaralgoritma hanya sah apabila ketiganya dilatih dan dievaluasi atas himpunan observasi, definisi target, dan pembagian data yang identik. Konsekuensinya, prapemrosesan pada penelitian ini tidak diperlakukan sebagai pekerjaan persiapan yang bersifat teknis semata, melainkan sebagai instrumen pengendali validitas internal: seluruh keputusan transformasi dipusatkan pada satu tabel fitur bersama, dan perbedaan kebutuhan bentuk masukan antaralgoritma diakomodasi oleh lapisan *adapter* tipis yang keluarannya diverifikasi kesetaraannya secara terprogram (lihat Subbab 4.14).

Prapemrosesan juga dirancang untuk dapat dijalankan ulang secara berkala atas data periode baru, karena model dimaksudkan untuk dipakai langsung oleh tim rantai pasok dalam siklus operasional mingguan, bukan sekadar dilaporkan sebagai hasil eksperimen. Persyaratan ini melahirkan dua konsekuensi desain yang tampak berulang sepanjang dokumen: seluruh artefak pemetaan (pengodean kategori, parameter standardisasi) dipersistensi ke berkas agar penomorannya stabil lintas pemutakhiran, dan setiap keanehan data yang tidak terduga menyebabkan program berhenti dengan galat eksplisit alih-alih diperbaiki secara diam-diam.

### 1.3 Konvensi notasi

Notasi berikut dipakai konsisten sepanjang dokumen.

| Simbol | Makna |
|---|---|
| $i$ | Kode barang (*stock keeping unit*, SKU) |
| $b$ | Cabang atau gerai |
| $p = (i, b)$ | Pasangan item–cabang; satu deret waktu tunggal |
| $\mathcal{P}$ | Himpunan seluruh pasangan aktif, $\lvert\mathcal{P}\rvert = 2.979$ |
| $t$ | Tanggal kalender |
| $q_{p,t}$ | Kuantitas barang keluar untuk pasangan $p$ pada tanggal $t$ |
| $L_{p,t}$ | Lead time, yaitu jumlah hari dari $t$ hingga hari pengiriman berikutnya |
| $s_{p,t}$ | Nomor segmen aktif; membatasi jangkauan seluruh operasi pergeseran |
| $y_{p,t}$ | Variabel target utama |
| $T_{\text{uji}}$ | Batas periode uji, yaitu 1 Desember 2025 |

Notasi $(t{-}k \ldots t{-}1)$ menyatakan jendela tertutup yang berakhir tepat satu hari sebelum $t$; notasi $(t{+}1 \ldots t{+}k)$ menyatakan jendela maju ketat yang dimulai satu hari setelah $t$. Kedua bentuk ini secara sengaja tidak pernah menyertakan $t$ itu sendiri, dengan alasan yang diuraikan pada Bab 7.

---

## 2. Deskripsi Data Awal

### 2.1 Sumber dan cakupan data

Data primer penelitian ini berupa **log transaksi barang keluar** ("Barang Keluar") yang diekspor dari sistem persediaan pusat distribusi Kebuli Yaman. Setiap baris log merepresentasikan satu *line item*, yakni satu jenis barang yang diserahkan kepada satu cabang pada satu tanggal.

Data diterima dalam lima berkas Microsoft Excel (`dataset/excel/*.xlsx`) yang masing-masing telah dikonversi menjadi berkas CSV berpadanan satu-ke-satu (`dataset/csv/*.csv`). Kelima berkas **memartisi sumbu waktu, bukan kategori barang**, dan tidak saling bertumpang tindih:

| Berkas | Periode cakupan |
|---|---|
| `jan-24.csv` | Januari 2024 |
| `feb-24.csv` | Februari 2024 |
| `mar-24.csv` | Maret 2024 |
| `apr-des-24.csv` | April – Desember 2024 |
| `jan-des-25.csv` | Januari – Desember 2025 |

Gabungan kelimanya menghasilkan berkas `dataset/dataset.csv` yang memuat **693.563 baris transaksi**, mencakup **109 kode barang** dan **67 cabang** sepanjang 1 Januari 2024 hingga 31 Desember 2025.

Distribusi kuantitas pada data mentah bersifat menjulur kanan secara tajam: nilai minimum 1, kuartil pertama 2, median 5, kuartil ketiga 34, rerata 30,39, dan maksimum 5.250 unit. Perlu dicatat bahwa **tidak terdapat satu pun baris bernilai nol** pada data mentah — log hanya merekam penyerahan barang yang benar-benar terjadi. Ketiadaan hari tanpa transaksi inilah yang menuntut konstruksi panel padat pada Subbab 4.5, karena hari tanpa permintaan merupakan informasi yang esensial bagi peramalan namun tidak terwakili secara eksplisit pada data sumber.

### 2.2 Kamus kolom data mentah

Berkas sumber menggunakan pemisah titik koma (`;`), pengodean UTF-8 dengan penanda *byte order mark*, dan baris tajuk berbahasa Indonesia. Ketujuh kolomnya diuraikan berikut ini.

1. **`Tanggal`** — untai teks berformat `DD Mon YYYY` (contoh: `01 Jan 2024`). Kolom ini menyatakan **tanggal pengambilan barang**, yaitu hari ketika barang secara fisik diserahkan kepada cabang. Kolom ini **bukan** tanggal pesanan masuk; implikasi metodologis dari perbedaan ini diuraikan tersendiri pada Subbab 2.5 karena menentukan batas atas akurasi yang dapat dicapai model mana pun.

2. **`Kategori Barang`** — untai teks yang mengelompokkan barang menurut tahap pengolahannya. **Dua belas nilai berbeda** muncul pada data mentah, yang pada panel akhir menyusut menjadi delapan: `Bahan Baku (RM)` untuk bahan baku, `Barang Dalam Process (WIP-1)` dan `Barang Semi FG (WIP-2)` untuk barang setengah jadi pada dua tingkat pengolahan, `Barang Jadi (FG)` untuk barang jadi, `Minuman - FG` dan `Snack (FG)` untuk dua lini produk jadi khusus, serta `Packaging` dan `Barang Umum` untuk barang pendukung nonpangan.

   Penyusutan tersebut bersumber dari dua sebab yang berbeda. Dua nilai — `Minuman` dan `Snack` — merupakan sinonim penulisan yang disatukan pada kanonikalisasi kategori (Subbab 4.3). Dua nilai lainnya — `Perlengkapan Resto` (1 SKU, 6 baris) dan `Tambahan` (2 SKU, 82 baris) — lenyap bukan karena penggabungan melainkan karena seluruh SKU penyusunnya tersaring oleh ambang riwayat minimum pada Subbab 4.5, sebab jumlah harinya terlalu sedikit untuk membentuk deret waktu yang dapat diramalkan.

3. **`Kode Barang`** — untai teks pengenal SKU dengan pola awalan tiga huruf diikuti tanda hubung dan lima digit, contohnya `FGS-00001` untuk barang jadi dan `PCG-00006` untuk kemasan. Kolom ini merupakan salah satu dari dua komponen kunci unit analisis. Konsistensi penulisannya tidak terjaga pada data mentah dan memerlukan normalisasi tersendiri (Subbab 4.3).

4. **`Nama Barang`** — untai teks nama produk dalam bahasa Indonesia, contohnya `Iga Sapi Kebuli` atau `Cup Sambal Loyang`. Kolom ini tidak dipakai sebagai kunci karena tidak dijamin unik maupun stabil, tetapi berperan penting sebagai **kriteria pembanding pada prosedur penggabungan kode barang bersyarat** (Subbab 4.3) dan dipertahankan hingga keluaran akhir untuk keperluan interpretasi hasil.

5. **`Nama Cabang`** — untai teks pengenal gerai dengan format `KY0NN - Kebuli Yaman <lokasi>`, contohnya `KY007 - Kebuli Yaman Cibubur`. Kolom ini merupakan komponen kedua kunci unit analisis. Satu gerai fisik dapat muncul dengan lebih dari satu penulisan sepanjang periode data — terutama akibat relokasi yang disertai penerbitan kode baru — sehingga memerlukan kanonikalisasi (Subbab 4.4).

6. **`Satuan`** — untai teks satuan ukur, dengan dua belas nilai berbeda pada data mentah: `Kg`, `Potong`, `Porsi`, `Botol`, `PCS`, `Pack`, `Ekor`, `Galon`, `Gr`, `Cup`, `Roll`, dan `Bungkus`. Tiga nilai terakhir sangat jarang — `Bungkus` hanya muncul pada satu baris, `Galon` pada tiga baris, dan `Roll` pada sembilan baris — sehingga SKU penyandangnya tersaring ambang riwayat minimum dan panel akhir hanya memuat sembilan satuan. Kolom ini tidak dipakai sebagai variabel prediktor karena bersifat konstan untuk setiap kode barang, namun **wajib dipertahankan hingga keluaran akhir** sebab kuantitas tidak bermakna tanpanya: nilai 3 pada satuan `Porsi` dan 3 pada satuan `Kg` merujuk pada besaran fisik yang sama sekali berbeda.

7. **`Kuantitas`** — bilangan yang menyatakan banyaknya barang yang diserahkan. Kolom ini merupakan **variabel dasar yang seluruh target dan sebagian besar fitur diturunkan darinya**. Nilainya tidak selalu bilangan bulat: satuan `Potong` memuat 6.510 baris bernilai pecahan, sedangkan satuan `PCS` dan `Botol` tidak memuat satu pun. Ketidakseragaman ini kelak ditangani secara empiris, bukan melalui daftar satuan yang ditetapkan secara manual (Subbab 4.7).

### 2.3 Berkas konfigurasi pendamping

Selain log transaksi, prapemrosesan mengonsumsi lima berkas konfigurasi yang dipelihara secara manual oleh peneliti bersama pemilik data. Berkas-berkas ini memuat pengetahuan domain yang tidak terkandung dalam log transaksi dan tidak dapat diturunkan darinya.

1. **`dataset/outlets.csv`** (65 baris) — data induk gerai, memuat nama kanonik, alamat, kecamatan, kota, serta tiga penanda ketersediaan kanal penjualan daring (Shopee, GoFood, GrabFood). Berkas ini berfungsi sebagai **otoritas tunggal atas keberadaan cabang**: cabang yang tidak tercatat di sini dianggap tidak lagi beroperasi dan barisnya dikeluarkan dari analisis.

2. **`dataset/outlet_mapping.csv`** (65 baris) — memetakan nama gerai lama ke nama kanonik baru, sekaligus memuat dua atribut logistik yang tidak ada pada `outlets.csv`: `kawasan` (bernilai 1 atau 2) dan `hari_pengiriman` (contoh: `Selasa dan Jumat`). Kedua atribut inilah dasar perhitungan lead time.

3. **`dataset/outlet_name_overrides.csv`** (19 baris) — koreksi manual atas nama cabang yang ambigu serta nilai kota yang keliru pada data induk. Koreksi diselesaikan melalui berkas eksplisit dan bukan melalui pencocokan kemiripan untai teks (*fuzzy matching*), sebab pencocokan kemiripan dapat menghasilkan penggabungan yang salah tanpa memunculkan galat.

4. **`dataset/event_driven_items.csv`** (70 baris, satu per SKU) — penanda biner apakah permintaan suatu SKU digerakkan oleh pemesanan acara. Lima SKU bertanda benar. Prosedur penurunan penanda ini diuraikan pada Subbab 4.12.

5. **`dataset/outlet_closures.csv`** (7 interval) — periode ketika suatu gerai tidak beroperasi, baik karena penutupan sementara maupun masa serah terima relokasi. Setiap baris memuat nama gerai, tanggal tutup, tanggal buka, dan alasan beserta sumber konfirmasinya. Berkas ini merupakan **satu-satunya otoritas** yang menentukan hari mana yang diperlakukan sebagai periode tidak beroperasi; deteksi otomatis atas jeda transaksi hanya berfungsi memunculkan kandidat untuk dikonfirmasi, tidak pernah menetapkan penutupan secara mandiri.

### 2.4 Anomali struktural pada data mentah

Enam anomali teridentifikasi pada berkas sumber dan seluruhnya ditangani secara eksplisit oleh kode prapemrosesan. Anomali-anomali ini dilaporkan karena reproduksi penelitian ini atas data serupa akan menghadapi persoalan yang sama.

1. **Penanda *byte order mark* UTF-8.** Seluruh berkas diawali karakter tak tampak `U+FEFF`, sehingga nama kolom pertama akan terbaca sebagai `﻿Tanggal` apabila berkas dibaca dengan pengodean UTF-8 biasa. Penanganannya adalah pembacaan dengan pengodean `utf-8-sig`.

2. **Skema kolom tidak seragam.** Berkas `jan-des-25.csv` memuat sembilan medan per baris, dua lebih banyak daripada empat berkas lainnya. Kedua medan tambahan selalu kosong dan merupakan artefak ekspor. Kode pemroses memotong skema menjadi tujuh kolom, namun **memunculkan galat apabila medan kedelapan atau kesembilan ternyata berisi data**, agar kehilangan informasi tidak terjadi secara diam-diam.

3. **Baris kosong di akhir berkas.** Tiga berkas diakhiri baris berisi pemisah tanpa nilai (`;;;;;;`). Baris demikian dibuang, bukan diperlakukan sebagai galat, karena telah dipastikan merupakan artefak ekspor Excel.

4. **Awalan `xxx.` pada kode dan nama barang.** Sebagian entri pada `apr-des-24.csv` diawali `xxx.`, contohnya `xxx.FGS-00003` dengan nama `xxx.Iga Sapi Kebuli`. Pemeriksaan menunjukkan awalan ini merupakan penanda internal pemilik data, bukan pembeda SKU.

5. **Ketidakseragaman pemisah pada kode barang.** Kode barang yang secara semantik sama dapat tertulis dengan titik maupun tanda hubung, contohnya `FGS.00047` dan `FGS-00047`. Penyeragaman naif atas anomali ini justru berbahaya, sebagaimana diuraikan pada Subbab 4.3.

6. **Kuantitas bernilai negatif.** Ditemukan pada cabang KY011 tertanggal 29 Februari 2024 pada ekspor awal. Pemilik data kemudian menyediakan ekspor Februari 2024 yang telah dikoreksi, dan pada data yang dipakai penelitian ini tidak lagi terdapat baris bernilai negatif. Meskipun demikian, asersi penjaminan mutu yang memeriksa hal ini tetap dipertahankan (Bab 8).

### 2.5 Batasan mendasar: tanggal pengambilan versus tanggal pemesanan

Satu karakteristik data menuntut pernyataan eksplisit karena membatasi akurasi yang dapat dicapai model mana pun dalam penelitian ini, terlepas dari algoritma dan fitur yang dipakai.

Kolom `Tanggal` merekam **tanggal barang diambil**, bukan tanggal pesanan diterima (dikonfirmasi pemilik data, 15 Agustus 2026). Seorang pelanggan yang memesan pada hari Senin untuk pengambilan hari Kamis akan menghasilkan satu baris bertanggal **Kamis**. Manajer gerai meneruskan pesanan tersebut ke kantor pusat pada hari Senin, tetapi tidak ada catatan apa pun yang ditulis sampai barang benar-benar keluar. Sistem *point of sale* memang dirancang demikian secara sengaja: pesanan dapat dibatalkan, sehingga hanya transaksi yang terealisasi yang direkam.

Tiga konsekuensi metodologis mengalir dari fakta ini.

Pertama, seluruh deret waktu pada penelitian ini berjalan pada **sumbu waktu pengambilan**. Fitur dan target konsisten satu sama lain pada sumbu tersebut, sehingga tidak terjadi ketidakselarasan internal. Namun keduanya mengukur permintaan yang *terealisasi*, bukan saat permintaan itu timbul.

Kedua, terdapat asimetri informasi yang tidak dapat dijembatani. Kantor pusat sudah mengetahui sebagian permintaan beberapa hari ke depan melalui pesanan yang diteruskan manajer gerai, sedangkan informasi tersebut tidak terekam pada satu pun berkas dalam penelitian ini. Model dengan demikian bersaing dengan pengetahuan yang secara struktural lebih baik daripada yang tersedia baginya.

Ketiga, dan inilah yang menyelaraskan penelitian dengan kebutuhan operasional: logika bisnis yang berlaku menempatkan model sebagai peramal permintaan **di luar pesanan awal**, karena komponen pesanan ditangani tim secara terpisah. Kuantitas pada data mentah mencampur kedua komponen tanpa penanda apa pun, sehingga pemisahannya tidak dapat dilakukan secara pasti. Nilai yang telah dipangkas pada penanganan pencilan (Subbab 4.7) merupakan pendekatan terdekat yang tersedia bagi komponen pesanan, dan inilah alasan disediakannya sepasang target — satu atas kuantitas mentah, satu atas kuantitas terpangkas (Subbab 3.3).

Uraian lebih lengkap atas batasan ini tersedia pada `docs/batasan-penelitian.md` butir B-1 hingga B-3.

---

## 3. Perumusan Variabel Target

### 3.1 Unit analisis

Unit analisis penelitian ini adalah **pasangan kode barang dan cabang, diamati per hari kalender**. Setiap pasangan $p = (i, b)$ diperlakukan sebagai satu deret waktu tersendiri, dan setiap baris data merepresentasikan satu titik waktu $t$ pada deret tersebut.

Pemilihan granularitas ini mengikuti kebutuhan operasional. Keputusan yang harus diambil tim rantai pasok bersifat spesifik hingga tingkat barang dan gerai — berapa banyak satu jenis barang harus dikirim ke satu gerai tertentu — sehingga agregasi ke tingkat cabang atau ke frekuensi mingguan akan menghasilkan angka yang tidak dapat langsung ditindaklanjuti. Konsekuensinya adalah data menjadi jauh lebih jarang (*sparse*): 54,15% baris panel bernilai nol, dan 75,43% pasangan tergolong *intermittent* atau *lumpy* menurut klasifikasi Syntetos-Boylan (Subbab 5.4).

Setelah seluruh prosedur penyaringan diterapkan, terdapat **2.979 pasangan aktif** yang terbentuk dari 70 kode barang dan 59 cabang. Jumlah ini lebih kecil daripada hasil perkalian kartesian keduanya (4.130), sebab tidak setiap cabang menjual setiap barang.

### 3.2 Definisi formal target utama

Angka yang benar-benar dibutuhkan tim rantai pasok bukanlah permintaan pada satu hari tertentu, melainkan **total permintaan sepanjang selang waktu hingga kiriman berikutnya tiba**. Angka inilah yang dituliskan pada surat jalan. Target utama penelitian ini dirumuskan mengikuti definisi operasional tersebut.

Pengiriman dari pusat distribusi berlangsung dua kali sepekan menurut jadwal tetap per kawasan:

| Kawasan | Hari pengiriman |
|---|---|
| Kawasan 1 | Senin dan Kamis |
| Kawasan 2 | Selasa dan Jumat |

Lead time $L_{p,t}$ didefinisikan sebagai jarak hari dari tanggal $t$ menuju hari pengiriman berikutnya bagi kawasan tempat cabang $b$ berada:

$$L_{p,t} = \min \{ d \in \{1, 2, \ldots, 7\} \;:\; (\text{dow}(t) + d) \bmod 7 \in D_b \}$$

dengan $\text{dow}(t)$ menyatakan indeks hari dalam pekan untuk tanggal $t$ dan $D_b$ menyatakan himpunan hari pengiriman kawasan cabang $b$. Perumusan ini bersifat **maju ketat**: apabila $t$ kebetulan jatuh tepat pada hari pengiriman, nilai yang dihasilkan adalah jarak menuju kejadian *berikutnya*, bukan nol. Sebagai ilustrasi, untuk cabang di Kawasan 1, transaksi hari Senin menghasilkan $L = 3$ (menuju Kamis) sedangkan transaksi hari Kamis menghasilkan $L = 4$ (menuju Senin pekan berikutnya).

Target utama, dinamai `target_lead_time_cumulative`, adalah jumlah kuantitas sepanjang jendela maju ketat sepanjang lead time:

$$y_{p,t} = \sum_{k=1}^{L_{p,t}} q_{p,\,t+k}$$

Tiga sifat perumusan ini perlu digarisbawahi. Pertama, **panjang jendela penjumlahan bervariasi antarbaris**, bergantung pada hari dalam pekan dan kawasan cabang. Pada data penelitian, $L$ mengambil empat nilai dengan distribusi yang relatif berimbang untuk $L \in \{1,2,3\}$ (masing-masing sekitar 429 ribu baris) dan lebih jarang untuk $L = 4$ (214 ribu baris). Kedua, jendela **tidak pernah menyertakan hari $t$ itu sendiri**, sehingga target selalu merupakan besaran yang belum terobservasi pada saat prediksi dilakukan. Ketiga, penjumlahan dibatasi dalam satu segmen aktif yang sama (Subbab 4.5), sehingga tidak ada target yang menjembatani periode ketika gerai tidak beroperasi.

Baris yang jendela targetnya melampaui tanggal terakhir data memperoleh nilai kosong, bukan jumlah parsial. Pada keseluruhan data hal ini terjadi pada 0,498% baris.

### 3.3 Variabel target pendamping

Selain target utama, tersedia sembilan variabel target lain yang seluruhnya diturunkan dari kuantitas yang sama.

**Target harian `target_h1` hingga `target_h7`** merupakan kuantitas pada $h$ hari ke depan, yakni $q_{p,\,t+h}$ untuk $h = 1, \ldots, 7$. Fungsinya bersifat penunjang: karena $L$ tidak pernah melebihi 4 pada data penelitian, horizon `target_h5` hingga `target_h7` tidak memiliki padanan operasional langsung. Ketujuhnya tetap dipertahankan agar prediksi kumulatif dapat didekomposisi menjadi kontribusi harian, yang diperlukan untuk menjelaskan hasil peramalan kepada pihak yang akan menindaklanjutinya.

**Target atas kuantitas terpangkas `target_lead_time_cumulative_capped`** dirumuskan identik dengan target utama, hanya saja penjumlahan dilakukan atas `Kuantitas_capped`, yakni kuantitas yang lonjakan ekstremnya telah dipangkas (Subbab 4.7). Keberadaan sepasang target ini merupakan konsekuensi langsung dari batasan pada Subbab 2.5: karena kuantitas mentah mencampur permintaan reguler dengan komponen pesanan tanpa penanda, sementara baris yang terpangkas merupakan pendekatan terdekat bagi komponen pesanan tersebut, maka pilihan target menentukan pertanyaan penelitian yang sesungguhnya dijawab. Perbedaannya tidak dapat diabaikan: baris terpangkas hanya mencakup 2,41% jendela uji Desember, tetapi menyumbang 11,5% dari total galat absolut.

### 3.4 Justifikasi pemilihan target

Alternatif yang lebih lazim dalam literatur peramalan permintaan adalah memprediksi permintaan satu hari ke depan, yakni `target_h1`, kemudian menjumlahkan prediksi harian untuk memperoleh angka kumulatif. Alternatif ini ditolak dengan dua pertimbangan.

Secara operasional, penjumlahan prediksi harian mengakumulasi galat setiap horizon, sedangkan besaran yang benar-benar menentukan keberhasilan pengiriman adalah jumlahnya, bukan komponennya. Melatih model langsung atas besaran yang dievaluasi menghilangkan lapisan akumulasi galat tersebut.

Secara statistik, penjumlahan atas jendela beberapa hari meredam kejarangan data. Permintaan harian pada tingkat item–cabang bernilai nol pada 54,15% baris, sedangkan target kumulatif memiliki median 1 dan kuartil ketiga 8 — distribusi yang masih menjulur tajam namun secara substansial lebih informatif untuk dipelajari model.

Konsekuensi yang diterima adalah kompleksitas implementasi: karena panjang jendela bervariasi per baris, penjumlahan maju tidak dapat dihitung dengan satu operasi *rolling* tunggal, melainkan memerlukan prosedur khusus yang diuraikan pada Subbab 4.8.

---

## 4. Tahapan Prapemrosesan

Prapemrosesan tersusun atas empat belas tahap berurutan. Tahap 1 hingga 12 menghasilkan tabel fitur beserta pembagian latih–uji, sedangkan tahap 13 dan 14 menyiapkan bentuk masukan spesifik bagi masing-masing keluarga algoritma. Setiap subbab berikut memaparkan tujuan, prosedur, keluaran, dan justifikasi metodologis tahap bersangkutan.

Gambaran ringkas alur keseluruhan disajikan berikut ini.

```
Berkas ekspor .xlsx / .csv
  │
  ├─  1  merge_dataset.py         Penggabungan lima periode
  ├─  2  aggregate_dataset.py     Agregasi baris duplikat
  ├─  3  normalize_items.py       Normalisasi kode dan satuan barang
  ├─  4  outlet_features.py       Penyaringan dan kanonikalisasi cabang
  ├─  5  build_panel.py           Panel harian padat dan segmentasi
  ├─  6  calendar_features.py     Fitur kalender dan hari besar
  ├─  7  outlier_handling.py      Deteksi dan pemangkasan lonjakan
  ├─  8  prepare_forecast_data.py Rekayasa fitur dan target
  ├─  9  export_featured          → featured.parquet (68 kolom)
  ├─ 10  split_train_test         Pembagian temporal + purging
  ├─ 11  export_splits            → train.parquet / test.parquet
  ├─ 12  run_qa_checks            Sebelas asersi penjaminan mutu
  │
  ├─ 13  modeling_prep.py         Penanda acara → segmentasi permintaan →
  │                               fold → imputasi → pengodean kategori
  │                               → model_input.parquet (82 kolom)
  └─ 14  adapter                  to_tabular()   → XGBoost, Random Forest
                                  to_sequences() → LSTM
                                  validate_contract() mengikat keduanya
```

### 4.1 Tahap 1 — Penggabungan periode

**Tujuan.** Menyatukan lima berkas ekspor yang memartisi sumbu waktu menjadi satu berkas tunggal berurutan kronologis.

**Prosedur.** Kelima berkas dibaca berurutan menurut periodenya, skema dipotong menjadi tujuh kolom, kolom `Tanggal` diuraikan dengan format `%d %b %Y` semata-mata untuk keperluan pengurutan, lalu hasilnya ditulis ke `dataset/dataset.csv` dengan format tanggal dan pemisah asli tetap dipertahankan.

**Keluaran.** Berkas `dataset/dataset.csv` berisi 693.563 baris.

**Justifikasi.** Tahap ini menerapkan prinsip **kegagalan eksplisit** yang berlaku sepanjang pipeline: program memunculkan galat dan berhenti apabila medan kedelapan atau kesembilan ternyata tidak kosong, atau apabila suatu tanggal gagal diuraikan. Pemilihan ini disengaja — data yang rusak harus terlihat, bukan lenyap tanpa jejak. Format asli dipertahankan pada keluaran agar pemilik data tetap dapat membuka berkas hasil gabungan sebagaimana berkas aslinya, sehingga verifikasi silang tetap dimungkinkan.

### 4.2 Tahap 2 — Agregasi baris duplikat

**Tujuan.** Menyatukan baris yang identik pada seluruh atribut kunci, sebab satu transaksi kadang terekam sebagai beberapa *line item* terpisah.

**Prosedur.** Kuantitas dijumlahkan atas pengelompokan menurut tanggal, kategori barang, kode barang, nama barang, cabang, dan satuan.

**Justifikasi.** Tanpa agregasi ini, konstruksi panel pada Tahap 5 akan menghadapi kunci ganda dan asersi ketunggalan baris per pasangan per tanggal akan gagal. Agregasi berupa penjumlahan, bukan pengambilan nilai pertama, karena kedua baris merepresentasikan penyerahan barang yang sama-sama terjadi.

### 4.3 Tahap 3 — Normalisasi kode barang

**Tujuan.** Menyatukan kode barang yang merujuk pada produk yang sama namun tertulis berbeda, tanpa secara keliru menyatukan produk yang sesungguhnya berbeda.

**Prosedur.** Diterapkan empat operasi berurutan.

*Pertama, normalisasi bentuk penulisan.* Awalan `xxx.` dibuang dan pemisah antara bagian huruf dan bagian angka diseragamkan menjadi tanda hubung, sehingga `FGS.00047` menjadi `FGS-00047`.

*Kedua, penggabungan bersyarat.* Dua kode hanya digabungkan apabila **nama barangnya juga bersesuaian** setelah normalisasi ringan berupa pembuangan awalan `xxx.`, perapian spasi, dan pembuangan anotasi dalam tanda kurung. Syarat tambahan ini merupakan keputusan metodologis terpenting pada tahap ini, dan diturunkan dari pemeriksaan data empiris, bukan dari asumsi. Penyeragaman pemisah tanpa syarat nama akan menabrakkan **lima pasang produk yang sama sekali berbeda** yang kebetulan berbagi digit — sebagai contoh, `FGS-00047` adalah *Kentang Mustofa Rumput Laut* dengan satuan `Pack`, sedangkan `FGS.00047` adalah *Air Isi Ulang* dengan satuan `Galon`. Adapun pembuangan awalan `xxx.` menghasilkan empat kelompok dengan nama tak bersesuaian, tiga di antaranya bersifat kosmetik semata (`250ml` terhadap `250 ml`), sedangkan satu kelompok (`Cendol Pandan - FG` terhadap `Cendol - FG`) tidak digabungkan karena perbedaan varian rasa tidak dapat disingkirkan hanya berdasarkan data.

*Ketiga, konversi satuan.* Dua item berawalan `xxx.` — Santan Cendol (`xxx.FGS.00070`) dan Gula Cendol (`xxx.FGS.00071`) — tercatat dalam satuan gram pada periode awal, sedangkan periode berikutnya mencatatnya dalam porsi. Faktor konversi 40 dan 30 gram per porsi tidak ditebak, melainkan diturunkan dari bukti empiris bahwa **setiap nilai mentah pada periode gram merupakan kelipatan bulat** dari faktor tersebut. Setelah konversi, deret keduanya menyambung mulus dengan periode berdenominasi porsi.

*Keempat, kanonikalisasi kategori dan pengecualian.* Sinonim kategori disatukan (`Minuman` menjadi `Minuman - FG`, `Snack` menjadi `Snack (FG)`), dengan satu pengecualian eksplisit untuk `FGS-00014` (Club Mineral 600 ml) yang semula tercatat sebagai barang setengah jadi namun sesungguhnya merupakan minuman. Empat item yang telah dihentikan penjualannya (`xxx.FGS.00066` hingga `xxx.FGS.00069`) dikeluarkan. Satu cabang dikeluarkan pula, yaitu `Kebab Saudagar - Kutabumi`, yang merupakan merek berbeda dan hanya menerbitkan 137 baris pada 20–31 Desember 2025.

Setelah keseluruhan operasi, dilakukan **agregasi ulang** pada tingkat kode ternormalisasi, sebab normalisasi dapat menjadikan baris yang semula berbeda memiliki kunci yang sama.

**Justifikasi.** Pilihan untuk menggabungkan hanya bila nama bersesuaian berimplikasi sebagian deret tetap terpecah meskipun mungkin sesungguhnya merupakan satu produk. Pilihan ini diambil karena **kedua jenis kesalahan tidak simetris**: penggabungan yang salah merusak label dan menghasilkan deret yang secara fisik tidak bermakna, sedangkan pemisahan yang salah hanya mengurangi panjang riwayat yang tersedia.

### 4.4 Tahap 4 — Penyaringan dan kanonikalisasi cabang

**Tujuan.** Memastikan setiap cabang pada data memiliki identitas tunggal yang konsisten sepanjang periode pengamatan, dan mengeluarkan cabang yang tidak lagi beroperasi.

**Prosedur.** Baris yang cabangnya tidak terdaftar pada `outlets.csv` dibuang; hal ini mencakup sepuluh cabang atau sekitar 10% baris. Selanjutnya setiap nilai `Nama Cabang` ditulis ulang menjadi nama kanonik gerainya, sehingga gerai yang terekam dengan dua penulisan berbeda — misalnya karena penerbitan kode baru pascarelokasi — menyatu menjadi satu riwayat berkelanjutan. Agregasi harian dijalankan sekali lagi untuk menjumlahkan baris yang bertabrakan akibat penggantian nama tersebut.

**Justifikasi.** Penetapan `outlets.csv` sebagai otoritas keberadaan cabang membawa konsekuensi bahwa "cabang yang telah tutup" tidak dapat dibedakan dari "cabang baru yang belum terdaftar". Risiko ini dimitigasi dengan mencetak daftar cabang yang dibuang pada setiap eksekusi, sehingga anomali segera terlihat oleh peneliti.

Dua keputusan menyangkut variabel lokasi perlu dicatat. Awalan `Kota ` dan `Kabupaten ` pada nilai kota **dipertahankan** dan tidak dipangkas, sebab kota dan kabupaten memiliki pola permintaan yang berbeda. Sebaliknya, kolom `Kecamatan` **tidak** dipakai sebagai variabel prediktor: dengan 56 nilai unik untuk 62 gerai, variabel ini nyaris identik dengan identitas gerai itu sendiri dan terlalu jarang untuk digeneralisasi oleh model global.

### 4.5 Tahap 5 — Konstruksi panel harian padat dan segmentasi

**Tujuan.** Mengubah log transaksi menjadi panel seimbang dengan satu baris per pasangan per hari kalender, sekaligus memastikan tidak ada operasi berbasis waktu yang menjembatani periode ketika gerai tidak beroperasi.

**Prosedur.** Setiap pasangan diindeks ulang menjadi satu baris per hari kalender sepanjang rentang aktifnya sendiri. Hari tanpa transaksi diisi kuantitas nol, sedangkan kolom deskriptif (`Kategori Barang`, `Nama Barang`, `Satuan`) diisi dengan propagasi maju kemudian mundur.

Dua mekanisme kemudian membatasi kepadatan tersebut.

*Penghapusan hari tidak beroperasi.* Hari yang jatuh di dalam interval penutupan yang tercatat pada `outlet_closures.csv` **tidak menghasilkan baris sama sekali**. Pengisian nol pada hari-hari tersebut akan memfabrikasi riwayat permintaan bernilai nol untuk gerai yang sesungguhnya tidak ada, yang selanjutnya akan mencemari seluruh statistik cabang.

*Penomoran segmen.* Setiap rangkaian tanggal aktif yang berurutan diberi nomor pada kolom `segment_id`. Segmen baru dimulai pada dua keadaan: ketika dua tanggal yang dipertahankan berjarak lebih dari satu hari — yang persis menandai tempat interval penutupan dihapus — dan pada tanggal *breakpoint*, yaitu relokasi yang terjadi di dalam rentang data. Pada kasus relokasi, gerai tidak pernah berhenti beroperasi sehingga tidak ada baris yang dibuang, namun tingkat permintaannya bergeser tajam: pengukuran atas tiga relokasi yang memiliki cukup data pascapindah menunjukkan rasio 2,2 hingga 2,6 kali.

Seluruh operasi bergeser pada tahap berikutnya — nilai lag, jendela *rolling*, pergeseran target, maupun jendela masukan LSTM — dikelompokkan menurut (pasangan, segmen), bukan menurut pasangan saja.

*Penyaringan riwayat minimum.* Pasangan yang memiliki kurang dari 60 hari data sebelum batas periode uji dikeluarkan seluruhnya. Ambang ini memastikan setiap pasangan yang dilatih memiliki riwayat memadai untuk membentuk fitur lag terpanjang (28 hari) beserta margin yang wajar.

**Keluaran.** Pada data penelitian, terbentuk 230 pasangan yang memiliki lebih dari satu segmen, dengan nomor segmen maksimum 3.

**Justifikasi.** Alternatif yang ditolak adalah menandai baris masa tutup dengan penanda biner lalu membuangnya belakangan. Alternatif tersebut tidak menyelesaikan persoalan inti, sebab fitur lag dan *rolling* akan telanjur dihitung melintasi periode tutup, sehingga model menerima masukan yang menyatakan permintaan nol selama berbulan-bulan padahal gerai memang tidak ada. Segmentasi menyelesaikannya pada akarnya, dengan konsekuensi berupa satu kolom tambahan dan kebutuhan pemanasan ulang fitur lag setelah gerai kembali beroperasi.

### 4.6 Tahap 6 — Fitur kalender

**Tujuan.** Menyediakan variabel yang menangkap musiman mingguan, bulanan, serta pengaruh hari besar keagamaan dan nasional.

**Prosedur.** Dua puluh kolom kalender dibangkitkan dari kolom `Tanggal` semata. Kelompok pertama bersifat deterministik penanggalan: hari dalam pekan, tanggal dalam bulan, bulan, dan penanda akhir pekan. Kelompok kedua menandai hari libur nasional Indonesia menggunakan pustaka `holidays`. Kelompok ketiga menangani empat peristiwa besar — Ramadan, Idulfitri, Iduladha, Hari Kemerdekaan, dan Tahun Baru — masing-masing dengan penanda biner disertai variabel jarak hari sebelum dan sesudah peristiwa.

Tanggal Ramadan, Idulfitri, dan Iduladha untuk tahun 2024 dan 2025 ditetapkan secara eksplisit di dalam kode dan telah diverifikasi silang terhadap keluaran pustaka `holidays` versi 0.83 serta terhadap penetapan pemerintah Indonesia.

**Justifikasi.** Variabel jarak hari dibatasi jendela ±14 hari untuk seluruh peristiwa kecuali Ramadan; di luar jendela tersebut nilainya kosong. Pembatasan ini disengaja: jarak 200 hari menuju Idulfitri tidak membawa informasi perilaku apa pun, sehingga membiarkannya bernilai numerik hanya menambah derau. Perlakuan atas nilai kosong yang dihasilkannya diuraikan pada Subbab 4.12.

Fungsi pemeriksaan cakupan tahun **memunculkan galat** apabila data memuat tahun yang tabel tanggalnya belum tersedia. Perilaku ini disengaja: memproses data tahun 2026 dengan tabel yang hanya mencakup 2024–2025 akan menghasilkan seluruh penanda Ramadan bernilai salah tanpa satu pun pesan galat.

### 4.7 Tahap 7 — Deteksi dan pemangkasan pencilan

**Tujuan.** Meredam pengaruh lonjakan permintaan ekstrem terhadap fitur riwayat, tanpa menghapus informasi bahwa lonjakan tersebut terjadi.

**Prosedur.** Prosedur terdiri atas tiga keputusan.

*Penetapan garis dasar per pasangan.* Bagi setiap pasangan dihitung median kuantitas atas hari-hari bernilai positif **dalam periode latih saja**. Pasangan dinyatakan layak dipangkas apabila memiliki sekurang-kurangnya 30 hari transaksi positif dan mediannya lebih besar dari nol. Hari bernilai nol dikecualikan dari perhitungan karena seluruhnya merupakan hari hasil pengisian panel, bukan transaksi yang sesungguhnya terjadi — sebagaimana telah dinyatakan pada Subbab 2.1, data mentah tidak pernah memuat nilai nol.

*Penetapan ambang lonjakan.* Suatu baris ditandai sebagai lonjakan apabila rasionya terhadap median pasangan mencapai 5,0 atau lebih. Rasio ini disimpan pada kolom `baseline_ratio` dan penandanya pada `is_spike`.

*Pemangkasan bersyarat.* Nilai dipangkas ke ambang, yakni median dikalikan 5,0, **hanya apabila baris tersebut tidak jatuh pada jendela peristiwa besar** (Ramadan, Idulfitri, Iduladha, Hari Kemerdekaan, atau Tahun Baru). Lonjakan pada masa Lebaran merupakan pola musiman yang justru harus dipelajari model, bukan pencilan yang harus diredam.

Bagi pasangan yang sepanjang riwayatnya hanya pernah bertransaksi dalam satuan bulat, nilai ambang dibulatkan **ke atas**, bukan ke nilai terdekat. Sifat "hanya bernilai bulat" ditentukan dari riwayat pasangan itu sendiri dan bukan dari nama satuannya, sebab satuan pada data ini tidak terbagi bersih menjadi diskret dan kontinu — satuan `Potong` memuat 6.510 baris pecahan sementara `PCS` dan `Botol` tidak memuat satu pun. Pembulatan ke atas dipilih karena kriteria keberhasilan operasional adalah gerai tidak kehabisan stok, sehingga nilai imbang diputuskan ke arah persediaan yang lebih banyak.

**Keluaran.** Pada data penelitian, 0,903% baris ditandai sebagai lonjakan. Kolom `baseline_ratio` memiliki median 0,333 dan nilai maksimum 200, serta kosong pada 14,391% baris yang pasangannya tidak memenuhi syarat kelayakan.

**Justifikasi.** Pemangkasan **tidak** diterapkan pada nilai yang dipakai membentuk `target_h1` hingga `target_h7` maupun `target_lead_time_cumulative`, sebab target harus merepresentasikan permintaan yang sesungguhnya terjadi. Nilai terpangkas hanya dipakai membentuk fitur riwayat — lag, *rolling*, statistik cabang — serta target tandingan `target_lead_time_cumulative_capped`. Pemisahan ini memastikan lonjakan tidak mencemari fitur namun tetap terevaluasi sebagai bagian dari kinerja model.

### 4.8 Tahap 8 — Rekayasa fitur

**Tujuan.** Membangkitkan seluruh variabel prediktor dan variabel target dari panel tersegmentasi yang telah dilengkapi fitur kalender.

**Prosedur.** Urutan langkah didefinisikan pada satu fungsi tunggal `engineer_features()`, dengan susunan sebagai berikut.

| Urutan | Fungsi | Keluaran |
|---|---|---|
| 1 | `add_targets` | `target_h1` … `target_h7` |
| 2 | `apply_region_features` | `kawasan`, `hari_pengiriman`, `lead_time_days` |
| 3 | `apply_outlet_features` | `kota`, `has_shopee`, `has_gofood`, `has_grabfood`, `can_order_online` |
| 4 | `add_relocation_feature` | `days_since_relocation` |
| 5 | `add_delivery_day_flag` | `is_delivery_day` |
| 6 | `add_target_window_weekend_days` | `target_window_weekend_days` |
| 7 | `add_lead_time_target` | `target_lead_time_cumulative` |
| 8 | `add_lead_time_target` (varian terpangkas) | `target_lead_time_cumulative_capped` |
| 9 | `add_lag_features` | `lag_1` … `lag_28` |
| 10 | `add_rolling_features` | rerata dan simpangan baku *rolling* 7/14/28 hari |
| 11 | `compute_branch_stats`, `apply_branch_stats`, `add_branch_age_days` | empat statistik cabang |

Dua aspek implementasi memerlukan penjelasan tersendiri.

*Perhitungan lead time.* Nilai `lead_time_days` bergantung semata-mata pada pasangan (hari dalam pekan, jadwal pengiriman), yang jumlah kombinasi berbedanya hanya sedikit. Perhitungan karenanya dilakukan sekali atas tabel kombinasi unik lalu digabungkan kembali ke data utama, bukan diterapkan baris demi baris atas 1,5 juta baris. Fungsi pengurai jadwal pengiriman **memunculkan galat** untuk token hari yang tidak dikenali, agar perubahan format atau salah ketik pada berkas konfigurasi segera terlihat alih-alih merusak nilai lead time secara diam-diam.

*Perhitungan target berjendela variabel.* Karena panjang jendela penjumlahan berbeda antarbaris, target tidak dapat dihitung dengan satu operasi *rolling*. Prosedur yang dipakai adalah: untuk setiap nilai $w$ yang muncul pada `lead_time_days`, dihitung satu kolom jumlah-maju melalui pembalikan urutan baris, penerapan `rolling(w).sum()`, lalu pembalikan kembali; selanjutnya nilai yang tepat dipilih per baris berdasarkan lead time-nya. Pendekatan ini menghindari perulangan Python atas 1,5 juta baris sekaligus tetap mendukung jendela berukuran variabel.

**Justifikasi penempatan urutan.** Urutan langkah tidak dapat dipertukarkan secara bebas. `apply_region_features` harus mendahului `add_lead_time_target` karena menyediakan `lead_time_days` yang menjadi panjang jendela target; ia juga harus mendahului `add_target_window_weekend_days` dengan alasan yang sama. Statistik cabang dihitung paling akhir karena mengonsumsi kuantitas terpangkas yang baru tersedia setelah Tahap 7.

Definisi urutan ini sengaja dipusatkan pada satu fungsi karena pengalaman menunjukkan bahaya penggandaannya. Pada Agustus 2026 ditemukan ketidaksesuaian jumlah kolom antara berkas hasil skrip dan berkas hasil *notebook*; penyebabnya adalah kedua jalur kode sama-sama menuliskan urutan langkah secara terpisah, dan *notebook* tidak pernah diperbarui ketika satu fungsi baru ditambahkan ke skrip. Sejak itu berlaku ketentuan bahwa hanya ada satu tempat yang mendefinisikan urutan langkah, dan *notebook* memanggil fungsi komposit tersebut alih-alih menuliskan ulang langkah-langkahnya.

### 4.9 Tahap 9 — Ekspor tabel fitur

Tabel hasil rekayasa fitur diekspor ke `dataset/model_ready/featured.parquet` dengan 1.502.522 baris dan 68 kolom. Sebelum ekspor, program memverifikasi bahwa seluruh kolom yang terdaftar pada konstanta `FEATURED_COLUMNS` benar-benar hadir.

Format Parquet dipilih menggantikan CSV atas dua pertimbangan: volume data 1,5 juta baris menjadikan CSV tidak efisien baik dari sisi ukuran berkas maupun waktu baca, dan Parquet mempertahankan tipe data sehingga kolom bertipe *boolean*, kategorik, dan *timestamp* tidak perlu diuraikan ulang pada setiap pembacaan. Konsekuensi yang diterima adalah berkas tidak dapat dibuka langsung dengan perangkat lunak lembar sebar.

Tabel ini disimpan sebagai artefak antara yang berdiri sendiri, terpisah dari berkas hasil pembagian latih–uji, karena keduanya memiliki daur hidup berbeda: tabel fitur menyatakan fakta tentang data dan berguna bagi analisis apa pun, sedangkan pembagian latih–uji merupakan keputusan eksperimen.

### 4.10 Tahap 10 — Pembagian latih–uji dan *purging*

**Tujuan.** Membagi data menjadi himpunan latih dan himpunan uji secara temporal, tanpa membiarkan label periode uji merembes ke himpunan latih.

**Prosedur.** Batas pembagian ditetapkan pada 1 Desember 2025. Himpunan latih memuat seluruh baris bertanggal sebelum batas tersebut, sedangkan himpunan uji memuat baris bulan Desember 2025.

Pembagian menurut tanggal saja ternyata belum memadai. Karena target menjumlahkan permintaan sepanjang jendela maju, baris yang bertanggal beberapa hari sebelum batas membawa label yang sebagiannya tersusun dari permintaan periode uji. Melatih model atas baris-baris tersebut berarti melatihnya atas label yang dibangun dari bulan yang justru hendak dievaluasi. Prosedur ***purging*** karenanya diterapkan: baris latih yang keseluruhan jendela targetnya tidak berakhir sebelum batas dibuang dari himpunan latih. Karena lead time tidak pernah melebihi empat hari, prosedur ini hanya menyingkirkan 0,2% hingga 0,8% baris latih pada setiap batas — kecil dari sisi volume, namun menentukan bagi klaim bahwa himpunan uji benar-benar terkunci.

Himpunan uji tidak pernah dikenai *purging*, sebab baris-baris tersebut merupakan objek evaluasi, bukan data latih.

**Keluaran.** Himpunan latih memuat 1.441.159 baris; himpunan uji memuat 55.046 baris. Keduanya memiliki 68 kolom yang sama.

**Justifikasi.** Pembagian dilakukan menurut waktu dan bukan secara acak. Pembagian acak atas data deret waktu akan membocorkan informasi masa depan secara masif, sebab baris tanggal 15 Desember dapat masuk himpunan latih sementara tanggal 14 Desember masuk himpunan uji. Konsekuensi yang diterima adalah periode uji hanya sepanjang satu bulan, yang dimitigasi melalui validasi *walk-forward* lima *fold* pada Tahap 13.

Baris yang jendela targetnya melampaui 31 Desember 2025 dibiarkan bernilai kosong, bukan dipersempit jendelanya. Pilihan ini mempertahankan seluruh 55.046 baris uji untuk horizon pendek, dengan konsekuensi cakupan yang lebih sedikit bagi horizon panjang pada penghujung bulan.

### 4.11 Tahap 11–12 — Ekspor pembagian dan penjaminan mutu

Kedua himpunan diekspor ke `train.parquet` dan `test.parquet`, kemudian dijalankan fungsi `run_qa_checks()` yang memuat sebelas asersi kualitas data. Asersi ini dipanggil baik dari jalur skrip maupun jalur *notebook*, sehingga kedua jalur terverifikasi setara. Rinciannya disajikan pada Bab 8.

### 4.12 Tahap 13 — Prapemrosesan pemodelan

**Tujuan.** Mengubah tabel fitur menjadi satu sumber kebenaran tunggal yang dikonsumsi ketiga keluarga algoritma, dengan menambahkan variabel yang mengodekan keputusan eksperimen.

Tahap ini menghasilkan berkas terpisah `model_input.parquet` dan bukan menambahkan kolom pada `featured.parquet`. Pemisahan ini beralasan: kedua berkas memiliki daur hidup berbeda. Tabel fitur menyatakan fakta tentang data, sedangkan berkas masukan model mengodekan keputusan eksperimen — batas *fold*, skema pengodean, definisi segmen permintaan — yang akan berubah berkali-kali sepanjang eksperimen. Dengan pemisahan ini, pipeline prapemrosesan data yang telah stabil tidak perlu dijalankan ulang setiap kali strategi validasi berubah.

Lima prosedur dijalankan berurutan.

**(a) Penanda barang digerakkan acara.** Kolom `is_event_driven` dilekatkan dari `event_driven_items.csv`. Program **memunculkan galat** apabila terdapat SKU pada data yang tidak memiliki entri pada berkas tersebut, sebab SKU baru yang muncul pada pemutakhiran berkala harus diklasifikasikan oleh pemilik data, bukan diasumsikan bukan-acara secara diam-diam.

Penurunan penanda ini dilakukan **dari bentuk permintaan, bukan dari nama barang**, sebab nama terbukti tidak andal pada kedua arah. Tanda tangan statistik pemesanan acara adalah *jarang namun borongan*: interval permintaan rerata sekurang-kurangnya 50 hari disertai rerata sekurang-kurangnya 30 unit pada hari barang bergerak. Barang lambat biasa memenuhi kriteria pertama namun tidak kriteria kedua, sebab hanya bergerak satu hingga dua unit.

Uji penentu yang akhirnya dipakai adalah **ko-okurensi pada hari-cabang yang sama** dengan SKU akikah yang telah dikonfirmasi pemilik data. Perilaku pemesanan akikah hanya muncul pada 0,84% hari-cabang aktif, sehingga ko-okurensi yang jauh melampaui angka tersebut menandakan perilaku pemesanan yang sama. Prosedur ini mengoreksi dua dugaan berbasis nama: `PCG-00028` (Cup 60 ml) berko-okurensi pada 100% hari geraknya sehingga praktis merupakan komponen paket akikah, sedangkan sembilan SKU Loyang — yang namanya mengandung kata "Box" — hanya berko-okurensi 0,9% hingga 1,4% sehingga tidak berkaitan dengan pesanan acara.

**(b) Segmentasi permintaan.** Setiap pasangan diklasifikasikan menurut skema **Syntetos-Boylan** berdasarkan dua statistik: *Average Demand Interval* (ADI), yakni rerata jarak hari antarpermintaan bukan-nol, dan kuadrat koefisien variasi (CV²) kuantitas bukan-nol. Ambang yang dipakai adalah nilai baku literatur, yaitu ADI = 1,32 dan CV² = 0,49.

|  | CV² < 0,49 | CV² ≥ 0,49 |
|---|---|---|
| **ADI < 1,32** | `smooth` | `erratic` |
| **ADI ≥ 1,32** | `intermittent` | `lumpy` |

Klasifikasi dihitung **dari periode latih saja**. Pasangan yang tidak pernah bergerak sepanjang periode latih diberi label `lumpy`, yakni kelas tersulit.

Kolom ini memiliki dua fungsi sekaligus: sebagai variabel prediktor, dan sebagai sumbu pelaporan metrik. Fungsi kedua penting secara metodologis, sebab galat absolut rerata yang dihitung global akan didominasi pasangan yang mayoritas nilainya nol; suatu model dapat tampak unggul padahal keunggulannya hanya terletak pada situasi ketika menebak nol memang mudah.

Penanda acara dan segmen permintaan keduanya diperlukan karena kejarangan dan sifat digerakkan-acara merupakan **dua sumbu berbeda yang kebetulan beririsan**. Suatu barang dapat jarang bergerak di sebagian cabang semata-mata karena volume cabang tersebut kecil, tanpa kaitan apa pun dengan pemesanan acara.

**(c) Penetapan *fold* validasi.** Lima *fold* validasi *walk-forward* berjendela membesar ditetapkan, dengan bulan validasi berturut-turut Juli hingga November 2025. Data latih *fold* ke-$k$ adalah seluruh baris bertanggal sebelum awal bulan validasinya, dikenai *purging* yang sama sebagaimana Subbab 4.10. Desember 2025 sengaja tidak menjadi *fold* mana pun, sebab merupakan himpunan uji akhir yang terkunci.

Skema ini dipilih menggantikan *holdout* tunggal karena periode uji hanya satu bulan dan secara musiman bersifat atipikal. Konsekuensi yang diterima adalah biaya komputasi lima kali lipat.

**(d) Imputasi yang mempertahankan makna.** Nilai kosong diisi dengan prinsip bahwa nilai pengganti tidak boleh menyamar sebagai observasi yang sah.

Variabel jarak peristiwa kalender diisi dengan nilai sentinel 99,0. Nilai ini dipilih karena harus melampaui setiap nilai yang sesungguhnya mungkin muncul: `days_until_ramadan` mencapai 70, sehingga sentinel bernilai 30 akan bertabrakan dengan observasi nyata. Pengisian dengan nol ditolak tegas, sebab nol pada kolom ini berarti "peristiwa terjadi hari ini" — makna yang akan keliru dilekatkan pada 84% hingga 97% baris.

Variabel `days_since_relocation` dan `baseline_ratio` diisi masing-masing dengan 0,0 dan 1,0, **disertai kolom indikator** `was_relocated` dan `has_baseline` yang menandai apakah nilai aslinya ada. Tanpa indikator, model tidak dapat membedakan "tidak pernah relokasi" dari "relokasi tepat hari ini".

Tiga belas variabel riwayat (lag dan *rolling*) diisi dengan 0,0, disertai dua kolom indikator: `missing_history_count` yang mencacah berapa banyak di antaranya yang semula kosong, dan `has_full_history` yang bernilai benar bila tidak ada satu pun yang kosong. Indikator ini diperlukan karena nol merupakan **nilai lag yang sah** pada data ini — 54% kuantitas bernilai nol — sehingga pengisian dengan nol tanpa indikator tidak dapat dibedakan dari "memang tidak ada permintaan hari itu". Kedua indikator bersifat lokal terhadap baris: banyaknya jendela yang gagal terbentuk merupakan fungsi monoton dari sejauh mana posisi baris di dalam segmennya, sehingga cacahnya sekaligus berfungsi sebagai ukuran ordinal atas panjang riwayat yang tersedia.

**(e) Pengodean variabel kategorik.** Tujuh kolom kategorik dipetakan ke indeks bilangan bulat, dengan pemetaan **dibentuk dari periode latih saja**. Nilai yang tidak dikenali dipetakan ke token `<UNKNOWN>` berindeks 0.

Pembentukan dari periode latih saja merupakan syarat kebenaran, bukan sekadar kerapian: suatu cabang yang baru muncul setelah batas periode uji tidak boleh masuk ke dalam pemetaan sama sekali. Namun syarat tersebut sendiri belum cukup menghadapi pemutakhiran data. Ketika batas periode uji bergeser maju, nilai yang semula berada di periode uji akan masuk ke periode latih dan bergabung ke pemetaan; pengurutan ulang atas seluruh himpunan nilai akan menggeser indeks setiap nilai yang berada sesudahnya. Pengukuran atas data penelitian menunjukkan bahwa masuknya enam SKU baru ke periode latih menggeser indeks 32 dari 70 SKU yang telah ada, yang secara diam-diam membatalkan model mana pun yang telah dilatih atas penomoran lama. Karena itu pemetaan yang telah tersimpan dimuat kembali dan **indeks yang telah diterbitkan dipertahankan**, sedangkan nilai baru ditambahkan sesudah indeks tertinggi. Nilai yang tidak lagi dipakai tetap memegang indeksnya, sebab membebaskan indeks tersebut untuk nilai baru akan mengarahkan model terlatih pada kategori yang keliru.

**Keluaran.** Berkas `model_input.parquet` dengan 1.502.522 baris dan 82 kolom, disertai berkas pemetaan kategori `category_mapping.json`.

### 4.13 Tahap 14 — *Adapter* dan kontrak antaralgoritma

**Tujuan.** Menyediakan bentuk masukan yang sesuai bagi masing-masing keluarga algoritma dari satu tabel yang sama, sekaligus menjamin secara terprogram bahwa keduanya memuat observasi yang identik.

**Panjang jendela retrospektif.** Ditetapkan 28 hari, selaras dengan lag terpanjang yang tersedia. Konsekuensinya, baris yang posisinya kurang dari 28 hari dari awal segmennya tidak dapat menjadi baris prediksi. Pada data penelitian hal ini menyingkirkan 90.388 baris, menyisakan 1.412.134 baris.

**Dua pemotongan yang dilakukan kedua *adapter*.** Pemotongan pertama adalah pemanasan tersebut. Pemotongan kedua adalah target: pada hari-hari terakhir suatu segmen, jendela target melampaui akhir data pasangan sehingga tidak ada target yang terbentuk. Baris-baris tersebut dibuang sebagai baris *prediksi*, namun **sengaja tidak dihapus dari kerangka data**, sebab tetap merupakan riwayat yang sah di dalam jendela baris-baris berikutnya; menghapusnya akan menyambung deret secara keliru dan mengubah apa yang dilihat LSTM. Setelah kedua pemotongan, tersisa **1.404.700 baris prediksi**.

***Adapter* tabular** (`to_tabular`) menghasilkan tabel datar untuk XGBoost dan Random Forest, dengan nilai kosong pada fitur dibiarkan apa adanya karena kedua algoritma tersebut menanganinya secara asali.

***Adapter* sekuens** (`to_sequences`) menghasilkan tensor berdimensi (jumlah baris, 28, jumlah fitur) untuk LSTM, dengan setiap jendela berakhir pada baris bersangkutan secara inklusif.

**Kontrak antar-*adapter*.** Fungsi `validate_contract()` memverifikasi lima hal: kedua *adapter* menghasilkan jumlah baris yang sama, himpunan kunci (pasangan, tanggal) yang identik, nilai target yang identik, pembagian *fold* yang identik, dan tidak adanya nilai `NaN` pada kedua blok fitur maupun pada label.

Verifikasi ini merupakan **instrumen pengendali validitas internal penelitian**, bukan sekadar pemeriksaan teknis. Tanpanya, pernyataan "LSTM lebih baik 8%" berpotensi sesungguhnya bermakna "LSTM dievaluasi atas 5% baris yang berbeda". Pemeriksaan ketiadaan `NaN` diperlukan secara terpisah karena kesetaraan baris saja tidak memadai: model pohon mengonsumsi `NaN` secara asali sementara LSTM mengubahnya menjadi *loss* bernilai `NaN`, sehingga tensor yang masih memuat nilai kosong akan ditambal pada saat pelatihan dan kedua model diam-diam berhenti melihat masukan yang sama.

**Daftar fitur tunggal.** Konstanta `FEATURE_COLS` mendefinisikan satu daftar variabel prediktor yang dipakai ketiga algoritma. Dua kolom sengaja dikecualikan darinya, yaitu `baseline_ratio` dan `is_spike`, sebab keduanya diturunkan dari kuantitas pada hari $t$ itu sendiri; menyertakannya akan memungkinkan model memulihkan permintaan hari $t$ dan menjadikan frasa "diketahui pada saat prediksi" bermakna dua hal berbeda dalam satu baris yang sama. Biaya pengecualian ini terukur kecil: galat absolut rerata garis dasar bergeser dari 12,99 menjadi 13,19 ketika hari $t$ diizinkan masuk.

**Standardisasi.** Parameter standardisasi (rerata dan simpangan baku per fitur) dibentuk **per *fold*, dari baris latih *fold* tersebut saja**, kemudian dipersistensi ke `scaler_params.json`. Pembentukan secara global akan membocorkan statistik Desember ke dalam *fold* Juli.

**Transformasi target.** Transformasi `log1p` atas target diperbolehkan dan pembalikannya dilakukan dengan `expm1`. Pembalikan ini bersifat eksak untuk model kuantil, sebab kuantil bersifat ekuivarian terhadap transformasi monoton, sehingga $\text{expm1}(q_\alpha(\text{log1p}(y))) = q_\alpha(y)$. Nilai parameter transformasi harus sama pada kedua *adapter*, dan ketidaksamaannya akan menggagalkan pemeriksaan kontrak.

---

## 5. Hasil Prapemrosesan

### 5.1 Dimensi artefak keluaran

| Artefak | Baris | Kolom | Keterangan |
|---|---|---|---|
| `dataset/dataset.csv` | 693.563 | 7 | Gabungan lima periode mentah |
| `dataset/model_ready/featured.parquet` | 1.502.522 | 68 | Bersih dan berfitur, belum dibagi |
| `dataset/model_ready/train.parquet` | 1.441.159 | 68 | Sebelum 1 Desember 2025, telah di-*purge* |
| `dataset/model_ready/test.parquet` | 55.046 | 68 | Desember 2025 |
| `dataset/model_ready/model_input.parquet` | 1.502.522 | 82 | Sumber kebenaran tunggal untuk pemodelan |
| `dataset/model_ready/category_mapping.json` | — | — | Pemetaan kategori ke indeks |
| `dataset/model_ready/scaler_params.json` | — | — | Parameter standardisasi per *fold* |

Peningkatan jumlah baris dari 693.563 pada data mentah menjadi 1.502.522 pada panel merupakan konsekuensi langsung konstruksi panel padat: hari tanpa transaksi yang tidak terwakili pada log kini hadir sebagai baris bernilai nol.

**Cakupan akhir.** 70 kode barang × 59 cabang menghasilkan 2.979 pasangan aktif, tersebar di 16 kota, sepanjang 731 hari kalender (1 Januari 2024 – 31 Desember 2025).

### 5.2 Statistik deskriptif variabel kuantitas dan target

| Statistik | `Kuantitas` | `Kuantitas_capped` | `target_lead_time_cumulative` |
|---|---|---|---|
| Jumlah observasi | 1.502.522 | 1.502.522 | 1.495.046 |
| Rerata | 13,93 | 13,79 | 30,78 |
| Simpangan baku | 43,22 | 42,52 | 99,99 |
| Minimum | 0 | 0 | 0 |
| Kuartil 1 | 0 | 0 | 0 |
| Median | 0 | 0 | 1 |
| Kuartil 3 | 4 | 4 | 8 |
| Maksimum | 1.435 | 1.435 | 3.067 |

Tiga karakteristik menonjol. Pertama, **54,15% baris panel bernilai nol**, yang menegaskan sifat *intermittent* data pada granularitas item–cabang harian. Kedua, distribusi menjulur kanan secara ekstrem — rerata target 30,78 terhadap median 1 — sehingga fungsi kerugian berbasis rerata akan didominasi oleh sebagian kecil observasi bervolume besar. Ketiga, pemangkasan pencilan berdampak kecil pada agregat (rerata bergeser 1,0%) namun terkonsentrasi: hanya 0,903% baris yang terpangkas.

### 5.3 Proporsi nilai hilang menurut keluarga fitur

Seluruh nilai hilang pada tabel fitur bersifat struktural, yakni timbul karena definisi variabel dan bukan karena kerusakan data. Tabel berikut menyajikan proporsinya beserta penyebabnya.

| Kolom | Hilang (%) | Penyebab struktural |
|---|---|---|
| `days_since_relocation` | 84,317 | Cabang tidak pernah direlokasi (hanya 9 cabang memiliki tanggal relokasi) |
| `baseline_ratio` | 14,391 | Pasangan tidak memenuhi syarat kelayakan garis dasar |
| `lag_28`, `roll_mean_28`, `roll_std_28` | 6,016 | Jendela 28 hari belum terbentuk pada awal segmen |
| `lag_21` | 4,517 | Idem, jendela 21 hari |
| `lag_14`, `roll_mean_14`, `roll_std_14` | 3,014 | Idem, jendela 14 hari |
| `lag_7`, `roll_mean_7`, `roll_std_7` | 1,508 | Idem, jendela 7 hari |
| `target_h7` | 1,508 | Jendela maju melampaui akhir data segmen |
| `target_h6` | 1,292 | Idem |
| `target_h5` | 1,077 | Idem |
| `target_h4` | 0,861 | Idem |
| `lag_3`, `target_h3` | 0,646 | Idem |
| `target_lead_time_cumulative` (dan varian terpangkas) | 0,498 | Idem |
| `lag_2`, `target_h2` | 0,431 | Idem |
| `lag_1`, `target_h1` | 0,215 | Idem |

Variabel jarak peristiwa kalender juga bernilai kosong di luar jendela ±14 hari, dengan proporsi 84% hingga 97% bergantung peristiwanya. Seluruh nilai kosong pada tabel di atas ditangani pada Tahap 13 dengan prosedur imputasi yang mempertahankan makna (Subbab 4.12 butir d); setelah imputasi, 93,98% baris memiliki riwayat lengkap (`has_full_history` bernilai benar) dan 85,61% baris memiliki garis dasar (`has_baseline` bernilai benar).

### 5.4 Distribusi variabel klasifikasi

**Segmentasi permintaan Syntetos-Boylan.** Diukur pada tingkat pasangan, yang merupakan tingkat penetapan klasifikasi:

| Segmen | Jumlah pasangan | Proporsi | Diukur per baris |
|---|---|---|---|
| `intermittent` | 1.304 | 43,77% | 38,67% |
| `lumpy` | 943 | 31,65% | 34,21% |
| `erratic` | 407 | 13,66% | 14,72% |
| `smooth` | 325 | 10,91% | 12,39% |

Temuan pokoknya adalah **75,43% pasangan tergolong *intermittent* atau *lumpy***, yakni dua kelas yang paling sulit diramalkan menurut literatur. Perbedaan antara proporsi per pasangan dan per baris mencerminkan bahwa pasangan `smooth` cenderung memiliki rentang aktif yang lebih panjang.

**Lead time.** Distribusinya relatif berimbang untuk tiga nilai pertama, dengan $L = 4$ lebih jarang karena hanya timbul pada satu konfigurasi hari dalam pekan per kawasan:

| `lead_time_days` | Jumlah baris | Proporsi |
|---|---|---|
| 1 | 429.125 | 28,56% |
| 2 | 429.596 | 28,59% |
| 3 | 429.623 | 28,59% |
| 4 | 214.178 | 14,25% |

**Hari pengiriman.** Sebanyak 28,57% baris jatuh pada hari pengiriman cabangnya sendiri. Angka ini penting bagi pelaporan metrik: hanya baris-baris tersebut yang merupakan momen keputusan yang sesungguhnya, dan pada baris tersebut lead time hanya pernah bernilai 3 atau 4.

**Kawasan.** Kawasan 2 mendominasi dengan 1.150.299 baris (76,56%), berbanding Kawasan 1 dengan 352.223 baris (23,44%).

**Tingkatan volume cabang.** Pembagian kuartil menghasilkan distribusi yang tidak seragam pada tingkat baris, sebab tingkatan ditetapkan per cabang sementara jumlah pasangan per cabang berbeda: `flagship` 405.868 baris, `large` 388.416, `small` 367.859, dan `medium` 340.379.

**Barang digerakkan acara.** Lima SKU bertanda benar, mencakup 4,37% baris.

**Distribusi *fold* validasi.** Kelima *fold* memuat 76.263, 76.266, 70.982, 69.392, dan 61.165 baris validasi secara berturut-turut. Sisanya sebanyak 1.148.454 baris tidak termasuk bulan validasi mana pun dan berfungsi semata-mata sebagai data latih.

**Segmentasi aktif.** Sebanyak 230 pasangan memiliki lebih dari satu segmen, dengan nomor segmen tertinggi 3. Kolom `branch_age_days` berkisar antara 0 hingga 730 hari.

---

## 6. Kamus Kolom Keluaran

Bab ini memaparkan seluruh 68 kolom `featured.parquet` beserta 14 kolom tambahan `model_input.parquet`, dikelompokkan menjadi sembilan keluarga fungsional. Setiap butir memuat nama kolom, tipe data, definisi formal, dan penjelasan mengenai perannya.

Kolom yang dipakai sebagai variabel prediktor oleh ketiga algoritma ditandai keterangan **[fitur]**; kolom yang berfungsi sebagai kunci, target, atau metadata pelaporan tidak diberi tanda tersebut.

### 6.1 Kunci identitas dan atribut deskriptif (7 kolom)

1. **`Kode Barang`** (untai teks) — pengenal SKU setelah normalisasi Tahap 3, berpola tiga huruf, tanda hubung, lima digit. Merupakan komponen pertama kunci unit analisis. Kolom ini tidak dipakai langsung sebagai fitur melainkan melalui bentuk terkodenya `Kode Barang_idx`, sebab algoritma pembelajaran mesin memerlukan masukan numerik. Terdapat 70 nilai berbeda.

2. **`Nama Cabang`** (untai teks) — pengenal gerai setelah kanonikalisasi Tahap 4, sehingga satu gerai fisik selalu diwakili satu nilai meskipun sepanjang periode data sempat menggunakan lebih dari satu kode. Merupakan komponen kedua kunci unit analisis. Sebagaimana kode barang, dipakai sebagai fitur melalui bentuk terkodenya. Terdapat 59 nilai berbeda.

3. **`Tanggal`** (timestamp) — tanggal kalender baris, yang sebagaimana ditegaskan pada Subbab 2.5 merupakan tanggal **pengambilan** barang. Berfungsi sebagai sumbu waktu bagi seluruh operasi pergeseran, sebagai dasar pembangkitan seluruh fitur kalender, dan sebagai kriteria pembagian latih–uji. Tidak dipakai sebagai fitur dalam bentuk aslinya, sebab nilai absolut tanggal tidak dapat digeneralisasi ke periode masa depan.

4. **`Kategori Barang`** (untai teks) — kelompok barang menurut tahap pengolahannya, dengan delapan nilai berbeda setelah kanonikalisasi. Kolom ini menyediakan sarana bagi model untuk berbagi kekuatan statistik antarbarang sejenis: SKU baru dengan riwayat pendek dapat memperoleh manfaat dari pola yang dipelajari atas kategorinya. Dipakai sebagai fitur melalui `Kategori Barang_idx`.

5. **`Nama Barang`** (untai teks) — nama produk dalam bahasa Indonesia. Tidak dipakai sebagai fitur maupun kunci, dan dipertahankan semata-mata untuk keperluan interpretasi hasil serta penelusuran anomali. Nilainya dipropagasi maju kemudian mundur saat konstruksi panel, sehingga hari hasil pengisian tetap memiliki nama yang benar.

6. **`Satuan`** (untai teks) — satuan ukur kuantitas, dengan sembilan nilai berbeda pada panel akhir. Tidak dipakai sebagai fitur karena bersifat konstan untuk setiap kode barang — sifat yang setelah Tahap 3 dijamin berlaku, termasuk bagi dua item yang semula tercatat dalam gram. Kolom ini dipertahankan hingga keluaran akhir atas permintaan pemilik data, karena kuantitas tidak dapat diinterpretasikan tanpanya.

7. **`segment_id`** (bilangan bulat) — nomor blok tanggal aktif yang berurutan milik satu pasangan, dimulai dari 1. Segmen baru dimulai setelah interval penutupan gerai atau pada tanggal relokasi yang teramati. Kolom ini bukan fitur, melainkan **variabel pengendali struktural**: seluruh operasi lag, *rolling*, pergeseran target, dan pembentukan jendela LSTM dikelompokkan menurut (pasangan, segmen), sehingga tidak satu pun di antaranya menjembatani periode ketika gerai tidak beroperasi. Bernilai maksimum 3 pada data penelitian.

### 6.2 Variabel kuantitas dan penanganan pencilan (4 kolom)

1. **`Kuantitas`** (pecahan) — banyaknya barang yang diserahkan pada baris bersangkutan; bernilai 0 pada hari hasil pengisian panel. Merupakan variabel dasar yang seluruh target diturunkan darinya. Kolom ini sendiri **tidak** dipakai sebagai fitur, sebab nilainya pada hari $t$ belum diketahui pada saat prediksi untuk hari $t$ dilakukan.

2. **`baseline_ratio`** (pecahan) — rasio `Kuantitas` terhadap median kuantitas positif pasangan pada periode latih. Bernilai kosong bagi pasangan yang tidak memenuhi syarat kelayakan, yakni memiliki kurang dari 30 hari transaksi positif atau median nol. Kolom ini **sengaja dikecualikan dari daftar fitur**: karena pembilangnya adalah kuantitas hari $t$ itu sendiri sementara seluruh fitur lag dan *rolling* berhenti pada $t-1$, menyertakannya akan memungkinkan model memulihkan permintaan hari $t$. Perannya adalah sebagai variabel diagnostik untuk penelaahan pencilan dan sebagai dasar penghitungan `is_spike`.

3. **`is_spike`** (boolean) — bernilai benar apabila pasangan memenuhi syarat kelayakan dan `baseline_ratio` mencapai 5,0 atau lebih. Sebagaimana `baseline_ratio`, dikecualikan dari daftar fitur dengan alasan yang sama. Perannya adalah menandai baris yang lonjakannya diredam, sehingga tahap pemodelan dapat memperlakukannya secara khusus — misalnya sebagai bobot sampel — dan sehingga pelaporan galat dapat dipisahkan antara baris lonjakan dan baris reguler. Bernilai benar pada 0,903% baris.

4. **`Kuantitas_capped`** (pecahan) — kuantitas yang lonjakan ekstremnya telah dipangkas ke median pasangan dikalikan 5,0, kecuali apabila baris jatuh pada jendela peristiwa besar. Bagi pasangan yang riwayatnya hanya bernilai bulat, ambang dibulatkan ke atas lalu dibatasi agar tidak melampaui nilai mentahnya. Kolom ini merupakan **sumber bagi seluruh fitur riwayat** — lag, *rolling*, dan statistik cabang — sehingga lonjakan tidak mencemari gambaran tingkat permintaan normal yang dilihat model. Invarian `Kuantitas_capped ≤ Kuantitas` dijamin oleh asersi penjaminan mutu.

### 6.3 Fitur kalender (20 kolom)

Seluruh kolom pada keluarga ini dibangkitkan dari `Tanggal` semata dan tidak memerlukan data historis apa pun, sehingga sepenuhnya bebas dari risiko kebocoran informasi dan selalu dapat dihitung untuk tanggal masa depan mana pun.

1. **`day_of_week`** (bilangan bulat, 0–6) **[fitur]** — indeks hari dalam pekan, dengan 0 menyatakan Senin. Menangkap musiman mingguan, yang pada data ini kuat: indeks volume pengambilan mencapai 143 pada hari Minggu berbanding 77 pada hari Senin.

2. **`day_of_month`** (bilangan bulat, 1–31) **[fitur]** — tanggal dalam bulan. Menangkap pola yang terkait siklus penggajian dan siklus penagihan.

3. **`month`** (bilangan bulat, 1–12) **[fitur]** — bulan dalam tahun. Menangkap musiman tahunan kasar, dengan catatan bahwa dua tahun data hanya menyediakan dua pengamatan per bulan sehingga daya generalisasinya terbatas.

4. **`is_weekend`** (boolean) **[fitur]** — bernilai benar untuk hari Sabtu dan Minggu. Secara teknis merupakan turunan `day_of_week`, namun disediakan tersendiri agar model berbasis pohon tidak perlu membentuk dua pemisahan untuk mencapai informasi yang sama.

5. **`is_national_holiday`** (boolean) **[fitur]** — bernilai benar pada hari libur nasional Indonesia menurut pustaka `holidays` untuk tahun 2024 dan 2025.

6. **`is_ramadan`** (boolean) **[fitur]** — bernilai benar sepanjang bulan Ramadan, dengan periode ditetapkan eksplisit: 11 Maret – 9 April 2024 dan 1 – 30 Maret 2025.

7. **`days_into_ramadan`** (pecahan, 0–29) **[fitur]** — banyaknya hari yang telah berlalu sejak awal Ramadan; kosong di luar bulan Ramadan. Variabel ini membedakan awal, pertengahan, dan akhir Ramadan, yang pola permintaannya berbeda secara substansial.

8. **`days_until_ramadan`** (pecahan, 0–70) **[fitur]** — banyaknya hari menuju awal Ramadan tahun berjalan; kosong pada dan setelah dimulainya Ramadan. Berbeda dengan variabel jarak peristiwa lainnya, kolom ini **tidak dibatasi jendela ±14 hari** dan terdefinisi untuk setiap tanggal sebelum Ramadan pada tahun yang sama, sehingga nilainya dapat mencapai 70. Fakta inilah yang menentukan pemilihan nilai sentinel 99,0 pada tahap imputasi.

9. **`is_eid_al_fitr`** (boolean) **[fitur]** — bernilai benar tepat pada hari Idulfitri, yakni 10 April 2024 dan 31 Maret 2025.

10. **`days_since_eid_al_fitr`** (pecahan, 0–14) **[fitur]** — banyaknya hari sejak Idulfitri, terdefinisi hanya dalam jendela 14 hari sesudahnya. Menangkap fase pemulihan permintaan pascalebaran.

11. **`days_until_eid_al_fitr`** (pecahan, 0–14) **[fitur]** — banyaknya hari menuju Idulfitri, terdefinisi hanya dalam jendela 14 hari sebelumnya. Menangkap lonjakan persiapan lebaran.

12. **`is_eid_al_adha`** (boolean) **[fitur]** — bernilai benar tepat pada hari Iduladha, yakni 17 Juni 2024 dan 6 Juni 2025.

13. **`days_since_eid_al_adha`** (pecahan, 0–14) **[fitur]** — banyaknya hari sejak Iduladha dalam jendela 14 hari.

14. **`days_until_eid_al_adha`** (pecahan, 0–14) **[fitur]** — banyaknya hari menuju Iduladha dalam jendela 14 hari. Peristiwa ini relevan secara khusus bagi jaringan gerai bermenu daging kambing dan sapi.

15. **`is_independence_day`** (boolean) **[fitur]** — bernilai benar pada 17 Agustus.

16. **`days_since_independence_day`** (pecahan, 0–14) **[fitur]** — banyaknya hari sejak 17 Agustus dalam jendela 14 hari.

17. **`days_until_independence_day`** (pecahan, 0–14) **[fitur]** — banyaknya hari menuju 17 Agustus dalam jendela 14 hari. Menangkap permintaan terkait perayaan dan acara kantor.

18. **`is_new_year`** (boolean) **[fitur]** — bernilai benar pada 1 Januari.

19. **`days_since_new_year`** (pecahan, 0–14) **[fitur]** — banyaknya hari sejak 1 Januari dalam jendela 14 hari.

20. **`days_until_new_year`** (pecahan, 0–14) **[fitur]** — banyaknya hari menuju 1 Januari berikutnya dalam jendela 14 hari, sehingga terdefinisi pada 18–31 Desember dan pada 1 Januari itu sendiri. Berbeda dengan variabel `days_until_*` lainnya yang merujuk peristiwa tahun berjalan, kolom ini beralih ke 1 Januari tahun berikutnya begitu tanggal berjalan melewatinya.

### 6.4 Variabel target (9 kolom)

Seluruh kolom pada keluarga ini merupakan variabel terikat dan **tidak satu pun boleh menjadi variabel prediktor**. Seluruhnya dihitung atas jendela maju ketat dalam segmen yang sama.

1. **`target_h1`** (pecahan) — kuantitas mentah pada hari $t+1$, yakni $q_{p,t+1}$. Kosong pada 0,215% baris, yakni baris terakhir setiap segmen.

2. **`target_h2`** (pecahan) — kuantitas mentah pada hari $t+2$. Kosong pada 0,431% baris.

3. **`target_h3`** (pecahan) — kuantitas mentah pada hari $t+3$. Kosong pada 0,646% baris.

4. **`target_h4`** (pecahan) — kuantitas mentah pada hari $t+4$. Kosong pada 0,861% baris. Horizon ini merupakan yang terpanjang yang memiliki padanan operasional langsung, sebab lead time tidak pernah melebihi 4.

5. **`target_h5`** (pecahan) — kuantitas mentah pada hari $t+5$. Kosong pada 1,077% baris.

6. **`target_h6`** (pecahan) — kuantitas mentah pada hari $t+6$. Kosong pada 1,292% baris.

7. **`target_h7`** (pecahan) — kuantitas mentah pada hari $t+7$. Kosong pada 1,508% baris. Ketiga horizon terakhir tidak memiliki padanan operasional langsung dan dipertahankan semata-mata agar prediksi kumulatif dapat didekomposisi menjadi kontribusi harian untuk keperluan penjelasan hasil.

8. **`target_lead_time_cumulative`** (pecahan) — **variabel target utama penelitian**, yaitu $\sum_{k=1}^{L_{p,t}} q_{p,t+k}$, jumlah kuantitas mentah sepanjang jendela maju ketat sepanjang lead time baris bersangkutan. Inilah angka yang dituliskan pada surat jalan dan yang menjadi objek optimasi ketiga algoritma. Kosong pada 0,498% baris, yakni baris yang jendela targetnya melampaui akhir data segmennya. Statistik deskriptifnya disajikan pada Subbab 5.2.

9. **`target_lead_time_cumulative_capped`** (pecahan) — target yang dirumuskan identik namun menjumlahkan `Kuantitas_capped`. Keberadaannya merupakan konsekuensi batasan pada Subbab 2.5: karena kuantitas mentah mencampur permintaan reguler dengan komponen pesanan tanpa penanda apa pun, sementara baris terpangkas merupakan pendekatan terdekat bagi komponen pesanan tersebut, maka pilihan antara kedua target menentukan pertanyaan penelitian yang sesungguhnya dijawab. Invarian `target_capped ≤ target` dijamin oleh asersi penjaminan mutu.

### 6.5 Fitur siklus pengiriman (5 kolom)

Keluarga ini mengodekan aturan logistik pusat distribusi dan merupakan kelompok fitur yang paling erat kaitannya dengan definisi target.

1. **`kawasan`** (bilangan bulat, 1 atau 2) **[fitur]** — kawasan logistik cabang, yang menentukan jadwal pengiriman: Kawasan 1 dikirim Senin dan Kamis, Kawasan 2 dikirim Selasa dan Jumat. Selain sebagai penentu lead time, kolom ini berfungsi sebagai pengelompokan geografis kasar. Kawasan 2 mencakup 76,56% baris.

2. **`hari_pengiriman`** (untai teks) **[fitur melalui bentuk terkode]** — jadwal pengiriman cabang dalam bentuk teks, contohnya `Selasa dan Jumat`. Secara informasi bersifat redundan terhadap `kawasan` pada data saat ini, namun dipertahankan sebagai kolom tersendiri agar penambahan kawasan baru dengan jadwal berbeda pada pemutakhiran mendatang tidak memerlukan perubahan struktur. Dipakai sebagai fitur melalui `hari_pengiriman_idx`.

3. **`lead_time_days`** (bilangan bulat, 1–4) **[fitur]** — jarak hari dari tanggal baris menuju hari pengiriman berikutnya, dihitung dengan rumus pada Subbab 3.2. Bersifat **maju ketat**, sehingga baris yang jatuh tepat pada hari pengiriman memperoleh jarak menuju kejadian berikutnya dan bukan nol. Kolom ini menentukan panjang jendela penjumlahan target, sehingga merupakan fitur yang paling langsung menjelaskan skala nilai target: baris dengan lead time 4 secara konstruksi memiliki ekspektasi target lebih besar daripada baris dengan lead time 1.

4. **`is_delivery_day`** (boolean) **[fitur]** — bernilai benar apabila tanggal baris merupakan salah satu hari pengiriman cabangnya sendiri. Bernilai benar pada 28,57% baris. Perannya bersifat ganda. Sebagai fitur, kolom ini menandai hari ketika gerai baru saja menerima kiriman sehingga perilaku permintaannya berbeda. Sebagai sumbu pelaporan, kolom ini mengisolasi **momen keputusan yang sesungguhnya**: kantor pusat hanya mengirim dua kali sepekan, sehingga hanya pada baris inilah suatu ramalan benar-benar menggerakkan tindakan. Pada baris tersebut lead time hanya pernah bernilai 3 atau 4. Pelatihan tetap dilakukan atas seluruh hari, namun metrik yang dirata-ratakan atas seluruh hari menjawab pertanyaan yang tidak diajukan siapa pun. Cabang yang tidak memiliki jadwal tercatat memperoleh nilai salah, bukan nilai kosong, sebab ketiadaan jadwal berarti tidak ada hari pengiriman dan bukan hari pengiriman yang tidak diketahui.

5. **`target_window_weekend_days`** (pecahan, 0–2) **[fitur]** — cacah hari Sabtu dan Minggu yang jatuh di dalam jendela target $(t{+}1 \ldots t{+}L_{p,t})$. Variabel ini penting karena target merupakan penjumlahan atas jendela berukuran variabel, dan **komposisi hari di dalam jendela lebih menentukan daripada panjangnya**. Dengan indeks volume pengambilan 143 pada hari Minggu berbanding 77 pada hari Senin, kiriman hari Kamis bagi cabang Kawasan 1 — yang menanggung Jumat hingga Senin — harus memikul sekitar 1,9 kali beban kiriman hari Senin yang menanggung Selasa hingga Kamis. Tanpa kolom ini, informasi yang sama hanya dapat dicapai model sebagai interaksi antara `day_of_week` dan `lead_time_days`, yang bagi model linear maupun LSTM jauh lebih sulit dipelajari.

### 6.6 Fitur outlet (6 kolom)

Keluarga ini bersumber dari data induk gerai yang bersifat statis, sehingga tidak membawa risiko kebocoran informasi temporal.

1. **`kota`** (untai teks) **[fitur melalui bentuk terkode]** — kota atau kabupaten tempat gerai berada, dengan 16 nilai berbeda. Awalan `Kota ` dan `Kabupaten ` sengaja dipertahankan karena keduanya memiliki karakteristik permintaan yang berbeda. Kolom ini memungkinkan model berbagi kekuatan statistik antargerai dalam pasar yang sama, khususnya bermanfaat bagi gerai baru yang riwayatnya masih pendek. Asersi penjaminan mutu memastikan tidak ada nilai `Unknown` dan tidak ada cabang yang memetakan ke lebih dari satu kota.

2. **`has_shopee`** (boolean) **[fitur]** — bernilai benar apabila gerai melayani pemesanan melalui Shopee.

3. **`has_gofood`** (boolean) **[fitur]** — bernilai benar apabila gerai melayani pemesanan melalui GoFood.

4. **`has_grabfood`** (boolean) **[fitur]** — bernilai benar apabila gerai melayani pemesanan melalui GrabFood. Ketiga kanal daring ini dipisah dan tidak digabung karena basis pelanggan serta pola promosi masing-masing platform berbeda, sehingga komposisi kanal membawa informasi yang lebih kaya daripada sekadar jumlahnya.

5. **`can_order_online`** (boolean) **[fitur]** — bernilai benar apabila sekurang-kurangnya satu dari ketiga kanal di atas tersedia. Bernilai kosong apabila salah satu kanal penyusunnya tidak diketahui, sebab dalam keadaan tersebut disjungsinya memang tidak dapat ditentukan. Disediakan sebagai ringkasan agar model tidak perlu membentuk tiga pemisahan untuk membedakan gerai yang sepenuhnya luring.

6. **`days_since_relocation`** (pecahan) **[fitur]** — selisih hari antara tanggal baris dan tanggal relokasi gerainya; bernilai negatif sebelum pindah, nol pada hari pindah, dan positif sesudahnya. Kosong pada 84,317% baris, yakni gerai yang tidak pernah direlokasi; hanya 9 cabang memiliki tanggal relokasi tercatat.

   Kolom ini menangani konsekuensi tak terhindarkan dari kanonikalisasi cabang. Karena penyatuan nama dilakukan **sebelum** penggabungan data induk gerai, atribut `kota` dan `kawasan` suatu cabang yang pernah pindah merefleksikan lokasi **saat ini** untuk **seluruh** riwayatnya, termasuk baris sebelum pindah yang sesungguhnya tercatat di kota lain. Alternatif berupa pembuangan riwayat prarelokasi ditolak, sebab justru menghapus satu-satunya contoh transisi yang dimiliki model. Kolom ini disediakan agar tahap pemodelan dapat memperhitungkan pergeseran rezim tersebut secara eksplisit. Perlu dicatat bahwa dari sembilan tanggal relokasi, empat bersifat eksak dan lima merupakan perkiraan batas bawah.

### 6.7 Fitur riwayat permintaan (13 kolom)

Keluarga ini merupakan sumber informasi utama bagi peramalan dan sekaligus keluarga yang paling rawan kebocoran informasi. Seluruhnya dihitung dari **`Kuantitas_capped`**, bukan kuantitas mentah, dan seluruhnya dikelompokkan menurut (pasangan, segmen) sehingga tidak menjangkau melintasi periode gerai tidak beroperasi.

1. **`lag_1`** (pecahan) **[fitur]** — kuantitas terpangkas satu hari sebelumnya, $q^{c}_{p,t-1}$. Merupakan prediktor tunggal terkuat bagi deret yang bersifat *smooth*. Kosong pada 0,215% baris, yakni baris pertama setiap segmen.

2. **`lag_2`** (pecahan) **[fitur]** — kuantitas terpangkas dua hari sebelumnya. Kosong pada 0,431% baris.

3. **`lag_3`** (pecahan) **[fitur]** — kuantitas terpangkas tiga hari sebelumnya. Kosong pada 0,646% baris. Ketiga lag pendek ini bersama-sama menangkap momentum jangka sangat pendek serta pola sisa persediaan pascakiriman.

4. **`lag_7`** (pecahan) **[fitur]** — kuantitas terpangkas tujuh hari sebelumnya, yakni **hari yang sama pada pekan sebelumnya**. Merupakan lag musiman mingguan dan karenanya bernilai khusus pada data yang musiman mingguannya kuat. Kosong pada 1,508% baris.

5. **`lag_14`** (pecahan) **[fitur]** — kuantitas terpangkas dua pekan sebelumnya pada hari yang sama. Kosong pada 3,014% baris.

6. **`lag_21`** (pecahan) **[fitur]** — kuantitas terpangkas tiga pekan sebelumnya pada hari yang sama. Kosong pada 4,517% baris.

7. **`lag_28`** (pecahan) **[fitur]** — kuantitas terpangkas empat pekan sebelumnya pada hari yang sama. Kosong pada 6,016% baris. Merupakan lag terpanjang, dan panjangnya inilah yang menetapkan jendela retrospektif 28 hari bagi LSTM serta menentukan banyaknya baris yang tersingkir sebagai baris pemanasan.

8. **`roll_mean_7`** (pecahan) **[fitur]** — rerata aritmetik `Kuantitas_capped` sepanjang jendela tujuh hari yang berakhir **satu hari sebelum** tanggal baris, yakni $(t{-}7 \ldots t{-}1)$. Pergeseran satu hari sebelum agregasi merupakan penjaga kebocoran yang esensial: tanpanya, nilai hari $t$ akan masuk ke dalam statistik yang dipakai memprediksi hari $t$ itu sendiri. Perannya adalah proksi tingkat permintaan jangka pendek yang lebih tahan derau dibandingkan lag tunggal. Kosong pada 1,508% baris.

9. **`roll_std_7`** (pecahan) **[fitur]** — simpangan baku pada jendela yang sama. Mengukur volatilitas jangka pendek, yang bagi model kuantil berguna untuk membedakan deret yang tingkatnya sama namun ketidakpastiannya berbeda. Kosong pada 1,508% baris.

10. **`roll_mean_14`** (pecahan) **[fitur]** — rerata pada jendela empat belas hari $(t{-}14 \ldots t{-}1)$, dengan penjaga pergeseran yang sama. Kosong pada 3,014% baris.

11. **`roll_std_14`** (pecahan) **[fitur]** — simpangan baku pada jendela empat belas hari. Kosong pada 3,014% baris.

12. **`roll_mean_28`** (pecahan) **[fitur]** — rerata pada jendela dua puluh delapan hari $(t{-}28 \ldots t{-}1)$. Merupakan estimasi tingkat permintaan jangka menengah yang paling stabil, dan mencakup tepat empat siklus mingguan penuh sehingga tidak bias terhadap hari tertentu dalam pekan. Kosong pada 6,016% baris.

13. **`roll_std_28`** (pecahan) **[fitur]** — simpangan baku pada jendela dua puluh delapan hari. Kosong pada 6,016% baris.

Ketiga panjang jendela disediakan bersama-sama karena masing-masing memiliki keseimbangan responsivitas terhadap stabilitas yang berbeda, dan panjang mana yang paling informatif berbeda antarsegmen permintaan: deret *smooth* lebih diuntungkan jendela pendek, sedangkan deret *intermittent* memerlukan jendela panjang agar estimasinya tidak sepenuhnya bernilai nol.

### 6.8 Statistik cabang (4 kolom)

Tiga kolom pertama pada keluarga ini **dibekukan dari periode latih saja**, sehingga permintaan bulan Desember suatu cabang tidak pernah memengaruhi fitur cabang itu sendiri.

1. **`branch_avg_daily_qty`** (pecahan) **[fitur]** — rerata total kuantitas terpangkas harian seluruh barang pada satu cabang, dihitung atas periode latih. Kolom ini menyatakan skala cabang secara absolut, yang memungkinkan model menormalisasi ekspektasinya: permintaan 10 unit merupakan angka besar bagi gerai kecil namun kecil bagi gerai unggulan.

2. **`branch_demand_cv`** (pecahan) **[fitur]** — koefisien variasi permintaan harian cabang, yaitu simpangan baku dibagi rerata, dihitung atas periode latih. Menyatakan seberapa dapat diprediksi suatu cabang secara umum, terlepas dari skalanya.

3. **`branch_volume_tier`** (kategorik terurut) **[fitur melalui bentuk terkode]** — tingkatan volume cabang hasil pembagian kuartil atas `branch_avg_daily_qty`, dengan empat nilai berurutan: `small`, `medium`, `large`, `flagship`. Pemeringkatan dilakukan atas peringkat nilai dan bukan atas nilainya langsung, sehingga cabang bervolume persis sama tetap terdistribusi ke kuartil yang berbeda alih-alih menyebabkan kegagalan pembentukan batas kuartil. Kolom ini menyediakan bentuk kategorik dari informasi yang sama dengan butir pertama, yang bagi model berbasis pohon lebih mudah dimanfaatkan sebagai dasar pemisahan.

4. **`branch_age_days`** (bilangan bulat, 0–730) **[fitur]** — selisih hari antara tanggal baris dan tanggal pertama cabang bersangkutan muncul pada data. Berbeda dengan ketiga kolom sebelumnya, kolom ini **aman secara inheren** dan tidak memerlukan pembekuan periode latih, sebab hanya membaca masa lalu cabang itu sendiri. Perannya adalah menandai fase pematangan gerai: gerai yang baru dibuka umumnya mengalami pertumbuhan permintaan yang tidak akan berulang setelah mapan.

### 6.9 Kolom tambahan pada `model_input.parquet` (14 kolom)

Keempat belas kolom berikut ditambahkan pada Tahap 13 dan tidak terdapat pada `featured.parquet`. Kolom-kolom inilah yang mengodekan keputusan eksperimen, sebagaimana dijelaskan pada Subbab 4.12.

1. **`is_event_driven`** (boolean) **[fitur]** — bernilai benar apabila permintaan SKU digerakkan oleh pemesanan acara, khususnya akikah. Lima SKU bertanda benar, mencakup 4,37% baris. Kolom ini masuk ke model sebagai satu fitur di antara sekitar empat puluh dan **bukan** sebagai penyaring, sehingga penanda yang keliru hanya menurunkan kualitas fitur dan tidak membuang data. Kolom ini sekaligus menandai batas informasi yang tidak dapat diatasi: permintaan SKU digerakkan-acara ditentukan oleh pemesanan pelanggan dan bukan oleh pola historis, sehingga tidak ada fitur lag maupun *rolling* yang dapat memprediksinya.

2. **`demand_segment`** (untai teks) **[fitur melalui bentuk terkode]** — kelas Syntetos-Boylan pasangan, bernilai `smooth`, `erratic`, `intermittent`, atau `lumpy`, ditetapkan dari periode latih saja. Distribusinya disajikan pada Subbab 5.4. Berfungsi ganda sebagai variabel prediktor dan sebagai **sumbu pelaporan metrik**, yang penting secara metodologis karena galat global didominasi pasangan yang mayoritas nilainya nol.

3. **`fold_id`** (pecahan, 1–5 atau kosong) — nomor *fold* validasi *walk-forward* yang bulan validasinya memuat tanggal baris. Bernilai kosong pada 1.148.454 baris yang tidak termasuk bulan validasi mana pun. Bukan fitur, melainkan variabel pengendali eksperimen: kedua *adapter* meneruskannya apa adanya, dan kesetaraannya antar-*adapter* diverifikasi oleh pemeriksaan kontrak.

4. **`was_relocated`** (boolean) **[fitur]** — indikator yang menandai apakah `days_since_relocation` semula memiliki nilai. Diperlukan karena kolom tersebut diimputasi dengan 0,0, sedangkan 0,0 merupakan nilai sah yang berarti "relokasi terjadi tepat hari ini". Bernilai benar pada 15,68% baris.

5. **`has_baseline`** (boolean) **[fitur]** — indikator yang menandai apakah `baseline_ratio` semula memiliki nilai, yakni apakah pasangan memenuhi syarat kelayakan garis dasar. Bernilai benar pada 85,61% baris.

6. **`missing_history_count`** (bilangan bulat, 0–13) **[fitur]** — cacah variabel riwayat yang semula bernilai kosong pada baris bersangkutan. Karena banyaknya jendela yang gagal terbentuk merupakan fungsi monoton dari sejauh mana posisi baris berada di dalam segmennya, kolom ini sekaligus berfungsi sebagai **ukuran ordinal atas panjang riwayat yang tersedia**, tanpa memerlukan pengelompokan maupun pengurutan tambahan. Bernilai 0 pada 1.412.134 baris.

7. **`has_full_history`** (boolean) **[fitur]** — bernilai benar apabila `missing_history_count` sama dengan nol. Bernilai benar pada 93,98% baris. Disediakan berdampingan dengan butir sebelumnya karena membedakan "riwayat lengkap" dari "riwayat hampir lengkap" merupakan pemisahan yang sering berguna dan tidak perlu dipelajari model dari nilai cacahnya.

8. **`Kode Barang_idx`** (bilangan bulat) **[fitur]** — indeks bilangan bulat hasil pengodean `Kode Barang`, dengan pemetaan dibentuk dari periode latih dan dipersistensi ke `category_mapping.json`. Indeks 0 dicadangkan bagi token `<UNKNOWN>`.

9. **`Nama Cabang_idx`** (bilangan bulat) **[fitur]** — indeks hasil pengodean `Nama Cabang`.

10. **`Kategori Barang_idx`** (bilangan bulat) **[fitur]** — indeks hasil pengodean `Kategori Barang`.

11. **`kota_idx`** (bilangan bulat) **[fitur]** — indeks hasil pengodean `kota`.

12. **`hari_pengiriman_idx`** (bilangan bulat) **[fitur]** — indeks hasil pengodean `hari_pengiriman`.

13. **`branch_volume_tier_idx`** (bilangan bulat) **[fitur]** — indeks hasil pengodean `branch_volume_tier`.

14. **`demand_segment_idx`** (bilangan bulat) **[fitur]** — indeks hasil pengodean `demand_segment`.

Ketujuh kolom berindeks di atas menggantikan kolom kategorik aslinya sebagai variabel prediktor. Pengodean berupa indeks bilangan bulat dipilih menggantikan *one-hot encoding* langsung karena tiga alasan: kardinalitas `Nama Cabang` dan `Kode Barang` cukup tinggi sehingga *one-hot* akan membengkakkan dimensi masukan secara tidak perlu; model berbasis pohon menangani indeks kategorik secara efisien; dan LSTM dapat mengonsumsinya melalui lapisan *embedding* yang justru lebih ekspresif daripada representasi *one-hot*. Persistensi pemetaan bersifat wajib bagi kebutuhan inferensi berkala yang stabil, dengan alasan yang diuraikan pada Subbab 4.12 butir (e).

---

## 7. Pengendalian Kebocoran Informasi

Kebocoran informasi (*data leakage*) adalah masuknya informasi masa depan ke dalam fitur atau proses pelatihan, yang menyebabkan hasil evaluasi terlalu optimistis dan tidak dapat direproduksi pada penggunaan nyata. Bab ini mengonsolidasikan sembilan mekanisme pengendalian yang tersebar sepanjang pipeline, sehingga dapat dikutip sebagai satu kesatuan pada bagian metodologi naskah.

**(1) Pergeseran satu hari sebelum agregasi *rolling*.** Seluruh statistik *rolling* dihitung atas kuantitas yang telah digeser satu hari, sehingga jendela bagi baris $t$ mencakup $(t{-}w \ldots t{-}1)$ dan tidak pernah menyertakan kuantitas hari $t$ itu sendiri. Tanpa mekanisme ini, fitur tidak akan dapat dihitung pada saat prediksi.

**(2) Pembekuan statistik cabang dari periode latih.** Ketiga statistik cabang dihitung setelah data disaring menjadi baris bertanggal sebelum batas periode uji. Penyaringan dilakukan **sebelum** agregasi apa pun, bukan sesudahnya, sehingga permintaan Desember suatu cabang tidak pernah memengaruhi fitur cabang tersebut.

**(3) Pembekuan garis dasar pencilan dari periode latih.** Median pasangan yang menjadi dasar penetapan lonjakan dihitung atas periode latih saja, sehingga baris uji dibandingkan terhadap garis dasar yang tidak pernah melihat periode uji.

**(4) Penetapan segmen permintaan dari periode latih.** Klasifikasi Syntetos-Boylan dihitung atas periode latih saja. Menurunkannya dari deret penuh akan membocorkan perilaku pascabatas menjadi fitur yang dipelajari model.

**(5) Pembentukan pemetaan kategori dan parameter standardisasi hanya atas data latih, per *fold*.** Suatu nilai kategorik yang baru muncul setelah batas tidak masuk ke pemetaan sama sekali. Parameter standardisasi dibentuk ulang pada setiap *fold* dari baris latih *fold* tersebut, sehingga statistik Desember tidak merembes ke *fold* Juli.

**(6) Target selalu berupa jendela maju ketat.** Seluruh variabel target menjumlahkan atau mengambil nilai pada $t+1$ dan sesudahnya, tidak pernah menyertakan hari berjalan.

**(7) *Purging* pada setiap batas temporal.** Baris latih yang jendela targetnya melampaui batas dibuang, baik pada pembagian latih–uji maupun pada setiap batas *fold*. Tanpa mekanisme ini, label baris-baris tersebut sebagiannya tersusun dari permintaan periode yang hendak dievaluasi.

**(8) Pengelompokan seluruh operasi bergeser menurut (pasangan, segmen).** Lag, *rolling*, pergeseran target, dan jendela LSTM tidak pernah melintasi periode gerai tidak beroperasi, sehingga dua sisi masa tutup tidak diperlakukan sebagai hari yang berurutan.

**(9) Pengecualian variabel yang mengandung informasi hari berjalan.** Kolom `baseline_ratio` dan `is_spike` dikeluarkan dari daftar fitur karena diturunkan dari kuantitas hari $t$. Pengecualian ini menjaga agar frasa "diketahui pada saat prediksi" bermakna tunggal di dalam satu baris.

Satu kolom, yaitu `branch_age_days`, aman secara inheren dan tidak memerlukan mekanisme khusus, sebab hanya membaca tanggal kemunculan pertama cabang itu sendiri.

---

## 8. Verifikasi dan Pengujian

### 8.1 Asersi penjaminan mutu

Fungsi `run_qa_checks()` menjalankan sebelas asersi atas tabel fitur, dan dipanggil baik dari jalur skrip maupun jalur *notebook* sehingga kedua jalur terverifikasi setara. Kegagalan salah satu asersi menghentikan pipeline dengan pesan berbahasa Indonesia yang menyebutkan jenis kegagalannya.

1. Tidak terdapat `Kuantitas` bernilai negatif.
2. Tidak terdapat baris duplikat pada kunci (kode barang, cabang, tanggal).
3. `Kuantitas_capped` tidak pernah melampaui `Kuantitas`.
4. `target_lead_time_cumulative_capped` tidak pernah melampaui `target_lead_time_cumulative`.
5. Tidak terdapat cabang dengan nilai `kota` sama dengan `Unknown`.
6. Tidak terdapat cabang tanpa nilai `kawasan`.
7. Tidak terdapat cabang yang memetakan ke lebih dari satu kota.
8. Tidak terdapat baris yang jatuh di dalam interval penutupan yang tercatat.
9. `segment_id` dimulai dari 1 pada setiap pasangan.
10. `segment_id` bersifat kontinu per pasangan, yakni cacah nilai uniknya sama dengan nilai maksimumnya.
11. Tidak terdapat lubang tanggal di dalam satu segmen, yakni selisih antartanggal berurutan selalu satu hari.

Asersi kesembilan hingga kesebelas menjaga invarian kepadatan yang menjadi landasan seluruh operasi pergeseran. Apabila invarian ini rusak, nilai `lag_7` tidak lagi dijamin merujuk tujuh hari kalender sebelumnya, dan seluruh fitur riwayat kehilangan maknanya tanpa satu pun galat yang muncul.

Sebagai pemeriksaan tambahan yang tidak bersifat menghentikan, pipeline memunculkan peringatan atas setiap jeda transaksi cabang sepanjang 14 hari atau lebih yang tidak dijelaskan `outlet_closures.csv`. Ambang ini dikalibrasi terhadap data nyata: jeda jinak terpanjang adalah 7 hari, sedangkan penutupan terkonfirmasi terpendek adalah 13 hari. Perlu ditegaskan bahwa **ambang ini menangkap kandidat, bukan mendefinisikan penutupan** — jeda di bawah 14 hari pun dapat merupakan penutupan sungguhan, dan hanya berkas konfigurasi yang memutuskan.

### 8.2 Strategi pengujian unit

Kode prapemrosesan dikembangkan dengan pendekatan *test-driven development*, yakni pengujian ditulis dan dipastikan gagal terlebih dahulu sebelum implementasi. Terdapat satu berkas pengujian per modul pipeline, dengan total **442 pengujian yang seluruhnya lulus** pada eksekusi verifikasi dokumen ini.

Lima kelompok pengujian dirancang khusus untuk melindungi keputusan metodologis yang diuraikan pada bab-bab sebelumnya.

*Pengujian antikebocoran.* Klasifikasi segmen permintaan harus menghasilkan label yang identik antara masukan berupa data latih saja dan masukan berupa data penuh — ketidaksamaan hasil membuktikan adanya kebocoran. Penetapan *fold* tidak boleh menempatkan tanggal validasi di dalam rentang latih *fold* yang sama. Parameter standardisasi harus benar-benar dibentuk ulang pada setiap *fold*.

*Pengujian imputasi.* Nilai kosong pada variabel jarak peristiwa harus menjadi sentinel dan **bukan** nol, sebab nol pada kolom tersebut memiliki makna yang sah.

*Pengujian kontrak antar-*adapter*.* Kedua *adapter* harus mengembalikan himpunan kunci (pasangan, tanggal) yang identik. Pengujian ini disertai kasus negatif: perusakan yang disengaja atas salah satu *adapter* **harus** menggagalkan asersi, sehingga terbukti bahwa pemeriksaan kontrak benar-benar berfungsi dan bukan sekadar selalu lulus.

*Pengujian stabilitas pemetaan.* Nilai kategorik yang tidak dikenali harus memetakan ke `<UNKNOWN>` tanpa menggeser indeks yang telah diterbitkan sebelumnya.

*Pengujian pemisahan target dan pemangkasan.* Nilai `target_h*` pada baris yang terpangkas harus tetap sama dengan nilai mentah yang digeser, yang membuktikan bahwa pemangkasan tidak merembes ke variabel target.

### 8.3 Reproduksi

Seluruh hasil pada dokumen ini dapat direproduksi dengan perintah berikut, dijalankan dari akar repositori.

```bash
# Pipeline prapemrosesan data, tahap 1–12
.venv/bin/python3 -m utils.data_preprocessing.prepare_forecast_data

# Prapemrosesan pemodelan, tahap 13–14
.venv/bin/python3 -m utils.modelling.modeling_prep

# Melalui notebook
jupyter nbconvert --to notebook --execute --inplace \
  notebook/data_processing.ipynb notebook/train_test_split.ipynb \
  notebook/modeling_prep.ipynb

# Seluruh pengujian unit
.venv/bin/python3 -m unittest discover -p "test_*.py" -v
```

Lingkungan yang dipakai adalah Python 3.9.6 dengan pustaka `pandas`, `pyarrow`, dan `holidays` sebagaimana tercantum pada `requirements.txt`.

---

## 9. Keterbatasan Metodologis

Bab ini memuat keterbatasan yang melekat pada rancangan prapemrosesan dan tidak dapat dihilangkan oleh perbaikan teknis. Uraian lengkap tersedia pada `docs/batasan-penelitian.md`.

### 9.1 Keterbatasan yang bersumber dari data

**Sumbu waktu pengambilan, bukan pemesanan.** Sebagaimana diuraikan pada Subbab 2.5, seluruh deret berjalan pada sumbu waktu pengambilan barang. Buku pesanan tidak tersedia dan tidak akan tersedia, sebab sistem sengaja tidak merekamnya. Konsekuensinya, model bersaing dengan pengetahuan operasional yang secara struktural lebih baik daripada yang tersedia baginya.

**Langit-langit informasi pada SKU digerakkan-acara.** Permintaan lima SKU bertanda `is_event_driven` ditentukan oleh pemesanan pelanggan dan bukan oleh pola historis. Tidak ada fitur lag maupun *rolling* yang dapat memprediksinya. Hal ini harus dinyatakan sebagai batas data, bukan sebagai kegagalan model.

**Periode uji sepanjang satu bulan yang secara musiman atipikal.** Desember 2025 merupakan satu-satunya periode uji terkunci. Mitigasinya berupa validasi *walk-forward* lima *fold* untuk pemilihan model, dengan Desember dipakai semata-mata untuk pelaporan angka akhir.

**Dua tahun data.** Cakupan 2024–2025 hanya menyediakan dua pengamatan per bulan kalender, sehingga musiman tahunan tidak dapat diestimasi secara meyakinkan.

### 9.2 Keterbatasan yang bersumber dari keputusan prapemrosesan

**Pasangan yang tersingkir oleh ambang riwayat minimum.** Sebanyak 842 pasangan tersingkir oleh ambang 60 hari dan **tidak memperoleh ramalan sama sekali**; strategi *fallback* bagi pasangan tersebut belum ditetapkan. Satu cabang, yakni KY073 Cilebut yang buka 19 Desember 2025, tersingkir seluruhnya karena tidak memiliki satu hari pun sebelum batas; cabang ini akan masuk dengan sendirinya pada pemutakhiran berikutnya tanpa perubahan kode.

**Ketiadaan kasus *cold start* pada himpunan uji.** Sebanyak 1.059 pasangan berhenti aktif sebelum Desember sehingga dilatih namun tidak pernah dievaluasi. Himpunan uji karenanya tidak memuat satu pun pasangan baru, sehingga kinerja model pada situasi *cold start* tidak terukur.

**Status 393 pasangan yang berhenti muncul pada kuartal keempat 2025.** Belum dapat dipastikan apakah merupakan penghentian penjualan yang sesungguhnya atau celah pelaporan. Saat ini diperlakukan sebagai penghentian sesungguhnya.

**Lima tanggal relokasi berupa perkiraan batas bawah.** Bagi lima gerai, kode lama tidak pernah berhenti muncul pada data sehingga tanggal pastinya tidak dapat diturunkan. Kelimanya memakai tanggal terakhir kemunculan kode lama sebagai proksi, sehingga nilai `days_since_relocation` bagi baris prarelokasinya bertanda benar namun besarannya terlalu kecil.

**Cakupan tabel kalender terbatas pada 2024–2025.** Pipeline akan gagal secara eksplisit apabila dijalankan atas data 2026 sebelum tabel Ramadan, Idulfitri, dan Iduladha diperluas. Kegagalan ini disengaja dan lebih baik daripada memproses data dengan seluruh penanda bernilai salah.

**Ketiadaan penanda pemisah antara permintaan reguler dan komponen pesanan.** Sepasang target disediakan sebagai pendekatan, namun pemisahan yang sesungguhnya tidak dapat dilakukan dari data yang ada.

---

## 10. Glosarium dan Rujukan

### 10.1 Glosarium

| Istilah | Definisi |
|---|---|
| **ADI** (*Average Demand Interval*) | Rerata jarak hari antarpermintaan bukan-nol; salah satu dari dua sumbu klasifikasi Syntetos-Boylan |
| ***Adapter*** | Lapisan tipis yang mengubah tabel fitur bersama menjadi bentuk masukan spesifik suatu algoritma |
| **CV²** | Kuadrat koefisien variasi kuantitas bukan-nol; sumbu kedua klasifikasi Syntetos-Boylan |
| ***Erratic*** | Kelas Syntetos-Boylan dengan ADI rendah namun CV² tinggi: sering bergerak, besarannya bervariasi tajam |
| ***Fold*** | Satu iterasi validasi silang; pada penelitian ini satu bulan validasi beserta seluruh data sebelumnya sebagai data latih |
| ***Intermittent*** | Kelas Syntetos-Boylan dengan ADI tinggi namun CV² rendah: jarang bergerak, besarannya relatif seragam |
| ***Leakage*** (kebocoran informasi) | Masuknya informasi masa depan ke dalam fitur atau proses pelatihan, menyebabkan evaluasi terlalu optimistis |
| **Lead time** | Jumlah hari dari tanggal transaksi hingga hari pengiriman berikutnya |
| ***Lumpy*** | Kelas Syntetos-Boylan dengan ADI dan CV² sama-sama tinggi: jarang bergerak dan besarannya bervariasi tajam; kelas tersulit |
| **Panel padat** | Tabel dengan satu baris per pasangan per hari kalender, hari tanpa transaksi diisi nol |
| **Pasangan** (*pair*) | Kombinasi (kode barang, cabang); satu deret waktu tunggal |
| ***Pinball loss*** | Fungsi kerugian regresi kuantil; bersifat asimetris antara prediksi kurang dan prediksi lebih |
| ***Purging*** | Pembuangan baris latih yang jendela targetnya melampaui batas temporal, guna mencegah label melintasi batas |
| **Segmen** | Blok tanggal aktif berkelanjutan milik satu pasangan, dipisahkan oleh periode gerai tidak beroperasi; ditandai `segment_id` |
| **Sentinel** | Nilai khusus yang menandai "tidak berlaku" tanpa menyamar sebagai observasi yang sah |
| ***Smooth*** | Kelas Syntetos-Boylan dengan ADI dan CV² sama-sama rendah: sering bergerak dengan besaran seragam; kelas termudah |
| ***Walk-forward*** | Validasi deret waktu dengan jendela latih yang membesar dan periode validasi selalu berada di masa depan |

### 10.2 Rujukan internal

| Berkas | Isi |
|---|---|
| `docs/dokumentasi-preprocessing-id.md` | Catatan rekayasa terperinci, memuat riwayat keputusan desain beserta alternatif yang ditolak |
| `docs/pipeline-overview.md` | Ikhtisar keempat belas tahap dalam bahasa Inggris |
| `docs/batasan-penelitian.md` | Uraian lengkap batasan penelitian, butir B-1 hingga B-9 |
| `docs/todolist-proyek.md` | Daftar konfirmasi pemilik data beserta statusnya |
| `docs/outlet_relocation_notes.md` | Catatan relokasi cabang beserta tingkat kepastian tanggalnya |
| `docs/superpowers/specs/2026-07-18-merge-dataset-design.md` | Spesifikasi tahap penggabungan |
| `docs/superpowers/specs/2026-07-21-forecast-data-prep-design.md` | Spesifikasi rekayasa fitur |
| `docs/superpowers/specs/2026-07-23-outlet-location-features-design.md` | Spesifikasi fitur lokasi gerai |
| `docs/superpowers/specs/2026-08-08-outlier-handling-design.md` | Spesifikasi penanganan pencilan |
| `docs/superpowers/specs/2026-08-08-lead-time-integration-design.md` | Spesifikasi integrasi lead time |
| `docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md` | Spesifikasi prapemrosesan pemodelan |
| `docs/superpowers/specs/2026-08-15-outlet-lifecycle-handling-design.md` | Spesifikasi penanganan siklus hidup gerai |

### 10.3 Rujukan metodologis

Klasifikasi permintaan yang dipakai pada Subbab 4.12 mengikuti Syntetos, A. A., & Boylan, J. E. (2005), *The accuracy of intermittent demand estimates*, International Journal of Forecasting, 21(2), 303–314, beserta ambang ADI = 1,32 dan CV² = 0,49 yang lazim dipakai dalam literatur turunannya.

Prosedur *purging* pada batas temporal mengikuti praktik baku validasi *walk-forward* untuk target yang jendelanya bertumpang tindih dengan batas pembagian data.
