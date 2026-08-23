# Alokasi kuantil tersegmentasi di bawah batasan service level agregat — design

## Status

**Draf, menunggu konfirmasi ulang pemilik data** (lihat "Pertanyaan terbuka").
Tidak mengubah satu baris kode pun sampai konfirmasi itu diterima secara
tertulis. Spec ini adalah rencana lanjutan yang **hanya dikerjakan setelah**
pemenang di antara Random Forest/XGBoost/LSTM ditetapkan lewat protokol §19
`metodologi-pemodelan-dan-pemilihan-model.md` — bukan bagian dari perbandingan
ketiga model itu sendiri.

**Pembaruan 2026-08-24.** Dua hal berubah sejak draf ini ditulis:

1. Reinterpretasi B-9 (pertanyaan terbuka nomor 1) **sudah tertulis** di
   `docs/batasan-penelitian.md` sebagai "Klarifikasi lanjutan (2026-08-22)" di
   bawah butir "Tingkat layanan target". Konfirmasi tertulis yang diminta sudah
   ada; yang masih menahan spec ini adalah pertanyaan terbuka nomor 2, 3, dan 4.
2. **Bagian 4 (perluasan multi-kuantil) sudah selesai lebih awal**, diwarisi
   dari migrasi evaluasi multi-kuantil — lihat catatan di Bagian 4 dan tabel
   urutan pengerjaan yang sudah diperbarui.

## Purpose

Model produksi saat ini memprediksi satu titik kuantil (0,9) yang seragam
untuk seluruh SKU, sesuai keputusan pemilik data 2026-08-16 yang tercatat di
dua tempat independen (`batasan-penelitian.md` B-9 dan
`2026-08-12-modeling-preprocessing-design.md` baris 92). Keputusan itu
menghasilkan kekurangan stok berkurang 73–76% dengan ongkos kelebihan stok
2,5–2,8× lipat (`docs/hasil-modeling-{rf,xgb,lstm}.md` §1) — pertukaran yang
secara matematis melekat pada kuantil 0,9 yang sama untuk setiap item,
terlepas dari perishability atau volatilitas demand-nya.

Spec ini merancang lapisan alokasi kuantil **per segmen** (kategori barang,
`demand_segment`, dan tier volume cabang) yang tetap menghormati janji
service level 0,9, dengan syarat janji itu dipenuhi secara **agregat di
level pengiriman**, bukan per SKU secara individual. Tujuannya mengurangi
beban kelebihan stok pada item yang mahal disimpan berlebih (terutama barang
cepat rusak) tanpa mengorbankan tingkat layanan keseluruhan yang sudah
dijanjikan ke bisnis.

## Reinterpretasi keputusan B-9 — bukan pembatalan

Kalimat keputusan yang tercatat di kedua sumber ("head office ships every
item in one consignment, so one service level governs the whole delivery")
secara literal berbicara soal **satu governance pengiriman**, bukan secara
eksplisit melarang variasi kuantil input di balik layar selama hasil
agregatnya tetap memenuhi janji 0,9. Pemilik data mengonfirmasi ulang
penafsiran ini secara langsung (percakapan proyek, 2026-08-22): yang
dimaksud "satu service level" adalah target agregat, karena setiap item
punya tren pasar berbeda, bahkan item yang sama bisa punya tren berbeda
antar cabang.

Spec ini karenanya **tidak membalikkan** B-9 — janji 0,9 ke bisnis tetap
utuh — melainkan mengubah cara janji itu dipenuhi secara teknis, dari
"kuantil 0,9 di setiap baris" menjadi "kuantil rata-rata tertimbang 0,9 di
level pengiriman, dengan variasi per segmen di baliknya". Bab "Pertanyaan
terbuka" di bawah tetap meminta konfirmasi tertulis eksplisit atas
reinterpretasi ini sebelum implementasi dimulai, terpisah dari persetujuan
lisan yang sudah diperoleh, mengikuti standar dokumentasi proyek ini yang
selalu menyertakan tanggal dan bentuk konfirmasi pemilik data untuk setiap
keputusan kunci.

## Non-goals

- Mengganti model pemenang atau mengubah kriteria pemilihan model (§17–18
  `metodologi-pemodelan-dan-pemilihan-model.md`) — spec ini berjalan
  **sesudah** pemenang dibekukan.
- Fitur pengurangan hasil prediksi dengan stock opname — di luar cakupan,
  ditunda sesuai arahan sebelumnya.
- Membangun peta resep/bill-of-materials untuk memecah harga menu ke
  komponen item — proyek data terpisah, hanya disinggung sebagai motivasi
  berkas konfigurasi di bawah.
- Mengubah `target_lead_time_cumulative` atau mekanisme purging/leakage yang
  sudah ada — seluruhnya dipakai apa adanya.

## Latar: mengapa satu kuantil menghasilkan overstock berat pada segmen tertentu

Ketiga model kuantil 0,9 (RF, XGBoost, LSTM) memangkas kekurangan stok
73–76% dengan ongkos kelebihan stok 2,5–2,8× (§16.4
`metodologi-pemodelan-dan-pemilihan-model.md`). Pertukaran ini seragam
karena kuantilnya seragam — item Barang Semi FG WIP-2 (Ayam Kebuli, Nasi
Kebuli, Sambal) yang masa simpannya harian menanggung beban kelebihan stok
yang sama proporsinya dengan Packaging yang praktis tidak kedaluwarsa.
Mengizinkan kuantil berbeda per segmen, dikalibrasi terhadap rasio biaya
understock/overstock masing-masing, dapat menekan beban tersebut pada
segmen yang mahal disimpan berlebih, sambil menaikkannya sedikit pada
segmen yang murah disimpan berlebih — dengan syarat rata-rata tertimbangnya
tetap kembali ke 0,9.

## Pendekatan

### 1. Dimensi segmentasi — sudah tersedia, tidak perlu preprocessing baru

Tiga dimensi berikut sudah ada sebagai kolom di `model_input.parquet`,
dihitung dari periode latih saja (aman dari kebocoran), dan sudah berperan
ganda sebagai fitur model *dan* sumbu pelaporan evaluasi:

| Dimensi | Kolom sumber | Granularitas |
|---|---|---|
| Kategori/perishability | `Kategori Barang_idx` | per SKU |
| Volatilitas demand | `demand_segment` | per **pasangan** (item × cabang) — sudah menjawab kebutuhan "tren pasar beda antar cabang untuk item yang sama" |
| Volume/variabilitas cabang | `branch_volume_tier`, `branch_demand_cv` | per cabang |

Segmen alokasi kuantil didefinisikan sebagai kombinasi
`(Kategori Barang, demand_segment)`, dengan `branch_volume_tier` sebagai
modifier opsional bila hasil awal masih terlalu kasar (lihat "Simulasi
kalibrasi" di bawah). Tidak ada kolom baru yang perlu ditambahkan ke
pipeline preprocessing untuk dimensi ini.

### 2. Berkas konfigurasi baru: `dataset/item_cost_margin.csv`

Mengikuti pola berkas konfigurasi yang sudah ada (`event_driven_items.csv`,
`outlet_closures.csv`) — dipelihara manual oleh tim SCM, divalidasi eksplisit
oleh pipeline, bukan dari pencocokan tebakan.

```
Kode Barang;unit_cost;unit_margin;shelf_life_days;salvage_value_ratio;cost_source;cost_confidence;last_updated;shelf_life_rank_override
```

| Kolom | Tipe | Wajib | Keterangan |
|---|---|---|---|
| `Kode Barang` | teks | Ya | Kunci penghubung, harus cocok dengan `Kode Barang` hasil normalisasi Tahap 3 |
| `unit_cost` | pecahan | Ya | Biaya produksi/pengadaan per unit `Satuan` barang tersebut |
| `unit_margin` | pecahan | Ya | Margin kotor per unit; proksi Cu (biaya understock) |
| `shelf_life_days` | bilangan bulat | Ya | Perkiraan masa simpan dalam hari. Item praktis tidak kedaluwarsa (Packaging, RM kering) diisi sentinel besar (365), **bukan dikosongkan** — pola sentinel yang sama dengan `days_until_ramadan` (§4.12(d) `metodologi-preprocessing.md`) |
| `salvage_value_ratio` | pecahan, 0–1 | Tidak, default 0 | Proporsi nilai yang masih terselamatkan bila tidak terjual sebelum rusak (mis. sisa nasi diolah ulang) |
| `cost_source` | teks kategorik | Ya | `alokasi_resep` \| `estimasi_tim_scm` \| `harga_pasar_langsung` |
| `cost_confidence` | teks kategorik | Ya | `tinggi` \| `sedang` \| `rendah` |
| `last_updated` | tanggal | Ya | Untuk mendeteksi entri yang sudah usang |
| `shelf_life_rank_override` | bilangan bulat 1–7, atau kosong | Tidak, default kosong | **Kolom baru.** Jika terisi, menggantikan `shelf_life_rank` kategori (Bagian 3, Proksi A) untuk SKU ini secara spesifik — dipakai ketika suatu SKU secara fisik tidak representatif terhadap kategorinya (lihat "Override tingkat SKU" di Bagian 3). Kosong berarti SKU tetap mewarisi peringkat kategorinya seperti biasa. Independen dari `cost_confidence`/`unit_cost` — SKU boleh punya `shelf_life_rank_override` terisi meski `unit_cost` masih kosong, karena keduanya menjawab pertanyaan berbeda (masa simpan fisik vs biaya ekonomi) |

**Baris tidak wajib lengkap semua SKU.** SKU yang belum punya baris, atau
barisnya ada tapi `cost_confidence = rendah` di bawah ambang yang
dikonfigurasi, jatuh ke jalur proksi (Bagian 3). Ini bukan mode
"semua-atau-tidak-sama-sekali" — keputusan dibuat **per SKU**, sejalan
dengan permintaan agar mekanismenya berjalan dinamis mengikuti ketersediaan
data.

**Validasi wajib saat dimuat** (mengikuti pola fail-loud pipeline yang
sudah ada, mis. `parse_delivery_days`, `load_closures`):

- `Kode Barang` yang tidak dikenali pipeline → error, bukan diabaikan diam-diam.
- `unit_margin < 0` atau `shelf_life_days <= 0` → error.
- `cost_confidence` di luar tiga nilai yang ditetapkan → error.
- Berkas tidak ada sama sekali → bukan error; seluruh SKU jatuh ke jalur
  proksi (berkas ini sepenuhnya opsional, sama seperti `outlet_closures.csv`).

**Validasi wajib saat memuat `shelf_life_rank_by_category.csv`** (berkas
kedua yang menyusun jalur proksi, mengikuti pola fail-loud yang sama):

- `Kategori Barang` pada baris data yang **tidak ditemukan** di berkas ini →
  error saat simulasi dijalankan, bukan default diam-diam ke peringkat
  tengah. Ini konsisten dengan pola `event_driven_items.csv` (§4.12 butir a
  `metodologi-preprocessing.md`): SKU/kategori baru yang muncul di
  pemutakhiran data harus diklasifikasikan eksplisit oleh manusia, bukan
  diasumsikan. Kasus konkret yang divalidasi mekanisme ini: reklasifikasi
  WIP-2 → Barang Jadi (FG) yang dikonfirmasi retroaktif (lihat "Pertanyaan
  terbuka" nomor 5) — seandainya validasi ini sudah aktif sebelum konfirmasi
  diterima, ia akan menangkap ketidaksesuaian tersebut secara otomatis
  alih-alih bergantung pada pemeriksaan manual seperti yang baru saja
  dilakukan.
- `shelf_life_rank` di luar rentang 1 sampai jumlah kategori aktif, atau
  ada duplikat `Kategori Barang` → error.

### 3. Jalur proksi — dipakai per SKU, bukan per keseluruhan berkas

Untuk SKU tanpa entri biaya (atau dengan `cost_confidence = rendah`),
critical ratio didekati dari kombinasi dua sumber, tanpa memerlukan data
harga:

**Proksi A — peringkat ordinal masa simpan**, digabung dengan **Proksi B —
elisitasi kualitatif tim SCM** dalam satu berkas konfigurasi,
`dataset/shelf_life_rank_by_category.csv` (satu baris per kategori, bukan
per SKU, supaya mudah diisi tim SCM tanpa menunggu data per-item):

```
Kategori Barang;shelf_life_rank;elisitasi_skala_1_5;cost_source;cost_confidence;last_updated;catatan
```

| Kolom | Tipe | Keterangan |
|---|---|---|
| `Kategori Barang` | teks | Delapan nilai kanonik pipeline: `Packaging`, `Barang Umum`, `Bahan Baku (RM)`, `Barang Dalam Process (WIP-1)`, `Barang Semi FG (WIP-2)`, `Barang Jadi (FG)`, `Minuman - FG`, `Snack (FG)` |
| `shelf_life_rank` | bilangan bulat 1–8 | 1 = masa simpan terpanjang, 8 = tersingkat. Peringkat rendah memetakan ke critical ratio di atas 0,9 (kuantil dinaikkan); peringkat tinggi ke critical ratio di bawah 0,9. Fungsi pemetaan peringkat → critical ratio adalah satu parameter yang dikalibrasi bersama λ (Bagian 4), bukan tabel tetap |
| `elisitasi_skala_1_5` | bilangan bulat 1–5 | Jawaban tim SCM atas pertanyaan "seberapa kali lebih merugikan kehabisan dibanding kelebihan stok yang terbuang", dipakai sebagai pengali tambahan atas `shelf_life_rank` ketika tersedia |
| `cost_source` | teks kategorik | `estimasi_umum` (penalaran awal tanpa masukan SCM), `estimasi_tim_scm`, atau `harga_pasar_langsung` — pola sama dengan `item_cost_margin.csv` |
| `cost_confidence` | teks kategorik | `tinggi` \| `sedang` \| `rendah`, dipakai jalur keputusan dinamis yang sama seperti `item_cost_margin.csv` |
| `last_updated` | tanggal | Tanggal baris terakhir diperbarui |
| `catatan` | teks | Alasan penalaran atau dasar angka, khususnya penting untuk baris `cost_source = estimasi_umum` supaya tim SCM tahu persis apa yang perlu dikoreksi |

**Isian awal (estimasi umum, diperbarui 2026-08-22 berdasarkan
`dataset/pemetaan-sku-per-kategori.csv`, belum divalidasi tim SCM)** —
berbeda dari draf pertama, isian ini sekarang bersandar pada bukti SKU dan
volume nyata per kategori, bukan penalaran nama kategori semata:

| Kategori Barang | shelf_life_rank | elisitasi_skala_1_5 | cost_confidence | Dasar |
|---|---:|---:|---|---|
| Packaging | 1 | 1 | sedang | 21 SKU (Lunch Box, Cup Sambal, Loyang, dll), non-pangan |
| Barang Umum | 2 | 1 | sedang | 1 SKU (Kalender), non-pangan |
| Bahan Baku (RM) | 3 | 2 | sedang | 1 SKU (Saffron Basmati), bumbu kering |
| Snack (FG) | 4 | 2 | sedang | 7 SKU (Kentang Mustofa, Kebab), kemasan/beku |
| Minuman - FG | 5 | 3 | sedang | 17 SKU, campuran botol pabrik dan Meet Jelly |
| Barang Dalam Process (WIP-1) | 6 | 4 | sedang | 1 SKU (Rabeg), komponen setengah jadi |
| Barang Jadi (FG) | 7 | 5 | sedang | 22 SKU, **tapi 96% volume kategori berasal dari 3 SKU masakan matang harian** (Sambal, Nasi Kebuli, Ayam Kebuli) — lihat "Catatan heterogenitas" di bawah |

**Skema tujuh kategori ini dikonfirmasi pemilik data (2026-08-22) sebagai
kategori update terbaru**, menggantikan skema delapan kategori (termasuk
`Barang Semi FG (WIP-2)` sebagai kategori tersendiri) yang tercatat di
`metodologi-preprocessing.md` §2.2/§6.9. Pertanyaan terbuka nomor 5 di
bawah, semula berstatus kritis, **ditutup** dengan resolusi ini.

**Catatan heterogenitas — `Barang Jadi (FG)`, dan mekanisme override yang
menanganinya.** Kategori ini secara volume didominasi tiga SKU masakan
matang harian (`FGS-00001` Ayam Kebuli, `FGS-00004` Nasi Kebuli, `FGS-00005`
Sambal — 96% dari 12,2 juta unit kategori, `pct_hari_nol` mendekati 0%),
yang membenarkan peringkat kategori 7 (tersingkat) secara volume-tertimbang.
Namun kategori yang sama juga memuat item jauh lebih awet secara fisik.
Alih-alih membiarkan minoritas ini ikut peringkat mayoritas, kolom
`shelf_life_rank_override` di `item_cost_margin.csv` (Bagian 2) dipakai
untuk memberi peringkat individual pada SKU-SKU berikut, diverifikasi
langsung dari `dataset/pemetaan-sku-per-kategori.csv`:

| Kode Barang | Nama | Satuan | `shelf_life_rank_override` | Alasan |
|---|---|---|---:|---|
| `FGS-00054` | India Salaam Basmati Rice @1kg | Kg | 3 | Beras mentah, setara Bahan Baku (RM) |
| `FGS-00012` | Samosa Beef Original (RM) | PCS | 4 | Produk beku kemasan, bukan masakan matang harian |
| `FGS-00013` | Samosa Beef Spicy (RM) | PCS | 4 | Sama seperti di atas |
| `FGS-00011` | Saus Extra Delmonte @8gr | PCS | 3 | Saus sachet kemasan pabrik |
| `FGS-00065` | Saus Tomat Delmonte @8gr | PCS | 3 | Sama seperti di atas |
| `FGS.00055` | Saus Lemon | Gr | 4 *(sementara, perlu konfirmasi)* | Belum jelas kemasan pabrik atau racikan sendiri — beri peringkat sementara yang konservatif sampai dikonfirmasi tim SCM |
| `FGS.00056` | Saus Spicy | Gr | 4 *(sementara, perlu konfirmasi)* | Sama seperti di atas |

Ketujuh SKU ini mewakili sekitar 133.500 unit (≈4% volume kategori). SKU
lain di `Barang Jadi (FG)` (termasuk `FGS.00048` Kambing Oven yang sudah
dikonfirmasi slow mover di B-9, tapi tetap masakan matang) **tidak** diberi
override, sehingga tetap mewarisi peringkat kategori 7 secara default. Dua
saus tanpa merek (`FGS.00055`, `FGS.00056`) sengaja ditandai "sementara"
karena namanya tidak memberi petunjuk sekuat "Saus Extra Delmonte" (jelas
produk pabrik bermerek) — perlu dikonfirmasi tim SCM apakah kemasan pabrik
atau racikan dapur sebelum dianggap final.

Pertanyaan terbuka nomor 6 (di bawah) **ditutup** dengan mekanisme dan
tabel ini — bukan lagi sekadar diusulkan, tapi sudah didesain konkret dan
siap dipakai begitu `item_cost_margin.csv` diisi.

Berkas lengkap dengan kolom `catatan` berisi rincian SKU dan volume per
kategori tersedia di `dataset/shelf_life_rank_by_category.csv`.

**Keputusan dinamis per SKU** (inti dari permintaan Anda), sekarang tiga
tingkat setelah `shelf_life_rank_override` ditambahkan:

```
untuk setiap SKU:
    jika ada baris di item_cost_margin.csv DAN cost_confidence != "rendah":
        gunakan unit_cost/unit_margin/shelf_life_days langsung → critical ratio presisi
        cr_source = "presisi"
    lain jika ada shelf_life_rank_override terisi di item_cost_margin.csv:
        gunakan peringkat override ini, bukan peringkat kategori
            → critical ratio taksiran, tapi granularitas SKU
        cr_source = "override_kategori"
    selainnya:
        gunakan shelf_life_rank kategori (Proksi A) × elisitasi SCM jika ada (Proksi B)
            → critical ratio taksiran, granularitas kategori
        cr_source = "taksiran"
    catat cr_source pada output simulasi, sehingga hasil akhir bisa
    disaring/dibedakan menurut tingkat kepercayaan dan granularitasnya
```

Biaya presisi (`unit_cost`/`unit_margin`) tetap didahulukan di atas
override peringkat, karena biaya langsung selalu lebih akurat daripada
peringkat ordinal apa pun. `shelf_life_rank_override` mengisi celah di
antara keduanya: lebih presisi daripada peringkat kategori, tapi tidak
memerlukan data biaya yang belum tentu tersedia (lihat B-10) — cukup
penilaian fisik masa simpan per SKU, seperti tujuh SKU yang sudah
diidentifikasi di "Catatan heterogenitas" di atas.

Begitu suatu SKU mendapat entri biaya presisi (`cost_confidence` naik dari
`rendah`, atau baris baru ditambahkan), simulasi dijalankan ulang dan SKU
tersebut otomatis pindah jalur pada run berikutnya tanpa perubahan kode —
inilah yang membuatnya "berjalan dinamis mengikuti ketersediaan data"
seperti yang diminta.

### 4. Perluasan model: multi-kuantil, bukan model baru

> **DIWARISI — selesai lebih awal (2026-08-24).** Bagian ini **tidak lagi
> menjadi pekerjaan yang tersisa untuk spec ini.** Migrasi evaluasi
> multi-kuantil
> (`2026-08-22-model-comparison-refactor-migration.md`, mengikuti
> `2026-08-22-multi-quantile-evaluation-design.md`) memindahkan perluasan
> multi-kuantil ke **hulu**: ketiganya diperluas sebagai bagian dari kriteria
> K1 yang baru, **sebelum** pemenang ditetapkan, karena K1 sekarang adalah
> rata-rata pinball loss lintas `QUANTILE_SET`. Konsekuensinya, model mana pun
> yang menang **sudah otomatis punya kapabilitas multi-kuantil** begitu tangga
> K1–K3 selesai.
>
> Teks asli di bawah **sengaja tidak dihapus**, supaya jejak keputusannya tetap
> terbaca: alasan biaya "jalankan hanya untuk pemenang" masuk akal ketika
> multi-kuantil hanya melayani spec ini, dan berhenti masuk akal begitu
> multi-kuantil menjadi kriteria pemilihan model itu sendiri. Yang berubah
> adalah premisnya, bukan analisis teknis per arsitekturnya — tabel di bawah
> tetap berlaku persis apa adanya, dan sudah diterapkan ke ketiga spec model.
>
> Satu penyesuaian pada tabel: contoh `[0.6, 0.7, 0.8, 0.9, 0.95]` untuk
> XGBoost adalah ilustrasi dari sebelum `QUANTILE_SET` didefinisikan. Nilai yang
> berlaku sekarang adalah `QUANTILE_SET` Tahap A (19 titik, spasi 0,05).

Dijalankan pada arsitektur pemenang saja (§21
`metodologi-pemodelan-dan-pemilihan-model.md` sudah menetapkan pola ini
untuk SHAP; spec ini mengikuti pola yang sama untuk alasan biaya yang
sama — menjalankan multi-kuantil untuk ketiga arsitektur berarti membayar
ongkos penjelasan untuk model yang tidak akan dipakai).

Perluasan teknis per arsitektur, tanpa mengubah data prep:

| Model | Perubahan |
|---|---|
| Random Forest (`quantile-forest`) | Tidak perlu dilatih ulang — baca kuantil lain langsung dari distribusi empiris di daun yang sama |
| XGBoost (`reg:quantileerror`) | `quantile_alpha` diisi daftar (mis. `[0.6, 0.7, 0.8, 0.9, 0.95]`) — satu fit menghasilkan banyak kuantil sekaligus, didukung native sejak versi yang sudah dipakai proyek ini |
| LSTM | Head output diperluas dari satu neuron menjadi satu neuron per titik kuantil; loss total = jumlah pinball loss di seluruh titik |

**Jika RF atau LSTM yang menang** (bukan XGBoost seperti usulan §18), poin
tabel di atas tetap berlaku — perluasan ini tidak bergantung pada model
mana yang menang di tangga §17–18.

### 5. Simulasi kalibrasi λ

Dijalankan sebagai skrip simulasi terpisah di atas prediksi multi-kuantil
yang sudah ada (model tidak dilatih ulang untuk tiap kombinasi), sama
seperti hyperparameter search yang sudah dilakukan proyek ini menggunakan
mesin evaluasi bersama.

```
untuk setiap segmen (Kategori Barang × demand_segment):
    critical_ratio_dasar[segmen] = dari item_cost_margin.csv (presisi) atau proksi (taksiran)

untuk λ dalam grid (mis. 0.5 sampai 1.5, langkah 0.05):
    critical_ratio_disesuaikan[segmen] = clip(critical_ratio_dasar[segmen] * λ, batas_bawah, batas_atas)
    kuantil_terpilih[segmen] = titik kuantil terdekat yang tersedia dari prediksi multi-kuantil
    jalankan evaluasi walk-forward dengan kuantil per segmen ini
    catat: fill_rate agregat, days_of_supply overstock total, shortfall per segmen

pilih λ yang membawa fill_rate agregat kembali ke ~0.90,
lalu laporkan trade-off pada λ tersebut per segmen
```

`batas_bawah`/`batas_atas` pada clip mencegah segmen mana pun jatuh ke
kuantil ekstrem (mis. di bawah 0,5 atau di atas 0,99) akibat salah kalibrasi
proksi — pagar pengaman, bukan bagian dari logika inti.

### 6. Metrik evaluasi: days of supply menggantikan rupiah untuk sementara

Karena `unit_cost`/`unit_margin` mungkin sebagian besar kosong sampai peta
resep tersedia, metrik evaluasi utama simulasi ini adalah **days of supply**,
bukan rupiah — menyelesaikan sekaligus masalah satuan campur (Kg, Porsi,
Botol, PCS) yang sudah dicatat sebagai keterbatasan di
`docs/hasil-modeling-{rf,xgb,lstm}.md` §1:

```
days_of_supply_overstock(pasangan) = overstock_units(pasangan) / roll_mean_28(pasangan)
days_of_shortfall(pasangan)        = shortfall_units(pasangan) / roll_mean_28(pasangan)
```

`roll_mean_28` dipilih sebagai penyebut (bukan `roll_mean_7`) karena sudah
didokumentasikan sebagai "estimasi tingkat permintaan jangka menengah yang
paling stabil" (§6.7 `metodologi-preprocessing.md`), dan sudah tersedia
sebagai kolom jadi — tidak perlu turunan baru.

Begitu berkas biaya terisi memadai (ambang diusulkan: `cost_confidence`
bukan `rendah` untuk segmen yang mewakili ≥ 50% volume), metrik evaluasi
utama beralih ke rupiah, dan days of supply diturunkan menjadi metrik
pendamping — persis pola yang sudah dipakai proyek ini untuk MAE (§15.2
`metodologi-pemodelan-dan-pemilihan-model.md`: dilaporkan untuk konteks,
bukan kriteria keputusan).

### 7. Perluasan sumbu evaluasi

`GROUP_COLS` pada mesin evaluasi bersama (`walk_forward.py`) perlu
diperluas dari `{demand_segment, is_delivery_day}` menjadi juga mencakup
`Kategori Barang` dan `branch_volume_tier`, karena keduanya menjadi dimensi
keputusan langsung di sini, bukan sekadar potongan tambahan seperti
sekarang. Karena tidak ada spec tersendiri untuk `walk_forward.py`/
`evaluation.py` yang bisa dirujuk, perubahan ini perlu diverifikasi
langsung terhadap kode saat implementasi, bukan diasumsikan dari dokumen.

## Urutan pengerjaan relatif terhadap rencana kerja yang sudah ada

Menyisip di antara butir yang sudah tercatat di §21
`metodologi-pemodelan-dan-pemilihan-model.md`:

**Diperbarui 2026-08-24** mengikuti migrasi evaluasi multi-kuantil.

| # | Pekerjaan | Sumber |
|---|---|---|
| **0a** | **Migrasi evaluasi multi-kuantil: ubah spec RF/XGB/LSTM, jalankan ulang ketiga notebook, tulis ulang `hasil-modeling-*.md`, revisi §15–19 dan §21** | **baru — `2026-08-22-model-comparison-refactor-migration.md`** |
| 1–3 | Bekukan §18 (dengan angka multi-kuantil), buka test set Desember, tulis hasil | sudah direncanakan — sekarang bergantung pada 0a |
| **3a** | **Spec ini: alokasi kuantil tersegmentasi pada model pemenang** — Bagian 4 sudah diwarisi dari 0a, jadi tinggal Bagian 2, 3, 5, 6, 7 | **baru** |
| 4 | Dekomposisi harian (`target_h1`…`target_h4`) | sudah direncanakan — independen, tidak saling bergantung dengan 3a |
| 5 | SHAP untuk pemenang saja | sudah direncanakan |

Butir 3a tetap diletakkan setelah Desember dibuka, bukan sebelumnya. Alasannya
sekarang lebih sempit daripada sebelumnya, dan perlu dinyatakan ulang dengan
tepat:

- **Alasan yang sudah tidak berlaku.** Sebelumnya butir ini beralasan bahwa
  tangga §17 "butuh kriteria kuantil 0,9 seragam yang tetap sampai G0–K4
  selesai". Itu tidak lagi benar — K1 justru sekarang dinilai lintas 19 titik
  kuantil sekaligus. Kekhawatiran bahwa multi-kuantil membuat perbandingan
  tidak apple to apple juga tidak terbukti: ketiga model diperluas ke
  `QUANTILE_SET` yang **sama persis**, pada baris dan fold yang sama, sehingga
  perbandingannya justru lebih setara, bukan kurang.
- **Alasan yang masih berlaku.** Yang tetap harus dijaga adalah **kuantil yang
  dipakai memproduksi angka** selama G0–K4 harus seragam antar model. Ketiga
  model dinilai pada grid identik; yang belum boleh masuk adalah alokasi kuantil
  **berbeda per segmen**, karena itu memberi satu model peta alokasi yang tidak
  dimiliki dua model lain. Jadi yang ditunda adalah *alokasinya*, bukan
  *kapabilitas multi-kuantilnya*.

**Konsekuensi bersih: migrasi ini memperpendek jalur menuju segmentasi kuantil,
bukan memperpanjangnya.** Bagian 4 — yang sebelumnya adalah pekerjaan pertama
butir 3a dan mensyaratkan menunggu pemenang ditetapkan — kini sudah selesai
sebelum 3a dimulai. Simulasi kalibrasi λ (Bagian 5) bisa langsung berjalan
begitu pemenang ditetapkan, di atas prediksi multi-kuantil yang sudah ada,
tanpa satu pun fit tambahan untuk memperluas modelnya lebih dulu.

## Testing

Mengikuti konvensi TDD proyek ini (test ditulis dan dipastikan gagal lebih
dulu):

- `load_item_cost_margin`: baris dengan `Kode Barang` tak dikenal → error;
  `unit_margin` negatif → error; `shelf_life_rank_override` di luar 1–7 →
  error; berkas tidak ada → dict kosong, bukan error.
- `resolve_critical_ratio(sku, ...)`: SKU dengan entri `cost_confidence`
  tinggi/sedang → memakai jalur presisi (`cr_source = "presisi"`); SKU
  tanpa biaya presisi tapi punya `shelf_life_rank_override` terisi →
  memakai jalur override (`cr_source = "override_kategori"`), **bukan**
  peringkat kategorinya; SKU tanpa keduanya → memakai jalur proksi kategori
  (`cr_source = "taksiran"`). Urutan prioritas diverifikasi eksplisit:
  SKU dengan `cost_confidence` presisi **dan** `shelf_life_rank_override`
  terisi sekaligus → tetap memakai jalur presisi, override diabaikan
  (biaya presisi selalu didahulukan).
- Kasus nyata `FGS-00054` (India Salaam Basmati Rice): tanpa override,
  critical ratio-nya memakai peringkat kategori `Barang Jadi (FG)` = 7;
  dengan `shelf_life_rank_override = 3` terisi, critical ratio-nya berubah
  memakai peringkat 3 — perbedaan hasil ini diverifikasi eksplisit sebagai
  regression test, bukan hanya memastikan kolomnya terbaca.
- Regresi dinamis: SKU yang semula tanpa entri biaya, setelah baris
  ditambahkan ke `item_cost_margin.csv`, berpindah dari `cr_source = taksiran`
  ke `cr_source = presisi` pada run berikutnya tanpa perubahan kode.
- `days_of_supply_overstock`/`days_of_shortfall`: hasil tidak terdefinisi
  (`NaN`, bukan `inf` atau error) ketika `roll_mean_28` pasangan bernilai 0.
- Simulasi λ: fill_rate agregat pada λ=1.0 (sebelum penyesuaian) harus
  mereproduksi angka fill_rate model pemenang yang sudah dilaporkan di
  `docs/hasil-modeling-*.md` — pemeriksaan konsistensi terhadap hasil yang
  sudah ada, bukan angka baru yang berdiri sendiri.
- Multi-kuantil XGBoost: prediksi pada kuantil 0,9 dari model multi-kuantil
  harus sama (dalam toleransi numerik) dengan prediksi model kuantil-0,9
  tunggal yang sudah ada — memastikan perluasan tidak diam-diam mengubah
  hasil yang sudah divalidasi.

## Out of scope

- Peta resep/bill-of-materials untuk memecah harga menu ke komponen item.
- Perubahan pada `target_lead_time_cumulative`, mekanisme purging, atau
  kontrak adapter yang sudah ada.
- Business-rule cap pasca-model (opsi "Jalur C" yang dibahas sebelum
  reinterpretasi B-9 dikonfirmasi) — tidak diperlukan lagi karena
  pendekatan struktural (spec ini) sudah disetujui sebagai jalur utama.
- Mengubah kriteria atau hasil tangga pemilihan model §17–18.
- Automasi elisitasi tim SCM (Proksi B) menjadi survei berulang — untuk
  sekarang berupa berkas CSV yang diisi manual, sama seperti berkas
  konfigurasi lain di pipeline ini.

## Pertanyaan terbuka

1. **Konfirmasi tertulis atas reinterpretasi B-9** dari pemilik data,
   dengan tanggal, mengikuti standar dokumentasi proyek ini — persetujuan
   lisan yang sudah diperoleh dalam diskusi perlu dituangkan ke
   `batasan-penelitian.md` atau berkas setara sebagai pembaruan status B-9,
   bukan hanya hidup di riwayat percakapan.
2. **Ambang `cost_confidence` untuk berpindah dari days of supply ke
   rupiah sebagai metrik utama** — diusulkan 50% volume pada confidence
   bukan-rendah (Bagian 6), tapi ini angka awal yang perlu disepakati, bukan
   keputusan final.
3. **Siapa yang mengisi `shelf_life_rank_by_category.csv`** dan
   `item_cost_margin.csv` — tim SCM disebut di permintaan awal, tapi perlu
   dipastikan siapa secara spesifik (PIC) dan target waktu pengisian awal,
   supaya butir 3a di rencana kerja punya perkiraan mulai yang realistis.
4. **Apakah `branch_volume_tier` benar-benar diperlukan sebagai dimensi
   tambahan**, atau `Kategori Barang × demand_segment` saja sudah cukup
   granular — diusulkan sebagai *modifier opsional* (Bagian 1) supaya tidak
   memecah segmen menjadi terlalu kecil untuk diestimasi stabil sebelum
   terbukti perlu.
5. ~~Status kategori `Barang Semi FG (WIP-2)`~~ — **DITUTUP (2026-08-22),
   dikonfirmasi bersih.** Tujuh kategori di
   `dataset/pemetaan-sku-per-kategori.csv` adalah skema kategori update
   terbaru, **diterapkan retroaktif ke seluruh data historis 2024-2025**
   (dikonfirmasi pemilik data). Konsekuensinya dua hal sekaligus:

   - Tidak ada satu baris pun di `model_input.parquet` yang nilai
     `Kategori Barang`-nya masih literal `Barang Semi FG (WIP-2)` —
     `shelf_life_rank_by_category.csv` dengan tujuh baris (tanpa WIP-2)
     sudah lengkap dan aman dipakai sebagai sumber join.
   - Entri `"Barang Semi FG (WIP-2)": 4` yang masih terlihat di
     `category_mapping.json` (dilampirkan pemilik proyek, 2026-08-22) **bukan
     tanda WIP-2 masih aktif**, melainkan indeks yatim yang dipertahankan
     sesuai kebijakan stabilitas pipeline sendiri (§4.12 butir e
     `metodologi-preprocessing.md`: "nilai yang tidak lagi dipakai tetap
     memegang indeksnya"). Tidak ada risiko fungsional terhadap model final
     yang sudah dilatih — kekhawatiran sebelumnya soal `category_mapping.json`
     tidak sinkron dengan skema baru **tidak terbukti**, murni deskripsi
     "delapan kategori" di dokumen lama yang sekarang usang secara redaksional,
     bukan cacat pada model atau datanya.
6. ~~Apakah `Barang Jadi (FG)` perlu override tingkat SKU~~ — **DITUTUP
   (2026-08-22).** Kolom `shelf_life_rank_override` ditambahkan ke
   `item_cost_margin.csv` (Bagian 2), dengan tujuh SKU kandidat sudah
   diidentifikasi dari `dataset/pemetaan-sku-per-kategori.csv` (lihat tabel
   di "Catatan heterogenitas", Bagian 3). Dua dari tujuh (`FGS.00055` Saus
   Lemon, `FGS.00056` Saus Spicy) masih perlu konfirmasi tim SCM apakah
   kemasan pabrik atau racikan sendiri sebelum peringkat override-nya
   dianggap final; lima lainnya cukup jelas dari nama SKU-nya.

## Documentation updates (in scope for this work)

- `docs/batasan-penelitian.md`: **B-9** mendapat sub-bagian klarifikasi baru
  bertanggal 2026-08-22 di bawah bullet "Tingkat layanan target" yang sudah
  ada — tidak menghapus atau mengubah teks 2026-08-16 yang sudah dikonfirmasi,
  hanya menambahkan paragraf "Klarifikasi lanjutan (2026-08-22)" dan
  "Konsekuensi" setelahnya. Dua butir baru ditambahkan setelah B-9 dan
  sebelum "Ringkasan untuk bab batasan": **B-10** (data biaya/margin per item
  belum tersedia, status terbuka dan fleksibel, dengan tabel pelacakan
  cakupan) dan **B-11** (metrik days of supply sebagai pengganti sementara
  rupiah, statusnya mengikuti B-10). "Ringkasan untuk bab batasan" di akhir
  dokumen mendapat satu paragraf penutup baru yang membedakan sifat B-10/B-11
  dari kesembilan butir sebelumnya. Naskah lengkap sudah disiapkan di
  `batasan-penelitian.md` (versi revisi terlampir bersama spec ini) — Claude
  Code cukup mengganti file lama dengan versi ini, bukan menyusun ulang dari
  awal.
- `dataset/item_cost_margin.csv`: file baru, belum ada isinya untuk 70 SKU
  penuh. Jalankan `generate_item_cost_margin_template.py` (terlampir) di
  environment yang punya akses ke `dataset/model_ready/model_input.parquet`
  untuk menghasilkan baris kosong bagi seluruh SKU aktual — skrip ini tidak
  bisa dijalankan dari sini karena membutuhkan akses ke data produksi.
- `dataset/shelf_life_rank_by_category.csv`: file baru, **sudah terisi
  penuh** (tujuh kategori, dengan `catatan` berbasis bukti SKU/volume dari
  `dataset/pemetaan-sku-per-kategori.csv`) — siap dipakai apa adanya, hanya
  perlu ditinjau tim SCM sebelum `cost_confidence` dinaikkan dari `sedang`.
- `dataset/pemetaan-sku-per-kategori.csv`: sumber data yang sudah ada,
  dikonfirmasi sebagai skema kategori aktif terbaru (2026-08-22) — tidak
  perlu diubah, hanya dirujuk.
- Tidak ada perubahan pada `category_mapping.json`, kode pipeline
  preprocessing, atau model yang sudah dilatih — dikonfirmasi tidak
  diperlukan (lihat "Pertanyaan terbuka" nomor 5).

## References

- `docs/batasan-penelitian.md` B-9 — keputusan kuantil 0,9 seragam yang
  direinterpretasi (bukan dibatalkan) oleh spec ini.
- `docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md` —
  sumber kedua keputusan yang sama, dan sumber `target_h1`…`target_h4`.
- `docs/metodologi-pemodelan-dan-pemilihan-model.md` §15–21 — kriteria
  metrik, tangga pemilihan model, dan rencana kerja tersisa yang disisipi
  spec ini.
- `docs/hasil-modeling-{rf,xgb,lstm}.md` — angka shortfall/overstock dasar
  yang menjadi motivasi spec ini.
- `dataset/shelf_life_rank_by_category.csv` — isian awal Proksi A/B
  (estimasi umum, 2026-08-22), menunggu koreksi tim SCM sebelum
  `cost_confidence` dinaikkan dari `sedang`.
- `dataset/pemetaan-sku-per-kategori.csv` — sumber bukti SKU/volume per
  kategori yang mendasari isian di atas, dan sumber konfirmasi skema tujuh
  kategori terbaru (menggantikan skema delapan kategori di
  `metodologi-preprocessing.md`).
