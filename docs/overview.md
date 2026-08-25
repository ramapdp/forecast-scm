# Overview Proyek — Demand Forecasting SCM Kebuli Yaman

Dokumen ini adalah titik masuk untuk memahami keseluruhan proyek: masalah
bisnis yang diselesaikan, keputusan besar yang sudah diambil, alur pipeline,
dan status setiap bagian. Untuk detail masing-masing bagian, dokumen ini
merujuk ke dokumen lain, tidak mengulang isinya.

Terakhir diperbarui: 2026-08-26.

---

## 1. Masalah bisnis

Kantor pusat mengirim stok ke outlet **dua kali seminggu**: kawasan 1 setiap
Senin dan Kamis, kawasan 2 setiap Selasa dan Jumat. Jumlah kirim ditentukan
tim SCM dari tren tiga bulan terakhir yang diolah di Excel, dikurangi hasil
stock opname.

Praktik ini menyebabkan outlet sering kehabisan stok, yang lalu ditambal
lewat **lateral transshipment** (transfer stok antar outlet). Transfer itu
menimbulkan biaya tambahan yang bersifat reaktif dan menggerus keuntungan.

**Tujuan proyek:** mengurangi stockout dengan tambahan overstock yang
terkendali, lewat ramalan permintaan kumulatif sampai pengiriman berikutnya.

## 2. Perumusan masalah

| Aspek | Keputusan |
|---|---|
| Granularitas | satu deret waktu per pasangan (`Kode Barang`, `Nama Cabang`), harian |
| Target | `target_lead_time_cumulative` — total permintaan dari besok sampai kiriman berikutnya (3 atau 4 hari, bervariasi per baris menurut kawasan dan hari transaksi) |
| Target latih vs nilai | **latih** di `..._capped`, **dinilai** di target mentah (keputusan 2026-08-24, §5 `metodologi-pemodelan-dan-pemilihan-model.md`) |
| Strategi model | satu model global per algoritma, dilatih lintas seluruh deret, dengan identitas item/cabang/kategori sebagai fitur |
| Validasi | walk-forward 5 fold (Juli–November 2025), test set Desember 2025 terkunci |
| Kandidat model | Random Forest, XGBoost, LSTM |

**Kenapa target kumulatif, bukan ramalan harian yang dijumlahkan.**
Menjumlahkan kuantil harian tidak sama dengan kuantil dari total, dan
cenderung melebih-lebihkan secara sistematis. Target dihitung langsung
sebagai satu angka kumulatif per baris, sehingga model mempelajari variansi
horizon 3–4 hari apa adanya, termasuk autokorelasi antar hari di dalamnya.

## 3. Kenapa forecasting kuantil, bukan ramalan titik

Metrik simetris (MAE, RMSE, MSE) memperlakukan kekurangan dan kelebihan stok
sebagai kesalahan yang sama beratnya. Dalam bisnis ini konsekuensinya jauh
berbeda: kekurangan memicu transshipment, kelebihan memicu biaya simpan atau
waste. Model yang dilatih pada loss simetris akan mengarah ke median, yang
secara matematis berarti kehabisan stok pada sekitar separuh kejadian.

Karena itu proyek ini memakai **pinball loss** dan meramalkan kuantil, bukan
titik tengah. Tingkat layanan target dikonfirmasi pemilik data: **kuantil
0,9**, dipahami sebagai komitmen agregat di level pengiriman (lihat B-9
`batasan-penelitian.md`).

## 4. Dua perubahan rencana besar (Agustus 2026)

### 4a. Evaluasi multi-kuantil untuk perbandingan model

Semula ketiga model dibandingkan pada satu titik, pinball@0,9. Itu
mengandung asumsi tak teruji: model yang unggul di satu titik belum tentu
unggul di rentang kuantil lain. Sejak 2026-08-22, kriteria utama (K1) adalah
**rata-rata pinball loss di seluruh `QUANTILE_SET`** (Tahap A: 19 titik
0,05–0,95).

Spec: `superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`
Rencana penerapan: `superpowers/specs/2026-08-22-model-comparison-refactor-migration.md`

### 4b. Alokasi kuantil tersegmentasi (pekerjaan lanjutan)

Kuantil 0,9 seragam untuk semua SKU memangkas stockout 73–76% tapi menaikkan
overstock 2,5–2,8×, karena barang cepat rusak menanggung beban yang sama
dengan kemasan yang praktis tidak kedaluwarsa. Rencananya kuantil bervariasi
per segmen (kategori × `demand_segment`), dengan rata-rata tertimbang tetap
kembali ke 0,9 secara agregat.

Dikerjakan **setelah** pemenang model ditetapkan.
Spec: `superpowers/specs/2026-08-22-segmented-quantile-allocation-design.md`

## 5. Alur pipeline

```
dataset/csv/*.csv
  → merge_dataset.py            gabung 5 berkas sumber → dataset.csv
  → normalize_items.py          normalisasi kode/nama, kanonikalisasi kategori,
                                exclude cabang/item, konversi satuan
  → build_panel.py              panel harian padat, segment_id per blok aktif
  → calendar_features.py        kalender, Ramadan/Idulfitri/Iduladha/17-an/Tahun Baru
  → outlier_handling.py         deteksi lonjakan + capping (target tetap mentah)
  → outlet_features.py          kota, kanal online, kawasan, lead_time_days
  → prepare_forecast_data.py    target, lag, rolling, statistik cabang
                                → featured.parquet → train/test.parquet
  → modeling_prep.py            encoding, demand_segment, fold_id
                                → model_input.parquet + category_mapping.json
  → walk_forward.py             runner 5 fold, kontrak baris identik antar model
  → model_{random_forest,xgboost,lstm}.py
  → evaluation.py               pinball per τ, K1, coverage, fill rate, crossing
```

Detail lengkap: `metodologi-preprocessing.md` (metodologi) dan
`dokumentasi-preprocessing-id.md` (dokumentasi teknis).

## 6. Peta dokumen

**Baca lebih dulu**

| Dokumen | Isi |
|---|---|
| `overview.md` | dokumen ini |
| `batasan-penelitian.md` | B-1…B-11, batasan yang tidak bisa dihilangkan dengan menulis kode lebih baik. Wajib dibaca sebelum menafsirkan hasil apa pun |
| `pipeline-overview.md` | status dan struktur pipeline |
| `temuan-eda.md` | temuan EDA atas `dataset.csv`: konteks bisnis, 13 kesimpulan, open questions, checklist pra-modeling. Narasi ini dulu ada di `notebook/eda.ipynb` |

**Metodologi**

| Dokumen | Isi |
|---|---|
| `metodologi-preprocessing.md` | keputusan preprocessing dan alasannya |
| `dokumentasi-preprocessing-id.md` | dokumentasi teknis per tahap |
| `metodologi-pemodelan-dan-pemilihan-model.md` | metrik, tangga keputusan K1–K3, protokol pembukaan test set (§19), rencana kerja (§21) |

**Hasil (angka, bukan rencana)**

`hasil-modeling-rf.md` — satu-satunya dokumen hasil yang berlaku (multi-kuantil,
2026-08-25).

> ⚠ Dokumen hasil XGBoost dan LSTM **sudah diarsipkan** ke `bak/` pada
> 2026-08-26: angkanya dari run kuantil-tunggal (19–20 Agustus) di atas data
> pra-reklasifikasi kategori, dan tidak sebanding dengan K1. Sampai Fase 3
> kedua model itu dijalankan, tidak ada dokumen hasil yang berlaku untuk
> keduanya — angka lama hanya boleh dibaca sebagai catatan sejarah di
> `bak/hasil-modeling-xgb.single-quantile.bak.md` dan
> `bak/hasil-modeling-lstm.single-quantile.bak.md`.
>
> Isi `docs/bak/`: dokumen hasil yang sudah digantikan atau ditinggalkan.
> Arsip artefaknya ada di `dataset/model_ready/bak/` (9 berkas) dan
> `models/bak/` (3 bundle).

**Spec desain** (`superpowers/specs/`, urut kronologis)

Tiga terbaru adalah yang mengatur pekerjaan saat ini:
`2026-08-22-multi-quantile-evaluation-design.md`,
`2026-08-22-model-comparison-refactor-migration.md`,
`2026-08-22-segmented-quantile-allocation-design.md`.

**Operasional**

`checklist-refresh-data-2026.md`, `todolist-proyek.md`,
`pertanyaan-data-owner.md`, `outlet_relocation_notes.md`

## 7. Berkas konfigurasi yang dipelihara manusia

Berkas berikut **bukan hasil generate** — isinya keputusan manusia yang
divalidasi keras oleh pipeline (kode tak dikenal atau nilai di luar rentang
memicu error, bukan diabaikan diam-diam).

| Berkas | Isi | Status |
|---|---|---|
| `event_driven_items.csv` | penanda SKU acara/aqiqah | 70 SKU, 3 dikonfirmasi pemilik data |
| `outlet_mapping.csv` | kawasan + jadwal kirim per cabang | lengkap |
| `outlet_closures.csv` | interval tutup per cabang | 7 baris |
| `outlet_name_overrides.csv` | resolusi nama/kota cabang | 19 baris |
| `shelf_life_rank_by_category.csv` | peringkat masa simpan per kategori (proksi B-10) | terisi, estimasi umum, menunggu tinjauan tim SCM |
| `item_cost_margin.csv` | biaya/margin per SKU + `shelf_life_rank_override` | 70 baris, kolom biaya **masih kosong** (B-10) |

## 8. Status per 2026-08-26

| Bagian | Status |
|---|---|
| Preprocessing | ✅ selesai, artefak terverifikasi (7 kategori, 0 SKU multi-kategori, 1.502.522 baris, fold cocok) |
| Reklasifikasi kategori WIP-2 → FG | ✅ selesai, 10 SKU, lewat `EXPLICIT_CATEGORY_OVERRIDES` |
| Gerbang konsistensi kategori | ✅ `utils/eda/verify_category_consistency.py` |
| Fase 1 — dokumentasi migrasi multi-kuantil | ✅ selesai |
| Fase 2 — implementasi kode multi-kuantil | ✅ selesai 2026-08-24 (kontrak `fit_predict` multi-kuantil; status jalannya ada di baris Fase 3 di bawah) |
| Prasyarat metodologis Fase 3 (target latih/nilai, kerapatan grid) | ✅ ditutup 2026-08-24 |
| Fase 3 — Random Forest | ✅ selesai 2026-08-25, K1 = 2,8621 (5 fold) / 2,8508 (fold bersih 1/2/4) |
| Fase 3 — XGBoost | 🔶 baru probe Tahap 0 (kandidat 0, CPU, 19.959 detik); pencarian penuh belum jalan |
| Fase 3 — LSTM | ⬜ belum jalan |
| Penulisan ulang `hasil-modeling-*.md` | 🔶 `-rf.md` selesai 2026-08-25; `-xgb.md` dan `-lstm.md` setelah Fase 3 masing-masing |
| Pembukaan test set Desember 2025 | ⬜ belum pernah dibuka, protokol §19 |
| Alokasi kuantil tersegmentasi | ⬜ setelah pemenang ditetapkan |
| Data biaya/margin per SKU | ⬜ terbuka (B-10), 0% volume terisi |

**`models/random_forest_q90.joblib` sudah versi Fase 3** (2026-08-25, 19 titik
kuantil, 1.349.011 baris latih). Berkas `models/*.single-quantile.bak.joblib`
adalah arsip pra-reklasifikasi berkriteria kuantil-tunggal — jangan dipakai
untuk inferensi. XGBoost dan LSTM belum punya model Fase 3.

## 9. Prinsip kerja yang dipegang di repo ini

Beberapa pola berulang yang menjelaskan kenapa kode ditulis seperti sekarang:

1. **Gagal keras, jangan diam-diam salah.** Nilai konfigurasi tak dikenal
   memicu error, bukan default. Contoh: `parse_delivery_days`,
   `load_closures`, penjaga checkpoint `quantile_set`, gerbang konsistensi
   kategori. Alasannya: kesalahan yang berhenti terlihat jauh lebih murah
   daripada kesalahan yang lolos ke laporan.
2. **Deteksi memberi peringatan, manusia memutuskan.** `detect_unrecorded_gaps`,
   rekonsiliasi sentinel `shelf_life_days`. Pipeline tidak pernah menebak
   sendiri klasifikasi yang seharusnya dikonfirmasi.
3. **Kesetaraan antar model dijaga di kontrak, bukan di niat.**
   `validate_contract()` memastikan ketiga model dinilai pada baris, kunci,
   target, dan fold yang identik.
4. **Batasan dinyatakan, bukan disembunyikan.** `batasan-penelitian.md`
   memuat batasan yang membatasi akurasi ketiga model sekaligus, termasuk
   yang tidak menguntungkan hasil.
5. **Nilai kategori pensiun tetap memegang indeksnya.** Membebaskan indeks
   akan mengarahkan model terlatih ke kategori yang keliru
   (`metodologi-preprocessing.md` §4.12(e)).
6. **TDD.** Tes ditulis dan dipastikan gagal karena alasan yang benar
   sebelum implementasi. 842 tes per 2026-08-26.
7. **Notebook berdiri sendiri, `utils/` tempat kode diuji.** Sejak 2026-08-26
   tiap notebook memuat sendiri fungsi yang dijalankannya, tanpa
   `from utils... import`; `utils/` + `test/` adalah tempat kode ditulis dan
   diuji, lalu disalin verbatim ke notebook. Tiap notebook ditutup sel opsional
   yang membandingkan salinannya dengan sumber di `utils/` dan menyebut fungsi
   yang menyimpang. Sel markdown dibatasi pada judul bagian; penjelasan hidup di
   docstring dan komentar. Notebook modeling dikecualikan sebagian — hanya kode
   modelnya yang diinline, mesin bersama tetap diimpor. Aturan lengkapnya di
   bagian "Notebook convention" `CLAUDE.md`.

## 10. Batasan yang paling memengaruhi pembacaan hasil

Tiga hal yang membatasi akurasi maksimum ketiga model, bukan disebabkan
pilihan arsitektur (rincian di `batasan-penelitian.md`):

1. **Buku pesanan tidak terekam** (B-1, B-2). Manajer outlet sudah mengetahui
   sebagian permintaan beberapa hari ke depan; informasi itu nyata, dipakai
   dalam operasi, dan tidak ada di data mana pun. Selisih tipis terhadap
   baseline naif harus dibaca sebagai keterbatasan informasi.
2. **Lingkup model dan isi target tidak berimpit** (B-3). Model dibutuhkan
   untuk permintaan di luar pesanan, tapi target memuat keduanya bercampur.
3. **Evaluasi berlaku untuk irisan yang lebih sempit** (B-4, B-5, B-6):
   1.920 dari 2.979 pasangan, 1–29 Desember 2025, dan hanya 29% baris uji
   yang merupakan momen keputusan sungguhan.
