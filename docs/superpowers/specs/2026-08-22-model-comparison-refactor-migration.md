# Migrasi perbandingan model ke evaluasi multi-kuantil — refactor plan

## Status

**Menggantikan pendekatan evaluasi di `metodologi-pemodelan-dan-pemilihan-model.md`
§15–18 sebelum test set Desember 2025 dibuka.** Test set belum pernah
dibuka (dikonfirmasi §1 `hasil-modeling-rf.md`: "Desember 2025 tidak
dibuka"), sehingga migrasi ini **tidak membuang hasil evaluasi out-of-sample
apa pun** — yang perlu diulang hanya pencarian hyperparameter dan
walk-forward di kelima fold latih (Juli–November 2025), bukan pengujian
final.

**Dokumen ini murni checklist eksekusi.** Untuk penjelasan apa yang
berubah dan kenapa (definisi K1/K2/K3 baru, mekanisme dua tahap penentuan
kuantil, dasar bukti dari jurnal), lihat
`2026-08-22-multi-quantile-evaluation-design.md` — dokumen ini tidak
mengulang isinya, hanya merujuknya per poin.

## Catatan eksekusi (2026-08-24)

Checklist ini mulai dijalankan 2026-08-24. Status per butir:

| Butir | Status |
|---|---|
| 1 — spec XGBoost | ✅ selesai |
| 2 — spec LSTM | ✅ selesai |
| 3 — spec Random Forest | ✅ selesai |
| 4 — `metodologi` §15–18 | ⚠️ sebagian: definisi K1/K2 di §15 dan §17 selesai, plus §19 dan §21 (perluasan cakupan, disetujui pemilik proyek 2026-08-24). §16 dan §18 diberi penanda "menunggu penggantian" dan ditulis ulang setelah butir 5 |
| 5 — `hasil-modeling-{rf,xgb,lstm}.md` | ⬜ menunggu notebook dijalankan ulang |
| 6 — spec segmentasi kuantil | ✅ selesai |
| 7 — `batasan-penelitian.md` / `pipeline-overview.md` | ⚠️ direvisi, lihat "Koreksi butir 7" di bawah |

**Prasyarat yang belum tercatat di checklist ini.** "Urutan eksekusi yang
disarankan" langkah 2 meminta ketiga notebook dijalankan ulang "sesuai spec yang
sudah diubah", tetapi tidak ada butir yang mencakup **perubahan kodenya**. Kode
saat ini masih kuantil tunggal dari ujung ke ujung: `evaluation.py`
`DEFAULT_ALPHA = 0.9` dengan `score()` mengembalikan satu angka pinball,
`walk_forward.py` dan `model_common.py` menerima `alpha: float` skalar, dan tidak
ada `QUANTILE_SET` di mana pun. Langkah 2 tidak dapat dijalankan sebelum
`evaluation.py`, `walk_forward.py`, `model_common.py`, `model_xgboost.py`,
`model_lstm.py`, `model_random_forest.py`, dan ketiga notebook diubah. Pekerjaan
itu dicatat sebagai butir 0b di §21 `metodologi-pemodelan-dan-pemilihan-model.md`.

**Koreksi angka kandidat.** Butir 1 di bawah menyebut "18 kandidat" untuk
XGBoost, dan tabel "Dampak teknis per model" di
`2026-08-22-multi-quantile-evaluation-design.md` menyebut angka yang sama. Itu
keliru — 18 adalah anggaran **Random Forest**. Anggaran XGBoost yang benar-benar
dijalankan adalah **30** (`dataset/model_ready/xgb_search_results.csv` berisi 30
baris; `docs/hasil-modeling-xgb.md` §"Pencarian hyperparameter"). LSTM = 12 sudah
benar. Angka yang berlaku: **XGBoost 30, LSTM 12.**

**Keputusan anggaran pencarian (2026-08-24).** Pertanyaan terbuka nomor 2 di spec
multi-kuantil, bagian anggaran, **ditutup**: anggaran dipertahankan pada 30
(XGBoost) dan 12 (LSTM), tidak dikurangi meskipun tiap kandidat kini memprediksi
19 kuantil sekaligus. Dasarnya konsisten dengan posisi proyek yang sudah
tertulis: ongkos komputasi sengaja dikesampingkan karena tujuannya menemukan
model terbaik. Untuk LSTM ini berarti N **dipatok** 12, bukan diturunkan ulang
dari formula anggaran — konsekuensinya plafon 8 jam kemungkinan terlampaui, dan
itu dicatat sebagai ongkos terukur, bukan kegagalan (lihat §2.2 spec LSTM).
Bagian *warm start* dari pertanyaan terbuka nomor 2 (peralihan Tahap A → Tahap B)
**tetap terbuka**.

**Koreksi butir 7.** Butir 7 menyatakan tidak ada perubahan pada
`batasan-penelitian.md` dengan alasan "B-9 berbicara soal kuantil 0,9 sebagai
komitmen bisnis, bukan kriteria pemilihan model". Alasan itu benar untuk isi
utama B-9, tetapi tidak untuk seluruh isinya: paragraf **"Konsekuensi"** di bawah
"Klarifikasi lanjutan (2026-08-22)" berbicara eksplisit tentang proses pemilihan
model — "klarifikasi ini tidak mengubah proses pemilihan model" — dan pernyataan
itu menjadi tidak benar setelah migrasi ini. Sebuah catatan koreksi bertanggal
2026-08-24 ditambahkan di bawah paragraf tersebut (disetujui pemilik proyek).
Teks 2026-08-16 dan 2026-08-22 tidak dihapus atau diubah. `pipeline-overview.md`
memang tidak berubah, sesuai butir 7.

**Prasyarat Random Forest yang perlu dibaca bersama butir 3.** "RF tidak perlu
retrain" berlaku untuk **pencarian hyperparameter**, bukan untuk artefak
terlatihnya. `models/random_forest_q90.joblib` basi sejak reclass kategori WIP-2
2026-08-22 (§0 `docs/pipeline-overview.md`, prasyarat §19
`docs/metodologi-pemodelan-dan-pemilihan-model.md`), jadi walk-forward RF dan fit
final-nya tetap harus dijalankan ulang — hanya `rf_best_params.json` yang dipakai
ulang apa adanya.

## Purpose

Menerapkan desain di `2026-08-22-multi-quantile-evaluation-design.md` ke
seluruh spec dan dokumen proyek yang sudah ada, dengan urutan dan rincian
per file yang eksplisit, supaya bisa dieksekusi langsung tanpa perlu
menafsirkan ulang desain metodologinya.

## Dampak pada spec segmentasi kuantil (`2026-08-22-segmented-quantile-allocation-design.md`)

Bagian "Urutan pengerjaan relatif terhadap rencana kerja yang sudah ada"
di spec tersebut perlu diperbarui:

- **Sebelum migrasi ini**: perluasan multi-kuantil dikerjakan *setelah*
  pemenang ditetapkan (Bagian 4 spec tersebut, sebagai pekerjaan lanjutan
  khusus model pemenang).
- **Setelah migrasi ini**: perluasan multi-kuantil sudah selesai dikerjakan
  untuk **ketiga model** sebagai bagian dari K1 yang baru, sebelum pemenang
  ditetapkan. Begitu pemenang dipilih di K1–K3 yang sudah direvisi, ia
  **sudah otomatis punya kapabilitas multi-kuantil** — Bagian 4 spec
  segmentasi kuantil menjadi pekerjaan yang sudah selesai (inherited),
  bukan pekerjaan yang masih perlu dilakukan. Simulasi kalibrasi λ (Bagian
  5 spec tersebut) bisa langsung dimulai begitu pemenang ditetapkan, tanpa
  menunggu perluasan model tambahan.

Ini mempercepat, bukan menambah, jalur menuju segmentasi kuantil —
konsekuensi baik dari migrasi ini yang layak dicatat eksplisit di spec
segmentasi kuantil supaya tidak terlihat seolah menambah beban kerja
berganda.

## Dampak per dokumen (ringkasan cepat, rincian eksekusi di "Documentation updates")

| Dokumen | Dampak |
|---|---|
| `metodologi-pemodelan-dan-pemilihan-model.md` | §15–18 direvisi: definisi K1/K2 (lihat `2026-08-22-multi-quantile-evaluation-design.md` Bagian 2–3), tabel hasil, kesimpulan tangga keputusan — semuanya perlu ditulis ulang dengan angka baru |
| `2026-08-18-random-forest-modeling-design.md` | Bagian evaluasi diperluas ke `QUANTILE_SET`; bagian pencarian hyperparameter **tidak berubah** (RF tidak perlu retrain) |
| `2026-08-19-xgboost-modeling-design.md` | `quantile_alpha` diubah ke daftar; pencarian hyperparameter diulang |
| `2026-08-19-lstm-modeling-design.md` | Arsitektur head diubah; pencarian hyperparameter diulang |
| `hasil-modeling-{rf,xgb,lstm}.md` | **Seluruh angka perlu digenerate ulang** — dokumen-dokumen ini adalah bukti hasil, bukan spec, sehingga tidak "direvisi" tapi dijalankan ulang lalu ditulis ulang dari nol mengikuti template yang sama |
| `2026-08-22-segmented-quantile-allocation-design.md` | Bagian urutan pengerjaan diperbarui (lihat "Dampak pada spec segmentasi kuantil" di atas); Bagian 4 ditandai selesai lebih awal |
| `batasan-penelitian.md` | Tidak ada perubahan isi — B-9 berbicara soal kuantil 0,9 sebagai *komitmen bisnis*, bukan kriteria pemilihan model, jadi tetap valid apa adanya |
| `pipeline-overview.md` | Tidak ada perubahan — migrasi ini di tahap pemodelan, bukan preprocessing |

## Documentation updates (in scope for this work)

1. **`docs/superpowers/specs/2026-08-19-xgboost-modeling-design.md`**: ganti
   `quantile_alpha=0.9` menjadi `quantile_alpha=QUANTILE_SET` (definisi di
   `2026-08-22-multi-quantile-evaluation-design.md` Bagian 1); catat bahwa
   18 kandidat pencarian perlu dijalankan ulang dengan objective baru.
2. **`docs/superpowers/specs/2026-08-19-lstm-modeling-design.md`**: ubah
   spesifikasi arsitektur head dari 1 neuron menjadi `len(QUANTILE_SET)`
   neuron, loss total = jumlah pinball loss lintas kuantil; catat bahwa
   pencarian hyperparameter (12 kandidat) perlu diulang.
3. **`docs/superpowers/specs/2026-08-18-random-forest-modeling-design.md`**:
   tambahkan catatan bahwa evaluasi walk-forward sekarang membaca seluruh
   titik `QUANTILE_SET` dari forest yang sama; **tidak ada perubahan pada
   bagian pencarian hyperparameter**.
4. **`docs/metodologi-pemodelan-dan-pemilihan-model.md`** §15–18: revisi
   definisi K1 (rata-rata pinball di `QUANTILE_SET`, bukan pinball@0,9
   tunggal — definisi lengkap di `2026-08-22-multi-quantile-evaluation-design.md`
   Bagian 2), revisi K2 (coverage dicek per kuantil, Bagian 3), tabel hasil
   tangga keputusan ditulis ulang setelah ketiga model dijalankan ulang.
5. **`docs/hasil-modeling-rf.md`, `docs/hasil-modeling-xgb.md`,
   `docs/hasil-modeling-lstm.md`**: dijalankan ulang penuh dari notebook
   masing-masing setelah perubahan 1–3 diterapkan, ditulis ulang mengikuti
   struktur yang sama (ringkasan, setup evaluasi, benchmark, pencarian
   hyperparameter, hasil walk-forward per fold/segmen/hari-kirim, model
   final, batasan) — bukan diedit sebagian, karena seluruh angka di
   dalamnya berubah.
6. **`docs/superpowers/specs/2026-08-22-segmented-quantile-allocation-design.md`**:
   perbarui bagian "Urutan pengerjaan relatif" sesuai "Dampak pada spec
   segmentasi kuantil" di atas; tandai Bagian 4 (perluasan multi-kuantil)
   sebagai *inherited/selesai lebih awal* begitu migrasi ini dijalankan,
   bukan dihapus (supaya jejak keputusan asli tetap terbaca).
7. Tidak ada perubahan pada `batasan-penelitian.md` atau
   `pipeline-overview.md` — dikonfirmasi di atas.

## Urutan eksekusi yang disarankan

Bukan bebas urutan — beberapa langkah bergantung pada langkah sebelumnya:

1. Terapkan poin 1–3 (ubah spec XGBoost, LSTM, RF) lebih dulu, karena
   poin 4–5 butuh spec model sudah final sebagai acuan implementasi.
2. Jalankan ulang notebook ketiga model sesuai spec yang sudah diubah.
3. Tulis ulang poin 5 (`hasil-modeling-*.md`) dari hasil run tersebut.
4. Baru revisi poin 4 (`metodologi-pemodelan-dan-pemilihan-model.md`
   §15–18), karena tabel hasil di situ mengutip angka dari poin 5.
5. Terakhir, poin 6 (spec segmentasi kuantil) — independen dari 1–5,
   bisa dikerjakan kapan saja, tapi logis ditutup terakhir karena ia
   mengonsumsi hasil dari langkah 1–4 (pemenang model + kapabilitas
   multi-kuantilnya).

## Out of scope

- Perubahan isi metodologi itu sendiri — sepenuhnya mengikuti
  `2026-08-22-multi-quantile-evaluation-design.md`, dokumen ini tidak
  mendefinisikan ulang apa pun.
- Alokasi kuantil tersegmentasi — tetap sepenuhnya di
  `2026-08-22-segmented-quantile-allocation-design.md`.
- Mengubah `target_lead_time_cumulative`, mekanisme purging, atau split
  train/test — seluruhnya dipakai apa adanya.

## References

- `docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`
  — sumber kebenaran metodologi untuk seluruh perubahan di dokumen ini.
- `docs/superpowers/specs/2026-08-22-segmented-quantile-allocation-design.md`
  — Bagian 4 dan bagian urutan pengerjaan yang terdampak migrasi ini.
- `docs/batasan-penelitian.md` B-9 — komitmen kuantil 0,9 yang tidak
  berubah isinya akibat migrasi ini.
- `docs/metodologi-pemodelan-dan-pemilihan-model.md` §15–18 — tangga
  keputusan yang menjadi target revisi dokumen ini.
