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
