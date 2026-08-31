# To-Do List — Optimasi Model (Terpisah dari Jalur Utama)

Daftar kerja untuk perbaikan **metodologi** yang meningkatkan kualitas model,
tapi **bukan** prasyarat untuk menyelesaikan pemilihan pemenang (Fase D/E di
`docs/todolist-proyek.md`). Sengaja dipisah ke berkas ini karena setiap butir
di sini mengubah **definisi target atau fitur**, yang berarti mengulang
seluruh pipeline data-prep **dan** melatih ulang ketiga model (RF, XGBoost,
LSTM) — ongkos yang sama besarnya dengan butir 0c yang baru saja selesai
(`docs/todolist-proyek.md`, 2026-08-25 s.d. 2026-08-28). Menjalankannya
sekarang berarti membuang hasil 0c yang baru selesai sebelum sempat dipakai
memilih pemenang.

**Urutan yang disarankan: kerjakan sesudah Fase E (pembekuan pemenang + buka
test set Desember) ditutup, bukan sebelum.** Kalau salah satu butir di sini
ternyata mengubah angka secara material, itu jadi siklus evaluasi berikutnya
di atas pemenang yang sudah dibekukan — bukan menyisipkan target bergerak ke
tengah proses seleksi yang sedang berjalan.

---

## 1. Validasi statistik ambang deteksi lonjakan (`SPIKE_RATIO_THRESHOLD`)

**Asal pertanyaan:** diskusi 2026-08-29 tentang kolom `baseline_ratio` /
`is_spike` / `Kuantitas_capped` di `utils/data_preprocessing/outlier_handling.py`.

**Status saat ini.** `SPIKE_RATIO_THRESHOLD = 5.0` (`outlier_handling.py:9`)
bukan hasil turunan statistik — ditetapkan lewat eyeballing manual di
`notebook/eda.ipynb` bagian 1.7 (`docs/superpowers/specs/2026-08-08-outlier-handling-design.md:15`:
*"carried over unchanged from the values already explored and eyeballed
against real output"*), lalu dikunci jadi konstanta pipeline. Angka ini
menentukan `Kuantitas_capped`, yang dipakai baik sebagai **fitur** (lag/
rolling/statistik cabang) **maupun** sebagai **target latih**
(`prepare_forecast_data.py:274-277`) — jadi bukan perubahan kosmetik.

`docs/analisis-lonjakan-permintaan.md` sudah menunjukkan kelemahan konkret
ambang rasio-tetap ini: karena rasionya relatif terhadap median tiap pasangan
(item, cabang), item bervolume kecil (mis. `Rice Bowl 600 ml`, median ~8
unit/hari) jauh lebih gampang melewati 5× dibanding item bervolume besar
(`Nasi Kebuli`, naik ke 1,92× median di hari yang sama tidak pernah tertandai)
— bias yang murni fungsi skala median pasangan, bukan fungsi seberapa besar
lonjakan sungguhan.

- [ ] **Pilih metode validasi/penurunan ambang yang lebih baku daripada
  eyeball.** Kandidat, dari yang paling murah ke paling ketat:
  1. **Robust Z-score (MAD)** — `modified_z = 0.6745 * (x - median) / MAD`,
     ambang umum `|modified_z| > 3.5` (Iglewicz & Hoaglin, 1993). Skalanya
     mengikuti sebaran tiap pair sendiri, bukan angka rasio tetap.
  2. **Percentile / Tukey's fences pada distribusi `baseline_ratio`
     itu sendiri** — mis. ambil top 0,5–1% sebagai spike, atau
     `Q3 + k*IQR` dari sebaran rasio (bukan dari sebaran `Kuantitas` mentah).
  3. **Model probabilitas count** (Poisson/Negative Binomial per pair) —
     tandai spike sebagai baris dengan `P(X ≥ observed)` di bawah ambang
     (mis. 0,01). Overdispersion kemungkinan besar terjadi di data ini
     (banyak nol + ekor panjang), jadi NB lebih masuk akal daripada Poisson
     murni — perlu dicek langsung ke data sebelum dipilih.
  4. **Elbow/inflection pada kurva %-baris-tertandai vs ambang kandidat**
     (3×, 4×, 5×, 7×, 10×, …) — cara termurah untuk memvalidasi bahwa 5,0
     bukan angka sembarang, tanpa mengganti seluruh metodologi.
- [ ] **Jalankan opsi termurah dulu (butir 4 di atas) sebagai sanity check**
  sebelum berkomitmen ke opsi 1–3 — kalau kurva %-tertandai menunjukkan 5,0
  memang dekat titik siku, ambang saat ini mungkin sudah cukup baik dan
  tidak perlu diganti; kalau tidak, itu alasan kuat untuk lanjut ke opsi 1–3.
- [ ] **Kalau ambang berubah**: seluruh pipeline harus diulang dari
  `.venv/bin/python3 -m utils.data_preprocessing.prepare_forecast_data`
  (regenerasi `featured.parquet` → `train/test.parquet`) → `modeling_prep.py`
  (`model_input.parquet`) → training ulang RF + XGBoost + LSTM (walk-forward +
  fit final). Angka di `docs/hasil-modeling-*.md` yang berlaku sebelum
  perubahan ini **tidak lagi sebanding** dengan run sesudahnya — sama seperti
  perubahan target capped vs mentah yang sudah pernah terjadi (A3 no. 2 di
  `todolist-proyek.md`).
- [ ] **Dokumentasikan keputusan akhir** (ambang baru atau tetap 5,0, dan
  alasannya) di `docs/analisis-lonjakan-permintaan.md` atau dokumen baru,
  dengan cara yang sama seperti keputusan target capped vs mentah didukung
  konfirmasi pemilik data (A3 no. 2 di `todolist-proyek.md`) — ambang deteksi
  spike memengaruhi target latih, jadi sebaiknya bukan keputusan sepihak tim
  teknis.

---

## 2. Kandidat lain (belum digali, catat kalau muncul)

- [ ] *(kosong — isi kalau ada usulan optimasi lain yang juga butuh
  retraining penuh dan karenanya layak masuk daftar terpisah ini alih-alih
  `todolist-proyek.md`.)*

---

*Sumber: diskusi 2026-08-29 tentang `utils/data_preprocessing/outlier_handling.py`;
`docs/analisis-lonjakan-permintaan.md`;
`docs/superpowers/specs/2026-08-08-outlier-handling-design.md`.*
