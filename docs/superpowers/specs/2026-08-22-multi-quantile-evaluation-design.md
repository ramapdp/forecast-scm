# Evaluasi multi-kuantil untuk perbandingan model forecasting — design

## Status

Spec metodologi. Untuk rencana penerapannya ke spec/dokumen proyek yang
sudah ada, lihat
`2026-08-22-model-comparison-refactor-migration.md` (dokumen terpisah,
merujuk balik ke spec ini sebagai sumber kebenaran metodologi).

## Purpose

Mendefinisikan pendekatan evaluasi model forecasting probabilistik
(Random Forest, XGBoost, LSTM) berdasarkan **rata-rata pinball loss di
banyak titik kuantil sekaligus**, menggantikan evaluasi pada satu titik
kuantil tunggal (0,9). Motivasi: model yang unggul di satu titik kuantil
tidak dijamin tetap unggul ketika dievaluasi di rentang kuantil yang lebih
luas, sementara tujuan proyek ini (memahami tren pasar dan memenuhi demand
forecast selama lead time secara menyeluruh) menuntut model yang teruji
kuat di seluruh rentang, bukan hanya satu titik service level.

## Latar belakang & justifikasi

Dengan ongkos komputasi sengaja dikesampingkan sebagai pertimbangan (tujuan
proyek adalah menemukan model terbaik, bukan model termurah), sumber
berikut mendasari keputusan ini:

1. **Kompetisi M5 Uncertainty** (Makridakis et al., 2021,
   [International Journal of Forecasting](https://consensus.app/papers/details/f2c4f9a65dfa586290e47bb56b52eb84/?utm_source=claude_desktop))
   — standar industri untuk forecasting demand ritel skala besar (42.840
   deret waktu penjualan Walmart) mengevaluasi model pada sembilan titik
   kuantil sekaligus lewat weighted scaled pinball loss yang dirata-ratakan,
   bukan satu titik. Desain kompetisi ini dijelaskan lebih lanjut di
   Makridakis et al. (2021),
   [The M5 competition: Background, organization, and implementation](https://consensus.app/papers/details/3f407c505de85568bcfc31a9e9cb269f/?utm_source=claude_desktop).
2. **Peringkat model terbukti berubah tergantung metrik/titik evaluasi**
   (Serafin et al., 2024,
   [Ranking probabilistic forecasting models with different loss functions](https://consensus.app/papers/details/812f35029318504d8516de885ce67b9d/?utm_source=claude_desktop))
   — model yang unggul di pinball@0,9 tidak dijamin tetap unggul di
   kuantil lain.
3. **Kalibrasi tak bersyarat vs bersyarat** (Gneiting & Resin, 2022,
   [Model Diagnostics and Forecast Evaluation for Quantiles](https://consensus.app/papers/details/8d390a5c683b555282973a237ee755fc/?utm_source=claude_desktop))
   — menilai model hanya di satu kuantil berisiko menangkap kalibrasi baik
   secara kebetulan di titik itu saja, tanpa menjamin kalibrasi baik di
   seluruh distribusi.
4. **Kelayakan teknis multi-kuantil pada XGBoost** (Sluijterman et al., 2024,
   [Composite quantile regression with XGBoost using the novel arctan pinball loss](https://consensus.app/papers/details/31fce8142b7d59b5bc93632421773376/?utm_source=claude_desktop))
   — `quantile_alpha` berupa daftar sudah merupakan pendekatan yang mapan
   di literatur, bukan improvisasi ad hoc.
5. **Rata-rata pinball loss pada grid padat mendekati CRPS** (Bröcker, 2012,
   [Evaluating raw ensembles with the continuous ranked probability score](https://consensus.app/papers/details/9e7a60d68c7c571395cbf055499987aa/?utm_source=claude_desktop))
   — hubungan formal antara *Continuous Ranked Probability Score* dan
   quantile score, mendasari desain grid di Bagian 1.
6. **Meminimalkan expected pinball loss di seluruh kuantil lebih efisien**
   daripada melatih pada satu titik (Narayan et al., 2024,
   [Expected Pinball Loss For Quantile Regression And Inverse CDF Estimation](https://consensus.app/papers/details/81988d96ace9541d8263c9d319a73d7a/?utm_source=claude_desktop)).
7. **Data-driven newsvendor mencapai service level tepat sasaran** dari data
   biaya aktual (van der Laan et al., 2022,
   [The data-driven newsvendor problem: Achieving on-target service-levels using distributionally robust chance-constrained optimization](https://consensus.app/papers/details/b228ad515ebe570ea63abdd74410a299/?utm_source=claude_desktop)),
   unggul khususnya pada target service level di atas 80% (Ye et al., 2021,
   [The New Data-Driven Newsvendor Problem with Service Level Constraint](https://consensus.app/papers/details/6ea069c92054575fb0405d81ef0e4d21/?utm_source=claude_desktop)).

## Keputusan desain

### 1. Set kuantil target — dua tahap, dinamis mengikuti ketersediaan data biaya

Set kuantil ditentukan lewat mekanisme dua tahap, mengikuti pola dinamis
yang sama dengan jalur presisi/proksi yang sudah dirancang untuk
`item_cost_margin.csv` di spec segmentasi kuantil — berubah otomatis
sesuai ketersediaan data, bukan keputusan sekali jalan.

**Tahap A (default, sebelum B-10 mencapai ambang) — grid padat merata,
mendekati CRPS.** Tidak butuh data biaya sama sekali, tidak ada keputusan
subjektif soal titik mana yang relevan:

```
QUANTILE_SET_A = [0.05, 0.10, 0.15, ..., 0.90, 0.95]   # 19 titik, spasi 0,05
```

**Tahap B (begitu B-10 mencapai ambang ≥80% volume dengan `cost_confidence`
bukan `rendah`) — grid diturunkan dari sebaran critical ratio aktual**:

```
untuk setiap segmen (Kategori Barang x demand_segment):
    critical_ratio[segmen] = resolve_critical_ratio(segmen, item_cost_margin.csv, ...)
QUANTILE_SET_B = persentil ke-10, 25, 50, 75, 90 dari seluruh critical_ratio[segmen]
```

`QUANTILE_SET_B` mencerminkan langsung kebutuhan alokasi kuantil
tersegmentasi yang sebenarnya akan dipakai di produksi, bukan tebakan.

**Mekanisme peralihan:**

```
QUANTILE_SET = QUANTILE_SET_B jika B-10 sudah mencapai ambang penutupan
               QUANTILE_SET_A selainnya
```

Peralihan ini memicu pengulangan K1–K3 (Bagian 2–4), sama seperti regresi
dinamis SKU di `item_cost_margin.csv` — begitu ambang tercapai, langkah
berikutnya otomatis memakai grid yang lebih presisi tanpa keputusan manual
ulang soal titik mana yang dipakai.

### 2. K1 (kriteria utama) — rata-rata pinball loss di seluruh set kuantil, generik terhadap ukuran grid

```
K1_score(model) = mean( pinball_loss(model, τ) untuk τ in QUANTILE_SET )
```

dihitung dengan disiplin walk-forward yang identik dengan protokol yang
sudah ada di `metodologi-pemodelan-dan-pemilihan-model.md` (fold 3 dan
fold 5 dipakai sebagai pemilih, ambang 2% dikalibrasi dari varians data)
— **hanya targetnya yang berubah**, dari pinball@0,9 tunggal menjadi
rata-rata seluruh titik di `QUANTILE_SET`, berapa pun jumlahnya (19 titik
di Tahap A, atau 5-7 titik di Tahap B). Skor per titik kuantil tetap
dilaporkan berdampingan (bukan disembunyikan di balik satu angka
rata-rata), mengikuti pola pemecahan `demand_segment`/`is_delivery_day`
yang sudah jadi kebiasaan dokumentasi proyek ini.

### 3. K2 (kalibrasi) — coverage dicek per kuantil, bukan hanya di 0,9

```
untuk setiap τ in QUANTILE_SET:
    coverage(model, τ) harus mendekati τ
```

Model yang kalibrasinya melenceng konsisten ke satu arah di **seluruh**
titik kuantil (bukan hanya di 0,9) adalah sinyal yang lebih kuat untuk
tersingkir di K2, dibanding pelencengan yang hanya tampak di satu titik.

### 4. K3 (ongkos/reprodusibilitas) — tidak berubah strukturnya

Tetap tie-breaker terakhir seperti sekarang, hanya dihitung di atas hasil
pencarian hyperparameter yang sudah diperluas ke multi-kuantil.

## Dampak teknis per model

| Model | Perubahan yang diperlukan | Ongkos relatif |
|---|---|---|
| Random Forest | **Tidak perlu retrain.** `quantile_forest` membaca kuantil berapa pun dari forest yang sama — pencarian hyperparameter yang sudah ada tetap valid, hanya evaluasi walk-forward perlu diulang untuk membaca seluruh titik `QUANTILE_SET` (19 di Tahap A, 5-7 di Tahap B) alih-alih 1 | Murah |
| XGBoost | `quantile_alpha` diubah dari skalar `0.9` menjadi daftar `QUANTILE_SET`; pencarian **30** kandidat perlu diulang dengan objective multi-kuantil | Sedang–tinggi |
| LSTM | Head output diperluas dari 1 neuron menjadi `len(QUANTILE_SET)` neuron, loss = jumlah pinball loss lintas kuantil; pencarian hyperparameter (**30 kandidat**, ruang 144, + 3 seed pada pemenang — lihat koreksi anggaran di bawah) perlu diulang pada arsitektur baru | Tinggi |

> **Koreksi 2026-08-24.** Baris XGBoost semula menyebut "18 kandidat" — itu
> anggaran Random Forest, bukan XGBoost. Anggaran XGBoost yang terukur adalah 30
> (`dataset/model_ready/xgb_search_results.csv`, 30 baris). Ruang pencarian LSTM
> juga dikoreksi dari 48 menjadi 144, sesuai `SEARCH_SPACE` di §2.1
> `2026-08-19-lstm-modeling-design.md`; angka 48 adalah ruang setelah dua dimensi
> dipotong oleh benchmark, sebagaimana tercatat di §18
> `metodologi-pemodelan-dan-pemilihan-model.md`.
>
> Random Forest tidak muncul di kolom "perubahan yang diperlukan" karena
> pencariannya tidak diulang — tetapi walk-forward dan fit final-nya **tetap**
> dijalankan ulang, karena bundle-nya basi akibat reclass kategori WIP-2
> 2026-08-22. Itu prasyarat yang berdiri sendiri, bukan konsekuensi migrasi ini.

**Koreksi 2026-08-24 (pemilik proyek) atas paragraf terakhir kutipan di atas.**
Kutipan dipertahankan sebagai jejak keputusan, tetapi kalimat "pencariannya
tidak diulang" **sudah tidak berlaku**: pencarian RF ikut dijalankan ulang, 18
kandidat, pada data pasca-reclass dan kriteria K1. Pembalikannya memang bukan
konsekuensi migrasi ini — persis seperti yang ditulis paragraf itu tentang
bundle — melainkan konsekuensi reclass WIP-2 yang sama: `rf_best_params.json`
dipilih 2026-08-18, di atas data pra-reclass. Lihat §Part 2
`2026-08-18-random-forest-modeling-design.md` dan §"Langkah 1–3"
`2026-08-22-model-comparison-refactor-migration.md`.

## Testing

- Regression test RF: prediksi kuantil 0,9 dari forest yang sudah dilatih
  harus identik (dalam toleransi numerik) dengan angka yang sudah tercatat
  sebelumnya — memastikan tidak ada perubahan perilaku model, hanya
  perluasan evaluasi.
- `K1_score`: dihitung benar sebagai rata-rata tak berbobot `pinball_loss`
  per kuantil di seluruh `QUANTILE_SET`, pada baris/fold yang identik.
- Coverage per kuantil: untuk kuantil τ, proporsi baris dengan
  `actual <= prediksi` harus dilaporkan terpisah per τ, bukan digabung
  jadi satu angka.
- XGBoost multi-kuantil: tidak ada *quantile crossing* (prediksi τ=0,7
  harus selalu ≤ prediksi τ=0,8 pada baris yang sama) — sejalan dengan
  temuan Sluijterman et al. (2024) bahwa pinball loss standar rawan
  crossing, motivasi mereka mengusulkan arctan pinball loss sebagai
  alternatif yang layak dipertimbangkan bila crossing signifikan.
- Mekanisme peralihan Tahap A → Tahap B: begitu B-10 disimulasikan
  mencapai ambang, `QUANTILE_SET` berubah dari `QUANTILE_SET_A` ke
  `QUANTILE_SET_B` tanpa perubahan kode, hanya berdasarkan status data.

## Out of scope

- Alokasi kuantil tersegmentasi itu sendiri (kategori/`demand_segment`) —
  spec ini hanya mendefinisikan *bagaimana model dibandingkan dan model
  mana yang menang*, bukan bagaimana kuantil akhirnya dialokasikan ke
  produksi. Itu tetap sepenuhnya di
  `2026-08-22-segmented-quantile-allocation-design.md`.
- Penerapan spec ini ke dokumen/file proyek yang sudah ada — lihat
  `2026-08-22-model-comparison-refactor-migration.md`.
- Mengubah `target_lead_time_cumulative`, mekanisme purging, atau split
  train/test — seluruhnya dipakai apa adanya.

## Pertanyaan terbuka

1. **Bagaimana pembobotan rata-rata di K1** — saat ini diusulkan tak
   berbobot (setiap titik kuantil kontribusinya sama), tapi bisa
   dipertimbangkan pembobotan lebih besar ke arah 0,9 (karena itu komitmen
   bisnis yang sudah dikonfirmasi eksplisit di B-9) dibanding titik lain
   yang lebih eksploratif. Relevan untuk Tahap A maupun Tahap B — di Tahap
   B, pembobotan alternatifnya bisa mengikuti volume segmen yang
   menyumbang tiap titik critical ratio, bukan tak berbobot rata.
2. **Skala anggaran pencarian hyperparameter XGBoost/LSTM untuk grid 19
   titik (Tahap A), dan pengulangan keduanya saat beralih ke Tahap B**:
   - ~~**Ukuran grid Tahap A (19 titik) jauh lebih besar**~~ — **DITUTUP
     (2026-08-24).** Anggaran **dipertahankan**: XGBoost 30 kandidat, LSTM 12
     kandidat (bukan 18/12 seperti tertulis semula — lihat koreksi di tabel
     "Dampak teknis per model"). Dasarnya adalah posisi proyek yang sudah
     tertulis di Bagian "Latar belakang & justifikasi": ongkos komputasi
     sengaja dikesampingkan karena tujuannya menemukan model terbaik.
     Mengurangi anggaran justru akan membuat run multi-kuantil dicari lebih
     sempit daripada run kuantil-tunggal yang digantikannya, sehingga
     perbandingan lama-baru tidak lagi setara. Untuk LSTM, N dipatok 12 dan
     tidak diturunkan ulang dari formula anggaran §2.2 spec LSTM;
     konsekuensinya plafon 8 jam kemungkinan terlampaui, dicatat sebagai ongkos
     terukur di `docs/hasil-modeling-lstm.md`.

     > **Direvisi 2026-08-24 (pemilik proyek, sesudah T-7).** Angka LSTM di
     > butir ini — 12 kandidat — **tidak lagi berlaku**. Anggaran LSTM
     > dinaikkan ke **30 kandidat** (setara XGBoost), ruang pencariannya
     > dipulihkan ke 144 di kode, dan konfigurasi terbaiknya diulang pada 3
     > seed. Yang berubah bukan posisi soal ongkos — itu tetap dikesampingkan —
     > melainkan penilaian bahwa ketimpangan anggaran tidak boleh dipertahankan
     > ketika ketiga model dicari ulang dari nol, karena kekalahan LSTM di K1
     > menjadi tidak dapat diatribusikan antara arsitektur dan kedangkalan
     > pencarian. Uraian lengkapnya di §2.2
     > `2026-08-19-lstm-modeling-design.md` dan §21
     > `docs/metodologi-pemodelan-dan-pemilihan-model.md`. Anggaran XGBoost (30)
     > dan RF (18) tidak berubah.
   - **Peralihan ke Tahap B memicu pengulangan pencarian dengan grid yang
     berbeda** (5-7 titik dari critical ratio, bukan 19 titik merata).
     Apakah pencarian Tahap B dimulai dari nol, atau bisa memanfaatkan
     hasil pencarian Tahap A sebagai titik awal (warm start) mengingat
     kedua grid tumpang tindih di beberapa titik (mis. keduanya kemungkinan
     mencakup sekitar 0,9) — berpotensi memangkas ongkos pengulangan tanpa
     mengurangi kualitas pencarian.

## References

- `docs/superpowers/specs/2026-08-22-segmented-quantile-allocation-design.md`
  — spec alokasi kuantil tersegmentasi yang menjadi konsumen hasil spec ini.
- `docs/batasan-penelitian.md` B-9, B-10, B-11 — komitmen kuantil 0,9 dan
  status data biaya yang menentukan Tahap A/B di Bagian 1.
- Makridakis, S., Spiliotis, E., Assimakopoulos, V., Chen, Z., Gaba, A.,
  Tsetlin, I., & Winkler, R. L. (2021). The M5 uncertainty competition:
  Results, findings and conclusions. *International Journal of
  Forecasting*.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2021). The M5
  competition: Background, organization, and implementation.
  *International Journal of Forecasting*.
- Serafin, T., et al. (2024). Ranking probabilistic forecasting models
  with different loss functions.
- Gneiting, T., & Resin, J. (2022). Model Diagnostics and Forecast
  Evaluation for Quantiles. *Annual Review of Statistics and Its
  Application*, 10.
- Sluijterman, L., et al. (2024). Composite quantile regression with
  XGBoost using the novel arctan pinball loss. *International Journal of
  Machine Learning and Cybernetics*.
- Bröcker, J. (2012). Evaluating raw ensembles with the continuous ranked
  probability score. *Quarterly Journal of the Royal Meteorological
  Society*.
- Narayan, T., et al. (2024). Expected Pinball Loss For Quantile
  Regression And Inverse CDF Estimation. *Transactions on Machine
  Learning Research*.
- van der Laan, N., et al. (2022). The data-driven newsvendor problem:
  Achieving on-target service-levels using distributionally robust
  chance-constrained optimization. *International Journal of Production
  Economics*.
- Ye, Y., et al. (2021). The New Data-Driven Newsvendor Problem with
  Service Level Constraint.
