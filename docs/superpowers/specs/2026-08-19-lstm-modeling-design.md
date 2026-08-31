# LSTM modeling — design

## Status — revisi multi-kuantil (2026-08-24)

Arsitektur head dan kriteria pencarian spec ini **direvisi** mengikuti
`docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`, lewat
checklist di
`docs/superpowers/specs/2026-08-22-model-comparison-refactor-migration.md`
(butir 2). Yang berubah: head output dari 1 neuron menjadi
`len(QUANTILE_SET)` neuron, loss total menjadi jumlah pinball loss lintas
kuantil, dan kriteria pencarian menjadi rata-rata pinball loss lintas kuantil.
Yang **tidak** berubah: ruang pencarian, protokol dua fit, delapan guard,
mekanisme windowing, dan seluruh arsitektur sekuensial di Part 1.

`QUANTILE_SET` saat ini berada di **Tahap A** — 19 titik merata
`[0.05, 0.10, ..., 0.90, 0.95]` — karena tabel pelacakan B-10
(`docs/batasan-penelitian.md`) masih mencatat 0% volume dengan entri biaya
presisi, jauh di bawah ambang ≥80% yang mengaktifkan Tahap B.

Seluruh angka terukur di bagian Background di bawah, dan di
`docs/hasil-modeling-lstm.md`, masih berasal dari run kuantil-0,9 tunggal dan
**akan digantikan** begitu pencarian dijalankan ulang.

## Purpose

Train and evaluate the third and last of the candidate models against
`dataset/model_ready/model_input.parquet`, through the same walk-forward
runner the Random Forest and XGBoost specs used.

The deliverable answers one question the first two could not:
**does a model that reads the raw 28-day sequence beat models that read a
hand-engineered summary of that same window, at predicting the conditional
quantiles of `target_lead_time_cumulative`?** Measured on K1 — the unweighted
mean pinball loss over every point in `QUANTILE_SET` — not on a single
quantile point.

This spec is the follow-up promised by
`docs/superpowers/specs/2026-08-19-xgboost-modeling-design.md`, whose non-goals
list "LSTM training — its own spec, consuming the same runner".

## Background

### What already exists

Everything this design needs is present and tested. Two pieces were written
during the preprocessing spec with this model as their intended consumer:

| Component | Location | What it gives us |
|---|---|---|
| `walk_forward.py` | `utils/` | fold boundaries, row eligibility, and scoring — model injected as a `fit_predict(train, valid) -> np.ndarray` callable |
| `eligible_rows()` | `utils/walk_forward.py` | the one definition of which rows any model may see |
| `purging.lookahead_safe_mask()` | `utils/purging.py` | rows whose whole target window stays strictly before a boundary |
| `evaluation.py` | `utils/` | the metrics and the three naive baselines |
| `FEATURE_COLS` | `utils/modeling_prep.py` | the 56 agreed model inputs, identical for every model |
| `impute_features()` | `utils/modeling_prep.py` | already applied to `model_input.parquet`; must not be applied again |
| `to_sequences()` | `utils/modeling_prep.py` | the readable reference windowing implementation |
| `fit_scaler()` / `apply_scaler()` | `utils/modeling_prep.py` | per-feature standardisation, fit per fold |
| `validate_contract()` | `utils/modeling_prep.py` | the tabular/sequence adapter agreement check |
| `model_common.py` | `utils/` | random search with checkpoint/resume, bundle I/O, `select_best()` |
| `model_xgboost.split_early_stopping()` | `utils/` | the purged training-tail split this spec moves to `model_common` |

The numbers this model is measured against, pooled over the five walk-forward
folds (345,547 validation rows):

| model | pinball@0.9 (5 folds) | pinball@0.9 (folds 1,2,4) | MAE | coverage |
|---|---:|---:|---:|---:|
| XGBoost | 2.390 | 2.400 | 14.31 | 0.909 |
| Random Forest | 2.410 | 2.403 | 15.64 | 0.932 |
| `naive_roll_mean_7` | 4.503 | — | 9.65 | — |

The two tree models are **practically tied**: 0.1% apart on the folds untouched
by model selection. That is the context this spec lands in — the interesting
outcome is not "which model wins" but whether a third model family moves the
number at all.

**Superseded (2026-08-24).** Those are single-quantile numbers, and the tie
above is exactly the situation the multi-quantile migration was designed to
break open: a ranking established at one quantile point is not evidence of a
ranking across the distribution (Serafin et al. 2024). The table this model is
actually compared against becomes the three models' mean pinball over
`QUANTILE_SET`, plus the per-quantile breakdown. The numbers above stay as the
old-criterion reference until the re-runs land.

### Measured facts this design rests on

All measured against `model_input.parquet` on 2026-08-19:

| Fact | Value | Why it matters |
|---|---|---|
| Panel size | 1,502,522 rows x 82 columns | sets every memory figure below |
| Segments | 3,236 (`Kode Barang` x `Nama Cabang` x `segment_id`) | 2,979 distinct pairs |
| Date gaps inside a segment | **0** | window position arithmetic *is* date arithmetic; no date lookup needed at batch time |
| Segments shorter than 29 days | 21 of 3,236 | produce no windows; already removed by `drop_warmup_rows()` |
| Dense sequence tensor | 1,502,522 x 28 x 56 float32 = **9.42 GB** | infeasible on a 16 GB machine — the reason for Part 1 |
| `Kategori Barang_idx` varies within a segment | **301 segments** | a per-segment categorical array would be silently wrong; see 1.1 |
| Other six `_idx` columns varying within a segment | 0 | — |
| `torch==2.8.0` cp39 arm64 wheel | 70 MB, verified via `pip download` | Python 3.9.6 is not a blocker |

### What is settled and inherited

Not reopened here:

- **Primary target** — `target_lead_time_cumulative`.
- **Service level commitment** — quantile **0.9, uniform across every SKU**
  (data owner, 2026-08-16), clarified 2026-08-22 as an *aggregate* commitment at
  the delivery level (B-9). That is the business promise the production model
  serves, and it is no longer the same thing as the model-comparison criterion.
- **Loss and comparison criterion** — mean pinball loss over `QUANTILE_SET`
  (K1), per
  `docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md`
  Bagian 2. Points are averaged unweighted, and every per-quantile score is
  reported alongside the mean.
- **Validation** — walk-forward, 5 expanding folds (Jul–Nov 2025). December
  2025 opens exactly once, after all three models have been compared.
- **Feature set** — `modeling_prep.FEATURE_COLS`, all 56 columns, identical
  for every model. `lag_*` and `roll_*` stay in even though they partly
  duplicate what the sequence already carries: identical features across
  models is the decision that makes the head-to-head legitimate.
- **Row eligibility and scoring** — owned by `walk_forward.py`.
- **Search folds** — 3 and 5. Not a consistency flourish: if the LSTM searched
  on different folds, the "folds 1, 2 and 4 are untouched by model selection"
  slice would collapse for *all three* models at once, and it is the only
  clean number the project has.

## Decisions

Settled during brainstorming 2026-08-19.

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Third contender under an identical protocol | Same features, target, folds, metric, and runner. Any other choice makes the three-way comparison unreportable |
| Framework | **PyTorch** (`torch==2.8.0`) | The deciding factor is bundle persistence: `state_dict()` is a plain dict of tensors and drops straight into the one-file joblib bundle RF and XGB already use, while a Keras model needs a separate `.keras` archive and would force an exception into `model_common.load_bundle()`. Secondary: memory stays flat across ~100 fits without manual `clear_session()`, windowing is plain numpy, and import cost in the test suite is ~1-2s rather than ~5s. Both frameworks have cp39 arm64 wheels (`tensorflow==2.20.0` + `keras 3.10.0`, `tensorflow-metal 1.2.0`), so this is a fit decision, not a feasibility one |
| Window memory | Contiguous panel matrix + end-position index, gathered per batch | The dense tensor is 9.42 GB on a 16 GB machine. The contiguous form is 294 MB and every window is a `sliding_window_view` slice of it |
| History access | `make_fit_predict(params, panel)` closes over the **full panel** | An LSTM window for a 1 July validation row needs 3–30 June feature rows, some of which sit in the purge gap and the 28-day warm-up and therefore appear in neither frame `run_fold` hands over. Safe because a window contains only *feature* rows dated <= H and every lag/rolling feature stops at H-1 — no target value ever enters a window. Purging protects against *training on* those rows' labels, not against reading their features. `walk_forward.py` stays untouched |
| Categorical path | Embedding per `_idx` column, concatenated at the head | XGBoost's search measured that `native` categorical handling beat `ordinal` on this data, i.e. treating an index as an ordered magnitude costs real score. Embeddings are the neural equivalent of the encoding that won |
| Categorical array shape | Per **panel row**, not per segment | `Kategori Barang_idx` changes inside 301 segments. A per-segment array would use a wrong value there, silently. Per-row costs 21 MB and removes the assumption instead of relying on it |
| Epoch count | Early stopping on a purged 30-day training tail, then **refit on all training rows** for the selected epoch count | Identical in shape to XGBoost's round protocol, and for the same reason: the validation fold cannot choose capacity without leaking, and permanently carving out the tail would train the LSTM on a different population than the other two models saw |
| Search budget | Derived from a measured benchmark, capped at 8 hours wall clock | A neural fit is far more expensive than a boosting round and the cost is not known in advance. Fixing N before measuring risks discovering at hour 12 that the search cannot finish |
| `log_target` | Searched | XGBoost measured that log hurts, but for a reason that does not transfer: there, log fought an already-correct pinball objective. Here the question is gradient scale on a target with maximum 3,067. Two opposing arguments means the answer must be measured |

### Non-goals

- **December 2025.** Not touched, not scored, not looked at.
- **The final test protocol.** Opening December requires three decisions that
  should not ride along inside a model spec — which model wins and by what
  criterion, over what range the winner is retrained, and how a one-shot
  evaluation is reported so there is no second attempt. It gets its own short
  spec once `docs/hasil-modeling-lstm.md` exists.
- SHAP or attention-based explainability. Winner only, afterwards.
- Seq2seq or multi-horizon output. The target is one number per row.
- Per-segment specialist models.
- Any change to `FEATURE_COLS`, to `modeling_prep.py`'s feature definitions, or
  to the 12 data-prep stages.
- Any change to the measured Random Forest or XGBoost results. The move in
  Part 1 must leave them reproducible.

## Part 1 — Architecture

```
model_input.parquet   (1,502,522 x 82 — the FULL panel, no cuts)
        |
        +--> walk_forward.eligible_rows()          <- prediction rows (unchanged)
        |
        +--> sequence_windows.build_index(panel)
                 values : float32 (1,502,522, 49)  contiguous, sorted (segment, date)  ->  294 MB
                 cats   : int16   (1,502,522, 7)   one row per panel row               ->   21 MB
                 ends   : int32   window end positions, one per eligible row
```

Four modules, with responsibilities that do not overlap.

### 1.1 `utils/sequence_windows.py` — new

Knows about memory and indices. Knows nothing about LSTMs.

**`build_index(panel) -> dict`**

Sorts the panel by `(Kode Barang, Nama Cabang, segment_id, Tanggal)` and
returns:

- `values` — the 49 dynamic feature columns (`FEATURE_COLS` minus the seven
  `_idx` columns) as one contiguous `float32` array.
- `cats` — the seven `_idx` columns as `int16`, **one row per panel row**.
  Read at the prediction row's own position, not across the window.
  `Kategori Barang_idx` changes inside 301 segments, so a per-segment array
  would be wrong there and nothing would raise. Per-row also gives the
  categorical values their natural meaning: the row's attributes on day H.
- `positions` — for each panel row, its zero-based position within its own
  segment, so eligibility (`position >= 28`) is a lookup rather than a
  recomputation.
- `dates`, `segment_key` — kept for the guards below.

**`window_ends(index, frame) -> np.ndarray`**

Maps a frame of prediction rows (a fold's `train` or `valid`) onto their
positions in `values`. The returned array is in the frame's own row order, so
predictions line up with `valid.index` without a join.

**`gather(values, ends_batch) -> np.ndarray`**

Returns `(B, 28, 49)`. Implemented as `sliding_window_view(values, 28, axis=0)`
— a zero-copy view — indexed by `ends_batch - 27` and transposed. Nothing is
copied until the batch itself is materialised.

Because the panel has **zero date gaps inside a segment** (measured), 28
consecutive array positions are exactly 28 consecutive calendar days. No date
arithmetic runs at batch time.

### 1.2 `modeling_prep.to_sequences()` — untouched

It stays as the readable reference implementation, and that is precisely its
job here: `test_sequence_windows.py` compares the fast path against it
**window for window** on small synthetic frames. The unreadable path is
verified by the readable one rather than by conviction.

`validate_contract()` likewise keeps working unchanged, on fixture-sized
frames.

### 1.3 `utils/model_lstm.py` — new

The model, the loss, the training loop, early stopping,
`make_fit_predict(params, panel)`, `fit_final`, `predict_bundle`,
`DEFAULT_PARAMS`, `SEARCH_SPACE`, and the path constants.

**Network**

```
49 dynamic features x 28 steps --> LSTM(hidden_size, num_layers, dropout) --> h_T --+
                                                                                    +--> concat --> MLP --> len(QUANTILE_SET)
7 _idx columns at row H --> 7 Embeddings --> concat (54 dims) --------------------+
```

The head is the only part of the network the multi-quantile migration touches:
one output neuron per point in `QUANTILE_SET` (19 under Tahap A) instead of one
neuron total. The LSTM trunk, the embeddings, and the concatenation are
unchanged, so the shared representation is learned once and every quantile
reads from it — which is the point of a composite head rather than
`len(QUANTILE_SET)` separate networks.

`num_embeddings` comes from `category_mapping.json` — the maximum index plus
one, which already includes the reserved `UNKNOWN_INDEX = 0` slot — and never
from the values that happen to appear in a fold's training rows. A branch that
opens after this model is trained maps to index 0 and must not index out of
bounds.

| column | `num_embeddings` | `embedding_dim` |
|---|---:|---:|
| `Kode Barang_idx` | 71 | 16 |
| `Nama Cabang_idx` | 60 | 16 |
| `kota_idx` | 17 | 9 |
| `Kategori Barang_idx` | 9 | 5 |
| `branch_volume_tier_idx` | 5 | 3 |
| `demand_segment_idx` | 5 | 3 |
| `hari_pengiriman_idx` | 3 | 2 |
| | | **54 total** |

`embedding_dim = min(16, (num_embeddings + 1) // 2)`, a fixed rule rather than
a search dimension.

**Loss**

Summed pinball across every point in `QUANTILE_SET`, applied directly. `pred`
is `(batch, len(QUANTILE_SET))`, `y` is `(batch, 1)` and broadcasts:

```python
alphas = torch.tensor(QUANTILE_SET).view(1, -1)   # (1, Q)
d = y - pred                                       # (batch, Q)
loss = torch.maximum(alphas * d, (alphas - 1) * d).sum(dim=1).mean()
```

Summing over quantiles and averaging over rows means each quantile head
contributes a gradient of its own scale, which is what keeps the extreme points
from being dominated by the middle of the grid. K1 is the *mean* over quantiles
rather than the sum; the two differ only by the constant factor
`len(QUANTILE_SET)`, so the training objective and the selection criterion
remain the same function up to that constant — the same property
`reg:quantileerror` gives XGBoost.

**Crossing.** Nothing in this loss forces the outputs to be monotone in τ, so
the same crossing check the XGBoost spec adds applies here: for every row, the
prediction at each τ must be ≤ the prediction at the next τ up. If crossing
turns out to be material, the documented options are a monotone
reparameterisation of the head (predict τ=0.05 plus non-negative increments)
or the arctan pinball loss of Sluijterman et al. (2024) — not a post-hoc sort,
which would hide the miscalibration rather than fix it.

**Scaling**

`modeling_prep.fit_scaler()` on the fold's training rows only, applied to the
whole `values` array including context rows. Applying a train-fitted scaler to
any row is safe; what leaks is *fitting* it outside the training window.

One scaler per fold, fit on **all** the fold's training rows and used by both
fits of the two-fit protocol. The early-stopping tail sits inside the training
window, so using its statistics cannot leak into validation; the only effect
is that early stopping is marginally optimistic about itself, and all it
decides is an epoch count. Using one scaler for both fits also means the two
fits see identically scaled inputs, which is what makes `best_epoch` transfer
between them.

### 1.4 `utils/model_common.py` — two changes

- `split_early_stopping()` **moves in** from `model_xgboost.py`. The mechanism
  is not XGBoost-specific. `model_xgboost` re-exports the name, so
  `test/test_model_xgboost.py` stays green with no assertion edited — the same
  pattern, and the same success criterion, as the Part 1 extraction in the
  XGBoost spec.
- `run_search()` is used as-is. No change.

### 1.5 Binding the panel to `run_search`

`model_common.run_search()` calls
`make_fit_predict(candidate, feature_cols=..., quantile=...)`. There is no
slot for the panel, and adding one would change a signature the other two
models already satisfy. So `model_lstm` supplies the panel by binding it:

**`bind_panel(panel) -> Callable`**

Runs `sequence_windows.build_index(panel)` **once**, then returns a callable
with exactly the signature `run_search` expects, closing over that index. The
index costs a sort of 1.5M rows; building it per candidate would repeat that
work N x 2 times for no reason.

The notebook therefore passes `make_fit_predict=model_lstm.bind_panel(panel)`
and `df=panel` — the same frame in both places, so the rows `run_search`
scores and the rows the windows are cut from cannot drift apart.

`run_search` catches `(MemoryError, ValueError)` and its docstring is explicit
that the tuple is narrow on purpose, because an unexpected exception type is a
bug and a bug must not be laundered into a NaN row. That tuple is **not
widened** for the LSTM: `RuntimeError` is what PyTorch raises for genuine bugs
as well as for out-of-memory, so catching it would hide the first kind. The
NaN-loss guard instead raises `ValueError` explicitly, which the existing tuple
already covers.

### 1.6 `utils/walk_forward.py` — zero changes

The LSTM enters through one injected callable, exactly as the other two do.
That is the structural guarantee — not a procedural one — that all three
models are scored on identical rows.

## Part 2 — Search and the two-fit protocol

### 2.1 Space

```python
SEARCH_SPACE = {
    "hidden_size":   [64, 128, 256],
    "num_layers":    [1, 2],
    "dropout":       [0.0, 0.2, 0.3],
    "learning_rate": [3e-4, 1e-3],
    "batch_size":    [1024, 2048],
    "log_target":    [False, True],
}
```

144 combinations — far smaller than XGBoost's 2,592, so the same number of
draws covers this space much more densely.

**All candidates must be re-run under the multi-quantile head (2026-08-24).**
The search space and the seed are unchanged, so the same draws are evaluated,
but every candidate now trains a `len(QUANTILE_SET)`-output head against a
summed pinball loss. The recorded scores in
`dataset/model_ready/lstm_search_results.csv` and the winner in
`lstm_best_params.json` do not carry over.

**Ruang pencarian dipulihkan ke 144 di kode (keputusan pemilik proyek,
2026-08-24).** Spec ini selalu menuliskan ruang 144 di atas, tetapi
`SEARCH_SPACE` di `utils/model_lstm.py` sempat dipangkas menjadi 48 —
`hidden_size` kehilangan 256 dan `num_layers` dikunci ke `[1]` — setelah ongkos
per epoch diukur pada 2026-08-19. Pemangkasan itu **keputusan ongkos, bukan
temuan**: tidak ada satu pun bukti bahwa lapisan kedua atau hidden 256 tidak
menolong, karena keduanya tidak pernah dicoba (dinyatakan terbuka di bagian 18
`metodologi-pemodelan-dan-pemilihan-model.md` dan di `hasil-modeling-lstm.md`).
Karena run multi-kuantil mencari ketiga model ulang dari nol, dua dimensi
kapasitas itu dikembalikan dan kode kembali sama dengan spec: ruang 144, bukan
48. `num_layers` juga dikembalikan ke `2` di `DEFAULT_PARAMS`, sesuai spec.

Deliberately absent, each because it already has a mechanism:

| Not searched | Why |
|---|---|
| epoch count | early stopping decides it per candidate per fold |
| `embedding_dim` | fixed rule from cardinality (1.3). Searching it would spend the tightest budget of the three models on seven numbers the cardinalities already determine |
| `grad_clip` | pinned at 1.0 — standard LSTM practice, not a research question |
| `weight_decay` | `dropout` already covers the regularisation axis |
| optimizer | Adam, fixed |

```python
DEFAULT_PARAMS = {
    "hidden_size":   128,
    "num_layers":    2,
    "dropout":       0.2,
    "learning_rate": 1e-3,
    "batch_size":    1024,
    "log_target":    False,
    "grad_clip":     1.0,
    "random_state":  42,
}
```

`DEFAULT_PARAMS` is the benchmark configuration, not a prediction of the
winner. Its job is to measure `sec_per_epoch` and `best_epoch` so 2.2 has
numbers to work with — the same role `DEFAULT_PARAMS` played in the XGBoost
benchmark.

`nn.LSTM` ignores its `dropout` argument when `num_layers=1`. So that the flag
is not meaningless across half the space, `dropout` is always applied in the
head MLP, and additionally between LSTM layers when `num_layers=2`.

### 2.2 Budget formula

The benchmark runs `DEFAULT_PARAMS` on fold 5 and measures `sec_per_epoch`,
`best_epoch`, and peak RSS. It runs on **CPU and MPS**, and the faster device
is used for the rest of the work and recorded in the results document.

One two-fit candidate costs roughly
`sec_per_epoch x (2 * best_epoch + patience)`. Fold 3's training window is
shorter than fold 5's, so charging both folds at fold 5's rate is
conservative:

```
N = floor(28_800 / (2 * sec_per_epoch * (2 * best_epoch + 5)))
```

clamped to `6 <= N <= 20`. The ceiling is 8 hours of wall clock.

The lower bound is not cosmetic. Below six draws the result is not a search,
and it is more honest to report it as "one manually chosen configuration" than
to call it one. If the formula lands under 6, that is the signal to shrink the
search space — not to quietly raise the ceiling.

**Budget under the multi-quantile head (decision 2026-08-24).** The formula
above produced **N = 12** on the single-output head. A 19-output head raises
`sec_per_epoch`, so re-deriving N from the same 8-hour ceiling would shrink the
search — and shrinking it would mean the LSTM is searched less thoroughly than
before precisely when its architecture changed, making the re-run not
comparable to the XGBoost and RF re-runs.

So **N is pinned at 12** for the multi-quantile re-run rather than re-derived.
The consequence is stated rather than hidden: the 8-hour wall-clock ceiling
will very likely be exceeded, and the actual wall clock is recorded in
`docs/hasil-modeling-lstm.md` as a measured cost, not treated as a failure.
This follows the project's standing position that computation cost is
deliberately set aside in favour of finding the best model, and closes the
budget half of open question 2 in the multi-quantile spec. The benchmark still
runs and still reports `sec_per_epoch` and `best_epoch` — they now document
what the multi-quantile head costs, instead of deciding N.

**Penyetaraan anggaran — menggantikan keputusan N = 12 di atas (pemilik
proyek, 2026-08-24, sesudah T-7).** Paragraf di atas dipertahankan sebagai
jejak keputusan, tetapi angkanya **tidak lagi berlaku**. Anggaran pencarian
LSTM dinaikkan dari 12 menjadi **30 kandidat**, setara XGBoost, dan
konfigurasi terbaiknya diulang pada **3 seed**.

Alasannya adalah validitas perbandingan, bukan ongkos. Ketimpangan anggaran
(RF 18, XGBoost 30, LSTM 12) selama ini tercatat sebagai keterbatasan yang
harus dibaca bersama hasilnya — dapat diterima ketika hanya LSTM yang berubah,
tetapi tidak lagi ketika **ketiga model dicari ulang penuh dari nol**. Pada
run multi-kuantil, ketimpangan itu berhenti menjadi warisan dan menjadi pilihan
yang diambil ulang secara sadar, jadi ia dibereskan sekalian.

Yang dipertaruhkan adalah pertanyaan inti penelitian ini: kalau LSTM kalah di
K1, kita harus bisa membedakan **arsitekturnya memang kurang cocok** untuk
persoalan ini dari **pencariannya yang paling dangkal di antara ketiganya**.
Dengan 12 draw pada ruang 144 (8,3% ruang) melawan 30 draw XGBoost, kekalahan
LSTM tidak dapat diatribusikan — dan atribusi itulah keluaran yang dicari.
Pada 30 draw, keduanya sama-sama mengambil 30 draw dan kesimpulan "kalah karena
arsitektur" menjadi klaim yang bisa dipertahankan.

Tiga hal berubah bersamaan, dan ketiganya bergerak ke arah yang sama:

| Yang berubah | Dari | Menjadi | Alasan |
|---|---|---|---|
| Jumlah kandidat | 12 | **30** | setara XGBoost; kekalahan LSTM jadi bisa diatribusikan |
| Ruang pencarian | 48 (kode) | **144** | `num_layers` dan `hidden_size` dipulihkan — dipotong karena ongkos, bukan karena terbukti tidak menolong (lihat bagian 2.1) |
| Pengulangan seed | tidak ada | **3 seed pada konfigurasi terbaik** | varians terukur langsung, bukan diduga dari selisih antar fold |

Pengulangan tiga seed menjawab keberatan yang berdiri sendiri: LSTM satu-satunya
model di perbandingan ini yang inisialisasi bobotnya acak, sehingga selisih K1
kecil antara LSTM dan model pohon selama ini tidak dapat dibedakan dari derau
seed. Sampai sekarang variansnya hanya bisa **diduga** dari selisih antar fold,
yang mencampur varians seed dengan varians data. Tiga seed pada konfigurasi
pemenang (fold pencarian yang sama, 3 dan 5) memisahkan keduanya: sebarannya
dilaporkan di `docs/hasil-modeling-lstm.md` dan dipakai membaca ambang 2% di K1.
Hanya konfigurasi pemenang yang diulang — mengulang ketiga puluh kandidat
berarti membayar tiga kali ongkos pencarian untuk menjawab pertanyaan yang
hanya relevan bagi pemenang.

Konsekuensi ongkos dinyatakan terbuka: ini menaikkan ongkos Fase 3 secara
signifikan, karena LSTM adalah model termahal per fit dan ketiga perubahan di
atas mengalikan ongkosnya (30/12 kandidat x head 19 keluaran, ditambah dua fit
tambahan untuk seed kedua dan ketiga), sebagian di antaranya kini boleh
menggambar `num_layers=2` dan `hidden_size=256` yang per epoch-nya jauh lebih
mahal — 259 s vs 104 s pada pengukuran 2026-08-19. Plafon 8 jam sudah
ditinggalkan pada keputusan sebelumnya; keputusan ini memperbesar
kelampauannya. Wall clock sebenarnya dicatat di `docs/hasil-modeling-lstm.md`
sebagai ongkos terukur.

#### Empat ketimpangan, satu tempat (ditutup 2026-08-24)

Tiga baris di tabel di atas adalah tiga ketimpangan yang selama ini dibawa
LSTM. **Ada yang keempat**, ditemukan saat migrasi multi-kuantil dikerjakan
(T-12, 2026-08-24), dan ia dicatat di sini bersama ketiganya karena keempatnya
menjawab pertanyaan yang sama — apakah hasil LSTM dapat diatribusikan ke
arsitekturnya — dan tersebar di empat tempat berarti tidak ada satu pun
pembaca yang melihat bobot gabungannya.

| # | Ketimpangan | Sebelum 2026-08-24 | Sesudah |
|---|---|---|---|
| 1 | Anggaran pencarian | 12 kandidat (RF 18, XGB 30) | **30**, setara XGBoost |
| 2 | Ruang pencarian | 48 titik — `num_layers` dan `hidden_size` dipangkas | **144**, dua dimensi kapasitas dipulihkan |
| 3 | Pengulangan seed | tidak ada, padahal satu-satunya model berinisialisasi acak | **3 seed** pada konfigurasi pemenang |
| 4 | **Cakupan uji** | **paling rendah dari ketiga model** — lihat di bawah | tes integrasi `run_fold` ditambahkan, setara XGBoost |

**Ketimpangan keempat: cakupan uji.** Sampai 2026-08-24
`test/test_model_lstm.py` memanggil `make_fit_predict(...)` langsung dan
memeriksa keluarannya sendiri; tidak satu pun tesnya melewati
`walk_forward.run_fold()` atau `model_common.run_search()`. Konsekuensinya
terlihat saat kontrak `fit_predict` berubah dari `(n,)` menjadi `(n, K)`:
`test_model_random_forest.py` dan `test_model_xgboost.py` langsung merah di
sebelas tes, `test_model_lstm.py` tetap hijau — **bukan karena `model_lstm.py`
sudah benar, melainkan karena tidak ada yang mengeceknya.**

Ini kualitatif berbeda dari ketimpangan 1–3. Ketiganya adalah keputusan
anggaran yang diambil sadar dan dicatat; yang keempat tidak pernah diputuskan
oleh siapa pun — ia celah yang tidak terlihat sampai sebuah perubahan kontrak
menyorotinya. Dan ia yang paling berbahaya bagi validitas perbandingan: dua
model lain punya jaring pengaman yang menangkap ketidakcocokan integrasi
sebelum sebuah run berjam-jam dimulai, LSTM tidak. Sebuah angka LSTM yang
salah karena kontrak akan tetap berbentuk benar.

Perbaikannya: `TestWalkForwardIntegration` di `test/test_model_lstm.py`
(2026-08-24), setara yang sudah dimiliki XGBoost — `run_fold` dijalankan
sungguhan lewat `bind_panel`, hasilnya diperiksa punya satu baris per titik
kuantil, dan satu tes memastikan pelanggaran bentuk memang ditolak runner-nya.
Ditambah `TestSeedRepeats` untuk protokol tiga seed di bagian 2.3.

Keempat baris ini bersama adalah alasan konkret penyetaraan anggaran
diputuskan. Ia bukan kemurahan hati terhadap LSTM: tanpa keempatnya, kalimat
"LSTM kalah karena arsitekturnya" tidak dapat dipertahankan pada dataset ini,
dan kalimat itulah keluaran yang dicari penelitian ini.

### 2.3 Protocol

- Scored on **folds 3 and 5** only, seed 42.
- Criterion: **pooled mean pinball over `QUANTILE_SET`** (K1), weighted by row
  count, so November — the smallest fold — does not count as much as September.
  The row-count weighting applies across folds; the averaging across quantile
  points inside each fold stays unweighted.
- **No subsampling.** Every training row of each fold is used.
- Each finished candidate is flushed to
  `dataset/model_ready/lstm_search_results.csv` immediately, and a restart
  resumes from it. At an 8-hour budget this is the single reason the mechanism
  exists.
- A candidate that raises — including one whose loss goes NaN — is recorded
  with NaN metrics and a populated `error` column rather than aborting the run.
- The stale-checkpoint guard carries over: resuming across a changed search
  space or seed is refused.

The winner is refit and reported across all five folds.

**Pengulangan tiga seed pada pemenang (2026-08-24).** Setelah `select_best()`
menetapkan konfigurasi terbaik, konfigurasi itu — dan hanya itu — dijalankan
ulang pada seed `42, 43, 44` di fold pencarian yang sama (3 dan 5), dengan
protokol dua fit yang identik. Yang dilaporkan adalah sebaran K1 ketiganya
(min, mean, max, dan rentangnya), disimpan ke
`dataset/model_ready/lstm_seed_repeats.csv`. Seed 42 dipakai apa adanya sebagai
salah satu dari ketiganya, sehingga baris pertamanya harus sama persis dengan
baris pemenang di `lstm_search_results.csv` — kalau tidak, ada
nondeterminisme yang belum tertangkap dan itu temuan tersendiri.

Rentang seed ini dibaca bersama ambang 2% di K1: kalau rentang antar seed
melebihi jarak K1 antara LSTM dan model pohon, jarak itu tidak dapat dibaca
sebagai perbedaan antar model. Model final tetap dilatih pada seed 42, supaya
`random_state` di `lstm_best_params.json` dan bundle-nya tetap satu angka yang
dapat direproduksi.

### 2.4 The two-fit protocol

1. Fit on `fit_rows` with the purged 30-day tail as the eval set,
   `patience = 5`, `MAX_EPOCHS = 100`. Record `best_epoch`.
2. Discard that model. Re-initialise from the same seed and fit on **all**
   training rows for exactly `best_epoch` epochs, with no early stopping.

Predictions come from the second model, inverted through
`modeling_prep.inverse_log_target()` when `log_target` is set, then clipped to
`>= 0`.

The early-stopping tail is about 5% of training rows (fold 5: 1,292,778 ->
1,224,830 + 65,140), so one epoch of the second fit contains ~5% more gradient
steps than one epoch of the first. That difference is accepted and stated in
the results document: pinning by *epoch* means "the same number of passes over
the data", which is the more meaningful invariant than a fixed step count.

### 2.5 Eight guards

What makes leakage dangerous in a sequence model is that it fails silently —
the tensor shapes stay correct either way. So the guards are assertions, not
care.

| # | Guard | When |
|---|---|---|
| G1 | For every `e`: `date[e] - date[e-27] == 27 days` **and** `segment[e] == segment[e-27]` | once, vectorised, in `build_index` |
| G2 | `TARGET_COL` is not among the 49 dynamic columns | in `build_index` |
| G3 | No training `end` is dated on or after the fold's month start | per fold |
| G4 | Early-stopping tail rows are not among the first fit's `ends` | per fold |
| G5 | No window contains a date on or after 2025-12-01 | per fold |
| G6 | Each window's maximum date equals its own prediction row's date — no window ever reaches forward | once, vectorised |
| G7 | `fit_predict` returns exactly `len(valid)` rows x `len(QUANTILE_SET)` columns, ordered to match `valid.index` | per fold |
| G8 | Predictions are non-negative; `log_target` is inverted **before** clipping | per fold |
| G9 | No quantile crossing: predictions are non-decreasing across `QUANTILE_SET` within every row | per fold |

G9 is new with the multi-quantile head (2026-08-24). It is reported as a rate
(fraction of rows with at least one inversion) rather than asserted to zero,
because a composite pinball head has no structural monotonicity guarantee — a
non-zero rate is a finding to write down and act on, not a crash.

G1 and G6 carry the most weight and both are cheap, because the panel is dense:
array position *is* date arithmetic.

Determinism: seed 42 through `torch.manual_seed`, a separate generator for
batch shuffling, and deterministic sort order in `build_index`. Windows are
shuffled across segments during batching — each window is self-contained, so
no ordering needs preserving between batches.

## Part 3 — Reporting

`docs/hasil-modeling-lstm.md`, in Indonesian, following the structure of
`docs/hasil-modeling-rf.md` and `docs/hasil-modeling-xgb.md`: summary,
evaluation setup, benchmark, search results, walk-forward per fold, per
`demand_segment`, per `is_delivery_day`, final model, reproduction,
limitations.

Two sections the earlier documents do not have.

**The budget formula with the measured numbers that filled it in.** 2.2 is
worthless if the `sec_per_epoch` and `best_epoch` it consumed are not written
down — the value of N would look arbitrary to anyone reading later.

**Three-way head-to-head.** Legitimate because all three were scored on
identical rows, guaranteed by `walk_forward.eligible_rows()`. Two slices:

1. All five folds.
2. **Folds 1, 2 and 4** — untouched by model selection for all three models,
   since all three searched on folds 3 and 5. This is the clean number.

Four asymmetries are stated in that section rather than buried. Three are
protocol:

| | Random Forest | XGBoost | LSTM |
|---|---|---|---|
| Search budget | 18 candidates | 30 candidates | 12 candidates (pinned, 2.2) |
| How capacity is chosen | tree count pinned | rounds by early stopping + refit | epochs by early stopping + refit |
| Categorical path | one-hot | native categorical | embeddings |
| Multi-quantile mechanism | read from the same leaves, no retrain | `quantile_alpha` list, one fit | one output neuron per τ, summed loss |

The last row is new with the multi-quantile migration and is the sharpest
asymmetry in the table: the Random Forest gets every quantile for free from a
search that never had to be repeated, while XGBoost and the LSTM both paid for
a full re-search. That is a property of the model families, not a protocol
choice, but a reader comparing K3 (cost and reproducibility) is entitled to see
it stated.

The third row is not unfairness — each was chosen by that model's own search or
by its model family's nature. It is written down so no reader assumes the three
were treated identically down to that layer.

The fourth is deeper and gets stated most plainly:

> **The LSTM sees more input built from the same features.** Random Forest and
> XGBoost receive a hand-engineered summary of the last 28 days — `lag_1`
> through `lag_28`, `roll_mean_*`, `roll_std_*`. The LSTM receives those 28
> days intact: 49 columns x 28 steps. The feature set is identical; the amount
> of information reaching the model is not.

That is not a protocol defect — it is the research question. But if the LSTM
wins, a reader is entitled to know it won with richer input; and if it loses
*despite* richer input, that is a far stronger finding than "the LSTM lost".

MAE is reported for context only, never as a winning criterion. Comparing a
quantile model's MAE against a mid-point baseline punishes it for doing exactly
what was asked. The mean pinball over `QUANTILE_SET` decides, with the
per-quantile breakdown reported beside it.

## Part 4 — Artifacts

| Artifact | Path | Versioned |
|---|---|---|
| Search results | `dataset/model_ready/lstm_search_results.csv` | No |
| Seed repeats on the winner | `dataset/model_ready/lstm_seed_repeats.csv` | No |
| Selected hyperparameters | `dataset/model_ready/lstm_best_params.json` | No |
| Full results table | `dataset/model_ready/lstm_walk_forward_results.csv` | No |
| Trained model | `models/lstm_q90.joblib` | No — `models/` is already gitignored |
| Notebook | `notebook/modeling_lstm.ipynb` | **Yes** |
| Results summary | `docs/hasil-modeling-lstm.md` | **Yes** |

The bundle is one joblib file, matching RF and XGB: `state_dict`,
`dynamic_cols`, `idx_cols`, `embedding_sizes`, `scaler`, `log_target`,
`best_epoch`, `quantiles`, `feature_cols`, `n_train`. `quantiles` holds the
whole of `QUANTILE_SET` in head order — the column order of the head is
unrecoverable from `state_dict` alone, so recording it is what makes the
bundle reloadable.

*Koreksi 2026-08-24 (implementasi butir 0b): kuncinya dinamai `quantiles`,
bukan `quantile`. Draf spec ini menulis "`quantile` kini memuat seluruh
QUANTILE_SET", yang berarti kunci bernama tunggal berisi sembilan belas
angka — dan ketiga model harus dibaca oleh kode pemuat yang sama. Ketiga
bundle kini memakai `quantiles`; `QUANTILE = 0.9` tetap ada di ketiga modul
sebagai konstanta service level B-9, terpisah dari grid evaluasi.* The `q90` in the filename is historical and left alone
so the artifact path stays stable. A model reloaded next
month against columns in a different order does not fail — it predicts
confidently from the wrong features, which is worse.

`notebook/modeling_lstm.ipynb` mirrors `modeling_xgb.ipynb`: benchmark, search,
final walk-forward, results. Outputs are cleared before commit, because the
evidence lives in the CSVs and in `docs/`, not in cell output.

## Part 5 — Testing

TDD, following the conventions of the existing suites. Small synthetic frames,
not the real parquet, so the tests stay fast and deterministic.

**`test/test_sequence_windows.py`**

- **Fast path equals reference path.** `build_index` + `gather` produce windows
  identical to `modeling_prep.to_sequences()`, window for window, on a
  synthetic frame. This is the most important test in the suite.
- G1 fires when the panel is given an artificial date gap, and when a window is
  asked to cross a `segment_id` boundary.
- G2 fires when `TARGET_COL` is smuggled into the dynamic columns.
- G6: each window's maximum date equals its prediction row's date.
- `ends` are ordered and correspond one-to-one with `eligible_rows()`' row
  order.
- `gather()` is correct for a **shuffled** batch of ends, not only a contiguous
  one.
- `cats` reads the prediction row's own values — verified on a synthetic
  segment where the category changes mid-series, the case that made 301 real
  segments unsafe.

**`test/test_model_lstm.py`** — the anti-leakage tests matter most, because
their failure mode is silent:

- No training `end` is dated on or after the fold's month start; no window
  contains a date on or after `TEST_START`.
- Early-stopping tail rows are absent from the first fit; the second fit sees
  `len(fit_rows) + len(es_rows)` rows and runs exactly `best_epoch` epochs.
- **Embeddings never index out of bounds.** `num_embeddings` comes from
  `category_mapping.json`, not from the fold's observed values; an unseen
  category falling to `UNKNOWN_INDEX = 0` stays valid. This failure would
  otherwise surface months later, when the 60th branch opens.
- `log_target` round-trips; predictions are non-negative; length equals
  `len(valid)` and the order matches `valid.index`.
- `predict_bundle()` forces the recorded column order: a column-shuffled input
  frame produces identical predictions.
- Same seed produces identical predictions.
- A training run whose loss goes NaN raises `ValueError`, so
  `run_search`'s existing `catch` tuple records it as a failed candidate rather
  than letting NaN predictions reach scoring.
- `bind_panel()` returns a callable matching `run_search`'s expected signature,
  and builds the window index exactly once no matter how many times that
  callable is invoked.

**`test/test_model_common.py`** gains coverage for the moved
`split_early_stopping()`.

**`test/test_model_xgboost.py` — not edited.** It must stay green through the
move, with no assertion changed. That is the regression test for Part 1.4.

## Part 6 — Dependencies

`requirements.txt` gains one line:

```
torch==2.8.0
```

The cp39 arm64 wheel was verified on 2026-08-19 via `pip download` against the
project's Python 3.9.6 venv (70 MB, `torch-2.8.0-cp39-none-macosx_11_0_arm64.whl`).

No other new dependency: `joblib`, `numpy`, `pandas` and `pyarrow` are already
pinned. No `libomp`-style external runtime is required.

## Part 7 — Risks

| Risk | Handling |
|---|---|
| Loss goes NaN — an LSTM on a heavy-tailed target (max 3,067) | `grad_clip=1.0`, plus a guard that fails that candidate with a populated `error` column instead of poisoning the whole search |
| MPS is slower than CPU, or numerically wrong | the benchmark measures both, the winner is used and recorded, and CPU is always the fallback |
| The budget formula lands below N=6 | the signal to shrink the search space, not to quietly raise the 8-hour ceiling |
| An 8-hour search is killed by the OS with no exception to catch | `model_common.run_search()` checkpoint/resume; the CSV doubles as the only progress signal available from inside a notebook cell |
| Overfitting to 3,236 segments | `dropout` is searched; early stopping runs per fold |
| Embeddings index out of range at inference | `num_embeddings` from `category_mapping.json`, tested explicitly |
| The Part 1.4 move silently changes XGBoost behaviour | `test/test_model_xgboost.py` runs unmodified; the XGB numbers stay reproducible from the same artifacts |
| The LSTM loses to XGBoost, to the Random Forest, or to a naive baseline | A legitimate result, reported as one. Given that RF and XGB are already tied at 2.40, "all three are equivalent" is a finding about this data's information ceiling, not a failure — and `docs/batasan-penelitian.md` B-1/B-2/B-3 already predicts where that ceiling comes from |

## References

- `docs/superpowers/specs/2026-08-19-xgboost-modeling-design.md` — the
  two-fit protocol, `model_common`, and the search protocol this spec reuses
- `docs/superpowers/specs/2026-08-18-random-forest-modeling-design.md` — the
  walk-forward runner and the settled quantile decision
- `docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md` — the
  shared feature table, the adapter contract, `to_sequences()`, and the
  lookback-28 decision
- `docs/superpowers/specs/2026-08-22-multi-quantile-evaluation-design.md` —
  the definition of `QUANTILE_SET` and of criteria K1/K2 this spec now targets
- `docs/superpowers/specs/2026-08-22-model-comparison-refactor-migration.md` —
  the checklist under which this spec was revised
- `docs/hasil-modeling-rf.md`, `docs/hasil-modeling-xgb.md` — the measured
  results this model is compared against
- `docs/batasan-penelitian.md` — B-1/B-2/B-3, the pickup-date information
  ceiling that bounds every model here
