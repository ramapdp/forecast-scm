# Modeling preprocessing — design

## Purpose

Take the existing `dataset/model_ready/featured.parquet` and produce a single,
verified `model_input.parquet` that XGBoost, Random Forest, and LSTM can all
consume through thin adapters, such that a comparison between the three is
methodologically defensible.

This spec covers **data preprocessing only** — everything up to the point where
a model can be trained. Training, hyperparameter tuning, model comparison, and
SHAP explainability are a separate follow-up spec.

## Background

### Business goal

The central SCM team ships to Region 1 outlets every Monday & Thursday and to
Region 2 outlets every Tuesday & Friday. The forecast they need is
**`target_lead_time_cumulative`**: total demand from tomorrow until the next
delivery arrives, per (item, branch). That number goes on the delivery note.

The model will be used directly by the SCM team, not just reported on. Two
consequences follow: the preprocessing must be re-runnable on fresh data
(weekly), and predictions must be explainable to the people acting on them.

### Current pipeline state

The data-prep pipeline (12 stages, `docs/pipeline-overview.md`) is complete and
covered by 195 tests. All data-owner confirmations tracked in
`docs/todolist-data-preprocessing.md` have been resolved and wired in.

### EDA findings that shaped this design

Run 2026-08-12 against the current parquet files.

**Scale.** 1,522,868 rows, 2024-01-01 → 2025-12-31. 2,979 (item, branch) pairs
from 70 SKUs across 59 branches over 731 days. Train (before 2025-12-01)
1,467,822 rows; test (December 2025) 55,046 rows.

**`featured.parquet` is stale.** It has 62 columns and is missing
`days_since_relocation`, while `train.parquet`/`test.parquet` have 63. Root
cause identified: `utils/prepare_forecast_data.py:224` calls
`outlet_features.add_relocation_feature` inside `build_featured_dataset()`, but
`notebook/data-processing.ipynb` never calls `build_featured_dataset()` — it
re-lists the nine constituent `prepare_forecast_data.*` calls manually, and that
manual sequence was never updated. The notebook ran last and overwrote the
script's correct output. See Part 1.

**Severe intermittency.** 54.8% of `Kuantitas` rows are zero (44.9% for
`target_lead_time_cumulative`). 776 pairs (23.6%) have >90% zero days, but they
account for only **0.47% of total volume**. Volume is extremely concentrated:
50% of all volume comes from 112 pairs (3.4%), 80% from 276 pairs (8.4%).

**Sparse pairs are lumpy, not dead.** The highest-volume sparse pairs are
Packaging items — `Lunch Box Aqiqah` (97% zero days, but averaging 87 pcs when
it does move), `Mika Bento`, `Cup 60 ml`. `Lunch Box Aqiqah` demand is driven by
customer aqiqah bookings, not by historical pattern; no lag or rolling feature
can predict it. Per-category sparsity: Barang Umum 100%, Bahan Baku (RM) 87.8%,
Snack (FG) 65.9%, Packaging 21.4%, Barang Jadi (FG) 12.5%.

Critically, **sparsity and event-drivenness are different axes that merely
overlap**. `Mika Bento` is sparse at some branches because those branches rarely
use it, not because it waits for an event. Conflating them would teach the model
the wrong signal.

**Test coverage gap.** The test period contains 1,920 of 2,979 pairs — 1,059
pairs stopped appearing before December 2025 and are therefore never evaluated.
Of those, **393 (37%) last appeared in 2025Q4**, immediately before the test
window. There are **0 new pairs** in December, so the test set itself has no
cold-start cases.

**NaN structure.** Ten calendar-proximity columns are 84.6–96.7% null
(`days_into_ramadan`, `days_until_ramadan`, and `days_since_`/`days_until_`
pairs for Eid al-Fitr, Eid al-Adha, Independence Day, and New Year),
`days_since_relocation` is
84.4% null, `baseline_ratio` 14.4%, and lag/rolling columns 1.4–5.5%.

**No ML dependencies exist yet.** `requirements.txt` has pandas, numpy,
openpyxl, pyarrow, holidays, nbconvert, matplotlib — no scikit-learn, xgboost,
or deep-learning framework, and no `models/` directory.

## Decisions

Settled during brainstorming with the data owner on 2026-08-12.

| Decision | Choice | Rationale |
|---|---|---|
| Primary target | `target_lead_time_cumulative` | The number SCM actually ships against |
| Auxiliary targets | `target_h1`…`target_h4` | Day-by-day decomposition of the primary number for explainability. `lead_time_days` never exceeds 4, so `target_h5`–`target_h7` are dropped by this pipeline. Preprocessing only *retains and validates* these columns; training models against them belongs to the modeling spec |
| Loss | **Quantile (pinball)**, not squared/absolute error | Stockout costs more than overstock, especially for FG. Mean regression stocks out ~50% of the time by construction |
| Service level | **Quantile 0.9, uniform across every SKU** | **Confirmed by the data owner 2026-08-16.** No per-category split: head office ships every item in one consignment, so one service level governs the whole delivery. The FG-vs-Packaging distinction considered earlier was rejected for that reason |
| Sparse pairs | Train all; add per-SKU `is_event_driven` flag **and** per-pair statistical segment | They answer different questions — see Part 3 |
| Booking data | Does not exist / not accessible | Event-driven SKUs have an information ceiling, not a model failure. Must be stated explicitly in reporting |
| Validation | Walk-forward, 5 expanding folds (Jul–Nov 2025); December 2025 opened exactly once for the final number | Tuning and winner selection on the same month would make the final figure optimistic. December is also atypical (Christmas / New Year) |
| Explainability | Daily decomposition (all models) + SHAP (winner only) | Decomposition answers "when is demand concentrated"; SHAP answers "why does the model believe this" |
| Architecture | Shared feature table + thin per-model adapters | Fair comparison enforced structurally, not by discipline |
| Notebook role | Stays the primary process driver and may write parquet, but must not re-list pipeline steps | Preserves the existing workflow while making the observed drift impossible |

### Non-goals

- Model training, tuning, comparison, SHAP computation — separate spec.
- Changing any of the 12 existing data-prep stages. This spec builds on top of
  `featured.parquet`, it does not modify how that file is produced (beyond the
  drift fix in Part 1, which changes no logic).
- Cold-start / fallback forecasting for pairs dropped by `MIN_HISTORY_DAYS`.
- Deployment scheduling and serving infrastructure.

## Part 1 — Fix the notebook ↔ script drift

The problem is not one forgotten column. It is that two code paths both encode
the pipeline's step order, and nothing keeps them in sync. `days_since_relocation`
is the first casualty to be noticed; more will follow while both orderings exist.

**Rule: exactly one place defines the step order —
`prepare_forecast_data.build_featured_dataset()`.** The notebook remains the
primary process driver and may still write parquet, but calls the composed
function instead of re-listing its steps.

1. Replace the nine sequential `prepare_forecast_data.*` cells in
   `notebook/data-processing.ipynb` with a single `build_featured_dataset()`
   call. QA and visualization cells stay as they are.
2. Regenerate all three parquet files and verify `featured.parquet` has 63
   columns including `days_since_relocation`.
3. Move the 7 QA assertions that currently live only in the notebook into
   `run_qa_checks()` in `prepare_forecast_data.py`, called from `main()` and
   from the notebook. This closes the long-standing 🟡 item in
   `docs/todolist-data-preprocessing.md` and prevents the mirror-image failure
   where the scripted path is never verified.

This must land before anything else in this spec: every later stage reads
`featured.parquet`, and verifying work built on a wrong file proves nothing.

## Part 2 — Architecture

One new module, `utils/modeling_prep.py`, plus two adapters.

```
dataset/model_ready/featured.parquet      63 columns, written by
        │                                 build_featured_dataset()
        ▼  utils/modeling_prep.py
   ┌──────────────────────────────────────────────┐
   │ 1. add_event_flag()       is_event_driven    │
   │ 2. classify_pairs()       demand_segment     │
   │ 3. assign_folds()         fold_id            │
   │ 4. encode_categoricals()  integer indices    │
   │ 5. validate_contract()    assertions         │
   └──────────────────────────────────────────────┘
        │
        ▼
dataset/model_ready/model_input.parquet   single source of truth
        │
        ├──► to_tabular()    → XGBoost, Random Forest
        └──► to_sequences()  → LSTM
```

`model_input.parquet` is a separate file rather than extra columns on
`featured.parquet` because the two have different lifecycles.
`featured.parquet` states facts about the data and is useful for any analysis.
`model_input.parquet` encodes experiment decisions — fold boundaries, encoding
scheme, segment definitions — which will change repeatedly during
experimentation. Separating them means the stable 195-test data-prep pipeline
does not need re-running every time a fold strategy changes.

All five functions are pure: DataFrame in, DataFrame out, no hidden I/O, so they
are unit-testable and callable from the notebook.

## Part 3 — Shared preprocessing components

### 3.1 `add_event_flag()` → `is_event_driven`

Reads `dataset/event_driven_items.csv` — 70 rows, one per SKU, filled in by the
data owner. Only `is_event_driven` is authoritative; the remaining columns are
supporting evidence to make the decision quick.

A draft was generated on 2026-08-12 (semicolon-delimited, UTF-8-BOM, matching
the repo's other data files) with rows ranked by `prioritas_cek`: 14 rows need a
real decision, 17 warrant a glance, 39 are clearly not event-driven.

**The draft is derived from demand shape, not from item names**, because names
proved unreliable in both directions. The signature of event ordering is *rare
but bulk*: `adi_rata2 >= 50` (moves roughly once every 50+ days) **and**
`rata2_saat_bergerak >= 30` (ships dozens of units when it does). Slow-moving
ordinary items share the first property but not the second — they move one or
two units at a time.

Two corrections the data produced against a name-based guess:

- **Box Loyang (`PCG-00006`/`00007`/`00008`) is almost certainly *not*
  event-driven.** Its statistics are identical to plain Loyang
  (`PCG-00003`/`00004`/`00005`) and Cup Sambal Loyang
  (`PCG-00011`/`00012`/`00013`) — 28.4% zero days, ADI 1.6, 59 branches for the
  Mini size. All three families move as a bundle, and Loyang Mini is close to
  smooth daily demand. Drafted `false`, flagged for confirmation.
- **`PCG-00027` (Mika Bento) and `PCG-00028` (Cup 60 ml) probably *are*
  event-driven**, despite names that suggest routine packaging. Both show the
  bulk signature — ADI 58.6 / 88.3 with mean 38.5 / 84.3 units when moving —
  the same shape as the confirmed Aqiqah items. Drafted `true`, flagged for
  confirmation.

Confirmed-by-name event SKUs, which the demand shape also supports:
`FGS-00018` and `FGS-00034` (Kambing Kebuli Aqiqah Betina / Jantan, ADI 85.9 /
132.5) and `PCG-00002` (Lunch Box Aqiqah, ADI 75.0).

A name rule would also mis-handle `Lunch Box` (`PCG-00001`), which contains
"Box" but is ordinary daily packaging.

### 3.2 `classify_pairs()` → `demand_segment`

Syntetos-Boylan classification per (item, branch) pair from two statistics:

- **ADI** — average interval between non-zero demand days
- **CV²** — squared coefficient of variation of non-zero quantities

| | CV² < 0.49 | CV² ≥ 0.49 |
|---|---|---|
| **ADI < 1.32** | `smooth` | `erratic` |
| **ADI ≥ 1.32** | `intermittent` | `lumpy` |

**Computed from the training period only.** Deriving segments from the full
series would leak future behaviour into a feature.

The column serves two purposes: it is a model input, and it is the axis along
which evaluation metrics are reported. Without per-segment reporting, a global
MAE dominated by mostly-zero pairs can make one model look like the winner when
it only won where predicting zero is easy.

### 3.3 `assign_folds()` → `fold_id`

`fold_id` marks the fold for which a row is the **validation** set (1 = Jul 2025
… 5 = Nov 2025). It is `NaN` for every row outside those five months — both rows
before July 2025 (train-only in all folds) and December 2025 rows (the locked
final test set, which no fold may touch). Training data for fold *k* is every
row dated before the start of fold *k*'s month.

```
                 TRAIN (2024-01 → 2025-11)               TEST
fold 1  ████████████████████░░ Jul
fold 2  ██████████████████████░░ Aug
fold 3  ████████████████████████░░ Sep
fold 4  ██████████████████████████░░ Oct
fold 5  ████████████████████████████░░ Nov
FINAL   ██████████████████████████████  🔒 Dec 2025

█ train   ░ validation   🔒 opened once, at the end
```

A single column fully represents the expanding-window scheme without
duplicating rows.

### 3.4 `encode_categoricals()` → integer indices + saved mapping

All categoricals become integer indices; the mapping is written to
`dataset/model_ready/category_mapping.json`.

Cardinalities are all small — `Kode Barang` 70, `Nama Cabang` 59, `kota` 16,
`Kategori Barang` 8, `demand_segment` 4, `branch_volume_tier` 4,
`hari_pengiriman` 2. Integer indices serve all three model families: XGBoost via
`enable_categorical`, Random Forest via one-hot expansion from the index, LSTM
via an embedding layer.

**The mapping is fit on training data only and persisted.** This is a
correctness requirement, not tidiness: SCM will run this weekly on new data, and
without a stored mapping a 60th branch opening next month would shift every
index and silently invalidate the model with no error raised. Unseen categories
at inference time map to a reserved `UNKNOWN` index.

### 3.5 `validate_contract()` → assertions

Fails loudly when: targets are null outside expected bounds, `fold_id` crosses a
month boundary incorrectly, the categorical mapping contains unknown values, or
the pair count differs from `featured.parquet`.

### 3.6 Note on target transformation

`target_lead_time_cumulative` is heavily right-skewed (99th percentile 488,
maximum 3,067). `log1p` normally conflicts with regression because the mean of
logs is not the log of the mean.

**Quantile regression is exempt.** Quantiles are equivariant under monotonic
transforms: the 0.9 quantile of `log1p(y)`, passed through `expm1`, is exactly
the 0.9 quantile of `y`. Training can therefore happen on the log scale — far
more stable, especially for the LSTM — and be inverted with no bias. This is a
fortunate consequence of the quantile-loss decision.

## Part 4 — Adapters and the cross-adapter contract

### 4.1 Lookback length: 28 days

| L | Train rows lost | **Test rows lost** |
|---|---|---|
| 7 | 1.37% | 0 |
| 14 | 2.74% | 0 |
| **28** | **5.48% (83,412)** | **0** |
| 56 | 10.95% | 0 |

No test rows are lost at any lookback — every pair alive in December 2025 has
far more than 28 days of prior history. Enforcing the identical-rows contract is
therefore effectively free: only per-series warm-up rows are sacrificed, and the
final evaluation keeps all 55,046 rows.

28 is preferred over 14 because `lag_28` is null on exactly the same 5.48% of
rows: cutting the L=28 warm-up also removes every lag/rolling NaN. One cut
solves two problems.

### 4.2 `to_tabular()` — XGBoost & Random Forest

Drop warm-up rows — those where the pair's own row index within its
date-sorted series is less than 28, i.e. the first 28 days of each pair's
history — then return `X`, `y`, `groups` (pair), and `fold_id`.
No scaling. NaNs are left in place — XGBoost handles them natively; Random
Forest gets light imputation inside the adapter.

### 4.3 `to_sequences()` — LSTM

Sliding window per pair producing a `(n_samples, 28, n_features)` tensor.

**NaN imputation.** The ten calendar-proximity columns are only defined within
a ±15-day window (±30 for Ramadan), so null means *"outside that window"*.
Imputing `0` would
assert `days_until_eid_al_fitr = 0`, i.e. *"today is Eid"*, on 96% of rows —
turning a useful feature into an actively harmful one. The sentinel must sit
outside the window and preserve ordinal meaning.

| Column | % null | What null means | Imputation |
|---|---|---|---|
| 10 event-proximity columns (`days_into_ramadan`, `days_until_ramadan`, and `days_since_`/`days_until_` for Eid al-Fitr, Eid al-Adha, Independence Day, New Year) | 84.6–96.7% | outside the proximity window | **`99`** — safely beyond every window, including Ramadan's ±30 |
| `days_since_relocation` | 84.4% | branch never relocated | `0` **+ indicator `was_relocated`** |
| `baseline_ratio` | 14.4% | pair ineligible for capping (<30 real days) | `1.0` **+ indicator `has_baseline`** |
| `lag_*`, `roll_*` | 1.4–5.5% | series warm-up | removed by the L=28 cut |

The two boolean indicators are required because `0` is a legitimate value for
`days_since_relocation` (it means "relocation day"); without an indicator,
"never relocated" and "relocated today" would be indistinguishable.

**Scaling.** Per-feature standardization, **fit on each fold's training data
only** and applied to that fold's validation set. Re-fit per fold rather than
once globally, otherwise December statistics leak into the July fold. Scaler
parameters are persisted alongside `category_mapping.json` for weekly inference.

`target_lead_time_cumulative` and raw quantity features are `log1p`-transformed
(safe per bagian 3.6).

**Sequences spanning a relocation date.** Four branches have exact relocation
dates mid-series, so some 28-day windows mix old-city and new-city demand
patterns. **No special handling is needed.** `days_since_relocation` is already
a per-row feature inside the sequence, so the LSTM observes it crossing from
negative to positive mid-window and can learn the regime shift itself. Dropping
those windows would discard the only transition examples the model has.

### 4.4 The contract

`validate_contract()` enforces three properties with hard assertions:

1. `to_tabular()` and `to_sequences()` return **identical** `(pair, date)` sets —
   not merely equal counts
2. Their `y` vectors are identical in value
3. Their `fold_id` assignments are identical

Without this, "LSTM is 8% better" could actually mean "LSTM was evaluated on a
different 5% of rows".

## Part 5 — Testing and QA

### Unit tests

`test/test_modeling_prep.py`, following the existing 195-test conventions,
written TDD (failing first). Required coverage:

- **Anti-leakage** — `classify_pairs()` produces identical segments given
  train-only vs. full data; `assign_folds()` never places a validation date
  inside the same fold's training range; scalers are re-fit per fold.
- **Imputation** — null `days_until_*` becomes `30`, **not** `0`. This test
  guards against the most dangerous bug in Part 4.
- **Adapter contract** — `to_tabular()` and `to_sequences()` return identical
  `(pair, date)` sets; deliberately corrupting one must fail the assertion.
- **Mapping stability** — an unseen category maps to `UNKNOWN` without shifting
  existing indices.

### Pipeline QA

`run_qa_checks()` in `prepare_forecast_data.py` (the 7 relocated notebook
assertions) plus `validate_contract()` in `modeling_prep.py`. Both are called
from the notebook **and** from the script, so either path is verified.

### Dependencies

`requirements.txt` gains scikit-learn (segmentation utilities, scaling,
Random Forest) at this stage. XGBoost and the deep-learning framework are added
by the modeling spec, not this one.

## Part 6 — Open confirmations

### Blocking — both closed 2026-08-16

| # | Item | Resolution |
|---|---|---|
| 1 | **`dataset/event_driven_items.csv`** — 70 SKUs marked event-driven or not | **Closed.** The 3 aqiqah SKUs (`FGS-00018`, `FGS-00034`, `PCG-00002`) were confirmed pre-order items by the data owner, matching the draft. The remaining 11 were settled from co-occurrence evidence rather than a second round of questions — see "How the last 11 SKUs were settled" below. No flag changed; the draft was right on all 70 |
| 2 | **Target service level** | **Closed — quantile 0.9, uniform across every SKU** (data owner, 2026-08-16). The per-category split is explicitly rejected: head office ships all items in one consignment, so one service level governs the delivery |

### How the last 11 SKUs were settled

Name-based reasoning had already failed in both directions, so the deciding
test was *what else moves on the same branch-day*. Aqiqah is the only
confirmed pre-order behaviour in the data, which makes it a usable reference
pattern: 0.84% of active branch-days carry an aqiqah SKU, so co-occurrence far
above that baseline is evidence of shared ordering behaviour.

| SKU | Days co-occurring with a confirmed aqiqah SKU | Lift vs. 0.84% baseline | Verdict |
|---|---|---|---|
| `PCG-00028` Cup 60 ml | **100%** (197/197 days, 100% of volume) | 118× | Event-driven — component of the aqiqah kit |
| `PCG-00027` Mika Bento | 51.6% of days, **93% of volume** (median 60 units on those days vs. 1 unit otherwise) | 61× | Event-driven — the 7% residual is single-unit dribble |
| 9 Loyang SKUs | 0.9%–1.4% of days; even their p99 days only reach 2.2%–3.7% | 1.1×–1.6× | **Not** event-driven — no association with pre-order behaviour |

Two structural findings came out of the same pass:

- **The 9 Loyang SKUs are 3 series, not 9.** Each size moves as a fixed
  bundle: `Loyang == Box Loyang` on 100% of active branch-days, and
  `Cup Sambal == 2 × Loyang` on 99.9%. One tray ships with one box and two
  sambal cups, deterministically. Total quantities confirm it exactly —
  `PCG-00003` and `PCG-00006` both total 85,898 units, `PCG-00011` totals
  171,896. Any modelling that treats them as nine independent series is
  fitting the same signal nine times.
- **The questionnaire's "Mini & Sedang daily, Besar for events" option is not
  supported.** Loyang Besar moves on 80.7% of the same branch-days as Loyang
  Sedang and shows no event partner; it is a less-popular size of the same
  daily product, not a different kind of item. Its lower activity (20% of days
  vs. 72%) is volume, not behaviour.

What this evidence *cannot* establish: whether an individual tray is ordered a
day ahead by the customer. The order date is not recorded anywhere
(`batasan-penelitian.md` B-1/B-2), so "not an event/aqiqah item" is the
strongest claim the data supports for the Loyang group. The residual risk is
small — `is_event_driven` enters the models as one feature among ~40, not as a
filter — so a wrong flag degrades a feature rather than dropping data.

### Important — defaults exist, but a wrong assumption is costly

| # | Item | Interim default |
|---|---|---|
| 3 | **393 pairs stopped in 2025Q4** — genuine discontinuation or reporting gap? | Treated as genuine |
| 4 | **1,059 pairs dead before December** — does SCM still need forecasts for them? | Trained, not evaluated (no December rows) |
| 5 | **842 pairs dropped by `MIN_HISTORY_DAYS = 60`** (carried over from the existing todolist) — where do new SKUs/branches get a forecast? | No forecast at all |
| 6 | **How often do new branches or SKUs appear?** | Assumed rare |
| 7 | ~~**`kawasan = 2` for Bintara, Citayam, Grand Wisata Bekasi** is still inferred~~ — **confirmed correct by the data owner 2026-08-16**; the inference from neighbouring Kota Bekasi/Kota Depok branches held | Region 2, Tuesday & Friday |
| 8 | **Who runs retraining, and how often?** | Weekly, manually via the notebook |

### Deferred — does not hold up the work

| # | Item |
|---|---|
| 9 | **5 relocation dates are still lower-bound proxies** (Mayor Oking, Cikarang Pusat, Teluk Pucung, Bukit Gading Balaraja, Grand Wisata Bekasi) — need exact dates when available |
| 10 | **`calendar_features.py` covers only 2024–2025** — must be extended before 2026 data arrives, or `check_year_coverage` raises and the pipeline fails hard |
| 11 | **`FGS.00048` (Kambing Oven) totals 4 units across 18 months at 1 branch** — noticed while building the event-driven draft. It shares number 00048 with `FGS-00048` (Kentang Mustofa Mie Goreng), separated only by dot vs. dash; `normalize_items.py` correctly keeps them apart because the names differ, so this is not a bug. But at that volume it is a candidate for `EXCLUDED_ITEMS` alongside the other discontinued SKUs — worth asking whether the item is still sold |

Nothing blocks the modelling phase any more. Items 3–6 and 8 keep their working
defaults; items 9–11 stay deferred.

## References

- `docs/pipeline-overview.md` — the 12 existing data-prep stages
- `docs/todolist-data-preprocessing.md` — resolved data-owner confirmations
- `docs/superpowers/specs/2026-08-08-lead-time-integration-design.md` —
  region/lead-time features and `target_lead_time_cumulative`
- `docs/superpowers/specs/2026-07-21-forecast-data-prep-design.md` — base
  data-prep design
