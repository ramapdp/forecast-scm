# Data Processing & Modelling Pipeline Overview

This document explains, end to end, how raw transactional exports become
model-ready data — through cleaning, feature engineering, and the
modeling-preprocessing stage that feeds XGBoost, Random Forest, and LSTM — and
what the model-training phase, which is not yet built, is expected to do with
it. For the detailed design rationale behind individual stages, see the dated
specs in `docs/superpowers/specs/`.

## 1. Inputs

- `dataset/excel/*.xlsx` — original goods-issued ("Barang Keluar") exports, one
  file per time period.
- `dataset/*.csv` — the same data as semicolon-delimited, UTF-8-BOM CSVs
  (`jan-24.csv`, `feb-24.csv`, `mar-24.csv`, `apr-des-24.csv`,
  `jan-des-25.csv`), each covering a non-overlapping date range from
  Jan 2024–Dec 2025.
- `outlets.json` / `dataset/outlets.csv` — outlet master data (city, online
  ordering channels) per branch, kept in sync by `sync_outlets.py`.
- `dataset/outlet_name_overrides.csv` — manual corrections for branch-name
  mismatches and ambiguous city values between the transactional data and the
  outlet master data.

## 2. Pipeline stages

Run via `notebook/data-processing.ipynb` (interactive/QA, stops at the cleaned
+ feature-engineered `featured.parquet`) followed by `notebook/train_test_split.ipynb`
(interactive, split + export), or `prepare_forecast_data.py` (scripted, runs
the whole thing end to end; `run_qa_checks()` fires on both paths — see stage
12). Order:

1. **Merge raw periods** — `merge_dataset.py` combines the 5 period CSVs into
   one `dataset/dataset.csv`, normalizing to a consistent 7-column schema.
2. **Aggregate duplicate rows** — `aggregate_dataset.py` sums `Kuantitas` for
   any rows sharing the same (date, category, item code, item name, branch,
   unit), collapsing duplicate line items.
3. **Normalize item codes & branches** — `normalize_items.py` strips `xxx.`
   prefixes, unifies separator punctuation in `Kode Barang`, conditionally
   merges codes that only differ by these cosmetic issues (only when the item
   name also agrees), converts a handful of `xxx.`-prefixed items whose
   `Kuantitas` was recorded in grams instead of `Porsi` (Santan Cendol/Gula
   Cendol, factor 40/30 gram per porsi, derived from every raw value being an
   exact integer multiple of that factor) so they merge cleanly into their
   later Porsi-denominated series, drops confirmed-discontinued items
   (Nasi Putih, Cendol Pandan, Ayam Crispy Original/Spicy — the latter two
   were previously force-merged into other SKUs via `EXPLICIT_ITEM_RENAMES`,
   which turned out to combine genuinely different products; both are now
   excluded outright and the renames table is empty), and re-aggregates
   quantities at the normalized-code level.
4. **Filter and canonicalize branches** — `outlet_features.filter_matched_branches`
   drops rows for branches with no corresponding entry in `outlets.csv` (i.e.
   branches that no longer operate), so downstream stages only see currently
   active outlets. `outlet_features.canonicalize_branch_names` then rewrites
   every matched `Nama Cabang` to its outlet's canonical name, so branches
   recorded under two different raw strings in the source data (e.g. a
   legacy short name used only in an early export period, resolved via
   `outlet_name_overrides.csv`) collapse into one continuous branch history
   instead of being split in two. `normalize_items.reaggregate_daily` runs
   again afterward to re-sum any `(item, date, branch)` rows that this
   renaming causes to collide.
5. **Build a dense daily panel** — `build_panel.py` reindexes each
   (item, branch) pair to one row per calendar day across its own
   first-to-last observed date range, filling gaps with `Kuantitas = 0` and
   forward-filling descriptive columns. Dates inside a closure interval
   recorded in `dataset/outlet_closures.csv` produce **no rows at all** — the
   outlet did not exist then, so zero-filling would fabricate demand history —
   and each contiguous run of kept dates is numbered into `segment_id`. Every
   shift-based feature downstream groups by (pair, segment) so no lag, rolling
   window, target, or LSTM sequence bridges a closure. A relocation date from
   `outlet_features.OBSERVED_RELOCATION_DATES` also starts a new segment, but
   keeps every row: the outlet never stopped trading, it moved to a different
   market, and its demand level shifts 2.18x-2.64x across the three moves with
   enough post-move data to measure. Only relocations observable inside the
   data qualify — the five lower-bound dates describe moves that happen after
   coverage ends, so breaking there would carve a one-day segment out of the
   test window. `detect_unrecorded_gaps`
   warns about transaction gaps of ≥14 missing days that the config does not
   explain, but never acts on them. Then `filter_min_history` drops
   pairs with fewer than 60 days of history before the Dec-2025 cutoff
   (insufficient for the longest lag/rolling window); it stays pair-level, since
   history either side of a closure is still real history.
6. **Calendar features** — `calendar_features.py` (`add_calendar_features`):
   day-of-week, day-of-month, month, weekend flag, Indonesian public
   holidays, and flags with days-until/days-since proximity features for
   the 4 high-season events — Ramadan / Eid al-Fitr, Eid al-Adha, Indonesian
   Independence Day (Aug 17), and New Year's Day (Jan 1). Runs before
   outlier handling (next step) so its event flags are available to decide
   which spikes are exempt from capping.
7. **Outlier / demand-spike handling** — `outlier_handling.py`:
   `compute_pair_baseline` computes each (item, branch) pair's historical
   median `Kuantitas` from real (non-zero-filled) train-period transactions
   only (pairs with fewer than 30 such days are marked ineligible and never
   capped). `apply_outlier_capping` flags rows at least 5x that median as
   `is_spike` and produces `Kuantitas_capped` — the spike clipped to
   `median * 5`, *unless* the row falls in a known high-season window
   (Ramadan/Eid al-Fitr/Eid al-Adha/Independence Day/New Year), in which
   case the value is left uncapped since it's treated as a real recurring
   pattern, not noise. `baseline_ratio` and `is_spike` are kept as columns but
   **excluded from `FEATURE_COLS`** — both are derived from the row's own day's
   `Kuantitas`, while every lag and rolling feature stops at H-1, so admitting
   them would make "known at prediction time" mean two things in one row. Raw
   `Kuantitas` is preserved unchanged for target computation, and
   `Kuantitas_capped` now also feeds a second target (stage 8).
8. **Feature engineering** — `prepare_forecast_data.py::build_featured_dataset`:
   - `add_targets`: forecast targets `target_h1`…`target_h7` (raw,
     **uncapped** `Kuantitas` shifted 1–7 days into the future — spikes are
     real demand the model should be evaluated against, not something to
     hide from the label), grouped by (pair, segment).
   - `outlet_features.apply_region_features`: joins `kawasan`/`hari_pengiriman`
     from `dataset/outlet_mapping.csv`, then computes `lead_time_days` per
     row — days from that row's transaction date to the *next* delivery day
     (Region 1 ships Monday & Thursday, Region 2 ships Tuesday & Friday),
     via `outlet_features.compute_lead_time_days`. Always strictly forward
     (never 0, even when the transaction date is itself a delivery day).
   - `outlet_features.apply_outlet_features`: joins static per-branch
     features — `kota`, `has_shopee`, `has_gofood`, `has_grabfood`, and the
     derived `can_order_online`.
   - `outlet_features.add_relocation_feature`: `days_since_relocation` for
     branches merged from a physically-relocated old branch code (negative
     before the relocation date, 0 on it, positive after; `NaN` for
     non-relocated branches). Exists because `canonicalize_branch_names` runs
     before the two steps above, so a relocated branch's `kota`/`kawasan`
     reflect its *current* location for its *entire* history, including
     pre-relocation rows recorded at the old (often different-city) location —
     this flag lets the modelling phase account for that regime shift.
   - `add_lead_time_target`: the core business target,
     `target_lead_time_cumulative` — sum of raw `Kuantitas` over the
     strictly-forward window `(H+1 .. H+lead_time_days)`, i.e. cumulative
     demand until the next delivery. Window length is variable per row
     (from `lead_time_days`).
   - `add_lag_features`: lagged quantities at 1, 2, 3, 7, 14, 21, 28 days,
     computed from `Kuantitas_capped` so a single extreme day doesn't
     dominate the lag inputs.
   - `add_rolling_features`: 7/14/28-day rolling mean & std (also from
     `Kuantitas_capped`), shifted by one day before the window is computed
     so today's value never leaks into its own rolling stats.
   - `compute_branch_stats` / `apply_branch_stats`: branch-level
     characteristics (average daily quantity, demand coefficient of
     variation, volume tier, branch age in days) computed from
     `Kuantitas_capped`, **only from the training period**, then
     frozen/applied to both splits, to avoid leaking future information
     into features.
9. **Export featured dataset** — `export_featured` writes
   `dataset/model_ready/featured.parquet`, the full unsplit cleaned +
   feature-engineered table (currently 1,503,120 rows × 67 columns). This is the file
   `notebook/train_test_split.ipynb` reads for the next stage.
10. **Train/test split** — `split_train_test`: train = everything before
   2025-12-01; test = December 2025. A `target_h{n}`/
   `target_lead_time_cumulative` is left as `NaN` rather than shrinking the
   test window when the target date would fall past 2025-12-31. Training rows
   whose lead-time window reaches into December are **purged**
   (`utils/purging.py`) rather than kept: their label is summed partly over
   test-period demand. 6,188 rows before the fix, 0 after. `fold_train_mask()`
   applies the same purge at each walk-forward fold boundary.
11. **Export splits** — `export_splits` writes `dataset/model_ready/train.parquet`
    and `dataset/model_ready/test.parquet`.
12. **QA checks** — `prepare_forecast_data.run_qa_checks()` runs from both the
    script and the notebook: no negative `Kuantitas`, no duplicate
    (item, branch, date) rows, `Kuantitas_capped` never exceeding raw, no
    `kota == "Unknown"`, no branch missing `kawasan`, no branch mapping to
    more than one city, no row inside a recorded closure interval, `segment_id`
    starting at 1 and contiguous per pair, and no date gap *within* a segment
    (the density invariant `shift` depends on). `main()` additionally asserts
    the output carries every column in `FEATURED_COLUMNS` (64). Notebook-only extras remain: a
    lag/rolling leakage spot-check, per-outlet date ranges, the visual QA
    section, and split-integrity checks in `train_test_split.ipynb`.
13. **Modeling preprocessing** (`utils/modeling_prep.py`, run via
    `notebook/modeling_prep.ipynb` or `python3 -m utils.modeling_prep`) —
    adds `is_event_driven` (per-SKU, from `dataset/event_driven_items.csv`),
    `demand_segment` (Syntetos-Boylan ADI/CV², computed from the training
    period only), `fold_id` (five expanding walk-forward folds over Jul–Nov
    2025; December stays unlabelled as the locked test set), meaning-preserving
    NaN imputation with the indicator columns `was_relocated` / `has_baseline`
    / `has_full_history` / `missing_history_count` (lag and rolling nulls are
    filled too — an LSTM window reaches back over warm-up rows, so leaving them
    put 5.43% of sequence windows out of reach while the tabular matrix stayed
    clean),
    and integer categorical indices with the mapping persisted to
    `dataset/model_ready/category_mapping.json`. Exports
    `dataset/model_ready/model_input.parquet` (1,503,120 rows × 81 columns).
    `FEATURE_COLS` pins the 56 columns all three models train on, deliberately
    excluding `baseline_ratio` and `is_spike`: both derive from the row's own
    day while every lag stops at H-1.
14. **Model adapters** — `to_tabular()` for XGBoost/Random Forest and
    `to_sequences()` for the LSTM (28-day windows ending at the prediction row
    inclusive). Both drop each pair's first 28 warm-up rows, which costs 5.93%
    of rows and 996 test rows (1.81%, nearly all at Bintara, which relocated on
    28 November). `validate_contract()` asserts both expose identical
    `(pair, date)` sets, targets, and fold assignments — and, unless
    `require_finite=False`, that neither feature block contains NaN, since a
    tree model consumes NaN natively while an LSTM turns it into NaN loss.
15. **Evaluation floor** — `utils/evaluation.py` provides pinball loss,
    quantile coverage, and three naive baselines, groupable by
    `demand_segment` or `is_delivery_day`. `roll_mean_7 × lead_time_days`
    reaches MAE 13.05 and pinball@0.9 6.61 on December with no model at all;
    its coverage of 0.61 against a 0.9 service level is the plainest argument
    for training on pinball rather than the mean.

```
raw .xlsx/.csv
  → merge_dataset.py           (dataset/dataset.csv)
  → aggregate_dataset.py       (dedup, in place)
  → normalize_items.py         (clean codes/branches)
  → outlet_features.filter_matched_branches
  → outlet_features.canonicalize_branch_names
  → normalize_items.reaggregate_daily (re-dedup after renaming)
  → build_panel.py             (dense daily panel per active segment,
                                 min-history filter)
  → calendar_features.py       (calendar/holiday/high-season features)
  → outlier_handling.py        (per-pair spike detection + capping)
  → prepare_forecast_data.py   (targets [raw] → region/lead-time → outlet
                                 features → lead-time target [raw] →
                                 lags/rolling/branch stats [capped])
  → export_featured            (dataset/model_ready/featured.parquet)
  → split_train_test → export_splits
  → dataset/model_ready/{train,test}.parquet

  → modeling_prep.py           (event flag → demand segment → folds →
                                 imputation → categorical encoding)
  → export_model_input         (dataset/model_ready/model_input.parquet)
  → to_tabular()   → XGBoost, Random Forest
    to_sequences() → LSTM          (bound by validate_contract())
```

Stages 1–12 are defined once each: `build_featured_dataset()` composes the
load→panel half, and `engineer_features()` the feature half. The notebook calls
those functions rather than re-listing their steps, so the two paths cannot
drift apart (they previously did — the notebook's hand-copied sequence missed
`add_relocation_feature`, silently exporting a 62-column `featured.parquet`).

## 3. What's already model-ready vs. still open

Note first that some constraints are not "still open" work at all — they are
properties of the data and the problem framing that no amount of code can fix.
`docs/batasan-penelitian.md` is the register for those; the most consequential
are that `Tanggal` records pickup rather than order date, that the order book is
never stored (orders can be cancelled), and that the business needs the model for
demand *outside* pre-orders while the target mixes both together.


Fully implemented and verified against the actual parquet output: all 14
stages above, including outlet/location features, region/lead-time features,
the cumulative lead-time target, outlier/demand-spike handling, and the
modeling-preprocessing stage with its two adapters.

Nothing is blocking the modelling phase any more — the last two items closed on
2026-08-16. What remains is routine hygiene:

- Re-run `prepare_forecast_data.py` whenever any pipeline script changes, so the
  exported parquet reflects the current code.
- `calendar_features.py` covers only 2024–2025 and must be extended before 2026
  data arrives, or `check_year_coverage` fails the pipeline hard.

Closed by the data owner on 2026-08-16:

- **Target service level: quantile 0.9, uniform across every SKU.** No
  per-category split — head office ships all items in one consignment, so a
  single service level governs the delivery. The FG-vs-Packaging distinction
  considered earlier is rejected.
- **`dataset/event_driven_items.csv` is final; no flag changed.** The owner
  confirmed the 3 aqiqah SKUs (`FGS-00018`, `FGS-00034`, `PCG-00002`), and the
  remaining 11 were settled from the data instead of a second round of
  questions. Co-occurrence with confirmed aqiqah SKUs (baseline: 0.84% of
  active branch-days) separates them cleanly: `PCG-00028` Cup 60 ml co-occurs
  on **100%** of its days, `PCG-00027` Mika Bento carries **93% of its volume**
  on aqiqah days, while the 9 Loyang SKUs sit at 0.9%–1.4% — no association at
  all. The draft's `true`/`true`/`false` split was right. Two by-products worth
  carrying into modelling: the 9 Loyang SKUs are really **3 series** (one tray
  ships with exactly one box and two sambal cups, holding on 99.9%+ of
  branch-days), and Loyang Besar is a less-popular size of the same daily
  product, not an event item. Details in the spec's Part 6.
- The 8 outlet `Kota Override` values in `dataset/outlet_name_overrides.csv` are
  confirmed correct, `KY001` Kutabumi included — it is `Kabupaten Tangerang`,
  even though the `Kecamatan` column in `outlets.csv` reads Jatiuwung. (The two
  duplicate-branch mappings in the same file — `KY069` → `KY011` "Bekasi
  Galaxy" and `TOD M1 Bandara` → `KY051` — were confirmed on 2026-08-10 as old
  code/name for the same branch, not distinct branches.)
- `dataset/outlet_mapping.csv` was missing 3 branches (Kebuli Yaman Bintara,
  Citayam, Grand Wisata Bekasi — matched fine against `outlets.csv` for
  `kota`/online-channel features, but absent from this file), leaving
  `kawasan`, `hari_pengiriman`, `lead_time_days`, and
  `target_lead_time_cumulative` null for their ~82k rows. Filled on 2026-08-11
  with `kawasan=2`/`hari_pengiriman=Selasa dan Jumat`, inferred from every other
  Kota Depok/Kota Bekasi branch in the file, and **confirmed correct on
  2026-08-16** — so the 82,068 affected rows (5.5% of the dataset) no longer
  rest on an inference. `batasan-penelitian.md` B-8 — the pre-relocation
  schedule — closed a day later: the owner confirmed on **2026-08-17** that
  `outlet_mapping.csv` *is* the schedule archive and that matching on the *new*
  outlet name is the valid rule, so the 205,513 pre-relocation rows (13.7%) are
  no longer a stated assumption. All nine relocated branches carry a non-empty
  `kawasan`/`hari_pengiriman` under their new names.

## 4. Expected modelling phase (not yet built)

Everything below is intentionally out of scope for the data-prep pipeline
and remains to be designed/implemented separately:

- **Model comparison**: Random Forest vs. XGBoost vs. LSTM, forecasting
  `target_lead_time_cumulative` — total demand until the next delivery — at
  (item, branch) daily granularity. `target_h1`…`target_h4` are retained as
  auxiliary targets so a prediction can be decomposed day by day for the SCM
  team; `target_h5`…`target_h7` go unused because `lead_time_days` never
  exceeds 4.
- **Quantile (pinball) loss** rather than mean regression: a stockout costs
  more than overstock, especially for FG, and a mean forecast stocks out
  roughly half the time by construction. All three families support this —
  XGBoost via `reg:quantileerror`, Random Forest via quantile forests, LSTM
  via a custom pinball loss. Training may use `log1p` (both adapters accept
  `log_target`), which is bias-free here because quantiles are equivariant
  under monotonic transforms.
- **Per-segment evaluation**: metrics reported by `demand_segment`, not just
  globally. 75.7% of pairs are intermittent or lumpy, so a single global MAE
  is dominated by pairs where predicting zero is easy.
- **Explainability**: day-by-day decomposition for all models, plus SHAP for
  the winner. LSTM has no TreeSHAP equivalent, so if it wins on accuracy its
  weaker explainability is part of the recommendation, not a disqualification.
- **Cold-start / fallback handling**: (item, branch) pairs dropped by the
  60-day minimum-history filter currently have no forecast at all — a
  fallback strategy (e.g. category-level averages) is undecided. Separately,
  1,059 of 2,979 pairs stopped appearing before December 2025 and are
  therefore never evaluated.
- **Evaluation metrics**: how pinball loss, fill rate, and waste should be
  weighted against each other is not yet defined.

Now implemented and no longer open: categorical encoding, LSTM sequence
windowing and scaling, and the walk-forward validation scheme — see stages
13–14 above.
