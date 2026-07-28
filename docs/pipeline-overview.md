# Data Processing & Modelling Pipeline Overview

This document explains, end to end, how raw transactional exports become
model-ready data, and what the (not-yet-built) modelling phase is expected to
do with it. For the detailed design rationale behind individual stages, see
the dated specs in `docs/superpowers/specs/`.

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

Run via `notebook/data-processing.ipynb` (interactive/QA) or
`prepare_forecast_data.py` (scripted, no QA assertions). Order:

1. **Merge raw periods** — `merge_dataset.py` combines the 5 period CSVs into
   one `dataset/dataset.csv`, normalizing to a consistent 7-column schema.
2. **Aggregate duplicate rows** — `aggregate_dataset.py` sums `Kuantitas` for
   any rows sharing the same (date, category, item code, item name, branch,
   unit), collapsing duplicate line items.
3. **Normalize item codes & branches** — `normalize_items.py` strips `xxx.`
   prefixes, unifies separator punctuation in `Kode Barang`, conditionally
   merges codes that only differ by these cosmetic issues (only when the item
   name also agrees), applies confirmed manual renames/exclusions, and
   re-aggregates quantities at the normalized-code level.
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
   forward-filling descriptive columns. Then `filter_min_history` drops
   pairs with fewer than 60 days of history before the Dec-2025 cutoff
   (insufficient for the longest lag/rolling window).
6. **Feature engineering** — `prepare_forecast_data.py`:
   - `add_targets`: forecast targets `target_h1`…`target_h7` (Kuantitas
     shifted 1–7 days into the future).
   - `add_lag_features`: lagged quantities at 1, 2, 3, 7, 14, 21, 28 days.
   - `add_rolling_features`: 7/14/28-day rolling mean & std, shifted by one
     day before the window is computed so today's value never leaks into
     its own rolling stats.
   - `calendar_features.py` (`add_calendar_features`): day-of-week,
     day-of-month, month, weekend flag, Indonesian public holidays, and
     flags with days-until/days-since proximity features for the 4 high-season
     events — Ramadan / Eid al-Fitr, Eid al-Adha, Indonesian Independence Day
     (Aug 17), and New Year's Day (Jan 1).
   - `compute_branch_stats` / `apply_branch_stats`: branch-level
     characteristics (average daily quantity, demand coefficient of
     variation, volume tier, branch age in days) computed **only from the
     training period** and then frozen/applied to both splits, to avoid
     leaking future information into features.
   - `outlet_features.apply_outlet_features`: joins static per-branch
     features — `kota`, `has_shopee`, `has_gofood`, `has_grabfood`, and the
     derived `can_order_online`.
7. **Train/test split** — `split_train_test`: train = everything before
   2025-12-01; test = December 2025. A `target_h{n}` is left as `NaN` rather
   than shrinking the test window when the target date would fall past
   2025-12-31.
8. **Export** — `export_splits` writes `dataset/model_ready/train.parquet`
   and `dataset/model_ready/test.parquet` (currently 49 columns).
9. **QA checks** (notebook only, not run by the plain script) — row-count
   sanity vs. the panel, no duplicate (item, branch, date) rows, no negative
   `Kuantitas`, Ramadan/Eid spot-checks on known dates, a lag/rolling-window
   leakage spot-check, and outlet-join sanity checks (no `kota == "Unknown"`,
   no branch mapping to more than one city).

```
raw .xlsx/.csv
  → merge_dataset.py           (dataset/dataset.csv)
  → aggregate_dataset.py       (dedup, in place)
  → normalize_items.py         (clean codes/branches)
  → outlet_features.filter_matched_branches
  → outlet_features.canonicalize_branch_names
  → normalize_items.reaggregate_daily (re-dedup after renaming)
  → build_panel.py             (dense daily panel, min-history filter)
  → prepare_forecast_data.py   (targets, lags, rolling stats, branch stats)
  → calendar_features.py       (calendar/holiday/high-season features)
  → outlet_features.apply_outlet_features
  → split_train_test → export_splits
  → dataset/model_ready/{train,test}.parquet
```

## 3. What's already model-ready vs. still open

Fully implemented and verified against the actual parquet output: all 9
stages above, including outlet/location features (these are already wired
into `prepare_forecast_data.py`, not a pending addition).

Still open before the data can be fully trusted for modelling:

- Re-run `prepare_forecast_data.py` whenever any pipeline script changes, so
  the exported parquet reflects the current code.
- A handful of outlet `Kota Override` values and one duplicate-branch mapping
  (`KY069` → `KY011`, "Bekasi Galaxy") in
  `dataset/outlet_name_overrides.csv` are best-guess corrections, not yet
  confirmed by the data owner.
- The 7 QA assertions currently live only in the notebook — a plain
  `python3 prepare_forecast_data.py` run does not re-verify them.

## 4. Expected modelling phase (not yet built)

Everything below is intentionally out of scope for the data-prep pipeline
and remains to be designed/implemented separately:

- **Model comparison**: Random Forest vs. XGBoost vs. LSTM, forecasting
  `target_h1`…`target_h7` at (item, branch) daily granularity.
- **Categorical encoding strategy** for `Kode Barang`, `Nama Cabang`,
  `Kategori Barang`, `kota` — currently exported as plain, unencoded
  identifiers; each model family will need its own encoding choice
  (e.g. target/frequency encoding for trees, embeddings for LSTM).
- **LSTM-specific prep**: sequence windowing per (item, branch) pair and
  numeric feature scaling — not needed for tree-based models but required
  for a recurrent model.
- **Validation strategy**: only a single December-2025 holdout exists today;
  a rolling-window / walk-forward validation scheme is a likely follow-up
  for more robust model selection.
- **Cold-start / fallback handling**: (item, branch) pairs dropped by the
  60-day minimum-history filter currently have no forecast at all — a
  fallback strategy (e.g. category-level averages) is undecided.
- **Evaluation metrics & horizon weighting**: how h1 vs. h7 forecast error
  should be weighted/compared across models is not yet defined.
