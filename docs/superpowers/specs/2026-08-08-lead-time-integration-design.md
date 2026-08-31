# Region / lead-time integration — design

## Purpose

Finish the region/lead-time integration flagged as the highest-priority gap in
`docs/todolist-data-preprocessing.md` (marked 🔴): wire the already-drafted
`outlet_features.apply_region_features` into the scripted pipeline, make
`lead_time_days` vary per row instead of being a flat constant, and build the
cumulative-demand-until-next-delivery target that is the core business goal
of this forecasting project (`eda.ipynb`'s "Business context" cell). Also
separates the "clean + engineer features" stage from the "split train/test"
stage into their own notebooks, matching this repo's existing
one-notebook-per-stage pattern (`merge_and_aggregate.ipynb` →
`data-processing.ipynb` → ...).

## Background

The central SCM team ships to **Region 1** (`kawasan = 1`) outlets every
**Monday & Thursday**, and to **Region 2** (`kawasan = 2`) outlets every
**Tuesday & Friday**. `dataset/outlet_mapping.csv` already carries `kawasan`
and `hari_pengiriman` ("Senin dan Kamis" / "Selasa dan Jumat") per branch, and
`outlet_features.apply_region_features` can already join those columns onto
`Nama Cabang`. Three things are unfinished (confirmed by reading the current
uncommitted state of `utils/outlet_features.py`,
`utils/prepare_forecast_data.py`, and `test/test_outlet_features.py`):

1. `apply_region_features` is not called anywhere in
   `prepare_forecast_data.py::main()`.
2. `lead_time_days` is a flat constant (`DEFAULT_LEAD_TIME_DAYS = 4`)
   regardless of `kawasan` or the transaction's day of week — asserted as
   current behavior by `test_lead_time_days_flat_default_regardless_of_kawasan`
   (`test/test_outlet_features.py:256`). The real business rule is variable:
   e.g. Region 1, transaction on Monday → 3-day window to Thursday;
   transaction on Thursday → 4-day window to the following Monday.
3. There is no target for "cumulative demand until the next delivery" —
   `add_targets` only produces `target_h1`…`target_h7` (fixed daily
   horizons).

Coverage check (done as part of this design, not assumed): all 54 branches
that survive `filter_matched_branches`/`canonicalize_branch_names` are present
in `dataset/outlet_mapping.csv` — no unmatched-branch case exists in the data
today, though the code should still degrade safely (propagate `NaN`) if that
ever changes.

Also discovered while designing this: `CLAUDE.md` documents the pipeline
entry point as `python3 prepare_forecast_data.py`, but the module lives at
`utils/prepare_forecast_data.py` and uses relative imports (`from . import
build_panel`, etc.), so it must be run as `python3 -m
utils.prepare_forecast_data`. This is corrected as part of the docs update
below.

## Components

### `utils/outlet_features.py`

- `INDONESIAN_WEEKDAYS`: mapping of Indonesian day names (`"Senin"` … `"Minggu"`)
  to `0`–`6` (Monday = 0, matching `pandas`' `.dt.weekday`).
- `parse_delivery_days(hari_pengiriman: str) -> set[int]`: splits on `"dan"`/
  `","`, strips whitespace, maps each token through `INDONESIAN_WEEKDAYS`.
  Raises `ValueError` on any unrecognized token — fail loud rather than
  silently producing a wrong/empty delivery-day set, so a future format
  change or typo in `outlet_mapping.csv` surfaces immediately instead of
  quietly corrupting `lead_time_days`.
- `compute_lead_time_days(day_of_week: int, delivery_days: set[int]) -> int`:
  smallest `d` in `1..7` such that `(day_of_week + d) % 7 in delivery_days`.
  Always strictly forward — if the transaction's own day is a delivery day,
  the result is the days to the *next* occurrence, not `0` (matches the
  documented examples: Region 1 Monday → 3, Region 1 Thursday → 4).
- `apply_region_features(df, region_df, branch_col="Nama Cabang", date_col="Tanggal")`:
  **signature change** — the `lead_time_days: int` override parameter is
  removed (it existed only to support the flat-constant behavior being
  removed here). Joins `kawasan`/`hari_pengiriman` as before, then computes
  `lead_time_days` per row from `(day_of_week, hari_pengiriman)`. Because
  there are at most a handful of distinct `(day_of_week, hari_pengiriman)`
  combinations in the whole dataset (≤ 7 weekdays × distinct delivery
  patterns), the per-combination result is computed once and merged back
  onto the full frame — not `.apply()`'d row-by-row over 1.3M+ rows.
  Unmatched branches keep `NaN` for `kawasan`/`hari_pengiriman`/
  `lead_time_days` (unchanged from current behavior).

### `utils/prepare_forecast_data.py`

- `add_lead_time_target(df, pair_cols=PAIR_COLS, date_col="Tanggal", qty_col="Kuantitas", lead_time_col="lead_time_days") -> df`:
  adds `target_lead_time_cumulative` = sum of raw `Kuantitas` over the
  strictly-forward window `(H+1 .. H+lead_time_days)` within the same
  (item, branch) pair. Uses raw `Kuantitas` (not `Kuantitas_capped`), matching
  `add_targets`'s existing rationale — spikes are real demand the model
  should be evaluated against, not hidden from the label.

  Implementation: for each distinct value `w` present in `lead_time_days`,
  compute a forward-sum column via reverse → `rolling(w, min_periods=w).sum()`
  → reverse-back (per pair group), then select the correct column per row
  with `np.select` keyed on that row's `lead_time_days`. Avoids a per-row
  Python loop over 1.3M+ rows while still supporting a variable window size.
  Rows whose `lead_time_days` is `NaN` (unmatched branch) or whose window
  runs past the last available date get `NaN` in the target — same pattern
  `target_h1`…`target_h7` already use for the December-2025 boundary.

- `build_featured_dataset(input_path=..., outlets_path=..., overrides_path=..., region_path=..., cutoff=TEST_START, min_history_days=..., min_pair_history=..., spike_ratio_threshold=...) -> pd.DataFrame`:
  new function, extracted from the current body of `main()` — every step from
  `load_and_normalize` through `apply_outlet_features`/`add_branch_age_days`
  (i.e. everything except `split_train_test`/`export_splits`). Returns the
  full cleaned + feature-engineered (pre-split) dataframe. This is the single
  source of truth for "cleansing" used by both `main()` and the new
  `train_test_split.ipynb` notebook (via the `featured.parquet` file it
  exports — see below).

- `export_featured(df, output_dir=MODEL_READY_DIR) -> None`: writes
  `dataset/model_ready/featured.parquet`, the pre-split intermediate
  artifact.

- `main()` becomes: `df = build_featured_dataset(...)` →
  `export_featured(df, output_dir)` → `train, test = split_train_test(df,
  cutoff)` → `export_splits(train, test, output_dir)`. Still one CLI command
  (`python3 -m utils.prepare_forecast_data`) that runs the full pipeline
  end-to-end, matching the documented "scripted, no QA assertions" usage.

### Pipeline order inside `build_featured_dataset`

```
... build_dense_panel → filter_min_history → calendar_features
  → outlier_handling (pair_baseline, capping)
  → add_targets                      (target_h1..h7, raw)
  → apply_region_features            (kawasan, hari_pengiriman, lead_time_days — now variable)
  → apply_outlet_features            (kota, has_shopee, has_gofood, has_grabfood, can_order_online)
  → add_lead_time_target             (target_lead_time_cumulative, raw)
  → add_lag_features / add_rolling_features   (Kuantitas_capped)
  → compute_branch_stats / apply_branch_stats / add_branch_age_days
```

`apply_region_features` and `apply_outlet_features` are adjacent (both are
static per-branch metadata joins); `add_lead_time_target` immediately follows
since it consumes `lead_time_days` from the step just before it. Relative to
`add_lag_features`/`add_rolling_features`/branch-stats there is no
dependency either way (those use `Kuantitas_capped`, unrelated to this
block), so this grouping is purely for readability.

### Notebooks

- `notebook/data-processing.ipynb`: unchanged in structure (cell-by-cell,
  one stage per cell/section, QA checks throughout). Gets new cells for the
  region/lead-time step, added in the position shown above. Continues to stop
  at the fully-featured (pre-split) dataframe, as it already does today —
  it currently never calls `split_train_test`/`export_splits`.
- `notebook/train_test_split.ipynb` (**new**): reads
  `dataset/model_ready/featured.parquet`, calls `split_train_test` +
  `export_splits`, plus QA specific to the split itself: train/test row
  counts, no dates crossing the cutoff into the wrong split, `target_h*`/
  `target_lead_time_cumulative` `NaN` rates near the December-2025 boundary.

## Testing

- `test/test_outlet_features.py`:
  - `parse_delivery_days`: valid inputs (`"Senin dan Kamis"`, `"Selasa dan
    Jumat"`), unrecognized token raises `ValueError`.
  - `compute_lead_time_days`: full 7-day × 2-`kawasan` matrix from the
    worked examples in `docs/todolist-data-preprocessing.md`.
  - `TestApplyRegionFeatures`: **replace**
    `test_lead_time_days_flat_default_regardless_of_kawasan` and
    `test_lead_time_days_override_via_parameter` (both assert the flat-default
    behavior being removed) with tests asserting `lead_time_days` varies
    correctly by `kawasan` + transaction day-of-week. Keep the existing
    join/no-fan-out/unmatched-branch-gets-`NaN` tests as-is.
- `test/test_prepare_forecast_data.py`:
  - `add_lead_time_target`: correct cumulative sum for a 3-day and a 4-day
    window, leakage-safety (window never includes the current day or any
    prior day), `NaN` at the end-of-data boundary, `NaN` propagation when
    `lead_time_days` is `NaN`.
- `notebook/train_test_split.ipynb`: verified via `jupyter nbconvert
  --to notebook --execute --allow-errors`, like the other notebooks — not
  unit-tested.

## Documentation updates (in scope for this work)

- `docs/pipeline-overview.md`: document the region/lead-time step, the new
  `target_lead_time_cumulative` target, the `featured.parquet` intermediate
  artifact, and the new `train_test_split.ipynb` notebook as a distinct
  pipeline stage.
- `docs/superpowers/specs/2026-07-21-forecast-data-prep-design.md`: update
  the sections that currently describe region mapping as absent.
- `docs/todolist-data-preprocessing.md`: check off the four items under
  the 🔴 heading (wiring, variable `lead_time_days`, cumulative target, doc updates).
- `CLAUDE.md`: add the new notebook and `featured.parquet` to the pipeline
  description; correct the run command to `python3 -m
  utils.prepare_forecast_data`.

## Out of scope

- Confirming `kawasan`/`hari_pengiriman` provenance with the data owner
  (tracked separately in `docs/todolist-data-preprocessing.md` under the 🟠 heading) — this
  design proceeds on the current `dataset/outlet_mapping.csv` as-is, per
  explicit user direction to not block on that confirmation.
- Re-running `eda.ipynb` section 5/7 to segment day-of-week/lead-time analysis by
  region — noted in the todolist as a nice-to-have follow-up, not required
  for the pipeline to produce correct model-ready data.
- Moving the 7 QA assertions from notebook-only into the script (tracked
  separately in `docs/todolist-data-preprocessing.md`, pre-existing gap
  unrelated to region/lead-time).
