# Outlier / demand-spike handling — design

## Purpose

Exploratory analysis in `notebook/eda.ipynb` (cells 24–27) found that ranking rows by raw `Kuantitas` (`nlargest(20, "Kuantitas")`) is biased toward `Satuan` with naturally large per-transaction counts (`Porsi`, `PCS`) and misses genuine demand spikes in items whose normal order size is small (e.g. `Botol`, `Gr` units) — on 2025-02-23, `Ayam Kebuli`/`Rice Bowl 600 ml` spiked at `KY001` alongside `Nasi Kebuli`/`Sambal - FG` but never appeared in the raw top-20 because their absolute counts stayed small. A relative measure, `baseline_ratio = Kuantitas / (that item-branch pair's own historical median)`, was designed there and validated against the actual data. This spec turns that exploratory logic into a tested pipeline stage that caps extreme per-row spikes before they distort downstream `lag_*`/`roll_*` input features and branch-level statistics, while explicitly preserving known recurring seasonal spikes (Ramadan, Eid al-Fitr, Eid al-Adha, Independence Day, New Year) and leaving forecast targets untouched.

## Key finding: `Kuantitas` is never 0 in raw data

Verified directly against `normalize_items.load_and_normalize()` output (692,993 rows): `min(Kuantitas) == 1`, no rows with `Kuantitas <= 0`. This matters because `build_panel.build_dense_panel()` (`utils/build_panel.py:26`) reindexes each `(Kode Barang, Nama Cabang)` pair to a dense daily range and fills gap days with `Kuantitas = 0`. Since raw transactions never record `0`, **every `Kuantitas == 0` row in the dense panel is a gap-fill day, never a real transaction** — so the baseline (median, count) needed for outlier detection can be computed straight from the dense panel by filtering to `Kuantitas > 0`, with no need to insert a stage before `build_dense_panel` or modify that function (which otherwise drops any column not in its explicit `pair_cols + [date_col, qty_col] + carry_cols` selection).

## Module: `utils/outlier_handling.py`

New module, same shape as `calendar_features.py`/`outlet_features.py`: pure functions, dedicated `test/test_outlier_handling.py`, wired into `prepare_forecast_data.py`'s `main()`. Follows the same compute-on-train/freeze/apply-to-both-splits pattern already used by `compute_branch_stats`/`apply_branch_stats`.

Constants: `MIN_PAIR_HISTORY = 30` (minimum real-transaction-day count per pair for a trustworthy median), `SPIKE_RATIO_THRESHOLD = 5.0` — both carried over unchanged from the values already explored and eyeballed against real output in `eda.ipynb` — and `EVENT_FLAG_COLS = ["is_ramadan", "is_eid_al_fitr", "is_eid_al_adha", "is_independence_day", "is_new_year"]`, the `calendar_features.py` column names an event-window exemption check reads.

### `compute_pair_baseline(df, cutoff=TEST_START, pair_cols=PAIR_COLS, date_col="Tanggal", qty_col="Kuantitas", min_history=MIN_PAIR_HISTORY) -> pd.DataFrame`

- Leakage guard, same shape as `compute_branch_stats`: filters to `Tanggal < cutoff` (train-only) **and** `Kuantitas > 0` (real transactions only, excludes gap-fill days) before computing anything.
- Groups by `pair_cols`, computing `count` and `median` of `qty_col`.
- `pair_eligible = (count >= min_history) & (median > 0)`.
- Returns `pair_cols + ["pair_median", "pair_eligible"]` — one row per pair.

### `apply_outlier_capping(df, baseline_df, ratio_threshold=SPIKE_RATIO_THRESHOLD, pair_cols=PAIR_COLS, qty_col="Kuantitas", event_cols=EVENT_FLAG_COLS) -> pd.DataFrame`

- Left-merges `baseline_df` onto `df` (train + test) by `pair_cols`.
- `baseline_ratio = Kuantitas / pair_median` where `pair_eligible`; `NaN` where not eligible (history too short/sparse to trust) or unmatched.
- `is_spike = pair_eligible & (baseline_ratio >= ratio_threshold)`.
- `in_event_window = is_ramadan | is_eid_al_fitr | is_eid_al_adha | is_independence_day | is_new_year` — read directly from columns already added by `calendar_features.add_calendar_features` (see pipeline-ordering change below). Deliberately scoped to only these 4 high-season flags, not the broader `is_national_holiday` — a spike on an arbitrary public holiday unrelated to the 4 known high-demand events is still treated as an anomaly candidate, not exempted.
- `should_cap = is_spike & ~in_event_window`.
- `Kuantitas_capped = where(should_cap, pair_median * ratio_threshold, Kuantitas)`.
- Output columns added to `df`: `Kuantitas_capped`, `baseline_ratio`, `is_spike`. Helper columns (`pair_median`, `pair_eligible`) are dropped before returning.

Both functions are pure/stateless like the rest of the pipeline's feature functions — no notebook-only logic, consistent with every other stage.

## Pipeline-ordering change in `prepare_forecast_data.py`

`calendar_features.add_calendar_features` moves earlier — from after `add_rolling_features` to right after `build_panel.filter_min_history` — because `apply_outlier_capping` needs its event-flag columns to decide which spikes are exempt. This is safe: `add_calendar_features` is a pure function of `Tanggal` with no dependency on lag/rolling/target columns.

```
... build_dense_panel → filter_min_history
  → calendar_features.add_calendar_features                          (moved earlier)
  → outlier_handling.compute_pair_baseline → apply_outlier_capping    (NEW)
  → add_targets(qty_col="Kuantitas")                                  (unchanged — raw, uncapped)
  → add_lag_features(qty_col="Kuantitas_capped")                      (changed from "Kuantitas")
  → add_rolling_features(qty_col="Kuantitas_capped")                  (changed from "Kuantitas")
  → compute_branch_stats(qty_col="Kuantitas_capped")                  (changed from "Kuantitas")
  → apply_branch_stats → add_branch_age_days → apply_outlet_features
  → split_train_test → export_splits
```

`add_targets`, `add_lag_features`, `add_rolling_features`, and `compute_branch_stats` already accept a `qty_col` parameter, so this is purely a change to the arguments `main()` passes them — no changes to those functions' bodies.

### Why targets stay uncapped but inputs get capped

Capping is applied to the columns that describe **past behavior** the model conditions on (`lag_*`, `roll_mean_*`/`roll_std_*`, `branch_avg_daily_qty`/`branch_demand_cv`/`branch_volume_tier`) — an extreme one-day spike in the input history shouldn't dominate a 7/14/28-day rolling window or a branch's overall demand-variability estimate. `target_h1`…`target_h7` (`add_targets`, still built from raw `Kuantitas`) are what the model is evaluated against, so they must reflect real observed demand, spike included — artificially capping the label would make the evaluation lie about how well the model predicts genuine demand surges.

### Why event-window spikes are exempt from capping

`baseline_ratio` is computed against each pair's **all-time** median, which is pulled down by ordinary low-season days. For an item that's highly seasonal (e.g. only sells meaningfully during Ramadan), that low all-time median makes its real, recurring Ramadan spike register as a very high ratio — capping it every year would flatten out exactly the seasonal pattern `calendar_features.py`'s Ramadan/Eid/Independence-Day/New-Year proximity features exist to let the model learn from. Exempting rows inside these known event windows keeps that signal intact in `Kuantitas_capped` (and therefore in `lag_*`/`roll_*`), while spikes that don't coincide with any known event — more likely data-entry errors or one-off bulk orders — are still capped.

## Leakage treatment

`pair_median`/`pair_eligible` are computed strictly from `Tanggal < cutoff` (train) rows, exactly like `compute_branch_stats`, then merged onto both train and test rows unchanged — test-period values never influence the baseline a test row is compared against.

## Testing (`test/test_outlier_handling.py`)

- Pair with fewer than `MIN_PAIR_HISTORY` real-transaction days in train → `pair_eligible = False`, `Kuantitas_capped == Kuantitas` (no capping), `baseline_ratio` is `NaN`.
- Pair with a spike (`baseline_ratio >= SPIKE_RATIO_THRESHOLD`) outside any event window → `Kuantitas_capped == pair_median * SPIKE_RATIO_THRESHOLD`, `is_spike == True`.
- Same-magnitude spike, but the row's `Tanggal` falls inside `is_ramadan`/`is_eid_al_fitr`/`is_eid_al_adha`/`is_independence_day`/`is_new_year` → `Kuantitas_capped == Kuantitas` (uncapped), `is_spike` still `True` (flag reflects detection, not the capping decision).
- `add_targets` output (`target_h1`…`target_h7`) built on a capped row still equals the pre-cap raw `Kuantitas` value shifted — confirms targets never see the capped column.
- `compute_pair_baseline` ignores rows with `Tanggal >= cutoff` and rows with `Kuantitas == 0` (gap-fill) when computing `pair_median`/count.
- Gap-fill day (`Kuantitas == 0`) passes through with `Kuantitas_capped == 0`, `is_spike == False` (0 is always far below any positive cap value, never flagged as a spike).

## Documentation updates

- `docs/pipeline-overview.md`: insert the new stage into the numbered pipeline-stages list and the flow diagram, between panel-building and feature engineering; note the `calendar_features` reordering.
- No changes needed to `CLAUDE.md` (already describes the pipeline at a level this stage doesn't affect).

## Out of scope

- Retroactively re-checking whether any of the 8 highest-ratio rows already surfaced in `eda.ipynb`'s exploratory output represent genuine data-entry errors vs. real bulk/catering orders — this spec caps mechanically by ratio + event-window, it doesn't adjudicate individual historical rows.
- A seasonal-aware baseline (e.g. day-of-week or day-of-month-specific median instead of a flat all-time median) — the event-window exemption is judged sufficient for the 4 known high-season drivers already modeled elsewhere in the pipeline; a more granular seasonal baseline is a possible future refinement if capping is found to still clip legitimate demand outside those 4 windows.
- Using `is_spike`/`baseline_ratio` as model *sample weights* or any other explicit consumption by a training script — this spec only produces the columns; how a future modeling stage uses them is undecided (per `docs/pipeline-overview.md`'s "Expected modelling phase (not yet built)").
- The `not_discontinued` (`xxx.` prefix) filter used in `eda.ipynb`'s exploration — not needed here since `normalize_items.py` already strips `xxx.` prefixes upstream of this stage, so no discontinued-item codes reach it.
