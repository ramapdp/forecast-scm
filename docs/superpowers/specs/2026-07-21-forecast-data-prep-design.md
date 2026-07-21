# Sales forecast model-comparison — data preparation design

## Purpose

Prepare `dataset/dataset.csv` (693,565 goods-issued rows, Jan 2024–Dec 2025) into model-ready train/test artifacts to support a later comparison of Random Forest, XGBoost, and LSTM for short-term (1–7 day) demand forecasting, at item × branch granularity. This spec covers **data preparation only** — model configuration, training, and comparison methodology are a separate follow-up design.

## Scope decisions

- **Category scope**: all `Kategori Barang` categories are in scope (not limited to finished goods) — this is a goods-issued forecast, not a customer-sales-only forecast.
- **Granularity**: one time series per (`Kode Barang`, `Nama Cabang`) pair, daily.
- **Horizon**: 1–7 days ahead.
- **Modeling strategy** (informs what data prep must produce, even though training itself is out of scope): one global model per algorithm, trained across all series at once with item/branch/category as features — not one model per series. Chosen because ~3,765 actual pairs (of 6,633 possible item×branch combinations) exist, many with limited history; a global model lets series share statistical strength.
- **Validation approach**: single time-based holdout. Test = predictions made for each day in December 2025 (the last full month in the data); train = everything before 2025-12-01. Chosen over rolling-window walk-forward validation to keep the first comparison pass fast to iterate across three model families; walk-forward is a natural follow-up if holdout results come out close/ambiguous between models.
- **Evaluation metrics** (computed downstream, in the modeling stage, but drives what targets data prep must emit): both WMAPE (volume-weighted, intuitive) and MASE (scale-free, standard in forecasting research) — needed because items span very different units (Kg, Porsi, PCS, Botol, etc.) and volumes.

## One shared table, not per-model pipelines

RF, XGBoost, and LSTM do not get separate data-preparation pipelines. All three are built from the same `train.parquet`/`test.parquet` produced by this pipeline, with only a thin, model-specific adapter applied at the modeling stage:

- **RF / XGBoost** consume the table close to as-is — flat rows, one per (item, branch, date) — with just categorical columns (`Kode Barang`, `Nama Cabang`, `Kategori Barang`) label/target-encoded.
- **LSTM** needs a structurally different input: the flat rows are windowed into 3D sequences (samples × timesteps × features), numeric features are scaled, and the categorical columns are integer-encoded for embedding lookup instead of label/target-encoded.

Keeping one shared table with per-model adapters (rather than three separate feature-engineering passes) matters for the comparison's validity: if lags/rolling stats/calendar features were computed three separate times, the three models could silently see slightly different data, undermining the comparison.

## Pipeline stages

Implemented as sequential cells in `data-processing.ipynb` (not separate scripts) — chosen to keep this exploratory/QA-heavy stage in one notebook rather than splitting into composable modules like `merge_dataset.py`/`aggregate_dataset.py`.

### 1. Load & normalize item codes

Read `dataset/dataset.csv`, parse `Tanggal`.

Normalize `Kode Barang` two ways, each independently validated before merging:

- **Strip `xxx.` prefix** (e.g. `xxx.FGS-00003` → `FGS-00003`).
- **Unify separator** (`.` → `-`, e.g. `FGS.00047` → `FGS-00047`).

**Conditional merge rule**: a normalized code is only merged into one series if every row sharing that normalized code also shares the same `Nama Barang` after light name-normalization (strip `xxx.` prefix, collapse whitespace, strip trailing parenthetical annotations like `(Menu pakai kode ...)`). If names disagree after that normalization, the normalization is **not** applied for that specific collision — each original raw `Kode Barang` is kept as its own distinct series.

This rule was derived from checking the actual data: the dot/dash separator swap alone collides 5 pairs of completely unrelated products that happen to share digits (e.g. `FGS-00047` = `Kentang Mustofa Rumput Laut`, Pack; `FGS.00047` = `Air Isi Ulang`, Galon — different products, must stay separate). The `xxx.`-prefix strip alone produces 4 name-mismatched groups, 3 of which are cosmetic (`Gula Asam 250ml` vs `250 ml`, `Kunyit Asam` likewise, `Club Mineral 330 ml` vs the same with a POS-menu-code annotation) and merge cleanly under light name-normalization; the 4th (`Cendol Pandan - FG` vs `Cendol - FG`) stays unmerged since a flavor difference can't be ruled out from the data alone.

After normalization, re-aggregate `Kuantitas` by (normalized `Kode Barang`, `Tanggal`, `Nama Cabang`) — normalization can cause previously-distinct rows to become the same key.

`Satuan` is not carried into the feature set (constant per item, non-informative as a model input) but is kept during QA to confirm that constancy.

### 2. Build dense daily panel

For each (normalized `Kode Barang`, `Nama Cabang`) pair, reindex to one row per calendar day spanning that pair's own first→last observed transaction date (not the full 2024–2025 dataset range), filling `Kuantitas = 0` for days with no recorded transaction. `Kategori Barang` / `Nama Barang` are forward-filled as constant per item.

Using each pair's own active range (rather than the full dataset range) avoids inventing years of fake "zero demand" history for branches that opened partway through the period or items introduced later.

### 3. Minimum-history filter

Drop pairs with fewer than 60 days of history before 2025-12-01 (the test window start) — insufficient to populate the 28-day lag/rolling features without excessive `NaN`s. Dropped pairs are out of scope for this comparison entirely, not imputed or cold-start-handled.

Pairs whose activity stopped before December 2025 (discontinued items, closed branches) naturally have no rows in the test window once dense-filled to their own last observed date — this is correct behavior, not an error, since forecasting a discontinued item/closed branch for December isn't meaningful.

### 4. Feature engineering

Computed per pair, in chronological order:

- **Targets**: `target_h1` … `target_h7` — `Kuantitas` shifted forward 1–7 days within the pair's series. All 7 horizons are emitted as separate columns so the modeling stage can choose direct multi-output or recursive 1-step forecasting later; that choice doesn't affect data prep.
- **Lag features**: `lag_1, lag_2, lag_3, lag_7, lag_14, lag_21, lag_28` (past `Kuantitas` values within the pair's series).
- **Rolling stats**: 7/14/28-day rolling mean and std of `Kuantitas`, computed on history strictly before the current day (shifted) to avoid leakage.
- **Calendar features**: day-of-week, day-of-month, month, is-weekend, Indonesian national holiday flag, Ramadan flag + days-into/until Ramadan, Eid al-Fitr flag + proximity, Eid al-Adha flag + proximity. Source: the `holidays` Python package's Indonesia calendar. Islamic holiday dates from this package are spot-checked against known 2024/2025 dates before being trusted, given their business relevance to a Middle Eastern (Yaman/Kebuli) food business.
- **Identifiers kept as plain categorical columns, not encoded**: normalized `Kode Barang`, `Nama Cabang`, `Kategori Barang`. Actual encoding (one-hot for RF/XGBoost, embedding index for LSTM) is a modeling-stage concern, since each algorithm wants a different encoding.
- **Outlet (branch) characteristic features** — every branch has its own scale, maturity, and demand stability, which plain identity (`Nama Cabang` as a category) doesn't capture on its own. No external outlet master data (city/region, size, opening date, demographics) is available yet, so these are derived purely from each branch's own transaction history:
  - `branch_avg_daily_qty` — average total quantity (summed across all items) per day for that branch.
  - `branch_volume_tier` — a bucketed version of the above (e.g. quartiles), giving tree models a clean split point.
  - `branch_age_days` — days between the branch's own first observed transaction and the current row's date. Dynamic (grows per row), unlike the other branch features below which are frozen constants.
  - `branch_demand_cv` — coefficient of variation of the branch's daily total quantity, as a volatility measure.

  **Leakage rule**: `branch_avg_daily_qty`, `branch_volume_tier`, and `branch_demand_cv` are computed using training-period rows only (dates before 2025-12-01), then frozen and joined onto that branch's rows in both train and test — so a branch's December test-period demand never influences its own branch-level feature values. `branch_age_days` is inherently leakage-safe since it only reads each branch's own past (its first transaction date, which for any surviving pair is well before the test window per the minimum-history filter).

### 5. Train/test split

- **Train**: all rows with date < 2025-12-01.
- **Test**: rows with date in December 2025. A given row's `target_h{n}` is `NaN` if `date + n days > 2025-12-31` (data doesn't extend that far) — those specific horizon/row combinations are excluded from that horizon's metric rather than shrinking the test window to force full coverage on `h7`.

### 6. Export

Write `dataset/model_ready/train.parquet` and `dataset/model_ready/test.parquet`. Parquet chosen over CSV: the dense panel is ~1.5M rows before the minimum-history filter, and Parquet preserves dtypes and compresses substantially better than CSV at that size.

### 7. QA checks (in-notebook, before export)

- Row-count sanity check against expected pair-day counts.
- No duplicate (pair, date) rows.
- No negative quantities.
- Ramadan/Eid flags land on the correct known 2024/2025 dates (spot-check).
- Leakage check: for a sampled row, confirm lag/rolling features only reference strictly earlier dates than the row's own date.

## Out of scope

- Model configuration, training, hyperparameter tuning, and cross-model comparison methodology (separate follow-up design).
- Categorical encoding strategy (one-hot / label / embedding) — deferred to the modeling stage.
- LSTM-specific sequence windowing and feature scaling — deferred to the modeling stage; see "One shared table, not per-model pipelines" above.
- Rolling-window walk-forward validation — noted as a possible follow-up if single-holdout results are ambiguous.
- Cold-start / fallback handling for series dropped by the minimum-history filter.
