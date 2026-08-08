# Outlier / Demand-Spike Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pipeline stage that detects and caps extreme per-row demand spikes (relative to each item-branch pair's own historical median) before they distort lag/rolling/branch-level input features, while leaving forecast targets uncapped and exempting known high-season events (Ramadan, Eid al-Fitr, Eid al-Adha, Independence Day, New Year) from capping.

**Architecture:** A new pure-function module, `utils/outlier_handling.py`, follows the same compute-on-train/freeze/apply-to-both-splits pattern already used by `compute_branch_stats`/`apply_branch_stats` in `utils/prepare_forecast_data.py`. It's wired into `prepare_forecast_data.main()` between panel-building and feature engineering, requiring `calendar_features.add_calendar_features` to move earlier in the pipeline so its event-flag columns exist before the capping decision is made.

**Tech Stack:** Python 3.9.6, pandas, unittest (existing project stack — see `CLAUDE.md`). No new dependencies.

## Global Constraints

- `MIN_PAIR_HISTORY = 30` — minimum real-transaction-day count (train-period, pre-cutoff) per `(Kode Barang, Nama Cabang)` pair before its median is trusted as a baseline.
- `SPIKE_RATIO_THRESHOLD = 5.0` — a row's `Kuantitas` is a "spike" when `Kuantitas / pair_median >= 5.0`.
- `EVENT_FLAG_COLS = ["is_ramadan", "is_eid_al_fitr", "is_eid_al_adha", "is_independence_day", "is_new_year"]` — rows where any of these is `True` are exempt from capping, even if flagged as a spike.
- Baseline statistics (`pair_median`, eligibility) must be computed strictly from `Tanggal < cutoff` (train-only) rows — never from test-period rows — matching the leakage guard already used by `compute_branch_stats`.
- `target_h1`…`target_h7` must always be built from raw, uncapped `Kuantitas` — never from `Kuantitas_capped`.
- Full spec: `docs/superpowers/specs/2026-08-08-outlier-handling-design.md`.

---

### Task 1: `compute_pair_baseline`

**Files:**
- Create: `utils/outlier_handling.py`
- Test: `test/test_outlier_handling.py`

**Interfaces:**
- Consumes: `build_panel.PAIR_COLS` (`["Kode Barang", "Nama Cabang"]`), `build_panel.TEST_START` (`pd.Timestamp("2025-12-01")`) — both already defined in `utils/build_panel.py`.
- Produces: `compute_pair_baseline(df, cutoff=TEST_START, pair_cols=PAIR_COLS, date_col="Tanggal", qty_col="Kuantitas", min_history=MIN_PAIR_HISTORY) -> pd.DataFrame` returning one row per pair with columns `pair_cols + ["pair_median", "pair_eligible"]`. Consumed by Task 2's `apply_outlier_capping`.

- [ ] **Step 1: Write the failing test**

Create `test/test_outlier_handling.py`:

```python
import unittest

import pandas as pd

from utils import outlier_handling


def _pair_rows(pair, qtys, start="2025-01-01"):
    n = len(qtys)
    return pd.DataFrame({
        "Kode Barang": [pair[0]] * n, "Nama Cabang": [pair[1]] * n,
        "Tanggal": pd.date_range(start, periods=n, freq="D"),
        "Kuantitas": qtys,
    })


class TestComputePairBaseline(unittest.TestCase):
    def test_eligible_pair_gets_correct_median(self):
        # 35 real-transaction days, mostly 10 with one high value — median
        # stays robust to the single outlier.
        qtys = [10] * 34 + [500]
        df = _pair_rows(("A", "X"), qtys, start="2025-01-01")
        result = outlier_handling.compute_pair_baseline(df, cutoff=pd.Timestamp("2025-12-01"))
        row = result[(result["Kode Barang"] == "A") & (result["Nama Cabang"] == "X")].iloc[0]
        self.assertEqual(row["pair_median"], 10.0)
        self.assertTrue(row["pair_eligible"])

    def test_pair_below_min_history_is_ineligible(self):
        qtys = [10] * 29  # 29 < MIN_PAIR_HISTORY (30)
        df = _pair_rows(("A", "X"), qtys, start="2025-01-01")
        result = outlier_handling.compute_pair_baseline(df, cutoff=pd.Timestamp("2025-12-01"))
        row = result[(result["Kode Barang"] == "A") & (result["Nama Cabang"] == "X")].iloc[0]
        self.assertFalse(row["pair_eligible"])

    def test_zero_fill_gap_days_do_not_count_toward_history(self):
        # 20 real transactions + 15 zero-quantity gap-fill rows = 35 panel
        # rows, but only 20 are real — below the 30-day minimum.
        real = _pair_rows(("A", "X"), [10] * 20, start="2025-01-01")
        gaps = _pair_rows(("A", "X"), [0] * 15, start="2025-01-21")
        df = pd.concat([real, gaps], ignore_index=True)
        result = outlier_handling.compute_pair_baseline(df, cutoff=pd.Timestamp("2025-12-01"))
        row = result[(result["Kode Barang"] == "A") & (result["Nama Cabang"] == "X")].iloc[0]
        self.assertFalse(row["pair_eligible"])

    def test_test_period_rows_excluded_from_baseline(self):
        train = _pair_rows(("A", "X"), [10] * 30, start="2025-10-01")
        test_period = _pair_rows(("A", "X"), [99999] * 5, start="2025-12-01")
        df = pd.concat([train, test_period], ignore_index=True)
        result = outlier_handling.compute_pair_baseline(df, cutoff=pd.Timestamp("2025-12-01"))
        row = result[(result["Kode Barang"] == "A") & (result["Nama Cabang"] == "X")].iloc[0]
        self.assertEqual(row["pair_median"], 10.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m unittest test.test_outlier_handling -v`
Expected: `ModuleNotFoundError: No module named 'utils.outlier_handling'` (or `ImportError`).

- [ ] **Step 3: Write minimal implementation**

Create `utils/outlier_handling.py`:

```python
import pandas as pd

from . import build_panel

PAIR_COLS = build_panel.PAIR_COLS
TEST_START = build_panel.TEST_START
MIN_PAIR_HISTORY = 30
SPIKE_RATIO_THRESHOLD = 5.0
EVENT_FLAG_COLS = [
    "is_ramadan", "is_eid_al_fitr", "is_eid_al_adha", "is_independence_day", "is_new_year",
]


def compute_pair_baseline(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_START,
    pair_cols: list[str] = PAIR_COLS,
    date_col: str = "Tanggal",
    qty_col: str = "Kuantitas",
    min_history: int = MIN_PAIR_HISTORY,
) -> pd.DataFrame:
    # Leakage guard: filter to strictly-pre-cutoff (training-period) rows
    # BEFORE computing any per-pair aggregate — mirrors compute_branch_stats.
    # Kuantitas == 0 rows are always build_panel gap-fill days (raw
    # Kuantitas is never 0 in the source data), never real transactions, so
    # excluding them here recovers the same median/count as computing on
    # the pre-panel transactional data directly.
    train = df[(df[date_col] < cutoff) & (df[qty_col] > 0)]
    stats = (
        train.groupby(pair_cols)[qty_col]
        .agg(pair_count="count", pair_median="median")
        .reset_index()
    )
    stats["pair_eligible"] = (stats["pair_count"] >= min_history) & (stats["pair_median"] > 0)
    return stats.drop(columns=["pair_count"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m unittest test.test_outlier_handling -v`
Expected: All 4 tests in `TestComputePairBaseline` PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/outlier_handling.py test/test_outlier_handling.py
git commit -m "feat: add compute_pair_baseline for outlier-handling stage"
```

---

### Task 2: `apply_outlier_capping`

**Files:**
- Modify: `utils/outlier_handling.py`
- Modify: `test/test_outlier_handling.py`

**Interfaces:**
- Consumes: `compute_pair_baseline`'s output (Task 1) — a DataFrame with `pair_cols + ["pair_median", "pair_eligible"]`. Also consumes `EVENT_FLAG_COLS`-named boolean columns that must already exist on the input `df` (produced in the real pipeline by `calendar_features.add_calendar_features`, see Task 3).
- Produces: `apply_outlier_capping(df, baseline_df, ratio_threshold=SPIKE_RATIO_THRESHOLD, pair_cols=PAIR_COLS, qty_col="Kuantitas", event_cols=EVENT_FLAG_COLS) -> pd.DataFrame` — returns `df` with three new columns added: `Kuantitas_capped`, `baseline_ratio`, `is_spike`. Consumed by Task 3's `prepare_forecast_data.main()`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_outlier_handling.py` (before the `if __name__ == "__main__":` line):

```python
def _with_event_flags(df, **flags):
    result = df.copy()
    for col in outlier_handling.EVENT_FLAG_COLS:
        result[col] = flags.get(col, False)
    return result


class TestApplyOutlierCapping(unittest.TestCase):
    def _baseline(self, pair=("A", "X"), median=10.0, eligible=True):
        return pd.DataFrame({
            "Kode Barang": [pair[0]], "Nama Cabang": [pair[1]],
            "pair_median": [median], "pair_eligible": [eligible],
        })

    def test_caps_spike_above_threshold_outside_event_window(self):
        df = _with_event_flags(_pair_rows(("A", "X"), [1000]))
        result = outlier_handling.apply_outlier_capping(df, self._baseline())
        self.assertEqual(result["Kuantitas_capped"].iloc[0], 50.0)  # 10 * 5.0
        self.assertTrue(result["is_spike"].iloc[0])
        self.assertEqual(result["baseline_ratio"].iloc[0], 100.0)

    def test_does_not_cap_value_below_threshold(self):
        df = _with_event_flags(_pair_rows(("A", "X"), [40]))  # ratio 4.0 < 5.0
        result = outlier_handling.apply_outlier_capping(df, self._baseline())
        self.assertEqual(result["Kuantitas_capped"].iloc[0], 40.0)
        self.assertFalse(result["is_spike"].iloc[0])

    def test_exempts_spike_inside_event_window(self):
        df = _with_event_flags(_pair_rows(("A", "X"), [1000]), is_ramadan=True)
        result = outlier_handling.apply_outlier_capping(df, self._baseline())
        self.assertEqual(result["Kuantitas_capped"].iloc[0], 1000.0)  # not capped
        self.assertTrue(result["is_spike"].iloc[0])  # still flagged as detected

    def test_ineligible_pair_is_never_capped(self):
        df = _with_event_flags(_pair_rows(("A", "X"), [1000]))
        baseline = self._baseline(median=10.0, eligible=False)
        result = outlier_handling.apply_outlier_capping(df, baseline)
        self.assertEqual(result["Kuantitas_capped"].iloc[0], 1000.0)
        self.assertFalse(result["is_spike"].iloc[0])
        self.assertTrue(pd.isna(result["baseline_ratio"].iloc[0]))

    def test_gap_fill_zero_quantity_row_is_not_a_spike(self):
        df = _with_event_flags(_pair_rows(("A", "X"), [0]))
        result = outlier_handling.apply_outlier_capping(df, self._baseline())
        self.assertEqual(result["Kuantitas_capped"].iloc[0], 0.0)
        self.assertFalse(result["is_spike"].iloc[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m unittest test.test_outlier_handling -v`
Expected: `AttributeError: module 'utils.outlier_handling' has no attribute 'apply_outlier_capping'`.

- [ ] **Step 3: Write minimal implementation**

Append to `utils/outlier_handling.py`:

```python
def apply_outlier_capping(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    ratio_threshold: float = SPIKE_RATIO_THRESHOLD,
    pair_cols: list[str] = PAIR_COLS,
    qty_col: str = "Kuantitas",
    event_cols: list[str] = EVENT_FLAG_COLS,
) -> pd.DataFrame:
    result = df.merge(baseline_df, on=pair_cols, how="left")
    result["pair_eligible"] = result["pair_eligible"].fillna(False)
    result["baseline_ratio"] = result[qty_col] / result["pair_median"]
    result.loc[~result["pair_eligible"], "baseline_ratio"] = float("nan")
    result["is_spike"] = result["pair_eligible"] & (result["baseline_ratio"] >= ratio_threshold)

    in_event_window = result[event_cols].any(axis=1)
    should_cap = result["is_spike"] & ~in_event_window
    cap_value = result["pair_median"] * ratio_threshold
    result["Kuantitas_capped"] = result[qty_col].where(~should_cap, cap_value)

    return result.drop(columns=["pair_median", "pair_eligible"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m unittest test.test_outlier_handling -v`
Expected: All 9 tests (4 from Task 1 + 5 from this task) PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/outlier_handling.py test/test_outlier_handling.py
git commit -m "feat: add apply_outlier_capping with event-window exemption"
```

---

### Task 3: Wire outlier handling into `prepare_forecast_data.main()`

**Files:**
- Modify: `utils/prepare_forecast_data.py:1-18` (imports/constants), `utils/prepare_forecast_data.py:153-183` (`main`)
- Modify: `test/test_prepare_forecast_data.py`

**Interfaces:**
- Consumes: `outlier_handling.compute_pair_baseline`, `outlier_handling.apply_outlier_capping`, `outlier_handling.MIN_PAIR_HISTORY`, `outlier_handling.SPIKE_RATIO_THRESHOLD` (Tasks 1–2). Consumes existing `add_lag_features(df, qty_col=...)`, `add_rolling_features(df, qty_col=...)`, `compute_branch_stats(df, cutoff=..., qty_col=...)` — all already accept a `qty_col` parameter, no signature changes needed.
- Produces: `main()`'s exported `train.parquet`/`test.parquet` gain three columns: `Kuantitas_capped`, `baseline_ratio`, `is_spike`.

- [ ] **Step 1: Write the failing test**

Add to `test/test_prepare_forecast_data.py`, after the existing `_branch_rows` helper (around line 279) and before `class TestMain`:

```python
def _branch_rows_with_quantities(branch, start, quantities):
    lines = []
    for date, qty in zip(pd.date_range(start, periods=len(quantities), freq="D"), quantities):
        lines.append(
            f"{date.strftime('%d %b %Y')};Barang Jadi (FG);FGS-00001;Widget;"
            f"{branch};Porsi;{qty}\n"
        )
    return lines
```

Add a new test class after `class TestMain`'s existing tests (still inside `class TestMain`, as an additional method — insert before the final `if __name__ == "__main__":` line):

```python
    def test_main_caps_lag_input_but_leaves_target_and_flags_spike(self):
        rows = ["Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"]
        # 90 days, 2025-08-01..2025-10-29, steady Kuantitas=10, except a
        # single spike of 1000 on 2025-08-31 (not a calendar event date).
        # Cutoff 2025-10-01 gives 61 pre-cutoff real-transaction days for
        # "KY001 - Branch", comfortably above MIN_PAIR_HISTORY (30) and
        # min_history_days (60).
        quantities = [10] * 90
        quantities[30] = 1000  # 2025-08-01 + 30 days = 2025-08-31
        rows += _branch_rows_with_quantities("KY001 - Branch", "2025-08-01", quantities)
        content = "".join(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "dataset.csv"
            input_path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            output_dir = Path(tmpdir) / "model_ready"
            outlets_path = _write_outlets_fixture(tmpdir, ["KY001 - Branch"])
            overrides_path = _write_empty_overrides_fixture(tmpdir)
            prepare_forecast_data.main(
                input_path=input_path,
                output_dir=output_dir,
                min_history_days=60,
                cutoff=pd.Timestamp("2025-10-01"),
                outlets_path=outlets_path,
                overrides_path=overrides_path,
            )
            train = pd.read_parquet(output_dir / "train.parquet")

        self.assertIn("Kuantitas_capped", train.columns)
        self.assertIn("baseline_ratio", train.columns)
        self.assertIn("is_spike", train.columns)

        spike_day = train[train["Tanggal"] == pd.Timestamp("2025-08-31")].iloc[0]
        self.assertTrue(spike_day["is_spike"])
        self.assertEqual(spike_day["Kuantitas_capped"], 50.0)  # 10 * SPIKE_RATIO_THRESHOLD

        day_before_spike = train[train["Tanggal"] == pd.Timestamp("2025-08-30")].iloc[0]
        self.assertEqual(day_before_spike["target_h1"], 1000)  # target uses RAW Kuantitas

        day_after_spike = train[train["Tanggal"] == pd.Timestamp("2025-09-01")].iloc[0]
        self.assertEqual(day_after_spike["lag_1"], 50.0)  # lag uses CAPPED Kuantitas

        branch_stats_row = train[train["Tanggal"] == pd.Timestamp("2025-09-15")].iloc[0]
        # Uncapped, the spike would pull the average toward ~26; capped it
        # stays close to the steady 10/day baseline.
        self.assertLess(branch_stats_row["branch_avg_daily_qty"], 15.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m unittest test.test_prepare_forecast_data.TestMain.test_main_caps_lag_input_but_leaves_target_and_flags_spike -v`
Expected: FAIL — `KeyError: 'Kuantitas_capped'` (or `'is_spike'`/`'baseline_ratio'`), since `main()` doesn't produce these columns yet.

- [ ] **Step 3: Wire the new stage into `main()`**

In `utils/prepare_forecast_data.py`, add the import near the top (after `from . import outlet_features`):

```python
from . import outlet_features
from . import outlier_handling
```

Replace the body of `main()` (currently `utils/prepare_forecast_data.py:153-183`) with:

```python
def main(
    input_path: str = normalize_items.RAW_DATA_FILE,
    output_dir: str = MODEL_READY_DIR,
    min_history_days: int = build_panel.MIN_HISTORY_DAYS,
    cutoff: pd.Timestamp = TEST_START,
    outlets_path: str = outlet_features.OUTLETS_FILE,
    overrides_path: str = outlet_features.OVERRIDES_FILE,
    min_pair_history: int = outlier_handling.MIN_PAIR_HISTORY,
    spike_ratio_threshold: float = outlier_handling.SPIKE_RATIO_THRESHOLD,
) -> None:
    outlets_df = outlet_features.load_outlets(outlets_path)
    overrides_df = outlet_features.load_overrides(overrides_path)
    df = normalize_items.load_and_normalize(input_path)
    df = outlet_features.filter_matched_branches(df, outlets_df, overrides_df)
    df = outlet_features.canonicalize_branch_names(df, outlets_df, overrides_df)
    df = normalize_items.reaggregate_daily(df)
    df = build_panel.build_dense_panel(df)
    df = build_panel.filter_min_history(df, cutoff=cutoff, min_days=min_history_days)
    df = calendar_features.add_calendar_features(df)
    pair_baseline = outlier_handling.compute_pair_baseline(
        df, cutoff=cutoff, min_history=min_pair_history
    )
    df = outlier_handling.apply_outlier_capping(
        df, pair_baseline, ratio_threshold=spike_ratio_threshold
    )
    df = add_targets(df)
    df = add_lag_features(df, qty_col="Kuantitas_capped")
    df = add_rolling_features(df, qty_col="Kuantitas_capped")
    branch_stats = compute_branch_stats(df, cutoff=cutoff, qty_col="Kuantitas_capped")
    df = apply_branch_stats(df, branch_stats)
    df = add_branch_age_days(df)
    df = apply_outlet_features(df, outlets_df, overrides_df)
    train, test = split_train_test(df, cutoff=cutoff)
    export_splits(train, test, output_dir)
    print(f"Wrote {len(train)} train rows and {len(test)} test rows to {output_dir}")
```

Note what changed from the current version: `calendar_features.add_calendar_features(df)` moved from after `add_rolling_features` to right after `filter_min_history`; the two new `outlier_handling` calls were inserted immediately after it; `add_lag_features`, `add_rolling_features`, and `compute_branch_stats` now pass `qty_col="Kuantitas_capped"`; `add_targets` is unchanged (still defaults to raw `Kuantitas`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m unittest test.test_prepare_forecast_data -v`
Expected: All tests PASS, including the new `test_main_caps_lag_input_but_leaves_target_and_flags_spike` and every pre-existing `TestMain` test (confirms the reordering didn't break canonicalization/branch-filtering/outlet-join behavior).

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v`
Expected: All tests across every module PASS (confirms no other module imports or depends on the old `main()` call order).

- [ ] **Step 6: Commit**

```bash
git add utils/prepare_forecast_data.py test/test_prepare_forecast_data.py
git commit -m "feat: wire outlier capping into prepare_forecast_data pipeline"
```

---

### Task 4: Update `docs/pipeline-overview.md`

**Files:**
- Modify: `docs/pipeline-overview.md`

**Interfaces:**
- Consumes: Final pipeline shape from Task 3 (stage order, new columns).
- Produces: Updated documentation only — no code.

- [ ] **Step 1: Update the numbered stage list**

In `docs/pipeline-overview.md`, replace step 6 (the "Feature engineering" bullet list, starting `6. **Feature engineering** — \`prepare_forecast_data.py\`:`) with:

```markdown
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
   pattern, not noise. `baseline_ratio` and `is_spike` are kept as features;
   raw `Kuantitas` is preserved unchanged for target computation.
8. **Feature engineering** — `prepare_forecast_data.py`:
   - `add_targets`: forecast targets `target_h1`…`target_h7` (raw,
     **uncapped** `Kuantitas` shifted 1–7 days into the future — spikes are
     real demand the model should be evaluated against, not something to
     hide from the label).
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
   - `outlet_features.apply_outlet_features`: joins static per-branch
     features — `kota`, `has_shopee`, `has_gofood`, `has_grabfood`, and the
     derived `can_order_online`.
```

Renumber the two steps that follow: the old step 7 ("Train/test split") becomes step 9, and the old step 8 ("Export") becomes step 10, changing the column count from 49 to 52 (the three new `Kuantitas_capped`/`baseline_ratio`/`is_spike` columns). The old step 9 ("QA checks") becomes step 11 — text unchanged.

- [ ] **Step 2: Update the flow diagram**

Replace the fenced flow diagram (currently):

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

with:

```
raw .xlsx/.csv
  → merge_dataset.py           (dataset/dataset.csv)
  → aggregate_dataset.py       (dedup, in place)
  → normalize_items.py         (clean codes/branches)
  → outlet_features.filter_matched_branches
  → outlet_features.canonicalize_branch_names
  → normalize_items.reaggregate_daily (re-dedup after renaming)
  → build_panel.py             (dense daily panel, min-history filter)
  → calendar_features.py       (calendar/holiday/high-season features)
  → outlier_handling.py        (per-pair spike detection + capping)
  → prepare_forecast_data.py   (targets [raw], lags/rolling/branch stats [capped])
  → outlet_features.apply_outlet_features
  → split_train_test → export_splits
  → dataset/model_ready/{train,test}.parquet
```

- [ ] **Step 3: Commit**

```bash
git add docs/pipeline-overview.md
git commit -m "docs: document outlier-handling stage in pipeline overview"
```
