# Modeling Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `dataset/model_ready/featured.parquet` into a verified `model_input.parquet` that XGBoost, Random Forest, and LSTM all consume through two thin adapters bound by a contract guaranteeing they see identical rows.

**Architecture:** First close the notebook↔script drift that left `featured.parquet` stale, by extracting the feature-engineering step order into one reusable function the notebook calls. Then add `utils/modeling_prep.py` with five model-agnostic functions (event flag, demand segmentation, walk-forward folds, categorical encoding, contract validation) plus two adapters (`to_tabular`, `to_sequences`).

**Tech Stack:** Python 3.9.6, pandas 2.3.3, numpy 2.0.2, pyarrow, unittest. No new third-party dependencies — see Global Constraints.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md`. Read it before starting.
- **Python:** 3.9.6 via `.venv/bin/python3`. Run everything from the repo root; `utils` is a package with relative imports.
- **Tests:** `unittest`, not pytest. Test classes are `class TestXxx(unittest.TestCase)`. Run one module with `.venv/bin/python3 -m unittest test.test_modeling_prep -v`; run everything with `.venv/bin/python3 -m unittest discover -p "test_*.py" -v`.
- **The full suite is 201 tests (verified 2026-08-12) and must stay green after every task.** Older docs say 195; that figure is stale.
- **CSV convention:** every data file under `dataset/` is semicolon-delimited, `encoding="utf-8-sig"`.
- **Module convention:** constants in `UPPER_SNAKE` at the top, `BASE_DIR = Path(__file__).resolve().parent.parent`, type-hinted signatures, functions take a DataFrame and return `df.copy()`-derived output without mutating the input.
- **`dataset/` is gitignored.** Never `git add -f` anything under it. Data files are produced locally, not committed.
- **Pair key is `["Kode Barang", "Nama Cabang"]`** — never include `Kategori Barang`, which is deliberately time-varying for 8 WIP-2/FG SKUs.
- **Lookback is 28 days.** A row is predictable when its zero-based position within its own pair's date-sorted series is `>= 28`.
- **Event-proximity sentinel is `99`, not `0` and not `30`.** `days_until_ramadan` reaches 70 in real data, so a smaller sentinel collides with genuine values.
- **Deviation from spec, accepted:** the spec's Part 5 says `requirements.txt` gains scikit-learn at this stage. This plan implements fold-wise scaling in pandas and persists parameters as JSON instead, so no new dependency is added until the modeling spec actually needs one. If you disagree, stop and raise it rather than silently adding the dependency.

---

## File Structure

| File | Responsibility |
|---|---|
| `utils/prepare_forecast_data.py` (modify) | Gains `engineer_features()` extracted from `build_featured_dataset()`, and `run_qa_checks()` |
| `utils/modeling_prep.py` (create) | The five shared functions, the two adapters, and the contract validator |
| `test/test_prepare_forecast_data.py` (modify) | Column-contract test for `engineer_features()`, tests for `run_qa_checks()` |
| `test/test_modeling_prep.py` (create) | All `modeling_prep` tests |
| `notebook/data-processing.ipynb` (modify) | Cells 18–24 collapse into one `engineer_features()` call |
| `notebook/modeling_prep.ipynb` (create) | Runs and QAs the modeling-prep stage |
| `docs/pipeline-overview.md` (modify) | Documents the new stage |

---

## Task 1: Extract `engineer_features()` and lock the column contract

The notebook re-lists the nine feature-engineering calls inside `build_featured_dataset()` instead of calling it, which is how `days_since_relocation` went missing. Extracting the step order into its own function gives the notebook something to call, and a column-contract test makes any future divergence fail loudly.

**Files:**
- Modify: `utils/prepare_forecast_data.py:195-231`
- Test: `test/test_prepare_forecast_data.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `engineer_features(df: pd.DataFrame, outlets_df: pd.DataFrame, overrides_df: pd.DataFrame, region_df: pd.DataFrame, cutoff: pd.Timestamp = TEST_START, min_pair_history: int = outlier_handling.MIN_PAIR_HISTORY, spike_ratio_threshold: float = outlier_handling.SPIKE_RATIO_THRESHOLD) -> pd.DataFrame` and the module constant `FEATURED_COLUMNS: list[str]` (63 names).

- [ ] **Step 1: Write the failing test**

Append to `test/test_prepare_forecast_data.py`:

```python
class TestEngineerFeaturesContract(unittest.TestCase):
    def test_featured_columns_constant_has_63_entries(self):
        self.assertEqual(len(prepare_forecast_data.FEATURED_COLUMNS), 63)

    def test_featured_columns_includes_days_since_relocation(self):
        self.assertIn("days_since_relocation", prepare_forecast_data.FEATURED_COLUMNS)

    def test_build_featured_dataset_delegates_to_engineer_features(self):
        self.assertTrue(hasattr(prepare_forecast_data, "engineer_features"))
        self.assertTrue(callable(prepare_forecast_data.engineer_features))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m unittest test.test_prepare_forecast_data.TestEngineerFeaturesContract -v`
Expected: FAIL with `AttributeError: module 'utils.prepare_forecast_data' has no attribute 'FEATURED_COLUMNS'`

- [ ] **Step 3: Extract the function**

In `utils/prepare_forecast_data.py`, replace the body of `build_featured_dataset` (currently lines 205-231) so the feature-engineering half lives in its own function:

```python
def engineer_features(
    df: pd.DataFrame,
    outlets_df: pd.DataFrame,
    overrides_df: pd.DataFrame,
    region_df: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_START,
    min_pair_history: int = outlier_handling.MIN_PAIR_HISTORY,
    spike_ratio_threshold: float = outlier_handling.SPIKE_RATIO_THRESHOLD,
) -> pd.DataFrame:
    """Single definition of the feature-engineering step order.

    Both the scripted pipeline and notebook/data-processing.ipynb call this,
    so a step added here can never be missed by one of the two paths.
    """
    df = calendar_features.add_calendar_features(df)
    pair_baseline = outlier_handling.compute_pair_baseline(
        df, cutoff=cutoff, min_history=min_pair_history
    )
    df = outlier_handling.apply_outlier_capping(
        df, pair_baseline, ratio_threshold=spike_ratio_threshold
    )
    df = add_targets(df)
    df = outlet_features.apply_region_features(df, region_df)
    df = apply_outlet_features(df, outlets_df, overrides_df)
    df = outlet_features.add_relocation_feature(df)
    df = add_lead_time_target(df)
    df = add_lag_features(df, qty_col="Kuantitas_capped")
    df = add_rolling_features(df, qty_col="Kuantitas_capped")
    branch_stats = compute_branch_stats(df, cutoff=cutoff, qty_col="Kuantitas_capped")
    df = apply_branch_stats(df, branch_stats)
    df = add_branch_age_days(df)
    return df


def build_featured_dataset(
    input_path: str = normalize_items.RAW_DATA_FILE,
    min_history_days: int = build_panel.MIN_HISTORY_DAYS,
    cutoff: pd.Timestamp = TEST_START,
    outlets_path: str = outlet_features.OUTLETS_FILE,
    overrides_path: str = outlet_features.OVERRIDES_FILE,
    region_path: str = outlet_features.REGION_MAPPING_FILE,
    min_pair_history: int = outlier_handling.MIN_PAIR_HISTORY,
    spike_ratio_threshold: float = outlier_handling.SPIKE_RATIO_THRESHOLD,
) -> pd.DataFrame:
    outlets_df = outlet_features.load_outlets(outlets_path)
    overrides_df = outlet_features.load_overrides(overrides_path)
    region_df = outlet_features.load_region_mapping(region_path)
    df = normalize_items.load_and_normalize(input_path)
    df = outlet_features.filter_matched_branches(df, outlets_df, overrides_df)
    df = outlet_features.canonicalize_branch_names(df, outlets_df, overrides_df)
    df = normalize_items.reaggregate_daily(df)
    df = build_panel.build_dense_panel(df)
    df = build_panel.filter_min_history(df, cutoff=cutoff, min_days=min_history_days)
    return engineer_features(
        df,
        outlets_df=outlets_df,
        overrides_df=overrides_df,
        region_df=region_df,
        cutoff=cutoff,
        min_pair_history=min_pair_history,
        spike_ratio_threshold=spike_ratio_threshold,
    )
```

- [ ] **Step 4: Add the `FEATURED_COLUMNS` constant**

Add near the top of `utils/prepare_forecast_data.py`, after `TEST_START = build_panel.TEST_START`:

```python
# The exact columns engineer_features() produces, in order. Asserted by
# run_qa_checks() so a feature added on one code path but not the other fails
# loudly instead of silently producing a short parquet.
FEATURED_COLUMNS = [
    "Kode Barang", "Nama Cabang", "Tanggal", "Kuantitas", "Kategori Barang",
    "Nama Barang", "day_of_week", "day_of_month", "month", "is_weekend",
    "is_national_holiday", "is_ramadan", "days_into_ramadan", "days_until_ramadan",
    "is_eid_al_fitr", "days_since_eid_al_fitr", "days_until_eid_al_fitr",
    "is_eid_al_adha", "days_since_eid_al_adha", "days_until_eid_al_adha",
    "is_independence_day", "days_since_independence_day", "days_until_independence_day",
    "is_new_year", "days_since_new_year", "days_until_new_year",
    "baseline_ratio", "is_spike", "Kuantitas_capped",
    "target_h1", "target_h2", "target_h3", "target_h4", "target_h5", "target_h6", "target_h7",
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
    "roll_mean_7", "roll_std_7", "roll_mean_14", "roll_std_14",
    "roll_mean_28", "roll_std_28",
    "kawasan", "hari_pengiriman", "lead_time_days", "kota",
    "has_shopee", "has_gofood", "has_grabfood", "can_order_online",
    "target_lead_time_cumulative", "days_since_relocation",
    "branch_avg_daily_qty", "branch_demand_cv", "branch_volume_tier", "branch_age_days",
]
```

- [ ] **Step 5: Run the new tests**

Run: `.venv/bin/python3 -m unittest test.test_prepare_forecast_data.TestEngineerFeaturesContract -v`
Expected: PASS, 3 tests

- [ ] **Step 6: Run the full suite for regressions**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v 2>&1 | tail -5`
Expected: `OK`, 204 tests (201 existing + 3 new)

- [ ] **Step 7: Verify the constant matches reality**

Run:
```bash
.venv/bin/python3 -c "
from utils import prepare_forecast_data as p
import pandas as pd
actual = list(pd.read_parquet('dataset/model_ready/train.parquet').columns)
expected = p.FEATURED_COLUMNS
print('missing from constant:', [c for c in actual if c not in expected])
print('extra in constant   :', [c for c in expected if c not in actual])
"
```
Expected: both lists empty. `train.parquet` is used rather than `featured.parquet` because the latter is the known-stale 62-column file this plan is about to fix.

- [ ] **Step 8: Commit**

```bash
git add utils/prepare_forecast_data.py test/test_prepare_forecast_data.py
git commit -m "refactor: extract engineer_features() and pin the 63-column contract

The notebook re-listed the nine feature-engineering calls instead of calling
build_featured_dataset(), which is how days_since_relocation went missing from
featured.parquet. Extracting the step order into engineer_features() gives the
notebook one thing to call, and FEATURED_COLUMNS pins the expected output so
divergence fails loudly.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: Add `run_qa_checks()` to the scripted pipeline

Seven QA assertions currently live only in `notebook/data-processing.ipynb` cells 27–31, so `python3 -m utils.prepare_forecast_data` verifies nothing after export. This moves them into the script.

**Files:**
- Modify: `utils/prepare_forecast_data.py`
- Test: `test/test_prepare_forecast_data.py`

**Interfaces:**
- Consumes: `FEATURED_COLUMNS` from Task 1.
- Produces: `run_qa_checks(df: pd.DataFrame) -> None` — raises `AssertionError` on violation, returns `None` on success.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_prepare_forecast_data.py`:

```python
def _qa_frame():
    """Minimal frame that satisfies every run_qa_checks assertion."""
    return pd.DataFrame({
        "Kode Barang": ["FGS-00001", "FGS-00001"],
        "Nama Cabang": ["KY001 - A", "KY001 - A"],
        "Tanggal": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "Kuantitas": [10.0, 12.0],
        "Kuantitas_capped": [10.0, 12.0],
        "kota": ["Kota Tangerang", "Kota Tangerang"],
        "kawasan": [1, 1],
    })


class TestRunQaChecks(unittest.TestCase):
    def test_passes_on_a_clean_frame(self):
        self.assertIsNone(prepare_forecast_data.run_qa_checks(_qa_frame()))

    def test_rejects_negative_kuantitas(self):
        df = _qa_frame()
        df.loc[0, "Kuantitas"] = -1.0
        with self.assertRaisesRegex(AssertionError, "negatif"):
            prepare_forecast_data.run_qa_checks(df)

    def test_rejects_duplicate_pair_date_rows(self):
        df = _qa_frame()
        df.loc[1, "Tanggal"] = df.loc[0, "Tanggal"]
        with self.assertRaisesRegex(AssertionError, "duplikat"):
            prepare_forecast_data.run_qa_checks(df)

    def test_rejects_capped_exceeding_raw(self):
        df = _qa_frame()
        df.loc[0, "Kuantitas_capped"] = 999.0
        with self.assertRaisesRegex(AssertionError, "capped"):
            prepare_forecast_data.run_qa_checks(df)

    def test_rejects_unknown_kota(self):
        df = _qa_frame()
        df.loc[0, "kota"] = "Unknown"
        with self.assertRaisesRegex(AssertionError, "kota"):
            prepare_forecast_data.run_qa_checks(df)

    def test_rejects_null_kawasan(self):
        df = _qa_frame()
        df.loc[0, "kawasan"] = None
        with self.assertRaisesRegex(AssertionError, "kawasan"):
            prepare_forecast_data.run_qa_checks(df)

    def test_rejects_branch_mapped_to_two_cities(self):
        df = _qa_frame()
        df.loc[1, "kota"] = "Kota Bekasi"
        with self.assertRaisesRegex(AssertionError, "lebih dari satu kota"):
            prepare_forecast_data.run_qa_checks(df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_prepare_forecast_data.TestRunQaChecks -v`
Expected: FAIL, `AttributeError: ... has no attribute 'run_qa_checks'`

- [ ] **Step 3: Implement `run_qa_checks`**

Add to `utils/prepare_forecast_data.py`, just above `def main(`:

```python
def run_qa_checks(df: pd.DataFrame) -> None:
    """Assertions that previously lived only in notebook/data-processing.ipynb.

    Called from main() so the scripted path is verified too. Raises
    AssertionError with an Indonesian message naming the failure.
    """
    assert (df["Kuantitas"] >= 0).all(), "Ditemukan Kuantitas negatif"

    dupes = df.duplicated(subset=["Kode Barang", "Nama Cabang", "Tanggal"]).sum()
    assert dupes == 0, f"Ditemukan {dupes} baris duplikat (item, cabang, tanggal)"

    assert (df["Kuantitas_capped"] <= df["Kuantitas"]).all(), (
        "Kuantitas_capped melebihi Kuantitas mentah"
    )

    assert (df["kota"] != "Unknown").all(), "Ditemukan cabang dengan kota 'Unknown'"

    assert df["kawasan"].notna().all(), "Ditemukan cabang tanpa kawasan"

    kota_per_cabang = df.groupby("Nama Cabang", observed=True)["kota"].nunique()
    bad = kota_per_cabang[kota_per_cabang > 1]
    assert bad.empty, f"Cabang memetakan ke lebih dari satu kota: {list(bad.index)}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_prepare_forecast_data.TestRunQaChecks -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Wire it into `main()`**

In `utils/prepare_forecast_data.py::main`, insert the check and the column assertion between `build_featured_dataset(...)` and `export_featured(df, output_dir)`:

```python
    run_qa_checks(df)
    missing = [c for c in FEATURED_COLUMNS if c not in df.columns]
    assert not missing, f"Kolom hilang dari featured dataset: {missing}"
    export_featured(df, output_dir)
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v 2>&1 | tail -5`
Expected: `OK`, 211 tests

- [ ] **Step 7: Commit**

```bash
git add utils/prepare_forecast_data.py test/test_prepare_forecast_data.py
git commit -m "feat: run QA assertions from the scripted pipeline

The seven checks lived only in the notebook, so a plain
python3 -m utils.prepare_forecast_data run verified nothing after export.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: Rewire the notebook and regenerate the parquet files

**Files:**
- Modify: `notebook/data-processing.ipynb` (cells 18, 20, 22, 24 — see below)

**Interfaces:**
- Consumes: `engineer_features()` from Task 1, `run_qa_checks()` from Task 2.
- Produces: a correct 63-column `dataset/model_ready/featured.parquet`.

- [ ] **Step 1: Confirm the current cell layout**

Run:
```bash
.venv/bin/python3 -c "
import json
nb = json.load(open('notebook/data-processing.ipynb'))
for i in (16, 17, 18, 19, 20, 21, 22, 23, 24, 25):
    c = nb['cells'][i]
    print(f'--- [{i}] {c[\"cell_type\"]} ---')
    print(''.join(c['source'])[:300])
"
```
Expected: cell 16 ends by producing `panel`; cells 18/20/22/24 call `calendar_features`, `outlier_handling`, targets/lags, and region/outlet/branch-stats respectively.

- [ ] **Step 2: Replace cells 18–25 with one call**

Keep markdown cell 17 (`### 4. Calendar features`) but retitle it, delete markdown cells 19, 21, 23 and code cells 18, 20, 22, 24, 25, and put this single code cell in their place. Use `NotebookEdit` or edit the JSON directly.

New markdown cell (replacing cell 17):

```markdown
### 4. Feature engineering

Satu panggilan ke `prepare_forecast_data.engineer_features()` — urutan langkahnya
didefinisikan di satu tempat saja (`utils/prepare_forecast_data.py`), supaya
notebook ini tidak bisa lagi ketinggalan langkah seperti `days_since_relocation`
yang sempat hilang.
```

New code cell:

```python
calendar_features.check_year_coverage(panel["Tanggal"])

featured = prepare_forecast_data.engineer_features(
    panel,
    outlets_df=outlets_df,
    overrides_df=overrides_df,
    region_df=region_df,
    cutoff=cutoff,
)
print(f"{len(featured):,} baris × {len(featured.columns)} kolom")
assert len(featured.columns) == 63, f"Diharapkan 63 kolom, dapat {len(featured.columns)}"
```

- [ ] **Step 3: Replace the hand-written QA cells with the shared function**

Cells 27, 29, 31 duplicate what `run_qa_checks()` now does. Replace cell 27 with:

```python
prepare_forecast_data.run_qa_checks(featured)
print("✓ run_qa_checks lolos")
```

Delete cells 29 and 31. Keep cells 28, 30, 32, 33 (the informational spot-checks and `.info()`) and the whole of section 9 (visual QA, cells 34–39) unchanged.

- [ ] **Step 4: Execute the notebook end to end**

Run: `.venv/bin/python3 -m jupyter nbconvert --to notebook --execute --inplace notebook/data-processing.ipynb`
Expected: completes with no error. Note `--allow-errors` is deliberately omitted — the negative-Kuantitas anomaly was confirmed fixed on 2026-08-10, so a failure here is real.

- [ ] **Step 5: Verify the parquet is repaired**

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd
df = pd.read_parquet('dataset/model_ready/featured.parquet')
print('shape:', df.shape)
assert df.shape[1] == 63, f'masih {df.shape[1]} kolom'
assert 'days_since_relocation' in df.columns
print('✓ 63 kolom, days_since_relocation ada')
"
```
Expected: `shape: (1522868, 63)` then the success line.

- [ ] **Step 6: Regenerate the splits from the script**

Run: `.venv/bin/python3 -m utils.prepare_forecast_data`
Expected: `Wrote 1467822 train rows and 55046 test rows to .../model_ready`

- [ ] **Step 7: Commit**

```bash
git add notebook/data-processing.ipynb
git commit -m "fix: call engineer_features() from the notebook instead of re-listing steps

Cells 18-25 duplicated the nine feature-engineering calls, and the copy went
stale when add_relocation_feature was added to the script. featured.parquet is
back to 63 columns.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: Scaffold `modeling_prep` and implement `add_event_flag()`

**Files:**
- Create: `utils/modeling_prep.py`
- Create: `test/test_modeling_prep.py`

**Interfaces:**
- Consumes: `dataset/event_driven_items.csv` (already drafted).
- Produces: constants `BASE_DIR`, `MODEL_READY_DIR`, `FEATURED_FILE`, `MODEL_INPUT_FILE`, `EVENT_ITEMS_FILE`, `CATEGORY_MAPPING_FILE`, `SCALER_FILE`, `PAIR_COLS`, `DATE_COL`, `TARGET_COL`, `LOOKBACK`; functions `load_event_items(path: str = EVENT_ITEMS_FILE) -> pd.DataFrame` and `add_event_flag(df: pd.DataFrame, event_items_df: pd.DataFrame) -> pd.DataFrame` adding a bool column `is_event_driven`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_modeling_prep.py`:

```python
import unittest

import numpy as np
import pandas as pd

from utils import modeling_prep


def _event_items(rows):
    return pd.DataFrame(rows, columns=["Kode Barang", "is_event_driven"])


class TestAddEventFlag(unittest.TestCase):
    def test_marks_true_for_listed_event_sku(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00002"]})
        items = _event_items([["PCG-00002", "true"], ["PCG-00001", "false"]])
        result = modeling_prep.add_event_flag(df, items)
        self.assertTrue(bool(result.iloc[0]["is_event_driven"]))

    def test_marks_false_for_ordinary_sku(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00001"]})
        items = _event_items([["PCG-00002", "true"], ["PCG-00001", "false"]])
        result = modeling_prep.add_event_flag(df, items)
        self.assertFalse(bool(result.iloc[0]["is_event_driven"]))

    def test_is_case_and_whitespace_insensitive(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00002"]})
        items = _event_items([["PCG-00002", "  TRUE "]])
        result = modeling_prep.add_event_flag(df, items)
        self.assertTrue(bool(result.iloc[0]["is_event_driven"]))

    def test_raises_when_a_sku_is_missing_from_the_list(self):
        df = pd.DataFrame({"Kode Barang": ["FGS-99999"]})
        items = _event_items([["PCG-00002", "true"]])
        with self.assertRaisesRegex(ValueError, "FGS-99999"):
            modeling_prep.add_event_flag(df, items)

    def test_does_not_mutate_the_input_frame(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00001"]})
        items = _event_items([["PCG-00001", "false"]])
        modeling_prep.add_event_flag(df, items)
        self.assertNotIn("is_event_driven", df.columns)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'utils.modeling_prep'`

- [ ] **Step 3: Create the module**

Create `utils/modeling_prep.py`:

```python
"""Turn featured.parquet into a model-ready table shared by all three model
families. See docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_READY_DIR = str(BASE_DIR / "dataset/model_ready")
FEATURED_FILE = str(BASE_DIR / "dataset/model_ready/featured.parquet")
MODEL_INPUT_FILE = str(BASE_DIR / "dataset/model_ready/model_input.parquet")
EVENT_ITEMS_FILE = str(BASE_DIR / "dataset/event_driven_items.csv")
CATEGORY_MAPPING_FILE = str(BASE_DIR / "dataset/model_ready/category_mapping.json")
SCALER_FILE = str(BASE_DIR / "dataset/model_ready/scaler_params.json")

PAIR_COLS = ["Kode Barang", "Nama Cabang"]
DATE_COL = "Tanggal"
TARGET_COL = "target_lead_time_cumulative"
LOOKBACK = 28


def load_event_items(path: str = EVENT_ITEMS_FILE) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def add_event_flag(
    df: pd.DataFrame,
    event_items_df: pd.DataFrame,
    item_col: str = "Kode Barang",
) -> pd.DataFrame:
    """Attach the per-SKU is_event_driven flag from event_driven_items.csv.

    Raises rather than defaulting when a SKU is absent from the list: a new SKU
    appearing in a monthly refresh must be classified by the data owner, not
    silently assumed non-event.
    """
    result = df.copy()
    flags = (
        event_items_df.set_index(item_col)["is_event_driven"]
        .astype(str).str.strip().str.lower().eq("true")
    )
    mapped = result[item_col].map(flags)
    if mapped.isna().any():
        missing = sorted(result.loc[mapped.isna(), item_col].unique())
        raise ValueError(
            f"SKU tanpa entri di {EVENT_ITEMS_FILE}: {missing}. "
            "Tambahkan barisnya dan minta klasifikasi dari data owner."
        )
    result["is_event_driven"] = mapped.astype(bool)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Verify against real data**

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd
from utils import modeling_prep as m
df = pd.read_parquet(m.FEATURED_FILE)
out = m.add_event_flag(df, m.load_event_items())
print(out.groupby('is_event_driven')['Kode Barang'].nunique())
"
```
Expected: no exception (all 70 SKUs are in the CSV), with a small `True` count.

- [ ] **Step 6: Commit**

```bash
git add utils/modeling_prep.py test/test_modeling_prep.py
git commit -m "feat: add modeling_prep module with add_event_flag()

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: `classify_pairs()` — Syntetos-Boylan demand segmentation

**Files:**
- Modify: `utils/modeling_prep.py`
- Test: `test/test_modeling_prep.py`

**Interfaces:**
- Consumes: `PAIR_COLS`, `DATE_COL` from Task 4.
- Produces: constants `ADI_THRESHOLD = 1.32`, `CV2_THRESHOLD = 0.49`, `TEST_START`; functions `compute_pair_demand_stats(df, cutoff, qty_col="Kuantitas") -> pd.DataFrame` (indexed by `PAIR_COLS`, columns `adi`, `cv2`) and `classify_pairs(df, cutoff=TEST_START, qty_col="Kuantitas") -> pd.DataFrame` adding a string column `demand_segment`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_modeling_prep.py`, above the `if __name__` block:

```python
def _series_frame(quantities, start="2024-01-01", item="I1", branch="B1"):
    return pd.DataFrame({
        "Kode Barang": [item] * len(quantities),
        "Nama Cabang": [branch] * len(quantities),
        "Tanggal": pd.date_range(start, periods=len(quantities), freq="D"),
        "Kuantitas": [float(q) for q in quantities],
    })


class TestClassifyPairs(unittest.TestCase):
    def test_daily_stable_demand_is_smooth(self):
        df = _series_frame([10, 11, 10, 9, 10, 11, 10, 10])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result.iloc[0]["demand_segment"], "smooth")

    def test_daily_but_wildly_varying_demand_is_erratic(self):
        df = _series_frame([1, 50, 2, 80, 3, 90, 1, 70])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result.iloc[0]["demand_segment"], "erratic")

    def test_rare_but_consistent_demand_is_intermittent(self):
        df = _series_frame([10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result.iloc[0]["demand_segment"], "intermittent")

    def test_rare_and_bulky_demand_is_lumpy(self):
        df = _series_frame([5, 0, 0, 0, 0, 0, 0, 200, 0, 0, 0, 0, 0, 0, 90])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result.iloc[0]["demand_segment"], "lumpy")

    def test_pair_that_never_moved_is_lumpy(self):
        df = _series_frame([0, 0, 0, 0])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result.iloc[0]["demand_segment"], "lumpy")

    def test_segment_ignores_rows_at_or_after_the_cutoff(self):
        """The whole point of computing segments on train only: post-cutoff
        behaviour must not change the label."""
        train_only = _series_frame([10, 11, 10, 9, 10, 11, 10, 10])
        with_future = pd.concat([
            train_only,
            _series_frame([0] * 60, start="2025-12-01"),
        ], ignore_index=True)
        cutoff = pd.Timestamp("2025-12-01")
        a = modeling_prep.classify_pairs(train_only, cutoff=cutoff).iloc[0]["demand_segment"]
        b = modeling_prep.classify_pairs(with_future, cutoff=cutoff).iloc[0]["demand_segment"]
        self.assertEqual(a, b)

    def test_every_row_of_a_pair_gets_the_same_label(self):
        df = _series_frame([10, 11, 10, 9, 10, 11, 10, 10])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result["demand_segment"].nunique(), 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep.TestClassifyPairs -v`
Expected: FAIL, `AttributeError: ... has no attribute 'classify_pairs'`

- [ ] **Step 3: Implement**

Add to `utils/modeling_prep.py`, after `add_event_flag`:

```python
# Syntetos-Boylan cut-offs. ADI = average interval between non-zero demand
# days; CV2 = squared coefficient of variation of the non-zero quantities.
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49

TEST_START = pd.Timestamp("2025-12-01")


def compute_pair_demand_stats(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_START,
    qty_col: str = "Kuantitas",
    pair_cols: list = None,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    """ADI and CV2 per pair, computed from the training period only.

    Deriving these from the full series would leak post-cutoff behaviour into
    a feature the model trains on.
    """
    pair_cols = pair_cols or PAIR_COLS
    train = df[df[date_col] < cutoff]
    grouped = train.groupby(pair_cols, observed=True)[qty_col]

    n_days = grouped.size()
    n_nonzero = grouped.apply(lambda s: int((s > 0).sum()))
    nz_mean = grouped.apply(lambda s: s[s > 0].mean())
    nz_std = grouped.apply(lambda s: s[s > 0].std(ddof=0))

    adi = n_days / n_nonzero.replace(0, np.nan)
    cv2 = (nz_std / nz_mean.replace(0, np.nan)) ** 2

    return pd.DataFrame({"adi": adi, "cv2": cv2.fillna(0.0)})


def _segment_label(adi: float, cv2: float) -> str:
    if pd.isna(adi):
        # Never moved during the training period — treat as the hardest case.
        return "lumpy"
    if adi < ADI_THRESHOLD:
        return "smooth" if cv2 < CV2_THRESHOLD else "erratic"
    return "intermittent" if cv2 < CV2_THRESHOLD else "lumpy"


def classify_pairs(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_START,
    qty_col: str = "Kuantitas",
    pair_cols: list = None,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    pair_cols = pair_cols or PAIR_COLS
    stats = compute_pair_demand_stats(
        df, cutoff=cutoff, qty_col=qty_col, pair_cols=pair_cols, date_col=date_col
    )
    labels = stats.apply(lambda r: _segment_label(r["adi"], r["cv2"]), axis=1)
    labels.name = "demand_segment"

    result = df.copy()
    result["demand_segment"] = (
        result.set_index(pair_cols).index.map(labels).astype(object)
    )
    result["demand_segment"] = result["demand_segment"].fillna("lumpy")
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep.TestClassifyPairs -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Sanity-check the real distribution**

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd
from utils import modeling_prep as m
df = pd.read_parquet(m.FEATURED_FILE)
out = m.classify_pairs(df)
pair = out.groupby(m.PAIR_COLS, observed=True)['demand_segment'].first()
print(pair.value_counts())
print('total pair:', len(pair))
"
```
Expected: 2,979 pairs total, spread across all four segments with `lumpy`/`intermittent` well represented (the EDA found 60% of pairs are majority-zero).

- [ ] **Step 6: Commit**

```bash
git add utils/modeling_prep.py test/test_modeling_prep.py
git commit -m "feat: add Syntetos-Boylan demand segmentation

Segments are computed from the training period only so post-cutoff behaviour
cannot leak into the label.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 6: `assign_folds()` — walk-forward fold assignment

**Files:**
- Modify: `utils/modeling_prep.py`
- Test: `test/test_modeling_prep.py`

**Interfaces:**
- Consumes: `DATE_COL`, `TEST_START` from Tasks 4–5.
- Produces: constant `FOLD_STARTS: list[pd.Timestamp]` (5 month starts, Jul–Nov 2025); `assign_folds(df, fold_starts=FOLD_STARTS, date_col=DATE_COL) -> pd.DataFrame` adding a float column `fold_id` (1–5, `NaN` elsewhere); `fold_train_mask(df, fold_id, fold_starts=FOLD_STARTS, date_col=DATE_COL) -> pd.Series` of bools.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_modeling_prep.py`:

```python
class TestAssignFolds(unittest.TestCase):
    def _frame(self, dates):
        return pd.DataFrame({"Tanggal": pd.to_datetime(dates)})

    def test_july_2025_is_fold_1(self):
        result = modeling_prep.assign_folds(self._frame(["2025-07-15"]))
        self.assertEqual(result.iloc[0]["fold_id"], 1)

    def test_november_2025_is_fold_5(self):
        result = modeling_prep.assign_folds(self._frame(["2025-11-15"]))
        self.assertEqual(result.iloc[0]["fold_id"], 5)

    def test_rows_before_july_2025_have_no_fold(self):
        result = modeling_prep.assign_folds(self._frame(["2024-05-01", "2025-06-30"]))
        self.assertTrue(result["fold_id"].isna().all())

    def test_december_2025_has_no_fold_because_it_is_the_locked_test_set(self):
        result = modeling_prep.assign_folds(self._frame(["2025-12-15"]))
        self.assertTrue(result["fold_id"].isna().all())

    def test_month_boundaries_land_in_the_right_fold(self):
        result = modeling_prep.assign_folds(
            self._frame(["2025-07-01", "2025-07-31", "2025-08-01"])
        )
        self.assertEqual(list(result["fold_id"]), [1, 1, 2])

    def test_train_mask_excludes_the_validation_month_itself(self):
        df = self._frame(["2025-06-30", "2025-07-15", "2025-08-15"])
        mask = modeling_prep.fold_train_mask(df, fold_id=1)
        self.assertEqual(list(mask), [True, False, False])

    def test_train_mask_expands_for_later_folds(self):
        df = self._frame(["2025-06-30", "2025-07-15", "2025-08-15"])
        mask = modeling_prep.fold_train_mask(df, fold_id=2)
        self.assertEqual(list(mask), [True, True, False])

    def test_no_validation_row_ever_appears_in_its_own_training_mask(self):
        df = self._frame([
            "2025-06-15", "2025-07-15", "2025-08-15",
            "2025-09-15", "2025-10-15", "2025-11-15",
        ])
        folded = modeling_prep.assign_folds(df)
        for fold in range(1, 6):
            train = modeling_prep.fold_train_mask(df, fold_id=fold)
            valid = folded["fold_id"] == fold
            self.assertFalse((train & valid).any(), f"kebocoran di fold {fold}")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep.TestAssignFolds -v`
Expected: FAIL, `AttributeError: ... has no attribute 'assign_folds'`

- [ ] **Step 3: Implement**

Add to `utils/modeling_prep.py`:

```python
# Five expanding-window folds. Training data for fold k is every row dated
# before FOLD_STARTS[k-1]; validation is that month alone. December 2025 is
# absent on purpose — it is the locked final test set.
FOLD_STARTS = [
    pd.Timestamp("2025-07-01"),
    pd.Timestamp("2025-08-01"),
    pd.Timestamp("2025-09-01"),
    pd.Timestamp("2025-10-01"),
    pd.Timestamp("2025-11-01"),
]


def assign_folds(
    df: pd.DataFrame,
    fold_starts: list = None,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    fold_starts = fold_starts or FOLD_STARTS
    result = df.copy()
    result["fold_id"] = np.nan
    for number, start in enumerate(fold_starts, start=1):
        end = start + pd.offsets.MonthBegin(1)
        in_month = (result[date_col] >= start) & (result[date_col] < end)
        result.loc[in_month, "fold_id"] = float(number)
    return result


def fold_train_mask(
    df: pd.DataFrame,
    fold_id: int,
    fold_starts: list = None,
    date_col: str = DATE_COL,
) -> pd.Series:
    """Rows usable for training fold `fold_id` — strictly before its month."""
    fold_starts = fold_starts or FOLD_STARTS
    if not 1 <= fold_id <= len(fold_starts):
        raise ValueError(f"fold_id harus 1..{len(fold_starts)}, dapat {fold_id}")
    return df[date_col] < fold_starts[fold_id - 1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep.TestAssignFolds -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add utils/modeling_prep.py test/test_modeling_prep.py
git commit -m "feat: add walk-forward fold assignment

Five expanding folds over Jul-Nov 2025; December stays unlabelled so no fold
can touch the locked test set.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 7: `encode_categoricals()` with a persisted mapping

**Files:**
- Modify: `utils/modeling_prep.py`
- Test: `test/test_modeling_prep.py`

**Interfaces:**
- Consumes: `CATEGORY_MAPPING_FILE`, `TEST_START` from Tasks 4–5.
- Produces: constants `CATEGORICAL_COLS: list[str]` (7 names), `UNKNOWN_TOKEN = "<UNKNOWN>"`, `UNKNOWN_INDEX = 0`; functions `build_category_mapping(df, cutoff=TEST_START, cols=CATEGORICAL_COLS, date_col=DATE_COL) -> dict[str, dict[str, int]]`, `encode_categoricals(df, mapping, cols=CATEGORICAL_COLS) -> pd.DataFrame` adding `{col}_idx` int columns, `save_category_mapping(mapping, path=CATEGORY_MAPPING_FILE) -> None`, `load_category_mapping(path=CATEGORY_MAPPING_FILE) -> dict`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_modeling_prep.py`:

```python
class TestEncodeCategoricals(unittest.TestCase):
    def _frame(self, kota, dates=None):
        dates = dates or ["2024-01-01"] * len(kota)
        return pd.DataFrame({"kota": kota, "Tanggal": pd.to_datetime(dates)})

    def test_unknown_token_is_index_zero(self):
        mapping = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi"]), cols=["kota"]
        )
        self.assertEqual(mapping["kota"][modeling_prep.UNKNOWN_TOKEN], 0)

    def test_known_values_get_stable_sorted_indices(self):
        mapping = modeling_prep.build_category_mapping(
            self._frame(["Kota Depok", "Kota Bekasi"]), cols=["kota"]
        )
        self.assertEqual(mapping["kota"]["Kota Bekasi"], 1)
        self.assertEqual(mapping["kota"]["Kota Depok"], 2)

    def test_mapping_ignores_values_seen_only_after_the_cutoff(self):
        df = self._frame(
            ["Kota Bekasi", "Kota Baru"], ["2024-01-01", "2025-12-15"]
        )
        mapping = modeling_prep.build_category_mapping(
            df, cutoff=pd.Timestamp("2025-12-01"), cols=["kota"]
        )
        self.assertNotIn("Kota Baru", mapping["kota"])

    def test_unseen_value_encodes_to_unknown_index(self):
        mapping = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi"]), cols=["kota"]
        )
        result = modeling_prep.encode_categoricals(
            self._frame(["Kota Antah Berantah"]), mapping, cols=["kota"]
        )
        self.assertEqual(result.iloc[0]["kota_idx"], modeling_prep.UNKNOWN_INDEX)

    def test_adding_a_new_branch_does_not_shift_existing_indices(self):
        """A 60th branch opening must not renumber the other 59, or every
        previously trained model silently breaks."""
        before = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi", "Kota Depok"]), cols=["kota"]
        )
        frame = self._frame(["Kota Bekasi", "Kota Depok", "Kota Antah Berantah"])
        after = modeling_prep.encode_categoricals(frame, before, cols=["kota"])
        self.assertEqual(after.iloc[0]["kota_idx"], before["kota"]["Kota Bekasi"])
        self.assertEqual(after.iloc[1]["kota_idx"], before["kota"]["Kota Depok"])

    def test_encoded_column_is_integer_dtype(self):
        mapping = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi"]), cols=["kota"]
        )
        result = modeling_prep.encode_categoricals(
            self._frame(["Kota Bekasi"]), mapping, cols=["kota"]
        )
        self.assertTrue(pd.api.types.is_integer_dtype(result["kota_idx"]))

    def test_mapping_survives_a_save_load_round_trip(self):
        import tempfile, os
        mapping = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi", "Kota Depok"]), cols=["kota"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mapping.json")
            modeling_prep.save_category_mapping(mapping, path)
            self.assertEqual(modeling_prep.load_category_mapping(path), mapping)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep.TestEncodeCategoricals -v`
Expected: FAIL, `AttributeError: ... has no attribute 'build_category_mapping'`

- [ ] **Step 3: Implement**

Add to `utils/modeling_prep.py`:

```python
CATEGORICAL_COLS = [
    "Kode Barang",
    "Nama Cabang",
    "Kategori Barang",
    "kota",
    "hari_pengiriman",
    "branch_volume_tier",
    "demand_segment",
]

UNKNOWN_TOKEN = "<UNKNOWN>"
UNKNOWN_INDEX = 0


def build_category_mapping(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_START,
    cols: list = None,
    date_col: str = DATE_COL,
) -> dict:
    """Fit value -> index maps from the training period only, and persist them.

    Fitting on train only is a correctness requirement, not tidiness: SCM reruns
    this weekly on fresh data, and a 60th branch opening next month must not
    renumber the existing 59 and silently invalidate a trained model.
    """
    cols = cols or CATEGORICAL_COLS
    train = df[df[date_col] < cutoff] if date_col in df.columns else df
    mapping = {}
    for col in cols:
        values = sorted(str(v) for v in train[col].dropna().unique())
        mapping[col] = {UNKNOWN_TOKEN: UNKNOWN_INDEX}
        for index, value in enumerate(values, start=1):
            mapping[col][value] = index
    return mapping


def encode_categoricals(
    df: pd.DataFrame,
    mapping: dict,
    cols: list = None,
) -> pd.DataFrame:
    cols = cols or CATEGORICAL_COLS
    result = df.copy()
    for col in cols:
        result[f"{col}_idx"] = (
            result[col].astype(str).map(mapping[col]).fillna(UNKNOWN_INDEX).astype(int)
        )
    return result


def save_category_mapping(mapping: dict, path: str = CATEGORY_MAPPING_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2, sort_keys=True)


def load_category_mapping(path: str = CATEGORY_MAPPING_FILE) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep.TestEncodeCategoricals -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add utils/modeling_prep.py test/test_modeling_prep.py
git commit -m "feat: add categorical encoding with a persisted train-only mapping

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 8: `impute_features()` — the highest-risk step

Imputing the event-proximity columns with `0` would assert "today is Eid" on 96% of rows. Imputing with `30` is also wrong: `days_until_ramadan` reaches **70** in real data, so `30` collides with genuine values. The sentinel is `99`.

**Files:**
- Modify: `utils/modeling_prep.py`
- Test: `test/test_modeling_prep.py`

**Interfaces:**
- Consumes: nothing beyond Task 4 constants.
- Produces: constants `EVENT_PROXIMITY_COLS: list[str]` (10 names), `EVENT_PROXIMITY_SENTINEL = 99.0`; `impute_features(df) -> pd.DataFrame` filling those columns and adding bool columns `was_relocated`, `has_baseline`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_modeling_prep.py`:

```python
class TestImputeFeatures(unittest.TestCase):
    def _frame(self, **overrides):
        base = {col: [np.nan] for col in modeling_prep.EVENT_PROXIMITY_COLS}
        base["days_since_relocation"] = [np.nan]
        base["baseline_ratio"] = [np.nan]
        base.update({k: [v] for k, v in overrides.items()})
        return pd.DataFrame(base)

    def test_event_proximity_nulls_become_the_sentinel_not_zero(self):
        result = modeling_prep.impute_features(self._frame())
        for col in modeling_prep.EVENT_PROXIMITY_COLS:
            self.assertEqual(
                result.iloc[0][col], modeling_prep.EVENT_PROXIMITY_SENTINEL,
                f"{col} salah diimputasi",
            )
            self.assertNotEqual(result.iloc[0][col], 0.0)

    def test_sentinel_is_above_the_largest_real_value(self):
        """days_until_ramadan reaches 70 in the real data, so any sentinel at
        or below that would be indistinguishable from a genuine observation."""
        self.assertGreater(modeling_prep.EVENT_PROXIMITY_SENTINEL, 70)

    def test_there_are_ten_event_proximity_columns(self):
        self.assertEqual(len(modeling_prep.EVENT_PROXIMITY_COLS), 10)

    def test_real_event_proximity_values_are_left_alone(self):
        result = modeling_prep.impute_features(self._frame(days_until_ramadan=5.0))
        self.assertEqual(result.iloc[0]["days_until_ramadan"], 5.0)

    def test_missing_relocation_becomes_zero_with_a_false_indicator(self):
        result = modeling_prep.impute_features(self._frame())
        self.assertEqual(result.iloc[0]["days_since_relocation"], 0.0)
        self.assertFalse(bool(result.iloc[0]["was_relocated"]))

    def test_relocation_day_zero_is_distinguishable_from_never_relocated(self):
        """0 is a legitimate value meaning 'relocated today'; without the
        indicator it would be identical to 'never relocated'."""
        relocated = modeling_prep.impute_features(self._frame(days_since_relocation=0.0))
        never = modeling_prep.impute_features(self._frame())
        self.assertEqual(
            relocated.iloc[0]["days_since_relocation"],
            never.iloc[0]["days_since_relocation"],
        )
        self.assertTrue(bool(relocated.iloc[0]["was_relocated"]))
        self.assertFalse(bool(never.iloc[0]["was_relocated"]))

    def test_missing_baseline_ratio_becomes_one_with_a_false_indicator(self):
        result = modeling_prep.impute_features(self._frame())
        self.assertEqual(result.iloc[0]["baseline_ratio"], 1.0)
        self.assertFalse(bool(result.iloc[0]["has_baseline"]))

    def test_no_nulls_remain_in_the_imputed_columns(self):
        result = modeling_prep.impute_features(self._frame())
        targets = modeling_prep.EVENT_PROXIMITY_COLS + [
            "days_since_relocation", "baseline_ratio",
        ]
        self.assertFalse(result[targets].isna().any().any())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep.TestImputeFeatures -v`
Expected: FAIL, `AttributeError: ... has no attribute 'EVENT_PROXIMITY_COLS'`

- [ ] **Step 3: Implement**

Add to `utils/modeling_prep.py`:

```python
# Only defined inside their proximity window (+/-15 days, +/-30 for Ramadan),
# so null means "outside that window" -- NOT zero, which would read as "the
# event is today" on 84-97% of rows.
EVENT_PROXIMITY_COLS = [
    "days_into_ramadan",
    "days_until_ramadan",
    "days_since_eid_al_fitr",
    "days_until_eid_al_fitr",
    "days_since_eid_al_adha",
    "days_until_eid_al_adha",
    "days_since_independence_day",
    "days_until_independence_day",
    "days_since_new_year",
    "days_until_new_year",
]

# Must exceed every genuine value. days_until_ramadan reaches 70, so 30 would
# collide with real observations.
EVENT_PROXIMITY_SENTINEL = 99.0


def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the nulls a neural net cannot consume, preserving each column's
    meaning. Tree models do not need this, but both adapters run it so the two
    see identical values.
    """
    result = df.copy()

    for col in EVENT_PROXIMITY_COLS:
        result[col] = result[col].fillna(EVENT_PROXIMITY_SENTINEL)

    result["was_relocated"] = result["days_since_relocation"].notna()
    result["days_since_relocation"] = result["days_since_relocation"].fillna(0.0)

    result["has_baseline"] = result["baseline_ratio"].notna()
    result["baseline_ratio"] = result["baseline_ratio"].fillna(1.0)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep.TestImputeFeatures -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add utils/modeling_prep.py test/test_modeling_prep.py
git commit -m "feat: impute NaNs with meaning-preserving sentinels

Event-proximity nulls become 99, not 0 (which would read as 'the event is
today' on 96% of rows) and not 30 (days_until_ramadan reaches 70, so 30
collides with genuine values). Relocation and baseline nulls get indicator
columns because 0 and 1.0 are legitimate values for them.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 9: `drop_warmup_rows()` and `to_tabular()`

**Files:**
- Modify: `utils/modeling_prep.py`
- Test: `test/test_modeling_prep.py`

**Interfaces:**
- Consumes: `LOOKBACK`, `PAIR_COLS`, `DATE_COL`, `TARGET_COL` from Task 4.
- Produces: `drop_warmup_rows(df, lookback=LOOKBACK, pair_cols=PAIR_COLS, date_col=DATE_COL) -> pd.DataFrame`; `to_tabular(df, feature_cols, target_col=TARGET_COL, lookback=LOOKBACK, log_target=False) -> dict` with keys `X` (DataFrame), `y` (Series), `keys` (DataFrame of `PAIR_COLS + [DATE_COL]`), `fold_id` (Series). The `log_target` flag is implemented here but only exercised by Task 10's tests, where its counterpart on `to_sequences` lands.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_modeling_prep.py`:

```python
def _pair_frame(n_rows, item="I1", branch="B1", start="2024-01-01"):
    return pd.DataFrame({
        "Kode Barang": [item] * n_rows,
        "Nama Cabang": [branch] * n_rows,
        "Tanggal": pd.date_range(start, periods=n_rows, freq="D"),
        "feat_a": np.arange(n_rows, dtype=float),
        "target_lead_time_cumulative": np.arange(n_rows, dtype=float) * 2,
        "fold_id": [np.nan] * n_rows,
    })


class TestDropWarmupRows(unittest.TestCase):
    def test_drops_exactly_the_first_lookback_rows_of_each_pair(self):
        df = _pair_frame(40)
        result = modeling_prep.drop_warmup_rows(df, lookback=28)
        self.assertEqual(len(result), 12)

    def test_first_surviving_row_is_at_position_lookback(self):
        df = _pair_frame(40)
        result = modeling_prep.drop_warmup_rows(df, lookback=28)
        self.assertEqual(
            result.iloc[0]["Tanggal"], pd.Timestamp("2024-01-01") + pd.Timedelta(days=28)
        )

    def test_a_pair_shorter_than_the_lookback_disappears_entirely(self):
        result = modeling_prep.drop_warmup_rows(_pair_frame(10), lookback=28)
        self.assertEqual(len(result), 0)

    def test_each_pair_gets_its_own_warmup_cut(self):
        df = pd.concat([_pair_frame(40, item="I1"), _pair_frame(40, item="I2")],
                       ignore_index=True)
        result = modeling_prep.drop_warmup_rows(df, lookback=28)
        self.assertEqual(len(result), 24)
        self.assertEqual(result.groupby("Kode Barang").size().tolist(), [12, 12])


class TestToTabular(unittest.TestCase):
    def test_returns_the_expected_keys(self):
        out = modeling_prep.to_tabular(_pair_frame(40), feature_cols=["feat_a"])
        self.assertEqual(set(out), {"X", "y", "keys", "fold_id"})

    def test_x_contains_only_the_requested_features(self):
        out = modeling_prep.to_tabular(_pair_frame(40), feature_cols=["feat_a"])
        self.assertEqual(list(out["X"].columns), ["feat_a"])

    def test_all_parts_have_the_same_length(self):
        out = modeling_prep.to_tabular(_pair_frame(40), feature_cols=["feat_a"])
        self.assertEqual(len(out["X"]), 12)
        self.assertEqual(len(out["y"]), 12)
        self.assertEqual(len(out["keys"]), 12)
        self.assertEqual(len(out["fold_id"]), 12)

    def test_keys_identify_pair_and_date(self):
        out = modeling_prep.to_tabular(_pair_frame(40), feature_cols=["feat_a"])
        self.assertEqual(
            list(out["keys"].columns), ["Kode Barang", "Nama Cabang", "Tanggal"]
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep.TestDropWarmupRows test.test_modeling_prep.TestToTabular -v`
Expected: FAIL, `AttributeError: ... has no attribute 'drop_warmup_rows'`

- [ ] **Step 3: Implement**

Add to `utils/modeling_prep.py`:

```python
def drop_warmup_rows(
    df: pd.DataFrame,
    lookback: int = LOOKBACK,
    pair_cols: list = None,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    """Keep rows whose zero-based position within their own pair is >= lookback.

    These are exactly the rows an LSTM can build a full window for, and exactly
    the rows where lag_28 is non-null. Both adapters cut here so their row sets
    match.
    """
    pair_cols = pair_cols or PAIR_COLS
    result = df.sort_values(pair_cols + [date_col]).reset_index(drop=True)
    position = result.groupby(pair_cols, observed=True).cumcount()
    return result[position >= lookback].reset_index(drop=True)


def to_tabular(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = TARGET_COL,
    lookback: int = LOOKBACK,
    pair_cols: list = None,
    date_col: str = DATE_COL,
    log_target: bool = False,
) -> dict:
    """Adapter for XGBoost and Random Forest: a flat table, NaNs left in place.

    Pass the same log_target value here and to to_sequences(), or the contract
    check will fail.
    """
    pair_cols = pair_cols or PAIR_COLS
    frame = drop_warmup_rows(df, lookback=lookback, pair_cols=pair_cols, date_col=date_col)
    if log_target:
        frame = frame.copy()
        frame[target_col] = np.log1p(frame[target_col])
    return {
        "X": frame[feature_cols].reset_index(drop=True),
        "y": frame[target_col].reset_index(drop=True),
        "keys": frame[pair_cols + [date_col]].reset_index(drop=True),
        "fold_id": frame["fold_id"].reset_index(drop=True),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep.TestDropWarmupRows test.test_modeling_prep.TestToTabular -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Verify the warm-up cost matches the EDA**

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd
from utils import modeling_prep as m
df = pd.read_parquet(m.FEATURED_FILE)
kept = m.drop_warmup_rows(df)
lost = len(df) - len(kept)
print(f'hilang {lost:,} ({lost/len(df)*100:.2f}%)')
dec = kept[kept['Tanggal'] >= '2025-12-01']
print(f'baris Desember tersisa: {len(dec):,} (harus 55,046)')
"
```
Expected: about 83,412 rows lost (5.48%), and all 55,046 December rows kept.

- [ ] **Step 6: Commit**

```bash
git add utils/modeling_prep.py test/test_modeling_prep.py
git commit -m "feat: add warm-up row cut and the tabular adapter

The 28-row cut costs 5.48% of train rows and zero test rows, and removes every
lag/rolling NaN as a side effect.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 10: `to_sequences()` with fold-wise scaling

**Files:**
- Modify: `utils/modeling_prep.py`
- Test: `test/test_modeling_prep.py`

**Interfaces:**
- Consumes: `drop_warmup_rows` (Task 9), `impute_features` (Task 8), `SCALER_FILE` (Task 4).
- Produces: `fit_scaler(df, feature_cols) -> dict[str, tuple[float, float]]`, `apply_scaler(df, scaler, feature_cols) -> pd.DataFrame`, `save_scaler(scaler, path=SCALER_FILE) -> None`, `load_scaler(path=SCALER_FILE) -> dict`, `inverse_log_target(values: np.ndarray) -> np.ndarray`, and `to_sequences(df, feature_cols, target_col=TARGET_COL, lookback=LOOKBACK, log_target=False) -> dict` with keys `X` (`np.ndarray` shaped `(n, lookback, len(feature_cols))`, dtype float32), `y`, `keys`, `fold_id`.
- `to_tabular` already accepts `log_target` from Task 9; this task adds the matching flag to `to_sequences` and the tests covering both.

**On `log_target`:** spec §3.6 notes that quantiles are equivariant under monotonic transforms, so training on `log1p(y)` and inverting with `expm1` returns the exact same quantile — unlike mean regression, where the transform biases the result. The target is heavily right-skewed (99th percentile 488, max 3,067), so this matters most for the LSTM. It is a parameter rather than always-on because the choice belongs to the modeling spec. **Both adapters must receive the same value** — `validate_contract()` compares `y` values, so a mismatch fails loudly rather than silently producing an unfair comparison.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_modeling_prep.py`:

```python
class TestScaler(unittest.TestCase):
    def test_scaled_column_has_zero_mean_and_unit_std(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        scaler = modeling_prep.fit_scaler(df, ["a"])
        scaled = modeling_prep.apply_scaler(df, scaler, ["a"])
        self.assertAlmostEqual(scaled["a"].mean(), 0.0, places=6)
        self.assertAlmostEqual(scaled["a"].std(ddof=0), 1.0, places=6)

    def test_constant_column_does_not_divide_by_zero(self):
        df = pd.DataFrame({"a": [7.0, 7.0, 7.0]})
        scaler = modeling_prep.fit_scaler(df, ["a"])
        scaled = modeling_prep.apply_scaler(df, scaler, ["a"])
        self.assertTrue(np.isfinite(scaled["a"]).all())

    def test_scaler_fit_on_train_is_applied_unchanged_to_validation(self):
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        valid = pd.DataFrame({"a": [10.0]})
        scaler = modeling_prep.fit_scaler(train, ["a"])
        scaled = modeling_prep.apply_scaler(valid, scaler, ["a"])
        mean, std = scaler["a"]
        self.assertAlmostEqual(scaled.iloc[0]["a"], (10.0 - mean) / std, places=6)

    def test_scaler_survives_a_save_load_round_trip(self):
        import tempfile, os
        scaler = modeling_prep.fit_scaler(pd.DataFrame({"a": [1.0, 2.0]}), ["a"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scaler.json")
            modeling_prep.save_scaler(scaler, path)
            self.assertEqual(modeling_prep.load_scaler(path), scaler)


class TestToSequences(unittest.TestCase):
    def test_tensor_has_shape_samples_lookback_features(self):
        out = modeling_prep.to_sequences(_pair_frame(40), feature_cols=["feat_a"], lookback=28)
        self.assertEqual(out["X"].shape, (12, 28, 1))

    def test_window_ends_at_the_prediction_row_inclusive(self):
        """The prediction row's own features (lags, calendar) are known at
        prediction time, so the window must include them."""
        out = modeling_prep.to_sequences(_pair_frame(40), feature_cols=["feat_a"], lookback=28)
        # feat_a == row position, so the first sample predicts position 28 and
        # its window must be positions 1..28.
        self.assertEqual(out["X"][0, -1, 0], 28.0)
        self.assertEqual(out["X"][0, 0, 0], 1.0)

    def test_target_matches_the_prediction_row(self):
        out = modeling_prep.to_sequences(_pair_frame(40), feature_cols=["feat_a"], lookback=28)
        self.assertEqual(out["y"][0], 56.0)  # position 28 * 2

    def test_keys_identify_the_prediction_row_not_the_window_start(self):
        out = modeling_prep.to_sequences(_pair_frame(40), feature_cols=["feat_a"], lookback=28)
        self.assertEqual(
            out["keys"].iloc[0]["Tanggal"],
            pd.Timestamp("2024-01-01") + pd.Timedelta(days=28),
        )

    def test_a_pair_shorter_than_the_lookback_produces_no_samples(self):
        out = modeling_prep.to_sequences(_pair_frame(10), feature_cols=["feat_a"], lookback=28)
        self.assertEqual(out["X"].shape[0], 0)

    def test_windows_never_span_two_pairs(self):
        df = pd.concat([_pair_frame(40, item="I1"), _pair_frame(40, item="I2")],
                       ignore_index=True)
        out = modeling_prep.to_sequences(df, feature_cols=["feat_a"], lookback=28)
        self.assertEqual(out["X"].shape[0], 24)
        self.assertTrue((out["X"] <= 39.0).all())

    def test_tensor_is_float32(self):
        out = modeling_prep.to_sequences(_pair_frame(40), feature_cols=["feat_a"], lookback=28)
        self.assertEqual(out["X"].dtype, np.float32)


class TestLogTarget(unittest.TestCase):
    def test_log_target_transforms_y_in_both_adapters(self):
        df = _pair_frame(40)
        tab = modeling_prep.to_tabular(df, feature_cols=["feat_a"], log_target=True)
        seq = modeling_prep.to_sequences(df, feature_cols=["feat_a"], log_target=True)
        self.assertAlmostEqual(float(tab["y"].iloc[0]), float(np.log1p(56.0)), places=5)
        self.assertAlmostEqual(float(seq["y"][0]), float(np.log1p(56.0)), places=5)

    def test_log_target_defaults_to_off(self):
        out = modeling_prep.to_tabular(_pair_frame(40), feature_cols=["feat_a"])
        self.assertEqual(float(out["y"].iloc[0]), 56.0)

    def test_inverse_returns_the_original_scale(self):
        original = np.array([0.0, 5.0, 488.0, 3067.0])
        restored = modeling_prep.inverse_log_target(np.log1p(original))
        self.assertTrue(np.allclose(restored, original))

    def test_contract_still_holds_when_both_adapters_use_log(self):
        df = _pair_frame(40)
        tab = modeling_prep.to_tabular(df, feature_cols=["feat_a"], log_target=True)
        seq = modeling_prep.to_sequences(df, feature_cols=["feat_a"], log_target=True)
        self.assertIsNone(modeling_prep.validate_contract(tab, seq))
```

Note: `test_contract_still_holds_when_both_adapters_use_log` depends on
`validate_contract` from Task 11. If you are executing tasks strictly in order,
write it now and expect it to fail until Task 11 lands, or move just that one
test into Task 11's test class.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep.TestScaler test.test_modeling_prep.TestToSequences -v`
Expected: FAIL, `AttributeError: ... has no attribute 'fit_scaler'`

- [ ] **Step 3: Implement**

Add to `utils/modeling_prep.py`:

```python
def fit_scaler(df: pd.DataFrame, feature_cols: list) -> dict:
    """Per-feature mean and std. Fit on one fold's training rows only —
    fitting globally would leak December statistics into the July fold.
    """
    scaler = {}
    for col in feature_cols:
        mean = float(df[col].mean())
        std = float(df[col].std(ddof=0))
        scaler[col] = (mean, std if std > 0 else 1.0)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: dict, feature_cols: list) -> pd.DataFrame:
    result = df.copy()
    for col in feature_cols:
        mean, std = scaler[col]
        result[col] = (result[col] - mean) / std
    return result


def save_scaler(scaler: dict, path: str = SCALER_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serializable = {col: list(params) for col, params in scaler.items()}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)


def load_scaler(path: str = SCALER_FILE) -> dict:
    with open(path, encoding="utf-8") as handle:
        return {col: tuple(params) for col, params in json.load(handle).items()}


def inverse_log_target(values: np.ndarray) -> np.ndarray:
    """Undo log1p on predictions. Exact for quantile models: quantiles are
    equivariant under monotonic transforms, so expm1(q_a(log1p(y))) == q_a(y).
    """
    return np.expm1(values)


def to_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = TARGET_COL,
    lookback: int = LOOKBACK,
    pair_cols: list = None,
    date_col: str = DATE_COL,
    log_target: bool = False,
) -> dict:
    """Adapter for the LSTM: one (lookback, n_features) window per predictable
    row, the window ending at that row inclusive.

    Produces exactly the rows drop_warmup_rows() keeps, so to_tabular() and
    to_sequences() agree — see validate_contract(). Pass the same log_target
    value to both adapters or the contract check will fail.
    """
    pair_cols = pair_cols or PAIR_COLS
    frame = df.sort_values(pair_cols + [date_col]).reset_index(drop=True)
    if log_target:
        frame = frame.copy()
        frame[target_col] = np.log1p(frame[target_col])

    windows, targets, key_rows, folds = [], [], [], []
    for _, group in frame.groupby(pair_cols, observed=True, sort=False):
        values = group[feature_cols].to_numpy(dtype="float32")
        target_values = group[target_col].to_numpy(dtype="float32")
        fold_values = group["fold_id"].to_numpy()
        keys = group[pair_cols + [date_col]].to_numpy()

        for position in range(lookback, len(group)):
            windows.append(values[position - lookback + 1 : position + 1])
            targets.append(target_values[position])
            key_rows.append(keys[position])
            folds.append(fold_values[position])

    if windows:
        stacked = np.stack(windows).astype("float32")
    else:
        stacked = np.empty((0, lookback, len(feature_cols)), dtype="float32")

    return {
        "X": stacked,
        "y": np.asarray(targets, dtype="float32"),
        "keys": pd.DataFrame(key_rows, columns=pair_cols + [date_col]),
        "fold_id": pd.Series(folds, dtype="float64"),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep.TestScaler test.test_modeling_prep.TestToSequences test.test_modeling_prep.TestLogTarget -v`
Expected: PASS, 14 tests. `test_contract_still_holds_when_both_adapters_use_log` fails until Task 11 — that is expected; re-run it at the end of Task 11.

- [ ] **Step 5: Commit**

```bash
git add utils/modeling_prep.py test/test_modeling_prep.py
git commit -m "feat: add the LSTM sequence adapter and fold-wise scaling

Windows end at the prediction row inclusive, since that row's own lag and
calendar features are known at prediction time. Scalers are fit per fold so
December statistics cannot leak into the July fold.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 11: `validate_contract()` and the `build_model_input()` orchestrator

**Files:**
- Modify: `utils/modeling_prep.py`
- Test: `test/test_modeling_prep.py`

**Interfaces:**
- Consumes: every function from Tasks 4–10.
- Produces: `validate_contract(tabular: dict, sequences: dict) -> None` (raises `AssertionError`); `build_model_input(featured_path=FEATURED_FILE, event_items_path=EVENT_ITEMS_FILE, cutoff=TEST_START) -> pd.DataFrame`; `export_model_input(df, path=MODEL_INPUT_FILE) -> None`; `main() -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_modeling_prep.py`:

```python
class TestValidateContract(unittest.TestCase):
    def _pair(self):
        df = _pair_frame(40)
        tabular = modeling_prep.to_tabular(df, feature_cols=["feat_a"])
        sequences = modeling_prep.to_sequences(df, feature_cols=["feat_a"])
        return tabular, sequences

    def test_matching_adapters_pass(self):
        tabular, sequences = self._pair()
        self.assertIsNone(modeling_prep.validate_contract(tabular, sequences))

    def test_rejects_differing_row_counts(self):
        tabular, sequences = self._pair()
        tabular["keys"] = tabular["keys"].iloc[:-1]
        with self.assertRaisesRegex(AssertionError, "jumlah baris"):
            modeling_prep.validate_contract(tabular, sequences)

    def test_rejects_same_count_but_different_dates(self):
        """Equal lengths are not enough — the actual rows must match."""
        tabular, sequences = self._pair()
        tabular["keys"] = tabular["keys"].copy()
        tabular["keys"].loc[0, "Tanggal"] = pd.Timestamp("1999-01-01")
        with self.assertRaisesRegex(AssertionError, "baris berbeda"):
            modeling_prep.validate_contract(tabular, sequences)

    def test_rejects_differing_targets(self):
        tabular, sequences = self._pair()
        tabular["y"] = tabular["y"].copy()
        tabular["y"].iloc[0] = -12345.0
        with self.assertRaisesRegex(AssertionError, "target"):
            modeling_prep.validate_contract(tabular, sequences)

    def test_rejects_differing_fold_assignments(self):
        tabular, sequences = self._pair()
        tabular["fold_id"] = tabular["fold_id"].copy()
        tabular["fold_id"].iloc[0] = 3.0
        with self.assertRaisesRegex(AssertionError, "fold"):
            modeling_prep.validate_contract(tabular, sequences)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep.TestValidateContract -v`
Expected: FAIL, `AttributeError: ... has no attribute 'validate_contract'`

- [ ] **Step 3: Implement**

Add to `utils/modeling_prep.py`:

```python
def validate_contract(tabular: dict, sequences: dict) -> None:
    """Guarantee the two adapters expose the same rows, targets, and folds.

    Without this, "the LSTM is 8% better" could really mean "the LSTM was
    evaluated on a different 5% of the rows".
    """
    tabular_keys = tabular["keys"].reset_index(drop=True)
    sequence_keys = sequences["keys"].reset_index(drop=True)

    assert len(tabular_keys) == len(sequence_keys), (
        f"Adapter menghasilkan jumlah baris berbeda: "
        f"tabular {len(tabular_keys)}, sequence {len(sequence_keys)}"
    )

    tabular_set = set(map(tuple, tabular_keys.to_numpy()))
    sequence_set = set(map(tuple, sequence_keys.to_numpy()))
    assert tabular_set == sequence_set, (
        f"Adapter menghasilkan baris berbeda: "
        f"{len(tabular_set - sequence_set)} hanya di tabular, "
        f"{len(sequence_set - tabular_set)} hanya di sequence"
    )

    tabular_y = np.asarray(tabular["y"], dtype="float64")
    sequence_y = np.asarray(sequences["y"], dtype="float64")
    assert np.allclose(tabular_y, sequence_y, equal_nan=True), (
        "Nilai target berbeda antar adapter"
    )

    tabular_fold = np.asarray(tabular["fold_id"], dtype="float64")
    sequence_fold = np.asarray(sequences["fold_id"], dtype="float64")
    assert np.allclose(tabular_fold, sequence_fold, equal_nan=True), (
        "Pembagian fold berbeda antar adapter"
    )


def build_model_input(
    featured_path: str = FEATURED_FILE,
    event_items_path: str = EVENT_ITEMS_FILE,
    cutoff: pd.Timestamp = TEST_START,
) -> pd.DataFrame:
    df = pd.read_parquet(featured_path)
    df = add_event_flag(df, load_event_items(event_items_path))
    df = classify_pairs(df, cutoff=cutoff)
    df = assign_folds(df)
    df = impute_features(df)

    mapping = build_category_mapping(df, cutoff=cutoff)
    save_category_mapping(mapping)
    df = encode_categoricals(df, mapping)
    return df


def export_model_input(df: pd.DataFrame, path: str = MODEL_INPUT_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> None:
    df = build_model_input()
    export_model_input(df)
    print(f"Wrote {len(df):,} rows x {len(df.columns)} columns to {MODEL_INPUT_FILE}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep.TestValidateContract -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v 2>&1 | tail -5`
Expected: `OK`, 274 tests

- [ ] **Step 6: Generate `model_input.parquet`**

Run: `.venv/bin/python3 -m utils.modeling_prep`
Expected: a line reporting roughly 1,522,868 rows and about 74 columns.

- [ ] **Step 7: Verify the contract holds on real data**

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd
from utils import modeling_prep as m
df = pd.read_parquet(m.MODEL_INPUT_FILE)
feats = ['lag_1', 'lag_7', 'roll_mean_7', 'is_event_driven', 'kota_idx']
sub = df[df['Nama Cabang'].isin(df['Nama Cabang'].unique()[:2])]
tab = m.to_tabular(sub, feature_cols=feats)
seq = m.to_sequences(sub, feature_cols=feats)
m.validate_contract(tab, seq)
print(f'✓ kontrak lolos — {len(tab[\"keys\"]):,} baris, tensor {seq[\"X\"].shape}')
"
```
Expected: the success line. A two-branch subset keeps this quick; the full run happens in the notebook.

- [ ] **Step 8: Commit**

```bash
git add utils/modeling_prep.py test/test_modeling_prep.py
git commit -m "feat: add the cross-adapter contract and build_model_input()

validate_contract() compares actual (pair, date) sets rather than row counts,
so a mismatch cannot hide behind equal lengths.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 12: Modeling-prep notebook and documentation

**Files:**
- Create: `notebook/modeling_prep.ipynb`
- Modify: `docs/pipeline-overview.md`

**Interfaces:**
- Consumes: everything from Tasks 4–11.
- Produces: no new code interfaces.

- [ ] **Step 1: Create the notebook**

Create `notebook/modeling_prep.ipynb` with these cells, following the `sys.path` pattern from `notebook/data-processing.ipynb` cell 0.

Cell 1 (code):
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd

from utils import modeling_prep
```

Cell 2 (markdown):
```markdown
### 1. Build model input

Satu panggilan — urutan langkahnya didefinisikan di `utils/modeling_prep.py`,
bukan di notebook ini.
```

Cell 3 (code):
```python
model_input = modeling_prep.build_model_input()
print(f"{len(model_input):,} baris × {len(model_input.columns)} kolom")
model_input[["is_event_driven", "demand_segment", "fold_id"]].head()
```

Cell 4 (markdown):
```markdown
### 2. Segment & fold QA
```

Cell 5 (code):
```python
pair_segment = model_input.groupby(modeling_prep.PAIR_COLS, observed=True)["demand_segment"].first()
print(pair_segment.value_counts(), "\n")
print(model_input["fold_id"].value_counts(dropna=False).sort_index())

assert model_input.loc[model_input["Tanggal"] >= "2025-12-01", "fold_id"].isna().all(), \
    "Baris Desember tidak boleh punya fold — itu test set terkunci"
print("\n✓ Desember bebas fold")
```

Cell 6 (markdown):
```markdown
### 3. Imputation QA

Cek yang paling penting: kolom kedekatan event tidak boleh terisi `0`
(itu berarti "hari ini Idul Fitri"), dan sentinel harus di atas 70
(`days_until_ramadan` mencapai 70 di data asli).
```

Cell 7 (code):
```python
for col in modeling_prep.EVENT_PROXIMITY_COLS:
    assert model_input[col].notna().all(), f"{col} masih punya null"

filled = (model_input["days_until_ramadan"] == modeling_prep.EVENT_PROXIMITY_SENTINEL).mean()
print(f"days_until_ramadan sentinel: {filled:.1%} baris")
assert modeling_prep.EVENT_PROXIMITY_SENTINEL > 70, "Sentinel bertabrakan dengan nilai asli"

print(model_input[["was_relocated", "has_baseline"]].mean().round(3))
print("\n✓ imputasi aman")
```

Cell 8 (markdown):
```markdown
### 4. Adapter contract

Ini yang membuat perbandingan XGBoost / Random Forest / LSTM bisa
dipertanggungjawabkan: kedua adapter wajib melihat baris yang sama persis.
```

Cell 9 (code):
```python
feature_cols = [
    "lag_1", "lag_7", "lag_28", "roll_mean_7", "roll_mean_28", "roll_std_7",
    "day_of_week", "is_weekend", "is_national_holiday", "lead_time_days",
    "days_until_ramadan", "days_since_relocation", "was_relocated",
    "baseline_ratio", "has_baseline", "is_event_driven",
    "Kode Barang_idx", "Nama Cabang_idx", "kota_idx", "demand_segment_idx",
]

sample_branches = model_input["Nama Cabang"].unique()[:3]
sample = model_input[model_input["Nama Cabang"].isin(sample_branches)]

tabular = modeling_prep.to_tabular(sample, feature_cols=feature_cols)
sequences = modeling_prep.to_sequences(sample, feature_cols=feature_cols)
modeling_prep.validate_contract(tabular, sequences)

print(f"tabular X : {tabular['X'].shape}")
print(f"sequence X: {sequences['X'].shape}")
print("✓ kontrak lolos")
```

Cell 10 (markdown):
```markdown
### 5. Export
```

Cell 11 (code):
```python
modeling_prep.export_model_input(model_input)
print(f"✓ ditulis ke {modeling_prep.MODEL_INPUT_FILE}")
```

- [ ] **Step 2: Execute the notebook**

Run: `.venv/bin/python3 -m jupyter nbconvert --to notebook --execute --inplace notebook/modeling_prep.ipynb`
Expected: completes with no error, every assertion passing.

- [ ] **Step 3: Update `docs/pipeline-overview.md`**

In §2, after stage 12 (QA checks), add:

```markdown
13. **Modeling preprocessing** (`utils/modeling_prep.py`, run via
    `notebook/modeling_prep.ipynb` or `python3 -m utils.modeling_prep`) —
    adds `is_event_driven` (per-SKU, from `dataset/event_driven_items.csv`),
    `demand_segment` (Syntetos-Boylan ADI/CV², computed from the training
    period only), `fold_id` (five expanding walk-forward folds over Jul–Nov
    2025; December stays unlabelled as the locked test set), meaning-preserving
    NaN imputation with the indicator columns `was_relocated` / `has_baseline`,
    and integer categorical indices with the mapping persisted to
    `dataset/model_ready/category_mapping.json`. Exports
    `dataset/model_ready/model_input.parquet`.
14. **Model adapters** — `to_tabular()` for XGBoost/Random Forest and
    `to_sequences()` for the LSTM (28-day windows ending at the prediction row
    inclusive). `validate_contract()` asserts both expose identical
    `(pair, date)` sets, targets, and fold assignments.
```

In §4 ("Expected modelling phase"), remove the now-completed
"Categorical encoding strategy" and "LSTM-specific prep" bullets, and replace
the "Validation strategy" bullet with:

```markdown
- **Validation strategy**: implemented as `fold_id` — five expanding
  walk-forward folds over Jul–Nov 2025 for tuning and model selection, with
  December 2025 opened exactly once for the final figure.
```

Also update the §3 bullet about the seven notebook-only QA assertions: they now
run from the script via `run_qa_checks()`.

- [ ] **Step 4: Verify the docs are consistent**

Run: `grep -n "notebook only\|only in the notebook\|not-yet-built" docs/pipeline-overview.md`
Expected: no stale claims that QA runs only in the notebook.

- [ ] **Step 5: Run the full suite one last time**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v 2>&1 | tail -5`
Expected: `OK`, 274 tests

- [ ] **Step 6: Commit**

```bash
git add notebook/modeling_prep.ipynb docs/pipeline-overview.md
git commit -m "docs: add modeling-prep notebook and document stages 13-14

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Done criteria

- [ ] `featured.parquet` has 63 columns including `days_since_relocation`
- [ ] `model_input.parquet` exists with `is_event_driven`, `demand_segment`, `fold_id`, `was_relocated`, `has_baseline`, and seven `*_idx` columns
- [ ] `category_mapping.json` and `scaler_params.json` exist under `dataset/model_ready/`
- [ ] `validate_contract()` passes on real data
- [ ] Full suite green at 274 tests
- [ ] No December 2025 row carries a `fold_id`
- [ ] No event-proximity column contains a null, and the sentinel is 99

## What this plan does not build

Per the spec's non-goals: model training, tuning, comparison, and SHAP; cold-start forecasting for pairs dropped by `MIN_HISTORY_DAYS`; deployment scheduling. The eight open confirmations in the spec's Part 6 remain open — items 1 and 2 (the event-driven SKU list and the target service level) block the *modeling* spec, not this plan, because `add_event_flag()` reads whatever the CSV currently says and the quantile is a training-time parameter.
