# Random Forest Modeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a 0.9-quantile Random Forest on `dataset/model_ready/model_input.parquet` using a walk-forward runner that XGBoost and the LSTM will later reuse unchanged.

**Architecture:** Two pure modules. `utils/walk_forward.py` owns everything that must be identical across model families — row eligibility, the 28-day warm-up cut, the null-target drop, and scoring against the naive baselines on identical rows — and takes the model as an injected `fit_predict(train_df, valid_df) -> np.ndarray` callable. `utils/model_random_forest.py` supplies that callable for a `quantile-forest` `RandomForestQuantileRegressor`, plus the search space, a memory estimator, and persistence. A thin notebook drives the runs.

**Tech Stack:** Python 3.9.6, pandas 2.3.3, numpy 2.0.2, scikit-learn 1.6.1, quantile-forest 1.4.2, joblib, unittest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-18-random-forest-modeling-design.md`. Read it before Task 1.
- Python is `.venv/bin/python3` (3.9.6). Never use a system interpreter.
- All modules live in the root-level `utils` package and use **relative imports** (`from . import evaluation`), matching the existing modules. Run scripts as modules from the repo root: `.venv/bin/python3 -m utils.<name>`.
- Tests are `unittest`, one file per module, in `test/`, named `test_<module>.py`, class-based, using small synthetic DataFrames — never the real parquet.
- Run all tests with `.venv/bin/python3 -m unittest discover -p "test_*.py" -v`; one module with `.venv/bin/python3 -m unittest test.test_walk_forward -v`.
- Target column: `target_lead_time_cumulative`. Quantile: **0.9**, uniform across every SKU.
- Feature list: `modeling_prep.FEATURE_COLS` (56 columns). Never redefine it.
- **December 2025 (`>= 2025-12-01`, i.e. `modeling_prep.TEST_START`) is locked.** It must not appear in any fold's training or validation set, and nothing in this plan scores against it.
- **Do not call `modeling_prep.impute_features()` anywhere in this plan.** `build_model_input()` already ran it; a second pass recomputes `was_relocated` from an already-filled column and sets it `True` on every row.
- Leaf-storage memory budget: **3 GB** (`MEMORY_BUDGET_BYTES = 3 * 1024 ** 3`). Every candidate is screened before it is fitted.
- Commit after every task. Commit messages end with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

---

## File Structure

| File | Responsibility |
|---|---|
| `utils/walk_forward.py` | **Create.** Model-agnostic walk-forward evaluation: row eligibility, fold splitting, scoring, baselines, pooled metrics |
| `utils/model_random_forest.py` | **Create.** QRF wrapper: memory estimator, one-hot expansion, `fit_predict` factory, search space, final fit, persistence |
| `test/test_walk_forward.py` | **Create.** Anti-leakage, identical-rows, and output-schema tests for the runner |
| `test/test_model_random_forest.py` | **Create.** Memory estimator, one-hot, log-target round-trip, non-negativity, reproducibility |
| `notebook/modeling_rf.ipynb` | **Create.** Thin driver: benchmark → search → final walk-forward → tables |
| `docs/hasil-modeling-rf.md` | **Create.** Results record: benchmark numbers, chosen hyperparameters, three result cuts |
| `requirements.txt` | **Modify.** Add scikit-learn, quantile-forest, joblib |
| `.gitignore` | **Modify.** Add `models/` |

---

### Task 1: Dependencies and environment

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Test: none (verified by a smoke command)

**Interfaces:**
- Consumes: nothing
- Produces: an importable `quantile_forest.RandomForestQuantileRegressor` in `.venv`, and a gitignored `models/` directory

- [ ] **Step 1: Add the dependencies**

Append to `requirements.txt`, after the existing `matplotlib` line:

```
scikit-learn==1.6.1
quantile-forest==1.4.2
joblib==1.5.3
```

- [ ] **Step 2: Install them**

Run: `.venv/bin/pip install -r requirements.txt`

Expected: installs `scikit-learn-1.6.1`, `quantile-forest-1.4.2`, `joblib-1.5.3`, `scipy-1.13.1`, `threadpoolctl-3.6.0`. `numpy` must stay at `2.0.2` — if pip reports upgrading numpy, stop and report it, because the rest of the pipeline is pinned against 2.0.2.

- [ ] **Step 3: Smoke-test the quantile API**

Run:

```bash
.venv/bin/python3 -c "
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
rng = np.random.default_rng(0)
X = rng.normal(size=(500, 3))
y = X[:, 0] * 2 + rng.normal(size=500)
m = RandomForestQuantileRegressor(n_estimators=20, max_samples_leaf=None, random_state=0).fit(X, y)
p90 = m.predict(X, quantiles=0.9)
p50 = m.predict(X, quantiles=0.5)
print('shape', p90.shape, 'q90 above q50 on', float((p90 >= p50).mean()) * 100, 'pct of rows')
"
```

Expected: `shape (500,)` and 100.0 pct. If the fraction is below 100, stop — the quantile call is not doing what the plan assumes.

- [ ] **Step 4: Gitignore the models directory**

Add `models/` to `.gitignore` on the line after `dataset/`.

Then run: `mkdir -p models && git check-ignore -v models`

Expected: output naming `.gitignore` and the `models/` pattern.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "$(cat <<'EOF'
build: add scikit-learn, quantile-forest and joblib for modeling

Random Forest predicts the locked 0.9 service level, which sklearn's own
forest cannot do — it minimizes squared or absolute error only. quantile-forest
reads the quantile off the distribution stored in each leaf.

models/ is gitignored: trained forests are large and fully reproducible.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Row eligibility — `eligible_rows()`

The single definition of "which rows any model may see". Everything else in the runner builds on it, and a test pins it to `to_tabular()` so the tabular and sequence paths cannot drift apart.

**Files:**
- Create: `utils/walk_forward.py`
- Test: `test/test_walk_forward.py`

**Interfaces:**
- Consumes: `modeling_prep.drop_warmup_rows()`, `modeling_prep.to_tabular()`, `modeling_prep.TEST_START`, `modeling_prep.DATE_COL`, `modeling_prep.TARGET_COL`, `modeling_prep.LOOKBACK`
- Produces: `walk_forward.eligible_rows(df, lookback=28, date_col="Tanggal", target_col="target_lead_time_cumulative", test_start=TEST_START) -> pd.DataFrame` — all original columns, index reset, December removed, warm-up removed, null targets removed

- [ ] **Step 1: Write the failing tests**

Create `test/test_walk_forward.py`:

```python
import unittest

import numpy as np
import pandas as pd

from utils import modeling_prep, walk_forward


def _panel(n_days=245, pairs=(("I1", "B1"), ("I2", "B1")), start="2025-05-01"):
    """Two pairs spanning 2025-05-01..2025-12-31, so every fold and the locked
    December window are represented.
    """
    rows = []
    for item, branch in pairs:
        for i, date in enumerate(pd.date_range(start, periods=n_days, freq="D")):
            rows.append({
                "Kode Barang": item,
                "Nama Cabang": branch,
                "segment_id": 1,
                "Tanggal": date,
                "target_lead_time_cumulative": float(i % 7),
                "lead_time_days": 3.0,
                "lag_1": float(i % 5),
                "roll_mean_7": float(i % 4),
                "demand_segment": "smooth",
                "is_delivery_day": bool(i % 2),
                "feat_a": float(i),
                "feat_b": float(i % 3),
            })
    return modeling_prep.assign_folds(pd.DataFrame(rows))


FEATURES = ["feat_a", "feat_b"]


class TestEligibleRows(unittest.TestCase):
    def test_drops_the_first_28_days_of_each_pair(self):
        result = walk_forward.eligible_rows(_panel())
        first = result[result["Kode Barang"] == "I1"]["Tanggal"].min()
        self.assertEqual(first, pd.Timestamp("2025-05-29"))

    def test_drops_every_december_row(self):
        result = walk_forward.eligible_rows(_panel())
        self.assertEqual(len(result[result["Tanggal"] >= modeling_prep.TEST_START]), 0)

    def test_drops_rows_with_a_null_target(self):
        panel = _panel()
        panel.loc[panel["Tanggal"] == pd.Timestamp("2025-07-15"), "target_lead_time_cumulative"] = np.nan
        result = walk_forward.eligible_rows(panel)
        self.assertEqual(len(result[result["Tanggal"] == pd.Timestamp("2025-07-15")]), 0)

    def test_keeps_every_original_column(self):
        panel = _panel()
        result = walk_forward.eligible_rows(panel)
        self.assertEqual(set(panel.columns), set(result.columns))

    def test_matches_to_tabular_row_for_row(self):
        """The contract: the tabular adapter and the runner must agree on the
        row set exactly, or a cross-model comparison compares different data.
        """
        panel = _panel()
        pre_december = panel[panel["Tanggal"] < modeling_prep.TEST_START]
        expected = modeling_prep.to_tabular(pre_december, FEATURES)["keys"]
        result = walk_forward.eligible_rows(panel)
        key_cols = ["Kode Barang", "Nama Cabang", "segment_id", "Tanggal"]
        self.assertEqual(
            set(map(tuple, expected[key_cols].to_numpy())),
            set(map(tuple, result[key_cols].to_numpy())),
        )

    def test_does_not_mutate_the_input_frame(self):
        panel = _panel()
        before = len(panel)
        walk_forward.eligible_rows(panel)
        self.assertEqual(len(panel), before)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_walk_forward -v`

Expected: FAIL with `ImportError: cannot import name 'walk_forward' from 'utils'`.

- [ ] **Step 3: Write the module**

Create `utils/walk_forward.py`:

```python
"""Walk-forward evaluation, with nothing in it that knows about a model.

Three models are being compared on this data, and a comparison is only worth
reporting if all three saw the same rows. That is not something discipline can
guarantee across three separate training scripts, so it is guaranteed here
instead: this module owns row eligibility, fold boundaries, and scoring, and
takes the model itself as an injected callable.

`fit_predict(train_df, valid_df) -> np.ndarray` is the entire model interface.
Anything a model needs beyond that — feature selection, imputation, target
transforms, scaling — belongs inside its own wrapper, because those are the
choices the comparison is meant to expose rather than hide.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import evaluation, modeling_prep

FOLDS = (1, 2, 3, 4, 5)

# The two axes a global number hides. A MAE dominated by mostly-zero pairs can
# crown a model that only won where predicting zero was easy, and delivery days
# are the rows that actually put goods on a truck.
GROUP_COLS = ("demand_segment", "is_delivery_day")


def eligible_rows(
    df: pd.DataFrame,
    lookback: int = modeling_prep.LOOKBACK,
    date_col: str = modeling_prep.DATE_COL,
    target_col: str = modeling_prep.TARGET_COL,
    test_start: pd.Timestamp = modeling_prep.TEST_START,
) -> pd.DataFrame:
    """Every row a model may see during walk-forward, all columns retained.

    Three cuts, in this order:

    1. December 2025 and later. Redundant with the fold definitions, and kept
       anyway — the cost of one accidental leak is the credibility of the
       final number, and a redundant guard is cheaper than that.
    2. Each segment's first `lookback` days, where the lag and rolling windows
       do not fit yet. Computed on the whole series, never within a fold,
       because a per-fold cut would delete a pair's first 28 days of every
       month.
    3. Rows with no target, which occur at the end of each segment where the
       lead-time window runs past the available data.

    Cuts 2 and 3 are exactly what `modeling_prep.to_tabular()` applies. This
    function reproduces them while keeping every column, because scoring needs
    `demand_segment`, `is_delivery_day` and the baseline inputs that the
    adapter drops. `test_matches_to_tabular_row_for_row` pins the two together.
    """
    frame = df[df[date_col] < test_start]
    frame = modeling_prep.drop_warmup_rows(frame, lookback=lookback, date_col=date_col)
    frame = frame[frame[target_col].notna()]
    return frame.reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_walk_forward -v`

Expected: 6 tests, all PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/walk_forward.py test/test_walk_forward.py
git commit -m "$(cat <<'EOF'
feat: add eligible_rows, the single definition of model-visible rows

Applies the same warm-up and null-target cuts as to_tabular() but keeps every
column, since scoring needs demand_segment, is_delivery_day and the baseline
inputs the adapter drops. A test pins the two row sets together so the tabular
and sequence paths cannot drift.

December is filtered here as well as by the fold definitions. Redundant on
purpose: one accidental leak costs the credibility of the final number.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Fold splitting — `prepare_fold()`

**Files:**
- Modify: `utils/walk_forward.py`
- Test: `test/test_walk_forward.py`

**Interfaces:**
- Consumes: `walk_forward.eligible_rows()`, `modeling_prep.fold_train_mask()`
- Produces: `walk_forward.prepare_fold(df, fold_id, prepared=False) -> dict` with keys `"train"` and `"valid"`, both `pd.DataFrame` sharing the index of the `eligible_rows()` frame. `prepared=True` means `df` has already been through `eligible_rows()`, which lets a five-fold run pay that cost once.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_walk_forward.py`, before the `if __name__` block:

```python
class TestPrepareFold(unittest.TestCase):
    def test_validation_is_exactly_the_folds_month(self):
        prepared = walk_forward.prepare_fold(_panel(), 1)
        valid = prepared["valid"]
        self.assertEqual(valid["Tanggal"].min(), pd.Timestamp("2025-07-01"))
        self.assertEqual(valid["Tanggal"].max(), pd.Timestamp("2025-07-31"))

    def test_no_training_row_reaches_the_fold_boundary(self):
        for fold_id in walk_forward.FOLDS:
            prepared = walk_forward.prepare_fold(_panel(), fold_id)
            boundary = modeling_prep.FOLD_STARTS[fold_id - 1]
            self.assertLess(prepared["train"]["Tanggal"].max(), boundary)

    def test_purging_removes_rows_whose_target_window_crosses_the_boundary(self):
        """lead_time_days is 3, so 2025-06-30 sums demand through 2025-07-03 —
        three days of the validation month.
        """
        prepared = walk_forward.prepare_fold(_panel(), 1)
        last = prepared["train"]["Tanggal"].max()
        self.assertLessEqual(last, pd.Timestamp("2025-06-27"))

    def test_no_fold_ever_sees_december(self):
        for fold_id in walk_forward.FOLDS:
            prepared = walk_forward.prepare_fold(_panel(), fold_id)
            for part in ("train", "valid"):
                late = prepared[part]["Tanggal"] >= modeling_prep.TEST_START
                self.assertEqual(int(late.sum()), 0, f"fold {fold_id} {part}")

    def test_training_window_expands_with_the_fold_number(self):
        sizes = [len(walk_forward.prepare_fold(_panel(), f)["train"]) for f in walk_forward.FOLDS]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], sizes[-1])

    def test_train_and_valid_never_share_a_row(self):
        prepared = walk_forward.prepare_fold(_panel(), 3)
        overlap = set(prepared["train"].index) & set(prepared["valid"].index)
        self.assertEqual(overlap, set())

    def test_prepared_flag_skips_the_second_cut(self):
        panel = _panel()
        once = walk_forward.eligible_rows(panel)
        from_raw = walk_forward.prepare_fold(panel, 2)["valid"]
        from_prepared = walk_forward.prepare_fold(once, 2, prepared=True)["valid"]
        pd.testing.assert_frame_equal(from_raw, from_prepared)

    def test_rejects_an_out_of_range_fold(self):
        with self.assertRaises(ValueError):
            walk_forward.prepare_fold(_panel(), 9)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_walk_forward -v`

Expected: FAIL with `AttributeError: module 'utils.walk_forward' has no attribute 'prepare_fold'`.

- [ ] **Step 3: Implement `prepare_fold()`**

Append to `utils/walk_forward.py`:

```python
def prepare_fold(
    df: pd.DataFrame,
    fold_id: int,
    prepared: bool = False,
    fold_col: str = "fold_id",
) -> dict:
    """Training and validation frames for one fold.

    Training is every eligible row strictly before the fold's month, purged:
    `target_lead_time_cumulative` sums over H+1..H+lead_time_days, so the last
    few days before a boundary carry a label built partly out of the month
    being validated. `modeling_prep.fold_train_mask` applies that purge.

    Both frames keep the index of the eligible-rows frame, so a caller can line
    predictions up against either without a join.
    """
    frame = df if prepared else eligible_rows(df)
    train_mask = modeling_prep.fold_train_mask(frame, fold_id)
    valid_mask = frame[fold_col] == fold_id
    return {"train": frame[train_mask], "valid": frame[valid_mask]}
```

`fold_train_mask` already raises `ValueError` for a fold outside 1..5, which satisfies `test_rejects_an_out_of_range_fold`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_walk_forward -v`

Expected: 14 tests, all PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/walk_forward.py test/test_walk_forward.py
git commit -m "$(cat <<'EOF'
feat: add prepare_fold with purged expanding training windows

Training for fold k is every eligible row strictly before fold k's month,
purged at the boundary because the lead-time target sums forward into it.
Tests assert no training row reaches its boundary, the window expands with the
fold number, and no fold touches December in either split.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Scoring — `run_fold()`, `run_walk_forward()`, `pooled_metric()`

**Files:**
- Modify: `utils/walk_forward.py`
- Test: `test/test_walk_forward.py`

**Interfaces:**
- Consumes: `walk_forward.prepare_fold()`, `evaluation.score()`, `evaluation.naive_predictions()`
- Produces:
  - `walk_forward.run_fold(df, fold_id, fit_predict, model_name="model", alpha=0.9, prepared=False) -> pd.DataFrame`
  - `walk_forward.run_walk_forward(df, fit_predict, folds=FOLDS, model_name="model", alpha=0.9) -> pd.DataFrame`
  - `walk_forward.pooled_metric(results, model_name, metric="pinball", folds=None) -> float`
  - Result schema: `model`, `fold_id`, `group_col`, `group_value`, `n`, `mae`, `pinball`, `coverage`, `fill_rate`, `shortfall_units`, `overstock_units`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_walk_forward.py`, before the `if __name__` block:

```python
def _perfect(train, valid):
    """A model that cheats. Used to assert the plumbing, not the modeling:
    a perfect prediction must score MAE 0, so any non-zero MAE means the
    runner mis-aligned predictions with labels.
    """
    return valid["target_lead_time_cumulative"].to_numpy(dtype=float)


def _constant(value):
    def fit_predict(train, valid):
        return np.full(len(valid), float(value))
    return fit_predict


class TestRunFold(unittest.TestCase):
    def test_a_perfect_model_scores_zero_error(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf")
        overall = results[(results["model"] == "rf") & results["group_col"].isna()]
        self.assertEqual(len(overall), 1)
        self.assertAlmostEqual(float(overall.iloc[0]["mae"]), 0.0)
        self.assertAlmostEqual(float(overall.iloc[0]["pinball"]), 0.0)

    def test_predictions_are_aligned_row_by_row_not_just_in_count(self):
        """Reversing the prediction vector must change the score. If it does
        not, the runner is comparing sorted or re-indexed values.
        """
        def reversed_model(train, valid):
            return _perfect(train, valid)[::-1]

        straight = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf")
        flipped = walk_forward.run_fold(_panel(), 1, reversed_model, model_name="rf")
        straight_mae = float(straight[straight["group_col"].isna() & (straight["model"] == "rf")].iloc[0]["mae"])
        flipped_mae = float(flipped[flipped["group_col"].isna() & (flipped["model"] == "rf")].iloc[0]["mae"])
        self.assertAlmostEqual(straight_mae, 0.0)
        self.assertGreater(flipped_mae, 0.0)

    def test_every_naive_baseline_is_scored_too(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf")
        self.assertEqual(
            set(results["model"].unique()),
            {"rf", "naive_zero", "naive_lag_1", "naive_roll_mean_7"},
        )

    def test_model_and_baselines_are_scored_on_identical_row_counts(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf")
        overall = results[results["group_col"].isna()]
        self.assertEqual(overall["n"].nunique(), 1)

    def test_reports_each_group_column(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf")
        self.assertEqual(
            set(results["group_col"].dropna().unique()),
            set(walk_forward.GROUP_COLS),
        )

    def test_group_row_counts_sum_to_the_overall_count(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf")
        rf = results[results["model"] == "rf"]
        overall = int(rf[rf["group_col"].isna()].iloc[0]["n"])
        for group_col in walk_forward.GROUP_COLS:
            grouped = rf[rf["group_col"] == group_col]
            self.assertEqual(int(grouped["n"].sum()), overall, group_col)

    def test_carries_every_metric_column(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf")
        for column in ["n", "mae", "pinball", "coverage", "fill_rate",
                       "shortfall_units", "overstock_units"]:
            self.assertIn(column, results.columns)

    def test_rejects_a_prediction_of_the_wrong_length(self):
        def short(train, valid):
            return np.zeros(len(valid) - 1)

        with self.assertRaisesRegex(ValueError, "panjang"):
            walk_forward.run_fold(_panel(), 1, short)


class TestRunWalkForward(unittest.TestCase):
    def test_covers_every_fold(self):
        results = walk_forward.run_walk_forward(_panel(), _perfect, model_name="rf")
        self.assertEqual(sorted(results["fold_id"].unique()), list(walk_forward.FOLDS))

    def test_a_huge_constant_overshoots_and_a_zero_undershoots(self):
        high = walk_forward.run_walk_forward(_panel(), _constant(1000), model_name="rf")
        low = walk_forward.run_walk_forward(_panel(), _constant(0), model_name="rf")
        self.assertAlmostEqual(walk_forward.pooled_metric(high, "rf", "coverage"), 1.0)
        self.assertLess(walk_forward.pooled_metric(low, "rf", "coverage"), 1.0)

    def test_pooled_metric_weights_folds_by_row_count(self):
        results = walk_forward.run_walk_forward(_panel(), _perfect, model_name="rf")
        self.assertAlmostEqual(walk_forward.pooled_metric(results, "rf", "pinball"), 0.0)

    def test_pooled_metric_can_be_restricted_to_the_search_folds(self):
        results = walk_forward.run_walk_forward(_panel(), _perfect, model_name="rf")
        value = walk_forward.pooled_metric(results, "rf", "pinball", folds=(3, 5))
        self.assertAlmostEqual(value, 0.0)

    def test_is_deterministic_for_a_deterministic_model(self):
        first = walk_forward.run_walk_forward(_panel(), _constant(5), model_name="rf")
        second = walk_forward.run_walk_forward(_panel(), _constant(5), model_name="rf")
        pd.testing.assert_frame_equal(first, second)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_walk_forward -v`

Expected: FAIL with `AttributeError: module 'utils.walk_forward' has no attribute 'run_fold'`.

- [ ] **Step 3: Implement the three functions**

Append to `utils/walk_forward.py`:

```python
METRIC_COLS = ("n", "mae", "pinball", "coverage", "fill_rate",
               "shortfall_units", "overstock_units")


def run_fold(
    df: pd.DataFrame,
    fold_id: int,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    model_name: str = "model",
    alpha: float = evaluation.DEFAULT_ALPHA,
    prepared: bool = False,
) -> pd.DataFrame:
    """Fit on one fold's training rows, score on its validation rows.

    The naive baselines are recomputed here, on this fold's exact validation
    rows, rather than quoted from evaluation.py's docstring. The floor moves
    with the data, and a floor measured on a different row set is precisely
    the error this module exists to prevent.
    """
    split = prepare_fold(df, fold_id, prepared=prepared)
    train, valid = split["train"], split["valid"]

    raw = np.asarray(fit_predict(train, valid), dtype=float)
    if raw.shape != (len(valid),):
        raise ValueError(
            f"fit_predict mengembalikan panjang {raw.shape}, "
            f"seharusnya ({len(valid)},)"
        )
    predictions = {model_name: pd.Series(raw, index=valid.index)}
    predictions.update(evaluation.naive_predictions(valid))

    actual = valid[modeling_prep.TARGET_COL]
    rows = []
    for name, prediction in predictions.items():
        rows.append({
            "model": name, "fold_id": fold_id,
            "group_col": None, "group_value": None,
            **evaluation.score(actual, prediction, alpha=alpha),
        })
        for group_col in GROUP_COLS:
            for value, index in valid.groupby(group_col, observed=True).groups.items():
                rows.append({
                    "model": name, "fold_id": fold_id,
                    "group_col": group_col, "group_value": str(value),
                    **evaluation.score(actual.loc[index], prediction.loc[index], alpha=alpha),
                })
    return pd.DataFrame(rows)


def run_walk_forward(
    df: pd.DataFrame,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    folds: tuple = FOLDS,
    model_name: str = "model",
    alpha: float = evaluation.DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Every fold, one long result frame. The eligibility cut runs once."""
    frame = eligible_rows(df)
    parts = [
        run_fold(frame, fold_id, fit_predict, model_name=model_name,
                 alpha=alpha, prepared=True)
        for fold_id in folds
    ]
    return pd.concat(parts, ignore_index=True)


def pooled_metric(
    results: pd.DataFrame,
    model_name: str,
    metric: str = "pinball",
    folds: Optional[tuple] = None,
) -> float:
    """One number across folds, weighted by row count.

    Every metric here is a per-row mean, so weighting by `n` reconstructs the
    value the metric would have had on the pooled rows. A plain average across
    folds would let November — the smallest fold — count as much as July.
    """
    rows = results[(results["model"] == model_name) & results["group_col"].isna()]
    if folds is not None:
        rows = rows[rows["fold_id"].isin(folds)]
    total = rows["n"].sum()
    if total == 0:
        return float("nan")
    return float((rows[metric] * rows["n"]).sum() / total)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_walk_forward -v`

Expected: 27 tests, all PASS.

- [ ] **Step 5: Run the whole suite to check nothing regressed**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v 2>&1 | tail -5`

Expected: `OK`, with the total count now above 195.

- [ ] **Step 6: Commit**

```bash
git add utils/walk_forward.py test/test_walk_forward.py
git commit -m "$(cat <<'EOF'
feat: add the walk-forward scoring loop with baselines on identical rows

run_fold scores the model and all three naive baselines on the same validation
rows, overall and split by demand_segment and is_delivery_day. Baselines are
recomputed per fold rather than quoted, so the floor moves with the data.

pooled_metric weights folds by row count, since every metric is a per-row mean
and a plain average would let November count as much as July.

A test reverses the prediction vector and requires the score to change, which
catches the misalignment a length check alone would miss.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: The Random Forest wrapper

**Files:**
- Create: `utils/model_random_forest.py`
- Test: `test/test_model_random_forest.py`

**Interfaces:**
- Consumes: `modeling_prep.FEATURE_COLS`, `modeling_prep.TARGET_COL`, `modeling_prep.inverse_log_target()`, `quantile_forest.RandomForestQuantileRegressor`
- Produces:
  - `model_random_forest.QUANTILE = 0.9`, `MEMORY_BUDGET_BYTES`, `DEFAULT_PARAMS`, `SEARCH_SPACE`, `IDX_COLS`
  - `assert_no_nan(frame, feature_cols) -> None`
  - `estimate_leaf_memory_bytes(params, n_train) -> int`
  - `expand_one_hot(train_X, valid_X, idx_cols=IDX_COLS) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `build_estimator(params) -> RandomForestQuantileRegressor`
  - `make_fit_predict(params=None, feature_cols=None, quantile=QUANTILE) -> Callable`

- [ ] **Step 1: Write the failing tests**

Create `test/test_model_random_forest.py`:

```python
import unittest

import numpy as np
import pandas as pd

from utils import model_random_forest as rf


FEATURES = ["feat_a", "feat_b", "cat_idx"]


def _frame(n=200, seed=0, target_scale=1.0):
    rng = np.random.default_rng(seed)
    feat_a = rng.normal(size=n)
    return pd.DataFrame({
        "feat_a": feat_a,
        "feat_b": rng.normal(size=n),
        "cat_idx": rng.integers(0, 3, size=n),
        "target_lead_time_cumulative": np.abs(feat_a * 10 + 20) * target_scale,
    })


class TestAssertNoNan(unittest.TestCase):
    def test_passes_on_a_clean_frame(self):
        rf.assert_no_nan(_frame(), FEATURES)

    def test_names_the_offending_column(self):
        frame = _frame()
        frame.loc[0, "feat_b"] = np.nan
        with self.assertRaisesRegex(ValueError, "feat_b"):
            rf.assert_no_nan(frame, FEATURES)


class TestEstimateLeafMemoryBytes(unittest.TestCase):
    def test_scales_linearly_with_tree_count(self):
        base = {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 50,
                "max_samples_leaf": 20, "max_samples": None}
        doubled = {**base, "n_estimators": 200}
        self.assertEqual(
            rf.estimate_leaf_memory_bytes(doubled, 1_000_000),
            2 * rf.estimate_leaf_memory_bytes(base, 1_000_000),
        )

    def test_a_bigger_min_samples_leaf_costs_less(self):
        small = {"n_estimators": 200, "max_depth": 30, "min_samples_leaf": 20,
                 "max_samples_leaf": 20, "max_samples": None}
        large = {**small, "min_samples_leaf": 200}
        self.assertLess(
            rf.estimate_leaf_memory_bytes(large, 1_000_000),
            rf.estimate_leaf_memory_bytes(small, 1_000_000),
        )

    def test_depth_caps_the_node_count(self):
        """A depth-12 tree cannot exceed 2**13 nodes however many rows it sees."""
        params = {"n_estimators": 1, "max_depth": 12, "min_samples_leaf": 1,
                  "max_samples_leaf": 1, "max_samples": None}
        self.assertLessEqual(
            rf.estimate_leaf_memory_bytes(params, 10_000_000),
            2 ** 13 * 8,
        )

    def test_bootstrap_fraction_reduces_the_estimate(self):
        full = {"n_estimators": 200, "max_depth": 30, "min_samples_leaf": 50,
                "max_samples_leaf": 20, "max_samples": None}
        half = {**full, "max_samples": 0.5}
        self.assertLess(
            rf.estimate_leaf_memory_bytes(half, 1_000_000),
            rf.estimate_leaf_memory_bytes(full, 1_000_000),
        )

    def test_the_configuration_the_spec_rejects_blows_the_budget(self):
        params = {"n_estimators": 200, "max_depth": 40, "min_samples_leaf": 1,
                  "max_samples_leaf": 1, "max_samples": None}
        self.assertGreater(
            rf.estimate_leaf_memory_bytes(params, 1_280_000),
            rf.MEMORY_BUDGET_BYTES,
        )

    def test_the_default_configuration_fits_the_budget(self):
        self.assertLess(
            rf.estimate_leaf_memory_bytes(rf.DEFAULT_PARAMS, 1_280_000),
            rf.MEMORY_BUDGET_BYTES,
        )


class TestExpandOneHot(unittest.TestCase):
    def test_row_count_and_order_are_unchanged(self):
        train, valid = _frame(60, seed=1), _frame(40, seed=2)
        train_X, valid_X = rf.expand_one_hot(train[FEATURES], valid[FEATURES], ["cat_idx"])
        self.assertEqual(len(train_X), 60)
        self.assertEqual(len(valid_X), 40)
        self.assertTrue((train_X["feat_a"].to_numpy() == train[FEATURES]["feat_a"].to_numpy()).all())

    def test_columns_match_between_train_and_valid(self):
        train, valid = _frame(60, seed=1), _frame(40, seed=2)
        train_X, valid_X = rf.expand_one_hot(train[FEATURES], valid[FEATURES], ["cat_idx"])
        self.assertEqual(list(train_X.columns), list(valid_X.columns))

    def test_a_category_absent_from_training_does_not_shift_columns(self):
        train = _frame(60, seed=1)
        train["cat_idx"] = 0
        valid = _frame(40, seed=2)
        valid["cat_idx"] = 7
        train_X, valid_X = rf.expand_one_hot(train[FEATURES], valid[FEATURES], ["cat_idx"])
        self.assertEqual(list(train_X.columns), list(valid_X.columns))
        self.assertEqual(int(valid_X["cat_idx_0"].sum()), 0)

    def test_the_index_column_itself_is_gone(self):
        train, valid = _frame(60, seed=1), _frame(40, seed=2)
        train_X, _ = rf.expand_one_hot(train[FEATURES], valid[FEATURES], ["cat_idx"])
        self.assertNotIn("cat_idx", train_X.columns)


class TestMakeFitPredict(unittest.TestCase):
    def _params(self, **overrides):
        return {"n_estimators": 20, "max_depth": 6, "min_samples_leaf": 5,
                "max_samples_leaf": 20, "random_state": 0, **overrides}

    def test_returns_one_prediction_per_validation_row(self):
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        predict = rf.make_fit_predict(self._params(), feature_cols=FEATURES)
        self.assertEqual(predict(train, valid).shape, (80,))

    def test_predictions_are_never_negative(self):
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        predict = rf.make_fit_predict(self._params(), feature_cols=FEATURES)
        self.assertTrue((predict(train, valid) >= 0).all())

    def test_the_high_quantile_sits_above_the_low_one(self):
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        low = rf.make_fit_predict(self._params(), feature_cols=FEATURES, quantile=0.1)
        high = rf.make_fit_predict(self._params(), feature_cols=FEATURES, quantile=0.9)
        self.assertTrue((high(train, valid) >= low(train, valid)).all())

    def test_log_target_returns_predictions_on_the_original_scale(self):
        """Quantiles are equivariant under log1p, so inverting must land back
        in the target's own range — not in log space.
        """
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        predict = rf.make_fit_predict(self._params(log_target=True), feature_cols=FEATURES)
        prediction = predict(train, valid)
        self.assertGreater(prediction.mean(), 5.0)

    def test_one_hot_runs_end_to_end(self):
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        predict = rf.make_fit_predict(self._params(one_hot=True), feature_cols=FEATURES)
        self.assertEqual(predict(train, valid).shape, (80,))

    def test_the_same_seed_gives_the_same_predictions(self):
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        predict = rf.make_fit_predict(self._params(), feature_cols=FEATURES)
        np.testing.assert_array_equal(predict(train, valid), predict(train, valid))

    def test_a_nan_feature_is_rejected_rather_than_imputed(self):
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        train.loc[0, "feat_a"] = np.nan
        predict = rf.make_fit_predict(self._params(), feature_cols=FEATURES)
        with self.assertRaisesRegex(ValueError, "feat_a"):
            predict(train, valid)

    def test_an_over_budget_configuration_is_refused_before_fitting(self):
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        predict = rf.make_fit_predict(
            self._params(n_estimators=200, max_depth=40, min_samples_leaf=1,
                         max_samples_leaf=50),
            feature_cols=FEATURES,
            memory_budget=1000,
        )
        with self.assertRaisesRegex(MemoryError, "budget"):
            predict(train, valid)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_model_random_forest -v`

Expected: FAIL with `ImportError: cannot import name 'model_random_forest' from 'utils'`.

- [ ] **Step 3: Write the module**

Create `utils/model_random_forest.py`:

```python
"""Random Forest at the 0.9 service level.

sklearn's own forest cannot do this. It minimizes squared or absolute error,
both of which target the middle of the distribution, and a forecast that is
right in the middle stocks out roughly half the time by construction. A
quantile regression forest keeps the training targets that reach each leaf and
reads the 0.9 quantile off that empirical distribution instead.

That storage is what makes this model's cost unusual. quantile-forest holds it
in a dense int64 array of shape
(n_estimators, max_node_count, n_outputs, max_samples_leaf), so the memory bill
is fixed by the hyperparameters before a single tree is grown — see
estimate_leaf_memory_bytes(). Deep trees with tiny leaves are unaffordable
here, which happens to agree with the statistics: a leaf holding one sample
cannot estimate a 0.9 quantile at all.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor

from . import modeling_prep

QUANTILE = 0.9

# Leaf storage above this is refused before the fit starts. Discovering the
# limit through the OOM killer twenty minutes into a fit is the alternative.
MEMORY_BUDGET_BYTES = 3 * 1024 ** 3

# The encoded categoricals, the only columns one-hot expansion touches.
IDX_COLS = [col for col in modeling_prep.FEATURE_COLS if col.endswith("_idx")]

DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 16,
    "min_samples_leaf": 50,
    "max_samples_leaf": 20,
    "max_features": "sqrt",
    "max_samples": None,
    "log_target": False,
    "one_hot": False,
    "random_state": 42,
}

# n_estimators is absent on purpose: forest quality is monotone in tree count,
# so searching it spends budget on a question with a known answer. It is pinned
# during the search and raised for the final fit.
#
# max_depth=None and min_samples_leaf below 20 are absent for the reason in the
# module docstring.
SEARCH_SPACE = {
    "max_depth": [12, 16, 20],
    "min_samples_leaf": [20, 50, 100, 200],
    "max_samples_leaf": [1, 20, 50],
    "max_features": ["sqrt", 0.3, 0.5, 1.0],
    "max_samples": [None, 0.5],
    "log_target": [False, True],
    "one_hot": [False, True],
}

ESTIMATOR_KEYS = ("n_estimators", "max_depth", "min_samples_leaf",
                  "max_samples_leaf", "max_features", "max_samples",
                  "random_state")


def assert_no_nan(frame: pd.DataFrame, feature_cols: list) -> None:
    """Fail loudly on a null the forest cannot consume.

    Deliberately not an imputation step. build_model_input() already ran
    impute_features(), and running it a second time would recompute
    was_relocated from a column that is now filled with 0.0, setting the
    indicator True on every row and erasing the distinction it exists to make.
    """
    counts = frame[feature_cols].isna().sum()
    offenders = counts[counts > 0]
    if len(offenders):
        raise ValueError(f"NaN pada fitur: {offenders.to_dict()}")


def estimate_leaf_memory_bytes(params: dict, n_train: int) -> int:
    """Upper bound on quantile-forest's leaf-value array, in bytes.

    bytes = n_estimators x node_count x max_samples_leaf x 8

    node_count is bounded twice: by depth, since a tree of depth d holds at
    most 2^(d+1) nodes, and by leaf size, since n rows split into leaves of at
    least L rows give at most 2n/L nodes including internal ones. The tighter
    bound wins.
    """
    fraction = params.get("max_samples") or 1.0
    n_bootstrap = n_train * fraction
    depth_bound = 2.0 ** (params["max_depth"] + 1)
    leaf_bound = 2.0 * n_bootstrap / params["min_samples_leaf"]
    node_count = min(depth_bound, leaf_bound)
    return int(params["n_estimators"] * node_count * params["max_samples_leaf"] * 8)


def expand_one_hot(
    train_X: pd.DataFrame,
    valid_X: pd.DataFrame,
    idx_cols: Optional[list] = None,
) -> tuple:
    """One-hot the encoded categoricals, with validation reindexed onto the
    training columns.

    The reindex is the point. A category that appears only in validation would
    otherwise add a column there and shift every column after it, so the forest
    would read the wrong feature at every position — silently, since the shapes
    still line up.
    """
    idx_cols = IDX_COLS if idx_cols is None else idx_cols
    present = [col for col in idx_cols if col in train_X.columns]
    train_out = pd.get_dummies(train_X, columns=present)
    valid_out = pd.get_dummies(valid_X, columns=present)
    valid_out = valid_out.reindex(columns=train_out.columns, fill_value=0)
    return train_out, valid_out


def build_estimator(params: dict) -> RandomForestQuantileRegressor:
    kwargs = {key: params[key] for key in ESTIMATOR_KEYS if key in params}
    return RandomForestQuantileRegressor(n_jobs=-1, **kwargs)


def make_fit_predict(
    params: Optional[dict] = None,
    feature_cols: Optional[list] = None,
    quantile: float = QUANTILE,
    memory_budget: int = MEMORY_BUDGET_BYTES,
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """The callable walk_forward.run_fold() injects.

    Feature selection, one-hot expansion and the target transform all live in
    here rather than in the runner, because they are model choices — the very
    things the three-way comparison is supposed to expose.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    feature_cols = feature_cols or modeling_prep.FEATURE_COLS

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
        assert_no_nan(train, feature_cols)
        assert_no_nan(valid, feature_cols)

        needed = estimate_leaf_memory_bytes(params, len(train))
        if needed > memory_budget:
            raise MemoryError(
                f"leaf storage {needed / 1024 ** 3:.1f} GB melebihi budget "
                f"{memory_budget / 1024 ** 3:.1f} GB untuk {params}"
            )

        train_X, valid_X = train[feature_cols], valid[feature_cols]
        if params["one_hot"]:
            train_X, valid_X = expand_one_hot(train_X, valid_X)

        y_train = train[modeling_prep.TARGET_COL].to_numpy(dtype=float)
        if params["log_target"]:
            y_train = np.log1p(y_train)

        model = build_estimator(params)
        model.fit(train_X.to_numpy(dtype=np.float32), y_train)
        prediction = model.predict(valid_X.to_numpy(dtype=np.float32), quantiles=quantile)
        if params["log_target"]:
            prediction = modeling_prep.inverse_log_target(prediction)
        # A negative shipment quantity is not a thing.
        return np.clip(np.asarray(prediction, dtype=float), 0.0, None)

    return fit_predict
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_model_random_forest -v`

Expected: 20 tests, all PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/model_random_forest.py test/test_model_random_forest.py
git commit -m "$(cat <<'EOF'
feat: add the 0.9-quantile Random Forest wrapper

Wraps quantile-forest behind the fit_predict callable the walk-forward runner
injects, with one-hot expansion and the log-target transform as parameters so
both are settled by measurement rather than assumption.

estimate_leaf_memory_bytes bounds the dense leaf array quantile-forest
allocates, and an over-budget configuration is refused before the fit rather
than found by the OOM killer. assert_no_nan checks the invariant instead of
imputing: model_input.parquet is already imputed, and a second pass would set
was_relocated True on every row.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Search space sampling and candidate selection

**Files:**
- Modify: `utils/model_random_forest.py`
- Test: `test/test_model_random_forest.py`

**Interfaces:**
- Consumes: `model_random_forest.SEARCH_SPACE`, `estimate_leaf_memory_bytes()`, `walk_forward.run_walk_forward()`, `walk_forward.pooled_metric()`
- Produces:
  - `sample_search_space(n_candidates=18, n_train=1_280_000, seed=42, memory_budget=MEMORY_BUDGET_BYTES) -> list[dict]`
  - `run_search(df, candidates, folds=(3, 5), alpha=0.9, model_name="random_forest", feature_cols=None) -> pd.DataFrame` with columns `candidate_id`, `pinball`, `mae`, `coverage`, `fill_rate`, plus one column per searched parameter
  - `select_best(search_results, candidates) -> dict`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_model_random_forest.py`, before the `if __name__` block:

```python
class TestSampleSearchSpace(unittest.TestCase):
    def test_returns_the_requested_number_of_candidates(self):
        self.assertEqual(len(rf.sample_search_space(18, n_train=1_280_000, seed=1)), 18)

    def test_candidates_are_distinct(self):
        candidates = rf.sample_search_space(18, n_train=1_280_000, seed=1)
        signatures = {tuple(sorted(c.items(), key=lambda kv: kv[0])) for c in candidates}
        self.assertEqual(len(signatures), 18)

    def test_the_same_seed_reproduces_the_same_list(self):
        first = rf.sample_search_space(10, n_train=1_280_000, seed=7)
        second = rf.sample_search_space(10, n_train=1_280_000, seed=7)
        self.assertEqual(first, second)

    def test_different_seeds_give_different_lists(self):
        first = rf.sample_search_space(10, n_train=1_280_000, seed=7)
        second = rf.sample_search_space(10, n_train=1_280_000, seed=8)
        self.assertNotEqual(first, second)

    def test_every_candidate_fits_the_memory_budget(self):
        for candidate in rf.sample_search_space(18, n_train=1_280_000, seed=1):
            self.assertLessEqual(
                rf.estimate_leaf_memory_bytes(candidate, 1_280_000),
                rf.MEMORY_BUDGET_BYTES,
                candidate,
            )

    def test_every_candidate_carries_a_full_parameter_set(self):
        for candidate in rf.sample_search_space(5, n_train=1_280_000, seed=1):
            for key in rf.DEFAULT_PARAMS:
                self.assertIn(key, candidate)

    def test_a_tiny_budget_that_admits_nothing_raises(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            rf.sample_search_space(18, n_train=1_280_000, seed=1, memory_budget=10)

    def test_only_searched_parameters_vary(self):
        candidates = rf.sample_search_space(18, n_train=1_280_000, seed=1)
        self.assertEqual({c["n_estimators"] for c in candidates},
                         {rf.DEFAULT_PARAMS["n_estimators"]})
        self.assertEqual({c["random_state"] for c in candidates},
                         {rf.DEFAULT_PARAMS["random_state"]})


class TestSelectBest(unittest.TestCase):
    def test_picks_the_lowest_pinball(self):
        candidates = [{"max_depth": 12}, {"max_depth": 16}, {"max_depth": 20}]
        results = pd.DataFrame({
            "candidate_id": [0, 1, 2],
            "pinball": [5.0, 3.0, 4.0],
        })
        self.assertEqual(rf.select_best(results, candidates), {"max_depth": 16})

    def test_ignores_a_candidate_that_failed(self):
        candidates = [{"max_depth": 12}, {"max_depth": 16}]
        results = pd.DataFrame({
            "candidate_id": [0, 1],
            "pinball": [np.nan, 4.0],
        })
        self.assertEqual(rf.select_best(results, candidates), {"max_depth": 16})

    def test_raises_when_every_candidate_failed(self):
        candidates = [{"max_depth": 12}]
        results = pd.DataFrame({"candidate_id": [0], "pinball": [np.nan]})
        with self.assertRaisesRegex(ValueError, "tidak ada kandidat"):
            rf.select_best(results, candidates)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_model_random_forest -v`

Expected: FAIL with `AttributeError: module 'utils.model_random_forest' has no attribute 'sample_search_space'`.

- [ ] **Step 3: Implement the three functions**

Append to `utils/model_random_forest.py` (and add `import random` and `from . import walk_forward` to the imports at the top):

```python
SEARCH_FOLDS = (3, 5)

# Enough training rows to size the memory screen realistically before the data
# is loaded — fold 5 trains on roughly this many rows.
TYPICAL_N_TRAIN = 1_280_000


def sample_search_space(
    n_candidates: int = 18,
    n_train: int = TYPICAL_N_TRAIN,
    seed: int = 42,
    memory_budget: int = MEMORY_BUDGET_BYTES,
    space: Optional[dict] = None,
) -> list:
    """Distinct, within-budget parameter sets drawn at random.

    Random rather than grid: the space is 1,152 combinations and only a few of
    its dimensions carry real signal, so random draws cover each dimension's
    range better than a truncated grid at the same cost.
    """
    space = SEARCH_SPACE if space is None else space
    rng = random.Random(seed)
    keys = sorted(space)
    seen, candidates = set(), []

    for _ in range(n_candidates * 200):
        if len(candidates) == n_candidates:
            break
        drawn = {key: rng.choice(space[key]) for key in keys}
        signature = tuple(drawn[key] for key in keys)
        if signature in seen:
            continue
        candidate = {**DEFAULT_PARAMS, **drawn}
        if estimate_leaf_memory_bytes(candidate, n_train) > memory_budget:
            continue
        seen.add(signature)
        candidates.append(candidate)

    if len(candidates) < n_candidates:
        raise ValueError(
            f"hanya {len(candidates)} dari {n_candidates} kandidat muat dalam "
            f"budget {memory_budget / 1024 ** 3:.1f} GB"
        )
    return candidates


def run_search(
    df: pd.DataFrame,
    candidates: list,
    folds: tuple = SEARCH_FOLDS,
    alpha: float = QUANTILE,
    model_name: str = "random_forest",
    feature_cols: Optional[list] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Score every candidate on the search folds only.

    A candidate that raises is recorded with NaN metrics rather than aborting
    the run: eighteen fits is a long afternoon, and losing all of it to the
    seventeenth configuration would be a poor trade.
    """
    frame = walk_forward.eligible_rows(df)
    rows = []
    for candidate_id, candidate in enumerate(candidates):
        record = {"candidate_id": candidate_id,
                  **{key: candidate[key] for key in sorted(SEARCH_SPACE)}}
        try:
            fit_predict = make_fit_predict(candidate, feature_cols=feature_cols,
                                           quantile=alpha)
            parts = [
                walk_forward.run_fold(frame, fold_id, fit_predict,
                                      model_name=model_name, alpha=alpha,
                                      prepared=True)
                for fold_id in folds
            ]
            results = pd.concat(parts, ignore_index=True)
            for metric in ("pinball", "mae", "coverage", "fill_rate"):
                record[metric] = walk_forward.pooled_metric(
                    results, model_name, metric=metric, folds=folds
                )
            record["error"] = None
        except (MemoryError, ValueError) as failure:
            for metric in ("pinball", "mae", "coverage", "fill_rate"):
                record[metric] = float("nan")
            record["error"] = str(failure)
        if verbose:
            print(f"[{candidate_id + 1}/{len(candidates)}] "
                  f"pinball={record['pinball']:.4f} {record['error'] or ''}")
        rows.append(record)
    return pd.DataFrame(rows)


def select_best(search_results: pd.DataFrame, candidates: list) -> dict:
    """The candidate with the lowest pooled pinball across the search folds.

    Pinball alone decides it. The service level is uniform across every SKU by
    the data owner's decision, so the selection criterion has to be uniform
    too — picking on a per-segment metric would optimize for a split the
    business does not make.
    """
    scored = search_results[search_results["pinball"].notna()]
    if scored.empty:
        raise ValueError("tidak ada kandidat yang berhasil dinilai")
    best_id = int(scored.loc[scored["pinball"].idxmin(), "candidate_id"])
    return candidates[best_id]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_model_random_forest -v`

Expected: 31 tests, all PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/model_random_forest.py test/test_model_random_forest.py
git commit -m "$(cat <<'EOF'
feat: add memory-screened random search over the RF parameter space

18 distinct candidates drawn with a fixed seed, each rejected before it is
fitted if its leaf storage would exceed the budget. Scored on folds 3 and 5
only; the reported number still comes from the full five-fold run, since
reporting the folds that chose the winner would be optimistic.

A candidate that raises is recorded with NaN metrics rather than aborting the
search, so one bad configuration does not cost the whole afternoon.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Final fit and persistence

**Files:**
- Modify: `utils/model_random_forest.py`
- Test: `test/test_model_random_forest.py`

**Interfaces:**
- Consumes: `build_estimator()`, `expand_one_hot()`, `purging.lookahead_safe_mask()`, `joblib`
- Produces:
  - `fit_final(df, params, feature_cols=None, n_estimators=FINAL_N_ESTIMATORS) -> dict` — a bundle `{"model", "params", "feature_cols", "columns", "quantile", "n_train"}`
  - `predict_bundle(bundle, frame) -> np.ndarray`
  - `save_bundle(bundle, path=MODEL_FILE) -> None`, `load_bundle(path=MODEL_FILE) -> dict`
  - `MODEL_FILE`, `BEST_PARAMS_FILE`, `RESULTS_FILE`, `SEARCH_FILE`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_model_random_forest.py`, before the `if __name__` block:

```python
import tempfile
from pathlib import Path


def _dated_frame(n=400, seed=3):
    frame = _frame(n, seed=seed)
    frame["Tanggal"] = pd.date_range("2025-01-01", periods=n, freq="D")
    frame["lead_time_days"] = 3.0
    return frame


class TestFitFinal(unittest.TestCase):
    def _params(self):
        return {"n_estimators": 20, "max_depth": 6, "min_samples_leaf": 5,
                "max_samples_leaf": 20, "random_state": 0}

    def test_bundle_records_what_prediction_needs(self):
        bundle = rf.fit_final(_dated_frame(), self._params(),
                              feature_cols=FEATURES, n_estimators=20)
        for key in ("model", "params", "feature_cols", "columns", "quantile", "n_train"):
            self.assertIn(key, bundle)
        self.assertEqual(bundle["feature_cols"], FEATURES)

    def test_training_stops_before_december(self):
        frame = _dated_frame(n=400)
        bundle = rf.fit_final(frame, self._params(), feature_cols=FEATURES, n_estimators=20)
        eligible = frame[frame["Tanggal"] < pd.Timestamp("2025-12-01")]
        self.assertLessEqual(bundle["n_train"], len(eligible))
        self.assertGreater(bundle["n_train"], 0)

    def test_the_boundary_is_purged(self):
        """lead_time_days is 3, so 2025-11-29 onward is contaminated."""
        frame = _dated_frame(n=400)
        bundle = rf.fit_final(frame, self._params(), feature_cols=FEATURES, n_estimators=20)
        safe = frame[frame["Tanggal"] <= pd.Timestamp("2025-11-27")]
        self.assertLessEqual(bundle["n_train"], len(safe))

    def test_final_tree_count_overrides_the_searched_one(self):
        bundle = rf.fit_final(_dated_frame(), self._params(),
                              feature_cols=FEATURES, n_estimators=33)
        self.assertEqual(bundle["params"]["n_estimators"], 33)

    def test_predict_bundle_returns_one_value_per_row(self):
        frame = _dated_frame()
        bundle = rf.fit_final(frame, self._params(), feature_cols=FEATURES, n_estimators=20)
        self.assertEqual(rf.predict_bundle(bundle, frame.head(25)).shape, (25,))

    def test_predict_bundle_is_non_negative(self):
        frame = _dated_frame()
        bundle = rf.fit_final(frame, self._params(), feature_cols=FEATURES, n_estimators=20)
        self.assertTrue((rf.predict_bundle(bundle, frame.head(25)) >= 0).all())

    def test_one_hot_bundle_predicts_when_a_column_order_differs(self):
        frame = _dated_frame()
        params = {**self._params(), "one_hot": True}
        bundle = rf.fit_final(frame, params, feature_cols=FEATURES, n_estimators=20)
        shuffled = frame.head(25)[list(reversed(FEATURES)) + ["Tanggal", "lead_time_days"]]
        self.assertEqual(rf.predict_bundle(bundle, shuffled).shape, (25,))

    def test_a_saved_bundle_predicts_identically_after_loading(self):
        frame = _dated_frame()
        bundle = rf.fit_final(frame, self._params(), feature_cols=FEATURES, n_estimators=20)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "bundle.joblib")
            rf.save_bundle(bundle, path)
            reloaded = rf.load_bundle(path)
        np.testing.assert_array_equal(
            rf.predict_bundle(bundle, frame.head(25)),
            rf.predict_bundle(reloaded, frame.head(25)),
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_model_random_forest -v`

Expected: FAIL with `AttributeError: module 'utils.model_random_forest' has no attribute 'fit_final'`.

- [ ] **Step 3: Implement persistence and the final fit**

Add `import json` and `from pathlib import Path` plus `import joblib` and `from . import purging` to the imports at the top of `utils/model_random_forest.py`, then append:

```python
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = str(BASE_DIR / "models/random_forest_q90.joblib")
BEST_PARAMS_FILE = str(BASE_DIR / "dataset/model_ready/rf_best_params.json")
SEARCH_FILE = str(BASE_DIR / "dataset/model_ready/rf_search_results.csv")
RESULTS_FILE = str(BASE_DIR / "dataset/model_ready/rf_walk_forward_results.csv")

# Raised from the searched 200. Forest quality is monotone in tree count, so
# the final model buys the variance reduction the search could not afford.
FINAL_N_ESTIMATORS = 400


def fit_final(
    df: pd.DataFrame,
    params: dict,
    feature_cols: Optional[list] = None,
    n_estimators: int = FINAL_N_ESTIMATORS,
    date_col: str = modeling_prep.DATE_COL,
    test_start: pd.Timestamp = modeling_prep.TEST_START,
) -> dict:
    """Fit on every eligible row before December, purged at that boundary.

    The bundle records the exact training column order alongside the model. A
    forest reloaded next week against columns in a different order does not
    fail — it predicts confidently from the wrong features, which is worse.
    """
    params = {**DEFAULT_PARAMS, **params, "n_estimators": n_estimators}
    feature_cols = feature_cols or modeling_prep.FEATURE_COLS

    frame = df[df[date_col] < test_start]
    frame = frame[purging.lookahead_safe_mask(frame, test_start, date_col=date_col)]
    assert_no_nan(frame, feature_cols)

    train_X = frame[feature_cols]
    if params["one_hot"]:
        train_X, _ = expand_one_hot(train_X, train_X)

    y_train = frame[modeling_prep.TARGET_COL].to_numpy(dtype=float)
    if params["log_target"]:
        y_train = np.log1p(y_train)

    model = build_estimator(params)
    model.fit(train_X.to_numpy(dtype=np.float32), y_train)
    return {
        "model": model,
        "params": params,
        "feature_cols": feature_cols,
        "columns": list(train_X.columns),
        "quantile": QUANTILE,
        "n_train": int(len(frame)),
    }


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predict with a fitted bundle, forcing the recorded column order."""
    params = bundle["params"]
    features = frame[bundle["feature_cols"]]
    if params["one_hot"]:
        features, _ = expand_one_hot(features, features)
    features = features.reindex(columns=bundle["columns"], fill_value=0)
    prediction = bundle["model"].predict(
        features.to_numpy(dtype=np.float32), quantiles=bundle["quantile"]
    )
    if params["log_target"]:
        prediction = modeling_prep.inverse_log_target(prediction)
    return np.clip(np.asarray(prediction, dtype=float), 0.0, None)


def save_bundle(bundle: dict, path: str = MODEL_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str = MODEL_FILE) -> dict:
    return joblib.load(path)


def save_best_params(params: dict, path: str = BEST_PARAMS_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, sort_keys=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_model_random_forest -v`

Expected: 39 tests, all PASS.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v 2>&1 | tail -5`

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add utils/model_random_forest.py test/test_model_random_forest.py
git commit -m "$(cat <<'EOF'
feat: add the final RF fit and a self-describing model bundle

fit_final trains on every eligible row before December, purged at that
boundary, and raises the tree count above the searched value since forest
quality is monotone in it.

The bundle records the exact training column order next to the model, and
predict_bundle reindexes onto it. A forest reloaded against reordered columns
does not fail — it predicts confidently from the wrong features.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Benchmark on real data

The first contact with the full dataset. Its purpose is to confirm the memory bound holds and to learn what one fit actually costs, before spending eighteen of them.

**Files:**
- Create: `notebook/modeling_rf.ipynb` (first two cells)
- Test: none — this task measures rather than asserts

**Interfaces:**
- Consumes: `model_random_forest.DEFAULT_PARAMS`, `walk_forward.prepare_fold()`, `model_random_forest.make_fit_predict()`
- Produces: measured wall time and peak RSS for one fold-5 fit, recorded in `docs/hasil-modeling-rf.md`

- [ ] **Step 1: Write the benchmark script**

Create `bench_rf.py` in this session's scratchpad directory (the path given in the
`Scratchpad Directory` section of the environment, not `/tmp`), containing:

```python
import resource
import time

import pandas as pd

from utils import model_random_forest as rf
from utils import modeling_prep, walk_forward

df = pd.read_parquet(modeling_prep.MODEL_INPUT_FILE)
split = walk_forward.prepare_fold(df, 5)
train, valid = split["train"], split["valid"]
print(f"train {len(train):,} rows, valid {len(valid):,} rows")

params = dict(rf.DEFAULT_PARAMS)
print("estimated leaf storage: "
      f"{rf.estimate_leaf_memory_bytes(params, len(train)) / 1024 ** 3:.2f} GB")

start = time.time()
prediction = rf.make_fit_predict(params)(train, valid)
elapsed = time.time() - start

peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
print(f"wall time {elapsed / 60:.1f} min")
print(f"peak RSS  {peak_bytes / 1024 ** 3:.2f} GB")
print(f"prediction mean {prediction.mean():.2f}, max {prediction.max():.2f}")
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python3 <scratchpad>/bench_rf.py` from the repo root, substituting
the scratchpad path from Step 1.

Expected: `train` around 1.2M rows, estimated leaf storage under 3 GB, and a wall time and peak RSS printed. Note all four numbers.

- [ ] **Step 3: Decide the search sizing**

Apply this rule with the measured wall time `T`:

- `T <= 6 min` — run all 18 candidates on both search folds at full training size. No change needed.
- `6 min < T <= 15 min` — keep 18 candidates but expect roughly `18 x 2 x T` minutes; run it in the background and check back.
- `T > 15 min` — subsample training rows to 40%, stratified by `demand_segment` with `random_state=42`, and record the fraction in the results document. Validation rows are never subsampled.

Write the decision and the measured numbers down now; they go into `docs/hasil-modeling-rf.md` in Task 10.

- [ ] **Step 4: Start the notebook**

Create `notebook/modeling_rf.ipynb` with three cells:

Cell 1 (setup — every notebook in this repo starts this way):

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd

from utils import evaluation, model_random_forest as rf
from utils import modeling_prep, walk_forward

df = pd.read_parquet(modeling_prep.MODEL_INPUT_FILE)
print(f"{len(df):,} rows x {df.shape[1]} columns")
```

Cell 2 (markdown):

```markdown
## Benchmark

One fit on fold 5's full training set, to confirm the leaf-storage bound holds
and to size the search. Numbers recorded in `docs/hasil-modeling-rf.md`.
```

Cell 3 — the body of `bench_rf.py` from Step 1, minus its imports.

- [ ] **Step 5: Commit**

```bash
git add notebook/modeling_rf.ipynb
git commit -m "$(cat <<'EOF'
feat: add the RF modeling notebook with the fold-5 benchmark

One fit on the full training set before spending eighteen of them, to confirm
the leaf-storage bound holds in practice and to size the search against a
measured wall time rather than a guess.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Run the search and the final walk-forward

**Files:**
- Modify: `notebook/modeling_rf.ipynb`
- Test: none — this task produces results

**Interfaces:**
- Consumes: `sample_search_space()`, `run_search()`, `select_best()`, `run_walk_forward()`, `make_fit_predict()`, `fit_final()`, `save_bundle()`, `save_best_params()`
- Produces: `dataset/model_ready/rf_search_results.csv`, `dataset/model_ready/rf_best_params.json`, `dataset/model_ready/rf_walk_forward_results.csv`, `models/random_forest_q90.joblib`

- [ ] **Step 1: Add the search cells**

Append to `notebook/modeling_rf.ipynb`:

Markdown cell:

```markdown
## Hyperparameter search

18 candidates, memory-screened, scored on folds 3 and 5 by pooled pinball@0.9.
The reported result comes from the full five-fold run below, not from here —
scoring on the folds that chose the winner would be optimistic.
```

Code cell:

```python
train_size = len(walk_forward.prepare_fold(df, 5)["train"])
candidates = rf.sample_search_space(18, n_train=train_size, seed=42)
search_results = rf.run_search(df, candidates, folds=rf.SEARCH_FOLDS)
search_results.to_csv(rf.SEARCH_FILE, index=False)
search_results.sort_values("pinball").head(10)
```

If the benchmark in Task 8 put you in the `T > 15 min` branch, replace the
third line with:

```python
sampled = (df.groupby("demand_segment", observed=True, group_keys=False)
             .apply(lambda part: part.sample(frac=0.4, random_state=42)))
search_results = rf.run_search(sampled, candidates, folds=rf.SEARCH_FOLDS)
```

- [ ] **Step 2: Run the search**

Run the cells. Expected: 18 progress lines, each printing a pinball value.
Verify before continuing:

```bash
.venv/bin/python3 -c "
import pandas as pd
from utils import model_random_forest as rf
r = pd.read_csv(rf.SEARCH_FILE)
print('candidates', len(r), 'failed', int(r['pinball'].isna().sum()))
print(r.sort_values('pinball').head(3).to_string())
"
```

Expected: 18 candidates, and **at most 2 failed**. If more than 2 failed, stop and report the `error` column — the memory screen or the parameter space is wrong, and continuing would pick a winner from a crippled search.

- [ ] **Step 3: Add the final walk-forward cells**

Append to the notebook:

Markdown cell:

```markdown
## Final walk-forward

The winning configuration across all five folds, against all three naive
baselines on identical rows.
```

Code cell:

```python
best = rf.select_best(search_results, candidates)
rf.save_best_params(best)
print(best)

fit_predict = rf.make_fit_predict(best)
results = walk_forward.run_walk_forward(df, fit_predict, model_name="random_forest")
results.to_csv(rf.RESULTS_FILE, index=False)

overall = results[results["group_col"].isna()]
overall.pivot_table(index="model", columns="fold_id",
                    values="pinball").round(3)
```

- [ ] **Step 4: Run it and check the model beats the floor**

Run: after the cells complete,

```bash
.venv/bin/python3 -c "
import pandas as pd
from utils import model_random_forest as rf, walk_forward
r = pd.read_csv(rf.RESULTS_FILE)
for model in r['model'].unique():
    print(f'{model:20s} pinball {walk_forward.pooled_metric(r, model, \"pinball\"):8.3f}'
          f'  mae {walk_forward.pooled_metric(r, model, \"mae\"):8.3f}'
          f'  coverage {walk_forward.pooled_metric(r, model, \"coverage\"):6.3f}')
"
```

Expected: four lines. `random_forest` coverage should sit near 0.90 — that is what training at the 0.9 quantile promises. Record whether `random_forest` pinball is below `naive_roll_mean_7`. **A loss is a legitimate result and must be reported as one, not tuned away.**

- [ ] **Step 5: Fit and save the final model**

Append to the notebook:

```python
bundle = rf.fit_final(df, best)
rf.save_bundle(bundle)
print(f"trained on {bundle['n_train']:,} rows, "
      f"{len(bundle['columns'])} columns, quantile {bundle['quantile']}")
```

Run it. Expected: a row count close to 1.4M and a saved file at `models/random_forest_q90.joblib`.

- [ ] **Step 6: Commit**

```bash
git add notebook/modeling_rf.ipynb
git commit -m "$(cat <<'EOF'
feat: add the RF search and final walk-forward runs to the notebook

Search on folds 3 and 5, winner reported across all five, then a final fit on
everything before the locked December window.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Record the results

The artifacts in `dataset/model_ready/` and `models/` are gitignored, so without this task the evidence exists only on one laptop.

**Files:**
- Create: `docs/hasil-modeling-rf.md`
- Modify: `notebook/modeling_rf.ipynb` (reporting cells)
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `rf.RESULTS_FILE`, `rf.SEARCH_FILE`, `rf.BEST_PARAMS_FILE`, the Task 8 benchmark numbers
- Produces: a committed results document

- [ ] **Step 1: Add the reporting cells**

Append to `notebook/modeling_rf.ipynb`:

Markdown cell:

```markdown
## Results

Three cuts, each against all three naive baselines on identical rows. A single
global number is misleading on data where 45% of targets are zero.
```

Code cell:

```python
results = pd.read_csv(rf.RESULTS_FILE)

print("=== per fold (overall) ===")
print(results[results["group_col"].isna()]
      .pivot_table(index="model", columns="fold_id", values="pinball").round(3))

for group_col in walk_forward.GROUP_COLS:
    print(f"\n=== per {group_col} (pooled over folds) ===")
    grouped = results[results["group_col"] == group_col]
    table = (grouped.assign(weighted=grouped["pinball"] * grouped["n"])
                    .groupby(["model", "group_value"], observed=True)
                    .apply(lambda part: part["weighted"].sum() / part["n"].sum())
                    .unstack())
    print(table.round(3))

print("\n=== coverage and fill rate (overall, pooled) ===")
for model in results["model"].unique():
    print(f"{model:20s} "
          f"coverage {walk_forward.pooled_metric(results, model, 'coverage'):6.3f}  "
          f"fill_rate {walk_forward.pooled_metric(results, model, 'fill_rate'):6.3f}  "
          f"shortfall {walk_forward.pooled_metric(results, model, 'shortfall_units'):9.1f}")
```

- [ ] **Step 2: Write the results document**

Create `docs/hasil-modeling-rf.md` using this exact skeleton, filling every
bracketed slot from the printed output. Leave no bracket behind.

```markdown
# Hasil modeling — Random Forest (quantile 0.9)

Dijalankan [TANGGAL] terhadap `dataset/model_ready/model_input.parquet`
([N] baris). Desain: `docs/superpowers/specs/2026-08-18-random-forest-modeling-design.md`.

Angka di bawah berasal dari walk-forward lima fold (Jul–Nov 2025).
**Desember 2025 belum dibuka** dan tidak dinilai di sini.

## Benchmark

| Ukuran | Nilai |
|---|---|
| Baris training fold 5 | [N] |
| Estimasi leaf storage | [X] GB |
| Peak RSS terukur | [X] GB |
| Waktu satu fit | [X] menit |
| Keputusan sizing | [penuh / subsample 40%] |

## Hyperparameter terpilih

| Parameter | Nilai |
|---|---|
| `max_depth` | [X] |
| `min_samples_leaf` | [X] |
| `max_samples_leaf` | [X] |
| `max_features` | [X] |
| `max_samples` | [X] |
| `log_target` | [X] |
| `one_hot` | [X] |
| `n_estimators` (final) | 400 |

Dipilih dari 18 kandidat, [K] gagal, berdasarkan pinball@0.9 gabungan di fold 3 dan 5.

## Per fold (pinball@0.9)

| Model | F1 Jul | F2 Agu | F3 Sep | F4 Okt | F5 Nov |
|---|---|---|---|---|---|
| random_forest | | | | | |
| naive_roll_mean_7 | | | | | |
| naive_lag_1 | | | | | |
| naive_zero | | | | | |

## Per demand_segment (pinball@0.9, digabung lintas fold)

| Model | smooth | erratic | intermittent | lumpy |
|---|---|---|---|---|
| random_forest | | | | |
| naive_roll_mean_7 | | | | |

## Kalibrasi

| Model | coverage | fill_rate | shortfall_units |
|---|---|---|---|
| random_forest | | | |
| naive_roll_mean_7 | | | |

Dilatih di quantile 0.9, jadi coverage seharusnya mendekati 0,90. [Satu kalimat:
apakah tercapai, dan jika tidak, di segmen mana meleset.]

## Bacaan

[2–4 kalimat. Apakah RF mengalahkan lantai naive, di segmen mana ia menang dan
di mana tidak, dan apakah kalibrasinya bisa dipercaya. Kekalahan dilaporkan apa
adanya — `evaluation.py` ada justru supaya hasil itu terdeteksi.]

## Batasan yang berlaku

SKU event-driven punya langit-langit informasi, bukan kegagalan model: tanggal
pesanan tidak tercatat di mana pun (`docs/batasan-penelitian.md` B-1/B-2).
Selisih performa di segmen `lumpy` harus dibaca dengan itu di kepala.

## Artefak (gitignored, bisa dibuat ulang)

- `models/random_forest_q90.joblib`
- `dataset/model_ready/rf_best_params.json`
- `dataset/model_ready/rf_search_results.csv`
- `dataset/model_ready/rf_walk_forward_results.csv`

Buat ulang dengan menjalankan `notebook/modeling_rf.ipynb` dari atas.
```

- [ ] **Step 3: Verify no placeholder survived**

Run: `grep -n '\[' docs/hasil-modeling-rf.md`

Expected: only markdown links, if any. Every `[X]`, `[N]`, `[K]`, `[TANGGAL]`
and bracketed instruction must be gone.

- [ ] **Step 4: Update CLAUDE.md**

In the "Project state" paragraph, after the sentence ending
`dataset/model_ready/{train,test}.parquet`, add:

```
Modeling sits on top of that: `utils/walk_forward.py` runs the shared five-fold
walk-forward evaluation every model is scored through, and
`utils/model_random_forest.py` supplies the first model — a 0.9-quantile
Random Forest. See `docs/superpowers/specs/2026-08-18-random-forest-modeling-design.md`
for the design and `docs/hasil-modeling-rf.md` for the measured results.
```

And in the "Commands" list, add:

```
- Run the Random Forest modeling notebook: `jupyter nbconvert --to notebook --execute --inplace notebook/modeling_rf.ipynb`
```

- [ ] **Step 5: Run the full suite one last time**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v 2>&1 | tail -5`

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add docs/hasil-modeling-rf.md notebook/modeling_rf.ipynb CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: record the Random Forest walk-forward results

Benchmark numbers, the chosen hyperparameters, and the three result cuts
against the naive baselines. The artifacts themselves are gitignored, so
without this the evidence would live on one laptop only.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## What this plan deliberately does not do

- **Open December 2025.** It stays locked until XGBoost and the LSTM exist and
  a winner has been chosen. No task here reads it.
- **Train XGBoost or the LSTM.** They consume `walk_forward.py` unchanged, in
  their own plans.
- **Compute SHAP.** Winner only, after the comparison.
- **Build per-segment specialist models.** Recorded in the spec as a possible
  follow-up, kept out so the three-model comparison stays like-for-like.
