# LSTM Modeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a 0.9-quantile LSTM as the third candidate model, scored through the existing walk-forward runner on rows identical to those the Random Forest and XGBoost saw.

**Architecture:** A new `utils/sequence_windows.py` turns the panel into one contiguous `float32` matrix plus an array of window end-positions, so 28-day windows are gathered per batch instead of being materialised (the dense tensor is 9.42 GB on a 16 GB machine). A new `utils/model_lstm.py` holds the network, the pinball loss, the two-fit training protocol, and the `bind_panel()` adapter that gives `model_common.run_search()` a callable of the signature it expects. `utils/walk_forward.py` is not touched.

**Tech Stack:** Python 3.9.6, PyTorch 2.8.0 (CPU/MPS, arm64), pandas 2.3.3, numpy 2.0.2, unittest.

## Global Constraints

Copied from `docs/superpowers/specs/2026-08-19-lstm-modeling-design.md`. Every task's requirements implicitly include this section.

- **Spec:** `docs/superpowers/specs/2026-08-19-lstm-modeling-design.md`. Read it before starting.
- **Never touch `utils/walk_forward.py`.** The LSTM enters through one injected callable.
- **Never touch `test/test_model_xgboost.py` or `test/test_model_random_forest.py`.** They must stay green with no assertion edited — that is the regression test for Task 1.
- **December 2025 is locked.** No row dated on or after `2025-12-01` may enter any fit, any window, or any reported metric.
- **Feature set is frozen:** `modeling_prep.FEATURE_COLS`, all 56 columns. Do not add, remove, or reorder.
- **`model_input.parquet` is already imputed.** Never call `modeling_prep.impute_features()` on it again.
- Quantile is **0.9**, uniform across every SKU. Selection criterion is **pooled pinball@0.9**, weighted by row count.
- Search folds are **3 and 5**, seed **42**. Not negotiable — folds 1, 2, 4 must stay untouched by model selection for all three models.
- Run everything as a module from the repo root: `.venv/bin/python3 -m utils.<name>`; the package uses relative imports.
- Tests: `unittest`, files `test/test_*.py`, small synthetic frames, never the real parquet.
- Run one module's tests: `.venv/bin/python3 -m unittest test.test_<name> -v`
- Run all tests: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v`
- Code comments and docstrings explain **why**, following the existing `utils/` style. Error messages are in Indonesian, matching `model_xgboost.py`.
- New dependency, exactly one line in `requirements.txt`: `torch==2.8.0`.

---

### Task 1: Move `split_early_stopping()` into `model_common`

**Files:**
- Modify: `utils/model_common.py` (add the function and its imports)
- Modify: `utils/model_xgboost.py:68-97` (delete the body, re-export the name)
- Test: `test/test_model_common.py`
- Untouched, must stay green: `test/test_model_xgboost.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `model_common.split_early_stopping(train: pd.DataFrame, tail_days: int = 30, date_col: str = "Tanggal") -> tuple[pd.DataFrame, pd.DataFrame]` returning `(fit_rows, es_rows)`. Task 6 and Task 7 call it.

- [ ] **Step 1: Write the failing test**

Append to `test/test_model_common.py`:

```python
class TestSplitEarlyStopping(unittest.TestCase):
    """The mechanism moved here from model_xgboost because it is not
    XGBoost-specific: any model that must choose its own capacity without
    reading the validation fold needs a purged training tail.
    """

    def _dated_frame(self, n=200, lead_time=3.0):
        rng = np.random.default_rng(11)
        return pd.DataFrame({
            "Tanggal": pd.date_range("2025-01-01", periods=n, freq="D"),
            "feat_a": rng.normal(size=n),
            "target_lead_time_cumulative": np.abs(rng.normal(size=n)) * 10,
            "lead_time_days": lead_time,
            "Kode Barang": "FGS-00001",
            "Nama Cabang": "KY001",
            "segment_id": 1,
        })

    def test_the_tail_is_the_last_thirty_days(self):
        train = self._dated_frame()
        _, es_rows = model_common.split_early_stopping(train, tail_days=30)
        self.assertEqual(len(es_rows), 30)
        self.assertEqual(es_rows["Tanggal"].max(), train["Tanggal"].max())

    def test_no_es_date_precedes_a_fit_date(self):
        fit_rows, es_rows = model_common.split_early_stopping(self._dated_frame())
        self.assertLess(fit_rows["Tanggal"].max(), es_rows["Tanggal"].min())

    def test_fit_rows_whose_label_window_crosses_the_tail_are_purged(self):
        # lead_time_days=3 means the label at H sums H+1..H+3, so the last
        # three days before the tail carry a label built inside the tail.
        train = self._dated_frame(lead_time=3.0)
        fit_rows, es_rows = model_common.split_early_stopping(train, tail_days=30)
        es_start = es_rows["Tanggal"].min()
        self.assertLessEqual(fit_rows["Tanggal"].max(),
                             es_start - pd.Timedelta(days=4))

    def test_an_empty_training_frame_raises(self):
        with self.assertRaises(ValueError):
            model_common.split_early_stopping(self._dated_frame().iloc[0:0])

    def test_a_window_too_short_for_the_tail_raises(self):
        with self.assertRaises(ValueError):
            model_common.split_early_stopping(self._dated_frame(n=20), tail_days=30)
```

Make sure `test/test_model_common.py` imports what these need. Check the top of the file; add any of these that are missing:

```python
import numpy as np
import pandas as pd

from utils import model_common
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m unittest test.test_model_common -v`
Expected: FAIL — `AttributeError: module 'utils.model_common' has no attribute 'split_early_stopping'`

- [ ] **Step 3: Move the function**

In `utils/model_common.py`, make sure these imports exist near the top (add only what is missing):

```python
import pandas as pd

from . import modeling_prep, purging
```

Then add, after `assert_no_nan()`:

```python
ES_TAIL_DAYS = 30


def split_early_stopping(
    train: pd.DataFrame,
    tail_days: int = ES_TAIL_DAYS,
    date_col: str = modeling_prep.DATE_COL,
) -> tuple:
    """Split a fold's training rows into fit rows and an early-stopping tail.

    The tail is the last `tail_days` calendar days. The purge on the fit side
    is not extra caution: `target_lead_time_cumulative` sums over
    H+1..H+lead_time_days, so a fit row dated within `lead_time_days` of the
    tail carries a label built partly out of the early-stopping window. Without
    the purge, early stopping would be reading a signal it had partly trained
    on and would stop too late — the identical leak `fold_train_mask()`
    prevents at the fold boundary, one scale down.

    Lives here rather than in one model's module because two models need it:
    XGBoost chooses a boosting-round count this way, the LSTM chooses an epoch
    count. Neither mechanism is specific to its model family.
    """
    if train.empty:
        raise ValueError("frame training kosong")

    es_start = train[date_col].max() - pd.Timedelta(days=tail_days - 1)
    es_rows = train[train[date_col] >= es_start]
    fit_rows = train[train[date_col] < es_start]
    fit_rows = fit_rows[purging.lookahead_safe_mask(fit_rows, es_start,
                                                    date_col=date_col)]

    if fit_rows.empty:
        raise ValueError(
            f"jendela training terlalu pendek untuk tail {tail_days} hari: "
            f"tidak ada baris tersisa untuk fit"
        )
    return fit_rows, es_rows
```

In `utils/model_xgboost.py`, delete the whole `split_early_stopping` function body (currently lines 68-97) and replace it with a re-export placed at the same spot:

```python
# Moved to model_common: the LSTM chooses its epoch count the same way.
# Re-exported so test/test_model_xgboost.py and notebook/modeling_xgb.ipynb
# keep working with no line changed.
split_early_stopping = model_common.split_early_stopping
```

Leave `ES_TAIL_DAYS = 30` in `model_xgboost.py` where it is — it is that module's default and other functions reference it.

- [ ] **Step 4: Run both test modules to verify**

Run: `.venv/bin/python3 -m unittest test.test_model_common test.test_model_xgboost -v`
Expected: PASS, all tests in both modules. `test_model_xgboost.py` must pass **without having been edited** — if it fails, the move changed behaviour and must be corrected, not the test.

- [ ] **Step 5: Commit**

```bash
git add utils/model_common.py utils/model_xgboost.py test/test_model_common.py
git commit -m "refactor: move split_early_stopping into model_common

The LSTM chooses its epoch count the same way XGBoost chooses its round
count, so the purged training tail is not XGBoost-specific.
model_xgboost re-exports the name; its test suite passes unedited."
```

---

### Task 2: `sequence_windows.build_index()` — the panel index and its guards

**Files:**
- Create: `utils/sequence_windows.py`
- Test: `test/test_sequence_windows.py`

**Interfaces:**
- Consumes: `modeling_prep.FEATURE_COLS`, `modeling_prep.LOOKBACK` (28), `modeling_prep.DATE_COL`, `modeling_prep.TARGET_COL`, `modeling_prep.SEGMENT_COLS`, `model_common.IDX_COLS`.
- Produces: `sequence_windows.build_index(panel, feature_cols=None, lookback=28, date_col="Tanggal", pair_cols=None) -> dict` with keys `values` (float32 `(N, 49)`), `cats` (int16 `(N, 7)`), `dates` (datetime64[D] `(N,)`), `positions` (int64 `(N,)`), `segment_code` (int64 `(N,)`), `lookup` (`pd.Series` mapping key → row position), `key_cols` (list), `feature_cols`, `dynamic_cols`, `idx_cols` (lists), `lookback` (int). Tasks 3, 6 and 7 consume it.

- [ ] **Step 1: Write the failing test**

Create `test/test_sequence_windows.py`:

```python
import unittest

import numpy as np
import pandas as pd

from utils import sequence_windows


def _panel(n_days=60, n_pairs=2, start="2025-01-01", seed=5):
    """A dense daily panel for a few pairs, shaped like model_input.parquet.

    Only the columns the windowing code reads are present; feature_cols is
    passed explicitly everywhere so the fixtures stay small.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for pair in range(n_pairs):
        parts.append(pd.DataFrame({
            "Tanggal": pd.date_range(start, periods=n_days, freq="D"),
            "Kode Barang": f"FGS-0000{pair}",
            "Nama Cabang": "KY001",
            "segment_id": 1,
            "feat_a": rng.normal(size=n_days).astype("float64"),
            "feat_b": rng.normal(size=n_days).astype("float64"),
            "cat_idx": rng.integers(0, 3, size=n_days),
            "target_lead_time_cumulative": np.abs(rng.normal(size=n_days)) * 10,
        }))
    return pd.concat(parts, ignore_index=True)


FEATURES = ["feat_a", "feat_b", "cat_idx"]


class TestBuildIndex(unittest.TestCase):
    def test_dynamic_and_categorical_columns_are_separated(self):
        index = sequence_windows.build_index(_panel(), feature_cols=FEATURES,
                                             lookback=7)
        self.assertEqual(index["dynamic_cols"], ["feat_a", "feat_b"])
        self.assertEqual(index["idx_cols"], ["cat_idx"])

    def test_values_are_contiguous_float32_and_cats_are_int16(self):
        index = sequence_windows.build_index(_panel(), feature_cols=FEATURES,
                                             lookback=7)
        self.assertEqual(index["values"].dtype, np.dtype("float32"))
        self.assertTrue(index["values"].flags["C_CONTIGUOUS"])
        self.assertEqual(index["cats"].dtype, np.dtype("int16"))
        self.assertEqual(index["values"].shape, (120, 2))
        self.assertEqual(index["cats"].shape, (120, 1))

    def test_rows_are_sorted_by_segment_then_date(self):
        panel = _panel().sample(frac=1.0, random_state=0).reset_index(drop=True)
        index = sequence_windows.build_index(panel, feature_cols=FEATURES,
                                             lookback=7)
        # positions restart at zero for each segment and never decrease within one
        self.assertEqual(index["positions"][0], 0)
        self.assertEqual(index["positions"].max(), 59)
        self.assertEqual(int((index["positions"] == 0).sum()), 2)

    def test_a_target_column_among_the_dynamic_columns_raises(self):
        """G2. The target must never become a window channel."""
        with self.assertRaises(ValueError) as caught:
            sequence_windows.build_index(
                _panel(),
                feature_cols=FEATURES + ["target_lead_time_cumulative"],
                lookback=7,
            )
        self.assertIn("target_lead_time_cumulative", str(caught.exception))

    def test_a_date_gap_inside_a_segment_raises(self):
        """G1. Window positions are only date arithmetic if the panel is dense."""
        panel = _panel()
        panel = panel.drop(index=30).reset_index(drop=True)
        with self.assertRaises(ValueError) as caught:
            sequence_windows.build_index(panel, feature_cols=FEATURES, lookback=7)
        self.assertIn("celah tanggal", str(caught.exception))

    def test_a_window_may_not_cross_a_segment_boundary(self):
        """G1. Two segments of the same pair are separated by a closure."""
        panel = _panel(n_days=40, n_pairs=1)
        second = panel.copy()
        second["segment_id"] = 2
        second["Tanggal"] = second["Tanggal"] + pd.Timedelta(days=100)
        both = pd.concat([panel, second], ignore_index=True)
        index = sequence_windows.build_index(both, feature_cols=FEATURES,
                                             lookback=7)
        # positions restart at 0 for the second segment, so no window built
        # from position >= lookback can reach back across the gap.
        self.assertEqual(int((index["positions"] == 0).sum()), 2)

    def test_the_lookup_maps_a_key_back_to_its_row_position(self):
        panel = _panel()
        index = sequence_windows.build_index(panel, feature_cols=FEATURES,
                                             lookback=7)
        key = ("FGS-00001", "KY001", 1, pd.Timestamp("2025-01-10"))
        position = index["lookup"].loc[key]
        self.assertEqual(index["dates"][position],
                         np.datetime64("2025-01-10", "D"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m unittest test.test_sequence_windows -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.sequence_windows'`

- [ ] **Step 3: Write the implementation**

Create `utils/sequence_windows.py`:

```python
"""Sliding 28-day windows over the panel, without ever materialising them.

The dense tensor this replaces is 1,502,522 x 28 x 56 float32 = 9.42 GB, on a
16 GB machine. The contiguous form below is 294 MB, and every window is a
`sliding_window_view` slice of it — nothing is copied until a batch is built.

Measured on `model_input.parquet` 2026-08-19: the panel has **zero date gaps
inside a segment**. That is what makes position arithmetic equal to date
arithmetic, so no dates are read at batch time. `build_index` re-checks the
property rather than trusting it, because if it ever stops holding, every
window silently spans the wrong days.

This module knows about memory and indices. It knows nothing about LSTMs.
"""

from typing import Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import modeling_prep


def _split_columns(feature_cols: list) -> tuple:
    idx_cols = [col for col in feature_cols if col.endswith("_idx")]
    dynamic_cols = [col for col in feature_cols if not col.endswith("_idx")]
    return dynamic_cols, idx_cols


def build_index(
    panel: pd.DataFrame,
    feature_cols: Optional[list] = None,
    lookback: int = modeling_prep.LOOKBACK,
    date_col: str = modeling_prep.DATE_COL,
    pair_cols: Optional[list] = None,
) -> dict:
    """One contiguous view of the whole panel, plus everything needed to
    address windows inside it.

    The panel passed here is the **full** frame, not `eligible_rows()`. A
    window for a 1 July validation row reaches back into June, over rows that
    the 28-day warm-up cut and the fold purge both remove — rows that appear
    in neither frame `walk_forward.run_fold()` hands a model. Reading their
    *features* is safe: every window ends at its own prediction row, and every
    lag and rolling feature stops at H-1, so no target value can enter a
    window. Purging protects against training on those rows' labels, which
    still never happens.
    """
    feature_cols = list(feature_cols or modeling_prep.FEATURE_COLS)
    dynamic_cols, idx_cols = _split_columns(feature_cols)

    # G2. A target channel inside a window would be a perfect predictor of
    # itself and would not change a single tensor shape.
    if modeling_prep.TARGET_COL in dynamic_cols:
        raise ValueError(
            f"{modeling_prep.TARGET_COL} tidak boleh menjadi kolom dinamis"
        )

    pair_cols = modeling_prep._resolve_pair_cols(panel, pair_cols)
    frame = panel.sort_values(pair_cols + [date_col]).reset_index(drop=True)

    values = np.ascontiguousarray(frame[dynamic_cols].to_numpy(dtype="float32"))
    cats = np.ascontiguousarray(frame[idx_cols].to_numpy(dtype="int16"))
    dates = frame[date_col].to_numpy("datetime64[D]")

    grouped = frame.groupby(pair_cols, observed=True, sort=False)
    positions = grouped.cumcount().to_numpy()
    segment_code = grouped.ngroup().to_numpy()

    _assert_dense(dates, positions, segment_code)
    _assert_windows_fit(dates, positions, segment_code, lookback)

    key_cols = list(pair_cols) + [date_col]
    lookup = pd.Series(
        np.arange(len(frame), dtype=np.int64),
        index=pd.MultiIndex.from_frame(frame[key_cols]),
    )

    return {
        "values": values,
        "cats": cats,
        "dates": dates,
        "positions": positions,
        "segment_code": segment_code,
        "lookup": lookup,
        "key_cols": key_cols,
        "feature_cols": feature_cols,
        "dynamic_cols": dynamic_cols,
        "idx_cols": idx_cols,
        "lookback": lookback,
    }


def _assert_dense(dates, positions, segment_code) -> None:
    """G1, first half: consecutive positions are consecutive days."""
    inside = positions > 0
    if not inside.any():
        return
    step = (dates[1:] - dates[:-1]).astype("timedelta64[D]").astype(np.int64)
    same_segment = segment_code[1:] == segment_code[:-1]
    bad = same_segment & (step != 1)
    if bad.any():
        first = int(np.flatnonzero(bad)[0]) + 1
        raise ValueError(
            f"celah tanggal di dalam segmen pada posisi {first} "
            f"({dates[first - 1]} -> {dates[first]}); "
            "aritmetika posisi tidak lagi sama dengan aritmetika tanggal"
        )


def _assert_windows_fit(dates, positions, segment_code, lookback) -> None:
    """G1 second half and G6: every usable window stays inside one segment,
    spans exactly `lookback` days, and never reaches past its own row.
    """
    ends = np.flatnonzero(positions >= lookback)
    if ends.size == 0:
        return
    starts = ends - lookback + 1
    if not np.array_equal(segment_code[starts], segment_code[ends]):
        raise ValueError("ada window yang melintasi batas segmen")
    span = (dates[ends] - dates[starts]).astype("timedelta64[D]").astype(np.int64)
    if not np.all(span == lookback - 1):
        raise ValueError(
            f"ada window yang tidak mencakup tepat {lookback} hari berurutan"
        )
```

Note `_resolve_pair_cols` is `modeling_prep`'s own private helper; calling it from inside the package is intentional so segment handling stays defined in exactly one place.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m unittest test.test_sequence_windows -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add utils/sequence_windows.py test/test_sequence_windows.py
git commit -m "feat: add sequence_windows.build_index with its density guards

The dense 28-day tensor is 9.42 GB on a 16 GB machine. This indexes the
panel instead: one contiguous 294 MB float32 matrix plus positions, with
assertions that consecutive array positions really are consecutive days
and that no window crosses a segment boundary."
```

---

### Task 3: `window_ends()` and `gather()` — addressing and batching

**Files:**
- Modify: `utils/sequence_windows.py` (append)
- Test: `test/test_sequence_windows.py` (append)

**Interfaces:**
- Consumes: `build_index()`'s dict from Task 2.
- Produces:
  - `sequence_windows.window_ends(index: dict, frame: pd.DataFrame) -> np.ndarray` — int64 row positions, **in the frame's own row order**.
  - `sequence_windows.gather(values: np.ndarray, ends: np.ndarray, lookback: int = 28) -> np.ndarray` — `(B, lookback, n_features)` float32. Takes the array, not the index, so a scaled copy can be passed.
  - Task 6 calls both.

- [ ] **Step 1: Write the failing test**

Append to `test/test_sequence_windows.py`:

```python
class TestWindowEnds(unittest.TestCase):
    def test_ends_follow_the_frames_own_row_order(self):
        panel = _panel()
        index = sequence_windows.build_index(panel, feature_cols=FEATURES,
                                             lookback=7)
        frame = panel.iloc[[70, 12, 95]].copy()
        ends = sequence_windows.window_ends(index, frame)
        self.assertEqual(len(ends), 3)
        self.assertEqual(
            [index["dates"][e] for e in ends],
            [np.datetime64(d, "D") for d in frame["Tanggal"]],
        )

    def test_a_row_absent_from_the_panel_raises(self):
        panel = _panel()
        index = sequence_windows.build_index(panel, feature_cols=FEATURES,
                                             lookback=7)
        frame = panel.iloc[[0]].copy()
        frame["Tanggal"] = pd.Timestamp("2030-01-01")
        with self.assertRaises(ValueError) as caught:
            sequence_windows.window_ends(index, frame)
        self.assertIn("tidak ditemukan", str(caught.exception))


class TestGather(unittest.TestCase):
    def test_a_window_is_the_lookback_rows_ending_at_the_end_position(self):
        panel = _panel()
        index = sequence_windows.build_index(panel, feature_cols=FEATURES,
                                             lookback=7)
        batch = sequence_windows.gather(index["values"], np.array([10]),
                                        lookback=7)
        self.assertEqual(batch.shape, (1, 7, 2))
        np.testing.assert_allclose(batch[0], index["values"][4:11])

    def test_gather_is_correct_for_a_shuffled_batch(self):
        panel = _panel()
        index = sequence_windows.build_index(panel, feature_cols=FEATURES,
                                             lookback=7)
        ends = np.array([40, 9, 77, 15])
        batch = sequence_windows.gather(index["values"], ends, lookback=7)
        for position, end in enumerate(ends):
            np.testing.assert_allclose(batch[position],
                                       index["values"][end - 6:end + 1])

    def test_the_fast_path_matches_to_sequences_window_for_window(self):
        """The unreadable path is verified by the readable one.

        modeling_prep.to_sequences() is the reference implementation; it is
        unusable at panel scale but exactly right at fixture scale.
        """
        panel = _panel(n_days=60, n_pairs=2)
        panel["fold_id"] = np.nan
        dynamic = ["feat_a", "feat_b"]

        reference = modeling_prep.to_sequences(
            panel, feature_cols=dynamic, lookback=7,
            pair_cols=["Kode Barang", "Nama Cabang", "segment_id"],
        )

        index = sequence_windows.build_index(panel, feature_cols=FEATURES,
                                             lookback=7)
        eligible = modeling_prep.drop_warmup_rows(panel, lookback=7)
        eligible = eligible[eligible["target_lead_time_cumulative"].notna()]
        ends = sequence_windows.window_ends(index, eligible)
        fast = sequence_windows.gather(index["values"], ends, lookback=7)

        self.assertEqual(fast.shape, reference["X"].shape)
        np.testing.assert_allclose(fast, reference["X"], rtol=1e-6)

    def test_cats_are_read_at_the_prediction_row_not_across_the_window(self):
        """Kategori Barang_idx changes inside 301 real segments, so a
        per-segment categorical array would be wrong there and silent.
        """
        panel = _panel(n_days=40, n_pairs=1)
        panel.loc[panel.index >= 20, "cat_idx"] = 2
        panel.loc[panel.index < 20, "cat_idx"] = 1
        index = sequence_windows.build_index(panel, feature_cols=FEATURES,
                                             lookback=7)
        self.assertEqual(int(index["cats"][19, 0]), 1)
        self.assertEqual(int(index["cats"][20, 0]), 2)
```

Add `modeling_prep` to the test module's imports:

```python
from utils import modeling_prep, sequence_windows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m unittest test.test_sequence_windows -v`
Expected: FAIL — `AttributeError: module 'utils.sequence_windows' has no attribute 'window_ends'`

- [ ] **Step 3: Write the implementation**

Append to `utils/sequence_windows.py`:

```python
def window_ends(index: dict, frame: pd.DataFrame) -> np.ndarray:
    """Row positions in `index["values"]` for a frame of prediction rows.

    Returned in the frame's **own row order**, so a caller can line
    predictions up against `valid.index` without a join — which is what
    `walk_forward.run_fold()` assumes when it wraps the array in a Series.
    """
    key = pd.MultiIndex.from_frame(frame[index["key_cols"]])
    ends = index["lookup"].reindex(key).to_numpy()
    missing = pd.isna(ends)
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} baris tidak ditemukan di panel; "
            "frame prediksi harus berasal dari panel yang sama dengan indeks"
        )
    return ends.astype(np.int64)


def gather(
    values: np.ndarray,
    ends: np.ndarray,
    lookback: int = modeling_prep.LOOKBACK,
) -> np.ndarray:
    """`(len(ends), lookback, n_features)` — the window ending at each position.

    `values` is taken as an argument rather than read from the index so a
    per-fold scaled copy can be passed without rebuilding anything.

    `sliding_window_view` costs nothing: it is a strided view over `values`.
    The only allocation is the batch itself, produced by the fancy index.
    """
    if len(ends) == 0:
        return np.empty((0, lookback, values.shape[1]), dtype="float32")
    if ends.min() < lookback - 1:
        raise ValueError(
            f"posisi akhir {int(ends.min())} terlalu awal untuk window "
            f"{lookback} hari"
        )
    windows = sliding_window_view(values, lookback, axis=0)
    return np.ascontiguousarray(
        windows[ends - lookback + 1].transpose(0, 2, 1)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m unittest test.test_sequence_windows -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Verify the fast path against the real panel once**

Run:

```bash
.venv/bin/python3 -c "
import numpy as np, pandas as pd
from utils import sequence_windows, walk_forward
panel = pd.read_parquet('dataset/model_ready/model_input.parquet')
index = sequence_windows.build_index(panel)
print('values', index['values'].shape, index['values'].nbytes/1e6, 'MB')
print('cats  ', index['cats'].shape, index['cats'].nbytes/1e6, 'MB')
ends = sequence_windows.window_ends(index, walk_forward.eligible_rows(panel))
print('ends  ', ends.shape)
batch = sequence_windows.gather(index['values'], ends[:4096])
print('batch ', batch.shape, batch.dtype)
"
```

Expected: `values (1502522, 49) 294.4... MB`, `cats (1502522, 7) 21.0... MB`, `ends (1145...,)` (any figure near 1.1-1.2M), `batch (4096, 28, 49) float32`. No exception means G1, G2 and G6 all hold on the real data.

- [ ] **Step 6: Commit**

```bash
git add utils/sequence_windows.py test/test_sequence_windows.py
git commit -m "feat: add window_ends and gather to sequence_windows

gather() is verified against modeling_prep.to_sequences() window for
window at fixture scale, so the fast path is checked by the readable
path rather than by conviction."
```

---

### Task 4: The network, the loss, and the embedding sizes

**Files:**
- Create: `utils/model_lstm.py`
- Modify: `requirements.txt` (add `torch==2.8.0`)
- Test: `test/test_model_lstm.py`

**Interfaces:**
- Consumes: `sequence_windows.build_index()`, `model_common.IDX_COLS`, `modeling_prep.load_category_mapping()`.
- Produces:
  - `model_lstm.QUANTILE = 0.9`, `ES_TAIL_DAYS = 30`, `EARLY_STOPPING_EPOCHS = 5`, `MAX_EPOCHS = 100`, `BUDGET_SECONDS = 28_800`, `MIN_CANDIDATES = 6`, `MAX_CANDIDATES = 20`, `DEFAULT_PARAMS`, `SEARCH_SPACE`
  - `model_lstm.pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantile: float = 0.9) -> torch.Tensor`
  - `model_lstm.embedding_sizes(mapping: dict = None, idx_cols: list = None) -> list[tuple[int, int]]`
  - `model_lstm.QuantileLSTM(n_dynamic: int, sizes: list, hidden_size: int, num_layers: int, dropout: float)` with `forward(x_dynamic, x_cats) -> torch.Tensor` of shape `(B,)`
  - `model_lstm.build_model(params: dict, n_dynamic: int, sizes: list, seed: int) -> QuantileLSTM`
  - `model_lstm.resolve_device(name: str = "cpu") -> torch.device`
  - `model_lstm.candidate_budget(sec_per_epoch: float, best_epoch: int, ...) -> int`
  - Tasks 5, 6, 7 consume all of these.

- [ ] **Step 1: Install the dependency**

Add the line `torch==2.8.0` to `requirements.txt`, then:

```bash
.venv/bin/pip install torch==2.8.0
.venv/bin/python3 -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Expected: `2.8.0 True` (or `False` for MPS on a non-Apple-Silicon machine — either is fine, CPU is the default).

- [ ] **Step 2: Write the failing test**

Create `test/test_model_lstm.py`:

```python
import unittest

import numpy as np
import torch

from utils import model_lstm


class TestPinballLoss(unittest.TestCase):
    def test_under_prediction_is_penalised_nine_times_harder(self):
        """alpha=0.9 means a shortfall costs 0.9 per unit and an overstock
        0.1 per unit — the asymmetry the whole project is built on.
        """
        target = torch.tensor([10.0])
        under = model_lstm.pinball_loss(torch.tensor([9.0]), target, 0.9)
        over = model_lstm.pinball_loss(torch.tensor([11.0]), target, 0.9)
        self.assertAlmostEqual(float(under), 0.9, places=5)
        self.assertAlmostEqual(float(over), 0.1, places=5)

    def test_a_perfect_prediction_costs_nothing(self):
        loss = model_lstm.pinball_loss(torch.tensor([5.0, 7.0]),
                                       torch.tensor([5.0, 7.0]), 0.9)
        self.assertAlmostEqual(float(loss), 0.0, places=6)


class TestEmbeddingSizes(unittest.TestCase):
    def test_sizes_come_from_the_mapping_not_from_observed_values(self):
        """num_embeddings must cover every index the saved mapping can emit,
        including UNKNOWN=0. A branch that opens after training would
        otherwise index out of bounds months later.
        """
        mapping = {
            "Kode Barang": {"<UNKNOWN>": 0, "A": 1, "B": 2},
            "Nama Cabang": {"<UNKNOWN>": 0, "KY001": 1},
        }
        sizes = model_lstm.embedding_sizes(
            mapping, idx_cols=["Kode Barang_idx", "Nama Cabang_idx"])
        self.assertEqual(sizes, [(3, 2), (2, 1)])

    def test_the_dimension_is_capped_at_sixteen(self):
        mapping = {"Kode Barang": {str(i): i for i in range(200)}}
        sizes = model_lstm.embedding_sizes(mapping, idx_cols=["Kode Barang_idx"])
        self.assertEqual(sizes, [(200, 16)])


class TestQuantileLSTM(unittest.TestCase):
    def _model(self, num_layers=2, dropout=0.2):
        return model_lstm.QuantileLSTM(
            n_dynamic=4, sizes=[(5, 3), (3, 2)],
            hidden_size=8, num_layers=num_layers, dropout=dropout)

    def test_forward_returns_one_value_per_row(self):
        model = self._model()
        x = torch.randn(6, 28, 4)
        c = torch.zeros(6, 2, dtype=torch.long)
        self.assertEqual(model(x, c).shape, (6,))

    def test_a_single_layer_model_still_applies_dropout_in_the_head(self):
        """nn.LSTM ignores dropout when num_layers=1, so the flag would be
        meaningless across half the search space if the head did not use it.
        """
        model = self._model(num_layers=1, dropout=0.5)
        self.assertEqual(model.lstm.dropout, 0.0)
        self.assertTrue(any(isinstance(layer, torch.nn.Dropout)
                            for layer in model.head))

    def test_the_highest_category_index_is_in_range(self):
        model = self._model()
        x = torch.randn(2, 28, 4)
        c = torch.tensor([[4, 2], [0, 0]], dtype=torch.long)
        self.assertEqual(model(x, c).shape, (2,))


class TestBuildModel(unittest.TestCase):
    def test_the_same_seed_produces_identical_initial_weights(self):
        params = {**model_lstm.DEFAULT_PARAMS, "hidden_size": 8, "num_layers": 1}
        a = model_lstm.build_model(params, n_dynamic=4, sizes=[(5, 3)], seed=42)
        b = model_lstm.build_model(params, n_dynamic=4, sizes=[(5, 3)], seed=42)
        for left, right in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(left, right)


class TestCandidateBudget(unittest.TestCase):
    def test_a_cheap_configuration_is_capped_at_twenty(self):
        self.assertEqual(
            model_lstm.candidate_budget(sec_per_epoch=1.0, best_epoch=2), 20)

    def test_the_formula_divides_the_budget_by_two_folds_of_two_fits(self):
        # per_fit = 60 * (2*10 + 5) = 1500s; two folds = 3000s; 28800/3000 = 9
        self.assertEqual(
            model_lstm.candidate_budget(sec_per_epoch=60.0, best_epoch=10), 9)

    def test_a_configuration_too_slow_for_six_candidates_raises(self):
        """Below six draws it is not a search. Clamping N up to six would
        silently overrun the 8-hour ceiling, so the operator is told to
        shrink the space instead.
        """
        with self.assertRaises(ValueError) as caught:
            model_lstm.candidate_budget(sec_per_epoch=600.0, best_epoch=20)
        self.assertIn("perkecil ruang search", str(caught.exception))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python3 -m unittest test.test_model_lstm -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.model_lstm'`

- [ ] **Step 4: Write the implementation**

Create `utils/model_lstm.py`:

```python
"""A 0.9-quantile LSTM, the third candidate model.

See docs/superpowers/specs/2026-08-19-lstm-modeling-design.md.

The one thing that separates this model from the other two is that it reads
the 28 days themselves rather than the hand-engineered summary of them that
`lag_*` and `roll_*` provide. The feature set is identical; the amount of
information reaching the model is not, and docs/hasil-modeling-lstm.md says so
in the head-to-head section rather than leaving a reader to discover it.
"""

from typing import Optional

import numpy as np
import torch
from torch import nn

from . import model_common, modeling_prep, sequence_windows

QUANTILE = 0.9

# Same purged tail as XGBoost: the epoch count is a capacity decision, and the
# validation fold is the one place it cannot be taken from.
ES_TAIL_DAYS = 30
EARLY_STOPPING_EPOCHS = 5
MAX_EPOCHS = 100

# Wall-clock ceiling the search budget is derived from, in seconds.
BUDGET_SECONDS = 28_800
MIN_CANDIDATES = 6
MAX_CANDIDATES = 20

DEFAULT_PARAMS = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "batch_size": 1024,
    "log_target": False,
    "grad_clip": 1.0,
    "random_state": 42,
}

SEARCH_SPACE = {
    "hidden_size": [64, 128, 256],
    "num_layers": [1, 2],
    "dropout": [0.0, 0.2, 0.3],
    "learning_rate": [3e-4, 1e-3],
    "batch_size": [1024, 2048],
    "log_target": [False, True],
}


def pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float = QUANTILE,
) -> torch.Tensor:
    """The training objective *is* the selection criterion.

    The same property `reg:quantileerror` gave XGBoost: what is optimised
    during training and what is scored during evaluation are one function, so
    a model cannot win the fit and lose the metric.
    """
    difference = target - prediction
    return torch.maximum(quantile * difference,
                         (quantile - 1.0) * difference).mean()


def embedding_sizes(
    mapping: Optional[dict] = None,
    idx_cols: Optional[list] = None,
) -> list:
    """`(num_embeddings, embedding_dim)` per `_idx` column.

    `num_embeddings` comes from `category_mapping.json` — the highest index
    plus one, which already covers the reserved UNKNOWN slot at 0 — and never
    from the values that happen to appear in a fold's training rows. A branch
    opening after this model is trained maps to 0 and must stay in range; the
    alternative fails months later, in production, with an index error.
    """
    mapping = mapping if mapping is not None else modeling_prep.load_category_mapping()
    idx_cols = idx_cols or model_common.IDX_COLS
    sizes = []
    for col in idx_cols:
        source = col[: -len("_idx")]
        num_embeddings = max(mapping[source].values()) + 1
        sizes.append((num_embeddings, min(16, (num_embeddings + 1) // 2)))
    return sizes


class QuantileLSTM(nn.Module):
    """49 dynamic channels through the LSTM, 7 categoricals through embeddings.

    The categoricals are read at the prediction row, not repeated across the
    window: `Kategori Barang_idx` changes inside 301 real segments, so "the
    segment's category" is not a well-defined thing to repeat.
    """

    def __init__(
        self,
        n_dynamic: int,
        sizes: list,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(count, dim) for count, dim in sizes]
        )
        self.lstm = nn.LSTM(
            input_size=n_dynamic,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # nn.LSTM ignores this when num_layers == 1, which is why the head
            # below always applies dropout: otherwise the searched flag would
            # be meaningless across half the space.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        width = hidden_size + sum(dim for _, dim in sizes)
        self.head = nn.Sequential(
            nn.Linear(width, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_dynamic: torch.Tensor, x_cats: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x_dynamic)
        last = output[:, -1, :]
        embedded = [layer(x_cats[:, position])
                    for position, layer in enumerate(self.embeddings)]
        return self.head(torch.cat([last, *embedded], dim=1)).squeeze(1)


def build_model(params: dict, n_dynamic: int, sizes: list, seed: int) -> QuantileLSTM:
    """Seeded construction, so the two fits of the two-fit protocol start
    from identical weights and `best_epoch` means the same thing in both.
    """
    torch.manual_seed(seed)
    return QuantileLSTM(
        n_dynamic=n_dynamic,
        sizes=sizes,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    )


def resolve_device(name: str = "cpu") -> torch.device:
    """CPU by default, deliberately.

    MPS has no fused LSTM kernel and at these hidden sizes is often slower
    than CPU, so the benchmark measures both and records which one won rather
    than a default silently choosing.
    """
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS tidak tersedia di mesin ini")
    return torch.device(name)


def candidate_budget(
    sec_per_epoch: float,
    best_epoch: int,
    budget_seconds: int = BUDGET_SECONDS,
    patience: int = EARLY_STOPPING_EPOCHS,
    minimum: int = MIN_CANDIDATES,
    maximum: int = MAX_CANDIDATES,
) -> int:
    """How many candidates fit inside the wall-clock ceiling.

    One two-fit candidate costs about
    `sec_per_epoch * (2 * best_epoch + patience)`, and each is scored on two
    folds. Fold 3's training window is shorter than fold 5's, so charging both
    at fold 5's measured rate is conservative.

    Below `minimum` this raises instead of clamping upward. Clamping up would
    be a silent overrun of the ceiling, and the spec is explicit that a
    too-small N is the signal to shrink the search space — a decision for the
    operator, not for this function.
    """
    per_fit = sec_per_epoch * (2 * best_epoch + patience)
    raw = int(budget_seconds // (2 * per_fit))
    if raw < minimum:
        raise ValueError(
            f"anggaran hanya cukup untuk {raw} kandidat (<{minimum}); "
            f"perkecil ruang search atau turunkan ongkos per fit — "
            f"jangan naikkan plafon {budget_seconds}s diam-diam"
        )
    return min(raw, maximum)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python3 -m unittest test.test_model_lstm -v`
Expected: PASS, 11 tests.

- [ ] **Step 6: Commit**

```bash
git add utils/model_lstm.py test/test_model_lstm.py requirements.txt
git commit -m "feat: add the LSTM network, pinball loss and budget formula

Embedding sizes come from category_mapping.json rather than a fold's
observed values, so a branch opening after training stays in range.
candidate_budget raises below six draws instead of clamping up, because
clamping up would silently overrun the 8-hour ceiling."
```

---

### Task 5: The training loop and early stopping

**Files:**
- Modify: `utils/model_lstm.py` (append)
- Test: `test/test_model_lstm.py` (append)

**Interfaces:**
- Consumes: Task 4's `QuantileLSTM`, `build_model`, `pinball_loss`; Task 3's `sequence_windows.gather`.
- Produces:
  - `model_lstm.scale_values(values: np.ndarray, scaler: dict, dynamic_cols: list) -> np.ndarray`
  - `model_lstm.run_epoch(model, optimizer, scaled, cats, ends, targets, params, quantile, generator, device, lookback) -> float`
  - `model_lstm.predict(model, scaled, cats, ends, device, lookback, batch_size=4096) -> np.ndarray`
  - `model_lstm.fit_with_early_stopping(...) -> tuple[QuantileLSTM, int]`
  - `model_lstm.fit_epochs(...) -> QuantileLSTM`
  - Task 6 calls all of these.

- [ ] **Step 1: Write the failing test**

Append to `test/test_model_lstm.py`:

```python
from utils import sequence_windows


def _tiny_index(n=80, seed=3):
    """A one-pair panel small enough to train on in a test."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    feat = rng.normal(size=n).astype("float64")
    panel = pd.DataFrame({
        "Tanggal": pd.date_range("2025-01-01", periods=n, freq="D"),
        "Kode Barang": "FGS-00001",
        "Nama Cabang": "KY001",
        "segment_id": 1,
        "feat_a": feat,
        "feat_b": rng.normal(size=n),
        "cat_idx": rng.integers(0, 3, size=n),
        "target_lead_time_cumulative": np.abs(feat * 5 + 10),
    })
    index = sequence_windows.build_index(
        panel, feature_cols=["feat_a", "feat_b", "cat_idx"], lookback=7)
    return panel, index


class TestScaleValues(unittest.TestCase):
    def test_each_column_is_standardised_by_its_own_statistics(self):
        _, index = _tiny_index()
        scaler = {"feat_a": (1.0, 2.0), "feat_b": (0.0, 1.0)}
        scaled = model_lstm.scale_values(index["values"], scaler,
                                         index["dynamic_cols"])
        np.testing.assert_allclose(scaled[:, 0],
                                   (index["values"][:, 0] - 1.0) / 2.0,
                                   rtol=1e-6)
        np.testing.assert_allclose(scaled[:, 1], index["values"][:, 1],
                                   rtol=1e-6)
        self.assertEqual(scaled.dtype, np.dtype("float32"))


class TestTrainingLoop(unittest.TestCase):
    def _setup(self):
        panel, index = _tiny_index()
        ends = np.arange(7, len(panel))
        targets = panel["target_lead_time_cumulative"].to_numpy("float32")[ends]
        params = {**model_lstm.DEFAULT_PARAMS, "hidden_size": 8,
                  "num_layers": 1, "batch_size": 16}
        model = model_lstm.build_model(params, n_dynamic=2, sizes=[(3, 2)], seed=42)
        return index, ends, targets, params, model

    def test_one_epoch_returns_a_finite_mean_loss(self):
        index, ends, targets, params, model = self._setup()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        loss = model_lstm.run_epoch(
            model, optimizer, index["values"], index["cats"], ends, targets,
            params, quantile=0.9,
            generator=torch.Generator().manual_seed(42),
            device=torch.device("cpu"), lookback=7)
        self.assertTrue(np.isfinite(loss))

    def test_a_non_finite_loss_raises_rather_than_poisoning_the_search(self):
        index, ends, targets, params, model = self._setup()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        poisoned = targets.copy()
        poisoned[0] = np.inf
        with self.assertRaises(ValueError) as caught:
            model_lstm.run_epoch(
                model, optimizer, index["values"], index["cats"], ends, poisoned,
                params, quantile=0.9,
                generator=torch.Generator().manual_seed(42),
                device=torch.device("cpu"), lookback=7)
        self.assertIn("NaN", str(caught.exception))

    def test_predict_returns_one_value_per_end(self):
        index, ends, _, params, model = self._setup()
        prediction = model_lstm.predict(
            model, index["values"], index["cats"], ends,
            device=torch.device("cpu"), lookback=7, batch_size=16)
        self.assertEqual(prediction.shape, (len(ends),))

    def test_early_stopping_reports_an_epoch_within_the_cap(self):
        index, ends, targets, params, model = self._setup()
        fit_ends, es_ends = ends[:50], ends[50:]
        fitted, best_epoch = model_lstm.fit_with_early_stopping(
            params, index, fit_ends, targets[:50], es_ends, targets[50:],
            quantile=0.9, sizes=[(3, 2)], device=torch.device("cpu"),
            max_epochs=6, patience=2, lookback=7)
        self.assertGreaterEqual(best_epoch, 1)
        self.assertLessEqual(best_epoch, 6)
        self.assertIsInstance(fitted, model_lstm.QuantileLSTM)

    def test_fit_epochs_runs_exactly_the_requested_number_of_epochs(self):
        index, ends, targets, params, model = self._setup()
        fitted = model_lstm.fit_epochs(
            params, index, ends, targets, epochs=3, quantile=0.9,
            sizes=[(3, 2)], device=torch.device("cpu"), lookback=7)
        self.assertEqual(fitted.epochs_run, 3)

    def test_the_same_seed_produces_identical_predictions(self):
        index, ends, targets, params, _ = self._setup()
        outputs = []
        for _ in range(2):
            fitted = model_lstm.fit_epochs(
                params, index, ends, targets, epochs=2, quantile=0.9,
                sizes=[(3, 2)], device=torch.device("cpu"), lookback=7)
            outputs.append(model_lstm.predict(
                fitted, index["values"], index["cats"], ends,
                device=torch.device("cpu"), lookback=7))
        np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m unittest test.test_model_lstm -v`
Expected: FAIL — `AttributeError: module 'utils.model_lstm' has no attribute 'scale_values'`

- [ ] **Step 3: Write the implementation**

Append to `utils/model_lstm.py`:

```python
def scale_values(values: np.ndarray, scaler: dict, dynamic_cols: list) -> np.ndarray:
    """Standardise the whole panel matrix with one fold's scaler.

    The scaler is fit on that fold's training rows only. Applying it to every
    row, context rows included, is safe — what leaks is *fitting* it outside
    the training window, never applying it.
    """
    mean = np.array([scaler[col][0] for col in dynamic_cols], dtype="float32")
    std = np.array([scaler[col][1] for col in dynamic_cols], dtype="float32")
    return ((values - mean) / std).astype("float32")


def _shuffled_batches(count: int, batch_size: int, generator) -> list:
    order = torch.randperm(count, generator=generator).numpy()
    return [order[start:start + batch_size] for start in range(0, count, batch_size)]


def _to_tensors(scaled, cats, ends, lookback, device):
    windows = sequence_windows.gather(scaled, ends, lookback=lookback)
    x_dynamic = torch.from_numpy(windows).to(device)
    x_cats = torch.from_numpy(cats[ends].astype("int64")).to(device)
    return x_dynamic, x_cats


def run_epoch(
    model: QuantileLSTM,
    optimizer,
    scaled: np.ndarray,
    cats: np.ndarray,
    ends: np.ndarray,
    targets: np.ndarray,
    params: dict,
    quantile: float,
    generator,
    device,
    lookback: int = modeling_prep.LOOKBACK,
) -> float:
    """One pass over the training windows, returning the mean loss.

    Windows are shuffled across segments. Each one is self-contained, so no
    ordering needs preserving between batches.
    """
    model.train()
    total, seen = 0.0, 0
    for batch in _shuffled_batches(len(ends), params["batch_size"], generator):
        x_dynamic, x_cats = _to_tensors(scaled, cats, ends[batch], lookback, device)
        y = torch.from_numpy(targets[batch].astype("float32")).to(device)

        optimizer.zero_grad()
        loss = pinball_loss(model(x_dynamic, x_cats), y, quantile)
        if not torch.isfinite(loss):
            # Fails this candidate through run_search's existing catch tuple.
            # RuntimeError is not raised here on purpose: PyTorch uses it for
            # genuine bugs as well as OOM, so widening the tuple would launder
            # bugs into NaN rows.
            raise ValueError(
                "loss LSTM menjadi NaN/inf — kandidat digagalkan"
            )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])
        optimizer.step()

        total += float(loss) * len(batch)
        seen += len(batch)
    return total / max(seen, 1)


@torch.no_grad()
def predict(
    model: QuantileLSTM,
    scaled: np.ndarray,
    cats: np.ndarray,
    ends: np.ndarray,
    device,
    lookback: int = modeling_prep.LOOKBACK,
    batch_size: int = 4096,
) -> np.ndarray:
    """Predictions in `ends` order, so they line up with the caller's frame."""
    model.eval()
    if len(ends) == 0:
        return np.empty(0, dtype="float32")
    parts = []
    for start in range(0, len(ends), batch_size):
        chunk = ends[start:start + batch_size]
        x_dynamic, x_cats = _to_tensors(scaled, cats, chunk, lookback, device)
        parts.append(model(x_dynamic, x_cats).cpu().numpy())
    return np.concatenate(parts)


def _evaluate(model, scaled, cats, ends, targets, quantile, device, lookback):
    prediction = predict(model, scaled, cats, ends, device=device, lookback=lookback)
    difference = targets.astype("float64") - prediction.astype("float64")
    return float(np.maximum(quantile * difference,
                            (quantile - 1.0) * difference).mean())


def fit_with_early_stopping(
    params: dict,
    index: dict,
    fit_ends: np.ndarray,
    fit_targets: np.ndarray,
    es_ends: np.ndarray,
    es_targets: np.ndarray,
    quantile: float,
    sizes: list,
    device,
    scaled: Optional[np.ndarray] = None,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_EPOCHS,
    lookback: int = modeling_prep.LOOKBACK,
) -> tuple:
    """Fit on the purged rows, stop on the tail, report the epoch that won.

    Under `log_target` the stopping metric is computed on the log scale. That
    is sound: early stopping only chooses an epoch count *within* one
    candidate. Candidates are compared to each other by pinball on the
    original scale, after inversion.
    """
    scaled = index["values"] if scaled is None else scaled
    model = build_model(params, len(index["dynamic_cols"]), sizes,
                        params["random_state"])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    generator = torch.Generator().manual_seed(params["random_state"])

    best_score, best_epoch, since_improvement = float("inf"), 1, 0
    for epoch in range(1, max_epochs + 1):
        run_epoch(model, optimizer, scaled, index["cats"], fit_ends, fit_targets,
                  params, quantile, generator, device, lookback)
        score = _evaluate(model, scaled, index["cats"], es_ends, es_targets,
                          quantile, device, lookback)
        if score < best_score:
            best_score, best_epoch, since_improvement = score, epoch, 0
        else:
            since_improvement += 1
            if since_improvement >= patience:
                break
    model.best_score = best_score
    return model, best_epoch


def fit_epochs(
    params: dict,
    index: dict,
    ends: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    quantile: float,
    sizes: list,
    device,
    scaled: Optional[np.ndarray] = None,
    lookback: int = modeling_prep.LOOKBACK,
) -> QuantileLSTM:
    """The second fit: same seed, all training rows, a fixed epoch count.

    One epoch here contains about 5% more gradient steps than one epoch of the
    first fit, because the early-stopping tail is back in. That is accepted:
    pinning by epoch means "the same number of passes over the data", which is
    the more meaningful invariant than a fixed step count.
    """
    scaled = index["values"] if scaled is None else scaled
    model = build_model(params, len(index["dynamic_cols"]), sizes,
                        params["random_state"])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    generator = torch.Generator().manual_seed(params["random_state"])

    for _ in range(epochs):
        run_epoch(model, optimizer, scaled, index["cats"], ends, targets,
                  params, quantile, generator, device, lookback)
    model.epochs_run = epochs
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m unittest test.test_model_lstm -v`
Expected: PASS, 18 tests.

- [ ] **Step 5: Commit**

```bash
git add utils/model_lstm.py test/test_model_lstm.py
git commit -m "feat: add the LSTM training loop and early stopping

A non-finite loss raises ValueError, which run_search's existing catch
tuple records as a failed candidate. RuntimeError is deliberately not
caught: PyTorch uses it for genuine bugs as well as OOM."
```

---

### Task 6: `make_fit_predict()` and `bind_panel()` — the two-fit protocol and its guards

**Files:**
- Modify: `utils/model_lstm.py` (append)
- Test: `test/test_model_lstm.py` (append)

**Interfaces:**
- Consumes: Tasks 1, 3, 4, 5.
- Produces:
  - `model_lstm.make_fit_predict(params=None, index=None, feature_cols=None, quantile=0.9, tail_days=30, max_epochs=100, patience=5, device_name="cpu", sizes=None) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]`, with `.best_epochs: list[int]` and `.index` recorded on the returned callable
  - `model_lstm.bind_panel(panel, feature_cols=None, lookback=28, device_name="cpu", sizes=None) -> Callable` matching `model_common.run_search`'s expected `make_fit_predict(candidate, feature_cols=..., quantile=...)` signature

`sizes` defaults to `embedding_sizes(idx_cols=index["idx_cols"])`, which reads the real `category_mapping.json`. Tests pass it explicitly, because a synthetic `cat_idx` fixture column has no entry in that file.
  - Task 7 and the notebook call `bind_panel`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_model_lstm.py`:

```python
import pandas as pd

from utils import modeling_prep, walk_forward


def _fold_panel(n_days=200, n_pairs=2, seed=7):
    """A panel long enough to reach fold 5 and survive a 7-day warm-up."""
    rng = np.random.default_rng(seed)
    parts = []
    for pair in range(n_pairs):
        feat = rng.normal(size=n_days)
        parts.append(pd.DataFrame({
            "Tanggal": pd.date_range("2025-05-01", periods=n_days, freq="D"),
            "Kode Barang": f"FGS-0000{pair}",
            "Nama Cabang": "KY001",
            "segment_id": 1,
            "feat_a": feat,
            "feat_b": rng.normal(size=n_days),
            "cat_idx": rng.integers(0, 3, size=n_days),
            "lead_time_days": 3.0,
            "demand_segment": "smooth",
            "is_delivery_day": True,
            "target_lead_time_cumulative": np.abs(feat * 5 + 10),
        }))
    panel = pd.concat(parts, ignore_index=True)
    return modeling_prep.assign_folds(panel)


FOLD_FEATURES = ["feat_a", "feat_b", "cat_idx"]
SMALL = {**model_lstm.DEFAULT_PARAMS, "hidden_size": 8, "num_layers": 1,
         "batch_size": 64}


class TestFitPredict(unittest.TestCase):
    def _index_and_split(self):
        panel = _fold_panel()
        index = sequence_windows.build_index(panel, feature_cols=FOLD_FEATURES,
                                             lookback=7)
        frame = walk_forward.eligible_rows(panel, lookback=7)
        split = walk_forward.prepare_fold(frame, 5, prepared=True)
        return panel, index, split["train"], split["valid"]

    def _fit_predict(self, index, **kwargs):
        # sizes is passed explicitly: the fixture's `cat_idx` has no entry in
        # the real category_mapping.json that embedding_sizes() would read.
        return model_lstm.make_fit_predict(
            SMALL, index=index, quantile=0.9, tail_days=30,
            sizes=[(3, 2)],
            max_epochs=kwargs.pop("max_epochs", 3),
            patience=kwargs.pop("patience", 2), **kwargs)

    def test_it_returns_one_prediction_per_validation_row(self):
        _, index, train, valid = self._index_and_split()
        prediction = self._fit_predict(index)(train, valid)
        self.assertEqual(prediction.shape, (len(valid),))

    def test_predictions_are_never_negative(self):
        _, index, train, valid = self._index_and_split()
        prediction = self._fit_predict(index)(train, valid)
        self.assertTrue((prediction >= 0).all())

    def test_no_training_row_is_dated_inside_the_validation_month(self):
        """G3. This checks the position mapping, not fold_train_mask —
        window_ends() returning wrong positions is the new failure mode.
        """
        _, index, train, valid = self._index_and_split()
        train_ends = sequence_windows.window_ends(index, train)
        self.assertLess(index["dates"][train_ends].max(),
                        np.datetime64(valid["Tanggal"].min(), "D"))

    def test_the_early_stopping_tail_is_absent_from_the_first_fit(self):
        """G4."""
        _, index, train, _ = self._index_and_split()
        fit_rows, es_rows = model_common.split_early_stopping(train, tail_days=30)
        fit_ends = set(sequence_windows.window_ends(index, fit_rows).tolist())
        es_ends = set(sequence_windows.window_ends(index, es_rows).tolist())
        self.assertEqual(fit_ends & es_ends, set())

    def test_no_window_reaches_into_december(self):
        """G5."""
        _, index, train, valid = self._index_and_split()
        ends = np.concatenate([
            sequence_windows.window_ends(index, train),
            sequence_windows.window_ends(index, valid),
        ])
        self.assertLess(index["dates"][ends].max(),
                        np.datetime64(modeling_prep.TEST_START, "D"))

    def test_the_best_epoch_of_each_fold_is_recorded(self):
        _, index, train, valid = self._index_and_split()
        fit_predict = self._fit_predict(index)
        fit_predict(train, valid)
        self.assertEqual(len(fit_predict.best_epochs), 1)
        self.assertGreaterEqual(fit_predict.best_epochs[0], 1)

    def test_log_target_round_trips_to_the_original_scale(self):
        _, index, train, valid = self._index_and_split()
        logged = model_lstm.make_fit_predict(
            {**SMALL, "log_target": True}, index=index, quantile=0.9,
            sizes=[(3, 2)], max_epochs=2, patience=2)(train, valid)
        self.assertTrue(np.isfinite(logged).all())
        self.assertTrue((logged >= 0).all())


class TestBindPanel(unittest.TestCase):
    def test_it_matches_run_searchs_expected_signature(self):
        panel = _fold_panel()
        make = model_lstm.bind_panel(panel, feature_cols=FOLD_FEATURES,
                                     lookback=7, sizes=[(3, 2)])
        fit_predict = make(SMALL, feature_cols=FOLD_FEATURES, quantile=0.9)
        self.assertTrue(callable(fit_predict))

    def test_the_index_is_built_once_and_reused(self):
        panel = _fold_panel()
        make = model_lstm.bind_panel(panel, feature_cols=FOLD_FEATURES,
                                     lookback=7, sizes=[(3, 2)])
        first = make(SMALL, quantile=0.9)
        second = make(SMALL, quantile=0.9)
        self.assertIs(first.index, second.index)

    def test_a_different_feature_list_raises(self):
        panel = _fold_panel()
        make = model_lstm.bind_panel(panel, feature_cols=FOLD_FEATURES,
                                     lookback=7, sizes=[(3, 2)])
        with self.assertRaises(ValueError) as caught:
            make(SMALL, feature_cols=["feat_a"], quantile=0.9)
        self.assertIn("berbeda dari yang dipakai membangun indeks",
                      str(caught.exception))
```

Add `model_common` to the test module's imports:

```python
from utils import model_common, model_lstm, modeling_prep, sequence_windows, walk_forward
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m unittest test.test_model_lstm -v`
Expected: FAIL — `AttributeError: module 'utils.model_lstm' has no attribute 'make_fit_predict'`

- [ ] **Step 3: Write the implementation**

Append to `utils/model_lstm.py`:

```python
def _target(frame, params: dict) -> np.ndarray:
    values = frame[modeling_prep.TARGET_COL].to_numpy(dtype="float64")
    return (np.log1p(values) if params["log_target"] else values).astype("float32")


def _assert_no_december(index: dict, ends: np.ndarray,
                        test_start=modeling_prep.TEST_START) -> None:
    """G5. Redundant with the fold definitions, and kept anyway: the cost of
    one accidental leak is the credibility of the final number.
    """
    if len(ends) and index["dates"][ends].max() >= np.datetime64(test_start, "D"):
        raise ValueError("ada window yang menyentuh Desember 2025")


def _assert_train_precedes_valid(index: dict, train_ends: np.ndarray,
                                 valid_ends: np.ndarray) -> None:
    """G3. Checks the *position mapping*, not `fold_train_mask` — the frames
    were already split correctly by `walk_forward`, so what this can still
    catch is `window_ends()` handing back the wrong rows.
    """
    if len(train_ends) and len(valid_ends):
        if index["dates"][train_ends].max() >= index["dates"][valid_ends].min():
            raise ValueError(
                "posisi window training tidak seluruhnya mendahului validasi"
            )


def make_fit_predict(
    params: Optional[dict] = None,
    index: Optional[dict] = None,
    feature_cols: Optional[list] = None,
    quantile: float = QUANTILE,
    tail_days: int = ES_TAIL_DAYS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_EPOCHS,
    device_name: str = "cpu",
    sizes: Optional[list] = None,
) -> "object":
    """The callable `walk_forward.run_fold()` injects.

    Two fits. The first runs on the purged fit rows with the 30-day tail as
    its eval set and reports the epoch that won. The second discards that
    model, re-initialises from the same seed, and trains on **every** training
    row for exactly that many epochs — so the model producing the reported
    predictions has seen the same population the Random Forest and XGBoost
    were trained on.

    Best epochs are recorded on the returned callable rather than returned,
    because `walk_forward` accepts predictions and nothing else — and their
    spread across folds is worth reporting.
    """
    if index is None:
        raise ValueError("make_fit_predict butuh indeks dari bind_panel()")
    params = {**DEFAULT_PARAMS, **(params or {})}
    device = resolve_device(device_name)
    # From category_mapping.json unless the caller supplies its own — tests do,
    # because a synthetic _idx fixture column has no entry in that file.
    sizes = sizes if sizes is not None else embedding_sizes(
        idx_cols=index["idx_cols"])
    lookback = index["lookback"]

    def fit_predict(train, valid) -> np.ndarray:
        model_common.assert_no_nan(train, index["feature_cols"])
        model_common.assert_no_nan(valid, index["feature_cols"])

        train_ends = sequence_windows.window_ends(index, train)
        valid_ends = sequence_windows.window_ends(index, valid)
        _assert_train_precedes_valid(index, train_ends, valid_ends)
        _assert_no_december(index, np.concatenate([train_ends, valid_ends]))

        # One scaler for the whole fold, used by both fits. The tail's
        # statistics sit inside the training window so nothing leaks into
        # validation; sharing it is what makes best_epoch transfer between
        # two fits that would otherwise see differently scaled inputs.
        scaler = modeling_prep.fit_scaler(train, index["dynamic_cols"])
        scaled = scale_values(index["values"], scaler, index["dynamic_cols"])

        fit_rows, es_rows = model_common.split_early_stopping(
            train, tail_days=tail_days)
        fit_ends = sequence_windows.window_ends(index, fit_rows)
        es_ends = sequence_windows.window_ends(index, es_rows)

        _, best_epoch = fit_with_early_stopping(
            params, index, fit_ends, _target(fit_rows, params),
            es_ends, _target(es_rows, params), quantile=quantile, sizes=sizes,
            device=device, scaled=scaled, max_epochs=max_epochs,
            patience=patience, lookback=lookback)
        fit_predict.best_epochs.append(int(best_epoch))

        model = fit_epochs(
            params, index, train_ends, _target(train, params), epochs=best_epoch,
            quantile=quantile, sizes=sizes, device=device, scaled=scaled,
            lookback=lookback)

        prediction = predict(model, scaled, index["cats"], valid_ends,
                             device=device, lookback=lookback)
        prediction = np.asarray(prediction, dtype="float64")
        if params["log_target"]:
            prediction = modeling_prep.inverse_log_target(prediction)
        # A negative shipment quantity is not a thing.
        return np.clip(prediction, 0.0, None)

    fit_predict.best_epochs = []
    fit_predict.index = index
    return fit_predict


def bind_panel(
    panel,
    feature_cols: Optional[list] = None,
    lookback: int = modeling_prep.LOOKBACK,
    device_name: str = "cpu",
    tail_days: int = ES_TAIL_DAYS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_EPOCHS,
    sizes: Optional[list] = None,
):
    """Give `model_common.run_search()` a callable of the signature it expects.

    `run_search` calls `make_fit_predict(candidate, feature_cols=...,
    quantile=...)`. There is no slot for the panel, and adding one would
    change a signature the other two models already satisfy — so the panel is
    bound here instead.

    The window index is built **once**. It costs a sort of 1.5M rows;
    rebuilding it per candidate would repeat that N x 2 times for nothing.
    """
    index = sequence_windows.build_index(panel, feature_cols=feature_cols,
                                         lookback=lookback)

    def make(params=None, feature_cols=None, quantile: float = QUANTILE):
        if feature_cols is not None and list(feature_cols) != index["feature_cols"]:
            raise ValueError(
                "feature_cols berbeda dari yang dipakai membangun indeks"
            )
        return make_fit_predict(params, index=index, quantile=quantile,
                                tail_days=tail_days, max_epochs=max_epochs,
                                patience=patience, device_name=device_name,
                                sizes=sizes)

    make.index = index
    return make
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m unittest test.test_model_lstm -v`
Expected: PASS, 28 tests.

- [ ] **Step 5: Run the whole suite — nothing else may have moved**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v 2>&1 | tail -5`
Expected: `OK`, with no failures in `test_model_xgboost` or `test_model_random_forest`.

- [ ] **Step 6: Commit**

```bash
git add utils/model_lstm.py test/test_model_lstm.py
git commit -m "feat: add the LSTM two-fit fit_predict and bind_panel

bind_panel builds the window index once and hands run_search a callable
of the signature it already expects, so walk_forward stays untouched and
all three models are still scored on identical rows."
```

---

### Task 7: `fit_final()`, `predict_bundle()`, and the search wrappers

**Files:**
- Modify: `utils/model_lstm.py` (append)
- Test: `test/test_model_lstm.py` (append)

**Interfaces:**
- Consumes: Tasks 1 and 3-6.
- Produces:
  - Path constants `MODEL_FILE`, `BEST_PARAMS_FILE`, `SEARCH_FILE`, `RESULTS_FILE`
  - `model_lstm.fit_final(df, params, feature_cols=None, ...) -> dict` (the bundle)
  - `model_lstm.predict_bundle(bundle, panel, frame) -> np.ndarray`
  - `model_lstm.save_bundle/load_bundle/save_best_params`
  - `model_lstm.SEARCH_FOLDS = (3, 5)`, `model_lstm.select_best`
  - `model_lstm.sample_search_space(n_candidates, seed=42, space=None) -> list`
  - `model_lstm.run_search(df, candidates, folds=SEARCH_FOLDS, ...) -> pd.DataFrame`
  - The notebook in Task 8 calls all of these.

- [ ] **Step 1: Write the failing test**

Append to `test/test_model_lstm.py`:

```python
class TestFitFinalAndBundle(unittest.TestCase):
    def _bundle(self):
        panel = _fold_panel()
        bundle = model_lstm.fit_final(
            panel, SMALL, feature_cols=FOLD_FEATURES, lookback=7,
            sizes=[(3, 2)], max_epochs=3, patience=2)
        return panel, bundle

    def test_the_bundle_records_everything_needed_to_reload(self):
        _, bundle = self._bundle()
        for key in ("state_dict", "params", "feature_cols", "dynamic_cols",
                    "idx_cols", "embedding_sizes", "scaler", "log_target",
                    "best_epoch", "quantile", "n_train", "lookback"):
            self.assertIn(key, bundle)
        self.assertEqual(bundle["quantile"], 0.9)
        self.assertEqual(bundle["feature_cols"], FOLD_FEATURES)

    def test_no_training_row_reaches_december(self):
        """fit_final takes its rows from walk_forward.eligible_rows(), which
        cuts December before anything else — so this checks the cut survived
        the extra purge and the window mapping.
        """
        panel, bundle = self._bundle()
        self.assertGreater(bundle["n_train"], 0)
        eligible = walk_forward.eligible_rows(panel, lookback=7)
        self.assertLess(eligible["Tanggal"].max(), modeling_prep.TEST_START)
        self.assertLessEqual(bundle["n_train"], len(eligible))

    def test_predict_bundle_returns_non_negative_values_per_row(self):
        panel, bundle = self._bundle()
        frame = walk_forward.eligible_rows(panel, lookback=7).head(20)
        prediction = model_lstm.predict_bundle(bundle, panel, frame)
        self.assertEqual(prediction.shape, (20,))
        self.assertTrue((prediction >= 0).all())

    def test_a_column_shuffled_frame_produces_identical_predictions(self):
        """The bundle forces the recorded column order. A model reloaded
        against a different layout does not fail — it predicts confidently
        from the wrong features, which is worse.
        """
        panel, bundle = self._bundle()
        frame = walk_forward.eligible_rows(panel, lookback=7).head(20)
        shuffled_panel = panel[list(reversed(panel.columns))]
        straight = model_lstm.predict_bundle(bundle, panel, frame)
        shuffled = model_lstm.predict_bundle(bundle, shuffled_panel, frame)
        np.testing.assert_allclose(straight, shuffled, rtol=1e-5)


class TestSearchWrappers(unittest.TestCase):
    def test_the_same_seed_reproduces_the_identical_candidate_list(self):
        first = model_lstm.sample_search_space(5, seed=42)
        second = model_lstm.sample_search_space(5, seed=42)
        self.assertEqual(first, second)

    def test_every_candidate_carries_the_defaults_it_did_not_draw(self):
        for candidate in model_lstm.sample_search_space(5, seed=42):
            self.assertEqual(candidate["random_state"], 42)
            self.assertEqual(candidate["grad_clip"], 1.0)
            self.assertIn(candidate["hidden_size"], [64, 128, 256])

    def test_the_search_folds_match_the_other_two_models(self):
        self.assertEqual(model_lstm.SEARCH_FOLDS, (3, 5))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m unittest test.test_model_lstm -v`
Expected: FAIL — `AttributeError: module 'utils.model_lstm' has no attribute 'fit_final'`

- [ ] **Step 3: Write the implementation**

Add `from pathlib import Path` and `from . import purging, walk_forward` to the imports at the top of `utils/model_lstm.py`, then append:

```python
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = str(BASE_DIR / "models/lstm_q90.joblib")
BEST_PARAMS_FILE = str(BASE_DIR / "dataset/model_ready/lstm_best_params.json")
SEARCH_FILE = str(BASE_DIR / "dataset/model_ready/lstm_search_results.csv")
RESULTS_FILE = str(BASE_DIR / "dataset/model_ready/lstm_walk_forward_results.csv")


def fit_final(
    df,
    params: dict,
    feature_cols: Optional[list] = None,
    lookback: int = modeling_prep.LOOKBACK,
    tail_days: int = ES_TAIL_DAYS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_EPOCHS,
    quantile: float = QUANTILE,
    device_name: str = "cpu",
    sizes: Optional[list] = None,
    date_col: str = modeling_prep.DATE_COL,
    test_start=modeling_prep.TEST_START,
) -> dict:
    """Fit on every eligible row before December, purged at that boundary.

    Eligibility comes from `walk_forward.eligible_rows`, not from a date
    filter written here: the rows this model is finally trained on have to be
    the rows it was scored on, and the scoring cuts are not just the date.

    The **windows**, though, are cut from the full `df` — the same reason
    `build_index` takes the whole panel. Context rows outside the eligible set
    are still legitimate history.
    """
    params = {**DEFAULT_PARAMS, **params}
    index = sequence_windows.build_index(df, feature_cols=feature_cols,
                                         lookback=lookback, date_col=date_col)
    device = resolve_device(device_name)
    sizes = sizes if sizes is not None else embedding_sizes(
        idx_cols=index["idx_cols"])

    frame = walk_forward.eligible_rows(df, lookback=lookback, date_col=date_col,
                                       test_start=test_start)
    frame = frame[purging.lookahead_safe_mask(frame, test_start, date_col=date_col)]
    model_common.assert_no_nan(frame, index["feature_cols"])

    ends = sequence_windows.window_ends(index, frame)
    _assert_no_december(index, ends, test_start=test_start)

    scaler = modeling_prep.fit_scaler(frame, index["dynamic_cols"])
    scaled = scale_values(index["values"], scaler, index["dynamic_cols"])

    fit_rows, es_rows = model_common.split_early_stopping(
        frame, tail_days=tail_days, date_col=date_col)
    _, best_epoch = fit_with_early_stopping(
        params, index,
        sequence_windows.window_ends(index, fit_rows), _target(fit_rows, params),
        sequence_windows.window_ends(index, es_rows), _target(es_rows, params),
        quantile=quantile, sizes=sizes, device=device, scaled=scaled,
        max_epochs=max_epochs, patience=patience, lookback=lookback)

    model = fit_epochs(params, index, ends, _target(frame, params),
                       epochs=best_epoch, quantile=quantile, sizes=sizes,
                       device=device, scaled=scaled, lookback=lookback)

    return {
        "state_dict": {key: value.cpu() for key, value
                       in model.state_dict().items()},
        "params": params,
        "feature_cols": index["feature_cols"],
        "dynamic_cols": index["dynamic_cols"],
        "idx_cols": index["idx_cols"],
        "embedding_sizes": sizes,
        "scaler": scaler,
        "log_target": params["log_target"],
        "best_epoch": int(best_epoch),
        "quantile": quantile,
        "lookback": lookback,
        "n_train": int(len(frame)),
    }


def predict_bundle(bundle: dict, panel, frame) -> np.ndarray:
    """Predict with a fitted bundle, forcing the recorded column order.

    `panel` is required and not optional: an LSTM cannot predict from a row on
    its own — it needs the 28 days behind it. Rebuilding the index from
    `bundle["feature_cols"]` is what pins the column order, so a panel whose
    columns arrive in a different order produces identical predictions.
    """
    index = sequence_windows.build_index(
        panel, feature_cols=bundle["feature_cols"], lookback=bundle["lookback"])
    device = resolve_device(bundle["params"].get("device", "cpu"))
    model = build_model(bundle["params"], len(bundle["dynamic_cols"]),
                        bundle["embedding_sizes"], bundle["params"]["random_state"])
    model.load_state_dict(bundle["state_dict"])
    model.to(device)

    scaled = scale_values(index["values"], bundle["scaler"], bundle["dynamic_cols"])
    ends = sequence_windows.window_ends(index, frame)
    prediction = np.asarray(
        predict(model, scaled, index["cats"], ends, device=device,
                lookback=bundle["lookback"]),
        dtype="float64",
    )
    if bundle["log_target"]:
        prediction = modeling_prep.inverse_log_target(prediction)
    return np.clip(prediction, 0.0, None)


def save_bundle(bundle: dict, path: str = MODEL_FILE) -> None:
    model_common.save_bundle(bundle, path)


def load_bundle(path: str = MODEL_FILE) -> dict:
    return model_common.load_bundle(path)


def save_best_params(params: dict, path: str = BEST_PARAMS_FILE) -> None:
    model_common.save_best_params(params, path)


# Identical to the Random Forest and XGBoost searches, and that is the point:
# if this model searched on different folds, "folds 1, 2 and 4 are untouched
# by model selection" would collapse for all three models at once.
SEARCH_FOLDS = (3, 5)

select_best = model_common.select_best


def sample_search_space(
    n_candidates: int,
    seed: int = 42,
    space: Optional[dict] = None,
) -> list:
    """Distinct parameter sets drawn at random from SEARCH_SPACE.

    No affordability screen: unlike the quantile forest there is no
    leaf-storage bound to screen against — a batch of 2048 windows is 11 MB
    whatever the hidden size.

    `n_candidates` has no default on purpose. It comes from
    `candidate_budget()` and its measured inputs, so hard-coding one here
    would invite skipping the measurement.
    """
    return model_common.sample_search_space(
        space=SEARCH_SPACE if space is None else space,
        defaults=DEFAULT_PARAMS,
        n_candidates=n_candidates,
        seed=seed,
        screen=None,
    )


def run_search(
    df,
    candidates: list,
    folds: tuple = SEARCH_FOLDS,
    alpha: float = QUANTILE,
    model_name: str = "lstm",
    feature_cols: Optional[list] = None,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
    device_name: str = "cpu",
    lookback: int = modeling_prep.LOOKBACK,
    sizes: Optional[list] = None,
):
    """Score every LSTM candidate on the search folds.

    `df` is the panel, passed to both `run_search` (which cuts eligible rows
    from it) and `bind_panel` (which cuts windows from it) — the same frame in
    both places, so the rows scored and the rows windowed cannot drift apart.
    """
    return model_common.run_search(
        df,
        candidates,
        make_fit_predict=bind_panel(df, feature_cols=feature_cols,
                                    lookback=lookback, device_name=device_name,
                                    sizes=sizes),
        search_space=SEARCH_SPACE,
        folds=folds,
        alpha=alpha,
        model_name=model_name,
        feature_cols=feature_cols,
        verbose=verbose,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m unittest test.test_model_lstm -v`
Expected: PASS, 35 tests.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v 2>&1 | tail -5`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add utils/model_lstm.py test/test_model_lstm.py
git commit -m "feat: add LSTM fit_final, predict_bundle and search wrappers

The bundle is one joblib file matching RF and XGB. predict_bundle takes
the panel explicitly, because an LSTM cannot predict from a row without
the 28 days behind it, and rebuilds the index from the recorded
feature_cols so column order is pinned."
```

---

### Task 8: The notebook, and the benchmark that sets the search budget

**Files:**
- Create: `notebook/modeling_lstm.ipynb`
- Artifacts produced: none yet beyond the benchmark's printed numbers

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: the measured `sec_per_epoch`, `best_epoch`, device choice and peak RSS that Task 9's `N_CANDIDATES` and Task 10's results document depend on.

- [ ] **Step 1: Create the notebook skeleton**

Create `notebook/modeling_lstm.ipynb` mirroring `notebook/modeling_xgb.ipynb`'s structure. Build it with this script so the JSON is well-formed:

```bash
.venv/bin/python3 - <<'NB'
import json, pathlib

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}

cells = [
    md("# Modeling — LSTM (quantile 0.9)\n\n"
       "Desain: `docs/superpowers/specs/2026-08-19-lstm-modeling-design.md`.\n"
       "Desember 2025 tidak dibuka di notebook ini."),
    code("import sys\n"
         "sys.path.insert(0, '..')\n\n"
         "import numpy as np\n"
         "import pandas as pd\n"
         "import torch\n\n"
         "from utils import model_lstm as lstm\n"
         "from utils import modeling_prep, walk_forward\n\n"
         "df = pd.read_parquet(modeling_prep.MODEL_INPUT_FILE)\n"
         "print(df.shape, torch.__version__, torch.backends.mps.is_available())"),

    md("## Benchmark\n\n"
       "Satu putaran dua-fit di fold 5 dengan `DEFAULT_PARAMS`, di CPU dan MPS.\n"
       "Yang diukur: detik per epoch, epoch tempat early stopping mendarat, dan\n"
       "peak RSS. Ketiganya yang mengisi rumus anggaran di §2.2 spec."),
    code("import resource, time\n\n"
         "frame = walk_forward.eligible_rows(df)\n"
         "split = walk_forward.prepare_fold(frame, 5, prepared=True)\n"
         "print('train', len(split['train']), 'valid', len(split['valid']))\n\n"
         "benchmark = {}\n"
         "for device_name in ('cpu', 'mps'):\n"
         "    try:\n"
         "        make = lstm.bind_panel(df, device_name=device_name)\n"
         "    except ValueError as failure:\n"
         "        print(device_name, 'dilewati:', failure)\n"
         "        continue\n"
         "    fit_predict = make(lstm.DEFAULT_PARAMS, quantile=lstm.QUANTILE)\n"
         "    started = time.time()\n"
         "    prediction = fit_predict(split['train'], split['valid'])\n"
         "    elapsed = time.time() - started\n"
         "    best_epoch = fit_predict.best_epochs[0]\n"
         "    # elapsed covers both fits: best_epoch + patience epochs in the\n"
         "    # first, best_epoch in the second.\n"
         "    epochs_run = 2 * best_epoch + lstm.EARLY_STOPPING_EPOCHS\n"
         "    benchmark[device_name] = {\n"
         "        'wall_seconds': elapsed,\n"
         "        'best_epoch': best_epoch,\n"
         "        'sec_per_epoch': elapsed / epochs_run,\n"
         "        'peak_rss_gb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,\n"
         "        'pred_mean': float(prediction.mean()),\n"
         "        'pred_max': float(prediction.max()),\n"
         "    }\n"
         "    print(device_name, benchmark[device_name])\n\n"
         "pd.DataFrame(benchmark).T"),

    md("## Anggaran pencarian\n\n"
       "`N` datang dari angka benchmark, bukan dari tebakan. Kalau rumusnya\n"
       "jatuh di bawah 6, `candidate_budget` melempar ValueError — itu sinyal\n"
       "untuk memperkecil ruang search, bukan menaikkan plafon 8 jam."),
    code("DEVICE = min(benchmark, key=lambda name: benchmark[name]['sec_per_epoch'])\n"
         "measured = benchmark[DEVICE]\n"
         "N_CANDIDATES = lstm.candidate_budget(\n"
         "    sec_per_epoch=measured['sec_per_epoch'],\n"
         "    best_epoch=measured['best_epoch'],\n"
         ")\n"
         "print('device terpilih:', DEVICE)\n"
         "print('sec_per_epoch  :', round(measured['sec_per_epoch'], 1))\n"
         "print('best_epoch     :', measured['best_epoch'])\n"
         "print('N_CANDIDATES   :', N_CANDIDATES)"),

    md("## Pencarian hyperparameter\n\n"
       "Fold 3 dan 5 saja, seed 42, kriteria pinball@0.9 gabungan berbobot\n"
       "jumlah baris. Checkpoint di-flush tiap kandidat selesai."),
    code("candidates = lstm.sample_search_space(N_CANDIDATES, seed=42)\n"
         "search_results = lstm.run_search(\n"
         "    df, candidates,\n"
         "    checkpoint_path=lstm.SEARCH_FILE,\n"
         "    device_name=DEVICE,\n"
         ")\n"
         "search_results.sort_values('pinball').head(10)"),

    md("## Walk-forward final\n\n"
       "Pemenang dijalankan ulang di kelima fold, lalu difit final."),
    code("best = lstm.select_best(search_results, candidates)\n"
         "lstm.save_best_params(best)\n"
         "print(best)\n\n"
         "make = lstm.bind_panel(df, device_name=DEVICE)\n"
         "fit_predict = make(best, quantile=lstm.QUANTILE)\n"
         "results = walk_forward.run_walk_forward(\n"
         "    df, fit_predict, model_name='lstm', alpha=lstm.QUANTILE)\n"
         "results.to_csv(lstm.RESULTS_FILE, index=False)\n"
         "print('best_epoch per fold:', fit_predict.best_epochs)"),
    code("bundle = lstm.fit_final(df, best, device_name=DEVICE)\n"
         "lstm.save_bundle(bundle)\n"
         "print(bundle['best_epoch'], bundle['n_train'])"),

    md("## Hasil"),
    code("results = pd.read_csv(lstm.RESULTS_FILE)\n"
         "overall = results[results['group_col'].isna()]\n"
         "for name in overall['model'].unique():\n"
         "    print(name, round(walk_forward.pooled_metric(results, name), 4))\n\n"
         "overall[overall['model'] == 'lstm']"),

    md("## Head-to-head tiga arah\n\n"
       "Sah karena ketiganya dinilai di baris identik — dijamin\n"
       "`walk_forward.eligible_rows()`. Potongan kedua (fold 1, 2, 4) adalah\n"
       "angka bersihnya: tidak ada model yang memakai fold itu untuk seleksi."),
    code("from utils import model_random_forest as rf\n"
         "from utils import model_xgboost as xgb\n\n"
         "tables = {\n"
         "    'lstm': pd.read_csv(lstm.RESULTS_FILE),\n"
         "    'xgboost': pd.read_csv(xgb.RESULTS_FILE),\n"
         "    'random_forest': pd.read_csv(rf.RESULTS_FILE),\n"
         "}\n"
         "rows = []\n"
         "for name, table in tables.items():\n"
         "    rows.append({\n"
         "        'model': name,\n"
         "        'pinball_5_folds': walk_forward.pooled_metric(table, name),\n"
         "        'pinball_folds_124': walk_forward.pooled_metric(\n"
         "            table, name, folds=(1, 2, 4)),\n"
         "        'mae_5_folds': walk_forward.pooled_metric(table, name, metric='mae'),\n"
         "        'coverage': walk_forward.pooled_metric(\n"
         "            table, name, metric='coverage'),\n"
         "    })\n"
         "pd.DataFrame(rows).sort_values('pinball_folds_124')"),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.6"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
pathlib.Path("notebook/modeling_lstm.ipynb").write_text(
    json.dumps(notebook, indent=1) + "\n")
print("written")
NB
```

- [ ] **Step 2: Verify the notebook is valid and its imports resolve**

Run:

```bash
.venv/bin/python3 -c "
import json
nb = json.load(open('notebook/modeling_lstm.ipynb'))
print(len(nb['cells']), 'cells')
"
.venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
from utils import model_lstm as lstm
print(lstm.SEARCH_FILE, lstm.RESULTS_FILE, lstm.MODEL_FILE)
"
```

Expected: `13 cells`, then the three paths under `dataset/model_ready/` and `models/`.

- [ ] **Step 3: Run only the benchmark cells**

Run:

```bash
.venv/bin/python3 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  --to notebook notebook/modeling_lstm.ipynb 2>&1 | tail -20
```

This runs the entire notebook, which takes hours. To run **only** through the budget cell first, execute cells 0-5 in a plain script instead:

```bash
.venv/bin/python3 - <<'BENCH' 2>&1 | tail -20
import json, sys
sys.path.insert(0, '.')
source = json.load(open('notebook/modeling_lstm.ipynb'))
cells = [c for c in source['cells'] if c['cell_type'] == 'code'][:3]
namespace = {}
for cell in cells:
    exec(''.join(cell['source']).replace("sys.path.insert(0, '..')",
                                         "sys.path.insert(0, '.')"), namespace)
BENCH
```

Expected: prints the panel shape, the fold-5 row counts, one line per device, and finally `device terpilih`, `sec_per_epoch`, `best_epoch`, `N_CANDIDATES`.

- [ ] **Step 4: Record the measured numbers**

Write them into `docs/superpowers/plans/2026-08-19-lstm-modeling.md` under a new `## Measured benchmark` heading at the bottom of this file: device chosen, `sec_per_epoch`, `best_epoch`, peak RSS, wall time, and the resulting `N_CANDIDATES`. Task 10 quotes them.

If `candidate_budget` raised, **stop and report to the human**: the search space needs shrinking, and that is their decision, not the implementer's.

- [ ] **Step 5: Commit**

```bash
git add notebook/modeling_lstm.ipynb docs/superpowers/plans/2026-08-19-lstm-modeling.md
git commit -m "feat: add the LSTM modeling notebook and record its benchmark

Benchmark measures CPU and MPS, and the search budget comes from the
measured sec_per_epoch rather than a guess."
```

---

### Task 9: Run the search, the walk-forward, and the final fit

**Files:**
- Modify: `notebook/modeling_lstm.ipynb` (executed, then outputs cleared)
- Produces (all gitignored): `dataset/model_ready/lstm_search_results.csv`, `lstm_best_params.json`, `lstm_walk_forward_results.csv`, `models/lstm_q90.joblib`

**Interfaces:**
- Consumes: Task 8's `N_CANDIDATES` and device choice.
- Produces: the three CSV/JSON artifacts Task 10 reads every number from.

- [ ] **Step 1: Run the full notebook**

Run:

```bash
.venv/bin/python3 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebook/modeling_lstm.ipynb
```

This takes hours. The search checkpoints after every candidate, so an interrupted run resumes from `dataset/model_ready/lstm_search_results.csv` when re-launched.

- [ ] **Step 2: Verify the artifacts exist and are complete**

Run:

```bash
.venv/bin/python3 -c "
import json
import pandas as pd
search = pd.read_csv('dataset/model_ready/lstm_search_results.csv')
print('kandidat:', len(search), '| gagal:', int(search['error'].notna().sum()))
print(search.sort_values('pinball')[['candidate_id','hidden_size','num_layers',
      'dropout','learning_rate','batch_size','log_target','pinball','mae',
      'coverage']].head(5).to_string(index=False))
results = pd.read_csv('dataset/model_ready/lstm_walk_forward_results.csv')
print('fold dinilai:', sorted(results['fold_id'].unique()))
print('model:', sorted(results['model'].unique()))
print(json.load(open('dataset/model_ready/lstm_best_params.json')))
"
ls -la models/lstm_q90.joblib
```

Expected: candidate count equals `N_CANDIDATES`, folds `[1,2,3,4,5]`, models include `lstm` plus the three naive baselines, and the bundle file exists.

- [ ] **Step 3: Verify no December row leaked into the reported metrics**

Run:

```bash
.venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
from utils import model_lstm as lstm, modeling_prep, walk_forward
df = pd.read_parquet(modeling_prep.MODEL_INPUT_FILE)
frame = walk_forward.eligible_rows(df)
print('tanggal maksimum baris layak:', frame['Tanggal'].max())
bundle = lstm.load_bundle()
print('n_train bundle:', bundle['n_train'], '| best_epoch:', bundle['best_epoch'])
assert frame['Tanggal'].max() < modeling_prep.TEST_START
print('OK — Desember tidak tersentuh')
"
```

Expected: a date in November 2025, then `OK — Desember tidak tersentuh`.

- [ ] **Step 4: Clear the notebook outputs and commit**

Run:

```bash
.venv/bin/python3 -m nbconvert --ClearOutputPreprocessor.enabled=True \
  --to notebook --inplace notebook/modeling_lstm.ipynb
git add notebook/modeling_lstm.ipynb
git commit -m "chore: run the LSTM notebook end to end

Outputs cleared before commit: the evidence lives in the CSVs under
dataset/model_ready/ and in docs/, not in cell output."
```

---

### Task 10: `docs/hasil-modeling-lstm.md` and the CLAUDE.md pointer

**Files:**
- Create: `docs/hasil-modeling-lstm.md`
- Modify: `CLAUDE.md` (the modeling paragraph and the Commands list)

**Interfaces:**
- Consumes: every artifact from Task 9, plus the benchmark numbers recorded in Task 8 Step 4.
- Produces: the written evidence. Nothing consumes it.

- [ ] **Step 1: Write the results document**

Create `docs/hasil-modeling-lstm.md` in **Indonesian**, following the structure of `docs/hasil-modeling-xgb.md`. Read that file first and match its tone and section order. Every number comes from an artifact — none may be estimated or recalled.

Required sections and their sources:

| Section | Where every number comes from |
|---|---|
| 1. Ringkasan | pooled pinball/MAE/coverage from `lstm_walk_forward_results.csv`; shortfall/overstock table for `lstm`, `xgboost`, `random_forest`, `naive_roll_mean_7` |
| 2. Setup evaluasi | the table from `hasil-modeling-xgb.md` §2 with the LSTM rows changed: implementation `torch==2.8.0`, architecture (hidden/layers/dropout from `lstm_best_params.json`), epoch count via early stopping + refit, device chosen |
| 3. Benchmark | the numbers recorded in Task 8 Step 4: CPU vs MPS `sec_per_epoch`, `best_epoch`, wall time, peak RSS, device chosen and why |
| 4. Anggaran pencarian | the formula from spec §2.2, the measured inputs, and the resulting N. State the asymmetry against XGB's 30 and RF's 18 plainly |
| 5. Pencarian hyperparameter | `lstm_search_results.csv` — top rows, bottom rows, the spread, and per-dimension medians for `log_target` and `num_layers` |
| 6. Hasil walk-forward | `lstm_walk_forward_results.csv` — per fold, per `demand_segment`, per `is_delivery_day`, with `best_epoch` per fold from the notebook output |
| 7. Head-to-head tiga arah | the three results CSVs, both slices (5 folds, and folds 1/2/4) |
| 8. Model final | `lstm_best_params.json` and the bundle's `best_epoch` / `n_train` |
| 9. Reproduksi | the nbconvert command, and the note that the search resumes from its checkpoint |
| 10. Batasan | `docs/batasan-penelitian.md` B-1/B-2/B-3, plus the four asymmetries |

Section 7 must contain, verbatim in substance, the four asymmetries from spec §3.2 — the three-row protocol table, and this paragraph:

> **LSTM melihat masukan yang lebih banyak dari fitur yang sama.** Random
> Forest dan XGBoost menerima ringkasan 28 hari terakhir yang sudah diringkas
> tangan — `lag_1` sampai `lag_28`, `roll_mean_*`, `roll_std_*`. LSTM menerima
> 28 harinya utuh: 49 kolom × 28 langkah. Set fiturnya identik; jumlah
> informasi yang sampai ke model tidak.
>
> Itu bukan cacat protokol, itu justru pertanyaan penelitiannya. Tapi kalau
> LSTM menang, pembaca berhak tahu bahwa ia menang dengan masukan yang lebih
> kaya — dan kalau ia kalah meski masukannya lebih kaya, itu temuan yang jauh
> lebih kuat daripada sekadar "LSTM kalah".

Also carry over the unit caveat that `hasil-modeling-xgb.md` §1 has, since the same mixed-unit sum appears:

> Catatan satuan: `shortfall_units` dan `overstock_units` menjumlahkan unit
> lintas SKU yang satuannya campur (Kg, Porsi, Botol, PCS, …), jadi angkanya
> sah untuk membandingkan antar model pada baris yang sama, tapi tidak punya
> makna fisik sebagai satu besaran tunggal.

And state, in §2 or §6, the 5% step-count difference between the two fits, as spec §2.4 requires.

- [ ] **Step 2: Verify every number in the document against its artifact**

Run:

```bash
.venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
from utils import model_lstm as lstm, model_xgboost as xgb
from utils import model_random_forest as rf, walk_forward
tables = {'lstm': lstm.RESULTS_FILE, 'xgboost': xgb.RESULTS_FILE,
          'random_forest': rf.RESULTS_FILE}
for name, path in tables.items():
    table = pd.read_csv(path)
    print(name,
          'pinball_5=', round(walk_forward.pooled_metric(table, name), 4),
          'pinball_124=', round(walk_forward.pooled_metric(table, name, folds=(1,2,4)), 4),
          'mae=', round(walk_forward.pooled_metric(table, name, metric='mae'), 3),
          'coverage=', round(walk_forward.pooled_metric(table, name, metric='coverage'), 4))
"
```

Compare each printed figure against the corresponding figure in the document. Any mismatch is a documentation bug — fix the document, never the artifact.

- [ ] **Step 3: Update CLAUDE.md**

In the modeling paragraph (the one beginning "`utils/model_common.py` holds the parts of that machinery..."), append after the XGBoost sentence:

```
`utils/sequence_windows.py` and `utils/model_lstm.py` supply the third model: a 0.9-quantile LSTM with embeddings for the seven categorical columns, whose epochs come from early stopping on the same purged 30-day tail, then a refit on the full training rows. The dense 28-day tensor would be 9.42 GB, so `sequence_windows` indexes the panel into one contiguous 294 MB float32 matrix and gathers windows per batch; `model_lstm.bind_panel()` closes over that index because a window for a fold's first validation row reaches back over rows that appear in neither frame `walk_forward` hands over. See `docs/superpowers/specs/2026-08-19-lstm-modeling-design.md` and `docs/hasil-modeling-lstm.md`. Requires `torch==2.8.0`; no external runtime needed.
```

In the Commands list, after the XGBoost notebook line, add:

```
- Run the LSTM modeling notebook (benchmark, budget, search, final walk-forward; takes hours): `.venv/bin/python3 -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook/modeling_lstm.ipynb`
```

- [ ] **Step 4: Run the whole test suite one last time**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v 2>&1 | tail -5`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add docs/hasil-modeling-lstm.md CLAUDE.md
git commit -m "docs: record the measured LSTM results

Includes the three-way head-to-head on both slices, and the four
asymmetries spec 3.2 requires stated rather than buried - including
that the LSTM reads the 28 days intact while the tree models read a
hand-engineered summary of them."
```

---

## Measured benchmark

_Filled in by Task 8, Step 4. Until then this section is empty by design — the search budget must come from a measurement, not from this document._

