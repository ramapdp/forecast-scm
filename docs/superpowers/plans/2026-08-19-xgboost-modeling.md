# XGBoost Modeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 0.9-quantile XGBoost as the second model in the three-way comparison, scored through the existing walk-forward runner on rows identical to the Random Forest's.

**Architecture:** The search protocol, checkpoint/resume, one-hot expansion and bundle I/O move out of `utils/model_random_forest.py` into a new `utils/model_common.py` that both models import. `utils/model_xgboost.py` then supplies the XGBoost half: `reg:quantileerror` at `quantile_alpha=0.9`, boosting rounds chosen by early stopping on a purged 30-day tail of each fold's training window and then refit on the full training rows, and categorical handling as a searched three-way flag. `utils/walk_forward.py` is not touched — XGBoost enters through the same injected `fit_predict(train, valid) -> np.ndarray` callable the Random Forest uses.

**Tech Stack:** Python 3.9.6, pandas 2.3.3, numpy 2.0.2, xgboost 2.1.4, joblib 1.5.3, unittest.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-08-19-xgboost-modeling-design.md`. Read it before Task 1.
- **December 2025 is locked.** No code in this plan may read, score, or fit on a row dated 2025-12-01 or later. `walk_forward.eligible_rows()` is the only sanctioned row filter.
- **`test/test_model_random_forest.py` may not be edited.** It must stay green through Task 1 with no assertion changed. That is the regression test for the extraction.
- Quantile is **0.9**, uniform across every SKU. Feature set is `modeling_prep.FEATURE_COLS`, identical for every model.
- Python is **3.9.6** — no `match`, no `X | Y` type unions at runtime, no `dict | dict`. Use `Optional[...]` from `typing`.
- xgboost pin is exactly **`xgboost==2.1.4`** (highest release with cp39 wheels; `reg:quantileerror` arrived in 2.0).
- Run tests from the repo root with `.venv/bin/python3 -m unittest ...`. `utils` is a package using relative imports.
- Error messages in `utils/` are written in Indonesian, matching the existing modules.
- Commit after every task. Work on branch `feat/xgboost-modeling`.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `utils/model_common.py` | Create (Task 1) | Model-agnostic search protocol, checkpoint/resume, one-hot expansion, NaN guard, bundle I/O |
| `utils/model_random_forest.py` | Modify (Task 1) | QRF specifics only; re-exports the moved names so its callers are unaffected |
| `test/test_model_common.py` | Create (Task 1) | Tests for the generalized helpers |
| `requirements.txt` | Modify (Task 2) | `xgboost==2.1.4` |
| `utils/model_xgboost.py` | Create (Tasks 2–6) | The XGBoost wrapper: ES split, encoding, two-fit protocol, final fit, bundle, search wrappers |
| `test/test_model_xgboost.py` | Create (Tasks 2–6) | Anti-leakage, encoding, two-fit, bundle tests |
| `notebook/modeling_xgb.ipynb` | Create (Task 7) | Benchmark, search, final walk-forward, results — thin, all logic in `utils/` |
| `docs/hasil-modeling-xgb.md` | Create (Task 8) | Measured results and the head-to-head against Random Forest |
| `CLAUDE.md` | Modify (Task 8) | Point at the new module, notebook, and results doc |

---

### Task 1: Extract `utils/model_common.py`

**Files:**
- Create: `utils/model_common.py`
- Modify: `utils/model_random_forest.py`
- Create: `test/test_model_common.py`
- Untouched (regression gate): `test/test_model_random_forest.py`

**Interfaces:**
- Consumes: `utils/walk_forward.py` (`eligible_rows`, `run_fold`, `pooled_metric`), `utils/modeling_prep.py` (`FEATURE_COLS`).
- Produces:
  - `model_common.assert_no_nan(frame: pd.DataFrame, feature_cols: list) -> None`
  - `model_common.IDX_COLS: list`
  - `model_common.expand_one_hot(train_X, valid_X, idx_cols: Optional[list] = None) -> tuple`
  - `model_common.sample_search_space(space: dict, defaults: dict, n_candidates: int, seed: int = 42, screen: Optional[Callable[[dict], bool]] = None, screen_label: str = "screen") -> list`
  - `model_common.run_search(df, candidates: list, make_fit_predict: Callable, search_space: dict, folds: tuple, alpha: float, model_name: str, feature_cols: Optional[list] = None, verbose: bool = True, checkpoint_path: Optional[str] = None, resume: bool = True, catch: tuple = (MemoryError, ValueError)) -> pd.DataFrame`
  - `model_common.select_best(search_results: pd.DataFrame, candidates: list) -> dict`
  - `model_common.save_bundle(bundle: dict, path: str) -> None`, `load_bundle(path: str) -> dict`, `save_best_params(params: dict, path: str) -> None`

- [ ] **Step 1: Read the module being split**

Run: `sed -n '1,120p' utils/model_random_forest.py` and skim the whole file. The functions moving out are `assert_no_nan`, `expand_one_hot`, `sample_search_space`, `run_search`, `_ordered`, `_assert_checkpoint_matches`, `select_best`, `save_bundle`, `load_bundle`, `save_best_params`, plus the `IDX_COLS` constant.

- [ ] **Step 2: Write the failing tests for the generalized helpers**

Create `test/test_model_common.py`:

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utils import model_common, modeling_prep


FEATURES = ["feat_a", "feat_b", "cat_idx"]

SPACE = {"alpha": [1, 2, 3], "beta": ["x", "y"]}
DEFAULTS = {"alpha": 1, "beta": "x", "pinned": 99}


def _panel(periods=245):
    """One pair's daily series, long enough that the 28-day warm-up cut leaves rows."""
    rows = []
    for i, date in enumerate(pd.date_range("2025-05-01", periods=periods, freq="D")):
        rows.append({
            "Kode Barang": "I1", "Nama Cabang": "B1", "segment_id": 1,
            "Tanggal": date,
            "target_lead_time_cumulative": float(i % 7),
            "lead_time_days": 3.0, "lag_1": float(i % 5),
            "roll_mean_7": float(i % 4), "demand_segment": "smooth",
            "is_delivery_day": bool(i % 2),
            "feat_a": float(i), "feat_b": float(i % 3), "cat_idx": i % 3,
        })
    return modeling_prep.assign_folds(pd.DataFrame(rows))


def _mean_fit_predict(params, feature_cols=None, quantile=0.9):
    """A stand-in model: no library, no fitting, one number per validation row."""
    def fit_predict(train, valid):
        return np.full(len(valid), float(params["alpha"]))
    return fit_predict


class TestSampleSearchSpace(unittest.TestCase):
    def test_returns_the_requested_number_of_candidates(self):
        self.assertEqual(len(model_common.sample_search_space(SPACE, DEFAULTS, 4)), 4)

    def test_the_same_seed_reproduces_the_same_list(self):
        first = model_common.sample_search_space(SPACE, DEFAULTS, 4, seed=7)
        second = model_common.sample_search_space(SPACE, DEFAULTS, 4, seed=7)
        self.assertEqual(first, second)

    def test_candidates_are_distinct(self):
        drawn = model_common.sample_search_space(SPACE, DEFAULTS, 6)
        signatures = {(c["alpha"], c["beta"]) for c in drawn}
        self.assertEqual(len(signatures), 6)

    def test_defaults_fill_the_unsearched_keys(self):
        for candidate in model_common.sample_search_space(SPACE, DEFAULTS, 3):
            self.assertEqual(candidate["pinned"], 99)

    def test_no_screen_rejects_nothing(self):
        self.assertEqual(
            len(model_common.sample_search_space(SPACE, DEFAULTS, 6, screen=None)), 6
        )

    def test_an_injected_screen_rejects_exactly_what_it_says(self):
        drawn = model_common.sample_search_space(
            SPACE, DEFAULTS, 2, screen=lambda params: params["alpha"] == 1
        )
        self.assertEqual({c["alpha"] for c in drawn}, {1})

    def test_a_screen_that_admits_nothing_raises_naming_the_screen(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            model_common.sample_search_space(
                SPACE, DEFAULTS, 4,
                screen=lambda params: False, screen_label="budget 3.0 GB",
            )


class TestRunSearch(unittest.TestCase):
    def _candidates(self):
        return [{**DEFAULTS, "alpha": a} for a in (1, 2)]

    def test_scores_a_space_it_has_never_seen(self):
        results = model_common.run_search(
            _panel(), self._candidates(), make_fit_predict=_mean_fit_predict,
            search_space=SPACE, folds=(1,), alpha=0.9, model_name="toy",
            feature_cols=FEATURES, verbose=False,
        )
        self.assertEqual(list(results["candidate_id"]), [0, 1])
        self.assertTrue(results["pinball"].notna().all())

    def test_records_the_searched_keys_of_that_space(self):
        results = model_common.run_search(
            _panel(), self._candidates(), make_fit_predict=_mean_fit_predict,
            search_space=SPACE, folds=(1,), alpha=0.9, model_name="toy",
            feature_cols=FEATURES, verbose=False,
        )
        for key in SPACE:
            self.assertIn(key, results.columns)

    def test_a_failing_candidate_is_recorded_not_raised(self):
        def exploding(params, feature_cols=None, quantile=0.9):
            def fit_predict(train, valid):
                raise ValueError("meledak")
            return fit_predict

        results = model_common.run_search(
            _panel(), self._candidates(), make_fit_predict=exploding,
            search_space=SPACE, folds=(1,), alpha=0.9, model_name="toy",
            feature_cols=FEATURES, verbose=False,
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(results["pinball"].isna().all())
        self.assertTrue(results["error"].str.contains("meledak").all())

    def test_an_uncaught_exception_type_propagates(self):
        """The catch list is deliberately narrow: a bug must not become a NaN row."""
        def exploding(params, feature_cols=None, quantile=0.9):
            def fit_predict(train, valid):
                raise KeyError("bug")
            return fit_predict

        with self.assertRaises(KeyError):
            model_common.run_search(
                _panel(), self._candidates(), make_fit_predict=exploding,
                search_space=SPACE, folds=(1,), alpha=0.9, model_name="toy",
                feature_cols=FEATURES, verbose=False,
            )

    def test_a_widened_catch_list_records_the_new_type(self):
        def exploding(params, feature_cols=None, quantile=0.9):
            def fit_predict(train, valid):
                raise KeyError("bug")
            return fit_predict

        results = model_common.run_search(
            _panel(), self._candidates(), make_fit_predict=exploding,
            search_space=SPACE, folds=(1,), alpha=0.9, model_name="toy",
            feature_cols=FEATURES, verbose=False, catch=(KeyError,),
        )
        self.assertTrue(results["pinball"].isna().all())


class TestRunSearchCheckpoint(unittest.TestCase):
    def _candidates(self):
        return [{**DEFAULTS, "alpha": a} for a in (1, 2, 3)]

    def _run(self, path, candidates=None, resume=True):
        return model_common.run_search(
            _panel(), candidates or self._candidates(),
            make_fit_predict=_mean_fit_predict, search_space=SPACE,
            folds=(1,), alpha=0.9, model_name="toy", feature_cols=FEATURES,
            verbose=False, checkpoint_path=path, resume=resume,
        )

    def test_writes_a_row_per_candidate_as_it_finishes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "checkpoint.csv")
            self._run(path)
            self.assertEqual(len(pd.read_csv(path)), 3)

    def test_a_finished_candidate_is_not_recomputed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "checkpoint.csv")
            self._run(path)
            calls = []

            def counting(params, feature_cols=None, quantile=0.9):
                calls.append(params["alpha"])
                return _mean_fit_predict(params)

            model_common.run_search(
                _panel(), self._candidates(), make_fit_predict=counting,
                search_space=SPACE, folds=(1,), alpha=0.9, model_name="toy",
                feature_cols=FEATURES, verbose=False, checkpoint_path=path,
            )
            self.assertEqual(calls, [])

    def test_results_come_back_in_candidate_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "checkpoint.csv")
            results = self._run(path)
            self.assertEqual(list(results["candidate_id"]), [0, 1, 2])

    def test_a_checkpoint_from_a_different_space_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "checkpoint.csv")
            self._run(path)
            other = [{**DEFAULTS, "alpha": a} for a in (7, 8, 9)]
            with self.assertRaisesRegex(ValueError, "tidak cocok"):
                self._run(path, candidates=other)


class TestBundleIO(unittest.TestCase):
    def test_a_bundle_round_trips(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "nested" / "bundle.joblib")
            model_common.save_bundle({"columns": ["a", "b"]}, path)
            self.assertEqual(model_common.load_bundle(path)["columns"], ["a", "b"])

    def test_best_params_are_written_sorted_and_readable(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "params.json")
            model_common.save_best_params({"b": 2, "a": 1}, path)
            written = Path(path).read_text(encoding="utf-8")
            self.assertLess(written.index('"a"'), written.index('"b"'))
            self.assertEqual(json.loads(written), {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_model_common -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.model_common'`

- [ ] **Step 4: Create `utils/model_common.py`**

Move the functions verbatim from `model_random_forest.py`, changing only what a second model forces. Preserve the draw order inside `sample_search_space` exactly — dedupe check first, then screen, then record — or the Random Forest's seed-42 candidate list changes and its recorded results stop reproducing.

```python
"""What every model in the comparison needs, and no model owns.

The search protocol, its checkpoint, the one-hot expansion and the bundle
format are not Random Forest ideas — they are how this project runs a search
and ships a model. Leaving them inside model_random_forest.py would mean
fixing the next checkpoint bug twice, then three times when the LSTM lands,
and would point every future model's import at a sibling model.

Nothing here knows what a model is. `run_search` takes the factory; the
screen that rejects an unaffordable candidate is injected too, because the
Random Forest's leaf-storage bound has no XGBoost analogue.
"""

import json
import random
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd

from . import modeling_prep, walk_forward

# The encoded categoricals, the only columns one-hot expansion touches.
IDX_COLS = [col for col in modeling_prep.FEATURE_COLS if col.endswith("_idx")]


def assert_no_nan(frame: pd.DataFrame, feature_cols: list) -> None:
    """Fail loudly on a null the estimator cannot consume.

    Deliberately not an imputation step. build_model_input() already ran
    impute_features(), and running it a second time would recompute
    was_relocated from a column that is now filled with 0.0, setting the
    indicator True on every row and erasing the distinction it exists to make.
    """
    counts = frame[feature_cols].isna().sum()
    offenders = counts[counts > 0]
    if len(offenders):
        raise ValueError(f"NaN pada fitur: {offenders.to_dict()}")


def expand_one_hot(
    train_X: pd.DataFrame,
    valid_X: pd.DataFrame,
    idx_cols: Optional[list] = None,
) -> tuple:
    """One-hot the encoded categoricals, with validation reindexed onto the
    training columns.

    The reindex is the point. A category that appears only in validation would
    otherwise add a column there and shift every column after it, so the model
    would read the wrong feature at every position — silently, since the shapes
    still line up.
    """
    idx_cols = IDX_COLS if idx_cols is None else idx_cols
    present = [col for col in idx_cols if col in train_X.columns]
    train_out = pd.get_dummies(train_X, columns=present)
    valid_out = pd.get_dummies(valid_X, columns=present)
    valid_out = valid_out.reindex(columns=train_out.columns, fill_value=0)
    return train_out, valid_out


def sample_search_space(
    space: dict,
    defaults: dict,
    n_candidates: int,
    seed: int = 42,
    screen: Optional[Callable[[dict], bool]] = None,
    screen_label: str = "screen",
) -> list:
    """Distinct parameter sets drawn at random, optionally screened.

    Random rather than grid: only a few dimensions of these spaces carry real
    signal, so random draws cover each dimension's range better than a
    truncated grid at the same cost.

    `screen` returns True for a candidate worth fitting. The Random Forest
    injects its leaf-storage bound here; XGBoost has no equivalent and passes
    None.
    """
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
        candidate = {**defaults, **drawn}
        if screen is not None and not screen(candidate):
            continue
        seen.add(signature)
        candidates.append(candidate)

    if len(candidates) < n_candidates:
        raise ValueError(
            f"hanya {len(candidates)} dari {n_candidates} kandidat lolos "
            f"{screen_label}"
        )
    return candidates


SEARCH_METRICS = ("pinball", "mae", "coverage", "fill_rate")


def run_search(
    df: pd.DataFrame,
    candidates: list,
    make_fit_predict: Callable,
    search_space: dict,
    folds: tuple,
    alpha: float,
    model_name: str,
    feature_cols: Optional[list] = None,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
    catch: tuple = (MemoryError, ValueError),
) -> pd.DataFrame:
    """Score every candidate on the search folds only.

    A candidate that raises one of `catch` is recorded with NaN metrics rather
    than aborting the run: a long search should not lose fourteen finished
    candidates to the fifteenth. `catch` is narrow on purpose — an unexpected
    exception type is a bug, and a bug must not be laundered into a NaN row.

    `checkpoint_path` extends that reasoning past what Python can catch. A
    search at this scale runs for hours, and an OS-level kill leaves no
    exception to handle — so every finished candidate is flushed to disk
    immediately, and the file doubles as the only progress signal available
    while the run is buried inside a notebook cell.

    `resume` is on by default because that is what the checkpoint is for. The
    stale-checkpoint guard is the price: resuming across a changed search space
    or seed would blend candidates from two different experiments and hand back
    a winner that was never actually evaluated.
    """
    frame = walk_forward.eligible_rows(df)
    rows = []
    completed = set()

    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        prior = pd.read_csv(checkpoint_path)
        _assert_checkpoint_matches(prior, candidates, search_space, checkpoint_path)
        rows = prior.to_dict("records")
        completed = {int(value) for value in prior["candidate_id"]}
        if verbose and completed:
            print(f"melanjutkan dari checkpoint: {len(completed)} kandidat sudah selesai",
                  flush=True)

    for candidate_id, candidate in enumerate(candidates):
        if candidate_id in completed:
            continue
        record = {"candidate_id": candidate_id,
                  **{key: candidate[key] for key in sorted(search_space)}}
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
            for metric in SEARCH_METRICS:
                record[metric] = walk_forward.pooled_metric(
                    results, model_name, metric=metric, folds=folds
                )
            record["error"] = None
        except catch as failure:
            for metric in SEARCH_METRICS:
                record[metric] = float("nan")
            record["error"] = str(failure)
        if verbose:
            print(f"[{candidate_id + 1}/{len(candidates)}] "
                  f"pinball={record['pinball']:.4f} {record['error'] or ''}",
                  flush=True)
        rows.append(record)
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            _ordered(rows).to_csv(checkpoint_path, index=False)
    return _ordered(rows)


def _ordered(rows: list) -> pd.DataFrame:
    """Candidate order, so select_best() can index `candidates` by position
    regardless of the order a resumed run happened to finish them in."""
    return (pd.DataFrame(rows)
            .sort_values("candidate_id")
            .reset_index(drop=True))


def _assert_checkpoint_matches(
    prior: pd.DataFrame,
    candidates: list,
    search_space: dict,
    path: str,
) -> None:
    """Refuse a checkpoint whose rows describe different candidates.

    Compares the searched parameters rather than trusting the file name. NaN
    stands in for None in the CSV, and booleans survive the round trip as
    numpy bools, so both are normalised before comparison.
    """
    for _, row in prior.iterrows():
        candidate_id = int(row["candidate_id"])
        if candidate_id >= len(candidates):
            raise ValueError(
                f"checkpoint {path} memuat candidate_id {candidate_id} "
                f"di luar {len(candidates)} kandidat saat ini"
            )
        for key in sorted(search_space):
            expected = candidates[candidate_id][key]
            actual = None if pd.isna(row[key]) else row[key]
            if isinstance(expected, bool):
                actual = bool(actual)
            elif isinstance(expected, (int, float)) and actual is not None:
                actual = type(expected)(actual)
            if expected != actual:
                raise ValueError(
                    f"checkpoint {path} tidak cocok dengan ruang pencarian: "
                    f"kandidat {candidate_id} punya {key}={actual}, "
                    f"seharusnya {expected}"
                )


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


def save_bundle(bundle: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str) -> dict:
    return joblib.load(path)


def save_best_params(params: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, sort_keys=True)
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_model_common -v`
Expected: PASS — every test in the module green.

- [ ] **Step 6: Rewire `utils/model_random_forest.py`**

Delete the moved definitions and the now-unused `json`/`random`/`joblib` imports. Add `from . import model_common` and the re-exports, then replace the two functions that needed generalizing with thin wrappers. The rest of the file (`QUANTILE`, `MEMORY_BUDGET_BYTES`, `DEFAULT_PARAMS`, `SEARCH_SPACE`, `ESTIMATOR_KEYS`, `estimate_leaf_memory_bytes`, `build_estimator`, `make_fit_predict`, `SEARCH_FOLDS`, `TYPICAL_N_TRAIN`, the path constants, `FINAL_N_ESTIMATORS`, `fit_final`, `predict_bundle`) stays exactly as it is.

Replace the moved block with:

```python
from . import model_common, modeling_prep, purging, walk_forward

# Re-exported so this module's callers — its test suite and modeling_rf.ipynb —
# keep working unchanged after the extraction. The definitions live in
# model_common.py because XGBoost and the LSTM need them too.
IDX_COLS = model_common.IDX_COLS
assert_no_nan = model_common.assert_no_nan
expand_one_hot = model_common.expand_one_hot
select_best = model_common.select_best
load_bundle = model_common.load_bundle


def sample_search_space(
    n_candidates: int = 18,
    n_train: int = TYPICAL_N_TRAIN,
    seed: int = 42,
    memory_budget: int = MEMORY_BUDGET_BYTES,
    space: Optional[dict] = None,
) -> list:
    """Distinct, within-budget parameter sets drawn at random.

    The budget screen is what makes this wrapper worth keeping: quantile-forest
    sizes its leaf-value array from the hyperparameters before a single tree is
    grown, so an unaffordable candidate can be refused without loading data.
    """
    def screen(candidate: dict) -> bool:
        return estimate_leaf_memory_bytes(candidate, n_train) <= memory_budget

    return model_common.sample_search_space(
        space=SEARCH_SPACE if space is None else space,
        defaults=DEFAULT_PARAMS,
        n_candidates=n_candidates,
        seed=seed,
        screen=screen,
        screen_label=f"budget {memory_budget / 1024 ** 3:.1f} GB",
    )


def run_search(
    df: pd.DataFrame,
    candidates: list,
    folds: tuple = SEARCH_FOLDS,
    alpha: float = QUANTILE,
    model_name: str = "random_forest",
    feature_cols: Optional[list] = None,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every Random Forest candidate on the search folds."""
    return model_common.run_search(
        df, candidates, make_fit_predict=make_fit_predict,
        search_space=SEARCH_SPACE, folds=folds, alpha=alpha,
        model_name=model_name, feature_cols=feature_cols, verbose=verbose,
        checkpoint_path=checkpoint_path, resume=resume,
    )


def save_bundle(bundle: dict, path: str = MODEL_FILE) -> None:
    model_common.save_bundle(bundle, path)


def save_best_params(params: dict, path: str = BEST_PARAMS_FILE) -> None:
    model_common.save_best_params(params, path)
```

Note the ordering constraint: `sample_search_space` and `run_search` reference `TYPICAL_N_TRAIN`, `MEMORY_BUDGET_BYTES`, `SEARCH_SPACE`, `SEARCH_FOLDS` and `make_fit_predict` at call time, but `save_bundle`/`save_best_params` reference `MODEL_FILE`/`BEST_PARAMS_FILE` as **default argument values**, which are evaluated at definition time. Put those two wrappers after the path constants near the end of the file.

- [ ] **Step 7: Run the Random Forest suite — the regression gate**

Run: `.venv/bin/python3 -m unittest test.test_model_random_forest -v`
Expected: PASS, all tests, with `test/test_model_random_forest.py` unmodified. If any test fails, the extraction changed behaviour — fix `utils/`, never the test.

- [ ] **Step 8: Confirm the seed-42 candidate list is byte-identical to what produced the recorded results**

Run:
```bash
.venv/bin/python3 -c "
from utils import model_random_forest as rf
import json, pandas as pd
drawn = rf.sample_search_space(18, n_train=1_280_000, seed=42)
saved = pd.read_csv('dataset/model_ready/rf_search_results.csv')
keys = sorted(rf.SEARCH_SPACE)
for i, row in saved.iterrows():
    for k in keys:
        want = drawn[int(row['candidate_id'])][k]
        got = None if pd.isna(row[k]) else row[k]
        if isinstance(want, bool): got = bool(got)
        elif isinstance(want, (int, float)) and got is not None: got = type(want)(got)
        assert want == got, (row['candidate_id'], k, want, got)
print('18 kandidat cocok dengan rf_search_results.csv')
"
```
Expected: `18 kandidat cocok dengan rf_search_results.csv`. This proves the refactor did not silently reshuffle the Random Forest's search, so its recorded results stay reproducible.

- [ ] **Step 9: Run the whole suite**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v 2>&1 | tail -5`
Expected: `OK`.

- [ ] **Step 10: Commit**

```bash
git add utils/model_common.py utils/model_random_forest.py test/test_model_common.py
git commit -m "refactor: lift the shared search protocol out of the RF module

The search loop, its checkpoint and resume, one-hot expansion and the bundle
format are not Random Forest ideas. XGBoost needs all of them and the LSTM
will too, so they move to utils/model_common.py before a second copy exists.

Two generalizations only: run_search takes the model factory and its search
space as arguments, and sample_search_space takes the affordability screen as
an injected callable — the QRF leaf-storage bound has no XGBoost analogue.
Draw order inside the sampler is unchanged, verified against the recorded
seed-42 candidate list, so the measured RF results still reproduce.

test/test_model_random_forest.py is untouched and green.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: The early-stopping split

**Files:**
- Modify: `requirements.txt`
- Create: `utils/model_xgboost.py`
- Create: `test/test_model_xgboost.py`

**Interfaces:**
- Consumes: `model_common`, `modeling_prep.DATE_COL`/`TARGET_COL`, `purging.lookahead_safe_mask`.
- Produces:
  - `xgb.QUANTILE = 0.9`, `xgb.ES_TAIL_DAYS = 30`, `xgb.EARLY_STOPPING_ROUNDS = 50`, `xgb.MAX_ROUNDS = 2000`
  - `xgb.DEFAULT_PARAMS: dict`, `xgb.SEARCH_SPACE: dict`, `xgb.ESTIMATOR_KEYS: tuple`
  - `xgb.split_early_stopping(train: pd.DataFrame, tail_days: int = ES_TAIL_DAYS, date_col: str = modeling_prep.DATE_COL) -> tuple` returning `(fit_rows, es_rows)`

- [ ] **Step 1: Install the dependency**

```bash
echo "xgboost==2.1.4" >> requirements.txt
.venv/bin/pip install xgboost==2.1.4
.venv/bin/python3 -c "import xgboost; print(xgboost.__version__)"
```
Expected: `2.1.4`.

- [ ] **Step 2: Confirm the quantile objective is actually reachable on this build**

Run:
```bash
.venv/bin/python3 -c "
import numpy as np
from xgboost import XGBRegressor
X = np.random.default_rng(0).normal(size=(500, 3))
y = np.abs(X[:, 0] * 10 + 20)
m = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.9,
                 tree_method='hist', n_estimators=20)
m.fit(X, y)
print('ok', m.predict(X[:3]))
"
```
Expected: `ok` and three numbers. If this fails, stop — the whole design rests on it.

- [ ] **Step 3: Write the failing tests**

Create `test/test_model_xgboost.py`:

```python
import unittest

import numpy as np
import pandas as pd

from utils import model_xgboost as xgb
from utils import modeling_prep, purging, walk_forward


FEATURES = ["feat_a", "feat_b", "cat_idx"]


def _dated_frame(n=400, seed=3, lead_time=3.0, start="2025-01-01"):
    """One pair's daily series, long enough to survive the 28-day warm-up cut."""
    rng = np.random.default_rng(seed)
    feat_a = rng.normal(size=n)
    return pd.DataFrame({
        "Tanggal": pd.date_range(start, periods=n, freq="D"),
        "feat_a": feat_a,
        "feat_b": rng.normal(size=n),
        "cat_idx": rng.integers(0, 3, size=n),
        "target_lead_time_cumulative": np.abs(feat_a * 10 + 20),
        "lead_time_days": lead_time,
        "Kode Barang": "FGS-00001",
        "Nama Cabang": "KY001",
        "segment_id": 1,
    })


class TestSplitEarlyStopping(unittest.TestCase):
    def test_the_tail_is_the_last_thirty_days(self):
        train = _dated_frame(200)
        _, es_rows = xgb.split_early_stopping(train, tail_days=30)
        self.assertEqual(len(es_rows), 30)
        self.assertEqual(es_rows["Tanggal"].max(), train["Tanggal"].max())

    def test_no_es_date_precedes_a_fit_date(self):
        fit_rows, es_rows = xgb.split_early_stopping(_dated_frame(200), tail_days=30)
        self.assertLess(fit_rows["Tanggal"].max(), es_rows["Tanggal"].min())

    def test_the_boundary_is_purged(self):
        """lead_time_days is 3, so the three fit rows nearest the tail carry a
        label built partly out of the early-stopping window."""
        train = _dated_frame(200, lead_time=3.0)
        fit_rows, es_rows = xgb.split_early_stopping(train, tail_days=30)
        es_start = es_rows["Tanggal"].min()
        self.assertTrue(
            (fit_rows["Tanggal"] + pd.Timedelta(days=3) < es_start).all()
        )

    def test_a_longer_lead_time_purges_more(self):
        short = xgb.split_early_stopping(_dated_frame(200, lead_time=1.0))[0]
        long = xgb.split_early_stopping(_dated_frame(200, lead_time=4.0))[0]
        self.assertLess(len(long), len(short))

    def test_the_two_parts_do_not_overlap(self):
        fit_rows, es_rows = xgb.split_early_stopping(_dated_frame(200))
        self.assertEqual(set(fit_rows.index) & set(es_rows.index), set())

    def test_a_training_window_too_short_to_split_is_refused(self):
        with self.assertRaisesRegex(ValueError, "terlalu pendek"):
            xgb.split_early_stopping(_dated_frame(20), tail_days=30)

    def test_an_empty_frame_is_refused(self):
        with self.assertRaisesRegex(ValueError, "kosong"):
            xgb.split_early_stopping(_dated_frame(0), tail_days=30)


class TestSearchSpace(unittest.TestCase):
    def test_n_estimators_is_not_searched(self):
        """Early stopping decides the round count; searching it wastes budget."""
        self.assertNotIn("n_estimators", xgb.SEARCH_SPACE)

    def test_encoding_offers_all_three_modes(self):
        self.assertEqual(set(xgb.SEARCH_SPACE["encoding"]),
                         {"ordinal", "native", "one_hot"})

    def test_defaults_cover_every_searched_key(self):
        for key in xgb.SEARCH_SPACE:
            self.assertIn(key, xgb.DEFAULT_PARAMS)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run to verify failure**

Run: `.venv/bin/python3 -m unittest test.test_model_xgboost -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.model_xgboost'`

- [ ] **Step 5: Create `utils/model_xgboost.py` with the constants and the split**

```python
"""XGBoost at the 0.9 service level.

`reg:quantileerror` optimizes the same pinball loss the models are selected
on, so training objective and selection criterion agree — which is not true of
a squared-error model asked for a high quantile afterwards.

What makes this wrapper more than a thin call is the round count. Boosting
overfits if it runs too long, so the number of rounds is itself a
regularization decision, and the obvious place to make it — the validation
fold — is exactly the place that would leak. Early stopping therefore runs on
a held-out tail of the training window, and the model is then refit on the
full training rows at the round count that tail chose. Two fits per fold, so
that XGBoost is finally trained on the same population the Random Forest saw
and the comparison stays like-for-like.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from xgboost.core import XGBoostError

from . import model_common, modeling_prep, purging, walk_forward

QUANTILE = 0.9

# The last 30 days of each fold's training window, held out to choose the
# round count. Long enough to cover a full delivery cycle and every weekday.
ES_TAIL_DAYS = 30
EARLY_STOPPING_ROUNDS = 50

# A ceiling, not a target: at learning_rate 0.03 the search needs room, and
# early stopping is what actually decides where a candidate lands.
MAX_ROUNDS = 2000

DEFAULT_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 10,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "encoding": "ordinal",
    "log_target": False,
    "random_state": 42,
}

# n_estimators is absent on purpose: early stopping decides it per candidate
# per fold, so searching it would spend budget on a question that already has
# a mechanism.
SEARCH_SPACE = {
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 10, 50],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "reg_lambda": [1.0, 10.0],
    "encoding": ["ordinal", "native", "one_hot"],
    "log_target": [False, True],
}

ESTIMATOR_KEYS = ("max_depth", "learning_rate", "min_child_weight",
                  "subsample", "colsample_bytree", "reg_lambda", "random_state")


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

- [ ] **Step 6: Run to verify the tests pass**

Run: `.venv/bin/python3 -m unittest test.test_model_xgboost -v`
Expected: PASS — every test in the module green.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt utils/model_xgboost.py test/test_model_xgboost.py
git commit -m "feat: add the XGBoost early-stopping split with its purge

The round count is a regularization decision, and the obvious place to make
it — the validation fold — is exactly the place that would leak. It is made
on a 30-day tail of the training window instead.

The purge between fit rows and tail is the part worth reading twice: the
target sums over H+1..H+lead_time, so without it the last few fit rows carry
labels built partly out of the early-stopping window and stopping happens too
late.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Categorical encoding, all three modes

**Files:**
- Modify: `utils/model_xgboost.py`
- Modify: `test/test_model_xgboost.py`

**Interfaces:**
- Consumes: `model_common.expand_one_hot`, `model_common.IDX_COLS`.
- Produces:
  - `xgb.encode(train_X: pd.DataFrame, valid_X: pd.DataFrame, encoding: str, idx_cols: Optional[list] = None) -> tuple` returning `(train_out, valid_out, enable_categorical: bool)`
  - `xgb.training_categories(train_X: pd.DataFrame, idx_cols: Optional[list] = None) -> dict` mapping column name to the sorted training category list
  - `xgb.apply_encoding(X: pd.DataFrame, encoding: str, columns: list, categories: dict, idx_cols: Optional[list] = None) -> tuple` returning `(X_out, enable_categorical)` — used at prediction time against a recorded layout

- [ ] **Step 1: Write the failing tests**

Append to `test/test_model_xgboost.py`, above the `if __name__` block:

```python
class TestEncode(unittest.TestCase):
    def _pair(self):
        train = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "cat_idx": [0, 1, 2]})
        valid = pd.DataFrame({"feat_a": [4.0, 5.0], "cat_idx": [1, 0]})
        return train, valid

    def test_ordinal_passes_the_index_through(self):
        train, valid = self._pair()
        train_out, valid_out, enable = xgb.encode(train, valid, "ordinal",
                                                  idx_cols=["cat_idx"])
        self.assertFalse(enable)
        self.assertEqual(list(train_out.columns), ["feat_a", "cat_idx"])
        self.assertEqual(list(train_out["cat_idx"]), [0, 1, 2])

    def test_native_makes_the_index_categorical(self):
        train, valid = self._pair()
        train_out, valid_out, enable = xgb.encode(train, valid, "native",
                                                  idx_cols=["cat_idx"])
        self.assertTrue(enable)
        self.assertEqual(str(train_out["cat_idx"].dtype), "category")
        self.assertEqual(str(valid_out["cat_idx"].dtype), "category")

    def test_native_gives_validation_the_training_categories(self):
        train, valid = self._pair()
        train_out, valid_out, _ = xgb.encode(train, valid, "native",
                                             idx_cols=["cat_idx"])
        self.assertEqual(list(train_out["cat_idx"].cat.categories),
                         list(valid_out["cat_idx"].cat.categories))

    def test_native_turns_an_unseen_category_into_a_null(self):
        """XGBoost consumes NaN natively; an unseen level must not become a
        different level's code."""
        train, valid = self._pair()
        valid = pd.DataFrame({"feat_a": [4.0], "cat_idx": [99]})
        _, valid_out, _ = xgb.encode(train, valid, "native", idx_cols=["cat_idx"])
        self.assertTrue(valid_out["cat_idx"].isna().all())

    def test_one_hot_expands_and_drops_the_index(self):
        train, valid = self._pair()
        train_out, valid_out, enable = xgb.encode(train, valid, "one_hot",
                                                  idx_cols=["cat_idx"])
        self.assertFalse(enable)
        self.assertNotIn("cat_idx", train_out.columns)
        self.assertEqual(list(train_out.columns), list(valid_out.columns))

    def test_every_mode_preserves_row_count_and_order(self):
        train, valid = self._pair()
        for encoding in ("ordinal", "native", "one_hot"):
            train_out, valid_out, _ = xgb.encode(train, valid, encoding,
                                                 idx_cols=["cat_idx"])
            self.assertEqual(len(train_out), 3, encoding)
            self.assertEqual(len(valid_out), 2, encoding)
            self.assertEqual(list(valid_out["feat_a"]), [4.0, 5.0], encoding)

    def test_a_validation_only_category_never_shifts_columns(self):
        train, valid = self._pair()
        valid = pd.DataFrame({"feat_a": [4.0, 5.0], "cat_idx": [1, 7]})
        for encoding in ("ordinal", "native", "one_hot"):
            train_out, valid_out, _ = xgb.encode(train, valid, encoding,
                                                 idx_cols=["cat_idx"])
            self.assertEqual(list(train_out.columns), list(valid_out.columns),
                             encoding)

    def test_an_unknown_encoding_is_refused(self):
        train, valid = self._pair()
        with self.assertRaisesRegex(ValueError, "encoding"):
            xgb.encode(train, valid, "embedding", idx_cols=["cat_idx"])


class TestApplyEncoding(unittest.TestCase):
    def _fit_layout(self, encoding):
        train = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "cat_idx": [0, 1, 2]})
        train_out, _, _ = xgb.encode(train, train, encoding, idx_cols=["cat_idx"])
        return (list(train_out.columns),
                xgb.training_categories(train, idx_cols=["cat_idx"]))

    def test_it_reproduces_the_training_columns_in_every_mode(self):
        for encoding in ("ordinal", "native", "one_hot"):
            columns, categories = self._fit_layout(encoding)
            frame = pd.DataFrame({"cat_idx": [2, 0], "feat_a": [9.0, 8.0]})
            out, _ = xgb.apply_encoding(frame, encoding, columns, categories,
                                        idx_cols=["cat_idx"])
            self.assertEqual(list(out.columns), columns, encoding)

    def test_a_shuffled_input_column_order_does_not_change_the_output(self):
        columns, categories = self._fit_layout("one_hot")
        frame = pd.DataFrame({"cat_idx": [2, 0], "feat_a": [9.0, 8.0]})
        out, _ = xgb.apply_encoding(frame, "one_hot", columns, categories,
                                    idx_cols=["cat_idx"])
        self.assertEqual(list(out.columns), columns)

    def test_native_restores_the_recorded_categories(self):
        columns, categories = self._fit_layout("native")
        frame = pd.DataFrame({"feat_a": [9.0], "cat_idx": [1]})
        out, enable = xgb.apply_encoding(frame, "native", columns, categories,
                                         idx_cols=["cat_idx"])
        self.assertTrue(enable)
        self.assertEqual(list(out["cat_idx"].cat.categories), categories["cat_idx"])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python3 -m unittest test.test_model_xgboost.TestEncode -v`
Expected: FAIL — `AttributeError: module 'utils.model_xgboost' has no attribute 'encode'`

- [ ] **Step 3: Implement the three modes**

Append to `utils/model_xgboost.py`:

```python
ENCODINGS = ("ordinal", "native", "one_hot")


def _present(frame: pd.DataFrame, idx_cols: Optional[list]) -> list:
    idx_cols = model_common.IDX_COLS if idx_cols is None else idx_cols
    return [col for col in idx_cols if col in frame.columns]


def training_categories(
    train_X: pd.DataFrame,
    idx_cols: Optional[list] = None,
) -> dict:
    """The category levels each encoded column had in training, sorted.

    Recorded in the bundle so a reloaded model rebuilds the identical dtype.
    Category codes are positional: rebuilding a column from a different level
    list silently renumbers every row.
    """
    return {
        col: sorted(train_X[col].dropna().unique().tolist())
        for col in _present(train_X, idx_cols)
    }


def _as_categorical(frame: pd.DataFrame, categories: dict) -> pd.DataFrame:
    out = frame.copy()
    for col, levels in categories.items():
        if col in out.columns:
            out[col] = out[col].astype(pd.CategoricalDtype(categories=levels))
    return out


def encode(
    train_X: pd.DataFrame,
    valid_X: pd.DataFrame,
    encoding: str,
    idx_cols: Optional[list] = None,
) -> tuple:
    """Prepare both matrices under one of the three searched encodings.

    Returns `(train_out, valid_out, enable_categorical)`. In every mode the
    validation columns are forced onto the training columns: a category
    present only in validation would otherwise shift every column after it,
    and the booster would read the wrong feature at each position — silently,
    since the shapes still line up.

    Under `native`, a level unseen in training becomes NaN rather than
    borrowing another level's code. XGBoost consumes NaN natively, so an
    unknown category is treated as missing, which is what it is.
    """
    if encoding == "ordinal":
        return train_X, valid_X.reindex(columns=train_X.columns), False
    if encoding == "native":
        categories = training_categories(train_X, idx_cols)
        train_out = _as_categorical(train_X, categories)
        valid_out = _as_categorical(valid_X.reindex(columns=train_X.columns),
                                    categories)
        return train_out, valid_out, True
    if encoding == "one_hot":
        train_out, valid_out = model_common.expand_one_hot(
            train_X, valid_X, idx_cols=_present(train_X, idx_cols)
        )
        return train_out, valid_out, False
    raise ValueError(f"encoding tidak dikenal: {encoding!r}, pilih dari {ENCODINGS}")


def apply_encoding(
    X: pd.DataFrame,
    encoding: str,
    columns: list,
    categories: dict,
    idx_cols: Optional[list] = None,
) -> tuple:
    """Encode a frame for prediction against a layout recorded at fit time.

    A booster reloaded next month against columns in a different order does
    not fail — it predicts confidently from the wrong features, which is
    worse. `columns` is the authority here, not whatever the caller passed in.
    """
    if encoding == "one_hot":
        out = pd.get_dummies(X, columns=_present(X, idx_cols))
    elif encoding == "native":
        out = _as_categorical(X, categories)
    elif encoding == "ordinal":
        out = X
    else:
        raise ValueError(f"encoding tidak dikenal: {encoding!r}, pilih dari {ENCODINGS}")

    out = out.reindex(columns=columns, fill_value=0)
    if encoding == "native":
        out = _as_categorical(out, categories)
    return out, encoding == "native"
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/bin/python3 -m unittest test.test_model_xgboost -v`
Expected: PASS — every test in the module green, including the new TestEncode and TestApplyEncoding classes.

- [ ] **Step 5: Commit**

```bash
git add utils/model_xgboost.py test/test_model_xgboost.py
git commit -m "feat: add the three searched categorical encodings

ordinal, native (enable_categorical) and one_hot behind one function, so the
search can settle the question with evidence instead of an assumption. All
three are affordable here — the largest categorical has 70 levels.

Every mode forces validation onto the training columns, and native maps an
unseen level to NaN rather than to another level's code. Both failure modes
are silent: the shapes still line up while the booster reads the wrong
feature.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The two-fit `make_fit_predict`

**Files:**
- Modify: `utils/model_xgboost.py`
- Modify: `test/test_model_xgboost.py`

**Interfaces:**
- Consumes: `xgb.split_early_stopping`, `xgb.encode`, `model_common.assert_no_nan`, `modeling_prep.inverse_log_target`.
- Produces:
  - `xgb.build_estimator(params: dict, n_estimators: int, enable_categorical: bool = False, early_stopping_rounds: Optional[int] = None) -> XGBRegressor`
  - `xgb.make_fit_predict(params: Optional[dict] = None, feature_cols: Optional[list] = None, quantile: float = QUANTILE, tail_days: int = ES_TAIL_DAYS, early_stopping_rounds: int = EARLY_STOPPING_ROUNDS, max_rounds: int = MAX_ROUNDS, idx_cols: Optional[list] = None) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]`
  - `idx_cols` defaults to `model_common.IDX_COLS` (the real `*_idx` feature names). Tests running on synthetic frames must pass their own, or every encoding silently becomes a no-op and tests nothing.
  - The returned callable carries `fit_predict.best_iterations: list` — one recorded round count per fold, in call order.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_model_xgboost.py`:

```python
class TestMakeFitPredict(unittest.TestCase):
    def _params(self, **overrides):
        return {**xgb.DEFAULT_PARAMS, "max_depth": 3, "learning_rate": 0.3,
                "min_child_weight": 1, "random_state": 0, **overrides}

    def _split(self, n=300):
        frame = _dated_frame(n)
        return frame.iloc[:250], frame.iloc[250:]

    def test_returns_one_prediction_per_validation_row(self):
        train, valid = self._split()
        prediction = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                          max_rounds=40)(train, valid)
        self.assertEqual(prediction.shape, (len(valid),))

    def test_predictions_are_never_negative(self):
        train, valid = self._split()
        prediction = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                          max_rounds=40)(train, valid)
        self.assertTrue((prediction >= 0).all())

    def test_the_high_quantile_sits_above_the_low_one(self):
        train, valid = self._split()
        high = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                    quantile=0.9, max_rounds=60)(train, valid)
        low = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                   quantile=0.1, max_rounds=60)(train, valid)
        self.assertGreater(high.mean(), low.mean())

    def test_every_encoding_runs_end_to_end(self):
        train, valid = self._split()
        for encoding in ("ordinal", "native", "one_hot"):
            prediction = xgb.make_fit_predict(
                self._params(encoding=encoding), feature_cols=FEATURES,
                max_rounds=40, idx_cols=["cat_idx"],
            )(train, valid)
            self.assertEqual(prediction.shape, (len(valid),), encoding)

    def test_one_hot_really_expands_on_this_frame(self):
        """Guards the test suite itself: without idx_cols the synthetic frame
        has no column matching the real IDX_COLS, so every encoding would
        quietly become a no-op and these tests would prove nothing."""
        train, _ = self._split()
        expanded, _, _ = xgb.encode(train[FEATURES], train[FEATURES], "one_hot",
                                    idx_cols=["cat_idx"])
        self.assertGreater(len(expanded.columns), len(FEATURES))

    def test_log_target_returns_predictions_on_the_original_scale(self):
        train, valid = self._split()
        logged = xgb.make_fit_predict(self._params(log_target=True),
                                      feature_cols=FEATURES, max_rounds=60)(train, valid)
        self.assertGreater(logged.mean(), 5.0)

    def test_the_same_seed_gives_the_same_predictions(self):
        train, valid = self._split()
        first = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                     max_rounds=40)(train, valid)
        second = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                      max_rounds=40)(train, valid)
        np.testing.assert_allclose(first, second)

    def test_a_nan_feature_is_rejected_rather_than_imputed(self):
        train, valid = self._split()
        train = train.copy()
        train.loc[train.index[0], "feat_a"] = np.nan
        with self.assertRaisesRegex(ValueError, "feat_a"):
            xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                 max_rounds=40)(train, valid)

    def test_the_round_count_is_recorded_per_call(self):
        train, valid = self._split()
        fit_predict = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                           max_rounds=40)
        fit_predict(train, valid)
        fit_predict(train, valid)
        self.assertEqual(len(fit_predict.best_iterations), 2)
        self.assertTrue(all(count >= 1 for count in fit_predict.best_iterations))

    def test_the_second_fit_sees_every_training_row(self):
        """The refit is the whole point: XGBoost must end up trained on the
        same population the Random Forest saw, tail included."""
        train, valid = self._split()
        seen = []
        original = xgb.build_estimator

        def spy(params, n_estimators, enable_categorical=False,
                early_stopping_rounds=None, quantile=xgb.QUANTILE):
            model = original(params, n_estimators,
                             enable_categorical=enable_categorical,
                             early_stopping_rounds=early_stopping_rounds,
                             quantile=quantile)
            real_fit = model.fit

            def fit(X, y, **kwargs):
                seen.append(len(X))
                return real_fit(X, y, **kwargs)

            model.fit = fit
            return model

        xgb.build_estimator = spy
        try:
            xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                 max_rounds=40)(train, valid)
        finally:
            xgb.build_estimator = original

        fit_rows, es_rows = xgb.split_early_stopping(train)
        self.assertEqual(seen[0], len(fit_rows))
        self.assertEqual(seen[1], len(train))

    def test_the_second_fit_uses_the_round_count_the_first_chose(self):
        train, valid = self._split()
        rounds = []
        original = xgb.build_estimator

        def spy(params, n_estimators, enable_categorical=False,
                early_stopping_rounds=None, quantile=xgb.QUANTILE):
            rounds.append(n_estimators)
            return original(params, n_estimators,
                            enable_categorical=enable_categorical,
                            early_stopping_rounds=early_stopping_rounds,
                            quantile=quantile)

        xgb.build_estimator = spy
        try:
            fit_predict = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                               max_rounds=40)
            fit_predict(train, valid)
        finally:
            xgb.build_estimator = original

        self.assertEqual(rounds[0], 40)
        self.assertEqual(rounds[1], fit_predict.best_iterations[0])


class TestWalkForwardIntegration(unittest.TestCase):
    def _panel(self, periods=245):
        rows = []
        for i, date in enumerate(pd.date_range("2025-05-01", periods=periods, freq="D")):
            rows.append({
                "Kode Barang": "I1", "Nama Cabang": "B1", "segment_id": 1,
                "Tanggal": date,
                "target_lead_time_cumulative": float(i % 7),
                "lead_time_days": 3.0, "lag_1": float(i % 5),
                "roll_mean_7": float(i % 4), "demand_segment": "smooth",
                "is_delivery_day": bool(i % 2),
                "feat_a": float(i), "feat_b": float(i % 3), "cat_idx": i % 3,
            })
        return modeling_prep.assign_folds(pd.DataFrame(rows))

    def test_it_plugs_into_run_fold_unchanged(self):
        results = walk_forward.run_fold(
            self._panel(), 1,
            xgb.make_fit_predict({**xgb.DEFAULT_PARAMS, "max_depth": 3,
                                  "min_child_weight": 1},
                                 feature_cols=FEATURES, max_rounds=30,
                                 tail_days=14),
            model_name="xgboost",
        )
        self.assertIn("xgboost", set(results["model"]))
        self.assertTrue(results["pinball"].notna().all())

    def test_no_training_row_reaches_december(self):
        frame = self._panel(periods=300)
        seen_max = []
        fit_predict = xgb.make_fit_predict(
            {**xgb.DEFAULT_PARAMS, "max_depth": 3, "min_child_weight": 1},
            feature_cols=FEATURES, max_rounds=20, tail_days=14,
        )

        def spy(train, valid):
            seen_max.append(train["Tanggal"].max())
            return fit_predict(train, valid)

        walk_forward.run_walk_forward(frame, spy, model_name="xgboost")
        for stamp in seen_max:
            self.assertLess(stamp, pd.Timestamp("2025-12-01"))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python3 -m unittest test.test_model_xgboost.TestMakeFitPredict -v`
Expected: FAIL — `AttributeError: module 'utils.model_xgboost' has no attribute 'make_fit_predict'`

- [ ] **Step 3: Implement the estimator builder and the two-fit callable**

Append to `utils/model_xgboost.py`:

```python
def build_estimator(
    params: dict,
    n_estimators: int,
    enable_categorical: bool = False,
    early_stopping_rounds: Optional[int] = None,
    quantile: float = QUANTILE,
) -> XGBRegressor:
    """A regressor whose training objective is the metric it is judged on."""
    kwargs = {key: params[key] for key in ESTIMATOR_KEYS if key in params}
    return XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        tree_method="hist",
        n_estimators=n_estimators,
        enable_categorical=enable_categorical,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=-1,
        **kwargs,
    )


def _target(frame: pd.DataFrame, params: dict) -> np.ndarray:
    values = frame[modeling_prep.TARGET_COL].to_numpy(dtype=float)
    return np.log1p(values) if params["log_target"] else values


def make_fit_predict(
    params: Optional[dict] = None,
    feature_cols: Optional[list] = None,
    quantile: float = QUANTILE,
    tail_days: int = ES_TAIL_DAYS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    max_rounds: int = MAX_ROUNDS,
    idx_cols: Optional[list] = None,
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """The callable walk_forward.run_fold() injects.

    Two fits. The first runs on the purged fit rows with the tail as its eval
    set and reports where early stopping landed. The second discards that
    booster and refits on every training row at exactly that round count, so
    the model that produces the reported predictions has seen the same
    population the Random Forest was trained on.

    Under `log_target`, the early-stopping metric is computed on the log
    scale. That is sound: early stopping only chooses a round count *within*
    one candidate. Candidates are compared to each other by pinball on the
    original scale, after inversion.

    Round counts are recorded on the returned callable rather than returned,
    because `walk_forward` accepts predictions and nothing else — and the
    spread of round counts across folds is worth reporting.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    feature_cols = feature_cols or modeling_prep.FEATURE_COLS

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
        model_common.assert_no_nan(train, feature_cols)
        model_common.assert_no_nan(valid, feature_cols)

        fit_rows, es_rows = split_early_stopping(train, tail_days=tail_days)
        fit_X, es_X, enable = encode(fit_rows[feature_cols], es_rows[feature_cols],
                                     params["encoding"], idx_cols=idx_cols)
        probe = build_estimator(params, max_rounds, enable_categorical=enable,
                                early_stopping_rounds=early_stopping_rounds,
                                quantile=quantile)
        probe.fit(fit_X, _target(fit_rows, params),
                  eval_set=[(es_X, _target(es_rows, params))], verbose=False)
        best_iteration = int(probe.best_iteration) + 1
        fit_predict.best_iterations.append(best_iteration)

        train_X, valid_X, enable = encode(train[feature_cols], valid[feature_cols],
                                          params["encoding"], idx_cols=idx_cols)
        model = build_estimator(params, best_iteration, enable_categorical=enable,
                                quantile=quantile)
        model.fit(train_X, _target(train, params), verbose=False)

        prediction = model.predict(valid_X)
        if params["log_target"]:
            prediction = modeling_prep.inverse_log_target(prediction)
        # A negative shipment quantity is not a thing.
        return np.clip(np.asarray(prediction, dtype=float), 0.0, None)

    fit_predict.best_iterations = []
    return fit_predict
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/bin/python3 -m unittest test.test_model_xgboost -v`
Expected: PASS — every test in the module green, including the new TestMakeFitPredict and TestWalkForwardIntegration classes.

- [ ] **Step 5: Commit**

```bash
git add utils/model_xgboost.py test/test_model_xgboost.py
git commit -m "feat: fit XGBoost twice so the refit sees every training row

Early stopping picks the round count on the purged tail, then that booster is
thrown away and the model refits on the full training rows at exactly that
count. Keeping the first booster would have scored XGBoost on a population
missing its most recent 30 days — a different population than the Random
Forest saw, which would contaminate the comparison with a data difference.

Round counts are recorded on the callable, since walk_forward accepts
predictions and nothing else.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Final fit, bundle, and prediction

**Files:**
- Modify: `utils/model_xgboost.py`
- Modify: `test/test_model_xgboost.py`

**Interfaces:**
- Consumes: `walk_forward.eligible_rows`, `purging.lookahead_safe_mask`, `modeling_prep.TEST_START`, `model_common.save_bundle`/`load_bundle`.
- Produces:
  - `xgb.MODEL_FILE`, `xgb.BEST_PARAMS_FILE`, `xgb.SEARCH_FILE`, `xgb.RESULTS_FILE` (str paths)
  - `xgb.fit_final(df, params: dict, feature_cols: Optional[list] = None, tail_days: int = ES_TAIL_DAYS, early_stopping_rounds: int = EARLY_STOPPING_ROUNDS, max_rounds: int = MAX_ROUNDS, idx_cols: Optional[list] = None, date_col: str = modeling_prep.DATE_COL, test_start: pd.Timestamp = modeling_prep.TEST_START) -> dict` with keys `model`, `params`, `feature_cols`, `columns`, `categories`, `idx_cols`, `encoding`, `log_target`, `best_iteration`, `quantile`, `n_train`
  - `xgb.predict_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray`
  - `xgb.save_bundle(bundle, path=MODEL_FILE)`, `xgb.save_best_params(params, path=BEST_PARAMS_FILE)`, `xgb.load_bundle(path=MODEL_FILE)`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_model_xgboost.py`:

```python
import tempfile
from pathlib import Path


class TestFitFinal(unittest.TestCase):
    def _params(self, **overrides):
        return {**xgb.DEFAULT_PARAMS, "max_depth": 3, "learning_rate": 0.3,
                "min_child_weight": 1, "random_state": 0, **overrides}

    def _bundle(self, frame=None, **overrides):
        return xgb.fit_final(frame if frame is not None else _dated_frame(400),
                             self._params(**overrides), feature_cols=FEATURES,
                             max_rounds=40, tail_days=14, idx_cols=["cat_idx"])

    def test_bundle_records_what_prediction_needs(self):
        bundle = self._bundle()
        for key in ("model", "params", "feature_cols", "columns", "categories",
                    "idx_cols", "encoding", "log_target", "best_iteration",
                    "quantile", "n_train"):
            self.assertIn(key, bundle)
        self.assertEqual(bundle["feature_cols"], FEATURES)
        self.assertEqual(bundle["quantile"], xgb.QUANTILE)

    def test_training_stops_before_december(self):
        frame = _dated_frame(400)
        bundle = self._bundle(frame)
        eligible = frame[frame["Tanggal"] < pd.Timestamp("2025-12-01")]
        self.assertLessEqual(bundle["n_train"], len(eligible))
        self.assertGreater(bundle["n_train"], 0)

    def test_the_december_boundary_is_purged(self):
        """lead_time_days is 3, so 2025-11-29 onward is contaminated."""
        frame = _dated_frame(400)
        bundle = self._bundle(frame)
        safe = frame[frame["Tanggal"] <= pd.Timestamp("2025-11-27")]
        self.assertLessEqual(bundle["n_train"], len(safe))

    def test_rows_without_a_target_are_dropped(self):
        frame = _dated_frame(400)
        blank = frame["Tanggal"].between("2025-06-01", "2025-06-05")
        frame.loc[blank, "target_lead_time_cumulative"] = np.nan
        bundle = self._bundle(frame)
        reference = self._bundle(_dated_frame(400))
        self.assertEqual(bundle["n_train"], reference["n_train"] - int(blank.sum()))

    def test_the_warmup_window_is_excluded(self):
        frame = _dated_frame(400)
        bundle = self._bundle(frame)
        expected = walk_forward.eligible_rows(frame)
        expected = expected[purging.lookahead_safe_mask(
            expected, pd.Timestamp("2025-12-01"))]
        self.assertEqual(bundle["n_train"], len(expected))

    def test_the_final_model_is_trained_on_every_eligible_row(self):
        """Tail included: the tail chooses the round count, then rejoins."""
        bundle = self._bundle()
        self.assertEqual(bundle["model"].n_estimators, bundle["best_iteration"])

    def test_predict_bundle_returns_one_value_per_row(self):
        frame = _dated_frame(400)
        bundle = self._bundle(frame)
        self.assertEqual(xgb.predict_bundle(bundle, frame).shape, (len(frame),))

    def test_predict_bundle_is_non_negative(self):
        frame = _dated_frame(400)
        bundle = self._bundle(frame)
        self.assertTrue((xgb.predict_bundle(bundle, frame) >= 0).all())

    def test_predict_bundle_ignores_the_input_column_order(self):
        for encoding in ("ordinal", "native", "one_hot"):
            frame = _dated_frame(400)
            bundle = self._bundle(frame, encoding=encoding)
            shuffled = frame[list(reversed(frame.columns))]
            np.testing.assert_allclose(
                xgb.predict_bundle(bundle, frame),
                xgb.predict_bundle(bundle, shuffled),
                err_msg=encoding,
            )

    def test_a_saved_bundle_predicts_identically_after_loading(self):
        frame = _dated_frame(400)
        bundle = self._bundle(frame)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "bundle.joblib")
            xgb.save_bundle(bundle, path)
            reloaded = xgb.load_bundle(path)
        np.testing.assert_allclose(xgb.predict_bundle(bundle, frame),
                                   xgb.predict_bundle(reloaded, frame))

    def test_log_target_bundles_predict_on_the_original_scale(self):
        frame = _dated_frame(400)
        raw = self._bundle(frame, log_target=False)
        logged = self._bundle(frame, log_target=True)
        self.assertLess(
            abs(xgb.predict_bundle(logged, frame).mean()
                - xgb.predict_bundle(raw, frame).mean()),
            20.0,
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python3 -m unittest test.test_model_xgboost.TestFitFinal -v`
Expected: FAIL — `AttributeError: module 'utils.model_xgboost' has no attribute 'fit_final'`

- [ ] **Step 3: Implement the final fit and prediction**

Append to `utils/model_xgboost.py`:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = str(BASE_DIR / "models/xgboost_q90.joblib")
BEST_PARAMS_FILE = str(BASE_DIR / "dataset/model_ready/xgb_best_params.json")
SEARCH_FILE = str(BASE_DIR / "dataset/model_ready/xgb_search_results.csv")
RESULTS_FILE = str(BASE_DIR / "dataset/model_ready/xgb_walk_forward_results.csv")


def fit_final(
    df: pd.DataFrame,
    params: dict,
    feature_cols: Optional[list] = None,
    tail_days: int = ES_TAIL_DAYS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    max_rounds: int = MAX_ROUNDS,
    idx_cols: Optional[list] = None,
    date_col: str = modeling_prep.DATE_COL,
    test_start: pd.Timestamp = modeling_prep.TEST_START,
) -> dict:
    """Fit on every eligible row before December, purged at that boundary.

    Eligibility comes from `walk_forward.eligible_rows`, not from a date filter
    written here. The rows this model is finally trained on have to be the rows
    it was scored on, and the scoring cuts are not just the date: the first 28
    days of each segment have no usable lag window, and the last few days have
    no target at all, because the lead-time sum runs past the end of the data.

    Same two-fit protocol as walk-forward. The bundle records the training
    column order, the encoding, and the category levels, because a booster
    reloaded next month against a different layout does not fail — it predicts
    confidently from the wrong features, which is worse.
    """
    params = {**DEFAULT_PARAMS, **params}
    feature_cols = feature_cols or modeling_prep.FEATURE_COLS

    frame = walk_forward.eligible_rows(df, date_col=date_col, test_start=test_start)
    frame = frame[purging.lookahead_safe_mask(frame, test_start, date_col=date_col)]
    model_common.assert_no_nan(frame, feature_cols)

    fit_rows, es_rows = split_early_stopping(frame, tail_days=tail_days,
                                             date_col=date_col)
    fit_X, es_X, enable = encode(fit_rows[feature_cols], es_rows[feature_cols],
                                 params["encoding"], idx_cols=idx_cols)
    probe = build_estimator(params, max_rounds, enable_categorical=enable,
                            early_stopping_rounds=early_stopping_rounds)
    probe.fit(fit_X, _target(fit_rows, params),
              eval_set=[(es_X, _target(es_rows, params))], verbose=False)
    best_iteration = int(probe.best_iteration) + 1

    train_X, _, enable = encode(frame[feature_cols], frame[feature_cols],
                                params["encoding"], idx_cols=idx_cols)
    model = build_estimator(params, best_iteration, enable_categorical=enable)
    model.fit(train_X, _target(frame, params), verbose=False)

    return {
        "model": model,
        "params": params,
        "feature_cols": feature_cols,
        "columns": list(train_X.columns),
        "categories": training_categories(frame[feature_cols], idx_cols=idx_cols),
        "idx_cols": idx_cols,
        "encoding": params["encoding"],
        "log_target": params["log_target"],
        "best_iteration": best_iteration,
        "quantile": QUANTILE,
        "n_train": int(len(frame)),
    }


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predict with a fitted bundle, forcing the recorded column order."""
    features, _ = apply_encoding(frame[bundle["feature_cols"]],
                                 bundle["encoding"], bundle["columns"],
                                 bundle["categories"],
                                 idx_cols=bundle["idx_cols"])
    prediction = bundle["model"].predict(features)
    if bundle["log_target"]:
        prediction = modeling_prep.inverse_log_target(prediction)
    return np.clip(np.asarray(prediction, dtype=float), 0.0, None)


def save_bundle(bundle: dict, path: str = MODEL_FILE) -> None:
    model_common.save_bundle(bundle, path)


def load_bundle(path: str = MODEL_FILE) -> dict:
    return model_common.load_bundle(path)


def save_best_params(params: dict, path: str = BEST_PARAMS_FILE) -> None:
    model_common.save_best_params(params, path)
```

Move the `from pathlib import Path` line up to the module's import block rather than leaving it mid-file.

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/bin/python3 -m unittest test.test_model_xgboost -v`
Expected: PASS — every test in the module green, including the new TestFitFinal class.

- [ ] **Step 5: Commit**

```bash
git add utils/model_xgboost.py test/test_model_xgboost.py
git commit -m "feat: add the final XGBoost fit and a self-describing bundle

Row population comes from walk_forward.eligible_rows plus the December purge,
so the shipped model is trained on exactly the rows the reported metrics
describe — not on every row before December, which would include warm-up rows
with no usable lag window and tail rows whose label is NaN.

The bundle carries the column order, the encoding, and the category levels.
Category codes are positional, so rebuilding a column from a different level
list silently renumbers every row.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The search wrappers

**Files:**
- Modify: `utils/model_xgboost.py`
- Modify: `test/test_model_xgboost.py`

**Interfaces:**
- Consumes: `model_common.sample_search_space`, `model_common.run_search`, `model_common.select_best`.
- Produces:
  - `xgb.SEARCH_FOLDS = (3, 5)`, `xgb.N_CANDIDATES = 30`
  - `xgb.sample_search_space(n_candidates: int = N_CANDIDATES, seed: int = 42, space: Optional[dict] = None) -> list`
  - `xgb.run_search(df, candidates, folds=SEARCH_FOLDS, alpha=QUANTILE, model_name="xgboost", feature_cols=None, verbose=True, checkpoint_path=None, resume=True) -> pd.DataFrame`
  - `xgb.select_best(search_results, candidates) -> dict` (re-export)

- [ ] **Step 1: Write the failing tests**

Append to `test/test_model_xgboost.py`:

```python
class TestSearchWrappers(unittest.TestCase):
    def test_the_default_budget_is_thirty_candidates(self):
        self.assertEqual(len(xgb.sample_search_space()), xgb.N_CANDIDATES)
        self.assertEqual(xgb.N_CANDIDATES, 30)

    def test_the_same_seed_reproduces_the_same_list(self):
        self.assertEqual(xgb.sample_search_space(6, seed=1),
                         xgb.sample_search_space(6, seed=1))

    def test_candidates_are_distinct(self):
        keys = sorted(xgb.SEARCH_SPACE)
        drawn = xgb.sample_search_space(20, seed=1)
        signatures = {tuple(c[k] for k in keys) for c in drawn}
        self.assertEqual(len(signatures), 20)

    def test_every_candidate_carries_a_full_parameter_set(self):
        for candidate in xgb.sample_search_space(5, seed=1):
            for key in xgb.DEFAULT_PARAMS:
                self.assertIn(key, candidate)

    def test_only_searched_parameters_vary(self):
        candidates = xgb.sample_search_space(20, seed=1)
        self.assertEqual({c["random_state"] for c in candidates},
                         {xgb.DEFAULT_PARAMS["random_state"]})

    def test_the_search_folds_are_three_and_five(self):
        self.assertEqual(xgb.SEARCH_FOLDS, (3, 5))

    def test_run_search_scores_every_candidate(self):
        rows = []
        for i, date in enumerate(pd.date_range("2025-05-01", periods=245, freq="D")):
            rows.append({
                "Kode Barang": "I1", "Nama Cabang": "B1", "segment_id": 1,
                "Tanggal": date,
                "target_lead_time_cumulative": float(i % 7),
                "lead_time_days": 3.0, "lag_1": float(i % 5),
                "roll_mean_7": float(i % 4), "demand_segment": "smooth",
                "is_delivery_day": bool(i % 2),
                "feat_a": float(i), "feat_b": float(i % 3), "cat_idx": i % 3,
            })
        panel = modeling_prep.assign_folds(pd.DataFrame(rows))
        candidates = [{**xgb.DEFAULT_PARAMS, "max_depth": d, "min_child_weight": 1}
                      for d in (3, 4)]
        results = xgb.run_search(panel, candidates, folds=(1,),
                                 feature_cols=FEATURES, verbose=False)
        self.assertEqual(list(results["candidate_id"]), [0, 1])
        self.assertTrue(results["pinball"].notna().all())
        for key in xgb.SEARCH_SPACE:
            self.assertIn(key, results.columns)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python3 -m unittest test.test_model_xgboost.TestSearchWrappers -v`
Expected: FAIL — `AttributeError: module 'utils.model_xgboost' has no attribute 'sample_search_space'`

- [ ] **Step 3: Implement the wrappers**

Append to `utils/model_xgboost.py`:

```python
SEARCH_FOLDS = (3, 5)

# More draws than the Random Forest's 18, because this space has more
# dimensions that genuinely move the score — not because XGBoost is being
# given an easier ride. The asymmetry is reported in docs/hasil-modeling-xgb.md.
N_CANDIDATES = 30

select_best = model_common.select_best


def sample_search_space(
    n_candidates: int = N_CANDIDATES,
    seed: int = 42,
    space: Optional[dict] = None,
) -> list:
    """Distinct parameter sets drawn at random from SEARCH_SPACE.

    No affordability screen: `hist` holds a quantized feature matrix — tens of
    megabytes at this size — so there is no analogue of the quantile forest's
    leaf-storage bound to screen against.
    """
    return model_common.sample_search_space(
        space=SEARCH_SPACE if space is None else space,
        defaults=DEFAULT_PARAMS,
        n_candidates=n_candidates,
        seed=seed,
        screen=None,
    )


def run_search(
    df: pd.DataFrame,
    candidates: list,
    folds: tuple = SEARCH_FOLDS,
    alpha: float = QUANTILE,
    model_name: str = "xgboost",
    feature_cols: Optional[list] = None,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every XGBoost candidate on the search folds.

    XGBoostError joins the caught types: a candidate whose parameter
    combination the library rejects should be recorded and skipped, exactly
    like an over-budget forest, rather than ending a multi-hour run.
    """
    return model_common.run_search(
        df, candidates, make_fit_predict=make_fit_predict,
        search_space=SEARCH_SPACE, folds=folds, alpha=alpha,
        model_name=model_name, feature_cols=feature_cols, verbose=verbose,
        checkpoint_path=checkpoint_path, resume=resume,
        catch=(MemoryError, ValueError, XGBoostError),
    )
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/bin/python3 -m unittest test.test_model_xgboost -v`
Expected: PASS — every test in the module green, including the new TestSearchWrappers class.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v 2>&1 | tail -5`
Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add utils/model_xgboost.py test/test_model_xgboost.py
git commit -m "feat: wire XGBoost into the shared search protocol

Thirty candidates from a 2,592-combination space, scored on folds 3 and 5,
checkpointed and resumable — the same protocol the Random Forest ran, now
supplied by model_common rather than copied.

No affordability screen: hist has no analogue of the quantile forest's leaf
storage. XGBoostError joins the caught exception types so a combination the
library rejects is recorded and skipped instead of ending a multi-hour run.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: The notebook

**Files:**
- Create: `notebook/modeling_xgb.ipynb`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything from Tasks 1–6, plus `walk_forward.run_walk_forward`, `walk_forward.pooled_metric`, `walk_forward.GROUP_COLS`.
- Produces: the four artifacts at `xgb.SEARCH_FILE`, `xgb.BEST_PARAMS_FILE`, `xgb.RESULTS_FILE`, `xgb.MODEL_FILE`.

- [ ] **Step 1: Write the notebook**

Create `notebook/modeling_xgb.ipynb` by writing this Python script and converting it. The notebook must stay thin — all logic lives in `utils/`, so the script path and the notebook path cannot diverge. Write `/tmp/build_xgb_nb.py`:

```python
import json
from pathlib import Path

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().split("\n")})

def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip().split("\n")})

md("""
# Modeling — XGBoost (quantile 0.9)

Desain: `docs/superpowers/specs/2026-08-19-xgboost-modeling-design.md`.
Rencana: `docs/superpowers/plans/2026-08-19-xgboost-modeling.md`.

Notebook ini tipis dengan sengaja. Semua logika ada di `utils/walk_forward.py`,
`utils/model_common.py` dan `utils/model_xgboost.py`, supaya jalur skrip dan
jalur notebook tidak bisa berbeda.

**Desember 2025 terkunci** dan tidak dinilai di sini.
""")

code("""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd

from utils import evaluation, model_xgboost as xgb
from utils import modeling_prep, walk_forward

df = pd.read_parquet(modeling_prep.MODEL_INPUT_FILE)
print(f"{len(df):,} rows x {df.shape[1]} columns")
""")

md("""
## Benchmark

Satu putaran dua-fit di fold 5 dengan `DEFAULT_PARAMS`, untuk mengukur ongkos
sebelum 60 fit pencarian dijalankan dan melihat di ronde berapa early stopping
mendarat. Angkanya dicatat di `docs/hasil-modeling-xgb.md`.
""")

code("""
import resource
import time

split = walk_forward.prepare_fold(df, 5)
train, valid = split["train"], split["valid"]
fit_rows, es_rows = xgb.split_early_stopping(train)
print(f"train {len(train):,} rows -> fit {len(fit_rows):,} + tail {len(es_rows):,}")
print(f"valid {len(valid):,} rows")

fit_predict = xgb.make_fit_predict(dict(xgb.DEFAULT_PARAMS))
start = time.time()
prediction = fit_predict(train, valid)
elapsed = time.time() - start

peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
print(f"best_iteration {fit_predict.best_iterations[0]} of {xgb.MAX_ROUNDS}")
print(f"wall time {elapsed / 60:.1f} min (both fits)")
print(f"peak RSS  {peak_bytes / 1024 ** 3:.2f} GB")
print(f"prediction mean {prediction.mean():.2f}, max {prediction.max():.2f}")
""")

md("""
## Pencarian hyperparameter

30 kandidat dari ruang 2.592 kombinasi, dinilai di fold 3 dan 5 dengan
pinball@0.9 gabungan. Hasil yang dilaporkan datang dari walk-forward lima fold
di bawah, bukan dari sini — menilai di fold yang memilih pemenang akan
optimistis.
""")

code("""
candidates = xgb.sample_search_space(xgb.N_CANDIDATES, seed=42)
search_results = xgb.run_search(df, candidates, folds=xgb.SEARCH_FOLDS,
                                checkpoint_path=xgb.SEARCH_FILE)
search_results.to_csv(xgb.SEARCH_FILE, index=False)
search_results.sort_values("pinball").head(10)
""")

md("""
## Walk-forward final

Konfigurasi pemenang di kelima fold, melawan ketiga baseline naive pada baris
yang identik.
""")

code("""
best = xgb.select_best(search_results, candidates)
xgb.save_best_params(best)
print(best)

fit_predict = xgb.make_fit_predict(best)
results = walk_forward.run_walk_forward(df, fit_predict, model_name="xgboost")
results.to_csv(xgb.RESULTS_FILE, index=False)

print("best_iteration per fold:", fit_predict.best_iterations)

overall = results[results["group_col"].isna()]
overall.pivot_table(index="model", columns="fold_id", values="pinball").round(3)
""")

code("""
bundle = xgb.fit_final(df, best)
xgb.save_bundle(bundle)
print(f"trained on {bundle['n_train']:,} rows, "
      f"{len(bundle['columns'])} columns, encoding {bundle['encoding']}, "
      f"{bundle['best_iteration']} rounds, quantile {bundle['quantile']}")
""")

md("""
## Hasil

Tiga potongan, masing-masing melawan ketiga baseline naive pada baris identik.
Satu angka global menyesatkan di data yang 44% targetnya nol.
""")

code("""
results = pd.read_csv(xgb.RESULTS_FILE)

print("=== per fold (overall) ===")
print(results[results["group_col"].isna()]
      .pivot_table(index="model", columns="fold_id", values="pinball").round(3))

for group_col in walk_forward.GROUP_COLS:
    print(f"\\n=== per {group_col} (pooled over folds) ===")
    grouped = results[results["group_col"] == group_col]
    table = (grouped.assign(weighted=grouped["pinball"] * grouped["n"])
                    .groupby(["model", "group_value"], observed=True)
                    .apply(lambda part: part["weighted"].sum() / part["n"].sum())
                    .unstack())
    print(table.round(3))

print("\\n=== coverage and fill rate (overall, pooled) ===")
for model in results["model"].unique():
    print(f"{model:20s} "
          f"coverage {walk_forward.pooled_metric(results, model, 'coverage'):6.3f}  "
          f"fill_rate {walk_forward.pooled_metric(results, model, 'fill_rate'):6.3f}  "
          f"shortfall {walk_forward.pooled_metric(results, model, 'shortfall_units'):9.1f}")
""")

md("""
## Head-to-head lawan Random Forest

Sah dilakukan karena kedua model dinilai di baris yang identik — dijamin
`walk_forward.eligible_rows()`, bukan oleh disiplin. Fold 1, 2, 4 adalah
potongan yang bersih: keduanya memilih pemenang di fold 3 dan 5.
""")

code("""
from utils import model_random_forest as rf

rf_results = pd.read_csv(rf.RESULTS_FILE)
combined = pd.concat([results, rf_results], ignore_index=True)

for label, folds in (("semua fold", None), ("fold 1/2/4 (bersih)", (1, 2, 4))):
    print(f"=== {label} ===")
    for model in ("xgboost", "random_forest", "naive_roll_mean_7"):
        rows = combined[(combined["model"] == model) & combined["group_col"].isna()]
        if rows.empty:
            continue
        print(f"{model:20s} "
              f"pinball {walk_forward.pooled_metric(combined, model, 'pinball', folds):6.3f}  "
              f"mae {walk_forward.pooled_metric(combined, model, 'mae', folds):7.3f}  "
              f"coverage {walk_forward.pooled_metric(combined, model, 'coverage', folds):6.3f}")
    print()
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "forecast-scm", "language": "python",
                       "name": "forecast-scm"},
        "language_info": {"name": "python", "version": "3.9.6"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path("notebook/modeling_xgb.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print("written")
```

Run: `.venv/bin/python3 /tmp/build_xgb_nb.py`
Expected: `written`.

- [ ] **Step 2: Verify the notebook parses and its imports resolve**

Run:
```bash
.venv/bin/python3 -c "
import json
nb = json.load(open('notebook/modeling_xgb.ipynb'))
print(len(nb['cells']), 'cells')
"
.venv/bin/python3 -c "from utils import model_xgboost as xgb, model_common, walk_forward; print('imports ok')"
```
Expected: `11 cells` and `imports ok`.

- [ ] **Step 3: Confirm the kernel name matches the Random Forest notebook**

Run: `.venv/bin/python3 -c "
import json
for name in ('modeling_rf', 'modeling_xgb'):
    nb = json.load(open(f'notebook/{name}.ipynb'))
    print(name, nb['metadata']['kernelspec']['name'])
"`
Expected: both print the same kernel name. If `modeling_rf.ipynb` uses a different one, edit `modeling_xgb.ipynb`'s `kernelspec` to match — commit `104a7b3` pointed the notebooks at the project venv kernel and the new notebook must follow.

- [ ] **Step 4: Update `CLAUDE.md`**

In the "Project state" section, after the sentence ending "`docs/hasil-modeling-rf.md` for the measured results.", add:

```markdown
`utils/model_common.py` holds the parts of that machinery no single model owns — the random search with its checkpoint/resume, one-hot expansion, and the bundle format — and `utils/model_xgboost.py` supplies the second model: a 0.9-quantile XGBoost (`reg:quantileerror`) whose boosting rounds come from early stopping on a purged 30-day tail of each fold's training window, then a refit on the full training rows so it is trained on the same population the forest saw. See `docs/superpowers/specs/2026-08-19-xgboost-modeling-design.md` and `docs/hasil-modeling-xgb.md`.
```

In the "Commands" section, after the Random Forest notebook line, add:

```markdown
- Run the XGBoost modeling notebook (benchmark, search, final walk-forward; takes hours): `.venv/bin/python3 -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook/modeling_xgb.ipynb`
```

- [ ] **Step 5: Commit**

```bash
git add notebook/modeling_xgb.ipynb CLAUDE.md
git commit -m "feat: add the XGBoost modeling notebook

Mirrors modeling_rf.ipynb — benchmark, search, final walk-forward, results —
and adds a head-to-head cell that pools both models' recorded result tables.
That comparison is only meaningful because walk_forward.eligible_rows()
guarantees both were scored on identical rows.

Thin on purpose: every line of logic lives in utils/, so the script path and
the notebook path cannot drift apart.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Run it and write the results

**Files:**
- Modify: `notebook/modeling_xgb.ipynb` (executed, then outputs cleared)
- Create: `docs/hasil-modeling-xgb.md`

**Interfaces:**
- Consumes: everything above.
- Produces: measured numbers; no new code interfaces.

**This task takes hours of wall time.** The search checkpoints after each candidate, so an interrupted run resumes from `dataset/model_ready/xgb_search_results.csv`.

- [ ] **Step 1: Confirm December is still untouched before spending the compute**

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd
from utils import walk_forward, modeling_prep
df = pd.read_parquet(modeling_prep.MODEL_INPUT_FILE)
frame = walk_forward.eligible_rows(df)
print('max eligible date:', frame['Tanggal'].max())
assert frame['Tanggal'].max() < pd.Timestamp('2025-12-01')
print('rows:', len(frame))
"
```
Expected: a max date in November 2025 and `rows: 345547`-ish. If December appears, stop and fix before running anything.

- [ ] **Step 2: Execute the notebook**

Run:
```bash
.venv/bin/python3 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebook/modeling_xgb.ipynb
```
Expected: completes without error. Monitor progress by tailing `dataset/model_ready/xgb_search_results.csv` — one row appears per finished candidate.

- [ ] **Step 3: Confirm all four artifacts exist**

Run: `ls -la dataset/model_ready/xgb_* models/xgboost_q90.joblib`
Expected: `xgb_search_results.csv` (30 rows), `xgb_best_params.json`, `xgb_walk_forward_results.csv`, `xgboost_q90.joblib`.

Then: `.venv/bin/python3 -c "
import pandas as pd
s = pd.read_csv('dataset/model_ready/xgb_search_results.csv')
print(len(s), 'candidates,', s['error'].notna().sum(), 'failed')
r = pd.read_csv('dataset/model_ready/xgb_walk_forward_results.csv')
print(sorted(r['model'].unique()), sorted(r['fold_id'].unique()))
"`
Expected: 30 candidates, the four model names, folds 1–5.

- [ ] **Step 4: Collect the numbers the write-up needs**

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd
from utils import walk_forward, model_xgboost as xgb, model_random_forest as rf

x = pd.read_csv(xgb.RESULTS_FILE); r = pd.read_csv(rf.RESULTS_FILE)
both = pd.concat([x, r], ignore_index=True)
for label, folds in (('all folds', None), ('folds 1/2/4', (1,2,4))):
    print('==', label)
    for m in sorted(both['model'].unique()):
        print(f'{m:20s}',
              ' '.join(f'{k} {walk_forward.pooled_metric(both, m, k, folds):9.3f}'
                       for k in ('pinball','mae','coverage','fill_rate',
                                 'shortfall_units','overstock_units')))
" | tee /tmp/xgb_summary.txt
```
Expected: a table you can transcribe. Also capture the per-fold and per-group slices the same way the Random Forest write-up presents them.

- [ ] **Step 5: Write `docs/hasil-modeling-xgb.md`**

Follow the structure of `docs/hasil-modeling-rf.md` — read it first and match its register: Indonesian, measured numbers only, every claim traceable to an artifact. Sections, in order:

1. **Ringkasan** — XGBoost's pooled pinball@0.9 against `naive_roll_mean_7` and against the Random Forest's 2.410, plus the shortfall/overstock trade in units, with the mixed-units caveat the RF document carries.
2. **Setup evaluasi** — the same table (data, features, target, folds, locked test, eliminated rows, validation rows, quantile, implementation), with `reg:quantileerror` and the two-fit round protocol named.
3. **Benchmark** — wall time for both fits, peak RSS, and where early stopping landed relative to `MAX_ROUNDS = 2000`. If it landed at the ceiling, say so plainly: it means the cap bound the search rather than the data did.
4. **Pencarian hyperparameter** — the 30 candidates, how many failed, the top and bottom of the table, and what the spread says. State explicitly which `encoding` won and whether the three modes separated at all — that is the question the flag existed to answer.
5. **Hasil walk-forward** — per fold, per `demand_segment`, per `is_delivery_day`, each against the three naive baselines, plus the folds-1/2/4 slice.
6. **Head-to-head lawan Random Forest** — both models on all five folds and on folds 1/2/4. State the two asymmetries in the section itself: **30 candidates vs 18**, and **early stopping vs a pinned tree count**. If the margin is within a few percent, say that these are live rival explanations rather than claiming a winner.
7. **Model final** — `n_train`, column count, encoding, `best_iteration`, file size and date.
8. **Reproduksi** — the nbconvert command and the artifact table, matching the RF document's.
9. **Batasan** — December still closed; the pickup-date ceiling (`docs/batasan-penelitian.md` B-1/B-2/B-3); MAE not comparable across a quantile model and a mid-point baseline; folds 3 and 5 chose the winner; and now two of three models exist, so the LSTM is still outstanding before any final recommendation.

- [ ] **Step 6: Clear the notebook outputs**

The evidence lives in the CSVs and in `docs/`, not in cell output that a stray "Clear All Outputs" can erase — the same reason `modeling_rf.ipynb` is committed clean.

Run: `.venv/bin/python3 -m nbconvert --ClearOutputPreprocessor.enabled=True --to notebook --inplace notebook/modeling_xgb.ipynb`
Expected: `[NbConvertApp] Writing ... notebook/modeling_xgb.ipynb`

- [ ] **Step 7: Run the full test suite one last time**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v 2>&1 | tail -5`
Expected: `OK`.

- [ ] **Step 8: Commit**

```bash
git add docs/hasil-modeling-xgb.md notebook/modeling_xgb.ipynb
git commit -m "docs: record the measured XGBoost results

Numbers from a full run of notebook/modeling_xgb.ipynb, kept in git rather
than in notebook cell output that a clear-all can erase.

The head-to-head against the Random Forest carries its own caveats in the
text: 30 search candidates against 18, and early stopping against a pinned
tree count. Both are live rival explanations for a narrow margin, so they are
stated where the comparison is made rather than buried in a limitations list.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Done when

- `.venv/bin/python3 -m unittest discover -p "test_*.py"` reports OK, with `test/test_model_random_forest.py` unmodified.
- `dataset/model_ready/xgb_search_results.csv` holds 30 scored candidates.
- `dataset/model_ready/xgb_walk_forward_results.csv` holds all four model names across folds 1–5.
- `models/xgboost_q90.joblib` exists and `xgb.predict_bundle()` runs against a reloaded copy.
- `docs/hasil-modeling-xgb.md` states XGBoost's pinball@0.9, the comparison against the Random Forest on folds 1/2/4, and both budget asymmetries.
- No artifact, log, or document contains a metric computed on a row dated 2025-12-01 or later.
