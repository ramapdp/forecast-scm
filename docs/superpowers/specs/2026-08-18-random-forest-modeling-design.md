# Random Forest modeling — design

## Purpose

Train and evaluate the first of the three candidate models against
`dataset/model_ready/model_input.parquet`, and build the walk-forward runner
that XGBoost and the LSTM will reuse unchanged.

The deliverable is a defensible answer to one question: **does a Random Forest
predicting the 0.9 quantile of `target_lead_time_cumulative` beat the naive
baselines, and by how much, per demand segment?**

This spec is the follow-up promised by
`docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md`, which
covered preprocessing only and explicitly deferred "training, hyperparameter
tuning, model comparison, and SHAP explainability".

## Background

### What already exists

The preprocessing spec landed in full. Every piece this design needs is
present and tested:

| Component | Location | What it gives us |
|---|---|---|
| `model_input.parquet` | `dataset/model_ready/` | 1,502,522 rows × 82 columns, 2024-01-01 → 2025-12-31, with `fold_id`, `demand_segment`, `is_event_driven`, encoded categoricals |
| `FEATURE_COLS` | `utils/modeling_prep.py` | the 56 agreed model inputs, with `baseline_ratio`/`is_spike` deliberately excluded as day-H leakage |
| `fold_train_mask()` | `utils/modeling_prep.py` | training rows for fold *k*, already purged via `purging.lookahead_safe_mask()` |
| `to_tabular()` | `utils/modeling_prep.py` | warm-up cut (28 days) + null-target drop + `X`/`y`/`keys`/`fold_id` |
| `impute_features()` | `utils/modeling_prep.py` | sentinel 99 for event-proximity nulls, indicators for `days_since_relocation` / `baseline_ratio` |
| `evaluation.py` | `utils/` | `mae`, `pinball_loss`, `quantile_coverage`, `fill_rate`, `shortfall_units`, `overstock_units`, and the three naive baselines |

Fold sizes: 76,263 / 76,266 / 70,982 / 69,392 / 61,165 validation rows for
folds 1–5. Everything before July 2025 (1,148,454 rows, `fold_id` NaN) is
train-only in every fold, and December 2025 is the locked final test set.

Demand segments: intermittent 581,088, lumpy 514,044, erratic 221,181, smooth
186,209.

### What is settled and inherited

These come from the preprocessing spec and are **not** reopened here:

- **Primary target** — `target_lead_time_cumulative`.
- **Loss and service level** — pinball loss at **quantile 0.9, uniform across
  every SKU** (data owner, 2026-08-16).
- **Validation** — walk-forward, 5 expanding folds (Jul–Nov 2025). December
  2025 opened exactly once, after all three models have been compared.
- **Feature set** — `modeling_prep.FEATURE_COLS`, identical for every model.

## Decisions

Settled during brainstorming 2026-08-18.

| Decision | Choice | Rationale |
|---|---|---|
| Quantile mechanism | `quantile-forest`'s `RandomForestQuantileRegressor` | sklearn's `RandomForestRegressor` minimizes squared/absolute error only — it has no quantile loss. A quantile regression forest (Meinshausen 2006) reads the 0.9 quantile off the empirical distribution stored in each leaf, which is the correct estimator for the locked service level |
| Scope | Shared walk-forward runner + Random Forest + hyperparameter tuning | The runner is written once and reused by XGBoost and the LSTM, so the comparison is enforced structurally rather than by discipline — the same argument the preprocessing spec made for its adapters |
| Tuning budget | Random search, 18 candidates, scored on folds 3 and 5; winner refit and reported over all 5 folds | A full grid over 5 folds at 1.28M training rows is not runnable on the available hardware. Reporting the search folds' own scores would be optimistic, so the reported number comes from the full walk-forward |
| Model granularity | One global model | `demand_segment_idx` is already a feature, so the trees can split on it themselves. Maximum data per model, one artifact to ship weekly. Metrics are still reported per segment |
| Categorical encoding | Ordinal indices by default, one-hot as a searched flag | The preprocessing spec assumed one-hot expansion (§3.4) but never tested it. High-cardinality one-hot is known to dilute split quality and feature importance in forests; making it a hyperparameter settles the question with evidence instead of assumption |
| Target scale | `log_target` as a searched flag | Quantiles are equivariant under monotonic transforms, so inversion is exact and unbiased (preprocessing spec §3.6). What is actually being tested is the effect on split selection under a target whose 99th percentile is 488 and maximum 3,067 |

### Non-goals

- **December 2025.** Not touched, not scored, not looked at. It opens once,
  after XGBoost and the LSTM exist and a winner has been chosen.
- XGBoost and LSTM training — they consume this spec's runner, in their own
  specs.
- SHAP explainability — winner only, after the comparison.
- Per-segment specialist models. Recorded as a possible follow-up experiment,
  deliberately out of scope so the three-model comparison stays like-for-like.
- Any change to the 12 data-prep stages or to `modeling_prep.py`'s feature
  definitions.
- Serving/retraining infrastructure. The weekly run stays manual via the
  notebook, per the preprocessing spec's working default (item 8).

## Part 1 — Architecture

Two new modules and one notebook.

```
dataset/model_ready/model_input.parquet
        │
        ▼  utils/walk_forward.py          model-agnostic
   ┌──────────────────────────────────────────────────────┐
   │ for fold in 1..5:                                    │
   │   train = fold_train_mask(df, fold)   ← purged       │
   │   valid = df[fold_id == fold]                        │
   │   to_tabular() on both  ← warm-up cut, null-target   │
   │   y_pred = fit_predict(train, valid)  ← injected     │
   │   evaluation.score(...) per fold / segment / delivery│
   │   evaluation baselines on the SAME validation rows   │
   └──────────────────────────────────────────────────────┘
        │                              ▲
        │                              │ fit_predict
        ▼                              │
   results DataFrame          utils/model_random_forest.py
   (one row per fold × group)      RandomForestQuantileRegressor
                                   + search space + refit/predict
```

`walk_forward.py` knows nothing about Random Forest. Its only contact with a
model is a callable:

```python
fit_predict(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> np.ndarray
```

Everything that must be identical across the three model families lives in the
runner: which rows are eligible for training, the 28-day warm-up cut, dropping
rows with no target, and how scores are computed. If the LSTM later writes its
own loop, "Random Forest is 8% better" stops being a verifiable claim.

Both modules are pure in the same sense as `modeling_prep.py`: DataFrame in,
DataFrame or array out, no hidden I/O. Persistence is a separate, explicit
function.

### 1.1 `utils/walk_forward.py`

```python
FOLDS = (1, 2, 3, 4, 5)

def prepare_fold(df, fold_id, feature_cols, log_target=False) -> dict
def run_fold(df, fold_id, fit_predict, feature_cols, ...) -> pd.DataFrame
def run_walk_forward(df, fit_predict, folds=FOLDS, ...) -> pd.DataFrame
```

`prepare_fold()` returns `{"train": ..., "valid": ...}`, each the dict shape
`to_tabular()` already produces (`X`, `y`, `keys`, `fold_id`), plus the raw
validation frame needed for grouped reporting and for the naive baselines.

**Row eligibility, stated once:**

- Training rows: `fold_train_mask(df, fold_id)` — strictly before the fold's
  month start, with lookahead purging on. Purging matters here precisely
  because `target_lead_time_cumulative` sums over `H+1 .. H+lead_time_days`,
  so the last few days before a boundary carry labels built partly from the
  validation month.
- Validation rows: `df["fold_id"] == fold_id`.
- Both then pass through `to_tabular()`, which drops each segment's first 28
  days and every row whose target is null.
- Rows dated on or after `TEST_START` (2025-12-01) are filtered out of the
  frame before any fold runs. This is redundant with the two rules above and
  is kept as a hard guard because the cost of being wrong is the credibility
  of the final number.

**Output schema.** One long DataFrame, one row per (fold × group × model-or-
baseline):

| Column | Meaning |
|---|---|
| `model` | `"random_forest"`, `"naive_zero"`, `"naive_lag_1"`, `"naive_roll_mean_7"` |
| `fold_id` | 1–5 |
| `group_col` / `group_value` | `None`, or `demand_segment` / `is_delivery_day` and its value |
| `n`, `mae`, `pinball`, `coverage`, `fill_rate`, `shortfall_units`, `overstock_units` | from `evaluation.score()` |

Baselines are recomputed inside every fold on that fold's exact validation
rows rather than quoted from `evaluation.py`'s docstring. The floor moves with
the data, and comparing against a floor measured on a different row set is the
same error the preprocessing spec's adapter contract exists to prevent.

### 1.2 `utils/model_random_forest.py`

```python
QUANTILE = 0.9
DEFAULT_PARAMS = {...}
SEARCH_SPACE = {...}

def expand_one_hot(X, mapping) -> pd.DataFrame
def make_fit_predict(params, feature_cols, quantile=QUANTILE) -> Callable
def sample_search_space(n_candidates, seed) -> list[dict]
def fit_final(df, params, feature_cols) -> RandomForestQuantileRegressor
def save_model(model, path) / load_model(path)
```

`make_fit_predict(params)` returns the callable the runner injects. Inside it,
in order:

1. `impute_features()` — `quantile-forest`, like sklearn's forests before
   1.4, does not accept NaN. The function already exists and is already
   tested, including the sentinel-99 rule that keeps "outside the Eid window"
   from being read as "today is Eid".
2. optional one-hot expansion of the seven `*_idx` columns, driven by the
   `one_hot` parameter; the expansion is fit on training columns and
   reindexed onto validation so an absent category cannot shift columns.
3. optional `log1p` of the target, driven by `log_target`; predictions are
   inverted with `expm1` (`modeling_prep.inverse_log_target`).
4. fit, then `predict(X_valid, quantiles=0.9)`, clipped at 0 — a negative
   shipment quantity is not a thing.

**`max_samples_leaf` is measured, not assumed.** This `quantile-forest`
parameter controls how many training targets each leaf retains, and therefore
whether the 0.9 quantile is read from a real within-leaf distribution or only
from variation across trees. Retaining more is the statistically right answer
for intermittent demand and the expensive one for memory. **The first
implementation step is a benchmark** — one fit on fold 5's full training set
(~1.28M rows), recording wall time and peak RSS at the storage setting under
consideration. The tuning grid size is fixed only after that number exists,
and the measured numbers are recorded in `docs/hasil-modeling-rf.md`.

## Part 2 — Hyperparameter search

### 2.1 Space

`n_estimators` is not searched. Forest quality is monotone in tree count, so
searching it wastes budget on a question with a known answer; it is pinned at
200 during the search and raised for the final refit.

| Parameter | Values |
|---|---|
| `max_depth` | `None`, 12, 20, 30 |
| `min_samples_leaf` | 1, 5, 20, 50 |
| `max_features` | `"sqrt"`, 0.3, 0.5, 1.0 |
| `max_samples` | `None`, 0.5 |
| `log_target` | `False`, `True` |
| `one_hot` | `False`, `True` |

18 candidates drawn with a fixed seed. Random search rather than grid because
the space is 512 combinations and the informative dimensions are few — random
sampling covers each dimension's range better than a truncated grid at equal
cost.

### 2.2 Protocol

Each candidate is scored on **fold 3 (September) and fold 5 (November)** only.
Two folds rather than one because a single month can favour a parameter set by
accident; folds 3 and 5 differ in both training size and season without being
adjacent.

Selection metric: **pooled pinball@0.9 over the two folds' validation rows**.
Pooled rather than averaged so folds contribute in proportion to their row
count. Per-segment pinball is recorded for every candidate but does not decide
the winner — the service level is uniform across SKUs by the data owner's
decision, so the selection criterion must be too.

The winner is then refit and run across **all five folds**, and that is the
reported result. Reporting the search folds' own scores would be optimistic
for the same reason December is locked.

If the benchmark in Part 1 shows a single full-fold fit is too slow for 18
candidates, the search subsamples training rows (stratified by
`demand_segment`, fixed seed) and the subsample fraction is recorded next to
the results. Validation rows are never subsampled — the score must describe
the whole fold.

## Part 3 — Reporting

A single global MAE is misleading on data where 45% of targets are zero, so
results are reported in three cuts, each against all three naive baselines on
identical rows:

1. **Per fold** — is performance stable across months, or carried by one?
2. **Per `demand_segment`** — a global MAE dominated by mostly-zero pairs can
   crown a model that only won where predicting zero is easy. This is the
   axis the preprocessing spec built `demand_segment` for.
3. **Per `is_delivery_day`** — the rows that actually drive a shipment.

Beyond `mae` and `pinball`, two metrics answer the business question directly
and are reported everywhere:

- **`coverage`** — the fraction of rows where the prediction met or exceeded
  actual demand. A 0.9-quantile model that covers 60% is miscalibrated
  regardless of its MAE.
- **`fill_rate`** — the fraction of demanded units actually covered.

`shortfall_units` and `overstock_units` are reported as the raw cost sides of
the same trade-off.

`docs/hasil-modeling-rf.md` is written after the run and holds the benchmark
numbers, the selected hyperparameters, and the three result cuts, so the
evidence for the write-up lives in git rather than only in a notebook output
cell.

## Part 4 — Artifacts

| Artifact | Path | Versioned |
|---|---|---|
| Trained forest | `models/random_forest_q90.joblib` | No — new gitignored directory; large and reproducible |
| Selected hyperparameters | `dataset/model_ready/rf_best_params.json` | No — lives with the other model-ready outputs |
| Full results table | `dataset/model_ready/rf_walk_forward_results.csv` | No |
| Results summary | `docs/hasil-modeling-rf.md` | **Yes** — the evidence for the write-up belongs in git |

`models/` is added to `.gitignore` alongside `dataset/`.

The saved forest records the feature column order and the `one_hot`/
`log_target` flags used, because a model loaded a month later with columns in
a different order fails silently rather than loudly.

## Part 5 — Testing

`test/test_walk_forward.py` and `test/test_model_random_forest.py`, written
TDD (failing first), following the conventions of the existing 195 tests.
Tests use small synthetic frames, not the real parquet, so they run fast and
deterministically.

**Anti-leakage** — the tests that matter most, because their failure mode is
silent:

- No training row for fold *k* is dated on or after fold *k*'s month start.
- Purging is active: a row whose lead-time window crosses the boundary is
  excluded from training.
- No row dated on or after 2025-12-01 appears in any fold's training or
  validation set, for any fold.

**Identical rows** — the model and every naive baseline are scored on exactly
the same `(pair, date)` set within a fold; deliberately perturbing one must
fail the assertion. This mirrors the adapter contract in the preprocessing
spec, for the same reason.

**Random Forest wrapper**:

- `log_target` round-trips exactly: `expm1` of the log-scale prediction equals
  the prediction obtained on the raw scale for a monotone toy case.
- `one_hot` changes column count but never row count or row order, and a
  category present in validation but absent from training does not shift
  columns.
- No NaN survives `impute_features()` for any column in `FEATURE_COLS`.
- Predictions are non-negative.

**Reproducibility** — a fixed `random_state` gives identical results across
two runs, and `sample_search_space()` with a fixed seed gives the identical
candidate list.

## Part 6 — Dependencies

`requirements.txt` gains:

```
scikit-learn==1.6.1
quantile-forest==1.4.2
joblib
```

Verified 2026-08-18 against the project's Python 3.9.6 arm64 venv: both
resolve to cp39 wheels (`quantile_forest-1.4.2-cp39-cp39-macosx_11_0_arm64`,
`scikit_learn-1.6.1-cp39-cp39-macosx_12_0_arm64`) and pull in
`scipy 1.13.1`, `joblib 1.5.3`, `threadpoolctl 3.6.0` without disturbing the
pinned `numpy==2.0.2`.

XGBoost and the deep-learning framework are added by their own specs.

## Part 7 — Risks

| Risk | Handling |
|---|---|
| A full-fold QRF fit is too slow or exceeds memory | The Part 1 benchmark runs first and the search is sized against it; stratified training subsample is the documented fallback |
| The 0.9-quantile forest is badly calibrated on intermittent pairs | `coverage` is reported per segment, so miscalibration is visible rather than hidden inside MAE |
| Event-driven SKUs are unpredictable by construction | Known and accepted — the preprocessing spec establishes this as an information ceiling (no order-date data exists, `batasan-penelitian.md` B-1/B-2). Reported, not fixed |
| Random Forest loses to a naive baseline | A legitimate result and worth reporting as one. `evaluation.py` exists precisely so that outcome is detectable |

## References

- `docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md` — the
  shared feature table, folds, segments, and the quantile-0.9 decision
- `docs/superpowers/specs/2026-08-08-lead-time-integration-design.md` —
  `target_lead_time_cumulative` and the region/lead-time features
- `docs/batasan-penelitian.md` — B-1/B-2/B-3, the pickup-date limitation
- `docs/pipeline-overview.md` — the 12 data-prep stages
- Meinshausen (2006), *Quantile Regression Forests*, JMLR 7:983–999
