# XGBoost modeling — design

## Purpose

Train and evaluate the second of the three candidate models against
`dataset/model_ready/model_input.parquet`, through the walk-forward runner the
Random Forest spec built for exactly this purpose.

The deliverable answers two questions: **does an XGBoost predicting the 0.9
quantile of `target_lead_time_cumulative` beat the naive baselines, and how
does it compare to the Random Forest on identical rows?**

This spec is the follow-up promised by
`docs/superpowers/specs/2026-08-18-random-forest-modeling-design.md`, whose
non-goals list "XGBoost and LSTM training — they consume this spec's runner,
in their own specs".

## Background

### What already exists

The Random Forest spec landed in full and its results are recorded in
`docs/hasil-modeling-rf.md`. Everything this design needs is present and
tested:

| Component | Location | What it gives us |
|---|---|---|
| `walk_forward.py` | `utils/` | fold boundaries, row eligibility, and scoring — model injected as a `fit_predict(train, valid) -> np.ndarray` callable |
| `eligible_rows()` | `utils/walk_forward.py` | the one definition of which rows any model may see: pre-December, past the 28-day warm-up, non-null target |
| `purging.lookahead_safe_mask()` | `utils/purging.py` | rows whose whole target window stays strictly before a boundary |
| `evaluation.py` | `utils/` | `mae`, `pinball_loss`, `quantile_coverage`, `fill_rate`, `shortfall_units`, `overstock_units`, and the three naive baselines |
| `FEATURE_COLS` | `utils/modeling_prep.py` | the 56 agreed model inputs, identical for every model |
| `model_random_forest.py` | `utils/` | the search protocol, checkpoint/resume, one-hot expansion, and bundle I/O this spec generalizes |

The Random Forest numbers this model is measured against, pooled over the five
walk-forward folds (345,547 validation rows): **pinball@0.9 2.410**, MAE 15.64,
coverage 0.932. Best naive baseline `naive_roll_mean_7`: pinball 4.503. On the
three folds untouched by model selection (1, 2, 4): RF pinball **2.403**.

Categorical cardinalities, measured on `model_input.parquet` 2026-08-19:
`Kode Barang_idx` 70, `Nama Cabang_idx` 59, `kota_idx` 16, `Kategori
Barang_idx` 8, `branch_volume_tier_idx` 4, `demand_segment_idx` 4,
`hari_pengiriman_idx` 2. One-hot expansion therefore turns 56 columns into
roughly 212 — affordable, which is why all three encodings below are
searchable rather than decided by assumption.

### What is settled and inherited

Not reopened here:

- **Primary target** — `target_lead_time_cumulative`.
- **Loss and service level** — pinball loss at quantile **0.9, uniform across
  every SKU** (data owner, 2026-08-16).
- **Validation** — walk-forward, 5 expanding folds (Jul–Nov 2025). December
  2025 opens exactly once, after all three models have been compared.
- **Feature set** — `modeling_prep.FEATURE_COLS`, identical for every model.
- **Row eligibility and scoring** — owned by `walk_forward.py`, not
  re-implemented here.

## Decisions

Settled during brainstorming 2026-08-19.

| Decision | Choice | Rationale |
|---|---|---|
| Quantile mechanism | `reg:quantileerror` with `quantile_alpha=0.9`, `tree_method="hist"` | XGBoost's native quantile objective, available since 2.0. It optimizes the same loss the models are selected on, so training objective and selection criterion agree |
| Code structure | Extract `utils/model_common.py` first, then write `utils/model_xgboost.py` against it | The search protocol, checkpoint/resume, one-hot expansion and bundle I/O are not RF-specific. Duplicating them means fixing every future bug twice, then three times when the LSTM lands; importing them from `model_random_forest` would point the dependency arrow the wrong way permanently |
| Boosting rounds | Early stopping on a held-out training tail, then **refit on the full training rows** at `best_iteration` | The validation fold cannot be used for early stopping without leaking. Carving the tail out permanently would cost XGBoost ~4% of its training rows — the most recent ones — and score it on a different population than the Random Forest saw. The second fit restores the exact row set, at 2x fit cost that `hist` can afford |
| Categorical encoding | `encoding` as a searched flag over `ordinal` / `native` / `one_hot` | Cardinalities are small enough that all three are affordable. `native` (`enable_categorical=True`) is XGBoost's structural advantage over the forest; `one_hot` matches the configuration the RF search actually chose. Fixing this by assumption would leave "XGB lost on algorithm or on encoding?" unanswerable |
| Tuning budget | Random search, **30 candidates**, scored on folds 3 and 5; winner refit and reported over all 5 folds | Same protocol as RF, more draws: XGBoost's space has more dimensions that genuinely move the score, so equal coverage needs more draws. The asymmetry against RF's 18 is a reportable caveat, not a hidden one |
| `n_estimators` | Absent from the search space; capped at `MAX_ROUNDS = 2000` | Early stopping already decides it per candidate per fold. Searching it would spend budget on a question that has a mechanism |
| Memory screen | None | The QRF leaf-storage bound has no XGBoost analogue — `hist` holds a quantized feature matrix, tens of MB at this size. `sample_search_space()` takes the screen as an injected callable so RF keeps its own and XGB passes `None` |
| Target scale | `log_target` as a searched flag | Same reasoning as RF: quantiles are equivariant under monotonic transforms, so inversion is exact; what is being tested is the effect on split selection |

### Non-goals

- **December 2025.** Not touched, not scored, not looked at. It opens once,
  after the LSTM exists and a winner has been chosen.
- LSTM training — its own spec, consuming the same runner.
- SHAP explainability — winner only, after all three models exist.
- Per-segment specialist models. Out of scope so the comparison stays
  like-for-like.
- Any change to `FEATURE_COLS`, to `modeling_prep.py`'s feature definitions, or
  to the 12 data-prep stages.
- Any change to the Random Forest's measured results. The refactor in Part 1
  must leave them reproducible.

## Part 1 — Architecture

```
model_input.parquet
        │
        ▼
walk_forward.eligible_rows()        ← the one row-eligibility definition
        │
        ├──► walk_forward.run_fold(fit_predict=rf.make_fit_predict(...))
        └──► walk_forward.run_fold(fit_predict=xgb.make_fit_predict(...))
                                          │
                                model_common.py  (search, checkpoint, bundle I/O)
```

### 1.1 `utils/model_common.py` — new

Moved out of `model_random_forest.py`, generalized only where a second model
forces it:

| Moved | Change on the way out |
|---|---|
| `assert_no_nan()` | none |
| `expand_one_hot()`, `IDX_COLS` | none |
| `select_best()` | none |
| `save_bundle()` / `load_bundle()` / `save_best_params()` | none — already take a `path` |
| `run_search()`, `_ordered()`, `_assert_checkpoint_matches()` | `search_space` and `make_fit_predict` become arguments instead of module globals |
| `sample_search_space()` | takes `space`, `defaults`, and `screen: Callable[[dict], bool] \| None`; RF injects its leaf-memory screen, XGB injects `None` |

`model_random_forest.py` keeps what is genuinely QRF-specific —
`estimate_leaf_memory_bytes()`, `build_estimator()`, `make_fit_predict()`,
`fit_final()`, `predict_bundle()`, `DEFAULT_PARAMS`, `SEARCH_SPACE`, its path
constants — and **re-exports the moved names**, so
`test/test_model_random_forest.py` and `notebook/modeling_rf.ipynb` keep
working without a single line changed. That the RF suite stays green with no
assertion edited is the refactor's success criterion, not a convenience.

### 1.2 `utils/model_xgboost.py` — new

**`split_early_stopping(train, tail_days=30) -> (fit_rows, es_rows)`**

`es_rows` is the last `tail_days` calendar days of the fold's training window;
`fit_rows` is everything before, filtered by
`purging.lookahead_safe_mask(fit_rows, es_start)`.

That purge is not extra caution. `target_lead_time_cumulative` sums over
H+1..H+`lead_time_days`, so a fit row dated within `lead_time_days` of
`es_start` carries a label built partly out of the early-stopping window —
the identical leak `fold_train_mask()` prevents at the fold boundary, one
scale down. Without it, early stopping would be reading a signal it partly
trained on and would stop too late.

**`encode(train_X, valid_X, encoding) -> (train_X, valid_X, enable_categorical)`**

One place for all three modes:

- `"ordinal"` — the `_idx` columns pass through as numerics.
- `"native"` — `_idx` columns cast to pandas `category` with the training
  categories, validation reindexed onto them; `enable_categorical=True`.
- `"one_hot"` — delegates to `model_common.expand_one_hot()`, whose reindex
  already handles a category seen only in validation.

In every mode validation columns are forced onto the training columns. A
category present only in validation would otherwise shift every column after
it, and the booster would read the wrong feature at each position — silently,
since the shapes still line up.

**`make_fit_predict(params) -> Callable[[train, valid], np.ndarray]`**

Two fits per fold:

1. Fit on `fit_rows` with `es_rows` as the eval set,
   `early_stopping_rounds=50`, `n_estimators=MAX_ROUNDS`. Record
   `best_iteration`.
2. Discard that booster. Refit on **all** training rows (`fit_rows` +
   `es_rows`) with `n_estimators=best_iteration`.

Predictions come from the second booster, clipped to `>= 0` — a negative
shipment quantity is not a thing. `log_target` inverts through
`modeling_prep.inverse_log_target()` before clipping.

With `log_target=True` the early-stopping metric is computed on the log scale.
That is sound: early stopping only chooses a round count *within* one
candidate. Candidates are compared only by pinball on the original scale,
after inversion.

**`fit_final(df, params)`**

The same two-fit protocol, tail taken from the last 30 days before December.
Row population comes from `walk_forward.eligible_rows()` then
`purging.lookahead_safe_mask(frame, TEST_START)` — the same rows the reported
metrics describe, for the same reason `rf.fit_final()` does it that way.

**`predict_bundle(bundle, frame)`**

The bundle records `columns`, `encoding`, `log_target`, `best_iteration`,
`quantile`, `feature_cols`, `n_train`. A booster reloaded next month against
columns in a different order does not fail — it predicts confidently from the
wrong features, which is worse.

### 1.3 What does not change

`walk_forward.py` is untouched. It keeps ownership of fold boundaries, row
eligibility and scoring; XGBoost enters through one injected callable. That is
the structural guarantee — not a procedural one — that RF and XGB are scored
on identical rows.

## Part 2 — Hyperparameter search

### 2.1 Space

```python
SEARCH_SPACE = {
    "max_depth":        [4, 6, 8, 10],
    "learning_rate":    [0.03, 0.05, 0.1],
    "min_child_weight": [1, 10, 50],
    "subsample":        [0.7, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "reg_lambda":       [1.0, 10.0],
    "encoding":         ["ordinal", "native", "one_hot"],
    "log_target":       [False, True],
}
```

2,592 combinations; 30 drawn at random with seed 42. Random rather than grid,
for the reason the RF spec gives: only a few dimensions carry real signal, so
random draws cover each dimension's range better than a truncated grid at the
same cost.

`n_estimators` is absent by design (see Decisions). `max_depth=None` has no
XGBoost equivalent worth searching under a `hist` tree method at this depth
range.

### 2.2 Protocol

Identical to the Random Forest search, which is the point of extracting
`model_common.run_search()`:

- Scored on **folds 3 and 5** only.
- Criterion: **pooled pinball@0.9**, weighted by row count rather than
  averaged flat, so November — the smallest fold — does not count as much as
  September.
- **No subsampling.** Every training row of each fold is used.
- Each finished candidate is flushed to
  `dataset/model_ready/xgb_search_results.csv` immediately, and a restart
  resumes from it. A run this long can be killed by the OS with no exception
  to catch, and the file doubles as the only progress signal available from
  inside a notebook cell.
- A candidate that raises is recorded with NaN metrics and a populated `error`
  column rather than aborting the run.
- The stale-checkpoint guard carries over: resuming across a changed search
  space or seed is refused, because it would blend two experiments and hand
  back a winner that was never evaluated.

The winner is refit and reported across all five folds.

## Part 3 — Reporting

`docs/hasil-modeling-xgb.md`, following the structure of
`docs/hasil-modeling-rf.md`: summary, evaluation setup, search results,
walk-forward per fold, per `demand_segment`, per `is_delivery_day`, final
model, reproduction, limitations. Written in Indonesian, like the RF results
document.

One section the RF document does not have:

**Head-to-head XGBoost vs Random Forest.** Legitimate because both were scored
on identical rows — guaranteed by `walk_forward.eligible_rows()`, not by
discipline. Two slices:

1. All five folds.
2. **Folds 1, 2 and 4** — untouched by model selection for *either* model,
   since both searched on folds 3 and 5. This is the clean number.

Two asymmetries must be stated in that section rather than buried:

- **Unequal search budget** — 30 candidates versus 18.
- **Unequal round protocol** — early stopping versus a pinned tree count.

If XGBoost wins narrowly, both are honest rival explanations. If it wins by a
margin comparable to RF's 46% over the best baseline, neither explains it.

MAE is reported for context only, never as a winning criterion — comparing a
0.9-quantile model's MAE against a mid-point baseline punishes it for doing
exactly what was asked. Pinball@0.9 decides.

## Part 4 — Artifacts

| Artifact | Path | Versioned |
|---|---|---|
| Search results | `dataset/model_ready/xgb_search_results.csv` | No |
| Selected hyperparameters | `dataset/model_ready/xgb_best_params.json` | No |
| Full results table | `dataset/model_ready/xgb_walk_forward_results.csv` | No |
| Trained booster | `models/xgboost_q90.joblib` | No — `models/` is already gitignored |
| Notebook | `notebook/modeling_xgb.ipynb` | **Yes** |
| Results summary | `docs/hasil-modeling-xgb.md` | **Yes** — the evidence for the write-up belongs in git |

`notebook/modeling_xgb.ipynb` mirrors `modeling_rf.ipynb`: benchmark, search,
final walk-forward, results. Outputs are cleared before commit, because the
evidence lives in the CSVs and in `docs/`, not in cell output that a stray
"Clear All Outputs" can erase.

## Part 5 — Testing

TDD, following the conventions of the existing suites. Small synthetic frames,
not the real parquet, so the tests stay fast and deterministic.

**`test/test_model_random_forest.py` — not edited.** It must stay green
through the Part 1 extraction, with no assertion changed. This is the
regression test for the refactor.

**`test/test_model_common.py`** — the moved helpers, plus what generalization
added:

- `sample_search_space()` with an injected `screen` rejects exactly the
  candidates the screen rejects, and with `screen=None` rejects none.
- `run_search()` calls the injected `make_fit_predict` and records the space's
  keys, for a space it has never seen before.
- A fixed seed reproduces the identical candidate list.
- The stale-checkpoint guard rejects a checkpoint whose rows describe
  different candidates.

**`test/test_model_xgboost.py`** — the anti-leakage tests matter most, because
their failure mode is silent:

- `split_early_stopping()` puts no `es_rows` date before any `fit_rows` date,
  and a fit row whose lead-time window crosses `es_start` is excluded.
- No training row, in either fit, is dated on or after the fold's month start.
- No row dated on or after 2025-12-01 reaches any fit.

And the wrapper's own contract:

- The second fit sees `len(fit_rows) + len(es_rows)` rows — the refit really
  is on the full training population.
- The second booster uses `n_estimators == best_iteration` from the first.
- All three encodings preserve row count and row order; a category present in
  validation but absent from training does not shift columns in any of them.
- `log_target` round-trips: `expm1` of the log-scale prediction equals the
  raw-scale prediction for a monotone toy case.
- `predict_bundle()` forces the recorded column order, and a shuffled input
  frame produces identical predictions.
- Predictions are non-negative.
- `fit_predict` returns exactly `len(valid)` values.

## Part 6 — Dependencies

`requirements.txt` gains:

```
xgboost==2.1.4
```

2.1.4 is the highest release with cp39 wheels — verified 2026-08-19 against
the project's Python 3.9.6 venv via `pip index versions xgboost`. That is
comfortably above the 2.0 release that introduced `reg:quantileerror`, so the
version ceiling costs this design nothing.

No other new dependency: `joblib`, `numpy`, `pandas` and `pyarrow` are already
pinned.

## Part 7 — Risks

| Risk | Handling |
|---|---|
| `best_iteration` differs wildly across folds, making the reported model unstable | It is recorded per fold in the results table, so the spread is visible rather than averaged away |
| The tail-30-day early-stopping set is unrepresentative (a holiday, a promo) | The tail moves with the fold, so five folds give five different tails; a pathological one shows up as an outlier fold in the per-fold table |
| Two fits per candidate makes the search too slow | `hist` on 1.3M x 56 is far cheaper per fit than the QRF benchmark's 6.6 minutes; if it still overruns, the checkpoint makes a partial search resumable and `MAX_ROUNDS` can be lowered |
| `reg:quantileerror` is calibrated badly on intermittent pairs | `coverage` is reported per segment, as for RF, so miscalibration is visible instead of hidden inside a global MAE |
| The Part 1 refactor silently changes RF behaviour | `test/test_model_random_forest.py` runs unmodified; the RF numbers in `docs/hasil-modeling-rf.md` stay reproducible from the same artifacts |
| XGBoost loses to the Random Forest, or to a naive baseline | A legitimate result and reported as one. The comparison exists to be informative, not to be won |

## References

- `docs/superpowers/specs/2026-08-18-random-forest-modeling-design.md` — the
  walk-forward runner, the search protocol, and the settled quantile decision
- `docs/hasil-modeling-rf.md` — the measured Random Forest results this model
  is compared against
- `docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md` — the
  shared feature table, folds, segments, and the quantile-0.9 decision
- `docs/batasan-penelitian.md` — B-1/B-2/B-3, the pickup-date information
  ceiling that bounds every model here
