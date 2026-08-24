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

Since 2026-08-24 that array is two-dimensional: `(len(valid), len(quantiles))`,
one column per point of `QUANTILE_SET`, in the order the caller passed. A
one-dimensional return is refused rather than broadcast — a point forecast
stretched across nineteen quantiles keeps every shape correct while quietly
scoring a model that was never asked to be probabilistic.
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

# ── ROW ELIGIBILITY ────────────────────────────────────────────────────────────────

def eligible_rows(
    df: pd.DataFrame,
    lookback: int = modeling_prep.LOOKBACK,
    date_col: str = modeling_prep.DATE_COL,
    target_col: str = modeling_prep.EVAL_TARGET_COL,
    train_target_col: str = modeling_prep.TRAIN_TARGET_COL,
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

    Cut 3 now reads both targets. Since 2026-08-24 a model trains on the capped
    target and is scored on the raw one, so a row missing either is a row the
    two halves of the comparison would disagree about: the adapter would drop
    it for a null label while the scorer still expected a prediction for it.
    Both columns are built by the same `add_lead_time_target()` over the same
    window, so in the real pipeline they go missing together — which is exactly
    why a disagreement here means something upstream is wrong and is raised
    rather than quietly reconciled.
    """
    for column in (target_col, train_target_col):
        if column not in df.columns:
            raise KeyError(
                f"kolom {column!r} tidak ada — walk-forward butuh target latih "
                f"({train_target_col}) dan target penilaian ({target_col})"
            )

    missing_eval = df[target_col].isna()
    missing_train = df[train_target_col].isna()
    if not missing_eval.equals(missing_train):
        n = int((missing_eval ^ missing_train).sum())
        raise ValueError(
            f"pola nilai kosong {target_col!r} dan {train_target_col!r} "
            f"berbeda di {n} baris — keduanya dibangun dari jendela yang sama, "
            f"jadi selisih ini menandakan cacat di prapemrosesan, bukan "
            f"sesuatu yang boleh diselaraskan diam-diam di sini"
        )

    frame = df[df[date_col] < test_start]
    frame = modeling_prep.drop_warmup_rows(frame, lookback=lookback, date_col=date_col)
    frame = frame[frame[target_col].notna() & frame[train_target_col].notna()]
    return frame.reset_index(drop=True)

# ── FOLD PREPARATION ──────────────────────────────────────────────────────────────

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


METRIC_COLS = ("n", "quantile", "mae", "pinball", "coverage", "fill_rate",
               "shortfall_units", "overstock_units", "crossing_rate")


def _scored_rows(
    actual: pd.Series,
    prediction: pd.DataFrame,
    quantiles: tuple,
    header: dict,
) -> list:
    """One row per quantile for a single (model, fold, group) cell.

    `crossing_rate` is a property of the whole prediction matrix, not of any
    one quantile, so it repeats down every row of the cell. The repetition is
    the lesser evil: the alternative is a second frame with a different shape,
    and every consumer would have to join it back to ask "did this model's
    distribution make sense here".

    The price of that choice is a rule consumers have to keep: **anything
    aggregating this column must filter to a single tau first**, or state
    explicitly that it is reading a per-cell constant. Averaging across the
    grid happens to return the right number and reads as if it had combined
    nineteen measurements; summing returns nineteen times the truth.
    `model_common.summarise_candidate()` reads it at the headline tau.
    """
    scored = evaluation.score_quantiles(actual, prediction, quantiles)
    crossing = evaluation.crossing_rate(prediction, quantiles)
    return [{**header, **record, "crossing_rate": crossing}
            for record in scored.to_dict("records")]

# ── SCORING PER FOLD ──────────────────────────────────────────────────────────────

def run_fold(
    df: pd.DataFrame,
    fold_id: int,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    model_name: str = "model",
    quantiles: tuple = evaluation.QUANTILE_SET_A,
    prepared: bool = False,
) -> pd.DataFrame:
    """Fit on one fold's training rows, score on its validation rows.

    The naive baselines are recomputed here, on this fold's exact validation
    rows, rather than quoted from evaluation.py's docstring. The floor moves
    with the data, and a floor measured on a different row set is precisely
    the error this module exists to prevent.

    The baselines stay point forecasts and are widened to the same matrix
    shape, so model and floor are scored by one code path at one grid — the
    only way the two numbers can be subtracted from each other honestly.
    """
    quantiles = tuple(quantiles)
    split = prepare_fold(df, fold_id, prepared=prepared)
    train, valid = split["train"], split["valid"]

    raw = np.asarray(fit_predict(train, valid), dtype=float)
    expected = (len(valid), len(quantiles))
    if raw.shape != expected:
        raise ValueError(
            f"fit_predict mengembalikan bentuk {raw.shape}, "
            f"seharusnya {expected} — satu kolom per titik QUANTILE_SET"
        )

    predictions = {
        model_name: evaluation.as_quantile_frame(raw, quantiles, index=valid.index)
    }
    for name, baseline in evaluation.naive_predictions(valid).items():
        predictions[name] = evaluation.as_quantile_frame(baseline, quantiles,
                                                         index=valid.index)

    # The raw target, always — never the capped one the model was fitted on.
    # Scoring a model on the same trimmed series it learned would report a
    # number the project cannot act on: the outlet faces raw demand.
    actual = valid[modeling_prep.EVAL_TARGET_COL]
    rows = []
    for name, prediction in predictions.items():
        rows.extend(_scored_rows(actual, prediction, quantiles, {
            "model": name, "fold_id": fold_id,
            "group_col": None, "group_value": None,
        }))
        for group_col in GROUP_COLS:
            for value, index in valid.groupby(group_col, observed=True).groups.items():
                rows.extend(_scored_rows(
                    actual.loc[index], prediction.loc[index], quantiles, {
                        "model": name, "fold_id": fold_id,
                        "group_col": group_col, "group_value": str(value),
                    }))
    return pd.DataFrame(rows)


def run_walk_forward(
    df: pd.DataFrame,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    folds: tuple = FOLDS,
    model_name: str = "model",
    quantiles: tuple = evaluation.QUANTILE_SET_A,
) -> pd.DataFrame:
    """Every fold, one long result frame. The eligibility cut runs once."""
    frame = eligible_rows(df)
    parts = [
        run_fold(frame, fold_id, fit_predict, model_name=model_name,
                 quantiles=quantiles, prepared=True)
        for fold_id in folds
    ]
    return pd.concat(parts, ignore_index=True)


# The only two metrics whose value survives being averaged across the grid.
# `pinball` averaged across every quantile is the definition of K1;
# `crossing_rate` is a per-cell constant repeated down the quantile rows, so
# averaging it returns itself.
CROSS_QUANTILE_METRICS = ("pinball", "crossing_rate")

# ── AGGREGASI LINTAS FOLD ─────────────────────────────────────────────────────────

def pooled_metric(
    results: pd.DataFrame,
    model_name: str,
    metric: str = "pinball",
    folds: Optional[tuple] = None,
    quantile: Optional[float] = None,
) -> float:
    """One number across folds, weighted by row count.

    Every metric here is a per-row mean, so weighting by `n` reconstructs the
    value the metric would have had on the pooled rows. A plain average across
    folds would let November — the smallest fold — count as much as July.

    `quantile` picks a single point of the grid. Leaving it None pools across
    every quantile too, which is meaningful for exactly two metrics: `pinball`,
    where it *is* K1 (`pooled_k1()` is the name to use when that is what you
    mean), and `crossing_rate`, which is one number per cell repeated down the
    quantile rows. For anything else it is refused rather than computed —
    averaging coverage at 0.05 against coverage at 0.95 produces a number that
    describes no quantile, and it would look entirely reasonable in a table.
    """
    if quantile is None and metric not in CROSS_QUANTILE_METRICS:
        raise ValueError(
            f"metrik {metric!r} tidak punya arti dirata-ratakan lintas kuantil; "
            f"sebutkan quantile=... (mis. evaluation.DEFAULT_ALPHA)"
        )
    rows = results[(results["model"] == model_name) & results["group_col"].isna()]
    if folds is not None:
        rows = rows[rows["fold_id"].isin(folds)]
    if quantile is not None:
        rows = rows[np.isclose(rows["quantile"].to_numpy(dtype=float), quantile)]
    total = rows["n"].sum()
    if total == 0:
        return float("nan")
    return float((rows[metric] * rows["n"]).sum() / total)


def pooled_k1(
    results: pd.DataFrame,
    model_name: str,
    folds: Optional[tuple] = None,
) -> float:
    """K1: mean pinball across the quantile grid, pooled across folds.

    The two averages commute here — every quantile is scored on the identical
    rows, so each fold contributes the same `n` at every point — which is why
    one weighted pass over all the rows gives exactly the mean of the
    per-quantile pooled values. Weighting by `n` still matters across *folds*.
    """
    return pooled_metric(results, model_name, metric="pinball", folds=folds)


def coverage_by_quantile(
    results: pd.DataFrame,
    model_name: str,
    folds: Optional[tuple] = None,
) -> pd.DataFrame:
    """K2 in one table: realised coverage against the quantile it promised.

    `gap` is signed, not absolute. The finding K2 exists to catch is a model
    that misses the *same way* at every point — a distribution shifted bodily
    up or down — and an absolute gap erases the sign that makes that visible,
    leaving it indistinguishable from noise scattered either side.
    """
    rows = results[(results["model"] == model_name) & results["group_col"].isna()]
    if folds is not None:
        rows = rows[rows["fold_id"].isin(folds)]
    table = []
    for tau in sorted(rows["quantile"].unique()):
        coverage = pooled_metric(results, model_name, metric="coverage",
                                 folds=folds, quantile=tau)
        table.append({"quantile": tau, "target": tau, "coverage": coverage,
                      "gap": coverage - tau})
    return pd.DataFrame(table)
