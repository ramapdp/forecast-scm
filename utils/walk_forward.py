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
