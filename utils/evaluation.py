"""Metrics and the naive baselines every model must beat.

A model score means nothing on its own. `roll_mean_7 x lead_time_days` — no
model, no training, one multiplication — reaches MAE 12.99 and pinball@0.9
6.56 on the December 2025 test window. Anything that does not clear that is
not worth deploying however sophisticated its architecture, and reporting it
alongside the models is what turns "XGBoost scored 11" into a claim a reader
can check.

The metrics here drop null targets rather than treating them as zero: the last
days of the test window have no lead-time target because January demand is not
in the data, and scoring a forecast against a label that does not exist would
quietly reward whichever model happens to predict smallest there.
"""

from typing import Optional

import numpy as np
import pandas as pd

TARGET_COL = "target_lead_time_cumulative"
LEAD_TIME_COL = "lead_time_days"
DEFAULT_ALPHA = 0.9

# Each baseline scales a backward-looking demand estimate by the number of days
# the shipment has to cover. Both estimates already exist as features, so these
# cost nothing to compute and are exactly what an outlet manager would do by
# hand.
NAIVE_BASELINES = {
    "naive_zero": None,
    "naive_lag_1": "lag_1",
    "naive_roll_mean_7": "roll_mean_7",
}


def _aligned(y_true: pd.Series, y_pred: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    keep = y_true.notna()
    return (
        y_true[keep].to_numpy(dtype="float64"),
        y_pred[keep].fillna(0.0).to_numpy(dtype="float64"),
    )


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    actual, predicted = _aligned(y_true, y_pred)
    if actual.size == 0:
        return float("nan")
    return float(np.abs(actual - predicted).mean())


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, alpha: float = DEFAULT_ALPHA) -> float:
    """Asymmetric loss: a shortfall costs `alpha`, an excess costs `1 - alpha`.

    At alpha 0.9 a stockout is nine times as expensive as the same amount of
    overstock, which is why a mean forecast — right half the time by
    construction — is the wrong target for a replenishment decision.
    """
    actual, predicted = _aligned(y_true, y_pred)
    if actual.size == 0:
        return float("nan")
    delta = actual - predicted
    return float(np.where(delta >= 0, alpha * delta, (alpha - 1.0) * delta).mean())


def shortfall_units(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Units demanded but not forecast — the stockout, in the unit shipped."""
    actual, predicted = _aligned(y_true, y_pred)
    if actual.size == 0:
        return float("nan")
    return float(np.maximum(0.0, actual - predicted).sum())


def overstock_units(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Units forecast beyond demand — the cost side of a high service level."""
    actual, predicted = _aligned(y_true, y_pred)
    if actual.size == 0:
        return float("nan")
    return float(np.maximum(0.0, predicted - actual).sum())


def fill_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Share of demanded units the forecast would have covered.

    This is the data owner's stated success criterion — "the outlet does not
    run out" — stated in units rather than in error magnitude, and it is the
    one number an outlet manager can check against their own experience.

    Shortfalls are summed before the ratio, so a surplus on one outlet-day
    cannot cancel a stockout on another: the goods are already at the wrong
    branch on the wrong day. A window with no demand at all scores 1.0, since
    nothing went unserved.
    """
    actual, predicted = _aligned(y_true, y_pred)
    if actual.size == 0:
        return float("nan")
    demanded = actual.sum()
    if demanded == 0:
        return 1.0
    return float(1.0 - np.maximum(0.0, actual - predicted).sum() / demanded)


def quantile_coverage(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Share of actuals at or below the forecast.

    Train at alpha and this should come back near alpha. Far above means
    systematic overstock; far below means the outlet runs dry more often than
    the service level promises.
    """
    actual, predicted = _aligned(y_true, y_pred)
    if actual.size == 0:
        return float("nan")
    return float((actual <= predicted).mean())


def naive_predictions(
    df: pd.DataFrame,
    lead_time_col: str = LEAD_TIME_COL,
) -> dict[str, pd.Series]:
    """One prediction series per naive baseline, indexed like `df`."""
    lead_time = df[lead_time_col].fillna(0)
    predictions = {}
    for name, source_col in NAIVE_BASELINES.items():
        if source_col is None:
            predictions[name] = pd.Series(0.0, index=df.index)
            continue
        predictions[name] = (df[source_col] * lead_time).fillna(0.0).clip(lower=0.0)
    return predictions


def score(
    y_true: pd.Series, y_pred: pd.Series, alpha: float = DEFAULT_ALPHA
) -> dict[str, float]:
    return {
        "n": int(pd.Series(y_true).notna().sum()),
        "mae": mae(y_true, y_pred),
        "pinball": pinball_loss(y_true, y_pred, alpha=alpha),
        "coverage": quantile_coverage(y_true, y_pred),
        "fill_rate": fill_rate(y_true, y_pred),
        "shortfall_units": shortfall_units(y_true, y_pred),
        "overstock_units": overstock_units(y_true, y_pred),
    }


def evaluate_baselines(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lead_time_col: str = LEAD_TIME_COL,
    alpha: float = DEFAULT_ALPHA,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Score every naive baseline, optionally split by a grouping column.

    Pass group_col="demand_segment" to get the per-segment floor, or
    "is_delivery_day" to get the floor on the rows that actually drive a
    shipment. A global number alone hides both.
    """
    predictions = naive_predictions(df, lead_time_col=lead_time_col)
    rows = []
    for name, prediction in predictions.items():
        if group_col is None:
            rows.append({"baseline": name, **score(df[target_col], prediction, alpha)})
            continue
        for group_value, index in df.groupby(group_col, observed=True).groups.items():
            rows.append({
                "baseline": name,
                group_col: group_value,
                **score(df.loc[index, target_col], prediction.loc[index], alpha),
            })
    return pd.DataFrame(rows)
