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

Since 2026-08-24 the comparison runs on a *set* of quantiles rather than on
0.9 alone (see 2026-08-22-multi-quantile-evaluation-design.md). The primitives
below are unchanged — `pinball_loss` still takes one alpha — and the
multi-quantile layer sits on top of them: `resolve_quantile_set()` decides
which grid is in force from the state of the cost data, `score_quantiles()`
scores a prediction matrix point by point, and `k1_score()` collapses that to
the one number models are ranked by. The per-point scores are always kept
beside the mean; a single number that hides nineteen is the thing this
migration was written to get away from.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

TARGET_COL = "target_lead_time_cumulative"
LEAD_TIME_COL = "lead_time_days"
ITEM_COL = "Kode Barang"

# The service level the business ships at (B-9). No longer the whole of the
# evaluation — K1 is now the mean over QUANTILE_SET — but still the point the
# results documents lead with, so it keeps a name of its own.
DEFAULT_ALPHA = 0.9

# Tahap A: 19 evenly spaced points, 0.05 to 0.95. Even spacing is the point —
# the mean pinball over a dense uniform grid approaches CRPS (Bröcker 2012),
# so this grid scores the whole predictive distribution rather than sampling
# one service level and hoping the ranking holds elsewhere.
QUANTILE_SET_A = tuple(round(0.05 * step, 2) for step in range(1, 20))

# Tahap B: the grid stops being uniform and becomes the spread of critical
# ratios the segmented allocation will actually ship at. Percentiles rather
# than every segment's ratio, because 200 segments would mean 200 fits.
QUANTILE_SET_B_PERCENTILES = (10, 25, 50, 75, 90)

# B-10 closes at 80% of volume carried by SKUs with a non-`rendah` cost entry.
# The same threshold decides which grid this module hands out, so the switch
# happens on data status alone — no code change, no second decision.
COST_COVERAGE_THRESHOLD = 0.80
COST_MARGIN_FILE = str(BASE_DIR / "dataset/item_cost_margin.csv")
IMPRECISE_CONFIDENCE = "rendah"

QuantileSet = tuple
Predictions = Union[pd.DataFrame, pd.Series, np.ndarray]

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


def cost_coverage_share(
    cost_table: pd.DataFrame,
    volume_by_sku: pd.Series,
    item_col: str = ITEM_COL,
) -> float:
    """Share of volume carried by SKUs with a precise cost entry.

    Volume-weighted rather than SKU-counted, because that is how B-10 states
    its closing criterion: seventy SKUs are not seventy equal votes, and the
    handful that move most of the goods are the ones whose critical ratio
    would actually shape a Tahap B grid.

    A SKU missing from the cost table counts as imprecise. Absence is exactly
    the state B-10 describes — no entry yet — and reading it any other way
    would let an incomplete file close the threshold by being incomplete.
    """
    volume = pd.Series(volume_by_sku, dtype="float64")
    total = volume.sum()
    if total <= 0:
        return 0.0
    confidence = (cost_table.set_index(item_col)["cost_confidence"]
                  .reindex(volume.index))
    precise = confidence.notna() & (confidence != IMPRECISE_CONFIDENCE)
    return float(volume[precise.to_numpy()].sum() / total)


def quantile_set_b(critical_ratios: pd.Series) -> QuantileSet:
    """The Tahap B grid: percentiles of the observed critical-ratio spread.

    Duplicates collapse. A cost table where every segment lands on the same
    ratio should produce a one-point grid and say so, not a five-point grid
    that scores the same quantile five times and reports the mean as if it
    covered a range.
    """
    values = pd.Series(critical_ratios, dtype="float64").dropna()
    if values.empty:
        raise ValueError("tidak ada critical ratio untuk membentuk Tahap B")
    points = np.percentile(values.to_numpy(), QUANTILE_SET_B_PERCENTILES)
    return tuple(sorted({round(float(point), 4) for point in points}))


def resolve_quantile_set(
    cost_table: Optional[pd.DataFrame] = None,
    volume_by_sku: Optional[pd.Series] = None,
    critical_ratios: Optional[pd.Series] = None,
    threshold: float = COST_COVERAGE_THRESHOLD,
    cost_margin_file: str = COST_MARGIN_FILE,
) -> QuantileSet:
    """Tahap A or Tahap B, decided by data status rather than by an edit.

    This is the whole switching mechanism from the multi-quantile spec: the
    grid follows `item_cost_margin.csv` as the SCM team fills it, the same way
    the segmented allocation's precise/proxy path does, so nobody has to
    remember to change a constant on the day the threshold is crossed.

    Passing no cost table at all means Tahap A. That is the honest default for
    a caller with no cost data in reach — a test frame, a fresh checkout —
    rather than a silent read of whatever happens to be on disk.

    Once the threshold *is* crossed, a missing `critical_ratios` raises. The
    tempting fallback — quietly staying on Tahap A — would keep evaluating on
    a grid the design has already superseded, and the results document would
    say Tahap B while the numbers came from Tahap A.
    """
    if cost_table is None or volume_by_sku is None:
        return QUANTILE_SET_A

    share = cost_coverage_share(cost_table, volume_by_sku)
    if share < threshold:
        return QUANTILE_SET_A
    if critical_ratios is None:
        raise NotImplementedError(
            f"cakupan biaya presisi {share:.1%} sudah mencapai ambang "
            f"{threshold:.0%} (B-10), jadi QUANTILE_SET seharusnya Tahap B — "
            f"tetapi critical_ratios tidak diberikan. Jalur critical ratio "
            f"belum diimplementasikan (butir 3a, "
            f"2026-08-22-segmented-quantile-allocation-design.md); "
            f"jangan diam-diam memakai Tahap A"
        )
    return quantile_set_b(critical_ratios)


def as_quantile_frame(
    predictions: Predictions,
    quantiles: QuantileSet = QUANTILE_SET_A,
    index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """One column per quantile, in `quantiles` order, whatever the caller has.

    A 1-D input is broadcast across every column. That is not a convenience:
    the naive baselines and any point forecast produce one number per row, and
    scoring them at every tau is what keeps the floor in the same units as K1.
    A point forecast simply pays the pinball penalty of pretending its single
    number is every quantile at once — which is exactly the claim it makes.
    """
    quantiles = tuple(quantiles)
    if isinstance(predictions, pd.DataFrame):
        if len(predictions.columns) != len(quantiles):
            raise ValueError(
                f"prediksi punya {len(predictions.columns)} kolom, "
                f"seharusnya {len(quantiles)} sesuai QUANTILE_SET"
            )
        frame = predictions.copy()
        frame.columns = list(quantiles)
        return frame

    values = np.asarray(predictions, dtype="float64")
    if index is None:
        index = (predictions.index if isinstance(predictions, pd.Series)
                 else pd.RangeIndex(len(values)))
    if values.ndim == 1:
        values = np.repeat(values[:, None], len(quantiles), axis=1)
    elif values.shape[1] != len(quantiles):
        raise ValueError(
            f"prediksi punya {values.shape[1]} kolom, "
            f"seharusnya {len(quantiles)} sesuai QUANTILE_SET"
        )
    return pd.DataFrame(values, columns=list(quantiles), index=index)


def score_quantiles(
    y_true: pd.Series,
    predictions: Predictions,
    quantiles: QuantileSet = QUANTILE_SET_A,
) -> pd.DataFrame:
    """One scored row per quantile — the long form K1 and K2 both read.

    Long rather than wide because every metric here is per-quantile: coverage
    at 0.3 and coverage at 0.9 answer different questions, and a wide frame
    with `coverage_q30` columns would have to be melted by every consumer
    anyway. Keeping the breakdown visible is the design's requirement (Bagian
    2): the mean is reported *beside* the per-point scores, never instead.
    """
    frame = as_quantile_frame(predictions, quantiles,
                              index=pd.Series(y_true).index)
    return pd.DataFrame([
        {"quantile": tau, **score(y_true, frame[tau], alpha=tau)}
        for tau in frame.columns
    ])


def k1_score(per_quantile: pd.DataFrame, metric: str = "pinball") -> float:
    """K1: the unweighted mean of the per-quantile pinball losses.

    Unweighted on purpose. Every quantile is scored on the identical rows, so
    there is no coverage difference to correct for, and weighting toward 0.9
    — tempting, since that is the shipped service level — would rebuild the
    single-point criterion this migration exists to replace. Open question 1
    of the multi-quantile spec keeps that option on the table; until it is
    decided, the average is flat and says so.
    """
    values = pd.Series(per_quantile[metric], dtype="float64").dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def crossing_rate(
    predictions: Predictions,
    quantiles: QuantileSet = QUANTILE_SET_A,
) -> float:
    """Share of rows whose predictions are not non-decreasing across quantiles.

    Reported, never asserted to zero. A composite pinball head has no
    structural monotonicity guarantee — each quantile is fitted by the same
    loss but nothing ties them together — so crossing is a measurement about
    the fitted model, and a rate above a few percent is the signal that the
    arctan pinball loss (Sluijterman et al. 2024) is worth its complexity.

    A row counts once however many inversions it contains: the question is
    "does this row's predictive distribution make sense", not "how badly".

    This is a property of the whole prediction *matrix*, not of any one tau —
    there is no such thing as "the crossing rate at 0.9". Where it is stored
    beside per-quantile rows (`walk_forward.run_fold`) it therefore repeats
    down every row of a cell, and **a consumer that aggregates must pick one
    tau** rather than averaging the column: averaging returns the same number
    but reads as if nineteen measurements had been combined, and summing
    returns nineteen times the truth. `walk_forward.pooled_metric` allows the
    unfiltered call for exactly this reason, and `model_common` reads it at
    the headline point.
    """
    frame = as_quantile_frame(predictions, quantiles)
    if frame.empty:
        return float("nan")
    values = frame.to_numpy(dtype="float64")
    if values.shape[1] < 2:
        return 0.0
    inverted = np.diff(values, axis=1) < 0
    return float(inverted.any(axis=1).mean())


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
    quantiles: QuantileSet = QUANTILE_SET_A,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Score every naive baseline at every quantile, optionally split by group.

    Pass group_col="demand_segment" to get the per-segment floor, or
    "is_delivery_day" to get the floor on the rows that actually drive a
    shipment. A global number alone hides both.

    The baselines stayed point forecasts — `roll_mean_7 x lead_time_days` has
    no quantile to vary — so each is scored against every tau as it stands.
    That is what makes the floor comparable to K1: both numbers are a mean
    over the same grid, and the gap between them is what a model bought.
    """
    predictions = naive_predictions(df, lead_time_col=lead_time_col)
    rows = []
    for name, prediction in predictions.items():
        if group_col is None:
            scored = score_quantiles(df[target_col], prediction, quantiles)
            rows.extend({"baseline": name, **record}
                        for record in scored.to_dict("records"))
            continue
        for group_value, index in df.groupby(group_col, observed=True).groups.items():
            scored = score_quantiles(df.loc[index, target_col],
                                     prediction.loc[index], quantiles)
            rows.extend({"baseline": name, group_col: group_value, **record}
                        for record in scored.to_dict("records"))
    return pd.DataFrame(rows)
