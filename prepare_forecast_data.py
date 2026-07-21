import pandas as pd
from pathlib import Path

import build_panel

PAIR_COLS = build_panel.PAIR_COLS
TEST_START = build_panel.TEST_START
BRANCH_COL = "Nama Cabang"
TARGET_HORIZONS = range(1, 8)
LAG_DAYS = [1, 2, 3, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]
MODEL_READY_DIR = "dataset/model_ready"


def add_targets(
    df: pd.DataFrame,
    pair_cols: list[str] = PAIR_COLS,
    date_col: str = "Tanggal",
    qty_col: str = "Kuantitas",
    horizons=TARGET_HORIZONS,
) -> pd.DataFrame:
    result = df.sort_values(pair_cols + [date_col]).reset_index(drop=True)
    grouped = result.groupby(pair_cols)[qty_col]
    for h in horizons:
        result[f"target_h{h}"] = grouped.shift(-h)
    return result


def add_lag_features(
    df: pd.DataFrame,
    pair_cols: list[str] = PAIR_COLS,
    date_col: str = "Tanggal",
    qty_col: str = "Kuantitas",
    lags: list[int] = LAG_DAYS,
) -> pd.DataFrame:
    result = df.sort_values(pair_cols + [date_col]).reset_index(drop=True)
    grouped = result.groupby(pair_cols)[qty_col]
    for lag in lags:
        result[f"lag_{lag}"] = grouped.shift(lag)
    return result


def add_rolling_features(
    df: pd.DataFrame,
    pair_cols: list[str] = PAIR_COLS,
    date_col: str = "Tanggal",
    qty_col: str = "Kuantitas",
    windows: list[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    result = df.sort_values(pair_cols + [date_col]).reset_index(drop=True)
    # Shift by 1 before rolling so the window for row t covers strictly
    # preceding days (t-window .. t-1), never t's own Kuantitas — this is
    # what makes the rolling stats safe to use as prediction-time features.
    result["_shifted_qty"] = result.groupby(pair_cols)[qty_col].shift(1)
    for window in windows:
        rolled = result.groupby(pair_cols)["_shifted_qty"].rolling(window, min_periods=window)
        result[f"roll_mean_{window}"] = rolled.mean().reset_index(level=pair_cols, drop=True)
        result[f"roll_std_{window}"] = rolled.std().reset_index(level=pair_cols, drop=True)
    return result.drop(columns=["_shifted_qty"])


def compute_branch_stats(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_START,
    branch_col: str = BRANCH_COL,
    date_col: str = "Tanggal",
    qty_col: str = "Kuantitas",
) -> pd.DataFrame:
    # Leakage guard: filter to strictly-pre-cutoff (training-period) rows
    # BEFORE computing any branch-level aggregate. Everything below this
    # line must only ever see `train`, never `df`.
    train = df[df[date_col] < cutoff]
    daily_totals = train.groupby([branch_col, date_col])[qty_col].sum().reset_index()
    stats = daily_totals.groupby(branch_col)[qty_col].agg(
        branch_avg_daily_qty="mean", branch_demand_std="std"
    ).reset_index()
    stats["branch_demand_cv"] = stats["branch_demand_std"] / stats["branch_avg_daily_qty"]

    # Deviation from the brief: the brief's exact call —
    # pd.qcut(stats["branch_avg_daily_qty"], q=4, labels=[4 names],
    # duplicates="drop") — raises ValueError whenever fewer than 4 branches
    # (or fewer than 4 distinct avg-qty values) are present in the training
    # data, because duplicates="drop" can collapse bin edges below the
    # fixed 4-label count. Both leakage-safety tests use a single branch,
    # which trips this every time. Fix: size the label list to the number
    # of bins that can actually be formed (bounded by branch count), and
    # rank-break ties first so pd.qcut never needs to drop duplicate edges.
    all_tier_labels = ["small", "medium", "large", "flagship"]
    n_bins = max(1, min(len(all_tier_labels), len(stats)))
    if n_bins == 1:
        # Use pd.Categorical to ensure dtype=category (same as qcut path) even for single bin
        stats["branch_volume_tier"] = pd.Categorical(
            [all_tier_labels[0]] * len(stats), categories=all_tier_labels
        )
    else:
        # Note: rank-based tie-breaking means branches with exactly equal avg volume can land in different tiers,
        # rather than crashing (duplicates='drop') or grouping together.
        stats["branch_volume_tier"] = pd.qcut(
            stats["branch_avg_daily_qty"].rank(method="first"),
            q=n_bins,
            labels=all_tier_labels[:n_bins],
        )
    return stats.drop(columns=["branch_demand_std"])
