import pandas as pd
from pathlib import Path

import build_panel
import normalize_items
import calendar_features
import outlet_features

BASE_DIR = Path(__file__).resolve().parent

PAIR_COLS = build_panel.PAIR_COLS
TEST_START = build_panel.TEST_START
BRANCH_COL = "Nama Cabang"
TARGET_HORIZONS = range(1, 8)
LAG_DAYS = [1, 2, 3, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]
MODEL_READY_DIR = str(BASE_DIR / "dataset/model_ready")


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


def apply_branch_stats(
    df: pd.DataFrame, branch_stats: pd.DataFrame, branch_col: str = BRANCH_COL
) -> pd.DataFrame:
    return df.merge(branch_stats, on=branch_col, how="left")


def apply_outlet_features(
    df: pd.DataFrame,
    outlets_df: pd.DataFrame,
    overrides_df: pd.DataFrame,
    branch_col: str = BRANCH_COL,
) -> pd.DataFrame:
    features = outlet_features.build_outlet_features(
        df[branch_col].unique().tolist(), outlets_df, overrides_df
    )
    return df.merge(features, on=branch_col, how="left")


def add_branch_age_days(
    df: pd.DataFrame, branch_col: str = BRANCH_COL, date_col: str = "Tanggal"
) -> pd.DataFrame:
    result = df.copy()
    first_date = result.groupby(branch_col)[date_col].transform("min")
    result["branch_age_days"] = (result[date_col] - first_date).dt.days
    return result


def split_train_test(
    df: pd.DataFrame, cutoff: pd.Timestamp = TEST_START, date_col: str = "Tanggal"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df[date_col] < cutoff].reset_index(drop=True)
    test = df[df[date_col] >= cutoff].reset_index(drop=True)
    return train, test


def export_splits(train: pd.DataFrame, test: pd.DataFrame, output_dir: str = MODEL_READY_DIR) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)


def main(
    input_path: str = normalize_items.RAW_DATA_FILE,
    output_dir: str = MODEL_READY_DIR,
    min_history_days: int = build_panel.MIN_HISTORY_DAYS,
    cutoff: pd.Timestamp = TEST_START,
    outlets_path: str = outlet_features.OUTLETS_FILE,
    overrides_path: str = outlet_features.OVERRIDES_FILE,
) -> None:
    outlets_df = outlet_features.load_outlets(outlets_path)
    overrides_df = outlet_features.load_overrides(overrides_path)
    df = normalize_items.load_and_normalize(input_path)
    df = outlet_features.filter_matched_branches(df, outlets_df, overrides_df)
    df = outlet_features.canonicalize_branch_names(df, outlets_df, overrides_df)
    df = normalize_items.reaggregate_daily(df)
    df = build_panel.build_dense_panel(df)
    df = build_panel.filter_min_history(df, cutoff=cutoff, min_days=min_history_days)
    df = add_targets(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = calendar_features.add_calendar_features(df)
    branch_stats = compute_branch_stats(df, cutoff=cutoff)
    df = apply_branch_stats(df, branch_stats)
    df = add_branch_age_days(df)
    df = apply_outlet_features(df, outlets_df, overrides_df)
    train, test = split_train_test(df, cutoff=cutoff)
    export_splits(train, test, output_dir)
    print(f"Wrote {len(train)} train rows and {len(test)} test rows to {output_dir}")


if __name__ == "__main__":
    main()
