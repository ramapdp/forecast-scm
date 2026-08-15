from typing import Optional

import numpy as np
import pandas as pd
from pathlib import Path

from . import build_panel
from . import normalize_items
from . import calendar_features
from . import outlet_features
from . import outlier_handling

BASE_DIR = Path(__file__).resolve().parent.parent

PAIR_COLS = build_panel.PAIR_COLS
SEGMENT_COL = build_panel.SEGMENT_COL
# The grouping key for every shift-based feature. Grouping by pair alone would
# let a lag, rolling window, or target reach across a closure gap and treat two
# sides of a months-long shutdown as consecutive days.
SEGMENT_COLS = PAIR_COLS + [SEGMENT_COL]
TEST_START = build_panel.TEST_START
BRANCH_COL = "Nama Cabang"
TARGET_HORIZONS = range(1, 8)
LAG_DAYS = [1, 2, 3, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]
MODEL_READY_DIR = str(BASE_DIR / "dataset/model_ready")

# The exact columns engineer_features() produces. Asserted by main() so a
# feature added on one code path but not the other fails loudly instead of
# silently producing a short parquet.
FEATURED_COLUMNS = [
    "Kode Barang", "Nama Cabang", "Tanggal", "Kuantitas", "Kategori Barang",
    "Nama Barang", "segment_id", "day_of_week", "day_of_month", "month", "is_weekend",
    "is_national_holiday", "is_ramadan", "days_into_ramadan", "days_until_ramadan",
    "is_eid_al_fitr", "days_since_eid_al_fitr", "days_until_eid_al_fitr",
    "is_eid_al_adha", "days_since_eid_al_adha", "days_until_eid_al_adha",
    "is_independence_day", "days_since_independence_day", "days_until_independence_day",
    "is_new_year", "days_since_new_year", "days_until_new_year",
    "baseline_ratio", "is_spike", "Kuantitas_capped",
    "target_h1", "target_h2", "target_h3", "target_h4", "target_h5", "target_h6", "target_h7",
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
    "roll_mean_7", "roll_std_7", "roll_mean_14", "roll_std_14",
    "roll_mean_28", "roll_std_28",
    "kawasan", "hari_pengiriman", "lead_time_days", "kota",
    "has_shopee", "has_gofood", "has_grabfood", "can_order_online",
    "target_lead_time_cumulative", "days_since_relocation",
    "branch_avg_daily_qty", "branch_demand_cv", "branch_volume_tier", "branch_age_days",
]


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


def add_lead_time_target(
    df: pd.DataFrame,
    pair_cols: list[str] = PAIR_COLS,
    date_col: str = "Tanggal",
    qty_col: str = "Kuantitas",
    lead_time_col: str = "lead_time_days",
) -> pd.DataFrame:
    result = df.sort_values(pair_cols + [date_col]).reset_index(drop=True)
    rev = result.iloc[::-1]

    # For row H we want sum(Kuantitas[H+1 .. H+w]). On the reversed-order
    # frame, "shift(1) then trailing rolling(w)" walks the opposite direction
    # from the real date axis, turning a trailing window into the forward
    # window we need — same shift(1) leakage guard as add_rolling_features,
    # just applied on the reversed axis. lead_time_days only takes a handful
    # of distinct values, so compute one forward-sum column per distinct
    # window rather than per row.
    distinct_windows = sorted(int(w) for w in result[lead_time_col].dropna().unique())
    fwd_cols = []
    for w in distinct_windows:
        col = f"_fwd_sum_{w}"
        fwd = rev.groupby(pair_cols)[qty_col].shift(1).rolling(w, min_periods=w).sum()
        result[col] = fwd.reindex(result.index)
        fwd_cols.append(col)

    if distinct_windows:
        conditions = [result[lead_time_col] == w for w in distinct_windows]
        choices = [result[c] for c in fwd_cols]
        result["target_lead_time_cumulative"] = np.select(conditions, choices, default=np.nan)
    else:
        result["target_lead_time_cumulative"] = np.nan
    return result.drop(columns=fwd_cols)


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


def export_featured(df: pd.DataFrame, output_dir: str = MODEL_READY_DIR) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "featured.parquet", index=False)


def engineer_features(
    df: pd.DataFrame,
    outlets_df: pd.DataFrame,
    overrides_df: pd.DataFrame,
    region_df: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_START,
    min_pair_history: int = outlier_handling.MIN_PAIR_HISTORY,
    spike_ratio_threshold: float = outlier_handling.SPIKE_RATIO_THRESHOLD,
) -> pd.DataFrame:
    """Single definition of the feature-engineering step order.

    Both the scripted pipeline and notebook/data-processing.ipynb call this,
    so a step added here can never be missed by one of the two paths.
    """
    df = calendar_features.add_calendar_features(df)
    pair_baseline = outlier_handling.compute_pair_baseline(
        df, cutoff=cutoff, min_history=min_pair_history
    )
    df = outlier_handling.apply_outlier_capping(
        df, pair_baseline, ratio_threshold=spike_ratio_threshold
    )
    df = add_targets(df, pair_cols=SEGMENT_COLS)
    df = outlet_features.apply_region_features(df, region_df)
    df = apply_outlet_features(df, outlets_df, overrides_df)
    df = outlet_features.add_relocation_feature(df)
    df = add_lead_time_target(df, pair_cols=SEGMENT_COLS)
    df = add_lag_features(df, pair_cols=SEGMENT_COLS, qty_col="Kuantitas_capped")
    df = add_rolling_features(df, pair_cols=SEGMENT_COLS, qty_col="Kuantitas_capped")
    branch_stats = compute_branch_stats(df, cutoff=cutoff, qty_col="Kuantitas_capped")
    df = apply_branch_stats(df, branch_stats)
    df = add_branch_age_days(df)
    return df


def build_featured_dataset(
    input_path: str = normalize_items.RAW_DATA_FILE,
    min_history_days: int = build_panel.MIN_HISTORY_DAYS,
    cutoff: pd.Timestamp = TEST_START,
    outlets_path: str = outlet_features.OUTLETS_FILE,
    overrides_path: str = outlet_features.OVERRIDES_FILE,
    region_path: str = outlet_features.REGION_MAPPING_FILE,
    closures_path: str = outlet_features.CLOSURES_FILE,
    min_pair_history: int = outlier_handling.MIN_PAIR_HISTORY,
    spike_ratio_threshold: float = outlier_handling.SPIKE_RATIO_THRESHOLD,
) -> pd.DataFrame:
    outlets_df = outlet_features.load_outlets(outlets_path)
    overrides_df = outlet_features.load_overrides(overrides_path)
    region_df = outlet_features.load_region_mapping(region_path)
    closures = outlet_features.load_closures(closures_path)
    df = normalize_items.load_and_normalize(input_path)
    df = outlet_features.filter_matched_branches(df, outlets_df, overrides_df)
    df = outlet_features.canonicalize_branch_names(df, outlets_df, overrides_df)
    df = normalize_items.reaggregate_daily(df)

    for finding in outlet_features.detect_unrecorded_gaps(df, closures):
        print(
            f"[WARN] Cabang {finding['branch']!r} punya gap {finding['gap_days']} hari "
            f"({finding['gap_start'].date()}..{finding['gap_end'].date()}) "
            f"belum tercatat di {closures_path} — hari-hari itu akan diisi nol."
        )

    df = build_panel.build_dense_panel(df, closures=closures)
    df = build_panel.filter_min_history(df, cutoff=cutoff, min_days=min_history_days)
    return engineer_features(
        df,
        outlets_df=outlets_df,
        overrides_df=overrides_df,
        region_df=region_df,
        cutoff=cutoff,
        min_pair_history=min_pair_history,
        spike_ratio_threshold=spike_ratio_threshold,
    )


def run_qa_checks(df: pd.DataFrame) -> None:
    """Assertions that previously lived only in notebook/data-processing.ipynb.

    Called from main() so the scripted path is verified too. Raises
    AssertionError with an Indonesian message naming the failure.
    """
    assert (df["Kuantitas"] >= 0).all(), "Ditemukan Kuantitas negatif"

    dupes = df.duplicated(subset=["Kode Barang", "Nama Cabang", "Tanggal"]).sum()
    assert dupes == 0, f"Ditemukan {dupes} baris duplikat (item, cabang, tanggal)"

    assert (df["Kuantitas_capped"] <= df["Kuantitas"]).all(), (
        "Kuantitas_capped melebihi Kuantitas mentah"
    )

    assert (df["kota"] != "Unknown").all(), "Ditemukan cabang dengan kota 'Unknown'"

    assert df["kawasan"].notna().all(), "Ditemukan cabang tanpa kawasan"

    kota_per_cabang = df.groupby("Nama Cabang", observed=True)["kota"].nunique()
    bad = kota_per_cabang[kota_per_cabang > 1]
    assert bad.empty, f"Cabang memetakan ke lebih dari satu kota: {list(bad.index)}"


def main(
    input_path: str = normalize_items.RAW_DATA_FILE,
    output_dir: str = MODEL_READY_DIR,
    min_history_days: int = build_panel.MIN_HISTORY_DAYS,
    cutoff: pd.Timestamp = TEST_START,
    outlets_path: str = outlet_features.OUTLETS_FILE,
    overrides_path: str = outlet_features.OVERRIDES_FILE,
    region_path: str = outlet_features.REGION_MAPPING_FILE,
    min_pair_history: int = outlier_handling.MIN_PAIR_HISTORY,
    spike_ratio_threshold: float = outlier_handling.SPIKE_RATIO_THRESHOLD,
) -> None:
    df = build_featured_dataset(
        input_path=input_path,
        min_history_days=min_history_days,
        cutoff=cutoff,
        outlets_path=outlets_path,
        overrides_path=overrides_path,
        region_path=region_path,
        min_pair_history=min_pair_history,
        spike_ratio_threshold=spike_ratio_threshold,
    )
    run_qa_checks(df)
    missing = [c for c in FEATURED_COLUMNS if c not in df.columns]
    assert not missing, f"Kolom hilang dari featured dataset: {missing}"
    export_featured(df, output_dir)
    train, test = split_train_test(df, cutoff=cutoff)
    export_splits(train, test, output_dir)
    print(f"Wrote {len(train)} train rows and {len(test)} test rows to {output_dir}")


if __name__ == "__main__":
    main()
