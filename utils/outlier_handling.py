import pandas as pd

from . import build_panel

PAIR_COLS = build_panel.PAIR_COLS
TEST_START = build_panel.TEST_START
MIN_PAIR_HISTORY = 30
SPIKE_RATIO_THRESHOLD = 5.0
EVENT_FLAG_COLS = [
    "is_ramadan", "is_eid_al_fitr", "is_eid_al_adha", "is_independence_day", "is_new_year",
]


def compute_pair_baseline(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_START,
    pair_cols: list[str] = PAIR_COLS,
    date_col: str = "Tanggal",
    qty_col: str = "Kuantitas",
    min_history: int = MIN_PAIR_HISTORY,
) -> pd.DataFrame:
    # Leakage guard: filter to strictly-pre-cutoff (training-period) rows
    # BEFORE computing any per-pair aggregate — mirrors compute_branch_stats.
    # Kuantitas == 0 rows are always build_panel gap-fill days (raw
    # Kuantitas is never 0 in the source data), never real transactions, so
    # excluding them here recovers the same median/count as computing on
    # the pre-panel transactional data directly.
    train = df[(df[date_col] < cutoff) & (df[qty_col] > 0)]
    stats = (
        train.groupby(pair_cols)[qty_col]
        .agg(pair_count="count", pair_median="median")
        .reset_index()
    )
    stats["pair_eligible"] = (stats["pair_count"] >= min_history) & (stats["pair_median"] > 0)
    return stats.drop(columns=["pair_count"])


def apply_outlier_capping(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    ratio_threshold: float = SPIKE_RATIO_THRESHOLD,
    pair_cols: list[str] = PAIR_COLS,
    qty_col: str = "Kuantitas",
    event_cols: list[str] = EVENT_FLAG_COLS,
) -> pd.DataFrame:
    result = df.merge(baseline_df, on=pair_cols, how="left")
    result["pair_eligible"] = result["pair_eligible"].fillna(False)
    result["baseline_ratio"] = result[qty_col] / result["pair_median"]
    result.loc[~result["pair_eligible"], "baseline_ratio"] = float("nan")
    result["is_spike"] = result["pair_eligible"] & (result["baseline_ratio"] >= ratio_threshold)

    in_event_window = result[event_cols].any(axis=1)
    should_cap = result["is_spike"] & ~in_event_window
    cap_value = result["pair_median"] * ratio_threshold
    result["Kuantitas_capped"] = result[qty_col].where(~should_cap, cap_value)

    return result.drop(columns=["pair_median", "pair_eligible"])
