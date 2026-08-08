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
