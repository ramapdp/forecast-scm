from typing import Optional

import pandas as pd

TEST_START = pd.Timestamp("2025-12-01")
MIN_HISTORY_DAYS = 60
PAIR_COLS = ["Kode Barang", "Nama Cabang"]
CARRY_COLS = ["Kategori Barang", "Nama Barang"]
SEGMENT_COL = "segment_id"


def filter_min_history(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_START,
    min_days: int = MIN_HISTORY_DAYS,
    pair_cols: list[str] = PAIR_COLS,
    date_col: str = "Tanggal",
) -> pd.DataFrame:
    pre_cutoff_counts = (
        df[df[date_col] < cutoff]
        .groupby(pair_cols)
        .size()
        .reset_index(name="pre_cutoff_days")
    )
    valid_pairs = pre_cutoff_counts[pre_cutoff_counts["pre_cutoff_days"] >= min_days][pair_cols]
    return df.merge(valid_pairs, on=pair_cols, how="inner").reset_index(drop=True)


def _drop_closed_dates(dates: pd.DatetimeIndex, intervals: list) -> pd.DatetimeIndex:
    values = pd.Series(dates)
    keep = pd.Series(True, index=values.index)
    for start, end in intervals:
        closed = values >= start if end is None else (values >= start) & (values < end)
        keep &= ~closed
    return pd.DatetimeIndex(values[keep])


def _segment_ids(dates: pd.Series) -> pd.Series:
    # A new segment begins wherever two kept dates are more than one day
    # apart — which is exactly where a closure interval was removed.
    starts_new_segment = dates.diff().dt.days.fillna(1) > 1
    return starts_new_segment.cumsum().astype(int) + 1


def build_dense_panel(
    df: pd.DataFrame,
    pair_cols: list[str] = PAIR_COLS,
    date_col: str = "Tanggal",
    qty_col: str = "Kuantitas",
    carry_cols: list[str] = CARRY_COLS,
    closures: Optional[dict] = None,
    branch_col: str = "Nama Cabang",
) -> pd.DataFrame:
    """Reindex each pair to a dense daily panel over its own active range.

    Days inside a recorded closure interval for the pair's branch produce no
    rows at all: the outlet did not exist then, so zero-filling them would
    fabricate demand history. Each contiguous run of kept dates is numbered
    into SEGMENT_COL, and callers group by pair + segment so no lag, rolling
    window, target shift, or LSTM sequence ever bridges a closure.

    closures=None reproduces the pre-segmentation behaviour exactly, with
    segment_id == 1 everywhere.
    """
    closures = closures or {}
    pieces = []
    for keys, group in df.groupby(pair_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = group.sort_values(date_col)
        full_range = pd.date_range(group[date_col].min(), group[date_col].max(), freq="D")

        branch = dict(zip(pair_cols, keys)).get(branch_col)
        active_range = _drop_closed_dates(full_range, closures.get(branch, []))
        if len(active_range) == 0:
            continue

        dense = group.set_index(date_col).reindex(active_range)
        dense[qty_col] = dense[qty_col].fillna(0)
        for col in carry_cols:
            dense[col] = dense[col].ffill().bfill()
        for pair_col, key in zip(pair_cols, keys):
            dense[pair_col] = key
        dense = dense.reset_index().rename(columns={"index": date_col})
        dense[SEGMENT_COL] = _segment_ids(dense[date_col])
        pieces.append(dense[pair_cols + [date_col, qty_col] + carry_cols + [SEGMENT_COL]])

    result = pd.concat(pieces, ignore_index=True)
    return result.sort_values(pair_cols + [date_col]).reset_index(drop=True)
