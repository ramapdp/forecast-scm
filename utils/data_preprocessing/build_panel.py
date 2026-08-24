from typing import Optional

import pandas as pd

TEST_START = pd.Timestamp("2025-12-01")
MIN_HISTORY_DAYS = 60
PAIR_COLS = ["Kode Barang", "Nama Cabang"]
CARRY_COLS = ["Kategori Barang", "Nama Barang", "Satuan"]
SEGMENT_COL = "segment_id"


# ── HISTORY FILTER ────────────────────────────────────────────────────────────

def filter_min_history(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_START,
    min_days: int = MIN_HISTORY_DAYS,
    pair_cols: list[str] = PAIR_COLS,
    date_col: str = "Tanggal",
) -> pd.DataFrame:
    """Buang pasangan (item, cabang) yang belum punya riwayat cukup sebelum cutoff.

    Pasangan dengan kurang dari min_days hari sebelum cutoff tidak punya cukup
    konteks untuk membentuk lag dan rolling features — melatih model pada mereka
    hanya menambah noise. Filter ini mencegah pasangan baru masuk ke set training
    sebelum siap.
    """
    pre_cutoff_counts = (
        df[df[date_col] < cutoff]
        .groupby(pair_cols)
        .size()
        .reset_index(name="pre_cutoff_days")
    )
    valid_pairs = pre_cutoff_counts[pre_cutoff_counts["pre_cutoff_days"] >= min_days][pair_cols]
    return df.merge(valid_pairs, on=pair_cols, how="inner").reset_index(drop=True)



# ── CLOSURE & SEGMENT HELPERS ─────────────────────────────────────────────────

def _drop_closed_dates(dates: pd.DatetimeIndex, intervals: list) -> pd.DatetimeIndex:
    """Hapus tanggal yang jatuh dalam interval penutupan cabang dari date range.

    Setiap interval adalah [start, end): start inklusif, end eksklusif.
    end=None berarti cabang masih tutup sampai akhir data.
    """
    values = pd.Series(dates)
    keep = pd.Series(True, index=values.index)
    for start, end in intervals:
        closed = values >= start if end is None else (values >= start) & (values < end)
        keep &= ~closed
    return pd.DatetimeIndex(values[keep])


def _segment_ids(dates: pd.Series, breakpoints: Optional[list] = None) -> pd.Series:
    """Beri nomor segmen berurutan pada deret tanggal yang mungkin terputus.

    Segmen baru dimulai di mana dua tanggal berurutan lebih dari 1 hari
    jaraknya (karena interval tutup dihapus), atau di tanggal breakpoint
    (relokasi cabang yang tidak menutup operasional tapi memutus pola demand).
    Segmen pertama selalu bernomor 1.
    """
    # A new segment begins wherever two kept dates are more than one day
    # apart — which is exactly where a closure interval was removed — or on a
    # breakpoint date, where the outlet kept trading but the demand series
    # itself broke (a relocation to a different market).
    starts_new_segment = dates.diff().dt.days.fillna(1) > 1
    if breakpoints:
        starts_new_segment |= dates.isin(breakpoints)
    # cumsum() counts the run boundaries before each row; a break on the very
    # first date would otherwise number that pair's only segment 2.
    starts_new_segment.iloc[0] = False
    return starts_new_segment.cumsum().astype(int) + 1



# ── PANEL CONSTRUCTION ────────────────────────────────────────────────────────────

def build_dense_panel(
    df: pd.DataFrame,
    pair_cols: list[str] = PAIR_COLS,
    date_col: str = "Tanggal",
    qty_col: str = "Kuantitas",
    carry_cols: list[str] = CARRY_COLS,
    closures: Optional[dict] = None,
    breakpoints: Optional[dict] = None,
    branch_col: str = "Nama Cabang",
) -> pd.DataFrame:
    """Reindex each pair to a dense daily panel over its own active range.

    Days inside a recorded closure interval for the pair's branch produce no
    rows at all: the outlet did not exist then, so zero-filling them would
    fabricate demand history. Each contiguous run of kept dates is numbered
    into SEGMENT_COL, and callers group by pair + segment so no lag, rolling
    window, target shift, or LSTM sequence ever bridges a closure.

    `breakpoints` maps a branch to dates that start a new segment while
    keeping every row. A relocation is the motivating case: the outlet never
    stopped trading, so nothing should be dropped, but it moved to a different
    market and its demand level shifts — measured at 2.2x-2.6x across the
    three relocations with enough post-move data to check. Letting lags and
    rolling windows reach back across that move feeds the model a month of
    inputs from the old location at roughly half the new level.

    closures=None and breakpoints=None reproduce the pre-segmentation
    behaviour exactly, with segment_id == 1 everywhere.
    """
    closures = closures or {}
    breakpoints = breakpoints or {}
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
        dense[SEGMENT_COL] = _segment_ids(dense[date_col], breakpoints.get(branch, []))
        pieces.append(dense[pair_cols + [date_col, qty_col] + carry_cols + [SEGMENT_COL]])

    result = pd.concat(pieces, ignore_index=True)
    return result.sort_values(pair_cols + [date_col]).reset_index(drop=True)
