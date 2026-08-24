import numpy as np
import pandas as pd

from . import build_panel

PAIR_COLS = build_panel.PAIR_COLS
TEST_START = build_panel.TEST_START
MIN_PAIR_HISTORY = 30
SPIKE_RATIO_THRESHOLD = 5.0
EVENT_FLAG_COLS = [
    "is_ramadan", "is_eid_al_fitr", "is_eid_al_adha", "is_independence_day", "is_new_year",
]


# ── BASELINE COMPUTATION ──────────────────────────────────────────────────────

def compute_pair_baseline(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_START,
    pair_cols: list[str] = PAIR_COLS,
    date_col: str = "Tanggal",
    qty_col: str = "Kuantitas",
    min_history: int = MIN_PAIR_HISTORY,
) -> pd.DataFrame:
    """Hitung median dan statistik baseline per pasangan (item, cabang) dari data training.

    Hanya memakai baris sebelum cutoff dan Kuantitas > 0 (hari nol adalah
    gap-fill dari build_panel, bukan transaksi nyata). Hasil median dipakai
    untuk mendeteksi spike: baris dengan rasio >= SPIKE_RATIO_THRESHOLD dianggap outlier.
    """
    # Leakage guard: filter to strictly-pre-cutoff (training-period) rows
    # BEFORE computing any per-pair aggregate — mirrors compute_branch_stats.
    # Kuantitas == 0 rows are always build_panel gap-fill days (raw
    # Kuantitas is never 0 in the source data), never real transactions, so
    # excluding them here recovers the same median/count as computing on
    # the pre-panel transactional data directly.
    train = df[(df[date_col] < cutoff) & (df[qty_col] > 0)]
    stats = (
        train.groupby(pair_cols)[qty_col]
        .agg(
            pair_count="count",
            pair_median="median",
            # Whether this item has only ever been issued in whole units.
            # Derived from the pair's own history rather than from Satuan:
            # the units are not cleanly split into discrete and continuous
            # (Potong carries 6,510 fractional rows in the real data while
            # PCS and Botol carry none), so the data answers this better
            # than any hand-written list of unit names could.
            pair_integer_only=lambda s: bool((s % 1 == 0).all()),
        )
        .reset_index()
    )
    stats["pair_eligible"] = (stats["pair_count"] >= min_history) & (stats["pair_median"] > 0)
    return stats.drop(columns=["pair_count"])

# ── OUTLIER CAPPING ───────────────────────────────────────────────────────────────

def apply_outlier_capping(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    ratio_threshold: float = SPIKE_RATIO_THRESHOLD,
    pair_cols: list[str] = PAIR_COLS,
    qty_col: str = "Kuantitas",
    event_cols: list[str] = EVENT_FLAG_COLS,
) -> pd.DataFrame:
    """Cap kuantitas outlier ke nilai threshold x median baseline pasangan.

    Spike yang jatuh dalam jendela event (Ramadan, Idul Fitri, dll.) TIDAK
    di-cap: lonjakan pada hari-hari itu adalah permintaan yang sah dan bukan
    noise yang perlu dibersihkan. Kolom Kuantitas_capped tetap <= Kuantitas
    asli — invariant ini diverifikasi oleh run_qa_checks().
    """
    result = df.merge(baseline_df, on=pair_cols, how="left")
    result["pair_eligible"] = result["pair_eligible"].fillna(False)
    result["baseline_ratio"] = result[qty_col] / result["pair_median"]
    result.loc[~result["pair_eligible"], "baseline_ratio"] = float("nan")
    result["is_spike"] = result["pair_eligible"] & (result["baseline_ratio"] >= ratio_threshold)

    in_event_window = result[event_cols].any(axis=1)
    should_cap = result["is_spike"] & ~in_event_window
    cap_value = result["pair_median"] * ratio_threshold

    # A median ending in .5 puts the cap on a half unit, which is meaningless
    # for an item only ever issued whole. Round up rather than to nearest: the
    # success criterion is that the outlet does not run out, so the tie goes
    # to more stock. Clipping back to the raw quantity keeps the
    # Kuantitas_capped <= Kuantitas invariant that run_qa_checks asserts --
    # ceil() can otherwise overshoot a fractional raw value just above the cap.
    if "pair_integer_only" in result.columns:
        whole_unit = result["pair_integer_only"].fillna(False).astype(bool)
        rounded = np.minimum(np.ceil(cap_value), result[qty_col])
        cap_value = cap_value.where(~whole_unit, rounded)
        result = result.drop(columns=["pair_integer_only"])

    result["Kuantitas_capped"] = result[qty_col].where(~should_cap, cap_value)

    return result.drop(columns=["pair_median", "pair_eligible"])
