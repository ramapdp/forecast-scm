"""Sliding 28-day windows over the panel, without ever materialising them.

The dense tensor this replaces is 1,502,522 x 28 x 56 float32 = 9.42 GB, on a
16 GB machine. The contiguous form below is 294 MB, and every window is a
`sliding_window_view` slice of it — nothing is copied until a batch is built.

Measured on `model_input.parquet` 2026-08-19: the panel has **zero date gaps
inside a segment**. That is what makes position arithmetic equal to date
arithmetic, so no dates are read at batch time. `build_index` re-checks the
property rather than trusting it, because if it ever stops holding, every
window silently spans the wrong days.

This module knows about memory and indices. It knows nothing about LSTMs.
"""

from typing import Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import modeling_prep


def _split_columns(feature_cols: list) -> tuple:
    idx_cols = [col for col in feature_cols if col.endswith("_idx")]
    dynamic_cols = [col for col in feature_cols if not col.endswith("_idx")]
    return dynamic_cols, idx_cols


def build_index(
    panel: pd.DataFrame,
    feature_cols: Optional[list] = None,
    lookback: int = modeling_prep.LOOKBACK,
    date_col: str = modeling_prep.DATE_COL,
    pair_cols: Optional[list] = None,
) -> dict:
    """One contiguous view of the whole panel, plus everything needed to
    address windows inside it.

    The panel passed here is the **full** frame, not `eligible_rows()`. A
    window for a 1 July validation row reaches back into June, over rows that
    the 28-day warm-up cut and the fold purge both remove — rows that appear
    in neither frame `walk_forward.run_fold()` hands a model. Reading their
    *features* is safe: every window ends at its own prediction row, and every
    lag and rolling feature stops at H-1, so no target value can enter a
    window. Purging protects against training on those rows' labels, which
    still never happens.
    """
    feature_cols = list(feature_cols or modeling_prep.FEATURE_COLS)
    dynamic_cols, idx_cols = _split_columns(feature_cols)

    # G2. A target channel inside a window would be a perfect predictor of
    # itself and would not change a single tensor shape.
    if modeling_prep.TARGET_COL in dynamic_cols:
        raise ValueError(
            f"{modeling_prep.TARGET_COL} tidak boleh menjadi kolom dinamis"
        )

    pair_cols = modeling_prep._resolve_pair_cols(panel, pair_cols)
    frame = panel.sort_values(pair_cols + [date_col]).reset_index(drop=True)

    values = np.ascontiguousarray(frame[dynamic_cols].to_numpy(dtype="float32"))
    cats = np.ascontiguousarray(frame[idx_cols].to_numpy(dtype="int16"))
    dates = frame[date_col].to_numpy("datetime64[D]")

    grouped = frame.groupby(pair_cols, observed=True, sort=False)
    positions = grouped.cumcount().to_numpy()
    segment_code = grouped.ngroup().to_numpy()

    _assert_dense(dates, positions, segment_code)
    _assert_windows_fit(dates, positions, segment_code, lookback)

    key_cols = list(pair_cols) + [date_col]
    lookup = pd.Series(
        np.arange(len(frame), dtype=np.int64),
        index=pd.MultiIndex.from_frame(frame[key_cols]),
    )

    return {
        "values": values,
        "cats": cats,
        "dates": dates,
        "positions": positions,
        "segment_code": segment_code,
        "lookup": lookup,
        "key_cols": key_cols,
        "feature_cols": feature_cols,
        "dynamic_cols": dynamic_cols,
        "idx_cols": idx_cols,
        "lookback": lookback,
    }


def _assert_dense(dates, positions, segment_code) -> None:
    """G1, first half: consecutive positions are consecutive days."""
    inside = positions > 0
    if not inside.any():
        return
    step = (dates[1:] - dates[:-1]).astype("timedelta64[D]").astype(np.int64)
    same_segment = segment_code[1:] == segment_code[:-1]
    bad = same_segment & (step != 1)
    if bad.any():
        first = int(np.flatnonzero(bad)[0]) + 1
        raise ValueError(
            f"celah tanggal di dalam segmen pada posisi {first} "
            f"({dates[first - 1]} -> {dates[first]}); "
            "aritmetika posisi tidak lagi sama dengan aritmetika tanggal"
        )


def _assert_windows_fit(dates, positions, segment_code, lookback) -> None:
    """G1 second half and G6: every usable window stays inside one segment,
    spans exactly `lookback` days, and never reaches past its own row.
    """
    ends = np.flatnonzero(positions >= lookback)
    if ends.size == 0:
        return
    starts = ends - lookback + 1
    if not np.array_equal(segment_code[starts], segment_code[ends]):
        raise ValueError("ada window yang melintasi batas segmen")
    span = (dates[ends] - dates[starts]).astype("timedelta64[D]").astype(np.int64)
    if not np.all(span == lookback - 1):
        raise ValueError(
            f"ada window yang tidak mencakup tepat {lookback} hari berurutan"
        )
