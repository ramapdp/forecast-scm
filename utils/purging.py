"""Keep a training window's labels on its own side of a time boundary.

`target_lead_time_cumulative` sums demand over `H+1 .. H+lead_time_days`, so a
row dated a few days before a cutoff carries a label partly built from days
after it. Training on those rows leaks the validation or test period into the
labels — small in volume (0.2-0.8% of training rows at each boundary here,
since lead_time_days never exceeds 4) but enough to undercut the claim that
the December test set is locked.

This is the standard purging step for walk-forward validation on targets whose
windows overlap the split.
"""

import pandas as pd


def lookahead_safe_mask(
    df: pd.DataFrame,
    boundary: pd.Timestamp,
    date_col: str = "Tanggal",
    lead_time_col: str = "lead_time_days",
) -> pd.Series:
    """True for rows whose whole target window stays strictly before `boundary`.

    A null lead time is safe: without one there is no lead-time target for the
    boundary to contaminate. Rows dated on or after the boundary come out
    False, which is harmless — callers combine this with their own date filter
    and never train on those rows anyway.
    """
    lead_time = pd.to_timedelta(df[lead_time_col].fillna(0), unit="D")
    return df[date_col] + lead_time < boundary
