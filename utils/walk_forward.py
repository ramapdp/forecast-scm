"""Walk-forward evaluation, with nothing in it that knows about a model.

Three models are being compared on this data, and a comparison is only worth
reporting if all three saw the same rows. That is not something discipline can
guarantee across three separate training scripts, so it is guaranteed here
instead: this module owns row eligibility, fold boundaries, and scoring, and
takes the model itself as an injected callable.

`fit_predict(train_df, valid_df) -> np.ndarray` is the entire model interface.
Anything a model needs beyond that — feature selection, imputation, target
transforms, scaling — belongs inside its own wrapper, because those are the
choices the comparison is meant to expose rather than hide.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import evaluation, modeling_prep

FOLDS = (1, 2, 3, 4, 5)

# The two axes a global number hides. A MAE dominated by mostly-zero pairs can
# crown a model that only won where predicting zero was easy, and delivery days
# are the rows that actually put goods on a truck.
GROUP_COLS = ("demand_segment", "is_delivery_day")


def eligible_rows(
    df: pd.DataFrame,
    lookback: int = modeling_prep.LOOKBACK,
    date_col: str = modeling_prep.DATE_COL,
    target_col: str = modeling_prep.TARGET_COL,
    test_start: pd.Timestamp = modeling_prep.TEST_START,
) -> pd.DataFrame:
    """Every row a model may see during walk-forward, all columns retained.

    Three cuts, in this order:

    1. December 2025 and later. Redundant with the fold definitions, and kept
       anyway — the cost of one accidental leak is the credibility of the
       final number, and a redundant guard is cheaper than that.
    2. Each segment's first `lookback` days, where the lag and rolling windows
       do not fit yet. Computed on the whole series, never within a fold,
       because a per-fold cut would delete a pair's first 28 days of every
       month.
    3. Rows with no target, which occur at the end of each segment where the
       lead-time window runs past the available data.

    Cuts 2 and 3 are exactly what `modeling_prep.to_tabular()` applies. This
    function reproduces them while keeping every column, because scoring needs
    `demand_segment`, `is_delivery_day` and the baseline inputs that the
    adapter drops. `test_matches_to_tabular_row_for_row` pins the two together.
    """
    frame = df[df[date_col] < test_start]
    frame = modeling_prep.drop_warmup_rows(frame, lookback=lookback, date_col=date_col)
    frame = frame[frame[target_col].notna()]
    return frame.reset_index(drop=True)
