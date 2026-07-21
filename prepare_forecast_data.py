import pandas as pd
from pathlib import Path

import build_panel

PAIR_COLS = build_panel.PAIR_COLS
TEST_START = build_panel.TEST_START
BRANCH_COL = "Nama Cabang"
TARGET_HORIZONS = range(1, 8)
LAG_DAYS = [1, 2, 3, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]
MODEL_READY_DIR = "dataset/model_ready"


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
