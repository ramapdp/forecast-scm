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
