"""Turn featured.parquet into a model-ready table shared by all three model
families. See docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_READY_DIR = str(BASE_DIR / "dataset/model_ready")
FEATURED_FILE = str(BASE_DIR / "dataset/model_ready/featured.parquet")
MODEL_INPUT_FILE = str(BASE_DIR / "dataset/model_ready/model_input.parquet")
EVENT_ITEMS_FILE = str(BASE_DIR / "dataset/event_driven_items.csv")
CATEGORY_MAPPING_FILE = str(BASE_DIR / "dataset/model_ready/category_mapping.json")
SCALER_FILE = str(BASE_DIR / "dataset/model_ready/scaler_params.json")

PAIR_COLS = ["Kode Barang", "Nama Cabang"]
DATE_COL = "Tanggal"
TARGET_COL = "target_lead_time_cumulative"
LOOKBACK = 28


def load_event_items(path: str = EVENT_ITEMS_FILE) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def add_event_flag(
    df: pd.DataFrame,
    event_items_df: pd.DataFrame,
    item_col: str = "Kode Barang",
) -> pd.DataFrame:
    """Attach the per-SKU is_event_driven flag from event_driven_items.csv.

    Raises rather than defaulting when a SKU is absent from the list: a new SKU
    appearing in a monthly refresh must be classified by the data owner, not
    silently assumed non-event.
    """
    result = df.copy()
    flags = (
        event_items_df.set_index(item_col)["is_event_driven"]
        .astype(str).str.strip().str.lower().eq("true")
    )
    mapped = result[item_col].map(flags)
    if mapped.isna().any():
        missing = sorted(result.loc[mapped.isna(), item_col].unique())
        raise ValueError(
            f"SKU tanpa entri di {EVENT_ITEMS_FILE}: {missing}. "
            "Tambahkan barisnya dan minta klasifikasi dari data owner."
        )
    result["is_event_driven"] = mapped.astype(bool)
    return result
