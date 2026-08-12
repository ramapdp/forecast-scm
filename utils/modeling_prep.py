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


# Syntetos-Boylan cut-offs. ADI = average interval between non-zero demand
# days; CV2 = squared coefficient of variation of the non-zero quantities.
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49

TEST_START = pd.Timestamp("2025-12-01")


def compute_pair_demand_stats(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_START,
    qty_col: str = "Kuantitas",
    pair_cols: list = None,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    """ADI and CV2 per pair, computed from the training period only.

    Deriving these from the full series would leak post-cutoff behaviour into
    a feature the model trains on.
    """
    pair_cols = pair_cols or PAIR_COLS
    train = df[df[date_col] < cutoff]
    grouped = train.groupby(pair_cols, observed=True)[qty_col]

    n_days = grouped.size()
    n_nonzero = grouped.apply(lambda s: int((s > 0).sum()))
    nz_mean = grouped.apply(lambda s: s[s > 0].mean())
    nz_std = grouped.apply(lambda s: s[s > 0].std(ddof=0))

    adi = n_days / n_nonzero.replace(0, np.nan)
    cv2 = (nz_std / nz_mean.replace(0, np.nan)) ** 2

    return pd.DataFrame({"adi": adi, "cv2": cv2.fillna(0.0)})


def _segment_label(adi: float, cv2: float) -> str:
    if pd.isna(adi):
        # Never moved during the training period — treat as the hardest case.
        return "lumpy"
    if adi < ADI_THRESHOLD:
        return "smooth" if cv2 < CV2_THRESHOLD else "erratic"
    return "intermittent" if cv2 < CV2_THRESHOLD else "lumpy"


def classify_pairs(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_START,
    qty_col: str = "Kuantitas",
    pair_cols: list = None,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    pair_cols = pair_cols or PAIR_COLS
    stats = compute_pair_demand_stats(
        df, cutoff=cutoff, qty_col=qty_col, pair_cols=pair_cols, date_col=date_col
    )
    labels = stats.apply(lambda r: _segment_label(r["adi"], r["cv2"]), axis=1)
    labels.name = "demand_segment"

    result = df.copy()
    result["demand_segment"] = (
        result.set_index(pair_cols).index.map(labels).astype(object)
    )
    result["demand_segment"] = result["demand_segment"].fillna("lumpy")
    return result


# Five expanding-window folds. Training data for fold k is every row dated
# before FOLD_STARTS[k-1]; validation is that month alone. December 2025 is
# absent on purpose — it is the locked final test set.
FOLD_STARTS = [
    pd.Timestamp("2025-07-01"),
    pd.Timestamp("2025-08-01"),
    pd.Timestamp("2025-09-01"),
    pd.Timestamp("2025-10-01"),
    pd.Timestamp("2025-11-01"),
]


def assign_folds(
    df: pd.DataFrame,
    fold_starts: list = None,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    fold_starts = fold_starts or FOLD_STARTS
    result = df.copy()
    result["fold_id"] = np.nan
    for number, start in enumerate(fold_starts, start=1):
        end = start + pd.offsets.MonthBegin(1)
        in_month = (result[date_col] >= start) & (result[date_col] < end)
        result.loc[in_month, "fold_id"] = float(number)
    return result


def fold_train_mask(
    df: pd.DataFrame,
    fold_id: int,
    fold_starts: list = None,
    date_col: str = DATE_COL,
) -> pd.Series:
    """Rows usable for training fold `fold_id` — strictly before its month."""
    fold_starts = fold_starts or FOLD_STARTS
    if not 1 <= fold_id <= len(fold_starts):
        raise ValueError(f"fold_id harus 1..{len(fold_starts)}, dapat {fold_id}")
    return df[date_col] < fold_starts[fold_id - 1]


CATEGORICAL_COLS = [
    "Kode Barang",
    "Nama Cabang",
    "Kategori Barang",
    "kota",
    "hari_pengiriman",
    "branch_volume_tier",
    "demand_segment",
]

UNKNOWN_TOKEN = "<UNKNOWN>"
UNKNOWN_INDEX = 0


def build_category_mapping(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_START,
    cols: list = None,
    date_col: str = DATE_COL,
) -> dict:
    """Fit value -> index maps from the training period only.

    Fitting on train only is a correctness requirement, not tidiness: SCM reruns
    this weekly on fresh data, and a 60th branch opening next month must not
    renumber the existing 59 and silently invalidate a trained model.
    """
    cols = cols or CATEGORICAL_COLS
    train = df[df[date_col] < cutoff] if date_col in df.columns else df
    mapping = {}
    for col in cols:
        values = sorted(str(v) for v in train[col].dropna().unique())
        mapping[col] = {UNKNOWN_TOKEN: UNKNOWN_INDEX}
        for index, value in enumerate(values, start=1):
            mapping[col][value] = index
    return mapping


def encode_categoricals(
    df: pd.DataFrame,
    mapping: dict,
    cols: list = None,
) -> pd.DataFrame:
    cols = cols or CATEGORICAL_COLS
    result = df.copy()
    for col in cols:
        result[f"{col}_idx"] = (
            result[col].astype(str).map(mapping[col]).fillna(UNKNOWN_INDEX).astype(int)
        )
    return result


def save_category_mapping(mapping: dict, path: str = CATEGORY_MAPPING_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2, sort_keys=True)


def load_category_mapping(path: str = CATEGORY_MAPPING_FILE) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
