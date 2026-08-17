"""Turn featured.parquet into a model-ready table shared by all three model
families. See docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import build_panel
from . import purging

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_READY_DIR = str(BASE_DIR / "dataset/model_ready")
FEATURED_FILE = str(BASE_DIR / "dataset/model_ready/featured.parquet")
MODEL_INPUT_FILE = str(BASE_DIR / "dataset/model_ready/model_input.parquet")
EVENT_ITEMS_FILE = str(BASE_DIR / "dataset/event_driven_items.csv")
CATEGORY_MAPPING_FILE = str(BASE_DIR / "dataset/model_ready/category_mapping.json")
SCALER_FILE = str(BASE_DIR / "dataset/model_ready/scaler_params.json")

PAIR_COLS = ["Kode Barang", "Nama Cabang"]
SEGMENT_COL = "segment_id"
SEGMENT_COLS = PAIR_COLS + [SEGMENT_COL]
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

# Single source, shared with the panel builder. A second literal here would
# let a refresh move one cutoff and not the other, splitting the panel and the
# model input onto different dates without any error.
TEST_START = build_panel.TEST_START


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
    purge: bool = True,
) -> pd.Series:
    """Rows usable for training fold `fold_id` — strictly before its month.

    Purging drops the last few days before the fold boundary, whose lead-time
    label is summed partly over the validation month itself. Same reasoning as
    prepare_forecast_data.split_train_test.
    """
    fold_starts = fold_starts or FOLD_STARTS
    if not 1 <= fold_id <= len(fold_starts):
        raise ValueError(f"fold_id harus 1..{len(fold_starts)}, dapat {fold_id}")
    boundary = fold_starts[fold_id - 1]
    mask = df[date_col] < boundary
    if purge and "lead_time_days" in df.columns:
        mask &= purging.lookahead_safe_mask(df, boundary, date_col=date_col)
    return mask


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
    existing: Optional[dict] = None,
) -> dict:
    """Fit value -> index maps from the training period only.

    Fitting on train only is a correctness requirement, not tidiness: a branch
    that only appears after the cutoff must not enter the mapping at all.

    That alone does not survive a refresh, though. When the cutoff moves
    forward, values that used to sit in the test period cross into the
    training period and join the mapping for real -- and re-sorting the whole
    set renumbers every value sorting after them. Measured on this dataset:
    six new SKUs entering training shift the index of 32 of the 70 existing
    ones, which silently invalidates any model already trained on the old
    numbering. Pass `existing` (normally the previously saved mapping) to keep
    every index already handed out and append new values after the highest one.
    """
    cols = cols or CATEGORICAL_COLS
    existing = existing or {}
    train = df[df[date_col] < cutoff] if date_col in df.columns else df
    mapping = {}
    for col in cols:
        values = sorted(str(v) for v in train[col].dropna().unique())
        # Retired values keep their index: freeing it up for a new value would
        # point an already-trained model at the wrong category.
        previous = dict(existing.get(col) or {})
        previous.setdefault(UNKNOWN_TOKEN, UNKNOWN_INDEX)
        next_index = max(previous.values()) + 1
        for value in values:
            if value not in previous:
                previous[value] = next_index
                next_index += 1
        mapping[col] = previous
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


def load_existing_mapping(path: str = CATEGORY_MAPPING_FILE) -> dict:
    """The saved mapping if there is one, otherwise an empty dict.

    Lets build_model_input() extend the numbering across refreshes on a
    machine that already has a mapping, while a first run on a clean checkout
    still builds one from scratch.
    """
    if not Path(path).exists():
        return {}
    return load_category_mapping(path)


# Only defined inside their proximity window (+/-15 days, +/-30 for Ramadan),
# so null means "outside that window" -- NOT zero, which would read as "the
# event is today" on 84-97% of rows.
EVENT_PROXIMITY_COLS = [
    "days_into_ramadan",
    "days_until_ramadan",
    "days_since_eid_al_fitr",
    "days_until_eid_al_fitr",
    "days_since_eid_al_adha",
    "days_until_eid_al_adha",
    "days_since_independence_day",
    "days_until_independence_day",
    "days_since_new_year",
    "days_until_new_year",
]

# Must exceed every genuine value. days_until_ramadan reaches 70, so 30 would
# collide with real observations.
EVENT_PROXIMITY_SENTINEL = 99.0


# Lag and rolling features, null for the first days of every segment because
# their window does not fit yet. drop_warmup_rows() removes the rows that are
# *predicted* from such a position, but an LSTM window still reaches back over
# them as context, so the tensor carries their nulls unless they are filled.
HISTORY_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
    "roll_mean_7", "roll_std_7", "roll_mean_14", "roll_std_14",
    "roll_mean_28", "roll_std_28",
]


def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the nulls a neural net cannot consume, preserving each column's
    meaning. Tree models do not need this, but both adapters run it so the two
    see identical values.
    """
    result = df.copy()

    for col in EVENT_PROXIMITY_COLS:
        result[col] = result[col].fillna(EVENT_PROXIMITY_SENTINEL)

    result["was_relocated"] = result["days_since_relocation"].notna()
    result["days_since_relocation"] = result["days_since_relocation"].fillna(0.0)

    result["has_baseline"] = result["baseline_ratio"].notna()
    result["baseline_ratio"] = result["baseline_ratio"].fillna(1.0)

    # 0 is a legitimate lag value here — 54% of Kuantitas is zero — so filling
    # with it would be indistinguishable from "no demand that day" without an
    # indicator. Both indicators are row-local: how many windows failed to fit
    # is a monotone function of how far into its segment the row sits, so the
    # count doubles as an ordinal measure of available history without needing
    # to group or sort.
    missing = result[HISTORY_COLS].isna()
    result["missing_history_count"] = missing.sum(axis=1).astype(int)
    result["has_full_history"] = result["missing_history_count"] == 0
    for col in HISTORY_COLS:
        result[col] = result[col].fillna(0.0)

    return result


# The single feature list all three models train on. Written down here so a
# comparison cannot rest on XGBoost and the LSTM having quietly picked
# different columns.
#
# Two deliberate exclusions. `baseline_ratio` is Kuantitas on the row's own day
# divided by a per-pair constant, and `is_spike` is derived from the same
# value; every lag and rolling feature stops at H-1, so including these would
# let the model recover day H's demand and make "known at prediction time" mean
# two different things within one row. Measured cost of dropping them: the
# roll_mean_7 baseline moves from MAE 12.99 to 13.19 when day H is allowed in,
# so the information is worth almost nothing here anyway.
FEATURE_COLS = [
    # demand history
    *HISTORY_COLS,
    "has_full_history", "missing_history_count",
    # calendar
    "day_of_week", "day_of_month", "month", "is_weekend", "is_national_holiday",
    "is_ramadan", "days_into_ramadan", "days_until_ramadan",
    "is_eid_al_fitr", "days_since_eid_al_fitr", "days_until_eid_al_fitr",
    "is_eid_al_adha", "days_since_eid_al_adha", "days_until_eid_al_adha",
    "is_independence_day", "days_since_independence_day", "days_until_independence_day",
    "is_new_year", "days_since_new_year", "days_until_new_year",
    # replenishment cycle
    "kawasan", "lead_time_days", "is_delivery_day", "target_window_weekend_days",
    # outlet
    "has_shopee", "has_gofood", "has_grabfood", "can_order_online",
    "branch_avg_daily_qty", "branch_demand_cv", "branch_age_days",
    "days_since_relocation", "was_relocated",
    # item
    "is_event_driven",
    # encoded categoricals
    "Kode Barang_idx", "Nama Cabang_idx", "Kategori Barang_idx", "kota_idx",
    "hari_pengiriman_idx", "branch_volume_tier_idx", "demand_segment_idx",
]


def _resolve_pair_cols(df: pd.DataFrame, pair_cols: Optional[list]) -> list:
    """Group by segment when the frame carries one, otherwise by pair.

    An explicit pair_cols argument always wins. Falling back on the column's
    presence keeps fixtures and callers that predate segmentation working,
    while guaranteeing that any frame built from the segmented panel never
    lets a warm-up cut or an LSTM window bridge a closure.
    """
    if pair_cols is not None:
        return pair_cols
    return SEGMENT_COLS if SEGMENT_COL in df.columns else PAIR_COLS


def drop_warmup_rows(
    df: pd.DataFrame,
    lookback: int = LOOKBACK,
    pair_cols: list = None,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    """Keep rows whose zero-based position within their own pair is >= lookback.

    These are exactly the rows an LSTM can build a full window for, and exactly
    the rows where lag_28 is non-null. Both adapters cut here so their row sets
    match.
    """
    pair_cols = _resolve_pair_cols(df, pair_cols)
    result = df.sort_values(pair_cols + [date_col]).reset_index(drop=True)
    position = result.groupby(pair_cols, observed=True).cumcount()
    return result[position >= lookback].reset_index(drop=True)


def to_tabular(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = TARGET_COL,
    lookback: int = LOOKBACK,
    pair_cols: list = None,
    date_col: str = DATE_COL,
    log_target: bool = False,
) -> dict:
    """Adapter for XGBoost and Random Forest: a flat table, NaNs left in place.

    Pass the same log_target value here and to to_sequences(), or the contract
    check will fail.
    """
    pair_cols = _resolve_pair_cols(df, pair_cols)
    frame = drop_warmup_rows(df, lookback=lookback, pair_cols=pair_cols, date_col=date_col)
    if log_target:
        frame = frame.copy()
        frame[target_col] = np.log1p(frame[target_col])
    return {
        "X": frame[feature_cols].reset_index(drop=True),
        "y": frame[target_col].reset_index(drop=True),
        "keys": frame[pair_cols + [date_col]].reset_index(drop=True),
        "fold_id": frame["fold_id"].reset_index(drop=True),
    }


def fit_scaler(df: pd.DataFrame, feature_cols: list) -> dict:
    """Per-feature mean and std. Fit on one fold's training rows only —
    fitting globally would leak December statistics into the July fold.
    """
    scaler = {}
    for col in feature_cols:
        mean = float(df[col].mean())
        std = float(df[col].std(ddof=0))
        scaler[col] = (mean, std if std > 0 else 1.0)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: dict, feature_cols: list) -> pd.DataFrame:
    result = df.copy()
    for col in feature_cols:
        mean, std = scaler[col]
        result[col] = (result[col] - mean) / std
    return result


def save_scaler(scaler: dict, path: str = SCALER_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serializable = {col: list(params) for col, params in scaler.items()}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)


def load_scaler(path: str = SCALER_FILE) -> dict:
    with open(path, encoding="utf-8") as handle:
        return {col: tuple(params) for col, params in json.load(handle).items()}


def inverse_log_target(values: np.ndarray) -> np.ndarray:
    """Undo log1p on predictions. Exact for quantile models: quantiles are
    equivariant under monotonic transforms, so expm1(q_a(log1p(y))) == q_a(y).
    """
    return np.expm1(values)


def to_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = TARGET_COL,
    lookback: int = LOOKBACK,
    pair_cols: list = None,
    date_col: str = DATE_COL,
    log_target: bool = False,
) -> dict:
    """Adapter for the LSTM: one (lookback, n_features) window per predictable
    row, the window ending at that row inclusive.

    Produces exactly the rows drop_warmup_rows() keeps, so to_tabular() and
    to_sequences() agree — see validate_contract(). Pass the same log_target
    value to both adapters or the contract check will fail.
    """
    pair_cols = _resolve_pair_cols(df, pair_cols)
    frame = df.sort_values(pair_cols + [date_col]).reset_index(drop=True)
    if log_target:
        frame = frame.copy()
        frame[target_col] = np.log1p(frame[target_col])

    windows, targets, key_rows, folds = [], [], [], []
    for _, group in frame.groupby(pair_cols, observed=True, sort=False):
        values = group[feature_cols].to_numpy(dtype="float32")
        target_values = group[target_col].to_numpy(dtype="float32")
        fold_values = group["fold_id"].to_numpy()
        keys = group[pair_cols + [date_col]].to_numpy()

        for position in range(lookback, len(group)):
            windows.append(values[position - lookback + 1 : position + 1])
            targets.append(target_values[position])
            key_rows.append(keys[position])
            folds.append(fold_values[position])

    if windows:
        stacked = np.stack(windows).astype("float32")
    else:
        stacked = np.empty((0, lookback, len(feature_cols)), dtype="float32")

    return {
        "X": stacked,
        "y": np.asarray(targets, dtype="float32"),
        "keys": pd.DataFrame(key_rows, columns=pair_cols + [date_col]),
        "fold_id": pd.Series(folds, dtype="float64"),
    }


def validate_contract(tabular: dict, sequences: dict, require_finite: bool = True) -> None:
    """Guarantee the two adapters expose the same rows, targets, and folds.

    Without this, "the LSTM is 8% better" could really mean "the LSTM was
    evaluated on a different 5% of the rows".

    require_finite also rejects NaN in either feature block. Matching rows are
    not enough on their own: a tree model consumes NaN natively while an LSTM
    turns it into NaN loss, so a tensor that still carries nulls gets patched
    at training time and the two models silently stop seeing the same inputs.
    Pass False only for a run that is not comparing against the LSTM.
    """
    if require_finite:
        tabular_nan = int(np.isnan(np.asarray(tabular["X"], dtype="float64")).sum())
        assert tabular_nan == 0, (
            f"Fitur tabular mengandung {tabular_nan} NaN — jalankan impute_features()"
        )
        sequence_nan = int(np.isnan(np.asarray(sequences["X"], dtype="float64")).sum())
        assert sequence_nan == 0, (
            f"Tensor sequence mengandung {sequence_nan} NaN — jalankan impute_features()"
        )

    tabular_keys = tabular["keys"].reset_index(drop=True)
    sequence_keys = sequences["keys"].reset_index(drop=True)

    assert len(tabular_keys) == len(sequence_keys), (
        f"Adapter menghasilkan jumlah baris berbeda: "
        f"tabular {len(tabular_keys)}, sequence {len(sequence_keys)}"
    )

    tabular_set = set(map(tuple, tabular_keys.to_numpy()))
    sequence_set = set(map(tuple, sequence_keys.to_numpy()))
    assert tabular_set == sequence_set, (
        f"Adapter menghasilkan baris berbeda: "
        f"{len(tabular_set - sequence_set)} hanya di tabular, "
        f"{len(sequence_set - tabular_set)} hanya di sequence"
    )

    tabular_y = np.asarray(tabular["y"], dtype="float64")
    sequence_y = np.asarray(sequences["y"], dtype="float64")
    assert np.allclose(tabular_y, sequence_y, equal_nan=True), (
        "Nilai target berbeda antar adapter"
    )

    tabular_fold = np.asarray(tabular["fold_id"], dtype="float64")
    sequence_fold = np.asarray(sequences["fold_id"], dtype="float64")
    assert np.allclose(tabular_fold, sequence_fold, equal_nan=True), (
        "Pembagian fold berbeda antar adapter"
    )


def build_model_input(
    featured_path: str = FEATURED_FILE,
    event_items_path: str = EVENT_ITEMS_FILE,
    cutoff: pd.Timestamp = TEST_START,
    mapping_path: str = CATEGORY_MAPPING_FILE,
) -> pd.DataFrame:
    df = pd.read_parquet(featured_path)
    df = add_event_flag(df, load_event_items(event_items_path))
    df = classify_pairs(df, cutoff=cutoff)
    df = assign_folds(df)
    df = impute_features(df)

    mapping = build_category_mapping(
        df, cutoff=cutoff, existing=load_existing_mapping(mapping_path)
    )
    save_category_mapping(mapping, mapping_path)
    df = encode_categoricals(df, mapping)
    return df


def export_model_input(df: pd.DataFrame, path: str = MODEL_INPUT_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> None:
    df = build_model_input()
    export_model_input(df)
    print(f"Wrote {len(df):,} rows x {len(df.columns)} columns to {MODEL_INPUT_FILE}")


if __name__ == "__main__":
    main()
