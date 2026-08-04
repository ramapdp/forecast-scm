import re
from typing import Optional, Union
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

OUTLETS_FILE = str(BASE_DIR / "dataset/outlets.csv")
OVERRIDES_FILE = str(BASE_DIR / "dataset/outlet_name_overrides.csv")

PREFIX_RE = re.compile(r"^KY\d+\s*-\s*", re.IGNORECASE)
TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")

CHANNEL_COLS = ["has_shopee", "has_gofood", "has_grabfood"]


def load_outlets(path: str = OUTLETS_FILE) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def load_overrides(path: str = OVERRIDES_FILE) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)


def _strip_for_matching(nama_cabang: str) -> str:
    stripped = PREFIX_RE.sub("", nama_cabang)
    stripped = TRAILING_PAREN_RE.sub("", stripped)
    return stripped.strip()


def match_branch_to_outlet(
    nama_cabang: str, outlets_df: pd.DataFrame, overrides_df: pd.DataFrame
) -> tuple[Optional[pd.Series], Optional[str]]:
    override_row = overrides_df[overrides_df["Nama Cabang"] == nama_cabang]
    if not override_row.empty:
        nama_outlet = override_row.iloc[0]["Nama Outlet"]
        kota_override = override_row.iloc[0]["Kota Override"]
        outlet_row = outlets_df[outlets_df["Nama Outlet"] == nama_outlet]
        if outlet_row.empty:
            return None, None
        return outlet_row.iloc[0], (kota_override if pd.notna(kota_override) and kota_override else None)

    stripped = _strip_for_matching(nama_cabang)
    candidates = outlets_df[
        outlets_df["Nama Outlet"].str.casefold().apply(lambda name: stripped.casefold() in name)
    ]
    if len(candidates) == 1:
        return candidates.iloc[0], None
    return None, None


def normalize_kota(kota: Optional[str], kota_override: Optional[str]) -> str:
    if kota_override:
        return kota_override.strip()
    if kota is None or (isinstance(kota, float) and pd.isna(kota)):
        return "Unknown"
    return kota.strip()


def _to_bool(value) -> Union[bool, float]:
    if pd.isna(value):
        return float("nan")
    return value.strip().lower() == "yes"


def filter_matched_branches(
    df: pd.DataFrame,
    outlets_df: pd.DataFrame,
    overrides_df: pd.DataFrame,
    branch_col: str = "Nama Cabang",
) -> pd.DataFrame:
    branches = df[branch_col].unique()
    matched = {
        b for b in branches
        if match_branch_to_outlet(b, outlets_df, overrides_df)[0] is not None
    }
    return df[df[branch_col].isin(matched)].reset_index(drop=True)


def canonicalize_branch_names(
    df: pd.DataFrame,
    outlets_df: pd.DataFrame,
    overrides_df: pd.DataFrame,
    branch_col: str = "Nama Cabang",
) -> pd.DataFrame:
    branches = df[branch_col].unique()
    canonical_names = {}
    for nama_cabang in branches:
        outlet_row, _ = match_branch_to_outlet(nama_cabang, outlets_df, overrides_df)
        canonical_names[nama_cabang] = (
            nama_cabang if outlet_row is None else outlet_row["Nama Outlet"]
        )
    result = df.copy()
    result[branch_col] = result[branch_col].map(canonical_names)
    return result


def build_outlet_features(
    branch_names: list[str], outlets_df: pd.DataFrame, overrides_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for nama_cabang in branch_names:
        outlet_row, kota_override = match_branch_to_outlet(nama_cabang, outlets_df, overrides_df)
        if outlet_row is None:
            rows.append(
                {
                    "Nama Cabang": nama_cabang,
                    "kota": "Unknown",
                    "has_shopee": float("nan"),
                    "has_gofood": float("nan"),
                    "has_grabfood": float("nan"),
                    "can_order_online": float("nan"),
                }
            )
            continue

        channels = {col: _to_bool(outlet_row[col]) for col in CHANNEL_COLS}
        can_order_online = (
            float("nan")
            if any(pd.isna(v) for v in channels.values())
            else any(channels.values())
        )
        rows.append(
            {
                "Nama Cabang": nama_cabang,
                "kota": normalize_kota(outlet_row["Kota"], kota_override),
                **channels,
                "can_order_online": can_order_online,
            }
        )
    return pd.DataFrame(rows)
