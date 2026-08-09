import re
from typing import Optional, Union
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

OUTLETS_FILE = str(BASE_DIR / "dataset/outlets.csv")
OVERRIDES_FILE = str(BASE_DIR / "dataset/outlet_name_overrides.csv")
REGION_MAPPING_FILE = str(BASE_DIR / "dataset/outlet_mapping.csv")

PREFIX_RE = re.compile(r"^KY\d+\s*-\s*", re.IGNORECASE)
TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")

CHANNEL_COLS = ["has_shopee", "has_gofood", "has_grabfood"]

INDONESIAN_WEEKDAYS = {
    "Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3,
    "Jumat": 4, "Sabtu": 5, "Minggu": 6,
}


def load_outlets(path: str = OUTLETS_FILE) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def load_overrides(path: str = OVERRIDES_FILE) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)


def load_region_mapping(path: str = REGION_MAPPING_FILE) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def parse_delivery_days(hari_pengiriman: str) -> set[int]:
    tokens = re.split(r"\bdan\b|,", hari_pengiriman)
    days = set()
    for token in tokens:
        name = token.strip()
        if not name:
            continue
        if name not in INDONESIAN_WEEKDAYS:
            raise ValueError(f"Unrecognized weekday token: {name!r} in {hari_pengiriman!r}")
        days.add(INDONESIAN_WEEKDAYS[name])
    return days


def compute_lead_time_days(day_of_week: int, delivery_days: set[int]) -> int:
    # Always strictly forward: if day_of_week is itself a delivery day, this
    # returns the days to the *next* occurrence, never 0.
    for d in range(1, 8):
        if (day_of_week + d) % 7 in delivery_days:
            return d
    raise ValueError(f"No delivery day found for day_of_week={day_of_week}")


def apply_region_features(
    df: pd.DataFrame,
    region_df: pd.DataFrame,
    branch_col: str = "Nama Cabang",
    date_col: str = "Tanggal",
) -> pd.DataFrame:
    # region_df's `new_name` matches Nama Cabang once branches are canonicalized
    # (see canonicalize_branch_names) — join is a straight rename, no fuzzy matching needed.
    features = region_df[["new_name", "kawasan", "hari_pengiriman"]].rename(
        columns={"new_name": branch_col}
    )
    result = df.merge(features, on=branch_col, how="left")

    # lead_time_days depends only on (day_of_week, hari_pengiriman) — a handful
    # of distinct combinations even across millions of rows — so compute a
    # small lookup table once and merge back, instead of .apply() per row.
    # itertuples() can't expose a leading-underscore column as an attribute
    # (renamed positionally instead), so avoid that name for the temp column.
    result["day_of_week_tmp"] = result[date_col].dt.weekday
    combos = result[["day_of_week_tmp", "hari_pengiriman"]].drop_duplicates().dropna()
    # list comprehension over itertuples, not .apply(axis=1) — .apply() on an
    # empty frame (all branches unmatched) can't infer the output is a Series
    # and returns an empty DataFrame instead, breaking the assignment below.
    combos["lead_time_days"] = [
        compute_lead_time_days(int(row.day_of_week_tmp), parse_delivery_days(row.hari_pengiriman))
        for row in combos.itertuples()
    ]
    result = result.merge(combos, on=["day_of_week_tmp", "hari_pengiriman"], how="left")
    return result.drop(columns=["day_of_week_tmp"])


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
