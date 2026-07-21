import pandas as pd

TEST_START = pd.Timestamp("2025-12-01")
MIN_HISTORY_DAYS = 60
PAIR_COLS = ["Kode Barang", "Nama Cabang"]
CARRY_COLS = ["Kategori Barang", "Nama Barang"]


def build_dense_panel(
    df: pd.DataFrame,
    pair_cols: list[str] = PAIR_COLS,
    date_col: str = "Tanggal",
    qty_col: str = "Kuantitas",
    carry_cols: list[str] = CARRY_COLS,
) -> pd.DataFrame:
    pieces = []
    for keys, group in df.groupby(pair_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = group.sort_values(date_col)
        full_range = pd.date_range(group[date_col].min(), group[date_col].max(), freq="D")
        dense = group.set_index(date_col).reindex(full_range)
        dense[qty_col] = dense[qty_col].fillna(0)
        for col in carry_cols:
            dense[col] = dense[col].ffill().bfill()
        for pair_col, key in zip(pair_cols, keys):
            dense[pair_col] = key
        dense = dense.reset_index().rename(columns={"index": date_col})
        pieces.append(dense[pair_cols + [date_col, qty_col] + carry_cols])
    result = pd.concat(pieces, ignore_index=True)
    return result.sort_values(pair_cols + [date_col]).reset_index(drop=True)
