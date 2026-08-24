"""
Membandingkan permintaan SESUDAH lonjakan yang di-cap dengan permintaan
SEBELUMNYA, pada pasangan (item, cabang) yang sama.

Pertanyaannya (butir A3 / G2 di docs/todolist-proyek.md): lonjakan yang
di-cap itu permintaan tambahan yang benar-benar terjadi, atau sekadar
pembelian borongan yang "meminjam" permintaan hari-hari berikutnya?

Kedua hipotesis punya jejak yang berbeda dan bisa dipisahkan dari data:

  restock / borongan -> hari-hari sesudah lonjakan turun di bawah level
                        sebelum lonjakan (stok ditarik dari gudang outlet)
  permintaan tambahan -> level sesudah lonjakan sama dengan sebelumnya;
                        lonjakan berdiri di ATAS garis dasar, bukan
                        menggantikan permintaan berikutnya

Jalankan sebagai modul dari root repo:
    .venv/bin/python3 -m utils.analyze_spike_recovery

Skrip ini hanya MEMBACA featured.parquet dan mencetak tabel; ia tidak
menulis artefak apa pun dan bukan bagian dari pipeline.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from . import build_panel, outlier_handling

PAIR_COLS = build_panel.PAIR_COLS
# Jendela tidak boleh menyeberangi masa tutup cabang, jadi segment_id ikut
# menjadi kunci -- sama seperti seluruh lag/rolling di pipeline.
SERIES_COLS = PAIR_COLS + ["segment_id"]
DATE_COL = "Tanggal"
QTY_COL = "Kuantitas"
WINDOW = 7
CLUSTER_MIN = 3

BASE_DIR = build_panel.__file__  # hanya untuk kejelasan path di main()
FEATURED_PATH = "dataset/model_ready/featured.parquet"
READ_COLS = [
    "Kode Barang", "Nama Cabang", "Tanggal", "segment_id", "Kuantitas",
    "Kuantitas_capped", "baseline_ratio", "Kategori Barang", "day_of_week",
]


def add_window_means(
    df: pd.DataFrame,
    window: int = WINDOW,
    exclude_capped: bool = False,
    series_cols: list[str] = SERIES_COLS,
    date_col: str = DATE_COL,
    qty_col: str = QTY_COL,
) -> pd.DataFrame:
    """Rata-rata harian `window` hari sebelum dan sesudah setiap baris.

    Hari baris itu sendiri tidak pernah ikut dihitung di kedua sisi. Baris
    di tepi segmen (jendela tidak penuh) mendapat NaN, bukan rata-rata
    sebagian: rata-rata dari 3 hari tidak sebanding dengan rata-rata dari 7.

    `exclude_capped=True` membuang hari yang juga di-cap dari kedua jendela,
    supaya lonjakan tetangga tidak menutupi penurunan yang sedang dicari.
    Kolom `pre_days`/`post_days` mencatat berapa hari yang benar-benar
    tersisa di tiap jendela.
    """
    result = df.sort_values(series_cols + [date_col]).reset_index(drop=True)

    if exclude_capped:
        counted = ~result["is_capped"].astype(bool)
    else:
        counted = pd.Series(True, index=result.index)
    values = result[qty_col].where(counted)
    weights = counted.astype(float)

    grouped = result.groupby(series_cols, sort=False)

    def _lead(series: pd.Series, how: str) -> pd.Series:
        # Jendela sesudah = jendela sebelum pada deret yang dibalik.
        rolled = series[::-1].shift(1).rolling(window, min_periods=1)
        return (rolled.sum() if how == "sum" else rolled.count())[::-1]

    def _lag(series: pd.Series, how: str) -> pd.Series:
        rolled = series.shift(1).rolling(window, min_periods=1)
        return rolled.sum() if how == "sum" else rolled.count()

    result["_pre_sum"] = grouped[values.name].transform(lambda s: _lag(values.loc[s.index], "sum"))
    result["_post_sum"] = grouped[values.name].transform(lambda s: _lead(values.loc[s.index], "sum"))
    result["pre_days"] = grouped[values.name].transform(
        lambda s: _lag(weights.loc[s.index], "sum")).fillna(0)
    result["post_days"] = grouped[values.name].transform(
        lambda s: _lead(weights.loc[s.index], "sum")).fillna(0)

    # Jendela penuh = `window` hari kalender tersedia di segmen ini, terlepas
    # dari berapa yang lolos filter `exclude_capped`.
    calendar = pd.Series(1.0, index=result.index)
    pre_calendar = grouped[values.name].transform(lambda s: _lag(calendar.loc[s.index], "sum"))
    post_calendar = grouped[values.name].transform(lambda s: _lead(calendar.loc[s.index], "sum"))
    # Tiap sisi dinilai sendiri; compare_windows() yang membuang baris
    # begitu salah satu sisinya kosong.
    pre_full = pre_calendar >= window
    post_full = post_calendar >= window

    result["pre_mean"] = (
        result["_pre_sum"] / result["pre_days"].replace(0, np.nan)).where(pre_full)
    result["post_mean"] = (
        result["_post_sum"] / result["post_days"].replace(0, np.nan)).where(post_full)
    return result.drop(columns=["_pre_sum", "_post_sum"])


def count_capped_per_branch_day(
    df: pd.DataFrame,
    branch_col: str = "Nama Cabang",
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    """Berapa banyak item yang di-cap di cabang-hari yang sama.

    Satu item sendirian condong ke permintaan organik; banyak item serentak
    adalah tanda pesanan besar yang dilayani sekaligus.
    """
    counts = (df[df["is_capped"].astype(bool)]
              .groupby([branch_col, date_col]).size()
              .rename("n_capped_same_day").reset_index())
    out = df.merge(counts, on=[branch_col, date_col], how="left")
    out["n_capped_same_day"] = out["n_capped_same_day"].fillna(0).astype(int)
    return out


def compare_windows(rows: pd.DataFrame) -> dict:
    """Ringkasan sesudah-vs-sebelum untuk sekumpulan baris lonjakan."""
    total = len(rows)
    sub = rows.dropna(subset=["pre_mean", "post_mean"])
    if sub.empty:
        return {"n_total": total, "n_compared": 0}
    pre, post = sub["pre_mean"], sub["post_mean"]
    ratio = (post / pre.replace(0, np.nan)).dropna()
    try:
        pvalue = float(scipy_stats.wilcoxon(post, pre, zero_method="wilcox").pvalue)
    except ValueError:  # semua selisih nol
        pvalue = float("nan")
    return {
        "n_total": total,
        "n_compared": len(sub),
        "spike_qty_median": float(sub["Kuantitas"].median()),
        "pre_mean": float(pre.mean()),
        "post_mean": float(post.mean()),
        "pre_median": float(pre.median()),
        "post_median": float(post.median()),
        # Selisih agregat: total unit 7 hari sesudah vs 7 hari sebelum.
        "delta_pct": float((post.sum() / pre.sum() - 1) * 100) if pre.sum() else float("nan"),
        "n_ratio": len(ratio),
        "ratio_median": float(ratio.median()) if len(ratio) else float("nan"),
        "ratio_p25": float(ratio.quantile(0.25)) if len(ratio) else float("nan"),
        "ratio_p75": float(ratio.quantile(0.75)) if len(ratio) else float("nan"),
        "share_post_below_pre": float((post < pre).mean()),
        "share_post_below_80pct": float((post < 0.8 * pre).mean()),
        "wilcoxon_p": pvalue,
    }


def _print_row(label: str, s: dict) -> None:
    if not s.get("n_compared"):
        print(f"{label:<44} (tidak ada baris dengan jendela penuh)")
        return
    print(f"{label:<44} n={s['n_compared']:>5}  sebelum={s['pre_mean']:>7.2f}  "
          f"sesudah={s['post_mean']:>7.2f}  selisih={s['delta_pct']:>+6.1f}%  "
          f"rasio med={s['ratio_median']:.3f}  turun={s['share_post_below_pre']:.1%}  "
          f"p={s['wilcoxon_p']:.3g}")


def main() -> None:
    df = pd.read_parquet(FEATURED_PATH, columns=READ_COLS)
    df["is_capped"] = df["Kuantitas_capped"] < df["Kuantitas"]
    baseline = outlier_handling.compute_pair_baseline(df)
    df = df.merge(baseline[PAIR_COLS + ["pair_median"]], on=PAIR_COLS, how="left")
    df = count_capped_per_branch_day(df)

    for exclude in (False, True):
        w = add_window_means(df, window=WINDOW, exclude_capped=exclude)
        pack = w[w["Kategori Barang"] == "Packaging"]
        capped = pack[pack["is_capped"]]
        mode = ("hari lonjakan lain IKUT dihitung" if not exclude
                else "hari lonjakan lain DIKELUARKAN dari jendela")
        print(f"\n=== Packaging di-cap, jendela {WINDOW} hari ({mode}) ===")
        _print_row("semua baris di-cap", compare_windows(capped))
        _print_row("lonjakan sendirian (1 item/cabang-hari)",
                   compare_windows(capped[capped["n_capped_same_day"] == 1]))
        _print_row(f"lonjakan serentak (>={CLUSTER_MIN} item)",
                   compare_windows(capped[capped["n_capped_same_day"] >= CLUSTER_MIN]))
        control = pack[(~pack["is_capped"]) & pack["baseline_ratio"].between(0.8, 1.25)]
        _print_row("kontrol: hari biasa (rasio 0,8-1,25x)", compare_windows(control))


if __name__ == "__main__":
    main()
