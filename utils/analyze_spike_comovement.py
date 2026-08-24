"""
Apakah lonjakan Packaging yang berdiri sendiri disertai permintaan makanan
yang ikut tinggi di cabang yang sama pada hari yang sama?

Ini pelengkap `analyze_spike_recovery.py` untuk butir A3/G2
(`docs/todolist-proyek.md`). Skrip sebelumnya menunjukkan lonjakan Packaging
bersifat aditif (tidak ada penurunan sesudahnya). Skrip ini memisahkan dua
sisa kemungkinan:

  hari ramai sungguhan -> Nasi Kebuli / Sambal / Ayam Kebuli di cabang itu
                          ikut di atas median-nya sendiri pada hari itu
  gerakan kemasan saja  -> makanan tetap di level biasa; yang bergerak hanya
                          stok kemasan (mis. penerimaan/penataan gudang)

Pembandingnya selalu cabang yang sama: untuk tiap pasangan (item, cabang),
rasio hari lonjakan diletakkan sebagai persentil di dalam sebaran rasio
pasangan itu sendiri pada hari biasa. Kalau tidak ada kaitan, persentil itu
seragam di [0,1] dengan rata-rata 0,5 -- jadi 0,5 adalah hipotesis nol yang
diuji, bukan angka yang dikarang.

Jalankan sebagai modul dari root repo:
    .venv/bin/python3 -m utils.analyze_spike_comovement

Hanya membaca featured.parquet dan mencetak tabel; tidak menulis artefak.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from . import build_panel

PAIR_COLS = build_panel.PAIR_COLS
BRANCH_DAY = ["Nama Cabang", "Tanggal"]
PACKAGING = "Packaging"

# Tiga item makanan bervolume tertinggi; ketiganya bergerak hampir setiap
# hari di hampir semua cabang, jadi rasionya punya sebaran pembanding.
FOOD_ITEMS = {
    "FGS-00004": "Nasi Kebuli",
    "FGS-00005": "Sambal - FG",
    "FGS-00001": "Ayam Kebuli (0.9)",
}
FEATURED_PATH = "dataset/model_ready/featured.parquet"
READ_COLS = [
    "Kode Barang", "Nama Barang", "Nama Cabang", "Tanggal", "Kuantitas",
    "Kuantitas_capped", "baseline_ratio", "is_spike", "Kategori Barang", "is_weekend",
    "day_of_week",
]


def standalone_packaging_spike_days(df: pd.DataFrame) -> pd.DataFrame:
    """Cabang-hari yang tepat satu barisnya di-cap, dan barang itu Packaging.

    Hitungannya lintas kategori: satu item Packaging melonjak sementara
    tidak ada barang lain di cabang itu yang melonjak pada hari yang sama.
    """
    capped = df[df["is_capped"].astype(bool)]
    per_day = capped.groupby(BRANCH_DAY).size().rename("n_capped").reset_index()
    solo = per_day[per_day["n_capped"] == 1][BRANCH_DAY]
    solo_rows = capped.merge(solo, on=BRANCH_DAY, how="inner")
    return (solo_rows[solo_rows["Kategori Barang"] == PACKAGING]
            [BRANCH_DAY + ["Kode Barang", "Nama Barang", "baseline_ratio", "is_weekend"]]
            .rename(columns={"Kode Barang": "packaging_code",
                             "Nama Barang": "packaging_name",
                             "baseline_ratio": "packaging_ratio"})
            .reset_index(drop=True))


def ordinary_branch_days(df: pd.DataFrame) -> pd.DataFrame:
    """Cabang-hari tanpa lonjakan apa pun -- termasuk lonjakan di jendela
    event yang sengaja tidak di-cap, karena hari seperti itu jelas bukan
    hari biasa meski `Kuantitas_capped` tidak menyentuhnya."""
    flagged = (df.groupby(BRANCH_DAY)["is_spike"].any()
               .rename("any_spike").reset_index())
    return flagged[~flagged["any_spike"]][BRANCH_DAY].reset_index(drop=True)


def percentile_within(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Posisi tiap nilai di dalam sebaran `reference`, 0..1 (midrank untuk
    nilai yang sama, supaya deret dengan banyak angka kembar tidak otomatis
    terdorong ke atas)."""
    ref = np.sort(reference.to_numpy(dtype=float))
    if ref.size == 0:
        return pd.Series(np.nan, index=values.index)
    v = values.to_numpy(dtype=float)
    below = np.searchsorted(ref, v, side="left")
    equal = np.searchsorted(ref, v, side="right") - below
    return pd.Series((below + 0.5 * equal) / ref.size, index=values.index)


def compare_item_ratios(
    item_rows: pd.DataFrame,
    spike_days: pd.DataFrame,
    ordinary: pd.DataFrame,
    ratio_col: str = "baseline_ratio",
    match_cols: tuple = (),
) -> dict:
    """Sebaran rasio item pada hari lonjakan Packaging vs hari biasa.

    Percentil dihitung per cabang, jadi cabang besar tidak menutupi cabang
    kecil dan tidak ada rasio antar-cabang yang dicampur begitu saja.

    `match_cols` menambah syarat kecocokan pada pembanding -- dipakai untuk
    `day_of_week`, karena 65% lonjakan Packaging sendirian jatuh di akhir
    pekan sementara hari biasa hanya 25%. Tanpa pencocokan itu, yang terukur
    bisa jadi cuma "akhir pekan lebih ramai", bukan efek lonjakannya.
    """
    group_cols = ["Nama Cabang"] + list(match_cols)
    spike = item_rows.merge(spike_days[BRANCH_DAY], on=BRANCH_DAY, how="inner")
    # Hanya kunci cabang-hari yang dipakai untuk menyaring; kolom lain
    # (day_of_week dsb.) selalu diambil dari baris item, bukan dari
    # tabel penyaring, supaya tidak ada kolom kembar yang menyamar.
    base = item_rows.merge(ordinary[BRANCH_DAY], on=BRANCH_DAY, how="inner")
    spike = spike.dropna(subset=[ratio_col])
    base = base.dropna(subset=[ratio_col])
    if spike.empty:
        return {"n_spike_days": 0}

    pcts, matched = [], 0
    base_by_branch = {key: g[ratio_col] for key, g in base.groupby(group_cols)}
    for branch, group in spike.groupby(group_cols):
        ref = base_by_branch.get(branch)
        if ref is None or len(ref) < 30:  # cabang tanpa pembanding memadai
            continue
        matched += len(group)
        pcts.append(percentile_within(group[ratio_col], ref))
    pct = pd.concat(pcts) if pcts else pd.Series(dtype=float)

    out = {
        "n_spike_days": len(spike),
        "n_with_reference": matched,
        "spike_ratio_median": float(spike[ratio_col].median()),
        "base_ratio_median": float(base[ratio_col].median()),
        "spike_share_ge_1_5": float((spike[ratio_col] >= 1.5).mean()),
        "base_share_ge_1_5": float((base[ratio_col] >= 1.5).mean()),
        "spike_share_ge_2": float((spike[ratio_col] >= 2.0).mean()),
        "base_share_ge_2": float((base[ratio_col] >= 2.0).mean()),
        "spike_share_zero": float((spike[ratio_col] == 0).mean()),
        "base_share_zero": float((base[ratio_col] == 0).mean()),
    }
    out["match_cols"] = list(match_cols)
    if len(pct) > 20:
        out["pct_mean"] = float(pct.mean())
        out["pct_median"] = float(pct.median())
        out["pct_share_above_p90"] = float((pct >= 0.9).mean())
        # H0: hari lonjakan Packaging = hari acak di cabang itu -> pct ~ U(0,1)
        try:
            out["pct_p_value"] = float(scipy_stats.wilcoxon(pct - 0.5).pvalue)
        except ValueError:  # semua persentil tepat 0,5 -> tidak ada selisih
            out["pct_p_value"] = float("nan")
    return out


def main() -> None:
    df = pd.read_parquet(FEATURED_PATH, columns=READ_COLS)
    df["is_capped"] = df["Kuantitas_capped"] < df["Kuantitas"]

    spike_days = standalone_packaging_spike_days(df)
    ordinary = ordinary_branch_days(df)
    print(f"Cabang-hari dengan lonjakan Packaging SENDIRIAN : {len(spike_days)}")
    print(f"Cabang-hari biasa (tanpa lonjakan apa pun)      : {len(ordinary)}")
    print(f"  akhir pekan: lonjakan sendirian {spike_days['is_weekend'].mean():.1%} vs "
          f"hari biasa {df.merge(ordinary, on=BRANCH_DAY)['is_weekend'].mean():.1%}")

    print("\nKemasan apa yang melonjak sendirian (10 teratas):")
    print(spike_days["packaging_name"].value_counts().head(10).to_string())

    for code, name in FOOD_ITEMS.items():
        rows = df[df["Kode Barang"] == code]
        s = compare_item_ratios(rows, spike_days, ordinary)
        dow = compare_item_ratios(rows, spike_days, ordinary, match_cols=["day_of_week"])
        print(f"\n=== {name} ({code}) ===")
        if not s["n_spike_days"]:
            print("  tidak ada hari lonjakan yang bisa dibandingkan")
            continue
        print(f"  hari lonjakan Packaging sendirian yang punya baris item ini: "
              f"{s['n_spike_days']} (dengan pembanding cabang: {s['n_with_reference']})")
        print(f"  rasio (kuantitas / median pasangan) median : "
              f"lonjakan {s['spike_ratio_median']:.3f} vs hari biasa {s['base_ratio_median']:.3f}")
        print(f"  pangsa rasio >= 1,5x : {s['spike_share_ge_1_5']:.1%} vs "
              f"{s['base_share_ge_1_5']:.1%} (hari biasa)")
        print(f"  pangsa rasio >= 2x   : {s['spike_share_ge_2']:.1%} vs "
              f"{s['base_share_ge_2']:.1%}")
        print(f"  pangsa nol (item tidak keluar) : {s['spike_share_zero']:.1%} vs "
              f"{s['base_share_zero']:.1%}")
        if "pct_mean" in s:
            print(f"  persentil di dalam cabangnya sendiri : rata2 {s['pct_mean']:.3f} | "
                  f"median {s['pct_median']:.3f} | >= p90: {s['pct_share_above_p90']:.1%} | "
                  f"p={s['pct_p_value']:.3g}   (H0 = 0,500)")
        if "pct_mean" in dow:
            print(f"  persentil DICOCOKKAN per hari-dalam-minggu : "
                  f"rata2 {dow['pct_mean']:.3f} | median {dow['pct_median']:.3f} | "
                  f">= p90: {dow['pct_share_above_p90']:.1%} | p={dow['pct_p_value']:.3g} "
                  f"(n={dow['n_with_reference']})")


    print("\nRincian per kemasan yang melonjak (persentil Nasi Kebuli, "
          "dicocokkan per hari-dalam-minggu):")
    nasi = df[df["Kode Barang"] == "FGS-00004"]
    for pkg in spike_days["packaging_name"].value_counts().head(4).index:
        subset = spike_days[spike_days["packaging_name"] == pkg]
        s = compare_item_ratios(nasi, subset, ordinary, match_cols=["day_of_week"])
        if s.get("pct_mean") is None:
            print(f"  {pkg:<24} n={s['n_spike_days']:>4}  (terlalu sedikit untuk persentil)")
            continue
        print(f"  {pkg:<24} n={s['n_spike_days']:>4}  rasio med={s['spike_ratio_median']:.2f}  "
              f"persentil rata2={s['pct_mean']:.3f}  p={s['pct_p_value']:.3g}")


if __name__ == "__main__":
    main()
