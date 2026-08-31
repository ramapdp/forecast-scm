"""Rearrangement kuantil (Chernozhukov et al. 2010) pada prediksi XGBoost, dan
hitung ulang K1/K2 dari walk-forward yang dijalankan ulang.

`xgb_walk_forward_results.csv` (run 2026-08-27) cuma skor teragregasi per
(model, fold, quantile) -- prediksi mentah per baris tidak pernah disimpan.
Jadi tidak ada jalan pintas post-hoc murni: rearrangement yang jujur lintas-
fold (leakage-safe, sebanding dengan K1 RF/LSTM yang dihitung dari walk-
forward asli) berarti mem-fit ulang kelima model fold XGBoost dengan
hyperparameter pemenang yang sama (`xgb_best_params.json`). Ongkosnya sama
dengan walk-forward asli, ~3 jam 1 menit (bagian 7 hasil-modeling-xgb.md).

Rearrangement-nya sendiri = urutkan (sort) 19 prediksi kuantil tiap baris
naik. QUANTILE_SET_A sudah berurutan naik (0.05..0.95), jadi sort per baris
menempatkan statistik urutan ke-k pada tau ke-k -- definisi rearrangement
Chernozhukov, dan menjamin crossing_rate = 0 secara struktural sesudahnya
(lihat catatan di model_xgboost.make_fit_predict soal kenapa sort TIDAK
dilakukan di sana secara default).

Prediksi mentah (sebelum sort) direkam lewat closure di sepanjang jalan,
supaya K1/crossing_rate SEBELUM rearrangement bisa dihitung ulang sebagai
pengecekan reproduksibilitas terhadap angka resmi (K1=2,9433,
crossing_rate=0,9767) tanpa mem-fit dua kali.

Jalankan dari repo root: .venv/bin/python3 xgb_rearrangement_walkforward.py
~3 jam di CPU Mac, TANPA checkpoint -- jangan diinterupsi.
"""
import json
import time

import numpy as np
import pandas as pd

from utils.modelling import evaluation, model_xgboost, modeling_prep, walk_forward

BEST_PARAMS_FILE = "dataset/model_ready/xgb_best_params.json"
QUANTILES = evaluation.QUANTILE_SET_A

df = pd.read_parquet("dataset/model_ready/model_input.parquet")

with open(BEST_PARAMS_FILE) as f:
    best = json.load(f)

raw_fit_predict = model_xgboost.make_fit_predict(best, device="cpu")

# posisi panggilan (1..5, urutan walk_forward.FOLDS) -> (actual, prediksi mentah)
captured = {}


def rearranged_fit_predict(train, valid):
    raw = raw_fit_predict(train, valid)
    captured[len(captured) + 1] = (
        valid[modeling_prep.EVAL_TARGET_COL].copy(),
        raw.copy(),
    )
    return np.sort(raw, axis=1)


t0 = time.time()
results = walk_forward.run_walk_forward(
    df, rearranged_fit_predict, model_name="xgboost_rearranged", quantiles=QUANTILES
)
elapsed = time.time() - t0

out_path = "dataset/model_ready/xgb_walk_forward_results_rearranged.csv"
results.to_csv(out_path, index=False)

k1_5fold = walk_forward.pooled_k1(results, "xgboost_rearranged")
k1_clean = walk_forward.pooled_k1(results, "xgboost_rearranged", folds=(1, 2, 4))
crossing_after = walk_forward.pooled_metric(
    results, "xgboost_rearranged", metric="crossing_rate"
)

print(f"selesai dalam {elapsed/3600:.2f} jam, disimpan ke {out_path}")
print("best_iteration per fold:", raw_fit_predict.best_iterations)
print()
print(f"K1 (5 fold, sesudah rearrangement)          : {k1_5fold:.4f}")
print(f"K1 (fold 1/2/4 bersih, sesudah rearrangement): {k1_clean:.4f}"
      "  <- bandingkan ke XGBoost asli 2,9433 dan RF 2,8508")
print(f"crossing_rate sesudah rearrangement          : {crossing_after:.4f}"
      "  <- harus 0 (sort menjamin monoton)")
print()

coverage = walk_forward.coverage_by_quantile(
    results, "xgboost_rearranged", folds=(1, 2, 4)
)
print("K2 (fold 1/2/4 bersih) sesudah rearrangement:")
print(coverage.to_string(index=False))

# --- pengecekan reproduksibilitas: K1/crossing_rate SEBELUM rearrangement,
# dari prediksi mentah yang sama, tanpa fit ulang ---
fold_ids = list(walk_forward.FOLDS)
raw_scored = []
for position, (actual, raw) in captured.items():
    fold_id = fold_ids[position - 1]
    scored = evaluation.score_quantiles(actual, raw, QUANTILES)
    scored["fold_id"] = fold_id
    raw_scored.append(scored)
raw_scored = pd.concat(raw_scored, ignore_index=True)


def _pooled_pinball(scored: pd.DataFrame, folds=None) -> float:
    rows = scored
    if folds is not None:
        rows = rows[rows["fold_id"].isin(folds)]
    total = rows["n"].sum()
    return float((rows["pinball"] * rows["n"]).sum() / total)


raw_k1_clean = _pooled_pinball(raw_scored, folds=(1, 2, 4))
print()
print(f"[cek reproduksibilitas] K1 SEBELUM rearrangement, fold 1/2/4: "
      f"{raw_k1_clean:.4f}  <- bandingkan ke angka resmi 2,9433")

raw_crossing_by_fold = {
    fold_ids[position - 1]: evaluation.crossing_rate(raw, QUANTILES)
    for position, (_, raw) in captured.items()
}
print(f"[cek reproduksibilitas] crossing_rate SEBELUM rearrangement per fold: "
      f"{raw_crossing_by_fold}  <- bandingkan ke angka resmi 0,9767")
