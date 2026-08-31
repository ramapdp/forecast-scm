"""Verifikasi jarak K1 RF<->LSTM dengan seed tambahan (bukan seed 42).
Jalankan dari repo root: .venv/bin/python3 lstm_seed_walkforward.py 43
~8,5 jam di CPU Mac, TANPA checkpoint -- jangan diinterupsi.
"""
import json
import sys
import time

import pandas as pd

from utils.modelling import evaluation, model_lstm, walk_forward

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 43
assert SEED != 42, "seed 42 sudah ada di lstm_walk_forward_results.csv"

BEST_PARAMS_FILE = "dataset/model_ready/lstm_best_params.json"
QUANTILES = evaluation.QUANTILE_SET_A

df = pd.read_parquet("dataset/model_ready/model_input.parquet")

with open(BEST_PARAMS_FILE) as f:
    best = json.load(f)
best["random_state"] = SEED

make = model_lstm.bind_panel(df, device_name="cpu")  # cpu wajib -- sebanding K3
fit_predict = make(best, quantiles=QUANTILES)

t0 = time.time()
results = walk_forward.run_walk_forward(
    df, fit_predict, model_name="lstm", quantiles=QUANTILES
)
elapsed = time.time() - t0

out_path = f"dataset/model_ready/lstm_walk_forward_results_seed{SEED}.csv"
results.to_csv(out_path, index=False)

k1_5fold = walk_forward.pooled_k1(results, "lstm")
k1_clean = walk_forward.pooled_k1(results, "lstm", folds=(1, 2, 4))

print(f"selesai dalam {elapsed/3600:.2f} jam, disimpan ke {out_path}")
print("best_epoch per fold:", fit_predict.best_epochs)
print(f"K1 (5 fold)          : {k1_5fold:.4f}")
print(f"K1 (fold 1/2/4 bersih): {k1_clean:.4f}  <- bandingkan ke RF 2.8508")
