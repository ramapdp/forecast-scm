#!/bin/bash
# Rantai Fase 3 XGBoost: Tahap A (benchmark) -> B (pencarian) -> C (walk-forward,
# fit final, hasil). Ketiganya menulis outputnya ke notebook/modeling_xgb.ipynb.
#
# Dipecah menjadi tiga proses, bukan satu `nbconvert --execute`, karena B
# berjalan berhari-hari sementara C tidak punya checkpoint sama sekali: satu
# proses berarti satu kill di jam ke-120 menghapus semuanya. run_cells.py
# menyimpan notebook setelah tiap sel, jadi hasil tiap tahap sudah ada di
# berkas .ipynb sebelum tahap berikutnya mulai.
#
#   ./run_xgb_all.sh [PID_TAHAP_A]
#
# PID opsional: kalau Tahap A sudah berjalan, rantai menunggunya selesai dulu.
# Tanpa PID, Tahap A dijalankan dari awal oleh skrip ini.
set -u

cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python3
NB=notebook/modeling_xgb.ipynb
OUT=dataset/model_ready
CHAIN="$OUT/xgb_run_chain.log"
N_CANDIDATES=30

log() { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN"; }

selesai_kandidat() {
  $PY -c "import pandas as pd; print(len(pd.read_csv('$OUT/xgb_search_results.csv')))" 2>/dev/null || echo 0
}

log "=== rantai XGBoost dimulai (pid $$) ==="

# ---------------------------------------------------------------- Tahap A
A_PID="${1:-}"
if [ -n "$A_PID" ]; then
  log "Tahap A sudah berjalan (PID $A_PID) — menunggu"
  while kill -0 "$A_PID" 2>/dev/null; do sleep 30; done
else
  log "Tahap A: benchmark (sel 2-12)"
  $PY run_cells.py "$NB" 2-12 >> "$OUT/xgb_run_A_benchmark.log" 2>&1
fi

if ! tail -5 "$OUT/xgb_run_A_benchmark.log" | grep -q "=== SELESAI"; then
  log "BERHENTI: Tahap A tidak berakhir dengan SELESAI — lihat xgb_run_A_benchmark.log"
  exit 1
fi
log "Tahap A selesai"

# ---------------------------------------------------------------- Tahap B
# Diulang hanya selama checkpoint bertambah. resume=True membuat pengulangan
# melanjutkan, bukan mengulang dari nol — tapi percobaan yang mati TANPA
# menyelesaikan satu kandidat pun adalah kegagalan yang berulang, dan
# mengulanginya hanya menunda pesan errornya berjam-jam.
B_OK=0
for percobaan in 1 2 3 4 5; do
  sebelum=$(selesai_kandidat)
  log "Tahap B percobaan $percobaan — $sebelum/$N_CANDIDATES kandidat sudah di checkpoint"
  if $PY run_cells.py "$NB" 2-10,14 >> "$OUT/xgb_run_B_search.log" 2>&1; then
    B_OK=1
    log "Tahap B selesai"
    break
  fi
  sesudah=$(selesai_kandidat)
  log "Tahap B berhenti tak wajar ($sebelum -> $sesudah kandidat)"
  if [ "$sesudah" -le "$sebelum" ]; then
    log "BERHENTI: tidak ada kemajuan pada percobaan ini — lihat xgb_run_B_search.log"
    exit 1
  fi
done
[ "$B_OK" -eq 1 ] || { log "BERHENTI: Tahap B gagal setelah 5 percobaan"; exit 1; }

# ------------------------------------------------- gerbang sebelum Tahap C
# C memilih pemenang. Memilih dari checkpoint yang berlubang menghasilkan
# angka yang tampak sepenuhnya wajar, jadi kelengkapannya diperiksa di sini
# dan bukan dipercaya.
$PY - "$OUT/xgb_search_results.csv" "$N_CANDIDATES" <<'PYCHECK' || exit 1
import sys
import pandas as pd

path, expected = sys.argv[1], int(sys.argv[2])
frame = pd.read_csv(path)
ids = sorted(frame["candidate_id"].astype(int))
hilang = sorted(set(range(expected)) - set(ids))
ganda = sorted({i for i in ids if ids.count(i) > 1})
gagal = frame[frame["pinball"].isna()]

print(f"checkpoint: {len(ids)}/{expected} kandidat, "
      f"device={sorted(frame['device'].unique())}")
if gagal.empty:
    print("kandidat gagal: tidak ada")
else:
    print(f"kandidat gagal (pinball NaN, tercatat & dilewati): "
          f"{sorted(gagal['candidate_id'].astype(int))}")
    for _, row in gagal.iterrows():
        print(f"  id {int(row['candidate_id'])}: {row['error']}")

if hilang or ganda:
    print(f"BERHENTI: checkpoint tidak lengkap — hilang {hilang}, ganda {ganda}")
    sys.exit(1)
if len(gagal) == expected:
    print("BERHENTI: seluruh kandidat gagal — tidak ada pemenang untuk dipilih")
    sys.exit(1)
PYCHECK
log "gerbang kelengkapan checkpoint lolos"

# ---------------------------------------------------------------- Tahap C
# Sel 14 ikut dijalankan lagi: `candidates` dan `search_results` harus ada di
# memori untuk select_best() di sel 16, dan dengan checkpoint penuh ia hanya
# membaca CSV. Tidak diulang kalau gagal — C tidak punya checkpoint, jadi
# percobaan kedua mengulang seluruhnya dan itu keputusan manusia.
log "Tahap C: walk-forward, fit final, hasil (sel 2-10,14,16-24)"
if $PY run_cells.py "$NB" 2-10,14,16-24 >> "$OUT/xgb_run_C_final.log" 2>&1; then
  log "Tahap C selesai — SELURUH RANTAI TUNTAS"
else
  log "BERHENTI: Tahap C gagal — lihat xgb_run_C_final.log"
  exit 1
fi
