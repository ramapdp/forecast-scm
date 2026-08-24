"""What every model in the comparison needs, and no model owns.

The search protocol, its checkpoint, the one-hot expansion and the bundle
format are not Random Forest ideas — they are how this project runs a search
and ships a model. Leaving them inside model_random_forest.py would mean
fixing the next checkpoint bug twice, then three times when the LSTM lands,
and would point every future model's import at a sibling model.

Nothing here knows what a model is. `run_search` takes the factory; the
screen that rejects an unaffordable candidate is injected too, because the
Random Forest's leaf-storage bound has no XGBoost analogue.
"""

import json
import random
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from . import evaluation, modeling_prep, purging, walk_forward

# The encoded categoricals, the only columns one-hot expansion touches.
IDX_COLS = [col for col in modeling_prep.FEATURE_COLS if col.endswith("_idx")]


# ── INPUT VALIDATION ──────────────────────────────────────────────────────────────

def assert_no_nan(frame: pd.DataFrame, feature_cols: list) -> None:
    """Gagalkan eksekusi jika ada nilai NaN pada kolom fitur yang akan masuk ke model.

    Sengaja tidak melakukan imputasi di sini. build_model_input() sudah
    menjalankan impute_features(). Melakukannya lagi akan merusak fitur
    seperti was_relocated yang memang dibiarkan NaN untuk menandakan
    sesuatu.
    """
    counts = frame[feature_cols].isna().sum()
    offenders = counts[counts > 0]
    if len(offenders):
        raise ValueError(f"NaN pada fitur: {offenders.to_dict()}")


ES_TAIL_DAYS = 30


# ── EARLY STOPPING SPLIT ──────────────────────────────────────────────────────────

def split_early_stopping(
    train: pd.DataFrame,
    tail_days: int = ES_TAIL_DAYS,
    date_col: str = modeling_prep.DATE_COL,
) -> tuple:
    """Pecah row training dalam satu fold menjadi himpunan fit dan tail early-stopping.

    Tail adalah `tail_days` hari kalender terakhir. Purge pada himpunan fit
    penting karena `target_lead_time_cumulative` menggunakan data dari masa depan
    (H+1..H+lead_time_days). Jika tidak ada purge, model akan berlatih dari sinyal
    yang bocor ke periode early-stopping.
    """
    if train.empty:
        raise ValueError("frame training kosong")

    es_start = train[date_col].max() - pd.Timedelta(days=tail_days - 1)
    es_rows = train[train[date_col] >= es_start]
    fit_rows = train[train[date_col] < es_start]
    fit_rows = fit_rows[purging.lookahead_safe_mask(fit_rows, es_start,
                                                    date_col=date_col)]

    if fit_rows.empty:
        raise ValueError(
            f"jendela training terlalu pendek untuk tail {tail_days} hari: "
            f"tidak ada baris tersisa untuk fit"
        )
    return fit_rows, es_rows


# ── TARGET ACCESS ─────────────────────────────────────────────────────────────────

def train_target(
    frame: pd.DataFrame,
    log_target: bool = False,
    column: str = modeling_prep.TRAIN_TARGET_COL,
) -> np.ndarray:
    """Label tempat setiap model belajar: target yang sudah di-cap, dengan opsi log.

    Semua model mengambil target melalui fungsi ini untuk memastikan konsistensi
    (misal: memastikan semua benar-benar menggunakan target yang di-cap).
    Skoring tidak lewat sini, melainkan membaca `EVAL_TARGET_COL` secara langsung.
    """
    if column not in frame.columns:
        raise KeyError(
            f"kolom target latih {column!r} tidak ada di frame — model dilatih "
            f"pada target capped sejak 2026-08-24"
        )
    values = frame[column].to_numpy(dtype=float)
    return np.log1p(values) if log_target else values


def target_provenance() -> dict:
    """Dua nama kolom target yang perlu disimpan bersama model bundle.

    Bundle yang tidak menyebutkan target mana yang melatihnya tidak bisa di-load
    tanpa menebak-nebak, yang berisiko pada evaluasi yang salah.
    """
    return {
        "train_target": modeling_prep.TRAIN_TARGET_COL,
        "eval_target": modeling_prep.EVAL_TARGET_COL,
    }


# ── ONE-HOT EXPANSION ─────────────────────────────────────────────────────────────

def expand_one_hot(
    train_X: pd.DataFrame,
    valid_X: pd.DataFrame,
    idx_cols: Optional[list] = None,
) -> tuple:
    """Ekspansi one-hot untuk fitur kategorikal. Kolom validasi disesuaikan dengan train.

    Hal ini penting karena kategori yang hanya muncul di validasi akan mengubah
    jumlah dan urutan kolom, membuat model salah membaca fitur secara diam-diam.
    """
    idx_cols = IDX_COLS if idx_cols is None else idx_cols
    present = [col for col in idx_cols if col in train_X.columns]
    train_out = pd.get_dummies(train_X, columns=present)
    valid_out = pd.get_dummies(valid_X, columns=present)
    valid_out = valid_out.reindex(columns=train_out.columns, fill_value=0)
    return train_out, valid_out


# ── HYPERPARAMETER SEARCH ─────────────────────────────────────────────────────────

def sample_search_space(
    space: dict,
    defaults: dict,
    n_candidates: int,
    seed: int = 42,
    screen: Optional[Callable[[dict], bool]] = None,
    screen_label: str = "screen",
) -> list:
    """Pilih parameter set secara acak dari search space, difilter oleh screen opsional.

    Pemilihan acak lebih efisien menemukan kombinasi optimal di ruang dimensi tinggi
    daripada grid search. Parameter `screen` dipakai untuk menolak kandidat yang
    misalnya melebih batas ukuran model memori (seperti di Random Forest).
    """
    rng = random.Random(seed)
    keys = sorted(space)
    seen, candidates = set(), []

    for _ in range(n_candidates * 200):
        if len(candidates) == n_candidates:
            break
        drawn = {key: rng.choice(space[key]) for key in keys}
        signature = tuple(drawn[key] for key in keys)
        if signature in seen:
            continue
        candidate = {**defaults, **drawn}
        if screen is not None and not screen(candidate):
            continue
        seen.add(signature)
        candidates.append(candidate)

    if len(candidates) < n_candidates:
        raise ValueError(
            f"hanya {len(candidates)} dari {n_candidates} kandidat lolos "
            f"{screen_label}"
        )
    return candidates


# `pinball` is K1 — the mean across the whole quantile grid — because that is
# what select_best() ranks on. The other three are read at one point of the
# grid and say which point in `headline_quantile`, since "mae" on its own stops
# meaning anything once there are nineteen of them.
SEARCH_METRICS = ("pinball", "mae_headline", "coverage_headline",
                  "fill_rate_headline", "coverage_gap", "crossing_rate")


def headline_quantile(quantiles: tuple) -> float:
    """The grid point the per-candidate summary columns are read at.

    0.9 when the grid contains it, which is every Tahap A run: it is the
    service level the business ships at (B-9), so it is the number a reader
    checks first. Tahap B grids come from critical ratios and need not contain
    0.9 at all, so the nearest point stands in rather than a KeyError three
    hours into a search.
    """
    return min(quantiles, key=lambda tau: abs(tau - evaluation.DEFAULT_ALPHA))


def summarise_candidate(results: pd.DataFrame, model_name: str, folds: tuple,
                        quantiles: tuple) -> dict:
    """The one row per candidate that lands in the search CSV.

    Public because the LSTM's seed-repeat protocol writes rows that have to be
    comparable with the search CSV column for column — the spec's own check is
    that the seed-42 repeat reproduces the winner's row exactly, and two
    summaries built by two functions could not support it.

    K1 decides the winner; the rest is what a reader needs to see *why* a
    candidate won or lost — whether it bought its pinball with a calibration
    drift (`coverage_gap`, signed, so a bodily shift is distinguishable from
    scatter) or with a distribution that crosses itself.
    """
    headline = headline_quantile(quantiles)
    calibration = walk_forward.coverage_by_quantile(results, model_name, folds=folds)
    return {
        "pinball": walk_forward.pooled_k1(results, model_name, folds=folds),
        "mae_headline": walk_forward.pooled_metric(
            results, model_name, metric="mae", folds=folds, quantile=headline),
        "coverage_headline": walk_forward.pooled_metric(
            results, model_name, metric="coverage", folds=folds, quantile=headline),
        "fill_rate_headline": walk_forward.pooled_metric(
            results, model_name, metric="fill_rate", folds=folds, quantile=headline),
        "coverage_gap": float(calibration["gap"].mean()),
        "crossing_rate": walk_forward.pooled_metric(
            results, model_name, metric="crossing_rate", folds=folds,
            quantile=headline),
        "headline_quantile": headline,
    }

# Where a model leaves the capacity its own early stopping picked, one value
# per fold. Two names because the two models that have such a thing named it
# after their own unit; both mean "how much model the data asked for", and a
# search that does not record it cannot explain its own wall clock afterwards.
CAPACITY_ATTRS = ("best_epochs", "best_iterations")


def _selected(candidates: list, only: Optional[Iterable[int]]) -> set:
    """Id kandidat yang menjadi tanggung jawab proses ini.

    Alasan keberadaannya adalah pemecahan pekerjaan antar mesin. `run_search`
    menomori kandidat lewat posisinya di `candidates`, jadi memotong daftar itu
    di sisi pemanggil akan menomori ulang dan dua shard tidak akan pernah bisa
    disatukan lewat id. `only` menjaga penomoran tetap absolut terhadap
    `sample_search_space(seed=...)`, yang deterministik di mesin mana pun.

    Seleksi di luar jangkauan atau kosong dilempar, bukan dijalankan sebagai
    no-op: shard kosong selesai dalam sedetik dan menulis CSV kosong, yang dari
    luar tidak dapat dibedakan dari shard yang berhasil. Lubangnya baru muncul
    saat penggabungan, berjam-jam sesudahnya.
    """
    if only is None:
        return set(range(len(candidates)))
    selected = {int(value) for value in only}
    out_of_range = sorted(value for value in selected
                          if value < 0 or value >= len(candidates))
    if out_of_range:
        raise ValueError(
            f"only memuat candidate_id di luar {len(candidates)} kandidat "
            f"saat ini: {out_of_range}"
        )
    if not selected:
        raise ValueError("only kosong — tidak ada kandidat untuk dijalankan")
    return selected


def run_search(
    df: pd.DataFrame,
    candidates: list,
    make_fit_predict: Callable,
    search_space: dict,
    folds: tuple,
    quantiles: tuple,
    model_name: str,
    feature_cols: Optional[list] = None,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
    only: Optional[Iterable[int]] = None,
    provenance: Optional[dict] = None,
    catch: tuple = (MemoryError, ValueError),
) -> pd.DataFrame:
    """Score every candidate on the search folds only.

    A candidate that raises one of `catch` is recorded with NaN metrics rather
    than aborting the run: a long search should not lose fourteen finished
    candidates to the fifteenth. `catch` is narrow on purpose — an unexpected
    exception type is a bug, and a bug must not be laundered into a NaN row.

    `checkpoint_path` extends that reasoning past what Python can catch. A
    search at this scale runs for hours, and an OS-level kill leaves no
    exception to handle — so every finished candidate is flushed to disk
    immediately, and the file doubles as the only progress signal available
    while the run is buried inside a notebook cell.

    `resume` is on by default because that is what the checkpoint is for. The
    stale-checkpoint guard is the price: resuming across a changed search space
    or seed would blend candidates from two different experiments and hand back
    a winner that was never actually evaluated.

    `only` memecah satu pencarian ke beberapa mesin: tiap mesin menjalankan
    subset candidate_id-nya sendiri sementara penomorannya tetap absolut,
    sehingga hasilnya dapat disatukan oleh `merge_shards()`.
    """
    selected = _selected(candidates, only)
    provenance = provenance or {}
    collisions = sorted({"candidate_id", *search_space} & set(provenance))
    if collisions:
        raise ValueError(
            f"kunci provenance bertabrakan dengan kolom pencarian: "
            f"{collisions}"
        )
    frame = walk_forward.eligible_rows(df)
    rows = []
    completed = set()

    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        prior = pd.read_csv(checkpoint_path)
        _assert_checkpoint_matches(prior, candidates, search_space, checkpoint_path)
        rows = prior.to_dict("records")
        completed = {int(value) for value in prior["candidate_id"]}
        if verbose and completed:
            print(f"melanjutkan dari checkpoint: {len(completed)} kandidat sudah selesai",
                  flush=True)

    for candidate_id, candidate in enumerate(candidates):
        if candidate_id not in selected or candidate_id in completed:
            continue
        record = {"candidate_id": candidate_id,
                  **{key: candidate[key] for key in sorted(search_space)},
                  **provenance}
        fit_predict, message = None, None
        started = time.perf_counter()
        try:
            fit_predict = make_fit_predict(candidate, feature_cols=feature_cols,
                                           quantiles=quantiles)
            parts = [
                walk_forward.run_fold(frame, fold_id, fit_predict,
                                      model_name=model_name, quantiles=quantiles,
                                      prepared=True)
                for fold_id in folds
            ]
            results = pd.concat(parts, ignore_index=True)
            record.update(summarise_candidate(results, model_name, folds,
                                              quantiles))
        except catch as failure:
            for metric in SEARCH_METRICS:
                record[metric] = float("nan")
            record["headline_quantile"] = headline_quantile(quantiles)
            message = str(failure)
        # Recorded outside the try so a candidate that died after forty minutes
        # still reports the forty minutes.
        record["best_epoch"] = reported_capacity(fit_predict)
        record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        record["error"] = message
        if verbose:
            print(f"[{candidate_id + 1}/{len(candidates)}] "
                  f"pinball={record['pinball']:.4f} "
                  f"epoch={record['best_epoch'] or '-'} "
                  f"{record['elapsed_seconds']:.0f}s {record['error'] or ''}",
                  flush=True)
        rows.append(record)
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            _ordered(rows).to_csv(checkpoint_path, index=False)
    return _ordered(rows)


def reported_capacity(fit_predict) -> Optional[str]:
    """Kapasitas model yang dipilih oleh early stopping per fold (mis. jumlah epoch).

    Disimpan sebagai string gabungan (contoh: '40,45,42,40,50') karena kandidat
    dinilai pada beberapa fold dan sebaran antar fold adalah metrik penting.
    Menggunakan rata-rata tunggal akan menyembunyikan fold yang berlatih dua kali
    lebih lama. None jika model tidak punya konsep epoch/iterasi (seperti RF) atau
    gagal di fold pertama.
    """
    for name in CAPACITY_ATTRS:
        values = getattr(fit_predict, name, None)
        if values:
            return ",".join(str(int(value)) for value in values)
    return None



# ── CHECKPOINT MANAGEMENT ─────────────────────────────────────────────────────────

def _ordered(rows: list) -> pd.DataFrame:
    """Mengurutkan kandidat supaya select_best() selalu membaca urutan yang benar,
    bahkan jika proses yang dilanjutkan (resume) menyelesaikannya secara acak."""
    return (pd.DataFrame(rows)
            .sort_values("candidate_id")
            .reset_index(drop=True))


def current_commit(default: str = "unknown",
                   cwd: Optional[str] = None) -> str:
    """Hash git pendek dari pohon kerja yang menjalankan proses ini.

    Ditulis ke tiap baris shard karena sebuah baris shard adalah bukti: angka
    yang tidak dapat ditelusuri ke versi kode yang melahirkannya tidak
    reprodusibel, dan mesin cloud yang menjalankan shard ini sifatnya
    sementara — ia tidak akan ada lagi saat angkanya dibaca.

    Mengembalikan `default` alih-alih melempar ketika git tidak tersedia sama
    sekali: kegagalan mencatat asal-usul tidak boleh menggagalkan pencarian
    delapan jam yang selain itu baik-baik saja. Yang hilang tercatat sebagai
    nilai `default` yang kasat mata, bukan sebagai sel kosong.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
            cwd=cwd or str(Path(__file__).resolve().parents[2]),
        )
    except (OSError, subprocess.SubprocessError):
        return default
    return completed.stdout.strip() or default


# A column no single-quantile checkpoint can have. Its absence is how a
# pre-2026-08-24 file is recognised.
CHECKPOINT_SCHEMA_COL = "headline_quantile"


def _assert_checkpoint_matches(
    prior: pd.DataFrame,
    candidates: list,
    search_space: dict,
    path: str,
) -> None:
    """Tolak checkpoint jika baris-barisnya mewakili ruang parameter yang berbeda.

    Pemeriksaan schema (`CHECKPOINT_SCHEMA_COL`) sangat penting karena file
    checkpoint lama (single-quantile) dapat lolos pengecekan parameter dan
    tanpa disadari menggunakan hasil pinball@0.9 sebagai nilai K1 yang baru.
    """
    if CHECKPOINT_SCHEMA_COL not in prior.columns:
        raise ValueError(
            f"checkpoint {path} berasal dari run kuantil tunggal (tidak ada "
            f"kolom {CHECKPOINT_SCHEMA_COL!r}) — angkanya pinball@0,9, bukan "
            f"K1. Hapus berkasnya atau jalankan dengan resume=False"
        )
    for _, row in prior.iterrows():
        candidate_id = int(row["candidate_id"])
        if candidate_id >= len(candidates):
            raise ValueError(
                f"checkpoint {path} memuat candidate_id {candidate_id} "
                f"di luar {len(candidates)} kandidat saat ini"
            )
        for key in sorted(search_space):
            expected = candidates[candidate_id][key]
            actual = None if pd.isna(row[key]) else row[key]
            if isinstance(expected, bool):
                actual = bool(actual)
            elif isinstance(expected, (int, float)) and actual is not None:
                actual = type(expected)(actual)
            if expected != actual:
                raise ValueError(
                    f"checkpoint {path} tidak cocok dengan ruang pencarian: "
                    f"kandidat {candidate_id} punya {key}={actual}, "
                    f"seharusnya {expected}"
                )


def merge_shards(paths: list, candidates: list, search_space: dict) -> pd.DataFrame:
    """Satu frame pencarian dari beberapa CSV shard, diverifikasi bukan dipercaya.

    Empat pemeriksaan, dan tiap satunya membalas satu cara pemecahan pekerjaan
    bisa gagal tanpa suara:

    1. `candidate_id` ganda — dua mesin diberi rentang yang tumpang tindih,
       sehingga satu kandidat dinilai dua kali dan `select_best()` memilih di
       antara baris kembar.
    2. Cakupan berlubang — satu shard tidak pernah selesai, dan yang dilaporkan
       sebagai "pencarian 30 kandidat" sebenarnya 22.
    3. Parameter yang tidak cocok dengan id yang diklaimnya — shard tertukar,
       atau lahir dari `seed` / ruang pencarian yang berbeda.
    4. Skema kuantil tunggal — CSV pra-2026-08-24 yang angkanya pinball@0,9,
       bukan K1.

    Pemeriksaan 3 dan 4 tidak ditulis ulang di sini: keduanya sudah menjadi
    `_assert_checkpoint_matches()`, dan dua definisi "cocok" yang hidup
    berdampingan pasti akan berbeda diam-diam suatu hari.

    Kolom di luar `search_space` — `device`, hash commit, metrik — dibawa apa
    adanya; ia justru yang membuat baris hasil dapat ditelusuri ke mesinnya.
    """
    if not paths:
        raise ValueError("tidak ada shard untuk digabungkan")

    merged = pd.concat([pd.read_csv(path) for path in paths],
                       ignore_index=True)
    ids = [int(value) for value in merged["candidate_id"]]

    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(
            f"candidate_id ganda di gabungan shard: {duplicates} — dua mesin "
            f"diberi rentang yang tumpang tindih"
        )

    missing = sorted(set(range(len(candidates))) - set(ids))
    if missing:
        raise ValueError(
            f"gabungan shard tidak menutup seluruh {len(candidates)} kandidat: "
            f"{missing} hilang — satu shard belum selesai atau tidak ikut "
            f"digabungkan"
        )

    _assert_checkpoint_matches(merged, candidates, search_space,
                               path="gabungan shard")
    return _ordered(merged.to_dict("records"))


def select_best(search_results: pd.DataFrame, candidates: list) -> dict:
    """Pilih kandidat terbaik berdasarkan skor K1 terendah lintas fold.

    K1 (mean pinball lintas quantile grid) adalah kriteria tunggal pemenang.
    Kalibrasi (K2) dan persilangan garis distribusi diukur tapi tidak ikut
    menentukan di tahap pencarian ini agar metodologi pencarian tetap bersih.
    """
    scored = search_results[search_results["pinball"].notna()]
    if scored.empty:
        raise ValueError("tidak ada kandidat yang berhasil dinilai")
    best_id = int(scored.loc[scored["pinball"].idxmin(), "candidate_id"])
    return candidates[best_id]



# ── MODEL BUNDLE I/O ──────────────────────────────────────────────────────────────

def save_bundle(bundle: dict, path: str) -> None:
    """Simpan objek bundel (model + scaler + meta) menggunakan joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str) -> dict:
    """Muat objek bundel dari disk menggunakan joblib."""
    return joblib.load(path)


def save_best_params(params: dict, path: str) -> None:
    """Simpan parameter hiperparameter terbaik ke format JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, sort_keys=True)
