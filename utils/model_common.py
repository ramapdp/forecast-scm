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
import time
from pathlib import Path
from typing import Callable, Optional

import joblib
import pandas as pd

from . import evaluation, modeling_prep, purging, walk_forward

# The encoded categoricals, the only columns one-hot expansion touches.
IDX_COLS = [col for col in modeling_prep.FEATURE_COLS if col.endswith("_idx")]


def assert_no_nan(frame: pd.DataFrame, feature_cols: list) -> None:
    """Fail loudly on a null the estimator cannot consume.

    Deliberately not an imputation step. build_model_input() already ran
    impute_features(), and running it a second time would recompute
    was_relocated from a column that is now filled with 0.0, setting the
    indicator True on every row and erasing the distinction it exists to make.
    """
    counts = frame[feature_cols].isna().sum()
    offenders = counts[counts > 0]
    if len(offenders):
        raise ValueError(f"NaN pada fitur: {offenders.to_dict()}")


ES_TAIL_DAYS = 30


def split_early_stopping(
    train: pd.DataFrame,
    tail_days: int = ES_TAIL_DAYS,
    date_col: str = modeling_prep.DATE_COL,
) -> tuple:
    """Split a fold's training rows into fit rows and an early-stopping tail.

    The tail is the last `tail_days` calendar days. The purge on the fit side
    is not extra caution: `target_lead_time_cumulative` sums over
    H+1..H+lead_time_days, so a fit row dated within `lead_time_days` of the
    tail carries a label built partly out of the early-stopping window. Without
    the purge, early stopping would be reading a signal it had partly trained
    on and would stop too late — the identical leak `fold_train_mask()`
    prevents at the fold boundary, one scale down.

    Lives here rather than in one model's module because two models need it:
    XGBoost chooses a boosting-round count this way, the LSTM chooses an epoch
    count. Neither mechanism is specific to its model family.
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


def expand_one_hot(
    train_X: pd.DataFrame,
    valid_X: pd.DataFrame,
    idx_cols: Optional[list] = None,
) -> tuple:
    """One-hot the encoded categoricals, with validation reindexed onto the
    training columns.

    The reindex is the point. A category that appears only in validation would
    otherwise add a column there and shift every column after it, so the model
    would read the wrong feature at every position — silently, since the shapes
    still line up.
    """
    idx_cols = IDX_COLS if idx_cols is None else idx_cols
    present = [col for col in idx_cols if col in train_X.columns]
    train_out = pd.get_dummies(train_X, columns=present)
    valid_out = pd.get_dummies(valid_X, columns=present)
    valid_out = valid_out.reindex(columns=train_out.columns, fill_value=0)
    return train_out, valid_out


def sample_search_space(
    space: dict,
    defaults: dict,
    n_candidates: int,
    seed: int = 42,
    screen: Optional[Callable[[dict], bool]] = None,
    screen_label: str = "screen",
) -> list:
    """Distinct parameter sets drawn at random, optionally screened.

    Random rather than grid: only a few dimensions of these spaces carry real
    signal, so random draws cover each dimension's range better than a
    truncated grid at the same cost.

    `screen` returns True for a candidate worth fitting. The Random Forest
    injects its leaf-storage bound here; XGBoost has no equivalent and passes
    None.
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
    """
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
        if candidate_id in completed:
            continue
        record = {"candidate_id": candidate_id,
                  **{key: candidate[key] for key in sorted(search_space)}}
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
    """The per-fold capacity a model chose, joined for one CSV cell.

    A string rather than a number because a candidate is scored on several
    folds and their spread is the interesting part — one number would have to
    be a mean, and a mean of two epochs hides the fold that ran twice as long.
    None for a model with no such notion, which is every Random Forest and any
    candidate that failed before finishing its first fold.
    """
    for name in CAPACITY_ATTRS:
        values = getattr(fit_predict, name, None)
        if values:
            return ",".join(str(int(value)) for value in values)
    return None


def _ordered(rows: list) -> pd.DataFrame:
    """Candidate order, so select_best() can index `candidates` by position
    regardless of the order a resumed run happened to finish them in."""
    return (pd.DataFrame(rows)
            .sort_values("candidate_id")
            .reset_index(drop=True))


# A column no single-quantile checkpoint can have. Its absence is how a
# pre-2026-08-24 file is recognised.
CHECKPOINT_SCHEMA_COL = "headline_quantile"


def _assert_checkpoint_matches(
    prior: pd.DataFrame,
    candidates: list,
    search_space: dict,
    path: str,
) -> None:
    """Refuse a checkpoint whose rows describe different candidates.

    Compares the searched parameters rather than trusting the file name. NaN
    stands in for None in the CSV, and booleans survive the round trip as
    numpy bools, so both are normalised before comparison.

    The schema check is the multi-quantile migration's half of this guard, and
    it matters more than it looks. The three search CSVs on disk were written
    by the single-quantile runs against the *same* search space and the *same*
    seed, so the parameter comparison below accepts them happily — and a
    resumed run would skip every candidate, hand back the old run's pinball@0.9
    numbers, and let them be written up as K1. Nothing about that would look
    wrong in the output.
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


def select_best(search_results: pd.DataFrame, candidates: list) -> dict:
    """The candidate with the lowest K1 across the search folds.

    K1 alone decides it — the mean pinball over the whole quantile grid, in
    the `pinball` column. The service level is uniform across every SKU by the
    data owner's decision, so the selection criterion has to be uniform too —
    picking on a per-segment metric would optimize for a split the business
    does not make.

    Calibration and crossing are recorded beside it but do not vote here. K2
    is a separate rung of the ladder applied to the finalists, and folding it
    into the search would collapse two criteria the methodology keeps apart.
    """
    scored = search_results[search_results["pinball"].notna()]
    if scored.empty:
        raise ValueError("tidak ada kandidat yang berhasil dinilai")
    best_id = int(scored.loc[scored["pinball"].idxmin(), "candidate_id"])
    return candidates[best_id]


def save_bundle(bundle: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str) -> dict:
    return joblib.load(path)


def save_best_params(params: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, sort_keys=True)
