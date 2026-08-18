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
from pathlib import Path
from typing import Callable, Optional

import joblib
import pandas as pd

from . import modeling_prep, walk_forward

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


SEARCH_METRICS = ("pinball", "mae", "coverage", "fill_rate")


def run_search(
    df: pd.DataFrame,
    candidates: list,
    make_fit_predict: Callable,
    search_space: dict,
    folds: tuple,
    alpha: float,
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
        try:
            fit_predict = make_fit_predict(candidate, feature_cols=feature_cols,
                                           quantile=alpha)
            parts = [
                walk_forward.run_fold(frame, fold_id, fit_predict,
                                      model_name=model_name, alpha=alpha,
                                      prepared=True)
                for fold_id in folds
            ]
            results = pd.concat(parts, ignore_index=True)
            for metric in SEARCH_METRICS:
                record[metric] = walk_forward.pooled_metric(
                    results, model_name, metric=metric, folds=folds
                )
            record["error"] = None
        except catch as failure:
            for metric in SEARCH_METRICS:
                record[metric] = float("nan")
            record["error"] = str(failure)
        if verbose:
            print(f"[{candidate_id + 1}/{len(candidates)}] "
                  f"pinball={record['pinball']:.4f} {record['error'] or ''}",
                  flush=True)
        rows.append(record)
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            _ordered(rows).to_csv(checkpoint_path, index=False)
    return _ordered(rows)


def _ordered(rows: list) -> pd.DataFrame:
    """Candidate order, so select_best() can index `candidates` by position
    regardless of the order a resumed run happened to finish them in."""
    return (pd.DataFrame(rows)
            .sort_values("candidate_id")
            .reset_index(drop=True))


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
    """
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
    """The candidate with the lowest pooled pinball across the search folds.

    Pinball alone decides it. The service level is uniform across every SKU by
    the data owner's decision, so the selection criterion has to be uniform
    too — picking on a per-segment metric would optimize for a split the
    business does not make.
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
