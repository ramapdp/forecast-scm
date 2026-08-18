"""Random Forest at the 0.9 service level.

sklearn's own forest cannot do this. It minimizes squared or absolute error,
both of which target the middle of the distribution, and a forecast that is
right in the middle stocks out roughly half the time by construction. A
quantile regression forest keeps the training targets that reach each leaf and
reads the 0.9 quantile off that empirical distribution instead.

That storage is what makes this model's cost unusual. quantile-forest holds it
in a dense int64 array of shape
(n_estimators, max_node_count, n_outputs, max_samples_leaf), so the memory bill
is fixed by the hyperparameters before a single tree is grown — see
estimate_leaf_memory_bytes(). Deep trees with tiny leaves are unaffordable
here, which happens to agree with the statistics: a leaf holding one sample
cannot estimate a 0.9 quantile at all.
"""

import json
import random
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor

from . import modeling_prep, purging, walk_forward

QUANTILE = 0.9

# Leaf storage above this is refused before the fit starts. Discovering the
# limit through the OOM killer twenty minutes into a fit is the alternative.
MEMORY_BUDGET_BYTES = 3 * 1024 ** 3

# The encoded categoricals, the only columns one-hot expansion touches.
IDX_COLS = [col for col in modeling_prep.FEATURE_COLS if col.endswith("_idx")]

DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 16,
    "min_samples_leaf": 50,
    "max_samples_leaf": 20,
    "max_features": "sqrt",
    "max_samples": None,
    "log_target": False,
    "one_hot": False,
    "random_state": 42,
}

# n_estimators is absent on purpose: forest quality is monotone in tree count,
# so searching it spends budget on a question with a known answer. It is pinned
# during the search and raised for the final fit.
#
# max_depth=None and min_samples_leaf below 20 are absent for the reason in the
# module docstring.
SEARCH_SPACE = {
    "max_depth": [12, 16, 20],
    "min_samples_leaf": [20, 50, 100, 200],
    "max_samples_leaf": [1, 20, 50],
    "max_features": ["sqrt", 0.3, 0.5, 1.0],
    "max_samples": [None, 0.5],
    "log_target": [False, True],
    "one_hot": [False, True],
}

ESTIMATOR_KEYS = ("n_estimators", "max_depth", "min_samples_leaf",
                  "max_samples_leaf", "max_features", "max_samples",
                  "random_state")


def assert_no_nan(frame: pd.DataFrame, feature_cols: list) -> None:
    """Fail loudly on a null the forest cannot consume.

    Deliberately not an imputation step. build_model_input() already ran
    impute_features(), and running it a second time would recompute
    was_relocated from a column that is now filled with 0.0, setting the
    indicator True on every row and erasing the distinction it exists to make.
    """
    counts = frame[feature_cols].isna().sum()
    offenders = counts[counts > 0]
    if len(offenders):
        raise ValueError(f"NaN pada fitur: {offenders.to_dict()}")


def estimate_leaf_memory_bytes(params: dict, n_train: int) -> int:
    """Upper bound on quantile-forest's leaf-value array, in bytes.

    bytes = n_estimators x node_count x max_samples_leaf x 8

    node_count is bounded twice: by depth, since a tree of depth d holds at
    most 2^(d+1) nodes, and by leaf size, since n rows split into leaves of at
    least L rows give at most 2n/L nodes including internal ones. The tighter
    bound wins.
    """
    fraction = params.get("max_samples") or 1.0
    n_bootstrap = n_train * fraction
    depth_bound = 2.0 ** (params["max_depth"] + 1)
    leaf_bound = 2.0 * n_bootstrap / params["min_samples_leaf"]
    node_count = min(depth_bound, leaf_bound)
    return int(params["n_estimators"] * node_count * params["max_samples_leaf"] * 8)


def expand_one_hot(
    train_X: pd.DataFrame,
    valid_X: pd.DataFrame,
    idx_cols: Optional[list] = None,
) -> tuple:
    """One-hot the encoded categoricals, with validation reindexed onto the
    training columns.

    The reindex is the point. A category that appears only in validation would
    otherwise add a column there and shift every column after it, so the forest
    would read the wrong feature at every position — silently, since the shapes
    still line up.
    """
    idx_cols = IDX_COLS if idx_cols is None else idx_cols
    present = [col for col in idx_cols if col in train_X.columns]
    train_out = pd.get_dummies(train_X, columns=present)
    valid_out = pd.get_dummies(valid_X, columns=present)
    valid_out = valid_out.reindex(columns=train_out.columns, fill_value=0)
    return train_out, valid_out


def build_estimator(params: dict) -> RandomForestQuantileRegressor:
    kwargs = {key: params[key] for key in ESTIMATOR_KEYS if key in params}
    return RandomForestQuantileRegressor(n_jobs=-1, **kwargs)


def make_fit_predict(
    params: Optional[dict] = None,
    feature_cols: Optional[list] = None,
    quantile: float = QUANTILE,
    memory_budget: int = MEMORY_BUDGET_BYTES,
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """The callable walk_forward.run_fold() injects.

    Feature selection, one-hot expansion and the target transform all live in
    here rather than in the runner, because they are model choices — the very
    things the three-way comparison is supposed to expose.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    feature_cols = feature_cols or modeling_prep.FEATURE_COLS

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
        assert_no_nan(train, feature_cols)
        assert_no_nan(valid, feature_cols)

        needed = estimate_leaf_memory_bytes(params, len(train))
        if needed > memory_budget:
            raise MemoryError(
                f"leaf storage {needed / 1024 ** 3:.1f} GB melebihi budget "
                f"{memory_budget / 1024 ** 3:.1f} GB untuk {params}"
            )

        train_X, valid_X = train[feature_cols], valid[feature_cols]
        if params["one_hot"]:
            train_X, valid_X = expand_one_hot(train_X, valid_X)

        y_train = train[modeling_prep.TARGET_COL].to_numpy(dtype=float)
        if params["log_target"]:
            y_train = np.log1p(y_train)

        model = build_estimator(params)
        model.fit(train_X.to_numpy(dtype=np.float32), y_train)
        prediction = model.predict(valid_X.to_numpy(dtype=np.float32), quantiles=quantile)
        if params["log_target"]:
            prediction = modeling_prep.inverse_log_target(prediction)
        # A negative shipment quantity is not a thing.
        return np.clip(np.asarray(prediction, dtype=float), 0.0, None)

    return fit_predict


SEARCH_FOLDS = (3, 5)

# Enough training rows to size the memory screen realistically before the data
# is loaded — fold 5 trains on roughly this many rows.
TYPICAL_N_TRAIN = 1_280_000


def sample_search_space(
    n_candidates: int = 18,
    n_train: int = TYPICAL_N_TRAIN,
    seed: int = 42,
    memory_budget: int = MEMORY_BUDGET_BYTES,
    space: Optional[dict] = None,
) -> list:
    """Distinct, within-budget parameter sets drawn at random.

    Random rather than grid: the space is 1,152 combinations and only a few of
    its dimensions carry real signal, so random draws cover each dimension's
    range better than a truncated grid at the same cost.
    """
    space = SEARCH_SPACE if space is None else space
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
        candidate = {**DEFAULT_PARAMS, **drawn}
        if estimate_leaf_memory_bytes(candidate, n_train) > memory_budget:
            continue
        seen.add(signature)
        candidates.append(candidate)

    if len(candidates) < n_candidates:
        raise ValueError(
            f"hanya {len(candidates)} dari {n_candidates} kandidat muat dalam "
            f"budget {memory_budget / 1024 ** 3:.1f} GB"
        )
    return candidates


def run_search(
    df: pd.DataFrame,
    candidates: list,
    folds: tuple = SEARCH_FOLDS,
    alpha: float = QUANTILE,
    model_name: str = "random_forest",
    feature_cols: Optional[list] = None,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every candidate on the search folds only.

    A candidate that raises is recorded with NaN metrics rather than aborting
    the run: eighteen fits is a long afternoon, and losing all of it to the
    seventeenth configuration would be a poor trade.

    `checkpoint_path` extends that reasoning past what Python can catch. A
    search at this scale runs for hours, and an OS-level kill leaves no
    exception to handle — so every finished candidate is flushed to disk
    immediately, and the file doubles as the only progress signal available
    while the run is buried inside a notebook cell.

    `resume` is on by default because that is what the checkpoint is for. A
    restart that recomputes finished candidates converts the checkpoint into a
    progress bar, which is not worth hours of CPU. The stale-checkpoint guard
    below is the price: resuming across a changed search space or seed would
    blend candidates from two different experiments and hand back a winner
    that was never actually evaluated.
    """
    frame = walk_forward.eligible_rows(df)
    rows = []
    completed = set()

    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        prior = pd.read_csv(checkpoint_path)
        _assert_checkpoint_matches(prior, candidates, checkpoint_path)
        rows = prior.to_dict("records")
        completed = {int(value) for value in prior["candidate_id"]}
        if verbose and completed:
            print(f"melanjutkan dari checkpoint: {len(completed)} kandidat sudah selesai",
                  flush=True)

    for candidate_id, candidate in enumerate(candidates):
        if candidate_id in completed:
            continue
        record = {"candidate_id": candidate_id,
                  **{key: candidate[key] for key in sorted(SEARCH_SPACE)}}
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
            for metric in ("pinball", "mae", "coverage", "fill_rate"):
                record[metric] = walk_forward.pooled_metric(
                    results, model_name, metric=metric, folds=folds
                )
            record["error"] = None
        except (MemoryError, ValueError) as failure:
            for metric in ("pinball", "mae", "coverage", "fill_rate"):
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


def _assert_checkpoint_matches(prior: pd.DataFrame, candidates: list, path: str) -> None:
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
        for key in sorted(SEARCH_SPACE):
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


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = str(BASE_DIR / "models/random_forest_q90.joblib")
BEST_PARAMS_FILE = str(BASE_DIR / "dataset/model_ready/rf_best_params.json")
SEARCH_FILE = str(BASE_DIR / "dataset/model_ready/rf_search_results.csv")
RESULTS_FILE = str(BASE_DIR / "dataset/model_ready/rf_walk_forward_results.csv")

# Raised from the searched 200. Forest quality is monotone in tree count, so
# the final model buys the variance reduction the search could not afford.
FINAL_N_ESTIMATORS = 400


def fit_final(
    df: pd.DataFrame,
    params: dict,
    feature_cols: Optional[list] = None,
    n_estimators: int = FINAL_N_ESTIMATORS,
    date_col: str = modeling_prep.DATE_COL,
    test_start: pd.Timestamp = modeling_prep.TEST_START,
) -> dict:
    """Fit on every eligible row before December, purged at that boundary.

    Eligibility comes from `walk_forward.eligible_rows`, not from a date filter
    written here. The rows this model is finally trained on have to be the rows
    it was scored on, and the scoring cuts are not just the date: the first 28
    days of each segment have no usable lag window, and the last few days have
    no target at all, because the lead-time sum runs past the end of the data.
    Selecting rows independently here silently trained the shipped model on a
    different population than the reported metrics describe — and, since the
    target cut was missing, on labels that were NaN.

    The bundle records the exact training column order alongside the model. A
    forest reloaded next week against columns in a different order does not
    fail — it predicts confidently from the wrong features, which is worse.
    """
    params = {**DEFAULT_PARAMS, **params, "n_estimators": n_estimators}
    feature_cols = feature_cols or modeling_prep.FEATURE_COLS

    frame = walk_forward.eligible_rows(df, date_col=date_col, test_start=test_start)
    frame = frame[purging.lookahead_safe_mask(frame, test_start, date_col=date_col)]
    assert_no_nan(frame, feature_cols)

    train_X = frame[feature_cols]
    if params["one_hot"]:
        train_X, _ = expand_one_hot(train_X, train_X)

    y_train = frame[modeling_prep.TARGET_COL].to_numpy(dtype=float)
    if params["log_target"]:
        y_train = np.log1p(y_train)

    model = build_estimator(params)
    model.fit(train_X.to_numpy(dtype=np.float32), y_train)
    return {
        "model": model,
        "params": params,
        "feature_cols": feature_cols,
        "columns": list(train_X.columns),
        "quantile": QUANTILE,
        "n_train": int(len(frame)),
    }


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predict with a fitted bundle, forcing the recorded column order."""
    params = bundle["params"]
    features = frame[bundle["feature_cols"]]
    if params["one_hot"]:
        features, _ = expand_one_hot(features, features)
    features = features.reindex(columns=bundle["columns"], fill_value=0)
    prediction = bundle["model"].predict(
        features.to_numpy(dtype=np.float32), quantiles=bundle["quantile"]
    )
    if params["log_target"]:
        prediction = modeling_prep.inverse_log_target(prediction)
    return np.clip(np.asarray(prediction, dtype=float), 0.0, None)


def save_bundle(bundle: dict, path: str = MODEL_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str = MODEL_FILE) -> dict:
    return joblib.load(path)


def save_best_params(params: dict, path: str = BEST_PARAMS_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, sort_keys=True)
