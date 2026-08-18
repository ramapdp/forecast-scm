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

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor

from . import model_common, modeling_prep, purging, walk_forward

QUANTILE = 0.9

# Leaf storage above this is refused before the fit starts. Discovering the
# limit through the OOM killer twenty minutes into a fit is the alternative.
MEMORY_BUDGET_BYTES = 3 * 1024 ** 3

# Re-exported so this module's callers — its test suite and modeling_rf.ipynb —
# keep working unchanged after the extraction. The definitions live in
# model_common.py because XGBoost and the LSTM need them too.
IDX_COLS = model_common.IDX_COLS
assert_no_nan = model_common.assert_no_nan
expand_one_hot = model_common.expand_one_hot
select_best = model_common.select_best
load_bundle = model_common.load_bundle

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

    The budget screen is what makes this wrapper worth keeping: quantile-forest
    sizes its leaf-value array from the hyperparameters before a single tree is
    grown, so an unaffordable candidate can be refused without loading data.
    """
    def screen(candidate: dict) -> bool:
        return estimate_leaf_memory_bytes(candidate, n_train) <= memory_budget

    return model_common.sample_search_space(
        space=SEARCH_SPACE if space is None else space,
        defaults=DEFAULT_PARAMS,
        n_candidates=n_candidates,
        seed=seed,
        screen=screen,
        screen_label=f"budget {memory_budget / 1024 ** 3:.1f} GB",
    )


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
    """Score every Random Forest candidate on the search folds."""
    return model_common.run_search(
        df, candidates, make_fit_predict=make_fit_predict,
        search_space=SEARCH_SPACE, folds=folds, alpha=alpha,
        model_name=model_name, feature_cols=feature_cols, verbose=verbose,
        checkpoint_path=checkpoint_path, resume=resume,
    )


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
    model_common.save_bundle(bundle, path)


def save_best_params(params: dict, path: str = BEST_PARAMS_FILE) -> None:
    model_common.save_best_params(params, path)
