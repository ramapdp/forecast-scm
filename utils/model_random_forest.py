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

from typing import Callable, Optional

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor

from . import modeling_prep

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
