"""XGBoost at the 0.9 service level.

`reg:quantileerror` optimizes the same pinball loss the models are selected
on, so training objective and selection criterion agree — which is not true of
a squared-error model asked for a high quantile afterwards.

What makes this wrapper more than a thin call is the round count. Boosting
overfits if it runs too long, so the number of rounds is itself a
regularization decision, and the obvious place to make it — the validation
fold — is exactly the place that would leak. Early stopping therefore runs on
a held-out tail of the training window, and the model is then refit on the
full training rows at the round count that tail chose. Two fits per fold, so
that XGBoost is finally trained on the same population the Random Forest saw
and the comparison stays like-for-like.
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from xgboost.core import XGBoostError

from . import evaluation, model_common, modeling_prep, purging, walk_forward

# The service level the business ships at (B-9). Kept as a scalar beside the
# grid because it is a promise to outlets, not an evaluation choice.
QUANTILE = 0.9

# The evaluation grid, taken from evaluation.py rather than restated, so a
# Tahap A -> Tahap B switch cannot leave this module fitting the old points.
QUANTILES = evaluation.QUANTILE_SET_A

# The last 30 days of each fold's training window, held out to choose the
# round count. Long enough to cover a full delivery cycle and every weekday.
ES_TAIL_DAYS = 30
EARLY_STOPPING_ROUNDS = 50

# A ceiling, not a target: at learning_rate 0.03 the search needs room, and
# early stopping is what actually decides where a candidate lands.
MAX_ROUNDS = 2000

DEFAULT_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 10,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "encoding": "ordinal",
    "log_target": False,
    "random_state": 42,
}

# n_estimators is absent on purpose: early stopping decides it per candidate
# per fold, so searching it would spend budget on a question that already has
# a mechanism.
SEARCH_SPACE = {
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 10, 50],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "reg_lambda": [1.0, 10.0],
    "encoding": ["ordinal", "native", "one_hot"],
    "log_target": [False, True],
}

ESTIMATOR_KEYS = ("max_depth", "learning_rate", "min_child_weight",
                  "subsample", "colsample_bytree", "reg_lambda", "random_state")


# Moved to model_common: the LSTM chooses its epoch count the same way.
# Re-exported so test/test_model_xgboost.py and notebook/modeling_xgb.ipynb
# keep working with no line changed.
split_early_stopping = model_common.split_early_stopping


ENCODINGS = ("ordinal", "native", "one_hot")


def _present(frame: pd.DataFrame, idx_cols: Optional[list]) -> list:
    idx_cols = model_common.IDX_COLS if idx_cols is None else idx_cols
    return [col for col in idx_cols if col in frame.columns]


def training_categories(
    train_X: pd.DataFrame,
    idx_cols: Optional[list] = None,
) -> dict:
    """The category levels each encoded column had in training, sorted.

    Recorded in the bundle so a reloaded model rebuilds the identical dtype.
    Category codes are positional: rebuilding a column from a different level
    list silently renumbers every row.
    """
    return {
        col: sorted(train_X[col].dropna().unique().tolist())
        for col in _present(train_X, idx_cols)
    }


def _as_categorical(frame: pd.DataFrame, categories: dict) -> pd.DataFrame:
    out = frame.copy()
    for col, levels in categories.items():
        if col in out.columns:
            out[col] = out[col].astype(pd.CategoricalDtype(categories=levels))
    return out


def encode(
    train_X: pd.DataFrame,
    valid_X: pd.DataFrame,
    encoding: str,
    idx_cols: Optional[list] = None,
) -> tuple:
    """Prepare both matrices under one of the three searched encodings.

    Returns `(train_out, valid_out, enable_categorical)`. In every mode the
    validation columns are forced onto the training columns: a category
    present only in validation would otherwise shift every column after it,
    and the booster would read the wrong feature at each position — silently,
    since the shapes still line up.

    Under `native`, a level unseen in training becomes NaN rather than
    borrowing another level's code. XGBoost consumes NaN natively, so an
    unknown category is treated as missing, which is what it is.
    """
    if encoding == "ordinal":
        return train_X, valid_X.reindex(columns=train_X.columns), False
    if encoding == "native":
        categories = training_categories(train_X, idx_cols)
        train_out = _as_categorical(train_X, categories)
        valid_out = _as_categorical(valid_X.reindex(columns=train_X.columns),
                                    categories)
        return train_out, valid_out, True
    if encoding == "one_hot":
        train_out, valid_out = model_common.expand_one_hot(
            train_X, valid_X, idx_cols=_present(train_X, idx_cols)
        )
        return train_out, valid_out, False
    raise ValueError(f"encoding tidak dikenal: {encoding!r}, pilih dari {ENCODINGS}")


def apply_encoding(
    X: pd.DataFrame,
    encoding: str,
    columns: list,
    categories: dict,
    idx_cols: Optional[list] = None,
) -> tuple:
    """Encode a frame for prediction against a layout recorded at fit time.

    A booster reloaded next month against columns in a different order does
    not fail — it predicts confidently from the wrong features, which is
    worse. `columns` is the authority here, not whatever the caller passed in.
    """
    if encoding == "one_hot":
        out = pd.get_dummies(X, columns=_present(X, idx_cols))
    elif encoding == "native":
        out = _as_categorical(X, categories)
    elif encoding == "ordinal":
        out = X
    else:
        raise ValueError(f"encoding tidak dikenal: {encoding!r}, pilih dari {ENCODINGS}")

    out = out.reindex(columns=columns, fill_value=0)
    if encoding == "native":
        out = _as_categorical(out, categories)
    return out, encoding == "native"


def build_estimator(
    params: dict,
    n_estimators: int,
    enable_categorical: bool = False,
    early_stopping_rounds: Optional[int] = None,
    quantiles: tuple = QUANTILES,
) -> XGBRegressor:
    """A regressor whose training objective is the metric it is judged on.

    `quantile_alpha` takes the whole grid rather than one point. Every
    quantile is then fitted inside one booster against the same loss K1
    averages, so training objective and selection criterion stay the same
    function — the property a squared-error model asked for a high quantile
    afterwards does not have, and the one that would be lost by fitting
    nineteen separate single-quantile boosters.
    """
    kwargs = {key: params[key] for key in ESTIMATOR_KEYS if key in params}
    return XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=np.asarray(quantiles, dtype=float),
        tree_method="hist",
        n_estimators=n_estimators,
        enable_categorical=enable_categorical,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=-1,
        **kwargs,
    )


def _target(frame: pd.DataFrame, params: dict) -> np.ndarray:
    return model_common.train_target(frame, log_target=params["log_target"])


def _as_matrix(prediction, n_rows: int, n_quantiles: int) -> np.ndarray:
    """XGBoost drops the second axis on a one-point grid; this puts it back.

    Tahap B can hand out a grid of one (evaluation.quantile_set_b() dedupes),
    and every consumer downstream indexes columns. A silently 1-D return there
    would fail far from its cause.
    """
    return np.asarray(prediction, dtype=float).reshape(n_rows, n_quantiles)


def make_fit_predict(
    params: Optional[dict] = None,
    feature_cols: Optional[list] = None,
    quantiles: tuple = QUANTILES,
    tail_days: int = ES_TAIL_DAYS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    max_rounds: int = MAX_ROUNDS,
    idx_cols: Optional[list] = None,
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """The callable walk_forward.run_fold() injects.

    Two fits. The first runs on the purged fit rows with the tail as its eval
    set and reports where early stopping landed. The second discards that
    booster and refits on every training row at exactly that round count, so
    the model that produces the reported predictions has seen the same
    population the Random Forest was trained on.

    Under `log_target`, the early-stopping metric is computed on the log
    scale. That is sound: early stopping only chooses a round count *within*
    one candidate. Candidates are compared to each other by pinball on the
    original scale, after inversion.

    Round counts are recorded on the returned callable rather than returned,
    because `walk_forward` accepts predictions and nothing else — and the
    spread of round counts across folds is worth reporting.

    Early stopping now watches the mean quantile loss across the whole grid
    rather than the loss at 0.9. One round count still serves every point,
    which is a real constraint: a booster cannot stop at round 300 for τ=0.05
    and round 700 for τ=0.95. The alternative — nineteen independently stopped
    boosters — buys per-point round counts at the price of nineteen fits and a
    guaranteed loss of the shared structure, and this project pays for the
    shared structure.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    feature_cols = feature_cols or modeling_prep.FEATURE_COLS
    quantiles = tuple(quantiles)

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
        model_common.assert_no_nan(train, feature_cols)
        model_common.assert_no_nan(valid, feature_cols)

        fit_rows, es_rows = split_early_stopping(train, tail_days=tail_days)
        fit_X, es_X, enable = encode(fit_rows[feature_cols], es_rows[feature_cols],
                                     params["encoding"], idx_cols=idx_cols)
        probe = build_estimator(params, max_rounds, enable_categorical=enable,
                                early_stopping_rounds=early_stopping_rounds,
                                quantiles=quantiles)
        probe.fit(fit_X, _target(fit_rows, params),
                  eval_set=[(es_X, _target(es_rows, params))], verbose=False)
        best_iteration = int(probe.best_iteration) + 1
        fit_predict.best_iterations.append(best_iteration)

        train_X, valid_X, enable = encode(train[feature_cols], valid[feature_cols],
                                          params["encoding"], idx_cols=idx_cols)
        model = build_estimator(params, best_iteration, enable_categorical=enable,
                                quantiles=quantiles)
        model.fit(train_X, _target(train, params), verbose=False)

        prediction = _as_matrix(model.predict(valid_X), len(valid_X),
                                len(quantiles))
        if params["log_target"]:
            prediction = modeling_prep.inverse_log_target(prediction)
        # A negative shipment quantity is not a thing. No sort: crossing is
        # measured by evaluation.crossing_rate(), and sorting here would drive
        # that measurement to zero without fixing anything.
        return np.clip(prediction, 0.0, None)

    fit_predict.best_iterations = []
    return fit_predict


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_FILE = str(BASE_DIR / "models/xgboost_q90.joblib")
BEST_PARAMS_FILE = str(BASE_DIR / "dataset/model_ready/xgb_best_params.json")
SEARCH_FILE = str(BASE_DIR / "dataset/model_ready/xgb_search_results.csv")
RESULTS_FILE = str(BASE_DIR / "dataset/model_ready/xgb_walk_forward_results.csv")


def fit_final(
    df: pd.DataFrame,
    params: dict,
    feature_cols: Optional[list] = None,
    quantiles: tuple = QUANTILES,
    tail_days: int = ES_TAIL_DAYS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    max_rounds: int = MAX_ROUNDS,
    idx_cols: Optional[list] = None,
    date_col: str = modeling_prep.DATE_COL,
    test_start: pd.Timestamp = modeling_prep.TEST_START,
) -> dict:
    """Fit on every eligible row before December, purged at that boundary.

    Eligibility comes from `walk_forward.eligible_rows`, not from a date filter
    written here. The rows this model is finally trained on have to be the rows
    it was scored on, and the scoring cuts are not just the date: the first 28
    days of each segment have no usable lag window, and the last few days have
    no target at all, because the lead-time sum runs past the end of the data.

    Same two-fit protocol as walk-forward. The bundle records the training
    column order, the encoding, and the category levels, because a booster
    reloaded next month against a different layout does not fail — it predicts
    confidently from the wrong features, which is worse.
    """
    params = {**DEFAULT_PARAMS, **params}
    feature_cols = feature_cols or modeling_prep.FEATURE_COLS
    quantiles = tuple(quantiles)

    frame = walk_forward.eligible_rows(df, date_col=date_col, test_start=test_start)
    frame = frame[purging.lookahead_safe_mask(frame, test_start, date_col=date_col)]
    model_common.assert_no_nan(frame, feature_cols)

    fit_rows, es_rows = split_early_stopping(frame, tail_days=tail_days,
                                             date_col=date_col)
    fit_X, es_X, enable = encode(fit_rows[feature_cols], es_rows[feature_cols],
                                 params["encoding"], idx_cols=idx_cols)
    probe = build_estimator(params, max_rounds, enable_categorical=enable,
                            early_stopping_rounds=early_stopping_rounds,
                            quantiles=quantiles)
    probe.fit(fit_X, _target(fit_rows, params),
              eval_set=[(es_X, _target(es_rows, params))], verbose=False)
    best_iteration = int(probe.best_iteration) + 1

    train_X, _, enable = encode(frame[feature_cols], frame[feature_cols],
                                params["encoding"], idx_cols=idx_cols)
    model = build_estimator(params, best_iteration, enable_categorical=enable,
                            quantiles=quantiles)
    model.fit(train_X, _target(frame, params), verbose=False)

    return {
        "model": model,
        "params": params,
        "feature_cols": feature_cols,
        "columns": list(train_X.columns),
        "categories": training_categories(frame[feature_cols], idx_cols=idx_cols),
        "idx_cols": idx_cols,
        "encoding": params["encoding"],
        "log_target": params["log_target"],
        "best_iteration": best_iteration,
        "quantiles": quantiles,
        **model_common.target_provenance(),
        "n_train": int(len(frame)),
    }


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predict with a fitted bundle, forcing the recorded column order.

    The grid comes from the bundle rather than this module's constant: the
    booster's output columns are fixed at fit time, and reading them against a
    grid that has since moved would relabel every column silently.
    """
    features, _ = apply_encoding(frame[bundle["feature_cols"]],
                                 bundle["encoding"], bundle["columns"],
                                 bundle["categories"],
                                 idx_cols=bundle["idx_cols"])
    prediction = _as_matrix(bundle["model"].predict(features), len(features),
                            len(bundle["quantiles"]))
    if bundle["log_target"]:
        prediction = modeling_prep.inverse_log_target(prediction)
    return np.clip(prediction, 0.0, None)


def save_bundle(bundle: dict, path: str = MODEL_FILE) -> None:
    model_common.save_bundle(bundle, path)


def load_bundle(path: str = MODEL_FILE) -> dict:
    return model_common.load_bundle(path)


def save_best_params(params: dict, path: str = BEST_PARAMS_FILE) -> None:
    model_common.save_best_params(params, path)


SEARCH_FOLDS = (3, 5)

# More draws than the Random Forest's 18, because this space has more
# dimensions that genuinely move the score — not because XGBoost is being
# given an easier ride. The asymmetry is reported in docs/hasil-modeling-xgb.md.
N_CANDIDATES = 30

select_best = model_common.select_best


def sample_search_space(
    n_candidates: int = N_CANDIDATES,
    seed: int = 42,
    space: Optional[dict] = None,
) -> list:
    """Distinct parameter sets drawn at random from SEARCH_SPACE.

    No affordability screen: `hist` holds a quantized feature matrix — tens of
    megabytes at this size — so there is no analogue of the quantile forest's
    leaf-storage bound to screen against.
    """
    return model_common.sample_search_space(
        space=SEARCH_SPACE if space is None else space,
        defaults=DEFAULT_PARAMS,
        n_candidates=n_candidates,
        seed=seed,
        screen=None,
    )


def run_search(
    df: pd.DataFrame,
    candidates: list,
    folds: tuple = SEARCH_FOLDS,
    quantiles: tuple = QUANTILES,
    model_name: str = "xgboost",
    feature_cols: Optional[list] = None,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every XGBoost candidate on the search folds.

    XGBoostError joins the caught types: a candidate whose parameter
    combination the library rejects should be recorded and skipped, exactly
    like an over-budget forest, rather than ending a multi-hour run.
    """
    return model_common.run_search(
        df, candidates, make_fit_predict=make_fit_predict,
        search_space=SEARCH_SPACE, folds=folds, quantiles=quantiles,
        model_name=model_name, feature_cols=feature_cols, verbose=verbose,
        checkpoint_path=checkpoint_path, resume=resume,
        catch=(MemoryError, ValueError, XGBoostError),
    )
