"""A 0.9-quantile LSTM, the third candidate model.

See docs/superpowers/specs/2026-08-19-lstm-modeling-design.md.

The one thing that separates this model from the other two is that it reads
the 28 days themselves rather than the hand-engineered summary of them that
`lag_*` and `roll_*` provide. The feature set is identical; the amount of
information reaching the model is not, and docs/hasil-modeling-lstm.md says so
in the head-to-head section rather than leaving a reader to discover it.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from . import model_common, modeling_prep, purging, sequence_windows, walk_forward

QUANTILE = 0.9

# Same purged tail as XGBoost: the epoch count is a capacity decision, and the
# validation fold is the one place it cannot be taken from.
ES_TAIL_DAYS = 30
EARLY_STOPPING_EPOCHS = 5
MAX_EPOCHS = 100

# Wall-clock ceiling the search budget is derived from, in seconds.
BUDGET_SECONDS = 28_800
MIN_CANDIDATES = 6
MAX_CANDIDATES = 20

DEFAULT_PARAMS = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "batch_size": 1024,
    "log_target": False,
    "grad_clip": 1.0,
    "random_state": 42,
}

SEARCH_SPACE = {
    "hidden_size": [64, 128, 256],
    "num_layers": [1, 2],
    "dropout": [0.0, 0.2, 0.3],
    "learning_rate": [3e-4, 1e-3],
    "batch_size": [1024, 2048],
    "log_target": [False, True],
}


def pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float = QUANTILE,
) -> torch.Tensor:
    """The training objective *is* the selection criterion.

    The same property `reg:quantileerror` gave XGBoost: what is optimised
    during training and what is scored during evaluation are one function, so
    a model cannot win the fit and lose the metric.
    """
    difference = target - prediction
    return torch.maximum(quantile * difference,
                         (quantile - 1.0) * difference).mean()


def embedding_sizes(
    mapping: Optional[dict] = None,
    idx_cols: Optional[list] = None,
) -> list:
    """`(num_embeddings, embedding_dim)` per `_idx` column.

    `num_embeddings` comes from `category_mapping.json` — the highest index
    plus one, which already covers the reserved UNKNOWN slot at 0 — and never
    from the values that happen to appear in a fold's training rows. A branch
    opening after this model is trained maps to 0 and must stay in range; the
    alternative fails months later, in production, with an index error.
    """
    mapping = mapping if mapping is not None else modeling_prep.load_category_mapping()
    idx_cols = idx_cols or model_common.IDX_COLS
    sizes = []
    for col in idx_cols:
        source = col[: -len("_idx")]
        num_embeddings = max(mapping[source].values()) + 1
        sizes.append((num_embeddings, min(16, (num_embeddings + 1) // 2)))
    return sizes


class QuantileLSTM(nn.Module):
    """49 dynamic channels through the LSTM, 7 categoricals through embeddings.

    The categoricals are read at the prediction row, not repeated across the
    window: `Kategori Barang_idx` changes inside 301 real segments, so "the
    segment's category" is not a well-defined thing to repeat.
    """

    def __init__(
        self,
        n_dynamic: int,
        sizes: list,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(count, dim) for count, dim in sizes]
        )
        self.lstm = nn.LSTM(
            input_size=n_dynamic,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # nn.LSTM ignores this when num_layers == 1, which is why the head
            # below always applies dropout: otherwise the searched flag would
            # be meaningless across half the space.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        width = hidden_size + sum(dim for _, dim in sizes)
        self.head = nn.Sequential(
            nn.Linear(width, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_dynamic: torch.Tensor, x_cats: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x_dynamic)
        last = output[:, -1, :]
        embedded = [layer(x_cats[:, position])
                    for position, layer in enumerate(self.embeddings)]
        return self.head(torch.cat([last, *embedded], dim=1)).squeeze(1)


def build_model(params: dict, n_dynamic: int, sizes: list, seed: int) -> QuantileLSTM:
    """Seeded construction, so the two fits of the two-fit protocol start
    from identical weights and `best_epoch` means the same thing in both.
    """
    torch.manual_seed(seed)
    return QuantileLSTM(
        n_dynamic=n_dynamic,
        sizes=sizes,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    )


def resolve_device(name: str = "cpu") -> torch.device:
    """CPU by default, deliberately.

    MPS has no fused LSTM kernel and at these hidden sizes is often slower
    than CPU, so the benchmark measures both and records which one won rather
    than a default silently choosing.
    """
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS tidak tersedia di mesin ini")
    return torch.device(name)


def candidate_budget(
    sec_per_epoch: float,
    best_epoch: int,
    budget_seconds: int = BUDGET_SECONDS,
    patience: int = EARLY_STOPPING_EPOCHS,
    minimum: int = MIN_CANDIDATES,
    maximum: int = MAX_CANDIDATES,
) -> int:
    """How many candidates fit inside the wall-clock ceiling.

    One two-fit candidate costs about
    `sec_per_epoch * (2 * best_epoch + patience)`, and each is scored on two
    folds. Fold 3's training window is shorter than fold 5's, so charging both
    at fold 5's measured rate is conservative.

    Below `minimum` this raises instead of clamping upward. Clamping up would
    be a silent overrun of the ceiling, and the spec is explicit that a
    too-small N is the signal to shrink the search space — a decision for the
    operator, not for this function.
    """
    per_fit = sec_per_epoch * (2 * best_epoch + patience)
    raw = int(budget_seconds // (2 * per_fit))
    if raw < minimum:
        raise ValueError(
            f"anggaran hanya cukup untuk {raw} kandidat (<{minimum}); "
            f"perkecil ruang search atau turunkan ongkos per fit — "
            f"jangan naikkan plafon {budget_seconds}s diam-diam"
        )
    return min(raw, maximum)


def scale_values(values: np.ndarray, scaler: dict, dynamic_cols: list) -> np.ndarray:
    """Standardise the whole panel matrix with one fold's scaler.

    The scaler is fit on that fold's training rows only. Applying it to every
    row, context rows included, is safe — what leaks is *fitting* it outside
    the training window, never applying it.
    """
    mean = np.array([scaler[col][0] for col in dynamic_cols], dtype="float32")
    std = np.array([scaler[col][1] for col in dynamic_cols], dtype="float32")
    return ((values - mean) / std).astype("float32")


def _shuffled_batches(count: int, batch_size: int, generator) -> list:
    order = torch.randperm(count, generator=generator).numpy()
    return [order[start:start + batch_size] for start in range(0, count, batch_size)]


def _to_tensors(scaled, cats, ends, lookback, device):
    windows = sequence_windows.gather(scaled, ends, lookback=lookback)
    x_dynamic = torch.from_numpy(windows).to(device)
    x_cats = torch.from_numpy(cats[ends].astype("int64")).to(device)
    return x_dynamic, x_cats


def run_epoch(
    model: QuantileLSTM,
    optimizer,
    scaled: np.ndarray,
    cats: np.ndarray,
    ends: np.ndarray,
    targets: np.ndarray,
    params: dict,
    quantile: float,
    generator,
    device,
    lookback: int = modeling_prep.LOOKBACK,
) -> float:
    """One pass over the training windows, returning the mean loss.

    Windows are shuffled across segments. Each one is self-contained, so no
    ordering needs preserving between batches.
    """
    model.train()
    total, seen = 0.0, 0
    for batch in _shuffled_batches(len(ends), params["batch_size"], generator):
        x_dynamic, x_cats = _to_tensors(scaled, cats, ends[batch], lookback, device)
        y = torch.from_numpy(targets[batch].astype("float32")).to(device)

        optimizer.zero_grad()
        loss = pinball_loss(model(x_dynamic, x_cats), y, quantile)
        if not torch.isfinite(loss):
            # Fails this candidate through run_search's existing catch tuple.
            # RuntimeError is not raised here on purpose: PyTorch uses it for
            # genuine bugs as well as OOM, so widening the tuple would launder
            # bugs into NaN rows.
            raise ValueError(
                "loss LSTM menjadi NaN/inf — kandidat digagalkan"
            )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])
        optimizer.step()

        total += float(loss) * len(batch)
        seen += len(batch)
    return total / max(seen, 1)


@torch.no_grad()
def predict(
    model: QuantileLSTM,
    scaled: np.ndarray,
    cats: np.ndarray,
    ends: np.ndarray,
    device,
    lookback: int = modeling_prep.LOOKBACK,
    batch_size: int = 4096,
) -> np.ndarray:
    """Predictions in `ends` order, so they line up with the caller's frame."""
    model.eval()
    if len(ends) == 0:
        return np.empty(0, dtype="float32")
    parts = []
    for start in range(0, len(ends), batch_size):
        chunk = ends[start:start + batch_size]
        x_dynamic, x_cats = _to_tensors(scaled, cats, chunk, lookback, device)
        parts.append(model(x_dynamic, x_cats).cpu().numpy())
    return np.concatenate(parts)


def _evaluate(model, scaled, cats, ends, targets, quantile, device, lookback):
    prediction = predict(model, scaled, cats, ends, device=device, lookback=lookback)
    difference = targets.astype("float64") - prediction.astype("float64")
    return float(np.maximum(quantile * difference,
                            (quantile - 1.0) * difference).mean())


def fit_with_early_stopping(
    params: dict,
    index: dict,
    fit_ends: np.ndarray,
    fit_targets: np.ndarray,
    es_ends: np.ndarray,
    es_targets: np.ndarray,
    quantile: float,
    sizes: list,
    device,
    scaled: Optional[np.ndarray] = None,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_EPOCHS,
    lookback: int = modeling_prep.LOOKBACK,
) -> tuple:
    """Fit on the purged rows, stop on the tail, report the epoch that won.

    Under `log_target` the stopping metric is computed on the log scale. That
    is sound: early stopping only chooses an epoch count *within* one
    candidate. Candidates are compared to each other by pinball on the
    original scale, after inversion.
    """
    scaled = index["values"] if scaled is None else scaled
    model = build_model(params, len(index["dynamic_cols"]), sizes,
                        params["random_state"])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    generator = torch.Generator().manual_seed(params["random_state"])

    best_score, best_epoch, since_improvement = float("inf"), 1, 0
    for epoch in range(1, max_epochs + 1):
        run_epoch(model, optimizer, scaled, index["cats"], fit_ends, fit_targets,
                  params, quantile, generator, device, lookback)
        score = _evaluate(model, scaled, index["cats"], es_ends, es_targets,
                          quantile, device, lookback)
        if score < best_score:
            best_score, best_epoch, since_improvement = score, epoch, 0
        else:
            since_improvement += 1
            if since_improvement >= patience:
                break
    model.best_score = best_score
    return model, best_epoch


def fit_epochs(
    params: dict,
    index: dict,
    ends: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    quantile: float,
    sizes: list,
    device,
    scaled: Optional[np.ndarray] = None,
    lookback: int = modeling_prep.LOOKBACK,
) -> QuantileLSTM:
    """The second fit: same seed, all training rows, a fixed epoch count.

    One epoch here contains about 5% more gradient steps than one epoch of the
    first fit, because the early-stopping tail is back in. That is accepted:
    pinning by epoch means "the same number of passes over the data", which is
    the more meaningful invariant than a fixed step count.
    """
    scaled = index["values"] if scaled is None else scaled
    model = build_model(params, len(index["dynamic_cols"]), sizes,
                        params["random_state"])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    generator = torch.Generator().manual_seed(params["random_state"])

    for _ in range(epochs):
        run_epoch(model, optimizer, scaled, index["cats"], ends, targets,
                  params, quantile, generator, device, lookback)
    model.epochs_run = epochs
    return model


def _target(frame, params: dict) -> np.ndarray:
    values = frame[modeling_prep.TARGET_COL].to_numpy(dtype="float64")
    return (np.log1p(values) if params["log_target"] else values).astype("float32")


def _assert_no_december(index: dict, ends: np.ndarray,
                        test_start=modeling_prep.TEST_START) -> None:
    """G5. Redundant with the fold definitions, and kept anyway: the cost of
    one accidental leak is the credibility of the final number.
    """
    if len(ends) and index["dates"][ends].max() >= np.datetime64(test_start, "D"):
        raise ValueError("ada window yang menyentuh Desember 2025")


def _assert_train_precedes_valid(index: dict, train_ends: np.ndarray,
                                 valid_ends: np.ndarray) -> None:
    """G3. Checks the *position mapping*, not `fold_train_mask` — the frames
    were already split correctly by `walk_forward`, so what this can still
    catch is `window_ends()` handing back the wrong rows.
    """
    if len(train_ends) and len(valid_ends):
        if index["dates"][train_ends].max() >= index["dates"][valid_ends].min():
            raise ValueError(
                "posisi window training tidak seluruhnya mendahului validasi"
            )


def make_fit_predict(
    params: Optional[dict] = None,
    index: Optional[dict] = None,
    feature_cols: Optional[list] = None,
    quantile: float = QUANTILE,
    tail_days: int = ES_TAIL_DAYS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_EPOCHS,
    device_name: str = "cpu",
    sizes: Optional[list] = None,
) -> "object":
    """The callable `walk_forward.run_fold()` injects.

    Two fits. The first runs on the purged fit rows with the 30-day tail as
    its eval set and reports the epoch that won. The second discards that
    model, re-initialises from the same seed, and trains on **every** training
    row for exactly that many epochs — so the model producing the reported
    predictions has seen the same population the Random Forest and XGBoost
    were trained on.

    Best epochs are recorded on the returned callable rather than returned,
    because `walk_forward` accepts predictions and nothing else — and their
    spread across folds is worth reporting.
    """
    if index is None:
        raise ValueError("make_fit_predict butuh indeks dari bind_panel()")
    params = {**DEFAULT_PARAMS, **(params or {})}
    device = resolve_device(device_name)
    # From category_mapping.json unless the caller supplies its own — tests do,
    # because a synthetic _idx fixture column has no entry in that file.
    sizes = sizes if sizes is not None else embedding_sizes(
        idx_cols=index["idx_cols"])
    lookback = index["lookback"]

    def fit_predict(train, valid) -> np.ndarray:
        model_common.assert_no_nan(train, index["feature_cols"])
        model_common.assert_no_nan(valid, index["feature_cols"])

        train_ends = sequence_windows.window_ends(index, train)
        valid_ends = sequence_windows.window_ends(index, valid)
        _assert_train_precedes_valid(index, train_ends, valid_ends)
        _assert_no_december(index, np.concatenate([train_ends, valid_ends]))

        # One scaler for the whole fold, used by both fits. The tail's
        # statistics sit inside the training window so nothing leaks into
        # validation; sharing it is what makes best_epoch transfer between
        # two fits that would otherwise see differently scaled inputs.
        scaler = modeling_prep.fit_scaler(train, index["dynamic_cols"])
        scaled = scale_values(index["values"], scaler, index["dynamic_cols"])

        fit_rows, es_rows = model_common.split_early_stopping(
            train, tail_days=tail_days)
        fit_ends = sequence_windows.window_ends(index, fit_rows)
        es_ends = sequence_windows.window_ends(index, es_rows)

        _, best_epoch = fit_with_early_stopping(
            params, index, fit_ends, _target(fit_rows, params),
            es_ends, _target(es_rows, params), quantile=quantile, sizes=sizes,
            device=device, scaled=scaled, max_epochs=max_epochs,
            patience=patience, lookback=lookback)
        fit_predict.best_epochs.append(int(best_epoch))

        model = fit_epochs(
            params, index, train_ends, _target(train, params), epochs=best_epoch,
            quantile=quantile, sizes=sizes, device=device, scaled=scaled,
            lookback=lookback)

        prediction = predict(model, scaled, index["cats"], valid_ends,
                             device=device, lookback=lookback)
        prediction = np.asarray(prediction, dtype="float64")
        if params["log_target"]:
            prediction = modeling_prep.inverse_log_target(prediction)
        # A negative shipment quantity is not a thing.
        return np.clip(prediction, 0.0, None)

    fit_predict.best_epochs = []
    fit_predict.index = index
    return fit_predict


def bind_panel(
    panel,
    feature_cols: Optional[list] = None,
    lookback: int = modeling_prep.LOOKBACK,
    device_name: str = "cpu",
    tail_days: int = ES_TAIL_DAYS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_EPOCHS,
    sizes: Optional[list] = None,
):
    """Give `model_common.run_search()` a callable of the signature it expects.

    `run_search` calls `make_fit_predict(candidate, feature_cols=...,
    quantile=...)`. There is no slot for the panel, and adding one would
    change a signature the other two models already satisfy — so the panel is
    bound here instead.

    The window index is built **once**. It costs a sort of 1.5M rows;
    rebuilding it per candidate would repeat that N x 2 times for nothing.
    """
    index = sequence_windows.build_index(panel, feature_cols=feature_cols,
                                         lookback=lookback)

    def make(params=None, feature_cols=None, quantile: float = QUANTILE):
        if feature_cols is not None and list(feature_cols) != index["feature_cols"]:
            raise ValueError(
                "feature_cols berbeda dari yang dipakai membangun indeks"
            )
        return make_fit_predict(params, index=index, quantile=quantile,
                                tail_days=tail_days, max_epochs=max_epochs,
                                patience=patience, device_name=device_name,
                                sizes=sizes)

    make.index = index
    return make


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = str(BASE_DIR / "models/lstm_q90.joblib")
BEST_PARAMS_FILE = str(BASE_DIR / "dataset/model_ready/lstm_best_params.json")
SEARCH_FILE = str(BASE_DIR / "dataset/model_ready/lstm_search_results.csv")
RESULTS_FILE = str(BASE_DIR / "dataset/model_ready/lstm_walk_forward_results.csv")


def fit_final(
    df,
    params: dict,
    feature_cols: Optional[list] = None,
    lookback: int = modeling_prep.LOOKBACK,
    tail_days: int = ES_TAIL_DAYS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_EPOCHS,
    quantile: float = QUANTILE,
    device_name: str = "cpu",
    sizes: Optional[list] = None,
    date_col: str = modeling_prep.DATE_COL,
    test_start=modeling_prep.TEST_START,
) -> dict:
    """Fit on every eligible row before December, purged at that boundary.

    Eligibility comes from `walk_forward.eligible_rows`, not from a date
    filter written here: the rows this model is finally trained on have to be
    the rows it was scored on, and the scoring cuts are not just the date.

    The **windows**, though, are cut from the full `df` — the same reason
    `build_index` takes the whole panel. Context rows outside the eligible set
    are still legitimate history.
    """
    params = {**DEFAULT_PARAMS, **params}
    index = sequence_windows.build_index(df, feature_cols=feature_cols,
                                         lookback=lookback, date_col=date_col)
    device = resolve_device(device_name)
    sizes = sizes if sizes is not None else embedding_sizes(
        idx_cols=index["idx_cols"])

    frame = walk_forward.eligible_rows(df, lookback=lookback, date_col=date_col,
                                       test_start=test_start)
    frame = frame[purging.lookahead_safe_mask(frame, test_start, date_col=date_col)]
    model_common.assert_no_nan(frame, index["feature_cols"])

    ends = sequence_windows.window_ends(index, frame)
    _assert_no_december(index, ends, test_start=test_start)

    scaler = modeling_prep.fit_scaler(frame, index["dynamic_cols"])
    scaled = scale_values(index["values"], scaler, index["dynamic_cols"])

    fit_rows, es_rows = model_common.split_early_stopping(
        frame, tail_days=tail_days, date_col=date_col)
    _, best_epoch = fit_with_early_stopping(
        params, index,
        sequence_windows.window_ends(index, fit_rows), _target(fit_rows, params),
        sequence_windows.window_ends(index, es_rows), _target(es_rows, params),
        quantile=quantile, sizes=sizes, device=device, scaled=scaled,
        max_epochs=max_epochs, patience=patience, lookback=lookback)

    model = fit_epochs(params, index, ends, _target(frame, params),
                       epochs=best_epoch, quantile=quantile, sizes=sizes,
                       device=device, scaled=scaled, lookback=lookback)

    return {
        "state_dict": {key: value.cpu() for key, value
                       in model.state_dict().items()},
        "params": params,
        "feature_cols": index["feature_cols"],
        "dynamic_cols": index["dynamic_cols"],
        "idx_cols": index["idx_cols"],
        "embedding_sizes": sizes,
        "scaler": scaler,
        "log_target": params["log_target"],
        "best_epoch": int(best_epoch),
        "quantile": quantile,
        "lookback": lookback,
        "n_train": int(len(frame)),
    }


def predict_bundle(bundle: dict, panel, frame) -> np.ndarray:
    """Predict with a fitted bundle, forcing the recorded column order.

    `panel` is required and not optional: an LSTM cannot predict from a row on
    its own — it needs the 28 days behind it. Rebuilding the index from
    `bundle["feature_cols"]` is what pins the column order, so a panel whose
    columns arrive in a different order produces identical predictions.
    """
    index = sequence_windows.build_index(
        panel, feature_cols=bundle["feature_cols"], lookback=bundle["lookback"])
    device = resolve_device(bundle["params"].get("device", "cpu"))
    model = build_model(bundle["params"], len(bundle["dynamic_cols"]),
                        bundle["embedding_sizes"], bundle["params"]["random_state"])
    model.load_state_dict(bundle["state_dict"])
    model.to(device)

    scaled = scale_values(index["values"], bundle["scaler"], bundle["dynamic_cols"])
    ends = sequence_windows.window_ends(index, frame)
    prediction = np.asarray(
        predict(model, scaled, index["cats"], ends, device=device,
                lookback=bundle["lookback"]),
        dtype="float64",
    )
    if bundle["log_target"]:
        prediction = modeling_prep.inverse_log_target(prediction)
    return np.clip(prediction, 0.0, None)


def save_bundle(bundle: dict, path: str = MODEL_FILE) -> None:
    model_common.save_bundle(bundle, path)


def load_bundle(path: str = MODEL_FILE) -> dict:
    return model_common.load_bundle(path)


def save_best_params(params: dict, path: str = BEST_PARAMS_FILE) -> None:
    model_common.save_best_params(params, path)


# Identical to the Random Forest and XGBoost searches, and that is the point:
# if this model searched on different folds, "folds 1, 2 and 4 are untouched
# by model selection" would collapse for all three models at once.
SEARCH_FOLDS = (3, 5)

select_best = model_common.select_best


def sample_search_space(
    n_candidates: int,
    seed: int = 42,
    space: Optional[dict] = None,
) -> list:
    """Distinct parameter sets drawn at random from SEARCH_SPACE.

    No affordability screen: unlike the quantile forest there is no
    leaf-storage bound to screen against — a batch of 2048 windows is 11 MB
    whatever the hidden size.

    `n_candidates` has no default on purpose. It comes from
    `candidate_budget()` and its measured inputs, so hard-coding one here
    would invite skipping the measurement.
    """
    return model_common.sample_search_space(
        space=SEARCH_SPACE if space is None else space,
        defaults=DEFAULT_PARAMS,
        n_candidates=n_candidates,
        seed=seed,
        screen=None,
    )


def run_search(
    df,
    candidates: list,
    folds: tuple = SEARCH_FOLDS,
    alpha: float = QUANTILE,
    model_name: str = "lstm",
    feature_cols: Optional[list] = None,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
    device_name: str = "cpu",
    lookback: int = modeling_prep.LOOKBACK,
    sizes: Optional[list] = None,
):
    """Score every LSTM candidate on the search folds.

    `df` is the panel, passed to both `run_search` (which cuts eligible rows
    from it) and `bind_panel` (which cuts windows from it) — the same frame in
    both places, so the rows scored and the rows windowed cannot drift apart.
    """
    return model_common.run_search(
        df,
        candidates,
        make_fit_predict=bind_panel(df, feature_cols=feature_cols,
                                    lookback=lookback, device_name=device_name,
                                    sizes=sizes),
        search_space=SEARCH_SPACE,
        folds=folds,
        alpha=alpha,
        model_name=model_name,
        feature_cols=feature_cols,
        verbose=verbose,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )
