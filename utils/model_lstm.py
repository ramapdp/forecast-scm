"""A 0.9-quantile LSTM, the third candidate model.

See docs/superpowers/specs/2026-08-19-lstm-modeling-design.md.

The one thing that separates this model from the other two is that it reads
the 28 days themselves rather than the hand-engineered summary of them that
`lag_*` and `roll_*` provide. The feature set is identical; the amount of
information reaching the model is not, and docs/hasil-modeling-lstm.md says so
in the head-to-head section rather than leaving a reader to discover it.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import (evaluation, model_common, modeling_prep, purging,
               sequence_windows, walk_forward)

# The service level the business ships at (B-9). Kept as a scalar beside the
# grid: one is a promise to outlets, the other is the head's output layout.
QUANTILE = 0.9

# The evaluation grid, taken from evaluation.py rather than restated, so the
# head cannot end up with an output layer that disagrees with the evaluator.
QUANTILES = evaluation.QUANTILE_SET_A

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
    "num_layers": 1,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "batch_size": 1024,
    "log_target": False,
    "grad_clip": 1.0,
    "random_state": 42,
}

# Restored to the full 144 points on 2026-08-24. `hidden_size=256` and
# `num_layers=2` were cut on 2026-08-19 because a 2-layer hidden-128 epoch
# costs 259 s against 104 s for its 1-layer twin — a cost decision, never a
# finding that depth or width did not help. Under the budget equalisation
# (spec §2.2) that cut stopped being affordable in a different currency: with
# two capacity dimensions missing, an LSTM loss at K1 could always be answered
# with "you never let it be bigger", and attribution is the output this
# comparison exists to produce. The wall-clock consequence is recorded in
# docs/hasil-modeling-lstm.md rather than absorbed by shrinking the space.
SEARCH_SPACE = {
    "hidden_size": [64, 128, 256],
    "num_layers": [1, 2],
    "dropout": [0.0, 0.2, 0.3],
    "learning_rate": [3e-4, 1e-3],
    "batch_size": [1024, 2048],
    "log_target": [False, True],
}

# Equal to XGBoost's draw count, and pinned rather than derived from
# `candidate_budget()` — see spec §2.2. Thirty against thirty is what makes
# "the LSTM lost" a statement about the architecture instead of about the
# search depth.
N_CANDIDATES = 30


def pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantiles: tuple = QUANTILES,
) -> torch.Tensor:
    """The training objective *is* the selection criterion.

    `prediction` is `(batch, len(quantiles))`, `target` is `(batch,)` and
    broadcasts across the grid. The sum runs over quantiles and the mean over
    rows: summing keeps each point's gradient at its own scale, so the extreme
    quantiles are not drowned out by the dense middle of the grid, while the
    mean over rows keeps the number independent of batch size.

    K1 is the *mean* over quantiles, so it and this loss differ by the constant
    `len(quantiles)` and by nothing else — which preserves the property
    `reg:quantileerror` gives XGBoost: a model cannot win the fit and lose the
    metric.
    """
    alphas = torch.as_tensor(quantiles, dtype=prediction.dtype,
                             device=prediction.device).view(1, -1)
    difference = target.view(-1, 1) - prediction
    return torch.maximum(alphas * difference,
                         (alphas - 1.0) * difference).sum(dim=1).mean()


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
        n_quantiles: int = len(QUANTILES),
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
        # One output neuron per quantile, on a shared trunk: every point reads
        # the same LSTM state and the same embeddings, which is what makes this
        # cheaper than len(QUANTILE_SET) separate networks and what lets the
        # quantiles inform each other. Nothing here forces the outputs to be
        # monotone in tau — crossing is measured, not designed away.
        self.head = nn.Sequential(
            nn.Linear(width, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_quantiles),
        )

    def forward(self, x_dynamic: torch.Tensor, x_cats: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x_dynamic)
        last = output[:, -1, :]
        embedded = [layer(x_cats[:, position])
                    for position, layer in enumerate(self.embeddings)]
        return self.head(torch.cat([last, *embedded], dim=1))


def build_model(params: dict, n_dynamic: int, sizes: list, seed: int,
                n_quantiles: int = len(QUANTILES)) -> QuantileLSTM:
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
        n_quantiles=n_quantiles,
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
    quantiles: tuple,
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
        loss = pinball_loss(model(x_dynamic, x_cats), y, quantiles)
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

        total += float(loss.detach()) * len(batch)
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
    """Predictions in `ends` order, one column per quantile point.

    Returned exactly as the head produced them — no sort. A post-hoc sort
    would drive `evaluation.crossing_rate()` to zero without making the
    distribution any more coherent, which is hiding the measurement rather
    than taking it.
    """
    model.eval()
    if len(ends) == 0:
        return np.empty((0, model.head[-1].out_features), dtype="float32")
    parts = []
    for start in range(0, len(ends), batch_size):
        chunk = ends[start:start + batch_size]
        x_dynamic, x_cats = _to_tensors(scaled, cats, chunk, lookback, device)
        parts.append(model(x_dynamic, x_cats).cpu().numpy())
    return np.concatenate(parts)


def _evaluate(model, scaled, cats, ends, targets, quantiles, device, lookback):
    """The early-stopping metric: mean pinball across the grid — K1's own
    definition, so the epoch that wins here is the epoch that wins the
    criterion the candidate is later ranked on.

    The mean rather than the training loss's sum. They order epochs
    identically (they differ by len(quantiles)), and reading the same number
    the results tables carry is worth more than saving a multiplication.
    """
    prediction = predict(model, scaled, cats, ends, device=device, lookback=lookback)
    alphas = np.asarray(quantiles, dtype="float64").reshape(1, -1)
    difference = (targets.astype("float64").reshape(-1, 1)
                  - prediction.astype("float64"))
    return float(np.maximum(alphas * difference,
                            (alphas - 1.0) * difference).mean())


def fit_with_early_stopping(
    params: dict,
    index: dict,
    fit_ends: np.ndarray,
    fit_targets: np.ndarray,
    es_ends: np.ndarray,
    es_targets: np.ndarray,
    quantiles: tuple,
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
                        params["random_state"], n_quantiles=len(quantiles))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    generator = torch.Generator().manual_seed(params["random_state"])

    best_score, best_epoch, since_improvement = float("inf"), 1, 0
    for epoch in range(1, max_epochs + 1):
        run_epoch(model, optimizer, scaled, index["cats"], fit_ends, fit_targets,
                  params, quantiles, generator, device, lookback)
        score = _evaluate(model, scaled, index["cats"], es_ends, es_targets,
                          quantiles, device, lookback)
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
    quantiles: tuple,
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
                        params["random_state"], n_quantiles=len(quantiles))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    generator = torch.Generator().manual_seed(params["random_state"])

    for _ in range(epochs):
        run_epoch(model, optimizer, scaled, index["cats"], ends, targets,
                  params, quantiles, generator, device, lookback)
    model.epochs_run = epochs
    return model


def _target(frame, params: dict) -> np.ndarray:
    values = model_common.train_target(frame)
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
    quantiles: tuple = QUANTILES,
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
    quantiles = tuple(quantiles)
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
            es_ends, _target(es_rows, params), quantiles=quantiles, sizes=sizes,
            device=device, scaled=scaled, max_epochs=max_epochs,
            patience=patience, lookback=lookback)
        fit_predict.best_epochs.append(int(best_epoch))

        model = fit_epochs(
            params, index, train_ends, _target(train, params), epochs=best_epoch,
            quantiles=quantiles, sizes=sizes, device=device, scaled=scaled,
            lookback=lookback)

        prediction = predict(model, scaled, index["cats"], valid_ends,
                             device=device, lookback=lookback)
        prediction = np.asarray(prediction, dtype="float64").reshape(
            len(valid_ends), len(quantiles))
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
    quantiles=...)`. There is no slot for the panel, and adding one would
    change a signature the other two models already satisfy — so the panel is
    bound here instead.

    The window index is built **once**. It costs a sort of 1.5M rows;
    rebuilding it per candidate would repeat that N x 2 times for nothing.
    """
    index = sequence_windows.build_index(panel, feature_cols=feature_cols,
                                         lookback=lookback)

    def make(params=None, feature_cols=None, quantiles: tuple = QUANTILES):
        if feature_cols is not None and list(feature_cols) != index["feature_cols"]:
            raise ValueError(
                "feature_cols berbeda dari yang dipakai membangun indeks"
            )
        return make_fit_predict(params, index=index, quantiles=quantiles,
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
SEED_REPEATS_FILE = str(BASE_DIR / "dataset/model_ready/lstm_seed_repeats.csv")

# The winner is re-run at each of these. 42 is included rather than assumed:
# its row has to come back identical to the winner's row in the search CSV,
# and a mismatch there is nondeterminism nobody has noticed yet — a finding of
# its own, not a rounding difference to wave through.
SEED_REPEATS = (42, 43, 44)


def fit_final(
    df,
    params: dict,
    feature_cols: Optional[list] = None,
    lookback: int = modeling_prep.LOOKBACK,
    tail_days: int = ES_TAIL_DAYS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_EPOCHS,
    quantiles: tuple = QUANTILES,
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
    quantiles = tuple(quantiles)
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
        quantiles=quantiles, sizes=sizes, device=device, scaled=scaled,
        max_epochs=max_epochs, patience=patience, lookback=lookback)

    model = fit_epochs(params, index, ends, _target(frame, params),
                       epochs=best_epoch, quantiles=quantiles, sizes=sizes,
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
        # The whole grid in head order, not a scalar. The column order of the
        # head is unrecoverable from `state_dict` alone, so without this a
        # reloaded model returns nineteen unlabelled numbers.
        "quantiles": quantiles,
        **model_common.target_provenance(),
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
    quantiles = tuple(bundle["quantiles"])
    model = build_model(bundle["params"], len(bundle["dynamic_cols"]),
                        bundle["embedding_sizes"], bundle["params"]["random_state"],
                        n_quantiles=len(quantiles))
    model.load_state_dict(bundle["state_dict"])
    model.to(device)

    scaled = scale_values(index["values"], bundle["scaler"], bundle["dynamic_cols"])
    ends = sequence_windows.window_ends(index, frame)
    prediction = np.asarray(
        predict(model, scaled, index["cats"], ends, device=device,
                lookback=bundle["lookback"]),
        dtype="float64",
    ).reshape(len(ends), len(quantiles))
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
    quantiles: tuple = QUANTILES,
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
        quantiles=quantiles,
        model_name=model_name,
        feature_cols=feature_cols,
        verbose=verbose,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )


def run_seed_repeats(
    df,
    params: dict,
    seeds: tuple = SEED_REPEATS,
    folds: tuple = SEARCH_FOLDS,
    quantiles: tuple = QUANTILES,
    model_name: str = "lstm",
    feature_cols: Optional[list] = None,
    lookback: int = modeling_prep.LOOKBACK,
    device_name: str = "cpu",
    sizes: Optional[list] = None,
    tail_days: int = ES_TAIL_DAYS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_EPOCHS,
    verbose: bool = True,
    output_path: Optional[str] = None,
):
    """Re-run one configuration across seeds, on the search folds.

    The LSTM is the only model here whose weights start random, so a small K1
    gap between it and a tree model has never been separable from seed noise —
    the spread across folds mixes seed variance with data variance and cannot
    be asked to answer this. Running the *winner* at three seeds on the same
    two folds measures it directly.

    Only the winner. Repeating all thirty candidates would pay three times the
    search bill to answer a question that only matters for the configuration
    actually being reported.

    The row layout is `summarise_candidate()`'s, deliberately: the seed-42 row
    must be comparable column for column with that candidate's row in
    `lstm_search_results.csv`, and any difference between them is
    nondeterminism rather than seed variance.
    """
    frame = walk_forward.eligible_rows(df, lookback=lookback)
    make = bind_panel(df, feature_cols=feature_cols, lookback=lookback,
                      device_name=device_name, sizes=sizes, tail_days=tail_days,
                      max_epochs=max_epochs, patience=patience)

    rows = []
    for seed in seeds:
        started = time.perf_counter()
        fit_predict = make({**params, "random_state": seed},
                           quantiles=quantiles)
        results = pd.concat(
            [walk_forward.run_fold(frame, fold_id, fit_predict,
                                   model_name=model_name, quantiles=quantiles,
                                   prepared=True)
             for fold_id in folds],
            ignore_index=True,
        )
        record = {
            "seed": seed,
            **model_common.summarise_candidate(results, model_name, folds,
                                               quantiles),
            "best_epoch": model_common.reported_capacity(fit_predict),
            "elapsed_seconds": round(time.perf_counter() - started, 1),
        }
        rows.append(record)
        if verbose:
            print(f"seed {seed}: K1={record['pinball']:.4f} "
                  f"epoch={record['best_epoch'] or '-'} "
                  f"{record['elapsed_seconds']:.0f}s", flush=True)

    table = pd.DataFrame(rows)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
    return table


def seed_spread(repeats) -> dict:
    """min / mean / max / range of K1 across the seeds.

    `range` is the number that decides how the K1 table may be read: if the
    spread across seeds of one configuration exceeds the K1 distance between
    the LSTM and a tree model, that distance is not a difference between
    models and the results document has to say so (spec §2.3).
    """
    values = pd.Series(repeats["pinball"], dtype="float64").dropna()
    if values.empty:
        return {"min": float("nan"), "mean": float("nan"),
                "max": float("nan"), "range": float("nan")}
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
    }
