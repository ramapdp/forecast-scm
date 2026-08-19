"""A 0.9-quantile LSTM, the third candidate model.

See docs/superpowers/specs/2026-08-19-lstm-modeling-design.md.

The one thing that separates this model from the other two is that it reads
the 28 days themselves rather than the hand-engineered summary of them that
`lag_*` and `roll_*` provide. The feature set is identical; the amount of
information reaching the model is not, and docs/hasil-modeling-lstm.md says so
in the head-to-head section rather than leaving a reader to discover it.
"""

from typing import Optional

import numpy as np
import torch
from torch import nn

from . import model_common, modeling_prep, sequence_windows

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
