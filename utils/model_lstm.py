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
