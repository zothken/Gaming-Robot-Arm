"""PyTorch-Helfer fuer Policy/Value-Modelle in Muehle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from gaming_robot_arm.games.mill.neural_features import MOVE_FEATURE_DIM, STATE_FEATURE_DIM

_CHECKPOINT_VERSION = 1


class MillPolicyValueNet(nn.Module):
    """Zwei-Turm-Policy/Value-Netzwerk.

    - Zustands-Turm kodiert einen Zustandsvektor.
    - Zug-Turm kodiert jeden legalen Zugvektor.
    - Policy-Logits entstehen aus Zustand/Zug-Kompatibilitaet.
    - Value-Head sagt den Ausgang in [-1, 1] fuer die Seite am Zug voraus.
    """

    def __init__(
        self,
        *,
        state_dim: int = STATE_FEATURE_DIM,
        move_dim: int = MOVE_FEATURE_DIM,
        hidden_dim: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim muss groesser als 0 sein.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout muss im Bereich [0.0, 1.0) liegen.")

        self.state_dim = state_dim
        self.move_dim = move_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.move_encoder = nn.Sequential(
            nn.Linear(move_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.move_logit_bias = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state_features: Tensor,
        move_features: Tensor,
        move_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Vorwaertsdurchlauf.

        Args:
          state_features: [batch, state_dim]
          move_features: [batch, max_moves, move_dim]
          move_mask: Bool-Tensor [batch, max_moves], True fuer gueltige Zuege.
        Returns:
          logits: [batch, max_moves]
          value: [batch] in [-1, 1]
        """

        state_hidden = self.state_encoder(state_features)  # [B, H]
        move_hidden = self.move_encoder(move_features)  # [B, M, H]

        logits = (move_hidden * state_hidden.unsqueeze(1)).sum(dim=-1)
        logits = logits + self.move_logit_bias(move_hidden).squeeze(-1)
        if move_mask is not None:
            logits = logits.masked_fill(~move_mask, -1e9)

        value = torch.tanh(self.value_head(state_hidden)).squeeze(-1)
        return logits, value


def select_torch_device(device_name: str) -> torch.device:
    name = device_name.strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA angefordert, aber kein CUDA-Geraet verfuegbar.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("device muss eines von: auto, cpu, cuda sein.")


def checkpoint_model_kwargs(model: MillPolicyValueNet) -> dict[str, Any]:
    return {
        "state_dim": model.state_dim,
        "move_dim": model.move_dim,
        "hidden_dim": model.hidden_dim,
        "dropout": model.dropout,
    }


def save_checkpoint(
    *,
    path: str | Path,
    model: MillPolicyValueNet,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    best_val_loss: float | None = None,
    extra_metrics: dict[str, float] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": _CHECKPOINT_VERSION,
        "model_kwargs": checkpoint_model_kwargs(model),
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "best_val_loss": best_val_loss,
        "extra_metrics": extra_metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, checkpoint_path)


def load_checkpoint(path: str | Path, *, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    if "model_state" not in payload or "model_kwargs" not in payload:
        raise ValueError(f"Ungueltiges Checkpoint-Format: {checkpoint_path}")
    return payload


__all__ = [
    "MillPolicyValueNet",
    "checkpoint_model_kwargs",
    "load_checkpoint",
    "save_checkpoint",
    "select_torch_device",
]
