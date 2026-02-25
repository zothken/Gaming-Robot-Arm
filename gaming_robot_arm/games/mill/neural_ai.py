"""Neuronaler Zug-Provider fuer Muehle auf Basis eines PyTorch-Policy/Value-Modells."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch

from gaming_robot_arm.games.common.interfaces import Move
from gaming_robot_arm.games.mill.neural_features import (
    MOVE_FEATURE_DIM,
    encode_legal_move_features,
    encode_state_features,
)
from gaming_robot_arm.games.mill.neural_model import MillPolicyValueNet, load_checkpoint, select_torch_device
from gaming_robot_arm.games.mill.rules import MillRules
from gaming_robot_arm.games.mill.state import MillState


def _move_sort_key(move: Move) -> tuple[str, str, str]:
    return (move.src or "", move.dst, move.capture or "")


@dataclass(slots=True)
class NeuralMillAI:
    """Muehle-KI mit Policy/Value-Modell und optionalem stochastischem Sampling."""

    model_path: str | Path = "models/champion/mill_champion.pt"
    random_tiebreak: bool = True
    temperature: float = 0.0
    seed: int | None = None
    device: str = "auto"
    _rng: random.Random = field(init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)
    _model: MillPolicyValueNet = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError("temperature muss >= 0 sein.")
        self._rng = random.Random(self.seed)
        self._device = select_torch_device(self.device)

        checkpoint = load_checkpoint(self.model_path, device=self._device)
        model_kwargs = dict(checkpoint["model_kwargs"])
        self._model = MillPolicyValueNet(**model_kwargs).to(self._device)
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()

    def choose_move(self, state: MillState, rules: MillRules, move_history: Sequence[Move]) -> Move:
        del move_history
        legal_moves = list(rules.legal_moves(state))
        if not legal_moves:
            raise ValueError("Keine legalen Zuege verfuegbar.")

        state_features = encode_state_features(state, rules).astype(np.float32)
        move_features = encode_legal_move_features(legal_moves).astype(np.float32)
        logits = self._policy_logits(state_features, move_features)

        if self.temperature > 0.0:
            return self._sample_by_temperature(legal_moves, logits)

        best_score = float(np.max(logits))
        best_indices = [idx for idx, score in enumerate(logits) if float(score) == best_score]
        if len(best_indices) == 1:
            return legal_moves[best_indices[0]]

        best_moves = [legal_moves[idx] for idx in best_indices]
        if self.random_tiebreak:
            return self._rng.choice(best_moves)
        return min(best_moves, key=_move_sort_key)

    def evaluate_state(self, state: MillState, rules: MillRules) -> float:
        """Liefert die Value-Schaetzung in [-1, 1] fuer `state.to_move`."""

        state_features = encode_state_features(state, rules).astype(np.float32)
        with torch.no_grad():
            state_batch = torch.from_numpy(state_features).unsqueeze(0).to(self._device)
            dummy_move = torch.zeros((1, 1, MOVE_FEATURE_DIM), dtype=torch.float32, device=self._device)
            dummy_mask = torch.ones((1, 1), dtype=torch.bool, device=self._device)
            _, value = self._model(state_batch, dummy_move, dummy_mask)
        return float(value.item())

    def _policy_logits(self, state_features: np.ndarray, move_features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_batch = torch.from_numpy(state_features).unsqueeze(0).to(self._device)
            move_batch = torch.from_numpy(move_features).unsqueeze(0).to(self._device)
            move_mask = torch.ones((1, move_features.shape[0]), dtype=torch.bool, device=self._device)
            logits, _ = self._model(state_batch, move_batch, move_mask)
            return logits.squeeze(0).cpu().numpy().astype(np.float32)

    def _sample_by_temperature(self, legal_moves: list[Move], logits: np.ndarray) -> Move:
        scaled_logits = logits / max(self.temperature, 1e-6)
        max_logit = float(np.max(scaled_logits))
        exp_values = np.exp(scaled_logits - max_logit)
        probs = exp_values / np.sum(exp_values)
        choice_index = self._rng.choices(range(len(legal_moves)), weights=probs.tolist(), k=1)[0]
        return legal_moves[choice_index]


__all__ = ["NeuralMillAI"]
