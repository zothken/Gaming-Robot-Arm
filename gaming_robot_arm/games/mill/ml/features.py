"""Feature-Encoding-Helfer fuer leichtgewichtige neuronale Muehle-Modelle."""

from __future__ import annotations

from typing import Dict

import numpy as np

from gaming_robot_arm.games.common.interfaces import Move, Player
from ..core.board import BOARD_LABELS
from ..core.constants import PIECES_PER_PLAYER
from ..core.rules import MillRules, count_pieces, other_player, phase_for_player
from ..core.state import MillState


_LABEL_TO_INDEX: Dict[str, int] = {label: idx for idx, label in enumerate(BOARD_LABELS)}
_PHASE_TO_INDEX: Dict[str, int] = {"placement": 0, "movement": 1, "flying": 2}

STATE_FEATURE_DIM = 83
MOVE_FEATURE_DIM = 77


def encode_state_features(state: MillState, rules: MillRules) -> np.ndarray:
    """Kodiert den Zustand aus Sicht des aktuellen Spielers."""

    features = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)
    player = state.to_move
    opponent = other_player(player)

    for label, owner in state.board.items():
        idx = _LABEL_TO_INDEX[label]
        if owner == player:
            features[idx] = 1.0
        elif owner == opponent:
            features[24 + idx] = 1.0
        else:
            features[48 + idx] = 1.0

    own_pieces = count_pieces(state.board, player)
    opp_pieces = count_pieces(state.board, opponent)
    features[72] = state.placed.get(player, 0) / PIECES_PER_PLAYER
    features[73] = state.placed.get(opponent, 0) / PIECES_PER_PLAYER
    features[74] = own_pieces / PIECES_PER_PLAYER
    features[75] = opp_pieces / PIECES_PER_PLAYER
    features[76] = min(state.plies_without_capture, 200) / 200.0

    own_phase = phase_for_player(state, player, settings=rules.settings)
    opp_phase = phase_for_player(state, opponent, settings=rules.settings)
    features[77 + _PHASE_TO_INDEX[own_phase]] = 1.0
    features[80 + _PHASE_TO_INDEX[opp_phase]] = 1.0

    return features


def outcome_for_player(winner: Player | None, player: Player) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0


def encode_move_features(move: Move) -> np.ndarray:
    features = np.zeros(MOVE_FEATURE_DIM, dtype=np.float32)

    if move.src is None:
        features[24] = 1.0
    else:
        features[_LABEL_TO_INDEX[move.src]] = 1.0

    features[25 + _LABEL_TO_INDEX[move.dst]] = 1.0

    if move.capture is None:
        features[73] = 1.0
    else:
        features[49 + _LABEL_TO_INDEX[move.capture]] = 1.0

    features[74] = 1.0 if move.src is None else 0.0
    features[75] = 1.0 if move.src is not None else 0.0
    features[76] = 1.0 if move.capture is not None else 0.0
    return features


def encode_legal_move_features(moves: list[Move]) -> np.ndarray:
    if not moves:
        return np.zeros((0, MOVE_FEATURE_DIM), dtype=np.float32)
    return np.stack([encode_move_features(move) for move in moves]).astype(np.float32)


__all__ = [
    "MOVE_FEATURE_DIM",
    "STATE_FEATURE_DIM",
    "encode_legal_move_features",
    "encode_move_features",
    "encode_state_features",
    "outcome_for_player",
]
