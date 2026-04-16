"""Spiellogik fuer Muehle (Nine Men's Morris)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .ai.builtin import AlphaBetaMillAI, HeuristicMillAI
from .core.constants import PIECES_PER_PLAYER, PLAYERS
from .core.rules import MillRules
from .core.session import MillGameSession
from .core.settings import MillRuleSettings
from .core.state import MillState

if TYPE_CHECKING:
    from .ai.neural import NeuralMillAI as NeuralMillAI


def __getattr__(name: str):
    """Lade optionale Neural-AI erst bei Bedarf, um Torch-Imports zu vermeiden."""
    if name != "NeuralMillAI":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from .ai.neural import NeuralMillAI as _NeuralMillAI
    except ModuleNotFoundError as exc:  # pragma: no cover - optionaler Abhaengigkeitspfad (numpy/torch fehlt)
        if exc.name not in {"numpy", "torch"}:
            raise
        _NeuralMillAI = None

    globals()["NeuralMillAI"] = _NeuralMillAI
    return _NeuralMillAI

__all__ = [
    "AlphaBetaMillAI",
    "HeuristicMillAI",
    "MillGameSession",
    "MillRuleSettings",
    "MillRules",
    "MillState",
    "NeuralMillAI",
    "PIECES_PER_PLAYER",
    "PLAYERS",
]
