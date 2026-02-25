"""Spiellogik fuer Muehle (Nine Men's Morris)."""

from .builtin_ai import AlphaBetaMillAI, HeuristicMillAI
from .constants import PIECES_PER_PLAYER, PLAYERS
from .rules import MillRules
from .session import MillGameSession
from .settings import MillRuleSettings
from .state import MillState


def __getattr__(name: str):
    """Lade optionale Neural-AI erst bei Bedarf, um Torch-Imports zu vermeiden."""
    if name != "NeuralMillAI":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from .neural_ai import NeuralMillAI as _NeuralMillAI
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
