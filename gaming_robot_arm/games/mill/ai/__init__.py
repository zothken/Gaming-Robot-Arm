"""KI-Implementierungen fuer Muehle."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .builtin import AlphaBetaMillAI, HeuristicMillAI

if TYPE_CHECKING:
    from .neural import NeuralMillAI as NeuralMillAI


def __getattr__(name: str):
    if name != "NeuralMillAI":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .neural import NeuralMillAI

    globals()["NeuralMillAI"] = NeuralMillAI
    return NeuralMillAI


__all__ = ["AlphaBetaMillAI", "HeuristicMillAI", "NeuralMillAI"]
