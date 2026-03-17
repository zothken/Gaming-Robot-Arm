"""KI-Implementierungen fuer Muehle."""

from .builtin import AlphaBetaMillAI, HeuristicMillAI


def __getattr__(name: str):
    if name != "NeuralMillAI":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .neural import NeuralMillAI

    globals()["NeuralMillAI"] = NeuralMillAI
    return NeuralMillAI


__all__ = ["AlphaBetaMillAI", "HeuristicMillAI", "NeuralMillAI"]
