"""Gemeinsame Selbstspiel-Helfer fuer Muehle-ML."""

from __future__ import annotations

from gaming_robot_arm.games.common.interfaces import Move


def move_key(move: Move) -> tuple[str, str, str]:
    return (move.src or "", move.dst, move.capture or "")


def target_index_or_raise(legal_moves: list[Move], chosen_move: Move) -> int:
    try:
        return legal_moves.index(chosen_move)
    except ValueError:
        chosen = move_key(chosen_move)
        for index, move in enumerate(legal_moves):
            if move_key(move) == chosen:
                return index
    raise RuntimeError("Neuraler Spieler erzeugte einen Zug, der nicht in der legalen Zugliste steht.")


__all__ = ["move_key", "target_index_or_raise"]
