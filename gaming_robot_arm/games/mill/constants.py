"""Gemeinsame Konstanten fuer Muehle (Nine Men's Morris)."""

from __future__ import annotations

from gaming_robot_arm.games.common.interfaces import Player

PLAYERS: tuple[Player, Player] = ("W", "B")
PIECES_PER_PLAYER = 9

__all__ = ["PIECES_PER_PLAYER", "PLAYERS"]
