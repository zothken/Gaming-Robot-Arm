"""Zustandscontainer fuer Muehle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from gaming_robot_arm.games.common.interfaces import Player


@dataclass(frozen=True)
class MillState:
    board: Dict[str, Optional[Player]]
    to_move: Player
    placed: Dict[Player, int]
    plies_without_capture: int = 0
    position_history: Tuple[str, ...] = ()


__all__ = ["MillState"]
