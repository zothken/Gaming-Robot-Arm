"""Leichtgewichtige Schnittstellen, die von Spiellogik-Modulen geteilt werden."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

Player = str


@dataclass(frozen=True)
class Move:
    """Ein Spielzug von `src` nach `dst`, optional mit gegnerischem Schlag."""

    player: Player
    src: Optional[str]
    dst: str
    capture: Optional[str] = None


class GameState(Protocol):
    """Minimaler Zustandsvertrag fuer rundenbasierte Spiele."""

    to_move: Player


class Rules(Protocol):
    """Minimaler Regelvertrag fuer rundenbasierte Spiele."""

    def initial_state(self) -> GameState: ...

    def legal_moves(self, state: GameState) -> Sequence[Move]: ...

    def apply_move(self, state: GameState, move: Move) -> GameState: ...

    def is_terminal(self, state: GameState) -> bool: ...

    def winner(self, state: GameState) -> Optional[Player]: ...
