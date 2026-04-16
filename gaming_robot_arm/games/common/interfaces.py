"""Leichtgewichtige Schnittstellen, die von Spiellogik-Modulen geteilt werden."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, Protocol, Sequence, TypeVar

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


_S = TypeVar("_S", bound=GameState)


class Rules(Protocol[_S]):
    """Minimaler Regelvertrag fuer rundenbasierte Spiele."""

    def initial_state(self) -> _S: ...

    def legal_moves(self, state: _S) -> Sequence[Move]: ...

    def apply_move(self, state: _S, move: Move) -> _S: ...

    def is_terminal(self, state: _S) -> bool: ...

    def winner(self, state: _S) -> Optional[Player]: ...
