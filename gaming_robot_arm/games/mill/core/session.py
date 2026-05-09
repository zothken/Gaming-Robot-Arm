"""Sitzungshelfer fuer Muehle-Zustand, Zughistorie und KI-Integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from gaming_robot_arm.games.common.interfaces import Move
from .rules import MillRules
from .state import MillState


class MillMoveProvider(Protocol):
    """Minimaler Adaptervertrag fuer KI-Zugauswahl in MillGameSession."""

    def choose_move(self, state: MillState, rules: MillRules, move_history: Sequence[Move]) -> Move: ...


@dataclass(slots=True)
class MillGameSession:
    """Zustandsbehafteter Match-Container fuer Muehle mit optionaler KI-Anbindung."""

    rules: MillRules = field(default_factory=MillRules)
    state: MillState = field(init=False)
    move_history: list[Move] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.state = self.rules.initial_state()

    def reset(self) -> MillState:
        self.state = self.rules.initial_state()
        self.move_history.clear()
        return self.state

    def legal_moves(self) -> Sequence[Move]:
        return self.rules.legal_moves(self.state)

    def apply_move(self, move: Move) -> MillState:
        self.state = self.rules.apply_move(self.state, move)
        self.move_history.append(move)
        return self.state

    def undo_last_move(self) -> Move | None:
        """Nimmt den letzten Halbzug zurueck. Gibt den entfernten Zug zurueck (oder None bei leerer History)."""
        if not self.move_history:
            return None
        popped = self.move_history.pop()
        self._replay_from_history()
        return popped

    def undo_n_moves(self, n: int) -> list[Move]:
        """Nimmt bis zu n Halbzuege zurueck. Rueckgabe in Pop-Reihenfolge (juengster zuerst)."""
        popped: list[Move] = []
        for _ in range(n):
            if not self.move_history:
                break
            popped.append(self.move_history.pop())
        self._replay_from_history()
        return popped

    def _replay_from_history(self) -> None:
        self.state = self.rules.initial_state()
        for mv in self.move_history:
            self.state = self.rules.apply_move(self.state, mv)

    def choose_ai_move(self, provider: MillMoveProvider) -> Move:
        return provider.choose_move(self.state, self.rules, self.move_history)

    def is_terminal(self) -> bool:
        return self.rules.is_terminal(self.state)

    def winner(self):
        return self.rules.winner(self.state)


__all__ = ["MillGameSession", "MillMoveProvider"]
