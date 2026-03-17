"""Regel-Engine fuer Muehle (Nine Men's Morris)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from gaming_robot_arm.games.common.interfaces import Move, Player, Rules
from .board import ADJACENT, BOARD_LABELS, MILLS_BY_POSITION
from .constants import PIECES_PER_PLAYER, PLAYERS
from .settings import MillRuleSettings
from .state import MillState


def other_player(player: Player) -> Player:
    return PLAYERS[1] if player == PLAYERS[0] else PLAYERS[0]


def empty_board() -> Dict[str, Optional[Player]]:
    return {label: None for label in BOARD_LABELS}


def count_pieces(board: Dict[str, Optional[Player]], player: Player) -> int:
    return sum(1 for owner in board.values() if owner == player)


def is_mill(board: Dict[str, Optional[Player]], player: Player, position: str) -> bool:
    for mill in MILLS_BY_POSITION[position]:
        if all(board[pos] == player for pos in mill):
            return True
    return False


def removable_positions(board: Dict[str, Optional[Player]], player: Player) -> List[str]:
    positions = [pos for pos, owner in board.items() if owner == player]
    if not positions:
        return []
    not_in_mill = [pos for pos in positions if not is_mill(board, player, pos)]
    return not_in_mill or positions


def state_signature(state: MillState) -> str:
    """Kompakte, deterministische Signatur fuer Dreifachwiederholungs-Pruefungen."""

    board_key = "".join(state.board[label] if state.board[label] is not None else "." for label in BOARD_LABELS)
    placed_key = ":".join(str(state.placed.get(player, 0)) for player in PLAYERS)
    return f"{board_key}|{state.to_move}|{placed_key}"


def phase_for_player(
    state: MillState,
    player: Player,
    *,
    settings: MillRuleSettings | None = None,
) -> str:
    active_settings = settings if settings is not None else MillRuleSettings()

    placed = state.placed.get(player, 0)
    if placed < PIECES_PER_PLAYER:
        return "placement"

    if active_settings.enable_flying:
        pieces = count_pieces(state.board, player)
        if pieces <= 3:
            return "flying"

    return "movement"


def _move_creates_mill(
    board: Dict[str, Optional[Player]],
    player: Player,
    dst: str,
) -> bool:
    return is_mill(board, player, dst)


def _with_piece_moved(
    board: Dict[str, Optional[Player]],
    src: Optional[str],
    dst: str,
    player: Player,
) -> Dict[str, Optional[Player]]:
    next_board = dict(board)
    if src is not None:
        next_board[src] = None
    next_board[dst] = player
    return next_board


@dataclass(slots=True)
class MillRules(Rules):
    """Regelimplementierung fuer Muehle."""

    settings: MillRuleSettings = field(default_factory=MillRuleSettings)

    def initial_state(self) -> MillState:
        state = MillState(
            board=empty_board(),
            to_move=PLAYERS[0],
            placed={PLAYERS[0]: 0, PLAYERS[1]: 0},
            plies_without_capture=0,
            position_history=(),
        )
        signature = state_signature(state)
        return MillState(
            board=state.board,
            to_move=state.to_move,
            placed=state.placed,
            plies_without_capture=state.plies_without_capture,
            position_history=(signature,),
        )

    def legal_moves(self, state: MillState) -> Sequence[Move]:
        player = state.to_move
        opponent = other_player(player)
        phase = phase_for_player(state, player, settings=self.settings)

        moves: List[Move] = []
        board = state.board

        if phase == "placement":
            empties = [pos for pos, owner in board.items() if owner is None]
            for dst in empties:
                next_board = _with_piece_moved(board, None, dst, player)
                if _move_creates_mill(next_board, player, dst):
                    captures = removable_positions(next_board, opponent)
                    if captures:
                        moves.extend(Move(player, None, dst, capture=cap) for cap in captures)
                    else:
                        moves.append(Move(player, None, dst))
                else:
                    moves.append(Move(player, None, dst))
            return moves

        pieces = [pos for pos, owner in board.items() if owner == player]
        empties = [pos for pos, owner in board.items() if owner is None]

        for src in pieces:
            destinations = empties if phase == "flying" else [n for n in ADJACENT[src] if board[n] is None]
            for dst in destinations:
                next_board = _with_piece_moved(board, src, dst, player)
                if _move_creates_mill(next_board, player, dst):
                    captures = removable_positions(next_board, opponent)
                    if captures:
                        moves.extend(Move(player, src, dst, capture=cap) for cap in captures)
                    else:
                        moves.append(Move(player, src, dst))
                else:
                    moves.append(Move(player, src, dst))

        return moves

    def apply_move(self, state: MillState, move: Move) -> MillState:
        player = state.to_move
        if move.player != player:
            raise ValueError("Spieler im Zug passt nicht zu state.to_move.")

        board = state.board
        phase = phase_for_player(state, player, settings=self.settings)
        opponent = other_player(player)

        next_board = dict(board)
        next_placed = dict(state.placed)

        if phase == "placement":
            if move.src is not None:
                raise ValueError("Setzzug darf keine Quelle (`src`) enthalten.")
            if move.dst not in next_board or next_board[move.dst] is not None:
                raise ValueError("Ziel (`dst`) eines Setzzugs muss ein freies Feld sein.")
            next_board[move.dst] = player
            next_placed[player] = next_placed.get(player, 0) + 1
        else:
            if move.src is None:
                raise ValueError("Bewegungszug erfordert eine Quelle (`src`).")
            if move.src not in next_board or next_board[move.src] != player:
                raise ValueError("Quelle (`src`) muss einen Stein des Spielers enthalten.")
            if move.dst not in next_board or next_board[move.dst] is not None:
                raise ValueError("Ziel (`dst`) muss frei sein.")
            if phase != "flying" and move.dst not in ADJACENT[move.src]:
                raise ValueError("Ziel (`dst`) muss angrenzen, ausser im Flying-Modus.")
            next_board[move.src] = None
            next_board[move.dst] = player

        formed_mill = _move_creates_mill(next_board, player, move.dst)
        if formed_mill:
            captures = removable_positions(next_board, opponent)
            if captures:
                if move.capture is None:
                    raise ValueError("Nach geschlossener Muehle ist ein Schlag erforderlich.")
                if move.capture not in captures:
                    raise ValueError("Schlagziel ist nicht schlagbar.")
                next_board[move.capture] = None
            elif move.capture is not None:
                raise ValueError("Schlag angegeben, aber keine gegnerischen Steine vorhanden.")
        elif move.capture is not None:
            raise ValueError("Ein Schlag ist nur bei geschlossener Muehle erlaubt.")

        next_plies_without_capture = 0 if move.capture is not None else state.plies_without_capture + 1
        next_state = MillState(
            board=next_board,
            to_move=opponent,
            placed=next_placed,
            plies_without_capture=next_plies_without_capture,
            position_history=state.position_history,
        )
        signature = state_signature(next_state)

        return MillState(
            board=next_state.board,
            to_move=next_state.to_move,
            placed=next_state.placed,
            plies_without_capture=next_state.plies_without_capture,
            position_history=(*state.position_history, signature),
        )

    def is_terminal(self, state: MillState) -> bool:
        if self._is_draw(state):
            return True

        for player in PLAYERS:
            if state.placed.get(player, 0) >= PIECES_PER_PLAYER:
                if count_pieces(state.board, player) < 3:
                    return True

        if state.placed.get(state.to_move, 0) >= PIECES_PER_PLAYER:
            return len(self.legal_moves(state)) == 0

        return False

    def winner(self, state: MillState) -> Optional[Player]:
        if self._is_draw(state):
            return None

        for player in PLAYERS:
            if state.placed.get(player, 0) >= PIECES_PER_PLAYER:
                if count_pieces(state.board, player) < 3:
                    return other_player(player)

        if state.placed.get(state.to_move, 0) >= PIECES_PER_PLAYER:
            if len(self.legal_moves(state)) == 0:
                return other_player(state.to_move)

        return None

    def draw_reason(self, state: MillState) -> Optional[str]:
        if self._is_draw_by_repetition(state):
            return "threefold_repetition"
        if self._is_draw_by_no_capture(state):
            return "no_capture_limit"
        return None

    def _is_draw(self, state: MillState) -> bool:
        return self._is_draw_by_repetition(state) or self._is_draw_by_no_capture(state)

    def _is_draw_by_repetition(self, state: MillState) -> bool:
        if not self.settings.enable_threefold_repetition:
            return False

        if state.position_history:
            current = state.position_history[-1]
            return state.position_history.count(current) >= 3

        return False

    def _is_draw_by_no_capture(self, state: MillState) -> bool:
        if not self.settings.enable_no_capture_draw:
            return False
        return state.plies_without_capture >= self.settings.no_capture_draw_plies


__all__ = ["MillRules", "MillRuleSettings", "phase_for_player", "state_signature"]
