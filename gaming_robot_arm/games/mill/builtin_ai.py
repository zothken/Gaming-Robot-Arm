"""Interne Muehle-KIs, die keine externe Engine benoetigen."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Sequence
from typing import Literal

from gaming_robot_arm.games.common.interfaces import Move, Player
from gaming_robot_arm.games.mill.board import ADJACENT, BOARD_LABELS, MILLS
from gaming_robot_arm.games.mill.rules import MillRules, count_pieces, other_player, state_signature
from gaming_robot_arm.games.mill.state import MillState


_WIN_SCORE = 1_000_000.0
_INF = float("inf")


def _move_sort_key(move: Move) -> tuple[str, str, str]:
    """Stabiler Tie-Break-Schluessel fuer deterministisches KI-Verhalten."""

    return (move.src or "", move.dst, move.capture or "")


@dataclass(slots=True)
class HeuristicMillAI:
    """Einfache einstufige Heuristik-KI.

    Die Policy bevorzugt direkte Siege und Schlaege, bestraft Zuege mit
    starken gegnerischen Antworten und loest Gleichstaende zufaellig oder
    deterministisch auf.
    """

    random_tiebreak: bool = True
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def choose_move(self, state: MillState, rules: MillRules, move_history: Sequence[Move]) -> Move:
        del move_history  # Wird fuer diese Basisheuristik nicht benoetigt.

        legal_moves = list(rules.legal_moves(state))
        if not legal_moves:
            raise ValueError("Keine legalen Zuege verfuegbar.")

        scored_moves = [(self._score_move(state, move, rules), move) for move in legal_moves]
        best_score = max(score for score, _ in scored_moves)
        best_moves = [move for score, move in scored_moves if score == best_score]

        if len(best_moves) == 1:
            return best_moves[0]

        if self.random_tiebreak:
            return self._rng.choice(best_moves)

        return min(best_moves, key=_move_sort_key)

    def _score_move(self, state: MillState, move: Move, rules: MillRules) -> float:
        player = state.to_move
        opponent = other_player(player)
        next_state = rules.apply_move(state, move)

        if rules.is_terminal(next_state):
            winner = rules.winner(next_state)
            if winner == player:
                return 1_000_000.0
            if winner is None:
                return 0.0
            return -1_000_000.0

        score = 0.0

        if move.capture is not None:
            score += 5_000.0

        piece_delta = count_pieces(next_state.board, player) - count_pieces(next_state.board, opponent)
        score += piece_delta * 300.0

        placed_delta = next_state.placed.get(player, 0) - next_state.placed.get(opponent, 0)
        score += placed_delta * 20.0

        opponent_moves = list(rules.legal_moves(next_state))
        score -= len(opponent_moves) * 8.0

        opponent_capture_options = sum(1 for opp_move in opponent_moves if opp_move.capture is not None)
        score -= opponent_capture_options * 600.0

        if self._has_immediate_winning_reply(next_state, rules, opponent):
            score -= 50_000.0

        return score

    @staticmethod
    def _has_immediate_winning_reply(state: MillState, rules: MillRules, player: Player) -> bool:
        if state.to_move != player:
            return False

        for move in rules.legal_moves(state):
            reply_state = rules.apply_move(state, move)
            if rules.is_terminal(reply_state) and rules.winner(reply_state) == player:
                return True

        return False


@dataclass(slots=True)
class _TTEntry:
    depth: int
    value: float
    bound: Literal["exact", "lower", "upper"]
    best_move: Move | None


@dataclass(slots=True)
class AlphaBetaMillAI:
    """Tiefenbegrenzte Alpha-Beta-KI mit einfachem Transposition-Cache."""

    depth: int = 3
    random_tiebreak: bool = True
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)
    _tt: dict[tuple[str, int, int, tuple[str, ...]], _TTEntry] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError("depth muss groesser als 0 sein.")
        self._rng = random.Random(self.seed)
        self._tt = {}

    def choose_move(self, state: MillState, rules: MillRules, move_history: Sequence[Move]) -> Move:
        del move_history  # Die Suche benoetigt nur den aktuellen Zustand.

        legal_moves = list(rules.legal_moves(state))
        if not legal_moves:
            raise ValueError("Keine legalen Zuege verfuegbar.")

        self._tt.clear()
        ordered_moves = sorted(legal_moves, key=lambda move: (move.capture is None, _move_sort_key(move)))

        best_score = -_INF
        best_moves: list[Move] = []
        alpha = -_INF
        beta = _INF

        for move in ordered_moves:
            next_state = rules.apply_move(state, move)
            score = -self._negamax(next_state, rules, self.depth - 1, -beta, -alpha)

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

            alpha = max(alpha, best_score)

        if len(best_moves) == 1:
            return best_moves[0]

        if self.random_tiebreak:
            return self._rng.choice(best_moves)

        return min(best_moves, key=_move_sort_key)

    def _negamax(
        self,
        state: MillState,
        rules: MillRules,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        if rules.is_terminal(state):
            winner = rules.winner(state)
            if winner is None:
                return 0.0
            if winner == state.to_move:
                return _WIN_SCORE + depth
            return -_WIN_SCORE - depth

        if depth == 0:
            return self._evaluate_state(state, rules, state.to_move)

        key = (state_signature(state), depth, state.plies_without_capture, state.position_history)
        alpha_start = alpha
        beta_start = beta
        cached = self._tt.get(key)
        tt_best_move: Move | None = None

        if cached is not None:
            tt_best_move = cached.best_move
            if cached.bound == "exact":
                return cached.value
            if cached.bound == "lower":
                alpha = max(alpha, cached.value)
            else:
                beta = min(beta, cached.value)
            if alpha >= beta:
                return cached.value

        legal_moves = list(rules.legal_moves(state))
        if not legal_moves:
            return -_WIN_SCORE - depth

        ordered_moves = sorted(legal_moves, key=lambda move: (move.capture is None, _move_sort_key(move)))
        if tt_best_move is not None and tt_best_move in ordered_moves:
            ordered_moves.remove(tt_best_move)
            ordered_moves.insert(0, tt_best_move)

        best_value = -_INF
        best_move: Move | None = None

        for move in ordered_moves:
            child = rules.apply_move(state, move)
            score = -self._negamax(child, rules, depth - 1, -beta, -alpha)
            if score > best_value:
                best_value = score
                best_move = move
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break

        bound: Literal["exact", "lower", "upper"] = "exact"
        if best_value <= alpha_start:
            bound = "upper"
        elif best_value >= beta_start:
            bound = "lower"
        self._tt[key] = _TTEntry(depth=depth, value=best_value, bound=bound, best_move=best_move)
        return best_value

    def _evaluate_state(self, state: MillState, rules: MillRules, player: Player) -> float:
        opponent = other_player(player)
        board = state.board

        piece_delta = count_pieces(board, player) - count_pieces(board, opponent)
        placed_delta = state.placed.get(player, 0) - state.placed.get(opponent, 0)
        mill_delta = self._count_closed_mills(board, player) - self._count_closed_mills(board, opponent)
        open_mill_delta = self._count_open_mills(board, player) - self._count_open_mills(board, opponent)
        blocked_delta = self._count_blocked_pieces(board, opponent) - self._count_blocked_pieces(board, player)
        mobility_delta = self._mobility(state, rules, player) - self._mobility(state, rules, opponent)

        return (
            piece_delta * 1_500.0
            + placed_delta * 40.0
            + mill_delta * 220.0
            + open_mill_delta * 110.0
            + blocked_delta * 35.0
            + mobility_delta * 12.0
        )

    @staticmethod
    def _count_closed_mills(board: dict[str, Player | None], player: Player) -> int:
        return sum(1 for mill in MILLS if all(board[pos] == player for pos in mill))

    @staticmethod
    def _count_open_mills(board: dict[str, Player | None], player: Player) -> int:
        count = 0
        for mill in MILLS:
            owners = [board[pos] for pos in mill]
            if owners.count(player) == 2 and owners.count(None) == 1:
                count += 1
        return count

    @staticmethod
    def _count_blocked_pieces(board: dict[str, Player | None], player: Player) -> int:
        blocked = 0
        for pos in BOARD_LABELS:
            if board[pos] != player:
                continue
            if all(board[n] is not None for n in ADJACENT[pos]):
                blocked += 1
        return blocked

    @staticmethod
    def _with_to_move(state: MillState, player: Player) -> MillState:
        if state.to_move == player:
            return state
        return MillState(
            board=state.board,
            to_move=player,
            placed=state.placed,
            plies_without_capture=state.plies_without_capture,
            position_history=state.position_history,
        )

    def _mobility(self, state: MillState, rules: MillRules, player: Player) -> int:
        query_state = self._with_to_move(state, player)
        return len(rules.legal_moves(query_state))


__all__ = ["AlphaBetaMillAI", "HeuristicMillAI"]
