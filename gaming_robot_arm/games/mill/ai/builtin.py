"""Interne Muehle-KIs, die keine externe Engine benoetigen."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Literal, Sequence

from gaming_robot_arm.games.common.interfaces import Move, Player

from ..core.board import ADJACENT, BOARD_LABELS, MILLS, MILLS_BY_POSITION
from ..core.constants import PIECES_PER_PLAYER
from ..core.rules import (
    MillRules,
    count_pieces,
    is_mill,
    other_player,
    phase_for_player,
    state_signature,
)
from ..core.state import MillState


_WIN_SCORE = 1_000_000.0
_INF = float("inf")


def _move_sort_key(move: Move) -> tuple[str, str, str]:
    """Stabiler Tie-Break-Schluessel fuer deterministisches KI-Verhalten."""

    return (move.src or "", move.dst, move.capture or "")


@dataclass(frozen=True, slots=True)
class _WeightProfile:
    piece_delta: float = 0.0
    closed_mill_delta: float = 0.0
    open_mill_delta: float = 0.0
    double_mill_delta: float = 0.0
    future_mobility_delta: float = 0.0
    legal_mobility_delta: float = 0.0
    blocked_delta: float = 0.0
    protected_piece_delta: float = 0.0


@dataclass(frozen=True, slots=True)
class _EvalFeatures:
    piece_delta: int = 0
    closed_mill_delta: int = 0
    open_mill_delta: int = 0
    double_mill_delta: int = 0
    future_mobility_delta: int = 0
    legal_mobility_delta: int = 0
    blocked_delta: int = 0
    protected_piece_delta: int = 0


_PLACEMENT_EARLY_WEIGHTS = _WeightProfile(
    piece_delta=900.0,
    closed_mill_delta=70.0,
    open_mill_delta=140.0,
    double_mill_delta=180.0,
    future_mobility_delta=55.0,
    blocked_delta=20.0,
)
_PLACEMENT_LATE_WEIGHTS = _WeightProfile(
    piece_delta=1200.0,
    closed_mill_delta=150.0,
    open_mill_delta=130.0,
    double_mill_delta=150.0,
    future_mobility_delta=35.0,
    blocked_delta=25.0,
)
_MOVEMENT_WEIGHTS = _WeightProfile(
    piece_delta=1500.0,
    closed_mill_delta=190.0,
    open_mill_delta=90.0,
    double_mill_delta=180.0,
    legal_mobility_delta=18.0,
    blocked_delta=50.0,
)
_FLYING_WEIGHTS = _WeightProfile(
    piece_delta=1800.0,
    closed_mill_delta=170.0,
    open_mill_delta=130.0,
    double_mill_delta=220.0,
    legal_mobility_delta=8.0,
    protected_piece_delta=80.0,
    blocked_delta=0.0,
)


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


def _placement_progress(state: MillState) -> float:
    total_placed = state.placed.get("W", 0) + state.placed.get("B", 0)
    max_total = 2 * PIECES_PER_PLAYER
    return min(1.0, max(0.0, total_placed / float(max_total)))


def _lerp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


def _interpolate_weight_profile(start: _WeightProfile, end: _WeightProfile, progress: float) -> _WeightProfile:
    return _WeightProfile(
        piece_delta=_lerp(start.piece_delta, end.piece_delta, progress),
        closed_mill_delta=_lerp(start.closed_mill_delta, end.closed_mill_delta, progress),
        open_mill_delta=_lerp(start.open_mill_delta, end.open_mill_delta, progress),
        double_mill_delta=_lerp(start.double_mill_delta, end.double_mill_delta, progress),
        future_mobility_delta=_lerp(start.future_mobility_delta, end.future_mobility_delta, progress),
        legal_mobility_delta=_lerp(start.legal_mobility_delta, end.legal_mobility_delta, progress),
        blocked_delta=_lerp(start.blocked_delta, end.blocked_delta, progress),
        protected_piece_delta=_lerp(start.protected_piece_delta, end.protected_piece_delta, progress),
    )


def _phase_weight_profile(state: MillState, rules: MillRules, player: Player) -> _WeightProfile:
    phase = phase_for_player(state, player, settings=rules.settings)
    if phase == "placement":
        return _interpolate_weight_profile(
            _PLACEMENT_EARLY_WEIGHTS,
            _PLACEMENT_LATE_WEIGHTS,
            _placement_progress(state),
        )
    if phase == "movement":
        return _MOVEMENT_WEIGHTS
    return _FLYING_WEIGHTS


def _count_closed_mills(board: dict[str, Player | None], player: Player) -> int:
    return sum(1 for mill in MILLS if all(board[pos] == player for pos in mill))


def _count_open_mills(board: dict[str, Player | None], player: Player) -> int:
    count = 0
    for mill in MILLS:
        owners = [board[pos] for pos in mill]
        if owners.count(player) == 2 and owners.count(None) == 1:
            count += 1
    return count


def _count_double_mills(board: dict[str, Player | None], player: Player) -> int:
    count = 0
    for position in BOARD_LABELS:
        if board[position] != player:
            continue
        open_mills = 0
        for mill in MILLS_BY_POSITION[position]:
            owners = [board[pos] for pos in mill]
            if owners.count(player) == 2 and owners.count(None) == 1:
                open_mills += 1
        if open_mills >= 2:
            count += 1
    return count


def _count_blocked_pieces(board: dict[str, Player | None], player: Player) -> int:
    blocked = 0
    for pos in BOARD_LABELS:
        if board[pos] != player:
            continue
        if all(board[n] is not None for n in ADJACENT[pos]):
            blocked += 1
    return blocked


def _count_protected_pieces(board: dict[str, Player | None], player: Player) -> int:
    return sum(1 for pos in BOARD_LABELS if board[pos] == player and is_mill(board, player, pos))


def _future_mobility(board: dict[str, Player | None], player: Player) -> int:
    total = 0
    for pos in BOARD_LABELS:
        if board[pos] != player:
            continue
        total += sum(1 for neighbor in ADJACENT[pos] if board[neighbor] is None)
    return total


def _legal_mobility(state: MillState, rules: MillRules, player: Player) -> int:
    return len(rules.legal_moves(_with_to_move(state, player)))


def _collect_eval_features(state: MillState, rules: MillRules, player: Player) -> _EvalFeatures:
    opponent = other_player(player)
    board = state.board
    return _EvalFeatures(
        piece_delta=count_pieces(board, player) - count_pieces(board, opponent),
        closed_mill_delta=_count_closed_mills(board, player) - _count_closed_mills(board, opponent),
        open_mill_delta=_count_open_mills(board, player) - _count_open_mills(board, opponent),
        double_mill_delta=_count_double_mills(board, player) - _count_double_mills(board, opponent),
        future_mobility_delta=_future_mobility(board, player) - _future_mobility(board, opponent),
        legal_mobility_delta=_legal_mobility(state, rules, player) - _legal_mobility(state, rules, opponent),
        blocked_delta=_count_blocked_pieces(board, opponent) - _count_blocked_pieces(board, player),
        protected_piece_delta=_count_protected_pieces(board, player) - _count_protected_pieces(board, opponent),
    )


def _evaluate_state_for_player(state: MillState, rules: MillRules, player: Player) -> float:
    weights = _phase_weight_profile(state, rules, player)
    features = _collect_eval_features(state, rules, player)
    return (
        features.piece_delta * weights.piece_delta
        + features.closed_mill_delta * weights.closed_mill_delta
        + features.open_mill_delta * weights.open_mill_delta
        + features.double_mill_delta * weights.double_mill_delta
        + features.future_mobility_delta * weights.future_mobility_delta
        + features.legal_mobility_delta * weights.legal_mobility_delta
        + features.blocked_delta * weights.blocked_delta
        + features.protected_piece_delta * weights.protected_piece_delta
    )


def _has_immediate_winning_reply(state: MillState, rules: MillRules, player: Player) -> bool:
    if state.to_move != player:
        return False

    for move in rules.legal_moves(state):
        reply_state = rules.apply_move(state, move)
        if rules.is_terminal(reply_state) and rules.winner(reply_state) == player:
            return True

    return False


def _prefer_safe_flying_mill_closures(
    state: MillState,
    rules: MillRules,
    legal_moves: list[Move],
) -> list[Move]:
    if phase_for_player(state, state.to_move, settings=rules.settings) != "flying":
        return legal_moves

    closing_moves = [move for move in legal_moves if move.capture is not None]
    if not closing_moves:
        return legal_moves

    safe_closing_moves: list[Move] = []
    for move in closing_moves:
        next_state = rules.apply_move(state, move)
        opponent = next_state.to_move
        if not _has_immediate_winning_reply(next_state, rules, opponent):
            safe_closing_moves.append(move)

    return safe_closing_moves or legal_moves


def _move_order_key(state: MillState, rules: MillRules, move: Move) -> tuple[object, ...]:
    player = state.to_move
    opponent = other_player(player)
    phase = phase_for_player(state, player, settings=rules.settings)
    next_state = rules.apply_move(state, move)
    board = next_state.board
    terminal_win = rules.is_terminal(next_state) and rules.winner(next_state) == player
    unsafe_flying_closure = (
        phase == "flying"
        and move.capture is not None
        and _has_immediate_winning_reply(next_state, rules, opponent)
    )
    eval_score = _evaluate_state_for_player(next_state, rules, player)
    double_mills = _count_double_mills(board, player)
    blocked_opponent = _count_blocked_pieces(board, opponent)
    legal_mobility = _legal_mobility(next_state, rules, player)
    future_mobility = _future_mobility(board, player)
    open_mills = _count_open_mills(board, player)
    protected = _count_protected_pieces(board, player)

    if phase == "placement":
        return (
            0 if terminal_win else 1,
            move.capture is None,
            -double_mills,
            -future_mobility,
            -open_mills,
            -eval_score,
            _move_sort_key(move),
        )

    if phase == "movement":
        return (
            0 if terminal_win else 1,
            move.capture is None,
            -double_mills,
            -blocked_opponent,
            -legal_mobility,
            -eval_score,
            _move_sort_key(move),
        )

    return (
        0 if terminal_win else 1,
        unsafe_flying_closure,
        move.capture is None,
        -double_mills,
        -protected,
        -open_mills,
        -legal_mobility,
        -eval_score,
        _move_sort_key(move),
    )


@dataclass(slots=True)
class HeuristicMillAI:
    """Einfache einstufige Heuristik-KI."""

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

        candidate_moves = _prefer_safe_flying_mill_closures(state, rules, legal_moves)
        scored_moves = [(self._score_move(state, move, rules), move) for move in candidate_moves]
        best_score = max(score for score, _ in scored_moves)
        best_moves = [move for score, move in scored_moves if score == best_score]

        if len(best_moves) == 1:
            return best_moves[0]

        if self.random_tiebreak:
            return self._rng.choice(best_moves)

        return min(best_moves, key=lambda move: _move_order_key(state, rules, move))

    def _score_move(self, state: MillState, move: Move, rules: MillRules) -> float:
        player = state.to_move
        opponent = other_player(player)
        phase = phase_for_player(state, player, settings=rules.settings)
        next_state = rules.apply_move(state, move)

        if rules.is_terminal(next_state):
            winner = rules.winner(next_state)
            if winner == player:
                return _WIN_SCORE
            if winner is None:
                return 0.0
            return -_WIN_SCORE

        score = _evaluate_state_for_player(next_state, rules, player)

        if move.capture is not None:
            score += 2_000.0 if phase == "placement" else 5_000.0

        if phase == "flying" and move.capture is not None and _has_immediate_winning_reply(next_state, rules, opponent):
            score -= 20_000.0

        if _has_immediate_winning_reply(next_state, rules, opponent):
            score -= 50_000.0

        return score


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
        candidate_moves = _prefer_safe_flying_mill_closures(state, rules, legal_moves)
        ordered_moves = sorted(candidate_moves, key=lambda move: _move_order_key(state, rules, move))

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

        return min(best_moves, key=lambda move: _move_order_key(state, rules, move))

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

        ordered_moves = sorted(legal_moves, key=lambda move: _move_order_key(state, rules, move))
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

    @staticmethod
    def _evaluate_state(state: MillState, rules: MillRules, player: Player) -> float:
        return _evaluate_state_for_player(state, rules, player)

    @staticmethod
    def _count_closed_mills(board: dict[str, Player | None], player: Player) -> int:
        return _count_closed_mills(board, player)

    @staticmethod
    def _count_open_mills(board: dict[str, Player | None], player: Player) -> int:
        return _count_open_mills(board, player)

    @staticmethod
    def _count_double_mills(board: dict[str, Player | None], player: Player) -> int:
        return _count_double_mills(board, player)

    @staticmethod
    def _count_blocked_pieces(board: dict[str, Player | None], player: Player) -> int:
        return _count_blocked_pieces(board, player)

    @staticmethod
    def _count_protected_pieces(board: dict[str, Player | None], player: Player) -> int:
        return _count_protected_pieces(board, player)

    @staticmethod
    def _future_mobility(board: dict[str, Player | None], player: Player) -> int:
        return _future_mobility(board, player)

    @staticmethod
    def _legal_mobility(state: MillState, rules: MillRules, player: Player) -> int:
        return _legal_mobility(state, rules, player)

    @staticmethod
    def _has_immediate_winning_reply(state: MillState, rules: MillRules, player: Player) -> bool:
        return _has_immediate_winning_reply(state, rules, player)


__all__ = ["AlphaBetaMillAI", "HeuristicMillAI"]
