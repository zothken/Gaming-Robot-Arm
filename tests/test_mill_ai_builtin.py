import unittest

from gaming_robot_arm.games.common.interfaces import Move
from gaming_robot_arm.games.mill import AlphaBetaMillAI, HeuristicMillAI, MillRules
from gaming_robot_arm.games.mill.ai.builtin import (
    _count_double_mills,
    _evaluate_state_for_player,
    _future_mobility,
    _phase_weight_profile,
)
from gaming_robot_arm.games.mill.core.board import BOARD_LABELS
from gaming_robot_arm.games.mill.core.state import MillState


def _board(**pieces: str) -> dict[str, str | None]:
    board: dict[str, str | None] = {label: None for label in BOARD_LABELS}
    for label, owner in pieces.items():
        board[label] = owner
    return board


class PlacementHeuristicTest(unittest.TestCase):
    def test_alphabeta_empty_board_prefers_connector(self) -> None:
        rules = MillRules()
        ai = AlphaBetaMillAI(depth=3, random_tiebreak=False)
        move = ai.choose_move(rules.initial_state(), rules, [])
        self.assertIn(move.dst, {"B2", "B4", "B6", "B8"})

    def test_heuristic_empty_board_prefers_connector(self) -> None:
        rules = MillRules()
        ai = HeuristicMillAI(random_tiebreak=False)
        move = ai.choose_move(rules.initial_state(), rules, [])
        self.assertIn(move.dst, {"B2", "B4", "B6", "B8"})

    def test_both_ais_block_immediate_placement_mill_threat(self) -> None:
        rules = MillRules()
        state = MillState(
            board=_board(A1="B", A2="B", C2="W"),
            to_move="W",
            placed={"W": 1, "B": 2},
        )
        for ai in (HeuristicMillAI(random_tiebreak=False), AlphaBetaMillAI(depth=3, random_tiebreak=False)):
            with self.subTest(ai=type(ai).__name__):
                self.assertEqual(ai.choose_move(state, rules, []), Move("W", None, "A3"))

    def test_future_mobility_ranks_connector_over_edge_and_corner(self) -> None:
        self.assertGreater(_future_mobility(_board(B2="W"), "W"), _future_mobility(_board(A2="W"), "W"))
        self.assertGreater(_future_mobility(_board(A2="W"), "W"), _future_mobility(_board(A1="W"), "W"))


class MovementHeuristicTest(unittest.TestCase):
    def test_double_mill_count_detects_shared_piece_structure(self) -> None:
        self.assertEqual(_count_double_mills(_board(B1="W", B3="W", B5="W"), "W"), 1)
        self.assertEqual(_count_double_mills(_board(B1="W", B5="W"), "W"), 0)

    def test_both_ais_prefer_move_that_creates_double_mill_engine(self) -> None:
        rules = MillRules()
        state = MillState(
            board=_board(
                A1="W",
                A2="W",
                A4="W",
                A5="W",
                A6="W",
                B2="W",
                C2="W",
                C5="W",
                C6="W",
                A3="B",
                A8="B",
                B1="B",
                B4="B",
                B5="B",
                B8="B",
                C1="B",
                C3="B",
                C4="B",
            ),
            to_move="W",
            placed={"W": 9, "B": 9},
        )
        expected = Move("W", "B2", "B3")
        for ai in (HeuristicMillAI(random_tiebreak=False), AlphaBetaMillAI(depth=3, random_tiebreak=False)):
            with self.subTest(ai=type(ai).__name__):
                self.assertEqual(ai.choose_move(state, rules, []), expected)

    def test_movement_evaluation_rewards_state_with_more_double_mills(self) -> None:
        rules = MillRules()
        stronger = MillState(
            board=_board(
                A1="W",
                A2="W",
                A4="W",
                A5="W",
                A6="W",
                B3="W",
                C2="W",
                C5="W",
                C6="W",
                A3="B",
                A8="B",
                B1="B",
                B4="B",
                B5="B",
                B8="B",
                C1="B",
                C3="B",
                C4="B",
            ),
            to_move="W",
            placed={"W": 9, "B": 9},
        )
        weaker = MillState(
            board=_board(
                A1="W",
                A2="W",
                A4="W",
                A5="W",
                A6="W",
                B2="W",
                C2="W",
                C5="W",
                C6="W",
                A3="B",
                A8="B",
                B1="B",
                B4="B",
                B5="B",
                B8="B",
                C1="B",
                C3="B",
                C4="B",
            ),
            to_move="W",
            placed={"W": 9, "B": 9},
        )
        self.assertGreater(_evaluate_state_for_player(stronger, rules, "W"), _evaluate_state_for_player(weaker, rules, "W"))


class FlyingHeuristicTest(unittest.TestCase):
    def test_flying_profile_disables_blocked_weight(self) -> None:
        rules = MillRules()
        state = MillState(
            board=_board(A4="W", B2="W", C4="W", A6="B", A7="B", B6="B"),
            to_move="W",
            placed={"W": 9, "B": 9},
        )
        weights = _phase_weight_profile(state, rules, "W")
        self.assertEqual(weights.blocked_delta, 0.0)
        self.assertEqual(weights.protected_piece_delta, 80.0)

    def test_both_ais_choose_terminal_flying_capture(self) -> None:
        rules = MillRules()
        state = MillState(
            board=_board(A4="W", B2="W", C4="W", A6="B", A7="B", B6="B"),
            to_move="W",
            placed={"W": 9, "B": 9},
        )
        for ai in (HeuristicMillAI(random_tiebreak=False), AlphaBetaMillAI(depth=3, random_tiebreak=False)):
            with self.subTest(ai=type(ai).__name__):
                move = ai.choose_move(state, rules, [])
                self.assertEqual((move.src, move.dst), ("B2", "B4"))
                self.assertIn(move.capture, {"A6", "A7", "B6"})

    def test_both_ais_avoid_unsafe_flying_closure(self) -> None:
        rules = MillRules()
        state = MillState(
            board=_board(A4="W", B2="W", C4="W", A6="B", A7="B", B6="B", C1="B"),
            to_move="W",
            placed={"W": 9, "B": 9},
        )
        expected = Move("W", "B2", "B4", capture="A6")
        for ai in (HeuristicMillAI(random_tiebreak=False), AlphaBetaMillAI(depth=3, random_tiebreak=False)):
            with self.subTest(ai=type(ai).__name__):
                self.assertEqual(ai.choose_move(state, rules, []), expected)


if __name__ == "__main__":
    unittest.main()
