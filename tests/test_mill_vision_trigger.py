import sys
import types
import unittest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from gaming_robot_arm.games.common.interfaces import Move
from gaming_robot_arm.games.mill.core.board import BOARD_LABELS
from gaming_robot_arm.games.mill.core.rules import MillRules
from gaming_robot_arm.games.mill.core.state import MillState
from gaming_robot_arm.games.mill.runtime.game_loop import build_parser
from gaming_robot_arm.games.mill.runtime.vision_bridge import (
    _VisionAutoTriggerStateMachine,
    infer_moves_from_observation,
)


def _board(**pieces: str) -> dict[str, str | None]:
    board: dict[str, str | None] = {label: None for label in BOARD_LABELS}
    for label, owner in pieces.items():
        board[label] = owner
    return board


class VisionTriggerParserTest(unittest.TestCase):
    def test_parser_defaults_to_auto_trigger(self) -> None:
        args = build_parser().parse_args(["--human-input", "vision"])
        self.assertEqual(args.mill_vision_trigger, "auto")

    def test_parser_accepts_manual_trigger(self) -> None:
        args = build_parser().parse_args(["--human-input", "vision", "--vision-trigger", "manual"])
        self.assertEqual(args.mill_vision_trigger, "manual")


class AutoTriggerStateMachineTest(unittest.TestCase):
    def test_baseline_must_be_confirmed_before_move_can_fire(self) -> None:
        trigger = _VisionAutoTriggerStateMachine()
        expected_board = _board()
        candidate_board = _board(A1="W")
        candidate_move = Move("W", None, "A1")

        for _ in range(3):
            self.assertIsNone(
                trigger.update(
                    expected_board=expected_board,
                    observed_board=candidate_board,
                    quiet=True,
                    matches=[candidate_move],
                )
            )
        self.assertEqual(trigger.state, "acquiring_baseline")

        for _ in range(2):
            self.assertIsNone(
                trigger.update(
                    expected_board=expected_board,
                    observed_board=expected_board,
                    quiet=True,
                    matches=[],
                )
            )
            self.assertEqual(trigger.state, "acquiring_baseline")

        self.assertIsNone(
            trigger.update(
                expected_board=expected_board,
                observed_board=expected_board,
                quiet=True,
                matches=[],
            )
        )
        self.assertEqual(trigger.state, "armed")

        self.assertIsNone(
            trigger.update(
                expected_board=expected_board,
                observed_board=candidate_board,
                quiet=True,
                matches=[candidate_move],
            )
        )
        self.assertIsNone(
            trigger.update(
                expected_board=expected_board,
                observed_board=candidate_board,
                quiet=True,
                matches=[candidate_move],
            )
        )
        accepted = trigger.update(
            expected_board=expected_board,
            observed_board=candidate_board,
            quiet=True,
            matches=[candidate_move],
        )
        self.assertEqual(accepted, candidate_move)

    def test_ambiguous_matches_never_auto_accept(self) -> None:
        trigger = _VisionAutoTriggerStateMachine()
        expected_board = _board()
        candidate_board = _board(A1="W")
        candidate_a = Move("W", None, "A1")
        candidate_b = Move("W", None, "A2")

        for _ in range(3):
            trigger.update(
                expected_board=expected_board,
                observed_board=expected_board,
                quiet=True,
                matches=[],
            )
        self.assertEqual(trigger.state, "armed")

        for _ in range(5):
            accepted = trigger.update(
                expected_board=expected_board,
                observed_board=candidate_board,
                quiet=True,
                matches=[candidate_a, candidate_b],
            )
            self.assertIsNone(accepted)
            self.assertEqual(trigger.state, "armed")


class InferMovesFromObservationTest(unittest.TestCase):
    def test_capture_board_state_maps_to_single_legal_move(self) -> None:
        rules = MillRules()
        state = MillState(
            board=_board(A1="W", A2="W", B1="B"),
            to_move="W",
            placed={"W": 2, "B": 1},
        )
        legal_moves = list(rules.legal_moves(state))
        capture_move = next(move for move in legal_moves if move.dst == "A3" and move.capture == "B1")

        observed_board = rules.apply_move(state, capture_move).board
        matches = infer_moves_from_observation(
            rules=rules,
            state=state,
            legal_moves=legal_moves,
            observed_board=observed_board,
        )

        self.assertEqual(matches, [capture_move])


if __name__ == "__main__":
    unittest.main()
