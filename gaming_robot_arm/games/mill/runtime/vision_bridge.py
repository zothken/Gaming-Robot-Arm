"""Vision-Bridge fuer spielbare Muehle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from gaming_robot_arm.config import CAMERA_INDEX
from gaming_robot_arm.games.common.interfaces import Move, Player
from gaming_robot_arm.games.mill.core.board import BOARD_LABELS
from gaming_robot_arm.games.mill.core.rules import MillRules
from gaming_robot_arm.games.mill.core.state import MillState
from gaming_robot_arm.utils.logger import logger

if TYPE_CHECKING:
    from gaming_robot_arm.vision.figure_detector import BoardCoordSource
    from gaming_robot_arm.vision.recording import RecordingSession


@dataclass(slots=True)
class _LiveVisionSession:
    """Offener Kamerakanal ohne Video-Writer fuer Vision-Scans."""

    camera: Any

    def read(self) -> Any:
        ok, frame = self.camera.read()
        if not ok:
            raise RuntimeError("Konnte Kameraframe nicht lesen.")
        return frame

    def write(self, _frame: Any) -> None:
        return


@dataclass(slots=True)
class MillVisionBridge:
    """Liest stabile Brettbelegung ueber die bestehende Figuren-Detektion."""

    board_pixels: dict[str, tuple[float, float]]
    labels_order: list[str]
    attempts: int = 6
    debug_assignments: bool = False
    camera_index: int = CAMERA_INDEX
    board_source: BoardCoordSource = "calibration"

    @classmethod
    def from_calibration(
        cls,
        *,
        attempts: int = 6,
        debug_assignments: bool = False,
        camera_index: int = CAMERA_INDEX,
    ) -> "MillVisionBridge":
        from gaming_robot_arm.calibration.calibration import load_board_pixels

        board_pixels = load_board_pixels()
        labels_order = sorted(board_pixels.keys())
        return cls(
            board_pixels={label: (float(u), float(v)) for label, (u, v) in board_pixels.items()},
            labels_order=labels_order,
            attempts=max(1, attempts),
            debug_assignments=debug_assignments,
            camera_index=camera_index,
            board_source="calibration",
        )

    @classmethod
    def for_live_session(
        cls,
        *,
        attempts: int = 6,
        debug_assignments: bool = False,
        camera_index: int = CAMERA_INDEX,
    ) -> "MillVisionBridge":
        return cls(
            board_pixels={},
            labels_order=list(BOARD_LABELS),
            attempts=max(1, attempts),
            debug_assignments=debug_assignments,
            camera_index=camera_index,
            board_source="calibration",
        )

    @staticmethod
    def _positions_to_board_pixels(positions: Sequence[tuple[int, int]]) -> dict[str, tuple[float, float]]:
        if len(positions) != len(BOARD_LABELS):
            raise ValueError(
                f"Ungueltige Anzahl Brettpositionen: {len(positions)} (erwartet: {len(BOARD_LABELS)})."
            )
        return {
            label: (float(x), float(y))
            for label, (x, y) in zip(BOARD_LABELS, positions)
        }

    def calibrate_temporary_board_pixels(
        self,
        *,
        session: RecordingSession | _LiveVisionSession,
        attempts: int = 6,
    ) -> None:
        from gaming_robot_arm.vision.mill_board_detector import detect_board_positions

        max_attempts = max(1, int(attempts))
        for attempt_idx in range(max_attempts):
            try:
                frame = session.read()
            except Exception as exc:
                logger.warning(
                    "Konnte keinen Kameraframe fuer temporaere Brettkalibrierung lesen (Versuch %s/%s): %s",
                    attempt_idx + 1,
                    max_attempts,
                    exc,
                )
                continue

            session.write(frame)
            try:
                positions, _annotated = detect_board_positions(frame, debug=False, return_bw=False)
            except Exception:
                logger.exception(
                    "Mill-Board-Detektor fehlgeschlagen (Versuch %s/%s).",
                    attempt_idx + 1,
                    max_attempts,
                )
                continue

            if len(positions) != len(BOARD_LABELS):
                logger.debug(
                    "Temporare Brettkalibrierung: %s/%s Positionen erkannt (Versuch %s/%s).",
                    len(positions),
                    len(BOARD_LABELS),
                    attempt_idx + 1,
                    max_attempts,
                )
                continue

            self.board_pixels = self._positions_to_board_pixels(positions)
            self.labels_order = list(BOARD_LABELS)
            self.board_source = "calibration"
            logger.info("Temporare Brettkalibrierung erstellt (%s Positionen).", len(self.board_pixels))
            return

        raise RuntimeError("Temporare Brettkalibrierung fehlgeschlagen.")

    def observe_board(
        self,
        *,
        session: RecordingSession | _LiveVisionSession | None = None,
    ) -> dict[str, Player | None]:
        from gaming_robot_arm.vision.figure_detector import detect_board_assignments

        observed: dict[str, Player | None] = {label: None for label in BOARD_LABELS}
        if len(self.board_pixels) != len(BOARD_LABELS):
            logger.warning("Vision-Scan uebersprungen: keine gueltige Brettkalibrierung geladen.")
            return observed

        assignments = detect_board_assignments(
            self.board_pixels,
            attempts=self.attempts,
            session=session,
            labels_order=self.labels_order,
            debug_assignments=self.debug_assignments,
            camera_index=self.camera_index,
            board_source=self.board_source,
        )

        for item in assignments:
            label = str(item.get("label", "")).upper()
            color = str(item.get("color", "")).lower()
            if label not in observed:
                continue
            if color in {"weiss", "white"}:
                observed[label] = "W"
            elif color in {"schwarz", "black"}:
                observed[label] = "B"
        return observed


def infer_moves_from_observation(
    *,
    rules: MillRules,
    state: MillState,
    legal_moves: Sequence[Move],
    observed_board: dict[str, Player | None],
) -> list[Move]:
    matches: list[Move] = []
    for move in legal_moves:
        candidate = rules.apply_move(state, move)
        if all(candidate.board[label] == observed_board.get(label) for label in BOARD_LABELS):
            matches.append(move)
    return matches


__all__ = ["MillVisionBridge", "infer_moves_from_observation"]
