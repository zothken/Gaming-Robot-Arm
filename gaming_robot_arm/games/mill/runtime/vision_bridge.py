"""Vision-Bridge fuer spielbare Muehle."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Sequence

import cv2
import numpy as np

from gaming_robot_arm.config import CAMERA_INDEX
from gaming_robot_arm.games.common.interfaces import Move, Player
from gaming_robot_arm.games.mill.core.board import BOARD_LABELS
from gaming_robot_arm.games.mill.core.rules import MillRules
from gaming_robot_arm.games.mill.core.state import MillState
from gaming_robot_arm.utils.logger import logger

if TYPE_CHECKING:
    from gaming_robot_arm.vision.figure_detector import BoardCoordSource
    from gaming_robot_arm.vision.recording import RecordingSession


VisionTriggerMode = Literal["manual", "auto"]

AUTO_BASELINE_TIMEOUT_S = 20.0
AUTO_MOVE_TIMEOUT_S = 60.0
AUTO_BASELINE_CONFIRMATIONS = 3
AUTO_MOVE_CONFIRMATIONS = 3
AUTO_MOTION_PADDING_PX = 60
AUTO_MOTION_THRESHOLD = 20
AUTO_MOTION_RATIO = 0.015
AUTO_QUIET_FRAMES = 5


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
class AutoMoveResult:
    """Ergebnis der automatischen Vision-Zugerkennung."""

    move: Move | None
    reason: str


@dataclass(slots=True)
class _VisionAutoTriggerStateMachine:
    """Testbare Zustandsmaschine fuer konservatives Auto-Triggering."""

    baseline_confirmations_required: int = AUTO_BASELINE_CONFIRMATIONS
    candidate_confirmations_required: int = AUTO_MOVE_CONFIRMATIONS
    state: Literal["acquiring_baseline", "armed", "candidate_pending"] = "acquiring_baseline"
    baseline_confirmations: int = 0
    candidate_confirmations: int = 0
    pending_move: Move | None = None

    def update(
        self,
        *,
        expected_board: Mapping[str, Player | None],
        observed_board: Mapping[str, Player | None] | None,
        quiet: bool,
        matches: Sequence[Move],
    ) -> Move | None:
        if not quiet or observed_board is None:
            if self.state == "acquiring_baseline":
                self.baseline_confirmations = 0
            else:
                self.state = "armed"
            self.candidate_confirmations = 0
            self.pending_move = None
            return None

        if all(expected_board.get(label) == observed_board.get(label) for label in BOARD_LABELS):
            self.pending_move = None
            self.candidate_confirmations = 0
            if self.state == "acquiring_baseline":
                self.baseline_confirmations += 1
                if self.baseline_confirmations >= self.baseline_confirmations_required:
                    self.state = "armed"
            else:
                self.state = "armed"
            return None

        self.baseline_confirmations = 0
        if self.state == "acquiring_baseline":
            return None

        if len(matches) != 1:
            self.state = "armed"
            self.pending_move = None
            self.candidate_confirmations = 0
            return None

        move = matches[0]
        if self.pending_move == move:
            self.candidate_confirmations += 1
        else:
            self.pending_move = move
            self.candidate_confirmations = 1

        self.state = "candidate_pending"
        if self.candidate_confirmations < self.candidate_confirmations_required:
            return None

        accepted = move
        self.state = "armed"
        self.pending_move = None
        self.candidate_confirmations = 0
        return accepted


@dataclass(slots=True)
class _BoardMotionGate:
    """Blockiert Auto-Triggering, solange im Brettbereich Bewegung sichtbar ist."""

    board_pixels: Mapping[str, tuple[float, float]]
    padding_px: int = AUTO_MOTION_PADDING_PX
    pixel_threshold: int = AUTO_MOTION_THRESHOLD
    motion_ratio_threshold: float = AUTO_MOTION_RATIO
    quiet_frames_required: int = AUTO_QUIET_FRAMES
    _previous_gray: np.ndarray | None = None
    _quiet_frames: int = 0

    def _roi_bounds(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        if not self.board_pixels:
            return None

        height, width = frame.shape[:2]
        xs = [coord[0] for coord in self.board_pixels.values()]
        ys = [coord[1] for coord in self.board_pixels.values()]
        min_x = max(0, int(np.floor(min(xs))) - self.padding_px)
        min_y = max(0, int(np.floor(min(ys))) - self.padding_px)
        max_x = min(width, int(np.ceil(max(xs))) + self.padding_px)
        max_y = min(height, int(np.ceil(max(ys))) + self.padding_px)
        if min_x >= max_x or min_y >= max_y:
            return None
        return min_x, min_y, max_x, max_y

    def update(self, frame: np.ndarray) -> tuple[bool, bool, float]:
        bounds = self._roi_bounds(frame)
        if bounds is None:
            return False, False, 0.0

        min_x, min_y, max_x, max_y = bounds
        roi = frame[min_y:max_y, min_x:max_x]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        previous = self._previous_gray
        self._previous_gray = gray
        if previous is None or previous.shape != gray.shape:
            self._quiet_frames = min(self._quiet_frames + 1, self.quiet_frames_required)
            return False, self._quiet_frames >= self.quiet_frames_required, 0.0

        diff = cv2.absdiff(gray, previous)
        _, thresh = cv2.threshold(diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)
        changed_ratio = float(cv2.countNonZero(thresh)) / float(max(1, thresh.size))
        motion = changed_ratio > self.motion_ratio_threshold

        if motion:
            self._quiet_frames = 0
        else:
            self._quiet_frames = min(self._quiet_frames + 1, self.quiet_frames_required)

        return motion, self._quiet_frames >= self.quiet_frames_required, changed_ratio


@dataclass(slots=True)
class MillVisionBridge:
    """Liest stabile Brettbelegung ueber die bestehende Figuren-Detektion."""

    board_pixels: dict[str, tuple[float, float]]
    labels_order: list[str]
    attempts: int = 6
    debug_assignments: bool = False
    camera_index: int = CAMERA_INDEX
    board_source: BoardCoordSource = "calibration"
    detector_backend: str = "hough"
    classifier_model_path: str = ""

    @classmethod
    def from_calibration(
        cls,
        *,
        attempts: int = 6,
        debug_assignments: bool = False,
        camera_index: int = CAMERA_INDEX,
        detector_backend: str = "hough",
        classifier_model_path: str = "",
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
            detector_backend=detector_backend,
            classifier_model_path=classifier_model_path,
        )

    @classmethod
    def for_live_session(
        cls,
        *,
        attempts: int = 6,
        debug_assignments: bool = False,
        camera_index: int = CAMERA_INDEX,
        detector_backend: str = "hough",
        classifier_model_path: str = "",
    ) -> "MillVisionBridge":
        return cls(
            board_pixels={},
            labels_order=list(BOARD_LABELS),
            attempts=max(1, attempts),
            debug_assignments=debug_assignments,
            camera_index=camera_index,
            board_source="calibration",
            detector_backend=detector_backend,
            classifier_model_path=classifier_model_path,
        )

    def _make_detector(self) -> Any:
        """Gibt den konfigurierten Detektor als Callable zurueck (oder None fuer HoughCircles)."""
        if self.detector_backend != "classifier":
            return None
        from gaming_robot_arm.vision.classifier_figure_detector import ClassifierFigureDetector

        return ClassifierFigureDetector(self.classifier_model_path)

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
            detector=self._make_detector(),
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

    def observe_move_automatically(
        self,
        *,
        rules: MillRules,
        state: MillState,
        legal_moves: Sequence[Move],
        session: RecordingSession | _LiveVisionSession | None,
        status_callback: Callable[[str], None] | None = None,
        baseline_timeout_s: float = AUTO_BASELINE_TIMEOUT_S,
        move_timeout_s: float = AUTO_MOVE_TIMEOUT_S,
    ) -> AutoMoveResult:
        from gaming_robot_arm.vision.figure_detector import BoardAssignmentStream
        from gaming_robot_arm.vision.recording import get_effective_camera_fps

        if session is None:
            return AutoMoveResult(move=None, reason="missing_session")
        if len(self.board_pixels) != len(BOARD_LABELS):
            logger.warning("Auto-Trigger uebersprungen: keine gueltige Brettkalibrierung geladen.")
            return AutoMoveResult(move=None, reason="missing_board_pixels")

        fps = get_effective_camera_fps(session.camera)
        window_frames = max(1, int(round(fps)))
        stream = BoardAssignmentStream(
            self.board_pixels,
            labels_order=self.labels_order,
            board_source=self.board_source,
            window_frames=window_frames,
            min_samples=window_frames,
            min_ratio=0.5,
            debug_assignments=self.debug_assignments,
            detector=self._make_detector(),
        )
        motion_gate = _BoardMotionGate(self.board_pixels)
        trigger = _VisionAutoTriggerStateMachine()
        baseline_deadline = perf_counter() + max(0.1, float(baseline_timeout_s))
        move_deadline: float | None = None
        last_status: str | None = None

        def emit_status(message: str) -> None:
            nonlocal last_status
            if status_callback is None or message == last_status:
                return
            status_callback(message)
            last_status = message

        emit_status("Vision kalibriert / warte auf ruhiges Brett")

        while True:
            now = perf_counter()
            if trigger.state == "acquiring_baseline":
                if now > baseline_deadline:
                    return AutoMoveResult(move=None, reason="baseline_timeout")
            elif move_deadline is not None and now > move_deadline:
                return AutoMoveResult(move=None, reason="move_timeout")

            try:
                frame = session.read()
            except Exception as exc:
                logger.warning("Konnte keinen Kameraframe fuer Auto-Trigger lesen: %s", exc)
                return AutoMoveResult(move=None, reason="frame_read_failed")

            session.write(frame)
            observation = stream.process_frame(frame)
            motion, quiet, changed_ratio = motion_gate.update(frame)
            if motion:
                logger.debug("Vision-Motion-Gate aktiv (veraendert=%.4f).", changed_ratio)

            previous_state = trigger.state
            matches: Sequence[Move] = []
            if observation.ready:
                matches = infer_moves_from_observation(
                    rules=rules,
                    state=state,
                    legal_moves=legal_moves,
                    observed_board=observation.stable_board,
                )

            move = trigger.update(
                expected_board=state.board,
                observed_board=observation.stable_board if observation.ready else None,
                quiet=quiet,
                matches=matches,
            )

            if previous_state == "acquiring_baseline" and trigger.state == "armed":
                move_deadline = perf_counter() + max(0.1, float(move_timeout_s))
                emit_status("Vision bereit, Zug ausfuehren")

            if move is not None:
                return AutoMoveResult(move=move, reason="ok")


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


__all__ = [
    "AutoMoveResult",
    "MillVisionBridge",
    "VisionTriggerMode",
    "_BoardMotionGate",
    "_LiveVisionSession",
    "_VisionAutoTriggerStateMachine",
    "infer_moves_from_observation",
]
