"""Robotik-Bridge fuer spielbare Muehle."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from gaming_robot_arm.calibration.mill_default_calibration import (
    MILL_BLACK_RESERVE_POSITIONS,
    MILL_PICK_Z,
    MILL_PLACE_Z,
    MILL_RESERVE_PICK_Z,
    MILL_WHITE_RESERVE_POSITIONS,
    get_mill_uarm_positions,
)
from gaming_robot_arm.config import REST_POS, SAFE_Z, UARM_PORT
from gaming_robot_arm.games.common.interfaces import Move, Player
from gaming_robot_arm.games.mill.core.board import BOARD_LABELS
from gaming_robot_arm.games.mill.core.constants import PIECES_PER_PLAYER
from gaming_robot_arm.games.mill.core.rules import other_player
from gaming_robot_arm.utils.logger import logger

if TYPE_CHECKING:
    from gaming_robot_arm.control.uarm_controller import UArmController


ROBOT_BOARD_MAPS = ("default", "homography")


def build_default_reserve_positions() -> dict[Player, list[tuple[float, float]]]:
    return {
        "W": list(MILL_WHITE_RESERVE_POSITIONS),
        "B": list(MILL_BLACK_RESERVE_POSITIONS),
    }


def load_robot_board_positions(source: str) -> dict[str, tuple[float, float]]:
    if source == "default":
        return get_mill_uarm_positions()

    if source != "homography":
        raise ValueError(f"Nicht unterstuetzte Quelle fuer Roboter-Brettkoordinaten: {source}")

    from gaming_robot_arm.utils.homography import img_to_robot, load_homography

    H, board_pixels = load_homography()
    if H is None or not board_pixels:
        raise RuntimeError("Homography-Brettmapping angefordert, aber Kalibrierdaten sind unvollstaendig.")

    missing_labels = [label for label in BOARD_LABELS if label not in board_pixels]
    if missing_labels:
        raise RuntimeError(f"Homography-Brettmapping hat fehlende Labels: {', '.join(missing_labels)}")

    mapped: dict[str, tuple[float, float]] = {}
    for label in BOARD_LABELS:
        u, v = board_pixels[label]
        mapped[label] = img_to_robot(H, float(u), float(v))
    return mapped


@dataclass(slots=True)
class MillRobotBridge:
    """Fuehrt Muehle-Zuege auf dem uArm mit kalibrierten Brettkoordinaten aus."""

    board_positions: dict[str, tuple[float, float]]
    port: str | None = UARM_PORT
    reserve_positions: Mapping[Player, tuple[float, float] | Sequence[tuple[float, float]]] | None = None
    reserve_pick_z: float = MILL_RESERVE_PICK_Z
    move_speed: int = 500
    _controller: UArmController | None = None
    _reserve_slots: dict[Player, list[tuple[float, float]]] = field(default_factory=dict, init=False)
    _reserve_indices: dict[Player, int] = field(default_factory=dict, init=False)
    _capture_reserve_indices: dict[Player, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._reserve_slots = {}
        reserves = self.reserve_positions or {}
        for player, value in reserves.items():
            slots: list[tuple[float, float]] = []

            if (
                isinstance(value, (tuple, list))
                and len(value) == 2
                and isinstance(value[0], (int, float))
                and isinstance(value[1], (int, float))
            ):
                x = float(value[0])
                y = float(value[1])
                slots.extend([(float(x), float(y)) for _ in range(PIECES_PER_PLAYER)])
            else:
                for point in value:
                    if not isinstance(point, (tuple, list)) or len(point) != 2:
                        raise ValueError(f"Reservepunkt fuer {player} muss aus zwei Werten bestehen.")
                    x, y = point[0], point[1]
                    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                        raise ValueError(f"Reservepunkt fuer {player} muss numerische X/Y-Werte enthalten.")
                    slots.append((float(x), float(y)))

            if slots:
                self._reserve_slots[player] = slots

        self._reserve_indices = {player: 0 for player in self._reserve_slots}
        self._capture_reserve_indices = {player: 0 for player in self._reserve_slots}

    def connect(self) -> None:
        if self._controller is not None:
            return

        from gaming_robot_arm.control import UArmController

        controller = UArmController(port=self.port)
        swift = controller.swift
        if swift is None:
            raise RuntimeError("uArm-Verbindung hergestellt, aber ohne Swift-API-Handle.")
        self._controller = controller
        logger.info("Roboter-Bridge mit uArm auf %s verbunden", swift.port)

    def close(self) -> None:
        if self._controller is None:
            return
        self._controller.disconnect()
        self._controller = None

    def execute_move(self, move: Move, *, player: Player) -> bool:
        if self._controller is None:
            self.connect()

        is_placement_move = move.src is None
        src_xy = self._resolve_move_source(move, player)
        dst_xy = self.board_positions.get(move.dst)
        src_pick_z = self.reserve_pick_z if is_placement_move else MILL_PICK_Z
        pick_lift_z = MILL_PLACE_Z if is_placement_move else SAFE_Z

        if src_xy is None:
            logger.warning(
                "Roboterzug fuer %s uebersprungen: keine Quellkoordinate fuer %s verfuegbar.",
                _format_move(move),
                player,
            )
            return False

        if dst_xy is None:
            logger.warning("Roboterzug uebersprungen: Ziel-Label %s hat keine Roboterkoordinate.", move.dst)
            return False

        try:
            self._pick_and_place(src_xy, dst_xy, src_pick_z=src_pick_z, pick_lift_z=pick_lift_z)
            if is_placement_move:
                self._advance_reserve_slot(player)

            if move.capture is not None:
                self._handle_capture(move.capture, captured_player=other_player(player))

            self._move_to(*REST_POS)
            return True
        except Exception:
            logger.exception("Roboterausfuehrung fuer Zug %s fehlgeschlagen", _format_move(move))
            try:
                controller = self._controller
                if controller is not None and controller.swift is not None:
                    swift = controller.swift
                    swift.set_pump(on=False)
                    self._move_to(*REST_POS)
            except Exception:
                logger.exception("Konnte Roboter nach Ausfuehrungsfehler nicht in sicheren Zustand bringen.")
            return False

    def _resolve_move_source(self, move: Move, player: Player) -> tuple[float, float] | None:
        if move.src is not None:
            return self.board_positions.get(move.src)
        return self._next_reserve_slot(player)

    def _next_reserve_slot(self, player: Player) -> tuple[float, float] | None:
        slots = self._reserve_slots.get(player)
        if not slots:
            return None

        next_index = self._reserve_indices.get(player, 0)
        if next_index >= len(slots):
            logger.warning(
                "Keine Reserve-Slots mehr fuer %s verfuegbar (%s/%s verbraucht).",
                player,
                next_index,
                len(slots),
            )
            return None
        return slots[next_index]

    def _advance_reserve_slot(self, player: Player) -> None:
        slots = self._reserve_slots.get(player)
        if not slots:
            return
        next_index = self._reserve_indices.get(player, 0)
        if next_index < len(slots):
            self._reserve_indices[player] = next_index + 1

    def _handle_capture(self, capture_label: str, *, captured_player: Player) -> None:
        capture_src = self.board_positions.get(capture_label)
        if capture_src is None:
            logger.warning("Schlag-Label %s hat keine Roboterkoordinate; Stein bitte manuell entfernen.", capture_label)
            return

        reserve_dst = self._next_capture_reserve_slot(captured_player)
        if reserve_dst is None:
            logger.warning(
                "Keine Reserve-Ablage fuer geschlagenen Stein %s (%s) verfuegbar.",
                capture_label,
                captured_player,
            )
            return

        self._pick_and_place(capture_src, reserve_dst, src_pick_z=MILL_PICK_Z)
        self._advance_capture_reserve_slot(captured_player)

    def _next_capture_reserve_slot(self, player: Player) -> tuple[float, float] | None:
        slots = self._reserve_slots.get(player)
        if not slots:
            return None

        next_index = self._capture_reserve_indices.get(player, 0)
        if next_index >= len(slots):
            logger.warning(
                "Keine Capture-Reserve-Slots mehr fuer %s verfuegbar (%s/%s belegt).",
                player,
                next_index,
                len(slots),
            )
            return None
        return slots[next_index]

    def _advance_capture_reserve_slot(self, player: Player) -> None:
        slots = self._reserve_slots.get(player)
        if not slots:
            return
        next_index = self._capture_reserve_indices.get(player, 0)
        if next_index < len(slots):
            self._capture_reserve_indices[player] = next_index + 1

    def _pick_and_place(
        self,
        src_xy: tuple[float, float],
        dst_xy: tuple[float, float],
        *,
        src_pick_z: float = MILL_PICK_Z,
        pick_lift_z: float = SAFE_Z,
    ) -> None:
        self._pick_at(src_xy, pick_z=src_pick_z, pick_lift_z=pick_lift_z)
        self._place_at(dst_xy)

    def _pick_at(
        self,
        xy: tuple[float, float],
        *,
        pick_z: float = MILL_PICK_Z,
        pick_lift_z: float = SAFE_Z,
    ) -> None:
        x, y = xy
        swift = self._require_swift()
        self._move_to(x, y, SAFE_Z)
        self._move_to(x, y, pick_z)
        swift.set_pump(on=True)
        time.sleep(0.2)
        self._move_to(x, y, pick_lift_z)
        if pick_lift_z != SAFE_Z:
            self._move_to(x, y, SAFE_Z)

    def _place_at(self, xy: tuple[float, float]) -> None:
        x, y = xy
        swift = self._require_swift()
        self._move_to(x, y, SAFE_Z)
        self._move_to(x, y, MILL_PLACE_Z)
        swift.set_pump(on=False)
        time.sleep(0.2)
        self._move_to(x, y, SAFE_Z)

    def _move_to(self, x: float, y: float, z: float) -> None:
        controller = self._require_controller()
        controller.move_to(float(x), float(y), float(z), speed=self.move_speed)

    def _require_controller(self) -> UArmController:
        if self._controller is None:
            self.connect()
        controller = self._controller
        if controller is None:
            raise RuntimeError("Roboter-Controller ist nicht verfuegbar.")
        return controller

    def _require_swift(self) -> Any:
        controller = self._require_controller()
        swift = controller.swift
        if swift is None:
            raise RuntimeError("Roboter-Controller hat keine aktive Swift-Verbindung.")
        return swift


def _format_move(move: Move) -> str:
    src = move.src if move.src is not None else "VORRAT"
    capture = f" x {move.capture}" if move.capture is not None else ""
    return f"{move.player}: {src} -> {move.dst}{capture}"


__all__ = [
    "MillRobotBridge",
    "ROBOT_BOARD_MAPS",
    "build_default_reserve_positions",
    "load_robot_board_positions",
]
