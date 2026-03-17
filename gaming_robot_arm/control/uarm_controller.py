import logging
from typing import Any, Optional

from uarm.utils.log import logger as uarm_logger
from uarm.wrapper import SwiftAPI

try:
    from gaming_robot_arm.calibration.mill_default_calibration import MILL_PICK_Z
    from gaming_robot_arm.config import (
        SAFE_Z,
        UARM_CALLBACK_THREADS,
        UARM_PORT,
    )
except ModuleNotFoundError:
    import os
    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(repo_root)
    from gaming_robot_arm.calibration.mill_default_calibration import MILL_PICK_Z
    from gaming_robot_arm.config import (
        SAFE_Z,
        UARM_CALLBACK_THREADS,
        UARM_PORT,
    )

logger = logging.getLogger(__name__)
# Reduziert laute uArm-SDK-Logs (unterdrueckt VERBOSE-Ausgaben).
uarm_logger.setLevel(logging.WARNING)

class UArmController:
    """Abstraktions-Wrapper um die uArm-Swift-API."""

    def __init__(
        self,
        port: Optional[str] = UARM_PORT,
        do_connect: bool = True,
        **swift_options: Any,
    ):
        self.default_port = port
        self.swift_options = dict(swift_options)
        self.swift_options.setdefault("callback_thread_pool_size", UARM_CALLBACK_THREADS)
        self.swift: Optional[SwiftAPI] = None

        if do_connect:
            self.connect(port=port)

    def connect(self, port: Optional[str] = None, **swift_options: Any) -> SwiftAPI:
        if self.swift is not None:
            return self.swift

        selected_port = port if port is not None else self.default_port
        options = {**self.swift_options, **swift_options}

        logger.info("Verbinde mit uArm Swift Pro...")
        swift = SwiftAPI(port=selected_port, **options)
        swift.waiting_ready()
        self.swift = swift
        logger.info("uArm bereit auf %s", swift.port)
        return swift

    def disconnect(self) -> None:
        if not self.swift:
            return

        swift = self.swift
        is_connected = getattr(swift, "connected", False)
        if is_connected:
            logger.info("Trenne Verbindung zum uArm...")
            swift.disconnect()
        else:
            logger.info("uArm bereits getrennt, ueberspringe disconnect().")
        self.swift = None

    def _require_swift(self) -> SwiftAPI:
        swift = self.swift
        if swift is None:
            raise RuntimeError("uArm ist nicht verbunden.")
        return swift

    def move_to(self, x: float, y: float, z: float, speed: int = 500) -> None:
        logger.debug(f"Bewege zu ({x:.1f}, {y:.1f}, {z:.1f})")
        swift = self._require_swift()
        swift.set_position(x=x, y=y, z=z, speed=speed, wait=True)

    def safe_move(self, x: float, y: float, z: float = MILL_PICK_Z, speed: int = 500) -> None:
        """Bewege sicher: vertikal auf SAFE_Z, dann horizontal, dann auf Ziel-Z."""
        swift = self._require_swift()
        current_pos = swift.get_position()

        if current_pos and len(current_pos) >= 2:
            current_x = float(current_pos[0])
            current_y = float(current_pos[1])
            self.move_to(current_x, current_y, SAFE_Z, speed=speed)
        else:
            logger.debug("Aktuelle Position unbekannt, ueberspringe vertikalen SAFE_Z-Hub.")

        self.move_to(x, y, SAFE_Z, speed=speed)
        if z != SAFE_Z:
            self.move_to(x, y, z, speed=speed)

    def open_gripper(self) -> None:
        logger.debug("Oeffne Greifer")
        swift = self._require_swift()
        swift.set_gripper(True)

    def close_gripper(self) -> None:
        logger.debug("Schliesse Greifer")
        swift = self._require_swift()
        swift.set_gripper(False)

    def emergency_stop(self) -> None:
        """Sofortiger Stopp (z. B. bei Handerkennung)."""
        swift = self.swift
        if swift:
            logger.warning("Notstopp ausgeloest!")
            swift.set_speed_factor(0)
