import logging
import math
import time
from typing import Any, Optional

from uarm.utils.log import logger as uarm_logger
from uarm.wrapper import SwiftAPI

try:
    from config import (
        PICK_Z,
        SAFE_Z,
        UARM_CALLBACK_THREADS,
        UARM_MAX_RADIUS,
        UARM_MIN_RADIUS,
        UARM_PORT,
    )
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import (
        PICK_Z,
        SAFE_Z,
        UARM_CALLBACK_THREADS,
        UARM_MAX_RADIUS,
        UARM_MIN_RADIUS,
        UARM_PORT,
    )

logger = logging.getLogger(__name__)
# Reduce noisy uArm SDK logging (suppress VERBOSE output).
uarm_logger.setLevel(logging.WARNING)


class WorkspaceError(ValueError):
    """Wird ausgeloest, wenn eine Zielposition ausserhalb des uArm-Arbeitsbereichs liegt."""


class UArmController:
    """High-level wrapper around the uArm Swift API."""

    def __init__(self, port: Optional[str] = UARM_PORT, do_connect: bool = True, **swift_options: Any):
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
        self.swift = SwiftAPI(port=selected_port, **options)
        self.swift.waiting_ready()
        logger.info("uArm bereit auf %s", self.swift.port)
        return self.swift

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

    # --- Basisbewegungen ---
    def _validate_workspace(self, x: float, y: float, z: float) -> None:
        """Prueft, ob das Ziel innerhalb der erlaubten Reichweite liegt."""
        radius = math.hypot(x, y)
        min_radius = max(0.0, UARM_MIN_RADIUS)
        max_radius = max(min_radius, UARM_MAX_RADIUS)
        if radius < min_radius:
            raise WorkspaceError(
                f"Zielradius {radius:.1f} mm liegt unterhalb der sicheren Mindestdistanz "
                f"von {min_radius:.1f} mm."
            )
        if radius > max_radius:
            raise WorkspaceError(
                f"Zielradius {radius:.1f} mm liegt ausserhalb der Reichweite "
                f"({max_radius:.1f} mm)."
            )

    def move_to(self, x: float, y: float, z: float, speed: int = 500):
        self._validate_workspace(x, y, z)
        logger.debug(f"Bewege zu ({x:.1f}, {y:.1f}, {z:.1f})")
        self.swift.set_position(x=x, y=y, z=z, speed=speed, wait=True)

    def safe_move(self, x: float, y: float):
        """Bewege sicher: erst hoch, dann horizontal, dann runter."""
        self.move_to(x, y, SAFE_Z)
        time.sleep(0.2)
        self.move_to(x, y, PICK_Z)

    def open_gripper(self):
        logger.debug("Oeffne Greifer")
        self.swift.set_gripper(True)

    def close_gripper(self):
        logger.debug("Schliesse Greifer")
        self.swift.set_gripper(False)

    def emergency_stop(self):
        """Sofortiger Stopp (z. B. bei Handerkennung)."""
        if self.swift:
            logger.warning("Notstopp ausgeloest!")
            self.swift.set_speed_factor(0)
