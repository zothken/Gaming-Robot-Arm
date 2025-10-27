import time
import logging
from uarm.wrapper import SwiftAPI
from config import SAFE_Z, PICK_Z, UARM_PORT

logger = logging.getLogger(__name__)

class UArmController:
    def __init__(self, port: str = UARM_PORT, do_connect: bool = True):
        self.swift = None
        if do_connect:
            self.connect(port)

    def connect(self, port: str | None = None):
        logger.info("Verbinde mit uArm Swift Pro...")
        self.swift = SwiftAPI(port=port)
        self.swift.waiting_ready()
        logger.info("uArm bereit auf %s", self.swift.port)

    def disconnect(self):
        if self.swift:
            logger.info("Trenne Verbindung zum uArm...")
            self.swift.disconnect()
            self.swift = None

    # --- Basisbewegungen ---
    def move_to(self, x: float, y: float, z: float, speed: int = 500):
        logger.debug(f"Bewege zu ({x:.1f}, {y:.1f}, {z:.1f})")
        self.swift.set_position(x=x, y=y, z=z, speed=speed, wait=True)

    def safe_move(self, x: float, y: float):
        """Bewege sicher: erst hoch, dann horizontal, dann runter."""
        self.move_to(x, y, SAFE_Z)
        time.sleep(0.2)
        self.move_to(x, y, PICK_Z)

    def open_gripper(self):
        logger.debug("Öffne Greifer")
        self.swift.set_gripper(True)

    def close_gripper(self):
        logger.debug("Schließe Greifer")
        self.swift.set_gripper(False)

    def emergency_stop(self):
        """Sofortiger Stopp (z. B. bei Handerkennung)."""
        if self.swift:
            logger.warning("⚠️ Notstopp ausgelöst!")
            self.swift.set_speed_factor(0)
