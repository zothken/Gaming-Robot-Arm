"""Skript: Zwei Brett-Positionen eingeben und Figur umsetzen."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import SAFE_Z, UARM_PORT
from control import UArmController
from utils.logger import logger

H_FILE = ROOT / "data/calibration/cam_to_robot_homography.json"
REST_POS = (1.0, 150.0, 100.0)
PICK_Z = 10.0
PLACE_Z = 15.0


def load_homography() -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    data = json.loads(H_FILE.read_text(encoding="utf-8"))
    return np.array(data["H"], dtype=np.float64), data.get("board_pixels", {})


def img_to_robot(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    vec = H @ np.array([u, v, 1.0])
    x, y = vec[:2] / vec[2]
    return float(x), float(y)


def prompt_label(board_pixels: dict[str, tuple[float, float]], prompt: str) -> str | None:
    while True:
        lbl = input(prompt).strip().upper()
        if lbl == "Q":
            return None
        if lbl in board_pixels:
            return lbl
        print(f"Unbekanntes Label '{lbl}'. Erlaubt: {', '.join(sorted(board_pixels.keys()))}")


def main() -> None:
    if not H_FILE.exists():
        raise SystemExit(f"Homography-Datei nicht gefunden: {H_FILE}. Bitte erst kalibrieren.")

    H, board_pixels = load_homography()
    if not board_pixels:
        raise SystemExit("Homography-Datei enthaelt keine board_pixels. Bitte Kalibrierung erneut ausfuehren.")

    controller = UArmController(port=UARM_PORT)
    swift = controller.swift
    if swift is None:
        raise SystemExit("uArm konnte nicht verbunden werden.")

    try:
        logger.info("Fuehre Reset aus.")
        swift.reset()
        time.sleep(0.2)

        logger.info("Fahre in Ruheposition zu x=%.1f y=%.1f z=%.1f", *REST_POS)
        controller.move_to(*REST_POS)

        print("Gib Start- und Ziel-Label ein (A1–C8). 'q' beendet.")
        start_lbl = prompt_label(board_pixels, "Start-Position: ")
        if start_lbl is None:
            logger.info("Abbruch durch Benutzer.")
            return
        target_lbl = prompt_label(board_pixels, "Ziel-Position: ")
        if target_lbl is None:
            logger.info("Abbruch durch Benutzer.")
            return

        start_u, start_v = board_pixels[start_lbl]
        target_u, target_v = board_pixels[target_lbl]
        start_x, start_y = img_to_robot(H, start_u, start_v)
        target_x, target_y = img_to_robot(H, target_u, target_v)

        logger.info("Starte Umsetzen von %s -> %s", start_lbl, target_lbl)

        # Position des Steins anfahren und ansaugen
        controller.move_to(start_x, start_y, SAFE_Z)
        controller.move_to(start_x, start_y, PICK_Z)
        swift.set_pump(on=True)
        time.sleep(0.2)

        # Sicher anheben und zur Zielposition wechseln
        controller.move_to(start_x, start_y, SAFE_Z)
        controller.move_to(target_x, target_y, SAFE_Z)

        # Stein ablegen
        controller.move_to(target_x, target_y, PLACE_Z)
        swift.set_pump(on=False)

        # Zur Ruheposition zurueckkehren
        controller.move_to(*REST_POS)
        logger.info("Bewegung abgeschlossen.")
    finally:
        controller.disconnect()
        print("Beendet.")


if __name__ == "__main__":
    main()
