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

from config import PICK_Z, PLACE_Z, REST_POS, SAFE_Z, UARM_PORT
from control import UArmController, WorkspaceError
from utils.logger import logger
from vision.figure_detector import detect_figures, estimate_assign_distance
from vision.recording import open_camera

H_FILE = ROOT / "data/calibration/cam_to_robot_homography.json"


def load_homography() -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    data = json.loads(H_FILE.read_text(encoding="utf-8"))
    return np.array(data["H"], dtype=np.float64), data.get("board_pixels", {})


def img_to_robot(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    vec = H @ np.array([u, v, 1.0])
    x, y = vec[:2] / vec[2]
    return float(x), float(y)


def prompt_label(
    board_pixels: dict[str, tuple[float, float]],
    prompt: str,
    allowed_labels: set[str] | None = None,
) -> str | None:
    allowed = {lbl.upper() for lbl in allowed_labels} if allowed_labels is not None else None
    allowed_hint = f"Erlaubt: {', '.join(sorted(allowed))}" if allowed else None

    while True:
        lbl = input(prompt).strip().upper()
        if lbl == "Q":
            return None
        if lbl not in board_pixels:
            print(f"Unbekanntes Label '{lbl}'. Erlaubt: {', '.join(sorted(board_pixels.keys()))}")
            continue
        if allowed is not None and lbl not in allowed:
            hint = allowed_hint or "Keine verfuegbaren Positionen"
            print(f"Label '{lbl}' momentan nicht zulaessig. {hint}")
            continue
        return lbl


def detect_board_assignments(
    board_pixels: dict[str, tuple[float, float]],
    attempts: int = 4,
) -> list[dict[str, object]]:
    base_dist = estimate_assign_distance(board_pixels)
    dist_steps = [base_dist, base_dist * 1.25, base_dist * 1.5, base_dist * 1.75]

    try:
        with open_camera() as cam:
            for idx in range(min(attempts, len(dist_steps))):
                ret, frame = cam.read()
                if not ret:
                    logger.warning("Kameraframe konnte nicht gelesen werden (Versuch %s).", idx + 1)
                    continue

                try:
                    _, _, _, _, assignments = detect_figures(
                        frame,
                        board_coords=board_pixels,
                        max_assign_dist=dist_steps[idx],
                        return_assignments=True,
                        debug_assignments=True,
                    )
                except Exception:
                    logger.exception("Fehler bei der Zuordnung der Figuren zum Brett.")
                    return []

                if assignments:
                    return assignments

    except RuntimeError as exc:
        logger.warning("Kamera konnte nicht geoeffnet werden: %s", exc)
        return []

    return []


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

        assignments = detect_board_assignments(board_pixels)
        assignments = sorted(assignments, key=lambda a: a["label"])
        occupied_labels = [a["label"] for a in assignments]
        all_labels = sorted(board_pixels.keys())
        if assignments:
            formatted = ", ".join(f"{a['label']} ({a['color']})" for a in assignments)
            print(f"Erkannte Figuren auf Positionen: {formatted}")
            print(f"Waehle Start-Position aus: {', '.join(occupied_labels)}")
        else:
            print("Keine belegten Positionen erkannt.")
            print(f"Start-Position frei waehlbar aus: {', '.join(all_labels)}")

        start_allowed = set(occupied_labels) if occupied_labels else None
        start_lbl = prompt_label(board_pixels, "Start-Position: ", allowed_labels=start_allowed)
        if start_lbl is None:
            logger.info("Abbruch durch Benutzer.")
            return

        occupied_set = set(occupied_labels)
        free_labels = sorted(lbl for lbl in board_pixels if lbl not in occupied_set and lbl != start_lbl)
        if free_labels:
            print(f"Freie Positionen (als Ziel moeglich): {', '.join(free_labels)}")
        else:
            print("Keine freien Positionen erkannt - Abbruch.")
            logger.info("Keine freien Ziel-Positionen verfuegbar.")
            return

        target_lbl = prompt_label(board_pixels, "Ziel-Position: ", allowed_labels=set(free_labels))
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
        controller.move_to(target_x, target_y, SAFE_Z)

        # Zur Ruheposition zurueckkehren
        controller.move_to(*REST_POS)
        logger.info("Bewegung abgeschlossen.")
    except WorkspaceError as exc:
        logger.error("Bewegung kann nicht ausgefuehrt werden: %s", exc)
        print(f"Ziel ausserhalb des Arbeitsbereichs: {exc}")
    finally:
        controller.disconnect()
        print("Beendet.")


if __name__ == "__main__":
    main()
