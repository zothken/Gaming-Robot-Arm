"""Skript: Zwei Brett-Positionen eingeben und Figur umsetzen."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Mapping, TypedDict

from gaming_robot_arm.config import CAMERA_INDEX, REST_POS, SAFE_Z, UARM_PORT
from gaming_robot_arm.calibration.mill_default_calibration import MILL_PICK_Z, MILL_PLACE_Z, MILL_UARM_POSITIONS
from gaming_robot_arm.control import UArmController
from gaming_robot_arm.games.mill.core.board import BOARD_LABELS
from gaming_robot_arm.utils.cli import prompt_board_label, prompt_recording_enabled
from gaming_robot_arm.utils.homography import fit_homography_from_correspondences, img_to_robot
from gaming_robot_arm.utils.logger import logger
from gaming_robot_arm.vision.detector_config import DEFAULT_FIGURE_PARAMS, FIGURE_DETECTOR_CONFIG_PATH
from gaming_robot_arm.vision.figure_detector import detect_board_assignments
from gaming_robot_arm.vision.mill_board_detector import detect_board_positions
from gaming_robot_arm.vision.recording import open_camera, recording_session

# Temporärer Korrekturwert: uArm fährt beim Anfahren der Figur zu weit in +Y.
PICKUP_Y_OFFSET_MM = 0.0


BoardPixels = dict[str, tuple[float, float]]


class BoardAssignment(TypedDict):
    label: str
    centroid: tuple[float, float]
    color: str


def _parse_assignment(raw_assignment: Mapping[str, object]) -> BoardAssignment | None:
    label = raw_assignment.get("label")
    centroid = raw_assignment.get("centroid")
    color = raw_assignment.get("color")
    if not isinstance(label, str) or not isinstance(color, str):
        return None
    if not isinstance(centroid, (tuple, list)) or len(centroid) != 2:
        return None
    u, v = centroid
    if not isinstance(u, (int, float)) or not isinstance(v, (int, float)):
        return None
    return {"label": label, "centroid": (float(u), float(v)), "color": color}


def _detect_initial_board_pixels(camera_index: int, attempts: int = 3) -> BoardPixels | None:
    """Öffnet Kamera, erkennt das Brett und gibt Pixel-Positionen im aktuellen Frame zurück."""
    try:
        with open_camera(camera_index=camera_index) as cam:
            for _ in range(5):  # Kamera warmlaufen lassen
                cam.read()
            for attempt in range(attempts):
                ret, frame = cam.read()
                if not ret:
                    logger.warning("Kameraframe konnte nicht gelesen werden (Versuch %d).", attempt + 1)
                    continue
                positions, _ = detect_board_positions(frame, debug=False)
                if len(positions) == len(BOARD_LABELS):
                    return {label: (float(x), float(y)) for label, (x, y) in zip(BOARD_LABELS, positions)}
                logger.warning(
                    "Brett-Detektion lieferte %d statt %d Positionen (Versuch %d).",
                    len(positions), len(BOARD_LABELS), attempt + 1,
                )
    except Exception:
        logger.exception("Fehler bei der Brett-Erkennung beim Start.")
    return None


def main() -> None:
    if not FIGURE_DETECTOR_CONFIG_PATH.exists():
        logger.warning(
            "Keine figure_detector_config.json gefunden – Standard-Parameter aktiv "
            "(brightness_split=%s). Falls Figuren falsch klassifiziert werden, bitte "
            "`python gaming_robot_arm/vision/figure_detector.py` ausfuehren und Einstellungen speichern.",
            DEFAULT_FIGURE_PARAMS["brightness_split"],
        )

    logger.info("Erkenne Brett-Positionen aus aktuellem Kamerabild...")
    board_pixels = _detect_initial_board_pixels(camera_index=CAMERA_INDEX)
    if board_pixels is None:
        raise SystemExit(
            "Brett konnte nicht erkannt werden. Bitte Brett vor die Kamera legen und neu starten.\n"
            "Tipp: `python gaming_robot_arm/vision/mill_board_detector.py` zum Testen der Brettdetektion."
        )

    H = fit_homography_from_correspondences(board_pixels, MILL_UARM_POSITIONS)
    if H is None:
        raise SystemExit(
            "Homographie konnte nicht berechnet werden. Bitte MILL_UARM_POSITIONS pruefen."
        )
    logger.info("Brett erkannt, Homographie berechnet (%d Positionen).", len(board_pixels))

    controller = UArmController(port=UARM_PORT)
    swift = controller.swift
    if swift is None:
        raise SystemExit("uArm konnte nicht verbunden werden.")

    record_enabled = prompt_recording_enabled()

    session_ctx = recording_session() if record_enabled else nullcontext()

    try:
        with session_ctx as session:
            if record_enabled:
                logger.info("Aufnahme aktiviert.")
            else:
                logger.info("Aufnahme deaktiviert.")

            logger.info("Fuehre Reset aus.")
            swift.reset()
            time.sleep(0.2)

            logger.info("Fahre in Ruheposition zu x=%.1f y=%.1f z=%.1f", *REST_POS)
            controller.move_to(*REST_POS)

            print("Druecke 'q' bei einer Eingabe, um zu beenden.")

            while True:
                raw_assignments = detect_board_assignments(
                    board_pixels,
                    session=session if record_enabled else None,
                    labels_order=sorted(board_pixels.keys()),
                    debug_assignments=False,
                    board_source="calibration",
                )
                assignments: list[BoardAssignment] = []
                for raw_assignment in raw_assignments:
                    parsed = _parse_assignment(raw_assignment)
                    if parsed is not None:
                        assignments.append(parsed)

                assignments = sorted(assignments, key=lambda a: a["label"])
                assignments_by_label = {a["label"]: a for a in assignments}
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
                start_lbl = prompt_board_label(
                    board_pixels,
                    "Start-Position: ",
                    allowed_labels=start_allowed,
                    cancel_labels={"Q"},
                )
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
                    continue

                target_lbl = prompt_board_label(
                    board_pixels,
                    "Ziel-Position: ",
                    allowed_labels=set(free_labels),
                    cancel_labels={"Q"},
                )
                if target_lbl is None:
                    logger.info("Abbruch durch Benutzer.")
                    return

                start_assignment = assignments_by_label.get(start_lbl)
                if not start_assignment or "centroid" not in start_assignment:
                    logger.info("Keine Figur fuer %s erkannt, starte erneute Erkennung.", start_lbl)
                    raw_retry_assignments = detect_board_assignments(
                        board_pixels,
                        session=session if record_enabled else None,
                        labels_order=sorted(board_pixels.keys()),
                        debug_assignments=False,
                        board_source="calibration",
                    )
                    for raw_assignment in raw_retry_assignments:
                        parsed = _parse_assignment(raw_assignment)
                        if parsed is not None:
                            assignments_by_label[parsed["label"]] = parsed
                    start_assignment = assignments_by_label.get(start_lbl)

                if not start_assignment or "centroid" not in start_assignment:
                    print(f"Figur auf {start_lbl} konnte nicht sicher erkannt werden. Bitte neu positionieren.")
                    logger.error("Keine erkannte Figur fuer %s gefunden.", start_lbl)
                    continue

                live_u, live_v = start_assignment["centroid"]
                logger.info("Erkannte Figur-Position fuer %s (live): u=%.1f v=%.1f", start_lbl, live_u, live_v)

                start_u, start_v = board_pixels[start_lbl]
                target_u, target_v = board_pixels[target_lbl]
                start_x, start_y = img_to_robot(H, start_u, start_v)
                target_x, target_y = img_to_robot(H, target_u, target_v)
                start_y_pick = start_y + PICKUP_Y_OFFSET_MM

                logger.info("Starte Umsetzen von %s -> %s", start_lbl, target_lbl)

                pump_on = False
                try:
                    # Position des Steins anfahren und ansaugen
                    controller.move_to(start_x, start_y_pick, SAFE_Z)
                    controller.move_to(start_x, start_y_pick, MILL_PICK_Z)
                    swift.set_pump(on=True)
                    pump_on = True
                    time.sleep(0.2)

                    # Sicher anheben und zur Zielposition wechseln
                    controller.move_to(start_x, start_y_pick, SAFE_Z)
                    controller.move_to(target_x, target_y, SAFE_Z)

                    # Stein ablegen
                    controller.move_to(target_x, target_y, MILL_PLACE_Z)
                    swift.set_pump(on=False)
                    pump_on = False
                    controller.move_to(target_x, target_y, SAFE_Z)

                    # Zur Ruheposition zurueckkehren
                    controller.move_to(*REST_POS)
                    logger.info("Bewegung abgeschlossen.")
                except Exception as exc:
                    logger.error("Bewegung kann nicht ausgefuehrt werden: %s", exc)
                    print(f"Bewegung fehlgeschlagen: {exc}")
                    try:
                        controller.move_to(*REST_POS)
                    except Exception:
                        logger.exception("Konnte nicht zur Ruheposition zurueckkehren.")
                finally:
                    if pump_on:
                        try:
                            swift.set_pump(on=False)
                        except Exception:
                            logger.exception("Konnte Pumpe nicht deaktivieren.")
    finally:
        controller.disconnect()
        print("Beendet.")


if __name__ == "__main__":
    main()
