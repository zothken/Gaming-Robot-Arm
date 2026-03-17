"""Kombinierte Steuerung: Freies Fahren und Positionswahl ueber Brett-Labels."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
import sys
from threading import Event, Thread

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gaming_robot_arm.calibration.mill_default_calibration import (
    MILL_PICK_Z,
    MILL_PLACE_Z,
    get_mill_uarm_positions,
)
from gaming_robot_arm.config import SAFE_Z, UARM_CALLBACK_THREADS, UARM_PORT
from gaming_robot_arm import VisionControlRuntime
from gaming_robot_arm.utils.logger import logger
from gaming_robot_arm.vision.recording import recording_session

H_FILE = ROOT / "gaming_robot_arm" / "calibration" / "cam_to_robot_homography.json"


def load_homography() -> tuple[np.ndarray | None, dict[str, tuple[float, float]]]:
    if not H_FILE.exists():
        logger.warning(
            "Homography-Datei nicht gefunden: %s. Pixel->Roboter-Mapping via Homography deaktiviert.",
            H_FILE,
        )
        return None, {}

    data = json.loads(H_FILE.read_text(encoding="utf-8"))
    board_pixels = data.get("board_pixels", {})
    if not board_pixels:
        logger.warning("Homography enthaelt keine board_pixels. Pixel->Roboter-Mapping deaktiviert.")
        return None, {}

    return np.array(data["H"], dtype=np.float64), board_pixels


def img_to_robot(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    vec = H @ np.array([u, v, 1.0])
    x, y = vec[:2] / vec[2]
    return float(x), float(y)


def handle_board_move(
    board_robot: dict[str, tuple[float, float]],
    H: np.ndarray | None,
    board_pixels: dict[str, tuple[float, float]],
    controller,
) -> bool:
    if not board_robot and (H is None or not board_pixels):
        logger.error("Keine Brett-Kalibrierung verfuegbar. Bitte Kalibrierung ausfuehren.")
        return False

    lbl = input("Brett-Position (A1-C8, leer=Abbrechen): ").strip().upper()
    if not lbl:
        return False

    if board_robot:
        if lbl not in board_robot:
            logger.warning(
                "Unbekanntes Label '%s'. Verfuegbare Labels: %s",
                lbl,
                ", ".join(sorted(board_robot.keys())),
            )
            return False
        x, y = board_robot[lbl]
        logger.info("Fahre zu %s → Roboter=(%.1f, %.1f, %.1f)", lbl, x, y, SAFE_Z)
        controller.move_to(x, y, SAFE_Z)
        return True

    if H is None or not board_pixels:
        logger.error("Keine Brett-Kalibrierung verfuegbar. Bitte Kalibrierung ausfuehren.")
        return False
    if lbl not in board_pixels:
        logger.warning(
            "Unbekanntes Label '%s'. Verfuegbare Labels: %s",
            lbl,
            ", ".join(sorted(board_pixels.keys())),
        )
        return False

    u, v = board_pixels[lbl]
    x, y = img_to_robot(H, u, v)
    logger.info("Fahre zu %s → Pixel=(%.1f, %.1f) → Roboter=(%.1f, %.1f, %.1f)", lbl, u, v, x, y, SAFE_Z)
    controller.move_to(x, y, SAFE_Z)
    return True


def main(port: str | None = UARM_PORT) -> None:
    H, board_pixels = load_homography()
    board_robot = get_mill_uarm_positions()

    with VisionControlRuntime(display=False) as runtime:
        controller = runtime.ensure_controller(
            port=port,
            callback_thread_pool_size=UARM_CALLBACK_THREADS,
        )
        swift = controller.swift

        if swift is None:
            logger.error("Keine Verbindung zum uArm verfuegbar.")
            return

        current_pos = swift.get_position()
        last_z_position = current_pos[2] if current_pos and len(current_pos) > 2 else SAFE_Z
        stored_prev_z = last_z_position
        z_toggle_active = False

        pump_on = False
        record_choice = input("Aufnahme speichern? (y/n): ").strip().lower()
        record_enabled = record_choice.startswith("y")

        with (recording_session() if record_enabled else nullcontext()) as session:
            stop_recording = Event() if record_enabled else None

            if record_enabled:
                assert session is not None
                assert stop_recording is not None
                active_session = session
                active_stop_recording = stop_recording

                def _recording_worker() -> None:
                    while not active_stop_recording.is_set():
                        try:
                            frame = active_session.read()
                        except RuntimeError as exc:
                            logger.warning("Kamera-Lesefehler, stoppe Aufnahme: %s", exc)
                            break
                        active_session.write(frame)

                recorder = Thread(target=_recording_worker, daemon=True)
                recorder.start()
                logger.info("Aufnahme laeuft: %s", active_session.output_path)
            else:
                recorder = None
                logger.info("Aufnahme deaktiviert.")

            logger.info(
                "Steuerung bereit: [c] Zu Koordinate fahren, [p] Zu Brettposition fahren, [z] Z umschalten, [s] Saugen, [r] Reset, [q] Beenden."
            )

            while True:
                cmd = input("Befehl (c/p/z/s/r/q): ").strip().lower()

                if cmd == "q":
                    logger.info("Beende Test-Routine.")
                    if stop_recording:
                        stop_recording.set()
                    if recorder:
                        recorder.join(timeout=2.0)
                    swift.disconnect()
                    break

                if cmd == "c":
                    try:
                        x = float(input("x: ").strip())
                        y = float(input("y: ").strip())
                        z = float(input("z: ").strip())
                    except ValueError:
                        logger.warning("Ungueltige Koordinaten, bitte erneut versuchen.")
                        continue

                    logger.info("Bewege zu x=%.1f y=%.1f z=%.1f", x, y, z)

                    safe_z = 50.0
                    current_pos = swift.get_position()
                    current_x = current_pos[0] if current_pos and len(current_pos) > 0 else None
                    current_y = current_pos[1] if current_pos and len(current_pos) > 1 else None
                    current_z = current_pos[2] if current_pos and len(current_pos) > 2 else safe_z

                    if current_x is not None and current_y is not None and current_z != safe_z:
                        swift.set_position(x=current_x, y=current_y, z=safe_z, wait=True)

                    swift.set_position(x=x, y=y, z=safe_z, wait=True)
                    if z != safe_z:
                        swift.set_position(x=x, y=y, z=z, wait=True)
                    last_z_position = z
                    stored_prev_z = last_z_position
                    z_toggle_active = False
                    continue

                if cmd == "p":
                    moved = handle_board_move(board_robot, H, board_pixels, controller)
                    if moved:
                        last_z_position = SAFE_Z
                        stored_prev_z = last_z_position
                        z_toggle_active = False
                    continue

                if cmd == "z":
                    current_pos = swift.get_position()
                    if not current_pos or len(current_pos) < 3:
                        logger.warning("Aktuelle Position unbekannt, Z-Toggle nicht moeglich.")
                        continue

                    current_x, current_y = current_pos[0], current_pos[1]
                    if not z_toggle_active:
                        stored_prev_z = last_z_position if last_z_position is not None else current_pos[2]
                        if pump_on:
                            target_z = MILL_PLACE_Z
                            logger.info("Pumpe aktiv – senke auf PLACE_Z=%.1f.", MILL_PLACE_Z)
                        else:
                            target_z = MILL_PICK_Z
                            logger.info("Wechsle auf PICK_Z=%.1f bei gleichem x/y.", MILL_PICK_Z)
                        z_toggle_active = True
                    else:
                        target_z = stored_prev_z if stored_prev_z is not None else SAFE_Z
                        z_toggle_active = False
                        logger.info("Stelle Hoehe z=%.1f wieder her.", target_z)

                    swift.set_position(x=current_x, y=current_y, z=target_z, wait=True)
                    last_z_position = target_z
                    if not z_toggle_active:
                        stored_prev_z = last_z_position
                    continue

                if cmd == "s":
                    pump_on = not pump_on
                    swift.set_pump(on=pump_on)
                    logger.info("Saugen %s.", "AN" if pump_on else "AUS")
                    continue

                if cmd == "r":
                    logger.info("Setze uArm zurueck.")
                    swift.reset()
                    if pump_on:
                        swift.set_pump(on=True)
                    continue

                logger.warning("Unbekannter Befehl: %s", cmd)


if __name__ == "__main__":
    main()
