from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from gaming_robot_arm.config import CALIBRATION_DIR, CAMERA_INDEX
from gaming_robot_arm.utils.logger import logger as base_logger
from gaming_robot_arm.games.mill.core.board import BOARD_LABELS
from gaming_robot_arm.vision.mill_board_detector import detect_board_positions
from gaming_robot_arm.vision.recording import open_camera
logger = base_logger.getChild("calibration")
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

# detect_board_positions() liefert A1–C8 in der Reihenfolge von BOARD_LABELS.
DETECT_IDX = list(range(len(BOARD_LABELS)))


@dataclass(slots=True)
class CalibrationStore:
    base_dir: Path = CALIBRATION_DIR

    @property
    def cam_to_robot_path(self) -> Path:
        return self.base_dir / "cam_to_robot_homography.json"

    def list_files(self) -> Iterable[Path]:
        return sorted(self.base_dir.glob("*"))


STORE = CalibrationStore()


def _label_positions(positions: List[Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    labeled = {}
    for lbl, idx in zip(BOARD_LABELS, DETECT_IDX):
        try:
            x, y = positions[idx]
        except IndexError:
            raise ValueError(f"Erwartete 24 Punkte, erhalten: {len(positions)}") from None
        labeled[lbl] = (int(x), int(y))
    return labeled


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def capture_board_pixels(camera_index: int = CAMERA_INDEX) -> Dict[str, Tuple[int, int]]:
    """Erfasst ein Frame, detektiert A1–C8 und speichert die Pixelkoordinaten."""
    with open_camera(camera_index=camera_index) as cam:
        cv2.namedWindow("Board Pixels", cv2.WINDOW_NORMAL)
        print("[INFO] Richte das Brett aus. 'c'=capture, 'q'=abbrechen.")
        while True:
            ok, frame = cam.read()
            if not ok:
                logger.error("Kein Kamerabild.")
                break

            positions, annotated = detect_board_positions(frame.copy(), debug=True)
            display = annotated if annotated is not None else frame
            cv2.imshow("Board Pixels", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != ord("c"):
                continue

            if positions is None or len(positions) != 24:
                logger.warning("Detektion liefert %s Punkte, erwarte 24.", len(positions) if positions else 0)
                continue

            labeled = _label_positions(positions)
            existing: dict = {}
            if STORE.cam_to_robot_path.exists():
                try:
                    existing = _load_json(STORE.cam_to_robot_path)
                except Exception:
                    pass
            payload = dict(existing)
            payload["labels_used"] = list(BOARD_LABELS)
            payload["board_pixels"] = labeled
            if "H" not in existing:
                payload["method"] = "board_pixels"
            STORE.cam_to_robot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("Brett-Pixel gespeichert unter %s.", STORE.cam_to_robot_path)
            cv2.destroyAllWindows()
            return labeled

    cv2.destroyAllWindows()
    raise RuntimeError("Konnte keine gültigen Brett-Pixel erfassen.")


def load_board_pixels() -> Dict[str, Tuple[int, int]]:
    """Laedt Brett-Pixel (A1–C8) aus cam_to_robot_homography.json."""
    if not STORE.cam_to_robot_path.exists():
        raise FileNotFoundError("Keine Brett-Pixel gefunden (cam_to_robot_homography.json fehlt).")
    data = _load_json(STORE.cam_to_robot_path)
    board_pixels = data.get("board_pixels", {})
    if not board_pixels:
        raise FileNotFoundError("cam_to_robot_homography.json enthaelt keine Brett-Pixel.")
    return {k: tuple(v) for k, v in board_pixels.items()}


def calibrate_homography() -> None:
    """Fit einer 2D-Homography Kamera-Pixel → Roboter-XY anhand von 4–6 Punkten."""
    try:
        pixels = load_board_pixels()
        logger.info("Geladene Brett-Pixel.")
    except FileNotFoundError:
        pixels = capture_board_pixels()

    print("Gib mindestens 4 Punktpaare ein: Label  Roboter-X  Roboter-Y (mm). Leerzeile beendet Eingabe.")
    img_pts: List[Tuple[float, float]] = []
    rob_pts: List[Tuple[float, float]] = []
    used_labels: List[str] = []

    while True:
        line = input("Label,X,Y (z.B. A1 123 456): ").strip()
        if not line:
            break
        parts = line.split()
        if len(parts) != 3:
            print("Bitte: Label X Y")
            continue
        lbl, x_str, y_str = parts
        lbl = lbl.upper()
        if lbl not in pixels:
            print(f"Unbekanntes Label {lbl}. Erlaubt: {', '.join(BOARD_LABELS)}")
            continue
        try:
            rx, ry = float(x_str), float(y_str)
        except ValueError:
            print("X/Y als Zahl angeben.")
            continue
        img_pts.append(pixels[lbl])
        rob_pts.append((rx, ry))
        used_labels.append(lbl)

    if len(img_pts) < 4:
        raise RuntimeError("Mindestens 4 Punktpaare nötig.")

    H, mask = cv2.findHomography(
        np.array(img_pts, dtype=np.float32),
        np.array(rob_pts, dtype=np.float32),
        method=cv2.RANSAC,
    )
    if H is None:
        raise RuntimeError("Homography konnte nicht berechnet werden.")

    payload = {
        "method": "homography_2d",
        "pairs": len(img_pts),
        "inliers": int(mask.sum()) if mask is not None else len(img_pts),
        "labels_used": used_labels,
        "board_pixels": pixels,
        "H": H.tolist(),
    }
    STORE.cam_to_robot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Homography gespeichert unter %s.", STORE.cam_to_robot_path)


def detect_live_board_pixels(
    session,
    attempts: int = 6,
) -> Dict[str, Tuple[float, float]]:
    """Detect board pixel positions from live camera frames.

    Tries up to `attempts` frames. Returns a dict mapping BOARD_LABELS
    to (x, y) float pixel coords on success.
    Raises RuntimeError if no attempt yields all 24 positions.

    `session` must expose a `.read()` method returning an np.ndarray frame.
    """
    from gaming_robot_arm.vision.mill_board_detector import detect_board_positions as _detect

    max_attempts = max(1, int(attempts))
    for attempt_idx in range(max_attempts):
        try:
            frame = session.read()
        except Exception as exc:
            logger.warning(
                "Konnte keinen Kameraframe lesen (Versuch %s/%s): %s",
                attempt_idx + 1,
                max_attempts,
                exc,
            )
            continue
        try:
            positions, _annotated = _detect(frame, debug=False, return_bw=False)
        except Exception:
            logger.exception(
                "Brett-Detektor fehlgeschlagen (Versuch %s/%s).",
                attempt_idx + 1,
                max_attempts,
            )
            continue
        if len(positions) != len(BOARD_LABELS):
            logger.debug(
                "Live-Brettkalibrierung: %s/%s Positionen (Versuch %s/%s).",
                len(positions),
                len(BOARD_LABELS),
                attempt_idx + 1,
                max_attempts,
            )
            continue
        board_pixels = {
            label: (float(x), float(y))
            for label, (x, y) in zip(BOARD_LABELS, positions)
        }
        logger.info("Live-Brettkalibrierung erfolgreich (%s Positionen).", len(board_pixels))
        return board_pixels

    raise RuntimeError("Live-Brettkalibrierung fehlgeschlagen.")


def main() -> None:
    print("=== Schlanke Kalibrierung (Board-Pixel & Homography) ===")
    print("1 - Brett-Pixel erfassen (A1–C8)")
    print("2 - Homography Kamera→Roboter fitten (nur XY)")
    print("3 - Dateien anzeigen")

    choice = input("Auswahl: ").strip()
    if choice == "1":
        capture_board_pixels()
    elif choice == "2":
        calibrate_homography()
    elif choice == "3":
        files = list(STORE.list_files())
        if not files:
            print("Keine Kalibrationsdateien vorhanden.")
        else:
            for file in files:
                print("→", file.name)
    else:
        print("Abgebrochen.")


if __name__ == "__main__":
    main()
