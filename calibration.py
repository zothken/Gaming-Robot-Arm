from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from config import CAMERA_INDEX, DATA_DIR
from utils.logger import logger as base_logger
from vision.board_detector import detect_board_positions
from vision.recording import open_camera
1
logger = base_logger.getChild("calibration")
CALIB_DIR = (Path(DATA_DIR) / "calibration").resolve()
CALIB_DIR.mkdir(parents=True, exist_ok=True)

# Feste Label-Reihenfolge, passend zu detect_board_positions()
BOARD_LABELS: List[str] = [
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
]
# indices ordnen die 24 output-Punkte aus detect_board_positions() in obige Labels ein
DETECT_IDX = [0, 4, 1, 7, 3, 5, 2, 6, 8, 12, 9, 15, 11, 13, 10, 14, 16, 20, 17, 23, 19, 21, 18, 22]


@dataclass(slots=True)
class CalibrationStore:
    base_dir: Path = CALIB_DIR

    @property
    def board_pixels_path(self) -> Path:
        return self.base_dir / "board_pixels.json"

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
            STORE.board_pixels_path.write_text(json.dumps(labeled, indent=2), encoding="utf-8")
            logger.info("Brett-Pixel gespeichert unter %s.", STORE.board_pixels_path)
            cv2.destroyAllWindows()
            return labeled

    cv2.destroyAllWindows()
    raise RuntimeError("Konnte keine gültigen Brett-Pixel erfassen.")


def load_board_pixels() -> Dict[str, Tuple[int, int]]:
    data = json.loads(STORE.board_pixels_path.read_text(encoding="utf-8"))
    return {k: tuple(v) for k, v in data.items()}


def calibrate_homography() -> None:
    """Fit einer 2D-Homography Kamera-Pixel → Roboter-XY anhand von 4–6 Punkten."""
    try:
        pixels = load_board_pixels()
        logger.info("Geladene Brett-Pixel aus %s.", STORE.board_pixels_path)
    except FileNotFoundError:
        pixels = capture_board_pixels()

    print("Gib mindestens 4 Punktpaare ein: Label  Robot-X  Robot-Y (mm). Leerzeile beendet Eingabe.")
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
