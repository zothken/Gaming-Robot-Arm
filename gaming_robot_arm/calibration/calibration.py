from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from gaming_robot_arm.config import CALIBRATION_DIR, CAMERA_INDEX
from gaming_robot_arm.utils.logger import logger as base_logger
from gaming_robot_arm.games.mill.board import BOARD_LABELS
from gaming_robot_arm.games.mill.mill_board_detector import detect_board_positions
from gaming_robot_arm.vision.recording import open_camera
logger = base_logger.getChild("calibration")
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

# detect_board_positions() liefert A1–C8 in der Reihenfolge von BOARD_LABELS.
DETECT_IDX = list(range(len(BOARD_LABELS)))


@dataclass(slots=True)
class CalibrationStore:
    base_dir: Path = CALIBRATION_DIR

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
            payload = {
                "method": "board_pixels",
                "labels_used": BOARD_LABELS,
                "board_pixels": labeled,
            }
            STORE.cam_to_robot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("Brett-Pixel gespeichert unter %s.", STORE.cam_to_robot_path)
            cv2.destroyAllWindows()
            return labeled

    cv2.destroyAllWindows()
    raise RuntimeError("Konnte keine gültigen Brett-Pixel erfassen.")


def _pixels_fit_score(
    board_pixels: Dict[str, Tuple[int, int]],
    *,
    frame_width: int,
    frame_height: int,
) -> float:
    """Heuristischer Score, wie gut board_pixels zu einer Frame-Groesse passen."""
    if not board_pixels or frame_width <= 0 or frame_height <= 0:
        return float("-inf")

    xs = [p[0] for p in board_pixels.values()]
    ys = [p[1] for p in board_pixels.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    inside = 0
    for x, y in board_pixels.values():
        if 0 <= x < frame_width and 0 <= y < frame_height:
            inside += 1
    inside_ratio = inside / max(1, len(board_pixels))

    # Overshoot relativ zur Frame-Groesse: je mehr ausserhalb, desto schlechter.
    overshoot = 0.0
    if min_x < 0:
        overshoot += abs(min_x) / frame_width
    if min_y < 0:
        overshoot += abs(min_y) / frame_height
    if max_x >= frame_width:
        overshoot += (max_x - (frame_width - 1)) / frame_width
    if max_y >= frame_height:
        overshoot += (max_y - (frame_height - 1)) / frame_height

    return float(inside_ratio - 0.5 * overshoot)


def _pixels_inside_ratio(
    board_pixels: Dict[str, Tuple[int, int]],
    *,
    frame_width: int,
    frame_height: int,
) -> float:
    if not board_pixels or frame_width <= 0 or frame_height <= 0:
        return 0.0
    inside = 0
    for x, y in board_pixels.values():
        if 0 <= x < frame_width and 0 <= y < frame_height:
            inside += 1
    return inside / max(1, len(board_pixels))


def load_board_pixels(
    *,
    frame_size: Tuple[int, int] | None = None,
) -> Dict[str, Tuple[int, int]]:
    """
    Laedt Brett-Pixel (A1–C8).

    Standardverhalten (frame_size=None): bevorzugt cam_to_robot_homography.json (falls vorhanden),
    sonst board_pixels.json.

    Wenn frame_size angegeben ist, wird die Quelle gewaehlt, deren board_pixels am besten in die
    aktuelle Frame-Groesse passt (hilft bei wechselnden Kamera-Aufloesungen).
    """
    candidates: list[tuple[str, Dict[str, Tuple[int, int]]]] = []

    if STORE.cam_to_robot_path.exists():
        data = _load_json(STORE.cam_to_robot_path)
        board_pixels = data.get("board_pixels", {})
        if board_pixels:
            candidates.append(
                (str(STORE.cam_to_robot_path.name), {k: tuple(v) for k, v in board_pixels.items()})
            )

    if STORE.board_pixels_path.exists():
        data = _load_json(STORE.board_pixels_path)
        if data:
            candidates.append((str(STORE.board_pixels_path.name), {k: tuple(v) for k, v in data.items()}))

    if not candidates:
        raise FileNotFoundError("Keine Brett-Pixel gefunden (cam_to_robot_homography.json/board_pixels.json).")

    if frame_size is None or len(candidates) == 1:
        # Rueckwaertskompatibel: zuerst Homography-Datei, sonst board_pixels.json.
        return candidates[0][1]

    frame_width, frame_height = frame_size
    default_name, default_pixels = candidates[0]
    # Konservativ: wenn die Standardquelle groesstenteils in den Frame passt, verwende sie weiter.
    if _pixels_inside_ratio(default_pixels, frame_width=frame_width, frame_height=frame_height) >= 0.8:
        return default_pixels

    scored = [
        (
            _pixels_fit_score(pixels, frame_width=frame_width, frame_height=frame_height),
            name,
            pixels,
        )
        for name, pixels in candidates
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best_name, best_pixels = scored[0]

    # Transparenz: warnen, wenn die "Standardquelle" nicht passt.
    if best_name != candidates[0][0]:
        logger.warning(
            "Brett-Pixel aus %s gewaehlt (bessere Passung fuer Frame %sx%s; score=%.3f).",
            best_name,
            frame_width,
            frame_height,
            best_score,
        )

    return best_pixels


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
