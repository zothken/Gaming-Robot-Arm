"""Gemeinsame Helfer fuer Homography-IO und Pixel→Roboter-Umrechnung."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from gaming_robot_arm.config import HOMOGRAPHY_PATH
from gaming_robot_arm.utils.logger import logger

DEFAULT_H_PATH = HOMOGRAPHY_PATH

def load_homography(path: Path | None = None) -> tuple[np.ndarray | None, dict[str, tuple[float, float]]]:
    """
    Laedt die Kamera→Roboter-Homography-Datei und die zugehoerigen Brettpixel.

    Rueckgabe: (H-Matrix oder None, board_pixels-Dict).
    """
    h_path = Path(path) if path is not None else DEFAULT_H_PATH
    if not h_path.exists():
        logger.warning("Homography-Datei nicht gefunden: %s", h_path)
        return None, {}

    data = json.loads(h_path.read_text(encoding="utf-8"))
    board_pixels = data.get("board_pixels", {})
    if not board_pixels:
        logger.warning("Homography enthaelt keine board_pixels: %s", h_path)
        return None, {}

    H = data.get("H")
    if not H:
        logger.warning("Homography-Datei enthaelt keine H-Matrix: %s", h_path)
        return None, {k: tuple(v) for k, v in board_pixels.items()}

    return np.array(H, dtype=np.float64), {k: tuple(v) for k, v in board_pixels.items()}


def img_to_robot(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    """Wandelt Pixelkoordinaten (u,v) ueber die Homography in Roboter-XY um."""
    vec = H @ np.array([u, v, 1.0])
    x, y = vec[:2] / vec[2]
    return float(x), float(y)

def fit_homography_from_correspondences(
    board_pixels: Mapping[str, tuple[float, float]],
    board_robot: Mapping[str, tuple[float, float]],
    *,
    min_pairs: int = 4,
) -> np.ndarray | None:
    """
    Fittet eine Homography von Pixelkoordinaten auf Roboterkoordinaten ueber gemeinsame Labels.

    Nuetzlich, wenn Brettpixel aus der Kamera detektiert werden, waehrend die
    Roboterpositionen ueber einen mechanischen Adapter fest vorgegeben sind.
    """
    if not board_pixels or not board_robot:
        logger.warning("Homography-Fit: board_pixels oder board_robot leer.")
        return None

    labels = [lbl for lbl in board_pixels.keys() if lbl in board_robot]
    if len(labels) < min_pairs:
        logger.warning(
            "Homography-Fit: nur %s gemeinsame Labels (min=%s).",
            len(labels),
            min_pairs,
        )
        return None

    src = np.array([board_pixels[lbl] for lbl in labels], dtype=np.float32)
    dst = np.array([board_robot[lbl] for lbl in labels], dtype=np.float32)

    H, _mask = cv2.findHomography(src, dst, method=cv2.RANSAC)
    if H is None:
        logger.warning("Homography-Fit fehlgeschlagen (cv2.findHomography).")
        return None
    return H


__all__ = ["DEFAULT_H_PATH", "fit_homography_from_correspondences", "img_to_robot", "load_homography"]
