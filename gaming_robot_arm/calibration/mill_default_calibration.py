"""Standard-uArm-Positionen fuer Muehle (Nine Men's Morris).

Dieses Modul definiert feste uArm-XY-Koordinaten (in mm) fuer die 24
Brett-Labels, die projektweit genutzt werden (A1-C8). Der mechanische Adapter
fixiert die relative Lage zwischen uArm und Brett, daher koennen diese
Koordinaten konstant bleiben.

Label-Reihenfolge (entspricht gaming_robot_arm.games.mill.mill_board_detector):
- A1..A8 = aeusseres Quadrat, im Uhrzeigersinn ab oben links
- B1..B8 = mittleres Quadrat, im Uhrzeigersinn ab oben links
- C1..C8 = inneres Quadrat, im Uhrzeigersinn ab oben links

Passe MILL_UARM_POSITIONS bei Bedarf auf dein vermessenes Setup an.
"""

from __future__ import annotations

from typing import Dict, Tuple

from gaming_robot_arm.games.mill.board import BOARD_LABELS

# Standardwerte (mm) fuer das mechanische Adapter-Setup.
# Durch eigene Messwerte ersetzen, falls der Adapter abweicht.
MILL_UARM_POSITIONS: Dict[str, Tuple[float, float]] = {
    "A1": (130.0, 90.0),
    "A2": (223.0, 90.0),
    "A3": (316.0, 90.0),
    "A4": (316.0, 0.0),
    "A5": (316.0, -90.0),
    "A6": (223.0, -90.0),
    "A7": (130.0, -90.0),
    "A8": (130.0, 0.0),
    "B1": (161.0, 60.0),
    "B2": (223.0, 60.0),
    "B3": (285.0, 60.0),
    "B4": (285.0, 0.0),
    "B5": (285.0, -60.0),
    "B6": (223.0, -60.0),
    "B7": (161.0, -60.0),
    "B8": (161.0, 0.0),
    "C1": (192.0, 30.0),
    "C2": (223.0, 30.0),
    "C3": (254.0, 30.0),
    "C4": (254.0, 0.0),
    "C5": (254.0, -30.0),
    "C6": (223.0, -30.0),
    "C7": (192.0, -30.0),
    "C8": (192.0, 0.0),
}


def get_mill_uarm_positions() -> Dict[str, Tuple[float, float]]:
    """Liefert eine defensive Kopie der festen uArm-Koordinaten."""
    return {label: (float(x), float(y)) for label, (x, y) in MILL_UARM_POSITIONS.items()}


__all__ = ["BOARD_LABELS", "MILL_UARM_POSITIONS", "get_mill_uarm_positions"]
