"""Standard-uArm-Positionen fuer Muehle (Nine Men's Morris).

Dieses Modul definiert feste uArm-XY-Koordinaten (in mm) fuer die 24
Brett-Labels, die projektweit genutzt werden (A1-C8). Der mechanische Adapter
fixiert die relative Lage zwischen uArm und Brett, daher koennen diese
Koordinaten konstant bleiben.

Zusaetzlich sind hier die 3x3-Reservekoordinaten je Farbe hinterlegt
(weiss links vom Brett, schwarz rechts vom Brett).

Label-Reihenfolge (entspricht gaming_robot_arm.vision.mill_board_detector):
- A1..A8 = aeusseres Quadrat, im Uhrzeigersinn ab oben links
- B1..B8 = mittleres Quadrat, im Uhrzeigersinn ab oben links
- C1..C8 = inneres Quadrat, im Uhrzeigersinn ab oben links

Passe MILL_UARM_POSITIONS bei Bedarf auf dein vermessenes Setup an.
"""

from __future__ import annotations

from typing import Dict, Tuple

from gaming_robot_arm.games.mill.core.board import BOARD_LABELS

# Standardwerte (mm) fuer das mechanische Adapter-Setup.
# Durch eigene Messwerte ersetzen, falls der Adapter abweicht.
MILL_UARM_POSITIONS: Dict[str, Tuple[float, float]] = {
    "A1": (130.0, 93.0),
    "A2": (223.0, 93.0),
    "A3": (316.0, 93.0),
    "A4": (316.0, 0.0),
    "A5": (316.0, -93.0),
    "A6": (223.0, -93.0),
    "A7": (130.0, -93.0),
    "A8": (130.0, 0.0),
    "B1": (161.0, 62.0),
    "B2": (223.0, 62.0),
    "B3": (285.0, 62.0),
    "B4": (285.0, 0.0),
    "B5": (285.0, -62.0),
    "B6": (223.0, -62.0),
    "B7": (161.0, -62.0),
    "B8": (161.0, 0.0),
    "C1": (192.0,31.0),
    "C2": (223.0,31.0),
    "C3": (254.0,31.0),
    "C4": (254.0, 0.0),
    "C5": (254.0, -31.0),
    "C6": (223.0, -31.0),
    "C7": (192.0, -31.0),
    "C8": (192.0, 0.0),
}

# Vorratskoordinaten fuer Setzzuege (3x3 je Farbe), zeilenweise.
# Weiss: Start (16,116), Rasterabstand 25 mm.
# Schwarz: Spiegelung in -Y mit identischen X-Werten.
MILL_WHITE_RESERVE_POSITIONS: tuple[tuple[float, float], ...] = (
    (16.0, 116.0),
    (16.0, 141.0),
    (16.0, 166.0),
    (41.0, 116.0),
    (41.0, 141.0),
    (41.0, 166.0),
    (66.0, 116.0),
    (66.0, 141.0),
    (66.0, 166.0),
)
MILL_BLACK_RESERVE_POSITIONS: tuple[tuple[float, float], ...] = (
    (16.0, -116.0),
    (16.0, -141.0),
    (16.0, -166.0),
    (41.0, -116.0),
    (41.0, -141.0),
    (41.0, -166.0),
    (66.0, -116.0),
    (66.0, -141.0),
    (66.0, -166.0),
)

# Pickhoehe fuer Reservepositionen (mm).
MILL_RESERVE_PICK_Z = 8.0

# Mill-spezifische Pick/Place-Hoehen (mm) fuer Brettfiguren.
MILL_PICK_Z = 16.0
MILL_PLACE_Z = 20.0


def get_mill_uarm_positions() -> Dict[str, Tuple[float, float]]:
    """Liefert eine defensive Kopie der festen uArm-Koordinaten."""
    return {label: (float(x), float(y)) for label, (x, y) in MILL_UARM_POSITIONS.items()}


__all__ = [
    "BOARD_LABELS",
    "MILL_UARM_POSITIONS",
    "MILL_WHITE_RESERVE_POSITIONS",
    "MILL_BLACK_RESERVE_POSITIONS",
    "MILL_RESERVE_PICK_Z",
    "MILL_PICK_Z",
    "MILL_PLACE_Z",
    "get_mill_uarm_positions",
]
