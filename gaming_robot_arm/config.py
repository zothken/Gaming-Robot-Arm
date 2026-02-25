"""
config.py - zentrale Konfiguration fuer das Gaming-Robot-Arm-Projekt
"""

from pathlib import Path

# Basisverzeichnisse
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent  # Projektwurzel (Repository-Wurzel)
CALIBRATION_DIR = PACKAGE_ROOT / "calibration"
RECORDINGS_DIR = PROJECT_ROOT / "Aufnahmen"

# Kalibrierung
HOMOGRAPHY_PATH = CALIBRATION_DIR / "cam_to_robot_homography.json"


# Kameraeinstellungen
CAMERA_INDEX = 0  # Index der Kamera (OpenCV)
FRAME_WIDTH: int | None = 1920  # None = native Kameraaufloesung verwenden
FRAME_HEIGHT: int | None = None  # None = native Kameraaufloesung verwenden
FRAME_RATE: float | None = None  # None = native Kamera-FPS verwenden

# Datenspeicherung
IMAGE_FORMAT = "jpg"  # Dateiformat fuer gespeicherte Bilder
SAVE_INTERVAL = 5  # Sekunden zwischen automatischen Frames

# uArm Parameter (Startwerte)
UARM_PORT = None  # Wird automatisch erkannt
UARM_CALLBACK_THREADS = 1  # Anzahl Callback-Threads der SwiftAPI
SAFE_Z = 15  # mm - sichere Hoehe ueber Spielfeld
PICK_Z = 17.0  # mm - Greifhoehe fuer Figuren
PLACE_Z = 23.0  # mm - Ablagehoehe fuer Figuren
REST_POS = (60.0, -110.0, 70.0)  # Standard-Ruheposition (x, y, z)

# Allgemeine Einstellungen
DEBUG = True  # Debugmodus fuer zusaetzliche Ausgaben
LOG_LEVEL = "DEBUG" if DEBUG else "INFO"  # Protokollstufe fuer Logger

# Linien-Erkennung (Brett-Detektor)
BOARD_LINE_PARAMS = {
    "hough_thresh": 52,
    "min_len_pct": 4,  # Prozent von max(H, W)
    "max_gap": 6,
    "angle_tol": 10,
    "bw_block": 33,
    "bw_C": 10,
    "bw_open": 2,
    "edge_close": 7,
    "blur_ksize": 19,
}

# Mill-Regeln (Backend-Schalter fuer spaeteres Einstellungsmenue)
MILL_ENABLE_FLYING = True
MILL_ENABLE_THREEFOLD_REPETITION = False
MILL_ENABLE_NO_CAPTURE_DRAW = False
MILL_NO_CAPTURE_DRAW_PLIES = 200

# # Linien-Erkennung (Brett-Detektor)
# BOARD_LINE_PARAMS = {
#     "hough_thresh": 59,  # Hough-Schwelle fuer Liniensegmente
#     "min_len_pct": 6,  # Prozent von max(H, W)
#     "max_gap": 7,  # Maximaler Lueckenschluss in Pixeln
#     "angle_tol": 7,  # Winkel-Toleranz fuer H/V-Selektion (Grad)
#     "bw_block": 17,  # Blockgroesse fuer adaptive Schwellwertbildung
#     "bw_C": 13,  # Konstante C fuer adaptive Schwellwertbildung
#     "bw_open": 3,  # Kernelgroesse fuer Oeffnung im BW-Bild
#     "edge_close": 3,  # Kernelgroesse fuer Schliessen der Kanten
#     "blur_ksize": 3,  # Groesse des Gauß-Blur-Kernels (ungerade)
# }



