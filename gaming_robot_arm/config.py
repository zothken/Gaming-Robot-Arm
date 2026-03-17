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
CAMERA_INDEX = 1  # Index der Kamera (OpenCV)
FRAME_WIDTH: int | None = 1920  # None = native Kameraaufloesung verwenden
FRAME_HEIGHT: int | None = 1080  # None = native Kameraaufloesung verwenden
FRAME_RATE: float | None = 30.0  # None = native Kamera-FPS verwenden

# Datenspeicherung
IMAGE_FORMAT = "jpg"  # Dateiformat fuer gespeicherte Bilder
SAVE_INTERVAL = 5  # Sekunden zwischen automatischen Frames

# uArm Parameter (Startwerte)
UARM_PORT = None  # Wird automatisch erkannt
UARM_CALLBACK_THREADS = 1  # Anzahl Callback-Threads der SwiftAPI
SAFE_Z = 80  # mm - sichere Hoehe ueber Spielfeld
REST_POS = (60.0, 110.0, 70.0)  # Standard-Ruheposition (x, y, z)

# Allgemeine Einstellungen
DEBUG = True  # Debugmodus fuer zusaetzliche Ausgaben
LOG_LEVEL = "DEBUG" if DEBUG else "INFO"  # Protokollstufe fuer Logger

