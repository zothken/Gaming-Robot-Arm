"""
config.py - zentrale Konfiguration fuer das Gaming-Robot-Arm-Projekt
"""

from pathlib import Path

# Basisverzeichnisse
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"  # Ziel fuer aufgezeichnete Einzelbilder

# Kameraeinstellungen
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RATE = 20

# Datenspeicherung
IMAGE_FORMAT = "jpg"
SAVE_INTERVAL = 5  # Sekunden zwischen automatischen Frames

# uArm Parameter (Startwerte)
UARM_PORT = None  # Wird automatisch erkannt
UARM_CALLBACK_THREADS = 1  # Anzahl Callback-Threads der SwiftAPI
SAFE_Z = 60  # mm - sichere Hoehe ueber Spielfeld
PICK_Z = 8.0  # mm - Greifhoehe fuer Figuren
PLACE_Z = 15.0  # mm - Ablagehoehe fuer Figuren
REST_POS = (10.0, 150.0, 100.0)  # Standard-Ruheposition (x, y, z)

# Arbeitsbereich des uArms (mm)
UARM_MIN_RADIUS = 60.0  # Naeher an der Basis drohen Kollisionen
UARM_MAX_RADIUS = 320.0  # Mechanische Reichweite laut Datenblatt

# Allgemeine Einstellungen
DEBUG = True
CONF_THRESHOLD = 0.5  # Mindestvertrauen fuer (zukuenftige) ML-basierte Detektoren
