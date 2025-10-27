"""
config.py – zentrale Konfiguration für das Gaming-Robot-Arm-Projekt
"""

from pathlib import Path

# Basisverzeichnisse
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Kameraeinstellungen
CAMERA_INDEX = 0          # Standardkamera
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RATE = 15

# Datenspeicherung
IMAGE_FORMAT = "jpg"
SAVE_INTERVAL = 5         # Sekunden zwischen automatischen Frames

# uArm Parameter (Startwerte)
UARM_PORT = None          # Wird automatisch erkannt
SAFE_Z = 60               # mm – sichere Höhe über Spielfeld
PICK_Z = 10               # mm – Greifhöhe

# KI / Modellpfade (Platzhalter)
MODEL_PATH = MODEL_DIR / "yolo_muehle.pt"
CONF_THRESHOLD = 0.5
DEVICE = "cpu"

# Allgemeine Einstellungen
DEBUG = True
