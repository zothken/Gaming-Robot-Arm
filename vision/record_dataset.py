"""CLI-Huelle zum Aufzeichnen von Datensatz-Frames ueber die Recording-Hilfsfunktionen."""

import argparse
from pathlib import Path

from config import CAMERA_INDEX, RAW_DATA_DIR
from vision.recording import record_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Einzelbilder fuer die Datensatz-Erstellung aufnehmen.")
    parser.add_argument(
        "--out",
        type=Path,
        default=RAW_DATA_DIR,
        help="Zielverzeichnis fuer die gespeicherten Frames.",
    )
    parser.add_argument("--duration", type=int, default=30, help="Aufnahmedauer in Sekunden.")
    parser.add_argument("--fps", type=int, default=10, help="Aufnahmefrequenz in Frames pro Sekunde.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Zu verwendender Kamera-Index.")
    args = parser.parse_args()

    saved = record_dataset(args.out, args.duration, args.fps, camera_index=args.camera)
    print(f"{saved} Frames in {args.out} gespeichert.")


if __name__ == "__main__":
    main()
