"""
record_dataset.py – sammelt Kameraaufnahmen als Datengrundlage
"""
import cv2
import time
import argparse
from pathlib import Path
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, RAW_DATA_DIR, IMAGE_FORMAT

def record_dataset(output_dir: Path, duration: int = 30, fps: int = 10):
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print(f"📹 Starte Aufnahme für {duration} Sekunden … (Speicherort: {output_dir})")
    start_time = time.time()
    count = 0

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        filename = output_dir / f"frame_{count:04d}.{IMAGE_FORMAT}"
        cv2.imwrite(str(filename), frame)
        count += 1
        time.sleep(1 / fps)

    cap.release()
    print(f"✅ Aufnahme abgeschlossen – {count} Bilder gespeichert.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kameraaufnahme für Datensammlung")
    parser.add_argument("--out", type=str, default=str(RAW_DATA_DIR), help="Zielordner")
    parser.add_argument("--duration", type=int, default=30, help="Aufnahmedauer in Sekunden")
    parser.add_argument("--fps", type=int, default=10, help="Bilder pro Sekunde")
    args = parser.parse_args()
    record_dataset(Path(args.out), args.duration, args.fps)
