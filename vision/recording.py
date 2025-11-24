"""Hilfsfunktionen fuer Kamerastreams und Videoaufzeichnungen."""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional
import sys

# Sicherstellen, dass das Projekt-Root im Python-Pfad liegt,
# falls das Modul direkt ausgefuehrt wird.
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_RATE,
    FRAME_WIDTH,
    IMAGE_FORMAT,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)
from utils.logger import logger

FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
RECORDINGS_DIR = PROJECT_ROOT / "Aufnahmen"


@dataclass
class RecordingSession:
    """Schlanke Kapselung aus Kamerastream und zugehoerigem Video-Writer."""

    camera: cv2.VideoCapture
    writer: cv2.VideoWriter
    output_path: Path
    _prefetched_frame: Optional[np.ndarray] = None

    def read(self) -> np.ndarray:
        """Gibt den naechsten Kameraframe zurueck oder wirft einen Fehler bei Lesefehlern."""
        if self._prefetched_frame is not None:
            frame = self._prefetched_frame
            self._prefetched_frame = None
            return frame

        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")
        return frame

    def write(self, frame: np.ndarray) -> None:
        """Schreibt einen Frame in den aktiven Video-Writer."""
        self.writer.write(frame)


def create_output_dir(base: Optional[Path] = None) -> Path:
    """Stellt das Ausgabe-Verzeichnis fuer Aufnahmen bereit und gibt es zurueck."""
    base_path = Path(base) if base is not None else RECORDINGS_DIR
    base_path.mkdir(parents=True, exist_ok=True)
    logger.debug("Output directory ready: %s", base_path)
    return base_path


@contextmanager
def open_camera(
    camera_index: int = CAMERA_INDEX,
    width: Optional[int] = None,
    height: Optional[int] = None,
    warmup_frames: int = 10,
) -> Iterator[cv2.VideoCapture]:
    """Context-Manager, der einen geoeffneten und vorkonfigurierten Kamerastream bereitstellt."""
    cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam.release()
        cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise RuntimeError(f"Failed to open camera with index {camera_index}.")

    if width is not None:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    actual_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if actual_width <= 0:
        actual_width = width or FRAME_WIDTH
    if actual_height <= 0:
        actual_height = height or FRAME_HEIGHT

    logger.info(
        "Camera %s initialized at %sx%s pixels.",
        camera_index,
        actual_width,
        actual_height,
    )

    if warmup_frames > 0:
        for _ in range(warmup_frames):
            if not cam.grab():
                break

    try:
        yield cam
    finally:
        cam.release()
        logger.info("Camera %s released.", camera_index)


def _create_video_writer(
    cam: cv2.VideoCapture,
    output_dir: Optional[Path] = None,
    fps_override: Optional[float] = None,
) -> tuple[cv2.VideoWriter, Path, Optional[np.ndarray]]:
    """Erzeugt einen Video-Writer fuer den uebergebenen Kamerastream und liefert ihn zurueck."""
    output_base = create_output_dir(output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_base / f"video_{timestamp}.mp4"

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    prefetched_frame: Optional[np.ndarray] = None
    if frame_width <= 0 or frame_height <= 0:
        ret, sample = cam.read()
        if ret:
            frame_height, frame_width = sample.shape[:2]
            prefetched_frame = sample
        if frame_width <= 0 or frame_height <= 0:
            frame_width = FRAME_WIDTH
            frame_height = FRAME_HEIGHT

    fps = fps_override or cam.get(cv2.CAP_PROP_FPS) or FRAME_RATE

    writer = cv2.VideoWriter(
        str(output_path),
        FOURCC,
        float(fps),
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}.")

    logger.info("Recording started: %s", output_path)
    return writer, output_path, prefetched_frame


@contextmanager
def recording_session(
    camera_index: int = CAMERA_INDEX,
    output_dir: Optional[Path] = None,
    fps_override: Optional[float] = None,
) -> Iterator[RecordingSession]:
    """
    Context-Manager, der eine RecordingSession mit Live-Kamera und Video-Writer bereitstellt.

    Gibt beim Verlassen alle Ressourcen frei und schliesst offene OpenCV-Fenster.
    """
    with open_camera(camera_index) as cam:
        writer, output_path, prefetched = _create_video_writer(cam, output_dir, fps_override)
        session = RecordingSession(
            camera=cam,
            writer=writer,
            output_path=output_path,
            _prefetched_frame=prefetched,
        )
        try:
            yield session
        finally:
            writer.release()
            cv2.destroyAllWindows()
            logger.info("Recording session closed: %s", output_path)


def record_dataset(
    output_dir: Path = RAW_DATA_DIR,
    duration: int = 30,
    fps: int = 10,
    camera_index: int = CAMERA_INDEX,
) -> int:
    """Nimmt fuer eine feste Dauer Einzelbilder auf und gibt die gespeicherte Anzahl zurueck."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame_interval = 1.0 / fps
    deadline = time.time() + duration
    next_capture = time.time()
    saved_frames = 0

    with open_camera(camera_index) as cam:
        logger.info(
            "Starting dataset capture: duration=%ss, target_fps=%s, destination=%s",
            duration,
            fps,
            output_path,
        )

        actual_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        while time.time() < deadline:
            now = time.time()
            if now < next_capture:
                time.sleep(min(frame_interval / 4, next_capture - now))
                continue

            ret, frame = cam.read()
            if not ret:
                logger.warning("Failed to read frame during dataset capture; stopping.")
                break

            filename = output_path / f"frame_{saved_frames:04d}.{IMAGE_FORMAT}"
            if actual_width <= 0 or actual_height <= 0:
                actual_height, actual_width = frame.shape[:2]
            if frame.shape[1] != actual_width or frame.shape[0] != actual_height:
                frame = cv2.resize(frame, (actual_width, actual_height), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(str(filename), frame)
            saved_frames += 1
            next_capture += frame_interval

    logger.info("Dataset capture complete. %s frames stored in %s.", saved_frames, output_path)
    return saved_frames


__all__ = [
    "RecordingSession",
    "create_output_dir",
    "open_camera",
    "record_dataset",
    "recording_session",
]


def _run_live_preview(camera_index: int = CAMERA_INDEX, stop_key: str = "q") -> None:
    """Einfacher Main-Loop, der das Live-Kamerabild ausgibt."""
    with open_camera(camera_index) as cam:
        logger.info("Live-Vorschau gestartet (Stop mit Taste '%s').", stop_key)
        while True:
            ret, frame = cam.read()
            if not ret:
                logger.warning("Konnte kein Frame lesen; beende Vorschau.")
                break

            cv2.imshow("Live Preview", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord(stop_key):
                logger.info("Benutzerabbruch per Taste '%s'.", stop_key)
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_live_preview()
