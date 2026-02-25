"""Hilfsfunktionen fuer Kamerastreams und Videoaufzeichnungen."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from gaming_robot_arm.config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_RATE,
    FRAME_WIDTH,
    RECORDINGS_DIR,
)
from gaming_robot_arm.utils.logger import logger

FOURCC = cv2.VideoWriter_fourcc(*"MJPG")

FALLBACK_FRAME_WIDTH = 1920
FALLBACK_FRAME_HEIGHT = 1080
FALLBACK_FRAME_RATE = 30.0


def _fallback_dim(*values: Optional[int], default: int) -> int:
    for value in values:
        if value is None:
            continue
        candidate = int(value)
        if candidate > 0:
            return candidate
    return default


def _fallback_fps(*values: Optional[float], default: float) -> float:
    for value in values:
        if value is None:
            continue
        candidate = float(value)
        if candidate > 0:
            return candidate
    return float(default)


def get_effective_camera_fps(
    cam: cv2.VideoCapture,
    *,
    override_fps: Optional[float] = None,
    configured_fps: Optional[float] = None,
) -> float:
    """Liefert eine robuste FPS-Schaetzung fuer Fenster/Writer (0er-Werte werden ignoriert)."""
    if configured_fps is None:
        configured_fps = FRAME_RATE
    cam_fps = cam.get(cv2.CAP_PROP_FPS) or 0.0
    return _fallback_fps(override_fps, cam_fps, configured_fps, default=FALLBACK_FRAME_RATE)


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
            raise RuntimeError("Konnte Kameraframe nicht lesen.")
        return frame

    def write(self, frame: np.ndarray) -> None:
        """Schreibt einen Frame in den aktiven Video-Writer."""
        self.writer.write(frame)


def create_output_dir(base: Optional[Path] = None) -> Path:
    """Stellt das Ausgabe-Verzeichnis fuer Aufnahmen bereit und gibt es zurueck."""
    base_path = Path(base) if base is not None else RECORDINGS_DIR
    base_path.mkdir(parents=True, exist_ok=True)
    logger.debug("Ausgabeverzeichnis bereit: %s", base_path)
    return base_path


@contextmanager
def open_camera(
    camera_index: int = CAMERA_INDEX,
    width: Optional[int] = FRAME_WIDTH,
    height: Optional[int] = FRAME_HEIGHT,
    warmup_frames: int = 10,
) -> Iterator[cv2.VideoCapture]:
    """Kontextmanager, der einen geoeffneten und vorkonfigurierten Kamerastream bereitstellt."""
    cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam.release()
        cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise RuntimeError(f"Konnte Kamera mit Index {camera_index} nicht oeffnen.")

    requested_width = width
    requested_height = height

    def _apply_resolution(target_width: int, target_height: int) -> tuple[int, int]:
        if target_width >= 1280 or target_height >= 720:
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(target_width))
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(target_height))
        actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return actual_w, actual_h

    # Viele Webcams liefern hohe Aufloesungen nur mit komprimiertem Stream (z.B. MJPG).
    # Daher FourCC vor der Groessensetzung versuchen.
    if requested_width is not None and requested_height is not None:
        actual_width, actual_height = _apply_resolution(requested_width, requested_height)
    else:
        actual_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if FRAME_RATE is not None and float(FRAME_RATE) > 0:
        cam.set(cv2.CAP_PROP_FPS, float(FRAME_RATE))
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Fallback: wenn die angeforderte Aufloesung offensichtlich nicht uebernommen wurde,
    # probiere 720p als naechstbesten HD-Modus.
    if requested_width is not None and requested_height is not None:
        if (actual_width > 0 and actual_height > 0) and (actual_width < 1280 or actual_height < 720):
            fallback_w, fallback_h = 1280, 720
            if (requested_width, requested_height) != (fallback_w, fallback_h):
                actual_width, actual_height = _apply_resolution(fallback_w, fallback_h)

    if actual_width <= 0:
        actual_width = _fallback_dim(requested_width, FRAME_WIDTH, default=FALLBACK_FRAME_WIDTH)
    if actual_height <= 0:
        actual_height = _fallback_dim(requested_height, FRAME_HEIGHT, default=FALLBACK_FRAME_HEIGHT)

    if requested_width is not None and requested_height is not None:
        if actual_width != requested_width or actual_height != requested_height:
            logger.warning(
                "Kamera %s mit %sx%s Pixeln initialisiert (angefordert: %sx%s).",
                camera_index,
                actual_width,
                actual_height,
                requested_width,
                requested_height,
            )
        else:
            logger.info(
                "Kamera %s mit %sx%s Pixeln initialisiert.",
                camera_index,
                actual_width,
                actual_height,
            )
    else:
        logger.info("Kamera %s mit %sx%s Pixeln initialisiert.", camera_index, actual_width, actual_height)

    if warmup_frames > 0:
        for _ in range(warmup_frames):
            if not cam.grab():
                break

    try:
        yield cam
    finally:
        cam.release()
        logger.info("Kamera %s freigegeben.", camera_index)


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
            frame_width = _fallback_dim(FRAME_WIDTH, default=FALLBACK_FRAME_WIDTH)
            frame_height = _fallback_dim(FRAME_HEIGHT, default=FALLBACK_FRAME_HEIGHT)

    fps = get_effective_camera_fps(cam, override_fps=fps_override)

    writer = cv2.VideoWriter(
        str(output_path),
        FOURCC,
        float(fps),
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Konnte Video-Writer unter {output_path} nicht oeffnen.")

    logger.info("Aufnahme gestartet: %s", output_path)
    return writer, output_path, prefetched_frame


@contextmanager
def recording_session(
    camera_index: int = CAMERA_INDEX,
    output_dir: Optional[Path] = None,
    fps_override: Optional[float] = None,
) -> Iterator[RecordingSession]:
    """
    Kontextmanager, der eine RecordingSession mit Live-Kamera und Video-Writer bereitstellt.

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
            logger.info("Aufnahmesitzung geschlossen: %s", output_path)


__all__ = [
    "RecordingSession",
    "create_output_dir",
    "get_effective_camera_fps",
    "open_camera",
    "recording_session",
]


def _run_live_preview(camera_index: int = CAMERA_INDEX, stop_key: str = "q") -> None:
    """Einfacher Main-Loop, der das Live-Kamerabild ausgibt."""
    win = "Live-Vorschau"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    with open_camera(camera_index) as cam:
        logger.info("Live-Vorschau gestartet (Stopp mit Taste '%s').", stop_key)
        window_sized = False
        while True:
            ret, frame = cam.read()
            if not ret:
                logger.warning("Konnte kein Frame lesen; beende Vorschau.")
                break

            if not window_sized:
                h, w = frame.shape[:2]
                max_w, max_h = 1280, 720
                scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
                cv2.resizeWindow(win, int(round(w * scale)), int(round(h * scale)))
                window_sized = True

            cv2.imshow(win, frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord(stop_key):
                logger.info("Benutzerabbruch per Taste '%s'.", stop_key)
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_live_preview()
