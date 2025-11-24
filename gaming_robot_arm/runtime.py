"""Laufzeit-Orchestrierung zwischen Vision- und Robotersteuerungs-Subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

from config import CAMERA_INDEX
from tracker import CentroidTracker
from utils.logger import logger
from vision.figure_detector import detect_figures
from vision.recording import recording_session
from vision.visualization import show_frames

from control import UArmController

FrameHandler = Callable[["VisionArtifacts", CentroidTracker, Optional[UArmController]], None]
StopCondition = Callable[[CentroidTracker, Optional[UArmController]], bool]


@dataclass(slots=True)
class VisionArtifacts:
    """Container mit Zwischenframes der Vision-Pipeline fuer nachgelagerte Komponenten."""

    processed: np.ndarray
    gray: np.ndarray
    blurred: np.ndarray
    thresh: np.ndarray


class VisionControlRuntime:
    """Koordiniert die Kamera-/Vision-Schleife und optional den Roboter-Controller."""

    def __init__(
        self,
        *,
        tracker: Optional[CentroidTracker] = None,
        controller: Optional[UArmController] = None,
        frame_handlers: Optional[Iterable[FrameHandler]] = None,
        display: bool = True,
    ) -> None:
        self.tracker = tracker or CentroidTracker()
        self.controller = controller
        self.display = display
        self._frame_handlers = list(frame_handlers or [])
        self._last_recording: Optional[Path] = None

    def __enter__(self) -> "VisionControlRuntime":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.shutdown()

    @property
    def last_recording(self) -> Optional[Path]:
        """Gibt den Pfad der juengsten Aufzeichnung zurueck, falls vorhanden."""
        return self._last_recording

    def add_handler(self, handler: FrameHandler) -> None:
        """Registriert eine Callback-Funktion fuer jeden verarbeiteten Frame."""
        self._frame_handlers.append(handler)

    def ensure_controller(self, *, port: Optional[str] = None, **swift_options) -> UArmController:
        """
        Erstellt den Roboter-Controller bei Bedarf und stellt die Verbindung her.

        Zusaetzliche Keyword-Argumente werden an die SwiftAPI weitergereicht.
        """
        if self.controller is None:
            self.controller = UArmController(port=port, do_connect=False, **swift_options)

        if self.controller.swift is None:
            self.controller.connect(port=port, **swift_options)

        return self.controller

    def shutdown(self) -> None:
        """Trennt den Controller, sofern eine Verbindung besteht."""
        if self.controller:
            self.controller.disconnect()
            self.controller = None

    def run(
        self,
        *,
        camera_index: int = CAMERA_INDEX,
        stop_key: Optional[str] = "q",
        stop_condition: Optional[StopCondition] = None,
    ) -> Optional[Path]:
        """
        Fuehrt die Vision-Hauptschleife aus, bis ein Stopp-Ereignis ausgeloest wird.

        Liefert den Pfad der aufgezeichneten Videodatei oder None bei fehlender Aufzeichnung.
        """
        logger.info("Starting Gaming Robot Arm runtime on camera index %s.", camera_index)

        try:
            with recording_session(camera_index=camera_index) as session:
                self._last_recording = session.output_path

                while True:
                    try:
                        frame = session.read()
                    except RuntimeError as exc:
                        logger.warning("Stopping capture due to camera error: %s", exc)
                        break

                    processed, gray, blurred, thresh = detect_figures(frame, tracker=self.tracker)
                    artifacts = VisionArtifacts(
                        processed=processed,
                        gray=gray,
                        blurred=blurred,
                        thresh=thresh,
                    )

                    if self.display:
                        show_frames(
                            {
                                "Grayscale": gray,
                                "Blurred": blurred,
                                "Thresholded": thresh,
                                "Processed": processed,
                            }
                        )

                    session.write(processed)
                    self._notify_handlers(artifacts)

                    if stop_condition and stop_condition(self.tracker, self.controller):
                        logger.info("Stop condition triggered; exiting runtime loop.")
                        break

                    if self.display and stop_key:
                        key = cv2.waitKey(1)
                        if key & 0xFF == ord(stop_key):
                            logger.info("Exit requested by user via key '%s'.", stop_key)
                            break
        except RuntimeError as exc:
            logger.error("Unable to start recording session: %s", exc)
            self._last_recording = None
            return None

        if self._last_recording:
            logger.info("Recording finished. Video stored at %s", self._last_recording)
        else:
            logger.info("Recording finished without video output.")

        return self._last_recording

    def _notify_handlers(self, artifacts: VisionArtifacts) -> None:
        """Ruft registrierte Frame-Handler mit robuster Fehlerbehandlung auf."""
        for handler in self._frame_handlers:
            try:
                handler(artifacts, self.tracker, self.controller)
            except Exception:  # pragma: no cover - handler failures should not stop the loop
                logger.exception("Frame handler %r raised an exception; continuing.", handler)
