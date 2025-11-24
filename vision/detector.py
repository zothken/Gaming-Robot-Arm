"""
detector.py - Vision-Modul fuer Figuren- und Handerkennung
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from config import CONF_THRESHOLD


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]


class Detector:
    def __init__(self, model_path=None):
        self.model = None  # spaeter: YOLO / TensorFlow-Modell laden
        self.labels = ["piece", "hand"]

    def preprocess(self, frame):
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def detect(self, frame) -> List[Detection]:
        """Dummy-Erkennung: erkennt grosse bewegte Bereiche als 'hand'."""
        processed = self.preprocess(frame)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                detections.append(Detection("hand", 0.6, (x, y, w, h), (cx, cy)))
        return detections

    def annotate(self, frame, detections: List[Detection]):
        for det in detections:
            x, y, w, h = det.bbox
            color = (0, 0, 255) if det.label == "hand" else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{det.label} {det.confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return frame
