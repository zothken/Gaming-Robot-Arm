"""Persistenz fuer Vision-Detektor-Parameter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


FIGURE_DETECTOR_CONFIG_PATH = Path(__file__).with_name("figure_detector_config.json")
DEFAULT_FIGURE_PARAMS: dict[str, int | float] = {
    "blur_ksize": 11,
    "thresh_block": 89,
    "thresh_c": 15,
    "min_radius": 30,
    "max_radius": 40,
    "hough_dp": 0.2,
    "hough_min_dist": 26,
    "hough_param1": 15,
    "hough_param2": 20,
    "brightness_split": 95,
}


def load_figure_params() -> dict[str, int | float]:
    if not FIGURE_DETECTOR_CONFIG_PATH.exists():
        return dict(DEFAULT_FIGURE_PARAMS)
    try:
        payload = json.loads(FIGURE_DETECTOR_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_FIGURE_PARAMS)
    if not isinstance(payload, dict):
        return dict(DEFAULT_FIGURE_PARAMS)
    merged = dict(DEFAULT_FIGURE_PARAMS)
    for key, value in payload.items():
        if key in merged and isinstance(value, (int, float)):
            merged[key] = value
    return merged


def save_figure_params(params: Mapping[str, int | float]) -> Path:
    serializable = {key: float(value) if isinstance(value, float) else int(value) for key, value in params.items()}
    FIGURE_DETECTOR_CONFIG_PATH.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return FIGURE_DETECTOR_CONFIG_PATH


__all__ = ["DEFAULT_FIGURE_PARAMS", "FIGURE_DETECTOR_CONFIG_PATH", "load_figure_params", "save_figure_params"]
