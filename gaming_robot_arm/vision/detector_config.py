"""Persistenz fuer Vision-Detektor-Parameter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

_VISION_DIR = Path(__file__).parent

FIGURE_DETECTOR_CONFIG_PATH = _VISION_DIR / "figure_detector_config.json"
BOARD_DETECTOR_CONFIG_PATH = _VISION_DIR / "board_detector_config.json"

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

DEFAULT_BOARD_PARAMS: dict[str, int | float] = {
    "blur_ksize": 21,
    "bw_block": 19,
    "bw_C": 10,
    "bw_open": 2,
    "morph_close": 15,
    "approx_eps_pct": 3,
    "min_area_pct": 1,
    "ema_alpha": 40,
}


def _load_params(
    path: Path,
    defaults: dict[str, int | float],
) -> dict[str, int | float]:
    if not path.exists():
        return dict(defaults)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(defaults)
    if not isinstance(payload, dict):
        return dict(defaults)
    merged = dict(defaults)
    for key, value in payload.items():
        if key in merged and isinstance(value, (int, float)):
            merged[key] = value
    return merged


def _save_params(path: Path, params: Mapping[str, int | float]) -> Path:
    serializable = {
        key: float(value) if isinstance(value, float) else int(value)
        for key, value in params.items()
    }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return path


def load_figure_params() -> dict[str, int | float]:
    return _load_params(FIGURE_DETECTOR_CONFIG_PATH, DEFAULT_FIGURE_PARAMS)


def save_figure_params(params: Mapping[str, int | float]) -> Path:
    return _save_params(FIGURE_DETECTOR_CONFIG_PATH, params)


def load_board_params() -> dict[str, int | float]:
    return _load_params(BOARD_DETECTOR_CONFIG_PATH, DEFAULT_BOARD_PARAMS)


def save_board_params(params: Mapping[str, int | float]) -> Path:
    return _save_params(BOARD_DETECTOR_CONFIG_PATH, params)


__all__ = [
    "DEFAULT_BOARD_PARAMS",
    "DEFAULT_FIGURE_PARAMS",
    "BOARD_DETECTOR_CONFIG_PATH",
    "FIGURE_DETECTOR_CONFIG_PATH",
    "load_board_params",
    "load_figure_params",
    "save_board_params",
    "save_figure_params",
]
