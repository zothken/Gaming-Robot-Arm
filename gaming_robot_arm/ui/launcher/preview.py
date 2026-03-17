"""Lazy Loader fuer Kamera-Preview-Integrationen."""

from __future__ import annotations


def load_board_overlay_detector():
    from gaming_robot_arm.vision.mill_board_detector import detect_board_positions

    return detect_board_positions


def load_figure_overlay_detector():
    from gaming_robot_arm.vision.figure_detector import detect_figures

    return detect_figures


def load_board_pixels_loader():
    from gaming_robot_arm.calibration.calibration import load_board_pixels

    return load_board_pixels


__all__ = ["load_board_overlay_detector", "load_board_pixels_loader", "load_figure_overlay_detector"]
