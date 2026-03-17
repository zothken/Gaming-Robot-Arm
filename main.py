"""Hauptstartpunkt fuer Laufzeit- und spielbare Spielmodi."""

from __future__ import annotations

from gaming_robot_arm.app import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
