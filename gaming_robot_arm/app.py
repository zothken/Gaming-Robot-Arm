"""Paketinterner Haupteinstieg fuer Gaming Robot Arm."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gaming_robot_arm import VisionControlRuntime
from gaming_robot_arm.config import CAMERA_INDEX


APP_MODES = ("ui", "vision-loop", "play-mill")


def _run_vision_loop(camera_index: int = CAMERA_INDEX) -> int:
    with VisionControlRuntime() as runtime:
        runtime.run(camera_index=camera_index)
    return 0


def _requested_mode(argv: Sequence[str] | None) -> str:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--mode", default="ui")
    args, _unknown = probe.parse_known_args(argv)
    return str(args.mode)


def build_parser(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Starter fuer Gaming Robot Arm.")
    parser.add_argument(
        "--mode",
        choices=APP_MODES,
        default="ui",
        help="Startet UI-Launcher, Vision-Loop oder den spielbaren Muehle-Loop.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=CAMERA_INDEX,
        help="Kamera-Index fuer die Vision-Funktionen.",
    )

    if _requested_mode(argv) == "play-mill":
        from gaming_robot_arm.games.mill.cli.play import add_mill_cli_arguments

        add_mill_cli_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(argv)
    args = parser.parse_args(argv)

    if args.mode == "ui":
        try:
            from gaming_robot_arm.ui import launch_launcher

            return launch_launcher(entry_script=Path(__file__).resolve())
        except ModuleNotFoundError as exc:
            if exc.name == "PySide6":
                print(f"UI-Launcher konnte nicht geladen werden: {exc}")
                print("Hinweis: Fuer die Desktop-UI PySide6 installieren (z.B. `pip install -e .[ui]`).")
                return 1
            print(f"UI-Launcher konnte nicht geladen werden: {exc}")
            return 1
        except Exception as exc:
            print(f"UI-Launcher konnte nicht geladen werden: {exc}")
            return 1

    if args.mode == "vision-loop":
        return _run_vision_loop(camera_index=args.camera_index)

    if args.mode == "play-mill":
        from gaming_robot_arm.games.mill.cli.play import run_mill_game

        return run_mill_game(args)

    raise ValueError(f"Nicht unterstuetzter Modus: {args.mode}")


__all__ = ["APP_MODES", "build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
