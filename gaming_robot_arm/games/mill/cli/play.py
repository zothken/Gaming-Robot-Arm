"""CLI-Einstieg fuer spielbare Muehle."""

from __future__ import annotations

from gaming_robot_arm.games.mill.runtime.game_loop import add_mill_cli_arguments, build_parser, main, run_mill_game

__all__ = ["add_mill_cli_arguments", "build_parser", "main", "run_mill_game"]


if __name__ == "__main__":
    raise SystemExit(main())
