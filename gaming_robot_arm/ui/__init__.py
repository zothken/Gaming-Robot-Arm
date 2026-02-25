"""Desktop UI launcher for the Gaming Robot Arm project."""

from __future__ import annotations

from pathlib import Path


def launch_launcher(entry_script: Path) -> int:
    from .launcher import launch_launcher as _launch_launcher

    return _launch_launcher(entry_script=entry_script)


__all__ = ["launch_launcher"]
