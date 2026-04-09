"""Desktop-Launcher als Paket."""

from __future__ import annotations

from pathlib import Path


def launch_launcher(*args, **kwargs):
    from .window import launch_launcher as _launch_launcher

    return _launch_launcher(*args, **kwargs)

__all__ = ["launch_launcher", "Path"]
