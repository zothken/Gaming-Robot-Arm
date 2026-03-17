"""Persistenz und Datentransfer fuer Launcher-Einstellungen."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from gaming_robot_arm.config import CAMERA_INDEX, UARM_PORT
from gaming_robot_arm.games.mill.core.settings import (
    MILL_ENABLE_FLYING,
    MILL_ENABLE_NO_CAPTURE_DRAW,
    MILL_ENABLE_THREEFOLD_REPETITION,
    MILL_NO_CAPTURE_DRAW_PLIES,
)


@dataclass(slots=True)
class LauncherSettings:
    mode: str = "play-mill"
    camera_index: int = CAMERA_INDEX

    mill_mode: str = "human-vs-ai"
    mill_human_color: str = "W"
    mill_human_input: str = "manual"
    mill_uarm_controlled_players: str = "legacy"
    mill_max_plies: int = 400
    mill_record_game: bool = False

    mill_flying: bool = MILL_ENABLE_FLYING
    mill_threefold_repetition: bool = MILL_ENABLE_THREEFOLD_REPETITION
    mill_no_capture_draw: bool = MILL_ENABLE_NO_CAPTURE_DRAW
    mill_no_capture_draw_plies: int = MILL_NO_CAPTURE_DRAW_PLIES

    mill_ai: str = "alphabeta"
    mill_ai_depth: int = 3
    mill_ai_model: str = "models/champion/mill_champion.pt"
    mill_ai_temperature: float = 0.0
    mill_ai_device: str = "auto"
    mill_random_tiebreak: bool = True
    mill_seed: int = 42

    mill_vision_attempts: int = 6
    mill_debug_vision: bool = False

    mill_uarm_port: str = "" if UARM_PORT is None else str(UARM_PORT)
    mill_uarm_enable_ai_moves: bool = False
    mill_uarm_move_both_players: bool = False
    mill_robot_speed: int = 500
    mill_robot_board_map: str = "default"

    @classmethod
    def from_payload(cls, payload: object) -> "LauncherSettings":
        base = asdict(cls())
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key in base:
                    base[key] = value
        try:
            return cls(**base)
        except TypeError:
            return cls()


def load_launcher_settings(path: Path) -> LauncherSettings:
    if not path.exists():
        return LauncherSettings()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return LauncherSettings()
    return LauncherSettings.from_payload(payload)


def save_launcher_settings(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["LauncherSettings", "load_launcher_settings", "save_launcher_settings"]
