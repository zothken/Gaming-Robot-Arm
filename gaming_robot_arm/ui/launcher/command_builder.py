"""Kommandoaufbau fuer den Desktop-Launcher."""

from __future__ import annotations

from pathlib import Path

from .settings import LauncherSettings


def _as_int(value: object, label: str, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} muss eine ganze Zahl sein") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{label} muss >= {minimum} sein")
    return parsed


def _as_float(value: object, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} muss eine Zahl sein") from exc


def build_command(settings: LauncherSettings, *, python_executable: str, entry_script: Path) -> list[str]:
    mode = str(settings.mode).strip()
    if mode not in {"vision-loop", "play-mill"}:
        raise ValueError("Modus muss 'vision-loop' oder 'play-mill' sein")

    cmd = [
        python_executable,
        "-u",
        str(entry_script),
        "--mode",
        mode,
        "--camera-index",
        str(_as_int(settings.camera_index, "Kameraindex", minimum=0)),
    ]
    if mode == "vision-loop":
        return cmd

    mill_mode = str(settings.mill_mode).strip()
    human_color = str(settings.mill_human_color).strip()
    human_input = str(settings.mill_human_input).strip()
    uarm_controlled_players = str(settings.mill_uarm_controlled_players).strip().lower()
    ai_backend = str(settings.mill_ai).strip()
    ai_model = str(settings.mill_ai_model).strip()
    ai_device = str(settings.mill_ai_device).strip()
    robot_board_map = str(settings.mill_robot_board_map).strip()

    if mill_mode not in {"human-vs-human", "human-vs-ai", "ai-vs-ai"}:
        raise ValueError("Ungültiger Mühle-Spielmodus")
    if human_color not in {"W", "B"}:
        raise ValueError("Mensch-Farbe muss W oder B sein")
    if human_input not in {"manual", "vision"}:
        raise ValueError("Mensch-Eingabe muss 'manual' oder 'vision' sein")
    if uarm_controlled_players not in {"none", "white", "black", "both", "legacy"}:
        raise ValueError("uArm-Support muss none, white, black, both oder legacy sein")
    if ai_backend not in {"heuristic", "alphabeta", "neural"}:
        raise ValueError("KI-Backend muss heuristic, alphabeta oder neural sein")
    if robot_board_map not in {"default", "homography"}:
        raise ValueError("Brett-Mapping muss default oder homography sein")

    def add_bool(flag: str, enabled: bool) -> None:
        cmd.append(f"--{flag}" if enabled else f"--no-{flag}")

    cmd.extend(["--game-mode", mill_mode])
    cmd.extend(["--human-color", human_color])
    cmd.extend(["--human-input", human_input])
    cmd.extend(["--max-plies", str(_as_int(settings.mill_max_plies, "Max. Halbzüge", minimum=0))])
    add_bool("record-game", bool(settings.mill_record_game))
    add_bool("flying", bool(settings.mill_flying))
    add_bool("threefold-repetition", bool(settings.mill_threefold_repetition))
    add_bool("no-capture-draw", bool(settings.mill_no_capture_draw))
    cmd.extend(
        [
            "--no-capture-draw-plies",
            str(_as_int(settings.mill_no_capture_draw_plies, "Remisgrenze (Halbzüge)", minimum=1)),
        ]
    )

    cmd.extend(["--ai", ai_backend])
    cmd.extend(["--ai-depth", str(_as_int(settings.mill_ai_depth, "AlphaBeta-Tiefe", minimum=1))])
    if ai_model:
        cmd.extend(["--ai-model", ai_model])
    cmd.extend(["--ai-temperature", str(_as_float(settings.mill_ai_temperature, "Temperatur"))])
    if ai_device:
        cmd.extend(["--ai-device", ai_device])
    add_bool("random-tiebreak", bool(settings.mill_random_tiebreak))
    cmd.extend(["--seed", str(_as_int(settings.mill_seed, "Seed"))])

    cmd.extend(["--vision-attempts", str(_as_int(settings.mill_vision_attempts, "Scan-Versuche", minimum=1))])
    add_bool("debug-vision", bool(settings.mill_debug_vision))

    uarm_port = str(settings.mill_uarm_port).strip()
    if uarm_port:
        cmd.extend(["--uarm-port", uarm_port])
    cmd.extend(["--uarm-controlled-players", uarm_controlled_players])
    add_bool("uarm-enable-ai-moves", bool(settings.mill_uarm_enable_ai_moves))
    add_bool("uarm-move-both-players", bool(settings.mill_uarm_move_both_players))
    cmd.extend(["--robot-speed", str(_as_int(settings.mill_robot_speed, "Robotergeschwindigkeit", minimum=1))])
    cmd.extend(["--robot-board-map", robot_board_map])
    return cmd


__all__ = ["build_command"]
