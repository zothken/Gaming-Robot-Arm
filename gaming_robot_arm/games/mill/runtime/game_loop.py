"""Spielbare Muehle-Schleife mit optionaler Vision- und Roboterintegration."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from gaming_robot_arm.config import (
    AUTOSAVE_PATH,
    CAMERA_INDEX,
    UARM_PORT,
)
from gaming_robot_arm.games.common.interfaces import Move, Player
from gaming_robot_arm.games.mill import (
    MillGameSession,
    MillRuleSettings,
    MillRules,
)
from gaming_robot_arm.games.mill.core.state import MillState
from gaming_robot_arm.games.mill.core.rules import phase_for_player
from gaming_robot_arm.games.mill.core.settings import (
    MILL_ENABLE_FLYING,
    MILL_ENABLE_NO_CAPTURE_DRAW,
    MILL_ENABLE_THREEFOLD_REPETITION,
    MILL_NO_CAPTURE_DRAW_PLIES,
)
from gaming_robot_arm.logger import logger
from .players import (
    AI_BACKENDS,
    GAME_MODES,
    HUMAN_INPUT_MODES,
    UARM_CONTROLLED_PLAYERS,
    PlayerController,
    build_player_controllers,
    require_ai_provider,
    resolve_uarm_players,
)
from .robot_bridge import (
    ROBOT_BOARD_MAPS,
    MillRobotBridge,
    build_default_reserve_positions,
    load_robot_board_positions,
)
from .vision_bridge import (
    AUTO_BASELINE_TIMEOUT_S,
    MillVisionBridge,
    VisionTriggerMode,
    _LiveVisionSession,
    _VisionPreviewSession,
    infer_moves_from_observation,
)
from .voice_bridge import VoiceBridge, VOICE_MOVE_TIMEOUT_S
from .signals import UndoSignal

if TYPE_CHECKING:
    from gaming_robot_arm.vision.recording import RecordingSession


def add_mill_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX, help="Kamera-Index fuer den Vision-Modus.")
    game_group = parser.add_argument_group("Game/rule settings")
    game_group.add_argument(
        "--game-mode",
        dest="mill_mode",
        choices=GAME_MODES,
        default="human-vs-ai",
        help="Spielmodus: human-vs-human, human-vs-ai oder ai-vs-ai.",
    )
    game_group.add_argument(
        "--human-color",
        dest="mill_human_color",
        choices=("W", "B"),
        default="W",
        help="Menschliche Seite bei --game-mode=human-vs-ai.",
    )
    game_group.add_argument(
        "--human-input",
        dest="mill_human_input",
        choices=HUMAN_INPUT_MODES,
        default="manual",
        help="Quelle fuer menschliche Zuege: Terminaleingabe oder Vision-Inferenz.",
    )
    game_group.add_argument(
        "--max-plies",
        dest="mill_max_plies",
        type=int,
        default=400,
        help="Sicherheitsgrenze fuer Halbzuege pro Partie (0 = keine Begrenzung).",
    )

    game_group.add_argument(
        "--flying",
        dest="mill_flying",
        action=argparse.BooleanOptionalAction,
        default=MILL_ENABLE_FLYING,
        help="Aktiviert Flying-Regel, wenn eine Seite drei Steine hat.",
    )
    game_group.add_argument(
        "--threefold-repetition",
        dest="mill_threefold_repetition",
        action=argparse.BooleanOptionalAction,
        default=MILL_ENABLE_THREEFOLD_REPETITION,
        help="Aktiviert Remisregel bei Dreifachwiederholung.",
    )
    game_group.add_argument(
        "--no-capture-draw",
        dest="mill_no_capture_draw",
        action=argparse.BooleanOptionalAction,
        default=MILL_ENABLE_NO_CAPTURE_DRAW,
        help="Aktiviert Remis nach langer Folge ohne Schlag.",
    )
    game_group.add_argument(
        "--no-capture-draw-plies",
        dest="mill_no_capture_draw_plies",
        type=int,
        default=MILL_NO_CAPTURE_DRAW_PLIES,
        help="Remis-Schwelle ohne Schlag in Halbzuegen.",
    )

    ai_group = parser.add_argument_group("AI settings")
    ai_group.add_argument(
        "--ai",
        dest="mill_ai",
        choices=AI_BACKENDS,
        default="alphabeta",
        help="KI-Backend fuer KI-Zuege.",
    )
    ai_group.add_argument(
        "--ai-depth",
        dest="mill_ai_depth",
        type=int,
        default=3,
        help="Suchtiefe fuer AlphaBeta-KI.",
    )
    ai_group.add_argument(
        "--random-tiebreak",
        dest="mill_random_tiebreak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aktiviert zufaellige Tie-Breaks bei gleicher Zugbewertung.",
    )
    ai_group.add_argument(
        "--seed",
        dest="mill_seed",
        type=int,
        default=42,
        help="Basis-Zufallsseed fuer KI-Provider.",
    )

    other_group = parser.add_argument_group("Other settings (uArm, logging, coordinates)")
    other_group.add_argument(
        "--vision-attempts",
        dest="mill_vision_attempts",
        type=int,
        default=6,
        help="Frame-Versuche pro Vision-Scan.",
    )
    other_group.add_argument(
        "--debug-vision",
        dest="mill_debug_vision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Aktiviert ausfuehrliches Logging der Vision-Zuordnung.",
    )
    other_group.add_argument(
        "--vision-preview",
        dest="mill_vision_preview",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Zeigt ein separates Fenster mit Live-Kamera und Figuren-Detektor-Overlay waehrend der Partie.",
    )
    other_group.add_argument(
        "--vision-trigger",
        dest="mill_vision_trigger",
        choices=("manual", "auto"),
        default="auto",
        help="Ausloeser fuer Vision-Zuege: manueller Scan per Enter oder automatischer Trigger auf Brettaenderung.",
    )
    other_group.add_argument(
        "--baseline-timeout-disabled",
        dest="mill_baseline_timeout_disabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deaktiviert den Baseline-Timeout beim Warten auf ein ruhiges Brett.",
    )
    other_group.add_argument(
        "--voice-move-timeout-disabled",
        dest="mill_voice_move_timeout_disabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deaktiviert den 60s-Timeout der Spracheingabe. Wartet unbegrenzt auf einen gueltigen Zug.",
    )
    other_group.add_argument(
        "--pre-move-vision-gate",
        dest="mill_pre_move_vision_gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wartet vor uArm-Bewegungen auf ein ruhiges Kamerabild (nur wenn Vision aktiv).",
    )
    other_group.add_argument(
        "--pre-move-quiet-timeout",
        dest="mill_pre_move_quiet_timeout",
        type=float,
        default=10.0,
        help="Maximale Wartezeit auf ein ruhiges Bild in Sekunden.",
    )
    other_group.add_argument(
        "--pre-move-delay",
        dest="mill_pre_move_delay",
        type=float,
        default=2.0,
        help="Feste Pause vor uArm-Bewegung (Fallback ohne Vision oder bei Gate-Timeout, 0 = aus).",
    )

    other_group.add_argument(
        "--uarm-port",
        dest="mill_uarm_port",
        type=str,
        default=UARM_PORT,
        help="Optionaler serieller Port fuer uArm.",
    )
    other_group.add_argument(
        "--record-game",
        dest="mill_record_game",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Nimmt die laufende Partie als Video auf (Datei unter Aufnahmen/).",
    )
    other_group.add_argument(
        "--uarm-enable-ai-moves",
        dest="mill_uarm_enable_ai_moves",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy-Flag: aktiviert Roboterausfuehrung fuer KI-Zuege.",
    )
    other_group.add_argument(
        "--uarm-move-both-players",
        dest="mill_uarm_move_both_players",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy-Flag: laesst den uArm Zuege beider Seiten ausfuehren.",
    )
    other_group.add_argument(
        "--uarm-controlled-players",
        dest="mill_uarm_controlled_players",
        choices=UARM_CONTROLLED_PLAYERS,
        default="legacy",
        help="Steuert, welche Farbe physisch vom uArm bewegt wird: none|white|black|both|legacy.",
    )
    other_group.add_argument(
        "--robot-speed",
        dest="mill_robot_speed",
        type=int,
        default=500,
        help="uArm-Bewegungsgeschwindigkeit fuer Greif-/Ablagevorgaenge.",
    )
    other_group.add_argument(
        "--robot-board-map",
        dest="mill_robot_board_map",
        choices=ROBOT_BOARD_MAPS,
        default="homography",
        help="Quelle der Roboter-Brettkoordinaten: feste Standardwerte oder Homography-Projektion.",
    )

    resume_group = parser.add_argument_group("Resume settings")
    resume_group.add_argument(
        "--resume",
        dest="mill_resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Setzt das Spiel vom letzten gespeicherten Spielstand fort (falls vorhanden).",
    )
    resume_group.add_argument(
        "--restore-board",
        dest="mill_restore_board",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Laesst den uArm den physischen Spielstand wiederherstellen (Figuren aus Reserve auf Felder). Erfordert --resume.",
    )


def _format_move(move: Move) -> str:
    src = move.src if move.src is not None else "VORRAT"
    capture = f" x {move.capture}" if move.capture is not None else ""
    return f"{move.player}: {src} -> {move.dst}{capture}"


def _emit_pre_move_warning(
    move: Move,
    *,
    vision_bridge: MillVisionBridge | None,
    vision_session: "RecordingSession | _LiveVisionSession | None",
    use_vision_gate: bool,
    quiet_timeout_s: float,
    fallback_delay_s: float,
    kind: str = "Zug",
) -> None:
    print(f"Warnung: uArm bewegt sich gleich ({kind} {_format_move(move)}) - Brettbereich freihalten!")

    gate_succeeded = False
    if use_vision_gate and vision_bridge is not None and vision_session is not None:
        gate_succeeded = vision_bridge.wait_for_quiet_scene(
            session=vision_session,
            timeout_s=quiet_timeout_s,
            status_callback=print,
        )

    if not gate_succeeded and fallback_delay_s > 0:
        time.sleep(fallback_delay_s)


def _format_board(board: dict[str, Player | None]) -> str:
    def node(label: str) -> str:
        owner = board.get(label)
        piece = owner if owner in {"W", "B"} else "O"
        return f"[{label}/{piece}]"

    positions: dict[str, tuple[int, int]] = {
        "A1": (0, 0),
        "A2": (0, 22),
        "A3": (0, 44),
        "B1": (2, 7),
        "B2": (2, 22),
        "B3": (2, 37),
        "C1": (4, 14),
        "C2": (4, 22),
        "C3": (4, 30),
        "A8": (6, 0),
        "B8": (6, 7),
        "C8": (6, 14),
        "C4": (6, 30),
        "B4": (6, 37),
        "A4": (6, 44),
        "C7": (8, 14),
        "C6": (8, 22),
        "C5": (8, 30),
        "B7": (10, 7),
        "B6": (10, 22),
        "B5": (10, 37),
        "A7": (12, 0),
        "A6": (12, 22),
        "A5": (12, 44),
    }

    height, width = 13, 50
    canvas = [[" "] * width for _ in range(height)]

    def center_x(label: str) -> int:
        return positions[label][1] + 3

    def draw_horizontal(left: str, right: str) -> None:
        row, left_col = positions[left]
        _, right_col = positions[right]
        start = min(left_col, right_col) + 6
        end = max(left_col, right_col) - 1
        for x in range(start, end + 1):
            canvas[row][x] = "-"

    def draw_vertical(top: str, bottom: str) -> None:
        top_row, _ = positions[top]
        bottom_row, _ = positions[bottom]
        x = center_x(top)
        start = min(top_row, bottom_row) + 1
        end = max(top_row, bottom_row) - 1
        for y in range(start, end + 1):
            canvas[y][x] = "|"

    for left, right in (
        ("A1", "A2"),
        ("A2", "A3"),
        ("B1", "B2"),
        ("B2", "B3"),
        ("C1", "C2"),
        ("C2", "C3"),
        ("A8", "B8"),
        ("B8", "C8"),
        ("C4", "B4"),
        ("B4", "A4"),
        ("C7", "C6"),
        ("C6", "C5"),
        ("B7", "B6"),
        ("B6", "B5"),
        ("A7", "A6"),
        ("A6", "A5"),
    ):
        draw_horizontal(left, right)

    for top, bottom in (
        ("A1", "A8"),
        ("A8", "A7"),
        ("A3", "A4"),
        ("A4", "A5"),
        ("B1", "B8"),
        ("B8", "B7"),
        ("B3", "B4"),
        ("B4", "B5"),
        ("C1", "C8"),
        ("C8", "C7"),
        ("C3", "C4"),
        ("C4", "C5"),
        ("A2", "B2"),
        ("B2", "C2"),
        ("C6", "B6"),
        ("B6", "A6"),
    ):
        draw_vertical(top, bottom)

    for label, (row, col) in positions.items():
        token = node(label)
        for idx, ch in enumerate(token):
            canvas[row][col + idx] = ch

    return "\n".join("".join(row).rstrip() for row in canvas)


def _prompt_human_move(legal_moves: Sequence[Move]) -> "Move | UndoSignal":
    print("Legale Zuege:")
    for idx, move in enumerate(legal_moves, start=1):
        print(f"  [{idx:02d}] {_format_move(move)}")

    while True:
        raw = _read_user_input(
            "Zugnummer waehlen, 'z' fuer Zurueck, oder 'q' zum Abbrechen: "
        ).strip().lower()
        if raw == "q":
            raise KeyboardInterrupt
        if raw in ("z", "zurueck", "zurück", "undo"):
            return UndoSignal()
        try:
            choice = int(raw)
        except ValueError:
            print("Bitte eine numerische Zugnummer, 'z' oder 'q' eingeben.")
            continue
        if 1 <= choice <= len(legal_moves):
            return legal_moves[choice - 1]
        print(f"Auswahl ausserhalb des Bereichs (1..{len(legal_moves)}).")


def _record_single_frame(session: RecordingSession | None) -> None:
    if session is None:
        return
    try:
        frame = session.read()
    except Exception as exc:
        logger.warning("Konnte keinen Frame fuer Spielaufzeichnung lesen: %s", exc)
        return
    session.write(frame)


def _read_user_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError as exc:
        raise KeyboardInterrupt from exc


def _scan_human_move_via_vision(
    *,
    session: MillGameSession,
    vision_bridge: MillVisionBridge,
    legal_moves: Sequence[Move],
    vision_session: RecordingSession | _LiveVisionSession | None,
    prompt: str,
) -> Move | None:
    _read_user_input(prompt)
    observed = vision_bridge.observe_board(session=vision_session)
    matches = infer_moves_from_observation(
        rules=session.rules,
        state=session.state,
        legal_moves=legal_moves,
        observed_board=observed,
    )
    if len(matches) == 1:
        move = matches[0]
        logger.info("Vision hat Zug erkannt: %s", _format_move(move))
        print(f"Vision hat Zug erkannt: {_format_move(move)}")
        return move

    if len(matches) == 0:
        logger.warning("Vision-Scan passt zu keinem legalen Zug; falle auf manuelle Auswahl zurueck.")
    else:
        logger.warning("Vision-Scan passt zu %s legalen Zuegen; manuelle Aufloesung erforderlich.", len(matches))
    return None


def _choose_human_move(
    *,
    session: MillGameSession,
    controller: PlayerController,
    vision_bridge: MillVisionBridge | None,
    voice_bridge: VoiceBridge | None = None,
    vision_session: RecordingSession | _LiveVisionSession | None = None,
    vision_trigger: VisionTriggerMode = "auto",
    baseline_timeout_disabled: bool = False,
    voice_move_timeout_disabled: bool = False,
) -> "Move | UndoSignal":
    legal_moves = list(session.legal_moves())
    if not legal_moves:
        raise RuntimeError("Keine legalen Zuege fuer den menschlichen Zug verfuegbar.")

    if controller.input_mode == "voice" and voice_bridge is not None:
        voice_timeout = float("inf") if voice_move_timeout_disabled else VOICE_MOVE_TIMEOUT_S
        result = voice_bridge.listen_for_move(legal_moves, timeout_s=voice_timeout)
        if isinstance(result, UndoSignal):
            return result
        if result is not None:
            return result
        print("Spracherkennung abgelaufen, wechsle auf manuelle Eingabe.")

    if controller.input_mode == "vision" and vision_bridge is not None:
        if vision_trigger == "auto":
            baseline_timeout_s = float("inf") if baseline_timeout_disabled else AUTO_BASELINE_TIMEOUT_S
            while True:
                result = vision_bridge.observe_move_automatically(
                    rules=session.rules,
                    state=session.state,
                    legal_moves=legal_moves,
                    session=vision_session,
                    status_callback=print,
                    baseline_timeout_s=baseline_timeout_s,
                )
                if result.move is not None:
                    logger.info("Vision hat Zug erkannt: %s", _format_move(result.move))
                    print(f"Vision hat Zug erkannt: {_format_move(result.move)}")
                    return result.move
                if result.reason == "recalibrate_requested":
                    print("Neukalibrierung angefordert – Brett wird neu erkannt …")
                    try:
                        vision_bridge.calibrate_temporary_board_pixels(
                            session=vision_session,
                            attempts=5,
                        )
                        print("Neukalibrierung abgeschlossen.")
                    except Exception as exc:
                        print(f"Neukalibrierung fehlgeschlagen: {exc}")
                    continue
                if result.reason == "undo_requested":
                    return UndoSignal()
                break

            logger.warning(
                "Automatische Vision-Zugerkennung fehlgeschlagen (%s); falle auf manuellen Scan zurueck.",
                result.reason,
            )
            print("Auto-Erkennung unklar, manueller Scan")
            move = _scan_human_move_via_vision(
                session=session,
                vision_bridge=vision_bridge,
                legal_moves=legal_moves,
                vision_session=vision_session,
                prompt="Auto-Erkennung unklar. Enter fuer manuellen Scan druecken: ",
            )
            if move is not None:
                return move
        else:
            move = _scan_human_move_via_vision(
                session=session,
                vision_bridge=vision_bridge,
                legal_moves=legal_moves,
                vision_session=vision_session,
                prompt="Zug auf dem realen Brett ausfuehren und dann Enter zum Scannen druecken: ",
            )
            if move is not None:
                return move

    return _prompt_human_move(legal_moves)


def _decide_undo_count(
    session: MillGameSession,
    controllers: dict[Player, PlayerController],
) -> int:
    """Bestimmt, wie viele Halbzuege bei einem Undo-Request entfernt werden.

    - Keine History -> 0
    - Zwei Menschen -> stets 1
    - Mensch vs. KI: Wenn der letzte Halbzug von der KI war und davor ein
      menschlicher Halbzug -> 2 (KI- und Mensch-Halbzug). Sonst 1.
    """
    if not session.move_history:
        return 0
    if all(c.kind == "human" for c in controllers.values()):
        return 1
    last = session.move_history[-1]
    last_controller = controllers[last.player]
    if last_controller.kind == "ai" and len(session.move_history) >= 2:
        prev = session.move_history[-2]
        if controllers[prev.player].kind == "human":
            return 2
    return 1


def _print_turn_header(session: MillGameSession) -> None:
    state = session.state
    white_pieces = sum(1 for owner in state.board.values() if owner == "W")
    black_pieces = sum(1 for owner in state.board.values() if owner == "B")
    white_phase = phase_for_player(state, "W", settings=session.rules.settings)
    black_phase = phase_for_player(state, "B", settings=session.rules.settings)

    print()
    print(f"Halbzug {len(session.move_history) + 1} | am_zug={state.to_move}")
    print(f"Weiss: steine={white_pieces} gesetzt={state.placed.get('W', 0)} phase={white_phase}")
    print(f"Schwarz: steine={black_pieces} gesetzt={state.placed.get('B', 0)} phase={black_phase}")
    print(_format_board(state.board))


def _state_to_dict(state: MillState) -> dict:
    return {
        "board": state.board,
        "to_move": state.to_move,
        "placed": state.placed,
        "plies_without_capture": state.plies_without_capture,
        "position_history": list(state.position_history),
    }


def _dict_to_state(d: dict) -> MillState:
    return MillState(
        board=d["board"],
        to_move=d["to_move"],
        placed=d["placed"],
        plies_without_capture=d["plies_without_capture"],
        position_history=tuple(d["position_history"]),
    )


def _move_to_dict(move: Move) -> dict:
    return {"player": move.player, "src": move.src, "dst": move.dst, "capture": move.capture}


def _dict_to_move(d: dict) -> Move:
    return Move(player=d["player"], src=d.get("src"), dst=d["dst"], capture=d.get("capture"))


def _save_progress(path: Path, session: MillGameSession) -> None:
    try:
        payload = {
            "version": 1,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "ply": len(session.move_history),
            "state": _state_to_dict(session.state),
            "move_history": [_move_to_dict(m) for m in session.move_history],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Autosave fehlgeschlagen (%s); Spiel laeuft weiter.", exc)


def _load_progress(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Autosave konnte nicht geladen werden (%s); starte neu.", exc)
        return None


def run_mill_game(args: argparse.Namespace) -> int:
    settings = MillRuleSettings(
        enable_flying=args.mill_flying,
        enable_threefold_repetition=args.mill_threefold_repetition,
        enable_no_capture_draw=args.mill_no_capture_draw,
        no_capture_draw_plies=args.mill_no_capture_draw_plies,
    )
    rules = MillRules(settings=settings)
    session = MillGameSession(rules=rules)

    if getattr(args, "mill_resume", False):
        data = _load_progress(AUTOSAVE_PATH)
        if data is not None:
            session.state = _dict_to_state(data["state"])
            session.move_history = [_dict_to_move(m) for m in data["move_history"]]
            print(f"Spielstand wiederhergestellt (Halbzug {len(session.move_history)}, am Zug: {session.state.to_move}).")
        else:
            print("Kein Autosave gefunden; starte neues Spiel.")

    controllers = build_player_controllers(args)
    ai_player_present = any(ctrl.kind == "ai" for ctrl in controllers.values())
    physical_board_mode = any(ctrl.kind == "human" and ctrl.input_mode == "vision" for ctrl in controllers.values())
    robot_controlled_players = resolve_uarm_players(args, ai_player_present=ai_player_present)
    robot_bridge_enabled = bool(robot_controlled_players)
    record_game = bool(args.mill_record_game)

    vision_bridge: MillVisionBridge | None = None
    if physical_board_mode:
        vision_bridge = MillVisionBridge.for_live_session(
            attempts=args.mill_vision_attempts,
            debug_assignments=args.mill_debug_vision,
            camera_index=args.camera_index,
        )
        logger.info("Vision-Bridge fuer menschliche Zug-Inferenz aktiviert.")

    voice_mode = any(ctrl.kind == "human" and ctrl.input_mode == "voice" for ctrl in controllers.values())
    voice_bridge: VoiceBridge | None = None
    if voice_mode:
        voice_bridge = VoiceBridge()
        logger.info("Voice-Bridge fuer menschliche Zug-Eingabe aktiviert.")

    robot_bridge: MillRobotBridge | None = None
    if robot_bridge_enabled:
        try:
            board_positions = load_robot_board_positions(args.mill_robot_board_map)
            robot_bridge = MillRobotBridge(
                board_positions=board_positions,
                port=args.mill_uarm_port,
                reserve_positions=build_default_reserve_positions(),
                move_speed=args.mill_robot_speed,
            )
            robot_bridge.connect()
            if getattr(args, "mill_restore_board", False) and getattr(args, "mill_resume", False):
                print("Stelle physischen Spielstand via uArm wieder her...")
                try:
                    robot_bridge.restore_board_state(session.state)
                    print("Spielbrett physisch wiederhergestellt.")
                except Exception as exc:
                    logger.warning("Physische Spielstand-Wiederherstellung fehlgeschlagen (%s); fahre fort.", exc)
        except Exception as exc:
            logger.warning("Roboter-Bridge nicht verfuegbar (%s); fahre ohne Roboterausfuehrung fort.", exc)
            robot_bridge = None

    print("Spielbare Muehle-Sitzung gestartet")
    human_input_desc = str(args.mill_human_input)
    if human_input_desc == "vision":
        human_input_desc = f"{human_input_desc} ({args.mill_vision_trigger})"
    print(f"Modus: {args.mill_mode} | KI: {args.mill_ai} | Menschliche Eingabe: {human_input_desc}")
    print("Jederzeit mit Strg+C oder per 'q' bei menschlicher Eingabe abbrechen.")

    try:
        with ExitStack() as stack:
            game_recording: RecordingSession | None = None
            vision_session: RecordingSession | _LiveVisionSession | None = None
            if record_game:
                try:
                    from gaming_robot_arm.vision.recording import recording_session

                    game_recording = stack.enter_context(recording_session(camera_index=args.camera_index))
                    print(f"Aufzeichnung aktiv: {game_recording.output_path}")
                    logger.info("Spielaufzeichnung aktiv: %s", game_recording.output_path)
                except Exception as exc:
                    logger.warning("Spielaufzeichnung konnte nicht gestartet werden (%s); fahre ohne Aufnahme fort.", exc)
                    print(f"Hinweis: Aufzeichnung deaktiviert ({exc})")
                    game_recording = None

            if physical_board_mode and vision_bridge is not None:
                if game_recording is not None:
                    vision_session = game_recording
                    logger.info("Vision nutzt den geoeffneten Aufnahme-Kanal als Live-Feed.")
                else:
                    try:
                        from gaming_robot_arm.vision.recording import open_camera

                        live_camera = stack.enter_context(open_camera(camera_index=args.camera_index))
                        vision_session = _LiveVisionSession(camera=live_camera)
                        logger.info("Vision-Live-Feed auf Kamera %s gestartet.", args.camera_index)
                    except Exception as exc:
                        logger.warning(
                            "Vision-Live-Feed konnte nicht gestartet werden (%s); verwende manuelle Eingabe.",
                            exc,
                        )
                        vision_bridge = None

            if physical_board_mode and vision_bridge is not None and vision_session is not None:
                try:
                    vision_bridge.calibrate_temporary_board_pixels(
                        session=vision_session,
                        attempts=5,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Live-Brettkalibrierung fehlgeschlagen – Brett sichtbar und gut beleuchtet? ({exc})"
                    ) from exc

            if (
                physical_board_mode
                and vision_bridge is not None
                and vision_session is not None
                and bool(getattr(args, "mill_vision_preview", False))
            ):
                preview_session = _VisionPreviewSession(vision_session, vision_bridge)
                stack.callback(preview_session.close)
                vision_session = preview_session
                logger.info("Vision-Preview-Fenster aktiviert.")

            while (args.mill_max_plies == 0 or len(session.move_history) < args.mill_max_plies) and not session.is_terminal():
                _print_turn_header(session)
                player = session.state.to_move
                controller = controllers[player]

                if robot_bridge is not None and player in robot_controlled_players:
                    print("Hinweis: uArm zieht in diesem Halbzug - Brettbereich vorbereiten.")

                if controller.kind == "ai":
                    provider = require_ai_provider(controller)
                    move = session.choose_ai_move(provider)
                    print(f"{controller.label} waehlt {_format_move(move)}")
                else:
                    result = _choose_human_move(
                        session=session,
                        controller=controller,
                        vision_bridge=vision_bridge,
                        voice_bridge=voice_bridge,
                        vision_session=vision_session,
                        vision_trigger=args.mill_vision_trigger,
                        baseline_timeout_disabled=bool(args.mill_baseline_timeout_disabled),
                        voice_move_timeout_disabled=bool(args.mill_voice_move_timeout_disabled),
                    )
                    if isinstance(result, UndoSignal):
                        n = _decide_undo_count(session, controllers)
                        if n == 0:
                            print("Kein Zug zum Zuruecknehmen vorhanden.")
                            continue
                        popped = session.undo_n_moves(n)
                        for mv in popped:
                            if robot_bridge is not None and mv.player in robot_controlled_players:
                                _emit_pre_move_warning(
                                    mv,
                                    vision_bridge=vision_bridge,
                                    vision_session=vision_session,
                                    use_vision_gate=bool(args.mill_pre_move_vision_gate),
                                    quiet_timeout_s=args.mill_pre_move_quiet_timeout,
                                    fallback_delay_s=args.mill_pre_move_delay,
                                    kind="Rueckgaengig",
                                )
                                ok = robot_bridge.reverse_move(mv, player=mv.player)
                                if not ok:
                                    print(
                                        f"Achtung: physische Umkehr von {_format_move(mv)} fehlgeschlagen."
                                    )
                            else:
                                print(
                                    f"Bitte Stein {mv.player} manuell zuruecksetzen "
                                    f"(rueckwaerts: {_format_move(mv)})."
                                )
                        _save_progress(AUTOSAVE_PATH, session)
                        print(f"{n} Halbzug/Halbzuege zurueckgenommen.")
                        continue
                    move = result
                    print(f"{controller.label} waehlt {_format_move(move)}")

                _record_single_frame(game_recording)
                session.apply_move(move)
                _save_progress(AUTOSAVE_PATH, session)

                if robot_bridge is not None and player in robot_controlled_players:
                    _emit_pre_move_warning(
                        move,
                        vision_bridge=vision_bridge,
                        vision_session=vision_session,
                        use_vision_gate=bool(args.mill_pre_move_vision_gate),
                        quiet_timeout_s=args.mill_pre_move_quiet_timeout,
                        fallback_delay_s=args.mill_pre_move_delay,
                    )
                    executed = robot_bridge.execute_move(move, player=player)
                    if not executed:
                        logger.warning(
                            "Roboterausfuehrung fuer %s uebersprungen/fehlgeschlagen; logisches Spiel laeuft weiter.",
                            _format_move(move),
                        )

                _record_single_frame(game_recording)

            _print_turn_header(session)

            if session.is_terminal():
                winner = session.winner()
                if winner is None:
                    draw_reason = rules.draw_reason(session.state) or "remis"
                    print(f"Ergebnis: remis ({draw_reason})")
                else:
                    print(f"Ergebnis: Sieger = {winner}")

                if robot_bridge is not None:
                    if winner is None:
                        _outcome = "draw"
                    elif winner in robot_controlled_players:
                        _outcome = "win"
                    else:
                        _outcome = "loss"
                    print("Roboter-Animation...")
                    robot_bridge.perform_game_end_animation(_outcome)
                    print("Brett aufräumen...")
                    robot_bridge.cleanup_board(session.state)

                try:
                    AUTOSAVE_PATH.unlink(missing_ok=True)
                except Exception:
                    pass
            elif args.mill_max_plies > 0:
                print(f"Ergebnis: nach {args.mill_max_plies} Halbzuegen gestoppt (nicht-terminaler Zustand).")

            return 0
    except KeyboardInterrupt:
        print("\nSitzung durch Benutzer abgebrochen.")
        return 130
    finally:
        if robot_bridge is not None:
            robot_bridge.close()
        if voice_bridge is not None:
            voice_bridge.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Startet eine spielbare Muehle-Sitzung (CLI + optionale Vision-/Roboter-Bridge).")
    add_mill_cli_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mill_max_plies < 0:
        parser.error("--max-plies muss >= 0 sein (0 = keine Begrenzung).")
    return run_mill_game(args)


if __name__ == "__main__":
    raise SystemExit(main())
