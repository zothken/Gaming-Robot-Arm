"""Spielbare Muehle-Schleife mit optionaler Vision- und Roboterintegration."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from gaming_robot_arm.config import (
    CAMERA_INDEX,
    UARM_PORT,
)
from gaming_robot_arm.games.common.interfaces import Move, Player
from gaming_robot_arm.games.mill import (
    MillGameSession,
    MillRuleSettings,
    MillRules,
)
from gaming_robot_arm.games.mill.core.rules import phase_for_player
from gaming_robot_arm.games.mill.core.settings import (
    MILL_ENABLE_FLYING,
    MILL_ENABLE_NO_CAPTURE_DRAW,
    MILL_ENABLE_THREEFOLD_REPETITION,
    MILL_NO_CAPTURE_DRAW_PLIES,
)
from gaming_robot_arm.utils.logger import logger
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
    MillVisionBridge,
    _LiveVisionSession,
    infer_moves_from_observation,
)

if TYPE_CHECKING:
    from gaming_robot_arm.vision.recording import RecordingSession


def add_mill_cli_arguments(parser: argparse.ArgumentParser) -> None:
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
        "--ai-model",
        dest="mill_ai_model",
        type=Path,
        default=Path("models/champion/mill_champion.pt"),
        help="Modell-Checkpoint-Pfad fuer neuronale KI.",
    )
    ai_group.add_argument(
        "--ai-temperature",
        dest="mill_ai_temperature",
        type=float,
        default=0.0,
        help="Temperatur fuer neuronales KI-Sampling.",
    )
    ai_group.add_argument(
        "--ai-device",
        dest="mill_ai_device",
        type=str,
        default="auto",
        help="Geraet fuer neuronale KI: auto/cpu/cuda/...",
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
        default="default",
        help="Quelle der Roboter-Brettkoordinaten: feste Standardwerte oder Homography-Projektion.",
    )


def _format_move(move: Move) -> str:
    src = move.src if move.src is not None else "VORRAT"
    capture = f" x {move.capture}" if move.capture is not None else ""
    return f"{move.player}: {src} -> {move.dst}{capture}"


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


def _prompt_human_move(legal_moves: Sequence[Move]) -> Move:
    print("Legale Zuege:")
    for idx, move in enumerate(legal_moves, start=1):
        print(f"  [{idx:02d}] {_format_move(move)}")

    while True:
        raw = _read_user_input("Zugnummer waehlen (oder 'q' zum Abbrechen): ").strip().lower()
        if raw == "q":
            raise KeyboardInterrupt
        try:
            choice = int(raw)
        except ValueError:
            print("Bitte eine numerische Zugnummer eingeben.")
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


def _choose_human_move(
    *,
    session: MillGameSession,
    controller: PlayerController,
    vision_bridge: MillVisionBridge | None,
    vision_session: RecordingSession | _LiveVisionSession | None = None,
) -> Move:
    legal_moves = list(session.legal_moves())
    if not legal_moves:
        raise RuntimeError("Keine legalen Zuege fuer den menschlichen Zug verfuegbar.")

    if controller.input_mode == "vision" and vision_bridge is not None:
        _read_user_input("Zug auf dem realen Brett ausfuehren und dann Enter zum Scannen druecken: ")
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
            return move

        if len(matches) == 0:
            logger.warning("Vision-Scan passt zu keinem legalen Zug; falle auf manuelle Auswahl zurueck.")
        else:
            logger.warning("Vision-Scan passt zu %s legalen Zuegen; manuelle Aufloesung erforderlich.", len(matches))

    return _prompt_human_move(legal_moves)


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


def run_mill_game(args: argparse.Namespace) -> int:
    settings = MillRuleSettings(
        enable_flying=args.mill_flying,
        enable_threefold_repetition=args.mill_threefold_repetition,
        enable_no_capture_draw=args.mill_no_capture_draw,
        no_capture_draw_plies=args.mill_no_capture_draw_plies,
    )
    rules = MillRules(settings=settings)
    session = MillGameSession(rules=rules)
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
        except Exception as exc:
            logger.warning("Roboter-Bridge nicht verfuegbar (%s); fahre ohne Roboterausfuehrung fort.", exc)
            robot_bridge = None

    print("Spielbare Muehle-Sitzung gestartet")
    print(f"Modus: {args.mill_mode} | KI: {args.mill_ai} | Menschliche Eingabe: {args.mill_human_input}")
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
                        attempts=1,
                    )
                except Exception as exc:
                    logger.warning(
                        "Temporare Live-Brettkalibrierung fehlgeschlagen (%s). Versuche gespeicherte Kalibrierung.",
                        exc,
                    )
                    try:
                        fallback_bridge = MillVisionBridge.from_calibration(
                            attempts=args.mill_vision_attempts,
                            debug_assignments=args.mill_debug_vision,
                            camera_index=args.camera_index,
                        )
                        vision_bridge.board_pixels = fallback_bridge.board_pixels
                        vision_bridge.labels_order = fallback_bridge.labels_order
                        vision_bridge.board_source = "calibration"
                        logger.warning("Nutze gespeicherte Brettkalibrierung als Fallback fuer diese Partie.")
                    except Exception as fallback_exc:
                        logger.warning(
                            "Keine Brettkalibrierung verfuegbar (%s); verwende manuelle Eingabe.",
                            fallback_exc,
                        )
                        vision_bridge = None

            while (args.mill_max_plies == 0 or len(session.move_history) < args.mill_max_plies) and not session.is_terminal():
                _print_turn_header(session)
                player = session.state.to_move
                controller = controllers[player]

                if controller.kind == "ai":
                    provider = require_ai_provider(controller)
                    move = session.choose_ai_move(provider)
                    print(f"{controller.label} waehlt {_format_move(move)}")
                else:
                    move = _choose_human_move(
                        session=session,
                        controller=controller,
                        vision_bridge=vision_bridge,
                        vision_session=vision_session,
                    )
                    print(f"{controller.label} waehlt {_format_move(move)}")

                _record_single_frame(game_recording)
                session.apply_move(move)

                if robot_bridge is not None and player in robot_controlled_players:
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
            elif args.mill_max_plies > 0:
                print(f"Ergebnis: nach {args.mill_max_plies} Halbzuegen gestoppt (nicht-terminaler Zustand).")

            return 0
    except KeyboardInterrupt:
        print("\nSitzung durch Benutzer abgebrochen.")
        return 130
    finally:
        if robot_bridge is not None:
            robot_bridge.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Startet eine spielbare Muehle-Sitzung (CLI + optionale Vision-/Roboter-Bridge).")
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX, help="Kamera-Index fuer den Vision-Modus.")
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
