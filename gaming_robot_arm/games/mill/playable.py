"""Spielbare Muehle-Schleife mit optionaler Vision- und Roboterintegration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Literal, Sequence

from gaming_robot_arm.config import (
    CAMERA_INDEX,
    MILL_ENABLE_FLYING,
    MILL_ENABLE_NO_CAPTURE_DRAW,
    MILL_ENABLE_THREEFOLD_REPETITION,
    MILL_NO_CAPTURE_DRAW_PLIES,
    PICK_Z,
    PLACE_Z,
    REST_POS,
    SAFE_Z,
    UARM_PORT,
)
from gaming_robot_arm.games.common.interfaces import Move, Player
from gaming_robot_arm.games.mill import (
    AlphaBetaMillAI,
    HeuristicMillAI,
    MillGameSession,
    MillRuleSettings,
    MillRules,
    NeuralMillAI,
)
from gaming_robot_arm.games.mill.board import BOARD_LABELS
from gaming_robot_arm.games.mill.rules import phase_for_player
from gaming_robot_arm.games.mill.session import MillMoveProvider
from gaming_robot_arm.utils.logger import logger

if TYPE_CHECKING:
    from gaming_robot_arm.control.uarm_controller import UArmController


GAME_MODES = ("human-vs-human", "human-vs-ai", "ai-vs-ai")
HUMAN_INPUT_MODES = ("manual", "vision")
AI_BACKENDS = ("heuristic", "alphabeta", "neural")
ROBOT_BOARD_MAPS = ("default", "homography")


@dataclass(slots=True)
class PlayerController:
    kind: Literal["human", "ai"]
    label: str
    input_mode: str = "manual"
    provider: MillMoveProvider | None = None


@dataclass(slots=True)
class MillVisionBridge:
    """Liest stabile Brettbelegung ueber die bestehende Figuren-Detektion."""

    board_pixels: dict[str, tuple[float, float]]
    labels_order: list[str]
    attempts: int = 6
    debug_assignments: bool = False
    camera_index: int = CAMERA_INDEX

    @classmethod
    def from_calibration(
        cls,
        *,
        attempts: int = 6,
        debug_assignments: bool = False,
        camera_index: int = CAMERA_INDEX,
    ) -> "MillVisionBridge":
        from gaming_robot_arm.calibration.calibration import load_board_pixels

        board_pixels = load_board_pixels()
        labels_order = sorted(board_pixels.keys())
        return cls(
            board_pixels={label: (float(u), float(v)) for label, (u, v) in board_pixels.items()},
            labels_order=labels_order,
            attempts=max(1, attempts),
            debug_assignments=debug_assignments,
            camera_index=camera_index,
        )

    def observe_board(self) -> dict[str, Player | None]:
        from gaming_robot_arm.vision.figure_detector import detect_board_assignments

        assignments = detect_board_assignments(
            self.board_pixels,
            attempts=self.attempts,
            labels_order=self.labels_order,
            debug_assignments=self.debug_assignments,
            camera_index=self.camera_index,
        )

        observed: dict[str, Player | None] = {label: None for label in BOARD_LABELS}
        for item in assignments:
            label = str(item.get("label", "")).upper()
            color = str(item.get("color", "")).lower()
            if label not in observed:
                continue
            if color in {"weiss", "white"}:
                observed[label] = "W"
            elif color in {"schwarz", "black"}:
                observed[label] = "B"
        return observed


@dataclass(slots=True)
class MillRobotBridge:
    """Fuehrt Muehle-Zuege auf dem uArm mit kalibrierten Brettkoordinaten aus."""

    board_positions: dict[str, tuple[float, float]]
    port: str | None = UARM_PORT
    reserve_positions: dict[Player, tuple[float, float]] | None = None
    capture_bin: tuple[float, float] | None = None
    move_speed: int = 500
    _controller: UArmController | None = None

    def connect(self) -> None:
        if self._controller is not None:
            return

        from gaming_robot_arm.control import UArmController

        controller = UArmController(port=self.port)
        if controller.swift is None:
            raise RuntimeError("uArm-Verbindung hergestellt, aber ohne Swift-API-Handle.")
        self._controller = controller
        logger.info("Roboter-Bridge mit uArm auf %s verbunden", controller.swift.port)

    def close(self) -> None:
        if self._controller is None:
            return
        self._controller.disconnect()
        self._controller = None

    def execute_move(self, move: Move, *, player: Player) -> bool:
        if self._controller is None:
            self.connect()

        src_xy = self._resolve_move_source(move, player)
        dst_xy = self.board_positions.get(move.dst)

        if src_xy is None:
            logger.warning(
                "Roboterzug fuer %s uebersprungen: keine Quellkoordinate fuer %s verfuegbar.",
                _format_move(move),
                player,
            )
            return False

        if dst_xy is None:
            logger.warning("Roboterzug uebersprungen: Ziel-Label %s hat keine Roboterkoordinate.", move.dst)
            return False

        try:
            self._pick_and_place(src_xy, dst_xy)

            if move.capture is not None:
                self._handle_capture(move.capture)

            self._move_to(*REST_POS)
            return True
        except Exception:
            logger.exception("Roboterausfuehrung fuer Zug %s fehlgeschlagen", _format_move(move))
            try:
                if self._controller is not None and self._controller.swift is not None:
                    self._controller.swift.set_pump(on=False)
                    self._move_to(*REST_POS)
            except Exception:
                logger.exception("Konnte Roboter nach Ausfuehrungsfehler nicht in sicheren Zustand bringen.")
            return False

    def _resolve_move_source(self, move: Move, player: Player) -> tuple[float, float] | None:
        if move.src is not None:
            return self.board_positions.get(move.src)

        reserves = self.reserve_positions or {}
        return reserves.get(player)

    def _handle_capture(self, capture_label: str) -> None:
        capture_src = self.board_positions.get(capture_label)
        if capture_src is None:
            logger.warning("Schlag-Label %s hat keine Roboterkoordinate; Stein bitte manuell entfernen.", capture_label)
            return

        if self.capture_bin is None:
            logger.warning("Schlagablage nicht konfiguriert; geschlagenen Stein %s bitte manuell entfernen.", capture_label)
            return

        self._pick_and_place(capture_src, self.capture_bin)

    def _pick_and_place(self, src_xy: tuple[float, float], dst_xy: tuple[float, float]) -> None:
        self._pick_at(src_xy)
        self._place_at(dst_xy)

    def _pick_at(self, xy: tuple[float, float]) -> None:
        x, y = xy
        swift = self._require_swift()
        self._move_to(x, y, SAFE_Z)
        self._move_to(x, y, PICK_Z)
        swift.set_pump(on=True)
        time.sleep(0.2)
        self._move_to(x, y, SAFE_Z)

    def _place_at(self, xy: tuple[float, float]) -> None:
        x, y = xy
        swift = self._require_swift()
        self._move_to(x, y, SAFE_Z)
        self._move_to(x, y, PLACE_Z)
        swift.set_pump(on=False)
        time.sleep(0.2)
        self._move_to(x, y, SAFE_Z)

    def _move_to(self, x: float, y: float, z: float) -> None:
        controller = self._require_controller()
        controller.move_to(float(x), float(y), float(z), speed=self.move_speed)

    def _require_controller(self) -> UArmController:
        if self._controller is None:
            self.connect()
        controller = self._controller
        if controller is None:
            raise RuntimeError("Roboter-Controller ist nicht verfuegbar.")
        return controller

    def _require_swift(self) -> Any:
        controller = self._require_controller()
        swift = controller.swift
        if swift is None:
            raise RuntimeError("Roboter-Controller hat keine aktive Swift-Verbindung.")
        return swift


def _xy_arg(raw: str) -> tuple[float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Erwartetes Format: X,Y (Beispiel: 180.0,-40.0)")

    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("X und Y muessen numerische Werte sein.") from exc


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
        "--uarm-move-both-players",
        dest="mill_uarm_move_both_players",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Laesst den uArm Zuege beider Seiten ausfuehren (nicht nur KI-Zuege).",
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
    other_group.add_argument(
        "--white-reserve",
        dest="mill_white_reserve",
        type=_xy_arg,
        default=None,
        metavar="X,Y",
        help="Vorrats-Aufnahmepunkt fuer weisse Setzzuege.",
    )
    other_group.add_argument(
        "--black-reserve",
        dest="mill_black_reserve",
        type=_xy_arg,
        default=None,
        metavar="X,Y",
        help="Vorrats-Aufnahmepunkt fuer schwarze Setzzuege.",
    )
    other_group.add_argument(
        "--capture-bin",
        dest="mill_capture_bin",
        type=_xy_arg,
        default=None,
        metavar="X,Y",
        help="Ablagekoordinate fuer geschlagene Steine.",
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


def _create_ai_provider(args: argparse.Namespace, *, seed_offset: int = 0) -> MillMoveProvider:
    ai_name = args.mill_ai
    seed = None if args.mill_seed is None else int(args.mill_seed) + int(seed_offset)

    if ai_name == "heuristic":
        return HeuristicMillAI(random_tiebreak=args.mill_random_tiebreak, seed=seed)

    if ai_name == "alphabeta":
        return AlphaBetaMillAI(depth=args.mill_ai_depth, random_tiebreak=args.mill_random_tiebreak, seed=seed)

    if ai_name == "neural":
        if NeuralMillAI is None:
            raise RuntimeError("NeuralMillAI nicht verfuegbar. Bitte ML-Abhaengigkeiten (torch, numpy) installieren.")
        return NeuralMillAI(
            model_path=args.mill_ai_model,
            random_tiebreak=args.mill_random_tiebreak,
            temperature=args.mill_ai_temperature,
            seed=seed,
            device=args.mill_ai_device,
        )

    raise ValueError(f"Nicht unterstuetztes KI-Backend: {ai_name}")


def _build_player_controllers(args: argparse.Namespace) -> dict[Player, PlayerController]:
    mode = args.mill_mode
    human_input = args.mill_human_input

    if mode == "human-vs-human":
        return {
            "W": PlayerController(kind="human", label="Mensch W", input_mode=human_input),
            "B": PlayerController(kind="human", label="Mensch B", input_mode=human_input),
        }

    if mode == "human-vs-ai":
        human_color = args.mill_human_color
        ai_color = "B" if human_color == "W" else "W"
        return {
            human_color: PlayerController(kind="human", label=f"Mensch {human_color}", input_mode=human_input),
            ai_color: PlayerController(kind="ai", label=f"KI {ai_color}", provider=_create_ai_provider(args)),
        }

    if mode == "ai-vs-ai":
        return {
            "W": PlayerController(kind="ai", label="KI W", provider=_create_ai_provider(args, seed_offset=0)),
            "B": PlayerController(kind="ai", label="KI B", provider=_create_ai_provider(args, seed_offset=1)),
        }

    raise ValueError(f"Nicht unterstuetzter Muehle-Modus: {mode}")


def _prompt_human_move(legal_moves: Sequence[Move]) -> Move:
    print("Legale Zuege:")
    for idx, move in enumerate(legal_moves, start=1):
        print(f"  [{idx:02d}] {_format_move(move)}")

    while True:
        raw = input("Zugnummer waehlen (oder 'q' zum Abbrechen): ").strip().lower()
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


def _infer_moves_from_observation(
    *,
    rules: MillRules,
    state,
    legal_moves: Sequence[Move],
    observed_board: dict[str, Player | None],
) -> list[Move]:
    matches: list[Move] = []
    for move in legal_moves:
        candidate = rules.apply_move(state, move)
        if all(candidate.board[label] == observed_board.get(label) for label in BOARD_LABELS):
            matches.append(move)
    return matches


def _choose_human_move(
    *,
    session: MillGameSession,
    controller: PlayerController,
    vision_bridge: MillVisionBridge | None,
) -> Move:
    legal_moves = list(session.legal_moves())
    if not legal_moves:
        raise RuntimeError("Keine legalen Zuege fuer den menschlichen Zug verfuegbar.")

    if controller.input_mode == "vision" and vision_bridge is not None:
        input("Zug auf dem realen Brett ausfuehren und dann Enter zum Scannen druecken: ")
        observed = vision_bridge.observe_board()
        matches = _infer_moves_from_observation(
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


def _require_ai_provider(controller: PlayerController) -> MillMoveProvider:
    provider = controller.provider
    if provider is None:
        raise RuntimeError(f"Fehlender KI-Provider fuer Controller '{controller.label}'.")
    return provider


def _load_robot_board_positions(source: str) -> dict[str, tuple[float, float]]:
    from gaming_robot_arm.calibration.mill_default_calibration import get_mill_uarm_positions

    if source == "default":
        return get_mill_uarm_positions()

    if source != "homography":
        raise ValueError(f"Nicht unterstuetzte Quelle fuer Roboter-Brettkoordinaten: {source}")

    from gaming_robot_arm.utils.homography import img_to_robot, load_homography

    H, board_pixels = load_homography()
    if H is None or not board_pixels:
        raise RuntimeError("Homography-Brettmapping angefordert, aber Kalibrierdaten sind unvollstaendig.")

    missing_labels = [label for label in BOARD_LABELS if label not in board_pixels]
    if missing_labels:
        raise RuntimeError(f"Homography-Brettmapping hat fehlende Labels: {', '.join(missing_labels)}")

    mapped: dict[str, tuple[float, float]] = {}
    for label in BOARD_LABELS:
        u, v = board_pixels[label]
        mapped[label] = img_to_robot(H, float(u), float(v))
    return mapped


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
    controllers = _build_player_controllers(args)
    physical_board_mode = any(ctrl.kind == "human" and ctrl.input_mode == "vision" for ctrl in controllers.values())
    robot_move_both_players = bool(args.mill_uarm_move_both_players)
    robot_bridge_enabled = physical_board_mode or robot_move_both_players

    vision_bridge: MillVisionBridge | None = None
    if physical_board_mode:
        try:
            vision_bridge = MillVisionBridge.from_calibration(
                attempts=args.mill_vision_attempts,
                debug_assignments=args.mill_debug_vision,
                camera_index=args.camera_index,
            )
            logger.info("Vision-Bridge fuer menschliche Zug-Inferenz aktiviert.")
        except Exception as exc:
            logger.warning("Vision-Bridge nicht verfuegbar (%s); verwende manuelle Eingabe.", exc)
            vision_bridge = None

    robot_bridge: MillRobotBridge | None = None
    if robot_bridge_enabled:
        try:
            board_positions = _load_robot_board_positions(args.mill_robot_board_map)
            reserve_positions: dict[Player, tuple[float, float]] = {}
            if args.mill_white_reserve is not None:
                reserve_positions["W"] = args.mill_white_reserve
            if args.mill_black_reserve is not None:
                reserve_positions["B"] = args.mill_black_reserve

            robot_bridge = MillRobotBridge(
                board_positions=board_positions,
                port=args.mill_uarm_port,
                reserve_positions=reserve_positions,
                capture_bin=args.mill_capture_bin,
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
        while (args.mill_max_plies == 0 or len(session.move_history) < args.mill_max_plies) and not session.is_terminal():
            _print_turn_header(session)
            player = session.state.to_move
            controller = controllers[player]

            if controller.kind == "ai":
                provider = _require_ai_provider(controller)
                move = session.choose_ai_move(provider)
                print(f"{controller.label} waehlt {_format_move(move)}")
            else:
                move = _choose_human_move(session=session, controller=controller, vision_bridge=vision_bridge)
                print(f"{controller.label} waehlt {_format_move(move)}")

            session.apply_move(move)

            should_execute_robot_move = robot_bridge is not None and (controller.kind == "ai" or robot_move_both_players)
            if should_execute_robot_move:
                executed = robot_bridge.execute_move(move, player=player)
                if not executed:
                    logger.warning("Roboterausfuehrung fuer %s uebersprungen/fehlgeschlagen; logisches Spiel laeuft weiter.", _format_move(move))

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
