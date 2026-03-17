"""Generischer Kopf-an-Kopf-Benchmark fuer Muehle-Zug-Provider-KIs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import inspect
import time
from typing import Any, Sequence

from gaming_robot_arm.games.mill import (
    AlphaBetaMillAI,
    HeuristicMillAI,
    MillGameSession,
    NeuralMillAI,
    MillRuleSettings,
    MillRules,
)


AI_REGISTRY: dict[str, type[Any]] = {
    "heuristic": HeuristicMillAI,
    "alphabeta": AlphaBetaMillAI,
}
NEURAL_AI_UNAVAILABLE_REASON: str | None = None
if NeuralMillAI is not None:
    AI_REGISTRY["neural"] = NeuralMillAI
else:
    NEURAL_AI_UNAVAILABLE_REASON = "neuronale KI nicht verfuegbar (numpy und/oder torch fehlen in dieser Python-Umgebung)"


@dataclass(slots=True)
class BenchmarkParticipant:
    key: str
    label: str
    ai_class: type[Any]
    base_kwargs: dict[str, Any]
    seed_base: int | None


@dataclass(slots=True)
class GameResult:
    index: int
    white_key: str
    black_key: str
    white_name: str
    black_name: str
    winner_key: str | None
    winner_name: str | None
    draw_reason: str | None
    plies: int
    duration_seconds: float


def parse_scalar(raw: str) -> Any:
    value = raw.strip()
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "none":
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def parse_kv_items(items: Sequence[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Ungueltiges --*-arg '{item}'. Erwartet wird SCHLUESSEL=WERT.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Ungueltiges --*-arg '{item}'. SCHLUESSEL darf nicht leer sein.")
        parsed[key] = parse_scalar(raw_value)
    return parsed


def resolve_ai_class(identifier: str) -> type[Any]:
    key = identifier.strip()
    registry_key = key.lower()
    if registry_key in AI_REGISTRY:
        return AI_REGISTRY[registry_key]

    if ":" not in key:
        available = ", ".join(sorted(AI_REGISTRY))
        raise ValueError(
            f"Unbekannte KI '{identifier}'. Verwende eine aus [{available}] oder gib "
            "eine vollqualifizierte Klasse als module.path:ClassName an."
        )

    module_name, class_name = key.rsplit(":", 1)
    if not module_name or not class_name:
        raise ValueError(f"Ungueltige KI-Spezifikation '{identifier}'. Erwartet: module.path:ClassName.")

    module = importlib.import_module(module_name)
    ai_class = getattr(module, class_name)
    if not inspect.isclass(ai_class):
        raise TypeError(f"Aufgeloeste KI '{identifier}' ist keine Klasse.")
    return ai_class


def list_available_ais() -> None:
    print("Interne KI-Bezeichner:")
    for key, ai_class in sorted(AI_REGISTRY.items()):
        print(f"  {key:<10} -> {ai_class.__module__}:{ai_class.__name__}")
    if NEURAL_AI_UNAVAILABLE_REASON is not None:
        print(f"  neural     -> {NEURAL_AI_UNAVAILABLE_REASON}")


def get_init_parameters(ai_class: type[Any]) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(ai_class).parameters)


def instantiate_ai(
    participant: BenchmarkParticipant,
    *,
    game_index: int,
    deterministic: bool,
    default_alphabeta_depth: int | None,
) -> Any:
    kwargs = dict(participant.base_kwargs)
    init_params = get_init_parameters(participant.ai_class)

    if (
        participant.ai_class is AlphaBetaMillAI
        and default_alphabeta_depth is not None
        and "depth" in init_params
        and "depth" not in kwargs
    ):
        kwargs["depth"] = default_alphabeta_depth

    if "seed" in init_params and "seed" not in kwargs and participant.seed_base is not None:
        kwargs["seed"] = participant.seed_base + game_index

    if deterministic and "random_tiebreak" in init_params and "random_tiebreak" not in kwargs:
        kwargs["random_tiebreak"] = False

    return participant.ai_class(**kwargs)


def play_game(
    *,
    index: int,
    white_key: str,
    black_key: str,
    white_name: str,
    black_name: str,
    white_ai: Any,
    black_ai: Any,
    rules: MillRules,
    max_plies: int,
) -> GameResult:
    session = MillGameSession(rules=rules)
    start = time.perf_counter()
    draw_reason: str | None = None

    while len(session.move_history) < max_plies and not session.is_terminal():
        provider = white_ai if session.state.to_move == "W" else black_ai
        move = session.choose_ai_move(provider)
        session.apply_move(move)

    duration_seconds = time.perf_counter() - start
    winner_color = session.winner() if session.is_terminal() else None

    if winner_color is None:
        if session.is_terminal():
            draw_reason = rules.draw_reason(session.state) or "terminal_draw"
        else:
            draw_reason = f"max_plies_{max_plies}"

    winner_key: str | None = None
    winner_name: str | None = None
    if winner_color == "W":
        winner_key = white_key
        winner_name = white_name
    elif winner_color == "B":
        winner_key = black_key
        winner_name = black_name

    return GameResult(
        index=index,
        white_key=white_key,
        black_key=black_key,
        white_name=white_name,
        black_name=black_name,
        winner_key=winner_key,
        winner_name=winner_name,
        draw_reason=draw_reason,
        plies=len(session.move_history),
        duration_seconds=duration_seconds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuehrt einen generischen Muehle-KI-Benchmark aus.")
    parser.add_argument("--list-ai", action="store_true", help="Listet interne KI-Bezeichner und beendet.")
    parser.add_argument("--games", type=int, default=10, help="Anzahl zu spielender Partien.")
    parser.add_argument("--max-plies", type=int, default=400, help="Harte Abbruchgrenze pro Partie.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Setzt random_tiebreak=False automatisch, falls von der KI-Klasse unterstuetzt.",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Standardtiefe fuer AlphaBetaMillAI, falls depth nicht explizit per --*-arg depth=... gesetzt ist.",
    )

    parser.add_argument("--ai-a", type=str, default="alphabeta", help="KI fuer Teilnehmer A.")
    parser.add_argument("--ai-b", type=str, default="heuristic", help="KI fuer Teilnehmer B.")
    parser.add_argument("--label-a", type=str, default=None, help="Anzeigelabel fuer Teilnehmer A.")
    parser.add_argument("--label-b", type=str, default=None, help="Anzeigelabel fuer Teilnehmer B.")
    parser.add_argument("--ai-a-arg", action="append", default=[], help="KI-Init-Argument fuer Teilnehmer A: SCHLUESSEL=WERT.")
    parser.add_argument("--ai-b-arg", action="append", default=[], help="KI-Init-Argument fuer Teilnehmer B: SCHLUESSEL=WERT.")
    parser.add_argument(
        "--ai-a-seed-base",
        type=int,
        default=10_000,
        help="Falls KI seed unterstuetzt und nicht gesetzt ist: seed = ai_a_seed_base + game_index.",
    )
    parser.add_argument(
        "--ai-b-seed-base",
        type=int,
        default=20_000,
        help="Falls KI seed unterstuetzt und nicht gesetzt ist: seed = ai_b_seed_base + game_index.",
    )
    parser.add_argument(
        "--alternate-colors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wechselt ab, welcher Teilnehmer jede Partie als Weiss startet.",
    )

    parser.add_argument(
        "--enable-flying",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aktiviert die Muehle-Flying-Regel.",
    )
    parser.add_argument(
        "--enable-threefold-repetition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aktiviert Remis durch Dreifachwiederholung.",
    )
    parser.add_argument(
        "--enable-no-capture-draw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aktiviert Remis durch Zuglimit ohne Schlag.",
    )
    parser.add_argument(
        "--no-capture-draw-plies",
        type=int,
        default=200,
        help="Halbzuglimit ohne Schlag, wenn --enable-no-capture-draw aktiv ist.",
    )
    return parser


def build_participant(
    *,
    key: str,
    identifier: str,
    label: str | None,
    arg_items: Sequence[str],
    seed_base: int | None,
) -> BenchmarkParticipant:
    ai_class = resolve_ai_class(identifier)
    base_kwargs = parse_kv_items(arg_items)
    default_label = f"{identifier} ({ai_class.__name__})"
    return BenchmarkParticipant(
        key=key,
        label=label or default_label,
        ai_class=ai_class,
        base_kwargs=base_kwargs,
        seed_base=seed_base,
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.games <= 0:
        raise ValueError("--games muss > 0 sein")
    if args.max_plies <= 0:
        raise ValueError("--max-plies muss > 0 sein")
    if args.no_capture_draw_plies <= 0:
        raise ValueError("--no-capture-draw-plies muss > 0 sein")
    if args.depth is not None and args.depth <= 0:
        raise ValueError("--depth muss > 0 sein, wenn gesetzt")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_ai:
        list_available_ais()
        return

    validate_args(args)

    participant_a = build_participant(
        key="A",
        identifier=args.ai_a,
        label=args.label_a,
        arg_items=args.ai_a_arg,
        seed_base=args.ai_a_seed_base,
    )
    participant_b = build_participant(
        key="B",
        identifier=args.ai_b,
        label=args.label_b,
        arg_items=args.ai_b_arg,
        seed_base=args.ai_b_seed_base,
    )

    rule_settings = MillRuleSettings(
        enable_flying=args.enable_flying,
        enable_threefold_repetition=args.enable_threefold_repetition,
        enable_no_capture_draw=args.enable_no_capture_draw,
        no_capture_draw_plies=args.no_capture_draw_plies,
    )
    rules = MillRules(settings=rule_settings)

    results: list[GameResult] = []
    for game_index in range(1, args.games + 1):
        a_is_white = (game_index % 2 == 1) if args.alternate_colors else True
        white_participant = participant_a if a_is_white else participant_b
        black_participant = participant_b if a_is_white else participant_a

        white_ai = instantiate_ai(
            white_participant,
            game_index=game_index,
            deterministic=args.deterministic,
            default_alphabeta_depth=args.depth,
        )
        black_ai = instantiate_ai(
            black_participant,
            game_index=game_index,
            deterministic=args.deterministic,
            default_alphabeta_depth=args.depth,
        )

        result = play_game(
            index=game_index,
            white_key=white_participant.key,
            black_key=black_participant.key,
            white_name=white_participant.label,
            black_name=black_participant.label,
            white_ai=white_ai,
            black_ai=black_ai,
            rules=rules,
            max_plies=args.max_plies,
        )
        results.append(result)

        outcome = result.winner_name or f"Remis ({result.draw_reason})"
        print(
            f"Partie {result.index:02d}: "
            f"{result.white_name} (W) gegen {result.black_name} (B) "
            f"-> {outcome}; halbzuege={result.plies}; zeit={result.duration_seconds:.2f}s"
        )

    wins_by_key = {
        participant_a.key: sum(1 for result in results if result.winner_key == participant_a.key),
        participant_b.key: sum(1 for result in results if result.winner_key == participant_b.key),
    }
    draws = sum(1 for result in results if result.winner_key is None)
    total_plies = sum(result.plies for result in results)
    total_time = sum(result.duration_seconds for result in results)

    print()
    print("Zusammenfassung")
    print(f"  Partien: {len(results)}")
    print(f"  {participant_a.label} siege: {wins_by_key[participant_a.key]}")
    print(f"  {participant_b.label} siege: {wins_by_key[participant_b.key]}")
    print(f"  Remis: {draws}")
    print(f"  Ø Halbzuege/Partie: {total_plies / len(results):.1f}")
    print(f"  Ø Zeit/Partie: {total_time / len(results):.2f}s")


if __name__ == "__main__":
    main()
