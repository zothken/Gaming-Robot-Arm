"""Erzeugt ueberwachte Muehle-Trainingsdaten aus AlphaBeta-Lehrer-Selbstspiel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

from gaming_robot_arm.games.mill.core.settings import (
    MILL_ENABLE_FLYING,
    MILL_ENABLE_NO_CAPTURE_DRAW,
    MILL_ENABLE_THREEFOLD_REPETITION,
    MILL_NO_CAPTURE_DRAW_PLIES,
)
from gaming_robot_arm.games.mill import AlphaBetaMillAI, MillGameSession, MillRuleSettings, MillRules
from gaming_robot_arm.games.mill.ml.features import (
    encode_legal_move_features,
    encode_state_features,
    outcome_for_player,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Erzeugt Muehle-Trainingssamples aus AlphaBeta-Selfplay.")
    parser.add_argument("--output", type=Path, default=Path("data/mill_teacher.jsonl"), help="Ausgabe-JSONL-Pfad.")
    parser.add_argument("--games", type=int, default=500, help="Anzahl Selbstspiel-Partien.")
    parser.add_argument("--teacher-depth", type=int, default=3, help="AlphaBeta-Tiefe fuer beide Seiten.")
    parser.add_argument("--max-plies", type=int, default=400, help="Harte Zuggrenze pro Partie.")
    parser.add_argument("--seed", type=int, default=1234, help="Basis-RNG-Seed.")
    parser.add_argument(
        "--teacher-random-tiebreak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verwendet zufaellige Tie-Breaks im Teacher zur Sample-Diversifizierung.",
    )
    parser.add_argument(
        "--alternate-colors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wechselt ab, welche Teacher-Instanz als Weiss startet.",
    )
    parser.add_argument(
        "--enable-flying",
        action=argparse.BooleanOptionalAction,
        default=MILL_ENABLE_FLYING,
        help="Aktiviert Flying-Regel.",
    )
    parser.add_argument(
        "--enable-threefold-repetition",
        action=argparse.BooleanOptionalAction,
        default=MILL_ENABLE_THREEFOLD_REPETITION,
        help="Aktiviert Remis durch Dreifachwiederholung.",
    )
    parser.add_argument(
        "--enable-no-capture-draw",
        action=argparse.BooleanOptionalAction,
        default=MILL_ENABLE_NO_CAPTURE_DRAW,
        help="Aktiviert Remis durch Zuglimit ohne Schlag.",
    )
    parser.add_argument(
        "--no-capture-draw-plies",
        type=int,
        default=MILL_NO_CAPTURE_DRAW_PLIES,
        help="Halbzuglimit ohne Schlag, wenn No-Capture-Remis aktiv ist.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.games <= 0:
        raise ValueError("--games muss > 0 sein")
    if args.teacher_depth <= 0:
        raise ValueError("--teacher-depth muss > 0 sein")
    if args.max_plies <= 0:
        raise ValueError("--max-plies muss > 0 sein")
    if args.no_capture_draw_plies <= 0:
        raise ValueError("--no-capture-draw-plies muss > 0 sein")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    rule_settings = MillRuleSettings(
        enable_flying=args.enable_flying,
        enable_threefold_repetition=args.enable_threefold_repetition,
        enable_no_capture_draw=args.enable_no_capture_draw,
        no_capture_draw_plies=args.no_capture_draw_plies,
    )
    rules = MillRules(settings=rule_settings)

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    total_samples = 0
    draws = 0
    teacher_wins = {"W": 0, "B": 0}

    with args.output.open("w", encoding="utf-8") as fp:
        for game_index in range(1, args.games + 1):
            session = MillGameSession(rules=rules)
            white_seed = rng.randint(0, 2**31 - 1)
            black_seed = rng.randint(0, 2**31 - 1)

            teacher_a = AlphaBetaMillAI(
                depth=args.teacher_depth,
                random_tiebreak=args.teacher_random_tiebreak,
                seed=white_seed,
            )
            teacher_b = AlphaBetaMillAI(
                depth=args.teacher_depth,
                random_tiebreak=args.teacher_random_tiebreak,
                seed=black_seed,
            )

            a_is_white = (game_index % 2 == 1) if args.alternate_colors else True
            white_ai = teacher_a if a_is_white else teacher_b
            black_ai = teacher_b if a_is_white else teacher_a

            pending_samples: list[dict[str, object]] = []
            while len(session.move_history) < args.max_plies and not session.is_terminal():
                provider = white_ai if session.state.to_move == "W" else black_ai
                legal_moves = list(rules.legal_moves(session.state))
                if not legal_moves:
                    break

                chosen_move = provider.choose_move(session.state, rules, session.move_history)
                try:
                    target_index = legal_moves.index(chosen_move)
                except ValueError as exc:
                    raise RuntimeError("Teacher erzeugte einen Zug, der nicht in der legalen Zugliste steht.") from exc

                pending_samples.append(
                    {
                        "player": session.state.to_move,
                        "state": encode_state_features(session.state, rules).tolist(),
                        "moves": encode_legal_move_features(legal_moves).tolist(),
                        "target_index": target_index,
                    }
                )
                session.apply_move(chosen_move)

            winner = session.winner() if session.is_terminal() else None
            if winner is None:
                draws += 1
            else:
                teacher_wins[winner] += 1

            for sample in pending_samples:
                sample_player = str(sample["player"])
                sample["outcome"] = outcome_for_player(winner, sample_player)
                del sample["player"]
                fp.write(json.dumps(sample, separators=(",", ":")) + "\n")
                total_samples += 1

            if game_index % 20 == 0 or game_index == args.games:
                print(
                    f"Erzeugte Partie {game_index}/{args.games} | "
                    f"samples={total_samples} remis={draws} "
                    f"W_siege={teacher_wins['W']} B_siege={teacher_wins['B']}"
                )

    elapsed = time.perf_counter() - start
    print()
    print("Datenerzeugung abgeschlossen")
    print(f"  Ausgabe: {args.output}")
    print(f"  Partien: {args.games}")
    print(f"  Samples: {total_samples}")
    print(f"  Remis: {draws}")
    print(f"  Siege Weiss: {teacher_wins['W']}")
    print(f"  Siege Schwarz: {teacher_wins['B']}")
    print(f"  Dauer: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
