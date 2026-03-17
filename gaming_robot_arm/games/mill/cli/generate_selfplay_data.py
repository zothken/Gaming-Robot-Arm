"""Erzeugt ueberwachte Muehle-Trainingsdaten aus neuronalen Selbstspiel-Partien."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

from gaming_robot_arm.games.mill.core.settings import (  # noqa: E402
    MILL_ENABLE_FLYING,
    MILL_ENABLE_NO_CAPTURE_DRAW,
    MILL_ENABLE_THREEFOLD_REPETITION,
    MILL_NO_CAPTURE_DRAW_PLIES,
)
from gaming_robot_arm.games.mill import MillGameSession, MillRuleSettings, MillRules, NeuralMillAI  # noqa: E402
from gaming_robot_arm.games.mill.ml.features import (  # noqa: E402
    encode_legal_move_features,
    encode_state_features,
    outcome_for_player,
)
from gaming_robot_arm.games.mill.ml.selfplay import target_index_or_raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Erzeugt Muehle-Trainingssamples aus neuronalem Selbstspiel.")
    parser.add_argument("--output", type=Path, default=Path("data/mill_selfplay.jsonl"), help="Ausgabe-JSONL-Pfad.")
    parser.add_argument("--games", type=int, default=500, help="Anzahl Selbstspiel-Partien.")
    parser.add_argument("--max-plies", type=int, default=400, help="Harte Zuggrenze pro Partie.")
    parser.add_argument("--seed", type=int, default=1234, help="Basis-RNG-Seed.")
    parser.add_argument("--model", type=Path, required=True, help="Neuraler Checkpoint-Pfad fuer Teilnehmer A.")
    parser.add_argument(
        "--opponent-model",
        type=Path,
        default=None,
        help="Optionaler Checkpoint-Pfad fuer Teilnehmer B; Standard ist --model.",
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling-Temperatur fuer Teilnehmer A.")
    parser.add_argument(
        "--opponent-temperature",
        type=float,
        default=None,
        help="Optionale Sampling-Temperatur fuer Teilnehmer B; Standard ist --temperature.",
    )
    parser.add_argument(
        "--random-tiebreak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aktiviert zufaellige Tie-Breaks im Greedy-Modus (temperature=0).",
    )
    parser.add_argument(
        "--alternate-colors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wechselt ab, welcher Teilnehmer als Weiss startet.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch-Geraet.")
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
    if args.max_plies <= 0:
        raise ValueError("--max-plies muss > 0 sein")
    if args.no_capture_draw_plies <= 0:
        raise ValueError("--no-capture-draw-plies muss > 0 sein")
    if args.temperature < 0:
        raise ValueError("--temperature muss >= 0 sein")
    if args.opponent_temperature is not None and args.opponent_temperature < 0:
        raise ValueError("--opponent-temperature muss >= 0 sein")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    if NeuralMillAI is None:
        raise RuntimeError("NeuralMillAI ist nicht verfuegbar. Bitte zuerst numpy + torch installieren.")

    rule_settings = MillRuleSettings(
        enable_flying=args.enable_flying,
        enable_threefold_repetition=args.enable_threefold_repetition,
        enable_no_capture_draw=args.enable_no_capture_draw,
        no_capture_draw_plies=args.no_capture_draw_plies,
    )
    rules = MillRules(settings=rule_settings)

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    opponent_model = args.opponent_model if args.opponent_model is not None else args.model
    opponent_temperature = args.opponent_temperature if args.opponent_temperature is not None else args.temperature

    # Modellobjekte ueber alle Partien behalten, um wiederholtes Checkpoint-Laden zu vermeiden.
    participant_a = NeuralMillAI(
        model_path=args.model,
        random_tiebreak=args.random_tiebreak,
        temperature=args.temperature,
        seed=rng.randint(0, 2**31 - 1),
        device=args.device,
    )
    participant_b = NeuralMillAI(
        model_path=opponent_model,
        random_tiebreak=args.random_tiebreak,
        temperature=opponent_temperature,
        seed=rng.randint(0, 2**31 - 1),
        device=args.device,
    )

    start = time.perf_counter()
    total_samples = 0
    draws = 0
    wins = {"W": 0, "B": 0}

    with args.output.open("w", encoding="utf-8") as fp:
        for game_index in range(1, args.games + 1):
            session = MillGameSession(rules=rules)
            a_is_white = (game_index % 2 == 1) if args.alternate_colors else True
            white_ai = participant_a if a_is_white else participant_b
            black_ai = participant_b if a_is_white else participant_a

            pending_samples: list[dict[str, object]] = []
            while len(session.move_history) < args.max_plies and not session.is_terminal():
                provider = white_ai if session.state.to_move == "W" else black_ai
                legal_moves = list(rules.legal_moves(session.state))
                if not legal_moves:
                    break

                chosen_move = provider.choose_move(session.state, rules, session.move_history)
                target_index = target_index_or_raise(legal_moves, chosen_move)

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
                wins[winner] += 1

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
                    f"W_siege={wins['W']} B_siege={wins['B']}"
                )

    elapsed = time.perf_counter() - start
    print()
    print("Selfplay-Datenerzeugung abgeschlossen")
    print(f"  Ausgabe: {args.output}")
    print(f"  Modell A: {args.model}")
    print(f"  Modell B: {opponent_model}")
    print(f"  Partien: {args.games}")
    print(f"  Samples: {total_samples}")
    print(f"  Remis: {draws}")
    print(f"  Siege Weiss: {wins['W']}")
    print(f"  Siege Schwarz: {wins['B']}")
    print(f"  Dauer: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
