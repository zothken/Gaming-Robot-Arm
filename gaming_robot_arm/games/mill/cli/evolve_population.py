"""Evolutionsbasiertes Training fuer die Muehle-Neural-KI gegen AlphaBeta.

Idee:
- Start mit zufaellig initialisierten Netzen (kein Teacher, kein Gradiententraining).
- Individuen spielen gegen AlphaBeta.
- Fitness basiert auf Match-Ergebnissen.
- Top-Individuen werden Eltern der naechsten Generation.
- Neue Generation entsteht aus Crossover + Mutation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
from typing import Any, NotRequired, TypedDict

from gaming_robot_arm.games.mill.core.settings import (
    MILL_ENABLE_FLYING,
    MILL_ENABLE_NO_CAPTURE_DRAW,
    MILL_ENABLE_THREEFOLD_REPETITION,
    MILL_NO_CAPTURE_DRAW_PLIES,
)
from gaming_robot_arm.games.mill import AlphaBetaMillAI, MillGameSession, MillRuleSettings, MillRules, NeuralMillAI
from gaming_robot_arm.games.mill.ml.evolution import clone_state_dict, lerp
from gaming_robot_arm.games.mill.ml.features import (
    encode_legal_move_features,
    encode_state_features,
)
from gaming_robot_arm.games.mill.ml.model import (
    MillPolicyValueNet,
    load_checkpoint,
    save_checkpoint,
    select_torch_device,
)

import numpy as np
import torch


@dataclass(slots=True)
class EvalStats:
    fitness: float
    overall_score: float
    white_score: float
    black_score: float
    wins: int
    draws: int
    losses: int
    white_wins: int
    white_losses: int
    black_wins: int
    black_losses: int
    avg_plies: float


@dataclass(slots=True)
class Individual:
    name: str
    state: dict[str, torch.Tensor]
    parent_a: str | None = None
    parent_b: str | None = None
    stats: EvalStats | None = None


@dataclass(slots=True)
class HeadToHeadStats:
    candidate_wins: int
    champion_wins: int
    draws: int
    candidate_wins_white: int
    candidate_wins_black: int
    champion_wins_white: int
    champion_wins_black: int
    avg_plies: float


class ModelKwargs(TypedDict):
    hidden_dim: int
    dropout: float
    state_dim: NotRequired[int]
    move_dim: NotRequired[int]


class EvolutionNeuralAI:
    """Leichtgewichtiger Move-Provider fuer evolutionaere Bewertung."""

    def __init__(
        self,
        *,
        model: MillPolicyValueNet,
        rng: random.Random,
        random_move_prob: float,
        temperature: float,
        device: torch.device,
    ) -> None:
        self._model = model
        self._rng = rng
        self._random_move_prob = random_move_prob
        self._temperature = temperature
        self._device = device

    def choose_move(self, state, rules: MillRules, move_history) -> Any:
        del move_history
        legal_moves = list(rules.legal_moves(state))
        if not legal_moves:
            raise ValueError("Keine legalen Zuege verfuegbar.")

        if self._rng.random() < self._random_move_prob:
            return self._rng.choice(legal_moves)

        state_features = encode_state_features(state, rules).astype(np.float32)
        move_features = encode_legal_move_features(legal_moves).astype(np.float32)

        with torch.no_grad():
            state_batch = torch.from_numpy(state_features).unsqueeze(0).to(self._device)
            move_batch = torch.from_numpy(move_features).unsqueeze(0).to(self._device)
            move_mask = torch.ones((1, move_features.shape[0]), dtype=torch.bool, device=self._device)
            logits, _ = self._model(state_batch, move_batch, move_mask)
            scores = logits.squeeze(0).cpu().numpy().astype(np.float32)

        if self._temperature > 0.0:
            scaled = scores / max(self._temperature, 1e-6)
            scaled = scaled - float(np.max(scaled))
            probs = np.exp(scaled)
            probs = probs / np.sum(probs)
            move_idx = self._rng.choices(range(len(legal_moves)), weights=probs.tolist(), k=1)[0]
            return legal_moves[move_idx]

        max_score = float(np.max(scores))
        best_indices = [idx for idx, score in enumerate(scores) if float(score) == max_score]
        return legal_moves[self._rng.choice(best_indices)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evolutions-Training der Muehle-Neural-KI gegen AlphaBeta.")
    parser.add_argument("--generations", type=int, default=20, help="Anzahl Generationen.")
    parser.add_argument("--population-size", type=int, default=24, help="Individuen pro Generation.")
    parser.add_argument("--parents", type=int, default=6, help="Anzahl selektierter Eltern pro Generation.")
    parser.add_argument("--elitism", type=int, default=4, help="Top-Individuen, die unveraendert uebernommen werden.")
    parser.add_argument("--games-per-individual", type=int, default=60, help="Partien je Individuum gegen AlphaBeta.")
    parser.add_argument("--draw-score", type=float, default=0.35, help="Fitness-Gutschrift pro Remis in [0, 1].")
    parser.add_argument(
        "--color-balance-weight",
        type=float,
        default=0.30,
        help="Gewichtung des schwaecheren Farb-Scores im Fitnesswert in [0, 1].",
    )
    parser.add_argument(
        "--color-imbalance-penalty",
        type=float,
        default=0.0,
        help="Optionaler Strafterm fuer Farb-Ungleichgewicht (>= 0).",
    )

    parser.add_argument(
        "--ab-depth-stages",
        type=str,
        default="2,3",
        help="Kommagetrennte AlphaBeta-Tiefen je Stage, z.B. '1,2,3'.",
    )
    parser.add_argument("--stage-generations", type=int, default=8, help="Generationen pro Stage-Tiefe.")
    parser.add_argument(
        "--ab-random-tiebreak",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Zufalls-Tiebreak im AlphaBeta-Gegner.",
    )
    parser.add_argument(
        "--fitness-ab-random-tiebreak",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Zufalls-Tiebreak im AlphaBeta waehrend Fitness-Evaluation.",
    )

    parser.add_argument(
        "--random-move-prob-start",
        type=float,
        default=0.6,
        help="Startwahrscheinlichkeit fuer komplett zufaelligen Zug.",
    )
    parser.add_argument(
        "--random-move-prob-end",
        type=float,
        default=0.02,
        help="Endwahrscheinlichkeit fuer komplett zufaelligen Zug.",
    )
    parser.add_argument("--temperature-start", type=float, default=0.8, help="Starttemperatur fuer Sampling.")
    parser.add_argument("--temperature-end", type=float, default=0.10, help="Endtemperatur fuer Sampling.")
    parser.add_argument("--mutation-std-start", type=float, default=0.03, help="Start-Std fuer Gewichtsmutation.")
    parser.add_argument("--mutation-std-end", type=float, default=0.005, help="End-Std fuer Gewichtsmutation.")
    parser.add_argument(
        "--fitness-random-move-prob",
        type=float,
        default=0.0,
        help="Zufallszugwahrscheinlichkeit fuer Fitness-Evaluation (unabhaengig vom Evolutions-Explorationsplan).",
    )
    parser.add_argument(
        "--fitness-temperature",
        type=float,
        default=0.05,
        help="Sampling-Temperatur fuer Fitness-Evaluation (unabhaengig vom Evolutions-Explorationsplan).",
    )
    parser.add_argument("--crossover-rate", type=float, default=0.6, help="Wahrscheinlichkeit fuer Tensor-Crossover.")

    parser.add_argument("--hidden-dim", type=int, default=192, help="Hidden-Dimension des Netzes.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout des Netzes.")
    parser.add_argument("--max-plies", type=int, default=400, help="Maximale Halbzuege pro Partie.")
    parser.add_argument("--seed", type=int, default=20260221, help="Globaler Seed.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch-Geraet.")
    parser.add_argument("--save-parents", type=int, default=4, help="Anzahl Top-Individuen, die pro Generation gespeichert werden.")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optionaler Start-Checkpoint fuer Population (ein exakter Seed + mutierte Klone).",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models/evolution_ab"),
        help="Wurzelverzeichnis fuer Evolutionsartefakte.",
    )
    parser.add_argument(
        "--champion-output",
        type=Path,
        default=Path("models/champion/mill_champion.pt"),
        help="Zielpfad fuer den finalen evolvierten Champion.",
    )
    parser.add_argument(
        "--promote-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Kopiert den besten evolvierten Checkpoint nach --champion-output.",
    )
    parser.add_argument(
        "--final-benchmark-games",
        type=int,
        default=50,
        help="Partien des finalen Benchmarks (best_overall vs alter Champion) vor einer Promotion.",
    )
    parser.add_argument(
        "--final-benchmark-draw-score",
        type=float,
        default=0.5,
        help="Remis-Gewichtung fuer den finalen Promotionsbenchmark.",
    )
    parser.add_argument(
        "--final-benchmark-threshold",
        type=float,
        default=0.5,
        help="Mindestscore des Kandidaten im finalen Benchmark fuer Promotion (strict > threshold).",
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


def _parse_depth_stages(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        depth = int(item)
        if depth <= 0:
            raise ValueError("Werte in --ab-depth-stages muessen > 0 sein.")
        values.append(depth)
    if not values:
        raise ValueError("--ab-depth-stages darf nicht leer sein.")
    return values


def validate_args(args: argparse.Namespace) -> None:
    if args.generations <= 0:
        raise ValueError("--generations muss > 0 sein.")
    if args.population_size <= 1:
        raise ValueError("--population-size muss > 1 sein.")
    if args.parents < 2:
        raise ValueError("--parents muss >= 2 sein.")
    if args.parents > args.population_size:
        raise ValueError("--parents darf population-size nicht uebersteigen.")
    if args.elitism < 0:
        raise ValueError("--elitism muss >= 0 sein.")
    if args.elitism > args.parents:
        raise ValueError("--elitism darf parents nicht uebersteigen.")
    if args.games_per_individual <= 0:
        raise ValueError("--games-per-individual muss > 0 sein.")
    if args.final_benchmark_games <= 0:
        raise ValueError("--final-benchmark-games muss > 0 sein.")
    if args.stage_generations <= 0:
        raise ValueError("--stage-generations muss > 0 sein.")
    if args.draw_score < 0.0 or args.draw_score > 1.0:
        raise ValueError("--draw-score muss in [0, 1] liegen.")
    if args.color_balance_weight < 0.0 or args.color_balance_weight > 1.0:
        raise ValueError("--color-balance-weight muss in [0, 1] liegen.")
    if args.color_imbalance_penalty < 0.0:
        raise ValueError("--color-imbalance-penalty muss >= 0 sein.")
    if args.final_benchmark_draw_score < 0.0 or args.final_benchmark_draw_score > 1.0:
        raise ValueError("--final-benchmark-draw-score muss in [0, 1] liegen.")
    if args.final_benchmark_threshold < 0.0 or args.final_benchmark_threshold > 1.0:
        raise ValueError("--final-benchmark-threshold muss in [0, 1] liegen.")
    if args.random_move_prob_start < 0.0 or args.random_move_prob_start > 1.0:
        raise ValueError("--random-move-prob-start muss in [0, 1] liegen.")
    if args.random_move_prob_end < 0.0 or args.random_move_prob_end > 1.0:
        raise ValueError("--random-move-prob-end muss in [0, 1] liegen.")
    if args.fitness_random_move_prob < 0.0 or args.fitness_random_move_prob > 1.0:
        raise ValueError("--fitness-random-move-prob muss in [0, 1] liegen.")
    if args.temperature_start < 0.0 or args.temperature_end < 0.0:
        raise ValueError("--temperature-start/--temperature-end muessen >= 0 sein.")
    if args.fitness_temperature < 0.0:
        raise ValueError("--fitness-temperature muss >= 0 sein.")
    if args.mutation_std_start < 0.0 or args.mutation_std_end < 0.0:
        raise ValueError("--mutation-std-start/--mutation-std-end muessen >= 0 sein.")
    if args.crossover_rate < 0.0 or args.crossover_rate > 1.0:
        raise ValueError("--crossover-rate muss in [0, 1] liegen.")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim muss > 0 sein.")
    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError("--dropout muss in [0, 1) liegen.")
    if args.max_plies <= 0:
        raise ValueError("--max-plies muss > 0 sein.")
    if args.no_capture_draw_plies <= 0:
        raise ValueError("--no-capture-draw-plies muss > 0 sein.")
    if args.save_parents <= 0:
        raise ValueError("--save-parents muss > 0 sein.")


def _build_random_individual(name: str, model_kwargs: ModelKwargs) -> Individual:
    model = MillPolicyValueNet(**model_kwargs)
    return Individual(name=name, state=clone_state_dict(model.state_dict()))


def _load_population_seed_from_checkpoint(
    *,
    checkpoint_path: Path,
) -> tuple[ModelKwargs, dict[str, torch.Tensor], int | None]:
    payload = load_checkpoint(checkpoint_path, device=torch.device("cpu"))
    ckpt_model_kwargs_raw = payload.get("model_kwargs", {})
    if not isinstance(ckpt_model_kwargs_raw, dict):
        raise ValueError(f"Checkpoint hat ungueltige model_kwargs: {checkpoint_path}")

    model_kwargs: ModelKwargs = {
        "state_dim": int(ckpt_model_kwargs_raw["state_dim"]),
        "move_dim": int(ckpt_model_kwargs_raw["move_dim"]),
        "hidden_dim": int(ckpt_model_kwargs_raw["hidden_dim"]),
        "dropout": float(ckpt_model_kwargs_raw["dropout"]),
    }
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint hat keinen model_state: {checkpoint_path}")
    generation_raw = payload.get("epoch")
    generation = int(generation_raw) if isinstance(generation_raw, int) else None
    return model_kwargs, clone_state_dict(model_state), generation


def _load_model_from_state(
    *,
    model_kwargs: ModelKwargs,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> MillPolicyValueNet:
    model = MillPolicyValueNet(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _play_candidate_vs_ab(
    *,
    candidate_ai: EvolutionNeuralAI,
    candidate_is_white: bool,
    rules: MillRules,
    max_plies: int,
    ab_depth: int,
    ab_random_tiebreak: bool,
    rng: random.Random,
) -> tuple[str | None, int]:
    session = MillGameSession(rules=rules)
    ab_ai = AlphaBetaMillAI(
        depth=ab_depth,
        random_tiebreak=ab_random_tiebreak,
        seed=rng.randint(0, 2**31 - 1),
    )
    candidate_color = "W" if candidate_is_white else "B"
    white_ai = candidate_ai if candidate_is_white else ab_ai
    black_ai = ab_ai if candidate_is_white else candidate_ai

    while len(session.move_history) < max_plies and not session.is_terminal():
        provider = white_ai if session.state.to_move == "W" else black_ai
        move = session.choose_ai_move(provider)
        session.apply_move(move)

    winner = session.winner() if session.is_terminal() else None
    if winner == candidate_color:
        return "win", len(session.move_history)
    if winner is None:
        return "draw", len(session.move_history)
    return "loss", len(session.move_history)


def _run_final_neural_benchmark(
    *,
    candidate_path: Path,
    champion_path: Path,
    rules: MillRules,
    games: int,
    max_plies: int,
    device_name: str,
) -> HeadToHeadStats:
    if NeuralMillAI is None:
        raise RuntimeError("NeuralMillAI ist nicht verfuegbar. Bitte numpy + torch installieren.")

    candidate_ai = NeuralMillAI(
        model_path=candidate_path,
        random_tiebreak=False,
        temperature=0.0,
        seed=11_111,
        device=device_name,
    )
    champion_ai = NeuralMillAI(
        model_path=champion_path,
        random_tiebreak=False,
        temperature=0.0,
        seed=22_222,
        device=device_name,
    )

    candidate_wins = 0
    champion_wins = 0
    draws = 0
    candidate_wins_white = 0
    candidate_wins_black = 0
    champion_wins_white = 0
    champion_wins_black = 0
    total_plies = 0

    for game_idx in range(1, games + 1):
        candidate_is_white = (game_idx % 2) == 1
        session = MillGameSession(rules=rules)
        white_ai = candidate_ai if candidate_is_white else champion_ai
        black_ai = champion_ai if candidate_is_white else candidate_ai

        while len(session.move_history) < max_plies and not session.is_terminal():
            provider = white_ai if session.state.to_move == "W" else black_ai
            move = session.choose_ai_move(provider)
            session.apply_move(move)

        total_plies += len(session.move_history)
        winner = session.winner() if session.is_terminal() else None
        if winner is None:
            draws += 1
            continue

        if candidate_is_white:
            if winner == "W":
                candidate_wins += 1
                candidate_wins_white += 1
            else:
                champion_wins += 1
                champion_wins_black += 1
        else:
            if winner == "B":
                candidate_wins += 1
                candidate_wins_black += 1
            else:
                champion_wins += 1
                champion_wins_white += 1

    return HeadToHeadStats(
        candidate_wins=candidate_wins,
        champion_wins=champion_wins,
        draws=draws,
        candidate_wins_white=candidate_wins_white,
        candidate_wins_black=candidate_wins_black,
        champion_wins_white=champion_wins_white,
        champion_wins_black=champion_wins_black,
        avg_plies=total_plies / games,
    )


def _evaluate_individual(
    *,
    individual: Individual,
    model_kwargs: ModelKwargs,
    device: torch.device,
    rules: MillRules,
    games_per_individual: int,
    max_plies: int,
    ab_depth: int,
    ab_random_tiebreak: bool,
    random_move_prob: float,
    temperature: float,
    draw_score: float,
    color_balance_weight: float,
    color_imbalance_penalty: float,
    seed: int,
) -> EvalStats:
    eval_rng = random.Random(seed)
    model = _load_model_from_state(model_kwargs=model_kwargs, state_dict=individual.state, device=device)

    wins = 0
    draws = 0
    losses = 0
    white_wins = 0
    white_losses = 0
    black_wins = 0
    black_losses = 0
    total_plies = 0

    for game_idx in range(games_per_individual):
        candidate_is_white = (game_idx % 2) == 0
        candidate_ai = EvolutionNeuralAI(
            model=model,
            rng=random.Random(eval_rng.randint(0, 2**31 - 1)),
            random_move_prob=random_move_prob,
            temperature=temperature,
            device=device,
        )

        outcome, plies = _play_candidate_vs_ab(
            candidate_ai=candidate_ai,
            candidate_is_white=candidate_is_white,
            rules=rules,
            max_plies=max_plies,
            ab_depth=ab_depth,
            ab_random_tiebreak=ab_random_tiebreak,
            rng=eval_rng,
        )
        total_plies += plies

        if outcome == "win":
            wins += 1
            if candidate_is_white:
                white_wins += 1
            else:
                black_wins += 1
        elif outcome == "draw":
            draws += 1
        else:
            losses += 1
            if candidate_is_white:
                white_losses += 1
            else:
                black_losses += 1

    overall_score = (wins + draw_score * draws) / games_per_individual

    white_games = (games_per_individual + 1) // 2
    black_games = games_per_individual // 2
    white_draws = white_games - white_wins - white_losses
    black_draws = black_games - black_wins - black_losses

    white_score = (
        (white_wins + draw_score * white_draws) / white_games if white_games > 0 else overall_score
    )
    black_score = (
        (black_wins + draw_score * black_draws) / black_games if black_games > 0 else overall_score
    )
    weakest_color_score = min(white_score, black_score)
    imbalance_penalty = color_imbalance_penalty * abs(white_score - black_score)
    fitness = (1.0 - color_balance_weight) * overall_score + color_balance_weight * weakest_color_score
    fitness = max(0.0, fitness - imbalance_penalty)

    return EvalStats(
        fitness=fitness,
        overall_score=overall_score,
        white_score=white_score,
        black_score=black_score,
        wins=wins,
        draws=draws,
        losses=losses,
        white_wins=white_wins,
        white_losses=white_losses,
        black_wins=black_wins,
        black_losses=black_losses,
        avg_plies=total_plies / games_per_individual,
    )


def _select_parent(parents: list[Individual], rng: random.Random) -> Individual:
    weights = [max((parent.stats.fitness if parent.stats is not None else 0.0), 0.0) + 1e-6 for parent in parents]
    return rng.choices(parents, weights=weights, k=1)[0]


def _breed_child_state(
    *,
    parent_a_state: dict[str, torch.Tensor],
    parent_b_state: dict[str, torch.Tensor],
    crossover_rate: float,
    mutation_std: float,
    rng: random.Random,
) -> dict[str, torch.Tensor]:
    child: dict[str, torch.Tensor] = {}
    for key, a_tensor in parent_a_state.items():
        b_tensor = parent_b_state[key]
        if a_tensor.dtype.is_floating_point:
            if rng.random() < crossover_rate:
                mask = torch.rand_like(a_tensor, dtype=torch.float32) < 0.5
                mixed = torch.where(mask, a_tensor, b_tensor)
            else:
                mixed = a_tensor if rng.random() < 0.5 else b_tensor
            child_tensor = mixed.clone()
            if mutation_std > 0.0:
                child_tensor = child_tensor + torch.randn_like(child_tensor) * mutation_std
        else:
            child_tensor = (a_tensor if rng.random() < 0.5 else b_tensor).clone()
        child[key] = child_tensor
    return child


def _save_state_as_checkpoint(
    *,
    path: Path,
    model_kwargs: ModelKwargs,
    state_dict: dict[str, torch.Tensor],
    generation: int,
    metrics: dict[str, float | int | str | bool],
) -> None:
    model = MillPolicyValueNet(**model_kwargs)
    model.load_state_dict({key: value.detach().cpu() for key, value in state_dict.items()})
    save_checkpoint(
        path=path,
        model=model,
        optimizer=None,
        epoch=generation,
        best_val_loss=None,
        extra_metrics={key: float(value) for key, value in metrics.items() if isinstance(value, (float, int, bool))},
    )


def _individual_sort_key(individual: Individual) -> tuple[float, float, int, int, int, float]:
    assert individual.stats is not None
    stats = individual.stats
    # Tie-breaker: bevorzugt Kandidaten, die bei gleichem W/D/L laenger ueberleben.
    return (
        stats.fitness,
        min(stats.white_score, stats.black_score),
        stats.wins,
        stats.draws,
        -stats.losses,
        stats.avg_plies,
    )


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    depth_stages = _parse_depth_stages(args.ab_depth_stages)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_torch_device(args.device)
    print(f"Verwende Geraet: {device}")

    rules = MillRules(
        settings=MillRuleSettings(
            enable_flying=args.enable_flying,
            enable_threefold_repetition=args.enable_threefold_repetition,
            enable_no_capture_draw=args.enable_no_capture_draw,
            no_capture_draw_plies=args.no_capture_draw_plies,
        )
    )

    model_kwargs: ModelKwargs = {
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    init_seed_state: dict[str, torch.Tensor] | None = None
    init_seed_generation: int | None = None
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.exists():
            raise FileNotFoundError(f"Init-Checkpoint nicht gefunden: {args.init_checkpoint}")
        ckpt_model_kwargs, init_seed_state, init_seed_generation = _load_population_seed_from_checkpoint(
            checkpoint_path=args.init_checkpoint
        )
        if (
            int(ckpt_model_kwargs["hidden_dim"]) != int(args.hidden_dim)
            or float(ckpt_model_kwargs["dropout"]) != float(args.dropout)
        ):
            print(
                "Hinweis: --init-checkpoint verwendet andere Architekturparameter; "
                "hidden_dim/dropout werden aus dem Checkpoint uebernommen."
            )
        model_kwargs = ckpt_model_kwargs
        print(
            f"Initialisiere Population aus Checkpoint: {args.init_checkpoint}"
            + (f" (checkpoint_epoch={init_seed_generation})" if init_seed_generation is not None else "")
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    generations_dir = args.output_root / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "evolution_summary.jsonl"
    best_overall_path = args.output_root / "best_overall.pt"

    if init_seed_state is None:
        population = [
            _build_random_individual(name=f"g001_i{idx + 1:03d}", model_kwargs=model_kwargs)
            for idx in range(args.population_size)
        ]
    else:
        population = [
            Individual(
                name="g001_seed001",
                state=clone_state_dict(init_seed_state),
            )
        ]
        init_rng = random.Random(args.seed + 7_777_777)
        while len(population) < args.population_size:
            child_idx = len(population) + 1
            child_state = _breed_child_state(
                parent_a_state=init_seed_state,
                parent_b_state=init_seed_state,
                crossover_rate=0.0,
                mutation_std=args.mutation_std_start,
                rng=init_rng,
            )
            population.append(
                Individual(
                    name=f"g001_seed{child_idx:03d}",
                    state=child_state,
                    parent_a="seed",
                    parent_b="seed",
                )
            )

    best_overall: Individual | None = None
    best_overall_generation = 0
    summary_path.write_text("", encoding="utf-8")

    for generation_idx in range(args.generations):
        generation = generation_idx + 1
        progress = generation_idx / max(args.generations - 1, 1)
        stage_idx = min(generation_idx // args.stage_generations, len(depth_stages) - 1)
        ab_depth = depth_stages[stage_idx]
        random_move_prob = lerp(args.random_move_prob_start, args.random_move_prob_end, progress)
        temperature = lerp(args.temperature_start, args.temperature_end, progress)
        mutation_std = lerp(args.mutation_std_start, args.mutation_std_end, progress)
        fitness_random_move_prob = args.fitness_random_move_prob
        fitness_temperature = args.fitness_temperature
        fitness_ab_random_tiebreak = args.fitness_ab_random_tiebreak

        print()
        print(
            f"=== Generation {generation:03d}/{args.generations:03d} | "
            f"ab_depth={ab_depth} random_move_prob={random_move_prob:.3f} "
            f"temperature={temperature:.3f} mutation_std={mutation_std:.4f} | "
            f"fitness_random_move_prob={fitness_random_move_prob:.3f} "
            f"fitness_temperature={fitness_temperature:.3f} "
            f"fitness_ab_random_tiebreak={fitness_ab_random_tiebreak} | "
            f"color_balance_weight={args.color_balance_weight:.2f} "
            f"color_imbalance_penalty={args.color_imbalance_penalty:.3f} ==="
        )

        for idx, individual in enumerate(population):
            eval_seed = args.seed + generation * 1_000_000 + idx * 10_000
            individual.stats = _evaluate_individual(
                individual=individual,
                model_kwargs=model_kwargs,
                device=device,
                rules=rules,
                games_per_individual=args.games_per_individual,
                max_plies=args.max_plies,
                ab_depth=ab_depth,
                ab_random_tiebreak=fitness_ab_random_tiebreak,
                random_move_prob=fitness_random_move_prob,
                temperature=fitness_temperature,
                draw_score=args.draw_score,
                color_balance_weight=args.color_balance_weight,
                color_imbalance_penalty=args.color_imbalance_penalty,
                seed=eval_seed,
            )
            print(
                f"{individual.name}: fitness={individual.stats.fitness:.3f} "
                f"(overall={individual.stats.overall_score:.3f} "
                f"Wscore={individual.stats.white_score:.3f} Bscore={individual.stats.black_score:.3f}) "
                f"W={individual.stats.wins} D={individual.stats.draws} L={individual.stats.losses} "
                f"W_asW={individual.stats.white_wins} W_asB={individual.stats.black_wins}"
            )

        all_stats = [individual.stats for individual in population]
        assert all(stat is not None for stat in all_stats)
        losses_set = {stat.losses for stat in all_stats if stat is not None}
        wins_set = {stat.wins for stat in all_stats if stat is not None}
        draws_set = {stat.draws for stat in all_stats if stat is not None}
        if len(losses_set) == 1 and len(wins_set) == 1 and len(draws_set) == 1:
            print(
                "Tie-Break-Fallback aktiv: alle Individuen haben identische W/D/L-Werte; "
                "Sortierung nach durchschnittlich ueberlebten Halbzuegen."
            )

        ranked = sorted(population, key=_individual_sort_key, reverse=True)
        parents = ranked[: args.parents]
        best = parents[0]
        assert best.stats is not None

        if best_overall is None or _individual_sort_key(best) > _individual_sort_key(best_overall):
            best_overall = Individual(
                name=best.name,
                state=clone_state_dict(best.state),
                parent_a=best.parent_a,
                parent_b=best.parent_b,
                stats=best.stats,
            )
            best_overall_generation = generation
            _save_state_as_checkpoint(
                path=best_overall_path,
                model_kwargs=model_kwargs,
                state_dict=best_overall.state,
                generation=generation,
                metrics={
                    "fitness": best.stats.fitness,
                    "overall_score": best.stats.overall_score,
                    "white_score": best.stats.white_score,
                    "black_score": best.stats.black_score,
                    "wins": best.stats.wins,
                    "draws": best.stats.draws,
                    "losses": best.stats.losses,
                },
            )

        gen_dir = generations_dir / f"gen_{generation:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        _save_state_as_checkpoint(
            path=gen_dir / "best.pt",
            model_kwargs=model_kwargs,
            state_dict=best.state,
            generation=generation,
            metrics={
                "fitness": best.stats.fitness,
                "overall_score": best.stats.overall_score,
                "white_score": best.stats.white_score,
                "black_score": best.stats.black_score,
                "wins": best.stats.wins,
                "draws": best.stats.draws,
                "losses": best.stats.losses,
            },
        )

        for parent_idx, parent in enumerate(parents[: min(args.save_parents, len(parents))], start=1):
            assert parent.stats is not None
            _save_state_as_checkpoint(
                path=gen_dir / f"parent_{parent_idx:02d}.pt",
                model_kwargs=model_kwargs,
                state_dict=parent.state,
                generation=generation,
                metrics={
                    "fitness": parent.stats.fitness,
                    "overall_score": parent.stats.overall_score,
                    "white_score": parent.stats.white_score,
                    "black_score": parent.stats.black_score,
                    "wins": parent.stats.wins,
                    "draws": parent.stats.draws,
                    "losses": parent.stats.losses,
                },
            )

        generation_summary: dict[str, Any] = {
            "generation": generation,
            "ab_depth": ab_depth,
            "random_move_prob": random_move_prob,
            "temperature": temperature,
            "mutation_std": mutation_std,
            "fitness_random_move_prob": fitness_random_move_prob,
            "fitness_temperature": fitness_temperature,
            "fitness_ab_random_tiebreak": fitness_ab_random_tiebreak,
            "color_balance_weight": args.color_balance_weight,
            "color_imbalance_penalty": args.color_imbalance_penalty,
            "best_name": best.name,
            "best_fitness": best.stats.fitness,
            "best_overall_score": best.stats.overall_score,
            "best_white_score": best.stats.white_score,
            "best_black_score": best.stats.black_score,
            "best_wins": best.stats.wins,
            "best_draws": best.stats.draws,
            "best_losses": best.stats.losses,
            "best_white_wins": best.stats.white_wins,
            "best_black_wins": best.stats.black_wins,
            "best_avg_plies": best.stats.avg_plies,
        }
        with summary_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(generation_summary, separators=(",", ":")) + "\n")

        print(
            f"Generation {generation:03d} best: {best.name} "
            f"fitness={best.stats.fitness:.3f} "
            f"(overall={best.stats.overall_score:.3f} "
            f"Wscore={best.stats.white_score:.3f} Bscore={best.stats.black_score:.3f}) "
            f"W={best.stats.wins} D={best.stats.draws} L={best.stats.losses}"
        )

        if generation == args.generations:
            break

        next_generation = generation + 1
        breeder_rng = random.Random(args.seed + 10_000_000 + generation)
        next_population: list[Individual] = []

        # Elitism: top N unveraendert uebernehmen.
        for elite_idx, elite in enumerate(parents[: args.elitism], start=1):
            next_population.append(
                Individual(
                    name=f"g{next_generation:03d}_elite{elite_idx:02d}",
                    state=clone_state_dict(elite.state),
                    parent_a=elite.name,
                    parent_b=elite.name,
                )
            )

        # Restliche Population aus Eltern erzeugen.
        while len(next_population) < args.population_size:
            parent_a = _select_parent(parents, breeder_rng)
            parent_b = _select_parent(parents, breeder_rng)
            if len(parents) > 1:
                guard = 0
                while parent_b is parent_a and guard < 8:
                    parent_b = _select_parent(parents, breeder_rng)
                    guard += 1

            child_state = _breed_child_state(
                parent_a_state=parent_a.state,
                parent_b_state=parent_b.state,
                crossover_rate=args.crossover_rate,
                mutation_std=mutation_std,
                rng=breeder_rng,
            )
            child_idx = len(next_population) + 1
            next_population.append(
                Individual(
                    name=f"g{next_generation:03d}_child{child_idx:03d}",
                    state=child_state,
                    parent_a=parent_a.name,
                    parent_b=parent_b.name,
                )
            )

        population = next_population

    print()
    print("Evolution abgeschlossen.")
    print(f"Summary: {summary_path}")
    print(f"Best overall: {best_overall_path}")
    if best_overall is not None and best_overall.stats is not None:
        print(
            f"Best overall generation={best_overall_generation} "
            f"fitness={best_overall.stats.fitness:.3f} "
            f"W={best_overall.stats.wins} D={best_overall.stats.draws} L={best_overall.stats.losses}"
        )

    if args.promote_final:
        if not best_overall_path.exists():
            raise FileNotFoundError(f"Evolutions-Champion fehlt: {best_overall_path}")

        if args.champion_output.exists():
            h2h = _run_final_neural_benchmark(
                candidate_path=best_overall_path,
                champion_path=args.champion_output,
                rules=rules,
                games=args.final_benchmark_games,
                max_plies=args.max_plies,
                device_name=args.device,
            )
            final_score = (
                h2h.candidate_wins + args.final_benchmark_draw_score * h2h.draws
            ) / args.final_benchmark_games
            print()
            print("Finaler Champion-Benchmark (Kandidat vs alter Champion)")
            print(f"  Partien: {args.final_benchmark_games}")
            print(f"  Kandidat siege: {h2h.candidate_wins}")
            print(f"    davon Weiss: {h2h.candidate_wins_white}, Schwarz: {h2h.candidate_wins_black}")
            print(f"  Alter Champion siege: {h2h.champion_wins}")
            print(f"    davon Weiss: {h2h.champion_wins_white}, Schwarz: {h2h.champion_wins_black}")
            print(f"  Remis: {h2h.draws}")
            print(f"  Ø Halbzuege/Partie: {h2h.avg_plies:.1f}")
            print(
                f"  Kandidat-Score: {final_score:.3f} "
                f"(draw_score={args.final_benchmark_draw_score:.2f}, "
                f"Schwelle: > {args.final_benchmark_threshold:.3f})"
            )

            if final_score > args.final_benchmark_threshold:
                args.champion_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(best_overall_path, args.champion_output)
                print(f"Champion aktualisiert: {args.champion_output}")
            else:
                print("Champion NICHT aktualisiert: finaler Benchmark nicht bestanden.")
        else:
            args.champion_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_overall_path, args.champion_output)
            print(
                f"Kein alter Champion vorhanden; Kandidat direkt uebernommen: {args.champion_output}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
