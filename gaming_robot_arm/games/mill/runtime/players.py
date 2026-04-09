"""Player- und KI-Aufloesung fuer spielbare Muehle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

from gaming_robot_arm.games.common.interfaces import Player
from gaming_robot_arm.games.mill import (
    AlphaBetaMillAI,
    HeuristicMillAI,
    NeuralMillAI,
)
from gaming_robot_arm.games.mill.core.session import MillMoveProvider


GAME_MODES = ("human-vs-human", "human-vs-ai", "ai-vs-ai")
HUMAN_INPUT_MODES = ("manual", "vision", "voice")
AI_BACKENDS = ("heuristic", "alphabeta", "neural")
UARM_CONTROLLED_PLAYERS = ("none", "white", "black", "both", "legacy")


@dataclass(slots=True)
class PlayerController:
    kind: Literal["human", "ai"]
    label: str
    input_mode: str = "manual"
    provider: MillMoveProvider | None = None


def create_ai_provider(args: argparse.Namespace, *, seed_offset: int = 0) -> MillMoveProvider:
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


def build_player_controllers(args: argparse.Namespace) -> dict[Player, PlayerController]:
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
            ai_color: PlayerController(kind="ai", label=f"KI {ai_color}", provider=create_ai_provider(args)),
        }

    if mode == "ai-vs-ai":
        return {
            "W": PlayerController(kind="ai", label="KI W", provider=create_ai_provider(args, seed_offset=0)),
            "B": PlayerController(kind="ai", label="KI B", provider=create_ai_provider(args, seed_offset=1)),
        }

    raise ValueError(f"Nicht unterstuetzter Muehle-Modus: {mode}")


def _legacy_uarm_players(args: argparse.Namespace, *, ai_player_present: bool) -> set[Player]:
    if bool(args.mill_uarm_move_both_players):
        return {"W", "B"}
    if not bool(args.mill_uarm_enable_ai_moves) or not ai_player_present:
        return set()

    if args.mill_mode == "human-vs-ai":
        ai_color: Player = "B" if args.mill_human_color == "W" else "W"
        return {ai_color}
    if args.mill_mode == "ai-vs-ai":
        return {"W", "B"}
    return set()


def resolve_uarm_players(args: argparse.Namespace, *, ai_player_present: bool) -> set[Player]:
    raw_value = str(getattr(args, "mill_uarm_controlled_players", "legacy")).strip().lower()
    if raw_value == "legacy":
        return _legacy_uarm_players(args, ai_player_present=ai_player_present)
    if raw_value == "none":
        return set()
    if raw_value == "white":
        return {"W"}
    if raw_value == "black":
        return {"B"}
    if raw_value == "both":
        return {"W", "B"}
    raise ValueError(f"Nicht unterstuetzter uArm-Spieler-Filter: {raw_value}")


def require_ai_provider(controller: PlayerController) -> MillMoveProvider:
    provider = controller.provider
    if provider is None:
        raise RuntimeError(f"Fehlender KI-Provider fuer Controller '{controller.label}'.")
    return provider


__all__ = [
    "AI_BACKENDS",
    "GAME_MODES",
    "HUMAN_INPUT_MODES",
    "PlayerController",
    "UARM_CONTROLLED_PLAYERS",
    "build_player_controllers",
    "create_ai_provider",
    "require_ai_provider",
    "resolve_uarm_players",
]
