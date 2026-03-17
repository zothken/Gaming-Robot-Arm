"""Kernbausteine der Muehle-Domain."""

from .board import ADJACENT, BOARD_LABELS, MILLS, MILLS_BY_POSITION, RINGS
from .constants import PIECES_PER_PLAYER, PLAYERS
from .rules import MillRules, count_pieces, other_player, phase_for_player, state_signature
from .session import MillGameSession, MillMoveProvider
from .settings import (
    DEFAULT_MILL_RULE_SETTINGS,
    MILL_ENABLE_FLYING,
    MILL_ENABLE_NO_CAPTURE_DRAW,
    MILL_ENABLE_THREEFOLD_REPETITION,
    MILL_NO_CAPTURE_DRAW_PLIES,
    MillRuleSettings,
)
from .state import MillState

__all__ = [
    "ADJACENT",
    "BOARD_LABELS",
    "DEFAULT_MILL_RULE_SETTINGS",
    "MILL_ENABLE_FLYING",
    "MILL_ENABLE_NO_CAPTURE_DRAW",
    "MILL_ENABLE_THREEFOLD_REPETITION",
    "MILL_NO_CAPTURE_DRAW_PLIES",
    "MILLS",
    "MILLS_BY_POSITION",
    "MillGameSession",
    "MillMoveProvider",
    "MillRuleSettings",
    "MillRules",
    "MillState",
    "PIECES_PER_PLAYER",
    "PLAYERS",
    "RINGS",
    "count_pieces",
    "other_player",
    "phase_for_player",
    "state_signature",
]
