"""Konfigurierbare Muehle-Regel-Toggles fuer Backend und spaetere GUI-Anbindung."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MillRuleSettings:
    """Laufzeitschalter fuer optionale Muehle-Regeln.

    Die Standardwerte erhalten das aktuelle Verhalten, waehrend erweiterte
    Remis-Regeln spaeter ueber ein Einstellungs-UI aktiviert werden koennen.
    """

    enable_flying: bool = True
    enable_threefold_repetition: bool = False
    enable_no_capture_draw: bool = False
    no_capture_draw_plies: int = 200

    def __post_init__(self) -> None:
        if self.no_capture_draw_plies <= 0:
            raise ValueError("no_capture_draw_plies muss groesser als null sein.")


DEFAULT_MILL_RULE_SETTINGS = MillRuleSettings()
MILL_ENABLE_FLYING = DEFAULT_MILL_RULE_SETTINGS.enable_flying
MILL_ENABLE_THREEFOLD_REPETITION = DEFAULT_MILL_RULE_SETTINGS.enable_threefold_repetition
MILL_ENABLE_NO_CAPTURE_DRAW = DEFAULT_MILL_RULE_SETTINGS.enable_no_capture_draw
MILL_NO_CAPTURE_DRAW_PLIES = DEFAULT_MILL_RULE_SETTINGS.no_capture_draw_plies

__all__ = [
    "DEFAULT_MILL_RULE_SETTINGS",
    "MILL_ENABLE_FLYING",
    "MILL_ENABLE_NO_CAPTURE_DRAW",
    "MILL_ENABLE_THREEFOLD_REPETITION",
    "MILL_NO_CAPTURE_DRAW_PLIES",
    "MillRuleSettings",
]
