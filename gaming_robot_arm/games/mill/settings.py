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


__all__ = ["MillRuleSettings"]
