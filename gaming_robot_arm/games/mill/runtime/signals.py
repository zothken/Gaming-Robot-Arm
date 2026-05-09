"""Sentinels fuer nicht-Move-Signale aus den menschlichen Eingabekanaelen."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UndoSignal:
    """Sentinel-Wert: menschliche Eingabe hat 'zurueck' gewaehlt.

    Wird von voice_bridge, _prompt_human_move und vision_bridge zurueckgegeben,
    damit der Game-Loop an einer Stelle Undo-Dispatch durchfuehren kann.
    """


__all__ = ["UndoSignal"]
