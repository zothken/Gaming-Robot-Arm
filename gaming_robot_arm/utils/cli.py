"""CLI-Helfer, die von Beispielskripten gemeinsam genutzt werden."""

from __future__ import annotations

from typing import Callable, Mapping

LabelMap = Mapping[str, tuple[float, float]]
FeedbackFn = Callable[[str], None]


def prompt_recording_enabled(prompt: str = "Aufnahme speichern? (y/n): ") -> bool:
    """Fragt nach Aufnahme und liefert True, wenn der Benutzer zustimmt."""
    return input(prompt).strip().lower().startswith("y")


def prompt_board_label(
    board_pixels: LabelMap,
    prompt: str,
    *,
    allowed_labels: set[str] | None = None,
    cancel_labels: set[str] | None = None,
    allow_empty: bool = False,
    repeat_until_valid: bool = True,
    feedback: FeedbackFn | None = None,
    unknown_label_hint: str = "Erlaubt",
) -> str | None:
    """Fragt ein Brett-Label ab und validiert es gegen bekannte/erlaubte Labels."""
    emit = feedback or print
    allowed = {lbl.upper() for lbl in allowed_labels} if allowed_labels else None
    cancel = {lbl.upper() for lbl in cancel_labels} if cancel_labels else set()
    all_labels = ", ".join(sorted(board_pixels.keys()))
    allowed_hint = f"{unknown_label_hint}: {', '.join(sorted(allowed))}" if allowed else None

    while True:
        raw = input(prompt)
        if allow_empty and not raw.strip():
            return None

        label = raw.strip().upper()
        if label in cancel:
            return None

        if label not in board_pixels:
            emit(f"Unbekanntes Label '{label}'. {unknown_label_hint}: {all_labels}")
            if repeat_until_valid:
                continue
            return None

        if allowed is not None and label not in allowed:
            hint = allowed_hint or "Keine verfuegbaren Positionen"
            emit(f"Label '{label}' momentan nicht zulaessig. {hint}")
            if repeat_until_valid:
                continue
            return None

        return label
