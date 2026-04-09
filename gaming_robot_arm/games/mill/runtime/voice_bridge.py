"""Sprachsteuerung fuer menschliche Muehle-Zuege.

  AudioProcess (stt.py)  -> text_queue -> CommandProcess (command_process.py) -> match_queue
  VoiceBridge.listen_for_move() liest aus match_queue und mappt auf legale Zuege.
"""

from __future__ import annotations

import re
import threading
from queue import Queue
from typing import TYPE_CHECKING, Sequence

from .stt import AudioProcess
from .mill_commands import MillCommands
from .command_process import CommandProcess

if TYPE_CHECKING:
    from gaming_robot_arm.games.common.interfaces import Move


# ---------------------------------------------------------------------------
# Zahlwort-Mapping (Fallback: Zugnummer nennen)
# ---------------------------------------------------------------------------

_GERMAN_NUMBERS: dict[str, int] = {
    "eins": 1, "ein": 1, "zwei": 2, "zwo": 2, "drei": 3, "vier": 4,
    "fünf": 5, "fuenf": 5, "sechs": 6, "sieben": 7, "acht": 8,
    "neun": 9, "zehn": 10, "elf": 11, "zwölf": 12, "zwoelf": 12,
    "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15, "sechzehn": 16,
    "siebzehn": 17, "achtzehn": 18, "neunzehn": 19, "zwanzig": 20,
}


def _parse_number(text: str) -> int | None:
    """Fallback: Zugnummer aus Text (Ziffer oder deutsches Zahlwort)."""
    m = re.search(r'\b(\d+)\b', text)
    if m:
        return int(m.group(1))
    lower = text.lower()
    for word, n in _GERMAN_NUMBERS.items():
        if word in lower:
            return n
    return None


def _match_positions_to_move(positions: list[str], legal_moves: Sequence[Move]) -> Move | None:
    """Mappt extrahierte Positionen auf einen eindeutigen legalen Zug."""
    if len(positions) == 0:
        return None
    if len(positions) == 1:
        # Setzphase: nur Zielfeld
        candidates = [m for m in legal_moves if m.dst == positions[0] and m.src is None]
        return candidates[0] if len(candidates) == 1 else None
    # Zugphase: src + dst (+ optionaler Schlag)
    src, dst = positions[0], positions[1]
    capture = positions[2] if len(positions) >= 3 else None
    candidates = [
        m for m in legal_moves
        if m.src == src and m.dst == dst and (capture is None or m.capture == capture)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _format_move(move: Move) -> str:
    if move.src is None:
        base = f"setze {move.dst}"
    else:
        base = f"{move.src} -> {move.dst}"
    if move.capture:
        base += f" x{move.capture}"
    return base


# ---------------------------------------------------------------------------
# VoiceBridge — Adapter (uebernimmt Rolle von Betreuer main.py)
# ---------------------------------------------------------------------------

class VoiceBridge:
    """Sprachsteuerung fuer Muehle-Zuege.

    - Thread 1: AudioProcess.recorder_transcription_thread()  (stt.py)
    - Thread 2: CommandProcess.process_sentence()             (command_process.py)
    - listen_for_move() blockiert auf match_queue und mappt Positionen auf Zuege.
    """

    def __init__(self) -> None:
        text_q: Queue[str] = Queue(maxsize=1)
        self._match_queue: Queue[list[str]] = Queue(maxsize=1)

        # -- Betreuer-Vorlage main.py Zeilen 8-16 --
        self._audio = AudioProcess(text_q)
        commands = MillCommands()
        self._processor = CommandProcess(text_q, self._match_queue, commands)

        # -- Betreuer-Vorlage main.py Zeilen 18-24 (ohne setupAudioDevice) --
        threading.Thread(
            target=self._audio.recorder_transcription_thread,
            daemon=True,
            name="voice-stt",
        ).start()
        threading.Thread(
            target=self._processor.process_sentence,
            daemon=True,
            name="voice-cmd",
        ).start()

    def listen_for_move(self, legal_moves: Sequence[Move]) -> Move:
        """Blockiert bis ein gueltiger Zug gesprochen wurde.

        Primaer: Positionen aus CommandProcess (z.B. ["A1", "B2"]).
        Fallback: Zugnummer aus der angezeigten Liste (z.B. "drei").
        """
        print("\nLegale Zuege:")
        for idx, move in enumerate(legal_moves, start=1):
            print(f"  [{idx:02d}] {_format_move(move)}")
        print("Bitte Zug sprechen (z.B. 'A1 nach B2' oder Zugnummer 'drei')...")

        while True:
            positions = self._match_queue.get()

            move = _match_positions_to_move(positions, legal_moves)
            if move is not None:
                print(f"Erkannt (Position): {positions} -> {_format_move(move)}")
                return move

            # Fallback: vielleicht wurde eine Zugnummer gesprochen, die nicht
            # als Brett-Position gematcht wurde. Rohtext ist nicht mehr verfuegbar,
            # also versuchen wir die gematchten Positionen als Zahlen zu deuten.
            if len(positions) == 1:
                number = _parse_number(positions[0])
                if number is not None and 1 <= number <= len(legal_moves):
                    move = legal_moves[number - 1]
                    print(f"Erkannt (Nummer): {positions[0]} -> Zug {number}: {_format_move(move)}")
                    return move

            print(f"Nicht als Zug erkannt: {positions}. Bitte erneut sprechen.")

    def shutdown(self) -> None:
        self._audio.recorder.shutdown()


__all__ = ["VoiceBridge"]
