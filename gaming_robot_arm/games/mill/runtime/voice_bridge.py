"""Sprachsteuerung fuer menschliche Muehle-Zuege.

  AudioProcess (stt.py)  -> text_queue -> CommandProcess (command_process.py) -> match_queue
  VoiceBridge.listen_for_move() liest aus match_queue und mappt auf legale Zuege.
"""

from __future__ import annotations

import re
import threading
from queue import Empty, Queue
from time import perf_counter
from typing import TYPE_CHECKING, Sequence

from .stt import AudioProcess
from .mill_commands import MillCommands
from .command_process import CommandProcess

if TYPE_CHECKING:
    from gaming_robot_arm.games.common.interfaces import Move


VOICE_MOVE_TIMEOUT_S = 60.0


def _parse_number(text: str) -> int | None:
    """Fallback: Zugnummer aus Text (Ziffer oder deutsches Zahlwort)."""
    m = re.search(r'\b(\d+)\b', text)
    if m:
        return int(m.group(1))
    lower = text.lower()
    for word, n in MillCommands.GERMAN_NUMBERS.items():
        if word in lower:
            return n
    return None


def _match_positions_to_move(
    positions: list[str], legal_moves: Sequence[Move]
) -> tuple[Move | None, str]:
    """Mappt extrahierte Positionen auf einen eindeutigen legalen Zug.

    Gibt (move, reason) zurueck.
    reason: "ok" | "illegal" | "ambiguous" | "no_positions"
    """
    if len(positions) == 0:
        return None, "no_positions"
    if len(positions) == 1:
        # Setzphase: nur Zielfeld
        candidates = [m for m in legal_moves if m.dst == positions[0] and m.src is None]
        if len(candidates) == 1:
            return candidates[0], "ok"
        return None, "ambiguous" if len(candidates) > 1 else "illegal"
    # Zugphase: src + dst (+ optionaler Schlag)
    src, dst = positions[0], positions[1]
    capture = positions[2] if len(positions) >= 3 else None
    candidates = [
        m for m in legal_moves
        if m.src == src and m.dst == dst and (capture is None or m.capture == capture)
    ]
    if len(candidates) == 1:
        return candidates[0], "ok"
    return None, "ambiguous" if len(candidates) > 1 else "illegal"


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

    def listen_for_move(
        self, legal_moves: Sequence[Move], timeout_s: float = VOICE_MOVE_TIMEOUT_S
    ) -> Move | None:
        """Blockiert bis ein gueltiger Zug gesprochen wurde oder das Zeitlimit ablaeuft.

        Primaer: Positionen aus CommandProcess (z.B. ["A1", "B2"]).
        Fallback: Zugnummer aus der angezeigten Liste (z.B. "drei").
        Gibt None zurueck wenn das Zeitlimit ablaeuft.
        """
        print("\nLegale Zuege:")
        for idx, move in enumerate(legal_moves, start=1):
            print(f"  [{idx:02d}] {_format_move(move)}")
        print("Bitte Zug sprechen (z.B. 'A1 nach B2' oder Zugnummer 'drei')...")

        deadline = perf_counter() + timeout_s

        while True:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                print("Spracherkennung: Zeitlimit abgelaufen.")
                return None

            try:
                positions = self._match_queue.get(timeout=remaining)
            except Empty:
                print("Spracherkennung: Zeitlimit abgelaufen.")
                return None

            move, reason = _match_positions_to_move(positions, legal_moves)
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

            pos_str = " -> ".join(positions)
            if reason == "illegal":
                print(f"Illegaler Zug: '{pos_str}' ist kein gueltiger Zug. Bitte erneut sprechen.")
            elif reason == "ambiguous":
                print(f"Mehrdeutig: '{pos_str}' – mehrere Zuege moeglich. Schlagfeld angeben (z.B. 'A1 B2 C3').")
            else:
                print(f"Nicht erkannt: {positions}. Bitte erneut sprechen.")

    def shutdown(self) -> None:
        self._audio.recorder.shutdown()


__all__ = ["VoiceBridge"]
