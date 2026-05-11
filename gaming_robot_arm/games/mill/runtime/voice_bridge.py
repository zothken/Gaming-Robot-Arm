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
from .signals import UndoSignal

if TYPE_CHECKING:
    from gaming_robot_arm.games.common.interfaces import Move


VOICE_MOVE_TIMEOUT_S = 60.0
VOICE_CONFIRM_TIMEOUT_S = 10.0


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


def _is_placement_phase(legal_moves: Sequence[Move]) -> bool:
    return all(m.src is None for m in legal_moves)


def _resolve_buffer(
    buffer: list[str], legal_moves: Sequence[Move], placement: bool
) -> tuple[Move | None, str]:
    """Wertet einen wachsenden Positions-Buffer gegen legale Zuege aus.

    Status:
      ok | need_dst | need_capture | illegal_src | illegal_dst |
      illegal_capture | ambiguous | empty
    """
    if not buffer:
        return None, "empty"

    if placement:
        dst = buffer[0]
        cands = [m for m in legal_moves if m.src is None and m.dst == dst]
        if not cands:
            return None, "illegal_dst"
        needs_capture = all(m.capture is not None for m in cands)
        if not needs_capture and len(cands) == 1:
            return cands[0], "ok"
        # Setzen + Muehle: zweite Position = Schlagfeld
        if len(buffer) < 2:
            return None, "need_capture"
        cap = buffer[1]
        final = [m for m in cands if m.capture == cap]
        if len(final) == 1:
            return final[0], "ok"
        return None, "ambiguous" if len(final) > 1 else "illegal_capture"

    # Zug-/Flugphase
    src = buffer[0]
    src_cands = [m for m in legal_moves if m.src == src]
    if not src_cands:
        return None, "illegal_src"
    if len(buffer) == 1:
        return None, "need_dst"

    dst = buffer[1]
    pair_cands = [m for m in src_cands if m.dst == dst]
    if not pair_cands:
        return None, "illegal_dst"

    needs_capture = all(m.capture is not None for m in pair_cands)
    if not needs_capture and len(pair_cands) == 1:
        return pair_cands[0], "ok"

    if len(buffer) < 3:
        return None, "need_capture"
    cap = buffer[2]
    final = [m for m in pair_cands if m.capture == cap]
    if len(final) == 1:
        return final[0], "ok"
    return None, "ambiguous" if len(final) > 1 else "illegal_capture"


def _format_move(move: Move) -> str:
    if move.src is None:
        base = f"setze {move.dst}"
    else:
        base = f"{move.src} -> {move.dst}"
    if move.capture:
        base += f" x{move.capture}"
    return base


# ---------------------------------------------------------------------------
# VoiceBridge — Adapter (uebernimmt Rolle von main.py)
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
    ) -> "Move | UndoSignal | None":
        """Blockiert bis ein gueltiger Zug gesprochen wurde oder das Zeitlimit ablaeuft.

        Primaer: Positionen aus CommandProcess (z.B. ["A1", "B2"]).
        Fallback: Zugnummer aus der angezeigten Liste (z.B. "drei").
        Sonderfall: 'zurueck' liefert UndoSignal nach Bestaetigung mit 'ja'.
        Gibt None zurueck wenn das Zeitlimit ablaeuft.
        """
        placement = _is_placement_phase(legal_moves)

        print("\nLegale Zuege:")
        for idx, move in enumerate(legal_moves, start=1):
            print(f"  [{idx:02d}] {_format_move(move)}")
        if placement:
            print("Bitte Setzfeld sprechen (z.B. 'B3') oder Zugnummer 'drei'; 'zurueck' fuer Ruecknahme...")
        else:
            print("Bitte Zug sprechen (Zugphase: erst Startfeld, dann Zielfeld) oder Zugnummer 'drei'; 'zurueck' fuer Ruecknahme...")

        deadline = perf_counter() + timeout_s
        buffer: list[str] = []

        while True:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                print("Spracherkennung: Zeitlimit abgelaufen.")
                return None

            try:
                positions = self._match_queue.get(timeout=None if remaining == float("inf") else remaining)
            except Empty:
                print("Spracherkennung: Zeitlimit abgelaufen.")
                return None

            if positions == ["__UNDO__"]:
                print("Ruecknahme erkannt. Bitte 'ja' oder 'nein' sagen zur Bestaetigung...")
                if self._await_confirmation(timeout_s=VOICE_CONFIRM_TIMEOUT_S):
                    print("Ruecknahme bestaetigt.")
                    return UndoSignal()
                print("Ruecknahme abgebrochen. Bitte Zug sprechen...")
                buffer.clear()
                continue

            # Bestaetigungs-Marker ohne vorherige Ruecknahme-Anforderung -> ignorieren
            if positions in (["__YES__"], ["__NO__"]):
                continue

            # Zugnummer-Fallback: nur bei leerem Buffer (erste Utterance).
            if not buffer and len(positions) == 1:
                number = _parse_number(positions[0])
                if number is not None and 1 <= number <= len(legal_moves):
                    chosen = legal_moves[number - 1]
                    print(f"Erkannt (Nummer): {positions[0]} -> Zug {number}: {_format_move(chosen)}")
                    return chosen

            buffer.extend(positions)
            move, status = _resolve_buffer(buffer, legal_moves, placement)

            if status == "ok":
                print(f"Erkannt: {buffer} -> {_format_move(move)}")
                return move
            if status == "need_dst":
                print(f"Startfeld {buffer[0]} erkannt. Bitte Zielfeld nennen...")
                continue
            if status == "need_capture":
                if placement:
                    print(
                        f"Setzen auf {buffer[0]} schliesst Muehle. "
                        f"Bitte gegnerischen Stein zum Entfernen nennen..."
                    )
                else:
                    print(
                        f"Zug {buffer[0]} -> {buffer[1]} schliesst Muehle. "
                        f"Bitte gegnerischen Stein zum Entfernen nennen..."
                    )
                continue
            if status == "illegal_src":
                print(f"Illegales Startfeld: '{buffer[0]}'. Bitte erneut sprechen.")
                buffer.clear()
                continue
            if status == "illegal_dst":
                if placement:
                    print(f"Illegales Setzfeld: '{buffer[0]}'. Bitte erneut sprechen.")
                    buffer.clear()
                else:
                    bad = buffer[1] if len(buffer) >= 2 else buffer[0]
                    print(
                        f"Illegales Zielfeld: '{bad}'. Startfeld {buffer[0]} bleibt, "
                        f"bitte Zielfeld nennen..."
                    )
                    del buffer[1:]
                continue
            if status == "illegal_capture":
                cap_idx = 1 if placement else 2
                print(
                    f"'{buffer[cap_idx]}' ist kein entfernbarer Stein. "
                    f"Bitte erneut nennen..."
                )
                del buffer[cap_idx:]
                continue
            if status == "ambiguous":
                print(f"Mehrdeutig: {buffer}. Bitte Schlagfeld zusaetzlich nennen.")
                continue

    def _await_confirmation(self, *, timeout_s: float) -> bool:
        """Wartet auf gesprochene Bestaetigung. True bei 'ja', False bei 'nein' oder Timeout."""
        deadline = perf_counter() + timeout_s
        while True:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                print("Bestaetigung: Zeitlimit abgelaufen.")
                return False
            try:
                positions = self._match_queue.get(timeout=remaining)
            except Empty:
                print("Bestaetigung: Zeitlimit abgelaufen.")
                return False
            if positions == ["__YES__"]:
                return True
            if positions == ["__NO__"]:
                return False
            # Andere Eingaben (Position oder erneutes Undo) ignorieren und weiterwarten

    def shutdown(self) -> None:
        self._audio.recorder.shutdown()


__all__ = ["VoiceBridge"]
