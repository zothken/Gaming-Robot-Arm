"""Interaktiver Test der Sprachsteuerung ohne laufendes Spiel.

Verwendung:
  python examples/test_voice_commands.py              # nutzt large-v3 (wie im Spiel)
  python examples/test_voice_commands.py --model tiny # schneller Download, geringere Genauigkeit
"""

import argparse
import logging
import os
import threading
import warnings
from queue import Queue

# Verbose-Output von Huggingface/Whisper/httpcore unterdrücken
logging.basicConfig(level=logging.CRITICAL)
for logger_name in ("httpcore", "httpx", "huggingface_hub", "faster_whisper",
                    "RealtimeSTT", "urllib3", "filelock"):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

from gaming_robot_arm.games.mill.runtime.stt import AudioProcess
from gaming_robot_arm.games.mill.runtime.mill_commands import MillCommands
from gaming_robot_arm.games.mill.runtime.command_process import CommandProcess


def main():
    parser = argparse.ArgumentParser(description="Sprachbefehl-Test")
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper-Modell (Standard: large-v3 wie im Spiel, 'tiny' fuer schnellen Test)",
    )
    args = parser.parse_args()

    print(f"=== Sprachbefehl-Test (Modell: {args.model}) ===")
    print("Beenden mit Ctrl+C.\n")
    print("Lade Modell...", flush=True)

    text_q: Queue[str] = Queue(maxsize=1)
    match_q: Queue[list[str]] = Queue(maxsize=1)

    audio = AudioProcess(text_q, model=args.model)
    commands = MillCommands()
    processor = CommandProcess(text_q, match_q, commands)

    threading.Thread(
        target=audio.recorder_transcription_thread,
        daemon=True,
        name="voice-stt",
    ).start()
    threading.Thread(
        target=processor.process_sentence,
        daemon=True,
        name="voice-cmd",
    ).start()

    print("Bereit. Bitte sprechen.\n")

    try:
        while True:
            matches = match_q.get()
            print(f">>> Erkannte Befehle: {matches}")
    except KeyboardInterrupt:
        print("\nTest beendet.")
        os._exit(0)


if __name__ == "__main__":
    main()
