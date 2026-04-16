"""Muehle-Befehle fuer die Sprachsteuerung (basiert auf Betreuer-Vorlage command.py)."""

from gaming_robot_arm.games.mill.core.board import BOARD_LABELS


class MillCommands:
    # Aktions-Verben, die Whisper erkennen soll
    VERBS: list[str] = ["setze", "schlage", "entferne", "nach", "von"]

    # Deutsche Zahlwörter für den Zugnummer-Fallback
    GERMAN_NUMBERS: dict[str, int] = {
        "eins": 1, "ein": 1, "zwei": 2, "zwo": 2, "drei": 3, "vier": 4,
        "fünf": 5, "fuenf": 5, "sechs": 6, "sieben": 7, "acht": 8,
        "neun": 9, "zehn": 10, "elf": 11, "zwölf": 12, "zwoelf": 12,
        "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15, "sechzehn": 16,
        "siebzehn": 17, "achtzehn": 18, "neunzehn": 19, "zwanzig": 20,
    }

    def __init__(self):
        self.cmd = {pos: pos for pos in BOARD_LABELS}
        self.command_list = sorted(self.cmd.keys())

    def build_initial_prompt(self) -> str:
        """Generiert den Whisper-Priming-Prompt aus allen bekannten Vokabeln."""
        positions = ", ".join(sorted(self.cmd.keys()))
        verbs = ", ".join(self.VERBS)
        numbers = ", ".join(self.GERMAN_NUMBERS.keys())
        return (
            f"Mühle Brettspiel. Positionen: {positions}. "
            f"Befehle: {verbs}. "
            f"Zahlen: {numbers}."
        )

    def get_command_list_for_llm(self):
        return self.command_list

    def get_command_list_as_text(self):
        return ", ".join(self.command_list)
