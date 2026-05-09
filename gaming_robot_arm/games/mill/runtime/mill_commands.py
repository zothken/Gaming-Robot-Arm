"""Muehle-Befehle fuer die Sprachsteuerung (basiert auf Betreuer-Vorlage command.py)."""

from gaming_robot_arm.games.mill.core.board import BOARD_LABELS


class MillCommands:
    # Aktions-Verben, die Whisper erkennen soll
    VERBS: list[str] = ["setze", "schlage", "entferne", "nach", "von", "zurück", "zurueck"]

    # Schluesselwoerter fuer Zugruecknahme (Undo)
    UNDO_KEYWORDS: list[str] = [
        "zurück", "zurueck",
        "rückgängig", "rueckgaengig",
        "zurücknehmen", "zuruecknehmen",
    ]

    # Bestaetigungsworte fuer Sprach-Bestaetigungs-Dialog
    CONFIRM_YES: list[str] = ["ja", "okay"]
    CONFIRM_NO: list[str] = ["nein", "abbrechen"]

    # Umgangssprachliche Ring-Aliase: außen=A, mitte=B, innen=C
    RING_ALIASES: dict[str, str] = {
        "außen": "A", "aussen": "A",
        "mitte": "B",
        "innen": "C",
    }

    # Buchstaben-Aliase: Whisper-Fehlerkennung ähnlich klingender Wörter → A/B/C
    LETTER_ALIASES: dict[str, str] = {
        "ah": "A", "aha": "A",
        "be": "B", "beh": "B",
        "tse": "C", "ce": "C", "zee": "C", "zeh": "C",
    }

    # Deutsche Zahlwörter für den Zugnummer-Fallback
    GERMAN_NUMBERS: dict[str, int] = {
        "eins": 1, "ein": 1, "zwei": 2, "zwo": 2, "drei": 3, "dreie": 3, "vier": 4,
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
        ring_aliases = ", ".join(f"{alias}={ring}" for alias, ring in self.RING_ALIASES.items())
        letter_aliases = ", ".join(f"{alias}={letter}" for alias, letter in self.LETTER_ALIASES.items())
        undo = ", ".join(self.UNDO_KEYWORDS)
        confirm = ", ".join(self.CONFIRM_YES + self.CONFIRM_NO)
        return (
            f"Mühle Brettspiel. Positionen: {positions}. "
            f"Ring-Aliase: {ring_aliases}. "
            f"Buchstaben-Aliase: {letter_aliases}. "
            f"Befehle: {verbs}. "
            f"Zurueck: {undo}. "
            f"Bestaetigung: {confirm}. "
            f"Zahlen: {numbers}."
        )

    def get_command_list_for_llm(self):
        return self.command_list

    def get_command_list_as_text(self):
        return ", ".join(self.command_list)
