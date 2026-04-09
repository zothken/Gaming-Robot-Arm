"""Muehle-Befehle fuer die Sprachsteuerung (basiert auf Betreuer-Vorlage command.py)."""

from gaming_robot_arm.games.mill.core.board import BOARD_LABELS


class MillCommands:
    def __init__(self):
        self.cmd = {pos: pos for pos in BOARD_LABELS}
        self.command_list = self._generate_command_list()

    def _generate_command_list(self):
        commands = list(self.cmd.keys())

        commands.sort()
        return commands

    def get_command_list_for_llm(self):
        return self.command_list

    def get_command_list_as_text(self):
        return ", ".join(self.command_list)
