import unittest
from pathlib import Path

from gaming_robot_arm.ui.launcher.command_builder import build_command
from gaming_robot_arm.ui.launcher.settings import LauncherSettings


class LauncherVisionTriggerTest(unittest.TestCase):
    def test_settings_default_to_auto_trigger(self) -> None:
        settings = LauncherSettings.from_payload({"mill_human_input": "vision"})
        self.assertEqual(settings.mill_vision_trigger, "auto")

    def test_command_builder_emits_auto_trigger_by_default(self) -> None:
        cmd = build_command(
            LauncherSettings(mill_human_input="vision"),
            python_executable="python",
            entry_script=Path("/tmp/main.py"),
        )
        idx = cmd.index("--vision-trigger")
        self.assertEqual(cmd[idx + 1], "auto")

    def test_command_builder_emits_manual_trigger_when_requested(self) -> None:
        cmd = build_command(
            LauncherSettings(mill_human_input="vision", mill_vision_trigger="manual"),
            python_executable="python",
            entry_script=Path("/tmp/main.py"),
        )
        idx = cmd.index("--vision-trigger")
        self.assertEqual(cmd[idx + 1], "manual")


if __name__ == "__main__":
    unittest.main()
