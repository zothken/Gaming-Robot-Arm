"""QProcess-Starthelfer fuer den Desktop-Launcher."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment


def start_qprocess(process: QProcess, *, cmd: list[str], project_root: Path) -> str | None:
    env = QProcessEnvironment.systemEnvironment()
    env.insert("PYTHONUNBUFFERED", "1")
    env.insert("PYTHONUTF8", "1")
    process.setProcessEnvironment(env)
    process.setWorkingDirectory(str(project_root))
    process.setProgram(cmd[0])
    process.setArguments(cmd[1:])
    process.start()
    if process.waitForStarted(1500):
        return None
    return process.errorString() or "Unbekannter Startfehler"


__all__ = ["start_qprocess"]
