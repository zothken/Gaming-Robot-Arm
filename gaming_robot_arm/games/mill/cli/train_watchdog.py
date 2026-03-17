"""Watchdog fuer Muehle-Datenerzeugung und/oder neuronales Training.

Einzelkommando-Modus (Altmodus; nur Training):
  gra-mill-watchdog --interval-seconds 900 -- \
    gra-mill-train --data data/mill_teacher_500.jsonl ...

Zwei-Phasen-Modus (Generierung -> Training):
  gra-mill-watchdog \
    --interval-seconds 900 \
    --data-path data/mill_teacher_500.jsonl \
    --checkpoint-dir models/checkpoints/mill_torch_500e \
    --generate-cmd "gra-mill-generate-teacher --games 500 --teacher-depth 3 --output data/mill_teacher_500.jsonl" \
    --train-cmd "gra-mill-train --data data/mill_teacher_500.jsonl --epochs 500 --batch-size 128 --output models/mill_torch_500e.pt --checkpoint-dir models/checkpoints/mill_torch_500e --checkpoint-every 5"
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Callable


_GENERATE_PROGRESS_RE = re.compile(
    r"Erzeugte Partie (?P<games_done>\d+)/(?P<games_total>\d+) \| "
    r"samples=(?P<samples>\d+) remis=(?P<draws>\d+) "
    r"W_siege=(?P<white_wins>\d+) B_siege=(?P<black_wins>\d+)"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watchdog-Wrapper fuer Muehle-Datenerzeugung/Training.")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=900,
        help="Heartbeat-Intervall waehrend eine Phase laeuft.",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=3,
        help="Maximale Neustartversuche nach Exit-Code != 0 (Einzelkommando-Modus und Standard-Trainingsmodus).",
    )
    parser.add_argument(
        "--generate-max-restarts",
        type=int,
        default=0,
        help="Maximale Neustartversuche fuer Datengenerierung im Zwei-Phasen-Modus.",
    )
    parser.add_argument(
        "--train-max-restarts",
        type=int,
        default=None,
        help="Maximale Neustartversuche fuer Training im Zwei-Phasen-Modus; Standard ist --max-restarts.",
    )
    parser.add_argument(
        "--restart-delay-seconds",
        type=int,
        default=20,
        help="Verzoegerung vor Neustart nach Fehler.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optionales Checkpoint-Verzeichnis fuer Trainings-Heartbeat (.pt-Aktualisierungsalter).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optionaler Datendatei-Pfad fuer Generierungs-Heartbeat (Groesse/Alter).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Verzeichnis fuer Watchdog- und Trainingslogs.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("."),
        help="Arbeitsverzeichnis fuer Phasenkommandos.",
    )
    parser.add_argument(
        "--generate-cmd",
        type=str,
        default=None,
        help="Vollstaendige Kommandozeile fuer Datengenerierungsphase (Zwei-Phasen-Modus).",
    )
    parser.add_argument(
        "--train-cmd",
        type=str,
        default=None,
        help="Vollstaendige Kommandozeile fuer Trainingsphase (Zwei-Phasen-Modus).",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Einzelkommando-Modus: auszufuehrendes Kommando, meist nach '--'.",
    )
    return parser


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def _latest_checkpoint(path: Path | None) -> tuple[Path | None, float | None]:
    if path is None or not path.exists() or not path.is_dir():
        return None, None
    pt_files = sorted(path.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pt_files:
        return None, None
    latest = pt_files[0]
    age_seconds = time.time() - latest.stat().st_mtime
    return latest, age_seconds


def _data_file_stats(path: Path | None) -> tuple[Path | None, int | None, float | None]:
    if path is None or not path.exists() or not path.is_file():
        return None, None, None
    stat = path.stat()
    age_seconds = time.time() - stat.st_mtime
    return path, int(stat.st_size), age_seconds


def _phase_progress(stop_state: dict[str, object], phase_name: str) -> dict[str, int | None] | None:
    progress_by_phase = stop_state.get("phase_progress")
    if not isinstance(progress_by_phase, dict):
        return None
    phase_progress = progress_by_phase.get(phase_name)
    if not isinstance(phase_progress, dict):
        return None
    return phase_progress


def _update_generation_progress(stop_state: dict[str, object], phase_name: str, line: str) -> None:
    match = _GENERATE_PROGRESS_RE.search(line)
    if match is None:
        return

    progress = _phase_progress(stop_state, phase_name)
    if progress is None:
        return

    for key, value in match.groupdict().items():
        progress[key] = int(value)


def _stream_output(
    proc: subprocess.Popen[str],
    out_fp,
    prefix: str,
    *,
    phase_name: str,
    stop_state: dict[str, object],
) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        _update_generation_progress(stop_state, phase_name, line)
        text = f"{prefix}{line}"
        out_fp.write(text)
        out_fp.flush()
        sys.stdout.write(text)
        sys.stdout.flush()


def _normalize_command(raw: list[str]) -> list[str]:
    if raw and raw[0] == "--":
        raw = raw[1:]
    if not raw:
        raise ValueError("Kein Kommando angegeben. Bitte nach '--' uebergeben.")
    return raw


def _split_command(raw: str, arg_name: str) -> list[str]:
    parts = shlex.split(raw, posix=(os.name != "nt"))
    if not parts:
        raise ValueError(f"{arg_name} darf nicht leer sein.")
    return parts


def _emit(line: str, *fps) -> None:
    for fp in fps:
        fp.write(line)
        fp.flush()
    sys.stdout.write(line)
    sys.stdout.flush()


def _monitor_checkpoint(checkpoint_dir: Path | None) -> str:
    latest_pt, age_seconds = _latest_checkpoint(checkpoint_dir)
    if latest_pt is None:
        return "latest_pt=keins"
    return f"latest_pt={latest_pt} latest_pt_age_s={int(age_seconds or 0)}"


def _monitor_data_file(data_path: Path | None, progress: dict[str, int | None] | None = None) -> str:
    path, size_bytes, age_seconds = _data_file_stats(data_path)
    if path is None:
        base = "data_file=keine"
    else:
        base = f"data_file={path} size_bytes={size_bytes} data_age_s={int(age_seconds or 0)}"

    if progress is None:
        return base

    games_done = progress.get("games_done")
    games_total = progress.get("games_total")
    samples = progress.get("samples")
    draws = progress.get("draws")
    white_wins = progress.get("white_wins")
    black_wins = progress.get("black_wins")
    if games_done is None:
        return base

    suffix = f" games_played={games_done}"
    if games_total is not None:
        suffix += f"/{games_total}"
    if samples is not None:
        suffix += f" samples={samples}"
    if draws is not None:
        suffix += f" draws={draws}"
    if white_wins is not None and black_wins is not None:
        suffix += f" wins_W={white_wins} wins_B={black_wins}"
    return base + suffix


def _infer_single_phase(command: list[str]) -> str:
    """Leitet fuer Einzelkommando-Modus heuristisch den Phasentyp ab."""

    joined = " ".join(command).lower()
    generate_markers = (
        "generate_teacher_data",
        "generate_selfplay_data",
        "gra-mill-generate-teacher",
        "gra-mill-generate-selfplay",
    )
    if any(marker in joined for marker in generate_markers):
        return "generate"
    return "train"


def _run_phase(
    *,
    phase_name: str,
    command: list[str],
    interval_seconds: int,
    restart_delay_seconds: int,
    max_restarts: int,
    workdir: Path,
    monitor_line: Callable[[], str],
    run_fp,
    status_fp,
    stop_state: dict[str, object],
) -> int:
    restart_count = 0
    while not bool(stop_state["stop_requested"]):
        start_ts = time.time()
        run_index = restart_count + 1
        progress_by_phase = stop_state.setdefault("phase_progress", {})
        if isinstance(progress_by_phase, dict):
            progress_by_phase[phase_name] = {
                "games_done": None,
                "games_total": None,
                "samples": None,
                "draws": None,
                "white_wins": None,
                "black_wins": None,
            }
        resolved_exe = shutil.which(command[0]) if command else None
        header = (
            f"[{_now_iso()}] START phase={phase_name} run={run_index} restart_count={restart_count} "
            f"cmd={shlex.join(command)} exe={resolved_exe or 'unaufgeloest'}\n"
        )
        _emit(header, run_fp, status_fp)

        proc = subprocess.Popen(
            command,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        stop_state["active_proc"] = proc
        stream_thread = threading.Thread(
            target=_stream_output,
            args=(proc, run_fp, ""),
            kwargs={"phase_name": phase_name, "stop_state": stop_state},
            daemon=True,
        )
        stream_thread.start()

        next_heartbeat_at = time.monotonic() + interval_seconds
        while not bool(stop_state["stop_requested"]):
            return_code = proc.poll()
            if return_code is not None:
                break

            now = time.monotonic()
            if now >= next_heartbeat_at:
                elapsed = int(time.time() - start_ts)
                heartbeat = (
                    f"[{_now_iso()}] HEARTBEAT phase={phase_name} run={run_index} elapsed_s={elapsed} "
                    f"{monitor_line()}\n"
                )
                _emit(heartbeat, status_fp)
                next_heartbeat_at = now + interval_seconds
                continue

            sleep_seconds = min(1.0, max(0.0, next_heartbeat_at - now))
            time.sleep(sleep_seconds)

        if bool(stop_state["stop_requested"]) and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

        return_code = proc.wait()
        stream_thread.join(timeout=5)
        stop_state["active_proc"] = None

        footer = f"[{_now_iso()}] EXIT phase={phase_name} run={run_index} code={return_code}\n"
        _emit(footer, run_fp, status_fp)

        if bool(stop_state["stop_requested"]):
            return 130

        if return_code == 0:
            _emit(f"[{_now_iso()}] {phase_name} erfolgreich abgeschlossen.\n", status_fp)
            return 0

        if restart_count >= max_restarts:
            _emit(
                f"[{_now_iso()}] Maximale Neustarts fuer {phase_name} erreicht ({max_restarts}). Breche ab.\n",
                status_fp,
            )
            return return_code if return_code != 0 else 1

        restart_count += 1
        _emit(
            f"[{_now_iso()}] {phase_name} fehlgeschlagen (code={return_code}). "
            f"Neustart in {restart_delay_seconds}s ({restart_count}/{max_restarts}).\n",
            status_fp,
        )
        time.sleep(restart_delay_seconds)

    return 130


def main() -> int:
    args = build_parser().parse_args()

    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds muss > 0 sein")
    if args.max_restarts < 0:
        raise ValueError("--max-restarts muss >= 0 sein")
    if args.generate_max_restarts < 0:
        raise ValueError("--generate-max-restarts muss >= 0 sein")
    if args.train_max_restarts is not None and args.train_max_restarts < 0:
        raise ValueError("--train-max-restarts muss >= 0 sein")
    if args.restart_delay_seconds < 0:
        raise ValueError("--restart-delay-seconds muss >= 0 sein")

    two_phase_mode = args.generate_cmd is not None or args.train_cmd is not None
    generate_cmd: list[str] = []
    train_cmd: list[str]
    if two_phase_mode:
        if args.generate_cmd is None or args.train_cmd is None:
            raise ValueError("Zwei-Phasen-Modus erfordert sowohl --generate-cmd als auch --train-cmd.")
        if args.command:
            raise ValueError("Bei --generate-cmd/--train-cmd kein positionsbasiertes Kommando uebergeben.")
        generate_cmd = _split_command(args.generate_cmd, "--generate-cmd")
        train_cmd = _split_command(args.train_cmd, "--train-cmd")
    else:
        train_cmd = _normalize_command(args.command)

    train_max_restarts = args.train_max_restarts if args.train_max_restarts is not None else args.max_restarts

    args.log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log = args.log_dir / f"mill_watchdog_{run_id}.log"
    state_log = args.log_dir / f"mill_watchdog_{run_id}.status.log"

    print(f"[{_now_iso()}] Watchdog startet (two_phase={two_phase_mode})")
    if two_phase_mode:
        print(f"[{_now_iso()}] Generierungs-Kommando: {shlex.join(generate_cmd)}")
        print(f"[{_now_iso()}] Trainings-Kommando: {shlex.join(train_cmd)}")
    else:
        print(f"[{_now_iso()}] Kommando: {shlex.join(train_cmd)}")
    print(f"[{_now_iso()}] Arbeitsverzeichnis: {args.workdir.resolve()}")
    print(f"[{_now_iso()}] Logs: {run_log} / {state_log}")

    stop_state: dict[str, object] = {
        "stop_requested": False,
        "active_proc": None,
    }

    def _handle_signal(signum, _frame) -> None:
        stop_state["stop_requested"] = True
        print(f"[{_now_iso()}] Signal {signum} empfangen, stoppe ...")
        active_proc = stop_state["active_proc"]
        if isinstance(active_proc, subprocess.Popen) and active_proc.poll() is None:
            active_proc.terminate()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    with run_log.open("a", encoding="utf-8") as run_fp, state_log.open("a", encoding="utf-8") as status_fp:
        if two_phase_mode:
            gen_rc = _run_phase(
                phase_name="generate",
                command=generate_cmd,
                interval_seconds=args.interval_seconds,
                restart_delay_seconds=args.restart_delay_seconds,
                max_restarts=args.generate_max_restarts,
                workdir=args.workdir,
                monitor_line=lambda: _monitor_data_file(args.data_path, _phase_progress(stop_state, "generate")),
                run_fp=run_fp,
                status_fp=status_fp,
                stop_state=stop_state,
            )
            if gen_rc != 0:
                return gen_rc

            train_rc = _run_phase(
                phase_name="train",
                command=train_cmd,
                interval_seconds=args.interval_seconds,
                restart_delay_seconds=args.restart_delay_seconds,
                max_restarts=train_max_restarts,
                workdir=args.workdir,
                monitor_line=lambda: _monitor_checkpoint(args.checkpoint_dir),
                run_fp=run_fp,
                status_fp=status_fp,
                stop_state=stop_state,
            )
            return train_rc

        single_phase = _infer_single_phase(train_cmd)
        if single_phase == "generate":
            monitor = lambda: _monitor_data_file(args.data_path, _phase_progress(stop_state, single_phase))
        else:
            monitor = lambda: _monitor_checkpoint(args.checkpoint_dir)
        train_rc = _run_phase(
            phase_name=single_phase,
            command=train_cmd,
            interval_seconds=args.interval_seconds,
            restart_delay_seconds=args.restart_delay_seconds,
            max_restarts=train_max_restarts,
            workdir=args.workdir,
            monitor_line=monitor,
            run_fp=run_fp,
            status_fp=status_fp,
            stop_state=stop_state,
        )
        return train_rc


if __name__ == "__main__":
    raise SystemExit(main())
