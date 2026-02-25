"""Automatisiert iterative Schleifen aus Selbstspiel -> Training -> Bewertung -> Promotion.

Dieses Skript orchestriert:
1) Selbstspiel-Datenerzeugung (neuronales Modell gegen neuronales Modell).
2) Watchdog-ueberwachtes Training vom aktuellen Champion aus.
3) Kandidat-vs-Champion-Benchmark.
4) Champion-Promotion, wenn der Kandidat besser abschneidet.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuehrt iterative Muehle-Selfplay-Trainingsschleifen aus.")
    parser.add_argument("--iterations", type=int, required=True, help="Anzahl auszufuehrender Iterationen.")
    parser.add_argument(
        "--initial-champion",
        type=Path,
        required=True,
        help="Initialer Champion-Checkpoint (.pt) fuer den Start von Iteration 1.",
    )
    parser.add_argument(
        "--champion-output",
        type=Path,
        default=Path("models/champion/mill_champion.pt"),
        help="Pfad, unter dem der befoerderte Champion gespeichert wird.",
    )
    parser.add_argument("--start-iteration", type=int, default=1, help="Startindex der Iteration.")

    parser.add_argument("--games-per-iter", type=int, default=300, help="Selbstspiel-Partien pro Iteration.")
    parser.add_argument(
        "--ab-games-per-iter",
        type=int,
        default=0,
        help="Zusaetzliche AlphaBeta-Teacher-Partien pro Iteration (0 deaktiviert).",
    )
    parser.add_argument(
        "--ab-teacher-depth",
        type=int,
        default=3,
        help="AlphaBeta-Tiefe fuer zusaetzliche Teacher-Partien.",
    )
    parser.add_argument("--temperature", type=float, default=0.35, help="Selbstspiel-Temperatur fuer beide Seiten.")
    parser.add_argument("--max-plies", type=int, default=400, help="Maximale Halbzuege pro Selbstspiel-Partie.")
    parser.add_argument(
        "--enable-flying",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aktiviert Flying-Regel fuer Selbstspiel-Erzeugung und Bewertung.",
    )
    parser.add_argument(
        "--enable-threefold-repetition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Aktiviert Remis durch Dreifachwiederholung fuer Selbstspiel-Erzeugung und Bewertung.",
    )
    parser.add_argument(
        "--enable-no-capture-draw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Aktiviert No-Capture-Remis fuer Selbstspiel-Erzeugung und Bewertung.",
    )
    parser.add_argument(
        "--no-capture-draw-plies",
        type=int,
        default=200,
        help="Halbzuglimit ohne Schlag, wenn No-Capture-Remis aktiv ist.",
    )

    parser.add_argument("--train-epochs", type=int, default=150, help="Maximale Trainings-Epochen pro Iteration.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training-Batch-Groesse.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Training-Lernrate.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Training-Weight-Decay.")
    parser.add_argument("--validation-split", type=float, default=0.15, help="Training-Validierungsanteil.")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Checkpoint-Intervall in Epochen.")
    parser.add_argument("--early-stopping-patience", type=int, default=15, help="Geduld fuer fruehes Stoppen.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4, help="Minimum-Delta fuer fruehes Stoppen.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch-Geraet.")

    parser.add_argument("--seed-base", type=int, default=20260220, help="Basis-Seed zur Ableitung von Seeds pro Iteration.")

    parser.add_argument("--eval-games", type=int, default=50, help="Kandidat-vs-Champion-Partien pro Iteration.")
    parser.add_argument(
        "--ab-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuehrt pro Iteration zusaetzliche Champion-vs-AlphaBeta-Benchmarks aus.",
    )
    parser.add_argument(
        "--ab-eval-games",
        type=int,
        default=20,
        help="Partien pro AlphaBeta-Benchmarktiefe (wenn --ab-eval aktiv ist).",
    )
    parser.add_argument(
        "--ab-depths",
        type=str,
        default="2,3",
        help="Kommagetrennte AlphaBeta-Tiefen zur Bewertung, z.B. '2,3,4'.",
    )
    parser.add_argument(
        "--draw-score",
        type=float,
        default=0.75,
        help="Punktgutschrift fuer ein Remis in der Promotionsbewertung (muss in [0.0, 1.0) liegen).",
    )
    parser.add_argument(
        "--promote-threshold",
        type=float,
        default=0.55,
        help=(
            "Mindestscore des Kandidaten fuer Promotion. "
            "Score ist (siege + draw_score * remis) / partien."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/selfplay"),
        help="Verzeichnis fuer Selbstspiel-Datensaetze pro Iteration.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/selfplay"),
        help="Verzeichnis fuer Modellausgaben pro Iteration.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("models/checkpoints/selfplay"),
        help="Wurzelverzeichnis fuer Checkpoint-Ordner pro Iteration.",
    )
    parser.add_argument(
        "--watchdog-log-root",
        type=Path,
        default=Path("logs/selfplay"),
        help="Wurzelverzeichnis fuer Watchdog-Logs pro Iteration.",
    )
    parser.add_argument(
        "--eval-log-dir",
        type=Path,
        default=Path("logs/selfplay_eval"),
        help="Verzeichnis fuer Benchmark-Ausgabelogs pro Iteration.",
    )
    parser.add_argument(
        "--replay-data",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optionales statisches Replay-Dataset (JSONL), das pro Iteration anteilig "
            "in den Trainingsdatensatz gemischt wird. Mehrfach angebbar."
        ),
    )
    parser.add_argument(
        "--replay-samples-per-file",
        type=int,
        default=15000,
        help="Maximale Anzahl zufaellig gezogener Samples pro --replay-data-Datei und Iteration.",
    )
    parser.add_argument(
        "--shuffle-merged-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mischt den zusammengefuehrten Iterationsdatensatz vor dem Training.",
    )

    parser.add_argument("--watchdog-interval-seconds", type=int, default=900, help="Watchdog-Heartbeat-Intervall.")
    parser.add_argument("--watchdog-generate-max-restarts", type=int, default=0, help="Maximale Watchdog-Restarts fuer Generierung.")
    parser.add_argument("--watchdog-train-max-restarts", type=int, default=3, help="Maximale Watchdog-Restarts fuer Training.")
    parser.add_argument("--watchdog-restart-delay-seconds", type=int, default=20, help="Watchdog-Restart-Verzoegerung.")

    parser.add_argument(
        "--python-exe",
        type=Path,
        default=None,
        help="Zu nutzendes Python-Executable; Standard ist der aktuelle Interpreter.",
    )
    return parser


def _cmd_to_string(parts: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return " ".join(shlex_quote(part) for part in parts)


def _bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def shlex_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:\\-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _run_command(cmd: list[str], *, cwd: Path, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print(f"$ {_cmd_to_string(cmd)}")
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=capture,
        check=False,
    )


def _parse_benchmark_summary(stdout: str, *, label_a: str, label_b: str) -> tuple[int, int, int]:
    a_match = re.search(
        rf"^\s*{re.escape(label_a)}\s+siege:\s*(\d+)\s*$",
        stdout,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    b_match = re.search(
        rf"^\s*{re.escape(label_b)}\s+siege:\s*(\d+)\s*$",
        stdout,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    draws_match = re.search(r"^\s*Remis:\s*(\d+)\s*$", stdout, flags=re.MULTILINE | re.IGNORECASE)
    if a_match is None or b_match is None or draws_match is None:
        raise RuntimeError("Konnte Benchmark-Zusammenfassung nicht parsen. Bitte Ausgabeformat pruefen.")
    return int(a_match.group(1)), int(b_match.group(1)), int(draws_match.group(1))


def _latest_watchdog_log(log_dir: Path) -> Path | None:
    logs = sorted(
        (p for p in log_dir.glob("mill_watchdog_*.log") if not p.name.endswith(".status.log")),
        key=lambda p: p.stat().st_mtime,
    )
    return logs[-1] if logs else None


def _phase_tail(log_text: str, phase_name: str) -> str:
    marker = f"START phase={phase_name}"
    idx = log_text.rfind(marker)
    if idx < 0:
        return log_text
    return log_text[idx:]


def _find_last_int(text: str, pattern: str) -> int | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    return int(matches[-1])


def _find_last_float(text: str, pattern: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    return float(matches[-1])


def _parse_watchdog_metrics(log_path: Path | None) -> dict[str, int | float | None]:
    metrics: dict[str, int | float | None] = {
        "gen_white_wins": None,
        "gen_black_wins": None,
        "gen_draws": None,
        "train_best_val_loss": None,
        "train_epochs_ran": None,
    }
    if log_path is None or not log_path.exists():
        return metrics

    text = log_path.read_text(encoding="utf-8", errors="replace")
    gen_text = _phase_tail(text, "generate")
    train_text = _phase_tail(text, "train")

    metrics["gen_white_wins"] = _find_last_int(gen_text, r"^\s*Siege Weiss:\s*(\d+)\s*$")
    metrics["gen_black_wins"] = _find_last_int(gen_text, r"^\s*Siege Schwarz:\s*(\d+)\s*$")
    metrics["gen_draws"] = _find_last_int(gen_text, r"^\s*Remis:\s*(\d+)\s*$")
    metrics["train_best_val_loss"] = _find_last_float(
        train_text,
        r"^\s*(?:Best val loss|Beste val loss):\s*([0-9]*\.?[0-9]+)\s*$",
    )

    epoch_matches = re.findall(r"^Epoch\s+(\d+)/(\d+)\s+\|", train_text, flags=re.MULTILINE)
    if epoch_matches:
        metrics["train_epochs_ran"] = int(epoch_matches[-1][0])

    return metrics


def _parse_benchmark_color_win_stats(benchmark_stdout: str) -> dict[str, int]:
    stats = {
        "bench_champion_wins_white": 0,
        "bench_champion_wins_black": 0,
        "bench_candidate_wins_white": 0,
        "bench_candidate_wins_black": 0,
    }
    game_pattern = re.compile(
        r"^Partie\s+\d+:\s+(?P<white>.+?) \(W\) gegen (?P<black>.+?) \(B\) -> (?P<outcome>.+?);",
        flags=re.MULTILINE,
    )

    for match in game_pattern.finditer(benchmark_stdout):
        white_name = match.group("white").strip()
        black_name = match.group("black").strip()
        outcome = match.group("outcome").strip()

        if outcome.lower().startswith("draw") or outcome.lower().startswith("remis"):
            continue

        winner_name: str | None = None
        winner_color: str | None = None
        if outcome == white_name:
            winner_name = white_name
            winner_color = "white"
        elif outcome == black_name:
            winner_name = black_name
            winner_color = "black"

        if winner_name == "champion":
            stats[f"bench_champion_wins_{winner_color}"] += 1
        elif winner_name == "candidate":
            stats[f"bench_candidate_wins_{winner_color}"] += 1

    return stats


def _parse_depths_csv(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            depth = int(item)
        except ValueError as exc:
            raise ValueError(f"Ungueltige Tiefe '{item}' in --ab-depths.") from exc
        if depth <= 0:
            raise ValueError("--ab-depths-Werte muessen > 0 sein")
        if depth not in values:
            values.append(depth)
    return values


def _parse_generation_summary(stdout: str) -> dict[str, int | None]:
    return {
        "white_wins": _find_last_int(stdout, r"^\s*Siege Wei(?:ss|ß):\s*(\d+)\s*$"),
        "black_wins": _find_last_int(stdout, r"^\s*Siege Schwarz:\s*(\d+)\s*$"),
        "draws": _find_last_int(stdout, r"^\s*Remis:\s*(\d+)\s*$"),
        "samples": _find_last_int(stdout, r"^\s*Samples:\s*(\d+)\s*$"),
    }


def _iter_jsonl_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if line:
                lines.append(line)
    return lines


def _sample_jsonl_lines(path: Path, max_samples: int, *, rng: random.Random) -> list[str]:
    if max_samples <= 0:
        return []

    reservoir: list[str] = []
    seen = 0
    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            if seen < max_samples:
                reservoir.append(line)
            else:
                replace_idx = rng.randint(0, seen)
                if replace_idx < max_samples:
                    reservoir[replace_idx] = line
            seen += 1
    return reservoir


def _merge_iteration_dataset(
    *,
    output_path: Path,
    selfplay_data_path: Path,
    ab_data_path: Path | None,
    replay_paths: list[Path],
    replay_samples_per_file: int,
    shuffle_merged_data: bool,
    rng: random.Random,
) -> dict[str, int]:
    merged_lines: list[str] = []
    source_counts: dict[str, int] = {}

    selfplay_lines = _iter_jsonl_lines(selfplay_data_path)
    merged_lines.extend(selfplay_lines)
    source_counts["selfplay"] = len(selfplay_lines)

    if ab_data_path is not None and ab_data_path.exists():
        ab_lines = _iter_jsonl_lines(ab_data_path)
        merged_lines.extend(ab_lines)
        source_counts["ab_teacher"] = len(ab_lines)
    else:
        source_counts["ab_teacher"] = 0

    replay_total = 0
    for replay_path in replay_paths:
        sampled_lines = _sample_jsonl_lines(
            replay_path,
            replay_samples_per_file,
            rng=rng,
        )
        replay_total += len(sampled_lines)
        merged_lines.extend(sampled_lines)
        source_counts[f"replay::{replay_path.name}"] = len(sampled_lines)
    source_counts["replay_total"] = replay_total

    if shuffle_merged_data and merged_lines:
        rng.shuffle(merged_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for line in merged_lines:
            fp.write(line + "\n")
    source_counts["merged_total"] = len(merged_lines)
    return source_counts


def _format_metric_value(value: object) -> str:
    if value is None:
        return "k.A."
    if isinstance(value, bool):
        return "ja" if value else "nein"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_iteration_table(iteration_summaries: list[dict[str, object]], ab_depths: list[int]) -> str:
    if not iteration_summaries:
        return "Keine Iterationszusammenfassungen verfuegbar."

    headers = ["Metrik"] + [str(summary["iteration"]) for summary in iteration_summaries]
    row_specs = [
        ("Gen Siege Weiss", "gen_white_wins"),
        ("Gen Siege Schwarz", "gen_black_wins"),
        ("Gen Remis", "gen_draws"),
        ("Train beste val_loss", "train_best_val_loss"),
        ("Train gelaufene Epochen", "train_epochs_ran"),
        ("Champion Siege (W)", "bench_champion_wins_white"),
        ("Champion Siege (B)", "bench_champion_wins_black"),
        ("Kandidat Siege (W)", "bench_candidate_wins_white"),
        ("Kandidat Siege (B)", "bench_candidate_wins_black"),
        ("Remis", "bench_draws"),
        ("Kandidat befoerdert", "promoted"),
    ]
    for depth in ab_depths:
        row_specs.extend(
            [
                (f"AB d{depth} champion siege", f"ab_d{depth}_wins"),
                (f"AB d{depth} remis", f"ab_d{depth}_draws"),
                (f"AB d{depth} champion niederlagen", f"ab_d{depth}_losses"),
            ]
        )

    rows: list[list[str]] = []
    for label, key in row_specs:
        row = [label]
        for summary in iteration_summaries:
            row.append(_format_metric_value(summary.get(key)))
        rows.append(row)

    widths = [0] * len(headers)
    all_rows = [headers, *rows]
    for row in all_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _render(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([_render(headers), separator, *(_render(row) for row in rows)])


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _build_iteration_table_latex(iteration_summaries: list[dict[str, object]], ab_depths: list[int]) -> str:
    if not iteration_summaries:
        return "% Keine Iterationszusammenfassungen verfuegbar."

    headers = ["Metrik"] + [str(summary["iteration"]) for summary in iteration_summaries]
    grouped_row_specs = [
        (
            "Datenerzeugung",
            "genbg",
            [
                ("Gen Siege Weiss", "gen_white_wins"),
                ("Gen Siege Schwarz", "gen_black_wins"),
                ("Gen Remis", "gen_draws"),
            ],
        ),
        (
            "Training",
            "trainbg",
            [
                ("Train beste val_loss", "train_best_val_loss"),
                ("Train gelaufene Epochen", "train_epochs_ran"),
            ],
        ),
        (
            "Benchmark",
            "benchbg",
            [
                ("Champion Siege (W)", "bench_champion_wins_white"),
                ("Champion Siege (B)", "bench_champion_wins_black"),
                ("Kandidat Siege (W)", "bench_candidate_wins_white"),
                ("Kandidat Siege (B)", "bench_candidate_wins_black"),
                ("Remis", "bench_draws"),
                ("Kandidat befoerdert", "promoted"),
            ],
        ),
    ]
    for depth in ab_depths:
        grouped_row_specs[-1][2].extend(
            [
                (f"AB d{depth} champion siege", f"ab_d{depth}_wins"),
                (f"AB d{depth} remis", f"ab_d{depth}_draws"),
                (f"AB d{depth} champion niederlagen", f"ab_d{depth}_losses"),
            ]
        )

    col_spec = "l" + ("c" * len(iteration_summaries))
    lines = [
        "% Im Vorspann benoetigt:",
        "% \\usepackage[table]{xcolor}",
        "% \\usepackage{array}",
        "\\setlength{\\arrayrulewidth}{1.1pt}",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\definecolor{headbg}{gray}{0.90}",
        "\\definecolor{genbg}{HTML}{EAF4FF}",
        "\\definecolor{trainbg}{HTML}{EAFBEA}",
        "\\definecolor{benchbg}{HTML}{FFF4E5}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
    ]

    header_cells = [_latex_escape(cell) for cell in headers]
    lines.append("\\rowcolor{headbg} " + " & ".join(header_cells) + r" \\")
    lines.append(r"\hline")

    for group_name, color_name, row_specs in grouped_row_specs:
        lines.append(
            f"\\rowcolor{{{color_name}}}\\multicolumn{{{len(headers)}}}{{l}}{{\\textbf{{{_latex_escape(group_name)}}}}} \\\\"
        )
        for label, key in row_specs:
            cells = [_latex_escape(label)]
            for summary in iteration_summaries:
                cells.append(_latex_escape(_format_metric_value(summary.get(key))))
            lines.append(f"\\rowcolor{{{color_name}}} " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations muss > 0 sein")
    if args.start_iteration <= 0:
        raise ValueError("--start-iteration muss > 0 sein")
    if args.games_per_iter <= 0:
        raise ValueError("--games-per-iter muss > 0 sein")
    if args.ab_games_per_iter < 0:
        raise ValueError("--ab-games-per-iter muss >= 0 sein")
    if args.ab_teacher_depth <= 0:
        raise ValueError("--ab-teacher-depth muss > 0 sein")
    if args.train_epochs <= 0:
        raise ValueError("--train-epochs muss > 0 sein")
    if args.eval_games <= 0:
        raise ValueError("--eval-games muss > 0 sein")
    if args.ab_eval_games <= 0:
        raise ValueError("--ab-eval-games muss > 0 sein")
    if args.draw_score < 0.0 or args.draw_score >= 1.0:
        raise ValueError("--draw-score muss in [0.0, 1.0) liegen")
    if args.promote_threshold < 0.0 or args.promote_threshold > 1.0:
        raise ValueError("--promote-threshold muss in [0.0, 1.0] liegen")
    if args.no_capture_draw_plies <= 0:
        raise ValueError("--no-capture-draw-plies muss > 0 sein")
    if args.replay_samples_per_file <= 0:
        raise ValueError("--replay-samples-per-file muss > 0 sein")
    ab_depths = _parse_depths_csv(args.ab_depths)
    if args.ab_eval and not ab_depths:
        raise ValueError("--ab-depths muss mindestens eine Tiefe enthalten, wenn --ab-eval aktiv ist.")
    active_ab_depths = ab_depths if args.ab_eval else []

    python_exe = args.python_exe if args.python_exe is not None else Path(sys.executable)
    if not python_exe.exists():
        raise FileNotFoundError(f"Python-Executable nicht gefunden: {python_exe}")

    initial_champion = args.initial_champion
    if not initial_champion.exists():
        raise FileNotFoundError(f"Initialer Champion nicht gefunden: {initial_champion}")

    args.champion_output.parent.mkdir(parents=True, exist_ok=True)
    if initial_champion.resolve() != args.champion_output.resolve():
        shutil.copy2(initial_champion, args.champion_output)
    current_champion = args.champion_output

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    args.watchdog_log_root.mkdir(parents=True, exist_ok=True)
    args.eval_log_dir.mkdir(parents=True, exist_ok=True)

    watchdog_script = REPO_ROOT / "scripts" / "mill" / "mill_train_watchdog.py"
    selfplay_script = REPO_ROOT / "scripts" / "mill" / "mill_generate_selfplay_data.py"
    teacher_script = REPO_ROOT / "scripts" / "mill" / "mill_generate_teacher_data.py"
    train_script = REPO_ROOT / "scripts" / "mill" / "mill_train_neural.py"
    benchmark_script = REPO_ROOT / "scripts" / "mill" / "mill_ai_benchmark.py"

    if not watchdog_script.exists():
        raise FileNotFoundError(f"Watchdog-Skript nicht gefunden: {watchdog_script}")
    if not selfplay_script.exists():
        raise FileNotFoundError(f"Selbstspiel-Skript nicht gefunden: {selfplay_script}")
    if not teacher_script.exists():
        raise FileNotFoundError(f"Teacher-Skript nicht gefunden: {teacher_script}")
    if not train_script.exists():
        raise FileNotFoundError(f"Training-Skript nicht gefunden: {train_script}")
    if not benchmark_script.exists():
        raise FileNotFoundError(f"Benchmark-Skript nicht gefunden: {benchmark_script}")
    for replay_path in args.replay_data:
        if not replay_path.exists():
            raise FileNotFoundError(f"--replay-data nicht gefunden: {replay_path}")

    iteration_summaries: list[dict[str, object]] = []

    for offset in range(args.iterations):
        iteration = args.start_iteration + offset
        iter_tag = f"iter_{iteration:03d}"
        print()
        print(f"=== Iteration {iter_tag} ===")
        print(f"Champion: {current_champion}")

        data_path = args.data_dir / f"{iter_tag}.jsonl"
        selfplay_data_path = args.data_dir / f"{iter_tag}_selfplay.jsonl"
        ab_data_path = args.data_dir / f"{iter_tag}_ab_teacher.jsonl"
        model_output = args.model_dir / f"{iter_tag}.pt"
        checkpoint_dir = args.checkpoint_root / iter_tag
        watchdog_log_dir = args.watchdog_log_root / iter_tag
        eval_log_path = args.eval_log_dir / f"{iter_tag}.log"

        gen_seed = args.seed_base + (offset * 2)
        train_seed = gen_seed + 1

        generate_selfplay_cmd = [
            str(python_exe),
            str(selfplay_script),
            "--model",
            str(current_champion),
            "--opponent-model",
            str(current_champion),
            "--games",
            str(args.games_per_iter),
            "--temperature",
            str(args.temperature),
            "--max-plies",
            str(args.max_plies),
            "--seed",
            str(gen_seed),
            "--device",
            args.device,
            _bool_flag("enable-flying", args.enable_flying),
            _bool_flag("enable-threefold-repetition", args.enable_threefold_repetition),
            _bool_flag("enable-no-capture-draw", args.enable_no_capture_draw),
            "--no-capture-draw-plies",
            str(args.no_capture_draw_plies),
            "--output",
            str(selfplay_data_path),
        ]

        selfplay_result = _run_command(generate_selfplay_cmd, cwd=REPO_ROOT, capture=True)
        if selfplay_result.stdout:
            print(selfplay_result.stdout, end="")
        if selfplay_result.stderr:
            print(selfplay_result.stderr, end="", file=sys.stderr)
        if selfplay_result.returncode != 0:
            print(
                f"Iteration {iter_tag} Selbstspiel-Datengenerierung fehlgeschlagen "
                f"(exit={selfplay_result.returncode})."
            )
            return selfplay_result.returncode
        selfplay_gen_metrics = _parse_generation_summary(selfplay_result.stdout or "")

        ab_gen_metrics = {"white_wins": 0, "black_wins": 0, "draws": 0, "samples": 0}
        if args.ab_games_per_iter > 0:
            generate_ab_cmd = [
                str(python_exe),
                str(teacher_script),
                "--games",
                str(args.ab_games_per_iter),
                "--teacher-depth",
                str(args.ab_teacher_depth),
                "--teacher-random-tiebreak",
                "--alternate-colors",
                "--max-plies",
                str(args.max_plies),
                "--seed",
                str(gen_seed + 10_000),
                _bool_flag("enable-flying", args.enable_flying),
                _bool_flag("enable-threefold-repetition", args.enable_threefold_repetition),
                _bool_flag("enable-no-capture-draw", args.enable_no_capture_draw),
                "--no-capture-draw-plies",
                str(args.no_capture_draw_plies),
                "--output",
                str(ab_data_path),
            ]
            ab_result = _run_command(generate_ab_cmd, cwd=REPO_ROOT, capture=True)
            if ab_result.stdout:
                print(ab_result.stdout, end="")
            if ab_result.stderr:
                print(ab_result.stderr, end="", file=sys.stderr)
            if ab_result.returncode != 0:
                print(
                    f"Iteration {iter_tag} AB-Teacher-Datengenerierung fehlgeschlagen "
                    f"(exit={ab_result.returncode})."
                )
                return ab_result.returncode
            ab_gen_metrics = _parse_generation_summary(ab_result.stdout or "")

        merge_rng = random.Random(args.seed_base + 1_000_000 + iteration)
        merged_counts = _merge_iteration_dataset(
            output_path=data_path,
            selfplay_data_path=selfplay_data_path,
            ab_data_path=ab_data_path if args.ab_games_per_iter > 0 else None,
            replay_paths=args.replay_data,
            replay_samples_per_file=args.replay_samples_per_file,
            shuffle_merged_data=args.shuffle_merged_data,
            rng=merge_rng,
        )
        print(
            f"Iteration {iter_tag} Datenmix: selfplay={merged_counts.get('selfplay', 0)} "
            f"ab_teacher={merged_counts.get('ab_teacher', 0)} "
            f"replay_total={merged_counts.get('replay_total', 0)} "
            f"gesamt={merged_counts.get('merged_total', 0)}"
        )

        train_cmd = [
            str(python_exe),
            str(train_script),
            "--data",
            str(data_path),
            "--init-model",
            str(current_champion),
            "--seed",
            str(train_seed),
            "--validation-split",
            str(args.validation_split),
            "--epochs",
            str(args.train_epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--early-stopping-patience",
            str(args.early_stopping_patience),
            "--early-stopping-min-delta",
            str(args.early_stopping_min_delta),
            "--device",
            args.device,
            "--output",
            str(model_output),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ]

        watchdog_cmd = [
            str(python_exe),
            str(watchdog_script),
            "--interval-seconds",
            str(args.watchdog_interval_seconds),
            "--max-restarts",
            str(args.watchdog_train_max_restarts),
            "--restart-delay-seconds",
            str(args.watchdog_restart_delay_seconds),
            "--data-path",
            str(data_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--log-dir",
            str(watchdog_log_dir),
            "--",
            *train_cmd,
        ]

        watchdog_result = _run_command(watchdog_cmd, cwd=REPO_ROOT, capture=False)
        if watchdog_result.returncode != 0:
            print(f"Iteration {iter_tag} ist waehrend Training fehlgeschlagen (exit={watchdog_result.returncode}).")
            return watchdog_result.returncode

        candidate_best = checkpoint_dir / "best.pt"
        if not candidate_best.exists():
            print(f"Iteration {iter_tag} hat keinen Kandidaten-Checkpoint erzeugt: {candidate_best}")
            return 1

        watchdog_metrics = _parse_watchdog_metrics(_latest_watchdog_log(watchdog_log_dir))
        watchdog_metrics["gen_white_wins"] = (
            int(selfplay_gen_metrics.get("white_wins") or 0) + int(ab_gen_metrics.get("white_wins") or 0)
        )
        watchdog_metrics["gen_black_wins"] = (
            int(selfplay_gen_metrics.get("black_wins") or 0) + int(ab_gen_metrics.get("black_wins") or 0)
        )
        watchdog_metrics["gen_draws"] = int(selfplay_gen_metrics.get("draws") or 0) + int(ab_gen_metrics.get("draws") or 0)

        benchmark_cmd = [
            str(python_exe),
            str(benchmark_script),
            "--ai-a",
            "neural",
            "--ai-a-arg",
            f"model_path={candidate_best}",
            "--ai-b",
            "neural",
            "--ai-b-arg",
            f"model_path={current_champion}",
            "--label-a",
            "candidate",
            "--label-b",
            "champion",
            "--games",
            str(args.eval_games),
            "--deterministic",
            "--alternate-colors",
            _bool_flag("enable-flying", args.enable_flying),
            _bool_flag("enable-threefold-repetition", args.enable_threefold_repetition),
            _bool_flag("enable-no-capture-draw", args.enable_no_capture_draw),
            "--no-capture-draw-plies",
            str(args.no_capture_draw_plies),
        ]

        benchmark_result = _run_command(benchmark_cmd, cwd=REPO_ROOT, capture=True)
        if benchmark_result.stdout:
            print(benchmark_result.stdout, end="")
        if benchmark_result.stderr:
            print(benchmark_result.stderr, end="", file=sys.stderr)
        eval_log_path.parent.mkdir(parents=True, exist_ok=True)
        eval_log_path.write_text((benchmark_result.stdout or "") + (benchmark_result.stderr or ""), encoding="utf-8")
        if benchmark_result.returncode != 0:
            print(f"Iteration {iter_tag} Benchmark fehlgeschlagen (exit={benchmark_result.returncode}).")
            return benchmark_result.returncode

        candidate_wins, champion_wins, draws = _parse_benchmark_summary(
            benchmark_result.stdout or "",
            label_a="candidate",
            label_b="champion",
        )
        benchmark_color_wins = _parse_benchmark_color_win_stats(benchmark_result.stdout or "")
        score = (candidate_wins + args.draw_score * draws) / args.eval_games
        print(
            f"Iteration {iter_tag} Bewertung: "
            f"kandidat_siege={candidate_wins} champion_siege={champion_wins} "
            f"remis={draws} remis_score={args.draw_score:.2f} score={score:.3f}"
        )

        promote = (candidate_wins > champion_wins) and (score >= args.promote_threshold)
        if promote:
            shutil.copy2(candidate_best, args.champion_output)
            current_champion = args.champion_output
            print(f"Promotion: Kandidat als neuer Champion akzeptiert (Schwelle={args.promote_threshold:.3f}).")
        else:
            print(
                "Keine Promotion: "
                f"kandidat_siege={candidate_wins}, champion_siege={champion_wins}, "
                f"score={score:.3f}, schwelle={args.promote_threshold:.3f}."
            )

        ab_eval_results: dict[int, dict[str, int]] = {}
        if args.ab_eval:
            for depth in active_ab_depths:
                ab_label = f"alphabeta_d{depth}"
                ab_eval_log_path = args.eval_log_dir / f"{iter_tag}_{ab_label}.log"
                ab_cmd = [
                    str(python_exe),
                    str(benchmark_script),
                    "--ai-a",
                    "neural",
                    "--ai-a-arg",
                    f"model_path={current_champion}",
                    "--ai-b",
                    "alphabeta",
                    "--ai-b-arg",
                    f"depth={depth}",
                    "--label-a",
                    "champion",
                    "--label-b",
                    ab_label,
                    "--games",
                    str(args.ab_eval_games),
                    "--deterministic",
                    "--alternate-colors",
                    _bool_flag("enable-flying", args.enable_flying),
                    _bool_flag("enable-threefold-repetition", args.enable_threefold_repetition),
                    _bool_flag("enable-no-capture-draw", args.enable_no_capture_draw),
                    "--no-capture-draw-plies",
                    str(args.no_capture_draw_plies),
                ]

                ab_result = _run_command(ab_cmd, cwd=REPO_ROOT, capture=True)
                if ab_result.stdout:
                    print(ab_result.stdout, end="")
                if ab_result.stderr:
                    print(ab_result.stderr, end="", file=sys.stderr)
                ab_eval_log_path.parent.mkdir(parents=True, exist_ok=True)
                ab_eval_log_path.write_text((ab_result.stdout or "") + (ab_result.stderr or ""), encoding="utf-8")
                if ab_result.returncode != 0:
                    print(f"Iteration {iter_tag} AB-Tiefe {depth} Benchmark fehlgeschlagen (exit={ab_result.returncode}).")
                    return ab_result.returncode

                champ_wins, ab_wins, ab_draws = _parse_benchmark_summary(
                    ab_result.stdout or "",
                    label_a="champion",
                    label_b=ab_label,
                )
                ab_eval_results[depth] = {
                    "wins": champ_wins,
                    "draws": ab_draws,
                    "losses": ab_wins,
                }
                print(
                    f"Iteration {iter_tag} AB d{depth}: "
                    f"champion_siege={champ_wins} remis={ab_draws} champion_niederlagen={ab_wins}"
                )

        iteration_summaries.append(
            {
                "iteration": iter_tag,
                "gen_white_wins": watchdog_metrics["gen_white_wins"],
                "gen_black_wins": watchdog_metrics["gen_black_wins"],
                "gen_draws": watchdog_metrics["gen_draws"],
                "train_best_val_loss": watchdog_metrics["train_best_val_loss"],
                "train_epochs_ran": watchdog_metrics["train_epochs_ran"],
                "bench_champion_wins_white": benchmark_color_wins["bench_champion_wins_white"],
                "bench_champion_wins_black": benchmark_color_wins["bench_champion_wins_black"],
                "bench_candidate_wins_white": benchmark_color_wins["bench_candidate_wins_white"],
                "bench_candidate_wins_black": benchmark_color_wins["bench_candidate_wins_black"],
                "bench_draws": draws,
                "promoted": promote,
                **{
                    f"ab_d{depth}_wins": ab_eval_results.get(depth, {}).get("wins")
                    for depth in active_ab_depths
                },
                **{
                    f"ab_d{depth}_draws": ab_eval_results.get(depth, {}).get("draws")
                    for depth in active_ab_depths
                },
                **{
                    f"ab_d{depth}_losses": ab_eval_results.get(depth, {}).get("losses")
                    for depth in active_ab_depths
                },
            }
        )

    print()
    print("Selbstspiel-Schleife abgeschlossen.")
    print(f"Champion: {current_champion}")
    print()
    print("Iterationszusammenfassung")
    print(_build_iteration_table(iteration_summaries, active_ab_depths))
    print()
    print("Iterationszusammenfassung (LaTeX)")
    print(_build_iteration_table_latex(iteration_summaries, active_ab_depths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
