"""Prueft Muehle-PyTorch-Checkpoint-Dateien (.pt)."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

# Erlaubt das direkte Ausfuehren dieses Skripts aus der Repository-Wurzel.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EPOCH_FILE_RE = re.compile(r"^epoch_(\d+)\.pt$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prueft Muehle-Torch-Checkpoints und Trainingsmetriken.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/mill_torch/last.pt"),
        help="Zu pruefende Checkpoint-Datei.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=None,
        help="Optionales Verzeichnis mit epoch_XXX.pt-Dateien fuer Verlaufsmetriken.",
    )
    parser.add_argument(
        "--max-history-rows",
        type=int,
        default=20,
        help="Maximale Anzahl an Epochenzeilen aus dem Verlauf.",
    )
    parser.add_argument(
        "--show-optimizer",
        action="store_true",
        help="Zeigt eine kompakte Zusammenfassung des Optimizer-Status, falls vorhanden.",
    )
    return parser


def _fmt_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _load_checkpoint(torch_mod: Any, path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {path}")
    payload = torch_mod.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Ungueltige Checkpoint-Nutzlast (dict erwartet): {path}")
    return payload


def _print_main_checkpoint_summary(checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
    print("Checkpoint-Info")
    print(f"  path: {checkpoint_path}")
    print(f"  version: {checkpoint.get('version', '-')}")
    print(f"  epoch: {checkpoint.get('epoch', '-')}")
    print(f"  best_val_loss: {_fmt_float(checkpoint.get('best_val_loss'))}")
    model_kwargs = checkpoint.get("model_kwargs", {})
    print(f"  model_kwargs: {model_kwargs if model_kwargs else '{}'}")

    metrics = checkpoint.get("extra_metrics", {})
    if isinstance(metrics, dict) and metrics:
        print("  extra_metrics:")
        for key in sorted(metrics):
            print(f"    {key}: {_fmt_float(metrics[key])}")
    else:
        print("  extra_metrics: {}")


def _tensor_summary_rows(model_state: Any) -> list[tuple[str, str, int, float, float, float, float, float]]:
    if not isinstance(model_state, dict):
        return []

    rows: list[tuple[str, str, int, float, float, float, float, float]] = []
    for name, tensor in sorted(model_state.items()):
        if not hasattr(tensor, "numel"):
            continue
        data = tensor.detach().float().cpu()
        numel = int(data.numel())
        if numel == 0:
            mean = std = min_value = max_value = l2_norm = 0.0
        else:
            mean = float(data.mean().item())
            std = float(data.std(unbiased=False).item())
            min_value = float(data.min().item())
            max_value = float(data.max().item())
            l2_norm = float(data.norm().item())
        shape = "x".join(str(dim) for dim in data.shape) or "skalar"
        rows.append((name, shape, numel, mean, std, min_value, max_value, l2_norm))
    return rows


def _print_tensor_summary(checkpoint: dict[str, Any]) -> None:
    rows = _tensor_summary_rows(checkpoint.get("model_state"))
    if not rows:
        print()
        print("Parameter-Statistik")
        print("  model_state fehlt oder ist leer.")
        return

    total_params = sum(row[2] for row in rows)
    print()
    print("Parameter-Statistik")
    print(f"  tensoren: {len(rows)}")
    print(f"  gesamt_parameter: {total_params}")
    print("  name | form | anzahl | mittel | std | min | max | l2")
    for name, shape, numel, mean, std, min_value, max_value, l2_norm in rows:
        print(
            "  "
            f"{name} | {shape} | {numel} | "
            f"{mean:.6f} | {std:.6f} | {min_value:.6f} | {max_value:.6f} | {l2_norm:.6f}"
        )


def _epoch_sort_key(path: Path) -> int:
    match = EPOCH_FILE_RE.match(path.name)
    if not match:
        return 10**9
    return int(match.group(1))


def _print_optimizer_summary(checkpoint: dict[str, Any]) -> None:
    optimizer_state = checkpoint.get("optimizer_state")
    print()
    print("Optimierer")
    if not isinstance(optimizer_state, dict):
        print("  optimizer_state: nicht vorhanden")
        return
    param_groups = optimizer_state.get("param_groups")
    state = optimizer_state.get("state")
    group_count = len(param_groups) if isinstance(param_groups, list) else 0
    state_count = len(state) if isinstance(state, dict) else 0
    print(f"  param_groups: {group_count}")
    print(f"  state_entries: {state_count}")
    if group_count > 0 and isinstance(param_groups[0], dict):
        sample = param_groups[0]
        learning_rate = sample.get("lr", "-")
        weight_decay = sample.get("weight_decay", "-")
        print(f"  group0.lr: {learning_rate}")
        print(f"  group0.weight_decay: {weight_decay}")


def _print_history(torch_mod: Any, history_dir: Path, max_rows: int) -> None:
    if max_rows <= 0:
        return
    if not history_dir.exists() or not history_dir.is_dir():
        print()
        print(f"Verlauf ({history_dir})")
        print("  kein Verlaufsverzeichnis gefunden")
        return

    epoch_files = sorted(
        [path for path in history_dir.glob("epoch_*.pt") if EPOCH_FILE_RE.match(path.name)],
        key=_epoch_sort_key,
    )
    if not epoch_files:
        print()
        print(f"Verlauf ({history_dir})")
        print("  keine epoch_XXX.pt-Dateien gefunden")
        return

    rows: list[tuple[int, Any, Any, Any, Any, Any]] = []
    for path in epoch_files:
        payload = _load_checkpoint(torch_mod, path)
        metrics = payload.get("extra_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        rows.append(
            (
                int(payload.get("epoch", _epoch_sort_key(path))),
                metrics.get("train_loss"),
                metrics.get("val_loss"),
                metrics.get("train_acc"),
                metrics.get("val_acc"),
                metrics.get("val_value_mse"),
            )
        )

    print()
    print(f"Verlauf ({history_dir})")
    print("  epoch | train_loss | val_loss | train_acc | val_acc | val_value_mse")
    for epoch, train_loss, val_loss, train_acc, val_acc, val_value_mse in rows[-max_rows:]:
        print(
            "  "
            f"{epoch:>5} | "
            f"{_fmt_float(train_loss):>10} | "
            f"{_fmt_float(val_loss):>8} | "
            f"{_fmt_float(train_acc):>9} | "
            f"{_fmt_float(val_acc):>7} | "
            f"{_fmt_float(val_value_mse):>13}"
        )


def main() -> None:
    args = build_parser().parse_args()

    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise SystemExit("PyTorch wird fuer die Checkpoint-Pruefung benoetigt. Bitte zuerst torch installieren.")

    checkpoint = _load_checkpoint(torch, args.checkpoint)
    _print_main_checkpoint_summary(checkpoint, args.checkpoint)
    _print_tensor_summary(checkpoint)
    if args.show_optimizer:
        _print_optimizer_summary(checkpoint)

    history_dir = args.history_dir if args.history_dir is not None else args.checkpoint.parent
    _print_history(torch, history_dir, max_rows=args.max_history_rows)


if __name__ == "__main__":
    main()
