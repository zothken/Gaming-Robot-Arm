"""Trainiert ein PyTorch-Policy/Value-Muehle-Modell aus JSONL-Teacher-Daten."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Iterable

# Erlaubt das direkte Ausfuehren dieses Skripts aus der Repository-Wurzel.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from gaming_robot_arm.games.mill.neural_features import MOVE_FEATURE_DIM, STATE_FEATURE_DIM
from gaming_robot_arm.games.mill.neural_model import (
    MillPolicyValueNet,
    load_checkpoint,
    save_checkpoint,
    select_torch_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trainiert ein Muehle-Policy/Value-Torch-Modell aus JSONL-Samples.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/mill_teacher.jsonl"),
        help="Eingabe-JSONL-Pfad, erzeugt durch mill_generate_teacher_data.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/mill_torch_v1.pt"),
        help="Ausgabe-Checkpoint-Pfad.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Trainings-Epochen.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch-Groesse.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW-Lernrate.")
    parser.add_argument("--value-loss-weight", type=float, default=0.5, help="Relatives Gewicht fuer Value-MSE-Loss.")
    parser.add_argument(
        "--draw-outcome-target",
        type=float,
        default=-0.1,
        help="Value-Ziel fuer Remis-Samples (statt 0.0). Muss in [-1, 1] liegen.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW-Weight-Decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradienten-Norm-Clipping; <=0 deaktiviert.")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validierungsanteil in [0, 0.5).")
    parser.add_argument("--seed", type=int, default=1234, help="Zufallsseed fuer Split/Shuffle/Initialisierung.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optionale Obergrenze fuer geladene Samples.")
    parser.add_argument("--hidden-dim", type=int, default=192, help="Hidden-Dimension fuer Zustands-/Zug-Tuerme.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout in den Modelltuermen [0, 1).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch-Geraet.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints/mill_torch"), help="Verzeichnis fuer periodische Checkpoints.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Speichert alle N Epochen einen Checkpoint.")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Stoppt frueh, wenn sich val_loss N Epochen in Folge nicht verbessert; <=0 deaktiviert.",)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0, help="Minimale val_loss-Verbesserung zum Zuruecksetzen des fruehen Stoppens.",)
    parser.add_argument("--init-model", type=Path, default=None, help="Optionaler Checkpoint fuer Modellgewichte zum Warmstart.",)
    parser.add_argument("--resume-training-state", action="store_true", help="Beim Warmstart auch Optimizer-State, Startepoche und best_val_loss aus Checkpoint uebernehmen.",)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs muss > 0 sein")
    if args.batch_size <= 0:
        raise ValueError("--batch-size muss > 0 sein")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate muss > 0 sein")
    if args.value_loss_weight < 0:
        raise ValueError("--value-loss-weight muss >= 0 sein")
    if args.draw_outcome_target < -1.0 or args.draw_outcome_target > 1.0:
        raise ValueError("--draw-outcome-target muss in [-1, 1] liegen")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay muss >= 0 sein")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim muss > 0 sein")
    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError("--dropout muss im Bereich [0, 1) liegen")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every muss > 0 sein")
    if not (0.0 <= args.validation_split < 0.5):
        raise ValueError("--validation-split muss in [0, 0.5) liegen")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples muss > 0 sein, wenn gesetzt")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience muss >= 0 sein")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("--early-stopping-min-delta muss >= 0 sein")
    if args.resume_training_state and args.init_model is None:
        raise ValueError("--resume-training-state erfordert --init-model")


class Sample:
    __slots__ = ("state", "moves", "target_index", "outcome")

    def __init__(self, state: np.ndarray, moves: np.ndarray, target_index: int, outcome: float):
        self.state = state
        self.moves = moves
        self.target_index = target_index
        self.outcome = outcome


def _float_array(values: Iterable[float], expected_len: int, field_name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.shape != (expected_len,):
        raise ValueError(f"{field_name}-Shape passt nicht: erhalten {arr.shape}, erwartet ({expected_len},)")
    return arr


def load_samples(
    path: Path,
    max_samples: int | None = None,
    *,
    draw_outcome_target: float = 0.0,
) -> tuple[list[Sample], int]:
    if not path.exists():
        raise FileNotFoundError(f"Trainingsdaten nicht gefunden: {path}")

    samples: list[Sample] = []
    draw_sample_count = 0
    with path.open("r", encoding="utf-8") as fp:
        for line_number, raw_line in enumerate(fp, start=1):
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            state = _float_array(payload["state"], STATE_FEATURE_DIM, "state")
            move_rows = np.asarray(payload["moves"], dtype=np.float32)
            if move_rows.ndim != 2 or move_rows.shape[1] != MOVE_FEATURE_DIM:
                raise ValueError(
                    f"moves-Shape passt nicht in Zeile {line_number}: "
                    f"erhalten {move_rows.shape}, erwartet (num_moves, {MOVE_FEATURE_DIM})"
                )
            if move_rows.shape[0] == 0:
                raise ValueError(f"moves darf nicht leer sein (Zeile {line_number})")

            target_index = int(payload["target_index"])
            if not (0 <= target_index < move_rows.shape[0]):
                raise ValueError(f"target_index ausserhalb des Bereichs in Zeile {line_number}")

            outcome = float(payload["outcome"])
            if outcome == 0.0:
                outcome = draw_outcome_target
                draw_sample_count += 1

            if outcome < -1.0 or outcome > 1.0:
                raise ValueError(f"outcome muss in [-1, 1] liegen (Zeile {line_number})")

            samples.append(Sample(state=state, moves=move_rows, target_index=target_index, outcome=outcome))
            if max_samples is not None and len(samples) >= max_samples:
                break

    if not samples:
        raise ValueError("Keine Samples aus dem Datensatz geladen.")
    return samples, draw_sample_count


def collate_batch(
    samples: list[Sample],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(samples)
    max_moves = max(sample.moves.shape[0] for sample in samples)

    state_batch = torch.from_numpy(np.stack([sample.state for sample in samples])).to(device=device, dtype=torch.float32)
    move_batch = torch.zeros((batch_size, max_moves, MOVE_FEATURE_DIM), dtype=torch.float32, device=device)
    move_mask = torch.zeros((batch_size, max_moves), dtype=torch.bool, device=device)
    target_batch = torch.zeros((batch_size,), dtype=torch.long, device=device)
    outcome_batch = torch.zeros((batch_size,), dtype=torch.float32, device=device)

    for idx, sample in enumerate(samples):
        move_count = sample.moves.shape[0]
        move_batch[idx, :move_count, :] = torch.from_numpy(sample.moves).to(device=device, dtype=torch.float32)
        move_mask[idx, :move_count] = True
        target_batch[idx] = sample.target_index
        outcome_batch[idx] = sample.outcome

    return state_batch, move_batch, move_mask, target_batch, outcome_batch


def iterate_minibatches(
    samples: list[Sample],
    *,
    batch_size: int,
    rng: random.Random,
    shuffle: bool,
) -> Iterable[list[Sample]]:
    indices = list(range(len(samples)))
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        yield [samples[idx] for idx in chunk]


def evaluate_dataset(
    *,
    model: MillPolicyValueNet,
    samples: list[Sample],
    batch_size: int,
    device: torch.device,
    value_loss_weight: float,
    rng: random.Random,
) -> tuple[float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0

    model.eval()
    total_loss = 0.0
    total_policy_correct = 0
    total_value_mse = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_samples in iterate_minibatches(samples, batch_size=batch_size, rng=rng, shuffle=False):
            states, moves, mask, targets, outcomes = collate_batch(batch_samples, device=device)
            logits, values = model(states, moves, mask)

            policy_loss = F.cross_entropy(logits, targets)
            value_mse = F.mse_loss(values, outcomes)
            loss = policy_loss + value_loss_weight * value_mse

            batch_count = targets.shape[0]
            total_loss += float(loss.item()) * batch_count
            total_value_mse += float(value_mse.item()) * batch_count
            total_policy_correct += int((torch.argmax(logits, dim=1) == targets).sum().item())
            total_count += batch_count

    return total_loss / total_count, total_policy_correct / total_count, total_value_mse / total_count


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_torch_device(args.device)
    print(f"Verwende Geraet: {device}")
    print(f"Lade Samples aus {args.data} ...")
    samples, draw_sample_count = load_samples(
        args.data,
        max_samples=args.max_samples,
        draw_outcome_target=args.draw_outcome_target,
    )
    print(f"{len(samples)} Samples geladen")
    if draw_sample_count > 0:
        print(
            f"Remis-Targets umkodiert: {draw_sample_count} Samples auf "
            f"{args.draw_outcome_target:.3f}"
        )

    rng = random.Random(args.seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    split_at = int(len(indices) * (1.0 - args.validation_split))
    if split_at <= 0 or split_at >= len(indices):
        raise ValueError("Ungueltiger Split: --validation-split anpassen oder mehr Samples verwenden.")

    train_samples = [samples[idx] for idx in indices[:split_at]]
    val_samples = [samples[idx] for idx in indices[split_at:]]

    checkpoint: dict[str, Any] | None = None
    start_epoch = 1
    best_val_loss = float("inf")

    if args.init_model is not None:
        checkpoint = load_checkpoint(args.init_model, device=device)
        model_kwargs = dict(checkpoint["model_kwargs"])
        model = MillPolicyValueNet(**model_kwargs).to(device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Warmstart von {args.init_model}")
    else:
        model_kwargs = {
            "state_dim": STATE_FEATURE_DIM,
            "move_dim": MOVE_FEATURE_DIM,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        }
        model = MillPolicyValueNet(**model_kwargs).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.resume_training_state:
        assert checkpoint is not None  # guaranteed by validate_args
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        else:
            print("Fortsetzen angefordert, aber Checkpoint ohne optimizer_state; nutze frischen Optimizer-State.")

        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = checkpoint_epoch + 1
        checkpoint_best_val = checkpoint.get("best_val_loss")
        if checkpoint_best_val is not None:
            best_val_loss = float(checkpoint_best_val)
        print(
            f"Trainingszustand von {args.init_model} fortgesetzt | "
            f"start_epoch={start_epoch} best_val_loss={best_val_loss:.4f}"
        )

    if start_epoch > args.epochs:
        print(
            f"Keine Epochen auszufuehren: start_epoch={start_epoch} ist groesser als --epochs={args.epochs}. "
            "Nichts zu trainieren."
        )
        return

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    no_improve_epochs = 0
    last_completed_epoch = start_epoch - 1

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_value_mse_sum = 0.0
        train_correct = 0
        train_count = 0

        for batch_samples in iterate_minibatches(train_samples, batch_size=args.batch_size, rng=rng, shuffle=True):
            states, moves, mask, targets, outcomes = collate_batch(batch_samples, device=device)
            logits, values = model(states, moves, mask)

            policy_loss = F.cross_entropy(logits, targets)
            value_mse = F.mse_loss(values, outcomes)
            loss = policy_loss + args.value_loss_weight * value_mse

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            batch_count = targets.shape[0]
            train_loss_sum += float(loss.item()) * batch_count
            train_value_mse_sum += float(value_mse.item()) * batch_count
            train_correct += int((torch.argmax(logits, dim=1) == targets).sum().item())
            train_count += batch_count

        train_loss = train_loss_sum / train_count
        train_acc = train_correct / train_count
        train_value_mse = train_value_mse_sum / train_count
        val_loss, val_acc, val_value_mse = evaluate_dataset(
            model=model,
            samples=val_samples,
            batch_size=args.batch_size,
            device=device,
            value_loss_weight=args.value_loss_weight,
            rng=rng,
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} train_value_mse={train_value_mse:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_value_mse={val_value_mse:.4f}"
        )
        last_completed_epoch = epoch

        epoch_metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_value_mse": train_value_mse,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_value_mse": val_value_mse,
        }

        improved = val_loss < (best_val_loss - args.early_stopping_min_delta)
        if improved:
            best_val_loss = val_loss
            no_improve_epochs = 0
            save_checkpoint(
                path=args.checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                extra_metrics=epoch_metrics,
            )
        else:
            no_improve_epochs += 1

        if epoch % args.checkpoint_every == 0:
            save_checkpoint(
                path=args.checkpoint_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss if best_val_loss != float("inf") else None,
                extra_metrics=epoch_metrics,
            )

        if args.early_stopping_patience > 0 and no_improve_epochs >= args.early_stopping_patience:
            print(
                "Fruehes Stoppen ausgeloest: "
                f"keine val_loss-Verbesserung > {args.early_stopping_min_delta:.6f} "
                f"ueber {args.early_stopping_patience} aufeinanderfolgende Epochen."
            )
            break

    save_checkpoint(
        path=args.output,
        model=model,
        optimizer=optimizer,
        epoch=last_completed_epoch,
        best_val_loss=best_val_loss,
        extra_metrics={"val_loss": best_val_loss},
    )
    shutil.copy2(args.output, args.checkpoint_dir / "last.pt")

    duration = time.perf_counter() - start
    print()
    print("Training abgeschlossen")
    print(f"  Modell: {args.output}")
    print(f"  Beste val loss: {best_val_loss:.4f}")
    print(f"  Trainingssamples: {len(train_samples)}")
    print(f"  Validierungssamples: {len(val_samples)}")
    print(f"  Dauer: {duration:.1f}s")


if __name__ == "__main__":
    main()
