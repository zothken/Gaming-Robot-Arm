"""Datensatz-Helfer fuer Muehle-Training."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch

from .features import MOVE_FEATURE_DIM, STATE_FEATURE_DIM


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
):
    indices = list(range(len(samples)))
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        yield [samples[idx] for idx in chunk]


__all__ = ["Sample", "collate_batch", "iterate_minibatches", "load_samples"]
