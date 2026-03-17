"""Training-Helfer fuer Muehle-Modelle."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from .dataset import collate_batch, iterate_minibatches
from .model import MillPolicyValueNet


def evaluate_dataset(
    *,
    model: MillPolicyValueNet,
    samples,
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


__all__ = ["evaluate_dataset"]
