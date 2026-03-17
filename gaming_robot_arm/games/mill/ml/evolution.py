"""Wiederverwendbare Evolutions-Helfer fuer Muehle."""

from __future__ import annotations

import torch


def clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in state_dict.items()}


def lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


__all__ = ["clone_state_dict", "lerp"]
