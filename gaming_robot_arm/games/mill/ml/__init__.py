"""ML-Bausteine fuer Muehle."""

from .features import (
    MOVE_FEATURE_DIM,
    STATE_FEATURE_DIM,
    encode_legal_move_features,
    encode_move_features,
    encode_state_features,
    outcome_for_player,
)
from .selfplay import move_key, target_index_or_raise

__all__ = [
    "MOVE_FEATURE_DIM",
    "STATE_FEATURE_DIM",
    "encode_legal_move_features",
    "encode_move_features",
    "encode_state_features",
    "move_key",
    "outcome_for_player",
    "target_index_or_raise",
]

try:
    from .dataset import Sample, collate_batch, iterate_minibatches, load_samples
    from .model import MillPolicyValueNet, checkpoint_model_kwargs, load_checkpoint, save_checkpoint, select_torch_device
    from .training import evaluate_dataset
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "MillPolicyValueNet",
            "Sample",
            "checkpoint_model_kwargs",
            "collate_batch",
            "evaluate_dataset",
            "iterate_minibatches",
            "load_checkpoint",
            "load_samples",
            "save_checkpoint",
            "select_torch_device",
        ]
    )
