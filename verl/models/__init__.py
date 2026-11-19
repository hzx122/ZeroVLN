"""Model adapters for the closed-loop navigation trainers."""

from .qwen36_action_head import (
    ActionHeadConfig,
    Qwen36ActionHead,
    extract_last_hidden_state,
    set_backbone_trainable,
)
from .qwen36_fast_slow_action_head import (
    FastSlowActionHeadConfig,
    Qwen36FastSlowActionHead,
)

__all__ = [
    "ActionHeadConfig",
    "Qwen36ActionHead",
    "extract_last_hidden_state",
    "set_backbone_trainable",
    "FastSlowActionHeadConfig",
    "Qwen36FastSlowActionHead",
]
