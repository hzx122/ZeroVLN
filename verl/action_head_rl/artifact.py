"""Training-side reader for vLLM four-query and MoE routing artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


_CAPTURE_DTYPES = {
    'query_hidden': torch.float32,
    'query_routed_experts': torch.int64,
    'query_positions': torch.int64,
    'all_routed_experts': torch.int64,
    'all_positions': torch.int64,
    'all_input_ids': torch.int64,
}


def own_query_capture_tensors(tensors):
    """Own persisted fields once, without retaining a whole safetensors mmap.

    Text and FM may share these immutable CPU tensors. Keep the established
    FP32/int64 values; this is ownership and sharing, not route compression.
    Other capture fields are consumed as scalars by the decision builder.
    """
    result = dict(tensors)
    for name, dtype in _CAPTURE_DTYPES.items():
        if name in result:
            result[name] = result[name].to(device='cpu', dtype=dtype, copy=True)
    return result


def query_routing_artifact_from_tensors(tensors):
    """Reuse an already owned capture in the text and action expert readers."""
    required = ('query_hidden', 'query_routed_experts', 'query_positions')
    missing = set(required) - set(tensors)
    if missing:
        raise ValueError(f'query artifact is missing {sorted(missing)}')
    return QueryRoutingArtifact(**{key: tensors.get(key) for key in _CAPTURE_DTYPES})


@dataclass(frozen=True)
class QueryRoutingArtifact:
    query_hidden: torch.Tensor
    query_routed_experts: torch.Tensor
    query_positions: torch.Tensor
    all_routed_experts: torch.Tensor | None = None
    all_positions: torch.Tensor | None = None
    all_input_ids: torch.Tensor | None = None

    def validate(self, *, hidden_size: int, layer_count: int,
                 top_k: int, require_full: bool, query_count: int = 4) -> None:
        if query_count <= 0:
            raise ValueError("query_count must be positive")
        if tuple(self.query_hidden.shape) != (query_count, hidden_size):
            raise ValueError(
                f"query_hidden must have shape ({query_count},{hidden_size}), got "
                f"{tuple(self.query_hidden.shape)}")
        if tuple(self.query_routed_experts.shape) != (query_count, layer_count, top_k):
            raise ValueError("query_routed_experts has an incompatible shape")
        if tuple(self.query_positions.shape) != (query_count,):
            raise ValueError("query_positions must have shape (query_count,)")
        if not torch.equal(
                self.query_positions[1:] - self.query_positions[:-1],
                torch.ones(query_count - 1, dtype=self.query_positions.dtype)):
            raise ValueError("the query positions must be consecutive")
        if require_full:
            if (self.all_routed_experts is None or self.all_positions is None
                    or self.all_input_ids is None):
                raise ValueError("full training requires all-token routing tensors")
            if self.all_routed_experts.ndim != 3:
                raise ValueError("all_routed_experts must be [token,layer,top_k]")
            if tuple(self.all_routed_experts.shape[1:]) != (layer_count, top_k):
                raise ValueError("all_routed_experts layer/top-k shape mismatch")
            if tuple(self.all_positions.shape) != (self.all_routed_experts.shape[0],):
                raise ValueError("all_positions does not align with all_routed_experts")
            if tuple(self.all_input_ids.shape) != (self.all_routed_experts.shape[0],):
                raise ValueError("all_input_ids does not align with all_routed_experts")
            if self.all_positions.unique().numel() != self.all_positions.numel():
                raise ValueError("all_positions contains duplicates")
            ordered = self.all_positions.sort().values
            if int(ordered[0]) != 0:
                raise ValueError("all_positions does not start at token zero")
            expected = torch.arange(
                int(ordered[0]), int(ordered[0]) + ordered.numel(),
                dtype=ordered.dtype)
            if not torch.equal(ordered, expected):
                raise ValueError("all_positions is not a contiguous token sequence")
            if not torch.equal(self.query_positions, ordered[-query_count:]):
                raise ValueError("query positions are not the final query_count routed tokens")


def load_query_routing_artifact(path: str | Path) -> QueryRoutingArtifact:
    source = Path(path)
    with safe_open(source, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        required = {"query_hidden", "query_routed_experts", "query_positions"}
        missing = required - keys
        if missing:
            raise ValueError(f"query artifact is missing {sorted(missing)}: {source}")
        tensors = {key: handle.get_tensor(key) for key in _CAPTURE_DTYPES if key in keys}
        return query_routing_artifact_from_tensors(own_query_capture_tensors(tensors))
