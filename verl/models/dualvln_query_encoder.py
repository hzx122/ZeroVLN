"""DualVLN-style configurable-query conditioner for training and export."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


class DualVlnQueryEncoder(nn.Module):
    """Append learned embeddings and return their final Qwen outputs.

    Four remains the backward-compatible DualVLN default. Scaled checkpoints
    carry their count explicitly so training and serving cannot silently
    disagree about the slow-to-fast latent interface.
    """

    DEFAULT_QUERY_COUNT = 4
    WEIGHTS_NAME = "dualvln_query_embeddings.pt"

    def __init__(self, hidden_size: int, init_std: float = 0.02,
                 query_count: int = DEFAULT_QUERY_COUNT):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not isinstance(query_count, int) or query_count <= 0:
            raise ValueError("query_count must be a positive integer")
        self.hidden_size = int(hidden_size)
        self.query_count = int(query_count)
        self.query_embeddings = nn.Parameter(
            torch.empty(self.query_count, self.hidden_size))
        nn.init.normal_(self.query_embeddings, mean=0.0, std=float(init_std))

    @property
    def QUERY_COUNT(self) -> int:
        """Compatibility accessor for older callers; value is instance-specific."""
        return self.query_count

    def append(self, prefix_embeds: Tensor,
               attention_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if prefix_embeds.ndim != 3 or prefix_embeds.shape[-1] != self.hidden_size:
            raise ValueError(
                f"prefix_embeds must have shape (B,L,{self.hidden_size})")
        batch, length, _ = prefix_embeds.shape
        queries = self.query_embeddings.to(
            device=prefix_embeds.device, dtype=prefix_embeds.dtype)
        queries = queries.unsqueeze(0).expand(batch, -1, -1)
        inputs = torch.cat((prefix_embeds, queries), dim=1)
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch, length), device=prefix_embeds.device, dtype=torch.long)
        if tuple(attention_mask.shape) != (batch, length):
            raise ValueError("attention_mask must match prefix_embeds[:2]")
        query_mask = torch.ones(
            (batch, self.query_count), device=attention_mask.device,
            dtype=attention_mask.dtype)
        return inputs, torch.cat((attention_mask, query_mask), dim=1)

    def forward(self, backbone: nn.Module, *, prefix_embeds: Tensor,
                attention_mask: Tensor | None = None, **model_kwargs: Any) -> Tensor:
        inputs, mask = self.append(prefix_embeds, attention_mask)
        outputs = backbone(
            inputs_embeds=inputs,
            attention_mask=mask,
            output_hidden_states=False,
            return_dict=True,
            **model_kwargs,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            # Causal-LM wrappers expose the final-normalized LM-head input in
            # their model output only when the base model is called. Fail
            # closed instead of silently taking logits or a pre-norm tensor.
            raise RuntimeError("backbone did not return final last_hidden_state")
        result = hidden[:, -self.query_count:, :]
        expected = (prefix_embeds.shape[0], self.query_count, self.hidden_size)
        if tuple(result.shape) != expected:
            raise RuntimeError(
                f"backbone query output must have shape {expected}, got {tuple(result.shape)}")
        return result

    def save_pretrained(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save({
            "query_embeddings": self.query_embeddings.detach().cpu(),
            "query_count": self.query_count,
            "hidden_size": self.hidden_size,
            "hidden_convention": "final_norm",
        }, output / self.WEIGHTS_NAME)

    @classmethod
    def from_pretrained(cls, input_dir: str | Path,
                        map_location: str | torch.device = "cpu"):
        # Accept both the save_pretrained directory and the resolved weights
        # file.  Ray propagates absolute artifact files to workers, while the
        # original SFT entry points pass the containing directory.
        source = Path(input_dir)
        path = source if source.is_file() else source / cls.WEIGHTS_NAME
        try:
            state = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=map_location)
        query_count = state.get("query_count")
        if not isinstance(query_count, int) or query_count <= 0:
            raise ValueError("query checkpoint has no valid query_count")
        if state.get("hidden_convention") != "final_norm":
            raise ValueError("query checkpoint hidden convention is not final_norm")
        embeddings = state.get("query_embeddings")
        expected = (query_count, int(state["hidden_size"]))
        if not torch.is_tensor(embeddings) or tuple(embeddings.shape) != expected:
            raise ValueError(f"query checkpoint embedding shape must be {expected}")
        model = cls(int(state["hidden_size"]), query_count=query_count)
        with torch.no_grad():
            model.query_embeddings.copy_(embeddings)
        return model
