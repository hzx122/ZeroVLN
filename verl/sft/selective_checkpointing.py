"""Bound activation retention without changing Qwen's forward math."""


def shared_token_budget(tokens, device):
    """Keep checkpoint/recompute policy identical across FSDP ranks."""
    import torch
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        value = torch.tensor(tokens, device=device, dtype=torch.int64)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return int(value.item())
    return tokens


class SelectiveCheckpointing:
    def __init__(self, layers, skip_every: int, max_tokens: int):
        if skip_every not in (2, 4) or max_tokens <= 0:
            raise ValueError('selective checkpointing requires skip_every=2/4 and max_tokens>0')
        self.layers = list(layers)
        self.original = [layer.gradient_checkpointing for layer in self.layers]
        if not self.layers or not all(self.original):
            raise ValueError('selective checkpointing requires all text layers checkpointed initially')
        self.skip_every, self.max_tokens = skip_every, max_tokens

    def apply(self, *, tokens: int, training: bool) -> int:
        if tokens <= 0:
            raise ValueError('token count must be positive')
        retain = training and tokens <= self.max_tokens
        skipped = 0
        for index, (layer, original) in enumerate(zip(self.layers, self.original)):
            skip = retain and index % self.skip_every == 0
            layer.gradient_checkpointing = original and not skip
            skipped += int(skip)
        return skipped
