"""Live fast-image tensors, bounded to the current training batch only.

Reuse the online visual encoder's processor and frozen-tower output contract.
No feature cache is persisted. The original fast tower is distinct from the slow tower.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from PIL import Image

LIVE_VISUAL = 'live_frozen_visual_v1'
FAST_INPUTS = {'fast_pixel_values', 'fast_image_grid_thw',
               'fast_reference_image_index', 'fast_current_image_index'}


def load_frozen_fast_tower(backbone, expected_hidden_size, *, device="cpu", dtype=torch.bfloat16):
    """Reuse the original online fast tower, checking its head contract before loading."""
    from transformers import AutoConfig
    from action_head.policy import Qwen35VisualTokenEncoder
    if not backbone:
        raise ValueError("live fast vision requires an explicit ACTION_HEAD_FAST_BACKBONE")
    config = AutoConfig.from_pretrained(backbone, trust_remote_code=True)
    width = int(config.vision_config.out_hidden_size)
    if width != int(expected_hidden_size):
        raise ValueError(f"fast visual width {width} != head visual width {expected_hidden_size}")
    encoder = Qwen35VisualTokenEncoder.from_pretrained(backbone, device=device, dtype=dtype)
    if any(parameter.requires_grad for parameter in encoder.vision_model.parameters()):
        raise ValueError("live fast tower must remain frozen")
    return encoder.vision_model


def verify_image(path, expected):
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError('live fast image must be an existing absolute file')
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('live fast image hash changed: '+str(path))
    return str(path)


def build_fast_inputs(processor, references, currents):
    """Decode each distinct batch image once, preserving window-to-image indices."""
    if not references or len(references) != len(currents):
        raise ValueError('live fast images require equal nonempty reference/current windows')
    paths, indices = [], {}
    def index(path):
        if path not in indices:
            indices[path] = len(paths)
            paths.append(path)
        return indices[path]
    ref = [index(path) for path in references]
    cur = [index(path) for path in currents]
    images = []
    try:
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        values = processor.image_processor(images=images, return_tensors='pt')
    finally:
        for image in images:
            image.close()
    return dict(fast_pixel_values=values['pixel_values'],
                fast_image_grid_thw=values['image_grid_thw'],
                fast_reference_image_index=torch.tensor(ref, dtype=torch.long),
                fast_current_image_index=torch.tensor(cur, dtype=torch.long))


def slice_fast_inputs(payload, selected):
    """Slice and rebase a heterogeneous anchor microbatch without cross-case reuse."""
    reference = payload['fast_reference_image_index'][selected]
    current = payload['fast_current_image_index'][selected]
    if not reference.numel() or reference.shape != current.shape:
        raise ValueError('live microbatch requires nonempty matching image indices')
    image_indices, inverse = torch.unique(torch.cat([reference, current]),
                                          sorted=True, return_inverse=True)
    grid = payload['fast_image_grid_thw']
    if grid.ndim != 2 or grid.shape[1] != 3 or torch.any(grid <= 0):
        raise ValueError('invalid live image grid')
    if torch.any(image_indices < 0) or torch.any(image_indices >= len(grid)):
        raise ValueError('live image index outside batch')
    counts = grid.prod(dim=1).long()
    if int(counts.sum()) != len(payload['fast_pixel_values']):
        raise ValueError('live image grid/pixel count mismatch')
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    pixels = torch.cat([payload['fast_pixel_values'][int(offsets[i]):int(offsets[i+1])]
                        for i in image_indices.tolist()])
    size = len(reference)
    return dict(fast_pixel_values=pixels, fast_image_grid_thw=grid[image_indices],
                fast_reference_image_index=inverse[:size],
                fast_current_image_index=inverse[size:])


@torch.no_grad()
def encode_fast_inputs(vision_model, pixel_values, grid, reference_index, current_index):
    """Run the bound original fast tower inside the root FSDP/DDP forward."""
    from action_head.policy import encode_processed_visual_batch
    parameter = next(vision_model.parameters())
    tokens = encode_processed_visual_batch(vision_model,
        pixel_values.to(device=parameter.device, dtype=parameter.dtype),
        grid.to(device=parameter.device))
    # Match the historical FP32 input conversion after frozen visual inference.
    tokens = [value.detach().float() for value in tokens]
    padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
    lengths = torch.tensor([len(value) for value in tokens], device=padded.device)
    mask = torch.arange(padded.shape[1], device=padded.device)[None] < lengths[:, None]
    reference_index = reference_index.to(device=padded.device, dtype=torch.long)
    current_index = current_index.to(device=padded.device, dtype=torch.long)
    return (padded[reference_index], padded[current_index],
            mask[reference_index], mask[current_index])
