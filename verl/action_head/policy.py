"""Native PyTorch policy adapter for the fast/slow action expert model."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import torch

from models.qwen36_fast_slow_action_head import Qwen36FastSlowActionHead

from .query_provider import VllmQueryHidden
from .runtime import OptionContext


class VisualTokenEncoder(Protocol):
    def __call__(self, rgb: np.ndarray) -> Any:
        """Return tokens or ``(tokens, mask)`` for one RGB observation."""


class Qwen35VisualTokenEncoder:
    """Load only Qwen3.5/3.6's frozen vision tower from a sharded checkpoint."""

    def __init__(self, processor: Any, vision_model: torch.nn.Module,
                 device: str | torch.device, dtype: torch.dtype):
        self.processor = processor
        self.device = torch.device(device)
        self.dtype = dtype
        self.vision_model = vision_model.to(
            device=self.device, dtype=dtype).eval()
        for parameter in self.vision_model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, *,
                        device: str | torch.device = "cpu",
                        dtype: torch.dtype = torch.float32):
        """Instantiate the visual tower without duplicating the language model."""
        import json
        from safetensors import safe_open
        from transformers import (
            AutoConfig,
            AutoProcessor,
            Qwen3_5MoeVisionModel,
            Qwen3_5VisionModel,
        )

        root = Path(model_dir)
        config = AutoConfig.from_pretrained(root, trust_remote_code=True)
        model_type = str(config.model_type)
        if model_type == "qwen3_5":
            vision_model = Qwen3_5VisionModel(config.vision_config)
        elif model_type == "qwen3_5_moe":
            vision_model = Qwen3_5MoeVisionModel(config.vision_config)
        else:
            raise ValueError(f"unsupported Qwen visual model type: {model_type}")

        index_path = root / "model.safetensors.index.json"
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        prefix = "model.visual."
        by_shard: dict[str, list[str]] = {}
        for full_name, shard in weight_map.items():
            if full_name.startswith(prefix):
                by_shard.setdefault(shard, []).append(full_name)
        if not by_shard:
            raise ValueError(f"checkpoint has no {prefix} weights: {root}")
        loaded = set()
        for shard, names in sorted(by_shard.items()):
            with safe_open(root / shard, framework="pt", device="cpu") as handle:
                state = {name[len(prefix):]: handle.get_tensor(name) for name in names}
            unexpected = vision_model.load_state_dict(state, strict=False).unexpected_keys
            if unexpected:
                raise ValueError(f"unexpected visual checkpoint keys: {unexpected[:5]}")
            loaded.update(state)
        missing = sorted(set(vision_model.state_dict()) - loaded)
        if missing:
            raise ValueError(f"missing visual checkpoint keys: {missing[:5]}")
        processor = AutoProcessor.from_pretrained(root, trust_remote_code=True)
        return cls(processor, vision_model, device, dtype)

    @torch.no_grad()
    def encode_batch(self, rgbs):
        """Encode multiple RGB frames in one vision-tower forward."""
        from PIL import Image

        images = [Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
                  for rgb in rgbs]
        if not images:
            return []
        inputs = self.processor.image_processor(
            images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            device=self.device, dtype=self.dtype)
        grid = inputs["image_grid_thw"].to(device=self.device)
        return encode_processed_visual_batch(self.vision_model, pixel_values, grid)

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray):
        return self.encode_batch([rgb])[0]


@torch.no_grad()
def encode_processed_visual_batch(vision_model, pixel_values, grid):
    """Shared online/live-training tower contract; no resize or token alteration."""
    output = vision_model(pixel_values, grid_thw=grid, return_dict=True)
    tokens = output.pooler_output
    if tokens.ndim != 2:
        raise ValueError(f"Qwen vision tower returned invalid tokens: {tuple(tokens.shape)}")
    merge = int(vision_model.config.spatial_merge_size)
    token_counts = (grid.prod(dim=1) // (merge * merge)).tolist()
    if sum(token_counts) != int(tokens.shape[0]):
        raise ValueError("Qwen vision tower token/grid mismatch: "
                         f"tokens={tokens.shape[0]} split={token_counts}")
    return list(tokens.split([int(value) for value in token_counts], dim=0))


@dataclass(frozen=True)
class TrajectoryTransform:
    """Explicit checkpoint-space to real metre/radian conversion."""

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_quantiles(cls, q01, q99, *, flip_y: bool = False,
                       flip_yaw: bool = False):
        low = np.asarray(q01, dtype=np.float64)
        high = np.asarray(q99, dtype=np.float64)
        if low.shape != (3,) or high.shape != (3,) or np.any(high <= low):
            raise ValueError("trajectory quantiles must be increasing 3-vectors")
        scale = (high - low) * 0.5
        offset = (high + low) * 0.5
        if flip_y:
            scale[1], offset[1] = -scale[1], -offset[1]
        if flip_yaw:
            scale[2], offset[2] = -scale[2], -offset[2]
        return cls(tuple(scale.tolist()), tuple(offset.tolist()))

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        scale = value.new_tensor(self.scale).view(1, 1, 3)
        offset = value.new_tensor(self.offset).view(1, 1, 3)
        return value * scale + offset


class TorchActionHeadPolicy:
    """Convert a runtime OptionContext into ``model.sample`` inputs.

    The slow condition and reference RGB are cached for the option lifetime.
    The current RGB is encoded on every call, which makes fast replanning use
    the observation returned by the preceding prefix execution.
    """

    def __init__(self, model: Qwen36FastSlowActionHead,
                 visual_encoder: VisualTokenEncoder,
                 *, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32,
                 trajectory_transform: TrajectoryTransform | None = None,
                 seed: int | None = None,
                 deduplicate_first_visual: bool = False):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = model.to(device=self.device, dtype=dtype).eval()
        self.visual_encoder = visual_encoder
        self.trajectory_transform = trajectory_transform or TrajectoryTransform()
        self.seed = seed
        self.deduplicate_first_visual = deduplicate_first_visual
        self.visual_frames_encoded = 0
        self.first_visual_frames_reused = 0
        if model.config.action_dim != 3:
            raise ValueError("runtime requires Action Head action_dim=3")

    @classmethod
    def from_pretrained(cls, checkpoint: str | Path,
                        visual_encoder: VisualTokenEncoder, **kwargs):
        model = Qwen36FastSlowActionHead.from_pretrained(
            checkpoint, map_location="cpu")
        return cls(model, visual_encoder, **kwargs)

    def _tensor(self, value: Any, name: str, hidden_size: int) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[0] != 1 \
                or tensor.shape[-1] != hidden_size:
            raise ValueError(
                f"{name} must have shape (L,{hidden_size}) or (1,L,{hidden_size}), "
                f"got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")
        # Qwen vision pooler outputs can be non-contiguous.  The PPU linear
        # kernel aborts the process for some such strides instead of raising a
        # Python error, so normalize the layout at this boundary.
        return tensor.contiguous()

    @staticmethod
    def _rgb(observation: Any) -> np.ndarray:
        value = observation.get("rgb") if isinstance(observation, Mapping) \
            else observation
        rgb = np.asarray(value)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.size == 0:
            raise ValueError(f"observation RGB must have shape (H,W,3), got {rgb.shape}")
        return rgb

    def _visual(self, observation: Any) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.visual_encoder(self._rgb(observation))
        tokens, mask = encoded if isinstance(encoded, tuple) else (encoded, None)
        tokens = self._tensor(tokens, "visual tokens", self.model.config.visual_hidden_size)
        if mask is None:
            mask = torch.ones(tokens.shape[:2], device=self.device, dtype=torch.bool)
        else:
            mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if tuple(mask.shape) != tuple(tokens.shape[:2]):
                raise ValueError("visual mask does not match visual tokens")
        return tokens, mask

    @torch.no_grad()
    def predict(self, option: OptionContext) -> np.ndarray:
        return self.predict_batch([option])[0]

    def _slow_condition_inputs(self, options):
        bundles = [option.slow_condition for option in options]
        if any(not isinstance(bundle, VllmQueryHidden) for bundle in bundles):
            raise TypeError("slow_condition must be VllmQueryHidden")
        cfg = self.model.config
        queries = [self._tensor(bundle.query_hidden, "query hidden",
                                cfg.qwen_hidden_size) for bundle in bundles]
        if any(value.shape[1] != cfg.query_num_queries for value in queries):
            raise ValueError("query hidden slot count does not match checkpoint config")
        return {"query_hidden_states": torch.cat(queries, dim=0)}

    @torch.no_grad()
    def predict_batch(self, options: list[OptionContext]) -> list[np.ndarray]:
        """Run one visual/model forward for independent online options."""
        if not options:
            return []
        started = time.perf_counter()
        bundles = [option.slow_condition for option in options]
        cfg = self.model.config
        slow_inputs = self._slow_condition_inputs(options)
        reference_rgbs = [self._rgb(option.slow_reference_observation)
                          for option in options]
        current_rgbs = [self._rgb(option.current_observation)
                        for option in options]
        missing = [index for index, option in enumerate(options)
                   if option.policy_cache.get("reference_visual") is None]
        reused = {index for index in missing
                  if self.deduplicate_first_visual
                  and np.array_equal(reference_rgbs[index], current_rgbs[index])}
        current_indices = [index for index in range(len(options)) if index not in reused]
        images = ([reference_rgbs[index] for index in missing]
                  + [current_rgbs[index] for index in current_indices])
        encoder = getattr(self.visual_encoder, "encode_batch", None)
        encoded = encoder(images) if encoder is not None \
            else [self.visual_encoder(image) for image in images]
        if len(encoded) != len(images):
            raise RuntimeError("visual encoder returned the wrong number of frames")
        self.visual_frames_encoded += len(images)
        self.first_visual_frames_reused += len(reused)

        def normalized(value):
            tokens, mask = value if isinstance(value, tuple) else (value, None)
            tokens = self._tensor(tokens, "visual tokens", cfg.visual_hidden_size)
            if mask is None:
                mask = torch.ones(tokens.shape[:2], device=self.device,
                                  dtype=torch.bool)
            else:
                mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
                if mask.ndim == 1:
                    mask = mask.unsqueeze(0)
            return tokens, mask

        for encoded_index, option_index in enumerate(missing):
            value = normalized(encoded[encoded_index])
            options[option_index].policy_cache["reference_visual"] = value
            options[option_index].policy_cache["reference_rgb_sha256"] = \
                hashlib.sha256(reference_rgbs[option_index].tobytes()).hexdigest()
        references = [option.policy_cache["reference_visual"] for option in options]
        currents = list(references)
        for index, value in zip(current_indices, encoded[len(missing):]):
            currents[index] = normalized(value)

        def pad(values):
            width = max(int(tokens.shape[1]) for tokens, _ in values)
            result = torch.zeros((len(values), width, values[0][0].shape[-1]),
                                 device=self.device, dtype=self.dtype)
            mask = torch.zeros((len(values), width), device=self.device,
                               dtype=torch.bool)
            for index, (tokens, token_mask) in enumerate(values):
                length = int(tokens.shape[1])
                result[index, :length] = tokens[0]
                mask[index, :length] = token_mask[0]
            return result, mask

        reference_tokens, reference_mask = pad(references)
        current_tokens, current_mask = pad(currents)
        initial_noise = None
        if self.seed is not None:
            noises = []
            for option in options:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(self.seed + option.chunks)
                noises.append(torch.randn(
                    (1, cfg.horizon, cfg.action_dim), device=self.device,
                    dtype=self.dtype, generator=generator))
            initial_noise = torch.cat(noises, dim=0)
        trajectory = self.model.sample(
            **slow_inputs,
            reference_visual_tokens=reference_tokens,
            reference_visual_mask=reference_mask,
            current_visual_tokens=current_tokens,
            current_visual_mask=current_mask,
            initial_noise=initial_noise,
        )
        trajectory = self.trajectory_transform.apply(trajectory)
        output = trajectory.float().cpu().numpy()
        for index, option in enumerate(options):
            option.internal_log.append({
                "event": "policy_predict", "chunk": option.chunks + 1,
                "request_id": bundles[index].request_id,
                "reference_rgb_sha256": option.policy_cache["reference_rgb_sha256"],
                "current_rgb_sha256": hashlib.sha256(
                    current_rgbs[index].tobytes()).hexdigest(),
                "inference_batch_size": len(options),
                "policy_wall_ms": (time.perf_counter() - started) * 1000,
            })
        return [value for value in output]


class DynamicBatchingPolicy:
    """Thread-safe online batching facade for multiple rollout actors."""

    def __init__(self, policy: TorchActionHeadPolicy, *, max_batch_size: int = 8,
                 batch_wait_ms: float = 8.0):
        if max_batch_size <= 0 or batch_wait_ms < 0:
            raise ValueError("invalid dynamic batching settings")
        self.policy = policy
        self.model = policy.model
        self.max_batch_size = max_batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self._queue = queue.Queue()
        self._stats_lock = threading.Lock()
        self._closed = False
        self._batches = []
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="action-head-dynamic-batcher")
        self._thread.start()

    def predict(self, option: OptionContext) -> np.ndarray:
        event = threading.Event()
        item = {"option": option, "event": event, "queued": time.monotonic()}
        self._queue.put(item)
        event.wait()
        if "error" in item:
            raise item["error"]
        return item["result"]

    def _run(self):
        while True:
            first = self._queue.get()
            if first is None:
                return
            items = [first]
            deadline = time.monotonic() + self.batch_wait_s
            while len(items) < self.max_batch_size:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if item is None:
                    self._queue.put(None)
                    break
                items.append(item)
            started = time.monotonic()
            try:
                results = self.policy.predict_batch(
                    [item["option"] for item in items])
                for item, result in zip(items, results):
                    item["result"] = result
            except Exception as exc:
                for item in items:
                    item["error"] = exc
            finished = time.monotonic()
            with self._stats_lock:
                self._batches.append({
                    "size": len(items), "forward_s": finished - started,
                    "oldest_queue_s": started - items[0]["queued"],
                })
            for item in items:
                item["event"].set()

    def stats(self) -> dict:
        with self._stats_lock:
            batches = list(self._batches)
        requests = sum(row["size"] for row in batches)
        return {
            "requests": requests, "batches": len(batches),
            "mean_batch_size": requests / len(batches) if batches else 0.0,
            "max_batch_size": max((row["size"] for row in batches), default=0),
            "forward_s": sum(row["forward_s"] for row in batches),
            "mean_forward_s": (sum(row["forward_s"] for row in batches) /
                               len(batches) if batches else 0.0),
            "visual_frames_encoded": getattr(self.policy, "visual_frames_encoded", 0),
            "first_visual_frames_reused": getattr(self.policy, "first_visual_frames_reused", 0),
        }

    def close(self):
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._thread.join(timeout=30)


class PolicyProxy:
    """Per-actor trace surface over one shared dynamically batched policy."""

    def __init__(self, policy):
        self._policy = policy

    def predict(self, option):
        return self._policy.predict(option)

    def __getattr__(self, name):
        return getattr(self._policy, name)
