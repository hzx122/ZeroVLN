"""Generated-token hidden-state exporter for vLLM 0.18 hybrid models.

This backports the finish-time export design of the upstream connector: keep
the cache until generation finishes, then export ``all_token_ids[:-1]`` from
the dedicated hidden-state cache group. vLLM 0.18's example connector writes
on every forward with prompt IDs only, so it cannot honor
``include_output_tokens``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import safetensors.torch
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
    extract_from_kv_cache,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class PendingSave:
    req_id: str
    filename: str
    token_ids: torch.Tensor
    block_ids: list[int]


@dataclass
class ActionHeadHiddenStatesMetadata(KVConnectorMetadata):
    pending_saves: list[PendingSave] = field(default_factory=list)


class ActionHeadHiddenStatesConnector(ExampleHiddenStatesConnector, SupportsHMA):
    """HMA connector that exports prompt and generated-token hidden states."""

    @staticmethod
    def _find_hidden_group(kv_cache_config: "KVCacheConfig | None") -> int:
        if kv_cache_config is None:
            return 0
        matches = [
            group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if any(name.startswith("cache_only_layers.") for name in group.layer_names)
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one cache_only hidden-state KV group, found "
                f"{matches}"
            )
        return matches[0]

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self._hidden_group_id = self._find_hidden_group(kv_cache_config)
        if kv_cache_config is not None:
            group = kv_cache_config.kv_cache_groups[self._hidden_group_id]
            self._block_size = group.kv_cache_spec.block_size
        self._pending_saves: dict[str, PendingSave] = {}
        self._kv_cache: torch.Tensor | None = None
        self._is_tp_rank_zero = True

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        super().register_kv_caches(kv_caches)
        self._is_tp_rank_zero = get_tensor_model_parallel_rank() == 0
        self._kv_cache = kv_caches[self.cache_layers[0]]
        if self._kv_cache.shape[1] != self._block_size:
            raise ValueError(
                "Hidden cache block mismatch: configured "
                f"{self._block_size}, tensor has {self._kv_cache.shape[1]}"
            )

    def save_kv_layer(self, *args, **kwargs: Any) -> None:
        # CacheOnlyAttentionLayer already writes each forward into the paged
        # buffer. Export once at request completion so generated states exist.
        return None

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        meta = ActionHeadHiddenStatesMetadata(
            pending_saves=list(self._pending_saves.values())
        )
        self._pending_saves.clear()
        for new_req in scheduler_output.scheduled_new_reqs:
            filename = os.path.join(
                self._storage_path, f"{new_req.req_id}.safetensors"
            )
            self._request_filenames[new_req.req_id] = filename
        return meta

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        filename = self._request_filenames.pop(req_id, None)
        if filename is None:
            filename = os.path.join(self._storage_path, f"{req_id}.safetensors")
        kv_params = getattr(request, "kv_transfer_params", None) or {}
        if not kv_params:
            sampling = getattr(request, "sampling_params", None)
            extra = getattr(sampling, "extra_args", None) or {}
            kv_params = extra.get("kv_transfer_params") or {}
        if kv_params.get("include_output_tokens", False):
            # The final sampled token was never fed through another forward, so
            # no corresponding hidden state exists in the paged cache.
            token_ids = torch.tensor(list(request.all_token_ids)[:-1])
        else:
            token_ids = torch.tensor(request.prompt_token_ids or [])
        self._pending_saves[req_id] = PendingSave(
            req_id=req_id,
            filename=filename,
            token_ids=token_ids,
            block_ids=list(block_ids),
        )
        # Keep blocks alive until the worker exports them in get_finished.
        return True, {"hidden_states_path": filename}

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids[self._hidden_group_id])

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        del finished_req_ids
        if not self.has_connector_metadata():
            return None, None
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, ActionHeadHiddenStatesMetadata):
            return None, None
        done: set[str] = set()
        for pending in metadata.pending_saves:
            if self._is_tp_rank_zero:
                if self._kv_cache is None:
                    raise RuntimeError("hidden-state KV cache was not registered")
                blocks = torch.tensor(pending.block_ids, dtype=torch.long)
                offsets = torch.arange(self._block_size, dtype=torch.long)
                slots = (
                    offsets.reshape(1, self._block_size)
                    + blocks.reshape(-1, 1) * self._block_size
                ).flatten().to(self._kv_cache.device)
                hidden = extract_from_kv_cache(
                    self._kv_cache, slots, pending.token_ids.shape[0]
                )
                os.makedirs(os.path.dirname(pending.filename), exist_ok=True)
                safetensors.torch.save_file(
                    {
                        "hidden_states": hidden.detach().cpu(),
                        "token_ids": pending.token_ids.detach().cpu(),
                    },
                    pending.filename,
                )
            done.add(pending.req_id)
        return done or None, None
