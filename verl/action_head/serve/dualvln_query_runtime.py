"""vLLM 0.18 latent-query tail with a zero-work ordinary-request path.

This extension is deliberately independent of ``extract_hidden_states``.  A
marked action expert request is forced to consume one internal placeholder four
times after ``</tool_call>``.  The model embedding for those four occurrences
is replaced by four independently learned vectors and only their final model
outputs are written to host-local storage.

Marked requests reserve query_count + 1 positions inside max_tokens.  At the
text limit, even an unfinished tool call transitions to the query tail; the
extra position lets the last query be forwarded before EOS is sampled.
"""
from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def _ids(name: str) -> tuple[int, ...]:
    raw = os.environ.get(name, "")
    try:
        result = tuple(int(value) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} must be comma-separated token ids") from exc
    if not result:
        raise RuntimeError(f"{name} must not be empty")
    return result


@dataclass
class QueryRequestState:
    request_id: str
    output_ids: list[int]
    query_count: int
    max_text_tokens: int
    forced_completion_ids: tuple[int, ...] = ()
    executable_policy: Any = None


# The logits processor lives in EngineCore while model hooks live in TP worker
# processes.  They deliberately keep separate process-local state.  The custom
# worker supplies the flattened token-row ranges for marked requests before
# each model execution; this avoids relying on a registry that cannot cross the
# process boundary in vLLM V1.
QUERY_STATES: dict[int, QueryRequestState] = {}
WORKER_QUERY_ROWS: list[tuple[int, int, str, int]] = []


def set_worker_query_rows(rows: list[tuple[int, int, str, int]]) -> None:
    """Publish flattened rows plus each request's logical token offset."""
    WORKER_QUERY_ROWS.clear()
    WORKER_QUERY_ROWS.extend(rows)


def _trailing_count(values: list[int], token_id: int, limit: int) -> int:
    count = 0
    for value in reversed(values):
        if value != token_id or count == limit:
            break
        count += 1
    return count


def _contains_subsequence(values: list[int], part: tuple[int, ...]) -> bool:
    if not part or len(part) > len(values):
        return False
    return any(tuple(values[start:start + len(part)]) == part
               for start in range(len(values) - len(part) + 1))


def active_query_rows() -> list[tuple[int, QueryRequestState, int]]:
    placeholder = _ids("ACTION_HEAD_QUERY_TOKEN_ID")[0]
    result = []
    for index, state in sorted(QUERY_STATES.items()):
        phase = _trailing_count(state.output_ids, placeholder, state.query_count)
        if 1 <= phase <= state.query_count:
            result.append((index, state, phase))
    return result


def _capture_request_rows_cpu(
    cache: dict[str, dict[int, torch.Tensor]],
    source: torch.Tensor,
    rows: list[tuple[int, int, str, int]],
    *,
    dtype: torch.dtype,
    bulk: bool,
) -> None:
    """Preserve logical-token indexing with one device transfer per span.

    ``rows`` follows GPUModelRunner's current flattened batch order. Copying
    each contiguous request span before indexing on the CPU avoids a blocking
    device-to-host transfer for every prefix token. The legacy path is kept
    for fixed-input A/B checks; both paths overwrite replayed positions in the
    same way and retain data for earlier prefill/decode spans.
    """
    for start, end, request_id, token_start in rows:
        captured = cache.setdefault(request_id, {})
        if bulk:
            if end > start:
                host_rows = source[start:end].detach().to("cpu", dtype=dtype)
                captured.update((token_start + offset, value)
                                for offset, value in enumerate(host_rows.unbind(0)))
        else:
            for row in range(start, end):
                captured[token_start + row - start] = source[row].detach().to(
                    "cpu", dtype=dtype)


def _capture_query_row(source: torch.Tensor, *, dtype: torch.dtype,
                       defer_host_copy: bool) -> torch.Tensor:
    """Own a query row before vLLM reuses its hidden/routing buffers.

    Deferred rows remain on the producing device until the entire query tail
    is complete. clone() is required even when the dtype already matches:
    both source buffers may be overwritten by the next decode iteration.
    """
    if defer_host_copy:
        return source.detach().to(dtype=dtype).clone()
    return source.detach().to("cpu", dtype=dtype)


def install_query_model_hooks(model_runner: Any, checkpoint: str,
                              cache_root: str) -> None:
    """Install query-only embedding replacement and final-output capture."""
    if getattr(model_runner, "_action_head_query_hook", False):
        return
    model = model_runner.model
    parameter = next(model.parameters())

    def load_learned() -> torch.Tensor:
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        query_count = state.get("query_count")
        if (not isinstance(query_count, int) or not 1 <= query_count <= 64
                or state.get("hidden_convention") != "final_norm"):
            raise RuntimeError("query checkpoint must be 1-64 slot final_norm")
        value = state["query_embeddings"]
        if value.ndim != 2 or value.shape[0] != query_count:
            raise RuntimeError("query checkpoint has invalid embedding shape")
        return value.to(device=parameter.device, dtype=parameter.dtype)

    learned = load_learned()
    query_count = int(learned.shape[0])
    checkpoint_stat = os.stat(checkpoint)
    checkpoint_version = (
        checkpoint_stat.st_mtime_ns, checkpoint_stat.st_size,
        checkpoint_stat.st_ino)
    hot_reload = os.environ.get("ACTION_HEAD_QUERY_HOT_RELOAD", "0") == "1"
    placeholder = _ids("ACTION_HEAD_QUERY_TOKEN_ID")[0]
    output_cache: dict[str, list[torch.Tensor | None]] = {}
    routing_cache: dict[str, list[torch.Tensor | None]] = {}
    position_cache: dict[str, list[torch.Tensor | None]] = {}
    full_routing_cache: dict[str, dict[int, torch.Tensor]] = {}
    full_token_cache: dict[str, dict[int, torch.Tensor]] = {}
    query_phases: dict[str, int] = {}
    pending: list[tuple[str, int, int, int]] = []
    capture_routing = os.environ.get("ACTION_HEAD_CAPTURE_ROUTING", "0") == "1"
    capture_full_routing = os.environ.get(
        "ACTION_HEAD_CAPTURE_FULL_ROUTING", "0") == "1"
    bulk_route_capture = os.environ.get(
        "ACTION_HEAD_BULK_ROUTE_CAPTURE", "0") == "1"
    defer_query_capture = os.environ.get(
        "ACTION_HEAD_DEFER_QUERY_CAPTURE", "0") == "1"
    if capture_full_routing and not capture_routing:
        raise RuntimeError(
            "ACTION_HEAD_CAPTURE_FULL_ROUTING=1 also requires "
            "ACTION_HEAD_CAPTURE_ROUTING=1")
    native_full_routing = False
    if os.environ.get("ACTION_HEAD_R3_CAPTURE_MODE", "legacy") != "legacy":
        from action_head.serve.native_r3_capture import capture_mode
        native_full_routing = capture_mode() == "native"
        if not capture_full_routing or not capture_routing:
            raise RuntimeError("native R3 modes require full-routing capture flags")
        if native_full_routing:
            # Scheduler exports full IDs/routes from its existing native SHM
            # reader. Workers still capture all four query hidden/routes.
            capture_full_routing = False
    original_embed = model.embed_input_ids
    original_forward = model_runner._model_forward

    def reload_query_checkpoint():
        """Explicit direct-sync barrier; ordinary request hot reload is unchanged."""
        nonlocal learned, checkpoint_version
        if not hot_reload or query_phases:
            raise RuntimeError("query publication requires hot reload and no pending query tail")
        before = os.stat(checkpoint)
        version = (before.st_mtime_ns, before.st_size, before.st_ino)
        with open(checkpoint, "rb") as handle:
            checkpoint_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        if version != checkpoint_version:
            value = load_learned()
            if tuple(value.shape) != tuple(learned.shape):
                raise RuntimeError("query publication cannot change query/hidden dimensions")
            learned = value
            checkpoint_version = version
        after = os.stat(checkpoint)
        if version != (after.st_mtime_ns, after.st_size, after.st_ino):
            raise RuntimeError("query checkpoint changed during direct publication")
        cpu = learned.detach().cpu().contiguous()
        return {"checkpoint_sha256": checkpoint_sha,
                "checkpoint_version": list(checkpoint_version),
                "learned_sha256": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
                "shape": list(cpu.shape), "dtype": str(cpu.dtype),
                "pending_query_phases": len(query_phases)}

    model._action_head_reload_query_checkpoint = reload_query_checkpoint

    def embed_input_ids(input_ids, *args, **kwargs):
        nonlocal learned, checkpoint_version
        # The trainer publishes this small file with os.replace between rollout
        # waves. Never change embeddings while a latent-query tail is in flight.
        if hot_reload and not query_phases:
            current_stat = os.stat(checkpoint)
            current_version = (
                current_stat.st_mtime_ns, current_stat.st_size,
                current_stat.st_ino)
            if current_version != checkpoint_version:
                learned = load_learned()
                checkpoint_version = current_version
        embeds = original_embed(input_ids, *args, **kwargs)
        if capture_full_routing:
            _capture_request_rows_cpu(
                full_token_cache, input_ids.reshape(-1), WORKER_QUERY_ROWS,
                dtype=torch.int64, bulk=bulk_route_capture)
        placeholder_rows = torch.nonzero(
            input_ids.reshape(-1) == placeholder).flatten().tolist()
        for row in placeholder_rows:
            request_row = next(((request_id, token_start + row - start)
                               for start, end, request_id, token_start
                               in WORKER_QUERY_ROWS if start <= row < end), None)
            # The reserved token may legally occur in an unmarked ordinary
            # request.  Only marked row ranges are part of the query protocol.
            if request_row is None:
                continue
            request_id, logical_position = request_row
            # vLLM V1 may call embed_input_ids twice for the same scheduled
            # row before one forward.  Treat that as an idempotent replay of
            # the same query slot, not as the next slot.
            replay_phase = next((phase for pending_request, phase, pending_row, _
                                 in pending
                                 if pending_request == request_id and pending_row == row),
                                None)
            phase = replay_phase or (query_phases.get(request_id, 0) + 1)
            if phase > query_count:
                raise RuntimeError(
                    f"request {request_id} produced more than {query_count} query slots")
            if learned.shape[1] != embeds.shape[-1]:
                raise RuntimeError("query embedding hidden size does not match Qwen")
            embeds[row] = learned[phase - 1]
            if replay_phase is None:
                query_phases[request_id] = phase
                pending.append((request_id, phase, row, logical_position))
        return embeds

    def forward(*args, **kwargs):
        hidden = original_forward(*args, **kwargs)
        if pending or (capture_full_routing and WORKER_QUERY_ROWS):
            if not isinstance(hidden, torch.Tensor):
                raise RuntimeError("Qwen vLLM forward did not return final hidden tensor")
            routed = None
            if capture_routing:
                from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
                    RoutedExpertsCapturer,
                )

                capturer = RoutedExpertsCapturer.get_instance()
                routed = None if capturer is None else capturer._device_buffer
                if routed is None:
                    raise RuntimeError(
                        "ACTION_HEAD_CAPTURE_ROUTING=1 requires "
                        "vLLM --enable-return-routed-experts")
            positions = kwargs.get("positions")
            if capture_full_routing:
                if positions is None:
                    raise RuntimeError("full route capture requires vLLM positions")
                _capture_request_rows_cpu(
                    full_routing_cache, routed, WORKER_QUERY_ROWS,
                    dtype=torch.int32, bulk=bulk_route_capture)
            for request_id, phase, row, logical_position in pending:
                slots = output_cache.setdefault(request_id, [None] * query_count)
                slots[phase - 1] = _capture_query_row(
                    hidden[row], dtype=torch.bfloat16,
                    defer_host_copy=defer_query_capture)
                if capture_routing:
                    route_slots = routing_cache.setdefault(
                        request_id, [None] * query_count)
                    position_slots = position_cache.setdefault(
                        request_id, [None] * query_count)
                    route_slots[phase - 1] = _capture_query_row(
                        routed[row], dtype=torch.int32,
                        defer_host_copy=defer_query_capture)
                    position_slots[phase - 1] = torch.tensor(
                        logical_position, dtype=torch.int64)
                if all(value is not None for value in slots):
                    if (not torch.distributed.is_initialized()
                            or torch.distributed.get_rank() == 0):
                        suffix = ".query.safetensors" if native_full_routing else ".safetensors"
                        output = Path(cache_root) / f"{request_id}{suffix}"
                        output.parent.mkdir(parents=True, exist_ok=True)
                        temporary = output.with_suffix(".safetensors.tmp")
                        tensors = {"query_hidden": torch.stack(slots).to("cpu")}
                        if capture_routing:
                            routes = routing_cache[request_id]
                            query_positions = position_cache[request_id]
                            if not all(value is not None for value in routes):
                                raise RuntimeError("incomplete routed-expert capture")
                            if not all(value is not None for value in query_positions):
                                raise RuntimeError("incomplete query-position capture")
                            tensors["query_routed_experts"] = torch.stack(routes).to("cpu")
                            tensors["query_positions"] = torch.stack(query_positions)
                            if capture_full_routing:
                                by_position = full_routing_cache.get(request_id, {})
                                if not by_position:
                                    raise RuntimeError("full routed-expert capture is empty")
                                ordered = sorted(by_position.items())
                                tensors["all_positions"] = torch.tensor(
                                    [item[0] for item in ordered], dtype=torch.int64)
                                tensors["all_routed_experts"] = torch.stack(
                                    [item[1] for item in ordered])
                                token_by_position = full_token_cache.get(request_id, {})
                                if set(token_by_position) != set(by_position):
                                    raise RuntimeError(
                                        "full token ids do not align with captured routes")
                                tensors["all_input_ids"] = torch.stack(
                                    [token_by_position[position]
                                     for position, _ in ordered])
                        # A compiled/CUDA-graph replay can execute the four
                        # query rows again before the HTTP response is handed
                        # back to the rollout worker.  The first artifact owns
                        # the complete prefill (positions start at zero), while
                        # a replay only sees the decode tail.  Never let that
                        # late partial replay replace the complete artifact.
                        final_exists = (native_full_routing and (
                            Path(cache_root) / f"{request_id}.safetensors").exists())
                        if not output.exists() and not final_exists:
                            save_file(tensors, str(temporary))
                            os.replace(temporary, output)
                    output_cache.pop(request_id, None)
                    routing_cache.pop(request_id, None)
                    position_cache.pop(request_id, None)
                    full_routing_cache.pop(request_id, None)
                    full_token_cache.pop(request_id, None)
                    query_phases.pop(request_id, None)
            pending.clear()
        return hidden

    model.embed_input_ids = embed_input_ids
    # Hook the runner boundary instead of the model instance. vLLM may wrap or
    # replace the latter during torch.compile/CUDA-graph preparation, whereas
    # execute_model always calls _model_forward and receives the final hidden
    # tensor here immediately before logits computation.
    model_runner._model_forward = forward
    model_runner._action_head_query_hook = True

    def discard_query(request_id: str) -> None:
        output_cache.pop(request_id, None)
        routing_cache.pop(request_id, None)
        position_cache.pop(request_id, None)
        full_routing_cache.pop(request_id, None)
        full_token_cache.pop(request_id, None)
        query_phases.pop(request_id, None)

    model_runner._action_head_discard_query = discard_query


# vLLM imports are kept below the hook helpers so unit tests can exercise the
# state logic without initializing the accelerator runtime.
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)


class DualVlnQueryLogitsProcessor(LogitsProcessor):
    """Force the internal latent-query tail only for marked requests."""

    @classmethod
    def validate_params(cls, params: SamplingParams):
        extra = params.extra_args or {}
        count = extra.get("action_head_query_count")
        if count is not None and (not isinstance(count, int) or not 1 <= count <= 64):
            raise ValueError("action_head_query_count must be an integer in [1, 64]")
        request_id = extra.get("action_head_query_request_id")
        if count is not None and (not isinstance(request_id, str) or not request_id):
            raise ValueError("marked query request has no request id")
        if count is not None:
            budget = params.max_tokens
            if type(budget) is not int or budget < count + 1:
                raise ValueError("max_tokens must reserve query_count + 1 positions")
        forced = extra.get("action_head_forced_completion_ids")
        if forced is not None:
            if os.environ.get("ACTION_HEAD_ALLOW_FORCED_COMPLETION", "0") != "1":
                raise ValueError("per-request forced completion is diagnostic-only")
            if count is None or not isinstance(forced, list) or not 1 <= len(forced) <= 4096:
                raise ValueError("forced completion requires a marked request and 1-4096 token ids")
            if any(type(value) is not int or value < 0 for value in forced):
                raise ValueError("forced completion ids must be non-negative integers")

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        del device, is_pin_memory
        self.max_model_len = getattr(
            getattr(vllm_config, "model_config", None), "max_model_len", None)
        self.tool_end_ids = _ids("ACTION_HEAD_TOOL_END_TOKEN_IDS")
        self.run_tool_ids = _ids("ACTION_HEAD_RUN_TOOL_TOKEN_IDS")
        self.placeholder_id = _ids("ACTION_HEAD_QUERY_TOKEN_ID")[0]
        self.eos_id = _ids("ACTION_HEAD_QUERY_EOS_TOKEN_ID")[0]
        forced = os.environ.get("ACTION_HEAD_FORCED_COMPLETION_IDS", "")
        self.forced_completion_ids = tuple(
            int(value) for value in forced.split(",") if value.strip())

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        if not batch_update:
            return
        # vLLM may remove and reuse the same batch slot in one update.
        for index in batch_update.removed:
            QUERY_STATES.pop(index, None)
        for index, params, prompt_ids, output_ids in batch_update.added:
            assert params is not None
            self.validate_params(params)
            extra = params.extra_args or {}
            count = extra.get("action_head_query_count")
            if count is not None:
                budget = params.max_tokens
                context_limit = getattr(self, "max_model_len", None)
                if context_limit is not None and prompt_ids is not None:
                    budget = min(budget, context_limit - len(prompt_ids))
                if budget < count + 1:
                    raise ValueError("remaining context cannot fit query_count + 1 positions")
                from action_head_rl.executable_policy import new_sampler_state
                executable_policy = new_sampler_state(extra, budget=budget, query_count=count)
                QUERY_STATES[index] = QueryRequestState(
                    extra["action_head_query_request_id"], output_ids, count,
                    budget - count - 1,
                    tuple(extra.get("action_head_forced_completion_ids", ())),
                    executable_policy=executable_policy)
            else:
                QUERY_STATES.pop(index, None)
        for source, destination, direction in batch_update.moved:
            left = QUERY_STATES.pop(source, None)
            right = QUERY_STATES.pop(destination, None)
            if left is not None:
                QUERY_STATES[destination] = left
            if direction == MoveDirectionality.SWAP and right is not None:
                QUERY_STATES[source] = right

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not QUERY_STATES:
            return logits
        for row, request in QUERY_STATES.items():
            if request.executable_policy is not None:
                from action_head_rl.executable_policy import apply_sampler_state
                apply_sampler_state(request.executable_policy, request.output_ids, logits[row],
                    placeholder=self.placeholder_id, eos=self.eos_id, query_count=request.query_count)
                continue
            trailing = _trailing_count(
                request.output_ids, self.placeholder_id, request.query_count)
            text_limit = len(request.output_ids) >= request.max_text_tokens
            forced = request.forced_completion_ids or self.forced_completion_ids
            # The reserved tail also takes precedence over diagnostic forcing.
            if not trailing and not text_limit and len(request.output_ids) < len(forced):
                target = forced[len(request.output_ids)]
                if target >= logits.shape[-1]:
                    raise ValueError("forced completion token outside model vocabulary")
                value = logits[row, target].clone()
                logits[row] = float("-inf")
                logits[row, target] = value
                continue
            tool_finished = tuple(request.output_ids[-len(self.tool_end_ids):]) \
                == self.tool_end_ids and _contains_subsequence(
                    request.output_ids, self.run_tool_ids)
            if trailing == request.query_count:
                target = self.eos_id
            elif trailing or text_limit or tool_finished:
                target = self.placeholder_id
            else:
                target = None
            if target is None:
                continue
            value = logits[row, target].clone()
            logits[row] = float("-inf")
            logits[row, target] = value
        return logits
