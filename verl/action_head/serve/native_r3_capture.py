"""Opt-in full R3 export from vLLM 0.18's existing KV-slot route reader.

``native_audit`` compares the same request's legacy artifact without replacing
it. ``native`` joins worker query-only tensors with native full routes. Both
modes run at Scheduler._get_routed_experts, before the request's KV blocks are
freed, and currently require synchronous, prefix-cache-disabled execution.
The supported query tails contain four or sixteen forwarded query slots.
No vLLM modules are imported until the opt-in scheduler hook is installed.
"""
from __future__ import annotations

import errno
import importlib.util
import json
import os
import time
from functools import wraps, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from action_head_rl.artifact import QueryRoutingArtifact


CAPTURE_MODE_ENV = "ACTION_HEAD_R3_CAPTURE_MODE"
FULL_KEYS = ("all_input_ids", "all_positions", "all_routed_experts")
QUERY_KEYS = ("query_hidden", "query_routed_experts", "query_positions")


def capture_mode() -> str:
    mode = os.environ.get(CAPTURE_MODE_ENV, "legacy")
    if mode not in ("legacy", "native_audit", "native"):
        raise ValueError(f"invalid {CAPTURE_MODE_ENV}: {mode!r}")
    return mode


def _comparison(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    """Exact CPU tensor comparison with bounded, coordinate-level diagnostics."""
    result: dict[str, Any] = {
        "expected_shape": list(expected.shape), "actual_shape": list(actual.shape),
        "expected_dtype": str(expected.dtype), "actual_dtype": str(actual.dtype),
        "equal": False,
    }
    if expected.device.type != "cpu" or actual.device.type != "cpu":
        raise ValueError("native R3 comparison only accepts CPU tensors")
    if expected.shape != actual.shape:
        return result
    different = expected != actual
    result["different_elements"] = int(different.sum())
    result["equal"] = expected.dtype == actual.dtype and torch.equal(expected, actual)
    result["first_differences"] = [
        {"index": index, "expected": expected[tuple(index)].item(),
         "actual": actual[tuple(index)].item()}
        for index in different.nonzero()[:8].tolist()
    ]
    return result


def _validate_artifact(tensors: dict[str, torch.Tensor], *, full: bool,
                       layer_count: int, top_k: int, query_count: int) -> None:
    hidden = tensors["query_hidden"]
    if hidden.ndim != 2:
        raise ValueError("query_hidden must be a two-dimensional tensor")
    artifact = QueryRoutingArtifact(**{
        key: tensors[key] for key in (*QUERY_KEYS, *FULL_KEYS) if key in tensors})
    artifact.validate(hidden_size=hidden.shape[1], layer_count=layer_count,
                      top_k=top_k, require_full=full, query_count=query_count)


@lru_cache(maxsize=1)
def _control_io():
    # Other training imports can leave an older helper on sys.path. Bind the
    # helper from this exact container checkout, without global Path patches.
    source = Path(__file__).resolve().parents[3] / 'scripts/train/dlc_control_io_retry.py'
    spec = importlib.util.spec_from_file_location('_native_r3_control_io_retry', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=16)
def _audit_spool(audit_root: str):
    # Diagnostics are append-only JSONL on the data disk. The local spool is
    # retained until archived; EngineCore never waits for a CPFS metadata write.
    import atexit
    import socket
    import tempfile
    import threading
    source = Path(__file__).resolve().parents[3] / 'scripts/train/service_log_spool.py'
    spec = importlib.util.spec_from_file_location('_native_r3_log_spool', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = Path(audit_root) / 'spooled' / f'{socket.gethostname()}-{os.getpid()}.jsonl'
    spool = module.LogSpool(destination, Path(tempfile.gettempdir()) / 'native-r3-audit-spool')
    spool.record_lock = threading.Lock()
    print(json.dumps(dict(event='native_r3_audit_spool', **spool.status())), flush=True)
    def finish():
        if not spool.close(timeout=30.):
            print(json.dumps(dict(event='native_r3_audit_retained', **spool.status())), flush=True)
    atexit.register(finish)
    return spool


def _spool_report(path, report):
    spool = _audit_spool(str(Path(path).parent))
    payload = (json.dumps(dict(report_path=str(path), report=report), sort_keys=True) + '\n').encode()
    state = spool.status()
    if max(state['pending_bytes'], state['allocated_bytes']) + len(payload) >= spool.max_bytes:
        raise RuntimeError('native R3 audit spool capacity exhausted; retained local audit: ' + str(spool.path))
    # A complete local audit record is mandatory; failures are not swallowed.
    with spool.record_lock:
        remaining = memoryview(payload)
        while remaining:
            written = spool.handle.write(remaining)
            if not written:
                raise OSError('native R3 audit local write returned zero bytes')
            remaining = remaining[written:]
        os.fsync(spool.handle.fileno())
    spool.wake.set()
    if state['error']:
        now = time.monotonic()
        if now - getattr(spool, '_last_warning', 0.) >= 30.:
            print(json.dumps(dict(event='native_r3_audit_archive_delayed', **state)), flush=True)
            spool._last_warning = now


def _write_report(path: Path, report: dict[str, Any]) -> None:
    if os.environ.get("ACTION_HEAD_R3_AUDIT_SPOOL") == "1":
        return _spool_report(path, report)
    # Match the repository's safetensor/checkpoint atomic-publication convention.
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    retry = _control_io()
    deadline = time.monotonic() + 20.
    def publish():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(payload)
        os.replace(temporary, path)
    payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
    # A brief shared-storage ENOSPC must not immediately kill EngineCore.
    # Keep publication mandatory: persistent errors still fail the request.
    delays = (.5, 1., 2., 4.)
    for attempt in range(len(delays) + 1):
        try:
            with retry.bounded(deadline):
                retry.retry_atomic_json(publish, path)
            return
        except OSError as exc:
            if exc.errno == errno.ENOSPC and attempt < len(delays):
                time.sleep(delays[attempt])
                continue
            detail = {"report_path": str(path), "temporary_path": str(temporary),
                      "attempts": attempt + 1, "errno": exc.errno}
            try:
                disk = os.statvfs(path.parent)
                detail.update(available_bytes=disk.f_bavail * disk.f_frsize,
                              available_inodes=disk.f_favail)
            except OSError as probe_error:
                detail["statvfs_error"] = str(probe_error)
            exc.add_note("native R3 audit write failed: " + json.dumps(detail))
            raise


class NativeR3Mismatch(RuntimeError):
    """The native and worker records describe different token-level behavior."""


def process_native_r3(
    request: Any, routed_experts: np.ndarray | None, *, mode: str,
    cache_root: Path, audit_root: Path, query_token_id: int, eos_token_id: int,
) -> dict[str, Any] | None:
    """Audit or publish one stopped request; never mutate the native route array.

    The caller must invoke vLLM's original _get_routed_experts first. Its return
    is a CPU copy of all request slots, including any previously computed tokens.
    Requests that never generated an action expert query tail need no artifact.
    """
    if mode == "legacy":
        return None
    if mode not in ("native_audit", "native"):
        raise ValueError(f"invalid native R3 mode: {mode!r}")
    extra = getattr(request.sampling_params, "extra_args", None) or {}
    if "action_head_query_count" not in extra:
        return None
    record_decision = extra.get("action_head_record_decision", False)
    if record_decision and mode != "native":
        raise ValueError("v6 decision capture requires native R3 mode")
    external_id = extra.get("action_head_query_request_id")
    if (not isinstance(external_id, str) or not external_id
            or Path(external_id).name != external_id or external_id in (".", "..")):
        raise ValueError("native R3 requires a safe external query request id")
    final = cache_root / f"{external_id}.safetensors"
    query_file = cache_root / f"{external_id}.query.safetensors"
    report_path = audit_root / f"{external_id}.json"
    report: dict[str, Any] = {
        "mode": mode, "request_id": request.request_id,
        "query_request_id": external_id,
        "request_num_tokens": int(request.num_tokens),
        "num_cached_tokens": int(getattr(request, "num_cached_tokens", 0)),
        "output_artifact": str(final), "report_path": str(report_path),
        "published": False,
    }
    try:
        count = extra["action_head_query_count"]
        if type(count) is not int or count not in (4, 16):
            raise ValueError("native full R3 requires four or sixteen query slots")
        all_ids = list(request.all_token_ids)
        output_ids = list(request.output_token_ids)
        finished_status = getattr(getattr(request, "status", None), "name", None)
        finish_reason = request.get_finished_reason() if callable(
            getattr(request, "get_finished_reason", None)) else None
        report.update(
            output_token_count=len(output_ids), output_token_tail=output_ids[-18:],
            last_output_token_id=output_ids[-1] if output_ids else None,
            prompt_token_count=len(all_ids) - len(output_ids),
            forwarded_token_count=int(request.num_tokens) - 1,
            request_status=finished_status, finish_reason=str(finish_reason) if finish_reason is not None else None,
            stop_reason=getattr(request, "stop_reason", None),
            num_nans_in_logits=getattr(request, "num_nans_in_logits", None),
            native_route_shape=list(routed_experts.shape) if isinstance(routed_experts, np.ndarray) else None)
        if len(all_ids) != request.num_tokens or not all_ids:
            raise ValueError("scheduler all_token_ids does not match num_tokens")
        # Only sampled output tokens can start the query protocol. Prompt text
        # is allowed to contain the reserved carrier token in ordinary content.
        if (not output_ids or len(output_ids) > len(all_ids)
                or all_ids[-len(output_ids):] != output_ids):
            raise ValueError("scheduler output tokens do not match all_token_ids")
        if record_decision and finished_status is not None and finished_status not in (
                "FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED"):
            raise ValueError("text decision capture requires a normally completed request")
        expected_tail = [query_token_id] * count + [eos_token_id]
        has_query_tail = output_ids[-len(expected_tail):] == expected_tail
        report["has_query_tail"] = has_query_tail
        if not has_query_tail:
            if (query_token_id in output_ids[-len(expected_tail):]
                    or final.exists() or query_file.exists()):
                raise ValueError("stopped request has an incomplete or invalid query tail")
            if not record_decision:
                report.update(status="skipped", reason="no_query_tail")
                _write_report(report_path, report)
                return report
        prefix_cache = os.environ.get("ACTION_HEAD_V8_PREFIX_CACHE", "0") == "1"
        if prefix_cache:
            from action_head.serve.continuous_context import validate_cached_routes
            report.update(validate_cached_routes(request, routed_experts, cache_root))
        elif report["num_cached_tokens"] > 0:
            raise ValueError("native R3 gate currently requires prefix caching off")
        if getattr(request, "num_external_computed_tokens", 0):
            raise ValueError("native R3 gate does not support external KV transfer")
        token_count = int(request.num_tokens) - 1
        if (not isinstance(routed_experts, np.ndarray)
                or routed_experts.dtype != np.int32
                or routed_experts.ndim != 3
                or routed_experts.shape[0] != token_count
                or min(routed_experts.shape[1:]) <= 0):
            raise ValueError("native routes must be int32 [num_tokens-1,layer,top_k]")
        native = {
            "all_input_ids": torch.tensor(all_ids[:token_count], dtype=torch.int64),
            "all_positions": torch.arange(token_count, dtype=torch.int64),
            "all_routed_experts": torch.from_numpy(routed_experts),
        }
        if prefix_cache:
            from action_head.serve.continuous_context import CACHE_KEY
            native[CACHE_KEY] = torch.tensor(list(extra[CACHE_KEY].encode("ascii")), dtype=torch.uint8)
        decision = {}
        if record_decision:
            from action_head_rl.decision_records import decision_capture_tensors
            if extra.get("action_head_forced_completion_ids") or os.environ.get("ACTION_HEAD_FORCED_COMPLETION_IDS"):
                raise ValueError("forced diagnostic answers cannot train the text policy")
            decision = decision_capture_tensors(
                all_ids=all_ids, output_ids=output_ids, token_count=token_count,
                query_count=count, has_query_tail=has_query_tail,
                temperature=float(request.sampling_params.temperature),
                finished_status=finished_status)
            from action_head_rl.executable_policy import capture_policy_tensors
            decision.update(capture_policy_tensors(extra, all_ids=all_ids,
                prompt=int(decision["decision_prompt_length"].item()),
                end=int(decision["decision_response_end"].item())))
            if "decision_single_token_target_id" in decision:
                from action_head_rl.decision_records import SINGLE_TOKEN_TARGET_MODE
                report.update(text_target_mode=SINGLE_TOKEN_TARGET_MODE,
                              target_token_id=int(decision["decision_single_token_target_id"].item()),
                              predictor_position=token_count - 1)
            if not has_query_tail:
                # Text-only decisions can be the first request in a fresh
                # service. No query worker has created the cache directory yet.
                final.parent.mkdir(parents=True, exist_ok=True)
                temporary = final.with_suffix(".safetensors.native.tmp")
                save_file({**native, **decision}, str(temporary))
                os.replace(temporary, final)
                report.update(status="pass", published=True, decision_only=True,
                              token_count=token_count)
                _write_report(report_path, report)
                return report
        source = final if mode == "native_audit" else (
            query_file if query_file.exists() else final)
        report.update(source_artifact=str(source), token_count=token_count,
                      native_route_shape=list(routed_experts.shape))
        tensors = load_file(str(source), device="cpu")
        _validate_artifact(tensors, full=mode == "native_audit",
                           layer_count=routed_experts.shape[1],
                           top_k=routed_experts.shape[2], query_count=count)
        comparisons = {
            "query_positions": _comparison(
                native["all_positions"][-count:], tensors["query_positions"]),
            "query_routed_experts": _comparison(
                native["all_routed_experts"][-count:], tensors["query_routed_experts"]),
        }
        if mode == "native_audit":
            comparisons.update({key: _comparison(tensors[key], native[key])
                                for key in FULL_KEYS})
        report["comparisons"] = comparisons
        if not all(value["equal"] for value in comparisons.values()):
            raise NativeR3Mismatch("native routes/token IDs differ from worker artifact")
        if mode == "native":
            complete = {key: tensors[key] for key in QUERY_KEYS}
            complete.update(native)
            complete.update(decision)
            _validate_artifact(complete, full=True,
                               layer_count=routed_experts.shape[1],
                               top_k=routed_experts.shape[2], query_count=count)
            if final.exists():
                previous = load_file(str(final), device="cpu")
                replay = {key: _comparison(previous[key], value)
                          for key, value in complete.items()}
                report["existing_artifact_comparisons"] = replay
                if not all(value["equal"] for value in replay.values()):
                    raise NativeR3Mismatch("existing full artifact differs on replay")
                report["already_published"] = True
            else:
                temporary = final.with_suffix(".safetensors.native.tmp")
                save_file(complete, str(temporary))
                os.replace(temporary, final)
            report["published"] = True
            query_file.unlink(missing_ok=True)
        report["status"] = "pass"
    except Exception as exc:
        report.update(status="mismatch" if isinstance(exc, NativeR3Mismatch) else "error",
                      error=f"{type(exc).__name__}: {exc}")
        try:
            _write_report(report_path, report)
        except OSError as audit_error:
            # Failure reporting must never replace the original routing error.
            exc.add_note(f"audit failure while recording original error: {audit_error!r}")
            for note in getattr(audit_error, "__notes__", []):
                exc.add_note(note)
        raise RuntimeError(f"native R3 {mode} failed; report: {report_path}") from exc
    _write_report(report_path, report)
    return report


def _validate_config(config: Any) -> None:
    prefix_cache = os.environ.get("ACTION_HEAD_V8_PREFIX_CACHE", "0") == "1"
    if prefix_cache and (capture_mode() != "native"
            or os.environ.get("ACTION_HEAD_CONTINUOUS_CONTEXT") != "1"
            or not config.cache_config.enable_prefix_caching
            or config.cache_config.mamba_cache_mode != "align"):
        raise ValueError("v8 cached R3 requires native continuous context and align prefix caching")
    if config.cache_config.enable_prefix_caching and not prefix_cache:
        raise ValueError("native R3 gate requires prefix caching off")
    if not config.model_config.enable_return_routed_experts:
        raise ValueError("native R3 requires --enable-return-routed-experts")
    if config.scheduler_config.async_scheduling or config.speculative_config is not None:
        raise ValueError("native R3 gate requires synchronous, non-speculative decoding")
    if getattr(config, "kv_transfer_config", None) is not None:
        raise ValueError("native R3 gate does not support KV connectors")
    parallel = config.parallel_config
    for name in ("data_parallel_size", "pipeline_parallel_size",
                 "decode_context_parallel_size", "prefill_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"native R3 gate requires {name}=1")


def _validate_route_capacity(scheduler: Any) -> None:
    """Check actual HMA specs instead of assuming the bootstrap guard is enough."""
    config = scheduler.kv_cache_config
    group = config.kv_cache_groups[scheduler.routed_experts_attn_gid]
    required = int(config.num_blocks) * int(group.kv_cache_spec.block_size)
    host = scheduler.routed_experts_reader._host_buffer_view
    if host is None or host.shape[0] < required:
        available = 0 if host is None else host.shape[0]
        raise ValueError(
            f"native route SHM too small: {available} slots < {required} required")
    print(f"[action-head] native R3 SHM slots={host.shape[0]} "
          f"required={required}", flush=True)


def install_native_r3_scheduler_hook(scheduler_class=None) -> bool:
    """Patch only an explicitly enabled process; legacy does no imports/patching."""
    mode = capture_mode()
    if mode == "legacy":
        return False
    if (os.environ.get("ACTION_HEAD_CAPTURE_ROUTING") != "1"
            or os.environ.get("ACTION_HEAD_CAPTURE_FULL_ROUTING") != "1"):
        raise ValueError("native R3 modes require routing and full-routing capture flags")
    cache_root = Path(os.environ["ACTION_HEAD_QUERY_CACHE"])
    if os.environ.get("ACTION_HEAD_V8_PREFIX_CACHE") == "1":
        from action_head.serve.continuous_context import install_parent_cache_limit_hook
        install_parent_cache_limit_hook()
    audit_root = Path(os.environ.get("ACTION_HEAD_R3_AUDIT_DIR")
                      or str(cache_root / "native_r3_audit"))
    # Reuse the worker's existing token-ID parser and contract.
    from action_head.serve.dualvln_query_runtime import _ids
    query_ids = _ids("ACTION_HEAD_QUERY_TOKEN_ID")
    eos_ids = _ids("ACTION_HEAD_QUERY_EOS_TOKEN_ID")
    if len(query_ids) != 1 or len(eos_ids) != 1 or query_ids == eos_ids:
        raise ValueError("native R3 requires distinct single query/EOS token IDs")
    if scheduler_class is None:
        from vllm.v1.core.sched.scheduler import Scheduler
        scheduler_class = Scheduler
    original = scheduler_class._get_routed_experts
    installed = getattr(original, "_action_head_native_r3_mode", None)
    if installed:
        if installed != mode:
            raise RuntimeError("cannot change native R3 mode in a running scheduler")
        return False
    original_init = scheduler_class.__init__

    @wraps(original_init)
    def initialize(self, vllm_config, *args, **kwargs):
        _validate_config(vllm_config)
        result = original_init(self, vllm_config, *args, **kwargs)
        _validate_route_capacity(self)
        return result

    @wraps(original)
    def get_routed_experts(self, request):
        routes = original(self, request)
        process_native_r3(
            request, routes, mode=mode, cache_root=cache_root, audit_root=audit_root,
            query_token_id=query_ids[0], eos_token_id=eos_ids[0])
        return routes

    get_routed_experts._action_head_native_r3_mode = mode
    scheduler_class.__init__ = initialize
    scheduler_class._get_routed_experts = get_routed_experts
    print(f"[action-head] native R3 mode={mode}, audit_dir={audit_root}", flush=True)
    return True
