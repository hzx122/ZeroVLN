"""Budget an already encoded chat request without changing its image window.

Only complete old assistant/feedback groups can be removed. Token counts come
from the serving model's /tokenize endpoint, using the actual request messages
and chat template options; character counts are never used as a fallback.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import time
import urllib.error
import urllib.request
import uuid
from .serving_error import TRANSPORT_ERRORS, serving_error


POLICY_VERSION = "paired_text_history_budget_v2_mm_cache_isolation"
SAFETY_TOKENS = 64
MAX_TOKENIZE_CALLS = 12


def context_budget_policy():
    return dict(version=POLICY_VERSION, tokenizer="serving_model_tokenize",
                safety_tokens=SAFETY_TOKENS, max_tokenize_calls=MAX_TOKENIZE_CALLS,
                completion_budget="unchanged_request_max_tokens",
                image_cache="separate_generate_and_per_probe_tokenize_uuids",
                retain="system_initial_task_all_retained_images_latest_feedback",
                remove="oldest_complete_image_free_assistant_feedback_groups")


class ContextBudgetExhausted(RuntimeError):
    episode_abort_reason = "context_budget_exhausted"

    def __init__(self, detail, audit=None):
        super().__init__(detail)
        self.episode_abort_detail = detail
        self.audit = audit or {}


class ContextBudgetServerRejected(ContextBudgetExhausted):
    episode_abort_reason = "context_budget_server_rejection"


def is_context_overflow(status, detail):
    if status != 400:
        return False
    text = detail.lower()
    return any(mark in text for mark in (
        "maximum context length", "context_length_exceeded",
        "longer than the maximum model length", "exceeds the maximum model length",
        "reduce the length of the messages"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _images(message):
    content = message.get("content")
    return sum(part.get("type") in ("image", "image_url")
               for part in content if isinstance(part, dict)) if isinstance(content, list) else 0


def _isolate_image_cache(payload, namespace, *, nonce=None):
    """Assign transport-only IDs on an owned copy, retaining every image URL.

    vLLM 0.18 /tokenize updates the shared API-side multimodal sender cache
    without sending features to EngineCore's receiver cache. Reusing those IDs
    in generation can therefore produce a receiver-cache miss and zero tokens.
    Generation IDs are assigned once before budgeting so all candidates share
    the exact audited request. By default each tokenize probe gets a fresh
    namespace; an explicit nonce can share tokenizer IDs within one budget
    transaction while keeping them disjoint from every generation request.
    """
    nonce = nonce or uuid.uuid4().hex
    image_index = 0
    for message in payload["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image", "image_url"):
                part["uuid"] = f"{namespace}:{nonce}:{image_index}"
                image_index += 1


def removable_message_groups(messages):
    """Return old complete turns; ambiguous structure is retained conservatively."""
    starts = [i for i, message in enumerate(messages) if message.get("role") == "assistant"]
    removable = []
    for start, end in zip(starts, starts[1:]):
        group = messages[start:end]
        # Prefix (system + original task) and the newest turn are never visited.
        # Tool roles must match the preceding assistant's complete call list.
        if len(group) < 2 or any(_images(message) for message in group):
            continue
        if any(message.get("role") not in ("user", "tool") for message in group[1:]):
            continue
        calls = group[0].get("tool_calls") or []
        tool_ids = [message.get("tool_call_id") for message in group[1:]
                    if message.get("role") == "tool"]
        call_ids = [call.get("id") for call in calls if isinstance(call, dict)]
        if calls or tool_ids:
            if (len(call_ids) != len(calls) or not all(call_ids)
                    or len(set(call_ids)) != len(call_ids)
                    or sorted(call_ids) != sorted(tool_ids, key=str)):
                continue
        removable.append(list(range(start, end)))
    return removable


def tokenize_request(vllm_url, payload, timeout, *, image_cache_nonce=None):
    # Same endpoint contract already used by the instruction synthesis scripts.
    body = {key: copy.deepcopy(payload[key]) for key in
            ("model", "messages", "chat_template_kwargs", "tools", "tool_choice")
            if key in payload}
    _isolate_image_cache(body, "tokenize", nonce=image_cache_nonce)
    body["add_generation_prompt"] = True
    endpoint = vllm_url.rstrip("/").removesuffix("/v1") + "/tokenize"
    request = urllib.request.Request(endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    # Tokenization is read-only. Retry only transient transport/overload
    # failures, within one total caller budget; never replay generation here.
    configured_timeout = os.environ.get('ACTION_HEAD_TOKENIZE_TIMEOUT_SECONDS')
    per_attempt = (None if timeout is None and configured_timeout is None
                   else float(configured_timeout or '30'))
    attempts = int(os.environ.get('ACTION_HEAD_TOKENIZE_ATTEMPTS', '1'))
    if (per_attempt is not None and per_attempt <= 0) or not 1 <= attempts <= 3:
        raise ValueError('invalid bounded tokenize retry settings')
    deadline = None if timeout is None else time.monotonic() + timeout
    for attempt in range(1, attempts + 1):
        remaining = per_attempt if deadline is None else min(per_attempt, deadline-time.monotonic())
        if remaining is not None and remaining <= 0:
            raise serving_error(body, 'tokenize_open', TimeoutError('tokenize deadline exhausted'))
        phase = 'tokenize_open'
        try:
            with urllib.request.urlopen(request, timeout=remaining) as response:
                phase = 'tokenize_read'
                return json.load(response)
        except TRANSPORT_ERRORS as exc:
            retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in (429, 502, 503, 504)
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            if not retryable or attempt == attempts or (deadline is not None and time.monotonic() >= deadline):
                raise serving_error(body, phase, exc) from exc
            print(json.dumps(dict(event='tokenize_transport_retry', endpoint=endpoint,
                phase=phase, attempt=attempt, attempts=attempts, error_type=type(exc).__name__,
                timeout_seconds=remaining, pid=os.getpid(), node=socket.gethostname(),
                progress_unavailable=True)), flush=True)
        except json.JSONDecodeError as exc:
            raise serving_error(body, 'tokenize_decode', exc, failure_kind='protocol') from exc


def budget_chat_request(vllm_url, payload, timeout, *, tokenizer=None, tokenize_image_cache="probe",
                        allow_history_trim=True):
    """Return (exact prepared request, compact audit), or abort before sampling.

    At most 12 serving-tokenizer probes. Binary search retains as much old text
    as possible under the usual monotone token count; every returned payload is
    independently measured to fit, even if counts are not monotone.
    """
    started = time.monotonic()
    if tokenize_image_cache not in ("probe", "budget"):
        raise ValueError("tokenize_image_cache must be probe or budget")
    if tokenize_image_cache == "budget" and tokenizer is not None:
        raise ValueError("budget image cache requires the real serving tokenizer")
    if tokenize_image_cache == "budget":
        # All probes retain identical images in identical order. Share only the
        # tokenizer transport IDs inside this single budget transaction. They
        # remain disjoint from generation and every other budget transaction.
        cache_nonce = uuid.uuid4().hex
        tokenizer = lambda url, body, limit: tokenize_request(
            url, body, limit, image_cache_nonce=cache_nonce)
    else:
        tokenizer = tokenizer or tokenize_request
    prepared = copy.deepcopy(payload)
    _isolate_image_cache(prepared, "generate")
    messages = prepared["messages"]
    completion = prepared.get("max_tokens")
    if type(completion) is not int or completion <= 0:
        raise ValueError("context budget requires positive integer request max_tokens")
    groups = removable_message_groups(messages) if allow_history_trim else []
    image_count = sum(_images(message) for message in messages)
    audit = dict(policy=context_budget_policy(), status="checking", tokenizer_calls=0,
                 completion_tokens=completion, images_before=image_count,
                 messages_before=len(messages), request_sha256_before=_digest(prepared),
                 client_timing=dict(schema="context_budget_client_timing_v1",
                     scope="client wall time; tokenize includes serialization, HTTP and server processing",
                     setup_seconds=time.monotonic() - started, probes=[]))
    if tokenize_image_cache == "budget":
        audit["policy"] = dict(audit["policy"],
            version="paired_text_history_budget_v3_transaction_mm_cache",
            image_cache="separate_generate_and_per_budget_tokenize_uuids")
    if not allow_history_trim:
        audit['policy'] = dict(audit['policy'], version='v8_append_only_context_budget_v1',
                              retain='all_messages_and_images', remove='none')
    measured = {}
    max_model_len = None

    def probe(removed_groups):
        nonlocal max_model_len
        if removed_groups in measured:
            return measured[removed_groups]
        if audit["tokenizer_calls"] >= MAX_TOKENIZE_CALLS:
            raise RuntimeError("context tokenizer probe bound exceeded")
        removed = {i for group in groups[:removed_groups] for i in group}
        candidate = dict(prepared, messages=[message for i, message in enumerate(messages)
                                             if i not in removed])
        audit["tokenizer_calls"] += 1
        probe_started = time.monotonic()
        timing = dict(removed_groups=removed_groups, messages=len(candidate["messages"]),
                      completed=False)
        audit["client_timing"]["probes"].append(timing)
        try:
            result = tokenizer(vllm_url, candidate, timeout)
            timing["completed"] = True
        finally:
            timing["seconds"] = time.monotonic() - probe_started
            audit["client_timing"]["elapsed_seconds"] = time.monotonic() - started
        tokens, limit = result.get("count"), result.get("max_model_len")
        timing.update(input_tokens=tokens, max_model_len=limit)
        if type(tokens) is not int or tokens < 0 or type(limit) is not int or limit <= 0:
            raise RuntimeError("invalid serving /tokenize count or max_model_len")
        if max_model_len is not None and limit != max_model_len:
            raise RuntimeError("serving max_model_len changed during context budgeting")
        max_model_len = limit
        measured[removed_groups] = (tokens, candidate, sorted(removed))
        return measured[removed_groups]

    def fits(tokens):
        return tokens + completion + SAFETY_TOKENS <= max_model_len

    tokens_before, _, _ = probe(0)
    audit.update(input_tokens_before=tokens_before, max_model_len=max_model_len)
    selected = 0
    if not fits(tokens_before):
        minimum_tokens, _, _ = probe(len(groups))
        if not fits(minimum_tokens):
            audit.update(status="protected_context_exceeds_budget",
                         minimum_input_tokens=minimum_tokens,
                         removable_groups=len(groups))
            raise ContextBudgetExhausted(
                f"protected context {minimum_tokens} + completion {completion} + safety "
                f"{SAFETY_TOKENS} exceeds model limit {max_model_len}; no request sampled", audit)
        low, high = 0, len(groups)
        # `high` is always a measured fitting request, including at the bound.
        while high - low > 1 and audit["tokenizer_calls"] < MAX_TOKENIZE_CALLS:
            middle = (low + high) // 2
            if fits(probe(middle)[0]):
                high = middle
            else:
                low = middle
        selected = high
    tokens_after, result, removed = measured[selected]
    images_after = sum(_images(message) for message in result["messages"])
    if images_after != image_count:
        raise AssertionError("context budget changed retained image window")
    audit.update(status="trimmed" if selected else "unchanged",
                 input_tokens_after=tokens_after, images_after=images_after,
                 messages_after=len(result["messages"]), removed_groups=selected,
                 removed_message_indices=removed, request_sha256_after=_digest(result))
    audit["client_timing"].update(elapsed_seconds=time.monotonic() - started,
        tokenize_seconds=sum(probe["seconds"] for probe in audit["client_timing"]["probes"]))
    return result, audit
