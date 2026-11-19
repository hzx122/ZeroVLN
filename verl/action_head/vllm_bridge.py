"""vLLM request/response bridge owned by the Verl Agent process."""
from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping
import uuid


DEFAULT_QUERY_COUNT = 4
QUERY_TOKEN_TEXT = "<|fim_prefix|>"


def strip_query_tail(text: str, *, query_token: str = QUERY_TOKEN_TEXT,
                     query_count: int = DEFAULT_QUERY_COUNT) -> tuple[str, int]:
    """Remove the internal vLLM query carrier suffix from returned text.

    Qwen's FIM token is a reserved single token but is not registered in this
    tokenizer's ``all_special_tokens`` list, so the OpenAI server can expose it
    in response text.  Persisted training prefixes must end at ``</tool_call>``;
    the learnable query embeddings are appended by the model wrapper itself.
    """
    value = str(text or "")
    count = 0
    while count < query_count and value.endswith(query_token):
        value = value[:-len(query_token)]
        count += 1
    if value.endswith(query_token):
        raise ValueError("response contains more query carrier tokens than expected")
    if count not in (0, query_count):
        raise ValueError(
            f"response contains a partial query tail: {count}/{query_count}")
    return value, count


def request_queries(payload: MutableMapping[str, Any], query_count: int,
                    request_id: str | None = None) -> str:
    """Mark one normal generation for an in-worker latent-query continuation.

    No hidden-state, speculative-decoding, KV-transfer, or scheduling option is
    added.  ``vllm_xargs`` is consumed only by the optional action expert worker
    extension; requests without this marker follow stock vLLM behavior.
    """
    if not isinstance(query_count, int) or not 1 <= query_count <= 64:
        raise ValueError("query_count must be an integer in [1, 64]")
    request_id = request_id or ("ahq-" + uuid.uuid4().hex)
    payload["request_id"] = request_id
    extra = dict(payload.get("vllm_xargs") or {})
    extra["action_head_query_count"] = query_count
    extra["action_head_query_request_id"] = request_id
    payload["vllm_xargs"] = extra
    return request_id


def request_four_queries(payload: MutableMapping[str, Any],
                         request_id: str | None = None) -> str:
    """Backward-compatible marker for the original four-query checkpoint."""
    return request_queries(payload, DEFAULT_QUERY_COUNT, request_id)


def capture_transfer_metadata(
        chunk: Mapping[str, Any], choice: Mapping[str, Any],
        destination: MutableMapping[str, Any]) -> None:
    transfer = chunk.get("kv_transfer_params") or choice.get("kv_transfer_params")
    if isinstance(transfer, dict):
        destination.update(transfer)


def response_snapshot(*, text: str, visible: str, usage: Mapping[str, Any],
                      query_request_id: str = "") -> Dict[str, Any]:
    """Runtime-only snapshot; callers must not append it to conversation."""
    return {
        "text": text,
        "visible": visible,
        "usage": dict(usage),
        "query_request_id": query_request_id,
    }


def semantic_request_snapshot(*, messages: Any, instruction: str) -> Dict[str, Any]:
    """Build runtime-only semantic provenance from the exact Agent replay.

    Only text that is meaningful to the action expert is named here.  The hidden
    provider aligns these strings against the prompt token ids exported by the
    same vLLM request; system prompt/schema boilerplate and assistant thinking
    are intentionally absent.
    """
    texts = [str(instruction)]
    for message in messages or ():
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        parts = content if isinstance(content, list) else [content]
        for part in parts:
            if not isinstance(part, Mapping) or part.get("type") != "text":
                continue
            value = str(part.get("text") or "")
            if value.startswith("[工具返回]\n"):
                texts.append(value)
    return {"messages": messages, "semantic_texts": texts}
