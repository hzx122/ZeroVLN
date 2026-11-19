"""Consume Qwen tool-call hidden states exported by a vLLM server.

The public Tool request remains only ``target/where/plan``.  This module is the
runtime-side bridge: it validates the safetensors artifact returned by the same
generation request, aligns generated tokens with the raw assistant reply, and
keeps only the JSON ``arguments`` object that carries the semantic contract.

The stock vLLM ExampleHiddenStatesConnector still writes the full requested
layer before this process selects the contract tokens.  It is a correctness
backend, not the final transport.  A production connector can perform the same
selection/compression in the worker while preserving this provider contract.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from safetensors import SafetensorError

from . import contract as A


Tokenize = Callable[[str], Sequence[int]]
ArtifactLoader = Callable[[str], Mapping[str, Any]]


@dataclass(frozen=True)
class VllmContractHidden:
    """Complete frozen-Qwen condition selected from one generation request."""

    hidden: np.ndarray
    token_ids: np.ndarray
    field_ids: np.ndarray
    semantic_hidden: np.ndarray
    semantic_token_ids: np.ndarray
    semantic_sources: tuple[str, ...]
    request_id: str
    source_path: str
    source_token_count: int
    prompt_token_count: int
    completion_token_count: int
    contract_text: str
    boundary_expansion: int
    alignment_mode: str = "token_subsequence"

    @property
    def contract_hidden(self) -> np.ndarray:
        return self.hidden

    @property
    def contract_token_ids(self) -> np.ndarray:
        return self.token_ids


class VllmTokenizerClient:
    """Small HTTP tokenizer client; it does not load a second Qwen model."""

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: float = 30.0):
        self.url = base_url.rstrip("/") + "/tokenize"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def __call__(self, text: str) -> Sequence[int]:
        payload = json.dumps({
            "model": self.model,
            "prompt": text,
            "add_special_tokens": False,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        token_ids = result.get("tokens")
        if token_ids is None:
            token_ids = result.get("token_ids")
        if not isinstance(token_ids, list) or any(
                not isinstance(token, int) for token in token_ids):
            raise RuntimeError("vLLM /tokenize did not return integer token ids")
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        payload = json.dumps({
            "model": self.model,
            "tokens": [int(token) for token in token_ids],
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url.rsplit("/tokenize", 1)[0] + "/detokenize",
            data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        text = result.get("prompt")
        if not isinstance(text, str):
            raise RuntimeError("vLLM /detokenize did not return prompt text")
        return text


def load_safetensors_numpy(path: str) -> Mapping[str, Any]:
    from safetensors import safe_open

    # vLLM exports model activations as bfloat16. NumPy has no portable native
    # bfloat16 dtype, so read through the torch backend and convert only at this
    # legacy NumPy boundary. The production in-worker connector stays in torch.
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {
            "token_ids": handle.get_tensor("token_ids").numpy(),
            "hidden_states": handle.get_tensor("hidden_states").float().numpy(),
        }


def _find_subsequence(sequence: Sequence[int], part: Sequence[int]) -> int:
    if not part:
        raise ValueError("assistant response tokenization is empty")
    last = len(sequence) - len(part)
    for start in range(last, -1, -1):
        if list(sequence[start:start + len(part)]) == list(part):
            return start
    raise ValueError("assistant response tokens do not align with hidden-state token_ids")


def _find_available_response_prefix(
    sequence: Sequence[int], part: Sequence[int], max_missing_suffix: int = 8
) -> tuple[int, int]:
    """Align a response whose final sampled token has no hidden state."""
    minimum = max(1, len(part) - max_missing_suffix)
    for kept in range(len(part), minimum - 1, -1):
        try:
            return _find_subsequence(sequence, part[:kept]), kept
        except ValueError:
            continue
    raise ValueError(
        "assistant response prefix does not align with generated hidden-state "
        f"tokens: generated_len={len(sequence)}, response_len={len(part)}, "
        f"generated_tail={list(sequence[-16:])}, "
        f"response_head={list(part[:16])}, response_tail={list(part[-16:])}"
    )


def _find_contract_token_core(
    sequence: Sequence[int], contract_ids: Sequence[int], max_boundary_trim: int = 3
) -> tuple[int, int, int, int]:
    """Locate JSON contract tokens despite BPE merges at its two boundaries."""
    for total_trim in range(0, 2 * max_boundary_trim + 1):
        for left_trim in range(0, min(max_boundary_trim, total_trim) + 1):
            right_trim = total_trim - left_trim
            if right_trim > max_boundary_trim:
                continue
            end = len(contract_ids) - right_trim if right_trim else len(contract_ids)
            core = contract_ids[left_trim:end]
            if not core:
                continue
            try:
                start = _find_subsequence(sequence, core)
            except ValueError:
                continue
            return start, start + len(core), left_trim, right_trim
    raise ValueError("arguments JSON tokens do not align with generated hidden states")


def _matching_contract_span(text: str, contract: Mapping[str, Any]) -> tuple[int, int]:
    """Return the character span of the matching tool call's arguments object."""
    for match in A.TOOL_CALL_RE.finditer(text or ""):
        raw = A.strip_code_fence(match.group(1).strip())
        try:
            call = json.loads(raw)
        except Exception:
            # Match agent_tools.parse_tool_calls: Qwen may omit exactly the
            # final outer brace while leaving the arguments object complete.
            if (raw.endswith("}") and
                    raw.count("{") == raw.count("}") + 1):
                try:
                    call = json.loads(raw + "}")
                except Exception:
                    continue
            else:
                continue
        normalized, _ = A.validate_action_head_arguments(call.get("arguments"))
        if call.get("name") != "run_action_head" or normalized != dict(contract):
            continue

        args_match = re.search(r'"arguments"\s*:\s*\{', match.group(0))
        if args_match is None:
            continue
        brace = match.start() + args_match.end() - 1
        depth, quoted, escaped = 0, False, False
        for index in range(brace, match.end()):
            char = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return brace, index + 1
    raise ValueError("matching run_action_head arguments object not found in response")


def _semantic_texts(request: Any) -> tuple[str, ...]:
    if not isinstance(request, Mapping):
        raise ValueError("slow request snapshot has no semantic provenance")
    values = request.get("semantic_texts")
    if not isinstance(values, (list, tuple)):
        raise ValueError("slow request snapshot has no semantic_texts")
    result = tuple(dict.fromkeys(
        value.strip() for value in values
        if isinstance(value, str) and value.strip()
    ))
    if not result:
        raise ValueError("slow request snapshot semantic_texts is empty")
    return result


def _aligned_core(sequence: Sequence[int], text: str, tokenize: Tokenize) \
        -> tuple[int, int]:
    ids = list(tokenize(text))
    start, end, _, _ = _find_contract_token_core(sequence, ids)
    return start, end


def _contract_field_ids(generated_ids: Sequence[int], contract_text: str,
                        contract: Mapping[str, Any], tokenize: Tokenize,
                        contract_start: int, contract_end: int) -> np.ndarray:
    """Label only value tokens; JSON keys/punctuation retain field id zero."""
    labels = np.zeros(contract_end - contract_start, dtype=np.int64)
    cursor = 0
    for field_id, key in enumerate(
            A.RUN_ACTION_HEAD_PARAMETERS["required"], start=1):
        field = re.search(r'"' + re.escape(key) + r'"\s*:\s*',
                          contract_text[cursor:])
        if field is None:
            raise ValueError(f"contract field text not found for {key}")
        value_start = cursor + field.end()
        try:
            value, value_end_rel = json.JSONDecoder().raw_decode(
                contract_text[value_start:])
        except Exception as exc:
            raise ValueError(f"contract value JSON invalid for field {key}") from exc
        if value != contract[key]:
            raise ValueError(f"contract value mismatch for field {key}")
        value_end = value_start + value_end_rel
        local_start, local_end = _aligned_core(
            generated_ids[contract_start:contract_end],
            contract_text[value_start:value_end], tokenize)
        labels[local_start:local_end] = field_id
        cursor = value_end
    return labels


def _decoded_offsets(token_ids: Sequence[int], detokenize: Callable) \
        -> tuple[str, list[int], list[int]]:
    """Map a non-canonical guided-decoding tokenization back to characters."""
    decoded = [""] + [detokenize(token_ids[:index])
                      for index in range(1, len(token_ids) + 1)]
    full = decoded[-1]
    starts = [len(decoded[index]) for index in range(len(token_ids))]
    ends = [len(decoded[index + 1]) for index in range(len(token_ids))]
    return full, starts, ends


def _overlapping_token_span(starts: Sequence[int], ends: Sequence[int],
                            char_start: int, char_end: int) -> tuple[int, int]:
    selected = [index for index, (start, end) in enumerate(zip(starts, ends))
                if end > char_start and start < char_end]
    if not selected:
        raise ValueError("character span has no corresponding hidden-state tokens")
    return selected[0], selected[-1] + 1


def _decoded_field_ids(contract: Mapping[str, Any], contract_text: str,
                       contract_char_start: int, local_start: int, local_end: int,
                       starts: Sequence[int], ends: Sequence[int]) -> np.ndarray:
    labels = np.zeros(local_end - local_start, dtype=np.int64)
    cursor = 0
    for field_id, key in enumerate(
            A.RUN_ACTION_HEAD_PARAMETERS["required"], start=1):
        field = re.search(r'"' + re.escape(key) + r'"\s*:\s*',
                          contract_text[cursor:])
        if field is None:
            raise ValueError(f"contract field text not found for {key}")
        value_start = cursor + field.end()
        try:
            value, value_end_rel = json.JSONDecoder().raw_decode(
                contract_text[value_start:])
        except Exception as exc:
            raise ValueError(f"contract value JSON invalid for field {key}") from exc
        if value != contract[key]:
            raise ValueError(f"contract value mismatch for field {key}")
        value_end = value_start + value_end_rel
        cursor = value_end
        absolute_start = contract_char_start + value_start
        absolute_end = contract_char_start + value_end
        for token_index in range(local_start, local_end):
            if ends[token_index] > absolute_start and starts[token_index] < absolute_end:
                labels[token_index - local_start] = field_id
    return labels


class VllmHiddenStateProvider:
    """Load and contract-select a one-request vLLM hidden-state artifact."""

    def __init__(self, cache_root: str, tokenize: Tokenize,
                 artifact_loader: ArtifactLoader = load_safetensors_numpy,
                 wait_seconds: float = 10.0, cleanup: bool = True,
                 boundary_expansion: int = 1):
        self.cache_root = os.path.realpath(cache_root)
        self.tokenize = tokenize
        self.artifact_loader = artifact_loader
        self.wait_seconds = float(wait_seconds)
        self.cleanup = bool(cleanup)
        self.boundary_expansion = int(boundary_expansion)
        self._decoded_offset_cache = {}
        if self.wait_seconds <= 0:
            raise ValueError("wait_seconds must be positive")
        if self.boundary_expansion < 0:
            raise ValueError("boundary_expansion must be non-negative")

    def _safe_path(self, value: Any) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("vLLM response has no hidden_states_path")
        path = os.path.realpath(value)
        if os.path.commonpath((self.cache_root, path)) != self.cache_root:
            raise ValueError("hidden_states_path is outside the configured cache root")
        return path

    def _load_when_ready(self, path: str) -> Mapping[str, Any]:
        deadline = time.monotonic() + self.wait_seconds
        last_error = None
        while time.monotonic() < deadline:
            try:
                return self.artifact_loader(path)
            except (FileNotFoundError, OSError, ValueError, SafetensorError) as exc:
                last_error = exc
                time.sleep(0.01)
        raise RuntimeError(f"hidden-state artifact was not ready: {last_error}")

    @staticmethod
    def _token_major(hidden: Any, token_count: int) -> np.ndarray:
        value = np.asarray(hidden)
        if value.ndim == 2 and value.shape[0] == token_count:
            return value
        if value.ndim != 3:
            raise ValueError(f"hidden_states must be rank 2 or 3, got {value.shape}")
        if value.shape[0] == token_count:
            if value.shape[1] != 1:
                raise ValueError("v1 requires exactly one extracted Qwen layer")
            return value[:, 0, :]
        if value.shape[1] == token_count:
            if value.shape[0] != 1:
                raise ValueError("v1 requires exactly one extracted Qwen layer")
            return value[0, :, :]
        raise ValueError(
            f"hidden_states token dimension does not match {token_count}: {value.shape}")

    def __call__(self, *, request: Any, response: Any,
                 contract: Mapping[str, Any]) -> VllmContractHidden:
        semantic_sources = _semantic_texts(request)
        if not isinstance(response, Mapping):
            raise ValueError("missing slow response snapshot")
        transfer = response.get("kv_transfer_params") or {}
        path = self._safe_path(transfer.get("hidden_states_path"))
        usage = response.get("usage") or {}
        prompt_count = int(usage.get("prompt_tokens") or 0)
        completion_count = int(usage.get("completion_tokens") or 0)
        if prompt_count <= 0 or completion_count <= 0:
            raise ValueError("vLLM response is missing prompt/completion token counts")

        artifact = self._load_when_ready(path)
        try:
            token_ids = np.asarray(artifact["token_ids"], dtype=np.int64).reshape(-1)
            hidden = self._token_major(artifact["hidden_states"], len(token_ids))
            # The last sampled token is not an input to another forward pass,
            # so no hidden state exists for it. vLLM's finish-time connector
            # therefore exports prompt + completion - 1 tokens.
            minimum_with_generated = prompt_count + completion_count - 1
            if len(token_ids) < minimum_with_generated:
                raise ValueError(
                    "vLLM artifact is prompt-only; generated contract hidden states "
                    "require include_output_tokens support")

            available_completion = len(token_ids) - prompt_count
            generated_ids = token_ids[-available_completion:].tolist()
            prompt_ids = token_ids[:prompt_count].tolist()
            text = str(response.get("text") or "")
            char_start, char_end = _matching_contract_span(text, contract)
            contract_text = text[char_start:char_end]
            contract_ids = list(self.tokenize(contract_text))
            alignment_mode = "token_subsequence"
            decoded_offsets = None
            try:
                core_start, core_end, left_trim, right_trim = _find_contract_token_core(
                    generated_ids, contract_ids
                )
            except ValueError:
                detokenize = getattr(self.tokenize, "decode", None)
                if detokenize is None:
                    raise
                cache_key = tuple(generated_ids)
                decoded = self._decoded_offset_cache.get(cache_key)
                if decoded is None:
                    decoded = _decoded_offsets(generated_ids, detokenize)
                    self._decoded_offset_cache[cache_key] = decoded
                decoded_text, starts, ends = decoded
                decoded_char_start, decoded_char_end = _matching_contract_span(
                    decoded_text, contract)
                core_start, core_end = _overlapping_token_span(
                    starts, ends, decoded_char_start, decoded_char_end)
                left_trim = right_trim = 0
                decoded_offsets = (
                    decoded_char_start, starts, ends)
                alignment_mode = "incremental_detokenize"
            # Include a one-token halo for boundary BPE merges. This directly
            # selects arguments JSON and is independent of Qwen tool-tag special
            # token encoding.
            local_start = max(
                0, core_start - max(self.boundary_expansion, left_trim)
            )
            local_end = min(
                len(generated_ids),
                core_end + max(self.boundary_expansion, right_trim),
            )
            if local_start >= local_end:
                raise ValueError("contract token span is empty after token alignment")

            generated_base = prompt_count
            indices = np.arange(
                generated_base + local_start,
                generated_base + local_end,
                dtype=np.int64,
            )
            if decoded_offsets is None:
                try:
                    field_ids = _contract_field_ids(
                        generated_ids, contract_text, contract, self.tokenize,
                        local_start, local_end)
                except ValueError:
                    detokenize = getattr(self.tokenize, "decode", None)
                    if detokenize is None:
                        raise
                    cache_key = tuple(generated_ids)
                    decoded = self._decoded_offset_cache.get(cache_key)
                    if decoded is None:
                        decoded = _decoded_offsets(generated_ids, detokenize)
                        self._decoded_offset_cache[cache_key] = decoded
                    decoded_text, starts, ends = decoded
                    decoded_char_start, decoded_char_end = _matching_contract_span(
                        decoded_text, contract)
                    core_start, core_end = _overlapping_token_span(
                        starts, ends, decoded_char_start, decoded_char_end)
                    local_start = max(0, core_start - self.boundary_expansion)
                    local_end = min(
                        len(generated_ids), core_end + self.boundary_expansion)
                    indices = np.arange(
                        generated_base + local_start,
                        generated_base + local_end,
                        dtype=np.int64,
                    )
                    field_ids = _decoded_field_ids(
                        contract, contract_text, decoded_char_start,
                        local_start, local_end, starts, ends)
                    alignment_mode = "incremental_detokenize"
            else:
                decoded_char_start, starts, ends = decoded_offsets
                field_ids = _decoded_field_ids(
                    contract, contract_text, decoded_char_start,
                    local_start, local_end, starts, ends)

            semantic_indices = []
            for semantic_text in semantic_sources:
                start_index, end_index = _aligned_core(
                    prompt_ids, semantic_text, self.tokenize)
                semantic_indices.extend(range(start_index, end_index))
            # Contract value tokens are semantic, but schema keys and JSON
            # punctuation are not.  Preserve stable order and remove overlap.
            semantic_indices.extend(
                int(indices[index]) for index in np.flatnonzero(field_ids))
            semantic_indices = np.asarray(
                list(dict.fromkeys(semantic_indices)), dtype=np.int64)
            if semantic_indices.size == 0:
                raise ValueError("semantic token selection is empty")
            request_id = hashlib.sha256(
                (path + "\0" + str(prompt_count) + "\0" + contract_text).encode()
            ).hexdigest()[:20]
            return VllmContractHidden(
                hidden=np.ascontiguousarray(hidden[indices]),
                token_ids=np.ascontiguousarray(token_ids[indices]),
                field_ids=field_ids,
                semantic_hidden=np.ascontiguousarray(hidden[semantic_indices]),
                semantic_token_ids=np.ascontiguousarray(token_ids[semantic_indices]),
                semantic_sources=semantic_sources,
                request_id=request_id,
                source_path=path,
                source_token_count=len(token_ids),
                prompt_token_count=prompt_count,
                completion_token_count=completion_count,
                contract_text=contract_text,
                boundary_expansion=self.boundary_expansion,
                alignment_mode=alignment_mode,
            )
        finally:
            if self.cleanup:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
