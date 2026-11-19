"""Read the four DualVLN-style query outputs from a local vLLM side channel.

The production vLLM extension writes only one ``(4, hidden_size)`` tensor for
an action expert request.  It never exports prompt/generated-token activations
and never uses ``extract_hidden_states``.  Files are request-scoped, local to
the host, and consumed exactly once.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from safetensors import SafetensorError, safe_open


DEFAULT_QUERY_COUNT = 4
HIDDEN_CONVENTION = "final_norm"


@dataclass(frozen=True)
class VllmQueryHidden:
    query_hidden: np.ndarray
    request_id: str
    source_path: str
    hidden_convention: str = HIDDEN_CONVENTION
    query_count: int = DEFAULT_QUERY_COUNT


def load_query_safetensors(path: str) -> Mapping[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {
            "query_hidden": handle.get_tensor("query_hidden").float().numpy(),
        }


class VllmQueryStateProvider:
    """Consume a latent-query artifact produced by the same slow request."""

    def __init__(self, cache_root: str, *, hidden_size: int,
                 query_count: int = DEFAULT_QUERY_COUNT,
                 artifact_loader=load_query_safetensors,
                 wait_seconds: float = 10.0, cleanup: bool = True):
        self.cache_root = os.path.realpath(cache_root)
        self.hidden_size = int(hidden_size)
        self.query_count = int(query_count)
        self.artifact_loader = artifact_loader
        self.wait_seconds = float(wait_seconds)
        self.cleanup = bool(cleanup)
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.query_count <= 0:
            raise ValueError("query_count must be positive")
        if self.wait_seconds <= 0:
            raise ValueError("wait_seconds must be positive")

    @staticmethod
    def _safe_request_id(value: Any) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("slow response snapshot has no query_request_id")
        if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
               for char in value):
            raise ValueError("query_request_id contains unsafe characters")
        return value

    def _path(self, request_id: str) -> str:
        path = os.path.realpath(os.path.join(self.cache_root, request_id + ".safetensors"))
        if os.path.commonpath((self.cache_root, path)) != self.cache_root:
            raise ValueError("query artifact path escapes cache root")
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
        raise RuntimeError(f"latent-query artifact was not ready: {last_error}")

    def __call__(self, *, request: Any, response: Any,
                 contract: Mapping[str, Any]) -> VllmQueryHidden:
        del request, contract
        if not isinstance(response, Mapping):
            raise ValueError("missing slow response snapshot")
        request_id = self._safe_request_id(response.get("query_request_id"))
        path = self._path(request_id)
        artifact = self._load_when_ready(path)
        try:
            hidden = np.asarray(artifact["query_hidden"])
            if hidden.shape != (self.query_count, self.hidden_size):
                raise ValueError(
                    "query_hidden must have shape "
                    f"({self.query_count},{self.hidden_size}), got {hidden.shape}")
            if not np.isfinite(hidden).all():
                raise ValueError("query_hidden contains NaN or Inf")
            # A content-derived id makes accidental stale-file reuse visible in logs.
            digest = hashlib.sha256(hidden.tobytes()).hexdigest()[:16]
            return VllmQueryHidden(
                query_hidden=np.ascontiguousarray(hidden),
                request_id=f"{request_id}:{digest}",
                source_path=path,
                query_count=self.query_count,
            )
        finally:
            if self.cleanup:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
