"""Compiler-cache setup shared by the learner and vLLM workers."""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys


AUTOTUNE_SEED_ENV = "ACTION_HEAD_TRITON_AUTOTUNE_SEED"
_LEARNER_CACHE_KEY = re.compile(r"^learner_world_(\d+)_rank_(\d+)$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_autotune_seed(seed: Path, target: Path, *, world_size: int,
                           rank: int) -> dict:
    """Copy validated decisions into a private writable rank cache."""
    manifest_path = seed / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing Triton seed manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "schema_version": 1,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "triton_version": metadata.version("triton"),
        "fla_core_version": metadata.version("fla-core"),
        "world_size": world_size,
    }
    mismatches = {key: {"expected": value, "actual": manifest.get(key)}
                  for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise RuntimeError(f"incompatible Triton seed manifest: {mismatches}")
    entries = manifest.get("rank_files", {}).get(str(rank))
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"Triton seed has no files for rank {rank}")
    source_root = seed / f"rank_{rank}" / "triton"
    target_root = target / "triton"
    copied = 0
    for entry in entries:
        relative = entry.get("relative")
        digest = entry.get("sha256")
        if not isinstance(relative, str) or not relative.endswith(".autotune.json"):
            raise RuntimeError(f"invalid Triton seed entry for rank {rank}: {entry!r}")
        source = source_root / relative
        if not source.is_file() or _sha256(source) != digest:
            raise RuntimeError(f"Triton seed file failed hash validation: {source}")
        destination = target_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if _sha256(destination) != digest:
                raise RuntimeError(f"conflicting local Triton decision: {destination}")
            continue
        shutil.copy2(source, destination)
        copied += 1
    return {
        "enabled": True, "status": "installed", "seed": str(seed),
        "manifest_sha256": _sha256(manifest_path), "rank": rank,
        "world_size": world_size, "decisions": len(entries), "copied": copied,
        "storage": "persistent-readonly-to-container-local-private",
    }


def configure_process_compile_cache(cache_key: str | None = None) -> dict:
    """Use isolated compiler caches beneath the configured local runtime root.

    Startup uses a PID fallback because distributed ranks are not known yet.
    Learner/vLLM initialization calls this again with a stable rank key before
    the first model kernel is compiled. A rank can reuse its own cache during
    the job without concurrent ranks replacing files. Cross-job seeds must be
    explicitly version-bound inputs copied into each new local runtime root.
    """
    root = os.environ.get("ACTION_HEAD_PROCESS_COMPILE_CACHE_ROOT")
    if not root:
        return {"enabled": False, "status": "no_process_cache_root"}
    if cache_key is None:
        cache_key = str(os.getpid())
    cache_key = ''.join(char if char.isalnum() or char in '._-' else '-'
                        for char in str(cache_key)).strip('-')
    if not cache_key:
        raise ValueError('compiler cache key must not be empty')
    root = Path(root) / socket.gethostname() / cache_key
    for key, directory in (
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"),
        ("PYTORCH_KERNEL_CACHE_PATH", "kernels"), ("XDG_CACHE_HOME", "xdg"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("DG_CACHE_DIR", "deep_gemm"), ("TORCH_EXTENSIONS_DIR", "extensions"),
    ):
        path = root / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    seed_value = os.environ.get(AUTOTUNE_SEED_ENV)
    match = _LEARNER_CACHE_KEY.fullmatch(cache_key)
    if not seed_value:
        return {"enabled": True, "status": "private_cache", "root": str(root)}
    if match is None:
        return {"enabled": True, "status": "seed_skipped_non_learner", "root": str(root)}
    audit = _install_autotune_seed(
        Path(seed_value), root, world_size=int(match.group(1)), rank=int(match.group(2)))
    return {**audit, "root": str(root)}
