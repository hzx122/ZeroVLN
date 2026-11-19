"""Opt-in, source-bound fix for FLA 0.5.0's unused NB specialization.

Only the two small-D kernel objects are replaced in the original FLA module.
Existing imports of l2norm_fwd/l2norm_bwd retain their original globals, so the
Qwen gated-delta-rule callers also use the replacement. No installed files,
autotune keys, candidate configurations, or numerical expressions are changed.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
from importlib import metadata
import json
import linecache
import os
from pathlib import Path
import threading
from types import ModuleType
from typing import Any

ENVIRONMENT_VARIABLE = "ACTION_HEAD_FLA_L2NORM_DYNAMIC_NB"
EXPECTED_FLA_VERSION = "0.5.0"
EXPECTED_SOURCE_SHA256 = "95885e4679b2b496004b8c1336b9ef0f5fc2eacc0faba4d4d491712ab28b676e"
KERNEL_NAMES = ("l2norm_fwd_kernel", "l2norm_bwd_kernel")
_MARKER = "_action_head_dynamic_nb_patch"
_LOCK = threading.RLock()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _patched_source(source: str) -> str:
    """Reuse the probe's AST change, rejecting any upstream contract drift."""
    if _sha256(source.encode()) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("FLA l2norm source SHA256 differs from the validated 0.5.0 source")
    tree = ast.parse(source)
    kernels = [node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name in KERNEL_NAMES]
    if tuple(node.name for node in kernels) != KERNEL_NAMES:
        raise RuntimeError("FLA l2norm must contain exactly the two expected kernels")
    for node in kernels:
        references = [item.lineno for statement in node.body for item in ast.walk(statement)
                      if isinstance(item, ast.Name) and item.id == "NB"]
        if references:
            raise RuntimeError(f"NB is used by {node.name}: {references}")
        arguments = [argument for argument in node.args.args if argument.arg == "NB"]
        if len(arguments) != 1 or ast.unparse(arguments[0].annotation) != "tl.constexpr":
            raise RuntimeError(f"Unexpected NB annotation in {node.name}")
        arguments[0].annotation = None
        jit_calls = [decorator for decorator in node.decorator_list
                     if isinstance(decorator, ast.Call)
                     and ast.unparse(decorator.func) == "triton.jit"]
        if len(jit_calls) != 1:
            raise RuntimeError(f"Unexpected JIT decorator in {node.name}")
        keywords = [keyword for keyword in jit_calls[0].keywords
                    if keyword.arg == "do_not_specialize"]
        if len(keywords) != 1 or ast.literal_eval(keywords[0].value) != ["T"]:
            raise RuntimeError(f"Unexpected specialization contract in {node.name}")
        keywords[0].value.elts.append(ast.Constant(value="NB"))
    replacement = ast.Module(body=kernels, type_ignores=[])
    ast.fix_missing_locations(replacement)
    return ast.unparse(replacement) + "\n"


def _config_signature(tuner: Any) -> list[dict[str, Any]]:
    signatures = []
    for config in tuner.configs:
        value = getattr(config, "all_kwargs", None)
        if value is None:
            value = {**config.kwargs, **{
                name: getattr(config, name) for name in
                ("num_warps", "num_stages", "num_ctas", "maxnreg")
                if getattr(config, name, None) is not None}}
        signatures.append(dict(value() if callable(value) else value))
    return signatures


def _validate_tuner(tuner: Any, *, patched: bool) -> None:
    if tuner.keys != ["D", "NB"]:
        raise RuntimeError("FLA l2norm autotune keys changed")
    expected_grid = [(warps, bt) for warps in (1, 2, 4, 8, 16)
                     for bt in (8, 16, 32, 64, 128)]
    actual_grid = [(config.num_warps, config.kwargs.get("BT")) for config in tuner.configs]
    if actual_grid != expected_grid:
        raise RuntimeError("FLA l2norm must retain all 25 original autotune candidates")
    jit = tuner.fn
    if jit.do_not_specialize != (["T", "NB"] if patched else ["T"]):
        raise RuntimeError("FLA l2norm JIT specialization flags do not match the requested state")
    if (jit.arg_names.index("NB") in jit.constexprs) == patched:
        raise RuntimeError("FLA l2norm NB constexpr state does not match the requested state")


def create_patched_kernels(module: ModuleType) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build audited kernels without mutating module; also used by the GPU gate.

    The virtual source stays in linecache for Triton's inspect.getsource and
    traceback support. It requires no shared or persistent writable cache.
    """
    version = metadata.version("fla-core")
    if version != EXPECTED_FLA_VERSION:
        raise RuntimeError(f"Unsupported fla-core version {version}; expected {EXPECTED_FLA_VERSION}")
    source_path = Path(module.__file__).resolve()
    source_bytes = source_path.read_bytes()
    if _sha256(source_bytes) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"Unsupported FLA l2norm source SHA256: {_sha256(source_bytes)}")
    for name in KERNEL_NAMES:
        _validate_tuner(getattr(module, name), patched=False)
    source = _patched_source(source_bytes.decode())
    patch_sha256 = _sha256(source.encode())
    filename = f"<action_head_fla_l2norm_dynamic_nb_{patch_sha256}>"
    # Triton obtains the decorated function's source via inspect.getsource.
    # mtime=None makes linecache retain this immutable process-local source.
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(module.__dict__)
    exec(compile(source, filename, "exec", dont_inherit=True), namespace)
    kernels = {name: namespace[name] for name in KERNEL_NAMES}
    for name, replacement in kernels.items():
        original = getattr(module, name)
        _validate_tuner(replacement, patched=True)
        if (_config_signature(original) != _config_signature(replacement)
                or original.cache_results != replacement.cache_results):
            raise RuntimeError(f"Autotune configuration changed for {name}")
    audit = {
        "enabled": True,
        "status": "constructed",
        "fla_core_version": version,
        "triton_version": metadata.version("triton"),
        "source_path": str(source_path),
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "patch_sha256": patch_sha256,
        "implementation_sha256": _sha256(Path(__file__).read_bytes()),
        "autotune_keys": ["D", "NB"],
        "candidate_count_per_kernel": 25,
        "do_not_specialize": ["T", "NB"],
        "kernels": list(KERNEL_NAMES),
        "runtime_source_storage": "process_local_linecache",
    }
    return kernels, audit


def install_l2norm_dynamic_nb_patch() -> dict[str, Any]:
    """Install once; fail closed on drift or an intervening kernel replacement."""
    with _LOCK:
        module = importlib.import_module("fla.modules.l2norm")
        installed = getattr(module, _MARKER, None)
        if installed is not None:
            if (_sha256(Path(module.__file__).read_bytes()) != EXPECTED_SOURCE_SHA256
                    or metadata.version("fla-core") != EXPECTED_FLA_VERSION):
                raise RuntimeError("FLA source/version changed after the NB patch was installed")
            for name in KERNEL_NAMES:
                if getattr(module, name) is not installed["kernels"][name]:
                    raise RuntimeError(f"Patched FLA kernel was replaced: {name}")
                _validate_tuner(getattr(module, name), patched=True)
            return {**installed["audit"], "status": "already_applied"}

        # FLA's gated-delta-rule module imports these wrappers by reference.
        # Keeping both wrappers and their globals intact updates those callers
        # even when FLA's own package imports them before this hook runs.
        for name in ("l2norm_fwd", "l2norm_bwd"):
            if getattr(module, name).__globals__ is not module.__dict__:
                raise RuntimeError(f"Unexpected FLA wrapper globals: {name}")
        kernels, audit = create_patched_kernels(module)
        module.__dict__.update(kernels)
        audit = {**audit, "status": "applied", "original_wrapper_globals_preserved": True}
        setattr(module, _MARKER, {"kernels": kernels, "audit": audit})
        return audit


def maybe_install_l2norm_dynamic_nb_patch() -> dict[str, Any]:
    """Training startup hook; disabled by default and accepts only 0 or 1."""
    value = os.environ.get(ENVIRONMENT_VARIABLE, "0")
    if value == "0":
        return {"enabled": False, "status": "disabled"}
    if value != "1":
        raise RuntimeError(f"{ENVIRONMENT_VARIABLE} must be 0 or 1, got {value!r}")
    try:
        audit = install_l2norm_dynamic_nb_patch()
    except Exception as error:
        print("[action_head_fla_l2norm_dynamic_nb] " + json.dumps({
            "enabled": True, "status": "failed", "rank": os.environ.get("RANK"),
            "error": str(error),
        }, sort_keys=True), flush=True)
        raise
    print("[action_head_fla_l2norm_dynamic_nb] " + json.dumps({
        **audit, "rank": os.environ.get("RANK"), "local_rank": os.environ.get("LOCAL_RANK"),
    }, sort_keys=True), flush=True)
    return audit
