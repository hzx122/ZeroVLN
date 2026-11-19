"""Source-bound removal of unused FLA batch-shape specializations.

FLA 0.5.0 marks a few shape arguments as ``tl.constexpr`` even though their
kernel bodies never read them. Triton consequently recompiles identical code
for every batch/sequence bucket. This opt-in patch keeps FLA's autotune keys
and candidate grids intact while making only those proven-unused arguments
runtime values. Installed package files are never modified.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib
from importlib import metadata
import inspect
import json
import linecache
import os
from pathlib import Path
import threading
from types import ModuleType
from typing import Any

from sft.fla_l2norm_dynamic_nb import _config_signature, _sha256


ENVIRONMENT_VARIABLE = "ACTION_HEAD_FLA_UNUSED_SHAPE_DYNAMIC"
EXPECTED_FLA_VERSION = "0.5.0"
_MARKER = "_action_head_unused_shape_patch"
_LOCK = threading.RLock()


@dataclass(frozen=True)
class ModulePatch:
    module_name: str
    source_sha256: str
    kernel_arguments: tuple[tuple[str, str], ...]
    wrapper_names: tuple[str, ...]


PATCHES = (
    ModulePatch(
        module_name="fla.modules.fused_norm_gate",
        source_sha256="d3a40dee12c22b8416d1cf61483c0e8eaa87e44765489b1b2f22c1525f0394cf",
        kernel_arguments=(
            ("layer_norm_gated_fwd_kernel", "NB"),
            ("layer_norm_gated_bwd_kernel", "NB"),
        ),
        wrapper_names=("layer_norm_gated_fwd", "layer_norm_gated_bwd"),
    ),
    ModulePatch(
        module_name="fla.ops.utils.cumsum",
        source_sha256="29488c52d18ef7464a44841cd5886312b89f504c5bf02a01e425f6554a22e71b",
        kernel_arguments=(
            ("chunk_local_cumsum_scalar_kernel", "B"),
            ("chunk_local_cumsum_vector_kernel", "B"),
            ("chunk_global_cumsum_scalar_kernel", "B"),
            ("chunk_global_cumsum_vector_kernel", "B"),
        ),
        wrapper_names=(
            "chunk_local_cumsum_scalar", "chunk_local_cumsum_vector",
            "chunk_global_cumsum_scalar", "chunk_global_cumsum_vector",
        ),
    ),
)


def _jit_function(kernel: Any) -> Any:
    value = kernel
    while not hasattr(value, "constexprs"):
        nested = getattr(value, "fn", None)
        if nested is None or nested is value:
            raise RuntimeError(f"Unable to find Triton JIT function below {type(kernel).__name__}")
        value = nested
    return value


def _autotuner(kernel: Any) -> Any | None:
    value = kernel
    while value is not None:
        if hasattr(value, "configs") and hasattr(value, "keys"):
            return value
        nested = getattr(value, "fn", None)
        if nested is None or nested is value:
            return None
        value = nested
    return None


def _patch_jit_decorator(node: ast.FunctionDef, argument: str) -> None:
    matches = []
    for index, decorator in enumerate(node.decorator_list):
        expression = decorator.func if isinstance(decorator, ast.Call) else decorator
        if ast.unparse(expression) == "triton.jit":
            matches.append((index, decorator))
    if len(matches) != 1:
        raise RuntimeError(f"Unexpected Triton JIT decorators for {node.name}")
    index, decorator = matches[0]
    if not isinstance(decorator, ast.Call):
        decorator = ast.Call(func=decorator, args=[], keywords=[])
        node.decorator_list[index] = decorator
    keywords = [item for item in decorator.keywords if item.arg == "do_not_specialize"]
    if not keywords:
        keyword = ast.keyword(arg="do_not_specialize", value=ast.List(elts=[], ctx=ast.Load()))
        decorator.keywords.append(keyword)
    elif len(keywords) == 1 and isinstance(keywords[0].value, (ast.List, ast.Tuple)):
        keyword = keywords[0]
    else:
        raise RuntimeError(f"Unexpected do_not_specialize for {node.name}")
    values = [ast.literal_eval(item) for item in keyword.value.elts]
    if argument in values:
        raise RuntimeError(f"{argument} is already do_not_specialize in {node.name}")
    keyword.value.elts.append(ast.Constant(argument))


def _patched_source(source: str, spec: ModulePatch) -> str:
    if _sha256(source.encode()) != spec.source_sha256:
        raise RuntimeError(f"FLA source SHA256 differs for {spec.module_name}")
    tree = ast.parse(source)
    by_name = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    selected = []
    for kernel_name, argument in spec.kernel_arguments:
        node = by_name.get(kernel_name)
        if node is None:
            raise RuntimeError(f"Missing expected kernel {spec.module_name}.{kernel_name}")
        loads = [item.lineno for statement in node.body for item in ast.walk(statement)
                 if isinstance(item, ast.Name) and item.id == argument
                 and isinstance(item.ctx, ast.Load)]
        if loads:
            raise RuntimeError(f"{argument} is used by {kernel_name}: {loads}")
        arguments = [item for item in node.args.args if item.arg == argument]
        if len(arguments) != 1 or ast.unparse(arguments[0].annotation) != "tl.constexpr":
            raise RuntimeError(f"Unexpected {argument} annotation in {kernel_name}")
        arguments[0].annotation = None
        _patch_jit_decorator(node, argument)
        selected.append(node)
    replacement = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(replacement)
    return ast.unparse(replacement) + "\n"


def create_patched_kernels(module: ModuleType, spec: ModulePatch) -> tuple[dict[str, Any], dict[str, Any]]:
    version = metadata.version("fla-core")
    if version != EXPECTED_FLA_VERSION:
        raise RuntimeError(f"Unsupported fla-core version {version}; expected {EXPECTED_FLA_VERSION}")
    source_path = Path(module.__file__).resolve()
    source_bytes = source_path.read_bytes()
    if _sha256(source_bytes) != spec.source_sha256:
        raise RuntimeError(f"Unsupported FLA source SHA256 for {spec.module_name}: {_sha256(source_bytes)}")
    source = _patched_source(source_bytes.decode(), spec)
    patch_sha256 = _sha256(source.encode())
    filename = f"<action_head_fla_unused_shape_{patch_sha256}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(module.__dict__)
    exec(compile(source, filename, "exec", dont_inherit=True), namespace)
    kernels = {name: namespace[name] for name, _ in spec.kernel_arguments}
    for name, argument in spec.kernel_arguments:
        original = getattr(module, name)
        replacement = kernels[name]
        original_jit = _jit_function(original)
        patched_jit = _jit_function(replacement)
        position = original_jit.arg_names.index(argument)
        if position not in original_jit.constexprs or position in patched_jit.constexprs:
            raise RuntimeError(f"Unexpected constexpr transition for {name}.{argument}")
        before = list(original_jit.do_not_specialize)
        after = list(patched_jit.do_not_specialize)
        if after != before + [argument]:
            raise RuntimeError(f"Unexpected do_not_specialize transition for {name}.{argument}")
        original_tuner = _autotuner(original)
        patched_tuner = _autotuner(replacement)
        if (original_tuner is None) != (patched_tuner is None):
            raise RuntimeError(f"Autotuner wrapper changed for {name}")
        if original_tuner is not None and (
                list(original_tuner.keys) != list(patched_tuner.keys)
                or _config_signature(original_tuner) != _config_signature(patched_tuner)
                or original_tuner.cache_results != patched_tuner.cache_results):
            raise RuntimeError(f"Autotune contract changed for {name}")
    audit = {
        "module": spec.module_name,
        "source_path": str(source_path),
        "source_sha256": spec.source_sha256,
        "patch_sha256": patch_sha256,
        "kernels": [{"name": name, "dynamic_argument": argument}
                    for name, argument in spec.kernel_arguments],
    }
    return kernels, audit


def install_fla_unused_shape_patch() -> dict[str, Any]:
    with _LOCK:
        if metadata.version("fla-core") != EXPECTED_FLA_VERSION:
            raise RuntimeError(f"Unsupported fla-core version {metadata.version('fla-core')}")
        audits = []
        for spec in PATCHES:
            module = importlib.import_module(spec.module_name)
            installed = getattr(module, _MARKER, None)
            if installed is not None:
                for name, _ in spec.kernel_arguments:
                    if getattr(module, name) is not installed["kernels"][name]:
                        raise RuntimeError(f"Patched FLA kernel was replaced: {spec.module_name}.{name}")
                audits.append({**installed["audit"], "status": "already_applied"})
                continue
            for wrapper_name in spec.wrapper_names:
                wrapper = inspect.unwrap(getattr(module, wrapper_name))
                if wrapper.__globals__ is not module.__dict__:
                    raise RuntimeError(f"Unexpected FLA wrapper globals: {spec.module_name}.{wrapper_name}")
            kernels, audit = create_patched_kernels(module, spec)
            module.__dict__.update(kernels)
            audit = {**audit, "status": "applied"}
            setattr(module, _MARKER, {"kernels": kernels, "audit": audit})
            audits.append(audit)
        return {
            "enabled": True,
            "status": "applied" if all(x["status"] == "applied" for x in audits) else "already_applied",
            "fla_core_version": EXPECTED_FLA_VERSION,
            "triton_version": metadata.version("triton"),
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "modules": audits,
        }


def maybe_install_fla_unused_shape_patch() -> dict[str, Any]:
    value = os.environ.get(ENVIRONMENT_VARIABLE, "0")
    if value == "0":
        return {"enabled": False, "status": "disabled"}
    if value != "1":
        raise RuntimeError(f"{ENVIRONMENT_VARIABLE} must be 0 or 1, got {value!r}")
    try:
        audit = install_fla_unused_shape_patch()
    except Exception as error:
        print("[action_head_fla_unused_shape] " + json.dumps({
            "enabled": True, "status": "failed", "rank": os.environ.get("RANK"),
            "error": str(error),
        }, sort_keys=True), flush=True)
        raise
    print("[action_head_fla_unused_shape] " + json.dumps({
        **audit, "rank": os.environ.get("RANK"), "local_rank": os.environ.get("LOCAL_RANK"),
    }, sort_keys=True), flush=True)
    return audit
