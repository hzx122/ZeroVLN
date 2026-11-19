"""Typed loader for the config-driven action expert scale sweep."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


HEAD_SCALE_FIELDS = (
    "query_num_queries",
    "visual_num_queries",
    "fusion_dim",
    "visual_layers",
    "fusion_layers",
    "fusion_heads",
    "flow_model_dim",
    "flow_layers",
    "flow_heads",
)


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ScaleArm:
    name: str
    head_overrides: Mapping[str, Any]
    max_steps: int
    run_name: str
    warm_start_checkpoint: str
    step_offset: int

    @property
    def training_steps(self) -> int:
        return self.max_steps - self.step_offset

    @classmethod
    def from_dict(cls, name: str, value: Mapping[str, Any]) -> "ScaleArm":
        architecture = dict(value.get("architecture") or {})
        missing = [field for field in HEAD_SCALE_FIELDS if field not in architecture]
        extra = sorted(set(architecture) - set(HEAD_SCALE_FIELDS)
                       - {"visual_hidden_size", "backbone_model_type"})
        if missing or extra:
            raise ValueError(
                f"scale {name} architecture fields mismatch; missing={missing}, extra={extra}")
        overrides = {
            field: _positive_int(f"scales.{name}.architecture.{field}", architecture[field])
            for field in HEAD_SCALE_FIELDS
        }
        if "visual_hidden_size" in architecture:
            overrides["visual_hidden_size"] = _positive_int(
                f"scales.{name}.architecture.visual_hidden_size", architecture["visual_hidden_size"])
        if "backbone_model_type" in architecture:
            model_type = architecture["backbone_model_type"]
            if not isinstance(model_type, str) or not model_type.strip():
                raise ValueError(f"scale {name} backbone_model_type must be a non-empty string")
            overrides["backbone_model_type"] = model_type
        if overrides["fusion_dim"] % overrides["fusion_heads"]:
            raise ValueError(f"scale {name} fusion_dim is not divisible by fusion_heads")
        if overrides["flow_model_dim"] % overrides["flow_heads"]:
            raise ValueError(f"scale {name} flow_model_dim is not divisible by flow_heads")
        run_name = str(value.get("run_name") or "")
        if not run_name:
            raise ValueError(f"scale {name} has no run_name")
        warm_start = str(value.get("warm_start_checkpoint") or "")
        step_offset = int(value.get("step_offset", 0))
        max_steps = _positive_int(f"scales.{name}.max_steps", value.get("max_steps"))
        if step_offset < 0 or step_offset >= max_steps:
            raise ValueError(f"scale {name} step_offset must be in [0, max_steps)")
        if bool(warm_start) != bool(step_offset):
            raise ValueError(
                f"scale {name} warm_start_checkpoint and step_offset must be set together")
        return cls(
            name=name,
            head_overrides=overrides,
            max_steps=max_steps,
            run_name=run_name,
            warm_start_checkpoint=warm_start,
            step_offset=step_offset,
        )


@dataclass(frozen=True)
class ScaleExperiment:
    source: Path
    sha256: str
    base_head_checkpoint: str
    base_query_checkpoint: str
    initialization_root: str
    manifest: str
    slow_model: str
    output_root: str
    action_head_as_tool_version: str
    nnodes: int
    nproc_per_node: int
    anchors_per_gpu: int
    learning_rate: float
    save_freq: int
    val_freq: int
    val_subset: int
    total_epochs: int | None
    seed: int
    extra_query_seed: int
    preserve_base_query_slots: int | None
    scales: Mapping[str, ScaleArm]

    def scale(self, name: str) -> ScaleArm:
        normalized = str(name).upper()
        try:
            return self.scales[normalized]
        except KeyError as error:
            raise ValueError(
                f"unknown scale {name!r}; choose one of {', '.join(self.scales)}") from error

    def initialization_dir(self, name: str) -> Path:
        return Path(self.initialization_root) / self.scale(name).name.lower()

    def output_dir(self, name: str) -> Path:
        return Path(self.output_root) / self.scale(name).run_name


def load_scale_experiment(path: str | Path) -> ScaleExperiment:
    source = Path(path).resolve()
    raw = source.read_bytes()
    value = json.loads(raw)
    if value.get("schema") != "action_head_scale_experiment_v1":
        raise ValueError("unsupported Action Head scale config schema")
    shared = dict(value.get("shared") or {})
    training = dict(value.get("training") or {})
    initialization = dict(value.get("initialization") or {})
    scales_value = value.get("scales")
    if not isinstance(scales_value, Mapping) or not scales_value:
        raise ValueError("scale config must contain a non-empty scales mapping")
    scales = {
        str(name).upper(): ScaleArm.from_dict(str(name).upper(), arm)
        for name, arm in scales_value.items()
    }

    def required(section: Mapping[str, Any], name: str) -> str:
        result = str(section.get(name) or "")
        if not result:
            raise ValueError(f"scale config field {name!r} is required")
        return result

    protocol = required(shared, "action_head_as_tool_version")
    if protocol not in ("v1", "v2", "v3", "v4.3", "v6", "v8"):
        raise ValueError("action_head_as_tool_version must be v1, v2, v3, v4.3, v6, or v8")
    learning_rate = training.get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive")
    preserve_slots_value = initialization.get("preserve_base_query_slots")
    if preserve_slots_value is not None:
        if (not isinstance(preserve_slots_value, int)
                or isinstance(preserve_slots_value, bool)
                or preserve_slots_value < 0):
            raise ValueError(
                "initialization.preserve_base_query_slots must be a non-negative integer")
    return ScaleExperiment(
        source=source,
        sha256=hashlib.sha256(raw).hexdigest(),
        base_head_checkpoint=required(initialization, "base_head_checkpoint"),
        base_query_checkpoint=required(initialization, "base_query_checkpoint"),
        initialization_root=required(initialization, "output_root"),
        manifest=required(shared, "manifest"),
        slow_model=required(shared, "slow_model"),
        output_root=required(shared, "output_root"),
        action_head_as_tool_version=protocol,
        nnodes=_positive_int("training.nnodes", training.get("nnodes")),
        nproc_per_node=_positive_int(
            "training.nproc_per_node", training.get("nproc_per_node")),
        anchors_per_gpu=_positive_int(
            "training.anchors_per_gpu", training.get("anchors_per_gpu")),
        learning_rate=float(learning_rate),
        save_freq=int(training.get("save_freq", -1)),
        val_freq=int(training.get("val_freq", 0)),
        val_subset=int(training.get("val_subset", 0)),
        total_epochs=(None if training.get("total_epochs") is None else
                      _positive_int("training.total_epochs", training["total_epochs"])),
        seed=int(initialization.get("seed", 20260826)),
        extra_query_seed=int(initialization.get("extra_query_seed", 20260903)),
        preserve_base_query_slots=preserve_slots_value,
        scales=scales,
    )


def head_structure_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """True when changing query-token count leaves all head tensors reusable."""
    left_values = dict(left)
    right_values = dict(right)
    left_values.pop("query_num_queries", None)
    right_values.pop("query_num_queries", None)
    return left_values == right_values
