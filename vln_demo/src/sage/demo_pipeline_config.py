"""Strict config loader for the Sage demo episode production pipeline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml


DEFAULT_CONFIG_PATH = str(
    Path(__file__).resolve().parents[2] / "config" / "sage_demo_pipeline.yaml")


def _read_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("pipeline config must be a YAML mapping: %s" % path)
    return data


def _parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ValueError("override must be dotted.key=value, got %r" % raw)
    key, value = raw.split("=", 1)
    parts = [p.strip() for p in key.split(".") if p.strip()]
    if not parts:
        raise ValueError("override key is empty: %r" % raw)
    return parts, yaml.safe_load(value)


def _compatible_type(old: Any, new: Any) -> bool:
    if isinstance(old, bool):
        return isinstance(new, bool)
    if isinstance(old, float):
        return isinstance(new, (int, float)) and not isinstance(new, bool)
    if isinstance(old, int):
        return isinstance(new, int) and not isinstance(new, bool)
    return isinstance(new, type(old))


def _validate_shape(config: Mapping[str, Any], schema: Mapping[str, Any],
                    prefix: str = "") -> None:
    missing = sorted(set(schema) - set(config))
    unknown = sorted(set(config) - set(schema))
    if missing or unknown:
        raise ValueError("config keys differ at %s: missing=%s unknown=%s" % (
            prefix or "<root>", missing, unknown))
    for key, expected in schema.items():
        dotted = "%s.%s" % (prefix, key) if prefix else key
        actual = config[key]
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                raise TypeError("%s must be a mapping" % dotted)
            _validate_shape(actual, expected, dotted)
        elif not _compatible_type(expected, actual):
            raise TypeError("%s expects %s, got %s" % (
                dotted, type(expected).__name__, type(actual).__name__))


def _apply_override(config: dict, raw: str) -> None:
    parts, value = _parse_override(raw)
    cursor: dict = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            raise KeyError("unknown pipeline config key: %s" % ".".join(parts))
        cursor = cursor[part]
    leaf = parts[-1]
    if leaf not in cursor:
        raise KeyError("unknown pipeline config key: %s" % ".".join(parts))
    old = cursor[leaf]
    if not _compatible_type(old, value):
        raise TypeError("override %s expects %s, got %s" % (
            ".".join(parts), type(old).__name__, type(value).__name__))
    cursor[leaf] = float(value) if isinstance(old, float) else value


def _require_positive(config: Mapping[str, Any], dotted: str) -> None:
    value: Any = config
    for part in dotted.split("."):
        value = value[part]
    if float(value) <= 0:
        raise ValueError("%s must be positive, got %r" % (dotted, value))


def validate_pipeline_config(config: Mapping[str, Any]) -> None:
    for dotted in (
        "runtime.episodes_per_scene", "camera.width", "camera.height",
        "camera.hfov_deg", "camera.height_m", "robot.physical_radius_m",
        "robot.replay_step_m", "sampling.start_min_spacing_m",
        "goal.room_margin_m", "goal.room_tail_m",
        "visibility.min_bbox_width_px", "visibility.min_bbox_height_px",
        "visibility.depth_sample_stride",
        "planner.min_accepted_smoothing_blend",
    ):
        _require_positive(config, dotted)
    if not 1.0 < float(config["camera"]["hfov_deg"]) < 179.0:
        raise ValueError("camera.hfov_deg must be in (1, 179)")
    if float(config["robot"]["planning_margin_m"]) < 0:
        raise ValueError("robot.planning_margin_m must be non-negative")
    if config["planner"]["mode"] not in ("ecm", "astar"):
        raise ValueError("planner.mode must be ecm or astar")
    blend = float(config["planner"]["min_accepted_smoothing_blend"])
    if not 0.0 <= blend <= 1.0:
        raise ValueError("planner.min_accepted_smoothing_blend must be in [0, 1]")
    if config["goal"]["terminal_mode"] not in (
            "radial_approach", "visibility_set"):
        raise ValueError("goal.terminal_mode must be radial_approach or visibility_set")
    if not config["runtime"]["scenes"]:
        raise ValueError("runtime.scenes must not be empty")
    for key in ("pair_euclidean_m", "pair_geodesic_m"):
        band = config["sampling"][key]
        if len(band) != 2 or float(band[0]) >= float(band[1]):
            raise ValueError("sampling.%s must be [min, max]" % key)
    if not config["sampling"]["goal_standoff_m"]:
        raise ValueError("sampling.goal_standoff_m must not be empty")
    cap = config["capability_profile"]
    for lo, hi in (("min_turns", "max_turns"),
                   ("min_doors", "max_doors"),
                   ("min_clearance_m", "max_clearance_m")):
        if float(cap[lo]) > float(cap[hi]):
            raise ValueError("capability_profile.%s must be <= %s" % (lo, hi))
    if int(cap["min_decisions"]) < 0:
        raise ValueError("capability_profile.min_decisions must be non-negative")
    if float(cap["start_target_visibility_max_arc_m"]) < 0:
        raise ValueError("capability_profile.start_target_visibility_max_arc_m must be non-negative")
    if int(cap["max_visible_same_family_distractors"]) < 0:
        raise ValueError("capability_profile.max_visible_same_family_distractors must be non-negative")
    for key in ("grounding_max_length_gap_m", "grounding_max_length_ratio",
                "grounding_min_goal_separation_m", "grounding_max_candidates_per_start"):
        if float(cap[key]) <= 0:
            raise ValueError("capability_profile.%s must be positive" % key)
    if float(cap["grounding_max_length_ratio"]) < 1.0:
        raise ValueError("capability_profile.grounding_max_length_ratio must be >= 1")
    for key in ("grounding_max_turn_gap", "grounding_max_door_gap"):
        if int(cap[key]) < 0:
            raise ValueError("capability_profile.%s must be non-negative" % key)


def load_pipeline_config(path: Optional[str] = None,
                         overrides: Iterable[str] = ()) -> tuple[dict, dict]:
    resolved_path = os.path.abspath(path or os.environ.get(
        "SAGE_DEMO_PIPELINE_CONFIG", DEFAULT_CONFIG_PATH))
    config = copy.deepcopy(_read_yaml(resolved_path))
    _validate_shape(config, _read_yaml(DEFAULT_CONFIG_PATH))
    applied = list(overrides)
    for raw in applied:
        _apply_override(config, raw)
    validate_pipeline_config(config)
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    meta = {
        "config_path": resolved_path,
        "overrides": applied,
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "uses_default_config": resolved_path == os.path.abspath(DEFAULT_CONFIG_PATH),
    }
    return config, meta


def pipeline_config_from_argv(argv: Optional[list[str]] = None) -> tuple[dict, dict]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    args, _ = parser.parse_known_args(argv)
    return load_pipeline_config(args.config, args.set)


def _main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--get", default=None,
                        help="print one dotted key instead of the full JSON")
    args = parser.parse_args(argv)
    config, meta = load_pipeline_config(args.config, args.set)
    if args.get:
        value: Any = config
        for part in args.get.split("."):
            value = value[part]
        if isinstance(value, list):
            print(",".join(str(x) for x in value))
        elif isinstance(value, bool):
            print("1" if value else "0")
        else:
            print(value)
        return
    print(json.dumps({"config": config, "meta": meta}, ensure_ascii=False,
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    _main()
