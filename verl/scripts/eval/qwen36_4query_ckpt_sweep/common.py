#!/usr/bin/env python3
"""Shared, dependency-light helpers for the Qwen3.6 checkpoint sweep."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable


EXPECTED_STEPS = tuple(range(200, 5001, 200))
STAGE_QUOTAS = {"p0": 17, "p1": 17, "p2": 16}


def load_subset(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cases = list(manifest.get("cases") or [])
    quotas = manifest.get("quotas") or {}
    counts = {str(stage): sum(row.get("stage") == stage for row in cases)
              for stage in quotas}
    if not cases or counts != quotas or len(cases) != int(manifest.get("case_count", -1)):
        raise RuntimeError(
            f"closed-loop subset mismatch: counts={counts} quotas={quotas} total={len(cases)}")
    return manifest, cases


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_anchor_subset(rows: list[dict[str, Any]], max_anchors: int,
                        seed: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Select a deterministic anchor subset and keep every window it owns."""
    anchors = sorted({str(row["slow_anchor_id"]) for row in rows})
    if max_anchors < 0:
        raise ValueError("max_anchors must be non-negative")
    if max_anchors and max_anchors < len(anchors):
        anchors = sorted(sorted(
            anchors,
            key=lambda anchor: (
                hashlib.sha256(f"{seed}:{anchor}".encode()).digest(), anchor),
        )[:max_anchors])
    selected = set(anchors)
    subset = [row for row in rows if str(row["slow_anchor_id"]) in selected]
    subset.sort(key=lambda row: (
        str(row["slow_anchor_id"]), int(row["fast_window_index"])))
    return subset, anchors


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def contract_direction(arguments: dict[str, Any]) -> str:
    raw_plan = arguments.get("plan")
    if isinstance(raw_plan, (list, tuple)):
        plan = " ".join(str(x) for x in raw_plan).lower()
    elif isinstance(raw_plan, str):
        plan = raw_plan.lower()
    else:
        plan = ""
    where = str(arguments.get("where", "")).lower()
    target = str(arguments.get("target", "")).lower()
    text = " ".join((plan, where, target))
    for value in ("left", "right"):
        if re.search(rf"\b(turn|bear|veer|rotate|pivot)\s+{value}\b", text):
            return value
    if re.search(r"\b(move|go|walk|head|proceed)\s+straight\b", text):
        return "straight"
    if re.search(r"\bstraight ahead\b|\bdirectly (?:in front|ahead)\b", text):
        return "straight"
    return "unspecified"


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def direction_from_endpoint(x_m: float, y_m: float, yaw_deg: float) -> str:
    # World headings cross the +/-180 degree boundary. The endpoint trace is
    # a relative rotation, so -329 degrees means +31 degrees rather than a
    # large right turn.
    yaw_deg = wrap_degrees(yaw_deg)
    if yaw_deg >= 3.0:
        return "left"
    if yaw_deg <= -3.0:
        return "right"
    if y_m >= 0.03:
        return "left"
    if y_m <= -0.03:
        return "right"
    if x_m >= 0.05:
        return "straight"
    return "degenerate"


def delta_endpoint(points: Iterable[Iterable[float]]) -> dict[str, float]:
    rows = [[float(v) for v in row] for row in points]
    x = sum(row[0] for row in rows)
    y = sum(row[1] for row in rows)
    yaw = math.degrees(sum(row[2] for row in rows))
    return {
        "x_m": x,
        "y_m": y,
        "yaw_deg": yaw,
        "path_m": sum(math.hypot(row[0], row[1]) for row in rows),
    }


def checkpoint_identity(checkpoint_dir: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_dir)
    match = re.fullmatch(r"global_step_(\d{6})", root.name)
    if not match:
        raise ValueError(f"invalid checkpoint directory name: {root.name}")
    report_path = root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    native = report.get("condition_mode") == "v5_native"
    head_path = root / "action_head" / "fast_slow_action_head.pt"
    query_path = (root / "context_aggregator.pt" if native else
                  root / "dualvln_query" / "dualvln_query_embeddings.pt")
    missing = [str(path) for path in (report_path, head_path, query_path)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete checkpoint, missing={missing}")
    step = int(match.group(1))
    rl_checkpoint = report.get("schema") == "action_head_ray_grpo_checkpoint"
    if rl_checkpoint:
        if (int(report.get("step", -1)) != step or report.get("train_scope") != "full"
                or report.get("publication_kind") != "checkpoint"
                or report.get("qwen_exported") is not True
                or report.get("qwen_export_dtype") not in ("preserve", "bf16")):
            raise ValueError(f"RL checkpoint report is not complete: {report_path}")
        qwen = root / "qwen"
        index_path = qwen / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = set(index.get("weight_map", {}).values())
        if (not shards or any(Path(name).name != name or
                              not (qwen / name).is_file() or
                              (qwen / name).stat().st_size <= 0 for name in shards)):
            raise ValueError(f"RL checkpoint Qwen export is incomplete: {qwen}")
    elif not report.get("ok") or int(report.get("global_step", -1)) != step:
        raise ValueError(f"checkpoint report is not complete: {report_path}")
    identity = {
        "step": step,
        "checkpoint_dir": str(root.resolve()),
        "action_head_dir": str((root / "action_head").resolve()),
        "report_sha256": sha256_file(report_path),
        "action_head_sha256": sha256_file(head_path),
    }
    if native:
        identity.update(condition_mode="v5_native", aggregator_path=str(query_path.resolve()),
                        aggregator_sha256=sha256_file(query_path))
    else:
        identity.update(query_dir=str((root / "dualvln_query").resolve()),
                        query_sha256=sha256_file(query_path))
    if rl_checkpoint:
        identity.update(qwen_dir=str(qwen.resolve()),
                        qwen_index_sha256=sha256_file(index_path),
                        checkpoint_schema=report["schema"])
    return identity


def mean(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else None


def rate(flags: Iterable[bool]) -> dict[str, Any]:
    rows = [bool(value) for value in flags]
    hits = sum(rows)
    return {"value": hits / len(rows) if rows else None,
            "numerator": hits, "denominator": len(rows)}
