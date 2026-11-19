"""CPU metric helpers shared by the companion evaluator and search reporting."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from common import mean

METRIC_KEYS = ("prefix_ade_m", "prefix_fde_m", "full_ade_m", "full_fde_m",
               "endpoint_yaw_error_deg")
METRIC_VERSIONS = ("legacy_bf16_v1", "fp32_v2")


def trajectory_metrics(pred: np.ndarray, gt: np.ndarray, valid_steps: int) -> dict:
    valid_steps = max(1, min(int(valid_steps), len(gt), len(pred)))
    pred_xy = np.cumsum(pred[:valid_steps, :2], axis=0)
    gt_xy = np.cumsum(gt[:valid_steps, :2], axis=0)
    distances = np.linalg.norm(pred_xy - gt_xy, axis=1)
    prefix_steps = min(4, valid_steps)
    pred_prefix = pred_xy[:prefix_steps]
    gt_prefix = gt_xy[:prefix_steps]
    prefix_distances = np.linalg.norm(pred_prefix - gt_prefix, axis=1)
    pred_yaw = float(np.degrees(pred[:valid_steps, 2].sum()))
    gt_yaw = float(np.degrees(gt[:valid_steps, 2].sum()))
    return {
        "prefix_ade_m": float(prefix_distances.mean()),
        "prefix_fde_m": float(prefix_distances[-1]),
        "full_ade_m": float(distances.mean()),
        "full_fde_m": float(distances[-1]),
        "endpoint_yaw_error_deg": abs((pred_yaw - gt_yaw + 180.0) % 360.0 - 180.0),
    }


def denormalize_fp32(value, transform):
    """Apply existing transform coefficients after casting, without BF16 math."""
    value = np.asarray(value, dtype=np.float32)
    return (value * np.asarray(transform.scale, dtype=np.float32)
            + np.asarray(transform.offset, dtype=np.float32))


def aggregate_windows(rows):
    """Preserve window means and additionally weight each task/scene equally."""
    if not rows:
        raise ValueError("No evaluated windows")
    result = {"window_mean": {key: mean(row["primary"][key] for row in rows)
                              for key in METRIC_KEYS}}
    for name, fields in (("task", ("scene", "task_id")), ("scene", ("scene",))):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[field] for field in fields)].append(row["primary"])
        result[name + "_count"] = len(groups)
        result[name + "_macro_mean"] = {
            key: mean(mean(row[key] for row in group) for group in groups.values())
            for key in METRIC_KEYS}
    return result
