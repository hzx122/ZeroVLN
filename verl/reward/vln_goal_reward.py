#!/usr/bin/env python3
"""Phase-1 reward for VLN goal-point GRPO.

The policy is asked to emit ``goal point: [x, y]`` (see the SFT answer format in
exp13-vln_ce_ar_fm). We parse that pair and score it against the ground-truth
goal point. Coordinates live in a fixed ~[0, 1000) canvas (matching the
<loc_x_0..999>/<loc_y_0..999> bins), NOT the 640x480 image, so distance is
normalized by the canvas diagonal, never clipped to image resolution.

Signature matches the verl custom-reward convention used across VLA-RLVR
(causal_mcq_reward.compute_score): returns a dict whose ``score`` is the training
signal and whose extra keys are logged to TensorBoard.
"""

from __future__ import annotations

import math
import re
from typing import Any

CANVAS = 1000.0                       # goal-point coordinate space edge (0..999 bins)
CANVAS_DIAG = CANVAS * math.sqrt(2.0)  # ~1414, used to normalize L2
SIGMA_FRAC = 0.15                      # gaussian width as a fraction of the diagonal
FORMAT_WEIGHT = 0.1                    # small bonus for well-formed output

# "goal point: [708, 427]" (case-insensitive, tolerant of spaces / floats / signs)
_GOAL_RE = re.compile(
    r"goal\s*point\s*[:=]?\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
    re.IGNORECASE,
)


def parse_goal_point(text: str) -> tuple[float, float] | None:
    """Extract the LAST ``goal point: [x, y]`` pair from a completion, or None."""
    if not text:
        return None
    matches = _GOAL_RE.findall(text)
    if not matches:
        return None
    x, y = matches[-1]
    return float(x), float(y)


def _gt_goal(ground_truth: Any, extra_info: Any) -> tuple[float, float] | None:
    for src in (ground_truth, extra_info):
        if isinstance(src, dict):
            g = src.get("goal_point")
            if isinstance(g, (list, tuple)) and len(g) == 2:
                try:
                    return float(g[0]), float(g[1])
                except (TypeError, ValueError):
                    continue
    return None


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: Any = None,
    **_: Any,
) -> dict[str, Any]:
    gt = _gt_goal(ground_truth, extra_info)
    pred = parse_goal_point(solution_str or "")
    format_ok = 1.0 if pred is not None else 0.0

    if gt is None:
        # No supervision available -> only reward well-formedness.
        return {"score": FORMAT_WEIGHT * format_ok, "format_ok": format_ok, "goal_l2": -1.0, "acc": 0.0}

    if pred is None:
        return {"score": 0.0, "format_ok": 0.0, "goal_l2": -1.0, "acc": 0.0}

    dist = math.hypot(pred[0] - gt[0], pred[1] - gt[1])
    norm = dist / CANVAS_DIAG
    goal_reward = math.exp(-(norm * norm) / (2.0 * SIGMA_FRAC * SIGMA_FRAC))
    score = FORMAT_WEIGHT * format_ok + (1.0 - FORMAT_WEIGHT) * goal_reward

    return {
        "score": float(score),
        "format_ok": format_ok,
        "goal_l2": float(dist),
        "goal_reward": float(goal_reward),
        "acc": 1.0 if norm <= SIGMA_FRAC else 0.0,  # within ~1 sigma counts as a hit
    }


if __name__ == "__main__":
    # tiny self-check
    gt = {"goal_point": [708, 427]}
    for s in [
        "goal point: [708, 427]; <flow_matching_policy>",
        "blah goal point: [700, 430] end",
        "goal point: [0, 0]",
        "no coordinates here",
    ]:
        print(f"{s!r:60s} -> {compute_score(ground_truth=gt, solution_str=s)}")
