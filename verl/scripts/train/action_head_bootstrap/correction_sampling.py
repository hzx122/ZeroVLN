"""Lightweight, deterministic dual-type sampling shared by merge/accumulation."""
from collections import Counter, OrderedDict
import random

TYPES = ("execution_relabel", "recovery")
CONTRACT = "v43_dual_anchor_turn_aligned_v1"


def grouped_anchors(rows):
    groups = OrderedDict()
    for row in rows:
        groups.setdefault(row["slow_anchor_id"], []).append(row)
    return groups


def type_stats(rows):
    windows = Counter(row.get("sample_type", "legacy_recovery") for row in rows)
    anchors = Counter()
    for group in grouped_anchors(rows).values():
        anchors[group[0].get("sample_type", "legacy_recovery")] += 1
    return {"anchors": dict(anchors), "windows": dict(windows)}


def is_dual(rows):
    dual = any(row.get("sample_type") in TYPES for row in rows)
    if dual:
        for group in grouped_anchors(rows).values():
            kinds = {row.get("sample_type") for row in group}
            if len(kinds) != 1 or not kinds <= set(TYPES):
                raise ValueError("mixed or unknown correction sample types")
            if any(row.get("decision_filter") != CONTRACT for row in group):
                raise ValueError("dual correction selection contract mismatch")
    return dual


def balanced_groups(rows, *, seed=0, previous=()):
    """Prefer the underrepresented anchor type; fall back without oversampling.

    Each anchor is visited once. Existing cumulative anchors count toward the
    1:1 target; a scarce type never causes duplicate data or extra collection.
    """
    if not is_dual(rows):
        return list(grouped_anchors(rows).values())
    if previous and not is_dual(previous):
        raise ValueError("cannot append dual data to a legacy cumulative pool")
    rng = random.Random(seed)
    buckets = {kind: [] for kind in TYPES}
    for group in grouped_anchors(rows).values():
        buckets[group[0]["sample_type"]].append(group)
    for groups in buckets.values():
        rng.shuffle(groups)
    counts = Counter(type_stats(previous)["anchors"])
    ordered = []
    while any(buckets.values()):
        kind = min((kind for kind in TYPES if buckets[kind]),
                   key=lambda kind: (counts[kind], TYPES.index(kind)))
        ordered.append(buckets[kind].pop())
        counts[kind] += 1
    return ordered


def select_fragments(rows, budget, *, seed=0, previous=()):
    groups = balanced_groups(rows, seed=seed, previous=previous)
    # Preserve the existing exact window cap; only the final anchor may be
    # partially selected. Its selected windows keep their original targets.
    selected, truncated = [], []
    for group in groups:
        remaining = budget - len(selected)
        if remaining <= 0:
            break
        selected.extend(group[:remaining])
        if len(group) > remaining:
            truncated.append(group[0]["slow_anchor_id"])
    return selected, truncated
