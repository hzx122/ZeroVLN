"""Align opt-in joint/FM/text forwards without inspecting rollout payloads.

Strict token/route/decision pairing belongs to the descriptor producer. This
module only copies record dictionaries and aligns all learner call sequences;
opaque tensor/object-store references are preserved and never resolved here.
"""
from __future__ import annotations

import math
from typing import Any


FAMILY_ORDER = ("joint", "fm", "text")


def shard_joint_policy_groups(
    groups: list[dict[str, Any]], world: int, *, logical_valid: float,
    balance_by_token_flops: bool = False,
    condition_batch_size: int = 1, max_padded_tokens: int = 0,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Return equal-length flat rank streams and their scheduling audit.

    Each descriptor has ``family``, unique ``group_id``, positive ``token_count``,
    ``fm_records`` and ``text_record``. Joint groups have both branches, FM groups
    only action chunks, and text groups one text record. Every rank executes the
    same family, full padded token width and FM chunk-loop count in each round.
    Rank padding runs a complete valid-shaped graph with both objective weights
    zero. Chunk padding adds FM records only, so text is never counted twice.

    ``logical_valid`` is the original pre-filter/pre-pairing global episode mass
    supplied by the trainer. Neither group counts nor padding renormalize it.
    """
    if isinstance(world, bool) or not isinstance(world, int) or world <= 0:
        raise ValueError("learner world size must be a positive integer")
    if isinstance(logical_valid, bool) or not math.isfinite(logical_valid) or logical_valid <= 0:
        raise ValueError("original logical_valid must be finite and positive")
    if not groups:
        raise ValueError("joint train batch contains no groups")
    if condition_batch_size not in (1, 2, 4, 8) or max_padded_tokens < 0:
        raise ValueError("invalid joint condition packing bounds")

    families: dict[str, list[dict[str, Any]]] = {name: [] for name in FAMILY_ORDER}
    group_ids: set[str] = set()
    condition_owners: dict[str, str] = {}
    for descriptor in groups:
        family, group_id = descriptor.get("family"), descriptor.get("group_id")
        width = descriptor.get("token_count")
        fm, text = descriptor.get("fm_records"), descriptor.get("text_record")
        if family not in FAMILY_ORDER:
            raise ValueError(f"unknown joint scheduling family: {family!r}")
        if not isinstance(group_id, str) or not group_id or group_id in group_ids:
            raise ValueError("group_id must be a unique nonempty string")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("descriptor token_count must be a positive integer")
        if not isinstance(fm, list):
            raise ValueError("descriptor fm_records must be a list")
        if ((family == "joint" and (not fm or not isinstance(text, dict)))
                or (family == "fm" and (not fm or text is not None))
                or (family == "text" and (fm or not isinstance(text, dict)))):
            raise ValueError("descriptor branches do not match its family")
        if any(not isinstance(row, dict) or row.get("record_kind") == "text_policy" for row in fm):
            raise ValueError("FM branch contains a text/non-record payload")
        if text is not None and text.get("record_kind") != "text_policy":
            raise ValueError("text branch must contain a text_policy record")
        records = fm + ([text] if text is not None else [])
        for row in records:
            condition_id = row.get("condition_id")
            if not isinstance(condition_id, str) or not condition_id:
                raise ValueError("every record requires an explicit condition_id")
            if condition_id in condition_owners and condition_owners[condition_id] != group_id:
                raise ValueError("a condition_id cannot belong to multiple groups")
            condition_owners[condition_id] = group_id
        if len({row["condition_id"] for row in fm}) > 1:
            raise ValueError("a descriptor cannot merge multiple FM conditions")
        if text is not None and fm and text["condition_id"] == fm[0]["condition_id"]:
            raise ValueError("joint text and FM branches require distinct condition identities")
        group_ids.add(group_id)
        families[family].append({**descriptor, "fm_records": list(fm)})

    rank_records: list[list[dict[str, Any]]] = [[] for _ in range(world)]
    stats: dict[str, Any] = {
        "logical_valid": logical_valid,
        "family_order": list(FAMILY_ORDER),
        "real_groups": {name: len(families[name]) for name in FAMILY_ORDER},
        "rank_padding_groups": {name: 0 for name in FAMILY_ORDER},
        "chunk_padding_records": 0,
        "rounds": [],
    }
    used_ids = group_ids | set(condition_owners)
    next_padding_id = 0

    def fresh_identity(label: str) -> str:
        nonlocal next_padding_id
        while True:
            identity = f"__joint_schedule_padding__/{label}/{next_padding_id}"
            next_padding_id += 1
            if identity not in used_ids:
                used_ids.add(identity)
                return identity

    def rank_padding(template: dict[str, Any]) -> dict[str, Any]:
        new_group_id = fresh_identity("group")
        fm_id, text_id = fresh_identity("fm"), fresh_identity("text")
        return {**template, "group_id": new_group_id,
                "fm_records": [dict(row, condition_id=fm_id, sample_valid=0.0, advantage=0.0)
                               for row in template["fm_records"]],
                "text_record": (dict(template["text_record"], condition_id=text_id,
                                     sample_valid=0.0, advantage=0.0)
                                if template["text_record"] is not None else None)}

    for family in FAMILY_ORDER:
        phase = families[family]
        if not phase:
            continue
        # Chunk-count ordering is the legacy non-token-balanced scheduling rule.
        phase.sort(key=lambda group: len(group["fm_records"]), reverse=True)
        template = min(phase, key=lambda group: (group["token_count"], len(group["fm_records"])))
        while len(phase) % world:
            phase.append(rank_padding(template))
            stats["rank_padding_groups"][family] += 1
        if balance_by_token_flops:
            import torch
            from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions
            costs = calculate_workload(torch.tensor([group["token_count"] for group in phase],
                                                    dtype=torch.int64)).tolist()
            order = sorted(range(len(phase)), key=lambda index: costs[index], reverse=True)
            ordered = [phase[index] for index in order]
            partitions = get_seqlen_balanced_partitions([int(costs[index]) for index in order],
                                                        world, equal_size=True)
            rank_groups = [[ordered[index] for index in partition] for partition in partitions]
        else:
            rank_groups = [phase[rank::world] for rank in range(world)]
        if len({len(rows) for rows in rank_groups}) != 1:
            raise RuntimeError("joint scheduling produced unequal rank group counts")
        # Merge adjacent globally aligned rounds, never independently on ranks.
        # Keep the SDE loop count fixed; changing it introduces extra zero work.
        group_index = 0
        while group_index < len(rank_groups[0]):
            def round_at(index):
                return [rank_groups[rank][index] for rank in range(world)]
            wave = round_at(group_index)
            width = max(group["token_count"] for group in wave)
            chunk_count = max(len(group["fm_records"]) for group in wave)
            batch_size = 1
            while (batch_size < condition_batch_size
                   and group_index + batch_size < len(rank_groups[0])):
                next_wave = round_at(group_index + batch_size)
                next_width = max(width, max(group["token_count"] for group in next_wave))
                if (max(len(group["fm_records"]) for group in next_wave) != chunk_count
                        or (max_padded_tokens and next_width * (batch_size + 1) > max_padded_tokens)):
                    break
                width = next_width
                batch_size += 1
            pack_id = len(stats["rounds"])
            stats["rounds"].append({"family": family, "pack_batch": pack_id,
                                    "token_width": width, "fm_chunks_per_rank": chunk_count,
                                    "records_per_rank": batch_size * (chunk_count + int(family != "fm")),
                                    **({"batch_size": batch_size} if condition_batch_size > 1 else {})})
            for offset in range(batch_size):
                for rank, group in enumerate(round_at(group_index + offset)):
                    rows = list(group["fm_records"])
                    while len(rows) < chunk_count:
                        rows.append(dict(rows[0], sample_valid=0.0, advantage=0.0))
                        stats["chunk_padding_records"] += 1
                    if group["text_record"] is not None:
                        rows.append(group["text_record"])
                    for row in rows:
                        rank_records[rank].append({**row, "_joint_family": family,
                            "_joint_group_id": group["group_id"], "_condition_pack_batch": pack_id,
                            "_condition_pack_token_width": width})
            group_index += batch_size
    stats["rank_record_counts"] = [len(rows) for rows in rank_records]
    if len(set(stats["rank_record_counts"])) != 1:
        raise RuntimeError("joint scheduling produced unequal flat rank stream lengths")
    return rank_records, stats
