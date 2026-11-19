"""Raw scheduling work counters; never inspect tensor values or loss weights."""
from __future__ import annotations

from numbers import Real


def condition_workload_metrics(groups, packed):
    """Count one actual root forward, including query tails and dummy work.

    ``packed`` is the existing anchor TensorDict. Its all_positions numel is
    the actual R3 route-token allocation, including cross-rank left padding.
    A zero-advantage real record is still nonpadding. Counts are not loss
    denominators or FLOPs. Tensor-valued validity metadata is left unknown;
    observing it must never add a device-to-host synchronization.
    """
    rows = [row for group in groups for row in group]

    def padding(row):
        value = row.get("sample_valid", 1.0)
        return value == 0 if isinstance(value, Real) else None

    def token_rows(group):
        ids = group[0].get("all_input_ids")
        if ids is not None and hasattr(ids, "numel"):
            return int(ids.numel())  # Shape metadata only, even on a device.
        return None

    dummy_rows = [padding(row) for row in rows]
    dummy_groups = []
    for group in groups:
        values = [padding(row) for row in group]
        dummy_groups.append(False if False in values else
                            None if None in values else True)
    tokens = [token_rows(group) for group in groups]
    positions = packed.get("all_positions")
    route_slots = int(positions.numel()) if positions is not None else None
    known_unpadded = sum(value for value in tokens if value is not None)
    complete = route_slots is not None and all(value is not None for value in tokens)
    fm_rows = [(row, dummy) for row, dummy in zip(rows, dummy_rows)
               if row.get("record_kind") != "text_policy"]
    return {
        "workload_root_forwards": 1,
        "workload_condition_count": len(groups),
        "workload_dummy_condition_count": sum(value is True for value in dummy_groups),
        "workload_nonpadding_condition_count": sum(value is False for value in dummy_groups),
        "workload_condition_classification_unknown": sum(value is None for value in dummy_groups),
        "workload_record_count": len(rows),
        "workload_dummy_record_count": sum(value is True for value in dummy_rows),
        "workload_record_classification_unknown": sum(value is None for value in dummy_rows),
        "workload_fm_chunk_rows": len(fm_rows),
        "workload_dummy_fm_chunk_rows": sum(dummy is True for _, dummy in fm_rows),
        "workload_unpadded_route_token_rows": known_unpadded,
        "workload_unpadded_route_conditions_unknown": sum(value is None for value in tokens),
        "workload_dummy_unpadded_route_token_rows": sum(
            value for value, dummy in zip(tokens, dummy_groups)
            if value is not None and dummy is True),
        "workload_packed_route_token_slots": route_slots if route_slots is not None else 0,
        "workload_route_batches_unavailable": int(route_slots is None),
        "workload_left_padding_route_token_slots": route_slots - known_unpadded if complete else 0,
        "workload_padding_route_batches_unavailable": int(not complete),
    }


def sum_workload_metrics(metric_rows):
    """Preserve raw additive counters instead of averaging microbatches."""
    result = {}
    for row in metric_rows:
        for key, value in row.items():
            if key.startswith("workload_"):
                result[key] = result.get(key, 0) + value
    return result
