"""Versioned FM training seeds; validation uses its own fixed seed contract."""
import hashlib


def fm_training_seed(base_seed, call_index, anchor_id, policy="legacy"):
    if policy == "legacy":
        return int(base_seed) + int(call_index) * 1000
    if policy != "anchor_v1":
        raise ValueError(f"unsupported FM training seed policy: {policy}")
    # Anchor identity separates ranks; visit index changes the draw on replay.
    # This remains identical across candidates with the same data order.
    key = f"fm-anchor-v1:{int(base_seed)}:{int(call_index)}:{anchor_id}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**63 - 1000000)
