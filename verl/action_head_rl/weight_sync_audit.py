"""Optional vLLM worker diagnostics for the in-place weight reload gate."""
from __future__ import annotations

import hashlib
import json

import torch


class WeightSyncAuditExtension:
    def action_head_weight_fingerprint(self):
        """Hash actual post-load TP tensors, including kernel repacking."""
        digest = hashlib.sha256()
        count = 0
        for name, parameter in sorted(self.model_runner.get_model().named_parameters()):
            value = parameter.detach().cpu().contiguous()
            digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
            count += value.numel()
        return {"sha256": digest.hexdigest(), "parameter_elements": count}
