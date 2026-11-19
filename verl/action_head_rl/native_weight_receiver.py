"""Use upstream MoE bucket loading, adding our query publication barrier."""
import hashlib
import os

import torch
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from action_head_rl.direct_weight_sync import record_metadata
from action_head_rl.native_colocated_weights import native_ipc_handle


class NativeColocatedWeightExtension(vLLMColocateWorkerExtension):
    def _get_zmq_handle(self):
        return native_ipc_handle(os.environ['ACTION_HEAD_DIRECT_WEIGHT_SYNC_ROOT'], int(self.local_rank))

    def _update_weights(self, weights, peft_config, base_sync_done):
        weights = list(weights)
        for name, value in weights:
            if name in self._native_weight_names:
                raise ValueError('duplicate native policy tensor: ' + name)
            self._native_weight_names.add(name)
            self._native_weight_bytes += value.nbytes
            record_metadata(self._native_weight_digest, name, value)
        return super()._update_weights(weights, peft_config, base_sync_done)

    def update_weights_from_ipc(self, *, use_shm=False):
        config = self.model_runner.model_config
        if config.quantization is not None or config.dtype != torch.bfloat16:
            raise ValueError('v8 native publication requires an unquantized BF16 policy')
        self._native_weight_names = set()
        self._native_weight_bytes = 0
        self._native_weight_digest = hashlib.sha256()
        super().update_weights_from_ipc(use_shm=use_shm)
        torch.cuda.synchronize(self.device)
        if not self._native_weight_names:
            raise RuntimeError('native receiver consumed no policy weights')
        return dict(tp_rank=int(self.local_rank), pid=os.getpid(),
            tensor_count=len(self._native_weight_names), tensor_bytes=self._native_weight_bytes,
            metadata_sha256=self._native_weight_digest.hexdigest())

    def reload_native_query(self, expected_sha256):
        reload_query = getattr(self.model_runner.model, '_action_head_reload_query_checkpoint', None)
        if reload_query is None:
            raise RuntimeError('native publication lost its query hook')
        query = reload_query()
        if query['checkpoint_sha256'] != expected_sha256 or query['pending_query_phases']:
            raise RuntimeError('native publication query version/drain mismatch')
        return dict(tp_rank=int(self.local_rank), pid=os.getpid(), query=query)
