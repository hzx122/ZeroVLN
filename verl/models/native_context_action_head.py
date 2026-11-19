"""v5 SFT: frozen native frontend, trainable context pooler, shared FM loss."""
import os

import torch
from torch import nn

from action_head.context_encoder import ContextAggregator, NativeQwenFrontend, CachedNativeFrontend
from models.qwen36_dualvln_action_head import Qwen36DualVlnActionHead
from models.qwen36_fast_slow_action_head import Qwen36FastSlowActionHead


class NativeContextTrainModel(Qwen36DualVlnActionHead):
    def __init__(self, frontend, aggregator, head):
        nn.Module.__init__(self)
        self.frontend = frontend.requires_grad_(False).eval()
        self.context_aggregator = aggregator
        self.action_head = head
        # Native [B,16,384] conditions bypass the old 2048->384 query adapter.
        self.action_head.slow_query_adapter.requires_grad_(False)
        self._batch_parity_checked = True

    @classmethod
    def from_environment(cls):
        from transformers import AutoConfig
        path = os.environ['SLOW_MODEL']
        cached = os.environ.get('ACTION_HEAD_NATIVE_CACHE')
        frontend = (CachedNativeFrontend if cached else NativeQwenFrontend)(path, device='cpu', dtype=torch.bfloat16)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(os.environ.get('ACTION_HEAD_NATIVE_SEED', '20260908')))
            aggregator = ContextAggregator()
        initial = os.environ.get('ACTION_HEAD_AGGREGATOR_CHECKPOINT')
        if initial:
            aggregator.load_state_dict(torch.load(initial, map_location='cpu', weights_only=True))
        head = Qwen36FastSlowActionHead.from_pretrained(os.environ['ACTION_HEAD_HEAD_CHECKPOINT'])
        result = cls(frontend, aggregator, head)
        result.config = AutoConfig.from_pretrained(path, local_files_only=True)
        # Only the frozen vision blocks need layer wrapping; no language model exists.
        # VERL also wraps nn.Embedding leaves. Do not select their parent
        # aggregator: its pre-order wrapper would otherwise shard modality
        # embeddings twice. Aggregator/head parameters belong to root FSDP.
        result._no_split_modules = ([type(frontend.vision_model.blocks[0]).__name__]
                                    if not cached else ['_QueryPooler'])
        return result

    @property
    def qwen(self):
        return self.frontend

    def train(self, mode=True):
        super().train(mode)
        self.frontend.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _forward_slow_anchors(self, slow, image_counts, pixel_counts):
        if image_counts is None or pixel_counts is None:
            raise ValueError('native SFT requires per-anchor image boundaries')
        if torch.any(image_counts > 11):
            raise ValueError('v5 input exceeds the shared 8+3 image window')
        if 'cached_visual_tokens' in slow or os.environ.get('ACTION_HEAD_NATIVE_BATCHED') == '1':
            with torch.no_grad():
                hidden = self.frontend(slow)
            return self.context_aggregator(hidden, slow['attention_mask'].bool(),
                                           slow['input_ids'].eq(self.frontend.image_id))
        conditions = []
        for index in range(slow['input_ids'].shape[0]):
            inputs = self._slice_slow_anchor(slow, index, image_counts, pixel_counts)
            with torch.no_grad():
                hidden = self.frontend(inputs)
            conditions.append(self.context_aggregator(
                hidden, inputs['attention_mask'].bool(),
                inputs['input_ids'].eq(self.frontend.image_id)))
        return torch.cat(conditions)

    def _encode_condition(self, slow, reference, current, reference_mask, current_mask):
        return self.action_head.encode_condition(
            None, reference, current, reference_mask, current_mask,
            slow_condition_tokens=slow)
