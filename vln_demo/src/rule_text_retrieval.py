"""Frozen text retrieval using the already installed Qwen embedding model.

The model retrieves category hypotheses, never geometry or navigation actions.
No scene IDs, task goals, training examples or per-scene aliases are inputs.
"""
import importlib.util
import os
import sys
from pathlib import Path
from threading import RLock

import numpy as np
from rule_lexicon import ALIASES

_ENCODERS = {}
_ENCODERS_LOCK = RLock()


def _shared_encoder(model_path):
    """One frozen CPU model per process; episode geometry never enters here."""
    key = (os.getpid(), str(model_path))
    with _ENCODERS_LOCK:
        if key not in _ENCODERS:
            _ENCODERS[key] = (_load_encoder(model_path), {}, RLock())
        return _ENCODERS[key]


def _load_encoder(model_path):
    import torch
    source = model_path / 'scripts/qwen3_vl_embedding.py'
    spec = importlib.util.spec_from_file_location('rule_qwen_embedding', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    encoder = module.Qwen3VLEmbedder.__new__(module.Qwen3VLEmbedder)
    encoder.default_instruction = 'Represent the identity and type of this indoor object.'
    encoder.max_length = 128
    encoder.model = module.Qwen3VLForEmbedding.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.float32,
        attn_implementation='sdpa').to('cpu').eval()
    encoder.processor = module.Qwen3VLProcessor.from_pretrained(
        str(model_path), local_files_only=True, padding_side='right')
    return encoder


class TextRetriever:
    def __init__(self, model_path):
        self.model_path = Path(model_path).resolve()
        # Share only the frozen text encoder and text-keyed vectors. Keep
        # per-episode semantic decisions and audit dictionaries independent.
        self.encoder, self.cache, self._encoder_lock = _shared_encoder(self.model_path)
        self.labels = tuple(ALIASES)
        self.min_score = .88
        self.min_margin = .02
        self.audit = {}

    def encode(self, texts):
        with self._encoder_lock:
            missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
            for start in range(0, len(missing), 16):
                batch = missing[start:start + 16]
                vectors = self.encoder.process([{'text': text} for text in batch]).float().cpu().numpy()
                self.cache.update(zip(batch, vectors))
            return np.stack([self.cache[text] for text in texts])

    def scores(self, queries, labels):
        return self.encode(queries) @ self.encode(labels).T

    def resolve_many(self, texts):
        if not texts:
            return {}
        missing = list(dict.fromkeys(text for text in texts if text not in self.audit))
        if missing:
            for text, scores in zip(missing, self.scores(missing, self.labels)):
                order = scores.argsort()[::-1]
                best, second = order[:2]
                margin = float(scores[best] - scores[second])
                self.audit[text] = dict(method='pretrained_text_retrieval', label=text,
                    family=self.labels[best], score=float(scores[best]), margin=margin,
                    accepted=bool(scores[best] >= self.min_score and margin >= self.min_margin),
                    alternatives=[dict(family=self.labels[i], score=float(scores[i])) for i in order[:3]])
        return {text: self.audit[text] if self.audit[text]['accepted'] else None for text in texts}
