"""Inference lifetime adapter for the existing authoritative v8 RL provider."""
import hashlib
import json
import os
import uuid
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def boundary_tokens(model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    return (tuple(tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)),
            tokenizer.convert_tokens_to_ids('<|im_end|>'))


class EvaluationContinuousContext:
    def __init__(self, args, query_count):
        from action_head_rl.rollout import DecisionRoutingQueryStateProvider
        model = json.loads(Path(os.environ['ACTION_HEAD_EVAL_BRAIN_CONFIG']).read_text())
        text = model.get('text_config', model)
        self.continuous = getattr(args, 'action_head_as_tool_version', 'v8') == 'v8'
        self.provider = DecisionRoutingQueryStateProvider(args.query_cache,
            hidden_size=int(text['hidden_size']), layer_count=int(text['num_hidden_layers']),
            top_k=int(text['num_experts_per_tok']), train_scope=None,
            query_count=query_count, continuous_context=True) if self.continuous else None
        if self.provider is not None:
            self.provider.v8_prefix_cache = os.environ.get('ACTION_HEAD_V8_PREFIX_CACHE') == '1'
            self.provider.begin_episode(0)
        self.audit = []
        self.args, self.query_count = args, query_count
        self.compaction_enabled = os.environ.get('ACTION_HEAD_EVAL_CONTEXT_COMPACTION') == 'complete_turn_stride'
        self.compactions = []
        self.expected_prompt_tokens = None
        if self.compaction_enabled and self.continuous:
            self.boundaries = boundary_tokens(str(Path(os.environ['ACTION_HEAD_EVAL_BRAIN_CONFIG']).parent))
        self._new_context()

    def _new_context(self):
        self.compactions = []
        self.expected_prompt_tokens = None
        self.episode_id = uuid.uuid4().hex
        self.tokenize_nonce = uuid.uuid4().hex
        self.context = None
        if self.compaction_enabled:
            from action_head.context_compaction import TurnContext
            self.context = TurnContext(
                recent=int(os.environ.get('ACTION_HEAD_EVAL_CONTEXT_RECENT', '8')),
                frequency=int(os.environ.get('ACTION_HEAD_EVAL_CONTEXT_FREQUENCY', '4')),
                reserve_tokens=int(os.environ.get('ACTION_HEAD_EVAL_CONTEXT_RESERVE', '8192')))

    def begin_episode(self):
        if self.provider is not None:
            self.provider.begin_episode(0)
        self.audit = []
        self._new_context()

    def prepare_messages(self, messages, *, decision_index):
        """Sample canonical full turns only at a verified token-window boundary."""
        from closedloop_eval import training_matched_payload
        from verl_access.context_budget import tokenize_request
        from action_head.serve.continuous_context import merge_prefix_token_ids
        physical_window = int(os.environ.get('VLLM_MAX_MODEL_LEN', '131072'))
        window = int(os.environ.get('ACTION_HEAD_EVAL_CONTEXT_WINDOW', str(physical_window)))
        if window != physical_window:
            raise ValueError('compaction must use the actual serving window, not an earlier threshold')
        active = self.context.active_messages(messages)
        payload = training_matched_payload(active, self.query_count)
        measured = tokenize_request(self.args.vllm_url, payload, self.args.timeout,
            image_cache_nonce=self.tokenize_nonce)
        if measured.get('max_model_len') != physical_window:
            raise ValueError('actual serving window differs from evaluation contract')
        ids = measured.get('tokens')
        if not isinstance(ids, list) or measured.get('count') != len(ids):
            raise ValueError('serving tokenizer did not return authoritative token IDs')
        records = self.provider.decision_records if self.provider is not None else []
        if records:
            marker, end = self.boundaries
            ids = merge_prefix_token_ids(ids, records[-1]['text_history_ids'].tolist(),
                assistant_marker=list(marker), end_token=end)
        def count_tokens(candidate):
            result = tokenize_request(self.args.vllm_url,
                training_matched_payload(candidate, self.query_count), self.args.timeout)
            if result.get('max_model_len') != physical_window:
                raise ValueError('serving window changed during compaction')
            return result['count']
        selected, audit = self.context.prepare_messages(messages, count_tokens=count_tokens,
            input_tokens=len(ids), max_model_len=window,
            max_completion_tokens=payload['max_tokens'])
        if audit['requires_new_native_context']:
            # This is a new conditional context, not a continuation of the old
            # token/R3 stream. No environment reset or tool call occurs here.
            if self.provider is not None:
                self.provider.end_episode()
                self.provider.begin_episode(0)
            self.tokenize_nonce = uuid.uuid4().hex
            self.compactions.append(dict(decision_index=decision_index, **audit))
        self.expected_prompt_tokens = audit['input_tokens_after']
        return selected

    def archive_messages(self, messages, *, case, seed):
        from action_head.protocol_v8 import strip_context_images
        from verl_access.vlm_policy import to_openai_messages
        from rule_teacher_training_handoff import save, sha
        root = Path(os.environ['ACTION_HEAD_EVAL_CONTEXT_ARCHIVE']).resolve()
        artifact_root = Path(os.environ.get('RL_ARTIFACT_ROOT', '/data/closed_loop_3dgs_rl')).resolve()
        if not root.is_relative_to(artifact_root):
            raise ValueError('canonical evaluation history must be on the data disk')
        path = root/(self.episode_id+'.json')
        # v6 legitimately retains fast intermediate frames in its canonical
        # history. Conversion removes private tags without pruning any frame.
        canonical = strip_context_images(messages) if self.continuous else messages
        save(path, dict(logical_id=case.get('logical_id'), seed=seed,
            messages=to_openai_messages(canonical),
            compactions=self.compactions, native_decisions=self.audit))
        return dict(path=str(path), sha256=sha(path))

    def capture_decision(self, **kwargs):
        previous = self.provider.decision_records
        old_ids = previous[-1]['text_history_ids'] if previous else None
        self.provider.capture_decision(**kwargs)
        record = self.provider.decision_records[-1]
        if (self.expected_prompt_tokens is not None
                and record['text_prompt_length'] != self.expected_prompt_tokens):
            raise ValueError('native prompt length differs from serving-tokenizer compaction accounting')
        ids = record['text_history_ids']
        matches = old_ids is None or ids[:len(old_ids)].tolist() == old_ids.tolist()
        if not matches:
            raise ValueError('v8 evaluation changed the authoritative parent prefix')
        self.audit.append(dict(decision_index=kwargs['decision_index'],
            context_segment_id=self.context.segment if self.context is not None else 0,
            query_request_id=kwargs['response']['query_request_id'],
            prompt_tokens=record['text_prompt_length'], history_tokens=len(ids),
            history_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            exact_parent_prefix=matches))
        # Inference needs only the last native parent and current query hidden.
        # Keep the provider's artifact ownership and release rules unchanged.
        self.provider.decision_records[:] = [dict(text_history_ids=ids,
            slow_response=kwargs['response'])]


def install(evaluator):
    context = EvaluationContinuousContext(evaluator.args, evaluator.runtime.query_count)
    if context.provider is not None:
        from verl_access.vlm_policy import encode_canonical_images
        evaluator.canonical_image_preparer = encode_canonical_images
        evaluator.decision_observer = context.capture_decision
        evaluator.continuous_prefix_provider = context.provider.continuous_prefix_ids
        evaluator.runtime.tool_executor.slow_condition_provider = context.provider
    if context.compaction_enabled:
        evaluator.evaluation_context_preparer = context.prepare_messages
        evaluator.evaluation_context_observer = context.archive_messages
    return context
