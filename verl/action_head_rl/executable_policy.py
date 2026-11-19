"""One executable v6 tool decision, with an explicit learned STOP choice.

The grammar depends only on the public consecutive-turn counter. Goal labels
never enter sampling. The exact token automaton is replayed by the learner;
syntax/query carriers have zero policy gradient. No rejected-call resampling.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os

SCHEMA = 'three-source-boundary-stop-v1'
POLICY_SCHEMA = 'v6-executable-tools-v1'
TOKENIZER = '/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B'
MAX_STRING_CHARS = 96
PREFIX = '<tool_call>{"name":"'


def enabled():
    public = os.environ.get('ACTION_HEAD_EXECUTABLE_POLICY', '')
    if public not in ('', POLICY_SCHEMA):
        raise ValueError('unknown executable inference policy')
    return public == POLICY_SCHEMA or os.environ.get('ACTION_HEAD_FINAL_CONTRACT', '') == SCHEMA


def request_policy(consecutive_turn_calls):
    if type(consecutive_turn_calls) is not int or consecutive_turn_calls < 0:
        raise ValueError('public consecutive turn count must be a nonnegative integer')
    return dict(schema=POLICY_SCHEMA, allow_turn=consecutive_turn_calls < 2)


def validate_request(value):
    # vLLM's public vllm_xargs schema accepts scalar values, not nested objects.
    # Decode the wire representation at both sampler and capture boundaries.
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError('invalid public executable-policy specification') from exc
    if (not isinstance(value, dict) or set(value) != {'schema', 'allow_turn'}
            or value['schema'] != POLICY_SCHEMA or type(value['allow_turn']) is not bool):
        raise ValueError('invalid public executable-policy specification')
    return value


def encode_request(value):
    """Keep the policy unchanged while meeting the actual HTTP request schema."""
    return json.dumps(validate_request(value), sort_keys=True, separators=(',', ':'))


def grammar(tool):
    # No escaped quotation/control characters are needed in these natural
    # language fields. A non-whitespace first character guarantees nonempty
    # arguments. Bounded strings also bound worst-case generation length.
    # Build the literals through JSON escaping rather than ambiguous EBNF
    # backslash literals; character-class support is verified below in tests.
    # Angle brackets can introduce nested tool/thinking delimiters into the
    # real XML-like parser even while the JSON is valid. Python's tool argument
    # validators strip Unicode whitespace as well as ASCII; a string composed
    # only of NBSP/ideographic spaces must never count as an executable reason.
    # XGrammar 0.1.29 miscompiles Unicode character-class ranges and optimizes
    # singleton high-byte classes into different UTF-8 literals. Use audited
    # non-singleton byte ranges instead (no overlong forms or surrogates).
    # The first scalar uses a conservative BMP subset excluding every Unicode
    # whitespace block. This is a fixed grammar alphabet, not all BMP/Unicode;
    # Chinese, Arabic, Russian and the U+0800 regression are explicitly tested.
    unicode = (r'[\xc2-\xdf] [\x80-\xbf] | [\xe0-\xe1] [\xa0-\xbf] [\x80-\xbf] | '
               r'[\xe1-\xec] [\x80-\xbf]{2} | [\xec-\xed] [\x80-\x9f] [\x80-\xbf] | '
               r'[\xee-\xef] [\x80-\xbf]{2}')
    nonspace = (r'[\xc3-\xdf] [\x80-\xbf] | [\xe0-\xe1] [\xa0-\xbf] [\x80-\xbf] | '
                r'[\xe1-\xe3] [\x82-\x99\x9b-\xbf] [\x80-\xbf] | '
                r'[\xe4-\xec] [\x80-\xbf]{2} | [\xec-\xed] [\x80-\x9f] [\x80-\xbf] | '
                r'[\xee-\xef] [\x80-\xbf]{2}')
    string = ('string ::= ' + json.dumps('"') + ' first character{0,'
              + str(MAX_STRING_CHARS - 1) + '} ' + json.dumps('"')
              + '\nfirst ::= ' + r'[\x21\x23-\x3b\x3d\x3f-\x5b\x5d-\x7f] | ' + nonspace
              + '\ncharacter ::= ' + r'[\x20-\x21\x23-\x3b\x3d\x3f-\x5b\x5d-\x7f] | ' + unicode)
    if tool == 'run':
        root = ('root ::= ' + json.dumps('{"target":') + ' string '
                + json.dumps(',"where":') + ' string '
                + json.dumps(',"plan":[') + ' string (' + json.dumps(',')
                + ' string){0,2} ' + json.dumps(']}'))
    elif tool == 'stop':
        root = 'root ::= ' + json.dumps('{"reason":') + ' string ' + json.dumps('}')
    elif tool == 'turn':
        angles = ' | '.join(json.dumps(str(n / 2).removesuffix('.0')) for n in range(1, 181))
        root = ('root ::= ' + json.dumps('{"direction":') + ' ('
                + json.dumps('"left"') + ' | ' + json.dumps('"right"') + ') '
                + json.dumps(',"degrees":') + ' angle ' + json.dumps('}')
                + '\nangle ::= ' + angles)
    else:
        raise ValueError('unsupported tool')
    return root + '\n' + string


class Codec:
    def __init__(self, tokenizer, vocab_size):
        from importlib.metadata import version
        import xgrammar as xgr
        self.tokenizer, self.vocab_size = tokenizer, int(vocab_size)
        self.prefix = tuple(tokenizer.encode(PREFIX, add_special_tokens=False))
        self.choice = {name: tokenizer.encode(name, add_special_tokens=False)
                       for name in ('stop', 'run', 'turn')}
        if any(len(ids) != 1 for ids in self.choice.values()):
            raise ValueError('final policy requires unique single-token stop/run/turn branches')
        self.choice = {name: ids[0] for name, ids in self.choice.items()}
        if len(set(self.choice.values())) != 3:
            raise ValueError('tool branch tokens must differ')
        self.suffix = {name: tuple(tokenizer.encode(
            ('_action_head' if name == 'run' else '') + '","arguments":',
            add_special_tokens=False)) for name in self.choice}
        self.end = tuple(tokenizer.encode('}</tool_call>', add_special_tokens=False))
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=self.vocab_size)
        compiler = xgr.GrammarCompiler(info, max_threads=1)
        self.compiled = {name: compiler.compile_grammar(grammar(name)) for name in self.choice}
        self.digest = hashlib.sha256(json.dumps(dict(
            schema=POLICY_SCHEMA, vocab_size=self.vocab_size, prefix=self.prefix,
            choice=self.choice, suffix=self.suffix, end=self.end,
            tokenizer_sha256=hashlib.sha256(
                tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            special_ids=sorted(tokenizer.all_special_ids),
            xgrammar_version=version('xgrammar'),
            grammar={name: grammar(name) for name in self.choice}),
            sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=2)
def codec():
    from transformers import AutoConfig, AutoTokenizer
    path = os.environ.get('ACTION_HEAD_FINAL_TOKENIZER', TOKENIZER)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    vocab_size = getattr(getattr(config, 'text_config', config), 'vocab_size')
    return Codec(tokenizer, vocab_size)


class DecisionGrammar:
    """Stateful, batch-slot-independent automaton; accepts only actual IDs."""
    def __init__(self, spec, *, tokenizer_codec=None):
        import xgrammar as xgr
        self.spec = validate_request(spec)
        self.codec = tokenizer_codec or codec()
        self.xgr = xgr
        self.output = []
        self.phase, self.offset, self.tool = 'prefix', 0, None
        self.matcher = None
        self.bitmask = xgr.allocate_token_bitmask(1, self.codec.vocab_size)

    @property
    def complete(self):
        return self.phase == 'complete'

    def allowed(self):
        """Return explicit IDs or XGrammar's packed mask and phase label."""
        if self.phase == 'prefix':
            return (self.codec.prefix[self.offset],), 'syntax'
        if self.phase == 'choice':
            return tuple(v for k, v in self.codec.choice.items()
                         if k != 'turn' or self.spec['allow_turn']), 'choice'
        if self.phase == 'suffix':
            return (self.codec.suffix[self.tool][self.offset],), 'syntax'
        if self.phase == 'body':
            self.matcher.fill_next_token_bitmask(self.bitmask)
            return self.bitmask, 'body'
        if self.phase == 'end':
            return (self.codec.end[self.offset],), 'syntax'
        raise ValueError('completed executable decision has no more text tokens')

    def accept(self, token):
        token = int(token)
        allowed, phase = self.allowed()
        if phase != 'body' and token not in allowed:
            raise ValueError('token is outside executable grammar')
        if phase == 'body':
            if not self.matcher.accept_token(token):
                raise ValueError('token is outside executable argument grammar')
            if self.matcher.is_terminated():
                self.phase, self.offset = 'end', 0
        elif self.phase == 'choice':
            self.tool = next(k for k, v in self.codec.choice.items() if v == token)
            self.phase, self.offset = 'suffix', 0
            self.matcher = self.xgr.GrammarMatcher(
                self.codec.compiled[self.tool], terminate_without_stop_token=True)
        else:
            self.offset += 1
            sequence = (self.codec.prefix if self.phase == 'prefix' else
                        self.codec.suffix[self.tool] if self.phase == 'suffix' else self.codec.end)
            if self.offset == len(sequence):
                self.phase = {'prefix': 'choice', 'suffix': 'body', 'end': 'complete'}[self.phase]
                self.offset = 0
        self.output.append(token)

    def sync(self, output_ids):
        if list(output_ids[:len(self.output)]) != self.output:
            raise ValueError('executable grammar output prefix changed')
        for token in output_ids[len(self.output):]:
            self.accept(token)

    def mask_logits(self, logits):
        import torch
        allowed, phase = self.allowed()
        if logits.numel() != self.codec.vocab_size:
            raise ValueError('sampler vocabulary differs from executable policy')
        if phase == 'body':
            self.xgr.apply_token_bitmask_inplace(logits.unsqueeze(0), self.bitmask.to(logits.device))
        else:
            keep = logits[list(allowed)].clone()
            logits.fill_(float('-inf'))
            logits[list(allowed)] = keep
        if not torch.isfinite(logits).any():
            raise ValueError('executable policy has no finite allowed token')


def new_sampler_state(extra, *, budget, query_count):
    spec = extra.get('action_head_executable_policy')
    if spec is None:
        if enabled():
            raise ValueError('final-policy request missing executable contract')
        return None
    spec = validate_request(spec)
    if extra.get('action_head_forced_completion_ids') or os.environ.get('ACTION_HEAD_FORCED_COMPLETION_IDS'):
        raise ValueError('diagnostic forcing is incompatible with final policy')
    # Five strings, up to 96 Unicode scalars each, at most four UTF-8 byte
    # tokens per scalar plus JSON wrappers. Fail before any action if the
    # context cannot contain a complete legal decision and latent tail.
    if budget < 2200 + query_count + 1:
        raise ValueError('remaining context is too short for bounded executable grammar')
    return DecisionGrammar(spec)


def apply_sampler_state(state, output_ids, logits, *, placeholder, eos, query_count):
    """Mask natural choices, then reuse the existing AH-only query tail."""
    natural = list(output_ids)
    trailing = 0
    while natural and natural[-1] == placeholder:
        natural.pop(); trailing += 1
    state.sync(natural)
    if not state.complete:
        if trailing:
            raise ValueError('query tail preceded a complete legal call')
        state.mask_logits(logits)
        return
    if trailing > query_count or (trailing and state.tool != 'run'):
        raise ValueError('invalid executable query tail')
    target = placeholder if state.tool == 'run' and trailing < query_count else eos
    keep = logits[target].clone()
    logits.fill_(float('-inf')); logits[target] = keep


def capture_policy_tensors(extra, *, all_ids, prompt, end):
    import torch
    spec = extra.get('action_head_executable_policy')
    if spec is None:
        if enabled():
            raise ValueError('final decision capture lacks executable sampling provenance')
        return {}
    spec = validate_request(spec)
    state = DecisionGrammar(spec)
    state.sync(all_ids[prompt:end])
    if not state.complete:
        raise ValueError('incomplete executable decision cannot enter training')
    return dict(decision_executable_policy=torch.tensor([1, int(spec['allow_turn'])], dtype=torch.int64),
                decision_executable_digest=torch.tensor(list(bytes.fromhex(state.codec.digest)), dtype=torch.uint8))


def validate_captured_policy(tensors):
    import torch
    metadata = tensors.get('decision_executable_policy')
    if metadata is None:
        if enabled():
            raise ValueError('final decision has no executable-policy metadata')
        return None
    if metadata.tolist() not in ([1, 0], [1, 1]):
        raise ValueError('unknown executable-policy version or turn availability')
    expected = torch.tensor(list(bytes.fromhex(codec().digest)), dtype=torch.uint8)
    if not torch.equal(tensors.get('decision_executable_digest', torch.empty(0)).cpu(), expected):
        raise ValueError('sampler/learner executable token grammar differs')
    return metadata.clone().cpu()
