"""Bounded, opt-in CPU audit export; never part of the policy objective.

The rollout producer owns raw tensors before Plasma packing. The driver later
commits the independently computed branch targets, bound to the raw file hash.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import math

import numpy as np
from PIL import Image
from pathlib import Path

import torch

from action_head_rl.joint_policy import annotate_rollout_decisions
from action_head_rl.text_policy import is_text_record


ARTIFACTS = Path('/data/zerovln/artifacts')
SCHEMA = 'v6-joint-complete-fixture-v1'
MAX_CAPTURE_EPISODES = 32


def fixture_capture_spec(root, *, count, capture_step, first_step, step, rollout_index):
    if not root or step != (capture_step or first_step) or rollout_index >= count:
        return None
    return {'directory': str(Path(root) / f'step_{step:06d}_rollout_{rollout_index:04d}'),
            'step': int(step), 'rollout_index': int(rollout_index)}


def validate_fixture_options(root, count, capture_step):
    if (type(count) is not int or not 1 <= count <= MAX_CAPTURE_EPISODES
            or type(capture_step) is not int or capture_step < 0):
        raise ValueError('fixture count must be an integer from 1 to 32 and step must be nonnegative')
    if root:
        path = Path(root)
        if not path.is_absolute() or not path.resolve().is_relative_to(ARTIFACTS) or path.resolve() == ARTIFACTS:
            raise ValueError('fixture outputs require an explicit experiment directory on the data disk')



def actor_source_binding(checkpoint, *, qwen, head, query):
    source = {'checkpoint': str(checkpoint) if checkpoint else None,
              'qwen': str(Path(qwen).resolve()), 'head': str(Path(head).resolve()),
              'query': str(Path(query).resolve())}
    paths = [Path(qwen) / 'config.json', Path(qwen) / 'model.safetensors.index.json',
             Path(head) / 'fast_slow_action_head_config.json']
    query_path = Path(query)
    paths.append(query_path if query_path.is_file() else query_path / 'dualvln_query_embeddings.pt')
    if checkpoint:
        paths.extend([Path(checkpoint) / 'report.json', Path(checkpoint).parent / 'contract.json'])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'fixture source metadata missing: {missing}')
    source['metadata_sha256'] = {str(path): _sha(path) for path in paths}
    return source


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        with temporary.open('x') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
        # Atomic publish-if-absent: even retries never replace audit evidence.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cpu_copy(value, memo=None):
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if torch.is_tensor(value):
        if value.device.type != 'cpu':
            raise ValueError('fixture capture only accepts already materialized CPU tensors')
        result = value.detach().clone()
    elif isinstance(value, dict):
        result = {key: _cpu_copy(item, memo) for key, item in value.items()}
    elif isinstance(value, list):
        result = [_cpu_copy(item, memo) for item in value]
    elif isinstance(value, tuple):
        result = tuple(_cpu_copy(item, memo) for item in value)
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('fixture cannot serialize object-array runtime handles')
        result = value.copy()
    elif isinstance(value, (str, int, float, bool, bytes, type(None), np.generic, Image.Image)):
        result = copy.deepcopy(value)
    else:
        raise TypeError(f'fixture excludes unsupported runtime object: {type(value).__name__}')
    memo[id(value)] = result
    return result


def _require_self_contained_images(image_parts):
    for part in image_parts:
        value = part.get('image_url', part.get('image'))
        if isinstance(value, dict):
            value = value.get('url')
        if isinstance(value, str) and value.startswith('data:image/'):
            continue
        if isinstance(value, (bytes, Image.Image, np.ndarray)):
            continue
        raise ValueError('fixture images must be inline, not volatile file/HTTP references')


def capture_update_records(root, records, *, rank, world_size, logical_weight, layout):
    """Archive the first materialized update shard before either branch mutates it.

    Unlike paired fixtures, this preserves complete trajectory targets and all
    FM noise/states after credit assignment and distributed padding. It is an
    explicit diagnostic input, never a replacement for on-policy rollout.
    """
    validate_fixture_options(root, 1, 0)
    if (not root or not records or not 0 <= rank < world_size
            or not math.isfinite(logical_weight) or logical_weight <= 0):
        raise ValueError('update replay needs a nonempty shard and original loss mass')
    versions = {row['policy_version'] for row in records}
    if len(versions) != 1:
        raise ValueError('update replay cannot mix policy versions')
    # This also rejects CUDA tensors and live Ray references; the producer
    # must finish the existing staging resolver before calling this function.
    saved = _cpu_copy(records)
    directory = Path(root) / f'rank_{rank:03d}'
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / 'records.pt'
    temporary = target.with_name(target.name + f'.tmp.{os.getpid()}')
    try:
        with temporary.open('xb') as handle:
            torch.save({'records': saved, 'logical_weight': logical_weight}, handle)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    report = dict(schema='v8-complete-update-replay-v1', rank=rank, world_size=world_size,
        records=len(records), policy_version=next(iter(versions)), logical_weight=logical_weight,
        input=str(target), input_sha256=_sha(target), layout=layout,
        text_indices=[i for i, row in enumerate(records) if is_text_record(row)],
        source_file=__file__, source_sha256=_sha(Path(__file__)), optimizer_updates=0)
    _atomic_json(directory / 'complete.json', report)
    return report


def _error_audit(directory, stage, exc):
    result = {'status': 'error', 'stage': stage, 'directory': str(directory),
              'error_type': type(exc).__name__, 'error': str(exc)}
    try:
        if directory.is_dir():
            _atomic_json(directory / f'{stage}_error.json', result)
    except Exception as audit_error:
        result['audit_write_error'] = f'{type(audit_error).__name__}: {audit_error}'
    return result


def capture_raw_fixture(spec, records, decisions, result, *, case, seed, policy_version):
    """Select one pair deterministically; every failure is returned as audit data."""
    directory = Path(spec['directory'])
    temporary = None
    try:
        # Reserve once: retries cannot overwrite an earlier success or failure.
        directory.mkdir(parents=True, exist_ok=False)
        source = spec.get('actor_source') or {}
        if not source.get('metadata_sha256') or not (
                source.get('checkpoint') or all(source.get(key) for key in ('qwen', 'head', 'query'))):
            if not source.get('error'):
                raise ValueError('fixture requires an explicit actor source with metadata hashes')
        if source.get('error'):
            raise ValueError(f"fixture source binding failed: {spec['actor_source']['error']}")
        if int(spec['step']) != int(policy_version) + 1:
            raise ValueError('fixture batch must immediately follow the captured policy version')
        fm, text = [dict(row) for row in records], [dict(row) for row in decisions]
        annotate_rollout_decisions(fm, text, result)
        selected = None
        for decision in sorted(text, key=lambda row: int(row['decision_index'])):
            if (decision.get('decision_action_head_call_count') != 1
                    or decision.get('decision_fm_option_count') != 1):
                continue
            chunks = [row for row in fm if row.get('native_request_id') == decision['native_request_id']]
            if chunks:
                selected = (chunks, decision)
                break
        if selected is None:
            audit = {'status': 'no_eligible_pair', 'directory': str(directory), **spec}
            _atomic_json(directory / 'capture.json', audit)
            return audit
        chunks, decision = selected
        anchor = chunks[0]
        fm_required = ('slow_request', 'slow_response', 'all_input_ids', 'all_routed_experts',
                       'all_positions', 'query_routed_experts', 'query_positions',
                       'reference_visual_tokens', 'reference_visual_mask', 'current_visual_tokens',
                       'current_visual_mask', 'states', 'successors', 'timesteps', 'noise_scales',
                       'signed_dt', 'var_floor', 'old_logprob', 'chunk_index', 'option_id')
        text_required = ('record_kind', 'all_input_ids', 'all_routed_experts', 'all_positions',
                         'text_response_mask', 'text_prompt_length', 'text_response_end', 'text_temperature')
        if any(key not in row for row in chunks for key in fm_required) or any(
                key not in decision for key in text_required):
            raise ValueError('fixture lacks required replay inputs')
        if (any(int(row.get('policy_version', policy_version)) != policy_version for row in chunks)
                or int(decision['policy_version']) != policy_version):
            raise ValueError('fixture records disagree with current policy version')
        q = len(anchor['query_routed_experts'])
        if q <= 0 or len({row['option_id'] for row in chunks}) != 1:
            raise ValueError('fixture pair must contain one option and a nonempty query tail')
        if len({row['chunk_index'] for row in chunks}) != len(chunks):
            raise ValueError('fixture pair contains duplicate FM chunks')
        for key in ('all_input_ids', 'all_routed_experts', 'all_positions'):
            if not torch.equal(torch.as_tensor(anchor[key])[:-q], torch.as_tensor(decision[key])):
                raise ValueError(f'fixture text/FM native prefix mismatch: {key}')
        if (anchor['slow_request'] != decision['slow_request']
                or anchor['slow_response'] != decision['slow_response']):
            raise ValueError('fixture text/FM slow request or response mismatch')
        messages = anchor['slow_request']['messages']
        image_parts = [part for message in messages if isinstance(message.get('content'), list)
                       for part in message['content'] if isinstance(part, dict)
                       and part.get('type') in ('image', 'image_url')]
        _require_self_contained_images(image_parts)
        payload = _cpu_copy({
            'schema': SCHEMA, 'capture': dict(spec), 'case': case, 'seed': int(seed),
            'policy_version': int(policy_version), 'native_request_id': decision['native_request_id'],
            'decision_index': int(decision['decision_index']), 'option_id': anchor['option_id'],
            'fm_records': chunks, 'text_record': decision,
            'image_input_audit': {'actual_image_parts': len(image_parts),
                                  'protocol': 'exact v6 slow_request, including any map; no image-window modification'},
        })
        final = directory / 'raw.pt'
        temporary = directory / f'raw.pt.tmp.{os.getpid()}'
        torch.save(payload, temporary)
        os.link(temporary, final)
        temporary.unlink()
        audit = {'status': 'raw_complete_pending_targets', 'schema': SCHEMA, **spec,
                 'raw_sha256': _sha(final), 'policy_version': int(policy_version),
                 'native_request_id': payload['native_request_id'], 'decision_index': payload['decision_index'],
                 'option_id': payload['option_id'], 'chunk_indices': [row['chunk_index'] for row in chunks],
                 'actual_image_parts': len(image_parts)}
        _atomic_json(directory / 'capture.json', audit)
        return audit
    except FileExistsError:
        return {'status': 'already_attempted', 'directory': str(directory), **spec}
    except Exception as exc:
        audit = _error_audit(directory, 'capture', exc)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except Exception as cleanup_error:
                audit['temporary_cleanup_error'] = f'{type(cleanup_error).__name__}: {cleanup_error}'
        return audit


def finalize_fixture_targets(audit, records, *, step, policy_version, credit_mode,
                             logical_denominator, critic_sha256=None):
    """Commit actual post-credit targets, without reading/resolving opaque tensors."""
    if audit.get('status') != 'raw_complete_pending_targets':
        return audit
    directory = Path(audit['directory'])
    try:
        if (int(audit['step']) != step or int(audit['policy_version']) != policy_version
                or logical_denominator <= 0):
            raise ValueError('fixture target policy/step/denominator mismatch')
        if (directory / 'complete.json').exists():
            return {**audit, 'status': 'already_finalized'}
        if _sha(directory / 'raw.pt') != audit['raw_sha256']:
            raise ValueError('fixture raw payload changed before target commit')
        chunks = [row for row in records if not is_text_record(row)
                  and row.get('option_id') == audit['option_id']]
        text = [row for row in records if is_text_record(row)
                and int(row['decision_index']) == int(audit['decision_index'])]
        if (len(text) != 1 or [row['chunk_index'] for row in chunks] != audit['chunk_indices']):
            raise ValueError('fixture targets do not match the captured decision/chunks')
        if any(int(row.get('policy_version', policy_version)) != policy_version for row in chunks + text):
            raise ValueError('fixture target records changed policy version')
        for row in chunks + text:
            expected = (f"step={step}/rollout={audit['rollout_index']}/text={row['decision_index']}"
                        if is_text_record(row) else
                        f"step={step}/rollout={audit['rollout_index']}/option={row['option_id']}")
            if row.get('condition_id') != expected:
                raise ValueError('fixture target episode/decision identity mismatch')
            if row.get('native_request_id', audit['native_request_id']) != audit['native_request_id']:
                raise ValueError('fixture target native request identity mismatch')
        fields = ('record_kind', 'joint_episode_id', 'native_request_id', 'slow_decision_index', 'condition_id', 'option_id', 'chunk_index', 'decision_index', 'policy_version',
                  'advantage', 'sample_valid', 'loss_weight', 'text_loss_coef', 'text_temperature')
        def targets(row):
            output = {key: row[key] for key in fields if key in row}
            output.setdefault('sample_valid', 1.0)
            output['loss_coef'] = float(row.get('text_loss_coef', 1.0))
            output['temperature'] = row.get('text_temperature')
            if any(not math.isfinite(float(output[key])) for key in ('advantage', 'loss_weight', 'sample_valid')):
                raise ValueError('fixture targets must be finite')
            return output
        value = {'schema': SCHEMA, 'step': step, 'policy_version': policy_version,
                 'credit_mode': credit_mode, 'critic_sha256': critic_sha256,
                 'native_request_id': audit['native_request_id'], 'decision_index': audit['decision_index'],
                 'actor_source': audit.get('actor_source'), 'rollout_index': audit['rollout_index'],
                 'logical_denominator': float(logical_denominator), 'raw_sha256': audit['raw_sha256'],
                 'fm_targets': [targets(row) for row in chunks], 'text_target': targets(text[0])}
        _atomic_json(directory / 'targets.json', value)
        complete = {**audit, 'status': 'complete', 'targets_sha256': _sha(directory / 'targets.json')}
        _atomic_json(directory / 'complete.json', complete)
        return complete
    except Exception as exc:
        return _error_audit(directory, 'targets', exc)
