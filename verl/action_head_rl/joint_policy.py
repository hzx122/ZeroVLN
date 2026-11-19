"""Explicit metadata pairing for the opt-in shared text/FM forward.

Pairing never resolves Ray payloads. Tensor/processor equality is checked on
every learner before the first FSDP forward, after the usual staging barrier.
"""
from collections import Counter, defaultdict
import math

from action_head_rl.text_policy import is_text_record


def episode_text_mass(records) -> float:
    import os
    if os.environ.get('ACTION_HEAD_FINAL_CONTRACT') == 'three-source-boundary-stop-v1':
        identities = set()
        for row in records:
            if not is_text_record(row):
                continue
            valid = float(row.get('sample_valid', 1.0))
            if valid not in (0., 1.):
                raise ValueError('final episode mass requires binary sample_valid')
            if not valid:
                continue
            identity = row.get('joint_episode_id')
            if not isinstance(identity, str) or not identity:
                raise ValueError('final episode normalization requires joint_episode_id')
            identities.add(identity)
        if not identities:
            raise ValueError('final objective requires original valid text episodes')
        return float(len(identities))
    mass = sum(float(row.get('sample_valid', 1.0)) * float(row.get('loss_weight', 1.0))
               for row in records if is_text_record(row))
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('joint policy requires positive original episode text mass')
    return mass


def annotate_rollout_decisions(records, decisions, result):
    """Attach only scalar provenance before the opaque payloads are packed."""
    turns = {int(turn['turn']): turn for turn in result.get('turns', [])}
    by_request = {}
    for row in decisions:
        request_id = str((row.get('slow_response') or {}).get('query_request_id') or '')
        if not request_id:
            continue
        if request_id in by_request:
            raise ValueError('duplicate native slow request in decision capture')
        by_request[request_id] = row
    options = defaultdict(set)
    for row in records:
        request_id = str((row.get('slow_response') or {}).get('query_request_id') or '')
        if request_id:
            options[request_id].add(row['option_id'])
    for request_id, decision in by_request.items():
        turn = turns.get(int(decision['decision_index']), {})
        fields = {
            'native_request_id': request_id,
            'slow_decision_index': int(decision['decision_index']),
            'decision_action_head_call_count': sum(
                call.get('name') == 'run_action_head' for call in turn.get('calls', [])),
            'decision_fm_option_count': len(options[request_id]),
        }
        decision.update(fields)
        for row in records:
            if str((row.get('slow_response') or {}).get('query_request_id') or '') == request_id:
                row.update(fields)
                row['token_count'] = len(row['all_input_ids'])
                row['r3_query_count'] = len(row['query_routed_experts'])


def build_joint_policy_groups(records):
    """Pair one AH option with one sampled decision; report all fallbacks."""
    fm = defaultdict(list)
    texts = []
    for row in records:
        if not row.get('condition_id'):
            raise ValueError('joint policy records require condition_id')
        (texts if is_text_record(row) else fm[row['condition_id']]).append(row)

    def identity(row):
        values = (row.get('joint_episode_id'), row.get('policy_version'),
                  row.get('native_request_id'), row.get('slow_decision_index'))
        return values if values[0] and values[1] is not None and values[2] and values[3] else None

    text_by_id = {}
    for row in texts:
        key = identity(row)
        if key is not None:
            if key in text_by_id:
                raise ValueError('joint policy has duplicate text decisions for one native request')
            text_by_id[key] = row
    options_by_id = defaultdict(list)
    for group in fm.values():
        key = identity(group[0])
        if any(identity(row) != key for row in group):
            raise ValueError('FM option mixes slow decision identities')
        if key is not None:
            options_by_id[key].append(group)
    groups, consumed, fallbacks = [], set(), Counter()
    for group in fm.values():
        anchor = group[0]
        key = identity(anchor)
        text = text_by_id.get(key)
        reason = None
        if key is None or text is None:
            reason = 'missing_native_text_pair'
        elif (len(options_by_id[key]) != 1
              or any(row.get('decision_action_head_call_count') != 1
                     or row.get('decision_fm_option_count') != 1 for row in (anchor, text))):
            reason = 'multi_or_unknown_action_head_decision'
        elif int(anchor.get('r3_query_count', 0)) <= 0:
            reason = 'missing_query_tail'
        elif int(anchor.get('token_count', 0)) != (
                int(text.get('token_count', 0)) + int(anchor.get('r3_query_count', 0))):
            reason = 'different_prefix_lengths'
        if reason is not None:
            fallbacks[reason] += 1
            text = None
        else:
            consumed.add(text['condition_id'])
        groups.append({'family': 'joint' if text is not None else 'fm',
                       'group_id': anchor['condition_id'],
                       'token_count': int(anchor['token_count']),
                       'fm_records': group, 'text_record': text})
    for row in texts:
        if row['condition_id'] not in consumed:
            groups.append({'family': 'text', 'group_id': row['condition_id'],
                           'token_count': int(row['token_count']),
                           'fm_records': [], 'text_record': row})
    return groups, {'paired_decisions': len(consumed),
                    'unpaired_text_decisions': len(texts) - len(consumed),
                    'fallback_fm_options': dict(fallbacks)}


def consecutive_joint_groups(records):
    groups = []
    seen = set()
    for row in records:
        key = row.get('_joint_group_id')
        family = row.get('_joint_family')
        if not key or family not in ('joint', 'fm', 'text'):
            raise ValueError('joint train stream lacks explicit group/family metadata')
        if not groups or key != groups[-1][0]['_joint_group_id']:
            if key in seen:
                raise ValueError('joint group is not contiguous')
            seen.add(key)
            groups.append([])
        if groups[-1] and family != groups[-1][0]['_joint_family']:
            raise ValueError('joint group contains different forward families')
        groups[-1].append(row)
    return groups


def validate_joint_preflight_reports(reports, rounds, *, world, logical_valid):
    """Driver-only barrier gate; called after all CPU RPCs and before FSDP."""
    expected = [{"family": row["family"], "width": row["token_width"],
                 "fm_chunks": row["fm_chunks_per_rank"], "records": row["records_per_rank"],
                 **({"batch_size": row["batch_size"]} if "batch_size" in row else {})}
                for row in rounds]
    if len(reports) != world or any(
            report["schedule"] != expected or report["logical_valid"] != logical_valid
            for report in reports):
        raise RuntimeError("joint preflight has divergent rank schedules or loss mass")
    signatures = reports[0].get("collective_signatures")
    if signatures is None or len(signatures) != len(rounds) or any(
            report.get("collective_signatures") != signatures for report in reports):
        raise RuntimeError("joint preflight has divergent actual collective/loop shapes")


def consecutive_joint_batches(groups):
    """Use the driver schedule verbatim, including token-cap singleton tails."""
    batches, seen = [], set()
    for group in groups:
        key = group[0]['_condition_pack_batch']
        if not batches or key != batches[-1][0][0]['_condition_pack_batch']:
            if key in seen:
                raise ValueError('joint pack batch is not contiguous')
            seen.add(key)
            batches.append([])
        previous = batches[-1]
        if previous and any(group[0][name] != previous[0][0][name]
                            for name in ('_joint_family', '_condition_pack_token_width')):
            raise ValueError('joint pack batch changed family or width')
        previous.append(group)
    return batches


def final_record_has_auxiliary_loss(row):
    """Preserve STOP/path supervision when navigation advantage is zero."""
    import os
    if os.environ.get('ACTION_HEAD_FINAL_CONTRACT') != 'three-source-boundary-stop-v1':
        return False
    values = {key: float(row.get(key, 0.)) for key in
              ('text_stop_weight', 'text_path_advantage', 'action_path_advantage')}
    if not all(math.isfinite(value) for value in values.values()) or values['text_stop_weight'] < 0:
        raise ValueError('final auxiliary objective metadata must be finite with nonnegative STOP weight')
    return values['text_stop_weight'] > 0 or any(values[key] != 0 for key in
        ('text_path_advantage', 'action_path_advantage'))
