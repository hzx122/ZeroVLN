"""Fixed native-record packing replay, with CPU preparation separate from CUDA.

Two to thirty-two genuine source16 options contribute once in fixed rank slots.
Remaining slots are zero-valid shape padding, never additional samples.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path

import torch

from action_head_rl.joint_audit import (
    ARTIFACTS, _same_source, process_alive, query_gpu_state, read_json,
    resolve_rpc, sha256, validate_source, wall_clock_deadline, write_new_json,
)
from action_head_rl.text_policy import is_text_record

SCHEMA = "v6-native-packing-replay-v1"
BATCH_SIZES = (1, 2, 4, 8)
THRESHOLDS = dict(gradient_atol=1e-6, gradient_rtol=1e-3,
                  delta_atol=1e-7, delta_rtol=1e-3)


def replay_worker_environment(source, *, token_cap, token_bucket):
    """Pin the learner overrides normally installed by trainer.main before Ray.

    A launcher plan alone does not contain those overrides. In particular,
    engine._build_module defaults full-route capture OFF and must reject that.
    This offline replay uses native captured routes and never starts capture.
    """
    return dict(
        ACTION_HEAD_HEAD_CHECKPOINT=source['binding']['head'],
        ACTION_HEAD_QUERY_CHECKPOINT=source['binding']['query'],
        ACTION_HEAD_TRAIN_SCOPE='full', ACTION_HEAD_TEXT_POLICY='1',
        ACTION_HEAD_JOINT_TEXT_FM_FORWARD='0', ACTION_HEAD_ROUTER_REPLAY='R3',
        ACTION_HEAD_CAPTURE_FULL_ROUTING='1', ACTION_HEAD_R3_CAPTURE_MODE='native',
        ACTION_HEAD_DEFER_QUERY_CAPTURE='0', ACTION_HEAD_OLD_LOGPROB_MODE='single_forward',
        ACTION_HEAD_OPTION_CONDITION_REUSE='1', ACTION_HEAD_DEFER_GRADIENT_SYNC='0',
        ACTION_HEAD_MULTI_CONDITION_BATCH_SIZE='1', ACTION_HEAD_MAX_CONDITION_TOKENS=str(token_cap),
        ACTION_HEAD_CONDITION_TOKEN_BUCKET_SIZE=str(token_bucket),
        ACTION_HEAD_FULL_WEIGHT_RESIDENCY_GIB='0', ACTION_HEAD_FROZEN_WEIGHT_RESIDENCY='0',
        ACTION_HEAD_VLM_LR='1e-6', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')


def validate_replay_environment(environment, source, *, token_cap, token_bucket):
    from action_head_rl.config import validate_training_contract
    expected = replay_worker_environment(source, token_cap=token_cap, token_bucket=token_bucket)
    mismatched = [key for key, value in expected.items() if environment.get(key) != value]
    if mismatched:
        raise ValueError(f'packing learner environment differs: {mismatched}')
    # The actual engine's reset guard, run on CPU before launch as well.
    validate_training_contract(train_scope=environment['ACTION_HEAD_TRAIN_SCOPE'],
        router_replay_mode=environment['ACTION_HEAD_ROUTER_REPLAY'],
        capture_full_routes=environment['ACTION_HEAD_CAPTURE_FULL_ROUTING'] == '1',
        rollout_backend='vllm')


def load_fixture(directory, *, source_binding):
    """Strict existing native capture schema, extended to all selected32 indices."""
    from action_head_rl.joint_fixture import SCHEMA as FIXTURE_SCHEMA
    directory = Path(directory).resolve()
    complete, targets = read_json(directory/'complete.json'), read_json(directory/'targets.json')
    if (complete.get('status') != 'complete' or complete.get('schema') != FIXTURE_SCHEMA
            or targets.get('schema') != FIXTURE_SCHEMA
            or complete.get('raw_sha256') != sha256(directory/'raw.pt')
            or complete.get('targets_sha256') != sha256(directory/'targets.json')
            or targets.get('raw_sha256') != complete['raw_sha256']):
        raise ValueError('native fixture hashes or completion changed')
    raw = torch.load(directory/'raw.pt', map_location='cpu', weights_only=False)
    capture = raw.get('capture', {})
    index = capture.get('rollout_index')
    if (type(index) is not int or index not in range(32) or targets.get('rollout_index') != index
            or raw.get('schema') != FIXTURE_SCHEMA or capture.get('step') != 17
            or targets.get('step') != 17 or targets.get('policy_version') != 16
            or raw.get('policy_version') != 16 or targets.get('logical_denominator') != 32
            or not all(_same_source(item.get('actor_source'), source_binding)
                       for item in (capture, complete, targets))):
        raise ValueError('native fixture source/version/denominator differs')
    for key in ('native_request_id', 'decision_index'):
        if raw.get(key) != targets.get(key) or raw.get(key) != complete.get(key):
            raise ValueError('native fixture decision identity differs')
    fm, fm_targets = raw['fm_records'], targets['fm_targets']
    if (not fm or len(fm) != len(fm_targets)
            or len({r['chunk_index'] for r in fm}) != len(fm)
            or [r['chunk_index'] for r in fm] != complete.get('chunk_indices')):
        raise ValueError('native fixture FM chunks differ')
    rows = []
    for record, target in [*zip(fm, fm_targets), (raw['text_record'], targets['text_target'])]:
        keys = ('decision_index', 'option_id', 'policy_version') if is_text_record(record) else ('option_id', 'chunk_index', 'policy_version')
        if any(record.get(k) != target.get(k) for k in keys):
            raise ValueError('native target belongs to another decision/chunk')
        for key in ('advantage', 'sample_valid', 'loss_weight'):
            if type(target.get(key)) not in (int, float) or not math.isfinite(target[key]):
                raise ValueError('native targets must be finite')
        if target['sample_valid'] != 1 or target['loss_weight'] <= 0 or not target.get('condition_id'):
            raise ValueError('native branch weight/identity is invalid')
        if is_text_record(record):
            if target.get('loss_coef') != target.get('text_loss_coef') or target.get('temperature') != target.get('text_temperature'):
                raise ValueError('native text target coefficients differ')
        elif target.get('loss_coef') != 1.:
            raise ValueError('native FM coefficient differs')
        rows.append(dict(record, **target))
    return dict(directory=str(directory), rows=rows, rollout_index=index, denominator=32.,
        credit_mode=targets.get('credit_mode'), critic_sha256=targets.get('critic_sha256'),
        complete_sha256=sha256(directory/'complete.json'), raw_sha256=complete['raw_sha256'],
        targets_sha256=complete['targets_sha256'])


def replay_streams(fixtures, *, world=12, slots=8, token_cap=40960, token_bucket=0):
    from itertools import groupby
    if world < 1 or slots != 8 or not 2 <= len(fixtures) <= min(32, world*slots) or token_cap <= 0 or token_bucket < 0:
        raise ValueError("replay requires 2..32 real fixtures, eight rank slots and a positive cap")
    indices = [row['rollout_index'] for row in fixtures]
    if indices != sorted(set(indices)):
        raise ValueError('native fixtures must be unique and sorted by original rollout index')
    if len(fixtures) < 8 and indices != [0, 1]:
        raise ValueError('only the fixed original two-pair shape audit may have fewer than eight fixtures')
    if len({row.get("credit_mode") for row in fixtures}) != 1:
        raise ValueError("do not mix fixture algorithms or overwrite their original targets")
    originals = [row for fixture in fixtures for row in fixture["rows"]]
    if not all(any(row["advantage"] != 0 and is_text_record(row) == text
                   for row in originals) for text in (False, True)):
        raise ValueError("fixed fixture is uninformative; do not replace samples or targets")
    rank_groups = []
    for rank in range(world):
        families = [[], []]
        for slot in range(slots):
            index = rank*slots + slot
            fixture = fixtures[index % len(fixtures)]
            pair = [[], []]
            for row in fixture['rows']:
                item = dict(row)
                for key in list(item):
                    if key.startswith('_condition_pack_') or key.startswith('_joint_'):
                        item.pop(key)
                if index >= len(fixtures):
                    item.update(condition_id=f"{row['condition_id']}/audit-padding={index}", sample_valid=0., advantage=0.)
                item['logical_batch_valid'] = 32.
                pair[int(is_text_record(item))].append(item)
            if not pair[0] or len(pair[1]) != 1:
                raise ValueError('each capture must contain one original FM option and one text decision')
            for family in (0, 1):
                families[family].append(pair[family])
        rank_groups.append(families[0] + families[1])
    # Fixed chunk-loop counts across ranks, preserving all real chunks.
    for column in range(slots):
        chunks = max(len(groups[column]) for groups in rank_groups)
        for groups in rank_groups:
            while len(groups[column]) < chunks:
                groups[column].append(dict(groups[column][-1], sample_valid=0., advantage=0.))
    variants = {}
    summaries = {}
    for size in BATCH_SIZES:
        groups_by_rank = [[[dict(row) for row in group] for group in groups] for groups in rank_groups]
        batch_id = 0
        for family_start in (0, slots):
            start = family_start
            while start < family_start + slots:
                end, width = start, 0
                while end < family_start + slots and end - start < size:
                    next_width = max(width, max(torch.as_tensor(groups[end][0]['all_input_ids']).numel()
                                                for groups in groups_by_rank))
                    if token_bucket:
                        next_width = math.ceil(next_width/token_bucket)*token_bucket
                    if end > start and next_width * (end-start+1) > token_cap:
                        break
                    width, end = next_width, end + 1
                for groups in groups_by_rank:
                    for group in groups[start:end]:
                        for row in group:
                            row.update(_condition_pack_batch=batch_id, _condition_pack_token_width=width)
                batch_id, start = batch_id + 1, end
        ranks = [[row for group in groups for row in group] for groups in groups_by_rank]
        rank0 = ranks[0]
        variants[str(size)] = ranks
        groups = groups_by_rank[0]
        batches = [list(part) for _, part in groupby(groups, key=lambda group: group[0]['_condition_pack_batch'])]
        summaries[str(size)] = dict(
            requested_batch_size=size, real_conditions_per_branch=len(fixtures),
            shape_conditions_per_rank_branch=slots,
            zero_valid_rank_count=max(0, world-math.ceil(len(fixtures)/slots)),
            forwards_per_rank=len(batches),
            actual_batch_sizes=[len(batch) for batch in batches],
            actual_padded_widths=[int(batch[0][0]['_condition_pack_token_width']) for batch in batches],
            padded_token_slots_per_rank=sum(len(batch) * int(batch[0][0]["_condition_pack_token_width"])
                                            for batch in batches),
            natural_valid_tokens=sum(int(torch.as_tensor(row[0]["all_input_ids"]).numel())
                                     for row in groups if row[0].get("sample_valid", 1.) > 0))
    return variants, summaries


def pack_stream(records, processor, size, token_cap):
    from action_head_rl.worker import (
        consecutive_record_groups, pack_condition_groups, record_condition_groups_to_tensordict,
    )
    from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
    groups = consecutive_record_groups(records, True)
    batches = pack_condition_groups(groups, size, token_cap)
    packed = []
    for batch in batches:
        data = record_condition_groups_to_tensordict(batch, processor, "cpu")
        maybe_fix_3d_position_ids(data)
        packed.append(data)
    return packed


def packed_signature(packed):
    return [{key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)}
             for key in sorted(data.keys())} for data in packed]


def collective_signature(packed):
    """Vision token/image counts may differ; branch presence must not."""
    return [dict(vision_branch=('pixel_values' in signature), image_grids=('image_grid_thw' in signature),
            tensors={key: value for key, value in signature.items()
             if key not in ('pixel_values', 'image_grid_thw', 'slow_image_count_per_anchor')})
            for signature in packed_signature(packed)]


def cpu_preflight(variants, source, token_cap):
    from transformers import AutoProcessor
    from action_head_rl.joint_audit import temporary_environment
    processor = AutoProcessor.from_pretrained(source["binding"]["qwen"], local_files_only=True)
    result = {}
    with temporary_environment(ACTION_HEAD_TEXT_POLICY="1"):
        for size, ranks in variants.items():
            signatures = []
            cache = {}
            for rows in ranks:
                # Padding copies share immutable native payload objects. Only
                # reuse a CPU signature for that exact same referenced input;
                # distinct captures/images still get their own real processor.
                key = tuple(tuple((k, id(v)) for k, v in sorted(row.items()) if k not in
                    ('condition_id', 'sample_valid', 'advantage')) for row in rows)
                if key not in cache:
                    packed = pack_stream(rows, processor, int(size), token_cap)
                    cache[key] = dict(collective=collective_signature(packed), tensors=packed_signature(packed))
                signatures.append(cache[key])
            if any(row['collective'] != signatures[0]['collective'] for row in signatures):
                raise ValueError('rank replay collective signatures differ')
            result[size] = signatures
    return result


def prepare_plan(*, fixture_dirs, source_checkpoint, runtime_plan, output,
                 after_process_records=(), token_cap=40960, token_bucket=1024, warmup=1, repeats=3,
                 timeout_seconds=7200, thresholds=None):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ARTIFACTS) or output == ARTIFACTS:
        raise ValueError("prepare requires a new artifact output directory")
    if warmup != 1 or repeats != 3 or timeout_seconds <= 0:
        raise ValueError("fixed replay protocol is one warmup and three alternating timing repeats")
    source = validate_source(source_checkpoint)
    fixtures = [load_fixture(path, source_binding=source["binding"]) for path in fixture_dirs]
    variants, schedule = replay_streams(fixtures, token_cap=token_cap, token_bucket=token_bucket)
    limits = dict(THRESHOLDS if thresholds is None else thresholds)
    if set(limits) != set(THRESHOLDS) or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                                           for x in limits.values()):
        raise ValueError("numerical thresholds must be fixed finite positive values")
    runtime_path = Path(runtime_plan).resolve()
    runtime = read_json(runtime_path)
    if runtime.get("initial_checkpoint") != source["checkpoint"]:
        raise ValueError("runtime and replay source differ")
    repo = Path(__file__).resolve().parents[2]
    upstream = repo.parent/'upstream'
    if not (upstream/'verl').is_dir():
        upstream = Path(runtime['snapshot_upstream'])
    environment = dict(runtime["environment"])
    old_cache = environment["ACTION_HEAD_LOCAL_RUNTIME_ROOT"]
    cache = "/tmp/ah_pack/" + hashlib.sha256(str(output).encode()).hexdigest()[:12]
    environment = {k: v.replace(old_cache, cache) for k, v in environment.items()}
    if (repo.parent/'fast_kernel_overlay').is_dir():
        environment['FAST_KERNEL_OVERLAY'] = str(repo.parent/'fast_kernel_overlay')
    environment['VERL_SFT_CONFIG_DIR'] = str(upstream/'verl/trainer/config')
    environment.update(CUDA_VISIBLE_DEVICES=",".join(map(str, range(12))),
        **replay_worker_environment(source, token_cap=token_cap, token_bucket=token_bucket))
    validate_replay_environment(environment, source, token_cap=token_cap, token_bucket=token_bucket)
    environment["PYTHONPATH"] = ":".join((environment["FAST_KERNEL_OVERLAY"], str(repo / "verl"),
        str(upstream), str(repo / "vln_demo/src")))
    contract = source["contract"]
    learner = dict(slow_model=source["binding"]["qwen"], learner_parameter_dtype="fp32",
        gradient_checkpointing=contract["gradient_checkpointing"], fsdp_size=contract["fsdp_size"],
        fsdp_forward_prefetch=contract["fsdp_forward_prefetch"], learning_rate=contract["action_head_learning_rate"],
        weight_decay=contract["optimizer_weight_decay"], opt_steps=22)
    prerequisites = []
    for filename in after_process_records:
        path = Path(filename).resolve()
        value = read_json(path)
        if type(value.get("pid")) is not int or not str(value.get("start_ticks", "")).isdigit():
            raise ValueError("prerequisite needs a real recorded PID and start_ticks")
        prerequisites.append(dict(path=str(path), sha256=sha256(path),
            identity=dict(pid=value["pid"], start_ticks=str(value["start_ticks"]))))
    if not prerequisites:
        raise ValueError("bind the actual preceding audit/trainer process records")
    files = [repo / "scripts/eval/audit_v6_packing.py", runtime_path]
    for directory in (repo / "verl/action_head_rl", repo / "verl/models", repo / "verl/sft"):
        files.extend(directory.rglob("*.py"))
    files.extend((upstream/'verl/workers/engine').rglob('*.py'))
    files.append(upstream/'verl/workers/engine_workers.py')
    files.extend(Path(environment['FAST_KERNEL_OVERLAY']).rglob('*.py'))
    qwen_bytes = read_json(Path(source["binding"]["qwen"]) / "model.safetensors.index.json")["metadata"]["total_size"]
    return dict(schema=SCHEMA, output=str(output), source=source,
        fixtures=[{k: v for k, v in fixture.items() if k not in ("rows", "descriptor")} for fixture in fixtures],
        world_size=12, physical_gpus=list(range(12)), renderer_gpu=15, logical_denominator=32.,
        batch_sizes=list(BATCH_SIZES), shape_slots=8, effective_real_conditions=len(fixtures), token_cap=token_cap, token_bucket=token_bucket,
        warmup=warmup, repeats=repeats, timeout_seconds=timeout_seconds, thresholds=limits,
        optimizer_updates_per_numeric_variant=1, timing_optimizer_updates=0,
        cpu_snapshot_estimate_bytes=int(qwen_bytes * 5.1),
        cpu_temporary_reserve_bytes=128*1024**3,
        cpu_required_headroom_bytes=int(qwen_bytes * 5.1)+128*1024**3,
        cpu_snapshot_scope="FP32 params + two Adam moments + baseline gradients + baseline parameter delta; 2% margin for head/query",
        cpu_memory_scope='Persistent-copy estimate is not total RSS peak; reserve128GiB additionally for CPU FP64 comparisons, records/layouts and framework. Gate is rechecked only when the GPU slot is idle.',
        learner=learner, environment=environment, prerequisites=prerequisites,
        sources_sha256={str(path): sha256(path) for path in sorted(set(files))},
        schedule=schedule, cpu_preflight=cpu_preflight(variants, source, token_cap),
        scope=f"{len(fixtures)} native first-eligible options with original {fixtures[0]['credit_mode']} targets and denominator32, padded to eight rank slots. Partial replay only; not full-trajectory or end-to-end throughput.")


def validate_plan(plan, *, preflight=True):
    if (plan.get("schema") != SCHEMA or plan.get("world_size") != 12
            or plan.get("logical_denominator") != 32. or not 2 <= plan.get("effective_real_conditions", 0) <= 32
            or plan.get("shape_slots") != 8 or plan.get("batch_sizes") != list(BATCH_SIZES)
            or plan.get("optimizer_updates_per_numeric_variant") != 1 or plan.get("timing_optimizer_updates") != 0):
        raise ValueError("packing audit protocol changed")
    if (plan.get('warmup') != 1 or plan.get('repeats') != 3
            or set(plan.get('thresholds', {})) != set(THRESHOLDS)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in plan['thresholds'].values())
            or plan['environment'].get('CUDA_VISIBLE_DEVICES') != ','.join(map(str, range(12)))
            or plan['environment'].get('ACTION_HEAD_JOINT_TEXT_FM_FORWARD') != '0'):
        raise ValueError('packing runtime or numerical protocol changed')
    validate_replay_environment(plan['environment'], plan['source'],
                               token_cap=plan['token_cap'], token_bucket=plan.get('token_bucket', 0))
    if 'fixture_collection' in plan:
        collection = plan['fixture_collection']
        actual = sorted(read_json(path)['rollout_index'] for path in Path(collection['root']).glob('step_000017_rollout_*/complete.json'))
        if actual != collection['complete_indices']:
            raise ValueError('capture completion set changed; prepare a new immutable plan')
    for path, expected in plan["sources_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"replay source changed: {path}")
    for row in plan["prerequisites"]:
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError("prerequisite process record changed")
    source = validate_source(plan["source"]["checkpoint"])
    if source != plan["source"]:
        raise ValueError("source16 metadata changed")
    fixtures = [load_fixture(row["directory"], source_binding=source["binding"]) for row in plan["fixtures"]]
    for old, current in zip(plan["fixtures"], fixtures):
        for key in ("raw_sha256", "targets_sha256", "complete_sha256"):
            if old[key] != current[key]:
                raise ValueError("native fixture changed")
    variants, schedule = replay_streams(fixtures, token_cap=plan["token_cap"], token_bucket=plan.get('token_bucket', 0))
    if schedule != plan["schedule"] or (preflight and cpu_preflight(variants, source, plan["token_cap"]) != plan["cpu_preflight"]):
        raise ValueError("replay CPU layout changed")
    return variants


def global_comparison(ranks):
    result = {}
    for size in map(str, BATCH_SIZES):
        result[size] = {}
        for metric in ('gradient_summary', 'parameter_delta_summary'):
            result[size][metric] = {}
            for branch in ('all', 'qwen', 'query', 'fm'):
                tables = [rank['variants'][size][metric][branch] for rank in ranks]
                values = {key: sum(row[key] for row in tables) for key in
                    ('baseline_squared', 'candidate_squared', 'error_squared', 'dot_product', 'missing_gradient_count')}
                base, current, error = (values[key] for key in ('baseline_squared', 'candidate_squared', 'error_squared'))
                values.update(relative_l2=math.sqrt(error/max(base, 1e-30)),
                    cosine=values['dot_product']/math.sqrt(base*current) if base*current > 0 else None,
                    norm_ratio=math.sqrt(current/base) if base > 0 else None)
                result[size][metric][branch] = values
    return result


def memory_headroom(*, meminfo_path=Path('/proc/meminfo'), cgroup_roots=(Path('/sys/fs/cgroup'), Path('/sys/fs/cgroup/memory'))):
    host = int(next(line.split()[1] for line in meminfo_path.read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    available, reports = host, []
    for root in cgroup_roots:
        v2 = (root/'memory.max').exists()
        limit_path = root/('memory.max' if v2 else 'memory.limit_in_bytes')
        used_path = root/('memory.current' if v2 else 'memory.usage_in_bytes')
        if not (limit_path.exists() and used_path.exists()) or limit_path.read_text().strip() == 'max':
            continue
        limit, used = int(limit_path.read_text()), int(used_path.read_text())
        stat_path = root/'memory.stat'
        stat = {key: int(value) for key, value in (line.split() for line in stat_path.read_text().splitlines())} if stat_path.exists() else {}
        def value(name):
            return stat.get(name, 0) if v2 else stat.get('total_'+name, stat.get(name, 0))
        inactive = value('inactive_file')
        file = value('file') if v2 else value('cache')
        dirty, writeback, shmem = value('file_dirty' if v2 else 'dirty'), value('file_writeback' if v2 else 'writeback'), value('shmem')
        clean = min(inactive, max(0, file-shmem-dirty-writeback))
        hard_free = max(0, limit-used)
        estimated = hard_free + clean
        reports.append(dict(path=str(root), limit=limit, current=used, hard_free=hard_free,
            inactive_file=inactive, file=file, shmem=shmem, dirty=dirty, writeback=writeback,
            estimated_clean_reclaimable=clean, working_set=max(0, used-clean), estimated_headroom=estimated))
        available = min(available, estimated)
    return dict(host_mem_available=host, cgroups=reports, conservative_headroom_bytes=available)


def execution_gate(plan, *, query_gpu=query_gpu_state):
    if any(process_alive(row["identity"]) for row in plan["prerequisites"]):
        raise RuntimeError("preceding trainer/audit process is still alive")
    devices, apps = query_gpu()
    if any(index not in devices for index in range(16)):
        raise RuntimeError("did not prove the physical GPU0..15 layout")
    if any(uuid in {devices[i] for i in range(15)} for _, uuid in apps):
        raise RuntimeError("GPU0..14 must be idle; renderer15 is left untouched")
    # CPU caches can be large even though no additional model lives on CUDA.
    memory = memory_headroom()
    available = memory['conservative_headroom_bytes']
    required = plan.get('cpu_required_headroom_bytes', plan['cpu_snapshot_estimate_bytes'])
    if available < required:
        raise RuntimeError("insufficient conservative CPU/cgroup headroom for the fixed replay snapshots")
    return dict(ready=True, gpu_uuids={str(i): devices[i] for i in range(12)},
                remaining_gpu_processes=apps, memory=memory,
                cpu_snapshot_margin_bytes=available-plan['cpu_snapshot_estimate_bytes'],
                cpu_required_headroom_margin_bytes=available-required)


def execute_plan(path, *, expected_sha256, cpu_receipt_sha256):
    """Fresh-interpreter GPU entry. No rollout, vLLM, publication or checkpoint write."""
    import statistics
    import time
    from types import SimpleNamespace
    path = Path(path).resolve()
    if sha256(path) != expected_sha256:
        raise ValueError("packing replay plan SHA differs")
    plan = read_json(path)
    receipt = Path(plan['output'])/'execute_cpu_preflight.json'
    if (sha256(receipt) != cpu_receipt_sha256 or read_json(receipt) != dict(
            schema=SCHEMA, plan_sha256=expected_sha256, cpu_validation='passed')):
        raise ValueError('GPU execution lacks its hash-bound CPU preflight receipt')
    if any(os.environ.get(k) != v for k, v in plan["environment"].items()):
        raise RuntimeError("execute must start in a fresh interpreter with the plan environment")
    variants = validate_plan(plan, preflight=False)
    if torch.cuda.is_initialized():
        raise RuntimeError('GPU driver unexpectedly initialized CUDA before the idle resource gate')
    gate = execution_gate(plan)
    output = Path(plan["output"])
    write_new_json(output / "attempt.json", dict(plan_sha256=expected_sha256, pid=os.getpid(),
        start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19], gate=gate))
    # Last idle gate precedes heavy worker imports, whose framework probes can
    # themselves initialize the driver's CUDA runtime. No CPU processor runs
    # in this interpreter; worker preparation runs only after resource launch.
    execution_gate(plan)
    import ray
    from action_head_rl.packing_audit_worker import PackingReplayAuditWorker
    from action_head_rl.trainer import learner_config
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    cache = Path(plan["environment"]["ACTION_HEAD_LOCAL_RUNTIME_ROOT"])
    if not cache.is_relative_to('/tmp/ah_pack'):
        raise ValueError("packing replay cache escaped its private local root")
    cache.mkdir(parents=True, exist_ok=True)
    learner, identities, result = None, [], None
    deadline = time.monotonic() + plan['timeout_seconds']
    def resolve(value):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('packing replay absolute RPC deadline expired')
        return resolve_rpc(value, timeout_seconds=remaining)
    try:
        started = time.perf_counter()
        with wall_clock_deadline(plan["timeout_seconds"], "packing Ray bootstrap"):
            ray.init(address='local', num_gpus=12, include_dashboard=False, _temp_dir=str(cache/'ray'),
                     runtime_env={'env_vars': plan['environment']})
            learner = RayWorkerGroup(resource_pool=RayResourcePool(process_on_nodes=[12], max_colocate_count=4),
                ray_cls_with_init=RayClassWithInitArgs(ray.remote(max_restarts=0, max_task_retries=0)(PackingReplayAuditWorker),
                    config=learner_config(SimpleNamespace(**plan['learner']))), device_name='cuda')
        identities = resolve(learner.packing_process_identity())
        write_new_json(output/'worker_processes.json', dict(plan_sha256=expected_sha256, ranks=identities))
        resolve(learner.execute_all_async('reset'))
        resolve(learner.execute_all_async('load_checkpoint', local_path=str(Path(plan['source']['checkpoint'])/'trainer_state'),
                                         del_local_after_load=False))
        initialized = time.perf_counter() - started
        fast = resolve(learner.execute_all_async('audit_qwen_fast_path'))
        rank_variants = [{size: ranks[rank] for size, ranks in variants.items()} for rank in range(12)]
        ready = resolve(learner.prepare_packing_replay(variants=rank_variants, plan=[plan]*12))
        write_new_json(output/'worker_ready_shapes.json', dict(plan_sha256=expected_sha256, ranks=ready))
        if (len(ready) != 12 or {row['rank'] for row in ready} != set(range(12)) or any(
                row['signatures'] != {size: ranks[row['rank']] for size, ranks in plan['cpu_preflight'].items()}
                for row in ready)):
            raise RuntimeError('worker input signatures differ from CPU preparation')
        ranks = resolve(learner.run_packing_replay(plan=dict(plan, plan_sha256=expected_sha256)))
        if len(ranks) != 12 or {row['rank'] for row in ranks} != set(range(12)):
            raise RuntimeError('packing replay did not return all ranks')
        times = {size: [max(rank['timings'][str(repeat)][size]['seconds'] for rank in ranks)
                       for repeat in range(plan['repeats'])] for size in variants}
        result = dict(schema=SCHEMA, plan_sha256=expected_sha256,
            status='passed' if all(row['passed'] for row in ranks) else 'failed',
            initialization_seconds=initialized, effective_real_conditions=plan['effective_real_conditions'],
            fast_path=fast,
            median_max_rank_seconds={size: statistics.median(value) for size, value in times.items()},
            max_rank_seconds_by_repeat=times, full_shard_comparison=global_comparison(ranks),
            schedule=plan['schedule'], scope=plan['scope'], ranks=ranks)
    except Exception as exc:
        write_new_json(output/'failure.json', dict(type=type(exc).__name__, error=str(exc), automatic_retry=False))
        raise
    finally:
        errors = []
        if learner is not None:
            for actor in learner._workers:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as exc:
                    errors.append(str(exc))
        if ray.is_initialized():
            try:
                ray.shutdown()
            except Exception as exc:
                errors.append(str(exc))
        remaining = [row for row in identities if process_alive(row)]
        stop_deadline = time.monotonic()+10
        while remaining and time.monotonic() < stop_deadline:
            time.sleep(.2)
            remaining = [row for row in identities if process_alive(row)]
        teardown = dict(owned_worker_count=len(identities), remaining_owned_processes=remaining,
                        errors=errors, gpu_idle_not_claimed=True,
                        note='Only owned Ray actors were killed. The driver must exit before external GPU-idle verification.')
        write_new_json(output/'teardown.json', teardown)
    if result is not None:
        result['teardown'] = teardown
        if teardown['remaining_owned_processes'] or teardown['errors']:
            result['status'] = 'failed_teardown'
        write_new_json(output/'result.json', result)
        return result
