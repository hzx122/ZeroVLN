"""D16 full R3 GPU gate using existing real-input, HTTP and service helpers.

Default runs legacy, native_audit and native in separate TP2 services. Serial
warmed requests compare all six tensors exactly; concurrent requests use their
own native audit and artifact contracts without cross-batch equality claims.
Use --validate-inputs for a CPU-only preflight. GPU execution is explicit via
--gpus or CUDA_VISIBLE_DEVICES must contain exactly --tp-size devices.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import socket
import statistics
import sys
import tempfile
import threading
import time


REPO = Path('/workspace/ZeroVLN')
BASE = Path('/data/zerovln/action_head/rl/'
            'd16_resume_acceptance_20260911/optimization')
QUERY = Path('/data/zerovln/action_head/sft_ckpt/'
             'action-head-v3-official-gvln-d16-randomhead-10ep-4node-gb64-20260908-r1/'
             'global_step_010000/dualvln_query/dualvln_query_embeddings.pt')
ARCHIVED_PROBE = REPO / 'tmp/rollout_perf_validation_20260910/native_r3/probe.py'
HTTP_HELPER = REPO / 'tmp/query_parity_probe_runtime/capture_vllm.py'
LOCAL_RUNTIME_PARENT = Path('/tmp/action_head_rl')
# Only these two orchestration fixes may differ when continuing the same probe.
CONTINUATION_SOURCE_EXCEPTIONS = {
    str(Path(__file__).resolve()): 'Add guarded continuation and ownership/cleanup handling.',
    str(REPO / 'verl/action_head_rl/vllm_service.py'): 'Repair shutdown imports; model execution is unchanged.',
}
KEYS = ('query_hidden', 'query_routed_experts', 'query_positions',
        'all_input_ids', 'all_positions', 'all_routed_experts')
AUDIT_KEYS = ('all_input_ids', 'all_positions', 'all_routed_experts',
              'query_positions', 'query_routed_experts')
RUNTIME_DIRS = {
    'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'xdg', 'HF_HOME': 'hf',
    'HF_HUB_CACHE': 'hf/hub', 'HUGGINGFACE_HUB_CACHE': 'hf/hub',
    'TRANSFORMERS_CACHE': 'hf/transformers', 'HF_MODULES_CACHE': 'hf/modules',
    'TRITON_CACHE_DIR': 'triton', 'TORCHINDUCTOR_CACHE_DIR': 'inductor',
    'CUDA_CACHE_PATH': 'cuda', 'PYTORCH_KERNEL_CACHE_PATH': 'kernels',
    'TORCH_EXTENSIONS_DIR': 'extensions', 'TORCH_HOME': 'torch',
    'ACTION_HEAD_COMPILE_CACHE_ROOT': 'compile',
    'ACTION_HEAD_PROCESS_COMPILE_CACHE_ROOT': 'process_compile',
    'VLLM_CACHE_ROOT': 'vllm', 'FLASHINFER_WORKSPACE_BASE': 'flashinfer',
    'DEEPGEMM_JIT_CACHE_DIR': 'deepgemm', 'NUMBA_CACHE_DIR': 'numba',
    'MPLCONFIGDIR': 'matplotlib', 'PYTHONPYCACHEPREFIX': 'pycache',
    'VLLM_RPC_BASE_PATH': 'ipc',
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def runtime_environment(root: Path) -> dict[str, str]:
    values = {key: str(root / suffix) for key, suffix in RUNTIME_DIRS.items()}
    for value in values.values():
        Path(value).mkdir(parents=True, exist_ok=True)
    values.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  ACTION_HEAD_ISOLATE_TP_CACHE='1', TORCHINDUCTOR_COMPILE_THREADS='1',
                  ACTION_HEAD_VLLM_WEIGHT_SYNC='restart',
                  ACTION_HEAD_VLLM_AUDIT_WEIGHTS='0', VLLM_ENABLE_PREFIX_CACHING='0',
                  ACTION_HEAD_QUERY_TOKEN_ID='248060', ACTION_HEAD_QUERY_EOS_TOKEN_ID='248046',
                  ACTION_HEAD_TOOL_END_TOKEN_IDS='248059',
                  ACTION_HEAD_RUN_TOOL_TOKEN_IDS='5917,7681,12756')
    return values


def input_fingerprint(provenance, *, data_identity=False):
    fingerprint = {key: provenance[key] for key in ('model', 'query_checkpoint', 'images',
                   'forced_completion_token_ids', 'input_sha256', 'query_count')}
    if data_identity:
        fingerprint['input_sha256'] = {
            key: value for key, value in fingerprint['input_sha256'].items()
            if key not in CONTINUATION_SOURCE_EXCEPTIONS}
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()


def validate_runtime_continuation(args, provenance):
    """Read-only checks before claiming a stopped local probe's retained cache."""
    prior_path = args.resume_runtime_from.resolve(strict=True)
    if prior_path.name != 'report.json' or not prior_path.is_relative_to(BASE.resolve()):
        raise ValueError('prior report must be a report.json under BASE')
    if prior_path.parent == args.output_dir:
        raise ValueError('continuation requires a new output directory')
    prior = json.loads(prior_path.read_text())
    runtime = Path(prior['runtime_root'])
    if (runtime.parent != LOCAL_RUNTIME_PARENT or not runtime.name.startswith('nr3-')
            or runtime.name == 'nr3-' or not runtime.is_dir()
            or runtime.resolve() != runtime or runtime.stat().st_uid != os.geteuid()):
        raise ValueError('prior runtime must be an owned real local nr3-* directory')
    if (prior.get('runtime_cleaned') is not False or prior.get('status') != 'error'
            or prior.get('output_dir') != str(prior_path.parent)
            or prior.get('gpus') != args.gpus
            or prior.get('hostname', socket.gethostname()) != socket.gethostname()):
        raise ValueError('prior experiment is not an unfinished matching local GPU probe')
    if ('legacy' in args.modes or not set(args.modes).issubset(prior['requested_modes'])
            or any(row.get('status') == 'pass' and row.get('mode') in args.modes
                   for row in prior.get('modes', []))):
        raise ValueError('continuation must request only unfinished native modes')
    for suffix in RUNTIME_DIRS.values():
        if not (runtime / suffix).resolve().is_relative_to(runtime):
            raise ValueError('runtime cache path escapes its owned directory')
    shutdown_path = prior_path.with_name('manual_shutdown.json')
    shutdown = json.loads(shutdown_path.read_text())
    tracked = shutdown.get('tracked_pids', [])
    if (shutdown.get('ok') is not True or shutdown.get('surviving_pids') != []
            or shutdown.get('gpu_ids') != args.gpus or not tracked
            or shutdown.get('hostname', socket.gethostname()) != socket.gethostname()
            or any(type(pid) is not int or pid <= 0 or Path(f'/proc/{pid}').exists()
                   for pid in tracked)):
        raise ValueError('manual shutdown must confirm matching GPUs and no live tracked PIDs')
    inputs_path = prior_path.with_name('inputs.json')
    if prior.get('inputs') != str(inputs_path):
        raise ValueError('prior input provenance must belong to the same output directory')
    previous_inputs = json.loads(inputs_path.read_text())
    previous_fingerprint = input_fingerprint(previous_inputs)
    if previous_inputs.get('input_fingerprint') != previous_fingerprint:
        raise ValueError('prior full input fingerprint is inconsistent')
    identity = input_fingerprint(provenance, data_identity=True)
    if identity != input_fingerprint(previous_inputs, data_identity=True):
        raise ValueError('continuation data/model/query/token/native-code identity differs')
    reference = args.reference_dir.resolve(strict=True)
    if not reference.is_relative_to(BASE.resolve()):
        raise ValueError('legacy reference must be under BASE')
    record = json.loads((reference / 'record.json').read_text())
    reference_fingerprint = record.get('input_fingerprint')
    allowed_reference = previous_inputs.get('continuation', {}).get(
        'reference_input_fingerprint', previous_fingerprint)
    if (record.get('status') != 'pass' or record.get('mode') != 'legacy'
            or reference_fingerprint != allowed_reference):
        raise ValueError('legacy reference must match the prior validated input lineage')
    return {
        'prior_report': str(prior_path), 'prior_inputs': str(inputs_path),
        'manual_shutdown': str(shutdown_path), 'runtime_root': str(runtime),
        'hostname': socket.gethostname(), 'gpus': args.gpus,
        'reference_directory': str(reference),
        'reference_input_fingerprint': reference_fingerprint,
        'prior_input_fingerprint': previous_fingerprint,
        'input_fingerprint': provenance['input_fingerprint'],
        'data_identity_sha256': identity,
        'source_changes': {
            path: {'before': previous_inputs['input_sha256'][path],
                   'after': provenance['input_sha256'][path], 'reason': reason}
            for path, reason in CONTINUATION_SOURCE_EXCEPTIONS.items()},
        'evidence_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (prior_path, inputs_path, shutdown_path,
                                         reference / 'record.json')},
        'ownership_transfer': {'from': str(prior_path.parent), 'to': str(args.output_dir),
                               'pid': os.getpid(), 'accepted_unix': time.time()},
    }


def claim_runtime(continuation):
    """Keep an exclusive process lock until shutdown and cache removal finish."""
    marker = Path(continuation['runtime_root']) / 'continuation_owner.json'
    if marker.is_symlink():
        raise ValueError('runtime ownership marker must not be a symlink')
    owner = marker.open('a+')
    try:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner.seek(0)
        previous = owner.read()
        if previous and json.loads(previous)['ownership_transfer']['to'] != continuation['ownership_transfer']['from']:
            raise ValueError('runtime has already been transferred to another experiment')
        owner.seek(0)
        owner.truncate()
        owner.write(json.dumps(continuation, indent=2) + '\n')
        owner.flush()
        os.fsync(owner.fileno())
    except Exception:
        owner.close()
        raise
    return owner


def load_inputs(args):
    """Reuse the archived loader, updating its provenance to the actual D16 input."""
    import torch

    archived = runpy.run_path(str(ARCHIVED_PROBE.resolve()))
    loader = archived['inputs']
    query = args.query_checkpoint.resolve()
    if query.is_dir():
        query = query / 'dualvln_query_embeddings.pt'
    loader.__globals__['QUERY'] = query
    if args.manifest:
        loader.__globals__['SOURCE'] = args.manifest.resolve()
    if args.model:
        loader.__globals__['MODEL'] = args.model.resolve()
    # Resolve compatibility links before recording or reading persistent inputs.
    for key in ('SOURCE', 'QUERY', 'MODEL', 'CAPTURE'):
        loader.__globals__[key] = loader.__globals__[key].resolve()
    provenance, forced = loader()
    source = next(json.loads(line) for line in Path(provenance['source_manifest']).read_text().splitlines()
                  if line.strip() and json.loads(line).get('split') == 'train')
    provenance['resolved_inputs'] = {path: str(Path(path).resolve())
                                     for path in provenance['input_sha256']}
    provenance['images'] = [str(Path(path).resolve()) for path in provenance['images']]
    for message in source['slow_messages']:
        for part in message['content'] if isinstance(message.get('content'), list) else []:
            if isinstance(part, dict) and part.get('type') == 'image_url':
                path = Path(part['image_url']['url'].removeprefix('file://')).resolve()
                part['image_url']['url'] = 'file://' + str(path)
    checkpoint = torch.load(query, map_location='cpu', weights_only=True)
    embeddings = checkpoint['query_embeddings']
    if (checkpoint.get('query_count') != 16 or embeddings.ndim != 2 or embeddings.shape[0] != 16
            or checkpoint.get('hidden_convention') != 'final_norm'):
        raise ValueError('GPU gate requires a D16 final_norm query checkpoint')
    config = json.loads((Path(provenance['model']) / 'config.json').read_text())
    text_config = config.get('text_config', config)
    if embeddings.shape[1] != text_config['hidden_size'] or len(forced) + 17 > 256:
        raise ValueError('query hidden size or D16 completion budget violates the model contract')
    provenance.update(query_count=16, query_shape=list(embeddings.shape),
                      layer_count=text_config['num_hidden_layers'],
                      top_k=text_config['num_experts_per_tok'], max_tokens=256,
                      forced_completion_reason='Replay the archived tool tokens followed by 16 query forwards and EOS.')
    provenance['helper_paths'] = {
        'archived_inputs': {'requested': str(ARCHIVED_PROBE), 'resolved': str(ARCHIVED_PROBE.resolve())},
        'http': {'requested': str(HTTP_HELPER), 'resolved': str(HTTP_HELPER.resolve())},
    }
    for path in (ARCHIVED_PROBE.resolve(), HTTP_HELPER.resolve(), Path(__file__).resolve(),
                 REPO / 'verl/action_head/serve/native_r3_capture.py',
                 REPO / 'verl/action_head/serve/dualvln_query_runtime.py',
                 REPO / 'verl/action_head_rl/vllm_service.py'):
        provenance['input_sha256'][str(path)] = archived['sha256'](path)
    # Keep data/model identity separate from mode-specific runtime flags.
    provenance['input_fingerprint'] = input_fingerprint(provenance)
    provenance['data_identity_sha256'] = input_fingerprint(provenance, data_identity=True)
    post_json = runpy.run_path(str(HTTP_HELPER.resolve()))['post_json']
    return provenance, source, forced, post_json


def validate_artifact(tensors, provenance, forced):
    import torch
    from action_head_rl.artifact import QueryRoutingArtifact
    from action_head.serve.native_r3_capture import _comparison

    if set(tensors) != set(KEYS):
        raise ValueError('full R3 artifact must contain exactly all six tensors')
    QueryRoutingArtifact(**tensors).validate(
        hidden_size=provenance['query_shape'][1], layer_count=provenance['layer_count'],
        top_k=provenance['top_k'], require_full=True, query_count=16)
    if not torch.isfinite(tensors['query_hidden']).all():
        raise ValueError('query_hidden contains non-finite values')
    token_ids = tensors['all_input_ids']
    expected_tail = torch.tensor(forced + [248060] * 16, dtype=torch.int64)
    if not torch.equal(token_ids[-len(expected_tail):], expected_tail):
        raise ValueError('all_input_ids does not end with the exact forced completion and 16 query tokens')
    checks = {
        'positions_from_zero': _comparison(torch.arange(len(token_ids)), tensors['all_positions']),
        'query_routes_equal_full_slice': _comparison(
            tensors['all_routed_experts'][tensors['query_positions']], tensors['query_routed_experts']),
    }
    if not all(value['equal'] for value in checks.values()):
        raise ValueError('full R3 token positions or query route slice mismatch')
    return checks


def validate_native_report(audit, mode: str, request_id: str, token_count: int):
    expected = AUDIT_KEYS if mode == 'native_audit' else ('query_positions', 'query_routed_experts')
    checks = audit.get('comparisons', {})
    if (audit.get('status') != 'pass' or audit.get('mode') != mode
            or audit.get('query_request_id') != request_id or audit.get('token_count') != token_count
            or audit.get('num_cached_tokens') != 0
            or not all(checks.get(key, {}).get('equal') is True for key in expected)
            or audit.get('published') is not (mode == 'native')):
        raise ValueError(f'{mode} same-request report is incomplete or failed')
    if mode == 'native' and not audit.get('source_artifact', '').endswith('.query.safetensors'):
        raise ValueError('native publication did not consume the worker query-only sidecar')


def compare_records(reference: Path, candidate: Path, *, continuation=None):
    from safetensors.torch import load_file
    from action_head.serve.native_r3_capture import _comparison

    left = json.loads((reference / 'record.json').read_text())
    right = json.loads((candidate / 'record.json').read_text())
    if left['input_fingerprint'] != right['input_fingerprint']:
        if (continuation is None
                or str(reference.resolve()) != continuation['reference_directory']
                or left['input_fingerprint'] != continuation['reference_input_fingerprint']
                or right['input_fingerprint'] != continuation['input_fingerprint']
                or right.get('data_identity_sha256') != continuation['data_identity_sha256']):
            raise ValueError('reference input/model/query/source fingerprint differs')
    expected = load_file(str(reference / 'query.safetensors'), device='cpu')
    actual = load_file(str(candidate / 'query.safetensors'), device='cpu')
    comparisons = {key: _comparison(expected[key], actual[key]) for key in KEYS}
    text_equal = left['clean_completion'] == right['clean_completion']
    result = {'reference': str(reference), 'candidate': str(candidate),
              'reference_input_fingerprint': left['input_fingerprint'],
              'input_fingerprint': right['input_fingerprint'],
              'continuation_data_identity_sha256': (
                  continuation['data_identity_sha256'] if continuation else None),
              'text_equal': text_equal, 'tensor_comparisons': comparisons,
              'pass': text_equal and all(value['equal'] for value in comparisons.values())}
    write_json(candidate / 'comparison.json', result)
    if not result['pass']:
        raise ValueError(f'fixed serial request differs from reference: {candidate}')
    return result


def capture_request(*, directory, mode, cache, audit_root, port, source, forced,
                    provenance, post_json, timeout, barrier=None):
    from safetensors.torch import load_file
    from action_head.vllm_bridge import request_queries, strip_query_tail

    directory.mkdir(parents=True, exist_ok=False)
    payload = {'model': 'Qwen3.6-35B', 'messages': source['slow_messages'], 'temperature': 0.0,
               'seed': 0, 'max_tokens': 256, 'chat_template_kwargs': {'enable_thinking': False}}
    request_id = request_queries(payload, 16)
    record = {'status': 'running', 'mode': mode, 'request_id': request_id, 'payload': payload,
              'input_fingerprint': provenance['input_fingerprint'],
              'data_identity_sha256': provenance['data_identity_sha256']}
    try:
        if barrier:
            barrier.wait(timeout=30)
        record['started_monotonic'] = time.monotonic()
        response = post_json(f'http://127.0.0.1:{port}/v1/chat/completions', payload, timeout)
        record['ended_monotonic'] = time.monotonic()
        record['http_seconds'] = record['ended_monotonic'] - record['started_monotonic']
        record['response'] = response
        clean, tail = strip_query_tail(response['choices'][0]['message']['content'], query_count=16)
        if tail != 16 or clean != source['slow_completion']:
            raise ValueError('response must contain the exact archived completion and all 16 query carriers')
        record.update(clean_completion=clean, query_tail_count=tail, usage=response.get('usage'))
        artifact = cache / f'{request_id}.safetensors'
        if not artifact.is_file():
            raise ValueError('complete artifact was not published before HTTP completion')
        tensors = load_file(str(artifact), device='cpu')
        record['artifact_checks'] = validate_artifact(tensors, provenance, forced)
        record['token_count'] = len(tensors['all_input_ids'])
        record['tensor_shapes'] = {key: list(value.shape) for key, value in tensors.items()}
        # Preserve every formal artifact before removing the request cache.
        shutil.copy2(artifact, directory / 'query.safetensors')
        if mode != 'legacy':
            audit_file = audit_root / f'{request_id}.json'
            audit = json.loads(audit_file.read_text())
            write_json(directory / 'native_report.json', audit)
            validate_native_report(audit, mode, request_id, record['token_count'])
            if mode == 'native' and (cache / f'{request_id}.query.safetensors').exists():
                raise ValueError('native sidecar remained after complete publication')
        record['status'] = 'pass'
    except Exception as error:
        record.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(directory / 'record.json', record)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expr-name', default='d16-native-r3-gpu-r1-20260912')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--query-checkpoint', type=Path, default=QUERY)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--modes', default='legacy,native_audit,native')
    parser.add_argument('--reference-dir', type=Path, help='Prior legacy warm_00 request directory')
    parser.add_argument('--resume-runtime-from', type=Path,
                        help='Failed local report.json with a confirmed manual shutdown and retained cache')
    parser.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    parser.add_argument('--tp-size', type=int, default=2)
    parser.add_argument('--port', type=int, default=28898)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--startup-timeout', type=int, default=1800)
    parser.add_argument('--max-num-batched-tokens', type=int, default=8192)
    parser.add_argument('--validate-inputs', action='store_true')
    args = parser.parse_args(argv)
    args.modes = args.modes.split(',')
    if (len(set(args.modes)) != len(args.modes)
            or any(mode not in ('legacy', 'native_audit', 'native') for mode in args.modes)):
        parser.error('--modes must be a unique comma-separated subset of legacy,native_audit,native')
    if 'legacy' in args.modes and args.modes[0] != 'legacy':
        parser.error('legacy must run first to establish the serial reference')
    if not args.validate_inputs and 'legacy' not in args.modes and args.reference_dir is None:
        parser.error('a mode-only run requires --reference-dir from an existing legacy run')
    if args.repeats < 2 or args.concurrency < 4:
        parser.error('at least two warmed repetitions and four concurrent requests are required')
    devices = [value.strip() for value in args.gpus.split(',') if value.strip()]
    if args.tp_size <= 0:
        parser.error('--tp-size must be positive')
    if (not args.validate_inputs
            and (len(devices) != args.tp_size or len(set(devices)) != args.tp_size)):
        parser.error('set --gpus or CUDA_VISIBLE_DEVICES to exactly --tp-size distinct GPUs')
    args.gpus = ','.join(devices)
    if (Path(args.expr_name).name != args.expr_name or args.expr_name in ('', '.', '..')):
        parser.error('--expr-name must be a single directory name')
    args.output_dir = (args.output_dir or BASE / args.expr_name).resolve()
    if not args.output_dir.is_relative_to(BASE.resolve()):
        parser.error('output directory must be under the authorized data-disk optimization root')
    if args.resume_runtime_from and (args.validate_inputs or 'legacy' in args.modes):
        parser.error('runtime continuation requires mode-only native execution with a legacy reference')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    local_parent = LOCAL_RUNTIME_PARENT
    local_parent.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix='nr3-', dir=local_parent))
    original_environment = dict(os.environ)
    report = {'status': 'running', 'modes': [], 'requested_modes': args.modes,
              'query_count': 16, 'prefix_caching': False, 'output_dir': str(args.output_dir),
              'runtime_root': str(runtime), 'gpus': args.gpus, 'hostname': socket.gethostname(),
              'tensor_parallel_size': args.tp_size,
              'comparison_scope': 'six tensors and text on fixed serial warmed requests; '
                                  'concurrent captures use same-request checks only'}
    service = None
    runtime_owner = None
    continuation = None
    try:
        os.environ.update(runtime_environment(runtime))
        os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.validate_inputs else args.gpus
        sys.path.insert(0, str(REPO / 'verl'))
        provenance, source, forced, post_json = load_inputs(args)
        if args.resume_runtime_from:
            continuation = validate_runtime_continuation(args, provenance)
            runtime_owner = claim_runtime(continuation)
            preflight_runtime = runtime
            runtime = Path(continuation['runtime_root'])
            report.update(runtime_root=str(runtime), continuation=continuation)
            provenance['continuation'] = continuation
            shutil.rmtree(preflight_runtime)
            os.environ.update(runtime_environment(runtime))
            write_json(args.output_dir / 'continuation.json', continuation)
        write_json(args.output_dir / 'inputs.json', provenance)
        report['inputs'] = str(args.output_dir / 'inputs.json')
        report['runtime_environment'] = {key: os.environ[key] for key in RUNTIME_DIRS}
        if args.validate_inputs:
            report['status'] = 'cpu_inputs_pass'
            return report
        # Fail before touching a service if another process already owns the port.
        with socket.socket() as probe:
            probe.bind(('0.0.0.0', args.port))
        from action_head_rl.vllm_service import VllmService
        os.environ['ACTION_HEAD_FORCED_COMPLETION_IDS'] = ','.join(map(str, forced))
        reference = args.reference_dir.resolve() if args.reference_dir else None
        for mode in args.modes:
            directory = args.output_dir / mode
            directory.mkdir()
            cache = runtime / mode / 'query_cache'
            audit_root = directory / 'native_r3_audit'
            os.environ.update(ACTION_HEAD_R3_CAPTURE_MODE=mode,
                              ACTION_HEAD_R3_AUDIT_DIR=str(audit_root))
            service = VllmService(
                repo=str(REPO), model=provenance['model'], query_checkpoint=provenance['query_checkpoint'],
                query_cache=str(cache), log_file=str(directory / 'vllm.log'), port=args.port,
                capture_full_routes=True, bulk_route_capture=True, startup_timeout=args.startup_timeout,
                max_num_seqs=max(16, args.concurrency),
                max_num_batched_tokens=args.max_num_batched_tokens,
                tensor_parallel_size=args.tp_size)
            mode_report = {'mode': mode, 'status': 'running', 'serial': [], 'concurrent': []}
            report['modes'].append(mode_report)
            started = time.monotonic()
            service.start()
            mode_report['startup_seconds'] = time.monotonic() - started
            common = dict(mode=mode, cache=cache, audit_root=audit_root, port=args.port,
                          source=source, forced=forced, provenance=provenance,
                          post_json=post_json, timeout=args.timeout)
            cold = capture_request(directory=directory / 'cold', **common)
            mode_report['cold_http_seconds'] = cold['http_seconds']
            for index in range(args.repeats):
                destination = directory / f'warm_{index:02d}'
                record = capture_request(directory=destination, **common)
                if reference is None:
                    reference = destination
                comparison = compare_records(reference, destination, continuation=continuation)
                mode_report['serial'].append({'directory': str(destination),
                    'http_seconds': record['http_seconds'], 'six_tensor_exact': comparison['pass']})
            barrier = threading.Barrier(args.concurrency)
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(capture_request,
                    directory=directory / f'concurrent_{index:02d}', barrier=barrier, **common)
                    for index in range(args.concurrency)]
                for future in futures:
                    mode_report['concurrent'].append(future.result())
            rows = mode_report['concurrent']
            overlap = min(row['ended_monotonic'] for row in rows) - max(row['started_monotonic'] for row in rows)
            if len({row['request_id'] for row in rows}) != args.concurrency or overlap <= 0:
                raise ValueError('concurrent requests did not have unique IDs and overlapping lifetimes')
            mode_report['concurrent_overlap_seconds'] = overlap
            mode_report['warm_http_median_seconds'] = statistics.median(
                row['http_seconds'] for row in mode_report['serial'])
            service.stop()
            service = None
            mode_report.update(status='pass', service_stopped=True)
            write_json(directory / 'report.json', mode_report)
            write_json(args.output_dir / 'report.json', report)
        report['status'] = 'pass'
        report['serial_reference'] = str(reference)
        return report
    except Exception as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            if service is not None:
                service.stop()
            report['service_stopped'] = True
        except Exception as error:
            report.update(status='error', service_stopped=False,
                          cleanup_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            # A failed shutdown can leave live writers; preserve their runtime
            # and report the failure instead of deleting a live process's files.
            try:
                if report.get('service_stopped'):
                    shutil.rmtree(runtime)
            finally:
                if runtime_owner is not None:
                    runtime_owner.close()
                report['runtime_cleaned'] = not runtime.exists()
                os.environ.clear()
                os.environ.update(original_environment)
                write_json(args.output_dir / 'report.json', report)


if __name__ == '__main__':
    result = main()
    print(json.dumps({'status': result['status'], 'output_dir': result['output_dir']}, ensure_ascii=False))
