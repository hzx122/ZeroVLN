"""Compare a native in-place load with a fresh service on the same checkpoint.

Reuse the archived real multimodal input probe, production query bridge,
vLLM model loader, and VllmService lifecycle. Never modify checkpoint weights.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, runpy, secrets, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
from action_head_rl.vllm_service import VllmService

REPO = Path('/workspace/ZeroVLN')
BASE = Path('/data/zerovln/action_head/rl/d16_resume_acceptance_20260911')
QUERY = Path('/data/zerovln/action_head/sft_ckpt/action-head-v3-official-gvln-d16-randomhead-10ep-4node-gb64-20260908-r1/global_step_010000/dualvln_query')
SLEEP_OUTPUT_ROOT = BASE / 'optimization/d16-head-residency-20260912'

def import_probe(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def legacy_main(args):
    import torch
    from safetensors.torch import load_file
    from action_head.vllm_bridge import request_queries, strip_query_tail
    from action_head_rl.trainer import _query_file

    output = args.output_dir.resolve()
    if not output.is_relative_to(BASE):
        raise ValueError('output must be under authorized acceptance root')
    output.mkdir(parents=True, exist_ok=False)
    local_parent = Path('/tmp/action_head_rl')
    local_parent.mkdir(parents=True, exist_ok=True)
    local_runtime = Path(tempfile.mkdtemp(prefix='inplace-sync-', dir=local_parent))
    (local_runtime / 'tmp').mkdir()
    os.environ['TMPDIR'] = str(local_runtime / 'tmp')
    archived = import_probe('archived_native_inputs', REPO / 'tmp/rollout_perf_validation_20260910/native_r3/probe.py')
    capture = import_probe('archived_capture_http', REPO / 'tmp/query_parity_probe_runtime/capture_vllm.py')
    provenance, forced = archived.inputs()
    source = next(json.loads(line) for line in archived.SOURCE.read_text().splitlines()
                  if line.strip() and json.loads(line).get('split') == 'train')
    os.environ.update(ACTION_HEAD_PROCESS_COMPILE_CACHE_ROOT=str(local_runtime / 'compile_cache'),
                      ACTION_HEAD_VLLM_WEIGHT_SYNC='inplace', ACTION_HEAD_VLLM_AUDIT_WEIGHTS='1',
                      ACTION_HEAD_FORCED_COMPLETION_IDS=','.join(map(str, forced)))
    report = {'status': 'running', 'target': str(args.target), 'query': str(QUERY),
              'input_provenance': provenance, 'query_count': 16}
    service = VllmService(repo=str(REPO), model=str(archived.MODEL), query_checkpoint=str(_query_file(str(QUERY))),
                          query_cache=str(local_runtime / 'query_cache'), log_file=str(output / 'vllm.log'),
                          port=28898, capture_full_routes=True, bulk_route_capture=True,
                          max_num_seqs=16, max_num_batched_tokens=8192)
    def fingerprint():
        return service._post('/collective_rpc', {'method': 'action_head_weight_fingerprint', 'timeout': 1800})['results']
    def request(label):
        payload = {'model': 'Qwen3.6-35B', 'messages': source['slow_messages'], 'temperature': 0.0,
                   'max_tokens': max(256, len(forced) + 20), 'chat_template_kwargs': {'enable_thinking': False}}
        request_id = request_queries(payload, 16)
        response = capture.post_json('http://127.0.0.1:28898/v1/chat/completions', payload, 600)
        text, tail = strip_query_tail(response['choices'][0]['message']['content'], query_count=16)
        if tail != 16:
            raise RuntimeError(f'expected 16 query carriers, got {tail}')
        artifact = local_runtime / 'query_cache' / (request_id + '.safetensors')
        deadline = time.monotonic() + 30
        while not artifact.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        tensors = load_file(str(artifact))
        (output / (label + '.json')).write_text(json.dumps({'request': payload, 'response': response,
                                                           'artifact': str(artifact)}, ensure_ascii=False))
        return text, tensors
    try:
        service.start()
        initial = fingerprint()
        report['initial_weights'] = initial
        request('initial')
        original_pid = service.process.pid
        started = time.monotonic()
        service.restart(str(args.target))
        report['inplace_seconds'] = time.monotonic() - started
        report['pid_preserved'] = service.process.pid == original_pid
        reloaded = fingerprint()
        report['inplace_weights'] = reloaded
        report['weights_changed'] = initial != reloaded
        reloaded_text, reloaded_tensors = request('inplace')
        service.stop()
        started = time.monotonic()
        service.start()
        report['fresh_start_seconds'] = time.monotonic() - started
        fresh = fingerprint()
        fresh_text, fresh_tensors = request('fresh')
        report['initial_weights'] = initial
        report['inplace_weights'] = reloaded
        report['fresh_weights'] = fresh
        report['weights_changed'] = initial != reloaded
        report['weights_match_fresh'] = reloaded == fresh
        report['text_match'] = reloaded_text == fresh_text
        report['tensor_comparison'] = {}
        if reloaded_tensors.keys() != fresh_tensors.keys():
            raise RuntimeError('query artifact field set changed')
        for name, left in reloaded_tensors.items():
            right = fresh_tensors[name]
            same = left.shape == right.shape and torch.equal(left, right)
            entry = {'shape': list(left.shape), 'exact': same}
            if left.shape == right.shape and left.numel() and left.is_floating_point():
                entry['max_abs'] = float((left.float() - right.float()).abs().max())
            report['tensor_comparison'][name] = entry
        report['status'] = 'pass' if (report['pid_preserved'] and report['weights_changed']
            and report['weights_match_fresh'] and report['text_match']
            and all(row['exact'] for row in report['tensor_comparison'].values())) else 'fail'
    except Exception as error:
        report['status'] = 'error'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        service.stop()
        shutil.rmtree(local_runtime, ignore_errors=True)
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if report['status'] != 'pass':
        raise RuntimeError('weight synchronization acceptance failed')

def compare_snapshots(reference, candidate, *, pointers):
    """Require all TP ranks and the real query/buffer state, not only file names."""
    def indexed(rows):
        result = {row['tp_rank']: row for row in rows}
        if len(rows) != 2 or set(result) != {0, 1}:
            raise ValueError('snapshot requires exactly TP ranks0,1')
        return result
    left, right = indexed(reference), indexed(candidate)
    comparisons = []
    for rank in sorted(left):
        old, new = left[rank], right[rank]
        checks = {'query': old['query']['sha256'] == new['query']['sha256'],
                  'query_shape': old['query']['shape'] == new['query']['shape'],
                  'query_dtype': old['query']['dtype'] == new['query']['dtype'],
                  'query_drained': new['query']['pending_query_phases'] == 0}
        for family in ('parameters', 'buffers'):
            checks[family] = old[family]['sha256'] == new[family]['sha256']
            if pointers:
                checks[family + '_pointers'] = (
                    [(row['name'], row['pointer']) for row in old[family]['tensors']]
                    == [(row['name'], row['pointer']) for row in new[family]['tensors']])
        if pointers:
            checks['query_pointer'] = old['query']['pointer'] == new['query']['pointer']
            checks['worker_pid'] = old['pid'] == new['pid']
        checks['hashes_present'] = all(row.get('hashes_included') is True for row in (old, new))
        comparisons.append({'rank': rank, 'checks': checks, 'pass': all(checks.values())})
    return {'ranks': comparisons, 'pass': all(row['pass'] for row in comparisons)}


def service_exit_result(api_code, expected_pids, exit_records, survivors, *, graceful,
                        shutdown_started=0.0):
    """SIGTERM after requested shutdown is explicit termination; SIGABRT fails."""
    observed = {int(row['pid']): int(row['returncode']) for row in exit_records}
    times = {int(row['pid']): row.get('observed_monotonic') for row in exit_records}
    missing = sorted(set(expected_pids) - set(observed))
    bad = {str(pid): observed[pid] for pid in expected_pids
           if pid in observed and (observed[pid] not in (0, -signal.SIGTERM)
              or (observed[pid] == -signal.SIGTERM
                  and (times[pid] is None or times[pid] < shutdown_started)))}
    passed = (graceful and api_code in (0, -signal.SIGTERM) and not missing
              and not bad and not survivors)
    return {'pass': passed, 'api_returncode': api_code,
            'explicit_shutdown': True, 'graceful_deadline_met': graceful,
            'expected_child_pids': sorted(expected_pids), 'child_exit_codes': observed,
            'missing_exit_evidence': missing, 'unexpected_exit_codes': bad,
            'surviving_pids': survivors,
            'accepted_exit_codes_after_requested_shutdown': [0, -signal.SIGTERM]}


def live_owned_processes(processes):
    import psutil
    alive = []
    for process in processes:
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                alive.append(process)
        except psutil.NoSuchProcess:
            pass
    return alive


def graceful_stop(service, snapshots, exit_root, directory, timeout=120):
    import psutil

    process = service.process
    if process is None:
        raise RuntimeError('shutdown acceptance requires a started service')
    expected = {row['pid'] for row in snapshots} | {row['parent_pid'] for row in snapshots}
    owned = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
    expected_live = expected.issubset({row.pid for row in live_owned_processes(owned)})
    started = time.monotonic()
    graceful = process.poll() is None and expected_live
    if process.poll() is None:
        # Signal only the API. Its native shutdown asks EngineCore/TP to drain.
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            graceful = False
    _, alive = psutil.wait_procs(owned[1:], timeout=max(0, timeout - (time.monotonic() - started)))
    survivors = [row.pid for row in live_owned_processes(alive)]
    evidence = [json.loads(path.read_text()) for path in exit_root.glob('*_*.json')
                if not path.name.startswith('installed_')]
    result = service_exit_result(process.poll(), expected, evidence, survivors,
                                 graceful=graceful, shutdown_started=started)
    result.update(seconds=time.monotonic() - started, tracked_pids=[row.pid for row in owned],
                  all_expected_alive_before_shutdown=expected_live)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'exit_report.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result['pass']:
        # Cleanup may terminate owned survivors, but cannot repair acceptance.
        sys.path.insert(0, str(REPO / 'verl/scripts/train/action_head_bootstrap'))
        sys.path.insert(0, str(REPO / 'vln_demo/src'))
        from collection_runtime import stop_services
        stop_services([row.pid for row in live_owned_processes(owned)], [],
                      directory / 'fallback_shutdown.json', os.environ['CUDA_VISIBLE_DEVICES'])
    service.stop()
    return result


def sleep_worker(args, runtime):
    """The only branch that imports Torch or starts the independently owned TP2."""
    from types import SimpleNamespace

    helpers = runpy.run_path(str(REPO / 'scripts/eval/action_head_d16_native_r3_probe.py'))
    output = args.output_dir.resolve()
    write = helpers['write_json']
    report = {'status': 'running', 'scope': 'real TP2 sleep1, same/changed weights, fixed D16/R3 payload',
              'sleep_cycles': [], 'service_exits': [], 'output_dir': str(output),
              'gpu_executed': not args.validate_inputs, 'gpus': args.gpus,
              'target': str(args.target.resolve()), 'query_checkpoint': str(QUERY),
              'thresholds': {'text_and_six_r3_tensors': 'exact', 'parameters_buffers_query': 'exact',
                             'same_process_pointers': 'exact', 'released_pool_fraction_min': 0.90},
              'parent_inplace_reference': str(BASE / 'optimization/inplace_sync_localcache_r5/report.json')}
    service = None
    latest = []
    exit_root = stage_output = None
    try:
        os.environ.update(helpers['runtime_environment'](runtime / 'driver'))
        provenance, source, forced, post_json = helpers['load_inputs'](SimpleNamespace(
            query_checkpoint=QUERY, manifest=None, model=None))
        if not (args.target / 'config.json').is_file():
            raise FileNotFoundError(args.target / 'config.json')
        if args.target.resolve() == Path(provenance['model']).resolve():
            raise ValueError('changed-weight target must differ from initial model')
        paths = [Path(__file__), REPO / 'scripts/eval/action_head_d16_sleep_weight_audit.py',
                 REPO / 'verl/action_head_rl/vllm_service.py', REPO / 'scripts/serve/vllm/vllm-vln-35B.sh']
        report['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        report['input_provenance'] = provenance
        report['target_config_sha256'] = hashlib.sha256((args.target / 'config.json').read_bytes()).hexdigest()
        write(output / 'inputs.json', report)
        if args.validate_inputs:
            report['status'] = 'inputs_validated'
            return report
        import torch
        from action_head_d16_sleep_weight_audit import tensor_sha
        query_file = QUERY / 'dualvln_query_embeddings.pt'
        expected_query = tensor_sha(torch.load(query_file, map_location='cpu', weights_only=True)[
            'query_embeddings'].to(torch.bfloat16))
        report['expected_query_tensor_sha256'] = expected_query

        def launch(label, model):
            nonlocal service, latest, exit_root, stage_output
            stage_output = output / label
            stage_output.mkdir()
            stage = runtime / label
            os.environ.update(helpers['runtime_environment'](stage))
            bootstrap = stage / 'diagnostic_bootstrap'
            bootstrap.mkdir()
            original_bootstrap = REPO / 'verl/action_head/serve/bootstrap/sitecustomize.py'
            (bootstrap / 'sitecustomize.py').write_text(
                'import runpy\nrunpy.run_path(' + repr(str(original_bootstrap)) + ')\n'
                'from action_head_d16_sleep_weight_audit import install_exit_audit\ninstall_exit_audit()\n')
            exit_root = stage_output / 'process_exits'
            exit_root.mkdir()
            os.environ.update(
                CUDA_VISIBLE_DEVICES=args.gpus,
                PYTHONPATH=str(REPO / 'scripts/eval') + os.pathsep + os.environ.get('PYTHONPATH', ''),
                ACTION_HEAD_VLLM_WEIGHT_SYNC='inplace', ACTION_HEAD_VLLM_AUDIT_WEIGHTS='1',
                ACTION_HEAD_VLLM_AUDIT_EXTENSION='action_head_d16_sleep_weight_audit.SleepWeightAuditExtension',
                ACTION_HEAD_VLLM_DIAGNOSTIC_BOOTSTRAP=str(bootstrap),
                ACTION_HEAD_SLEEP_EXIT_AUDIT_DIR=str(exit_root),
                ACTION_HEAD_R3_CAPTURE_MODE='legacy',
                ACTION_HEAD_FORCED_COMPLETION_IDS=','.join(map(str, forced)),
                VLLM_GPU_MEMORY_UTILIZATION=str(args.gpu_memory_utilization))
            import socket
            with socket.socket() as port_check:
                port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                port_check.bind(('0.0.0.0', args.port))
            latest = []
            service = VllmService(repo=str(REPO), model=str(model), query_checkpoint=str(query_file),
                query_cache=str(stage / 'query_cache'), log_file=str(stage_output / 'vllm.log'),
                port=args.port, capture_full_routes=True, bulk_route_capture=True,
                max_num_seqs=16, max_num_batched_tokens=8192,
                startup_timeout=args.startup_timeout, enable_sleep_mode=True)
            write(stage_output / 'launch.json', {'model': str(model), 'runtime_root': str(stage),
                'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                'sleep_mode': True, 'gpu_memory_utilization': args.gpu_memory_utilization,
                'compile_root': os.environ['ACTION_HEAD_PROCESS_COMPILE_CACHE_ROOT']})
            service.start()

        def snapshot(label, hashes=True):
            nonlocal latest
            response = service._post('/collective_rpc', {
                'method': 'action_head_sleep_snapshot', 'kwargs': {'include_hashes': hashes},
                'timeout': args.startup_timeout})
            rows = response.get('results', [])
            if len(rows) != 2 or {row['tp_rank'] for row in rows} != {0, 1}:
                raise RuntimeError('missing/duplicate TP2 snapshot')
            rows.sort(key=lambda row: row['tp_rank'])
            if any(not row['enable_sleep_mode'] or row['query']['pending_query_phases']
                   or (hashes and row['query']['sha256'] != expected_query) for row in rows):
                raise RuntimeError('sleep/query version or drain gate failed')
            write(stage_output / (label + '.json'), rows)
            latest = rows
            return rows

        def request(label):
            directory = stage_output / label
            helpers['capture_request'](directory=directory, mode='legacy',
                cache=Path(service.query_cache), audit_root=stage_output / 'unused_native_audit',
                port=args.port, source=source, forced=forced, provenance=provenance,
                post_json=post_json, timeout=args.timeout)
            # Preserve the validated artifact at its durable location only.
            record = json.loads((directory / 'record.json').read_text())
            (Path(service.query_cache) / (record['request_id'] + '.safetensors')).unlink()
            return directory

        def sleep_cycle(cycle, baseline, reference):
            before = snapshot(f'cycle_{cycle}_before', hashes=False)
            started = time.monotonic()
            service.sleep(level=1)
            sleep_seconds = time.monotonic() - started
            asleep = snapshot(f'cycle_{cycle}_asleep', hashes=False)
            released = [right['free_bytes'] - left['free_bytes'] for left, right in zip(before, asleep)]
            release_pass = all(left['allocator_bytes'] > 0
                and left['allocator_bytes_by_tag'].get('weights', 0) > 0
                and left['allocator_bytes_by_tag'].get('kv_cache', 0) > 0
                and freed >= left['allocator_bytes'] * .90 for left, freed in zip(before, released))
            started = time.monotonic()
            service.wake_workers()
            wake_seconds = time.monotonic() - started
            awake = snapshot(f'cycle_{cycle}_awake')
            state_check = compare_snapshots(baseline, awake, pointers=True)
            cycle_report = {'cycle': cycle, 'sleep_seconds': sleep_seconds, 'wake_seconds': wake_seconds,
                'model': service.model,
                'released_bytes_by_rank': released, 'release_pass': release_pass,
                'scheduler_stayed_paused': service.is_paused(), 'state_comparison': state_check}
            report['sleep_cycles'].append(cycle_report)
            write(output / 'worker_report.json', report)
            if not release_pass or not state_check['pass'] or not cycle_report['scheduler_stayed_paused']:
                raise RuntimeError('sleep/wake memory, state or paused barrier gate failed')
            service._post('/reset_mm_cache')
            service.resume()
            after_request = request(f'cycle_{cycle}_request')
            cycle_report['artifact_comparison'] = helpers['compare_records'](reference, after_request)
            service.pause()

        launch('sleep_service', provenance['model'])
        request('warm_00')
        reference = request('warm_01')
        service.pause()
        initial = snapshot('initial')
        sleep_cycle(0, initial, reference)

        # A real changed model is loaded while scheduler remains paused.
        # Reuse the previously accepted r5 target instead of synthesizing weights.
        started = time.monotonic()
        service.reload_weights(str(args.target), resume=False)
        report['inplace_seconds'] = time.monotonic() - started
        changed = snapshot('changed_weights')
        report['weights_changed'] = any(old['parameters']['sha256'] != new['parameters']['sha256']
                                        for old, new in zip(initial, changed))
        report['reload_scheduler_stayed_paused'] = service.is_paused()
        report['reload_pointer_stable'] = all(
            [(p['name'], p['pointer']) for p in old[family]['tensors']]
            == [(p['name'], p['pointer']) for p in new[family]['tensors']]
            for old, new in zip(initial, changed) for family in ('parameters', 'buffers'))
        if not all(report[key] for key in ('weights_changed', 'reload_scheduler_stayed_paused', 'reload_pointer_stable')):
            raise RuntimeError('changed weight/pointer/version barrier failed')
        provenance = dict(provenance, model=str(args.target.resolve()), policy_version=1,
                          target_config_sha256=report['target_config_sha256'])
        provenance['input_fingerprint'] = helpers['input_fingerprint'](provenance)
        provenance['data_identity_sha256'] = helpers['input_fingerprint'](provenance, data_identity=True)
        write(output / 'target_inputs.json', provenance)
        service.resume()
        request('changed_warm_00')
        changed_request = request('changed_warm_01')
        service.pause()
        changed = snapshot('changed_after_requests')
        # The second cycle follows the real in-place update, exercising that
        # updated kernel storage is still owned by the sleepable weight pool.
        sleep_cycle(1, changed, changed_request)
        changed = snapshot('changed_after_second_sleep')
        report['service_exits'].append(graceful_stop(service, latest, exit_root, stage_output))
        service = None
        if not report['service_exits'][-1]['pass']:
            raise RuntimeError('sleep service failed complete process-exit gate')

        launch('fresh_target', args.target)
        request('warm_00')
        fresh_request = request('warm_01')
        service.pause()
        fresh = snapshot('fresh_after_requests')
        report['fresh_weight_comparison'] = compare_snapshots(changed, fresh, pointers=False)
        report['fresh_artifact_comparison'] = helpers['compare_records'](changed_request, fresh_request)
        report['service_exits'].append(graceful_stop(service, latest, exit_root, stage_output))
        service = None
        if not report['fresh_weight_comparison']['pass'] or not report['service_exits'][-1]['pass']:
            raise RuntimeError('fresh-target weights or process-exit gate failed')
        report['status'] = 'checks_passed'
    except BaseException as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
    finally:
        if service is not None:
            try:
                if latest and service.process is not None and service.process.poll() is None:
                    report['service_exits'].append(graceful_stop(service, latest, exit_root, stage_output))
                else:
                    service.stop()
            except BaseException as error:
                report['cleanup_error'] = f'{type(error).__name__}: {error}'
                try:
                    service.stop()
                except BaseException as fallback:
                    report['fallback_cleanup_error'] = f'{type(fallback).__name__}: {fallback}'
        write(output / 'worker_report.json', report)
    return report


def sleep_gate_main(args):
    output = args.output_dir.resolve()
    if not output.is_relative_to(SLEEP_OUTPUT_ROOT):
        raise ValueError(f'sleep gate output must be under {SLEEP_OUTPUT_ROOT}')
    if args.sleep_worker:
        if not args.runtime or not args.owner_token or not output.is_dir():
            raise ValueError('sleep worker requires a supervisor-owned runtime')
        owner = json.loads((args.runtime / 'owner.json').read_text())
        if owner != {'token': args.owner_token, 'output': str(output)}:
            raise ValueError('runtime ownership mismatch')
        result = sleep_worker(args, args.runtime)
        return 0 if result['status'] in ('checks_passed', 'inputs_validated') else 1
    output.mkdir(parents=True, exist_ok=False)
    args.runtime_parent.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix='sleep-tp2-', dir=args.runtime_parent)).resolve()
    token = secrets.token_hex(16)
    (runtime / 'owner.json').write_text(json.dumps({'token': token, 'output': str(output)}))
    child = None
    result = {}
    tracked = {}
    try:
        import psutil
        helpers = runpy.run_path(str(REPO / 'scripts/eval/action_head_d16_native_r3_probe.py'))
        environment = dict(os.environ)
        environment.update(helpers['runtime_environment'](runtime / 'driver'))
        environment['CUDA_VISIBLE_DEVICES'] = args.gpus
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                   '--sleep-worker', '--runtime', str(runtime), '--owner-token', token]
        with (output / 'worker.log').open('x') as log:
            child = subprocess.Popen(command, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            started = time.monotonic()
            while child.poll() is None:
                try:
                    parent = psutil.Process(child.pid)
                    for process in [parent, *parent.children(recursive=True)]:
                        tracked[process.pid] = process
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() - started > args.overall_timeout:
                    raise subprocess.TimeoutExpired(command, args.overall_timeout)
                time.sleep(.5)
        result = json.loads((output / 'worker_report.json').read_text())
        result['worker_status'] = result['status']
        if child.returncode != 0:
            result['status'] = 'process_abort' if child.returncode < 0 else 'process_error'
        elif result['status'] == 'checks_passed':
            result['status'] = 'pass'
    except BaseException as error:
        result.update(status='supervisor_error', error=f'{type(error).__name__}: {error}')
    finally:
        alive = live_owned_processes(tracked.values())
        if alive:
            # Only the owned group/descendants are stopped; acceptance stays fail.
            sys.path.insert(0, str(REPO / 'verl/scripts/train/action_head_bootstrap'))
            sys.path.insert(0, str(REPO / 'vln_demo/src'))
            from collection_runtime import stop_services
            stop_services([process.pid for process in alive], [],
                          output / 'supervisor_shutdown.json', args.gpus)
            if child is not None:
                child.wait(timeout=10)
            if result.get('status') in ('pass', 'inputs_validated'):
                result.update(status='cleanup_error', error='owned processes survived worker exit')
        owner = json.loads((runtime / 'owner.json').read_text())
        if owner != {'token': token, 'output': str(output)}:
            raise ValueError('refusing runtime cleanup after ownership changed')
        shutil.rmtree(runtime)
        result.update(child_returncode=child.returncode if child else None,
                      process_exit_verified=child is not None and child.returncode is not None,
                      runtime_root=str(runtime), runtime_cleaned=not runtime.exists())
        (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'report': str(output / 'report.json')}, ensure_ascii=False))
    return 0 if result['status'] in ('pass', 'inputs_validated') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--target', type=Path, default=BASE / 'full_r3/global_step_000001/qwen')
    parser.add_argument('--sleep-gate', '--sleep', action='store_true', dest='sleep_gate')
    parser.add_argument('--validate-inputs', action='store_true')
    parser.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    parser.add_argument('--port', type=int, default=28898)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--startup-timeout', type=int, default=1800)
    parser.add_argument('--overall-timeout', type=float, default=7200)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.70)
    parser.add_argument('--runtime-parent', type=Path, default=Path('/tmp/action_head_rl'))
    parser.add_argument('--sleep-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--runtime', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--owner-token', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sleep_gate:
        devices = [value.strip() for value in args.gpus.split(',') if value.strip()]
        if not args.validate_inputs and (len(devices) != 2 or len(set(devices)) != 2):
            parser.error('sleep gate requires exactly two distinct GPUs')
        args.gpus = ','.join(devices)
        if not 0 < args.gpu_memory_utilization < 1:
            parser.error('gpu-memory-utilization must be in (0,1)')
        return sleep_gate_main(args)
    if args.validate_inputs or args.sleep_worker:
        parser.error('sleep diagnostic flags require --sleep-gate')
    return legacy_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
