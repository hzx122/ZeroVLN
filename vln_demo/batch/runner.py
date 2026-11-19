"""Isolated episode runner. Remote render/model services; local per-episode state."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from urllib.request import Request, ProxyHandler, build_opener

from workspace import REPO, new_workspace, archive, sha256
from recovery import allocate_attempt, prepare_run, import_results, load_results
from model_gateway import model_gateway

NAV = Path('/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation')
DEFAULT_ARTIFACTS = Path('/opt/external/user-d5121be6/SpatialOnboarding/artifacts')
NODE = '/opt/external/user-d5121be6/tools/node/bin/node'


def request(url, token, data=None):
    headers = {'X-VLN-Batch-Token': token}
    if data is not None:
        data = json.dumps(data).encode()
        headers['Content-Type'] = 'application/json'
    with build_opener(ProxyHandler({})).open(Request(url, headers=headers, data=data), timeout=15) as response:
        return json.load(response)


def live_group(group):
    for file in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = file.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] != 'Z':
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def stop(process):
    if process is None:
        return
    # Reaping the group leader does not mean its writers have exited.
    for sig, seconds in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + seconds
        while live_group(process.pid) and time.monotonic() < deadline:
            process.poll()
            time.sleep(.1)
        if not live_group(process.pid):
            break
    process.wait(timeout=5)
    if live_group(process.pid):
        raise RuntimeError('episode process group still writing after shutdown')


def manifest_rows(dataset, split, role, languages, limit):
    path = NAV / 'data/datasets' / dataset / split / (f'{split}_{role}.json.gz' if dataset == 'rxr' else f'{split}.json.gz')
    with gzip.open(path, 'rt') as stream:
        episodes = json.load(stream)['episodes']
    result = []
    for ep in episodes:
        language = ep['instruction'].get('language', 'en')
        if languages != '*' and language not in languages.split(','):
            continue
        row = dict(dataset=dataset, split=split, role=role if dataset == 'rxr' else 'guide', language=language,
                   scene=ep['scene_id'].split('/')[-2], episode_id=str(ep['episode_id']), seed=0,
                   instruction=ep['instruction']['instruction_text'], source_file=str(path))
        row['logical_id'] = sha256(json.dumps(row, sort_keys=True).encode())[:20]
        result.append(row)
        if limit and len(result) >= limit:
            break
    return result


def finish_agent_exit(workspace, returncode, url, token):
    try:
        report = json.loads((workspace / '.runtime/agent-result.json').read_text())
    except (OSError, ValueError):
        return 'agent_exit_without_report'
    if returncode != 0 or report.get('termination') not in ('normal', 'tool_limit'):
        return 'agent_' + report.get('termination', 'error')
    request(url + '/batch-finish', token, {})
    return 'agent_tool_limit' if report['termination'] == 'tool_limit' else 'agent_completed'


def rolling_results(rows, workers, execute):
    """Refill free slots across the entire manifest without a batch barrier."""
    queued = iter(rows)
    submitted = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()

        def refill():
            nonlocal submitted
            while len(pending) < workers:
                row = next(queued, None)
                if row is None:
                    break
                pending.add(pool.submit(execute, row))
                submitted += 1

        refill()
        yield [], len(pending), len(rows) - submitted
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            # Submit replacements before reporting/CPFS writes. A slow episode
            # never holds up the other slots, even across dataset boundaries.
            refill()
            yield [future.result() for future in done], len(pending), len(rows) - submitted


def episode_expired(started, timeout, now=None):
    """Zero disables the episode wall clock; transport/startup checks are separate."""
    return timeout > 0 and (time.monotonic() if now is None else now) - started >= timeout


def episode(row, args, run, destination):
    # Exclusive attempt path and fresh OS processes; no reset/reuse between rows.
    from agent_models import model_choice
    selected = model_choice(args.agent_profile)
    local = Path(tempfile.mkdtemp(prefix='vln-episode-', dir=args.local_root))
    workspace = local / 'workspace'
    template = new_workspace(run, workspace)
    controller_output = local / 'controller'
    controller_output.mkdir()
    token = uuid.uuid4().hex
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    from service_env import service_env
    env = service_env(os.environ, args.model_url, args.controller_model_url, args.render_url)
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT', 'LOCAL_WORLD_SIZE'):
        env.pop(key, None)
    env.update(VLN_BATCH_TOKEN=token, VLN_BATCH_OUTPUT=str(controller_output),
        VLN_BATCH_MODEL_URL=args.model_url, VLN_BATCH_CONTROLLER_MODEL_URL=args.controller_model_url,
        VLN_BATCH_MODEL_PROVIDER=selected['provider'], VLN_BATCH_MODEL_ID=selected['model_id'],
        VLN_HABITAT_RENDER_URL=args.render_url,
        VLN_BATCH_SEED=str(row['seed']), VLN_RENDER_MODE=args.render_mode,
        PI_VLN_URL=url, VLN_BATCH_INSTRUCTION=row['instruction'],
        VLN_BATCH_MAX_TOOL_CALLS=str(args.max_tool_calls),
        DEMO_CONFIG=str(REPO / 'vln_demo/src/vln_demo.yaml'),
        DEMO_DEFAULT_MODEL='qwen35b', DEMO_DEFAULT_DRIVER='react', DEMO_DEFAULT_MODE='think',
        DEMO_MAX_TOOL_CALLS='10', DEMO_BACKEND='rxr' if row['dataset']=='rxr' else 'habitat',
        RXR_LANGUAGES='*', RXR_EPISODES_PER_SCENE='0', RXR_ROLES=row['role'],
        DEMO_SYSPROMPT_FILE=str(local / 'unused-prompt'), PYTHONDONTWRITEBYTECODE='1',
        TMPDIR=str(local / 'tmp'), XDG_CACHE_HOME=str(local / 'cache'),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    if row['dataset'] == 'objectnav':
        from objectnav_batch import controller_environment
        env.update(controller_environment(row, controller_output, args.render_url))
    for key in ('HF_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH',
                'PYTORCH_KERNEL_CACHE_PATH', 'TORCH_EXTENSIONS_DIR', 'VLLM_CACHE_ROOT'):
        env[key] = str(local / 'cache' / key.lower())
    for key in ('TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TRITON_CACHE_DIR',
                'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'PYTORCH_KERNEL_CACHE_PATH',
                'TORCH_EXTENSIONS_DIR', 'VLLM_CACHE_ROOT'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    controller = agent = None
    result = dict(row, state='failed', reason='initialization', template=template,
                  navigation_success=False, model=selected['model_id'], model_provider=selected['provider'],
                  model_url=args.upstream_model_url,
                  render_url=args.render_url, render_mode=args.render_mode, controller_model='Qwen3.6-35B',
                  controller_model_url=args.controller_model_url,
                  isolation='private-user-mount-pid-namespaces-readonly-host',
                  habitat_config=env['DEMO_CONFIG'],
                  habitat_config_sha256=sha256(Path(env['DEMO_CONFIG']).read_bytes()),
                  protocol='external_reads_and_network_allowed')
    started = time.monotonic()
    print(json.dumps({'episode': row['logical_id'], 'phase': 'controller_start'}), flush=True)
    try:
        command = ['bash', str(REPO / 'vln_demo/src/run_console.sh'), str(port),
                   '--split', row['split'], '--scene', ('rxr-' if row['dataset']=='rxr' else '') + row['scene'],
                   '--episode_id', row['episode_id'], '--out', str(controller_output)]
        with (local / 'controller.log').open('w') as log:
            controller = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                          env=env, start_new_session=True)
        deadline = time.monotonic() + 900
        while True:
            if controller.poll() is not None:
                raise RuntimeError(f'private controller exited {controller.returncode}')
            try:
                evaluation = request(url + '/batch-result', token)
            except (OSError, ValueError):
                evaluation = {}
            if evaluation.get('state') == 'failed':
                result.update(evaluation)
                raise RuntimeError('private evaluator rejected episode: ' + str(evaluation.get('reason')))
            try:
                state = request(url + '/status', token)
                frame = request(url + '/frame-snapshot', token)
                if frame.get('ok') and state.get('status') == 'idle':
                    if str(frame['source']['episode_id']) != row['episode_id'] or not frame['source']['scene'].endswith(row['scene']):
                        raise RuntimeError('private controller loaded the wrong episode')
                    break
            except (OSError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('private controller readiness timed out')
            time.sleep(1)
        actual = request(url + '/batch-config', token)
        if row['dataset'] == 'rxr':
            geometry = actual.get('geometry') or {}
            if any(abs(geometry.get(k, -1)-v) > 1e-5 for k, v in
                   dict(agent_height=.88, agent_radius=.18, navmesh_height=.88, navmesh_radius=.18).items()):
                raise RuntimeError('RxR collision body/navmesh differs from official dataset dimensions')
        expected = dict(model='qwen35b', driver='react', mode='think', maxtools=10, temp=None, nact=4)
        if any(actual['config'].get(key) != value for key, value in expected.items()):
            raise RuntimeError('private controller differs from audited web configuration')
        endpoint, model_id = actual['model_endpoint']
        if endpoint.rstrip('/').removesuffix('/v1') != args.controller_model_url.rstrip('/').removesuffix('/v1') or model_id != 'Qwen3.6-35B':
            raise RuntimeError('private controller model routing mismatch')
        (controller_output / 'runtime-config.json').write_text(json.dumps(actual, ensure_ascii=False, indent=2))
        command = [sys.executable, '-B', str(REPO / 'vln_demo/batch/sandbox.py'),
                   '--workspace', str(workspace), '--', NODE, str(REPO / 'vln_demo/batch/agent.mjs')]
        with (local / 'agent-events.jsonl').open('w') as log:
            agent = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        print(json.dumps({'episode': row['logical_id'], 'phase': 'agent_started', 'pid': agent.pid}), flush=True)
        agent_started = time.monotonic()
        while agent.poll() is None:
            evaluation = request(url + '/batch-result', token)
            if evaluation.get('state') in ('finished', 'failed'):
                result.update(evaluation)
                break
            if episode_expired(agent_started, args.timeout):
                result['reason'] = 'timeout'
                break
            time.sleep(1)
        # A normal pi-agent response ends the episode, just as in the interactive
        # workflow. The evaluator translates that into Habitat STOP externally.
        if result['state'] != 'finished' and result['reason'] != 'timeout' and agent.poll() is not None:
            result['reason'] = finish_agent_exit(workspace, agent.returncode, url, token)
        # STOP may be pending at agent exit; wait briefly for the main sim thread.
        for _ in range(180):
            evaluation = request(url + '/batch-result', token)
            if evaluation.get('state') in ('finished', 'failed'):
                result.update(evaluation)
                break
            if evaluation.get('state') != 'stop_requested':
                break
            time.sleep(1)
        if result['state'] != 'finished' and result['reason'] in ('agent_completed', 'agent_tool_limit'):
            result['reason'] = 'evaluation_not_finished'
        result['agent_returncode'] = agent.poll()
        if result['state'] == 'finished':
            try:
                agent.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            stop(agent)
        finally:
            stop(controller)
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    agent_report = workspace / '.runtime/agent-result.json'
    if agent_report.exists():
        report = json.loads(agent_report.read_text())
        result.update(agent_termination=report.get('termination'), tool_calls=report.get('toolCalls'),
                      agent_turns=report.get('turns'), max_tool_calls=args.max_tool_calls)
    sessions = list((workspace / '.runtime/sessions').glob('*.jsonl'))
    if len(sessions) == 1 and not sessions[0].is_symlink():
        shutil.copyfile(sessions[0], local / 'session.jsonl')
        result['session_file'] = 'session.jsonl'
    # Runtime caches are disposable and must remain on local disk, not CPFS.
    for cache in (local / 'tmp', local / 'cache', workspace / '.runtime/tmp', workspace / '.runtime/cache'):
        if cache.is_symlink():
            cache.unlink()
        elif cache.is_dir():
            shutil.rmtree(cache)
    final_agents = workspace / 'AGENTS.md'
    agents_template = REPO / 'third_party/pi-agent/.pi/workspaces/SpatialOnboarding/AGENTS.md'
    result['final_agents_sha256'] = (sha256(final_agents.read_bytes())
        if final_agents.is_file() and (not final_agents.is_symlink()
            or final_agents.resolve() == agents_template.resolve()) else None)
    if final_agents.is_symlink():
        result['final_agents_symlink'] = os.readlink(final_agents)
    for attempt in range(3):
        try:
            archive(local, destination, result, agents_template=agents_template,
                    project_config=REPO / 'vln_demo/config/pi/SpatialOnboarding')
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    shutil.rmtree(local)
    print(json.dumps({'episode': row['logical_id'], 'state': result['state'], 'reason': result['reason'],
                      'navigation_success': result['navigation_success'], 'archive': str(destination)}), flush=True)
    result['archive'] = str(destination)
    return result


def safe_episode(row, args, run):
    # Infrastructure errors are local to one episode. Retry with a fresh workspace;
    # never count a partial archive as a valid result or stop the other 63 slots.
    last = None
    for retry in range(2):
        destination = allocate_attempt(run, row)
        try:
            return episode(row, args, run, destination)
        except Exception as exc:
            last = dict(row, state='failed', reason='infrastructure_error', navigation_success=False,
                        error=f'{type(exc).__name__}: {exc}', archive=None)
            (destination.parent / ('.' + destination.name + '.error.json')).write_text(json.dumps(last)+'\n')
            print('VLN_EPISODE_ERROR ' + json.dumps(last), flush=True)
    return last


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--datasets', default='r2r,rxr')
    parser.add_argument('--split', default='val_unseen')
    parser.add_argument('--role', choices=['guide', 'follower'], default='guide')
    parser.add_argument('--languages', default='*')
    parser.add_argument('--limit-per-dataset', type=int, default=0)
    parser.add_argument('--render-mode', choices=('display', 'batch'), default='batch')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--timeout', type=int, default=0, help='optional episode wall-clock seconds; 0 disables')
    parser.add_argument('--max-tool-calls', type=int, default=100)
    parser.add_argument('--model-url', required=True)
    parser.add_argument('--agent-profile', default='next')
    parser.add_argument('--controller-model-url', required=True)
    parser.add_argument('--render-url', required=True)
    parser.add_argument('--local-root', default='/tmp/vln-batch')
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 0 or args.max_tool_calls < 1:
        parser.error('workers must be positive; timeout must be >= 0 (0 disables)')
    if args.workers > 32:
        parser.error('workers must not exceed 32')
    run = args.run_dir.resolve()
    if not run.is_relative_to(DEFAULT_ARTIFACTS):
        parser.error('run-dir must be under SpatialOnboarding/artifacts')
    Path(args.local_root).mkdir(parents=True, exist_ok=True)
    if args.manifest:
        if args.datasets == 'objectnav':
            from objectnav_batch import load_manifest
            rows = load_manifest(args.manifest)
        else:
            rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    else:
        rows = [row for dataset in args.datasets.split(',') for row in manifest_rows(
            dataset, args.split, args.role, args.languages, args.limit_per_dataset)]
    if not rows or len({row['logical_id'] for row in rows}) != len(rows):
        raise ValueError('manifest empty or contains duplicate episode identities')
    for row in rows:
        for key in ('dataset', 'split', 'role', 'language', 'logical_id'):
            if not row[key] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in row[key]):
                raise ValueError('unsafe manifest component: ' + key)
    configuration = vars(args).copy()
    configuration.update(image=os.environ.get('DOCKER_IMAGE_URL'),
                         git_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip())
    lock, template = prepare_run(run, rows, configuration)
    import_results(args.resume_from, run, rows)
    saved = load_results(run, rows)
    results = list(saved.values())
    remaining = [row for row in rows if row['logical_id'] not in saved]
    started = time.monotonic()
    remote_model_url = args.model_url
    args.upstream_model_url = remote_model_url
    from contextlib import nullcontext
    from agent_models import model_choice
    selected = model_choice(args.agent_profile)
    gate = nullcontext('') if selected.get('gateway') else model_gateway(remote_model_url, args.workers, run)
    with gate as gateway:
        args.model_url = gateway
        for completed, in_flight, queued in rolling_results(
                remaining, args.workers, lambda row: safe_episode(row, args, run)):
            results.extend(completed)
            progress = dict(episodes=len(rows), archived=sum(bool(r.get('archive')) for r in results),
                            processed=len(results), archive_errors=sum(not r.get('archive') for r in results),
                            pending=in_flight + queued, workers=args.workers, in_flight=in_flight, queued=queued,
                            successes=sum(r['navigation_success'] for r in results),
                            elapsed_seconds=round(time.monotonic()-started, 1))
            partial = run / '.progress.json.tmp'
            partial.write_text(json.dumps(progress)+'\n')
            partial.replace(run / 'progress.json')
            print('VLN_BATCH_PROGRESS ' + json.dumps(progress), flush=True)
    summary = dict(episodes=len(results), completed=sum(r['state']=='finished' for r in results),
                   successes=sum(r['navigation_success'] for r in results), results=results)
    (run / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (run / 'exports').mkdir(exist_ok=True)
    (run / 'exports/successful-episodes.jsonl').write_text(''.join(
        json.dumps(r,ensure_ascii=False)+'\n' for r in results if r['navigation_success']))
    if all(r.get('archive') for r in results):
        (run / '_SUCCESS.json').write_text(json.dumps({'archive_complete': True, 'episodes': len(results)})+'\n')
    lock.close()
    return 0 if all(r.get('archive') for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
