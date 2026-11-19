#!/usr/bin/env python3
"""Inspect the historical external handoff protocol; execution is disabled.

The real step22 attempt lost its Ray actors while their owner was SIGSTOP'ed.
Use the trainer's --stop-after-step or normal opt_steps endpoint instead.
Historical helpers remain for reading evidence and testing failure cleanup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ARTIFACTS = Path('/data/zerovln/artifacts')
REPO = Path(__file__).resolve().parents[2]


def driver_identity(pid: int, expected_ticks: str, run: Path, *,
                    proc_root=Path('/proc'), allow_stopped=False) -> dict:
    root = proc_root / str(pid)
    fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    ticks, state = fields[19], fields[0]
    argv = (root / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    if ticks != str(expected_ticks):
        raise RuntimeError('driver PID was reused or start_ticks do not match')
    if state in ('Z', 'X') or (state in ('T', 't') and not allow_stopped):
        raise RuntimeError(f'driver is not a live, independently running process: state={state}')
    module = next((argv[i + 1] for i, value in enumerate(argv[:-1]) if value == '-m'), None)
    outputs = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == '--output-dir']
    outputs += [value.split('=', 1)[1] for value in argv if value.startswith('--output-dir=')]
    if module != 'action_head_rl.trainer' or len(outputs) != 1 or Path(outputs[0]).resolve() != run:
        raise RuntimeError('PID is not the expected trainer with this exact output directory')
    return {'pid': pid, 'start_ticks': ticks, 'state': state, 'run_dir': str(run)}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Only a not-yet-completed final append may be ignored.
            if index != len(lines) - 1 or data.endswith(b'\n'):
                raise
    return rows


def validate_contract(contract: dict, *, world: int, target_step: int) -> int:
    if world != 12 or contract.get('learner_world_size') != world:
        raise ValueError('this handoff is restricted to the verified 12-rank local topology')
    if (contract.get('train_scope') != 'full'
            or contract.get('learner_parameter_dtype') != 'fp32'
            or contract.get('episodes_per_step') != 32
            or contract.get('reward', {}).get('normalization') != 'same_case_trajectory_outcome_grpo'):
        raise ValueError('requires full FP32 episode-GRPO with 32 rollouts per step')
    if 'tool_credit' in contract or contract.get('tool_credit_mode', 'episode_grpo') != 'episode_grpo':
        raise ValueError('tool-credit runs must save their critic through the trainer checkpoint path')
    if contract.get('overlap_rollout_learner', False) or contract.get('async_rollout', False):
        raise ValueError('the rollout boundary guard cannot protect overlapping/asynchronous learner updates')
    start = int(contract.get('resume', {}).get('start_step', 0))
    if not start < target_step:
        raise ValueError('target step must follow the source run resume step')
    return start


def completed_target_event(events: list[dict], *, start: int, target: int) -> dict | None:
    if not events:
        return None
    steps = [int(event['step']) for event in events]
    if steps[-1] > target:
        raise RuntimeError('missed target: a later optimizer update is already committed')
    if steps != list(range(start + 1, steps[-1] + 1)):
        raise ValueError('source metrics are not contiguous from resume.start_step + 1')
    if steps[-1] < target:
        return None
    event = events[-1]
    update = event.get('updates', {})
    if not math.isfinite(float(update.get('grad_norm', float('nan')))) or update.get('optimizer_steps') != 1:
        raise ValueError('target metric does not confirm one finite optimizer update')
    return event


def validate_boundary(events: list[dict], rows: list[dict], *, start: int,
                      target: int, episodes: int = 32) -> tuple[dict, list[dict]] | None:
    event = completed_target_event(events, start=start, target=target)
    if event is None:
        return None
    future = [row for row in rows if int(row['step']) > target]
    if len(future) >= episodes or any(int(row['step']) != target + 1 for row in future):
        raise RuntimeError('next rollout batch is complete; the next learner update may have begun')
    completed = [row for row in rows if start < int(row['step']) <= target]
    for step in range(start + 1, target + 1):
        batch = [row for row in completed if int(row['step']) == step]
        if len(batch) != episodes or not all(row.get('result', {}).get('evaluation_valid') is True for row in batch):
            raise ValueError(f'step {step} lacks exactly {episodes} valid completed rollouts')
    return event, completed


def learner_counters(worker):
    """Executed on each live Ray learner; reads state without a model forward."""
    states = worker.engine.optimizer.state.values()
    return {'rank': int(worker.rank),
            'optimizer_steps': sorted({float(state['step'].item())
                                       for state in states if 'step' in state}),
            'optimizer_moment_dtypes': sorted({str(value.dtype) for state in states
                for key, value in state.items() if key in ('exp_avg', 'exp_avg_sq')}),
            'scheduler_last_epoch': int(worker.engine.lr_scheduler.last_epoch)}


def validate_counters(rows, *, target, world):
    if len(rows) != world or sorted(row['rank'] for row in rows) != list(range(world)):
        raise ValueError('not all expected unique learner ranks responded')
    if not all(row['optimizer_steps'] == [float(target)]
               and row['scheduler_last_epoch'] == target
               and row['optimizer_moment_dtypes'] == ['torch.float32'] for row in rows):
        raise ValueError(f'live Adam/scheduler/FP32 state mismatch: {rows}')


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def build_plan(args, *, inspect_driver=True) -> dict:
    run, dest = args.run_dir.resolve(), args.dest.resolve()
    if (not run.is_relative_to(ARTIFACTS) or run == ARTIFACTS
            or not dest.is_relative_to(ARTIFACTS) or dest == ARTIFACTS):
        raise ValueError('source and destination must be experiment directories on the data disk')
    if dest == run or dest.is_relative_to(run) or run.is_relative_to(dest):
        raise ValueError('handoff destination must be separate from the source run')
    if dest.exists():
        raise FileExistsError(f'refusing to modify existing handoff destination: {dest}')
    if args.driver_pid <= 0 or not str(args.driver_start_ticks).isdigit():
        raise ValueError('a positive driver PID and explicit start_ticks are required')
    if not args.actor_prefix or not args.actor_prefix.endswith(':') or not args.ray_namespace:
        raise ValueError('explicit Ray namespace and actor prefix ending in colon are required')
    if not args.ray_address or args.ray_address in ('auto', 'local'):
        raise ValueError('an explicit already-running Ray address is required')
    if not math.isfinite(args.deadline_seconds) or args.deadline_seconds <= 0:
        raise ValueError('deadline-seconds must be finite and positive')
    contract = json.loads((run / 'contract.json').read_text())
    start = validate_contract(contract, world=args.world, target_step=args.target_step)
    adopt_paused = bool(getattr(args, 'adopt_paused_driver', False))
    identity = (driver_identity(args.driver_pid, args.driver_start_ticks, run,
                                allow_stopped=adopt_paused)
                if inspect_driver else None)
    if adopt_paused and identity is not None and identity['state'] not in ('T', 't'):
        raise RuntimeError('adopt-paused-driver requires the verified driver to be stopped')
    return {'schema': 'v6_local_exact_handoff_plan_v1', 'default_action': 'read_only_plan',
            'source_run': str(run), 'destination': str(dest), 'source_start_step': start,
            'target_step': args.target_step, 'expected_new_steps': list(range(start + 1, args.target_step + 1)),
            'expected_rollouts': (args.target_step - start) * 32,
            'source_contract_sha256': hashlib.sha256((run / 'contract.json').read_bytes()).hexdigest(),
            'driver': identity or {'pid': args.driver_pid, 'start_ticks': str(args.driver_start_ticks)},
            'ray_address': args.ray_address, 'ray_namespace': args.ray_namespace,
            'actor_names': [args.actor_prefix + str(rank) for rank in range(args.world)],
            'learner_world_size': args.world, 'deadline_seconds': args.deadline_seconds,
            'adopt_paused_driver': adopt_paused,
            'source_logs_writable': False, 'checkpoint_kind': 'FP32 actor + optimizer + scheduler + RNG',
            'success_action': 'SIGINT verified old driver, then SIGCONT in finally',
            'failure_action': 'do not claim saved; SIGCONT verified old driver in finally'}


def verify_saved_checkpoint(checkpoint: Path, *, target: int, world: int) -> dict:
    # The CLI also works without inheriting a training launcher's PYTHONPATH.
    sys.path[:0] = [str(REPO / 'verl'), '/workspace/verl']
    import torch
    from safetensors import safe_open
    from action_head_rl.checkpoint import resolve_resume_checkpoint
    state = checkpoint / 'trainer_state'
    resolved = resolve_resume_checkpoint(mode='resume_path', output_dir=checkpoint.parent,
        resume_from_path=str(checkpoint), train_scope='full', learner_world_size=world,
        allow_weights_only=False)
    if not resolved.exact:
        raise ValueError('checkpoint lacks full optimizer/scheduler/RNG state')
    report = json.loads((checkpoint / 'report.json').read_text())
    if report.get('publication_kind') != 'checkpoint' or report.get('qwen_export_dtype') != 'preserve':
        raise ValueError('handoff published a lossy inference copy')
    extra_audit = []
    for rank in range(world):
        optim = state / f'optim_world_size_{world}_rank_{rank}.pt'
        extra_file = state / f'extra_state_world_size_{world}_rank_{rank}.pt'
        if optim.stat().st_size <= 0 or extra_file.stat().st_size <= 0:
            raise ValueError('empty distributed state shard')
        extra = torch.load(extra_file, map_location='cpu', weights_only=False)
        if extra.get('lr_scheduler', {}).get('last_epoch') != target:
            raise ValueError(f'saved scheduler rank {rank} does not match target')
        if not {'cpu', 'cuda', 'numpy', 'random'} <= set(extra.get('rng', {})):
            raise ValueError(f'saved RNG rank {rank} is incomplete')
        extra_audit.append({'rank': rank, 'scheduler_last_epoch': target,
                            'optimizer_bytes': optim.stat().st_size, 'rng_keys': sorted(extra['rng'])})
    index = json.loads((checkpoint / 'qwen/model.safetensors.index.json').read_text())
    tensor_count = 0
    for shard in sorted(set(index['weight_map'].values())):
        with safe_open(checkpoint / 'qwen' / shard, framework='pt', device='cpu') as source:
            for key in source.keys():
                dtype = source.get_slice(key).get_dtype()
                if dtype.startswith(('F', 'BF')) and dtype != 'F32':
                    raise ValueError(f'Qwen checkpoint contains a non-FP32 floating tensor: {key} {dtype}')
                tensor_count += 1
    return {'ranks': extra_audit, 'qwen_tensors_checked': tensor_count}


def execute(plan: dict):
    import ray
    run, dest = Path(plan['source_run']), Path(plan['destination'])
    target, start = plan['target_step'], plan['source_start_step']
    pid, ticks = plan['driver']['pid'], plan['driver']['start_ticks']
    world = plan['learner_world_size']
    deadline = time.monotonic() + plan['deadline_seconds']
    print(json.dumps({'event': 'waiting_for_completed_step', 'target_step': target,
                      'source_start_step': start, 'driver_pid': pid,
                      'deadline_seconds': plan['deadline_seconds']}), flush=True)
    def identity(allow_stopped=False):
        return driver_identity(pid, ticks, run, allow_stopped=allow_stopped)
    def boundary():
        events, rows = read_jsonl(run / 'metrics.jsonl'), read_jsonl(run / 'rollouts.jsonl')
        result = validate_boundary(events, rows, start=start, target=target)
        return events, rows, result
    adopt_paused = plan.get('adopt_paused_driver', False)
    if not adopt_paused:
        previous_metrics_signature = None
        while True:
            identity()
            metrics_path = run / 'metrics.jsonl'
            try:
                metadata = metrics_path.stat()
                signature = (metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
            except FileNotFoundError:
                signature = None
            if signature != previous_metrics_signature:
                previous_metrics_signature = signature
                events = read_jsonl(metrics_path)
                if completed_target_event(events, start=start, target=target) is not None:
                    # Read the larger rollout log only after a completed target
                    # metric appears. Re-read it again after the actual STOP.
                    events, rows, result = boundary()
                    if result is not None:
                        break
            if time.monotonic() > deadline:
                raise TimeoutError('target optimizer step did not finish before the handoff deadline')
            time.sleep(.25)
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('safe handoff requires Linux pidfd signals to prevent PID reuse races')
    paused = connected = False
    pidfd = os.pidfd_open(pid)
    def send(signum):
        identity(allow_stopped=paused)
        signal.pidfd_send_signal(pidfd, signum)
    def all_ranks(refs):
        # All calls must be submitted before waiting: publication and checkpoint
        # save contain distributed collectives and cannot run rank by rank.
        ray.wait(refs, num_returns=len(refs))
        results, failures = [], []
        for ref in refs:
            try:
                results.append(ray.get(ref))
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise RuntimeError(f'{len(failures)} learner RPCs failed; no handoff committed') from failures[0]
        return results
    try:
        if adopt_paused:
            if identity(allow_stopped=True)['state'] not in ('T', 't'):
                raise RuntimeError('adopt-paused-driver requires the verified driver to be stopped')
            # Ownership starts only after pidfd + PID/ticks/output + T/t checks.
            # Every later failure goes through finally's SIGCONT, without an
            # intervening resume/re-pause window at the optimizer boundary.
            paused = True
        else:
            send(signal.SIGSTOP)
            paused = True
        stop_deadline = time.monotonic() + 20
        while identity(allow_stopped=True)['state'] not in ('T', 't'):
            if time.monotonic() > stop_deadline:
                raise TimeoutError('driver did not enter a confirmed stopped state')
            time.sleep(.01)
        # Re-read after the actual stop; a batch may have completed between the
        # first observation and the signal. Never touch learner state in that case.
        events, rows, result = boundary()
        if result is None:
            raise RuntimeError('completed boundary disappeared after pausing')
        event, completed = result
        if hashlib.sha256((run / 'contract.json').read_bytes()).hexdigest() != plan['source_contract_sha256']:
            raise RuntimeError('source contract changed after planning')
        ray.init(address=plan['ray_address'], namespace=plan['ray_namespace'], logging_level='ERROR')
        connected = True
        actors = [ray.get_actor(name, namespace=plan['ray_namespace']) for name in plan['actor_names']]
        counters = all_ranks([actor.execute_with_func_generator.remote(func=learner_counters) for actor in actors])
        validate_counters(counters, target=target, world=world)
        dest.mkdir(parents=True, exist_ok=False)
        (dest / 'contract.json').write_bytes((run / 'contract.json').read_bytes())
        atomic_json(dest / 'handoff_plan.json', plan)
        atomic_json(dest / 'live_optimizer_counters.json', counters)
        publications = [row for row in all_ranks([actor.publish_policy.remote(
            output_dir=str(dest), step=target, export_qwen=True,
            transport_only=False, qwen_export_dtype='preserve')
            for actor in actors]) if row is not None]
        if len(publications) != 1:
            raise ValueError('expected exactly one rank-zero publication')
        checkpoint = dest / f'global_step_{target:06d}'
        state = checkpoint / 'trainer_state'
        all_ranks([actor.save_checkpoint.remote(local_path=str(state), global_step=target,
                    max_ckpt_to_keep=None) for actor in actors])
        after = all_ranks([actor.execute_with_func_generator.remote(func=learner_counters) for actor in actors])
        validate_counters(after, target=target, world=world)
        audit = verify_saved_checkpoint(checkpoint, target=target, world=world)
        report = json.loads((checkpoint / 'report.json').read_text())
        report['trainer_state'] = {'format': 'verl_fsdp_optimizer_extra', 'path': str(state), 'world_size': world}
        report['handoff'] = {'source_run': str(run), 'source_resume_step': start,
                             'source_step': target, 'driver_identity': plan['driver']}
        atomic_json(checkpoint / 'report.json', report)
        atomic_json(dest / 'saved_state_audit.json', audit)
        derived = {**event, 'checkpoint': publications[0], 'checkpoint_committed': True,
                   'checkpoint_origin': 'live_actor_rpc_handoff',
                   'original_event_source': str(run / 'metrics.jsonl')}
        (dest / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in events[:-1] + [derived]))
        (dest / 'rollouts.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in completed))
        from action_head_rl.checkpoint import write_latest_checkpoint
        write_latest_checkpoint(dest, target)
        atomic_json(dest / 'handoff_result.json', {'state': 'checkpoint_saved', 'source': str(run),
            'checkpoint': str(checkpoint), 'step': target, 'driver_identity': plan['driver'],
            'saved_at': time.time(), 'grad_norm': event['updates']['grad_norm']})
        send(signal.SIGINT)
    except BaseException as exc:
        if dest.exists():
            failed = {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                      'time': time.time(), 'source_run': str(run), 'target_step': target}
            atomic_json(dest / 'handoff_failure.json', failed)
            # A later failure (for example signaling the old driver) must not
            # leave a success state for the candidate's handoff monitor.
            atomic_json(dest / 'handoff_result.json', failed)
        raise
    finally:
        if paused:
            try:
                send(signal.SIGCONT)
            except (FileNotFoundError, ProcessLookupError):
                pass
        os.close(pidfd)
        if connected:
            ray.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--target-step', type=int, default=22)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--driver-pid', type=int, required=True)
    parser.add_argument('--driver-start-ticks', required=True)
    parser.add_argument('--ray-address', required=True)
    parser.add_argument('--ray-namespace', required=True)
    parser.add_argument('--actor-prefix', required=True)
    parser.add_argument('--world', type=int, default=12)
    parser.add_argument('--deadline-seconds', type=float, default=3600)
    parser.add_argument('--adopt-paused-driver', action='store_true',
                        help='adopt an already T/t driver with exact PID/ticks/output; always resume it in finally')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.execute:
        parser.error('external SIGSTOP handoff is disabled after a real Ray owner RPC failure; '
                     'configure trainer --stop-after-step or opt_steps before starting')
    plan = build_plan(args)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
