"""Matched, update-free throughput measurements using the production rollout path.

Uses the trainer's case/seed contract, the same stateful actors, R3 provider,
FastPolicy and bounded scheduler. No learner or optimizer is created. Each
variant gets a service warmup before timing and writes per-episode evidence.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from action_head_rl.scheduling import iter_grouped_rollouts
from action_head_rl.vllm_service import VllmService


def run_benchmark(args):
    import numpy as np
    import ray
    from action_head_rl.rollout import ActionHeadRolloutWorker, FastPolicyWorker
    from action_head_rl.trainer import (
        _driver_memory_gb, _query_file, independent_episode_specs,
        load_smoke_cases, require_valid_rollouts,
    )

    variants = {
        "baseline16": dict(actors=16, bulk=False, pipeline=False, scheduler="wave", groups=2),
        "bulk16": dict(actors=16, bulk=True, pipeline=False, scheduler="wave", groups=2),
        "pipeline16": dict(actors=16, bulk=True, pipeline=True, scheduler="completion", groups=4),
        "pipeline64": dict(actors=64, bulk=True, pipeline=True, scheduler="completion", groups=8),
    }
    names = args.benchmark_variants.split(",")
    if any(name not in variants for name in names):
        raise ValueError(f"unknown benchmark variant; choose {list(variants)}")
    benches = [url.strip() for url in args.bench_urls.split(",") if url.strip()]
    max_actors = max(variants[name]["actors"] for name in names)
    if len(set(benches[:max_actors])) != max_actors:
        raise ValueError(f"benchmark needs {max_actors} distinct healthy gym URLs")
    if args.resume_mode != "disable" or args.fast_policy_workers < 1:
        raise ValueError("benchmark requires fresh weights and separate FastPolicy")
    cases = load_smoke_cases(args.smoke_cases)[:args.num_groups_per_step]
    if len(cases) != args.num_groups_per_step or args.group_size < 2:
        raise ValueError("insufficient cases or invalid GRPO group size")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    query = output / "current_query.pt"
    shutil.copy2(_query_file(args.query_checkpoint), query)
    contract = vars(args) | {"training_performed": False, "policy_version": 0,
                             "rollout_semantics": "independent_full_episode"}
    (output / "benchmark_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    runtime_vars = {key: value for key, value in os.environ.items() if key in {
        "PYTHONPATH", "LD_LIBRARY_PATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
        "PYTORCH_KERNEL_CACHE_PATH", "ACTION_HEAD_RAY_CACHE_ROOT",
    }}
    runtime_vars.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                        MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    ray.init(address=args.ray_address, runtime_env={"env_vars": runtime_vars})
    service = None
    actors, fast = [], []
    summaries = []

    def groups(specs, workers, config):
        return iter_grouped_rollouts(
            specs, workers, group_size=args.group_size,
            max_in_flight_groups=config["groups"], scheduler=config["scheduler"],
            submit=lambda actor, spec: actor.rollout.remote(spec[2], spec[3]),
            wait_one=lambda refs: ray.wait(list(refs), num_returns=1)[0][0],
            resolve=ray.get)

    try:
        for variant_index, name in enumerate(names):
            config = variants[name]
            variant = output / f"{variant_index:02d}_{name}"
            variant.mkdir()
            service = ray.remote(num_gpus=2)(VllmService).remote(
                repo=args.repo, model=args.slow_model, query_checkpoint=str(query),
                query_cache=str(variant / "query_cache"),
                log_file=str(variant / "vllm.log"), port=args.vllm_port,
                capture_full_routes=True, bulk_route_capture=config["bulk"],
                max_num_seqs=args.vllm_max_num_seqs,
                max_num_batched_tokens=args.vllm_max_num_batched_tokens)
            url = ray.get(service.start.remote())
            fast_cls = ray.remote(num_gpus=1, max_concurrency=256)(FastPolicyWorker)
            fast = [fast_cls.remote(
                checkpoint=args.head_checkpoint, backbone=args.fast_backbone,
                max_batch_size=args.fast_policy_max_batch_size,
                batch_wait_ms=args.fast_policy_batch_wait_ms,
                sde_steps=args.sde_steps, sde_noise=args.sde_noise,
                policy_version=0, pipeline_batches=config["pipeline"])
                for _ in range(args.fast_policy_workers)]
            actor_cls = ray.remote(num_cpus=1)(ActionHeadRolloutWorker)
            actors = [actor_cls.remote(
                repo=args.repo, checkpoint=args.head_checkpoint,
                backbone=args.fast_backbone, slow_model=args.slow_model,
                query_cache=str(variant / "query_cache"), bench_url=benches[index],
                vllm_url=url, train_scope=args.train_scope,
                max_turns=(args.max_tool_calls if args.train_scope == "full"
                           else args.max_turns),
                max_tool_calls=args.max_tool_calls,
                one_tool_call_per_turn=args.train_scope == "full",
                max_completion_tokens=(512 if args.train_scope == "full" else 0),
                fast_step_limit=(4 if args.train_scope == "full"
                                 else args.fm_chunks_per_episode),
                sde_steps=args.sde_steps, sde_noise=args.sde_noise,
                terminal_success_reward=args.terminal_success_reward,
                direction_match_reward=args.direction_match_reward,
                collision_penalty=args.collision_penalty,
                discount_gamma=args.discount_gamma,
                fast_policy_worker=fast[index % len(fast)], worker_id=index,
                policy_version=0) for index in range(config["actors"])]
            # Every actor is initialized before the timed run. Warmup uses the
            # same number of episodes in all variants and a separate seed range.
            ray.get([actor.reload_action_head.remote(args.head_checkpoint, 0)
                     for actor in actors])
            warm_specs = independent_episode_specs(cases[:2], step=0, group_size=args.group_size)
            for _, _, rows in groups(warm_specs, actors, config):
                require_valid_rollouts(rows, step=0)
            del rows
            before = ray.get([worker.stats.remote() for worker in fast])
            specs = independent_episode_specs(cases, step=1, group_size=args.group_size)
            started = time.perf_counter()
            count = chunks = successes = 0
            episode_times, slow_times = [], []
            with (variant / "episodes.jsonl").open("w") as handle:
                for group_index, group_specs, rows in groups(specs, actors, config):
                    require_valid_rollouts(rows, step=1)
                    for spec, row in zip(group_specs, rows):
                        digest = hashlib.sha256()
                        for chunk in row["chunks"]:
                            if int(chunk["policy_version"]) != 0:
                                raise RuntimeError("benchmark consumed a different policy version")
                            for key in ("states", "successors"):
                                digest.update(chunk[key].numpy().tobytes())
                        count += 1
                        chunks += len(row["chunks"])
                        successes += bool(row["result"].get("success"))
                        episode_times.append(row["episode_seconds"])
                        slow_times.append(row["slow_qwen_seconds"])
                        handle.write(json.dumps({
                            "group_index": group_index, "sample_index": spec[1],
                            "seed": row["seed"], "case": row["case"],
                            "reward": row["reward"], "result": row["result"],
                            "chunks": len(row["chunks"]), "sde_sha256": digest.hexdigest(),
                            "episode_seconds": row["episode_seconds"],
                            "slow_qwen_seconds": row["slow_qwen_seconds"],
                        }) + "\n")
                    handle.flush()
                    print(json.dumps({"variant": name, "episodes": count,
                                      "total": len(specs), **_driver_memory_gb()}), flush=True)
                del rows
            seconds = time.perf_counter() - started
            after = ray.get([worker.stats.remote() for worker in fast])
            requests = sum(b["requests"] - a["requests"] for a, b in zip(before, after))
            batches = sum(b["batches"] - a["batches"] for a, b in zip(before, after))
            summary = {"variant": name, "config": config, "episodes": count,
                       "chunks": chunks, "successes": successes, "rollout_seconds": seconds,
                       "episodes_per_second": count / seconds,
                       "episode_p50": float(np.percentile(episode_times, 50)),
                       "episode_p95": float(np.percentile(episode_times, 95)),
                       "slow_qwen_seconds_sum": sum(slow_times),
                       "fast_mean_batch": requests / batches if batches else 0,
                       "fast_stats_before": before, "fast_stats_after": after,
                       **_driver_memory_gb()}
            summaries.append(summary)
            (variant / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
            for actor in actors + fast:
                ray.kill(actor)
            actors, fast = [], []
            ray.get(service.stop.remote())
            ray.kill(service)
            service = None
            gc.collect()
        (output / "DONE").write_text("all benchmark variants completed; no training performed\n")
    finally:
        for actor in actors + fast:
            ray.kill(actor)
        if service is not None:
            ray.get(service.stop.remote())
            ray.kill(service)
        ray.shutdown()


def request_benchmark_helpers():
    """Reuse the independent GPU gate and trusted real-payload loader."""
    import runpy
    repo = Path(__file__).resolve().parents[2]
    native = runpy.run_path(str(repo / 'scripts/eval/action_head_d16_native_r3_probe.py'))
    replay = runpy.run_path(str(repo / 'scripts/eval/action_head_d16_learner_replay_probe.py'))
    return native, replay


def prepare_request_bank(directory, model, query, *, source_world_size=10, target_completion_tokens=76):
    """Preserve original JPEG bytes and prefixes; use one real completion bucket."""
    import base64
    from collections import Counter
    import torch
    from tokenizers import Tokenizer

    native, replay = request_benchmark_helpers()
    directory, model, query = Path(directory).resolve(), Path(model).resolve(), Path(query).resolve()
    if query.is_dir():
        query = query / 'dualvln_query_embeddings.pt'
    expected = {f'records_rank_{rank:03d}.pt' for rank in range(source_world_size)}
    if {path.name for path in directory.glob('records_rank_*.pt')} != expected:
        raise ValueError('all original source rank payloads are required')
    tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
    saved_query = torch.load(query, map_location='cpu', weights_only=True)
    if saved_query.get('query_count') != 16 or saved_query.get('hidden_convention') != 'final_norm':
        raise ValueError('request benchmark requires the original D16 final_norm query checkpoint')
    config = json.loads((model / 'config.json').read_text())
    config = config.get('text_config', config)
    rows, shards, seen = [], [], set()
    for name in sorted(expected):
        path = directory / name
        records, digest = replay['load_shard'](path)
        shards.append({'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size})
        for record in records:
            condition = record['condition_id']
            if not record.get('sample_valid', 1) or condition in seen:
                continue
            seen.add(condition)
            messages = record['slow_request']['messages']
            response = record['slow_response']
            prompt_tokens = response['usage']['prompt_tokens']
            token_ids = record['all_input_ids'].tolist()
            if (record['query_routed_experts'].shape[0] != 16
                    or token_ids[-16:] != [248060] * 16
                    or len(token_ids) != prompt_tokens + response['usage']['completion_tokens'] - 1):
                raise ValueError('source request has an incomplete D16 token sequence')
            forced = token_ids[prompt_tokens:-16]
            if not forced or tokenizer.decode(forced, skip_special_tokens=False).strip() != response['text'].strip():
                raise ValueError('source completion does not match its exact captured token IDs')
            images = []
            for message in messages:
                content = message.get('content')
                for part in content if isinstance(content, list) else []:
                    if part.get('type') == 'image_url':
                        url = part['image_url']['url']
                        if not url.startswith('data:image/jpeg;base64,'):
                            raise ValueError('source must preserve the original production JPEG data URLs')
                        raw = base64.b64decode(url.split(',', 1)[1], validate=True)
                        if not raw.startswith(b'\xff\xd8'):
                            raise ValueError('source image is not JPEG')
                        images.append(hashlib.sha256(raw).hexdigest())
            if not images:
                raise ValueError('each source request must contain real multimodal input')
            rows.append({'condition_id': condition, 'messages': messages,
                         'prefix_token_ids': token_ids[:prompt_tokens],
                         'prompt_tokens': prompt_tokens, 'image_sha256': images,
                         'prefix_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                         'original_completion': response['text'], 'original_completion_token_ids': forced,
                         'source_shard': str(path), 'source_query_request_id': response['query_request_id']})
        del records
    if len({row['prefix_sha256'] for row in rows}) < 2:
        raise ValueError('a diverse request benchmark requires at least two distinct real prefixes')
    selected = min(rows, key=lambda row: abs(len(row['original_completion_token_ids']) + 17 - target_completion_tokens))
    bank = {'model': str(model), 'query_checkpoint': str(query),
            'query_shape': list(saved_query['query_embeddings'].shape),
            'layer_count': config['num_hidden_layers'], 'top_k': config['num_experts_per_tok'],
            'query_count': 16, 'rows': rows, 'source_shards': shards,
            'forced_completion_token_ids': selected['original_completion_token_ids'],
            'forced_completion': selected['original_completion'],
            'completion_source_condition': selected['condition_id'],
            'completion_tokens': len(selected['original_completion_token_ids']) + 17,
            'conditions': len(rows), 'distinct_prefixes': len({row['prefix_sha256'] for row in rows}),
            'distinct_images': len({image for row in rows for image in row['image_sha256']}),
            'images_per_condition': dict(Counter(len(row['image_sha256']) for row in rows)),
            'jpeg_contract': 'Reuse original data URLs byte-for-byte; production encoding was JPEG90 optimize=True.',
            'replay_contract': 'Diverse real prefixes with one fixed real completion bucket; not exact original trajectories.',
            'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (query, model / 'config.json', model / 'tokenizer.json')}}
    bank['input_fingerprint'] = hashlib.sha256(json.dumps(bank, sort_keys=True).encode()).hexdigest()
    return bank


def bounded_request_round(jobs, concurrency, request):
    """Keep HTTP requests supplied using the existing completion scheduler."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        groups = iter_grouped_rollouts(
            jobs, list(range(concurrency)), group_size=1, max_in_flight_groups=len(jobs),
            submit=lambda worker, job: pool.submit(request, job),
            wait_one=lambda futures: next(iter(wait(futures, return_when=FIRST_COMPLETED).done)),
            resolve=lambda future: future.result(), scheduler='completion')
        return [rows[0] for _, _, rows in groups]


def parse_request_metrics(raw):
    """Read only relevant vLLM Prometheus counters and gauges."""
    import re
    names = {'vllm:mm_cache_queries_total', 'vllm:mm_cache_hits_total',
             'vllm:generation_tokens_total', 'vllm:prompt_tokens_total',
             'vllm:num_requests_running', 'vllm:num_requests_waiting',
             'vllm:request_queue_time_seconds_sum', 'vllm:request_queue_time_seconds_count'}
    values = {}
    for line in raw.splitlines():
        match = re.match(r'([^\s{]+)(?:\{.*\})?\s+([\d.eE+\-]+)(?:\s|$)', line)
        if match and match[1] in names:
            values[match[1]] = values.get(match[1], 0.0) + float(match[2])
    return values


def request_metrics(url):
    import urllib.request
    with urllib.request.urlopen(url.rstrip('/') + '/metrics', timeout=10) as response:
        raw = response.read().decode()
    return parse_request_metrics(raw)


def request_capture_settings(mode):
    """Resolve the benchmark variant without changing production defaults."""
    if mode in ('legacy_defer', 'native_defer'):
        return {'capture_mode': mode.removesuffix('_defer'), 'defer_query_capture': True}
    if mode in ('legacy', 'native_audit', 'native'):
        return {'capture_mode': mode, 'defer_query_capture': False}
    raise ValueError(f'unsupported request benchmark mode: {mode}')


def run_request_benchmark(args):
    """No Ray cluster, gym, learner, optimizer or production output is touched."""
    import copy
    import socket
    import statistics
    import tempfile
    import threading
    from action_head.vllm_bridge import request_queries, strip_query_tail
    from action_head.serve.bench_query_isolation import run as stream_request

    if not args.modes or args.modes[0] not in ('legacy', 'native'):
        raise ValueError('request benchmark requires an undeferred legacy or native baseline')
    baseline = args.modes[0]
    native, _ = request_benchmark_helpers()
    output = args.output_dir.resolve()
    if not output.is_relative_to(native['BASE'].resolve()):
        raise ValueError('output must be inside the authorized optimization data root')
    output.mkdir(parents=True, exist_ok=False)
    local_parent = Path('/tmp/action_head_rl')
    local_parent.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix='rtp-', dir=local_parent))
    original_environment = dict(os.environ)
    write = native['write_json']
    service = None
    report = {'status': 'running', 'output_dir': str(output), 'runtime_root': str(runtime),
              'gpus': args.gpus, 'modes': [], 'baseline_mode': baseline, 'prefix_caching': False,
              'capture_settings': {mode: request_capture_settings(mode) for mode in args.modes},
              'tensor_parallel_size': args.tp_size,
              'max_num_seqs': args.max_num_seqs,
              'max_num_batched_tokens': args.max_num_batched_tokens,
              'timing_scope': 'Streaming HTTP with already encoded production JPEGs; artifact checks are outside throughput timing.'}
    source_paths = (Path(__file__).resolve(), native['REPO'] / 'verl/action_head/serve/bench_query_isolation.py',
                    native['REPO'] / 'verl/action_head/serve/dualvln_query_runtime.py',
                    native['REPO'] / 'verl/action_head/serve/native_r3_capture.py',
                    native['REPO'] / 'verl/action_head_rl/vllm_service.py')
    report['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    try:
        os.environ.update(native['runtime_environment'](runtime))
        os.environ.update(CUDA_VISIBLE_DEVICES='' if args.validate_inputs else args.gpus,
                          ACTION_HEAD_DEFER_QUERY_CAPTURE='0', VLLM_SERVER_DEV_MODE='1')
        import torch
        from safetensors.torch import load_file
        bank = prepare_request_bank(args.slow_request_shards, args.model, args.query_checkpoint)
        write(output / 'request_bank.json', bank)
        report['bank'] = {key: value for key, value in bank.items() if key != 'rows'}
        if args.validate_inputs:
            report['status'] = 'cpu_inputs_pass'
            return report
        with socket.socket() as probe:
            probe.bind(('0.0.0.0', args.port))
        os.environ['ACTION_HEAD_FORCED_COMPLETION_IDS'] = ','.join(map(str, bank['forced_completion_token_ids']))
        unique = list({row['prefix_sha256']: index for index, row in enumerate(bank['rows'])}.values())

        def request(index):
            source = bank['rows'][index % len(bank['rows'])]
            payload = {'model': 'Qwen3.6-35B', 'messages': copy.deepcopy(source['messages']),
                       'stream': True, 'stream_options': {'include_usage': True},
                       'temperature': 0.0, 'seed': 0, 'max_tokens': 256,
                       'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': True}}
            request_id = request_queries(payload, 16)
            response = stream_request(url, payload['model'], '', args.timeout, payload=payload)
            clean, count = strip_query_tail(response['text'], query_count=16)
            if (count != 16 or clean.strip() != bank['forced_completion'].strip()
                    or response['usage'].get('prompt_tokens') != source['prompt_tokens']
                    or response['completion_tokens'] != bank['completion_tokens']):
                raise ValueError('streaming response changed the fixed prefix/completion/query workload')
            return {'source_index': index % len(bank['rows']), 'request_index': index, 'request_id': request_id,
                    'input_fingerprint': bank['input_fingerprint'], 'clean_completion': clean,
                    **response}

        def validate(record, destination=None):
            source = bank['rows'][record['source_index']]
            artifact = cache / (record['request_id'] + '.safetensors')
            tensors = load_file(str(artifact), device='cpu')
            native['validate_artifact'](tensors, bank, bank['forced_completion_token_ids'])
            expected_ids = source['prefix_token_ids'] + bank['forced_completion_token_ids'] + [248060] * 16
            if not torch.equal(tensors['all_input_ids'], torch.tensor(expected_ids)):
                raise ValueError('HTTP processing changed the captured original prefix token IDs')
            if capture_mode != 'legacy':
                audit = json.loads((audit_root / (record['request_id'] + '.json')).read_text())
                native['validate_native_report'](audit, capture_mode, record['request_id'], len(expected_ids))
                if (cache / (record['request_id'] + '.query.safetensors')).exists():
                    raise ValueError('native publication left a query-only sidecar')
            if destination:
                destination.mkdir(parents=True)
                shutil.copy2(artifact, destination / 'query.safetensors')
                write(destination / 'record.json', record)
            artifact.unlink()

        for mode in args.modes:
            settings = request_capture_settings(mode)
            capture_mode = settings['capture_mode']
            directory = output / mode
            directory.mkdir()
            cache, audit_root = runtime / mode / 'query_cache', directory / 'native_r3_audit'
            os.environ.update(ACTION_HEAD_R3_CAPTURE_MODE=capture_mode,
                              ACTION_HEAD_DEFER_QUERY_CAPTURE=str(int(settings['defer_query_capture'])),
                              ACTION_HEAD_R3_AUDIT_DIR=str(audit_root))
            service = VllmService(repo=str(native['REPO']), model=bank['model'],
                query_checkpoint=bank['query_checkpoint'], query_cache=str(cache),
                log_file=str(directory / 'vllm.log'), port=args.port, capture_full_routes=True,
                bulk_route_capture=True, max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=args.max_num_batched_tokens,
                startup_timeout=args.startup_timeout,
                tensor_parallel_size=args.tp_size)
            started = time.perf_counter()
            service.start()
            url = f'http://127.0.0.1:{args.port}'
            result = {'mode': mode, **settings, 'startup_seconds': time.perf_counter() - started,
                      'serial_numeric_gate': [], 'rounds': []}
            report['modes'].append(result)
            cold = request(unique[0]); validate(cold)
            result['cold_request'] = cold
            for index in unique:
                record = request(index)
                destination = directory / 'numeric_gate' / f'prefix_{index:03d}'
                validate(record, destination)
                if mode != baseline:
                    comparison = native['compare_records'](
                        output / baseline / 'numeric_gate' / f'prefix_{index:03d}', destination)
                    result['serial_numeric_gate'].append(comparison)
            # Compile warm, then clear both sides of the image cache before the
            # first timed round. Later rounds intentionally retain their cache.
            service._post('/reset_mm_cache')
            service._post('/reset_encoder_cache')
            result['cache_reset_endpoints'] = ['/reset_mm_cache', '/reset_encoder_cache']
            for iteration in range(args.rounds):
                before = request_metrics(url)
                samples, sampling_errors, done = [], [], threading.Event()
                def sample_metrics():
                    while not done.wait(2):
                        try:
                            samples.append(request_metrics(url))
                        except Exception as error:
                            sampling_errors.append(str(error))
                sampler = threading.Thread(target=sample_metrics, daemon=True)
                sampler.start()
                started = time.perf_counter()
                try:
                    rows = bounded_request_round(list(range(args.requests)), args.concurrency, request)
                    elapsed = time.perf_counter() - started
                finally:
                    done.set(); sampler.join(timeout=11)
                after = request_metrics(url)
                # Validate the whole timed batch only after the final HTTP
                # request completes, so CPU checks do not delay task refill.
                observed_prefixes = set()
                for record in rows:
                    prefix = bank['rows'][record['source_index']]['prefix_sha256']
                    record['prefix_seen_earlier_in_round'] = prefix in observed_prefixes
                    observed_prefixes.add(prefix)
                    validate(record)
                delta = {key: after[key] - before[key] for key in after
                         if key.endswith('_total') and key in before}
                queries = delta.get('vllm:mm_cache_queries_total', 0)
                hit_rate = (delta.get('vllm:mm_cache_hits_total', 0) / queries if queries else None)
                row = {'iteration': iteration, 'image_cache_phase': 'first_after_reset' if iteration == 0 else 'repeat',
                       'requests': len(rows), 'concurrency': args.concurrency, 'wall_seconds': elapsed,
                       'requests_per_second': len(rows) / elapsed,
                       'completion_tokens_per_second': sum(r['completion_tokens'] for r in rows) / elapsed,
                       'latency_median_seconds': statistics.median(r['total_s'] for r in rows),
                       'ttft_median_seconds': statistics.median(r['ttft_s'] for r in rows),
                       'mm_cache_hit_rate': hit_rate, 'metrics_before': before, 'metrics_after': after,
                       'metrics_delta': delta, 'metric_samples': samples, 'metric_sampling_errors': sampling_errors,
                       'distinct_prefixes': bank['distinct_prefixes'], 'distinct_images': bank['distinct_images'],
                       'first_prefix_occurrences': sum(not r['prefix_seen_earlier_in_round'] for r in rows),
                       'repeated_prefix_occurrences': sum(r['prefix_seen_earlier_in_round'] for r in rows),
                       'workload_is_formal_cache_match': False, 'all_request_artifacts_valid': True}
                result['rounds'].append(row)
                write(directory / f'round_{iteration:02d}_requests.json', rows)
                write(output / 'report.json', report)
            service.stop(); service = None
            result['status'] = 'pass'
            write(directory / 'report.json', result)
        report['status'] = 'pass'
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
            report.update(status='error', service_stopped=False, cleanup_error=str(error))
            raise
        finally:
            if report.get('service_stopped'):
                shutil.rmtree(runtime)
            report['runtime_cleaned'] = not runtime.exists()
            os.environ.clear(); os.environ.update(original_environment)
            write(output / 'report.json', report)


def request_benchmark_main(argv=None):
    import argparse
    native, replay = request_benchmark_helpers()
    parser = argparse.ArgumentParser(description='Real-prefix streaming D16 R3 throughput; no training.')
    parser.add_argument('--slow-request-shards', type=Path, default=replay['DEFAULT_INPUT'])
    parser.add_argument('--model', type=Path, default=replay['SLOW_MODEL'])
    parser.add_argument('--query-checkpoint', type=Path, default=native['QUERY'])
    parser.add_argument('--output-dir', type=Path, default=native['BASE'] / 'd16-real-prefix-throughput-r1-20260912')
    parser.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    parser.add_argument('--tp-size', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--port', type=int, default=28897)
    parser.add_argument('--concurrency', type=int, choices=(24, 32), default=32)
    parser.add_argument('--max-num-seqs', type=int, default=64)
    parser.add_argument('--max-num-batched-tokens', type=int, default=32768)
    parser.add_argument('--requests', type=int, default=96)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--modes', default='legacy,native')
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--startup-timeout', type=int, default=1800)
    parser.add_argument('--validate-inputs', action='store_true')
    args = parser.parse_args(argv)
    args.modes = args.modes.split(',')
    if (args.modes[0] not in ('legacy', 'native') or len(args.modes) != len(set(args.modes))
            or any(mode not in ('legacy', 'legacy_defer', 'native_audit', 'native', 'native_defer')
                   for mode in args.modes)):
        parser.error('modes must start with legacy or native and contain unique supported R3 modes')
    devices = [value.strip() for value in args.gpus.split(',') if value.strip()]
    if not args.validate_inputs and (len(devices) != args.tp_size
                                     or len(set(devices)) != args.tp_size):
        parser.error(f'exactly {args.tp_size} explicit GPUs are required for TP{args.tp_size}')
    if args.max_num_seqs < args.concurrency:
        parser.error('max-num-seqs must cover request concurrency')
    if args.max_num_batched_tokens <= 0:
        parser.error('max-num-batched-tokens must be positive')
    args.gpus = ','.join(devices)
    if args.requests < 96 or args.rounds < 3:
        parser.error('at least 96 requests per round and three rounds are required')
    return run_request_benchmark(args)


if __name__ == '__main__':
    result = request_benchmark_main()
    print(json.dumps({'status': result['status'], 'output_dir': result['output_dir']}))
