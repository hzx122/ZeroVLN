"""Probe-only strict determinism around the existing full learner replay.

Use VERL's per-worker initialization recipe, then upgrade its warning-only
policy to errors during the complete replay. This does not patch kernels,
change training defaults, or silently retry unsupported operations.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import traceback


BASE = Path('/data/zerovln/action_head/rl/'
            'd16_resume_acceptance_20260911/optimization')
VERL_HELPER = Path('/workspace/verl/verl/workers/engine/utils.py')
ENV_KEYS = ('PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG', 'FLASH_ATTENTION_DETERMINISTIC',
            'CUDA_LAUNCH_BLOCKING', 'HCCL_DETERMINISTIC', 'CLOSE_MATMUL_K_SHIFT',
            'NCCL_ALGO', 'NCCL_PROTO', 'ACTION_HEAD_VLM_LR')


def configure_strict_determinism(config):
    """Enable the existing initialization recipe in each future FSDP worker."""
    return replace(config, engine_config=replace(
        config.engine_config, full_determinism=True))


def _callable_description(function):
    if function is None:
        return None
    function = inspect.unwrap(function)
    result = {'module': getattr(function, '__module__', None),
              'qualname': getattr(function, '__qualname__', type(function).__qualname__)}
    try:
        result['source'] = inspect.getsourcefile(function)
    except (TypeError, OSError):
        result['source'] = None
    return result


def capture_runtime_determinism(worker):
    import torch

    def query(function):
        try:
            return function()
        except Exception as error:
            return {'unavailable': f'{type(error).__name__}: {error}'}

    engine = worker.engine
    module = engine.module
    qwen = module.qwen
    text_config = getattr(qwen.config, 'text_config', qwen.config)
    cuda = torch.backends.cuda
    report = {'torch_version': torch.__version__,
        'full_determinism_config': bool(engine.engine_config.full_determinism),
        'seed': engine.engine_config.seed,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cudnn': {key: query(lambda key=key: getattr(torch.backends.cudnn, key))
                  for key in ('enabled', 'deterministic', 'benchmark', 'allow_tf32')},
        'matmul': {key: query(lambda key=key: getattr(cuda.matmul, key))
                   for key in ('allow_tf32', 'allow_bf16_reduced_precision_reduction',
                               'allow_fp16_reduced_precision_reduction')},
        'sdpa': {name: query(getattr(cuda, name)) for name in
                 ('flash_sdp_enabled', 'mem_efficient_sdp_enabled', 'math_sdp_enabled', 'cudnn_sdp_enabled')
                 if hasattr(cuda, name)},
        'environment': {key: os.environ.get(key) for key in ENV_KEYS},
        'experts_implementation': getattr(text_config, '_experts_implementation', None),
        'attention_implementation': getattr(text_config, '_attn_implementation', None),
        'gradient_checkpointing': bool(getattr(qwen, 'is_gradient_checkpointing', False)),
        'layers': []}
    for index, layer in enumerate(module.query_forward._backbone().layers):
        linear = getattr(layer, 'linear_attn', None)
        row = {'layer': index, 'class': type(layer).__qualname__,
               'checkpointing': bool(getattr(layer, 'gradient_checkpointing', False)),
               'experts_forward': _callable_description(layer.mlp.experts.forward)}
        if linear is not None:
            row['linear_attention'] = {name: _callable_description(getattr(linear, name, None))
                for name in ('chunk_gated_delta_rule', 'causal_conv1d_fn', 'recurrent_gated_delta_rule')}
            row['linear_attention']['norm_class'] = type(linear.norm).__qualname__
            row['linear_attention']['conv_fallback_class'] = type(linear.conv1d).__qualname__
        report['layers'].append(row)
    return report


@contextmanager
def strict_determinism_audit(worker, output_dir):
    """Record active flags and preserve the first failure without retrying it."""
    import torch

    output = Path(output_dir).resolve()
    if not output.is_relative_to(BASE.resolve()):
        raise ValueError('determinism report must be under the authorized optimization root')
    rank = int(worker.engine.rank)
    path = output / 'determinism_audit' / f'rank_{rank:03d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'determinism audit already exists: {path}')
    original_enabled = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    report = {'status': 'initializing', 'rank': rank,
              'scope': 'probe only; full initialization recipe plus strict replay errors',
              'unsupported_operations_retried': False,
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__).resolve(), VERL_HELPER)}}

    def write():
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    try:
        if worker.engine.action_head_train_scope.value != 'full':
            raise ValueError('strict determinism diagnostic requires full train scope')
        if not worker.engine.engine_config.full_determinism:
            raise ValueError('VERL full_determinism must be enabled before worker initialization')
        report['after_verl_initialization'] = capture_runtime_determinism(worker)
        if not original_enabled:
            raise RuntimeError('VERL full_determinism initialization did not enable deterministic algorithms')
        torch.use_deterministic_algorithms(True, warn_only=False)
        report['during_strict_replay'] = capture_runtime_determinism(worker)
        if (not report['during_strict_replay']['deterministic_algorithms']
                or report['during_strict_replay']['warn_only']):
            raise RuntimeError('strict deterministic policy did not take effect')
        report['status'] = 'running'
        write()
        yield report
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        raise
    finally:
        torch.use_deterministic_algorithms(original_enabled, warn_only=original_warn_only)
        report['restored_policy'] = {
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'warn_only': torch.is_deterministic_algorithms_warn_only_enabled()}
        write()
