"""Read-only checkpoint browsing and transactional, device-thread-owned loading."""
from pathlib import Path
import hashlib
import json
import queue
import threading
import os
import time
from collections import deque

ARTIFACT_ROOT = Path('/data/zerovln/artifacts')


def contained_path(value='', root=ARTIFACT_ROOT):
    root = Path(root).resolve(strict=True)
    path = Path(value) if value else root
    path = (path if path.is_absolute() else root / path).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError('只能选择 artifacts 内的目录')
    return path


def inspect_checkpoint(value, root=ARTIFACT_ROOT):
    path = contained_path(value, root)
    snapshot = path.parent if path.name == 'action_head' else path
    snapshot = contained_path(snapshot, root)

    def required(relative):
        target = (snapshot / relative).resolve(strict=True)
        if not target.is_relative_to(Path(root).resolve()) or not target.is_file():
            raise ValueError(f'权重文件不在 artifacts 内：{relative}')
        if not target.stat().st_size:
            raise ValueError(f'权重文件为空：{relative}')
        return target

    head = required('action_head/fast_slow_action_head.pt')
    config_file = required('action_head/fast_slow_action_head_config.json')
    config = json.loads(config_file.read_text())
    report_file = required('report.json')
    report = json.loads(report_file.read_text())
    if report.get('ok') is not True:
        raise ValueError('checkpoint report 尚未通过或未完成发布')
    step = report.get('global_step')
    if snapshot.name.startswith('global_step_') and snapshot.name != f'global_step_{int(step):06d}':
        raise ValueError('checkpoint 目录与 report 步数不一致')
    native = report.get('condition_mode') == 'v5_native'
    condition = required('context_aggregator.pt' if native else
                         'dualvln_query/dualvln_query_embeddings.pt')
    version = report.get('action_head_as_tool_version')
    if not version:
        raise ValueError('checkpoint report 缺少 action_head_as_tool_version')
    files = (head, config_file, condition, report_file)
    identity = hashlib.sha256('\n'.join(
        f'{p}:{p.stat().st_size}:{p.stat().st_mtime_ns}' for p in files).encode()).hexdigest()
    return dict(path=str(snapshot), checkpoint=str(head.parent), condition=str(condition),
                condition_mode='v5_native' if native else 'query', version=version,
                step=step, query_count=0 if native else int(config['query_num_queries']),
                visual_hidden_size=int(config['visual_hidden_size']),
                hidden_size=int(config['qwen_hidden_size']),
                frontend_model=report.get('frontend_model'), identity=identity)


def browse_checkpoints(value='', root=ARTIFACT_ROOT):
    """List one level only; do not recursively scan datasets or load tensors."""
    path = contained_path(value, root)
    root = Path(root).resolve()
    directories = []
    for child in sorted(path.iterdir(), key=lambda p: p.name):
        if child.name.startswith('.') or child.name.endswith('.publishing'):
            continue
        try:
            resolved = contained_path(child, root)
        except (OSError, ValueError):
            continue
        directories.append(dict(name=child.name, path=str(resolved)))
    checkpoint, error = None, ''
    if (path / 'action_head').is_dir() or path.name == 'action_head':
        try:
            checkpoint = inspect_checkpoint(path, root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            error = str(exc)
    return dict(ok=True, root=str(root), path=str(path),
                parent=str(path.parent) if path != root else None,
                directories=directories, checkpoint=checkpoint, error=error)


def discover_checkpoints(root=ARTIFACT_ROOT, *, max_dirs=12000, max_seconds=20):
    """Bounded metadata-only discovery; reuse the loader's snapshot validation.

    Skip known non-checkpoint trees and stop descending at snapshots. Resolve
    symlinks inside the allowed root, deduplicating both aliases and cycles.
    """
    root = contained_path('', root)
    pending = deque([(root, 0)])
    seen, checkpoints, rejected, warnings = set(), [], [], []
    started = time.monotonic()
    skip = {'logs', 'tensorboard', 'cache', 'tmp', 'data', 'datasets', 'shards',
            'evaluation', 'eval', 'validation', 'analysis', 'diagnostics', 'launch',
            'submission_history', 'rollouts', 'frames', 'videos', 'images',
            'optimizer', 'actor', 'critic', 'wandb', 'audit', 'tests', 'test_artifacts',
            'raw', 'renders', 'observations', 'episodes', 'trajectories'}
    while pending:
        if len(seen) >= max_dirs or time.monotonic() - started >= max_seconds:
            warnings.append('扫描达到上限，列表可能不完整；可使用高级目录浏览补充选择')
            break
        candidate, depth = pending.popleft()
        try:
            path = contained_path(candidate, root)
            if path in seen:
                continue
            seen.add(path)
            with os.scandir(path) as entries:
                children = {entry.name: Path(entry.path) for entry in entries
                            if not entry.name.startswith('.') and not entry.name.endswith('.publishing')
                            and entry.is_dir()}
            if 'action_head' in children or path.name.startswith('global_step_'):
                try:
                    info = inspect_checkpoint(path, root)
                    parts = list(path.parent.relative_to(root).parts)
                    if 'checkpoints' in parts:
                        index = parts.index('checkpoints')
                        if index and all(part.startswith(parts[0]) for part in parts[index + 1:]):
                            parts = parts[:index]
                    # Collapse repeated experiment names around checkpoints/.
                    labels = []
                    for part in parts:
                        if part != 'checkpoints' and (not labels or part != labels[-1]):
                            labels.append(part)
                    info.update(experiment=' / '.join(labels) or path.name,
                                group=str(path.parent.relative_to(root)),
                                modified_at=(path / 'report.json').stat().st_mtime)
                    checkpoints.append(info)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    rejected.append(dict(path=str(path), reason=str(exc)))
                continue
            if depth >= 8:
                if children:
                    warnings.append(f'目录层级过深，未继续扫描：{path.relative_to(root)}')
                continue
            for name, child in sorted(children.items()):
                non_weights = depth > 0 and (name.startswith(('eval_', 'eval-', 'validation_',
                                                              'test_artifacts', 'render_', 'rollout_'))
                                             or name.endswith(('_data', '_cache', '_logs', '_eval')))
                if name not in skip and not non_weights:
                    if name == 'checkpoints' or 'checkpoints' in path.relative_to(root).parts:
                        pending.appendleft((child, depth + 1))
                    else:
                        pending.append((child, depth + 1))
        except (OSError, ValueError) as exc:
            warnings.append(f'{candidate.name}: {exc}')
    checkpoints.sort(key=lambda item: (item['modified_at'], int(item['step'] or 0)), reverse=True)
    return dict(ok=True, root=str(root), checkpoints=checkpoints, rejected=rejected,
                warnings=warnings, scanned_dirs=len(seen), scanned_at=time.time())


_catalog_lock = threading.Lock()
_catalog = None


def checkpoint_catalog(*, refresh=False):
    global _catalog
    with _catalog_lock:
        if refresh or _catalog is None or time.time() - _catalog['scanned_at'] > 60:
            _catalog = discover_checkpoints()
        return _catalog


class RuntimeController:
    """One persistent owner thread for loading and every executor operation.

    An unsuccessful load leaves the active runtime intact. A pending load is
    immutable; callers cannot enqueue multiple GPU loads by double-clicking.
    The factory may share frozen encoders across checkpoints on this thread.
    """
    def __init__(self, factory, publish, transition_lock=None):
        self.factory, self.publish = factory, publish
        self.lock = threading.Lock()
        self.transition_lock = transition_lock or threading.RLock()
        self.tasks = queue.Queue()
        self.active = None
        self.info = dict(state='unloaded', active=None, pending=None, error='')
        self.worker = None
        self.started = None
        self.phase_started = None
        self.warming = False
        self.encoder_started = None
        self.info.update(encoder_state='unloaded', encoder_phase='', encoder_elapsed_s=0,
                         encoder_error='')

    def status(self):
        with self.lock:
            result = dict(self.info)
            if self.started is not None and result['state'] == 'loading':
                result['elapsed_s'] = round(time.monotonic() - self.started, 2)
            if result['encoder_state'] == 'loading':
                result['encoder_elapsed_s'] = round(time.monotonic() - self.encoder_started, 2)
            return result

    def report_progress(self, phase):
        with self.lock:
            if self.warming:
                self.info['encoder_phase'] = phase
                return
            now = time.monotonic()
            if self.phase_started is not None:
                self.info['timings'].append(dict(phase=self.info['phase'],
                                                seconds=round(now - self.phase_started, 2)))
            self.phase_started = now
            self.info['phase'] = phase

    def _start_worker(self):
        if self.worker is None:
            self.worker = threading.Thread(target=self._work, daemon=True,
                                           name='manual-action-head-owner')
            self.worker.start()

    def prewarm(self, settings):
        with self.lock:
            if self.info['encoder_state'] in ('loading', 'ready'):
                return
            self.encoder_started = time.monotonic()
            self.info.update(encoder_state='loading', encoder_phase='预加载默认小脑编码器',
                             encoder_error='')
            self._start_worker()
            self.tasks.put(('warm', dict(settings), None))

    def load(self, selection, settings):
        with self.lock:
            if self.info['state'] == 'loading':
                raise ValueError('小脑正在加载，请等待完成')
            self._start_worker()
            self.started = self.phase_started = time.monotonic()
            self.info.update(state='loading', pending=dict(selection), error='',
                             phase='准备加载', timings=[], elapsed_s=0)
            self.tasks.put(('load', (dict(selection), dict(settings)), None))

    def _work(self):
        while True:
            method, payload, reply = self.tasks.get()
            if method == 'warm':
                self.warming = True
                try:
                    self.factory.prewarm(payload)
                    with self.lock:
                        self.info.update(encoder_state='ready', encoder_phase='默认小脑编码器已就绪')
                except Exception as exc:
                    with self.lock:
                        self.info.update(encoder_state='error', encoder_error=str(exc))
                finally:
                    with self.lock:
                        self.info['encoder_elapsed_s'] = round(time.monotonic() - self.encoder_started, 2)
                    self.warming = False
                continue
            if method == 'load':
                selection, settings = payload
                try:
                    # Re-selecting the active immutable snapshot does not load again.
                    if self.info['active'] and self.info['active']['identity'] == selection['identity']:
                        runtime = self.active
                    else:
                        runtime = self.factory(selection, settings)
                    with self.transition_lock:
                        self.publish(runtime, self, selection)
                        self.report_progress('加载完成')
                        with self.lock:
                            self.active = runtime
                            self.info['elapsed_s'] = round(time.monotonic() - self.started, 2)
                            self.info.update(state='loaded', active=selection, pending=None, error='')
                except Exception as exc:
                    with self.lock:
                        self.info['elapsed_s'] = round(time.monotonic() - self.started, 2)
                        self.info.update(state='error', pending=None, error=str(exc))
                continue
            try:
                if self.active is None:
                    raise RuntimeError('请先选择并加载小脑 checkpoint')
                target = self.active.tool_executor
                result = getattr(target, method)(**payload) if hasattr(target, method) else None
                reply.put((result, None))
            except Exception as exc:
                reply.put((None, exc))

    def _call(self, method, kwargs):
        reply = queue.Queue(maxsize=1)
        self.tasks.put((method, kwargs, reply))
        value, error = reply.get()
        if error:
            raise error
        return value

    def execute(self, **kwargs):
        return self._call('execute', kwargs)

    def end_episode(self):
        return self._call('end_episode', {})
