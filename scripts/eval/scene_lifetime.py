"""Keep bounded cached scenes alive until their concurrent requests finish."""
from collections import Counter
from contextlib import contextmanager
from functools import wraps
import threading


class SceneLifetime:
    def __init__(self, manager):
        self.manager = manager
        self.condition = threading.Condition(manager._lock)
        self.users = Counter()
        self.original_get = manager.get_scene

    def get_scene(self, scene_id):
        m = self.manager
        with self.condition:
            # Unknown scenes must retain the original immediate error behavior.
            if scene_id in m._available_scenes:
                while (scene_id not in m._scene_cache
                       and len(m._scene_cache) >= m.max_total_scenes
                       and all(self.users[s] for s in m._scene_cache)):
                    self.condition.wait()
            return self.original_get(scene_id)

    @contextmanager
    def lease(self, scene_id):
        with self.condition:
            scene = self.get_scene(scene_id)
            self.users[scene_id] += 1
        try:
            yield scene
        finally:
            with self.condition:
                self.users[scene_id] -= 1
                if not self.users[scene_id]:
                    del self.users[scene_id]
                self.condition.notify_all()

    def evict(self):
        with self.condition:
            candidates = [(value[2], name) for name, value in
                          self.manager._scene_cache.items() if not self.users[name]]
            if not candidates:
                return None
            scene_id = min(candidates)[1]
            self.unload(scene_id)
            return scene_id

    def unload(self, scene_id):
        with self.condition:
            if self.users[scene_id]:
                return False
            item = self.manager._scene_cache.pop(scene_id, None)
            if item is None:
                return False
            self.manager._gpu_scene_count[item[1]] -= 1
            # Drop the cache's reference, never delete fields on a shared object.
            # This also protects requests already in flight during installation.
            del item
            self.condition.notify_all()
            return True


def install(manager, app):
    """Install before serving, or on the specifically owned live renderer."""
    if getattr(manager, '_scene_lifetime', None) is not None:
        return manager._scene_lifetime
    from flask import jsonify, request

    lifetime = SceneLifetime(manager)
    with manager._lock:
        manager.get_scene = lifetime.get_scene
        manager._evict_oldest_scene = lifetime.evict
        manager._unload_scene = lifetime.unload
        manager.unload_scene = lifetime.unload

        def guarded(handler):
            @wraps(handler)
            def call(*args, **kwargs):
                data = request.get_json() if request.is_json else {}
                scene_id = data.get('scene_id') or request.args.get('scene_id')
                if not scene_id:
                    return handler(*args, **kwargs)
                # Preserve the existing HTTP error contract for acquisition.
                lease = lifetime.lease(scene_id)
                try:
                    lease.__enter__()
                except ValueError as error:
                    return jsonify({'error': str(error)}), 400
                except Exception as error:
                    return jsonify({'error': f'加载场景失败: {error}'}), 500
                try:
                    return handler(*args, **kwargs)
                finally:
                    lease.__exit__(None, None, None)
            return call

        for name in ('render_gs', 'render_gs_xyz_euler', 'render_gs_depth',
                     'render_gs_multi'):
            if name in app.view_functions:
                app.view_functions[name] = guarded(app.view_functions[name])
        manager._scene_lifetime = lifetime
    return lifetime
