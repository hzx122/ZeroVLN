# -*- coding: utf-8 -*-
"""vln_demo 的 gym 风格 HTTP 服务（单 session，纯 CPU sim，不含 vLLM）。

stdlib http.server 实现，零新依赖（gordon venv 没 gymnasium/FastAPI）。
HTTP 模式复用 vln_model.py:33/426/1107 的 ThreadingHTTPServer + Handler。

线程安全：habitat env 非线程安全，全局一把锁串行化所有写操作。

端点：
    GET  /health                  -> {"ok": true}
    GET  /action_space             -> JSON schema
    GET  /observation_space       -> JSON schema
    GET  /spec                    -> {scenes, scene_eps, splits}
    POST /reset   {scene,episode_id,instruction,seed}  -> {obs, info}
    POST /step    {action}                            -> {obs,reward,terminated,truncated,info}
    GET  /render?mode=rgb|map      -> image/jpeg bytes
    GET  /frame.jpg /map.jpg       -> 同上（兼容 webui 调试）
    POST /close                    -> {"closed": true}
    GET  /                         -> 最小状态页（debug）

verl rollout worker 经 client.VLNRemoteClient 调用，等价于本地 gym env。
"""
import os
import sys
import json
import io
import threading
import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# 确保能 import vln_gym 包（src_dir/vln_gym/...）
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import vln_gym.env as E  # noqa: E402
import vln_gym.fm_action as FMA  # noqa: E402  (启动时跑一次自检)

# 全局单 env + 锁（habitat env 非线程安全）
_ENV = None
_LOCK = threading.Lock()
_SESSION_LOCK = threading.RLock()
_SESSIONS = {}
_SPEC = {"scenes": [], "scene_eps": {}, "splits": []}


def _jpg_bytes(arr, q=85):
    from PIL import Image
    img = Image.fromarray(arr).convert("RGB") if not isinstance(arr, Image.Image) \
        else arr.convert("RGB")
    b = io.BytesIO()
    img.save(b, "JPEG", quality=q)
    return b.getvalue()


def _ep_scene(e):
    """场景名：zy 后端用 .scene 字段（abotn 的 scene_id.split('/')[-2] 会错名），
    r2r/rxr 用 scene_id.split('/')[-2]。"""
    return getattr(e, "scene", None) or e.scene_id.split("/")[-2]


def _build_env(scene, episode_id, split, max_steps, port, backend="r2r", *, defer_initial_reset=False):
    kwargs = {'defer_initial_reset': True} if defer_initial_reset else {}
    env = E.VLNEnv(scene=scene, episode_id=episode_id, split=split,
                   max_steps=max_steps, render_mode="rgb", backend=backend, **kwargs)
    scene_eps = {}
    for e in env.env.episodes:
        scene_eps.setdefault(_ep_scene(e), []).append(str(e.episode_id))
    for sc in scene_eps:
        scene_eps[sc].sort(key=lambda x: int(x) if x.isdigit() else x)
    _SPEC["scenes"] = sorted(scene_eps)
    _SPEC["scene_eps"] = scene_eps
    _SPEC["splits"] = [split]
    _SPEC["backend"] = backend
    _SPEC["backends"] = list(E.ALL_BACKENDS)
    return env


def _rebuild_env(backend, scene, episode_id):
    """重建 env（切 backend 或长寿命 env 重建）。调用方须持 _LOCK。旧 env 先 close。
    max_steps/port/split 取自 _SPEC（main() 启动时写入）。"""
    global _ENV
    if _ENV is not None:
        try:
            _ENV.close()
        except Exception:
            pass
    _ENV = _build_env(scene, episode_id, _SPEC["splits"][0] if _SPEC["splits"] else "val_unseen",
                      _SPEC.get("max_steps", 500), _SPEC.get("port", 8899), backend)
    return _ENV


class Handler(BaseHTTPRequestHandler):
    server_version = "VLNGym/1.0"

    def log_message(self, *a):
        pass  # quiet

    def _send(self, body, ctype="application/json", code=200, headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_img(self, arr):
        self._send(_jpg_bytes(arr), "image/jpeg")

    def _err(self, code, msg, **details):
        self._send({"error": str(msg), **details}, "application/json", code)

    def _reset_error(self, error, backend):
        from sage.navmesh_support import SourceCaseGeometryUnsupported
        # Only explicit original-case geometry guards are rule abstentions.
        # A missing asset or a renderer/environment failure remains HTTP 500.
        unsupported = (backend == 'sage' and isinstance(error, ValueError) and
            str(error) == 'original SAGE pose is not supported; automatic XY snapping is forbidden')
        unsupported_geometry = backend == 'sage' and isinstance(error, SourceCaseGeometryUnsupported)
        kind = ('unsupported_original_pose' if unsupported else
                'unsupported_original_geometry' if unsupported_geometry else 'environment_error')
        return self._err(422 if unsupported or unsupported_geometry else 500,
            f"reset failed: {error}", error_kind=kind)

    def do_OPTIONS(self):
        self._send({"ok": True})

    def do_GET(self):
        return self._dispatch_session(self._do_GET)

    def do_POST(self):
        return self._dispatch_session(self._do_POST)

    def _dispatch_session(self, handler):
        """Reuse the original HTTP operations on one privately owned live env.

        The server remains single-threaded, preserving Habitat/EGL ownership.
        Only /sessions/<id>/ URLs opt in; legacy routes retain their boot env.
        Sessions are created by reset and released by close, never recreated
        between depths. No pose teleportation or serialized physics is used.
        """
        global _ENV
        with _SESSION_LOCK:
            parts = self.path.split('/', 3)
            if len(parts) != 4 or parts[1] != 'sessions':
                return handler()
            session = parts[2]
            if not session or len(session) > 80 or not all(c.isalnum() or c in '_-' for c in session):
                return self._err(400, 'invalid session id')
            original_path, original_input, boot_env = self.path, self.rfile, _ENV
            self.path = '/' + parts[3]
            endpoint = urlparse(self.path).path
            try:
                if session not in _SESSIONS and endpoint == '/reset':
                    raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                    self.rfile = io.BytesIO(raw)
                    try:
                        body = json.loads(raw.decode('utf-8')) if raw else {}
                        if not isinstance(body, dict):
                            raise ValueError('reset body must be an object')
                    except (UnicodeDecodeError, ValueError) as error:
                        return self._err(400, f"bad json: {error}")
                    backend = body.get('backend') or getattr(boot_env, 'backend', 'r2r')
                    try:
                        env = _build_env(
                            body.get('scene') or boot_env.scene,
                            body.get('episode_id') or boot_env.episode_id,
                            _SPEC['splits'][0], _SPEC.get('max_steps', 500),
                            _SPEC.get('port', 8899), backend)
                    except Exception as error:
                        import traceback
                        print(f"[vln_gym] session reset failed:\n{traceback.format_exc()}", flush=True)
                        return self._reset_error(error, backend)
                    _SESSIONS[session] = env
                if session not in _SESSIONS:
                    if endpoint not in ('/health','/spec','/action_space','/observation_space'):
                        return self._err(409, 'session must be reset before execution')
                    return handler()
                _ENV = _SESSIONS[session]
                if endpoint == '/close':
                    # Release ownership before returning the close response.
                    _SESSIONS.pop(session)
                return handler()
            finally:
                if session in _SESSIONS and _ENV is not boot_env:
                    _SESSIONS[session] = _ENV
                    if endpoint == '/close':
                        _SESSIONS.pop(session, None)
                _ENV = boot_env
                self.path, self.rfile = original_path, original_input
                if endpoint == '/close' and session not in _SESSIONS:
                    callback = globals().get('_on_session_closed')
                    if callback is not None:
                        callback()

    def _do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p == "/health":
            self._send({"ok": True, "env_loaded": _ENV is not None,
                        "session_contract": "independent_gym_sessions_v1",
                        "active_sessions": len(_SESSIONS)})
            return
        if p == "/telemetry":
            with _LOCK:
                if _ENV is None or getattr(_ENV, '_initial_reset_deferred', False):
                    self._send({"ready": False})
                else:
                    self._send({"ready": True, "scene": _ENV.scene,
                                "episode_id": _ENV.episode_id,
                                "steps_used": _ENV.step_id,
                                "metrics": {key: value for key, value in _ENV.env.get_metrics().items()
                                    if isinstance(value, (str, int, float, bool)) or value is None
                                    or key == "official_object_success_detail"}})
            return
        if p == "/action_space":
            self._send(E.VLNEnv.action_space)
            return
        if p == "/observation_space":
            self._send(E.VLNEnv.observation_space)
            return
        if p == "/spec":
            self._send(_SPEC)
            return
        if (getattr(_ENV, '_initial_reset_deferred', False)
                and p in ('/frame.jpg', '/map.jpg', '/vln-map.jpg', '/render')):
            return self._err(409, 'environment must be reset before rendering')
        if p == "/frame.jpg":
            with _LOCK:
                arr = _ENV.render("rgb") if _ENV else None
            if arr is None:
                self._err(503, "env not ready")
            else:
                self._send_img(arr)
            return
        if p in ("/map.jpg", "/vln-map.jpg"):
            with _LOCK:
                show_goal = (p == "/map.jpg" and
                             parse_qs(u.query).get("show_goal", ["1"])[0] != "0")
                arr = _ENV.render("map", show_goal=show_goal) if _ENV else None
            if arr is None:
                self._err(503, "no map")
            else:
                self._send_img(arr)
            return
        if p == "/render":
            q = parse_qs(u.query)
            mode = (q.get("mode") or ["rgb"])[0]
            with _LOCK:
                arr = _ENV.render(mode) if _ENV else None
            if arr is None:
                self._err(503, f"render {mode} unavailable")
            else:
                self._send_img(arr)
            return
        if p == "/" or p == "/index.html":
            self._send(self._status_page(), "text/html; charset=utf-8")
            return
        self._err(404, f"unknown path {p}")

    def _do_POST(self):
        u = urlparse(self.path)
        p = u.path
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception as e:
            self._err(400, f"bad json: {e}")
            return
        if p == "/reset":
            with _LOCK:
                try:
                    be = body.get("backend")
                    cur = getattr(_ENV, "backend", "r2r") if _ENV else None
                    if be and be != cur:
                        # 切 backend = 重建 env（不同 sim/render server）。
                        sc = body.get("scene") or (_ENV.scene if _ENV else None)
                        ep = body.get("episode_id") or (_ENV.episode_id if _ENV else "0")
                        _rebuild_env(be, sc, ep)
                    obs, info = _ENV.reset(
                        scene=body.get("scene"),
                        episode_id=body.get("episode_id"),
                        instruction=body.get("instruction"),
                        seed=body.get("seed"))
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f"[vln_gym] reset failed:\n{tb}", flush=True)
                    self._reset_error(e, be or cur)
                    return
            self._send({"obs": obs, "info": info})
            return
        if p == "/step":
            if getattr(_ENV, '_initial_reset_deferred', False):
                return self._err(409, 'environment must be reset before execution')
            action = body.get("action")
            with _LOCK:
                try:
                    obs, reward, terminated, truncated, info = _ENV.step(action)
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f"[vln_gym] step failed (action={action!r}):\n{tb}",
                          flush=True)
                    self._err(500, f"step failed: {e}")
                    return
            self._send({"obs": obs, "reward": reward,
                        "terminated": terminated, "truncated": truncated,
                        "info": info})
            return
        if p == "/close":
            with _LOCK:
                if _ENV:
                    _ENV.close()
            self._send({"closed": True})
            return
        self._err(404, f"unknown path {p}")

    def _status_page(self):
        with _LOCK:
            env = _ENV
            ready = env is not None
            backend = getattr(env, "backend", "r2r") if env else "-"
            scene = getattr(env, "scene", "-") if env else "-"
            ep = getattr(env, "episode_id", "-") if env else "-"
            step = getattr(env, "step_id", 0) if env else 0
            dist = "-"
            if env is not None:
                try:
                    m = env.env.get_metrics()
                    d = m.get("distance_to_goal", float("nan"))
                    dist = "-" if d != d else f"{float(d):.2f} m"
                except Exception:
                    pass
        return f"""<!doctype html><html><head><meta charset=utf-8>
<title>VLN Gym Bench</title>
<style>body{{font:14px/1.6 -apple-system,monospace;background:#0d1117;color:#e6edf3;padding:24px}}
.pill{{display:inline-block;padding:2px 10px;border-radius:11px;background:#161b22;border:1px solid #2d333b}}
h1{{font-size:16px}} img{{max-width:640px;display:block;margin:10px 0;border:1px solid #2d333b}}</style></head>
<body><h1>VLN Gym Bench (单 session, 纯 CPU, 无 vLLM)</h1>
<p><span class=pill>状态 {('就绪' if ready else '加载中')}</span>
<span class=pill>后端 <b>{backend}</b></span>
<span class=pill>场景 <b>{scene}</b></span><span class=pill>回合 <b>{ep}</b></span>
<span class=pill>步数 <b>{step}</b></span><span class=pill>距目标 <b>{dist}</b></span></p>
<p>用法见 <code>/action_space</code> <code>/observation_space</code> <code>/spec</code>，
POST <code>/reset</code> / <code>/step</code>。</p>
<img src=/frame.jpg><img src=/map.jpg>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("DEMO_HTTP_PORT", "8899")))
    ap.add_argument("--scene", default=os.environ.get("DEMO_SCENE", "2azQ1b91cZZ"))
    ap.add_argument("--episode_id", default=os.environ.get("DEMO_EPISODE_ID", "11"))
    ap.add_argument("--split", default=os.environ.get("DEMO_SPLIT", "val_unseen"))
    ap.add_argument("--max_steps", type=int, default=int(os.environ.get("DEMO_MAX_STEPS", "500")))
    ap.add_argument("--backend", default=os.environ.get("DEMO_BACKEND", "r2r"),
                    choices=list(E.ALL_BACKENDS),
                    help="sim backend: r2r(默认) / rxr / rs3d / hsim / abotn / sage / mall")
    args = ap.parse_args()

    # 存进 _SPEC 供 /reset 切 backend 重建时复用同口径。
    _SPEC["max_steps"] = args.max_steps
    _SPEC["port"] = args.port

    global _ENV
    print(f"[vln_gym] booting: backend={args.backend} scene={args.scene} "
          f"ep={args.episode_id} split={args.split} port={args.port}", flush=True)
    _ENV = _build_env(args.scene, args.episode_id, args.split,
                      args.max_steps, args.port, args.backend,
                      defer_initial_reset=(args.backend == 'sage' and
                          os.environ.get('VLN_GYM_DEFER_INITIAL_RESET') == '1'))
    print(f"[vln_gym] env ready, {_SPEC['scenes'].__len__()} scenes, "
          f"port {args.port}", flush=True)

    srv = HTTPServer(("0.0.0.0", args.port), Handler)
    # 不用 ThreadingHTTPServer：habitat/Magnum 的 EGL GL context 是
    # thread-local，handler 线程 live 调 env.step/reset 会 'no current context'
    # 然后 SIGABRT。单线程 HTTPServer 保证所有 env 访问都在创建 env 的主线程。
    # 代价：一个慢 step 会阻塞 /health，但单 session RL 场景可接受。
    print(f"[vln_gym] HTTP listening on 0.0.0.0:{args.port}  (single-threaded, "
          f"open http://127.0.0.1:{args.port}/)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with _LOCK:
            if _ENV:
                _ENV.close()


if __name__ == "__main__":
    main()
