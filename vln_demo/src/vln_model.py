# -*- coding: utf-8 -*-
"""Interactive VLM-controlled VLN demo (web control panel).

Design notes
------------
* One worker thread owns the simulator and the model loop. HTTP handler threads
  only flip flags in CMD and read STATE, so habitat is never touched
  concurrently.
* Switching scene/episode does NOT rebuild the habitat Env. Setting
  ``env.current_episode`` flips ``_episode_from_iter_on_reset`` off, so the
  following ``env.reset()`` keeps that episode and ``reconfigure()`` swaps the
  scene in place. Rebuilding used to re-parse the whole R2R split every time,
  which is what made switching feel broken.
* Model replies are streamed token by token off the vLLM SSE endpoint, which
  also makes Stop/Load responsive: the stream loop checks the abort flag
  between chunks instead of blocking for a whole generation.
* A STOP action ends the run loop but is NOT stepped into the simulator, so the
  agent stays alive and drivable afterwards. That means the simulator's own
  success/spl are never triggered; we report distance-to-goal and re-derive the
  verdict from it, against whatever arrival radius the asset set uses (R2R's
  3 m, ABotN-POIBench's 2 m, PointBench's 0.5 m) -- see ``finish_run``.
* Eight asset sets, and by default **all of them at once** (``--backend all``):

    ``habitat``  the original R2R/MP3D task, through habitat-lab.
    ``rxr``      RxR on the very same 11 MP3D scenes, and therefore on the very
                 same simulator: a second episode list, not a second Env
                 (see rxr.py -- it is the ``current_episode`` setter that makes
                 an Env accept episodes from another dataset).
    ``3drs``     our own 3D-Gaussian captures -- rendered over HTTP by the 3DGS
                 server, walked with our own navmesh kinematics, because habitat
                 cannot open those at all (see rs3d/README.md).
    ``partnr``   HSSD synthetic homes with the official PARTNR episodes.
    ``hab``      the two hab2/ReplicaCAD benchmark stages, free roam.
    ``poi``      ABotN-POIBench: real streets in Gaussians, Chinese POI
                 instructions, ground-truth paths.
    ``in``       ABotN-PointBench indoor, point goals.
    ``out``      ABotN-PointBench outdoor, point goals over uneven ground.

  ``partnr``/``hab`` run on habitat_sim directly rather than habitat-lab, since
  those downloads ship scenes rather than a VLN task (see hsim/README.md). The
  three ABotN sets are Gaussians again but neither habitat nor a navmesh: they
  walk an occupancy grid and render off a *second* 3DGS server, because
  spherical-harmonic degree is fixed per server process and theirs differs from
  3DRS's (see abotn/README.md). The page asks for an asset set first and only
  then a scene, and the chosen scene is what selects the simulator:
  ``ACTIVE["env"]`` follows the episode. habitat and habitat_sim are therefore
  imported lazily, only when a backend that needs them is wanted, and everything
  below this point talks to whichever env it is handed.
"""
import sys
import pathlib
import os
import io
import base64
import binascii
import json
import hashlib
import time
import math
import threading
import queue
import weakref
import copy
import argparse
import socket
import signal
import select
import struct
import fcntl
import termios
import subprocess
import http.client
import action_head_checkpoints as AHC
import urllib.request
import urllib.error
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

NAV_DIR = os.environ.get(
    "RYNNBRAIN_NAV_DIR",
    "/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation",
)
DEMO_CONFIG = os.environ.get("DEMO_CONFIG", str(pathlib.Path(__file__).resolve().with_name("vln_demo.yaml")))
sys.path.insert(0, os.path.join(NAV_DIR, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
VERL_ROOT = os.path.join(REPO_ROOT, "verl")
if VERL_ROOT not in sys.path:
    sys.path.insert(0, VERL_ROOT)
os.environ.setdefault("RYNNVLN_LANGUAGE", "en")

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import prompts as P
import webui
import agent_tools as A
import visual_navigation_tools as VNT
from remote_habitat.policy import mode as render_mode
import lightnav_driver as LND
from action_head import vllm_bridge as AHV
from action_head import protocol as AHP
from action_head import protocol_v3 as AHV3

# habitat is imported lazily, by the habitat backend only. The 3DRS backend
# (see rs3d/env.py) renders 3D Gaussians over HTTP and runs its own navmesh
# kinematics; habitat cannot open those scenes at all, so importing it up here
# would make the backend that works depend on the one that does not.
base = None
base_maps = None


def load_habitat():
    """Import the RynnBrain evaluator + habitat's map helpers on first use."""
    global base, base_maps
    if base is None:
        import rynnvln_eval as _base
        from habitat.utils.visualizations import maps as _maps
        from remote_habitat.client import install as install_remote_render
        install_remote_render()
        from vln_geometry import register as register_vln_geometry
        register_vln_geometry()
        base, base_maps = _base, _maps
    return base


FWD, LEFT, RIGHT, STOP = P.FWD, P.LEFT, P.RIGHT, P.STOP
SUCCESS_RADIUS = 3.0  # R2R convention, used for the reported success flag
GOLDEN_ONLY_MSG = "走专家轨迹功能只支持 SAGE"

# Printed when --backend 3drs cannot reach the renderer. The demo does not start
# the server itself: it holds Gaussians on the GPU and is meant to outlive many
# console runs (see rs3d/README.md).
RS3D_SERVER_HINT = (
    "cd /workspace/agentic_sim && "
    "venv_render/bin/python ABot-Navigation/render_sim_dynamic.py "
    "--scenes_root 3drs_gs_root --port 7036 --gpus 0 --sh_degree 0")

# A second render server, not the same one: --sh_degree is fixed per process and
# ABotN's plys carry 45 SH coefficients (degree 3) where 3DRS's carry none, so
# one process cannot load both. Build the scenes_root first with
# ``python3 -m abotn.make_scenes_root``.
ABOTN_SERVER_HINT = (
    "cd /workspace/agentic_sim && "
    "venv_render/bin/python repo/ABot-Navigation/render_server/render_sim_dynamic.py "
    "--scenes_root abotn_gs_root --port 7037 --gpus 1 --sh_degree 3")

SAGE_SERVER_HINT = (
    "先转场景 ply 再起服务（都在本仓 vln_demo 下，注意 --sh_degree 3，是独立进程；"
    "生产 sage_gs_root 已全量 nurec 重建 944 场景，直接起服务即可，无需再转）：\n"
    "  cd closed_loop_3dgs_rl/vln_demo/src && "
    "PYTHONPATH=. .venv_fm/bin/python -m sage.make_scenes_root --split train --source nurec\n"
    "  PYTHONPATH=render_server:submodules/diff-plane-rasterization:submodules/simple-knn:render_libs "
    ".venv_fm/bin/python render_server/render_sim_dynamic.py "
    "--scenes_root /data/zerovln/SAGE-3D/sage_gs_root "
    "--port 7038 --gpus 0 --sh_degree 3")

# mall_floor2 的 ply 是 degree 0（与 3DRS 同系），但 scenes_root 不同（在 zekang 盘），
# 所以要独立的 render server 进程。场景布局
# <scenes_root>/mall_floor2/gs/block0/point_cloud.ply（6.4G）。
MALL_SERVER_HINT = (
    "cd /workspace/agentic_sim && "
    "venv_render/bin/python ABot-Navigation/render_sim_dynamic.py "
    "--scenes_root /opt/external/user-9656ce61/ppu_sim/3drs_gs_root "
    "--port 7039 --gpus 3 --sh_degree 0 --max_scenes_per_gpu 2")
# mall 现在是 start_all.sh 起的第 4 个渲染服务（PPU3/7039，独立进程独立卡），
# 不再复用 3DRS 那个 7036 槽——见 start_all.sh 头注释的卡位连续约定。

# ReAct driver budgets. Env steps are NOT capped -- vln_demo.yaml sets
# max_episode_steps: 0 (habitat's "no limit") and the loop below no longer
# checks args.max_steps, so the robot can keep moving as long as the
# conversation lasts. Turns and tool calls are still bounded, generously: they
# are what a stuck model burns without going anywhere, and 100 turns of
# keep_thinking history is already the scale where the context window, not the
# counter, is the thing that ends the run.
# The classic driver is unaffected -- its loop bounds itself on args.max_steps.
MAX_TURNS = 100
# FM driver: how many 32-waypoint chunks one run may emit. Each chunk re-plans
# from the live frame, so this is the navigation-length ceiling; args.max_steps
# (sim sub-steps) still bounds the run below it.
MAX_FM_CHUNKS = 64
# The tool-call budget is per-run and set from the page (P.DEFAULT_MAX_TOOL_CALLS
# by default) rather than fixed here: it is the one limit the model is actually
# shown, so it is also the one worth turning up or down between runs.
# Shared by both Tool profiles.  The first action expert experiment uses the
# previously agreed eight-frame single-view history; this must never vary with
# the selected tool profile or it would contaminate the A/B.
KEEP_LAST_IMAGES = int(os.environ.get("RYNNVLN_HISTORY_IMAGES", "8"))
if KEEP_LAST_IMAGES <= 0:
    raise SystemExit("RYNNVLN_HISTORY_IMAGES 必须是正整数")

# Strict Agent-side ablation switch.  The ReAct loop below is shared; selecting
# a profile changes only the tool schemas shown to Qwen and the matching tool
# implementation available to it.
ACTION_HEAD_TOOL_EXECUTOR = None
ACTION_HEAD_AGENT_RUNTIME = None
ACTION_HEAD_LOAD_STATE = "unloaded"
ACTION_HEAD_LOAD_ERROR = ""
ACTION_HEAD_LOAD_LOCK = threading.Lock()
ACTION_HEAD_DATASET_SPLIT = ""
ACTION_HEAD_GYM_PROCESS = None
ACTION_HEAD_RUNTIMES = {}
ACTION_HEAD_EXECUTORS = {}
ACTION_HEAD_ACTIVE_VARIANT = ""
MANUAL_HEAD_CONTROLLER = None
HEAD_CONTROL_LOCK = threading.RLock()
HEAD_EPISODE_ACTIVE = threading.Event()


def manual_head_status():
    return (MANUAL_HEAD_CONTROLLER.status() if MANUAL_HEAD_CONTROLLER else
            dict(state='unloaded', active=None, pending=None, error=''))


def manual_head_settings(selection):
    # Only deployment resources are inherited. Legacy variant checkpoints and
    # ACTION_HEAD_CHECKPOINT must never override the user's explicit selection.
    config_path = os.environ.get('ACTION_HEAD_CONFIG',
        os.path.join(REPO_ROOT, 'vln_demo/config/action_head_runtime.json'))
    document = json.loads(pathlib.Path(config_path).read_text())
    settings = dict(document.get('defaults') or {})
    settings.update((document.get('splits') or {}).get(ACTION_HEAD_DATASET_SPLIT, {}))
    settings.update(document.get('manual') or {})
    defaults = {
        'device': 'cuda:0', 'visual_device': 'cuda:0',
        'query_encoder_model': '/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B',
    }
    for key, value in defaults.items():
        settings.setdefault(key, value)
    # The checkpoint's visual dimension identifies the existing two backbones.
    backbones = {2560: '/opt/external/user-e7a01947/Model/Qwen3.5-4B',
                 2048: '/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B'}
    if selection['visual_hidden_size'] not in backbones:
        raise ValueError('不支持的视觉特征维度')
    settings['backbone'] = backbones[selection['visual_hidden_size']]
    if selection.get('frontend_model'):
        settings['query_encoder_model'] = selection['frontend_model']
    for key, env in {'gym_url': 'ACTION_HEAD_GYM_URL', 'backbone': 'ACTION_HEAD_BACKBONE',
                     'device': 'ACTION_HEAD_DEVICE', 'visual_device': 'ACTION_HEAD_VISUAL_DEVICE',
                     'query_device': 'ACTION_HEAD_QUERY_DEVICE',
                     'query_encoder_model': 'ACTION_HEAD_QUERY_ENCODER_MODEL'}.items():
        if os.environ.get(env):
            settings[key] = os.environ[env]
    if not settings.get('gym_url'):
        raise ValueError(f'当前 split={ACTION_HEAD_DATASET_SPLIT} 未配置 gym_url')
    settings['checkpoint'] = selection['checkpoint']
    return settings


def publish_manual_head(runtime, executor, selection):
    global ACTION_HEAD_AGENT_RUNTIME, ACTION_HEAD_TOOL_EXECUTOR
    global ACTION_HEAD_ACTIVE_VARIANT, ACTION_HEAD_LOAD_STATE, ACTION_HEAD_LOAD_ERROR
    with HEAD_CONTROL_LOCK:
        if CFG.get('driver') == 'action_head':
            ACTION_HEAD_AGENT_RUNTIME, ACTION_HEAD_TOOL_EXECUTOR = runtime, executor
            ACTION_HEAD_ACTIVE_VARIANT = selection['path']
            ACTION_HEAD_LOAD_STATE, ACTION_HEAD_LOAD_ERROR = 'loaded', ''
            CFG['action_head_as_tool_version'] = selection['version']
            set_state(action_head_as_tool_version=selection['version'])
        push_event('sys', text='小脑已加载：' + selection['path'])


def load_manual_head(path):
    global MANUAL_HEAD_CONTROLLER
    with HEAD_CONTROL_LOCK:
        if HEAD_EPISODE_ACTIVE.is_set() or STATE['status'] in ('running', 'manual', 'loading') or any(
                CMD.get(key) for key in ('start', 'manual_start', 'switch', 'point_move')):
            raise ValueError('请先停止当前回合，再切换小脑')
        if CFG['driver'] != 'action_head':
            raise ValueError('请先选择 FM Head 驱动')
        selection = AHC.inspect_checkpoint(path)
        AHP.normalize_version(selection['version'])
        settings = manual_head_settings(selection)
        ensure_manual_head_controller()
        MANUAL_HEAD_CONTROLLER.load(selection, settings)
        push_event('sys', text='正在加载小脑：' + selection['path'])
        return selection


def ensure_manual_head_controller():
    global MANUAL_HEAD_CONTROLLER
    with HEAD_CONTROL_LOCK:
        if MANUAL_HEAD_CONTROLLER is None:
            from action_head_manual_runtime import ManualRuntimeFactory
            MANUAL_HEAD_CONTROLLER = AHC.RuntimeController(
                ManualRuntimeFactory(interrupted, action_head_progress,
                    load_progress=lambda phase: MANUAL_HEAD_CONTROLLER.report_progress(phase)),
                publish_manual_head, transition_lock=HEAD_CONTROL_LOCK)
        return MANUAL_HEAD_CONTROLLER


def is_action_head_driver(driver=None):
    return (driver or CFG.get("driver")) in (
        "action_head", "action_head_untrained", "action_head_2800",
        "action_head_10ep", "action_head_v43_7000", "action_head_v5_native", "action_head_v5_cached", "action_head_planner")


def action_head_variant(driver=None):
    d = driver or CFG.get("driver")
    if d == "action_head_planner":
        return "semantic_planner"
    if d == "action_head_untrained":
        return "untrained"
    if d == "action_head_2800":
        return "trained_2800"
    if d == "action_head_10ep":
        return "trained_10ep"
    if d == "action_head_v43_7000":
        return "trained_v43_7000"
    if d == "action_head_v5_native":
        return "v5_native"
    if d == "action_head_v5_cached":
        return "v5_native_cached"
    return "trained"


def _action_head_settings(variant=None):
    """Resolve deployment settings from the registry, with optional overrides."""
    config_path = pathlib.Path(os.environ.get(
        "ACTION_HEAD_CONFIG",
        os.path.join(REPO_ROOT, "vln_demo/config/action_head_runtime.json")))
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RuntimeError(f"Action Head 配置不存在：{config_path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Action Head 配置无法读取：{config_path}（{exc}）") from None
    settings = dict(document.get("defaults") or {})
    settings.update((document.get("variants") or {}).get(
        variant or action_head_variant(), {}))
    settings.update((document.get("splits") or {}).get(
        ACTION_HEAD_DATASET_SPLIT, {}))
    # Environment variables remain useful for one-off experiments, but are no
    # longer required to make the normal console deployment loadable.
    overrides = {
        "checkpoint": os.environ.get("ACTION_HEAD_CHECKPOINT"),
        "backbone": os.environ.get("ACTION_HEAD_BACKBONE"),
        "gym_url": os.environ.get("ACTION_HEAD_GYM_URL"),
        "query_cache": os.environ.get("ACTION_HEAD_QUERY_CACHE"),
        "device": os.environ.get("ACTION_HEAD_DEVICE"),
        "visual_device": os.environ.get("ACTION_HEAD_VISUAL_DEVICE"),
        "seed": os.environ.get("ACTION_HEAD_SEED"),
    }
    settings.update({key: value for key, value in overrides.items() if value})
    required = (("gym_url", "slow_vllm_url", "slow_vllm_model")
                if (variant or action_head_variant()) == "semantic_planner"
                else ("checkpoint", "backbone", "gym_url"))
    if (variant or action_head_variant()) in ("v5_native", "v5_native_cached"):
        required += ("context_encoder_model", "context_aggregator_checkpoint")
    missing = [key for key in required
               if not settings.get(key)]
    if missing:
        raise RuntimeError(
            f"Action Head 配置缺少 {', '.join(missing)}（split={ACTION_HEAD_DATASET_SPLIT}）")
    for key in (key for key in ("checkpoint", "backbone") if key in required):
        if not pathlib.Path(settings[key]).exists():
            raise RuntimeError(f"Action Head {key} 路径不存在：{settings[key]}")
    return settings


def _action_head_gym_alive(url, timeout=1.0):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health",
                                    timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return bool(payload.get("ok") and payload.get("env_loaded"))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _ensure_action_head_gym(settings):
    """Lazily start the split-matched localhost gym when configured to do so."""
    global ACTION_HEAD_GYM_PROCESS
    url = str(settings["gym_url"])
    if _action_head_gym_alive(url):
        return
    if not settings.get("auto_start_gym"):
        raise RuntimeError(f"Action Head gym 不可用：{url}")
    port = int(settings.get("gym_port") or urlparse(url).port or 0)
    if not port:
        raise RuntimeError(f"Action Head gym 端口无效：{url}")
    log_path = pathlib.Path(REPO_ROOT) / "tmp" / f"action_head_gym_{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    gym_env = os.environ.copy()
    gym_env.update({
        "DEMO_HTTP_PORT": str(port),
        "DEMO_BACKEND": "sage",
        "DEMO_SPLIT": ACTION_HEAD_DATASET_SPLIT,
        "SAGE_SPLIT": ACTION_HEAD_DATASET_SPLIT,
        "SAGE_EPISODES_PER_SCENE": "0",
        "SAGE_EPS_PER_SCENE": "0",
        "SAGE_COLLISION": "usd_rooms",
        "SAGE_FURNITURE_COLLISION": "none",
        "SAGE_ROBOT_RADIUS": "0.07",
        "SAGE_CAMERA_HEIGHT": "1.2",
        "SAGE_HFOV": "115",
    })
    push_event("sys", text=f"FM Head 正在启动配套环境（{ACTION_HEAD_DATASET_SPLIT}）…")
    log_file = open(log_path, "ab", buffering=0)
    ACTION_HEAD_GYM_PROCESS = subprocess.Popen(
        ["bash", os.path.join(REPO_ROOT, "scripts/demo/vln_gym_server.sh")],
        cwd=REPO_ROOT, env=gym_env, stdin=subprocess.DEVNULL,
        stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if _action_head_gym_alive(url):
            return
        if ACTION_HEAD_GYM_PROCESS.poll() is not None:
            raise RuntimeError(
                f"Action Head gym 启动失败（日志：{log_path}）")
        time.sleep(0.5)
    raise RuntimeError(f"Action Head gym 启动超时（日志：{log_path}）")


def active_tool_profile(driver=None):
    """Select only the tools; the surrounding ReAct loop stays identical."""
    selected = driver or CFG.get("driver")
    if is_action_head_driver(selected):
        return A.get_tool_profile(
            "action_head",
            action_head_as_tool_version=CFG.get(
                "action_head_as_tool_version", "v1"))
    return A.get_tool_profile({
        "embodied_point": "embodied_point",
        "unilavira_bbox": "unilavira_bbox",
    }.get(selected, "primitive"))


def action_head_tool_version(driver=None):
    return (AHP.normalize_version(CFG.get("action_head_as_tool_version"))
            if is_action_head_driver(driver) else "v1")


def install_action_head_tool_executor(executor):
    """Install the production bridge without importing Torch into the web UI.

    ``executor`` must expose RunActionHeadExecutor.execute().  Keeping this an
    explicit dependency prevents the action_head profile from silently falling
    back to primitive motion when no checkpoint/feature provider is configured.
    """
    global ACTION_HEAD_TOOL_EXECUTOR
    ACTION_HEAD_TOOL_EXECUTOR = executor


def install_action_head_runtime(variant=None):
    """Load the separately deployed FM head from the runtime registry."""
    variant = variant or action_head_variant()
    if variant in ACTION_HEAD_RUNTIMES:
        return ACTION_HEAD_RUNTIMES[variant]
    settings = _action_head_settings(variant)
    if variant == "semantic_planner":
        from semantic_planner_driver import SemanticPlannerRuntime
        runtime = SemanticPlannerRuntime(
            gym_url=settings["gym_url"],
            vlm_url=settings["slow_vllm_url"],
            vlm_model=settings["slow_vllm_model"],
            artifact_dir=os.environ["SEMANTIC_PLANNER_ARTIFACT_DIR"],
            render_url=os.environ.get("SEMANTIC_PLANNER_RENDER_URL", "http://127.0.0.1:7038"),
            interrupted=interrupted, progress_callback=action_head_progress)
        ACTION_HEAD_RUNTIMES[variant] = runtime
        return runtime
    visible_devices = str(settings.get("visible_devices") or "").strip()
    if visible_devices:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", visible_devices)
    _ensure_action_head_gym(settings)
    import torch
    from action_head.agent_runtime import build_action_head_agent_runtime
    from action_head.policy import TrajectoryTransform

    stats_low = [-0.10273411870002747, -0.10309220850467682,
                 -0.5668500065803528]
    stats_high = [0.1031249612569809, 0.10309189558029175,
                  0.5747644901275635]
    transform = TrajectoryTransform.from_quantiles(
        stats_low, stats_high, flip_y=True, flip_yaw=True)
    runtime = build_action_head_agent_runtime(
        checkpoint=settings["checkpoint"],
        backbone_model=settings["backbone"],
        query_cache_root=settings.get("query_cache", "/dev/shm/action_head_queries"),
        gym_url=settings["gym_url"],
        device=settings.get("device", "cpu"),
        dtype=(torch.float32 if str(settings.get("device", "cpu")).startswith("cpu")
               else torch.bfloat16),
        visual_device=settings.get("visual_device", "cuda:0"),
        visual_dtype=torch.bfloat16,
        seed=int(settings.get("seed", 20260824)),
        trajectory_transform=transform,
        context_encoder_model=settings.get("context_encoder_model"),
        context_aggregator_checkpoint=settings.get("context_aggregator_checkpoint"),
        context_image_cache=bool(settings.get("context_image_cache", False)),
        interrupted=interrupted,
        progress_callback=action_head_progress,
    )
    ACTION_HEAD_RUNTIMES[variant] = runtime
    return runtime


class _ThreadBoundActionHeadExecutor:
    """Run every PPU call on the thread that loaded the FM Head.

    The PPU context is thread-affine on this deployment.  Loading on a short
    lived background thread and later forwarding on the Agent thread can end
    the whole process without a Python exception.
    """

    def __init__(self, target, owner_queue):
        self.target = target
        self.owner_queue = owner_queue

    def execute(self, **kwargs):
        return self._submit("execute", kwargs)

    def end_episode(self):
        if not hasattr(self.target, "end_episode"):
            return None
        return self._submit("end_episode", {})

    def _submit(self, method, kwargs):
        done = threading.Event()
        task = {"method": method, "kwargs": kwargs, "done": done}
        self.owner_queue.put(task)
        while not done.wait(0.1):
            pass
        if "error" in task:
            raise task["error"]
        return task["result"]


def _spawn_action_head_loader(variant):
    """Spawn the PPU-owner loader thread for one variant.

    Installs the runtime, builds its thread-bound executor, then loops forever
    on that thread as the sole owner of the variant's PPU context.  HTTP and
    Agent threads submit execute() calls through owner_queue, so multiple
    variants can coexist (one loader thread each) and a driver switch between
    cached variants is just an executor rebind.  The active runtime is only
    rebound when the *current* driver maps to this variant, so prewarming a
    variant while the user is on `react` does not steal the active slot.
    """
    def _run():
        global ACTION_HEAD_AGENT_RUNTIME, ACTION_HEAD_TOOL_EXECUTOR
        global ACTION_HEAD_ACTIVE_VARIANT, ACTION_HEAD_LOAD_STATE, ACTION_HEAD_LOAD_ERROR
        try:
            runtime = install_action_head_runtime(variant)
        except Exception as exc:
            with ACTION_HEAD_LOAD_LOCK:
                if ACTION_HEAD_ACTIVE_VARIANT == variant:
                    ACTION_HEAD_LOAD_STATE = "error"
                    ACTION_HEAD_LOAD_ERROR = str(exc)
            push_event("err", text=f"FM Head（{variant}）加载失败：{exc}")
            return
        owner_queue = queue.Queue()
        executor = _ThreadBoundActionHeadExecutor(runtime.tool_executor, owner_queue)
        with ACTION_HEAD_LOAD_LOCK:
            ACTION_HEAD_EXECUTORS[variant] = executor
            if is_action_head_driver() and action_head_variant() == variant:
                ACTION_HEAD_AGENT_RUNTIME = runtime
                ACTION_HEAD_TOOL_EXECUTOR = executor
                ACTION_HEAD_ACTIVE_VARIANT = variant
                ACTION_HEAD_LOAD_STATE = "loaded"
        push_event("sys", text=f"FM Head（{variant}）加载完成")

        # Keep the loader alive as the sole owner of the PPU context.  HTTP and
        # Agent threads submit work here and remain responsive for status/map
        # polling while inference and prefix execution proceed.
        while True:
            task = owner_queue.get()
            try:
                task["result"] = getattr(runtime.tool_executor, task["method"])(**task["kwargs"])
            except BaseException as exc:
                task["error"] = exc
            finally:
                task["done"].set()

    loader = threading.Thread(target=_run, name=f"action-head-loader-{variant}",
                              daemon=True)
    loader.start()
    return loader


def request_action_head_load():
    """Switch to this driver's variant instantly if cached, else lazy-load it."""
    global ACTION_HEAD_AGENT_RUNTIME, ACTION_HEAD_TOOL_EXECUTOR, ACTION_HEAD_ACTIVE_VARIANT
    global ACTION_HEAD_LOAD_STATE, ACTION_HEAD_LOAD_ERROR
    if CFG.get('driver') == 'action_head':
        # Selection is staged in the browser; only its confirm button loads.
        if MANUAL_HEAD_CONTROLLER and MANUAL_HEAD_CONTROLLER.active is not None:
            status = MANUAL_HEAD_CONTROLLER.status()
            publish_manual_head(MANUAL_HEAD_CONTROLLER.active, MANUAL_HEAD_CONTROLLER, status['active'])
        return
    variant = action_head_variant()
    with ACTION_HEAD_LOAD_LOCK:
        if variant in ACTION_HEAD_RUNTIMES and variant in ACTION_HEAD_EXECUTORS:
            ACTION_HEAD_AGENT_RUNTIME = ACTION_HEAD_RUNTIMES[variant]
            ACTION_HEAD_TOOL_EXECUTOR = ACTION_HEAD_EXECUTORS[variant]
            ACTION_HEAD_ACTIVE_VARIANT = variant
            ACTION_HEAD_LOAD_STATE = "loaded"
            push_event("sys", text=f"FM Head（{variant}）已加载")
            return
        if ACTION_HEAD_LOAD_STATE == "loading" and ACTION_HEAD_ACTIVE_VARIANT == variant:
            push_event("sys", text="FM Head 加载中…")
            return
        ACTION_HEAD_ACTIVE_VARIANT = variant
        ACTION_HEAD_AGENT_RUNTIME = None
        ACTION_HEAD_TOOL_EXECUTOR = None
        ACTION_HEAD_LOAD_STATE = "loading"
        ACTION_HEAD_LOAD_ERROR = ""
    push_event("sys", text=f"FM Head（{variant}）加载中…")
    _spawn_action_head_loader(variant)


def prewarm_action_head_runtimes():
    """Warm independent encoders when configured, without selecting a checkpoint."""
    if os.environ.get('ACTION_HEAD_PREWARM_ENCODERS', '1') == '0':
        return
    try:
        settings = manual_head_settings(dict(visual_hidden_size=2560, checkpoint=''))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[action-head] 跳过自动预热：{exc}", flush=True)
        return
    ensure_manual_head_controller().prewarm(settings)

# Does an assistant turn carry its <think> block into history? See
# A.assistant_history_content for the two modes and A.THINKING_HISTORY_MODES.
# Upstream (agentic) defaults to strip_thinking; this demo defaults to keep
# because navigation reasoning is cumulative -- the model spends a turn working
# out "the doorway is on my left, so turn before advancing" and under strip that
# conclusion is gone by the next turn, leaving only the tool call as evidence it
# ever happened.
#
# The cost is paid in context, and it is not small: reasoning is by far the
# longest part of a reply in think mode. Both are survivable here because the
# window is 131072 tokens and images are already capped at KEEP_LAST_IMAGES, but
# unlike the image budget this one grows monotonically with turn count -- so the
# per-turn [agent] line prints the server's own prompt_tokens, and blowing the
# window shows up there before it shows up as a failed request.
# RYNNVLN_THINKING_HISTORY=strip_thinking switches back for an A/B.
THINKING_HISTORY = os.environ.get("RYNNVLN_THINKING_HISTORY", "keep_thinking")
if THINKING_HISTORY not in A.THINKING_HISTORY_MODES:
    raise SystemExit(f"RYNNVLN_THINKING_HISTORY 必须是 "
                     f"{'/'.join(A.THINKING_HISTORY_MODES)}，收到 {THINKING_HISTORY!r}")

# Token accounting from the last stream_chat, for the per-turn [agent] line.
# A module global rather than a return value so run_model's call site stays
# untouched; safe because the sim runs on a single worker thread, one request at
# a time -- the same reason the actuation swap needs no lock.
LAST_USAGE = {}
# Correlation id for the four-query local side channel. It is runtime-only and
# never appended to the Agent conversation.
LAST_QUERY_REQUEST_ID = ""
LAST_INFERENCE_TIMING = {}
# Captured once from the yaml at startup. check_actuation() compares against
# this, never against the live values -- reading those back would happily adopt
# a leaked step size as the new baseline.
BASE_AMOUNTS = {}

LOCK = threading.Lock()
# Set on the way out so the /stream.mjpg threads stop waiting on FRAME_CV and
# let the process go, rather than leaning on daemon-thread teardown.
SHUTDOWN = threading.Event()
# Signalled whenever a new frame is published, so /stream.mjpg can
# push it instead of waiting for the next status poll. Shares LOCK -- one mutex
# for all shared state keeps the ordering trivial.
FRAME_CV = threading.Condition(LOCK)
# Timestamps of the last frames, for the fps read-out. Length only bounds how
# long the average looks back; the status handler ignores samples older than a
# second so a paused sim reads 0 rather than "whatever it did before it paused".
FRAME_TS = deque(maxlen=40)
STATE = {
    # frame_jpg is always the clean model/tool observation. display_frame_jpg
    # may contain a world-anchored navigation marker and is only streamed to
    # the human-facing first-person pane.
    "frame_jpg": None, "display_base_jpg": None,
    "display_frame_jpg": None, "frame_size": [0, 0],
    "evaluation": {}, "map_jpg": None, "tool_map_jpg": None, "map_heading_deg": 0.0,
    "seq": 0, "display_seq": 0, "map_seq": 0,
    "instruction": "", "status": "idle", "step": 0, "dist": float("nan"),
    "result": "", "scene": "", "episode_id": "",
    "actions": [], "stream": {"active": False, "text": "", "reasoning": "",
                               "n": 0, "dt": 0.0},
    # Measured relative pose for the active run. ``dist``/``bearing`` are polar
    # displacement from its start pose; ``yaw`` is final heading change.
    "motion": {"dist": 0.0, "bearing": 0.0, "yaw": 0.0, "steps": 0},
    "moves": [],
    "driver": os.environ.get("DEMO_DEFAULT_DRIVER", P.DEFAULT_DRIVER),
    "action_head_as_tool_version": "v1",
    "model": P.DEFAULT_MODEL,
    "react_reference": {"active": False, "kind": "", "name": "",
                        "point": None, "bbox": None, "version": 0},
    # which asset set is driving; the page shows the label in a header pill so
    # an MP3D id and a 3DRS id are never confused for one another
    "backend": "", "backend_label": "", "scene_title": "", "scene_name": "",
    # manual driving is live: the page then reads the continuous trail
    # (moves/motion) rather than the legacy 0.25 m-per-entry action list, which
    # cannot express a hold of arbitrary length.
    "manual": False,
    # 前端锁：一个后台 agent 在调试这台控制台时，人从浏览器点一下就能把它的场景/
    # 参数改掉，而 agent 完全看不见。锁上之后拒绝普通网页发来的 /cmd、参考图和
    # /sysprompt；pi-agent 工具请求携带内部来源标记，仍可执行移动。状态在服务端，
    # 不在浏览器：这样它跨标签页、跨刷新一致，而且一个没刷新的旧页面也绕不过去。
    # DEMO_LOCK=1 让控制台一起来就是锁着的。
    "locked": os.environ.get("DEMO_LOCK", "") == "1",
}
# Lossless source for the published first-person frame.  Web streaming uses the
# JPEG in STATE, but bbox annotation must be drawn on the exact clean RGB that
# produced it rather than on a JPEG decode.  Its source token is updated in the
# same critical section as STATE["frame_jpg"] / STATE["seq"], so an image and
# its identity can never be observed from different simulator steps.
CURRENT_FRAME = {"image": None, "depth": None,
                 "scene": "", "episode_id": "", "seq": 0}
# Temporary 3D visual-tool target. It is projected afresh for every rendered
# sub-step and never written into CURRENT_FRAME or model observations.
ACTIVE_VISUAL_TARGET = {"marker": None}
# LightNav may return two independent image points: APOS (the immediate
# action/affordance location) and OPOS (the object location).  Depth-backed
# backends turn them into world anchors and refresh() reprojects them on every
# browser frame.  This display-only state never enters CURRENT_FRAME.
ACTIVE_LIGHTNAV_POINTING = {"markers": [], "trajectory": None}
# A rejected screen-space point remains visible briefly so both a person and
# the model operator can see exactly what failed validation. It belongs only
# to the browser display layer: canonical observations and saved images stay
# clean. ``version`` prevents an older timer from clearing a newer marker.
TRANSIENT_SCREEN_MARKER = {
    "point": None, "expires_at": 0.0, "version": 0,
}
# Browser-only replay generation. Canonical state jumps straight to the final
# observation while Gaussian-rendered in-between turn frames arrive here later.
DISPLAY_ANIMATION = {"version": 0}
# 拒绝时回给前端的原话；前端把它原样弹出来，所以措辞只有这一处。
LOCKED_MSG = "后台 Agent 正在调试，只允许 Agent 自主操作，已禁止用户前端任何交互"
AGENT_CONTROL_HEADER = "X-VLN-Control-Source"
AGENT_CONTROL_VALUE = "pi-agent"
CMD = {"start": False, "stop": False, "switch": None, "manual": None,
       "manual_start": False, "reset": False, "clearlog": False, "goto": None,
       "walk_golden": False, "point_move": None}

# Optional first-turn visual context for the ReAct driver. The JSON-safe
# metadata lives in STATE so every browser tab sees the same selection; the PIL
# image stays here because /status must never try to serialise megabytes of RGB.
REACT_REFERENCE = {"image": None}
MAX_REFERENCE_POST_BYTES = 12 * 1024 * 1024
MAX_REFERENCE_IMAGE_BYTES = 8 * 1024 * 1024
MAX_REFERENCE_PIXELS = 16_000_000

# ---- hold-to-drive ---------------------------------------------------------
# Manual driving is continuous: how far the robot goes is how long you hold the
# key, not how many times you tapped it. The page therefore reports the *set* of
# keys currently down and keeps re-reporting it (every HOLD_BEAT ms, see
# web/js/panels/manual.js) instead of sending one event per press. Two reasons
# for the heartbeat rather than plain keydown/keyup:
#   * a keyup lost to alt-tab, a closed lid or a dropped wifi link would
#     otherwise leave the robot walking until it hit a wall,
#   * it makes the command idempotent, so a retried or reordered request cannot
#     turn into extra motion.
# HOLD["t"] is when the last report landed; anything older than HOLD_TIMEOUT is
# read as "released".
HOLD = {"keys": frozenset(), "t": 0.0}
HOLD_TIMEOUT = float(os.environ.get("DEMO_HOLD_TIMEOUT", "0.6"))
# Speeds, not per-tick quanta: the per-tick step is speed x elapsed time, so the
# robot covers the same ground per second whether it is running on ABotN's
# Gaussians at ~12 fps or on habitat's software GL at ~3. 1 m/s and 45 deg/s --
# a brisk walk, which is fast enough not to feel like wading across a plaza and
# still slow enough to steer a 2 m corridor.
HOLD_SPEED_MPS = float(os.environ.get("DEMO_HOLD_SPEED", "1.0"))
HOLD_TURN_DPS = float(os.environ.get("DEMO_HOLD_TURN", "45"))
# A tick that took longer than this was a hiccup (first render of a scene, a
# GC pause, the worker busy elsewhere), not time spent holding the key. Without
# the clamp a two-second stall would teleport the robot 2 m on resume.
HOLD_MAX_DT = 0.30
# Smallest move one tick will issue, and the floor handed to drive_forward in
# place of its own MIN_FWD_M. It has to sit below one tick's worth of travel on
# the *fastest* backend or it, not env.step, becomes the frame-rate ceiling:
# habitat renders a tick in ~35 ms, which at 1 m/s is 0.035 m, so MIN_FWD_M's
# 0.05 would have held habitat to 12 fps when it can do 28. Undershooting
# distance is not a risk -- whatever a tick does not spend is carried to the next
# one, so the speed is right either way; only the frame rate changes.
HOLD_MIN_M = float(os.environ.get("DEMO_HOLD_MIN_M", "0.02"))
# temp=None means "use the mode's preset temperature"; the UI sends a number
# only when the user actually moves the slider off auto.
CFG = {"tpl": P.DEFAULT_TEMPLATE,
       # action expert data construction uses enable_thinking=False.  Keep the
       # generic console default unchanged, while allowing its dedicated
       # launcher to select instruct and avoid an accidental long <think> turn.
       "mode": os.environ.get("DEMO_DEFAULT_MODE", P.DEFAULT_MODE), "nact": 4,
       "temp": None, "frompose": True, "instruction": "",
       "driver": os.environ.get("DEMO_DEFAULT_DRIVER", P.DEFAULT_DRIVER),
       # Explicit hyperparameter. It is intentionally not read from an
       # environment variable so one visible argument controls every v2 path.
       "action_head_as_tool_version": "v1",
       "model": P.DEFAULT_MODEL,
       "maxtools": int(os.environ.get(
           "DEMO_MAX_TOOL_CALLS", str(P.DEFAULT_MAX_TOOL_CALLS))),
       # 走专家轨迹的弧长 resample 步长（米）。默认读 env SAGE_GOLDEN_STEP_M，
       # 页面「专家步长」控件可调（set_golden_step，0.02~1.0m）。
       "golden_step": float(os.environ.get("SAGE_GOLDEN_STEP_M", "0.1") or "0.1"),
       "golden_stride": 4,
       # Point-Goal 前瞻距离（米）：专家 path 上前方 waypoint 选点弧长，sage 全
       # 室内默认 3m。env SAGE_POINTGOAL_LOOKAHEAD_M / 页面「前瞻距离」可调。
       "pointgoal_lookahead": float(os.environ.get("SAGE_POINTGOAL_LOOKAHEAD_M", "3") or "3"),
       "lightnav_url": os.environ.get("LIGHTNAV_SERVER_URL", LND.DEFAULT_URL),
       # "" = use prompts.agent_system_prompt()'s built-in text. Anything else is
       # a prompt hand-edited in the 提示词 panel and is sent verbatim.
       "sysprompt": ""}

# A hand-edited system prompt outlives the console it was typed into: editing it
# is a minutes-long job and restarting to load a new asset set is a routine one,
# so keeping it only in memory would lose the work at the least convenient time.
# Outside the repo on purpose -- it is per-machine scratch, not something to
# commit or to diff against the default.
SYSPROMPT_FILE = os.environ.get(
    "DEMO_SYSPROMPT_FILE",
    os.path.join(os.path.expanduser("~"), ".cache", "vln_demo",
                 "system_prompt.txt"))
# Only a runaway-paste guard. The real ceiling is the model's context window,
# which a prompt this long would eat on its own.
SYSPROMPT_MAX_CHARS = 100000

# The web terminal is intentionally just one shared PTY. It starts in the
# SpatialOnboarding workspace and runs ``pi`` as soon as the shell is ready.
# The only extra operation exposed by the page is restarting the PTY, which
# opens a fresh shell and runs ``pi`` again.
TERMINAL_CWD = os.environ.get(
    "DEMO_TERMINAL_CWD",
    "/opt/external/user-d5121be6/SpatialOnboarding/workspace")
TERMINAL_MAX_BUFFER = 4 * 1024 * 1024
TERMINAL_MAX_INPUT = 64 * 1024


class WebTerminal:
    """One browser-visible bash PTY with byte-exact output and resize support."""

    def __init__(self, cwd):
        self.configured_cwd = os.path.abspath(cwd)
        self.workspace_reset_error = None
        self.cwd = cwd if os.path.isdir(cwd) else os.getcwd()
        self.cv = threading.Condition()
        self.restart_lock = threading.Lock()
        self.proc = None
        self.master = None
        self.generation = 0
        self.base = 0
        self.output = bytearray()
        self.exit_code = None
        self.started = False

    def _append_locked(self, data):
        if not data:
            return
        self.output.extend(data)
        extra = len(self.output) - TERMINAL_MAX_BUFFER
        if extra > 0:
            del self.output[:extra]
            self.base += extra
        self.cv.notify_all()

    def _reader(self, master, proc, generation):
        while True:
            try:
                ready, _, _ = select.select([master], [], [], 0.5)
                if not ready:
                    if proc.poll() is not None:
                        break
                    continue
                data = os.read(master, 65536)
                if not data:
                    break
            except (OSError, ValueError):
                break
            with self.cv:
                if generation != self.generation:
                    return
                self._append_locked(data)
        try:
            code = proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            code = proc.poll()
        with self.cv:
            if generation != self.generation:
                return
            self.exit_code = code
            self._append_locked(
                f"\r\n[terminal exited: {code}] 点击“重启终端”重新打开。\r\n".encode())

    @staticmethod
    def _winsize(fd, cols, rows):
        cols = max(20, min(500, int(cols)))
        rows = max(5, min(300, int(rows)))
        fcntl.ioctl(fd, termios.TIOCSWINSZ,
                    struct.pack("HHHH", rows, cols, 0, 0))

    def _terminate(self, proc, master):
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        if master is not None:
            try:
                os.close(master)
            except OSError:
                pass

    def restart(self, cols=100, rows=30):
        with self.restart_lock:
            return self._restart(cols, rows)

    def reset_workspace(self, generation, cols=100, rows=30):
        from terminal_workspace import rotate_workspace, terminate_tree
        with self.restart_lock:
            if not isinstance(generation, int) or generation != self.generation:
                raise ValueError('终端已发生变化，请等待刷新后再重试')
            if self.workspace_reset_error:
                raise ValueError(self.workspace_reset_error)
            if os.path.abspath(self.cwd) != self.configured_cwd:
                raise ValueError('终端没有使用指定工作区，已取消清空')

            def stop():
                with self.cv:
                    proc, master = self.proc, self.master
                    self.proc = self.master = None
                    self.started = True  # polls must not reopen a workspace mid-rotation
                    self.generation += 1
                    self.cv.notify_all()
                try:
                    terminate_tree(proc)
                except Exception as exc:
                    self.workspace_reset_error = f'上次终端停止未完成，需检查残留进程后重试：{exc}'
                    raise
                finally:
                    if master is not None:
                        os.close(master)

            receipt = rotate_workspace(
                self.cwd,
                os.path.join(REPO_ROOT, 'third_party/pi-agent/.pi/workspaces/SpatialOnboarding/AGENTS.md'),
                os.path.join(REPO_ROOT, 'vln_demo/config/pi/SpatialOnboarding'),
                os.path.join(os.path.dirname(self.configured_cwd), 'workspace-backups'), stop)
            try:
                result = self._restart(cols, rows)
            except Exception as exc:
                raise OSError(f'工作区已重置，备份在 {receipt["backup"]}；终端启动失败：{exc}') from exc
            result['workspace_reset'] = receipt
            return result

    def _restart(self, cols, rows):
        with self.cv:
            old_proc, old_master = self.proc, self.master
            self.proc = self.master = None
            self.generation += 1
            generation = self.generation
            self.base = 0
            self.output.clear()
            self.exit_code = None
            self.started = True
            self.cv.notify_all()
        self._terminate(old_proc, old_master)

        master, slave = os.openpty()
        self._winsize(slave, cols, rows)
        env = os.environ.copy()
        env.update({
            "TERM": "xterm-256color",
            "COLORTERM": "truecolor",
            # --norc keeps startup quiet, so provide the useful part of a
            # normal shell prompt explicitly: user, host and current directory.
            "PS1": r"\u@\h:\w\$ ",
        })
        # Do not source the DSW login/profile scripts here: they print a large
        # mount banner and can consume keystrokes typed while they are still
        # starting. The console process already supplies the environment and
        # PATH (including /usr/local/bin/pi), so a clean interactive bash is
        # both faster and closer to an embedded editor terminal.
        command = ["/usr/bin/setsid", "--wait", "--ctty", "/bin/bash",
                   "--noprofile", "--norc", "-i"]
        try:
            proc = subprocess.Popen(
                command, cwd=self.cwd, env=env,
                stdin=slave, stdout=slave, stderr=slave, close_fds=True)
        except Exception:
            os.close(master)
            os.close(slave)
            raise
        finally:
            try:
                os.close(slave)
            except OSError:
                pass
        with self.cv:
            stale = generation != self.generation
            if not stale:
                self.proc, self.master = proc, master
                self.cv.notify_all()
        if stale:
            self._terminate(proc, master)
            return self.snapshot(0, 0, 0)
        threading.Thread(target=self._reader,
                         args=(master, proc, generation), daemon=True).start()
        # Bash changes terminal modes while it initializes, which may flush
        # keystrokes sent in the tiny window before its first prompt. Do not
        # report the PTY as ready to the browser until it has produced output.
        with self.cv:
            deadline = time.monotonic() + 1
            while (generation == self.generation and not self.output
                   and proc.poll() is None and time.monotonic() < deadline):
                self.cv.wait(deadline - time.monotonic())
            output_before_command = len(self.output)
        # Start the agent immediately. A terminal restart follows this same
        # path, so it also launches a fresh pi process without another click.
        self.write(b"pi\r")
        # Usually the PTY echoes immediately. Waiting briefly makes the
        # command part of the restart response as well as a later
        # long-poll update, so the UI never appears blank after restart.
        with self.cv:
            deadline = time.monotonic() + 0.25
            while (generation == self.generation
                   and len(self.output) == output_before_command
                   and proc.poll() is None and time.monotonic() < deadline):
                self.cv.wait(deadline - time.monotonic())
        return self.snapshot(0, 0, 0)

    def ensure(self):
        with self.cv:
            should_start = not self.started
            if should_start:
                # Reserve the initial start before releasing the condition, so
                # two browser tabs opened together cannot each replace the PTY.
                self.started = True
        if should_start:
            self.restart()

    def snapshot(self, since, generation, wait=15):
        self.ensure()
        deadline = time.monotonic() + max(0, min(float(wait), 20))
        with self.cv:
            while (generation == self.generation
                   and self.base <= since <= self.base + len(self.output)
                   and since == self.base + len(self.output)
                   and self.proc is not None and self.proc.poll() is None):
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self.cv.wait(left)
            end = self.base + len(self.output)
            reset = (generation != self.generation
                     or since < self.base or since > end)
            start = self.base if reset else since
            data = bytes(self.output[start - self.base:])
            running = self.proc is not None and self.proc.poll() is None
            return {"ok": True, "generation": self.generation,
                    "cursor": end, "reset": reset,
                    "output": base64.b64encode(data).decode("ascii"),
                    "running": running, "exit_code": self.exit_code,
                    "cwd": self.cwd, "workspace_reset_supported": True}

    def write(self, data):
        if not isinstance(data, (bytes, bytearray)) or not data:
            return
        if len(data) > TERMINAL_MAX_INPUT:
            raise ValueError("terminal input too large")
        with self.cv:
            master = self.master
            running = self.proc is not None and self.proc.poll() is None
        if master is None or not running:
            raise ValueError("terminal is not running")
        view = memoryview(data)
        while view:
            written = os.write(master, view)
            view = view[written:]

    def resize(self, cols, rows):
        with self.cv:
            master = self.master
        if master is not None:
            self._winsize(master, cols, rows)

    def close(self):
        with self.cv:
            proc, master = self.proc, self.master
            self.proc = self.master = None
            self.generation += 1
            self.cv.notify_all()
        self._terminate(proc, master)


TERMINAL = WebTerminal(TERMINAL_CWD)


def load_sysprompt():
    """Restore a previously saved custom system prompt into CFG."""
    try:
        with open(SYSPROMPT_FILE, encoding="utf-8") as fh:
            text = fh.read().strip()
    except OSError:
        return ""
    CFG["sysprompt"] = text[:SYSPROMPT_MAX_CHARS]
    return CFG["sysprompt"]


def save_sysprompt(text):
    """Persist (or clear, when `text` is empty) the custom system prompt.

    Returns the stored value. Write failures are reported to the caller rather
    than raised: a read-only home should cost you persistence, not the edit you
    just made -- CFG is already updated by then and the run will use it.
    """
    text = (text or "").strip()[:SYSPROMPT_MAX_CHARS]
    CFG["sysprompt"] = text
    try:
        os.makedirs(os.path.dirname(SYSPROMPT_FILE), exist_ok=True)
        if text:
            with open(SYSPROMPT_FILE, "w", encoding="utf-8") as fh:
                fh.write(text)
        elif os.path.exists(SYSPROMPT_FILE):
            os.remove(SYSPROMPT_FILE)
        return text, ""
    except OSError as exc:
        return text, str(exc)


EVENTS = deque(maxlen=400)
EV_NEXT = 0
EV_MIN = 0
# Model-input snapshots are kept out of /status: prompts and image bytes would
# make the 250 ms poll enormous. The event log carries only an id; the UI fetches
# the first full input or a later turn's incremental input through its endpoint.
DEBUG_REQUESTS = {}
DEBUG_REQUEST_ORDER = deque()
DEBUG_REQUEST_NEXT = 0
DEBUG_REQUEST_LIMIT = 128
DEBUG_IMAGES = {}
DEBUG_IMAGE_ORDER = deque()
DEBUG_IMAGE_OBJECTS = {}
DEBUG_IMAGE_NEXT = 0
DEBUG_IMAGE_LIMIT = 128
FOG_DIM = float(os.environ.get("DEMO_FOG_DIM", "0.82"))


# ----------------------------------------------------------------------------
# state helpers
# ----------------------------------------------------------------------------
def set_state(**kw):
    with LOCK:
        STATE.update(kw)


def push_event(t, **kw):
    global EV_NEXT, EV_MIN
    with LOCK:
        kw["t"] = t
        kw["i"] = EV_NEXT
        EV_NEXT += 1
        EVENTS.append(kw)
        if EVENTS[0]["i"] > EV_MIN:
            EV_MIN = EVENTS[0]["i"]


def clear_events():
    global EV_MIN
    with LOCK:
        EVENTS.clear()
        EV_MIN = EV_NEXT
        DEBUG_REQUESTS.clear()
        DEBUG_REQUEST_ORDER.clear()
        DEBUG_IMAGES.clear()
        DEBUG_IMAGE_ORDER.clear()
        DEBUG_IMAGE_OBJECTS.clear()


def react_reference_snapshot():
    """Return a detached ``(image, metadata)`` pair for one ReAct run."""
    with LOCK:
        image = REACT_REFERENCE["image"]
        return (image.copy() if image is not None else None,
                dict(STATE["react_reference"]))


def clear_react_reference(only_kind=None):
    """Clear the shared reference, optionally only when it is one kind."""
    with LOCK:
        current = STATE["react_reference"]
        if only_kind and current.get("kind") != only_kind:
            return False
        version = int(current.get("version") or 0) + 1
        REACT_REFERENCE["image"] = None
        STATE["react_reference"] = {
            "active": False, "kind": "", "name": "", "point": None,
            "bbox": None,
            "version": version,
        }
    return True


def _store_react_reference(image, kind, name="", point=None, bbox=None,
                           source=None):
    with LOCK:
        version = int(STATE["react_reference"].get("version") or 0) + 1
        meta = {
            "active": True,
            "kind": kind,
            "name": name,
            "point": point,
            "bbox": bbox,
            "version": version,
        }
        if source:
            meta.update(source)
        REACT_REFERENCE["image"] = image.copy()
        STATE["react_reference"] = meta
        return dict(meta)


def _uploaded_reference(value):
    if not isinstance(value, str) or not value:
        raise ValueError("缺少 image_data")
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("图片不是有效的 base64") from exc
    if not raw or len(raw) > MAX_REFERENCE_IMAGE_BYTES:
        raise ValueError("图片必须大于 0 且不超过 8 MB")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.width * opened.height > MAX_REFERENCE_PIXELS:
                raise ValueError("图片像素数不能超过 1600 万")
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("无法解码图片，仅支持 Pillow 可读取的图片格式") from exc
    return image


def set_react_reference(payload):
    """Apply one UI/API reference request and return JSON-safe metadata.

    ``upload`` keeps an arbitrary image as an extra first-turn reference.
    ``point`` and ``bbox`` use 0-to-1000 normalized coordinates and draw on a
    snapshot of the current simulator frame. ``clear`` removes any kind.
    """
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON object")
    kind = str(payload.get("kind") or "").strip().lower()
    if kind == "clear":
        clear_react_reference()
        with LOCK:
            return dict(STATE["react_reference"])
    if kind == "upload":
        image = _uploaded_reference(payload.get("image_data"))
        name = os.path.basename(str(payload.get("name") or "参考图"))[:120]
        return _store_react_reference(image, "upload", name=name)
    if kind not in ("point", "bbox"):
        raise ValueError("kind 必须是 upload、point、bbox 或 clear")

    field = "point" if kind == "point" else "bbox"
    expected = 2 if kind == "point" else 4
    raw_coords = payload.get(field)
    if not isinstance(raw_coords, (list, tuple)) or len(raw_coords) != expected:
        shape = "[x,y]" if kind == "point" else "[x1,y1,x2,y2]"
        raise ValueError(f"{field} 必须是 {shape}")
    try:
        coords = [float(v) for v in raw_coords]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 坐标必须是数字") from exc
    valid = all(math.isfinite(v) and v.is_integer() and 0 <= v <= 1000
                for v in coords)
    if kind == "bbox" and valid:
        valid = coords[0] < coords[2] and coords[1] < coords[3]
    if not valid and kind == "point":
        raise ValueError("point 必须是两个 0 到 1000 的归一化整数 [x,y]")
    if not valid:
        raise ValueError("bbox 必须是整数并满足 0≤x1<x2≤1000 且 0≤y1<y2≤1000")
    coords = [int(v) for v in coords]
    with LOCK:
        frame = CURRENT_FRAME["image"]
        image = frame.copy() if frame is not None else None
        depth = CURRENT_FRAME["depth"]
        depth = depth.copy() if depth is not None else None
        source = {key: CURRENT_FRAME[key]
                  for key in ("scene", "episode_id", "seq")}
    if image is None:
        raise ValueError("当前还没有可标注的第一视角画面")

    # A drawn point/bbox carries the atomic snapshot token returned by
    # /frame-snapshot.  Rejecting a stale token is essential: otherwise a box
    # drawn on frame A is silently painted at the same coordinates on frame B
    # if the simulator moves before the user presses Start.  API callers that
    # type coordinates without opening the picker may omit the token and target
    # the current frame, preserving the existing API.
    requested = payload.get("source")
    if requested is not None:
        if not isinstance(requested, dict):
            raise ValueError(f"{field} source 必须是 frame snapshot object")
        try:
            requested_seq = int(requested.get("seq"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} source 缺少有效 seq") from exc
        same_source = (
            requested_seq == source["seq"]
            and str(requested.get("scene", "")) == str(source["scene"])
            and str(requested.get("episode_id", "")) == str(source["episode_id"])
        )
        if not same_source:
            raise ValueError("标记所依据的画面已经变化，请重新观察后标记")
    if kind == "point":
        if depth is None:
            raise ValueError("point 缺少同帧 metric depth")
        try:
            VNT.point_sample(
                {"rgb": np.asarray(image), "depth": depth}, coords)
        except (TypeError, ValueError):
            show_invalid_point(coords, seconds=2.0)
            raise
        # Never attach a point sampled from a frame that moved concurrently.
        with LOCK:
            current_source = {key: CURRENT_FRAME[key]
                              for key in ("scene", "episode_id", "seq")}
        if current_source != source:
            raise ValueError("标记采样期间画面已经变化，请重新观察后选点")
    w, h = image.size
    draw = ImageDraw.Draw(image)
    width = max(3, min(w, h) // 100)
    if kind == "point":
        u = round(coords[0] * (w - 1) / 1000)
        v = round(coords[1] * (h - 1) / 1000)
        radius = max(8, min(w, h) // 40)
        draw.ellipse((u-radius, v-radius, u+radius, v+radius),
                     outline=(255, 32, 32), width=width)
        draw.line((u-radius, v, u+radius, v), fill=(255, 32, 32), width=width)
        draw.line((u, v-radius, u, v+radius), fill=(255, 32, 32), width=width)
        return _store_react_reference(image, "point", name="归一化视觉点",
                                      point=coords, source=source)
    xy = [round(coords[0] * (w - 1) / 1000),
          round(coords[1] * (h - 1) / 1000),
          round(coords[2] * (w - 1) / 1000),
          round(coords[3] * (h - 1) / 1000)]
    draw.rectangle(xy, outline=(255, 0, 0), width=width)
    return _store_react_reference(image, "bbox", name="归一化 bbox",
                                  bbox=coords, source=source)


def react_reference_jpeg():
    image, _ = react_reference_snapshot()
    return jpg_bytes(image, q=92) if image is not None else None


def frame_snapshot_payload():
    """Return one atomic, JSON-safe picker snapshot and its source token."""
    with LOCK:
        data = STATE["frame_jpg"]
        size = list(STATE["frame_size"])
        source = {key: CURRENT_FRAME[key]
                  for key in ("scene", "episode_id", "seq")}
    if data is None:
        raise ValueError("当前还没有第一视角画面")
    return {
        "ok": True,
        "image_data": "data:image/jpeg;base64," + base64.b64encode(data).decode(),
        "width": size[0],
        "height": size[1],
        "source": source,
    }


def map_snapshot_payload():
    """Return one atomic tool-map image with matching compass metadata."""
    with LOCK:
        data = STATE["tool_map_jpg"]
        heading = float(STATE["map_heading_deg"])
        scene_name = STATE["scene_name"] or STATE["scene_title"] or STATE["scene"]
        map_seq = STATE["map_seq"]
    if data is None:
        raise ValueError("当前还没有俯视地图")
    return {
        "ok": True,
        "image_data": "data:image/jpeg;base64," + base64.b64encode(data).decode(),
        "scene_name": scene_name,
        "map_heading_deg": heading,
        "map_seq": map_seq,
    }


def _draw_invalid_screen_point(image, point):
    """Draw a conspicuous rejected-point glyph on a display-only image."""
    image = image.convert("RGB").copy()
    w, h = image.size
    u = round(float(point[0]) * (w - 1) / 1000.0)
    v = round(float(point[1]) * (h - 1) / 1000.0)
    radius = max(10, min(w, h) // 28)
    width = max(4, min(w, h) // 80)
    draw = ImageDraw.Draw(image)
    # A dark halo keeps the red X legible on bright floors and red furniture.
    halo = width + 4
    draw.ellipse((u-radius, v-radius, u+radius, v+radius),
                 outline=(20, 20, 20), width=halo)
    draw.line((u-radius, v-radius, u+radius, v+radius),
              fill=(20, 20, 20), width=halo)
    draw.line((u-radius, v+radius, u+radius, v-radius),
              fill=(20, 20, 20), width=halo)
    draw.ellipse((u-radius, v-radius, u+radius, v+radius),
                 outline=(255, 35, 35), width=width)
    draw.line((u-radius, v-radius, u+radius, v+radius),
              fill=(255, 35, 35), width=width)
    draw.line((u-radius, v+radius, u+radius, v-radius),
              fill=(255, 35, 35), width=width)
    return image


def _display_with_transient_marker(base_jpg, point):
    if base_jpg is None or point is None:
        return base_jpg
    with Image.open(io.BytesIO(base_jpg)) as image:
        return jpg_bytes(_draw_invalid_screen_point(image, point))


def _clear_invalid_point(version):
    """Timer callback; only the timer belonging to the latest marker wins."""
    with FRAME_CV:
        if TRANSIENT_SCREEN_MARKER["version"] != version:
            return
        TRANSIENT_SCREEN_MARKER["version"] = version + 1
        TRANSIENT_SCREEN_MARKER["point"] = None
        TRANSIENT_SCREEN_MARKER["expires_at"] = 0.0
        STATE["display_frame_jpg"] = (STATE["display_base_jpg"]
                                      or STATE["frame_jpg"])
        STATE["display_seq"] += 1
        FRAME_CV.notify_all()


def show_invalid_point(point, seconds=2.0):
    """Immediately show a rejected normalized point in the browser stream."""
    coords = VNT.parse_point_2d(point)
    with FRAME_CV:
        version = int(TRANSIENT_SCREEN_MARKER["version"]) + 1
        TRANSIENT_SCREEN_MARKER.update(
            point=list(coords), expires_at=time.monotonic() + seconds,
            version=version)
        base = STATE["display_base_jpg"] or STATE["frame_jpg"]
        STATE["display_frame_jpg"] = _display_with_transient_marker(base, coords)
        STATE["display_seq"] += 1
        FRAME_CV.notify_all()
    timer = threading.Timer(max(0.0, float(seconds)),
                            _clear_invalid_point, args=(version,))
    timer.daemon = True
    timer.start()


def _async_turn_render_spec(env, direction, degrees):
    """Return immutable GS render inputs, or None for non-Gaussian backends."""
    sim = getattr(env, "sim", None)
    client = getattr(sim, "client", None)
    if client is None or not hasattr(client, "render_one") or not hasattr(sim, "_yaw"):
        return None
    state = sim.get_agent_state()
    eye = np.asarray(state.position, dtype=float).copy()
    scene = getattr(sim, "scene", None)
    if hasattr(sim, "camera_z"):
        # ABotN/SAGE expose Habitat-shaped (x, z, -y) agent states, but
        # observe() sends native z-up (x, y, z) to their Euler renderer.
        # The state already includes ground + camera height.
        eye = np.array([eye[0], -eye[2], eye[1]])
    elif scene is not None:
        # RS3DSim stores the agent foot position; its observe() adds eye height.
        eye[1] += float(getattr(sim, "sensor_height", 0.0))
    else:
        # MallSim's AgentState already stores CAMERA_HEIGHT.
        scene = getattr(env, "scene", None)
    if not scene:
        return None
    sign = 1.0 if direction == "left" else -1.0
    return {
        "sim": sim, "client": client, "scene": scene, "eye": eye,
        "start_yaw": float(sim._yaw), "delta": sign * math.radians(degrees),
    }


def replay_turn_async(spec, degrees):
    """Render/display intermediate GS views without delaying the next LLM turn."""
    if spec is None:
        return None
    count = max(1, math.ceil(float(degrees) / A.TURN_RENDER_STEP_DEG - 1e-9))
    if count <= 1:
        return None
    with FRAME_CV:
        DISPLAY_ANIMATION["version"] += 1
        version = DISPLAY_ANIMATION["version"]

    def replay():
        # The final endpoint is already canonical and model-visible. Replay the
        # missing poses only in the MJPEG layer, then restore that final frame.
        for index in range(1, count):
            yaw = spec["start_yaw"] + spec["delta"] * index / count
            started = time.monotonic()
            try:
                lock = getattr(spec["sim"], "_lock", None)
                if lock is None:
                    rgb, _ = spec["client"].render_one(
                        spec["scene"], spec["eye"], yaw, want_depth=False)
                else:
                    with lock:
                        rgb, _ = spec["client"].render_one(
                            spec["scene"], spec["eye"], yaw, want_depth=False)
                frame = jpg_bytes(rgb if isinstance(rgb, Image.Image)
                                  else Image.fromarray(rgb))
            except Exception as exc:
                print(f"[display] turn replay failed: {exc}", flush=True)
                break
            with FRAME_CV:
                if DISPLAY_ANIMATION["version"] != version:
                    return
                point = TRANSIENT_SCREEN_MARKER["point"]
                STATE["display_frame_jpg"] = _display_with_transient_marker(
                    frame, point)
                STATE["display_seq"] += 1
                FRAME_CV.notify_all()
            remaining = 1.0 / 25.0 - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
        with FRAME_CV:
            if DISPLAY_ANIMATION["version"] != version:
                return
            point = TRANSIENT_SCREEN_MARKER["point"]
            STATE["display_frame_jpg"] = _display_with_transient_marker(
                STATE["display_base_jpg"], point)
            STATE["display_seq"] += 1
            FRAME_CV.notify_all()

    thread = threading.Thread(target=replay, name="vln-turn-replay", daemon=True)
    thread.start()
    return thread


def publish_frame(image, frame_jpg, step_id, distance, map_jpg=None,
                  tool_map_jpg=None, map_heading_deg=None,
                  display_frame_jpg=None, depth=None,
                  preserve_display=False):
    """Atomically publish clean RGB, encoded views, metrics and frame token.

    The bird's-eye pane has its own counter: it is redrawn at a fraction of the
    first-person rate (see refresh), and a client that re-fetched it once per
    RGB frame would spend most of its bandwidth on an unchanged image.
    """
    with FRAME_CV:
        DISPLAY_ANIMATION["version"] += 1
        seq = STATE["seq"] + 1
        display_base = display_frame_jpg or frame_jpg
        point = TRANSIENT_SCREEN_MARKER["point"]
        if (point is not None
                and time.monotonic() >= TRANSIENT_SCREEN_MARKER["expires_at"]):
            point = None
            TRANSIENT_SCREEN_MARKER["version"] += 1
            TRANSIENT_SCREEN_MARKER["point"] = None
            TRANSIENT_SCREEN_MARKER["expires_at"] = 0.0
        STATE.update(frame_jpg=frame_jpg, display_base_jpg=display_base,
                     frame_size=list(image.size), seq=seq, step=step_id,
                     dist=distance)
        if not preserve_display:
            STATE["display_frame_jpg"] = _display_with_transient_marker(
                display_base, point)
            STATE["display_seq"] += 1
        if map_jpg is not None:
            STATE["map_jpg"] = map_jpg
            STATE["tool_map_jpg"] = tool_map_jpg  # never fall back to the GT display map
            if map_heading_deg is not None:
                STATE["map_heading_deg"] = float(map_heading_deg) % 360.0
            STATE["map_seq"] += 1
        CURRENT_FRAME.update(image=image.copy(),
                             depth=(np.asarray(depth, dtype=np.float32).copy()
                                    if depth is not None else None),
                             scene=STATE["scene"],
                             episode_id=STATE["episode_id"], seq=seq)
        FRAME_TS.append(time.time())
        FRAME_CV.notify_all()


def frame_rate():
    """Frames per second over the recent past, 0.0 when nothing is moving."""
    with LOCK:
        ts = list(FRAME_TS)
    if len(ts) < 2 or time.time() - ts[-1] > 1.0:
        return 0.0
    span = ts[-1] - ts[0]
    return round((len(ts) - 1) / span, 1) if span > 1e-6 else 0.0


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------
def jpg_bytes(img, q=85):
    b = io.BytesIO()
    img.convert("RGB").save(b, "JPEG", quality=q)
    return b.getvalue()


def observation_image(obs):
    """Return the canonical, unannotated RGB view for every image consumer."""
    rgb = obs["rgb"]
    if isinstance(rgb, Image.Image):
        return rgb.convert("RGB").copy()
    return Image.fromarray(rgb).convert("RGB")


def visual_tool_preview(obs, name, arguments):
    """Annotate the exact pre-motion frame for a point/bbox tool log event."""
    image = observation_image(obs)
    w, h = image.size
    draw = ImageDraw.Draw(image)
    width = max(3, min(w, h) // 100)
    if name == "move_to_point":
        x, y = VNT.parse_point_2d(arguments.get("point_2d"))
        u = round(x * (w - 1) / 1000.0)
        v = round(y * (h - 1) / 1000.0)
        radius = max(8, min(w, h) // 40)
        draw.ellipse((u-radius, v-radius, u+radius, v+radius),
                     outline=(255, 32, 32), width=width)
        draw.line((u-radius, v, u+radius, v), fill=(255, 32, 32), width=width)
        draw.line((u, v-radius, u, v+radius), fill=(255, 32, 32), width=width)
    elif name == "move_to_bbox":
        box = VNT.parse_bbox_2d(arguments.get("bbox_2d"))
        xy = [round(box[0] * (w - 1) / 1000.0),
              round(box[1] * (h - 1) / 1000.0),
              round(box[2] * (w - 1) / 1000.0),
              round(box[3] * (h - 1) / 1000.0)]
        draw.rectangle(xy, outline=(255, 32, 32), width=width)
        # The Uni-LaViRA Cobot executor projects this bottom-centre ray, not
        # the geometric centre of the box.  Show the actual depth sampling
        # location so the log preview and motion semantics cannot disagree.
        u, v = round((xy[0] + xy[2]) / 2.0), xy[3]
        radius = max(8, min(w, h) // 40)
        draw.ellipse((u-radius, v-radius, u+radius, v+radius),
                     outline=(255, 210, 0), width=width)
        draw.line((u-radius, v, u+radius, v), fill=(255, 210, 0), width=width)
        draw.line((u, v-radius, u, v+radius), fill=(255, 210, 0), width=width)
    else:
        raise ValueError(f"{name} 不是视觉定位工具")
    return _debug_image(image)


def set_active_visual_target(marker):
    """Set/clear the human-only world-space marker used by ``refresh``."""
    with LOCK:
        ACTIVE_VISUAL_TARGET["marker"] = dict(marker) if marker else None


def _marker_occluded(obs, projected_depth, u, v):
    """Use the current metric depth to distinguish visible and hidden goals."""
    depth = obs.get("depth") if isinstance(obs, dict) else None
    if depth is None:
        return False
    values = np.asarray(depth, dtype=np.float32).squeeze()
    if values.ndim != 2:
        return False
    row = int(np.clip(round(v), 0, values.shape[0]-1))
    col = int(np.clip(round(u), 0, values.shape[1]-1))
    radius = 2
    patch = values[max(0, row-radius):row+radius+1,
                   max(0, col-radius):col+radius+1]
    valid = patch[np.isfinite(patch) & (patch > 1e-4)]
    return bool(valid.size and float(np.median(valid)) < projected_depth-0.15)


def visual_target_frame(env, obs, image, marker=None):
    """Draw a perspective, depth-aware world marker for the browser only.

    The footprint ring lies in the ground plane and the pin has a real metric
    height. Both are reprojected from world coordinates at every simulator
    sub-step, so this is a small 3D scene annotation rather than a fixed screen
    overlay. Dashed/dim geometry means the current depth image occludes it.
    """
    if marker is None:
        with LOCK:
            current = ACTIVE_VISUAL_TARGET["marker"]
            marker = dict(current) if current else None
    if not marker or not marker.get("world_xyz"):
        return image

    target = tuple(float(x) for x in marker["world_xyz"])
    centre = VNT.project_world_point(env, target, image.size)
    if centre is None:
        return image
    u, v, forward = centre
    w, h = image.size
    # Do not paint an off-screen HUD arrow: the marker is allowed to leave the
    # view exactly like a physical landmark would.
    margin = max(w, h) * 0.15
    if not (-margin <= u < w+margin and -margin <= v < h+margin):
        return image

    occluded = _marker_occluded(obs, forward, u, v)
    colour = (255, 195, 0) if marker.get("kind") == "bbox" else (0, 235, 255)
    if occluded:
        colour = tuple(int(c*0.48) for c in colour)
    outline = (18, 18, 18)
    draw = ImageDraw.Draw(image)
    line_width = max(2, min(w, h)//180)

    # A metric ring on the horizontal plane makes the perspective/depth
    # explicit: it becomes an ellipse, expands on approach and rotates with the
    # camera. It also visualises the controller's actual stopping radius.
    radius_m = max(0.08, float(marker.get("goal_radius_m", 0.2)))
    ring = []
    for angle in np.linspace(0.0, 2.0*math.pi, 33):
        point = (target[0] + radius_m*math.cos(float(angle)), target[1],
                 target[2] + radius_m*math.sin(float(angle)))
        projected = VNT.project_world_point(env, point, image.size)
        ring.append(None if projected is None else projected[:2])
    segments, segment = [], []
    for point in ring + [None]:
        if point is not None:
            segment.append(point)
        elif len(segment) >= 2:
            segments.append(segment)
            segment = []
        else:
            segment = []
    for index, points in enumerate(segments):
        if occluded:
            for i in range(0, len(points)-1, 2):
                draw.line((points[i], points[i+1]), fill=outline,
                          width=line_width+2)
                draw.line((points[i], points[i+1]), fill=colour,
                          width=line_width)
        else:
            draw.line(points, fill=outline, width=line_width+2, joint="curve")
            draw.line(points, fill=colour, width=line_width, joint="curve")

    top = VNT.project_world_point(
        env, (target[0], target[1]+0.38, target[2]), image.size)
    if top is not None:
        stalk = ((u, v), (top[0], top[1]))
        draw.line(stalk, fill=outline, width=line_width+3)
        draw.line(stalk, fill=colour, width=line_width+1)
        head_r = max(4, min(13, int(90.0/max(forward, 0.2))))
        draw.ellipse((top[0]-head_r, top[1]-head_r,
                      top[0]+head_r, top[1]+head_r),
                     fill=colour, outline=outline, width=line_width)
    label = ("BBOX 3D" if marker.get("kind") == "bbox" else "POINT 3D")
    if occluded:
        label += " · OCCLUDED"
    label += f" · {forward:.2f}m"
    tx = int(np.clip(u+10, 3, max(3, w-150)))
    ty = int(np.clip((top[1] if top else v)-18, 3, max(3, h-18)))
    draw.rectangle((tx-3, ty-2, min(w-1, tx+8*len(label)+3), ty+14),
                   fill=(12, 12, 12))
    draw.text((tx, ty), label, fill=colour)
    return image


def set_active_lightnav_pointing(markers, trajectory=None, restore_display=False):
    """Replace LightNav's browser-only point and trajectory world anchors.

    ``restore_display`` removes the last source-frame overlay immediately when
    a run ends or errors.  Canonical frame bytes and CURRENT_FRAME are never
    changed.
    """
    with FRAME_CV:
        ACTIVE_LIGHTNAV_POINTING["markers"] = [dict(marker) for marker in markers]
        ACTIVE_LIGHTNAV_POINTING["trajectory"] = (
            {key: ([tuple(point) for point in value]
                   if isinstance(value, list) else value)
             for key, value in trajectory.items()} if trajectory else None)
        if restore_display and STATE["frame_jpg"] is not None:
            point = TRANSIENT_SCREEN_MARKER["point"]
            STATE["display_base_jpg"] = STATE["frame_jpg"]
            STATE["display_frame_jpg"] = _display_with_transient_marker(
                STATE["frame_jpg"], point)
            STATE["display_seq"] += 1
            FRAME_CV.notify_all()


def lightnav_camera_height(env, default=0.5):
    """Best available optical-centre height above the local ground plane."""
    try:
        state = env.sim.get_agent_state()
        body_y = float(state.position[1])
        sensors = getattr(state, "sensor_states", {}) or {}
        for key in ("depth", "depth_sensor", "rgb", "rgb_sensor"):
            sensor = sensors.get(key)
            if sensor is not None and getattr(sensor, "position", None) is not None:
                height = float(sensor.position[1])-body_y
                if 0.02 < height < 10.0:
                    return height
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    for value in (
            getattr(getattr(env, "sim", None), "sensor_height", None),
            getattr(getattr(getattr(env, "sim", None), "agents", [None])[0],
                    "agent_config", None)):
        if value is None:
            continue
        if not isinstance(value, (int, float)):
            value = getattr(value, "sensor_height", None)
        try:
            height = float(value)
        except (TypeError, ValueError):
            continue
        if 0.02 < height < 10.0:
            return height
    return float(default)


def lightnav_world_trajectory(env, waypoints, image_size):
    """Convert one metric ego trajectory into a display-only world ribbon."""
    try:
        array = np.asarray(waypoints, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 3
            or not np.all(np.isfinite(array[:, :3]))):
        return None
    array = array[:, :3]
    hfov = VNT.camera_hfov(env)
    cam_height = lightnav_camera_height(env)
    offset = LND.trajectory_bottom_edge_depth(image_size, hfov, cam_height)
    width, height = image_size
    half_width = LND.TRAJECTORY_WIDTH_FRAC*width/max(1, height)*cam_height
    path = np.vstack((np.zeros((1, 3), dtype=np.float64), array))
    ground_y = VNT.camera_world_y(env)-cam_height

    def lift(forward, lateral):
        # LightNav +lateral is camera-left; VNT local_x is camera-right.
        target_xz, _ = VNT.target_habitat_xz(
            env, -float(lateral), float(forward)+offset)
        return (float(target_xz[0]), float(ground_y), float(target_xz[1]))

    return {
        "left": [lift(row[0], row[1]+half_width) for row in path],
        "right": [lift(row[0], row[1]-half_width) for row in path],
        "centres": [lift(row[0], row[1]) for row in array],
        "headings": [lift(row[0]+0.10*math.cos(row[2]),
                           row[1]+0.10*math.sin(row[2])) for row in array],
        "count": int(array.shape[0]),
    }


def lightnav_trajectory_frame(env, image, trajectory):
    """Reproject a predicted world ribbon onto the current browser frame."""
    if not trajectory:
        return image
    left = [VNT.project_world_point(env, point, image.size)
            for point in trajectory.get("left", [])]
    right = [VNT.project_world_point(env, point, image.size)
             for point in trajectory.get("right", [])]
    centres = [VNT.project_world_point(env, point, image.size)
               for point in trajectory.get("centres", [])]
    headings = [VNT.project_world_point(env, point, image.size)
                for point in trajectory.get("headings", [])]
    if len(left) < 2 or len(right) != len(left):
        return image
    width, height = image.size
    limit = 8*max(width, height)

    def pixel(value):
        if value is None or not all(math.isfinite(float(v)) for v in value[:2]):
            return None
        return (int(np.clip(round(value[0]), -limit, limit)),
                int(np.clip(round(value[1]), -limit, limit)))

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    denom = max(1, len(left)-2)
    for index in reversed(range(len(left)-1)):
        points = (pixel(left[index]), pixel(left[index+1]),
                  pixel(right[index+1]), pixel(right[index]))
        if any(point is None for point in points):
            continue
        t = index/denom
        colour = tuple(round(LND.TRAJECTORY_NEAR[k]
                             + (LND.TRAJECTORY_FAR[k]-LND.TRAJECTORY_NEAR[k])*t)
                       for k in range(3))
        alpha = round(220*(1.0 if t <= 0.5 else max(0.0, 2.0*(1.0-t))**1.15))
        draw.polygon(points, fill=(*colour, alpha))
    rail_width = max(2, round(2*height/270.0))
    for edge in (left, right):
        points = [pixel(value) for value in edge]
        for index in range(len(points)-1):
            if points[index] is not None and points[index+1] is not None:
                draw.line((points[index], points[index+1]),
                          fill=(*LND.TRAJECTORY_RAIL, 225), width=rail_width)
    radius = max(3, min(width, height)//100)
    for centre, heading in zip(centres, headings):
        p0, p1 = pixel(centre), pixel(heading)
        if p0 is not None and p1 is not None:
            draw.line((p0, p1), fill=(255, 255, 255, 235),
                      width=max(1, radius//2))
    for value in centres:
        point = pixel(value)
        if point is None:
            continue
        u, v = point
        draw.ellipse((u-radius, v-radius, u+radius, v+radius),
                     fill=(*LND.TRAJECTORY_NEAR, 235),
                     outline=(255, 255, 255, 245), width=max(1, radius//3))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def lightnav_world_markers(env, obs, pointing, image_size):
    """Lift valid APOS/OPOS pixels into world space when metric depth exists."""
    points = LND.pointing_pixels(pointing, image_size)
    if not points:
        return []
    width, height = image_size
    markers = []
    for channel, (u0, v0) in points.items():
        point_2d = [
            1000.0 * u0 / max(1, width-1),
            1000.0 * v0 / max(1, height-1),
        ]
        try:
            u, v, depth_m, size = VNT.point_sample(obs, point_2d)
            local_x, local_y, local_z = VNT.local_waypoint(
                u, v, depth_m, size, VNT.camera_hfov(env))
            target_xz, _ = VNT.target_habitat_xz(env, local_x, local_z)
            target_y = VNT.camera_world_y(env)-local_y
        except (TypeError, ValueError, AttributeError):
            # Pointing is optional visualization metadata. Lack of metric depth
            # (notably the remote MuJoCo camera) must not fail policy execution.
            continue
        markers.append({
            "channel": channel,
            "world_xyz": (float(target_xz[0]), float(target_y),
                          float(target_xz[1])),
            "source_pixel": (int(u), int(v)),
            "depth_m": float(depth_m),
        })
    return markers


def lightnav_pointing_frame(env, obs, image, markers=None):
    """Reproject LightNav APOS/OPOS world anchors onto a browser frame."""
    if markers is None:
        with LOCK:
            markers = [dict(marker)
                       for marker in ACTIVE_LIGHTNAV_POINTING["markers"]]
    if not markers:
        return image
    width, height = image.size
    draw = ImageDraw.Draw(image)
    stroke = max(2, min(width, height)//180)
    radius = max(8, min(width, height)//38)
    outline = (15, 18, 22)
    for marker in markers:
        projected = VNT.project_world_point(
            env, tuple(marker.get("world_xyz") or ()), image.size)
        if projected is None:
            continue
        u, v, forward = projected
        margin = max(width, height)*0.15
        if not (-margin <= u < width+margin and -margin <= v < height+margin):
            continue
        channel = marker.get("channel")
        colour = LND.POINTING_COLOURS.get(channel, (240, 240, 240))
        if _marker_occluded(obs, forward, u, v):
            colour = tuple(int(value*0.48) for value in colour)
        draw.ellipse((u-radius, v-radius, u+radius, v+radius),
                     outline=outline, width=stroke+2)
        draw.ellipse((u-radius, v-radius, u+radius, v+radius),
                     outline=colour, width=stroke)
        draw.line((u-radius-4, v, u+radius+4, v), fill=outline, width=stroke+2)
        draw.line((u, v-radius-4, u, v+radius+4), fill=outline, width=stroke+2)
        draw.line((u-radius-4, v, u+radius+4, v), fill=colour, width=stroke)
        draw.line((u, v-radius-4, u, v+radius+4), fill=colour, width=stroke)
        label = f"{str(channel).upper()} · {forward:.2f}m"
        label_width = 8*len(label)+6
        tx = (u+radius+6 if u+radius+label_width < width
              else max(3, u-radius-label_width))
        tx = int(tx)
        ty = int(np.clip(v-radius, 3, max(3, height-17)))
        draw.rectangle((tx-3, ty-2, min(width-1, tx+8*len(label)+3), ty+14),
                       fill=outline)
        draw.text((tx, ty), label, fill=colour)
    return image


def show_lightnav_prediction(env, image, prediction):
    """Publish trajectory plus APOS/OPOS on the exact policy-input frame."""
    annotated = LND.draw_prediction_overlay(
        image, prediction.waypoints, prediction.pointing,
        hfov_deg=VNT.camera_hfov(env), cam_height_m=lightnav_camera_height(env))
    with FRAME_CV:
        STATE["display_frame_jpg"] = jpg_bytes(annotated)
        STATE["display_seq"] += 1
        FRAME_CV.notify_all()


def _debug_image(image):
    """Store one PIL image once and return JSON-safe preview metadata."""
    global DEBUG_IMAGE_NEXT
    oid = id(image)
    with LOCK:
        cached = DEBUG_IMAGE_OBJECTS.get(oid)
        if cached and cached[0]() is image and cached[1] in DEBUG_IMAGES:
            key = cached[1]
            data = DEBUG_IMAGES[key]
            return {"type": "image", "url": f"/debug-image/{key}.jpg",
                    "width": data[1], "height": data[2]}
    try:
        rgb = image.convert("RGB")
        width, height = rgb.size
        data = jpg_bytes(rgb, q=90)
    except Exception as exc:
        return {"type": "text", "text": f"[调试预览无法编码图片：{exc}]"}
    with LOCK:
        key = DEBUG_IMAGE_NEXT
        DEBUG_IMAGE_NEXT += 1
        DEBUG_IMAGES[key] = (data, width, height)
        DEBUG_IMAGE_ORDER.append(key)
        try:
            DEBUG_IMAGE_OBJECTS[oid] = (weakref.ref(image), key)
        except TypeError:
            pass
        while len(DEBUG_IMAGE_ORDER) > DEBUG_IMAGE_LIMIT:
            old = DEBUG_IMAGE_ORDER.popleft()
            DEBUG_IMAGES.pop(old, None)
            for obj_id, cached in list(DEBUG_IMAGE_OBJECTS.items()):
                if cached[1] == old:
                    DEBUG_IMAGE_OBJECTS.pop(obj_id, None)
        return {"type": "image", "url": f"/debug-image/{key}.jpg",
                "width": width, "height": height}


def capture_model_request(messages, turn, driver, display_from=0,
                          hidden_roles=()):
    """Snapshot the new visible input for one model call.

    The wire request still contains ``messages`` in full.  For a multi-turn
    ReAct conversation, however, the log should not print that full history on
    every turn: earlier input and assistant output are already immediately
    above it in the timeline.  ``display_from`` selects only newly appended
    messages and ``hidden_roles`` removes the assistant reply which is rendered
    by its own model-output event.
    """
    global DEBUG_REQUEST_NEXT
    snapshot, image_count = [], 0
    visible = [(index, message) for index, message in enumerate(messages)
               if index >= display_from
               and message.get("role") not in hidden_roles]
    for index, message in visible:
        parts = []
        content = message.get("content", "")
        if not isinstance(content, list):
            content = [{"type": "text", "text": str(content)}]
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image" \
                    and part.get("image") is not None:
                rendered = _debug_image(part["image"])
                image_count += rendered.get("type") == "image"
                parts.append(rendered)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append({"type": "text", "text": str(part.get("text") or "")})
            else:
                try:
                    text = json.dumps(part, ensure_ascii=False, default=str)
                except Exception:
                    text = str(part)
                parts.append({"type": "text", "text": text})
        snapshot.append({"index": index + 1,
                         "role": str(message.get("role") or "unknown"),
                         "parts": parts})
    with LOCK:
        request_id = DEBUG_REQUEST_NEXT
        DEBUG_REQUEST_NEXT += 1
        DEBUG_REQUESTS[request_id] = {
            "id": request_id, "turn": turn, "driver": driver,
            "message_count": len(snapshot), "image_count": image_count,
            "total_message_count": len(messages),
            "messages": snapshot,
        }
        DEBUG_REQUEST_ORDER.append(request_id)
        while len(DEBUG_REQUEST_ORDER) > DEBUG_REQUEST_LIMIT:
            DEBUG_REQUESTS.pop(DEBUG_REQUEST_ORDER.popleft(), None)
    push_event("request", request_id=request_id, turn=turn, driver=driver,
               message_count=len(snapshot), total_message_count=len(messages),
               image_count=image_count)
    return request_id


def debug_request(request_id):
    with LOCK:
        return DEBUG_REQUESTS.get(request_id)


def debug_image(image_id):
    with LOCK:
        item = DEBUG_IMAGES.get(image_id)
        return item[0] if item else None


# Minimum seconds between two bird's-eye redraws; see refresh().
MAP_MIN_DT = float(os.environ.get("DEMO_MAP_MIN_DT", "0.25"))
MAP_LAST = [0.0]
# How to read a click on the map back into world coordinates. The page reports
# clicks as fractions of the displayed image, which survives every rescaling
# between here and the browser -- but not render_map()'s conditional rot90, and
# not the fact that habitat's grid is not the env-drawn maps' pixel grid. Both
# are recorded here as the map is drawn, because that is the only place that
# knows which of the two paths produced the picture on screen.
MAP_GEOM = {"kind": "", "rot": False, "shape": (0, 0)}


def render_map(tdm, show_goal=False):
    if base_maps is None or not isinstance(tdm, dict) or "map" not in tdm:
        return None
    from habitat_topdown import agent_map_raster
    # Display overlays live on copies; agent maps also strip legacy annotations.
    m = (np.asarray(tdm["map"], dtype=np.uint8) if show_goal
         else agent_map_raster(tdm["map"]))
    fog = tdm.get("fog_of_war_mask")
    if fog is not None:
        fog = np.asarray(fog)
    rgb = base_maps.colorize_topdown_map(m, fog, fog_of_war_desat_amount=FOG_DIM)
    coords = tdm.get("agent_map_coord")
    angles = tdm.get("agent_angle")
    if coords is not None:
        coords = np.atleast_2d(np.asarray(coords))
        angles = (np.atleast_1d(np.asarray(angles)) if angles is not None
                  else [0.0] * len(coords))
        for i, pos in enumerate(coords):
            rgb = base_maps.draw_agent(
                image=rgb, agent_center_coord=(int(pos[0]), int(pos[1])),
                agent_rotation=float(angles[i]),
                agent_radius_px=max(4, min(rgb.shape[0], rgb.shape[1]) // 40))
    if rgb.shape[0] > rgb.shape[1]:
        rgb = np.rot90(rgb, 1)
        MAP_GEOM.update(kind="habitat", rot=True, shape=m.shape[:2])
    else:
        MAP_GEOM.update(kind="habitat", rot=False, shape=m.shape[:2])
    return rgb


def top_down_view(env, info, show_goal=False):
    """Bird's-eye view for the right-hand pane.

    habitat ships one inside ``get_metrics()`` as a dict of map codes; the 3DRS
    env rasterises its own from the navmesh instead, so ask the env first and
    only fall back to the habitat measure.
    """
    own = getattr(env, "render_top_down", None)
    if own is not None:
        MAP_GEOM.update(kind="env", rot=False, shape=(0, 0))
        return own(show_goal=show_goal)
    tdm = info.get("top_down_map")
    if show_goal:
        from vlnce_training import reference_map
        tdm = reference_map(env, tdm, base_maps)
    return render_map(tdm, show_goal=show_goal)


def map_heading_degrees(env, info):
    """Current arrow heading on the displayed map, clockwise from screen-up."""
    own = getattr(env, "map_heading_degrees", None)
    if own is not None:
        return float(own()) % 360.0
    tdm = info.get("top_down_map")
    if not isinstance(tdm, dict):
        return 0.0
    angles = tdm.get("agent_angle")
    if angles is None:
        return 0.0
    angle = float(np.atleast_1d(np.asarray(angles))[0])
    # Habitat flips its source sprite vertically before rotating it, so its
    # zero-angle arrow points down (south), not up.  Convert that native
    # convention to a clockwise compass bearing on the final displayed image.
    heading = 180.0 - math.degrees(angle)
    m = np.asarray(tdm.get("map"))
    if m.ndim >= 2 and m.shape[0] > m.shape[1]:
        heading -= 90.0
    return heading % 360.0


def map_click_world(env, fx, fy):
    """Where a click at ``(fx, fy)`` on the bird's-eye view lands, in world metres.

    ``fx``/``fy`` are fractions of the displayed map's width/height -- the page
    cannot know what the pixels meant, and normalising is what makes the answer
    independent of refresh()'s resize and of the browser's own scaling.

    Two inverses, picked by whatever drew the last map (see MAP_GEOM). The
    env-drawn maps each own theirs (they own the forward transform too). For
    habitat the transform lives in habitat's own ``maps`` module, so it is
    undone here: first render_map()'s rot90, then grid -> world.

    Returns a habitat-shaped ``[x, y, z]``, or None if there is no navigable
    point near the click.
    """
    fx = min(1.0, max(0.0, float(fx)))
    fy = min(1.0, max(0.0, float(fy)))
    if MAP_GEOM["kind"] == "env":
        own = getattr(env, "map_to_world", None)
        return None if own is None else own(fx, fy)
    if MAP_GEOM["kind"] != "habitat" or base_maps is None:
        return None
    rows, cols = MAP_GEOM["shape"]
    if rows < 2 or cols < 2:
        return None
    # np.rot90(A, 1) puts A[j, C-1-i] at B[i, j]: a click at (col fx, row fy) of
    # B is row fx, col 1-fy of A. Worked in fractions so it holds after resizing.
    nr, nc = (fx, 1.0 - fy) if MAP_GEOM["rot"] else (fy, fx)
    z, x = base_maps.from_grid(nr * (rows - 1), nc * (cols - 1), (rows, cols),
                               pathfinder=env.sim.pathfinder)
    y = float(env.sim.get_agent_state().position[1])
    p = np.asarray(env.sim.pathfinder.snap_point(
        np.array([x, y, z], dtype=np.float32)), dtype=float)
    return None if not np.all(np.isfinite(p)) else p


def do_goto(env, obs, fx, fy):
    """Teleport to a clicked point on the map. Returns the new observations.

    Deliberately keeps the heading: the click says where to stand, and a jump
    that also spun the camera would make it hard to tell which way you now face.
    """
    try:
        p = map_click_world(env, fx, fy)
    except Exception as exc:
        push_event("err", text=f"地图定位失败：{exc}")
        return obs
    if p is None:
        push_event("sys", text="那个位置不可到达（点在可走区域里再试一次）")
        return obs
    own = getattr(env, "teleport", None)
    try:
        if own is not None:
            new = own(p)
        else:
            new = _habitat_teleport(env, p)
    except Exception as exc:
        push_event("err", text=f"传送失败：{exc}")
        return obs
    obs = new or obs
    st = env.sim.get_agent_state().position
    push_event("sys", text="已传送到 (%.2f, %.2f)" % (float(st[0]), float(st[2])))
    refresh(env, obs, STATE["step"], force_map=True)
    return obs


def _habitat_teleport(env, p):
    """habitat's half of do_goto: move the agent, then re-run the measures.

    The measures are what draw the bird's-eye view, and they only update inside
    env.step() -- without this the map would keep showing the old position until
    something else moved. ``_previous_xy_location`` is cleared first so
    TopDownMap does not join the two positions with a line straight through the
    walls in between.
    """
    rot = env.sim.get_agent_state().rotation
    env.sim.set_agent_state(p, rot)
    obs = env.sim.get_observations_at(p, rot, keep_agent_at_new_pose=True)
    tdm = getattr(env.task.measurements, "measures", {}).get("top_down_map")
    if tdm is not None and getattr(tdm, "_previous_xy_location", None) is not None:
        tdm._previous_xy_location = [None] * len(tdm._previous_xy_location)
    env.task.measurements.update_measures(
        episode=env.current_episode, action=None, task=env.task,
        observations=obs)
    return obs


def refresh(env, obs, step_id, force_map=False, preserve_display=False):
    """Re-encode the views and publish the current metrics.

    The bird's-eye pane is throttled to MAP_MIN_DT. Colourising, rotating,
    resizing and re-encoding it costs about four times what the first-person
    frame costs (measured: 3.9 ms vs 1.0 ms), and no one reads a minimap twelve
    times a second -- so it is redrawn a few times a second and the saved
    milliseconds go into the view you actually steer by. ``force_map`` overrides
    that at the moments where a stale map would look like a bug: a fresh
    episode, the end of a run, letting go of the keys.
    """
    if isinstance(obs, dict) and obs.get('_render_deferred'):
        # Pose/metrics advance, but no stale image gets a new frame identity.
        return
    info = env.get_metrics()
    set_state(evaluation=goal_evaluation(info))
    cur = observation_image(obs)
    frame = jpg_bytes(cur)
    # Keep the canonical frame pristine for model input, vln_observe, bbox
    # coordinates and workspace snapshots. Only the browser stream receives
    # this independently encoded world-space annotation.
    with LOCK:
        active_marker = ACTIVE_VISUAL_TARGET["marker"]
        active_marker = dict(active_marker) if active_marker else None
        lightnav_markers = [dict(marker)
                            for marker in ACTIVE_LIGHTNAV_POINTING["markers"]]
        lightnav_trajectory = ACTIVE_LIGHTNAV_POINTING["trajectory"]
    display_image = cur.copy()
    if active_marker:
        display_image = visual_target_frame(env, obs, display_image, active_marker)
    if lightnav_trajectory:
        display_image = lightnav_trajectory_frame(
            env, display_image, lightnav_trajectory)
    if lightnav_markers:
        display_image = lightnav_pointing_frame(
            env, obs, display_image, lightnav_markers)
    display_frame = (jpg_bytes(display_image)
                     if active_marker or lightnav_markers or lightnav_trajectory else frame)
    now = time.time()
    mjpg = None
    tool_mjpg = None
    map_heading = None
    if force_map or STATE["map_jpg"] is None or now - MAP_LAST[0] >= MAP_MIN_DT:
        MAP_LAST[0] = now
        mrgb = top_down_view(env, info, show_goal=True)
        if mrgb is not None:
            mp = Image.fromarray(mrgb).convert("RGB")
            h = 420
            w = max(1, int(mp.width * h / max(1, mp.height)))
            mjpg = jpg_bytes(mp.resize((w, h)))
            # Derive the model view independently from the unannotated state.
            tool_mrgb = top_down_view(env, info, show_goal=False)
            if tool_mrgb is not None:
                tool_mp = Image.fromarray(tool_mrgb).convert("RGB")
                tool_mjpg = jpg_bytes(tool_mp.resize((w, h)))
            map_heading = map_heading_degrees(env, info)
    publish_frame(cur, frame, step_id,
                  float(info.get("distance_to_goal", float("nan"))),
                  mjpg, tool_mjpg, map_heading,
                  display_frame_jpg=display_frame,
                  depth=obs.get("depth") if isinstance(obs, dict) else None,
                  preserve_display=preserve_display)


def refresh_action_head_remote(obs, step_id):
    """Publish RGB and map from the same remote Gym that moved the agent."""
    cur = observation_image(obs)
    metrics = ACTION_HEAD_AGENT_RUNTIME.ui_metrics()
    set_state(evaluation=goal_evaluation(metrics))
    display_distance = float(metrics.get("distance_to_goal", float("nan")))
    map_jpg = tool_map_jpg = None
    try:
        remote_map = ACTION_HEAD_AGENT_RUNTIME.render_map(show_goal=True)
        h = 420
        w = max(1, int(remote_map.width * h / max(1, remote_map.height)))
        map_jpg = jpg_bytes(remote_map.resize((w, h)))
        tool_map = ACTION_HEAD_AGENT_RUNTIME.render_map(show_goal=False)
        tool_map_jpg = jpg_bytes(tool_map.resize((w, h)))
    except Exception as exc:
        # publish_frame's generic fallback reuses map_jpg for tool_map_jpg.
        # Do not publish the oracle map if its goal-free counterpart failed.
        map_jpg = tool_map_jpg = None
        push_event("err", text=f"Action Head 远程小地图获取失败：{exc}")
    publish_frame(
        cur, jpg_bytes(cur), step_id, display_distance,
        map_jpg=map_jpg, tool_map_jpg=tool_map_jpg,
        depth=obs.get("depth") if isinstance(obs, dict) else None,
    )


def action_head_progress(option, executed):
    """Publish each executed fast prefix while the Tool still owns control."""
    with LOCK:
        step_id = int(STATE.get("step", 0)) + max(0, int(executed.sim_steps))
        motion = dict(STATE.get("motion") or {})
    motion["steps"] = step_id
    motion["path_dist"] = round(
        float(motion.get("path_dist", 0.0))
        + max(0.0, float(executed.moved_m)), 3)
    set_trail(motion=motion)
    refresh_action_head_remote(executed.observation, step_id)
    push_event(
        "steps",
        text=(f"FM Head chunk {option.chunks}/4："
              f"{int(executed.sim_steps)} 步，移动 {float(executed.moved_m):.2f}m"
              + ("，受阻" if executed.blocked else "")),
    )


def set_trail(actions=None, moves=None, motion=None):
    """Publish whichever trail views changed. `actions` is the legacy int list;
    `moves`/`motion` are the ReAct driver's continuous equivalents."""
    kw = {}
    if actions is not None:
        kw["actions"] = list(actions)
    if moves is not None:
        kw["moves"] = list(moves)
    if motion is not None:
        kw["motion"] = dict(motion)
    if kw:
        set_state(**kw)


def relative_pose_motion(start_xy, start_yaw, end_xy, end_yaw):
    """Return the planar SE(2) change in start-heading polar coordinates.

    ``dist`` is the straight-line start-to-end distance, not distance walked.
    ``bearing`` is the displacement direction relative to the starting view:
    right is positive and left is negative. ``yaw`` is the final heading
    change with the same sign convention.
    """
    dx = float(end_xy[0]) - float(start_xy[0])
    dz = float(end_xy[1]) - float(start_xy[1])
    dist = math.hypot(dx, dz)
    if dist < 0.0005:
        bearing = 0.0
    else:
        heading = math.radians(float(start_yaw))
        forward = dx * math.sin(heading) - dz * math.cos(heading)
        right = dx * math.cos(heading) + dz * math.sin(heading)
        bearing = math.degrees(math.atan2(right, forward))
    return {
        "dist": round(dist, 3),
        "bearing": round(A.wrap180(bearing), 1),
        "yaw": round(A.wrap180(float(end_yaw) - float(start_yaw)), 1),
    }


def update_pose_motion(env, start_xy, start_yaw, motion):
    motion.update(relative_pose_motion(
        start_xy, start_yaw, A._xz(env), A._yaw(env)))
    return motion


# ----------------------------------------------------------------------------
# FM head (in-process, no vLLM)
# ----------------------------------------------------------------------------
# The FM driver does not talk to a vLLM server: the FM head only runs through
# the forked-transformers model in-process (generate -> forward -> ActionFormer
# -> Euler ODE), so the chat-completions path the other two drivers use cannot
# serve it. This loads the SocialNav FMInfer once and reuses it -- the same
# loader path verl_access/verify_fm.py uses, so the import works wherever that
# script already runs (i.e. the SocialNav venv + GPU).
_FM = {"infer": None, "denorm": None, "norm": None, "anonymous": None}

# FM head anonymous stats + denorm（fm_policy._ZYT / _denorm_delta 的本地镜像）。
# proxy 路径（gordon-venv 控制台经 FM_INFER_URL 走 HTTP sidecar）不能 import fm_policy
# （gordon venv hub 1.11.0 与 forked transformers pin hub<1.0 冲突），故 denorm/anonymous 本地留一份，
# sidecar /predict 也直接返回 denorm 后的 real，控制台 run_fm 两边都能用。
_FM_ZYT = {
    "dx":   {"q01": -0.10273411870002747, "q99":  0.1031249612569809},
    "dy":   {"q01": -0.10309220850467682, "q99":  0.10309189558029175},
    "dyaw": {"q01": -0.5668500065803528,  "q99":  0.5747644901275635},
}


def _fm_denorm_delta(norm_deltas, stats):
    """(32,3) 归一化 delta -> 真实米/弧度 delta（同 fm_policy._denorm_delta）。"""
    q01 = np.array([stats["dx"]["q01"], stats["dy"]["q01"], stats["dyaw"]["q01"]], np.float32)
    q99 = np.array([stats["dx"]["q99"], stats["dy"]["q99"], stats["dyaw"]["q99"]], np.float32)
    return (np.asarray(norm_deltas, np.float32) + 1.0) / 2.0 * (q99 - q01) + q01


class FMInferProxy:
    """经 HTTP 调 fm_infer_server.py sidecar 的 FMInfer 替身（venv 无关）。

    gordon-venv 控制台装不了 SocialNav forked transformers → in-process 加载不了
    FM head。改成 .venv_fm 起 `fm_infer_server.py` sidecar，控制台经 FM_INFER_URL
    POST /predict（图+指令+input_waypoint）拿 32×3 delta。predict 签名与 FMInfer 一致，
    run_fm 无感切换。sidecar 也返回 denorm 后的 real，省得本地再算。
    """

    def __init__(self, url, device="proxy"):
        self.url = url.rstrip("/")
        self.device = device
        self.action_chunk = 32
        self.action_dim = 3
        self.num_flow_steps = 3
        self._import_urllib()

    def _import_urllib(self):
        import urllib.request
        self._urllib = urllib.request

    @property
    def _alive(self):
        try:
            with self._urllib.urlopen(self.url + "/health", timeout=3) as r:
                return json.loads(r.read()).get("loaded") is True
        except Exception:
            return False

    def predict(self, frames, input_wp_norm, instruction, max_new_tokens=64):
        b64 = []
        for fr in frames:
            buf = io.BytesIO()
            fr.convert("RGB").save(buf, "JPEG", quality=90)
            b64.append(base64.b64encode(buf.getvalue()).decode())
        payload = json.dumps({
            "images": b64,
            "instruction": instruction or "",
            "input_waypoint": np.asarray(input_wp_norm, np.float32).tolist(),
        }).encode()
        req = self._urllib.Request(self.url + "/predict", data=payload,
                                   headers={"Content-Type": "application/json"})
        with self._urllib.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
        if "error" in out:
            raise RuntimeError("fm-sidecar: " + out["error"])
        return np.array(out["delta"], np.float32), list(out.get("gen_ids", []))


def _load_fm():
    """Lazy-load the FM head singleton. Returns the cache dict or None.

    两条路径：
    * **proxy**（``FM_INFER_URL`` env 非空）→ 不 import fm_policy，建 ``FMInferProxy``
      走 HTTP sidecar（``fm_infer_server.py``，.venv_fm 起）。gordon-venv 控制台用这条
      ——gordon venv hub 1.11.0 装不了 SocialNav forked transformers（pin hub<1.0），in-process
      会报 ``huggingface-hub>=0.34.0,<1.0 ... found 1.11.0``；sidecar 在 .venv_fm 里加载，
      控制台只经 HTTP 拿 delta。denorm/anonymous 用本地镜像 ``_fm_denorm_delta``/``_FM_ZYT``
      （sidecar /predict 也直返 denorm 后 real，省本地算）。
    * **in-process**（无 FM_INFER_URL）→ 默认走 closed_loop 训练配置那份 fm_policy
      （``<repo>/verl/workers/fm_policy.py`` + ``<repo>/third_party/SocialNav``），含
      PROMPT_VARIANT/_ZYT/0729 base。需 .venv_fm + GPU（见 start_sage_fm_viz.sh）。
    """
    if _FM["infer"] is not None:
        return _FM
    try:
        url = os.environ.get("FM_INFER_URL", "").strip()
        if url:
            infer = FMInferProxy(url)
            ckpt = CFG.get("fm_ckpt") or os.environ.get("FM_CKPT") or "sidecar"
            _FM.update(infer=infer, denorm=_fm_denorm_delta, norm=None, anonymous=_FM_ZYT)
            push_event("sys", text=f"FM 头走 sidecar（{url}，{os.path.basename(ckpt)}）")
            return _FM
        _REPO = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))   # closed_loop_3dgs_rl
        _SOCIAL = os.environ.get(
            "DEMO_FM_SOCIAL", os.path.join(_REPO, "third_party", "SocialNav"))
        _VERL = os.environ.get("DEMO_FM_VERL", _REPO)
        for p in (_SOCIAL, os.path.join(_SOCIAL, "train_clean"),
                  os.path.join(_VERL, "verl", "workers"),
                  os.path.join(_VERL, "verl", "bench")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from fm_policy import (FMInfer, _denorm_delta, _norm_input_wp,
                               _ZYT, FM_CKPT)  # verl/workers
        ckpt = CFG.get("fm_ckpt") or os.environ.get("FM_CKPT") or FM_CKPT
        device = CFG.get("fm_device") or os.environ.get("FM_DEVICE", "cuda")
        print(f"[fm] loading FM head: {ckpt} on {device}", flush=True)
        infer = FMInfer(ckpt=ckpt, device=device)
        _FM.update(infer=infer, denorm=_denorm_delta, norm=_norm_input_wp,
                   anonymous=_ZYT)
        push_event("sys", text=f"FM 头已加载（{os.path.basename(ckpt)}，"
                               f"chunk={infer.action_chunk}×{infer.action_dim}）")
        return _FM
    except Exception as exc:
        import traceback
        traceback.print_exc()
        push_event("err", text=f"FM 模型加载失败：{exc}（需 SocialNav 环境 + GPU，或起 fm sidecar 设 FM_INFER_URL）")
        return None


# ----------------------------------------------------------------------------
# vLLM streaming
# ----------------------------------------------------------------------------
def apply_model(ev, key=None):
    """Point `ev` at the selected backend. Returns the label for logging.

    Both servers run continuously; switching is just a URL swap, so it takes
    effect on the next request with no reload. Called at the top of every run
    rather than only on change -- that way a run always uses what the dropdown
    said when Start was pressed, even if the two got out of sync.
    """
    key = key or CFG.get("model") or P.DEFAULT_MODEL
    url, name = _model_endpoint(key)
    ev.vllm_server_url = url
    ev.args.vllm_model_name = name
    return P.MODELS.get(key, P.MODELS[P.DEFAULT_MODEL])[0]


def _endpoint_serves_model(url, want, timeout=3.0):
    """Return whether an OpenAI-compatible endpoint serves ``want``."""
    try:
        with urllib.request.urlopen(url + "/v1/models", timeout=timeout) as r:
            served = {m.get("id") for m in json.load(r).get("data", [])}
    except Exception:
        return False
    if want not in served:
        print(f"!! {url} 服务的是 {sorted(served)}, 期望 {want} !!", flush=True)
        return False
    return True


def model_alive(key, timeout=3.0):
    """True if that backend answers /v1/models *and* serves the expected name.

    Checking the name and not just the port is the point: both backends used to
    be served as "default", so pointing the UI at the wrong URL still produced
    fluent answers -- from the other model, with nothing anywhere saying so.
    Now a mismatch fails here instead of silently scoring the wrong model.
    """
    url, want = _model_endpoint(key)
    return _endpoint_serves_model(url, want, timeout=timeout)


def brain_readiness():
    """Prepare both independent frontends before entry; no head is selected."""
    key = CFG.get("model") or P.DEFAULT_MODEL
    if P.is_fm_model(key) or P.is_lightnav_model(key):
        key = P.DEFAULT_MODEL
    ready = model_alive(key, timeout=2.0)
    encoder = manual_head_status()
    encoder_state = encoder.get('encoder_state', 'unloaded')
    if encoder_state in ('loading', 'error'):
        phase = encoder.get('encoder_error') if encoder_state == 'error' else encoder.get('encoder_phase')
        return dict(ready=False, model=key, label=P.MODELS[key][0], brain_ready=ready,
                    encoder=encoder, message=('大脑已就绪；' if ready else '大脑加载中；')
                    + f"小脑编码器：{phase}（{encoder.get('encoder_elapsed_s', 0):.0f} 秒）")
    return dict(ready=ready, model=key, label=P.MODELS[key][0],
                message=("大脑已就绪，正在进入控制台…" if ready else
                         "正在等待大脑服务就绪，加载完成后自动进入控制台。"))


def _action_head_slow_endpoint():
    """Resolve the vLLM endpoint shared by action expert and plain ReAct."""
    urls = os.environ.get("RYNNVLN_VLLM_SERVER_URLS", "").split(",")
    url = next((value.strip() for value in urls if value.strip()), "")
    settings = _action_head_settings()
    if not url:
        url = str(settings.get("slow_vllm_url") or "")
    if not url:
        raise RuntimeError("Action Head 配置缺少 slow_vllm_url")
    name = os.environ.get(
        "RYNNVLN_VLLM_MODEL_NAME",
        str(settings.get("slow_vllm_model") or "Qwen3.6-35B"))
    return url.rstrip("/"), name


def _model_endpoint(key):
    """The selected brain alone determines its endpoint, regardless of head state."""
    return P.model_endpoint(key)


class ContextOverflow(RuntimeError):
    """The rendered conversation no longer fits the model's window.

    With env steps uncapped this is the expected way a long ReAct run ends, so
    the driver treats it as an ending rather than an error: everything up to
    that point really happened and deserves the normal episode summary.

    vLLM answers such a request with HTTP 400 and a message naming the two
    numbers, e.g. "This model's maximum context length is 131072 tokens.
    However, you requested 133500 tokens". `.detail` carries that text through
    to the UI banner, trimmed.
    """

    def __init__(self, detail=""):
        super().__init__(detail or "context length exceeded")
        self.detail = detail


# Substrings vLLM uses when a request does not fit. Matched case-insensitively
# against the 400 body; anything else that 400s is a real bug (bad sampling
# params, unknown model) and must not be dressed up as a graceful ending.
_OVERFLOW_MARKS = ("maximum context length", "context_length_exceeded",
                   "longer than the maximum model length",
                   "reduce the length of the messages")


def _classify_http_error(exc):
    """urllib HTTPError -> (ContextOverflow | RuntimeError). Never retried:
    a 400 is a verdict on the request, and resending it verbatim eight seconds
    later just wastes the time before the user sees why."""
    try:
        body = exc.read().decode("utf-8", "replace")
    except Exception:
        body = ""
    detail = body.strip()
    try:  # vLLM wraps it as {"error": {"message": ...}} or {"message": ...}
        obj = json.loads(body)
        err = obj.get("error", obj)
        detail = (err.get("message") if isinstance(err, dict) else None) or detail
    except Exception:
        pass
    detail = " ".join(detail.split())[:300]
    low = detail.lower()
    if exc.code == 400 and any(m in low for m in _OVERFLOW_MARKS):
        return ContextOverflow(detail)
    return RuntimeError(f"HTTP {exc.code}: {detail or exc.reason}")


def _post_json_interruptible(url, data, headers, timeout, abort):
    """POST JSON while allowing Stop to cancel a non-streaming response.

    vLLM exposes hidden-transfer metadata only on the aggregate response, so
    action expert cannot use the normal SSE loop.  Keep that response format but
    wait for it in a helper thread; closing the socket on abort also tells vLLM
    to stop generating the now-unused request.
    """
    parsed = urlparse(url)
    connection_cls = (http.client.HTTPSConnection
                      if parsed.scheme == "https"
                      else http.client.HTTPConnection)
    connection = connection_cls(
        parsed.hostname, parsed.port, timeout=timeout)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    done = threading.Event()
    outcome = {}

    def request_worker():
        try:
            connection.request("POST", path, body=data, headers=headers)
            response = connection.getresponse()
            outcome["status"] = response.status
            outcome["reason"] = response.reason
            outcome["body"] = response.read()
        except Exception as exc:
            outcome["error"] = exc
        finally:
            connection.close()
            done.set()

    threading.Thread(target=request_worker,
                     name="action-head-vllm-request", daemon=True).start()
    while not done.wait(0.1):
        if not abort():
            continue
        try:
            if connection.sock is not None:
                connection.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        connection.close()
        return None, True
    if "error" in outcome:
        raise outcome["error"]
    body = outcome.get("body", b"")
    status = int(outcome.get("status", 0))
    if status >= 400:
        raise urllib.error.HTTPError(
            url, status, outcome.get("reason", ""), None,
            io.BytesIO(body))
    return json.loads(body.decode("utf-8")), False


def stream_chat(ev, conv, mode, temperature, on_delta, abort,
                preserve_thinking=False, capture_queries=False, tool_specs=None):
    """POST to an OpenAI-compatible SSE endpoint (vLLM or SGLang).

    ``reasoning_content`` and ordinary ``content`` are kept as separate streams.
    SGLang's qwen3 reasoning parser emits the former; vLLM models may instead
    put an in-band ``<think>`` block in the latter, which callers normalize with
    :func:`agent_tools.split_thinking`. Keeping the channels separate ensures
    reasoning can be displayed without ever being parsed as a tool call.

    Sampling comes from P.MODES[mode] (see curl-Instruct.sh / curl-Thinking.sh),
    not from the evaluator defaults -- those capped max_tokens at 2048, which
    truncated reasoning replies before they reached the "答：" line.

    preserve_thinking is what actually makes THINKING_HISTORY="keep_thinking"
    work: without it both chat templates re-split every assistant message on
    </think> and drop the reasoning, so a raw reply in history renders exactly
    like a stripped one. Only the ReAct path passes it; the classic path is
    single-turn, so there is no history for it to affect.

    Returns ``(content, reasoning, aborted, finish_reason)``. Also records the
    server's token accounting in LAST_USAGE. Raises on transport failure so the
    caller can surface it in the UI.
    """
    params, thinking = P.sampling_params(mode, temperature)
    tpl_kwargs = {"enable_thinking": thinking}
    if preserve_thinking:
        tpl_kwargs["preserve_thinking"] = True
    payload = {
        "model": ev.args.vllm_model_name,
        "messages": ev._to_openai_messages(conv),
        "stream": True,
        # include_usage adds a final chunk carrying prompt_tokens -- the only
        # honest measure of how big the conversation actually got, since the
        # client cannot count what the template renders.
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": tpl_kwargs,
    }
    if tool_specs:
        payload['tools'] = [dict(type='function', function=dict(name=spec.name,
            description=spec.description, parameters=spec.parameters)) for spec in tool_specs]
        payload['tool_choice'] = 'auto'
    global LAST_QUERY_REQUEST_ID, LAST_INFERENCE_TIMING
    LAST_QUERY_REQUEST_ID = ""
    LAST_INFERENCE_TIMING = {}
    capture_queries = capture_queries and getattr(ACTION_HEAD_AGENT_RUNTIME, "requires_query_tail", True)
    if capture_queries:
        query_count = int(getattr(
            ACTION_HEAD_AGENT_RUNTIME, "query_count", AHV.DEFAULT_QUERY_COUNT))
        LAST_QUERY_REQUEST_ID = AHV.request_queries(payload, query_count)
    payload.update(params)

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if ev.args.vllm_api_key:
        headers["Authorization"] = f"Bearer {ev.args.vllm_api_key}"
    url = f"{ev.vllm_server_url}/v1/chat/completions"

    last_err = None
    for attempt in range(1, max(1, ev.args.vllm_retries) + 1):
        request_started = time.perf_counter()
        tool_ready_at = None
        raw_content = ""
        query_carriers = 0
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        chunks, reasoning_chunks = [], []
        native_calls = {}
        aborted, got_any, finish = False, False, None
        LAST_USAGE.clear()
        try:
            resp = urllib.request.urlopen(req, timeout=ev.args.vllm_timeout)
        except urllib.error.HTTPError as exc:
            # The server answered, and its answer is that the request is wrong.
            # Raise straight away -- notably ContextOverflow, which run_agent
            # catches and turns into a clean end-of-episode.
            raise _classify_http_error(exc) from None
        except Exception as exc:  # connection-level failure -> retry
            last_err = exc
            time.sleep(min(2 ** attempt, 8))
            continue
        try:
            for raw in resp:
                if abort():
                    aborted = True
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except Exception:
                    continue
                # The usage chunk arrives last and carries choices: [], so read
                # it before the choices[0] access below.
                if obj.get("usage"):
                    LAST_USAGE.update(obj["usage"])
                choices = obj.get("choices") or [{}]
                fr = choices[0].get("finish_reason")
                if fr:
                    finish = fr
                delta = choices[0].get("delta") or {}
                for call in delta.get('tool_calls') or []:
                    target = native_calls.setdefault(call.get('index', 0), dict(name='', arguments=''))
                    function = call.get('function') or {}
                    target['name'] += function.get('name') or ''
                    target['arguments'] += function.get('arguments') or ''
                    got_any = True
                reasoning_piece = (delta.get("reasoning_content")
                                   or delta.get("reasoning") or "")
                content_piece = delta.get("content") or ""
                raw_content += content_piece
                query_carriers += content_piece.count("<|fim_prefix|>")
                if tool_ready_at is None and "</tool_call>" in raw_content:
                    tool_ready_at = time.perf_counter()
                if capture_queries and content_piece:
                    # Query placeholders are model-internal carrier tokens.
                    # Keep them out of the visible stream and replay history;
                    # their final hidden outputs arrive through the side cache.
                    content_piece = content_piece.replace("<|fim_prefix|>", "")
                if reasoning_piece:
                    got_any = True
                    reasoning_chunks.append(reasoning_piece)
                    on_delta(reasoning_piece, "reasoning")
                if content_piece:
                    got_any = True
                    chunks.append(content_piece)
                    on_delta(content_piece, "content")
        except Exception as exc:
            if not got_any:
                last_err = exc
                try:
                    resp.close()
                except Exception:
                    pass
                time.sleep(min(2 ** attempt, 8))
                continue
            # partial output already shown; keep whatever arrived
        finally:
            try:
                resp.close()
            except Exception:
                pass
        if native_calls and not aborted:
            for _, call in sorted(native_calls.items()):
                # Only complete native calls become executable; truncated JSON
                # remains a transport/protocol failure, never a partial action.
                arguments = json.loads(call['arguments'])
                if not call['name'] or not isinstance(arguments, dict):
                    raise ValueError('invalid native navigation tool call')
                content = '<tool_call>' + json.dumps(dict(name=call['name'], arguments=arguments)) + '</tool_call>'
                chunks.append(content)
                on_delta(content, 'content')
        ended = time.perf_counter()
        LAST_INFERENCE_TIMING = {
            "request_wall_ms": (ended - request_started) * 1000,
            "tool_text_ready_ms": ((tool_ready_at-request_started)*1000 if tool_ready_at else None),
            "post_tool_stream_ms": ((ended-tool_ready_at)*1000 if tool_ready_at else None),
            "tool_ready_perf": tool_ready_at,
            "query_tail_requested": bool(capture_queries),
            "query_carriers_observed": query_carriers,
            "messages": len(payload["messages"]),
            "prompt_tokens": LAST_USAGE.get("prompt_tokens"),
            "messages_sha256": hashlib.sha256(json.dumps(
                payload["messages"], sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "images": sum(part.get("type") == "image_url"
                          for message in payload["messages"]
                          for part in (message["content"] if isinstance(message.get("content"), list) else [])
                          if isinstance(part, dict)),
        }
        timing_root = os.environ.get("ACTION_HEAD_TIMING_TRACE_DIR", "")
        if timing_root:
            trace_dir = pathlib.Path(timing_root)
            trace_dir.mkdir(parents=True, exist_ok=True)
            with (trace_dir / "slow_requests.jsonl").open("a") as trace:
                trace.write(json.dumps({"driver": CFG.get("driver"),
                    "scene": STATE.get("scene"), "episode_id": STATE.get("episode_id"),
                    "time_ns": time.time_ns(), "completion": "".join(chunks),
                    "usage": dict(LAST_USAGE), **LAST_INFERENCE_TIMING}, ensure_ascii=False) + "\n")
        return "".join(chunks), "".join(reasoning_chunks), aborted, finish
    raise RuntimeError(
        f"vLLM streaming failed after {ev.args.vllm_retries} attempts: {last_err}")


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):

    def _is_agent_control(self):
        """True only for control calls emitted by the pi-agent extension."""
        return self.headers.get(AGENT_CONTROL_HEADER, "") == AGENT_CONTROL_VALUE

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, body, ctype, code=200):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/readiness":
            self._send(json.dumps(brain_readiness(), ensure_ascii=False),
                       "application/json; charset=utf-8")
        elif u.path == "/frame.jpg":
            self._img(STATE["frame_jpg"])
        elif u.path == "/frame-snapshot":
            try:
                payload = frame_snapshot_payload()
                self._send(json.dumps(payload, ensure_ascii=False),
                           "application/json; charset=utf-8")
            except ValueError as exc:
                self._send(json.dumps({"ok": False, "err": str(exc)},
                                      ensure_ascii=False),
                           "application/json; charset=utf-8", 404)
        elif u.path == "/vln-map-snapshot":
            try:
                payload = map_snapshot_payload()
                self._send(json.dumps(payload, ensure_ascii=False),
                           "application/json; charset=utf-8")
            except ValueError as exc:
                self._send(json.dumps({"ok": False, "err": str(exc)},
                                      ensure_ascii=False),
                           "application/json; charset=utf-8", 404)
        elif u.path == "/react-reference.jpg":
            self._img(react_reference_jpeg())
        elif u.path.startswith("/debug-image/") and u.path.endswith(".jpg"):
            try:
                image_id = int(u.path.rsplit("/", 1)[1][:-4])
            except ValueError:
                image_id = -1
            data = debug_image(image_id)
            if data is None:
                self._send("debug image not found", "text/plain; charset=utf-8", 404)
            else:
                self._img(data)
        elif u.path.startswith("/debug-request/"):
            try:
                request_id = int(u.path.rsplit("/", 1)[1])
            except ValueError:
                request_id = -1
            request = debug_request(request_id)
            if request is None:
                self._send(json.dumps({"ok": False, "err": "调试快照已过期"},
                                      ensure_ascii=False),
                           "application/json; charset=utf-8", 404)
            else:
                self._send(json.dumps({"ok": True, "request": request},
                                      ensure_ascii=False),
                           "application/json; charset=utf-8")
        elif u.path == "/map.jpg":
            self._img(STATE["map_jpg"])
        elif u.path == "/vln-map.jpg":
            self._img(STATE["tool_map_jpg"])
        elif u.path == "/stream.mjpg":
            self._stream("display_frame_jpg", "display_seq")
        elif u.path == "/stream_map.mjpg":
            self._stream("map_jpg", "map_seq")
        elif u.path == "/status":
            self._send(json.dumps(self._status(q), ensure_ascii=False),
                       "application/json; charset=utf-8")
        elif u.path == "/terminal":
            try:
                since = int(q.get("since", ["0"])[0])
                generation = int(q.get("generation", ["0"])[0])
                wait = float(q.get("wait", ["15"])[0])
                result = TERMINAL.snapshot(since, generation, wait)
                self._send(json.dumps(result), "application/json; charset=utf-8")
            except (ValueError, OSError) as exc:
                self._send(json.dumps({"ok": False, "err": str(exc)},
                                      ensure_ascii=False),
                           "application/json; charset=utf-8", 400)
        elif u.path == "/cmd":
            self._send(self._cmd(q, agent_control=self._is_agent_control()),
                       "text/plain; charset=utf-8")
        elif u.path == "/action-head/checkpoints":
            try:
                payload = AHC.browse_checkpoints(q.get('path', [''])[0])
            except (OSError, ValueError) as exc:
                payload = dict(ok=False, error=str(exc))
            self._send(json.dumps(payload, ensure_ascii=False),
                       'application/json; charset=utf-8')
        elif u.path == "/action-head/catalog":
            try:
                payload = AHC.checkpoint_catalog(refresh=q.get('refresh', ['0'])[0] == '1')
            except (OSError, ValueError) as exc:
                payload = dict(ok=False, error=str(exc))
            self._send(json.dumps(payload, ensure_ascii=False),
                       'application/json; charset=utf-8')
        elif u.path == "/prompt":
            self._send(json.dumps(self._prompt(q), ensure_ascii=False),
                       "application/json; charset=utf-8")
        elif u.path == "/config.json":
            # everything the page needs that only the running console knows:
            # which asset sets are mounted, their scenes and episodes, the
            # dropdown registries. Built once in main().
            self._send(json.dumps(self.server.web_config, ensure_ascii=False),
                       "application/json; charset=utf-8")
        elif u.path == "/traj_cases":
            # One reference trajectory's full instruction set (sage backend):
            # read-only, no lock (it only queries the dataset).
            self._send(json.dumps(self._traj_cases(q), ensure_ascii=False),
                       "application/json; charset=utf-8")
        elif u.path.startswith("/static/"):
            got = webui.static(u.path[len("/static/"):])
            if got is None:
                self._send("not found", "text/plain; charset=utf-8", 404)
            else:
                self._send(got[0], got[1])
        else:
            self._send(webui.page(), "text/html; charset=utf-8")

    def do_POST(self):
        """Handle body-sized console settings: system prompt and ReAct image."""
        u = urlparse(self.path)
        if u.path not in ("/sysprompt", "/react-reference", "/terminal"):
            self._send("not found", "text/plain; charset=utf-8", 404)
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        max_body = TERMINAL_MAX_INPUT * 2 if u.path == "/terminal" \
            else MAX_REFERENCE_POST_BYTES
        if n < 0 or n > max_body:
            self.close_connection = True
            self._send(json.dumps({"ok": False, "err": "请求体过大"},
                                  ensure_ascii=False),
                       "application/json; charset=utf-8", 413)
            return
        raw = self.rfile.read(n) if n > 0 else b""
        # The terminal is a standalone PTY, not a simulator control. It is kept
        # outside the page lock on purpose so locking VLN controls cannot freeze
        # an active shell halfway through a command.
        if u.path == "/terminal":
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
                op = payload.get("op")
                if op == "restart":
                    result = TERMINAL.restart(payload.get("cols", 100),
                                              payload.get("rows", 30))
                elif op == "reset_workspace":
                    origin = self.headers.get('Origin')
                    if origin and urlparse(origin).netloc != self.headers.get('Host'):
                        raise ValueError('工作区重置只接受同源页面请求')
                    result = TERMINAL.reset_workspace(payload.get("generation", -1),
                                                      payload.get("cols", 100), payload.get("rows", 30))
                elif op == "input":
                    if "binary" in payload:
                        data = base64.b64decode(payload["binary"], validate=True)
                    else:
                        data = str(payload.get("data") or "").encode("utf-8")
                    TERMINAL.write(data)
                    result = {"ok": True}
                elif op == "resize":
                    TERMINAL.resize(payload.get("cols", 100),
                                    payload.get("rows", 30))
                    result = {"ok": True}
                else:
                    raise ValueError("terminal op must be input, resize, restart or reset_workspace")
                self._send(json.dumps(result), "application/json; charset=utf-8")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError,
                    binascii.Error, OSError, RuntimeError) as exc:
                self._send(json.dumps({"ok": False, "err": str(exc)},
                                      ensure_ascii=False),
                           "application/json; charset=utf-8", 400)
            return
        # 锁住时普通网页不能改参考图或 system prompt；pi-agent 设置 bbox/参考图是
        # 执行移动的一部分，所以带内部来源标记的 /react-reference 继续放行。先读完
        # body 再拒，不然 keep-alive 连接里的字节会被当成下一条请求的 HTTP 头。
        with LOCK:
            locked = STATE["locked"]
        agent_control = (u.path == "/react-reference"
                         and self._is_agent_control())
        if locked and not agent_control:
            self._send(json.dumps({"ok": False, "locked": True,
                                   "err": LOCKED_MSG}, ensure_ascii=False),
                       "application/json; charset=utf-8")
            return
        if u.path == "/react-reference":
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
                meta = set_react_reference(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._send(json.dumps({"ok": False, "err": str(exc)},
                                      ensure_ascii=False),
                           "application/json; charset=utf-8", 400)
                return
            if meta["active"]:
                if meta["kind"] == "upload":
                    label = f"上传图 {meta['name']}"
                elif meta["kind"] == "point":
                    label = f"带视觉点的当前画面 {meta['point']}"
                else:
                    label = f"带红框的当前画面 {meta['bbox']}"
                push_event("sys", text=f"已设置移动参考图：{label}")
            else:
                push_event("sys", text="已清除移动参考图")
            self._send(json.dumps({"ok": True, "reference": meta}, ensure_ascii=False),
                       "application/json; charset=utf-8")
            return

        body = raw.decode("utf-8", "replace")
        text, err = save_sysprompt(body)
        # Logged as an event so the run log records which prompt a run used --
        # otherwise a custom prompt is invisible in the transcript and a strange
        # run is impossible to explain after the fact.
        if text:
            push_event("sys", text="已启用自定义 system prompt（%d 字）%s"
                                   % (len(text),
                                      "" if P.prompt_has_tools(text)
                                      else "⚠ 里面没有工具说明，模型可能不会调用工具"))
        else:
            push_event("sys", text="system prompt 已恢复默认")
        self._send(json.dumps({"ok": True, "custom": bool(text),
                               "len": len(text),
                               "tools": P.prompt_has_tools(
                                   P.agent_system_prompt(
                                       (None if AHP.protocol_module(
                                           action_head_tool_version()) is not None
                                        else text),
                                       tool_specs=active_tool_profile().tool_specs,
                                       action_head_as_tool_version=(
                                           action_head_tool_version()))),
                               "saved": not err, "err": err},
                              ensure_ascii=False),
                   "application/json; charset=utf-8")

    def _img(self, data):
        if data is None:
            self._send(b"", "image/jpeg", 204)
        else:
            self._send(data, "image/jpeg")

    def _stream(self, jpg_key, seq_key):
        """Push frames as multipart JPEG, one part per frame produced.

        This is the frame-rate fix. The view used to be repainted by the 250 ms
        status poll -- the browser only ever saw four frames a second no matter
        how many the simulator rendered, and every one of them cost an extra
        round trip. Here the frame goes out the moment publish_frame() wakes the
        condition, so what
        you watch is the simulator's real rate (~12 fps on the Gaussian
        backends, whatever software GL manages on habitat).

        Costs one thread per open stream for as long as the tab is open, which
        is affordable: the console is single-user and the server is a
        ThreadingHTTPServer with daemon threads. The 2 s wake-up is a keepalive
        -- it re-sends the current frame so a client that died quietly is
        noticed via the write instead of parking a thread forever.
        """
        self.close_connection = True     # a multipart stream has no length
        bound = b"vlnframe"
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=" + bound.decode())
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        try:
            self.wfile.write(b"--" + bound + b"\r\n")
            while not SHUTDOWN.is_set():
                with FRAME_CV:
                    if STATE[seq_key] == last:
                        FRAME_CV.wait(2.0)
                    data, last = STATE[jpg_key], STATE[seq_key]
                if data is None:
                    time.sleep(0.1)
                    continue
                # Complete the part now: browsers may otherwise hold a still
                # frame until the next frame supplies its trailing boundary.
                self.wfile.write(b"Content-Type: image/jpeg"
                                 b"\r\nContent-Length: "
                                 + str(len(data)).encode() + b"\r\n\r\n"
                                 + data + b"\r\n--" + bound + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            pass

    def _status(self, q):
        try:
            since = int(q.get("since", ["0"])[0])
        except ValueError:
            since = 0
        with LOCK:
            s = {k: STATE[k] for k in ("instruction", "status", "step", "dist",
                                       "result", "scene", "episode_id", "seq",
                                       "map_seq", "map_heading_deg", "manual", "locked",
                                       "actions", "stream", "motion", "moves",
                                       "driver", "model", "action_head_as_tool_version",
                                       "react_reference", "evaluation",
                                       "backend", "backend_label",
                                       "scene_title", "scene_name")}
            evs = list(EVENTS)
            ev_next, ev_min = EV_NEXT, EV_MIN
        s["fps"] = frame_rate()
        s["golden_step"] = float(CFG.get("golden_step") or 0.1)
        s["golden_stride"] = CFG["golden_stride"]
        s["pointgoal_lookahead"] = float(CFG.get("pointgoal_lookahead") or 3.0)
        s['action_head'] = manual_head_status()
        s['invalid_tool_call_policy'] = 'feedback'
        if not isinstance(s["dist"], float) or math.isnan(s["dist"]):
            s["dist"] = None
        reset_log = since < ev_min or since > ev_next
        s["events"] = evs if reset_log else [e for e in evs if e["i"] >= since]
        s["since"] = ev_next
        s["reset_log"] = reset_log
        s["st"] = P.action_stats(s["actions"])
        return s

    def _prompt(self, q):
        """Render the prompt exactly as the active driver would, for preview.

        Uses the live instruction / action trail when the caller doesn't
        override them, so the preview shows the real prompt -- including the
        action-history block as it currently stands -- not a canned sample.
        """
        driver = q.get("driver", [""])[0] or CFG["driver"]
        instr = (q.get("instruction", [""])[0].strip()
                 or CFG["instruction"] or STATE["instruction"]
                 or "（这里会填入你输入的导航指令）")
        if (driver == "react" or is_action_head_driver(driver)
                or driver in ("embodied_point", "unilavira_bbox")):
            # The ReAct driver has no per-turn template: the system prompt plus
            # the first user turn is the whole of what it is told up front.
            label, desc = P.DRIVERS[driver][0], P.DRIVERS[driver][1]
            reference_kind = q.get("reference_kind", [""])[0]
            if reference_kind not in ("upload", "point", "bbox"):
                reference_kind = None
            profile = active_tool_profile(driver)
            version = action_head_tool_version(driver)
            modern = AHP.protocol_module(version) is not None
            sys_edit_text = P.agent_system_prompt(
                None if modern else CFG["sysprompt"],
                tool_specs=profile.tool_specs,
                action_head_as_tool_version=version)
            sys_text = P.agent_system_prompt(
                None if modern else CFG["sysprompt"],
                None if modern else reference_kind,
                tool_specs=profile.tool_specs,
                action_head_as_tool_version=version)
            # Labelled with the role names so it is unambiguous that these are
            # two separate messages (see the msgs list in run_agent) rather than
            # one blob of text. The labels exist only in this preview -- the
            # request itself carries the two bodies with nothing prepended.
            if reference_kind == "upload":
                user_preview = ("[当前起点第一人称画面]\n"
                                + P.agent_current_view_text() + "\n\n"
                                + "[额外参考图]\n\n"
                                + P.agent_task_text(instr))
            elif reference_kind == "bbox":
                user_preview = ("[当前第一视角画面｜已绘制红色 bbox]\n"
                                "上面这张图片是机器人当前看到的第一视角画面。"
                                "用户指定的 bbox 已作为红色矩形框直接绘制在图片上。\n"
                                "红框内的物体或区域是本次局部移动的视觉目标；"
                                "请结合完整画面判断可行路径。\n\n"
                                + "bbox 使用 0 到 1000 归一化整数坐标：[x1,y1,x2,y2]；"
                                  "左上角为 (0,0)，x 向右增大，y 向下增大。\n\n"
                                + P.agent_task_text(instr))
            elif reference_kind == "point":
                user_preview = ("[当前第一视角画面｜已绘制红色视觉 point]\n"
                                "point 是当前画面中的视觉目标像素；只要该像素具有"
                                "有效同帧深度即可使用。\n"
                                "point 使用 0 到 1000 归一化整数坐标 [x,y]；"
                                "左上角为 (0,0)，x 向右增大，y 向下增大。\n\n"
                                + P.agent_task_text(instr))
            else:
                user_preview = ("[当前起点第一人称画面]\n"
                                + P.agent_current_view_text() + "\n\n"
                                + P.agent_task_text(instr))
            text = ("system prompt：\n\n" + sys_text + "\n\n"
                    + "user prompt：\n\n" + user_preview)
            # Read the budget off the query like nact below, not off CFG: the page
            # fires set_maxtools and loadPrompt together, so trusting CFG would
            # show whichever request happened to land first.
            try:
                mt = max(P.TOOL_BUDGET_MIN,
                         min(P.TOOL_BUDGET_MAX, int(q.get("maxtools", [""])[0])))
            except (ValueError, TypeError):
                mt = CFG["maxtools"]
            return {"tpl": "react", "label": label, "desc": desc,
                    "nact": P.MAX_CALLS_PER_TURN, "nhist": 1,
                    "maxtools": mt,
                    # sys_text is handed over separately from `text` because the
                    # page also uses it to prefill the editor -- feeding the
                    # labelled blob back in would save "system prompt：" as part
                    # of the prompt.
                    "sys_text": sys_text,
                    "sys_edit_text": sys_edit_text,
                    "reference_kind": reference_kind or "none",
                    "sys_custom": bool(CFG["sysprompt"]),
                    "sys_tools": P.prompt_has_tools(sys_edit_text),
                    "instruction": instr, "text": text}
        if driver == "fm":
            # The FM driver builds its prompt inside FMInfer (SocialNav's
            # "<image> <input_waypoint> anonymous Instruction: ... Predict the goal
            # point and trajectory." format), so there is no template/text to
            # edit here. Show what will be fed and the model ckpt instead.
            label, desc = P.DRIVERS["fm"][0], P.DRIVERS["fm"][1]
            ckpt = (CFG.get("fm_ckpt") or os.environ.get("FM_CKPT")
                    or "<fm_policy 默认 FM_CKPT>")
            text = (f"FM 头（进程内加载，不走 vLLM）：\nckpt: {ckpt}\n"
                    f"指令: {instr}\n"
                    f"每步：当前画面 + 零 input_waypoint → 预测 32×3 ego 轨迹 → "
                    f"denorm → 逐 waypoint 落 sim（turn→fwd→turn 分解 + 子步防穿墙）")
            return {"tpl": "fm", "label": label, "desc": desc,
                    "nact": 32, "nhist": 1, "maxtools": 0,
                    "instruction": instr, "text": text}
        if driver == "lightnav":
            label, desc = P.DRIVERS["lightnav"][0], P.DRIVERS["lightnav"][1]
            url = CFG.get("lightnav_url") or LND.DEFAULT_URL
            text = (f"LightNav-0 独立策略服务：\nserver: {url}\n"
                    f"指令: {instr}\n"
                    "每轮：当前干净 RGB + 指令 → 服务端 SlowFast 历史 → "
                    "10×3 [forward, lateral, yaw] waypoint；控制台执行首个有效点。\n"
                    "原生 Habitat 投影：dt=0.1s，forward∈[0,0.25]m，"
                    "yaw∈[-30°,30°]，lateral 丢弃。")
            return {"tpl": "lightnav", "label": label, "desc": desc,
                    "nact": 1, "nhist": 64, "maxtools": 0,
                    "instruction": instr, "text": text}
        tpl = q.get("tpl", [""])[0] or CFG["tpl"]
        if tpl not in P.TEMPLATES:
            tpl = P.DEFAULT_TEMPLATE
        try:
            nact = max(1, min(8, int(q.get("nact", [""])[0])))
        except (ValueError, TypeError):
            nact = CFG["nact"]
        with LOCK:
            actions = list(STATE["actions"])
        label, desc, _ = P.TEMPLATES[tpl]
        # num_history only affects the "you see N images" wording. The real
        # value lives on the evaluator (main()-local), so mirror its default
        # here -- close enough for a preview, and it never gates behaviour.
        nhist = min(len(actions), 16)
        return {"tpl": tpl, "label": label, "desc": desc,
                "nact": nact, "nhist": nhist, "instruction": instr,
                "text": P.build_prompt(tpl, instr, nhist, nact, actions)}

    def _traj_cases(self, q):
        """One reference trajectory's full instruction cases (sage backend).

        Read-only, lock-free: it only asks the dataset, never changes state.
        Frontend calls it when the user opens the \"全部用例\" picker; refused
        for backends that do not ship per-trajectory instruction sets.
        """
        scene = q.get("scene", [""])[0].strip()
        traj = q.get("traj", [""])[0].strip()
        dimension = q.get("dimension", [""])[0].strip().lower()
        if not scene or not traj:
            return {"ok": False, "err": "need scene+traj"}
        env = ENVS.get(backend_of(scene))
        fn = getattr(env, "traj_cases", None)
        if fn is None:
            return {"ok": False, "err": "该后端不支持按轨迹取用例"}
        try:
            return {"ok": True, "scene": scene, "traj": traj,
                    "dimension": dimension,
                    "cases": fn(scene, traj, dimension=dimension or None)}
        except Exception as exc:
            return {"ok": False, "err": str(exc)}

    def _cmd(self, q, agent_control=False):
        with HEAD_CONTROL_LOCK:
            return self._cmd_serial(q, agent_control=agent_control)

    def _cmd_serial(self, q, agent_control=False):
        do = q.get("do", [""])[0]

        # 锁在这里把关，而不是只在页面上。前端那道拦截是为了给出提示，可它挡不住
        # 一个开着的旧标签页、一个后退回来的历史页面，或者手敲的 URL —— 那些恰恰
        # 是「agent 在调试时被人从背后改掉参数」最容易发生的方式。lock/unlock 自己
        # 必须放行，否则锁上就没法解开了。pi-agent 工具带内部来源标记，也不受这个
        # 只针对人工网页操作的锁影响。
        if do not in ("lock", "unlock") and not agent_control:
            with LOCK:
                if STATE["locked"]:
                    return "locked: " + LOCKED_MSG

        if do == 'load_action_head':
            try:
                load_manual_head(q.get('path', [''])[0])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return str(exc)
            return 'ok'
        if do in ('start', 'manual_start', 'set_driver', 'set_model'):
            if HEAD_EPISODE_ACTIVE.is_set() or STATE['status'] == 'running':
                return '请先停止当前回合'
        if do in ('start', 'manual_start'):
            wanted = P.canonical_driver(q.get('driver', [''])[0] or CFG['driver'])
            if wanted == 'action_head':
                head = manual_head_status()
                if head['state'] == 'loading':
                    return '小脑正在加载，请等待完成'
                if not head['active']:
                    return '请先选择小脑 checkpoint 并确认加载'

        def _num(key, cast, lo, hi, default):
            try:
                return max(lo, min(hi, cast(q.get(key, [""])[0])))
            except (ValueError, TypeError):
                return default

        def _temp(key):
            """'' or 'auto' -> None (follow the mode preset), else clamp."""
            raw = q.get(key, [""])[0].strip().lower()
            if raw in ("", "auto"):
                return None
            try:
                return max(0.0, min(2.0, float(raw)))
            except ValueError:
                return CFG["temp"]

        if do in ("lock", "unlock"):
            want = do == "lock"
            with LOCK:
                changed = STATE["locked"] != want
                STATE["locked"] = want
                if want:
                    # 锁上的瞬间松开手动驾驶：不然按着 W 时按锁，后面的心跳全被拒，
                    # 机器人要一直走到 HOLD_TIMEOUT 才停。
                    HOLD["keys"] = frozenset()
                    HOLD["t"] = time.time()
            if changed:
                push_event("sys", text=("🔒 已锁定前端：" + LOCKED_MSG) if want
                                       else "🔓 已解锁前端，恢复人工操作")
            return "ok"
        if do in ("start", "manual_start"):
            # Blank box means "use the episode's own instruction", so clear the
            # override instead of keeping the last one -- otherwise emptying the
            # box would silently re-run whatever was typed before it. The worker
            # falls back to ep.instruction.instruction_text when this is "".
            CFG["instruction"] = q.get("instruction", [""])[0].strip()
            tpl = q.get("tpl", [""])[0]
            if tpl in P.TEMPLATES:
                CFG["tpl"] = tpl
            mode = q.get("mode", [""])[0]
            if mode in P.MODES:
                CFG["mode"] = mode
            drv = P.canonical_driver(q.get("driver", [""])[0])
            if drv in P.DRIVERS:
                # An explicit pick wins: drop any driver guard_driver() stashed
                # away, so returning to a goal-bearing asset set does not undo it.
                CFG.pop("forced_from", None)
                CFG["driver"] = drv
                set_state(driver=drv)
            mdl = q.get("model", [""])[0]
            if mdl in P.MODELS:
                CFG["model"] = mdl
                set_state(model=mdl)
            # 模型→驱动锁定（防前端绕过）：FM 模型锁 fm+ckpt；非 FM 模型在 fm 则退回。
            # 在 set_driver 之后再 enforce 一次，模型选择胜出。
            _enforce_model_driver_lock(announce=True)
            # ``Start`` may carry the driver's first action expert selection in
            # the same request, bypassing the separate set_driver endpoint.
            # Begin the non-blocking lazy load here, before queueing the run,
            # so the model log immediately records loading/completion/failure.
            if is_action_head_driver(CFG["driver"]):
                request_action_head_load()
            CFG["nact"] = _num("nact", int, 1, 8, CFG["nact"])
            CFG["maxtools"] = _num("maxtools", int, P.TOOL_BUDGET_MIN,
                                   P.TOOL_BUDGET_MAX, CFG["maxtools"])
            if "temp" in q:
                CFG["temp"] = _temp("temp")
            if "frompose" in q:
                CFG["frompose"] = q["frompose"][0] == "1"
            CMD["start" if do == "start" else "manual_start"] = True
            return "ok"
        if do == "point_move":
            if not agent_control:
                return "point_move 只允许 pi-agent 调用"
            try:
                point = VNT.parse_point_2d([
                    q.get("x", [""])[0], q.get("y", [""])[0]])
                source_seq = int(q.get("seq", [""])[0])
            except (TypeError, ValueError) as exc:
                return f"bad point_move: {exc}"
            with LOCK:
                current_source = (STATE["scene"], STATE["episode_id"],
                                  STATE["seq"])
                reference = dict(STATE["react_reference"])
            requested_source = (
                q.get("scene", [""])[0], q.get("episode_id", [""])[0],
                source_seq)
            if requested_source != current_source:
                return "point_move 观察画面已过期，请重新 vln_observe"
            if (reference.get("kind") != "point"
                    or [float(v) for v in reference.get("point") or []] != point
                    or (reference.get("scene"), reference.get("episode_id"),
                        reference.get("seq")) != current_source):
                return "point_move 缺少匹配当前画面的同帧 point"
            CMD["point_move"] = {
                "point": point,
                "instruction": q.get("instruction", [""])[0].strip(),
                "source": current_source,
            }
            return "ok"
        if do == "screen_point_move":
            try:
                point = VNT.parse_point_2d([
                    q.get("x", [""])[0], q.get("y", [""])[0]])
                with LOCK:
                    source = {key: CURRENT_FRAME[key]
                              for key in ("scene", "episode_id", "seq")}
                set_react_reference({
                    "kind": "point", "point": point, "source": source,
                })
            except (TypeError, ValueError) as exc:
                message = f"双击 point 未执行：{exc}"
                push_event("err", text=message)
                return message
            current_source = (source["scene"], source["episode_id"],
                              source["seq"])
            CMD["point_move"] = {
                "point": point,
                "instruction": "移动到网页双击指定的视觉点。",
                "source": current_source,
            }
            return "ok"
        if do == "stop":
            CMD["stop"] = True
            return "ok"
        if do == "reset":
            CMD["reset"] = True
            return "ok"
        if do == "clearlog":
            CMD["clearlog"] = True
            return "ok"
        if do == "walk_golden":
            # This is a SAGE dataset/debugging facility, not a generic movement
            # command.  Reject it before it reaches the simulator worker: ABotN
            # happens to expose sim.xy/sim.yaw too, but has no SAGE golden-path
            # implementation and used to crash the entire console here.
            if STATE["backend"] != "sage":
                return GOLDEN_ONLY_MSG
            # 沿当前用例的专家(golden)参考轨迹自动行驶到终点：复用 run_fm 的
            # 逐步出画面机制（drive_turn/drive_forward + on_step -> refresh），
            # 不走 vLLM/FM，轨迹来自 env._golden_pts（与 top-down 地图画的橙线同源）。
            CMD["walk_golden"] = True
            return "ok"
        if do == "set_mode":
            mode = q.get("mode", [""])[0]
            if mode in P.MODES:
                CFG["mode"] = mode
            return "ok"
        if do == "set_driver":
            drv = P.canonical_driver(q.get("driver", [""])[0])
            if drv not in P.DRIVERS:
                return "bad driver"
            # 独立策略模型与驱动双向锁定，普通 chat 模型不能偷偷绕过自身端点。
            if drv == "fm" and not P.is_fm_model(CFG.get("model")):
                return "FM·流匹配 驱动需先选 VLN-FM 模型"
            if P.is_fm_model(CFG.get("model")) and drv != "fm":
                return "VLN-FM 模型驱动锁死 FM·流匹配，不可切其他驱动"
            if drv == "lightnav" and not P.is_lightnav_model(CFG.get("model")):
                return "LightNav 驱动需先选择 LightNav-0 模型"
            if P.is_lightnav_model(CFG.get("model")) and drv != "lightnav":
                return "LightNav-0 模型驱动锁死 LightNav-0·轨迹，不可切其他驱动"
            # An explicit pick wins: drop any driver guard_driver() stashed
            # away, so returning to a goal-bearing asset set does not undo it.
            CFG.pop("forced_from", None)
            CFG["driver"] = drv
            set_state(driver=drv)
            if is_action_head_driver(drv):
                request_action_head_load()
            return "ok"
        if do == "set_model":
            mdl = q.get("model", [""])[0]
            if mdl not in P.MODELS:
                return "bad model"
            CFG["model"] = mdl
            set_state(model=mdl)
            # 模型→驱动锁定：FM 模型锁 fm + 设 ckpt；非 FM 模型若在 fm 则退回默认。
            _enforce_model_driver_lock(announce=True)
            return "ok"
        if do == "set_tpl":
            tpl = q.get("tpl", [""])[0]
            if tpl in P.TEMPLATES:
                CFG["tpl"] = tpl
            return "ok"
        if do == "set_nact":
            CFG["nact"] = _num("n", int, 1, 8, CFG["nact"])

        if do == "set_maxtools":
            CFG["maxtools"] = _num("n", int, P.TOOL_BUDGET_MIN,
                                   P.TOOL_BUDGET_MAX, CFG["maxtools"])
            return "ok"
        if do == "set_temp":
            CFG["temp"] = _temp("t")
            return "ok"
        if do == "set_frompose":
            CFG["frompose"] = q.get("on", ["1"])[0] == "1"
            return "ok"
        if do == "set_golden_step":
            # 走专家轨迹的弧长 resample 步长（米），0.02~1.0。run_golden 读
            # CFG["golden_step"]；/status 回带当前值供前端控件回显。
            CFG["golden_step"] = _num("m", float, 0.02, 1.0, CFG["golden_step"])
            return "ok"
        if do == "set_golden_stride":
            stride = _num("n", int, 1, 8, CFG["golden_stride"])
            if stride not in (1, 2, 4, 8):
                return "bad golden stride"
            CFG["golden_stride"] = stride
            return "ok"
        if do == "set_pointgoal_lookahead":
            # Point-Goal 前瞻距离（米），0.5~10。run_golden 投前瞻 waypoint。
            CFG["pointgoal_lookahead"] = _num("m", float, 0.5, 10.0,
                                              CFG["pointgoal_lookahead"])
            return "ok"
        if do == "hold":
            # The whole held set, every heartbeat -- not a delta. See HOLD.
            keys = frozenset(c for c in q.get("k", [""])[0].lower()
                             if c in "wasd")
            with LOCK:
                HOLD["keys"] = keys
                HOLD["t"] = time.time()
            return "ok"
        if do == "manual":
            # Discrete single steps, kept for scripted pokes (curl) and for
            # comparing against the fixed action space the benchmarks use. The
            # page drives with `hold` instead, so W/A/S/D from the browser are
            # continuous; only X still comes through here.
            # "s" is not a sim action -- there is no backward action id (see
            # agent_tools' module docstring). It is a pseudo-action the worker
            # turns into one negative-amount move_forward step.
            amap = {"w": FWD, "a": LEFT, "d": RIGHT, "x": STOP, "s": "back"}
            a = q.get("a", [""])[0]
            if a in amap:
                CMD["manual"] = amap[a]
                return "ok"
            return "bad action"
        if do == "goto":
            # Double-click on the bird's-eye view. Fractions of the displayed
            # image, not pixels: the page has no way to know what one pixel of
            # that JPEG meant, and fractions survive every rescaling in between.
            # The worker does the actual jump -- it owns the simulator.
            try:
                fx = float(q.get("x", [""])[0])
                fy = float(q.get("y", [""])[0])
            except (ValueError, TypeError):
                return "need x+y in 0..1"
            CMD["goto"] = (fx, fy)
            return "ok"
        if do == "switch":
            scene = q.get("scene", [""])[0].strip()
            ep = q.get("episode_id", [""])[0].strip()
            if scene and ep:
                CMD["switch"] = (scene, ep)
                return "ok"
            return "need scene+episode_id"
        return "?"


# ----------------------------------------------------------------------------
# simulator control
# ----------------------------------------------------------------------------
def scene_of(episode):
    """The console's id for an episode's scene.

    Prefer the episode's own ``scene``: ABotN prefixes its ids per benchmark
    (``poi-``/``in-``/``out-``) precisely because four scene directories appear
    in two benchmarks at once, so deriving the id from the path would strip the
    prefix and collide those four. habitat's own episodes have no such field and
    fall back to the path, which is where their id has always come from.
    """
    return getattr(episode, "scene", None) or episode.scene_id.split("/")[-2]


def load_episode(env, episode, args):
    """Point the Env at `episode` and reset. No Env rebuild -- see module docs."""
    # A point/bbox is tied to pixels from the previous episode's current frame.
    # Uploaded semantic references are independent and deliberately persist.
    clear_react_reference(only_kind="bbox")
    clear_react_reference(only_kind="point")
    from vln_geometry import configure_episode
    configure_episode(env, episode)
    env.current_episode = episode
    obs = env.reset()
    sc = scene_of(episode)
    desc = (episode.info or {}).get("scene_description", "")
    set_state(instruction=episode.instruction.instruction_text, status="loading",
              result="", scene=sc, episode_id=str(episode.episode_id),
              scene_title=SCENE_TITLES.get(sc, ""),
              scene_name=SCENE_NAMES.get(sc, SCENE_TITLES.get(sc) or sc),
              actions=[], moves=[],
              motion={"dist": 0.0, "bearing": 0.0,
                      "yaw": 0.0, "steps": 0})
    if desc:
        push_event("sys", text="场景：" + desc)
    # Drop any typed-in override: it belonged to the previous episode. STATE's
    # instruction (the dataset text) is what the box shows and what an empty
    # box falls back to, so a fresh episode starts on its own instruction.
    CFG["instruction"] = ""
    refresh(env, obs, 0, force_map=True)
    set_state(status="idle")
    return obs


def select_episode(env, scene, ep_id):
    for e in env.episodes:
        if scene_of(e) == scene and str(e.episode_id) == str(ep_id):
            return e
    return None


def interrupted():
    return CMD["stop"] or CMD["switch"] is not None or CMD["reset"]


def local_ips():
    """Non-loopback IPv4s this box answers to, for the startup banner.

    The server has always bound 0.0.0.0, but the banner used to print only
    127.0.0.1, which read as "loopback only" -- so peers on other machines
    didn't know which IP to point at. Listing the real ones fixes that without
    changing the bind.
    """
    seen, out = set(), []
    try:
        # Trick: connect a (no-flow) UDP socket to read the source IP the
        # routing table would pick for an off-box destination.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip not in seen and not ip.startswith("127."):
                seen.add(ip); out.append(ip)
        finally:
            s.close()
    except OSError:
        pass
    # Add any other non-loopback IPs the interfaces carry (a pod typically has
    # several, and the reachable one depends on which network the peer is on).
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ip = info[4][0]
            if ip not in seen and not ip.startswith("127."):
                seen.add(ip); out.append(ip)
    except socket.gaierror:
        pass
    return out or ["<本机 IP>"]


def check_actuation(env, defaults=None):
    """Verify nobody leaked a mutated step size.

    agent_tools restores the amount in a finally, so this should never fire --
    but a leak is invisible in the UI and every manual increment would inherit
    it (hold-to-drive sets its own amount per tick and would be measuring
    against a poisoned baseline), so it is worth an explicit check at every
    entry point. Repairs and reports rather than raising.
    """
    defaults = defaults or BASE_AMOUNTS
    if not defaults:
        return True
    bad = []
    for name, want in defaults.items():
        act = A.actuation(env, name)
        if abs(float(act.amount) - want) > 1e-6:
            bad.append(f"{name} {act.amount:g}→{want:g}")
            act.amount = want
    if bad:
        push_event("err", text="⚠ 步长被改动过，已恢复：" + "，".join(bad))
    return not bad


# ----------------------------------------------------------------------------
# hold-to-drive
# ----------------------------------------------------------------------------
def held_keys():
    """Keys the page says are down, with the heartbeat watchdog applied."""
    with LOCK:
        if HOLD["keys"] and time.time() - HOLD["t"] > HOLD_TIMEOUT:
            HOLD["keys"] = frozenset()
        return HOLD["keys"]


def new_hold_state():
    """Per-burst bookkeeping for drive_hold. A burst is one press-and-hold."""
    return {"on": False, "t": 0.0, "deg": 0.0, "m": 0.0, "steps": 0, "id": 0,
            "blocked": False}


def hold_chip(moves, st, chip):
    """Append a trail chip, or grow the one this burst already owns.

    One second of holding W is a dozen sub-moves. As separate chips they would
    bury the model's own tool calls under a wall of "前进 0.05m", so a press is
    a chip: same burst id, same kind, same direction -> add to the running
    total. The id is what keeps a later burst (or the model's own moves) from
    being folded into an earlier one.
    """
    chip["h"] = st["id"]
    last = moves[-1] if moves else None
    if (last is not None and last.get("h") == chip["h"]
            and last["kind"] == chip["kind"]
            and last.get("dir") == chip.get("dir")):
        if "m" in chip:
            last["m"] = round(last["m"] + chip["m"], 3)
            last["blocked"] = chip["blocked"]
        else:
            last["deg"] = round(last["deg"] + chip["deg"], 1)
    else:
        moves.append(chip)


def drive_hold(env, keys, st, obs):
    """One tick of hold-to-drive: turn (A/D) and/or move (W/S). Returns obs.

    Distance and angle come from *elapsed time* rather than from a per-press
    quantum, which is the whole point: hold W for two seconds and you walk
    1.2 m, tap it and you shuffle. Each tick issues at most one env step per
    axis, so the loop runs as fast as the simulator can render and the frame
    rate -- not the step size -- is what makes it feel continuous.

    A/D and W/S compose: holding W and A together turns and walks in the same
    tick, i.e. drives an arc, at the cost of two env steps for that tick.
    Holding both of an opposed pair cancels, as it does in any game.

    Both axes accumulate rather than issuing immediately, because both have a
    floor below which the primitive refuses to step: drive_turn drops anything
    under MIN_TURN_DEG on the floor, and a distance under HOLD_MIN_M is not
    worth an env step. Carrying the remainder is what makes the speed mean the
    same thing at 28 fps as at 12 -- issue max(floor, speed*dt) instead and a
    fast backend drives at whatever floor/dt happens to be, which measured 1.38
    m/s on habitat against an asked-for 0.6.
    """
    now = time.time()
    dt = min(HOLD_MAX_DT, max(0.0, now - st["t"]))
    st["t"] = now
    moves, motion = list(STATE["moves"]), dict(STATE["motion"])
    acted = False

    turn = ("a" in keys) - ("d" in keys)      # +1 = left, matching the key pair
    if turn:
        st["deg"] += HOLD_TURN_DPS * dt
        if st["deg"] >= A.MIN_TURN_DEG:
            deg = min(A.MAX_TURN_DEG, st["deg"])
            st["deg"] = 0.0
            direction = "left" if turn > 0 else "right"
            obs2, info = A.drive_turn(env, direction, deg)
            obs, acted = obs2 or obs, True
            # Net yaw is signed right-positive and wrapped, same as the ReAct
            # driver's, so the summary line means one thing whoever was driving.
            motion["yaw"] = A.wrap180(motion["yaw"] - info["turned_deg"] * turn)
            motion["steps"] += info["n_steps"]
            st["steps"] += info["n_steps"]
            hold_chip(moves, st, {"kind": "turn", "dir": direction,
                                  "deg": info["turned_deg"]})
    else:
        st["deg"] = 0.0

    fwd = ("w" in keys) - ("s" in keys)
    if fwd and not env.episode_over:
        # Accumulate like the turn above, and for the same reason: on a fast
        # backend a tick is ~35 ms, so speed*dt is 0.02 m and MIN_FWD_M would
        # round every single tick up to 0.05 -- driving at twice the asked speed
        # on habitat and at the asked speed on 3DGS. Banking the remainder makes
        # the speed mean the same thing whatever the frame rate.
        st["m"] += HOLD_SPEED_MPS * dt
        if st["m"] >= HOLD_MIN_M:
            meters = min(A.MAX_FWD_M, st["m"])
            st["m"] = 0.0
            drive = A.drive_forward if fwd > 0 else A.drive_backward
            obs2, info = drive(env, meters, min_m=HOLD_MIN_M)
            obs, acted = obs2 or obs, True
            motion["dist"] += info["moved_m"]
            motion["steps"] += info["n_steps"]
            st["steps"] += info["n_steps"]
            hold_chip(moves, st, {"kind": "fwd" if fwd > 0 else "back",
                                  "m": info["moved_m"],
                                  "blocked": bool(info["blocked"])})
            # One line per time it jams, not one per tick: a held key against a
            # wall would otherwise fill the log at the frame rate.
            if info["blocked"] and not st["blocked"]:
                push_event("sys", text="前方受阻，松开键换个方向"
                                       if fwd > 0 else "后方受阻，退不动了")
            st["blocked"] = bool(info["blocked"])
    elif not fwd:
        st["m"] = 0.0
        st["blocked"] = False

    if acted:
        motion["dist"] = round(motion["dist"], 3)
        set_trail(moves=moves, motion=motion)
        refresh(env, obs, motion["steps"])
    else:
        # Both accumulators are still under their floor, so this tick did no
        # work at all. The caller skips the loop's own sleep to keep the frame
        # rate at whatever env.step allows, so yield here or a fast backend
        # spins a core while the remainder fills up.
        time.sleep(0.005)
    return obs


def goal_verdict(metrics):
    """Use the evaluator's object-region signal when present; never fall back."""
    dist = float(metrics.get("distance_to_goal", float("nan")))
    radius = float(metrics.get("success_radius", float("nan")))
    if "official_object_success" in metrics:
        detail = metrics.get("official_object_success_detail") or {}
        hit = (bool(metrics["official_object_success"])
               if detail.get("available") else None)
        return dist, float(detail.get("expansion_radius_m", 1.0)), hit
    if not math.isfinite(radius):
        radius = SUCCESS_RADIUS
    return dist, radius, dist == dist and dist <= radius


def goal_evaluation(metrics):
    """UI-only scoring description shared by local and remote final logs."""
    dist, radius, hit = goal_verdict(metrics)
    official = "official_object_success" in metrics
    detail = metrics.get("official_object_success_detail") or {}
    available = hit is not None if official else math.isfinite(dist)
    if official:
        criterion = f"目标物体包围盒四周扩张{radius:g}m（含边界，不要求主动stop）"
        reason = str(detail.get("reason") or "目标物体信息不可用") if not available else ""
    else:
        criterion = f"距目标点≤{radius:g}m"
        reason = "本场景无目标点" if not available else ""
    shortest = metrics.get("shortest_path_m")
    walked = metrics.get("path_length_m")
    spl = None
    if available and shortest is not None and walked is not None:
        shortest, walked = max(0.0, float(shortest)), max(0.0, float(walked))
        if math.isfinite(shortest) and math.isfinite(walked):
            spl = float(bool(hit)) * shortest / max(walked, shortest, 1e-12)
    return dict(kind="object_region" if official else "point_goal",
                available=available, success=bool(hit) if available else None,
                label=("成功" if hit else "失败") if available else "无法判定",
                criterion=criterion, reason=reason, spl=spl,
                reference_distance_m=dist if math.isfinite(dist) else None,
                target_object_id=detail.get("target_object_id"),
                expanded_bbox_xyxy=detail.get("expanded_bbox_xyxy"),
                contract_sha256=detail.get("contract_sha256"))


def finish_metrics(metrics, why, step_id):
    evaluation = goal_evaluation(metrics)
    dist = evaluation["reference_distance_m"]
    summary = f"{why} · 本次 {step_id} 步 · 判定{evaluation['label']}"
    if evaluation["available"]:
        summary += "（" + evaluation["criterion"] + "）"
        if evaluation["target_object_id"]:
            summary += " · 目标物体 " + evaluation["target_object_id"]
    else:
        summary += "：" + evaluation["reason"]
    if dist is not None:
        distance_label = "距参考终点（仅供参考）" if evaluation["kind"] == "object_region" else "距目标"
        summary += f" · {distance_label} {dist:.2f} m"
    if evaluation["spl"] is not None:
        summary += f" · SPL {evaluation['spl']:.3f}"
    push_event("done", text=summary, evaluation=evaluation)
    set_state(status="done", result=f"{step_id} 步 / {evaluation['label']}",
              evaluation=evaluation)
    return dist if dist is not None else float("nan")


def finish_run(env, why, step_id):
    return finish_metrics(env.get_metrics(), why, step_id)


def run_model(env, ev, instruction, args, obs, trail):
    """Drive the agent with the model until STOP / Stop button / step cap."""
    tpl, nact = CFG["tpl"], CFG["nact"]
    check_actuation(env)
    rgb_list = []
    step_id = 0
    start_xy, start_yaw = A._xz(env), A._yaw(env)
    motion = {"dist": 0.0, "bearing": 0.0, "yaw": 0.0, "steps": 0}
    set_state(status="running", result="")
    set_trail(actions=trail, moves=[], motion=motion)
    push_event("instr", text=instruction, tpl=P.TEMPLATES[tpl][0],
               frompose=CFG["frompose"])
    refresh(env, obs, step_id)

    stop_requested = False
    model_turn = 0
    while step_id < args.max_steps and not env.episode_over:
        if interrupted():
            CMD["stop"] = False
            set_state(status="stopped")
            push_event("sys", text="— 已停止 —")
            return obs, trail

        rgb_list.append(observation_image(obs))
        hist = rgb_list[:-1]
        if len(hist) > ev.max_history_frames:
            idx = np.linspace(0, len(hist) - 1, ev.max_history_frames, dtype=int)
            hist = [hist[j] for j in idx]
        conv = P.build_conversation(tpl, instruction, hist, rgb_list[-1], nact, trail)
        model_turn += 1
        capture_model_request(conv, model_turn, "legacy")

        t0 = time.time()
        buf, reasoning_buf = [], []

        def on_delta(piece, channel="content", _buf=buf,
                     _reasoning=reasoning_buf, _t0=t0):
            (_reasoning if channel == "reasoning" else _buf).append(piece)
            set_state(stream={"active": True, "text": "".join(_buf),
                              "reasoning": "".join(_reasoning),
                              "n": len(_buf) + len(_reasoning),
                              "dt": time.time() - _t0})

        set_state(stream={"active": True, "text": "", "reasoning": "",
                          "n": 0, "dt": 0.0})
        try:
            text, server_reasoning, aborted, finish = stream_chat(
                ev, conv, CFG["mode"], CFG["temp"], on_delta, interrupted)
        except Exception as exc:
            set_state(stream={"active": False, "text": "", "n": 0, "dt": 0.0},
                      status="error")
            push_event("err", text=f"模型请求失败：{exc}")
            return obs, trail
        dt = time.time() - t0
        set_state(stream={"active": False, "text": "", "n": 0, "dt": 0.0})

        reasoning, visible = A.split_thinking(text, server_reasoning)
        seq = P.parse_actions(visible, nact)
        push_event("model", text=visible.strip() or "(空响应)",
                   reasoning=(reasoning or "").strip(), dt=dt,
                   turn=model_turn, actions=seq, abort=aborted)
        if aborted:
            CMD["stop"] = False
            set_state(status="stopped")
            push_event("sys", text="— 已停止（推理中断）—")
            return obs, trail
        # length-capped replies usually lose the trailing "答：" line, so say so
        # rather than silently treating the missing answer as a stop.
        if finish == "length":
            mx = P.MODES.get(CFG["mode"], {}).get("max_tokens", "?")
            push_event("sys", text=f"⚠ 输出达到上限 {mx} tokens 被截断"
                                   f"{'，未解析出动作' if not seq else ''}")
        if not seq:
            push_event("sys", text="未解析出动作，本次视为停止")
            seq = [STOP]

        done_names = []
        for action in seq:
            if action == STOP:
                stop_requested = True
                break
            if interrupted():
                break
            obs = env.step(action)
            step_id += 1
            trail.append(action)
            done_names.append(P.ZH_ACTION[action])
            motion["steps"] = step_id
            update_pose_motion(env, start_xy, start_yaw, motion)
            set_trail(actions=trail, motion=motion)
            refresh(env, obs, step_id)
            if render_mode() == 'display':
                time.sleep(args.step_delay)
            if env.episode_over:
                break
        if done_names:
            push_event("steps", text=" → ".join(done_names))
        if stop_requested:
            break

    why = ("模型输出停止" if stop_requested else
           ("回合结束" if env.episode_over else f"达到步数上限 {args.max_steps}"))
    refresh(env, obs, step_id, force_map=True)
    # Publish the terminal RGB-D before exposing a terminal status.  Tool
    # clients snapshot as soon as they observe ``done``; the old ordering let
    # them capture the penultimate seq and then reject that image as stale when
    # this final refresh landed a few milliseconds later.
    finish_run(env, why, step_id)
    return obs, trail


def _img_part(img):
    w, h = img.size
    return {"type": "image", "image": img, "width": w, "height": h}


def _cur_img(obs):
    return observation_image(obs)


def react_first_user_content(obs, instruction, reference_image=None,
                             reference_meta=None,
                             action_head_as_tool_version="v1"):
    """Build the first ReAct user turn, including optional visual context."""
    meta = reference_meta or {}
    version = AHP.normalize_version(action_head_as_tool_version)
    protocol = AHP.protocol_module(version)
    task_text = (protocol.task_text(instruction) if protocol is not None
                 else P.agent_task_text(instruction))
    current_view_text = (protocol.current_view_text() if protocol is not None
                         else P.agent_current_view_text())
    if reference_image is not None and meta.get("kind") == "upload":
        return [
            _img_part(_cur_img(obs)),
            {"type": "text", "text": current_view_text},
            {"type": "text", "text": "【额外参考图｜第 2 张】"
             "仅用于理解目标外观或视觉参照，不是机器人当前画面。"},
            _img_part(reference_image),
            {"type": "text", "text": task_text},
        ]
    if reference_image is not None and meta.get("kind") == "bbox":
        return [
            _img_part(reference_image),
            {"type": "text", "text":
             "【当前第一视角画面｜已绘制红色 bbox】\n"
             "上面这张图片是机器人当前看到的第一视角画面。用户指定的 bbox "
             "已作为红色矩形框直接绘制在图片上。\n"
             "红框内的物体或区域是本次局部移动的视觉目标；"
             "请结合完整画面判断可行路径。\n\n"
             "bbox 使用 0 到 1000 归一化整数坐标："
             f"{meta['bbox']}；左上角为 (0,0)，x 向右增大，y 向下增大。\n\n"
             + task_text},
        ]
    if reference_image is not None and meta.get("kind") == "point":
        return [
            _img_part(reference_image),
            {"type": "text", "text":
             "【当前第一视角画面｜已绘制红色视觉 point】\n"
             "上面这张图片是机器人当前看到的第一视角画面。"
             "红色十字圆环是用户指定的视觉目标像素；它可以位于物体、"
             "墙面、家具或地面上，只要该像素具有有效同帧深度。\n\n"
             "point 使用 0 到 1000 归一化整数坐标："
             f"{meta['point']}；左上角为 (0,0)，x 向右增大，y 向下增大。"
             "该坐标只作为视觉目标；请使用经典 ReAct 的转向、前进、后退工具"
             "逐步接近标记区域，不要把它直接作为深度 waypoint 执行。\n\n" + task_text},
        ]
    current_part = _img_part(_cur_img(obs))
    if protocol is not None:
        current_part = protocol.tagged_image_part(current_part, kind="slow")
    return [
        current_part,
        {"type": "text", "text": current_view_text + "\n\n" + task_text},
    ]


def action_head_condition_messages(history, sent, runtime):
    """Share the brain's image window while retaining its full text history."""
    del history, runtime
    return list(sent)


def run_agent(env, ev, instruction, args, obs, trail):
    """Keep checkpoint switching locked until episode device cleanup finishes."""
    HEAD_EPISODE_ACTIVE.set()
    try:
        return _run_agent_with_cleanup(env, ev, instruction, args, obs, trail)
    finally:
        HEAD_EPISODE_ACTIVE.clear()


def _run_agent_with_cleanup(env, ev, instruction, args, obs, trail):
    """Release episode features even after stop, early return or failure."""
    executor = ACTION_HEAD_TOOL_EXECUTOR if is_action_head_driver() else None
    episode_id = getattr(getattr(env, "current_episode", None), "episode_id", None)
    driver = CFG.get("driver")
    try:
        return _run_agent_episode(env, ev, instruction, args, obs, trail)
    finally:
        cleanup = getattr(executor, "end_episode", None)
        if cleanup is not None:
            released = cleanup()
            if released is not None:
                record = dict(released, episode_id=episode_id, driver=driver,
                              finished_at=time.time())
                timing_root = os.environ.get("ACTION_HEAD_TIMING_TRACE_DIR")
                if timing_root:
                    destination = pathlib.Path(timing_root)
                    destination.mkdir(parents=True, exist_ok=True)
                    with (destination / "episode_cache_cleanup.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                push_event("sys", text="episode 图片特征缓存已释放", cache_cleanup=record)


def _run_agent_episode(env, ev, instruction, args, obs, trail):
    """ReAct driver: the model calls tools, we drive the sim in real units.

    Same signature and return contract as run_model so main() can pick either.
    The model gets metres and degrees; agent_tools turns those into sub-steps
    (<=0.25 m, see its docstring for why) and reports back what actually
    happened rather than what was asked for.
    """
    check_actuation(env)
    set_active_visual_target(None)
    profile = active_tool_profile()
    action_head_as_tool_version = (
        AHP.normalize_version(CFG.get("action_head_as_tool_version"))
        if profile.name == "action_head" else "v1")
    protocol = AHP.protocol_module(action_head_as_tool_version)
    if profile.name == "action_head":
        if ACTION_HEAD_AGENT_RUNTIME is None:
            raise RuntimeError("Action Head runtime is not installed")
        episode = getattr(env, "current_episode", None)
        episode_id = getattr(episode, "episode_id", None)
        scene = getattr(env, "scene", None)
        try:
            remote_obs, reset_info = ACTION_HEAD_AGENT_RUNTIME.reset_environment(
                scene=scene,
                episode_id=str(episode_id) if episode_id is not None else None,
                instruction=instruction,
            )
        except Exception as exc:
            set_state(status="error")
            push_event("err", text=f"Action Head 远端环境 reset 失败：{exc}")
            return obs, trail
        if remote_obs.get("rgb") is None:
            set_state(status="error")
            push_event("err", text="Action Head 远端环境 reset 未返回 RGB")
            return obs, trail
        obs = remote_obs
        push_event(
            "sys",
            text=("Action Head 远端环境已同步："
                  f"{reset_info.get('scene', scene)}/"
                  f"{reset_info.get('episode_id', episode_id)}"),
        )
    # Snapshotted, like run_model does with tpl/nact: turning the dial mid-run
    # would move the budget the model was already told about.
    max_tools = CFG["maxtools"]
    step_id = 0
    calls_used = 0
    moves = []
    start_xy, start_yaw = A._xz(env), A._yaw(env)
    motion = {"dist": 0.0, "bearing": 0.0, "yaw": 0.0,
              "steps": 0, "path_dist": 0.0}
    reference_image, reference_meta = react_reference_snapshot()
    if protocol is not None:
        # Versioned system prompts and training messages are canonical. Optional UI
        # reference modes alter that contract, so they stay available to v1
        # but are not injected into an official v2/v3 run.
        reference_image, reference_meta = None, {}
    if reference_meta.get("kind") in ("point", "bbox"):
        with LOCK:
            current_source = (STATE["scene"], STATE["episode_id"], STATE["seq"])
        marked_source = (reference_meta.get("scene"),
                         reference_meta.get("episode_id"),
                         reference_meta.get("seq"))
        if marked_source != current_source:
            reference_image = None
            stale_kind = reference_meta.get("kind")
            clear_react_reference(only_kind=stale_kind)
            push_event("sys", text="视觉标记起点图对应的画面已变化，本次 ReAct 未使用它")
    set_state(status="running", result="")
    set_trail(actions=[], moves=[], motion=motion)
    push_event("instr", text=instruction, tpl="ReAct 工具调用",
               frompose=CFG["frompose"])
    if profile.name == "action_head":
        refresh_action_head_remote(obs, step_id)
    else:
        refresh(env, obs, step_id)

    if reference_image is not None and reference_meta.get("kind") == "upload":
        push_event("sys", text=f"本次 ReAct 使用额外参考图：{reference_meta['name']}")
    elif reference_image is not None and reference_meta.get("kind") == "bbox":
        push_event("sys", text=f"本次 ReAct 使用带红框的当前画面：{reference_meta['bbox']}")
    elif reference_image is not None and reference_meta.get("kind") == "point":
        push_event("sys", text=f"本次 ReAct 使用带视觉点的当前画面：{reference_meta['point']}")
    first_user_content = react_first_user_content(
        obs, instruction, reference_image, reference_meta,
        action_head_as_tool_version=action_head_as_tool_version)
    reference_kind = (reference_meta.get("kind")
                      if reference_image is not None else None)

    msgs = [
        {"role": "system",
         "content": [{"type": "text",
                      "text": P.agent_system_prompt(
                          (None if protocol is not None
                           else CFG["sysprompt"]), reference_kind,
                          tool_specs=profile.tool_specs,
                          action_head_as_tool_version=action_head_as_tool_version)}]},
        {"role": "user",
         "content": first_user_content},
    ]

    def on_step(new_obs, _i):
        nonlocal step_id
        step_id += 1
        motion["steps"] = step_id
        update_pose_motion(env, start_xy, start_yaw, motion)
        set_trail(moves=moves, motion=motion)
        refresh(env, new_obs, step_id)
        if render_mode() == 'display':
            time.sleep(args.step_delay)

    why, stopped = "", False
    recovery_turn_available = False
    consecutive_turn_calls = 0
    trajectory_map = (protocol.new_trajectory_map_state()
                      if action_head_as_tool_version == "v6" else None)
    review_factory = getattr(protocol, "STOP_REVIEW_FACTORY", None)
    stop_review = review_factory() if review_factory else None
    display_from = 0
    for turn in range(MAX_TURNS):
        if calls_used >= max_tools:
            why = f"达到工具调用上限 {max_tools}"
            break
        if interrupted():
            CMD["stop"] = False
            set_state(status="stopped")
            push_event("sys", text="— 已停止 —")
            return obs, trail
        if env.episode_over:
            why = "回合结束"
            break

        sent = (protocol.strip_context_images(msgs, KEEP_LAST_IMAGES)
                if action_head_as_tool_version in ("v4.5", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6") else
                A.strip_context_images_v2(msgs, KEEP_LAST_IMAGES)
                if protocol is not None
                else A.strip_old_images(msgs, KEEP_LAST_IMAGES))
        if trajectory_map is not None:
            sent = protocol.inject_trajectory_map(sent, trajectory_map)
        # Brain and independent encoder share the same effective 8+3+map window.
        # Snapshot before appending this turn's assistant reply below.
        condition_messages = action_head_condition_messages(
            msgs, sent, ACTION_HEAD_AGENT_RUNTIME)
        # Printed rather than pushed as an event: this is for watching how the
        # conversation grows, and it would be noise in the user-facing log.
        # With env steps uncapped, the context window is the practical ceiling,
        # so this line plus the ← one below are the only warning before a run
        # ends on an overflow.
        print(f"[agent] 第 {turn + 1}/{MAX_TURNS} 轮 → 消息 {len(sent)} 条  图像 "
              f"{A.count_images(sent)} 张  工具已用 {calls_used}/{max_tools}"
              f"  仿真步 {step_id}", flush=True)
        capture_model_request(
            sent, turn + 1, "react", display_from=display_from,
            hidden_roles=("assistant",) if display_from else ())
        # The assistant reply produced below is appended to msgs, followed by
        # fresh tool/user observations.  On the next turn only that new suffix
        # is considered for display; the assistant part is already the output
        # card directly below this input card.
        display_from = len(sent)
        t0 = time.time()
        buf, reasoning_buf = [], []

        def on_delta(piece, channel="content", _buf=buf,
                     _reasoning=reasoning_buf, _t0=t0):
            (_reasoning if channel == "reasoning" else _buf).append(piece)
            set_state(stream={"active": True, "text": "".join(_buf),
                              "reasoning": "".join(_reasoning),
                              "n": len(_buf) + len(_reasoning),
                              "dt": time.time() - _t0})

        set_state(stream={"active": True, "text": "", "reasoning": "",
                          "n": 0, "dt": 0.0})
        try:
            text, server_reasoning, aborted, finish = stream_chat(
                ev, sent, CFG["mode"], CFG["temp"], on_delta, interrupted,
                preserve_thinking=THINKING_HISTORY == "keep_thinking",
                capture_queries=profile.name == "action_head",
                tool_specs=list(profile.tools_by_name.values())
                    if CFG.get('model') in P.MODEL_NATIVE_TOOL_CALLS else None)
        except ContextOverflow as exc:
            # The expected ending for a long run now that nothing else bounds
            # one. Not an error: everything up to here really happened, so the
            # episode gets its normal summary instead of a red banner.
            set_state(stream={"active": False, "text": "", "n": 0, "dt": 0.0})
            push_event("sys", text=f"⚠ 上下文超出模型窗口，回合在第 {turn + 1} 轮结束"
                                   f"（{exc.detail}）")
            why = f"上下文超出窗口（第 {turn + 1} 轮）"
            break
        except Exception as exc:
            set_state(stream={"active": False, "text": "", "n": 0, "dt": 0.0},
                      status="error")
            push_event("err", text=f"模型请求失败：{exc}")
            return obs, trail
        dt = time.time() - t0
        set_state(stream={"active": False, "text": "", "n": 0, "dt": 0.0})

        # prompt_tokens is the server's own count of the rendered conversation,
        # which is the only number that reflects what keep_thinking actually
        # costs -- the client cannot see what the chat template emits.
        pt = LAST_USAGE.get("prompt_tokens")
        model_window = P.model_context_window(CFG.get("model"))
        print(f"[agent] 第 {turn + 1} 轮 ← prompt "
              f"{pt if pt is not None else '?'} tok"
              f"{f' (窗口 {100 * pt / model_window:.0f}%)' if pt else ''}"
              f"  输出 {LAST_USAGE.get('completion_tokens', '?')} tok"
              f"  用时 {dt:.1f}s  历史 {THINKING_HISTORY}", flush=True)

        reasoning, visible = A.split_thinking(text, server_reasoning)
        calls = A.parse_tool_calls(visible)
        push_event("model", text=visible.strip() or "(空响应)",
                   reasoning=(reasoning or "").strip(), dt=dt,
                   turn=turn + 1, actions=[], abort=aborted)
        if aborted:
            CMD["stop"] = False
            set_state(status="stopped")
            push_event("sys", text="— 已停止（推理中断）—")
            return obs, trail
        if finish == "length":
            mx = P.MODES.get(CFG["mode"], {}).get("max_tokens", "?")
            push_event("sys", text=f"⚠ 输出达到上限 {mx} tokens 被截断"
                                   f"{'，未解析出工具调用' if not calls else ''}")

        # Under keep_thinking this is the raw reply, </think> and all -- which
        # only survives because stream_chat also sends preserve_thinking; see
        # THINKING_HISTORY and A.assistant_history_content.
        history_text = text
        if server_reasoning:
            history_text = (f"<think>\n{server_reasoning}\n</think>\n"
                            f"{visible}")
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": A.assistant_history_content(
                         history_text, visible, THINKING_HISTORY) or visible}]})

        tools_by_name = profile.tools_by_name
        parsed = calls
        unknown = [c["name"] for c in calls if c["name"] not in tools_by_name]
        calls = [c for c in calls if c["name"] in tools_by_name]
        format_error = A.tool_call_batch_error(visible, parsed, tools_by_name)
        if format_error:
            calls_used += 1
            raw = format_error
            msgs.append(A.tool_call_feedback_message(
                raw, max(0, max_tools - calls_used), protocol))
            push_event("obs", name="invalid_tool_call", ok=False,
                       text=A.compact_json(raw), tool_call_error=raw)
            if calls_used >= max_tools:
                why = f"达到工具调用上限 {max_tools}"
                break
            continue

        # Two ceilings: per turn, and whatever is left of the run's budget. The
        # second one matters because the budget is set on the page and can be
        # smaller than a batch -- without it, "最多还可以调用 1 次" would be
        # followed by four calls actually running.
        # v4's turn is observationally meaningful only if the slow system sees
        # the post-turn RGB before choosing its next action.  Execute exactly
        # one tool from each assistant response to preserve that causal order.
        per_turn_limit = 1 if action_head_as_tool_version in ("v4", "v4.1", "v4.2", "v4.3", "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6") \
            else P.MAX_CALLS_PER_TURN
        keep = min(per_turn_limit, max(0, max_tools - calls_used))
        dropped = 0
        if len(calls) > keep:
            dropped = len(calls) - keep
            calls = calls[:keep]

        results = []   # [(text, needs_image)]
        modern_trace_results = {}
        fatal_action_head_error = False
        recovery_episode_ended = False
        for call in calls:
            if interrupted() or env.episode_over:
                break
            name, argd = call["name"], call["arguments"]
            event = {"name": name, "args": argd}
            if name in ("move_to_point", "move_to_bbox"):
                try:
                    event["visual"] = visual_tool_preview(obs, name, argd)
                except (TypeError, ValueError):
                    # The executor below reports malformed geometry. Keep the
                    # call row but do not draw a guessed/misleading target.
                    pass
            push_event("tool", **event)
            calls_used += 1

            stop_rejection = A.rejected_stop_result(argd) if name == "stop" else None
            if stop_rejection is not None:
                txt = A.compact_json(stop_rejection)
                results.append((txt, False))
                push_event("obs", name=name, ok=False, text=txt)
                continue

            if name == "stop":
                review = (stop_review.consider(
                    instruction=instruction, remaining_calls=max_tools - calls_used,
                    remaining_decisions=MAX_TURNS - turn - 1)
                    if stop_review is not None else None)
                if review is not None:
                    txt = A.compact_json(review)
                    results.append((txt, False))
                    push_event("obs", name=name, ok=False, text=txt)
                    continue
                stopped = True
                why = "模型调用 stop：" + str(argd.get("reason") or "")
                break

            if name == "run_action_head":
                if ACTION_HEAD_TOOL_EXECUTOR is None:
                    raw = {
                        **({"schema": protocol.TOOL_RESULT_SCHEMA,
                            "option_id": f"turn-{turn + 1}-call-{calls_used}",
                            "completion": "unknown", "chunks": 0,
                            "path_m": 0.0, "net_m": None, "blocked": False,
                            "collision": False, "steps_used": 0}
                           if protocol is not None else {}),
                        "tool": "run_action_head",
                        "ok": False,
                        "return_reason": "error",
                        "error": "Action Head executor is not installed",
                    }
                    if action_head_as_tool_version in ("v3", "v4", "v4.1", "v4.2", "v4.3", "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6"):
                        raw["direction_alignment"] = AHV3.direction_alignment(
                            argd, None)
                    result = A.ToolResult(
                        False, A.compact_json(raw), raw=raw,
                        error=raw["error"])
                else:
                    result = ACTION_HEAD_TOOL_EXECUTOR.execute(
                        option_id=f"turn-{turn + 1}-call-{calls_used}",
                        instruction=instruction,
                        arguments=argd,
                        current_observation=obs,
                        # Both drivers use the same text history/image window.
                        slow_request_snapshot=AHV.semantic_request_snapshot(
                            messages=condition_messages, instruction=instruction),
                        slow_response_snapshot={**AHV.response_snapshot(
                            text=text, visible=visible, usage=LAST_USAGE,
                            query_request_id=LAST_QUERY_REQUEST_ID),
                            "timing": dict(LAST_INFERENCE_TIMING), "driver": CFG.get("driver"),
                            "scene": STATE.get("scene"), "episode_id": STATE.get("episode_id")},
                        action_head_as_tool_version=action_head_as_tool_version,
                    )
                if result.ok:
                    # Rejected AH arguments must not clear the turn streak.
                    consecutive_turn_calls, _ = A.enforce_v43_consecutive_turn_limit(
                        action_head_as_tool_version, name, consecutive_turn_calls)
                if result.observation is not None:
                    obs = result.observation
                    used = max(0, int((result.raw or {}).get("steps_used", 0)))
                    moved = max(0.0, float((result.raw or {}).get(
                        "path_m" if protocol is not None
                        else "moved_m", 0.0)))
                    step_id += used
                    motion["steps"] = step_id
                    motion["path_dist"] = round(
                        float(motion.get("path_dist", 0.0)) + moved, 3)
                    moves.append({
                        "kind": "action_head",
                        "m": moved,
                        "steps": used,
                        "chunks": int((result.raw or {}).get("chunks", 0)),
                        "blocked": bool((result.raw or {}).get("blocked", False)),
                    })
                    set_trail(moves=moves, motion=motion)
                    refresh_action_head_remote(obs, step_id)
                if trajectory_map is not None and result.ok:
                    trajectory_map.update_action_head(result.trace_frames)
                if protocol is not None:
                    modern_trace_results[len(results)] = result
                results.append((result.model_text, result.with_image))
                if action_head_as_tool_version in ("v4.1", "v4.9", "v4.10"):
                    recovery_turn_available = (
                        ((result.raw or {}).get("direction_alignment") or {})
                        .get("match") is False)
                push_event("obs", name=name, ok=result.ok,
                           text=result.model_text)
                if (not result.ok
                        and (result.raw or {}).get("return_reason") == "error"):
                    fatal_action_head_error = True
                    push_event("err", text=(
                        "Action Head 基础设施错误，已停止本轮，避免重复消耗工具预算："
                        + str((result.raw or {}).get("error") or "unknown error")))
                    break

            elif (name == "move_backward" and profile.name == "action_head"
                  and action_head_as_tool_version in ("v4.9", "v4.10")):
                from action_head.backoff_recovery import validate_backoff
                distance, raw = validate_backoff(msgs, argd)
                if raw is not None:
                    txt = A.compact_json(raw)
                    results.append((txt, False))
                    push_event("obs", name=name, ok=False, text=txt)
                    continue
                try:
                    obs2, raw = ACTION_HEAD_AGENT_RUNTIME.execute_backoff(distance=distance)
                except Exception as exc:
                    raw = dict(schema="backoff_tool_result_v1", tool=name,
                               ok=False, return_reason="error", error=str(exc))
                    txt = A.compact_json(raw)
                    results.append((txt, False))
                    push_event("obs", name=name, ok=False, text=txt)
                    fatal_action_head_error = True
                    break
                recovery_turn_available = False
                obs = obs2 or obs
                used = max(0, int(raw.get("steps_used", 0)))
                step_id += used
                motion["steps"] = step_id
                motion["path_dist"] -= float(raw.get("moved_m", 0.0))
                moves.append(dict(kind="back", m=raw.get("moved_m", 0.),
                                  req=distance, blocked=raw.get("blocked", False), steps=used))
                refresh_action_head_remote(obs, step_id)
                txt = A.compact_json(raw)
                results.append((txt, True))
                push_event("obs", name=name, ok=True, text=txt)

                if raw.get("return_reason") == "episode_ended":
                    recovery_episode_ended = True
                    break

            elif name in ("move_forward", "move_backward"):
                back = name == "move_backward"
                meters, note = A.clamp_distance(
                    argd.get("distance"), A.MAX_BACK_M if back else None)
                if meters is None:
                    txt, raw = A.error_observation(name, note, motion, None)
                    results.append((txt, False))
                    push_event("obs", name=name, ok=False, text=txt)
                    continue
                drive = A.drive_backward if back else A.drive_forward
                obs2, info = drive(env, meters, on_step, interrupted)
                obs = obs2 or obs
                # Keep the model-facing mileage ledger separate from the
                # start-to-end polar displacement published in ``dist``.
                motion["path_dist"] += -info["moved_m"] if back else info["moved_m"]
                moves.append({"kind": "back" if back else "fwd",
                              "m": info["moved_m"],
                              "req": info["requested_m"],
                              "blocked": bool(info["blocked"])})
                txt, raw = A.move_observation(info, motion, None, note)
                results.append((txt, True))
                push_event("obs", name=name, ok=True,
                           blocked=bool(info["blocked"]), text=txt)

            elif name in ("move_to_point", "move_to_bbox"):
                field = "point_2d" if name == "move_to_point" else "bbox_2d"
                mode = "point" if name == "move_to_point" else "bbox"
                try:
                    coords = (VNT.parse_point_2d(argd.get(field)) if mode == "point"
                              else VNT.parse_bbox_2d(argd.get(field)))

                    def on_visual_target(marker):
                        # Publish the selected 3D point before the first turn,
                        # then on_step() reprojects it for every movement frame.
                        set_active_visual_target(marker)
                        refresh(env, obs, step_id)

                    try:
                        obs2, raw = VNT.execute(
                            env, obs, mode, coords, on_step=on_step,
                            abort=interrupted, on_target=on_visual_target,
                            on_invalid_point=show_invalid_point)
                    finally:
                        # The already-published terminal frame keeps its marker
                        # until another frame arrives, but unrelated later
                        # actions can never inherit a stale world target.
                        set_active_visual_target(None)
                except (TypeError, ValueError) as exc:
                    if getattr(exc, 'terminal_observation', None) is not None:
                        obs = exc.terminal_observation
                        refresh(env, obs, step_id, force_map=True)
                    txt, raw = A.error_observation(name, str(exc), motion, None)
                    results.append((txt, False))
                    push_event("obs", name=name, ok=False, text=txt)
                    continue
                obs = obs2 or obs
                moved = float(raw.get("moved_m", 0.0))
                reached = bool(raw.get("reached", raw.get("ok", False)))
                motion["path_dist"] += moved
                moves.append({"kind": mode, "m": moved,
                              "depth_m": raw.get("depth_m"),
                              "blocked": (bool(raw.get("blocked", False))
                                          or not reached)})
                raw["steps_used"] = int(raw.get("n_steps", 0))
                raw["total_forward_m"] = round(float(motion["path_dist"]), 3)
                raw["net_yaw_deg"] = round(float(motion["yaw"]), 2)
                txt = A.compact_json(raw)
                results.append((txt, True))
                push_event("obs", name=name, ok=reached,
                           blocked=bool(raw.get("blocked", False)), text=txt)

            elif name == "turn":
                if profile.name == "action_head":
                    turn_args, note = AHP.validate_turn_arguments(
                        argd, action_head_as_tool_version)
                    if turn_args is None:
                        raw = AHP.rejected_turn_result(note)
                        txt = A.compact_json(raw)
                        results.append((txt, False))
                        push_event("obs", name=name, ok=False, text=txt)
                        continue
                    direction, degrees = turn_args["direction"], turn_args["degrees"]
                else:
                    # Preserve the separate legacy primitive-tool contract.
                    direction = A.parse_direction(argd.get("direction"))
                    degrees, note = A.clamp_degrees(argd.get("degrees"))
                if direction is None or degrees is None:
                    msg = note or "direction 必须是 left 或 right"
                    txt, raw = A.error_observation(name, msg, motion, None)
                    results.append((txt, False))
                    push_event("obs", name=name, ok=False, text=txt)
                    continue
                if (profile.name == "action_head"
                        and action_head_as_tool_version in ("v4", "v4.1", "v4.2", "v4.3", "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6")):
                    if (action_head_as_tool_version in ("v4.1", "v4.9", "v4.10")
                            and not recovery_turn_available):
                        raw = {
                            "schema": "turn_tool_result_v2", "tool": "turn",
                            "ok": False, "return_reason": "turn_not_available",
                            "error": ("turn is available only once after the immediately "
                                      "preceding run_action_head direction mismatch"),
                            "steps_used": 0,
                            "required_next_tool": "run_action_head_or_stop",
                            "turn_available_after_result": False,
                        }
                        txt = A.compact_json(raw)
                        results.append((txt, False))
                        push_event("obs", name=name, ok=False, text=txt)
                        continue
                    # Count only valid, available turns; total tool budget was
                    # already charged before validation, including rejections.
                    consecutive_turn_calls, turn_limit_rejection = (
                        A.enforce_v43_consecutive_turn_limit(
                            action_head_as_tool_version, name, consecutive_turn_calls))
                    if turn_limit_rejection is not None:
                        txt = A.compact_json(turn_limit_rejection)
                        results.append((txt, False))
                        push_event("obs", name=name, ok=False, text=txt)
                        continue
                    try:
                        if action_head_as_tool_version in ("v4.1", "v4.9", "v4.10"):
                            obs2, raw = ACTION_HEAD_AGENT_RUNTIME.execute_recovery_turn(
                                direction=direction, degrees=degrees)
                            recovery_turn_available = False
                        else:
                            obs2, raw = ACTION_HEAD_AGENT_RUNTIME.execute_turn(
                                direction=direction, degrees=degrees)
                    except Exception as exc:
                        raw = {
                            "schema": "turn_tool_result_v1", "tool": "turn",
                            "ok": False, "direction": direction,
                            "requested_deg": degrees, "steps_used": 0,
                            "error": str(exc),
                        }
                        txt = A.compact_json(raw)
                        results.append((txt, False))
                        push_event("obs", name=name, ok=False, text=txt)
                        fatal_action_head_error = True
                        break
                    obs = obs2 or obs
                    used = max(0, int(raw.get("steps_used", 0)))
                    turned = float(raw.get("turned_deg", 0.0))
                    step_id += used
                    motion["steps"] = step_id
                    motion["yaw"] = round(A.wrap180(
                        float(motion.get("yaw", 0.0)) +
                        (turned if direction == "left" else -turned)), 1)
                    moves.append({"kind": "turn", "dir": direction,
                                  "deg": turned, "steps": used})
                    refresh_action_head_remote(obs, step_id)
                    txt = A.compact_json(raw)
                    results.append((txt, True))
                    push_event("obs", name=name, ok=True, text=txt)
                    if trajectory_map is not None:
                        trajectory_map.update_turn(raw)
                else:
                    replay_spec = _async_turn_render_spec(env, direction, degrees)
                    if replay_spec is not None:
                        # Gaussian backends can render arbitrary poses. Execute
                        # the requested turn in one simulator action, then
                        # replay intermediate views in the browser stream.
                        obs2, info = A.drive_turn(
                            env, direction, degrees, abort=interrupted)
                        step_id += int(info["n_steps"])
                        motion["steps"] = step_id
                        update_pose_motion(env, start_xy, start_yaw, motion)
                    else:
                        obs2, info = A.drive_turn(
                            env, direction, degrees, on_step,
                            abort=interrupted)
                    obs = obs2 or obs
                    moves.append({"kind": "turn", "dir": direction,
                                   "deg": info["turned_deg"]})
                    if replay_spec is not None:
                        animate = info["turned_deg"] > A.TURN_RENDER_STEP_DEG
                        set_trail(moves=moves, motion=motion)
                        refresh(env, obs, step_id, preserve_display=animate)
                        if animate:
                            replay_turn_async(replay_spec, info["turned_deg"])
                    txt, raw = A.turn_observation(info, motion, None,
                                                   direction, note)
                    results.append((txt, True))
                    push_event("obs", name=name, ok=True, text=txt)

            if render_mode() == 'batch' and profile.name != 'action_head':
                refresh(env, obs, step_id, force_map=True)
            set_trail(moves=moves, motion=motion)

        if dropped:
            results.append((f"本轮只执行了前 {keep} 个工具调用，"
                            f"其余 {dropped} 个已忽略。", False))
        if unknown:
            results.append(("已忽略未知工具：" + "、".join(unknown), False))

        if stopped:
            break
        if fatal_action_head_error:
            why = "Action Head 基础设施错误"
            break
        if recovery_episode_ended:
            why = "远端环境已结束"
            break
        if not results:
            why = why or "无可执行的工具调用"
            break

        # The frame rides on the LAST result of the batch -- that is the "next
        # frame after the actions ran" the model is waiting for. Blocked moves
        # also carry one so it can see what is in the way.
        budget_left = max(0, max_tools - calls_used)
        for i, (txt, wants_img) in enumerate(results):
            if protocol is not None:
                body = (protocol.append_tool_budget(txt, budget_left)
                        if i == len(results) - 1 else txt)
                content = [{"type": "text", "text": "[Tool result]\n" + body}]
                traced = modern_trace_results.get(i)
                if traced is not None:
                    frames = tuple(getattr(traced, "trace_frames", ()) or ())
                    for frame in frames:
                        content.append({"type": "text", "text": protocol.chunk_text(
                            option_id=frame["option_id"], chunk=frame["chunk"],
                            actual_end_pose=frame.get("actual_end_pose"),
                            path_m=frame["path_m"], blocked=frame["blocked"],
                            collision=frame["collision"],
                            no_progress=frame["no_progress"])})
                        frame_obs = frame.get("observation")
                        if frame_obs is not None:
                            kind = ("final_slow" if frame["final_slow_observation"]
                                    else "fast_intermediate")
                            content.append(protocol.tagged_image_part(
                                _img_part(_cur_img(frame_obs)), kind=kind,
                                option_id=frame["option_id"], chunk=frame["chunk"]))
                elif wants_img and i == len(results) - 1:
                    # v4 turn returns the newly observed current frame in the
                    # same tool-result message, preserving result/image pairing.
                    content.append(protocol.tagged_image_part(
                        _img_part(_cur_img(obs)), kind="slow"))
            else:
                body = A.append_tool_budget(txt, budget_left) \
                    if i == len(results) - 1 else txt
                content = [{"type": "text", "text": "[工具返回]\n" + body}]
                if wants_img and i == len(results) - 1:
                    content.insert(0, _img_part(_cur_img(obs)))
            msgs.append({"role": "user", "content": content})

        if calls_used >= max_tools:
            why = f"达到工具调用上限 {max_tools}"
            break
    else:
        why = f"达到轮数上限 {MAX_TURNS}"

    check_actuation(env)
    set_active_visual_target(None)
    if profile.name == "action_head":
        moved = float(motion.get("path_dist", 0.0))
        metrics = ACTION_HEAD_AGENT_RUNTIME.ui_metrics()
        finish_metrics(metrics,
                       f"{why or '结束'} · 远端仿真 · Action Head累计执行 {moved:.2f} m",
                       step_id)
        refresh_action_head_remote(obs, step_id)
    else:
        refresh(env, obs, step_id, force_map=True)
        finish_run(env, why or "结束", step_id)
    return obs, trail


def direct_point_start_text(point_2d):
    """Describe direct point execution without assuming a model backend."""
    return f"pi-agent 已给出视觉点 {point_2d}，直接执行视觉点导航"


def run_direct_point(env, instruction, point_2d, args, obs, trail):
    """Execute a pi-agent-selected image point without another model round trip."""
    check_actuation(env)
    set_active_visual_target(None)
    step_id = 0
    moves = []
    start_xy, start_yaw = A._xz(env), A._yaw(env)
    motion = {"dist": 0.0, "bearing": 0.0, "yaw": 0.0,
              "steps": 0, "path_dist": 0.0}
    set_state(status="running", result="")
    set_trail(actions=[], moves=[], motion=motion)
    push_event("instr", text=instruction, tpl="pi-agent 直接 point",
               frompose=True)
    push_event("sys", text=direct_point_start_text(point_2d))
    refresh(env, obs, step_id)

    def on_step(new_obs, _i):
        nonlocal step_id
        step_id += 1
        motion["steps"] = step_id
        update_pose_motion(env, start_xy, start_yaw, motion)
        set_trail(moves=moves, motion=motion)
        refresh(env, new_obs, step_id)
        if render_mode() == 'display':
            time.sleep(args.step_delay)

    def on_visual_target(marker):
        set_active_visual_target(marker)
        refresh(env, obs, step_id)

    try:
        obs2, raw = VNT.execute(
            env, obs, "point", point_2d, on_step=on_step,
            abort=interrupted, on_target=on_visual_target,
            on_invalid_point=show_invalid_point)
    except (TypeError, ValueError) as exc:
        if getattr(exc, 'terminal_observation', None) is not None:
            obs = exc.terminal_observation
            refresh(env, obs, step_id, force_map=True)
        message = str(exc)
        set_state(status="error", result=message)
        push_event("err", text=message)
        return obs, trail
    finally:
        set_active_visual_target(None)
        clear_react_reference(only_kind="point")

    obs = obs2 or obs
    moved = float(raw.get("moved_m", 0.0))
    reached = bool(raw.get("reached", raw.get("ok", False)))
    motion["path_dist"] = moved
    motion["steps"] = int(raw.get("n_steps", step_id))
    update_pose_motion(env, start_xy, start_yaw, motion)
    moves.append({"kind": "point", "m": moved,
                  "depth_m": raw.get("depth_m"),
                  "blocked": (bool(raw.get("blocked", False)) or not reached)})
    set_trail(moves=moves, motion=motion)
    result = (f"已直接到达 point {point_2d}" if reached else
              f"直接 point {point_2d} 未到达："
              f"{raw.get('reason') or 'unknown'}")
    refresh(env, obs, motion["steps"], force_map=True)
    if reached:
        set_state(status="done", result=result)
        push_event("done", text=result + f" · 本次 {motion['steps']} 步")
    else:
        set_state(status="error", result=result)
        push_event("err", text=result)
    return obs, trail


def run_lightnav(env, ev, instruction, args, obs, trail):
    """Drive any mounted simulator through the external LightNav-0 service.

    The WebSocket connection is kept for the whole episode because LightNav's
    SlowFast history lives server-side.  Exactly one native velocity-relevant
    waypoint is applied before the next observation/inference round, matching
    its Habitat policy's feedback frequency.  ``lightnav_driver`` performs and
    logs the unavoidable velocity-to-primitive projection.
    """
    del ev  # LightNav owns inference; the console's chat-model client is unused.
    check_actuation(env)
    set_active_visual_target(None)
    set_active_lightnav_pointing([], restore_display=True)
    step_id = 0
    policy_step = 0
    moves = []
    start_xy, start_yaw = A._xz(env), A._yaw(env)
    motion = {"dist": 0.0, "bearing": 0.0, "yaw": 0.0,
              "steps": 0, "path_dist": 0.0}
    set_state(status="running", result="")
    set_trail(actions=[], moves=[], motion=motion)
    push_event("instr", text=instruction, tpl="LightNav-0 轨迹",
               frompose=CFG["frompose"])
    refresh(env, obs, step_id)

    client = LND.LightNavClient(
        args.lightnav_url, timeout=args.lightnav_timeout,
        client_id="closed_loop_console")
    try:
        client.reset()
    except Exception as exc:
        client.close()
        set_state(status="error", result=str(exc))
        push_event("err", text=f"LightNav 服务连接/重置失败：{exc}")
        return obs, trail

    push_event("sys", text=(f"LightNav 已连接：{args.lightnav_url}；"
                            "沿用原生 dt=0.1s / 0.25m / 30° 动作口径"))

    def on_step(new_obs, _i):
        nonlocal step_id, obs
        if new_obs is not None:
            obs = new_obs
        step_id += 1
        motion["steps"] = step_id
        update_pose_motion(env, start_xy, start_yaw, motion)
        set_trail(moves=moves, motion=motion)
        refresh(env, obs, step_id)
        if render_mode() == 'display':
            time.sleep(args.step_delay)

    why = ""
    try:
        while policy_step < args.max_steps:
            if interrupted():
                CMD["stop"] = False
                set_state(status="stopped")
                push_event("sys", text="— 已停止 —")
                return obs, trail
            if env.episode_over:
                why = "回合结束"
                break
            if step_id >= args.max_steps:
                why = f"达到仿真步数上限 {args.max_steps}"
                break

            policy_step += 1
            frame = observation_image(obs)
            # The remote MuJoCo backend carries the exact camera JPEG used by
            # LightNav's native client. Other backends are encoded from their
            # canonical clean RGB as usual.
            wire_frame = (obs.get("_lightnav_jpeg")
                          if isinstance(obs, dict) else None) or frame
            capture_model_request([
                {"role": "user", "content": [
                    _img_part(frame),
                    {"type": "text", "text":
                     "【LightNav-0 WebSocket 输入】\n"
                     f"instruction: {instruction}\n"
                     "服务端保留本回合 SlowFast 图像历史并预测 10×3 ego waypoints。"},
                ]},
            ], policy_step, "lightnav")
            set_state(stream={"active": True, "text": "LightNav 推理中…",
                              "n": 0, "dt": 0.0})
            t0 = time.time()
            try:
                prediction = client.predict(wire_frame, instruction)
            except Exception as exc:
                set_state(stream={"active": False, "text": "", "n": 0, "dt": 0.0},
                          status="error", result=str(exc))
                push_event("err", text=f"LightNav 推理失败：{exc}")
                return obs, trail
            wall_dt = time.time() - t0
            set_state(stream={"active": False, "text": "", "n": 0, "dt": wall_dt})

            # Show the complete 10x3 metric ego trajectory plus pointing on the
            # exact frame the policy consumed.  The ribbon is also lifted into
            # world space so it reprojects correctly on movement frames; depth-
            # backed APOS/OPOS get the same treatment. This touches only the
            # browser display stream, never the canonical observation.
            markers = lightnav_world_markers(
                env, obs, prediction.pointing, frame.size)
            trajectory = lightnav_world_trajectory(
                env, prediction.waypoints, frame.size)
            set_active_lightnav_pointing(markers, trajectory=trajectory)
            show_lightnav_prediction(env, frame, prediction)

            try:
                projected = LND.project_waypoint(prediction.waypoints)
            except (TypeError, ValueError) as exc:
                set_state(status="error", result=str(exc))
                push_event("err", text=f"LightNav waypoint 无效：{exc}")
                return obs, trail

            source_yaw_deg = math.degrees(projected.source_yaw_rad)
            loss = []
            if abs(projected.lateral_dropped_m) > 1e-6:
                loss.append(f"drop lateral {projected.lateral_dropped_m:+.3f}m")
            if abs(projected.forward_clipped_m) > 1e-6:
                loss.append(f"clip fwd {projected.forward_clipped_m:+.3f}m")
            if abs(projected.yaw_clipped_deg) > 1e-4:
                loss.append(f"clip yaw {projected.yaw_clipped_deg:+.1f}°")
            push_event(
                "model",
                text=(f"policy {policy_step} / seq {prediction.seq} · "
                      f"wp[{projected.index}] "
                      f"({projected.source_forward_m:+.3f},"
                      f"{projected.source_lateral_m:+.3f},"
                      f"{source_yaw_deg:+.1f}°) → "
                      f"({projected.forward_m:.3f}m,{projected.yaw_deg:+.1f}°) · "
                      f"stop={prediction.stop} visible={prediction.visible} · "
                      f"{prediction.latency_ms:.1f}ms"
                      + ((" · adapter: " + ", ".join(loss)) if loss else "")),
                turn=policy_step, dt=wall_dt, actions=[], abort=False,
                raw=prediction.raw_text, pointing=prediction.pointing,
            )

            # End only this driver run.  Do NOT forward STOP to the simulator:
            # every other console driver keeps the episode alive so the user
            # can continue from the current pose, while env.step(STOP) makes
            # Habitat-shaped backends (including LightNav MuJoCo) permanently
            # episode_over until reset.  The model's stop remains visible in
            # the log/result; finish_run derives any success verdict itself.
            if (prediction.stop or
                    (abs(projected.forward_m) <= 1e-9
                     and abs(projected.yaw_deg) < A.MIN_TURN_DEG)):
                why = "模型输出 STOP" if prediction.stop else "模型输出零速度"
                break

            log = LND.execute_projected_waypoint(
                env, projected, A, on_step=on_step, abort=interrupted)
            if log.get("observation") is not None:
                obs = log["observation"]
            motion["path_dist"] += float(log.get("moved_m", 0.0))
            update_pose_motion(env, start_xy, start_yaw, motion)
            moves.append({
                "kind": "lightnav",
                "m": float(log.get("moved_m", 0.0)),
                "yaw_deg": float(projected.yaw_deg),
                "lateral_m": float(projected.source_lateral_m),
                "waypoint_index": int(projected.index),
                "blocked": bool(log.get("blocked")),
                "collided": bool(log.get("collided")),
                "adapter_loss": {
                    "lateral_dropped_m": float(projected.lateral_dropped_m),
                    "forward_clipped_m": float(projected.forward_clipped_m),
                    "yaw_clipped_deg": float(projected.yaw_clipped_deg),
                },
            })
            set_trail(moves=moves, motion=motion)
            if log.get("terminated_mid"):
                why = "回合结束" if env.episode_over else "移动被中断"
                break
        else:
            why = f"达到策略步数上限 {args.max_steps}"
    finally:
        client.close()
        set_active_lightnav_pointing([], restore_display=True)

    check_actuation(env)
    refresh(env, obs, step_id, force_map=True)
    finish_run(env, why or "结束", step_id)
    return obs, trail


def run_fm(env, ev, instruction, args, obs, trail):
    """FM driver: the closed-loop 3DGS flow-matching head drives the sim.

    Same signature and return contract as run_model / run_agent so main() can
    pick it. Unlike those two it does NOT hit a vLLM server: the FM head is
    loaded in-process by _load_fm() (forked transformers + torch, GPU).

    Per-step recipe mirrors verl_nav's ODE deployment (eval_ode_env_reward /
    extract_cond), which is the deployment that actually improved env reward:
    current RGB frame + a zero input_waypoint -> FMInfer.predict -> 32x3
    normalised ego (dx,dy,dyaw) -> _denorm_delta (anonymous stats) -> execute the
    chunk via vln_gym.fm_action.execute_waypoint, which reuses
    agent_tools.drive_turn/drive_forward (the same primitives run_agent uses,
    so it works on every backend). The model re-plans from the live frame each
    chunk, so it keeps reacting to what it sees instead of replaying one blind
    trajectory.

    Why zero history rather than the gps-delta history fm_policy.run_fm_case
    (case 5) builds: the trainer's own GRPO rollout uses zero input_waypoint
    + a single current frame (cond cached per episode), and that is the recipe
    validated to improve reward; building gps deltas needs a backend-correct
    world pose that habitat-vs-3DGS backends disagree on (see the gym env's
    world_xy/gps comment), so the history path is an open risk the trainer
    sidesteps and we sidestep too.
    """
    fm = _load_fm()
    if fm is None or fm["infer"] is None:
        set_state(status="error")
        return obs, trail
    check_actuation(env)
    step_id = 0
    moves = []
    start_xy, start_yaw = A._xz(env), A._yaw(env)
    motion = {"dist": 0.0, "bearing": 0.0, "yaw": 0.0,
              "steps": 0, "path_dist": 0.0}
    set_state(status="running", result="")
    set_trail(actions=[], moves=[], motion=motion)
    push_event("instr", text=instruction, tpl="FM 流匹配",
               frompose=CFG["frompose"])
    refresh(env, obs, step_id)

    def on_step(new_obs, _i):
        nonlocal step_id
        step_id += 1
        motion["steps"] = step_id
        update_pose_motion(env, start_xy, start_yaw, motion)
        set_trail(moves=moves, motion=motion)
        refresh(env, new_obs, step_id)
        if render_mode() == 'display':
            time.sleep(args.step_delay)

    wp_zero = np.zeros((1, 2), np.float32)
    FMA = _fm_action()
    if FMA is None:
        set_state(status="error")
        return obs, trail
    denorm, anonymous = fm["denorm"], fm["anonymous"]
    infer = fm["infer"]

    why = ""
    for chunk in range(MAX_FM_CHUNKS):
        if interrupted():
            CMD["stop"] = False
            set_state(status="stopped")
            push_event("sys", text="— 已停止 —")
            return obs, trail
        if env.episode_over:
            why = "回合结束"
            break
        if step_id >= args.max_steps:
            why = f"达到步数上限 {args.max_steps}"
            break

        frame = _cur_img(obs)
        capture_model_request([
            {"role": "user", "content": [
                _img_part(frame),
                {"type": "text", "text":
                 "【FM 专用结构化输入】\ninput_waypoint: zeros([1, 2])\n"
                 f"anonymous Instruction: {instruction}\nPredict a 32×3 ego trajectory chunk."},
            ]},
        ], chunk + 1, "fm")
        t0 = time.time()
        set_state(stream={"active": True, "text": "FM 推理中…",
                          "n": 0, "dt": 0.0})
        try:
            norm_delta, _gen_ids = infer.predict([frame], wp_zero, instruction)
        except Exception as exc:
            set_state(stream={"active": False, "text": "", "n": 0, "dt": 0.0},
                      status="error")
            push_event("err", text=f"FM 推理失败：{exc}")
            return obs, trail
        dt = time.time() - t0
        set_state(stream={"active": False, "text": "", "n": 0, "dt": 0.0})
        real = denorm(norm_delta, anonymous)
        nonzero = bool(np.abs(norm_delta).max() > 1e-3)
        push_event("model",
                   text=f"chunk {chunk + 1}/{MAX_FM_CHUNKS} "
                        f"norm[{norm_delta.min():.2f},{norm_delta.max():.2f}] "
                        f"nonzero={nonzero} 用时 {dt:.1f}s",
                   turn=chunk + 1, dt=dt, actions=[], abort=False)

        chunk_collided = False
        for wp in real:
            if env.episode_over:
                why = "回合结束"
                break
            if interrupted():
                break
            log = FMA.execute_waypoint(
                env, float(wp[0]), float(wp[1]), float(wp[2]), A,
                on_step=on_step, abort=interrupted, render=True)
            motion["path_dist"] += float(log.get("moved_m", 0.0))
            moves.append({"kind": "fm", "dx": float(wp[0]), "dy": float(wp[1]),
                          "dyaw": float(wp[2]),
                          "moved_m": float(log.get("moved_m", 0.0)),
                          "collided": bool(log.get("collided")),
                          "blocked": bool(log.get("blocked"))})
            if log.get("collided"):
                chunk_collided = True
            set_trail(moves=moves, motion=motion)
            if log.get("terminated_mid"):
                why = "回合结束"
                break
            # Standalone FM auto-stop follows SAGE's object region. Other
            # backends retain their point-goal stop radius override.
            _m = env.get_metrics()
            _d = float(_m.get("distance_to_goal", float("nan")))
            _r = float(_m.get("success_radius", float("nan")))
            _stop_r = _r
            _fm_stop = float(os.environ.get("FM_STOP_RADIUS", "0") or 0)
            if _fm_stop > 0:
                _stop_r = max(_r, _fm_stop) if (_r == _r) else _fm_stop
            _object_contract = "official_object_success" in _m
            _strict = goal_verdict(_m)[2]
            _should_stop = (_strict is True if _object_contract else
                            _d == _d and _stop_r == _stop_r and _d <= _stop_r)
            if _should_stop:
                try:
                    env.step(STOP)   # 注册 _stopped → episode_over，finish_run 报 success
                except Exception:
                    pass
                why = ("进入目标物体成功区域（FM 自动停）" if _object_contract else
                       "到达目标（FM 自动停）" if _strict
                       else f"接近目标 d={_d:.2f}m（停半径 {_stop_r:g}m，未进严格 {_r:g}m）")
                push_event("sys", text=why + "，FM 已自动停")
                break
        if interrupted():
            CMD["stop"] = False
            set_state(status="stopped")
            push_event("sys", text="— 已停止 —")
            return obs, trail
        if why:
            break
    else:
        why = f"达到 chunk 上限 {MAX_FM_CHUNKS}"

    check_actuation(env)
    refresh(env, obs, step_id, force_map=True)
    finish_run(env, why or "结束", step_id)
    return obs, trail


def _fm_action():
    """vln_gym.fm_action -- pure math (decompose/denormalize/execute_waypoint),
    no habitat import. Imported lazily so the legacy/react paths never pay for
    it and so a missing vln_gym does not break page startup."""
    try:
        from vln_gym import fm_action as FMA
        return FMA
    except Exception as exc:
        push_event("err", text=f"无法导入 vln_gym.fm_action：{exc}")
        return None


def _arc_resample_full(pts, step_m=0.1):
    """把世界系 (x, y) 折线按弧长 resample 成 ~step_m 等距点，镜像
    ``verl/sft/build_abotn_sft_data.arc_resample_future`` 的插值（累积弧长 +
    np.interp），保留首尾两点。SFT builder 用同一套 resample 把 golden path
    切成 0.1m/步的单步 delta GT，这里用它让「走专家轨迹」的步长划分与 SFT
    数据完全同口径。"""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts
    xs, ys = pts[:, 0], pts[:, 1]
    seg = np.hypot(xs[1:] - xs[:-1], ys[1:] - ys[:-1])
    total = float(seg.sum())
    if total < 1e-6:
        return pts
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(math.ceil(total / step_m)))
    targets = np.linspace(0.0, total, n)
    rx = np.interp(targets, cum, xs)
    ry = np.interp(targets, cum, ys)
    return np.stack([rx, ry], axis=1)


def run_golden(env, ev, instruction, args, obs, trail):
    """沿专家轨迹逐步做物理检查，按回放档位发布图像。

    golden_stride=1保留逐步核验；默认每4步渲染一次。运动采样点和碰撞
    检查不因显示帧率变化，终点、受阻点和中断点始终发布真实位姿画面。

    步长划分对齐 SFT 数据口径：把 golden 折线按弧长 resample 成 ~0.1m/步
    （env ``SAGE_GOLDEN_STEP_M`` 可调，默认 0.1，与
    ``build_*_sft_data.arc_resample_future`` 的 step_m 一致），逐点走到终点。
    每个resample点都执行 ``A.drive_turn`` 和 ``A.drive_forward``，保留
    原有子步与碰撞检查，但跳过中间渲染。到显示采样点时统一observe一次，
    ``capture_model_request`` 记录该点的帧和SFT口径单步ego delta，随后
    ``refresh`` 发布同一帧。精细档逐点记录，快速档只记录显示采样点。

    不碰 vLLM/FM，轨迹取 ``env._golden_pts(ep)``（top-down 地图橙线：默认官方
    ``ep.waypoints``，``SAGE_GOLDEN_ASTAR=1`` 时是 A* 物理可执行重规划）。转向
    用 sage ``sim.yaw``（+x→+y），行进用 ``agent_tools.drive_turn/drive_forward``
    （与 react/fm 同一套原语，碰撞/滑行/stall 一致）。仅 sage 有
    ``_golden_pts``，其它后端报「不支持」。
    """
    golden_fn = getattr(env, "_golden_pts", None)
    if ACTIVE.get("backend") != "sage" or not callable(golden_fn):
        push_event("err", text=GOLDEN_ONLY_MSG)
        set_state(status="idle")
        return obs, trail
    ep = env.current_episode
    wps = golden_fn(ep)
    if not wps or len(wps) < 2:
        push_event("err", text="这个用例没有专家轨迹"
                            "（waypoints 为空或仅一点）")
        set_state(status="idle")
        return obs, trail

    step_m = float(CFG.get("golden_step") or 0.1)
    if step_m <= 0:
        step_m = 0.1
    # Point-Goal 前瞻距离（米）：专家 path 上前方 waypoint 选点弧长，室内默认
    # 3m（sage 全室内），env SAGE_POINTGOAL_LOOKAHEAD_M / 页面「前瞻距离」可调。
    look_m = float(CFG.get("pointgoal_lookahead") or 3.0)
    if look_m <= 0:
        look_m = 3.0
    # 弧长 resample：dense 点 P[0..M]，P[0]≈start、P[M]=goal。A* 折线稀疏
    # （string-pull 后 5~8 点），官方 waypoints 已 ~0.15m 密，两者统一成 0.1m。
    P = _arc_resample_full([(float(w[0]), float(w[1])) for w in wps], step_m)
    M = len(P) - 1
    # 各点切向 heading（=到下一点的方位，末点沿用前一段）
    head = np.empty(len(P), dtype=np.float32)
    head[:-1] = np.arctan2(P[1:, 1] - P[:-1, 1], P[1:, 0] - P[:-1, 0])
    head[-1] = head[-2] if len(P) > 1 else 0.0

    def _wrap(rad):
        return ((rad + math.pi) % (2.0 * math.pi)) - math.pi

    # 从回合起点出发（A* 路径即从 start 规划，P[0]≈start）。
    obs = env.reset()
    sim = env.sim
    trail = []
    check_actuation(env)
    step_id = 0
    yaw0 = float(sim.yaw)
    moves = []
    motion = {"dist": 0.0, "yaw": 0.0, "steps": 0}
    env._pointgoal_wp = None      # 走专家轨迹时地图上画的前瞻 waypoint
    set_state(status="running", result="")
    set_trail(actions=[], moves=[], motion=motion)
    push_event("instr", text=f"沿专家轨迹行驶（{len(wps)} 参考点 → "
                            f"{M + 1} 步 × {step_m:g}m）",
               tpl="专家轨迹", frompose=False)
    refresh(env, obs, step_id)

    why = ""
    for k in range(1, M + 1):
        if interrupted():
            break
        if env.episode_over:
            why = "回合结束"
            break
        if step_id >= args.max_steps:
            why = f"达到步数上限 {args.max_steps}"
            break
        tx, ty = float(P[k][0]), float(P[k][1])

        # 1) 原地转向，朝当前→目标点方位（sage +x→+y 系，与 sim.yaw 同系）。
        cx, cy = sim.xy
        dx, dy = tx - cx, ty - cy
        dist = math.hypot(dx, dy)
        if dist < 0.005:
            pass  # 重合点，跳过行进仍出帧
        else:
            target_yaw = math.atan2(dy, dx)
            dyaw = _wrap(target_yaw - float(sim.yaw))
            deg = abs(math.degrees(dyaw))
            if deg >= A.MIN_TURN_DEG:
                direction = "left" if dyaw > 0.0 else "right"
                obs2, _tinfo = A.drive_turn(env, direction, deg,
                                            on_step=None, render=False,
                                            abort=interrupted)
                if obs2 is not None:
                    obs = obs2
                if env.episode_over:
                    why = "回合结束"
                    break
                if interrupted():
                    break
            # 2) 行进该步（≤step_m+，resample 后通常一步到位）。drive_forward
            #    仍按 SUBSTEP_M 细分和逐步碰撞；这里只关闭中间页面帧，走完按
            #    真实位姿若仍有残量补一步。
            guard = 0
            while dist >= A.MIN_FWD_M and not env.episode_over \
                    and not interrupted() and step_id < args.max_steps \
                    and guard < 64:
                guard += 1
                obs2, info = A.drive_forward(env, dist, on_step=None,
                                             abort=interrupted, render=False)
                if obs2 is not None:
                    obs = obs2
                moved = float(info.get("moved_m", 0.0))
                motion["dist"] = round(float(motion["dist"]) + moved, 3)
                moves.append({"kind": "golden", "moved_m": moved,
                               "blocked": bool(info.get("blocked"))})
                set_trail(moves=moves, motion=motion)
                if info.get("blocked"):
                    why = "专家轨迹受阻（撞障碍，未到达终点）"
                    break
                cx, cy = sim.xy
                dist = math.hypot(tx - cx, ty - cy)
            if why:
                break

        step_id += 1
        motion["steps"] = step_id
        motion["yaw"] = round(math.degrees(_wrap(float(sim.yaw) - yaw0)), 1)
        set_trail(moves=moves, motion=motion)
        if k % CFG["golden_stride"] and k != M:
            continue
        obs = sim.observe()

        # 3) 到达 resample 点 k：把当前帧塞进模型输出框（与 VLM 每 turn 出图
        #    同入口），文本给 SFT 口径单步 ego delta（relative_future 同款
        #    R(-yaw0) 旋转 + 相邻差），让用户能逐帧核验「专家在这一步看到啥、
        #    该输出啥 delta」。
        h_prev = float(head[k - 1])
        wdx, wdy = float(P[k][0]) - float(P[k - 1][0]), \
            float(P[k][1]) - float(P[k - 1][1])
        c, s = math.cos(h_prev), math.sin(h_prev)
        ego_dx = c * wdx + s * wdy
        ego_dy = -s * wdx + c * wdy
        ego_dyaw = _wrap(float(head[k]) - h_prev)
        _m = env.get_metrics()
        _d = float(_m.get("distance_to_goal", float("nan")))
        _dstr = f"{_d:.2f}m" if _d == _d else "NaN"
        # 4) Point-Goal 前瞻 waypoint（世界 x,y）：专家 path 上 ~look_m 前方、
        #    traversable+LOS 的中间 waypoint。存 env._pointgoal_wp 供 top-down
        #    地图画青色标记（render_top_down 读它）；**图像不画点**——FOV/可见
        #    性几何限制下不一定可见，按用户定调：不可见就只在地图上可视化。
        try:
            from sage.pointgoal import lookahead_waypoint as _lh
            env._pointgoal_wp = _lh(env._occ_for(env.current_episode.scene),
                                    P, sim.xy, look_m, env.robot_radius)
        except Exception as exc:
            env._pointgoal_wp = None
            push_event("err", text=f"Point-Goal 前瞻失败：{exc}")
        _wptxt = ""
        _wp = env._pointgoal_wp
        if _wp is not None:
            _wptxt = (f"\nPoint-Goal 前瞻 waypoint: ({_wp[0]:.2f},{_wp[1]:.2f})"
                      f" 距 {look_m:g}m 前方 (safe+LOS)")
        try:
            frame = _cur_img(obs)
            capture_model_request(
                [{"role": "user", "content": [
                    _img_part(frame),
                    {"type": "text", "text":
                     f"【专家轨迹 step {k}/{M}】\n"
                     f"GT delta(SFT 口径): dx={ego_dx:.3f} dy={ego_dy:.3f} "
                     f"dyaw={math.degrees(ego_dyaw):.1f}°\n"
                     f"位姿: ({float(sim.xy[0]):.2f},{float(sim.xy[1]):.2f}) "
                     f"yaw={math.degrees(float(sim.yaw)):.1f}° "
                     f"距参考终点={_dstr}{_wptxt}"},
                ]}],
                k, "golden")
        except Exception as exc:
            push_event("err", text=f"走专家轨迹出帧失败：{exc}")
        refresh(env, obs, step_id)
        if render_mode() == 'display':
            time.sleep(args.step_delay)

    # Hidden physics steps may have ended at an obstacle or user stop.
    # Never publish an old frame paired with the new pose/depth.
    obs = sim.observe()
    if interrupted():
        CMD["stop"] = False
        set_state(status="stopped")
        push_event("sys", text="— 已停止 —")
        check_actuation(env)
        refresh(env, obs, step_id, force_map=True)
        return obs, trail

    # 不向 simulator 发送 STOP。STOP 会把 Sage 回合永久置为 episode_over，
    # 导致 worker 随后拒绝 WASD 手动驾驶。和其它 driver 一样，只结束本次
    # driver；成功与 SPL 由 finish_run 按后端评测口径独立计算，agent
    # 保持在专家轨迹终点且仍可继续手动移动。
    check_actuation(env)
    env._pointgoal_wp = None      # 清掉前瞻标记，走完不再画
    refresh(env, obs, step_id, force_map=True)
    finish_run(env, why or "专家轨迹走完", step_id)
    return obs, trail


def build_habitat(args):
    """The original backend: habitat Env over the R2R split + RynnBrain evaluator."""
    base = load_habitat()
    # This evaluator is created here to configure the Habitat environment; the
    # console's model dropdown points it at the selected server later via
    # apply_model().  Do not make simulator startup depend on a legacy seed URL.
    vllm_server_urls = os.environ.get("RYNNVLN_VLLM_SERVER_URLS", "").strip()
    eval_args = argparse.Namespace(
        local_rank=0, model_path=os.environ.get("MODEL_PATH", ""),
        habitat_config_path=DEMO_CONFIG, eval_split=args.split, output_path=args.out,
        num_future_steps=4, num_frames=32, save_video=False, num_history=8,
        max_eval_episodes_per_rank=0,
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "2048")),
        prompt_style="embodied_r1", num_actions=4, max_history_frames=16,
        open_loop_steps=0,
        inference_backend="vllm_server" if vllm_server_urls else "hf",
        vllm_server_urls=vllm_server_urls,
        vllm_model_name=os.environ.get("RYNNVLN_VLLM_MODEL_NAME",
                                       P.model_endpoint(P.DEFAULT_MODEL)[1]),
        vllm_timeout=300, vllm_retries=3,
        vllm_temperature=float(os.environ.get("RYNNVLN_VLLM_TEMPERATURE", "0")),
        vllm_top_p=None, vllm_api_key="",
        vllm_image_format=os.environ.get("RYNNVLN_VLLM_IMAGE_FORMAT", "JPG"),
        vllm_image_quality=int(os.environ.get("RYNNVLN_VLLM_IMAGE_QUALITY", "90")),
        enable_thinking=os.environ.get("RYNNVLN_ENABLE_THINKING", "0") == "1",
        model_max_length=4096, world_size=1, rank=0, gpu=0, port="1111",
        dist_url="env://", device="cpu",
    )
    base.init_distributed_mode(eval_args)

    ev = base.VLNEvaluator(config_path=DEMO_CONFIG, split=args.split, env_num=1,
                           output_path=args.out, model=None, processor=None,
                           epoch=0, args=eval_args)
    from habitat_topdown import register as register_floor_map
    import habitat
    register_floor_map()
    with habitat.config.read_write(ev.config):
        map_config = ev.config.habitat.task.measurements.top_down_map
        map_config.type = "FloorTopDownMap"
        map_config.draw_shortest_path = False
        map_config.draw_view_points = False
        map_config.draw_goal_positions = False
        map_config.draw_goal_aabbs = False
    return ev.config_env(), ev


def build_rxr(args):
    """RxR val_unseen, driven by the habitat backend's simulator.

    No second simulator and no second Env: RxR's 11 val_unseen scenes are R2R's,
    so this is one episode list wrapped around the habitat backend's Env. The
    parsing and the sharing are both explained in rxr.py -- read that first.
    """
    import rxr as _rxr
    load_habitat()
    host, ev = ENVS.get("habitat"), EVS.get("habitat")
    if host is None:
        # --backend rxr on its own: there is nothing to share with, so bring up
        # the R2R Env anyway and keep it private. Same simulator either way; the
        # only cost is that its episode list goes unused.
        host, ev = build_habitat(args)
    eps, missing = _rxr.load_episodes(split=args.split)
    if not eps:
        raise SystemExit(
            "RxR 没有可用回合：确认 %s 存在，且 RXR_LANGUAGES=%s 能匹配上"
            % (_rxr.DATA_PATH.format(split=args.split, role=_rxr.ROLES.split(",")[0]),
               _rxr.LANGUAGES))
    if missing:
        print("[sim] RxR 跳过 %d 个场景（MP3D 网格不在磁盘上）：%s"
              % (len(missing), ",".join(missing)), flush=True)
    print("[sim] RxR %s（%s，每场景 %s 个）：%d 回合 / %d 场景，共用 R2R 的仿真器"
          % (args.split, _rxr.LANGUAGES, _rxr.EPISODES_PER_SCENE or "全部",
             len(eps), len({scene_of(e) for e in eps})), flush=True)
    return _rxr.SharedSimEnv(host, eps), ev


def build_vlnce_train(args, family, role="guide"):
    """Mount full training splits on the already configured Habitat simulator."""
    import vlnce_training

    load_habitat()
    keys = ("habitat", "rxr", "habitat_train", "rxr_train_guide",
            "rxr_train_follower")
    name = next((key for key in keys if key in ENVS), None)
    if name is None:
        host_args = copy.copy(args)
        host_args.split = "train"
        host, ev = build_habitat(host_args)
    else:
        host, ev = ENVS[name], EVS[name]
    episodes, missing = vlnce_training.load_episodes(
        family, r2r_config=host._config.dataset, role=role)
    if not episodes:
        raise RuntimeError("%s train has no episodes with available MP3D meshes"
                           % family)
    print("[sim] %s train %s: %d episodes / %d scenes; missing meshes: %s"
          % (family, role if family == "rxr" else "",
             len(episodes), len({scene_of(ep) for ep in episodes}),
             ",".join(missing) or "none"), flush=True)
    return vlnce_training.SharedSimEnv(host, episodes), ev


def build_rs3d(args):
    """3DRS backend: 3DGS render server for vision, .navmesh for kinematics.

    No habitat, and no dataset: 3DRS ships scenes, so episodes are sampled
    spawn poses (see rs3d.env) and goal-relative metrics stay NaN.  Sensor and
    actuation values are read off vln_demo.yaml so both backends move the robot
    the same way.
    """
    from rs3d.env import RS3DEnv, list_scenes
    from rs3d.model_client import from_env as model_client_from_env

    scenes = [s for s in args.scenes.split(",") if s] or list_scenes(args.rs3d_root)
    sim_cfg = habitat_sim_params(DEMO_CONFIG)
    print(f"[sim] 3DRS 后端：{len(scenes)} 个场景，渲染服务 {args.render_url}", flush=True)
    env = RS3DEnv(scenes, rs3d_root=args.rs3d_root, render_url=args.render_url,
                  episodes_per_scene=args.episodes_per_scene,
                  want_depth=True, **sim_cfg)
    if not env.client.ping():
        raise SystemExit(
            f"渲染服务 {args.render_url} 没有响应。先起它：\n"
            f"  {RS3D_SERVER_HINT}")
    return env, model_client_from_env()


def build_hsim(args, key):
    """One of the habitat_sim asset sets under ``agentic_sim/benchmark``.

    Same simulator as the R2R backend but without habitat-lab: these downloads
    ship scenes (and, for PARTNR, episodes) rather than a VLN task, so ``hsim``
    talks to habitat_sim directly. Sensor and actuation values come from the demo
    yaml so every backend moves the robot identically; rendering is CPU, like the
    R2R one. Imported here rather than at module scope because pulling in
    habitat_sim requires ``LD_LIBRARY_PATH`` to have been set before exec.
    """
    from hsim.datasets import available
    from hsim.env import HabSceneEnv
    from rs3d.model_client import from_env as model_client_from_env

    sets = available()
    if key not in sets:
        raise SystemExit("这台机器上没有 %s 的资产文件" % key)
    ds = sets[key]
    scenes = [s for s in args.hsim_scenes.split(",") if s] or None
    env = HabSceneEnv(ds, scenes, want_depth=True,
                      episodes_per_scene=args.hsim_episodes_per_scene,
                      **habitat_sim_params(DEMO_CONFIG))
    print("[sim] %s 后端：%d 回合 / %d 场景" % (
        ds.name, len(env.episodes), len({e.scene for e in env.episodes})),
        flush=True)
    return env, model_client_from_env()


def build_abotn(args, key):
    """One of the ABotN Gaussian benchmarks (POIBench / PointBench).

    Vision is a 3DGS render server as with 3DRS, but a *second instance*: the
    ABotN plys are spherical-harmonic degree 3 while 3DRS's are degree 0, and
    the server fixes that per process. Kinematics come from each scene's
    occupancy grid rather than a navmesh. Unlike every other non-R2R set these
    ship real episodes -- start pose, goal point, reference path, and for
    POIBench a Chinese instruction -- so the metrics are real numbers.
    """
    from abotn.datasets import available
    from abotn.env import ABotNEnv
    from rs3d.model_client import from_env as model_client_from_env

    sets = available(args.abotn_root)
    if key not in sets:
        raise SystemExit("这台机器上没有 %s 的资产文件（找的是 %s）"
                         % (key, args.abotn_root))
    ds = sets[key]
    scenes = [s for s in args.abotn_scenes.split(",") if s] or None
    sim_cfg = habitat_sim_params(DEMO_CONFIG)
    env = ABotNEnv(ds, scenes, render_url=args.abotn_render_url,
                   episodes_per_scene=args.abotn_episodes_per_scene,
                   want_depth=True, **sim_cfg)
    if not env.client.ping():
        raise SystemExit(
            "渲染服务 %s 没有响应。先起它（注意 --sh_degree 3，和 3DRS 那个不是同一个进程）：\n"
            "  %s" % (args.abotn_render_url, ABOTN_SERVER_HINT))
    print("[sim] %s：%d 回合 / %d 场景" % (
        ds.label, len(env.episodes), len({e.scene for e in env.episodes})),
        flush=True)
    return env, model_client_from_env()


def build_sage(args):
    """SAGE-3D GVLN: InteriorGS Gaussian scenes + dense z-up reference paths.

    Same shape as :func:`build_abotn` -- a second degree-3 render server (port
    7038 by default) plus occupancy-grid kinematics -- but one backend with a
    ``--sage_split`` selector (train / instr / scene / traj) rather than three
    asset sets, since only the scored trajectories differ across splits. GVLN
    ships start pose + goal + reference path + English instruction, so the
    metrics are real numbers.

    The default split is ``train`` (929 scenes / ~900+ scenes visible in the
    viz), not ``instr`` (15): viz is meant to browse the whole asset, and the
    console parses one episode per scene's task list up front. ``train`` is 929
    scenes x ~2385 tasks ~= 2.2M episodes, so the default also caps
    ``--sage_episodes_per_scene`` at 5 (per-scene task parse, not after) -- the
    dominant startup cost is the one-time per-scene ``frame_offset`` solve, not
    episode count. Set ``SAGE_SPLIT=instr`` for the 15-scene Instruction-Unseen
    split, or ``SAGE_EPISODES_PER_SCENE=0`` to take every task.
    """
    from sage.datasets import available
    from sage.env import SageEnv
    from rs3d.model_client import from_env as model_client_from_env

    sets = available(args.sage_root)
    if args.sage_split not in sets:
        raise SystemExit(
            "这台机器上没有 SAGE-3D %s split 的资产（找的是 %s）。先转场景：\n  %s"
            % (args.sage_split, args.sage_root, SAGE_SERVER_HINT))
    ds = sets[args.sage_split]
    scenes = [s for s in args.sage_scenes.split(",") if s] or None
    sim_cfg = habitat_sim_params(DEMO_CONFIG)
    # Sage's official Isaac environment uses a 1.2 m camera and an 8 mm USD
    # lens (~105.3 deg HFOV). Do not inherit Habitat/Navid's 120 deg sensor.
    sim_cfg["hfov"] = float(os.environ.get("SAGE_HFOV", "105.3"))
    env = SageEnv(ds, scenes, render_url=args.sage_render_url,
                  episodes_per_scene=args.sage_episodes_per_scene,
                  want_depth=True, **sim_cfg)
    if not env.client.ping():
        raise SystemExit(
            "渲染服务 %s 没有响应（SAGE 用独立的 degree-3 服务，不是 3DRS/ABotN 那个）：\n"
            "  %s" % (args.sage_render_url, SAGE_SERVER_HINT))
    print("[sim] %s：%d 回合 / %d 场景；collision=%s furniture=%s radius=%.2fm" % (
        ds.label, len(env.episodes), len({e.scene for e in env.episodes}),
        env.collision_mode, env.furniture_mode, env.robot_radius), flush=True)
    return env, model_client_from_env()


def build_mall(args):
    """mall_floor2（万科里商场，local-user-06 自采 GS）：真实可评分的新环境。

    1225 个真实起点 × 商铺目标（指令「前往<店名>」，到店 2 m 算到达），
    运动学是膨胀 walk mask 上的严格语义（0.25 m 步进 / 15° 转向 / 撞墙整体
    拒绝，无滑动），与 zekang 闭环训练用的 /move 服务逐字一致。相机口径
    1280×720、hfov 90°（fx=640）、绝对高 1.5 m——是训练数据标定值，**不从
    vln_demo.yaml 继承**（那是 640×480/120° 的 R2R 口径）。渲染与 3DRS 同
    /render_gs 协议，GSRenderClient 直接复用；degree-0 ply 但 scenes_root 在
    zekang 盘，需独立服务进程。
    """
    from mall.assets import MallAssets
    from mall.env import MallEnv
    from rs3d.model_client import from_env as model_client_from_env

    ds = MallAssets(args.mall_assets_dir)
    if not ds.available():
        raise SystemExit(
            "这台机器上没有 mall 资产（找的是 %s，应有 walk.json / "
            "floor_meta.json / mall_walk_union_topo.npy）" % ds.root)
    ds.load()
    env = MallEnv(ds, render_url=args.mall_render_url,
                  episodes=args.mall_episodes)
    if not env.client.ping():
        raise SystemExit(
            "渲染服务 %s 没有响应（mall 是 degree-0，但 scenes_root 与 3DRS "
            "不同，需独立进程）：\n  %s" % (args.mall_render_url, MALL_SERVER_HINT))
    print("[sim] mall_floor2：%d 回合（%d 起点 × %d 商铺目标），资产 %s" % (
        len(env.episodes), len(ds.starts), len(ds.goal_shops()), ds.root),
        flush=True)
    return env, model_client_from_env()


def build_lepu(args):
    """Lepu office assets share mall's locnav_arr simulator and render service."""
    from lepu.assets import LepuAssets
    from lepu.env import LepuEnv
    from rs3d.model_client import from_env as model_client_from_env

    ds = LepuAssets(args.lepu_assets_dir)
    if not ds.available():
        raise SystemExit("Lepu 资产缺失：%s（需要 walk.json、floor_meta.json、"
                         "lepu_walk_union_g45.npy）" % ds.root)
    ds.load()
    env = LepuEnv(ds, render_url=args.lepu_render_url, episodes=args.lepu_episodes)
    if not env.client.ping():
        raise SystemExit("Lepu 与 mall 共用的渲染服务无响应：%s" % args.lepu_render_url)
    print("[sim] Lepu：%d 回合，%d 个房间与设施目标，资产 %s" % (
        len(env.episodes), len(ds.goal_shops()), ds.root), flush=True)
    return env, model_client_from_env()


def build_lightnav_mujoco(args):
    """Mount LightNav-0's bundled MuJoCo scene through its remote web API.

    The server owns the Python 3.11 MuJoCo/CasADi environment.  Keeping it out
    of process is intentional: the console's environment only implements the
    same habitat-shaped facade as the 3DGS backends and has no MuJoCo import.
    """
    from lightnav_mujoco.env import LightNavMujocoEnv
    from rs3d.model_client import from_env as model_client_from_env

    sim_cfg = habitat_sim_params(DEMO_CONFIG)
    env = LightNavMujocoEnv(
        args.lightnav_mujoco_url,
        timeout=args.lightnav_mujoco_timeout,
        forward_step=float(sim_cfg.get("forward_step", 0.25)),
        turn_angle=float(sim_cfg.get("turn_angle", 30.0)),
        linear_speed=args.lightnav_mujoco_linear_speed,
        angular_speed=args.lightnav_mujoco_angular_speed,
    )
    print("[sim] LightNav MuJoCo：1 场景（远端 %s；无目标指标）" %
          args.lightnav_mujoco_url, flush=True)
    return env, model_client_from_env()


# Which simulator each scene belongs to, and which one is driving right now.
# Two asset sets can be mounted at the same time (``--backend both``); picking a
# scene in the UI is what switches simulator, so every driver call has to read
# the env out of here rather than closing over one.
def build_objectnav(args, category=None):
    from objectnav.env import ObjectNavEnv
    from rs3d.model_client import from_env
    return ObjectNavEnv(category=category), from_env()

BACKENDS = {
    "habitat": {
        "label": "R2R-CE / Val Unseen",
        "desc": "R2R val_unseen：真人写的导航指令 + ground-truth 路径，指标可评分。",
        "goalless": False,
        "build": lambda a: build_habitat(a),
    },
    "rxr": {
        "label": "RxR-CE / Val Unseen / Guide",
        "desc": "RxR val_unseen 英文 guide 标注：还是那 11 个 MP3D 场景（和 R2R 共用一个"
                "仿真器），但指令长得多、一段一段描述沿途看到什么，指标可评分。",
        "goalless": False,
        "build": lambda a: build_rxr(a),
    },
    "habitat_train": {
        "label": "R2R-CE / Train",
        "desc": "R2R_VLNCE_v1 / train / English / MP3D / Habitat",
        "goalless": False,
        "build": lambda a: build_vlnce_train(a, "r2r"),
    },
    "rxr_train_guide": {
        "label": "RxR-CE / Train / Guide",
        "desc": "RxR_VLNCE_v0 / train / Guide / English, Hindi, Telugu / MP3D / Habitat",
        "goalless": False,
        "build": lambda a: build_vlnce_train(a, "rxr", "guide"),
    },
    "rxr_train_follower": {
        "label": "RxR-CE / Train / Follower",
        "desc": "RxR_VLNCE_v0 / train / Follower / English, Hindi, Telugu / MP3D / Habitat",
        "goalless": False,
        "build": lambda a: build_vlnce_train(a, "rxr", "follower"),
    },
    "3drs": {
        "label": "自采 3DGS (3DRS)",
        "desc": "自采高斯溅射场景，视觉走 3DGS 渲染服务。指令框预填资产自带的中文描述。",
        "goalless": True,
        "build": lambda a: build_rs3d(a),
    },
    "partnr": {
        "label": "PARTNR / HSSD (ABot)",
        "desc": "HSSD 合成户型 + PARTNR 官方回合，起点与英文任务指令都是数据集原文；"
                "PARTNR 评的是重排命题不是路径，所以没有路径指标。",
        "goalless": True,
        # Off the dropdown by default, same as `hab` below: the HSSD render is
        # poor enough that it misrepresents what the console can do, and it is
        # the slowest set to mount (software GL) for something nobody demos.
        # Still reachable on purpose with DEMO_BACKEND=partnr.
        "hidden": True,
        "build": lambda a: build_hsim(a, "partnr"),
    },
    "hab": {
        "label": "Habitat-HAB / ReplicaCAD (ABot)",
        "desc": "hab2 基准的两个 ReplicaCAD 场景，只有场景没有导航回合，"
                "出生点是采样的、指令要自己写——纯自由漫游。",
        "goalless": True,
        # Off the dropdown until the render is fixed: alone in its own process
        # both ReplicaCAD scenes look right, but mounted next to the other six
        # the picture comes out a red wash, and switching to one of them killed
        # the console outright (silent death mid-request, no traceback --
        # habitat_sim aborting inside C++). Not worth showing a broken asset set
        # that can also take the working ones down with it. Still reachable on
        # purpose with DEMO_BACKEND=hab, which is where the debugging happens.
        "hidden": True,
        "build": lambda a: build_hsim(a, "hab"),
    },
    "poi": {
        "label": "POI 导航 / ABotN-POIBench",
        "desc": "真实街区高斯场景 + 中文 POI 指令（「前往111医药馆」），"
                "带真值轨迹，到店门口 2 m 算到达，指标可评分。",
        "goalless": False,
        "build": lambda a: build_abotn(a, "poi"),
    },
    "in": {
        "label": "室内点导航 / ABotN-PointBench",
        "desc": "室内高斯户型，目标是一个坐标点而不是一句话；"
                "指令框里写的是坐标和直线距离，到达阈值 0.5 m。",
        "goalless": False,
        "build": lambda a: build_abotn(a, "in"),
    },
    "out": {
        "label": "室外点导航 / ABotN-PointBench",
        "desc": "室外街区/公园高斯场景，地面有起伏，目标同样是坐标点，"
                "到达阈值 0.5 m。",
        "goalless": False,
        "build": lambda a: build_abotn(a, "out"),
    },
    "sage": {
        "label": "SAGE-3D GVLN (InteriorGS)",
        "desc": "InteriorGS 室内高斯场景 + GVLN 稠密真值轨迹，英文因果/指代导航指令，"
                "起点/目标/参考路径俱全，指标可评分；capability split 在一个页面统一承载"
                "Perception/Grounding/Planning/Recovery/Execution。",
        "goalless": False,
        "build": lambda a: build_sage(a),
    },
    "mall": {
        "label": "mall_floor2 / 万科里商场 (3DGS)",
        "desc": "自采商场高斯场景（1280×720，hfov 90°，相机高 1.5 m），"
                "1225 个真实起点 × 商铺目标（「前往OCHIRLY」），膨胀 walk mask 严格运动学"
                "（0.25 m 步进 / 15° 转向 / 撞墙整体拒绝），到店门口 2 m 算到达，指标可评分。",
        "goalless": False,
        "build": lambda a: build_mall(a),
    },
    "lepu": {
        "label": "Lepu 办公层 / jW00XG5o3g (3DGS)",
        "desc": "7381 个真实起点、19 个房间与设施目标；与万科里商场共用仿真器和"
                "渲染服务。使用膨胀可走域，抵达半径 2 m；无拓扑图，SPL 距离按欧氏估计。",
        "goalless": False,
        "build": lambda a: build_lepu(a),
    },
    "lightnav_mujoco": {
        "label": "LightNav-0 / MuJoCo",
        "desc": "LightNav-0 自带的 ProcTHOR MuJoCo 场景；Python 3.11 服务独立运行，"
                "控制台经 HTTP/WebSocket 复用第一视角、位姿和运动控制。无目标点，"
                "右侧地图仅显示里程计轨迹。",
        "goalless": True,
        "build": lambda a: build_lightnav_mujoco(a),
    },
    "objectnav": {
        "label": "ObjectNav / HM3D · MP3D · HabitatGS",
        "desc": "Gary SAM3 candidates / selected goal viewpoint scoring (not official ObjectNav)",
        # Batch controllers select an episode explicitly; the UI uses Gary's categories.
        "hidden": True,
        "build": build_objectnav,
    },
}
from objectnav.categories import backend_definitions as objectnav_backend_definitions
BACKENDS.update(objectnav_backend_definitions(build_objectnav, bool(os.environ.get("OBJECTNAV_MANIFEST"))))
ACTIVE = {"env": None, "ev": None, "backend": ""}
SCENE_TITLES = {}              # scene id -> short readable title when available
SCENE_NAMES = {}               # scene id -> observe name, numbered iff duplicated
ENVS = {}                      # backend -> env
EVS = {}                       # backend -> RynnBrain evaluator (rxr borrows habitat's)
SCENE_BACKEND = {}             # scene id -> backend


def resolve_backends(spec):
    """``--backend`` value -> list of backend keys, in dropdown order.

    ``all`` is every asset set that is not marked ``hidden``, ``both`` keeps
    meaning the original pair (R2R + 3DRS) so existing invocations behave as
    before, and a comma-separated list picks exactly what you name -- naming a
    hidden one explicitly is allowed, that is the point of naming it.
    """
    if spec == "all":
        return [k for k, b in BACKENDS.items() if not b.get("hidden")]
    if spec == "both":
        return ["habitat", "3drs"]
    names = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [n for n in names if n not in BACKENDS]
    if bad or not names:
        raise SystemExit("--backend 认不出 %s；可选：%s，或 both / all"
                         % (",".join(bad) or "空值", "/".join(BACKENDS)))
    return names


def disambiguate_scene_names(scenes, titles):
    """Build stable observe names once for the mounted scene catalogue.

    Human-readable titles stay untouched when unique. Duplicate titles get a
    deterministic 1..N suffix ordered by the opaque scene id; the id itself is
    deliberately not exposed to the agent as part of the readable name.
    """
    groups = {}
    for scene in scenes:
        name = titles.get(scene) or scene
        groups.setdefault(name, []).append(scene)

    names = {}
    for name, ids in groups.items():
        ordered = sorted(ids)
        if len(ordered) == 1:
            names[ordered[0]] = name
        else:
            for number, scene in enumerate(ordered, 1):
                names[scene] = "%s %d" % (name, number)
    return names


def mount_backends(args):
    """Build every requested backend; returns the scene -> backend map.

    A backend that fails to come up is reported and skipped rather than taken
    down with the others: losing the render server should not cost you the
    habitat scenes, and not having habitat installed should not cost you 3DRS.
    """
    wanted = resolve_backends(args.backend)
    # Build in BACKENDS order however they were named: rxr shares the habitat
    # backend's simulator, so habitat has to be up first or rxr quietly builds a
    # second one. Dropdown order comes from BACKENDS anyway (see main()).
    wanted = [k for k in BACKENDS if k in set(wanted)]
    for name in wanted:
        try:
            env, ev = BACKENDS[name]["build"](args)
        except BaseException as exc:                      # SystemExit included
            if len(wanted) == 1:
                raise
            print("[sim] %s 后端起不来，跳过：%s" % (BACKENDS[name]["label"], exc),
                  flush=True)
            continue
        ENVS[name] = env
        EVS[name] = ev
        for ep in env.episodes:
            SCENE_BACKEND[scene_of(ep)] = name
    if not ENVS:
        raise SystemExit("没有一个后端起得来")
    # stream_chat only needs a server url, a few args and _to_openai_messages, so
    # one client serves both simulators; prefer habitat's evaluator when present
    # because the classic driver also reads ev.max_history_frames off it.
    ACTIVE["ev"] = EVS.get("habitat") or next(iter(EVS.values()))
    return SCENE_BACKEND


def use_backend(name):
    """Point the console at one of the mounted simulators."""
    if name not in ENVS or name == ACTIVE["backend"]:
        return ACTIVE["env"]
    global BASE_AMOUNTS
    ACTIVE["backend"] = name
    ACTIVE["env"] = ENVS[name]
    BASE_AMOUNTS = A.default_amounts(ACTIVE["env"])
    set_state(backend=name, backend_label=BACKENDS[name]["label"])
    return ACTIVE["env"]


def backend_of(scene, default=None):
    return SCENE_BACKEND.get(scene, default or ACTIVE["backend"])


def is_goalless(backend=None):
    """True when the active asset set ships no ground-truth path to score."""
    return bool(BACKENDS.get(backend or ACTIVE["backend"], {}).get("goalless"))


def guard_driver(announce=False):
    """Force a goal-independent driver on goalless asset sets.

    The classic driver scores its trajectory against the episode's ground-truth
    path; 3DRS has none, so leaving it selectable would just produce a run whose
    numbers mean nothing. ReAct and FM both drive without a goal to score
    against, so either is fine on a goalless set. The override is remembered,
    not destructive: switching back to an asset set with goals restores
    whatever the dropdown last chose. Returns the driver actually in force.
    """
    if is_goalless():
        if CFG["driver"] not in (
                "react", "action_head", "action_head_untrained",
                "action_head_2800", "action_head_10ep", "action_head_v43_7000", "action_head_v5_native", "action_head_v5_cached", "action_head_planner", "embodied_point",
                "unilavira_bbox", "fm", "lightnav"):
            CFG["forced_from"] = CFG["driver"]
            CFG["driver"] = "react"
            set_state(driver="react")
            if announce:
                push_event("sys", text="这套资产没有目标点，经典驱动无从评分，已切回 ReAct")
    elif CFG.get("forced_from"):
        CFG["driver"] = CFG.pop("forced_from")
        set_state(driver=CFG["driver"])
        if announce:
            push_event("sys", text="这套资产有目标点，已恢复驱动：" + CFG["driver"])
    return CFG["driver"]


def _enforce_model_driver_lock(announce=False):
    """Lock independently served policy models to their matching drivers.

    * FM 模型（``P.is_fm_model``）→ 驱动强制 fm，并把该模型的 ckpt 写进
      ``CFG["fm_ckpt"]``（_load_fm 优先读它）；非 fm 驱动被覆盖。
    * LightNav-0 模型 → 驱动强制 lightnav，不走普通 chat endpoint。
    * 普通模型 → 不能使用 fm/lightnav，若残留则退回 DEFAULT_DRIVER。
    返回强制后的 driver。set_model/set_driver/start/dispatch 都调一次，保证硬约束。
    """
    model = CFG.get("model") or P.DEFAULT_MODEL
    if P.is_fm_model(model):
        ck = P.fm_ckpt_for(model)
        if ck:
            CFG["fm_ckpt"] = ck   # _load_fm: CFG["fm_ckpt"] or FM_CKPT env or fm_policy 默认
        if CFG.get("driver") != "fm":
            CFG["driver"] = "fm"
            set_state(driver="fm")
            if announce:
                push_event("sys", text="VLN-FM 模型驱动已锁 FM·流匹配")
    elif P.is_lightnav_model(model):
        if CFG.get("driver") != "lightnav":
            CFG["driver"] = "lightnav"
            set_state(driver="lightnav")
            if announce:
                push_event("sys", text="LightNav-0 模型驱动已锁 LightNav-0·轨迹")
    else:
        if CFG.get("driver") in ("fm", "lightnav"):
            old_driver = CFG.get("driver")
            CFG["driver"] = P.DEFAULT_DRIVER
            set_state(driver=P.DEFAULT_DRIVER)
            if announce:
                name = "FM" if old_driver == "fm" else "LightNav"
                push_event("sys", text=f"{name} 驱动需选择对应模型，已切回 " +
                           P.DRIVERS[P.DEFAULT_DRIVER][0])
    return CFG["driver"]


def habitat_sim_params(config_path):
    """Read sensor/actuation values out of the demo's habitat yaml.

    Parsed as plain YAML rather than through hydra: the point is for the 3DRS
    backend to move and see exactly like the habitat one without importing
    habitat to find out how. Falls back to this file's defaults if the yaml
    moved or the keys are absent.
    """
    import yaml
    out = {}
    try:
        p = config_path if os.path.isabs(config_path) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), config_path)
        with open(p) as f:
            cfg = (yaml.safe_load(f) or {}).get("habitat", {}).get("simulator", {})
        sens = ((cfg.get("agents", {}) or {}).get("main_agent", {})
                .get("sim_sensors", {}) or {}).get("rgb_sensor", {}) or {}
        for key, src in (("width", "width"), ("height", "height"), ("hfov", "hfov")):
            if src in sens:
                out[key] = sens[src]
        if "forward_step_size" in cfg:
            out["forward_step"] = float(cfg["forward_step_size"])
        if "turn_angle" in cfg:
            out["turn_angle"] = float(cfg["turn_angle"])
        v0 = cfg.get("habitat_sim_v0", {}) or {}
        if "allow_sliding" in v0:
            out["allow_sliding"] = bool(v0["allow_sliding"])
    except Exception as exc:
        print(f"[sim] 读不出 {config_path}（{exc}），用内置默认值", flush=True)
    return out


def main():
    global ACTION_HEAD_DATASET_SPLIT
    batch_tick = lambda env: None
    if os.environ.get("VLN_BATCH_TOKEN"):
        # RynnBrain also ships a vln_demo.py; load the batch adapter by file so
        # its module cannot shadow this repository's directory.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "vln_batch_controller", os.path.join(REPO_ROOT, "vln_demo/batch/controller.py"))
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        batch_tick = adapter.install(sys.modules[__name__])
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--action-head-as-tool-version", choices=("v1", "v2", "v3", "v4", "v4.1", "v4.2", "v4.3", "v4.4", "v4.5", "v4.6", "v4.7", "v4.8", "v4.9", "v4.10", "v4.11", "v6"), default="v1",
        help="Versioned Action Head-as-Tool prompt/result/context protocol. "
             "Default v1 preserves the existing behavior.")
    ap.add_argument("--backend", default=os.environ.get("DEMO_BACKEND", "all"),
                    help="挂哪几套资产：all（默认，除隐藏的以外全挂）/"
                         " both（habitat+3drs）/ 逗号分隔的 "
                         + "、".join(BACKENDS)
                         + "（hab 目前渲染坏了，只能显式点名）；"
                           "页面上先选资产集再选场景")
    ap.add_argument("--hsim_scenes", default=os.environ.get("HSIM_SCENES", ""),
                    help="PARTNR/ReplicaCAD 场景白名单（逗号分隔，默认全部）")
    ap.add_argument("--hsim_episodes_per_scene", type=int,
                    default=int(os.environ.get("HSIM_EPISODES_PER_SCENE", "3")),
                    help="PARTNR 每个场景最多取几个官方回合（ReplicaCAD 恒为 1）")
    ap.add_argument("--render_url",
                    default=os.environ.get("RS3D_RENDER_URL", "http://127.0.0.1:7036"),
                    help="3DRS 后端的 3DGS 渲染服务地址")
    ap.add_argument("--rs3d_root", default=os.environ.get(
        "RS3D_ROOT", "/workspace/agentic_sim/3DRS"))
    ap.add_argument("--scenes", default=os.environ.get("RS3D_SCENES", ""),
                    help="3DRS 后端场景白名单（逗号分隔，默认全部 143 个）")
    ap.add_argument("--episodes_per_scene", type=int,
                    default=int(os.environ.get("RS3D_EPISODES_PER_SCENE", "1")),
                    help="3DRS 后端每个场景采样多少个出生点（默认 1）")
    ap.add_argument("--abotn_root", default=os.environ.get(
        "ABOTN_ROOT", "/workspace/agentic_sim/zytoss"),
        help="ABotN 三个 benchmark 目录的父目录")
    ap.add_argument("--abotn_render_url",
                    default=os.environ.get("ABOTN_RENDER_URL", "http://127.0.0.1:7037"),
                    help="ABotN 的 3DGS 渲染服务地址（sh_degree 3，和 3DRS 那个不是同一个进程）")
    ap.add_argument("--abotn_scenes", default=os.environ.get("ABOTN_SCENES", ""),
                    help="ABotN 场景白名单（逗号分隔，默认全部）")
    ap.add_argument("--abotn_episodes_per_scene", type=int,
                    default=int(os.environ.get("ABOTN_EPISODES_PER_SCENE", "0")),
                    help="ABotN 每个场景最多取几个官方回合（0 = 全要，共 628 个）")
    ap.add_argument("--sage_root", default=os.environ.get(
        "SAGE_ROOT", "/data/zerovln/SAGE-3D"))
    ap.add_argument("--sage_split", default=os.environ.get("SAGE_SPLIT", "train"),
                    help="GVLN split；可用 split 由 sage.datasets.available() 校验，"
                         "包括 train/instr/scene/traj/demo 及 capability construction/evaluation split")
    ap.add_argument("--sage_render_url",
                    default=os.environ.get("SAGE_RENDER_URL", "http://127.0.0.1:7038"))
    ap.add_argument("--sage_scenes", default=os.environ.get("SAGE_SCENES", ""),
                    help="SAGE 场景白名单（逗号分隔，默认整个 split）")
    ap.add_argument("--sage_episodes_per_scene", type=int,
                    default=int(os.environ.get("SAGE_EPISODES_PER_SCENE", "5")),
                    help="SAGE 每个场景最多取几个 GVLN 回合（0 = 全要；train 默认 5 截断 2.2M "
                         "回合的 eager 解析，frame_offset 一次性 ~8min 仍是主成本）")
    ap.add_argument("--mall_assets_dir", default=os.environ.get(
        "MALL_ASSETS_DIR",
        "/opt/external/user-9656ce61/grpo/locnav_grpo/exp0822/assets_arr5/mall"),
        help="mall_floor2 资产目录（walk.json / floor_meta.json / "
             "mall_walk_union_topo.npy，可选 walk_grpo.json 拓扑图供 SPL）")
    ap.add_argument("--mall_render_url",
                    default=os.environ.get("MALL_RENDER_URL", "http://127.0.0.1:7039"),
                    help="mall 的 3DGS 渲染服务地址（--sh_degree 0，scenes_root 指到 "
                         "zekang 的 ppu_sim/3drs_gs_root，独立进程 PPU3/7039）")
    ap.add_argument("--mall_episodes", type=int,
                    default=int(os.environ.get("MALL_EPISODES", "0")),
                    help="mall 起点回合数上限（0 = 全要，共 1225 个起点）")
    ap.add_argument("--lepu_assets_dir", default=os.environ.get(
        "LEPU_ASSETS_DIR",
        "/opt/external/user-9656ce61/grpo/locnav_grpo/exp0822/assets_arr5/lepu"))
    ap.add_argument("--lepu_render_url", default=os.environ.get(
        "LEPU_RENDER_URL", os.environ.get("MALL_RENDER_URL", "http://127.0.0.1:7039")),
        help="Lepu 与 mall 共用的 3DGS 渲染服务")
    ap.add_argument("--lepu_episodes", type=int, default=int(os.environ.get("LEPU_EPISODES", "0")),
                    help="Lepu 起点回合数上限（0 = 全部 7381 个起点）")
    ap.add_argument("--scene", default=os.environ.get("DEMO_SCENE", ""),
                    help="初始场景；留空时优先打开 ABotN 室内点导航")
    ap.add_argument("--episode_id", default=os.environ.get("DEMO_EPISODE_ID", "11"))
    ap.add_argument("--split", default=os.environ.get("DEMO_SPLIT", "val_unseen"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("DEMO_HTTP_PORT", "5000")))
    ap.add_argument("--max_steps", type=int, default=int(os.environ.get("DEMO_MAX_STEPS", "500")))
    ap.add_argument("--step_delay", type=float,
                    default=float(os.environ.get("DEMO_STEP_DELAY", "0.05")))
    ap.add_argument("--out", default=os.environ.get(
        "DEMO_OUTPUT_PATH",
        str(pathlib.Path(__file__).resolve().parents[1] / "outputs" / "model")))
    ap.add_argument("--fm_ckpt", default=os.environ.get("FM_CKPT", ""),
                    help="FM 驱动加载的带 FM 头的模型 checkpoint（留空用 "
                         "fm_policy.FM_CKPT 默认值）。仅 fm driver 用，进程内加载。")
    ap.add_argument("--fm_device", default=os.environ.get("FM_DEVICE", "cuda"),
                    help="FM 头推理设备（cuda / cuda:0 / cpu）。仅 fm driver 用。")
    ap.add_argument("--lightnav_url",
                    default=os.environ.get("LIGHTNAV_SERVER_URL", LND.DEFAULT_URL),
                    help="LightNav-0 独立推理服务 WebSocket 地址。仅 lightnav driver 用。")
    ap.add_argument("--lightnav_timeout", type=float,
                    default=float(os.environ.get("LIGHTNAV_SERVER_TIMEOUT", "30")),
                    help="LightNav 单次连接/推理超时秒数。")
    ap.add_argument("--lightnav_mujoco_url",
                    default=os.environ.get("LIGHTNAV_MUJOCO_URL", "http://127.0.0.1:18088"),
                    help="LightNav 自带 MuJoCo web server 地址。")
    ap.add_argument("--lightnav_mujoco_timeout", type=float,
                    default=float(os.environ.get("LIGHTNAV_MUJOCO_TIMEOUT", "5")))
    ap.add_argument("--lightnav_mujoco_linear_speed", type=float,
                    default=float(os.environ.get("LIGHTNAV_MUJOCO_LINEAR_SPEED", "2.5")),
                    help="远端 primitive 前进速度 m/s。")
    ap.add_argument("--lightnav_mujoco_angular_speed", type=float,
                    default=float(os.environ.get(
                        "LIGHTNAV_MUJOCO_ANGULAR_SPEED", str(math.radians(300.0)))),
                    help="远端 primitive 转向速度 rad/s。")
    args = ap.parse_args()
    CFG['driver'] = P.canonical_driver(CFG['driver'])
    set_state(driver=CFG['driver'])
    CFG["action_head_as_tool_version"] = args.action_head_as_tool_version
    set_state(action_head_as_tool_version=args.action_head_as_tool_version)
    split_label = args.split.replace("_", " ").title()
    BACKENDS["habitat"]["label"] = "R2R-CE / " + split_label
    BACKENDS["habitat"]["desc"] = "R2R_VLNCE_v1 / %s / English / MP3D / Habitat" % args.split
    import rxr as _rxr
    BACKENDS["rxr"]["label"] = "RxR-CE / %s / %s" % (split_label, _rxr.ROLES.title())
    BACKENDS["rxr"]["desc"] = "RxR_VLNCE_v0 / %s / %s / %s / MP3D / Habitat" % (
        args.split, _rxr.ROLES, _rxr.LANGUAGES)
    ACTION_HEAD_DATASET_SPLIT = args.sage_split
    if args.fm_ckpt:
        CFG["fm_ckpt"] = args.fm_ckpt
    if args.fm_device:
        CFG["fm_device"] = args.fm_device
    CFG["lightnav_url"] = args.lightnav_url

    if is_action_head_driver(CFG["driver"]):
        request_action_head_load()

    mount_backends(args)

    # Episodes are the union of every mounted set, but the page keeps the sets
    # apart: an MP3D id, an HSSD number and a 3DRS hash are indistinguishable in
    # one flat list, so the user picks the asset set first.
    scene_eps, per_set, episode_dimensions = {}, {}, {}
    episode_languages = {}
    for name, e in ENVS.items():
        for ep_ in e.episodes:
            sc = scene_of(ep_)
            episode_id = str(ep_.episode_id)
            scene_eps.setdefault(sc, []).append(episode_id)
            per_set.setdefault(name, set()).add(sc)
            language = getattr(getattr(ep_, "instruction", None), "language", "")
            if language:
                episode_languages.setdefault(sc, {})[episode_id] = language
            instruction_type = str(getattr(ep_, "info", {}).get(
                "instruction_type", ""))
            if (instruction_type.lower().startswith("capability/") or
                    instruction_type.lower().startswith("curriculum/")):
                dimension = instruction_type.split("/", 1)[1].strip().lower()
                if dimension:
                    episode_dimensions.setdefault(sc, {})[episode_id] = dimension
    for sc in scene_eps:
        scene_eps[sc].sort(key=lambda x: int(x) if x.isdigit() else x)
    # Opaque ids (3DRS hashes, HSSD numbers) get a readable title from whatever
    # the asset set itself says about the scene; MP3D has no caption, keeps its id.
    scene_titles = SCENE_TITLES
    for name, e in ENVS.items():
        if hasattr(e, "title_of"):
            for sc in per_set.get(name, ()):
                ttl = e.title_of(sc)
                if ttl:
                    scene_titles[sc] = ttl
    asset_sets = [(name, BACKENDS[name]["label"], BACKENDS[name]["desc"],
                   sorted(per_set[name], key=lambda s: (scene_titles.get(s, ""), s)))
                  for name in BACKENDS if per_set.get(name)]
    scenes = sorted(scene_eps)
    # observe needs a readable name that is also safe as a notes key. Compute
    # the duplicate groups once at startup, not on every status/observe call.
    SCENE_NAMES.clear()
    SCENE_NAMES.update(disambiguate_scene_names(scenes, scene_titles))

    # Keep asset-set ordering stable, but make the indoor ABotN set the landing
    # page when no scene was explicitly requested. If that optional backend did
    # not mount, retain the historical first-available fallback.
    indoor_scenes = next((items for name, _, _, items in asset_sets
                          if name == "in"), ())
    default_scene = indoor_scenes[0] if indoor_scenes else asset_sets[0][3][0]
    cur_scene = args.scene if args.scene in scene_eps else default_scene
    env = use_backend(backend_of(cur_scene))
    ep = (select_episode(env, cur_scene, args.episode_id)
          or select_episode(env, cur_scene, scene_eps[cur_scene][0]))
    if ep is None:
        raise SystemExit(f"no episode for scene={cur_scene}")
    ev = ACTIVE["ev"]
    # 启动默认模型可 env 覆盖（start_sage_fm_viz.sh 设 DEMO_DEFAULT_MODEL=vlnfm0729）；
    # 再强制模型→驱动锁（FM 模型锁 fm+ckpt）+ goalless 守一道。
    _dm = os.environ.get("DEMO_DEFAULT_MODEL")
    if _dm and _dm in P.MODELS:
        CFG["model"] = _dm
        set_state(model=_dm)
    _enforce_model_driver_lock(announce=True)
    guard_driver()

    # Confirm the action-space keys match the task action indices we hand to
    # env.step() (they do -- 1/2/3 -- but a silent mismatch would send the robot
    # the wrong way). BASE_AMOUNTS was snapshotted by use_backend().
    for name, e in ENVS.items():
        ids = {n: A.action_id(e, n)
               for n in ("move_forward", "turn_left", "turn_right")}
        want = {"move_forward": FWD, "turn_left": LEFT, "turn_right": RIGHT}
        if ids != want:
            print(f"!! {name} action id mismatch: sim={ids} task={want} !!", flush=True)
        print(f"[sim] {BACKENDS[name]['label']}: {len(e.episodes)} 回合  "
              f"步长 {A.default_amounts(e)}  action_id {ids}", flush=True)

    # The dropdown owns the model endpoint from here on. Report both servers at
    # startup so a missing one is obvious before the first run rather than at
    # Start time.
    if not (P.is_fm_model(CFG["model"])
            or P.is_lightnav_model(CFG["model"])):
        apply_model(ev)
    for k, (lbl, url, _, _) in P.MODELS.items():
        if P.is_fm_model(k):
            state = "进程内 FM（不走 vLLM）" + ("  ← 默认" if k == CFG['model'] else "")
            print(f"[model] {lbl:18s} {state}", flush=True)
            continue
        if P.is_lightnav_model(k):
            state = (f"独立策略 {args.lightnav_url}"
                     + ("  ← 默认" if k == CFG['model'] else ""))
            print(f"[model] {lbl:18s} {state}", flush=True)
            continue
        print(f"[model] {lbl:14s} {url}  "
              f"{'在线' if model_alive(k) else '离线（未起服务）'}"
              f"{'  ← 默认' if k == CFG['model'] else ''}", flush=True)

    # Announce a restored custom prompt at startup: a run that behaves oddly
    # because of a prompt edited days ago is otherwise very hard to explain.
    if load_sysprompt():
        print(f"[prompt] 使用自定义 system prompt（{len(CFG['sysprompt'])} 字，"
              f"{SYSPROMPT_FILE}）"
              + ("" if P.prompt_has_tools(CFG["sysprompt"])
                 else "  ⚠ 里面没有 <tools>，模型可能不会调用工具"), flush=True)

    srv = ThreadingHTTPServer(("127.0.0.1" if os.environ.get("VLN_BATCH_TOKEN") else "0.0.0.0", args.port), Handler)
    srv.daemon_threads = True
    srv.web_config = webui.config(scenes, cur_scene, scene_eps,
                                  P.template_list(), CFG["tpl"],
                                  P.mode_list(), CFG["mode"],
                                  P.driver_list(), CFG["driver"],
                                  P.model_list(), CFG["model"],
                                  CFG["maxtools"],
                                  P.TOOL_BUDGET_MIN, P.TOOL_BUDGET_MAX,
                                  asset_sets=asset_sets,
                                  scene_titles=scene_titles,
                                  hold_speed=HOLD_SPEED_MPS,
                                  hold_turn=HOLD_TURN_DPS,
                                  nact=CFG["nact"],
                                  episode_dimensions=episode_dimensions,
                                  episode_languages=episode_languages,
                                  cur_dimension=episode_dimensions.get(
                                      cur_scene, {}).get(str(ep.episode_id), ""))
    prewarm_action_head_runtimes()
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    obs = load_episode(env, ep, args)
    trail = []
    hold = new_hold_state()

    print("=" * 70, flush=True)
    print(f"VLN 控制台  scene={cur_scene} ep={ep.episode_id}  模板={CFG['tpl']}")
    ips = local_ips()
    host = ips[0] if ips else "127.0.0.1"
    print(f"打开 http://{host}:{args.port}"
          + (f"   (其它可达 IP: {', '.join(ips[1:])})" if len(ips) > 1 else ""))
    print("本机:   http://127.0.0.1:%d   (ssh -L %d:127.0.0.1:%d <host>)"
          % (args.port, args.port, args.port))
    print("=" * 70, flush=True)

    try:
        while True:
            batch_tick(env)
            if CMD["clearlog"]:
                CMD["clearlog"] = False
                clear_events()

            if CMD["switch"] is not None:
                scene, ep_id = CMD["switch"]
                CMD["switch"] = None
                CMD["stop"] = False
                set_state(status="loading", result="")
                push_event("sys", text=f"正在加载 {scene} 回合 {ep_id} …")
                t0 = time.time()
                # The scene decides the simulator: the two asset sets are mounted
                # side by side and their ids look alike, so route on the map
                # rather than on whatever was loaded last.
                target = backend_of(scene)
                nenv = ENVS.get(target, env)
                nep = select_episode(nenv, scene, ep_id)
                if nep is None and getattr(nenv, "episode_for", None):
                    # Pre-loaded list had no such case (SAGE_EPS truncation):
                    # materialise it on demand from the full task parse.
                    nep = nenv.episode_for(scene, ep_id)
                if nep is None:
                    push_event("err", text=f"{scene} 中没有回合 {ep_id}，保持不变")
                    set_state(status="idle")
                else:
                    try:
                        switched = target != ACTIVE["backend"]
                        env = use_backend(target)
                        guard_driver(announce=True)
                        obs = load_episode(env, nep, args)
                        ep, cur_scene = nep, scene
                        trail = []
                        hold = new_hold_state()
                        if switched:
                            push_event("sys", text="已切到资产集："
                                       + BACKENDS[target]["label"])
                        txt = (f"已加载 {cur_scene} 回合 {ep.episode_id}"
                               f"（{time.time() - t0:.1f}s）")
                        # No dangling colon for a captionless 3DRS scene, and no
                        # second copy of the caption when it is already the
                        # instruction -- load_episode logs it as 场景：… .
                        itext = ep.instruction.instruction_text
                        if itext and itext != (ep.info or {}).get(
                                "scene_description", ""):
                            txt += "：" + itext
                        push_event("sys", text=txt)
                    except Exception as exc:
                        push_event("err", text=f"加载失败：{exc}")
                        set_state(status="error")
                continue

            if CMD["reset"]:
                CMD["reset"] = False
                CMD["stop"] = False
                obs = load_episode(env, ep, args)
                trail = []
                hold = new_hold_state()
                push_event("sys", text="已回到回合起点")
                continue

            if CMD["goto"] is not None:
                fx, fy = CMD["goto"]
                CMD["goto"] = None
                obs = do_goto(env, obs, fx, fy)
                continue

            if CMD["manual_start"]:
                CMD["manual_start"] = False
                trail = []
                hold = new_hold_state()
                set_trail(actions=trail, moves=[],
                          motion={"dist": 0.0, "bearing": 0.0,
                                  "yaw": 0.0, "steps": 0})
                set_state(status="manual", manual=True, result="")
                push_event("sys", text="手动模式：按住 W/A/S/D 驾驶（按多久走多远），"
                                       "点「开始」交给模型接管")
                continue

            # Hold-to-drive. Checked before CMD["manual"] and before start so a
            # key that is down keeps the robot moving at the simulator's own
            # rate: the branch continues without touching the tail sleep, so the
            # loop turns exactly as fast as env.step returns.
            keys = held_keys()
            if CMD["start"] or CMD["manual_start"]:
                keys = frozenset()      # a pending run wins over a stuck key
            if keys and env.episode_over:
                if not hold["on"]:
                    hold["on"] = True
                    push_event("sys", text="回合已结束，请点「回到起点」或切换回合")
            elif keys:
                if not hold["on"]:
                    # Once per press, not once per tick: it walks the whole
                    # action space and would be pure overhead at 12 Hz.
                    check_actuation(env)
                    hold.update(on=True, t=time.time(), deg=0.0, m=0.0,
                                blocked=False, id=hold["id"] + 1)
                    set_state(status="manual", manual=True, result="")
                obs = drive_hold(env, keys, hold, obs)
                continue
            elif hold["on"]:
                # Released. Redraw once with the map forced so the bird's-eye
                # view matches where you actually stopped, rather than lagging
                # by up to MAP_MIN_DT.
                hold["on"] = False
                refresh(env, obs, STATE["motion"]["steps"], force_map=True)
                continue

            if CMD["manual"] is not None:
                act = CMD["manual"]
                CMD["manual"] = None
                if env.episode_over:
                    push_event("sys", text="回合已结束，请点「回到起点」或切换回合")
                elif act == STOP:
                    set_state(status="idle")
                    push_event("sys", text="手动停止")
                else:
                    check_actuation(env)   # discrete step, fixed benchmark size
                    if act == "back":
                        # One SUBSTEP_M step backwards. drive_backward owns the
                        # negative amount and its finally-restore, so the shared
                        # action_space is left exactly as it was found.
                        obs2, info = A.drive_backward(env, A.SUBSTEP_M)
                        obs = obs2 or obs
                        # The classic int trail has no code for "backward"
                        # (0.25 m / 30 deg per entry is the whole vocabulary),
                        # so record it in the continuous trail instead of
                        # mislabelling it as a forward step.
                        moves = list(STATE["moves"]) + [
                            {"kind": "back", "m": info["moved_m"],
                             "req": info["requested_m"],
                             "blocked": bool(info["blocked"])}]
                        set_trail(moves=moves)
                        if info["blocked"]:
                            push_event("sys", text="后方受阻，没退动")
                    else:
                        obs = env.step(act)
                        trail.append(act)
                        set_trail(actions=trail)
                    set_state(status="manual", manual=True)
                    refresh(env, obs, len(trail), force_map=True)
                continue

            if CMD["walk_golden"]:
                CMD["walk_golden"] = False
                CMD["stop"] = False
                try:
                    obs, trail = run_golden(env, ev, "", args, obs, trail)
                except Exception as exc:
                    # One optional debug action must never take the HTTP server
                    # and pi-agent terminal down with the simulator worker.
                    push_event("err", text=f"走专家轨迹失败：{exc}")
                    set_state(status="error")
                    print(f"[golden] failed: {exc!r}", flush=True)
                continue

            if CMD["point_move"] is not None:
                request = CMD["point_move"]
                CMD["point_move"] = None
                CMD["stop"] = False
                current_source = (STATE["scene"], STATE["episode_id"],
                                  STATE["seq"])
                if tuple(request["source"]) != current_source:
                    message = "point_move 开始前画面已变化，未执行移动"
                    set_state(status="error", result=message)
                    push_event("err", text=message)
                    clear_react_reference(only_kind="point")
                    continue
                trail = []
                hold = new_hold_state()
                set_state(manual=False)
                obs, trail = run_direct_point(
                    env, request["instruction"], request["point"],
                    args, obs, trail)
                continue

            if CMD["start"]:
                with HEAD_CONTROL_LOCK:
                    CMD["start"] = False
                    CMD["stop"] = False
                    set_state(status='running')
                # Empty box -> the episode's own instruction. That is the
                # dataset text this episode is scored against, so leaving the
                # box blank is the way to reproduce the benchmark condition.
                instr = (CFG.get("instruction") or "").strip() or \
                    ep.instruction.instruction_text
                if not instr:
                    # A 3DRS scene with no metainfo caption: nothing to prefill,
                    # and we do not invent a task, so there is nothing to run
                    # until you type it.
                    push_event("err", text="这个场景没有自带描述，"
                                           "请在指令框里写一条再开始（或者用手动驾驶）")
                    set_state(status="idle")
                    continue
                # FM loads in-process and LightNav talks to its own WebSocket;
                # neither uses the selected chat-model endpoint.
                driver = _enforce_model_driver_lock(announce=True)  # 模型→驱动锁（FM 模型锁 fm）
                driver = guard_driver(announce=True)                # goalless 资产再守一道
                if driver == "fm":
                    if _load_fm() is None:
                        set_state(status="error")
                        continue
                    label = "FM·流匹配"
                elif driver == "lightnav":
                    label = "LightNav-0"
                else:
                    if is_action_head_driver(driver) and ACTION_HEAD_AGENT_RUNTIME is None:
                        # The /cmd start handler has already requested a lazy
                        # load before putting this run on the worker queue.
                        # Do not start a second loader here: besides wasting
                        # resources, an earlier failure used to be mislabeled
                        # as "still loading".  Keep this fallback only for
                        # non-HTTP callers that can set CMD directly.
                        with ACTION_HEAD_LOAD_LOCK:
                            load_state = ACTION_HEAD_LOAD_STATE
                            load_error = ACTION_HEAD_LOAD_ERROR
                        if load_state == "unloaded":
                            request_action_head_load()
                            push_event("sys", text="FM Head 已开始加载；加载完成后再点击开始")
                        elif load_state == "loading":
                            push_event("sys", text="FM Head 正在加载；加载完成后再点击开始")
                        else:
                            push_event("err", text=(
                                "FM Head 尚未加载，已取消本次开始："
                                + (load_error or "加载状态异常")))
                        set_state(status="idle")
                        continue
                    try:
                        label = apply_model(ev)
                    except Exception as exc:
                        push_event("err", text=f"模型端点配置失败：{exc}")
                        set_state(status="error")
                        continue
                    if not model_alive(CFG["model"]):
                        # Two causes now: nothing listening, or something listening
                        # that serves a different model. Both mean "don't run" --
                        # the console log line says which.
                        push_event("err", text=f"大脑服务 {label} 尚未就绪或连接已中断"
                                               f"（{ev.vllm_server_url}，期望模型名 "
                                               f"{ev.args.vllm_model_name}），"
                                               f"请等待大脑服务就绪后重试。")
                        set_state(status="error")
                        continue
                if not CFG["frompose"]:
                    obs = load_episode(env, ep, args)
                # step count and frame history always restart at 0
                trail = []
                hold = new_hold_state()
                set_state(manual=False)
                set_trail(actions=trail, moves=[],
                          motion={"dist": 0.0, "bearing": 0.0,
                                  "yaw": 0.0, "steps": 0})
                print(f"[run] {cur_scene} ep{ep.episode_id} model={label} "
                      f"driver={driver} tpl={CFG['tpl']} "
                      f"nact={CFG['nact']} maxtools={CFG['maxtools']} "
                      f"sys={'自定义' if CFG['sysprompt'] else '默认'}"
                      f": {instr}", flush=True)
                drive = {"react": run_agent, "action_head": run_agent,
                         "action_head_untrained": run_agent,
                         "embodied_point": run_agent,
                         "unilavira_bbox": run_agent,
                         "lightnav": run_lightnav,
                         "action_head_2800": run_agent,
                         "action_head_10ep": run_agent,
                         "action_head_v43_7000": run_agent,
                         "action_head_v5_native": run_agent,
                         "action_head_v5_cached": run_agent,
                         "action_head_planner": run_agent,
                         "fm": run_fm}.get(driver, run_model)
                obs, trail = drive(env, ev, instr, args, obs, trail)
                continue

            if CMD["stop"]:
                CMD["stop"] = False
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        SHUTDOWN.set()
        TERMINAL.close()
        with FRAME_CV:
            FRAME_CV.notify_all()   # release the /stream.mjpg threads
        srv.shutdown()


if __name__ == "__main__":
    main()
