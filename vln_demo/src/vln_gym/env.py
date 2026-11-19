# -*- coding: utf-8 -*-
"""VLNEnv — gym 风格的 VLN-CE 闭环 RL 环境（duck-typed，不依赖 gymnasium）。

gordon venv 没装 gymnasium，所以本类不继承 gym.Env，但严格遵循 gym core API
（reset→(obs,info); step→(obs,reward,terminated,truncated,info); render; close;
action_space/observation_space），verl rollout worker 可以把它当本地 env 用，
或经 server.py 暴露成 HTTP 后由 client.VLNRemoteClient 代理。

设计参考：
- sim 构造照抄 vln_model.py:1032-1110 的 eval_args Namespace + VLNEvaluator +
  config_env()，但 model=None/processor=None（bench 纯 sim，不含 vLLM）。
- 连续动作复用 agent_tools.drive_forward/drive_turn/drive_backward（子步、
  碰撞、stall、finally-restore 全继承，不重写）。
- FM (dx,dy,dyaw) 轨迹复用 fm_action.execute_waypoint（turn→fwd→turn 分解）。

动作分层（用户拍板"分层支持不同形式的模型输出"）：
    step(int)              -> discrete: 0/1/2/3 (stop/forward/turn_left/turn_right)
    step({"name","args"})  -> tool:     move_forward/turn/move_backward/stop
    step({"mode":"fm_trajectory","waypoints":[[dx,dy,dyaw]*32],"normalized":bool})
                           -> 整个 chunk 一次性执行
"""
import os
import sys
import io
import json
import math
import base64
import argparse
import time

import numpy as np
from PIL import Image

# 延迟导入：rynnvln_eval / habitat_sim / agent_tools 在 __init__ 里才 import，
# 这样 client/server 导入本模块时不触发 habitat 加载（libOpenGL 等只在 sim
# 进程里需要，由 launcher 的 LD_LIBRARY_PATH 解决）。

# 动作索引（与 prompts.py:21 一致：FWD=1 LEFT=2 RIGHT=3 STOP=0）
STOP, FWD, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = {0: "stop", 1: "move_forward", 2: "turn_left", 3: "turn_right"}
SUCCESS_RADIUS = 3.0  # R2R success_distance，与 vln_model.py:55 一致


def _jpg_b64(rgb, quality=85):
    img = Image.fromarray(rgb).convert("RGB") if not isinstance(rgb, Image.Image) \
        else rgb.convert("RGB")
    b = io.BytesIO()
    img.save(b, "JPEG", quality=quality)
    return base64.b64encode(b.getvalue()).decode("ascii")


def _success_of(metrics):
    """success 读数；rs3d/hsim 是 free-roam，success=NaN（无 GT goal），视作未成功。
    abotn/rxr/r2r 是 0/1 实数，照常 bool。"""
    s = metrics.get("success", False)
    if isinstance(s, float) and s != s:  # NaN
        return False
    return bool(s)


# ------------------------------------------------------------------ backends
# r2r = R2R/MP3D (habitat via rynnvln_eval)；rxr = R2R 仿真器换 RxR 回合表；
# rs3d/hsim/abotn = zhouyang 周末新增的 habitat.Env-shaped 后端（见 rs3d/ hsim/
# abotn/）。zy 后端刻意做成 habitat 形：有 .episodes/.current_episode/.reset()→dict
# /.step(int)→dict /episode_over/@property/.get_metrics()/.close()，且 sim.agents[0].
# agent_config.action_space 是 Dict[int,ActionSpec(name,actuation.amount)]——所以
# agent_tools.action_id/actuation/default_amounts/drive_* 对它们零改可用，obs 只缺
# gps/compass（由 _zy_pose 从 sim pose 合成）。
ZY_BACKENDS = ("rs3d", "hsim", "abotn", "sage", "mall", "lepu")
ALL_BACKENDS = ("r2r", "rxr") + ZY_BACKENDS


def _build_zy_env(backend, scene, render_url=None):
    """构造一个 zhouyang 新后端 env（habitat.Env-shaped），不碰 rynnvln_eval/habitat。

    每个后端的资产集/渲染服务由 env var 选（数据根已在 rs3d/hsim/abotn/sage 里改指本机）：
      rs3d  : RS3D_RENDER_URL(默认 7036)，场景取自 RS3D_ROOT
      hsim  : HSIM_SET=partnr|hab（habitat_sim 直接，无渲染服务）
      abotn : ABOTN_RENDER_URL(默认 7037)、ABOTN_SET=poi|in|out
      sage  : SAGE_RENDER_URL(默认 7038)、SAGE_SPLIT=train|instr|scene|traj
      mall  : MALL_RENDER_URL(默认 7039)、MALL_ASSETS_DIR（zekang mall_floor2 资产目录）
      lepu  : LEPU_RENDER_URL(默认跟随 mall)、LEPU_ASSETS_DIR（Lepu 办公层资产）
    """
    if backend == "rs3d":
        from rs3d.env import RS3DEnv, list_scenes, RS3D_ROOT
        url = render_url or os.environ.get(
            "RS3D_RENDER_URL", "http://127.0.0.1:7036")
        all_sc = list_scenes(RS3D_ROOT)
        # scene 可能是 r2r 默认(MP3D id)，不在 3DRS 场景表里——回退到首个场景。
        sc = scene if scene in all_sc else (all_sc[0] if all_sc else scene)
        return RS3DEnv([sc], render_url=url)
    if backend == "hsim":
        from hsim.env import HabSceneEnv
        from hsim.datasets import available as hsim_available
        sets = hsim_available()
        if not sets:
            raise RuntimeError("hsim: 无可用资产集（PARTNR/Habitat-HAB 都不在 "
                               "AGENTIC_SIM_ROOT/benchmark 下）")
        key = os.environ.get("HSIM_SET", "partnr").lower()
        ds = sets.get("partnr" if key.startswith("part") else "hab") \
            or next(iter(sets.values()))
        all_sc = ds.list_scenes() if hasattr(ds, "list_scenes") else None
        # 只在能确认 scene 在数据集场景表里时才过滤；否则 None=全场景
        # （r2r 默认场景名是 MP3D id，不在 PARTNR/HSSD 表里，不能当过滤器）。
        scs = [scene] if (all_sc and scene in all_sc) else None
        return HabSceneEnv(ds, scenes=scs)
    if backend == "abotn":
        from abotn.env import ABotNEnv
        from abotn.datasets import available as abotn_available
        sets = abotn_available()
        if not sets:
            raise RuntimeError("abotn: 无可用 benchmark（POIBench/PointBench 都不在 "
                               "ABOTN_ROOT 下）")
        key = os.environ.get("ABOTN_SET", "poi").lower()
        ds = sets.get(key) or next(iter(sets.values()))
        all_sc = ds.list_scenes()
        scs = [scene] if scene in all_sc else None
        return ABotNEnv(ds, scenes=scs)
    if backend == "sage":
        # sage = SAGE-3D GVLN，abotn 模板（独立 degree-3 render server 7038 +
        # occupancy 栅格运动学，不引入 Isaac）。split 由 SAGE_SPLIT 选，render
        # url 由 SAGE_RENDER_URL 选——bench 侧与 console build_sage 同口径。
        # episodes_per_scene 由 SAGE_EPS_PER_SCENE 选（>0 限量解析，cap 大 split
        # 的启动开销：train 929 场景 × ~2385 task ≈ 2.2M，全 eager ~6min+2GB/bench；
        # 限 10 → ~5s+10MB）。小 split（instr/scene/traj）留 0=全量。
        from sage.env import SageEnv
        from sage.datasets import all_sets as sage_all_sets, available as sage_available
        key = os.environ.get("SAGE_SPLIT", "instr").lower()
        ds = sage_all_sets().get(key)
        # A configured valid split does not require scanning every unrelated
        # dataset. Keep the existing fallback for missing/unavailable names.
        if ds is None or not ds.available():
            sets = sage_available()
            if not sets:
                raise RuntimeError("sage: 无可用 split（train/instr/scene/traj 都不在 "
                                   "SAGE_ROOT 下；先转场景 ply："
                                   "make_scenes_root --split <s> --source nurec）")
            ds = sets.get(key) or next(iter(sets.values()))
        all_sc = ds.list_scenes()
        scs = [scene] if scene in all_sc else None
        url = render_url or os.environ.get(
            "SAGE_RENDER_URL", "http://127.0.0.1:7038")
        eps_per_scene = int(os.environ.get("SAGE_EPS_PER_SCENE", "0"))
        # Dedicated rollout/build clients know exact case IDs. Materialize one
        # initial episode and reuse SageEnv.episode_for for subsequent resets.
        # Keep the full catalog for consoles and explicit episode allow-lists.
        lazy = (os.environ.get("SAGE_LAZY_EPISODES", "0") == "1"
                and not os.environ.get("SAGE_EPISODE_KEEP", "").strip())
        if os.environ.get("SAGE_BENCH_PROTOCOL_ROOT") and not lazy:
            # The companion's first Gym exposes the complete /spec catalog;
            # its selected startup scene must not restrict the benchmark pool.
            scs = all_sc
        from vln_gym.observation_profile import requested_profile
        profile = requested_profile()
        camera_args = ({'width': profile['width'], 'height': profile['height'],
                        'hfov': profile['hfov_deg']} if profile else {})
        env = SageEnv(ds, scenes=(scs or all_sc[:1]) if lazy else scs,
                      render_url=url, episodes_per_scene=1 if lazy else eps_per_scene,
                      **camera_args)
        if profile:
            env.sim.z_offset = profile['height_m']
            env.sim.camera_pitch = math.radians(profile['pitch_deg'])
        env._lazy_episode_lookup = lazy
        return env
    if backend == "mall":
        # mall = mall_floor2 万科里商场（local-user-06 自采 GS）：1225 真实起点
        # × 商铺目标，膨胀 walk mask 严格运动学（0.25m/15°/撞墙整体拒绝），
        # 到店 2m 算到达，指标可评分。资产目录由 MALL_ASSETS_DIR 选，渲染服务
        # 由 MALL_RENDER_URL 选（与 3DRS 同 /render_gs 协议）。
        from mall.assets import MallAssets
        from mall.env import MallEnv
        ds = MallAssets()
        if not ds.available():
            raise RuntimeError(
                f"mall: 资产目录缺文件（{ds.root}），应有 walk.json / "
                "floor_meta.json / mall_walk_union_topo.npy")
        ds.load()
        url = render_url or os.environ.get(
            "MALL_RENDER_URL", "http://127.0.0.1:7039")
        return MallEnv(ds, render_url=url)
    if backend == "lepu":
        from lepu.assets import LepuAssets
        from lepu.env import LepuEnv
        ds = LepuAssets()
        if not ds.available():
            raise RuntimeError(f"Lepu 资产缺失：{ds.root}")
        return LepuEnv(ds, render_url=render_url)
    raise ValueError(f"unknown zy backend {backend!r}")


class VLNEnv:
    """单 session 的 VLN-CE gym 环境。线程不安全（habitat env 非线程安全），
    server.py 用全局锁串行化。"""

    metadata = {"render_modes": ["rgb", "map"]}

    # ---- action / observation schema（JSON 可序列化，HTTP 直接返回）----
    action_space = {
        "type": "layered",
        "modes": {
            "discrete": {"type": "int", "values": [0, 1, 2, 3],
                         "meaning": ["stop", "move_forward", "turn_left", "turn_right"]},
            "tool": {"type": "object",
                     "fields": {"name": "move_forward|turn|move_backward|stop",
                                "arguments": "dict"},
                     "tools": [
                         {"name": "move_forward",
                          "arguments": {"distance": "float m, 0.05~1.0"}},
                         {"name": "move_backward",
                          "arguments": {"distance": "float m, 0.05~1.0"}},
                         {"name": "turn",
                          "arguments": {"direction": "left|right",
                                        "degrees": "float 0.5~90"}},
                         {"name": "stop", "arguments": {"reason": "str"}},
                     ]},
            "fm_trajectory": {"type": "object",
                              "fields": {"waypoints": "[[dx,dy,dyaw]*32]",
                                         "normalized": "bool, default true"}},
        },
        "note": "step(int)=discrete; step({name,arguments})=tool; "
                "step({mode:'fm_trajectory',waypoints,normalized})=fm",
    }
    observation_space = {
        "type": "dict",
        "fields": {
            "rgb_b64": "str (base64 jpg, 640x480)",
            "depth_b64": "str (base64, optional)",
            "gps": "[x, y]",
            "yaw_deg": "float",
            "step": "int",
            "instruction": "str",
            "scene": "str",
            "episode_id": "str",
        },
    }

    def __init__(self, scene="2azQ1b91cZZ", episode_id="11",
                 split="val_unseen", max_steps=500, instruction=None,
                 render_mode="rgb", nav_dir=None, config=None,
                 output_path=None, backend="r2r", defer_initial_reset=False):
        self.backend = backend
        self.scene = scene
        self.episode_id = str(episode_id)
        self.split = split
        self.max_steps = int(max_steps)
        self._instruction_override = instruction
        self.render_mode = render_mode

        nav_dir = nav_dir or os.environ.get(
            "RYNNBRAIN_NAV_DIR",
            "/opt/navigation-runtime/"
            "VLMEvalKit/3rdparty/RynnBrain/navigation")
        # vln_demo/src 在 sys.path 上，agent_tools / prompts 才能 import
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for p in (os.path.join(nav_dir, "src"), src_dir):
            if p not in sys.path:
                sys.path.insert(0, p)

        # agent_tools 纯 python（仅 json/math/re/dataclasses），对 habitat 和 zy
        # 后端都零改可用——zy env 刻意做成 habitat.Env 形（见 _build_zy_env 注释）。
        import agent_tools as A
        self.A = A

        if backend in ZY_BACKENDS:
            # rs3d/hsim/abotn：直接构造，不碰 rynnvln_eval/habitat-maps/VLNEvaluator。
            self.base = None
            self.base_maps = None
            self.ev = None
            self.env = _build_zy_env(backend, scene)
        else:
            # r2r / rxr：原 habitat 路径。rxr 复用 R2R 仿真器，只换回合表。
            # bench 不含 vLLM，但 VLNEvaluator 的 args 仍带 model_path 字段；
            # 给个占位避免任何 KeyError（model=None 时不会真去加载）。
            # Read the default from this container's pinned Git checkout.
            # Rebuilding a backend must not reread a mutable YAML copy in the
            # Gym working directory (a failed runtime copy caused reset HTTP500).
            if config is None:
                config = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      os.pardir, "vln_demo.yaml")
                config = os.path.abspath(config)
            os.environ.setdefault("MODEL_PATH", "/dummy/vln_gym_no_vllm")
            import rynnvln_eval as base
            from habitat.utils.visualizations import maps as base_maps
            self.base = base
            self.base_maps = base_maps
            eval_args = argparse.Namespace(
                local_rank=0, model_path=os.environ.get("MODEL_PATH", ""),
                habitat_config_path=config, eval_split=split,
                output_path=output_path or "/tmp/vln_gym_out",
                num_future_steps=4, num_frames=32, save_video=False, num_history=8,
                max_eval_episodes_per_rank=0, max_new_tokens=2048,
                prompt_style="embodied_r1", num_actions=4, max_history_frames=16,
                open_loop_steps=0, inference_backend="vllm_server",
                vllm_server_urls="http://127.0.0.1:1",  # 占位过校验；bench 不调 generate_response，永不命中
                vllm_model_name="",
                vllm_timeout=300, vllm_retries=3, vllm_temperature=0, vllm_top_p=None,
                vllm_api_key="", vllm_image_format="JPG", vllm_image_quality=90,
                enable_thinking=False, model_max_length=4096,
                world_size=1, rank=0, gpu=0, port="1111", dist_url="env://",
                device="cpu", distributed=False,
            )
            os.makedirs(eval_args.output_path, exist_ok=True)
            # Gym sessions are local CPU simulators, not distributed model
            # workers. RynnBrain's single-process initializer still replaces
            # builtins.print; calling it for every session nests wrappers and
            # doubles the underlying print calls at each environment creation.
            self.ev = base.VLNEvaluator(config_path=config, split=split, env_num=1,
                                        output_path=eval_args.output_path,
                                        model=None, processor=None, epoch=0,
                                        args=eval_args)
            from vln_gym.native_episode_cache import enabled, cached_habitat_env
            self._native_episode_cache_enabled = enabled()
            self.env = (cached_habitat_env(self.ev) if self._native_episode_cache_enabled
                        else self.ev.config_env())
            if backend == "rxr":
                # SharedSimEnv 把 RxR 回合表 bolt 到 R2R habitat env 上，
                # reset/step/get_metrics 全走 habitat，下游契约不变。
                # load_episodes 返回 (episodes, missing)，只要 episodes。
                import rxr
                _ep, _missing = rxr.load_episodes(split=split)
                if not _ep:
                    raise RuntimeError(
                        f"rxr: 无可用回合（split={split}，missing={list(_missing)}）")
                self.env = rxr.SharedSimEnv(self.env, _ep)

        # 通用：动作 amount + action_id 校验（zy env 同形，照用）。
        self.BASE_AMOUNTS = A.default_amounts(self.env)
        ids = {n: A.action_id(self.env, n)
               for n in ("move_forward", "turn_left", "turn_right")}
        want = {"move_forward": FWD, "turn_left": LEFT, "turn_right": RIGHT}
        if ids != want:
            print(f"!! action id mismatch: sim={ids} task={want} !!", flush=True)

        self._episode_obj = None
        self.step_id = 0
        self._last_obs = None
        # Dedicated SAGE collection services are reset by the original case
        # request. A bad bootstrap pose must not kill the entire actor pool.
        # Consoles and ordinary clients retain their eager reset.
        if not isinstance(defer_initial_reset, bool) or (defer_initial_reset and backend != 'sage'):
            raise ValueError('deferred initial reset is only supported for SAGE services')
        self._initial_reset_deferred = defer_initial_reset
        if not defer_initial_reset:
            self.reset(scene=scene, episode_id=episode_id, instruction=instruction)

    # ---------------------------------------------------------------- reset
    def _select_episode(self, scene, ep_id):
        if self.backend == "sage" and (self._episode_obj is not None
                or getattr(self.env, '_lazy_episode_lookup', False)
                or os.environ.get('SAGE_NAVMESH_ROOT')):
            # Collection knows the exact ID, including the first lazy reset.
            # Do not substitute the first materialized case at initialization.
            scene = self.env.ds.qualify(self.env.ds.bare(scene))
            if scene not in self.env.ds.list_scenes():
                return None
            return self.env.episode_for(scene, str(ep_id))
        # 场景名匹配带前缀容忍：rxr 的 scene 是 "rxr-<mp3d>"，abotn 是 "poi-<id>"，
        # 调用方传的可能是裸 id（r2r 默认 2azQ1b91cZZ）。
        def _esc(e):
            return getattr(e, "scene", None) or e.scene_id.split("/")[-2]

        def _scene_match(esc, sc):
            return esc == sc or esc.endswith("-" + sc)

        from vln_gym.native_episode_cache import FrozenEpisodes
        if isinstance(self.env.episodes, FrozenEpisodes):
            return self.env.episodes.find(scene, ep_id, fallback=self.backend == 'rxr')

        # 1. 精确 (scene, ep_id)
        for e in self.env.episodes:
            if _scene_match(_esc(e), scene) and str(e.episode_id) == str(ep_id):
                return e
        # 2. 同 scene 的任意回合（zy/rxr 的 ep_id 命名不统一，smoke 友好）
        if self.backend in ZY_BACKENDS or self.backend == "rxr":
            for e in self.env.episodes:
                if _scene_match(_esc(e), scene):
                    return e
            return self.env.episodes[0] if self.env.episodes else None
        return None

    def reset(self, *, scene=None, episode_id=None, instruction=None,
              seed=None, options=None):
        """重置到指定 episode（或当前 episode）。返回 (obs, info)。"""
        scene = scene or self.scene
        episode_id = str(episode_id) if episode_id is not None else self.episode_id
        ep = self._select_episode(scene, episode_id)
        if ep is None:
            avail = [(str(e.episode_id),
                      getattr(e, "scene", None) or e.scene_id.split("/")[-2])
                     for e in self.env.episodes[:5]]
            raise ValueError(f"no episode scene={scene} id={episode_id}; "
                             f"sample={avail}")
        self.scene, self.episode_id = scene, episode_id
        self._episode_obj = ep
        self.env.current_episode = ep
        try:
            obs = self.env.reset()
        except (IndexError, RuntimeError) as e:
            # habitat reconfigure 偶发 self.agents 空列表（长寿命 env 累积触发，
            # 文档「重启即恢复」）。在进程内重建 env 再试一次，避免外置 policy
            # 每次都要重启 bench server。zy 后端无 config_env，直接重抛由调用方处理。
            if self.backend in ZY_BACKENDS or self.ev is None:
                raise
            print(f"[vln_gym] reset {type(e).__name__}: {e}; recreating env",
                  flush=True)
            try:
                self.env.close()
            except Exception:
                pass
            if getattr(self, '_native_episode_cache_enabled', False):
                from vln_gym.native_episode_cache import cached_habitat_env
                self.env = cached_habitat_env(self.ev)
                if self.backend == 'rxr':
                    import rxr
                    episodes, _ = rxr.load_episodes(split=self.split)
                    self.env = rxr.SharedSimEnv(self.env, episodes)
            else:
                self.env = self.ev.config_env()
            self.env.current_episode = ep
            self.BASE_AMOUNTS = self.A.default_amounts(self.env)
            obs = self.env.reset()
        self.step_id = 0
        self._last_obs = obs
        self._initial_reset_deferred = False
        if instruction is not None:
            self._instruction_override = instruction
        self._check_actuation()
        info = self._metrics(obs)
        info["reset"] = True
        actual_ep = self.env.current_episode
        info["actual_episode_id"] = str(actual_ep.episode_id)
        info["actual_scene"] = (getattr(actual_ep, "scene", None)
                                or actual_ep.scene_id.split("/")[-2])
        return self._observe(obs), info

    # ----------------------------------------------------------------- step
    def step(self, action):
        """分层动作分发。返回 (obs, reward, terminated, truncated, info)。"""
        self._check_actuation()
        mode, payload = self._classify(action)
        collided = False
        blocked = False
        tool_result = None
        waypoint_log = None
        terminated_mid = False

        if mode == "discrete":
            a = int(payload)
            if self.env.episode_over:
                terminated_mid = True
            else:
                self._last_obs = self.env.step(a)
                self.step_id += 1
        elif mode == "tool":
            name = payload["name"]
            args = payload.get("arguments", {})
            if name == "stop":
                # Sage success is defined as ``stopped && within radius``.
                # Execute the environment STOP instead of merely terminating
                # the HTTP wrapper, otherwise every primitive-tool rollout is
                # reported as failure even when the agent stops at the goal.
                if not self.env.episode_over:
                    self._last_obs = self.env.step(STOP)
                    self.step_id += 1
                terminated_mid = True
            elif name == "move_forward":
                d = self.A.clamp_distance(args.get("distance"))[0]
                self._last_obs, finfo = self.A.drive_forward(self.env, d)
                tool_result = dict(finfo)
                collided = finfo.get("collided", False)
                blocked = finfo.get("blocked", False)
                self.step_id += int(finfo.get("n_steps", 0))
                if finfo.get("reason") == "episode_over":
                    terminated_mid = True
            elif name == "move_backward":
                d = self.A.clamp_distance(args.get("distance"),
                                          maximum=self.A.MAX_BACK_M)[0]
                self._last_obs, finfo = self.A.drive_backward(self.env, d)
                tool_result = dict(finfo)
                collided = finfo.get("collided", False)
                blocked = finfo.get("blocked", False)
                self.step_id += int(finfo.get("n_steps", 0))
                if finfo.get("reason") == "episode_over":
                    terminated_mid = True
            elif name == "turn":
                direction = self.A.parse_direction(args.get("direction"))
                degrees = self.A.clamp_degrees(args.get("degrees"))[0]
                self._last_obs, tinfo = self.A.drive_turn(
                    self.env, direction, degrees)
                tool_result = dict(tinfo)
                self.step_id += int(tinfo.get("n_steps", 0))
                if tinfo.get("reason") == "episode_over":
                    terminated_mid = True
            else:
                raise ValueError(f"unknown tool {name!r}")
        elif mode == "fm_trajectory":
            wps = payload["waypoints"]
            normalized = payload.get("normalized", True)
            # render_final_only：整条 chunk 的几十个 sub-step 全程不渲染（碰撞/stall
            # 照旧，读 sim pose 不读 obs），执行到 chunk 实际停下的位姿后**只渲一帧**。
            # 把 render HTTP 从"每 sub-step 一次"降到"每 chunk 一次"——闭环 RL 只需
            # 终点观测，中间帧丢弃。默认 False 保持 UI/ReAct 逐帧动画不变。
            render_final_only = bool(payload.get("render_final_only", False))
            exec_render = not render_final_only
            from . import fm_action as FMA
            seconds = payload.get("execution_seconds_remaining")
            deadline = None
            if seconds is not None:
                if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError("execution budget must be finite and positive")
                deadline = time.monotonic() + seconds
            execution_budget_exhausted = False
            def execution_abort():
                nonlocal execution_budget_exhausted
                execution_budget_exhausted = deadline is not None and time.monotonic() >= deadline
                return execution_budget_exhausted
            real_wps = FMA.denormalize(wps, normalized=normalized)
            # P0 chunk-frame 协议：32 个 delta 都在 chunk 起点 fixed frame（官方
            # SocialNav data.py:denormalize_and_cumsum 用 cumsum 还原轨迹）。先 cumsum
            # 成 chunk-frame 绝对轨迹，每步读真实 sim world pose 换算成从当前 pose
            # 出发的 local-ego (dx,dy,dyaw) 再喂 execute_waypoint——避免旧"把
            # fixed-frame delta 当 rolling-local delta 逐条执行"导致转弯后二次旋转
            # （golden probe 实测终点偏差 mean 0.64m / max 2.4m）。
            traj = FMA.chunk_frame_trajectory(real_wps)  # (N,3) chunk-frame abs
            start_pose = self._world_pose_rad()           # (x0,y0,yaw0) world
            waypoint_log = []
            holder = {"obs": self._last_obs}  # drive_* 回调写最新 obs 进来
            def _cap(obs, _i, _h=holder):
                _h["obs"] = obs
            for k in range(len(traj)):
                if execution_abort():
                    break
                if self.env.episode_over:
                    terminated_mid = True
                    break
                # 当前真实 world pose → chunk frame（碰撞/stall 后真实位姿，不靠积分）
                wpose = self._world_pose_rad()
                cur_chunk = FMA.world_to_chunk_pose(
                    (wpose[0], wpose[1]), wpose[2], (start_pose[0], start_pose[1]),
                    start_pose[2])
                dx, dy, dyaw = FMA.target_to_local_command(traj[k], cur_chunk)
                private_geometry = getattr(self, '_private_waypoint_geometry', None)
                native_before = private_geometry() if private_geometry else None
                log = FMA.execute_waypoint(
                    self.env, dx, dy, dyaw, self.A, on_step=_cap,
                    render=exec_render, abort=execution_abort if deadline is not None else None)
                log["chunk_target"] = [float(traj[k, 0]), float(traj[k, 1]),
                                       float(traj[k, 2])]
                log["cur_chunk"] = list(cur_chunk)
                if private_geometry:
                    log['oracle_native_start_position'] = native_before
                    log['oracle_native_end_position'] = private_geometry()
                waypoint_log.append(log)
                if log.get("collided"):
                    collided = True
                if log.get("blocked"):
                    blocked = True
                if log.get("terminated_mid"):
                    terminated_mid = True
                    break
                self.step_id += 1  # Count the attempted waypoint before collision exit.
                if (os.environ.get('ACTION_HEAD_EXECUTION_PROFILE', 'v7_r11_collision') == 'v7_r11_collision'
                        and (log.get('blocked') or log.get('collided') or log.get('aborted'))):
                    break
            if render_final_only:
                # 全程未渲染 → 按停下的真实位姿渲一帧作终点观测
                self._last_obs = self._render_current()
            elif holder["obs"] is not None:
                self._last_obs = holder["obs"]
        else:
            raise ValueError(f"cannot classify action: {action!r}")

        # 取最新观察（drive_* / env.step 都已写进 self._last_obs）
        obs = self._last_obs if self._last_obs is not None else self.env.reset()

        metrics = self.env.get_metrics()
        terminated = _success_of(metrics) or terminated_mid \
            or self.env.episode_over
        truncated = self.max_steps > 0 and self.step_id >= self.max_steps
        reward = self._reward(metrics)
        info = self._metrics(obs, extra={
            "mode": mode, "collided_this_step": bool(collided),
            "blocked_this_step": bool(blocked),
            # Exact result already produced by agent_tools.drive_*.
            # The HTTP policy reuses agent_tools.move_observation /
            # turn_observation from this value, so console and bench expose the
            # same model-visible execution facts without leaking env metrics.
            "tool_result": tool_result,
            "execution_budget_exhausted": bool(mode == "fm_trajectory" and execution_budget_exhausted),
            "waypoint_log": waypoint_log,
            "raw_action": action, "action_id": self._action_echo(mode, action),
            "episode_over": bool(self.env.episode_over),
        })
        return self._observe(obs), reward, terminated, truncated, info

    # ------------------------------------------------------------ classify
    def _classify(self, action):
        """把 action 归类到 (mode, payload)。
        payload 对 discrete 是 int，对 tool/fm 是原始 dict（含 name/waypoints）。"""
        if isinstance(action, bool):  # bool 是 int 子类，先挡掉
            raise ValueError(f"bool action invalid: {action!r}")
        if isinstance(action, int):
            return "discrete", action
        if isinstance(action, str) and action.lstrip("-").isdigit():
            return "discrete", int(action)
        if isinstance(action, dict):
            if action.get("mode") == "fm_trajectory" or "waypoints" in action:
                return "fm_trajectory", action
            if "name" in action:
                return "tool", action
            if action.get("mode") == "discrete":
                return "discrete", int(action.get("action", action.get("value", 0)))
        if isinstance(action, (list, tuple)) and action and \
                isinstance(action[0], (list, tuple)):
            # 裸 [[dx,dy,dyaw]*N] 当 fm_trajectory
            return "fm_trajectory", {"waypoints": action, "normalized": True}
        raise ValueError(f"cannot classify action: {action!r}")

    def _action_echo(self, mode, action):
        if mode == "discrete":
            a = int(action if not isinstance(action, dict) else action.get("value", 0))
            return {"mode": "discrete", "action": a,
                    "name": ACTION_NAMES.get(a, "?")}
        if mode == "tool":
            return {"mode": "tool", **action}
        return {"mode": "fm_trajectory",
                "n_waypoints": len(action.get("waypoints", []))}

    # ------------------------------------------------------------ reward/info
    def _reward(self, metrics):
        """默认稠密 reward = -distance_to_goal（越近越高，NaN→0）。
        显式标注：verl 侧 reward 函数可覆盖此字段，bench 只如实返回指标。"""
        d = metrics.get("distance_to_goal", float("nan"))
        try:
            d = float(d)
        except (TypeError, ValueError):
            d = float("nan")
        if d != d:  # NaN
            return 0.0
        return -d

    def _metrics(self, obs, extra=None):
        m = self.env.get_metrics()
        from vln_gym.observation_profile import validate_environment
        camera = validate_environment(self.env)
        out = {
            "success": _success_of(m),
            "spl": float(m.get("spl", 0.0) or 0.0),
            "oracle_success": bool(m.get("oracle_success", False)),
            "distance_to_goal": float(m.get("distance_to_goal", float("nan")) or 0.0),
            "success_radius": float(m.get("success_radius", float("nan"))),
            "oracle_navigation_error": float(m.get("oracle_navigation_error", 0.0) or 0.0),
            "steps_used": int(self.step_id),
            "scene": self.scene,
            "episode_id": self.episode_id,
        }
        if camera is not None:
            out['observation_profile'] = camera
        for key in ("path_length_m", "shortest_path_m", "collisions"):
            if key in m:
                out[key] = m[key]
        # Evaluator-only SAGE terminal signal.  It is not included in model
        # observation text, but survives the bench API for reward assignment.
        if "official_object_success" in m:
            out["official_object_success"] = bool(m["official_object_success"])
            out["official_object_success_detail"] = m.get(
                "official_object_success_detail", {})
        # gps/yaw 来自 obs（zy 后端由 _enrich_zy_pose 补齐）
        obs = self._enrich_zy_pose(obs)
        if obs is not None:
            gps = obs.get("gps")
            if gps is not None:
                out["gps"] = [float(gps[0]), float(gps[1])]
            compass = obs.get("compass")
            if compass is not None:
                out["yaw_deg"] = float(compass[0])
        # world 系 pose（abotn/rs3d get_metrics 的 position=[x,y,z]、heading_deg 都在
        # backend world 系，与 SFT 数据 build_abotn_sft_data 的 (x,y,yaw) 同系）——闭环
        # 历史 ego 对齐必须用 world 系而非 habitat 的 gps=[x,-z]，否则旋转方向错。透出
        # 给 trainer 的 HistoryBuffer 用。缺省容错（某后端无此字段则不写）。
        pos = m.get("position")
        if pos is not None and len(pos) >= 2:
            out["world_xy"] = [float(pos[0]), float(pos[1])]
        hd = m.get("heading_deg")
        if hd is not None:
            out["world_yaw_deg"] = float(hd)
        if "oracle_world_pose" in m:
            # Explicitly enabled privileged executor telemetry only. This is
            # excluded from observations and from model-visible tool feedback.
            out["oracle_world_pose"] = list(m["oracle_world_pose"])
        for key in ('sage_floor_audit', 'ground_z_m', 'navmesh_manifest_sha256'):
            if key in m:
                out[key] = m[key]
        if os.environ.get("SAGE_RULE_CREDIT", "0") == "1" and self.backend == "sage":
            from sage.rule_credit_state import snapshot
            out["rule_credit_state"] = snapshot(self.env, obs, m)
        if extra:
            out.update(extra)
        return out

    # ------------------------------------------------------------ observe
    def _enrich_zy_pose(self, obs):
        """zy 后端 obs 只有 rgb(+depth)，无 gps/compass。从 sim pose 合成，
        复用 agent_tools._xz/_yaw（habitat 坐标，zy env 刻意 habitat 形），
        让下游 obs 契约与 R2R 一致。r2r/rxr 的 obs 本身带 gps/compass，原样返回。"""
        if obs is None or self.backend not in ZY_BACKENDS:
            return obs
        if obs.get("gps") is None or obs.get("compass") is None:
            try:
                gps = list(self.A._xz(self.env))      # (x, z)
                yaw = float(self.A._yaw(self.env))    # 度
                obs = dict(obs)
                if obs.get("gps") is None:
                    obs["gps"] = [float(gps[0]), float(gps[1])]
                if obs.get("compass") is None:
                    obs["compass"] = [yaw]
            except Exception as e:
                print(f"[vln_gym] _enrich_zy_pose failed: {e}", flush=True)
        return obs

    def _render_current(self):
        """按当前真实位姿渲一帧，不移动 agent（render_final_only 的 chunk 终点观测）。

        所有后端的 sim 都有 observe()（abotn/env.py:215、rs3d/env.py:191 同签名），
        直接调它拿当前位姿的 rgb+gps+compass。取不到就回退上一帧 obs。"""
        sim = getattr(self.env, "sim", None)
        if sim is not None and hasattr(sim, "observe"):
            return sim.observe()
        return self._last_obs

    def _world_pose_rad(self):
        """读当前 world 系 pose (x, y, yaw_rad)，用于 chunk-frame 转换。

        优先 get_metrics 的 position/heading_deg（abotn/rs3d/hsim，与 SFT 数据
        build_abotn_sft_data 的 (x,y,yaw) 同系）；缺则回退 obs 的 gps/compass
        （r2r/rxr，_enrich_zy_pose 已补齐）。两路皆无则报错——chunk-frame 执行
        必须有 world pose，不能静默退回 rolling-local（那是 P0 bug）。
        """
        m = self.env.get_metrics()
        pos = m.get("position")
        hd = m.get("heading_deg")
        if pos is not None and hd is not None and len(pos) >= 2:
            return (float(pos[0]), float(pos[1]), math.radians(float(hd)))
        obs = self._enrich_zy_pose(dict(self._last_obs or {}))
        gps = obs.get("gps")
        compass = obs.get("compass")
        if gps is not None and compass is not None:
            return (float(gps[0]), float(gps[1]), math.radians(float(compass[0])))
        raise RuntimeError(
            "fm_trajectory chunk-frame 执行需要 world pose（position/heading_deg "
            "或 gps/compass），当前后端未提供")

    def _observe(self, obs):
        """构造 JSON 可序列化的 observation dict（rgb 走 base64 jpg）。"""
        if obs is None:
            return {}
        obs = self._enrich_zy_pose(obs)
        o = {"rgb_b64": _jpg_b64(obs["rgb"]),
             "step": self.step_id,
             "instruction": self._instruction(),
             "scene": self.scene, "episode_id": self.episode_id}
        # depth 是 float32，PIL 存不了 jpg，且 VLM/FM 路径不需要；不编码。
        # 如需 depth 可后续走 .npy base64，目前只用 rgb。
        gps = obs.get("gps")
        if gps is not None:
            o["gps"] = [float(gps[0]), float(gps[1])]
        compass = obs.get("compass")
        if compass is not None:
            o["yaw_deg"] = float(compass[0])
        return o

    def _instruction(self):
        if self._instruction_override:
            return self._instruction_override
        if self._episode_obj is not None:
            return self._episode_obj.instruction.instruction_text
        return ""

    # ------------------------------------------------------------- render
    def render(self, mode="rgb", show_goal=False):
        """rgb → PIL Image / np array；map → top-down map np array。None 表示无。"""
        obs = self._last_obs
        if obs is None:
            return None
        if mode == "rgb":
            return obs["rgb"]
        if mode == "map":
            # zy 后端 top_down_map=None（无 habitat 地图格式），直接用其 render_top_down。
            if self.backend in ZY_BACKENDS:
                arr = self.env.render_top_down(show_goal=show_goal)
                return np.asarray(arr) if arr is not None else None
            tdm = self.env.get_metrics().get("top_down_map")
            if show_goal:
                from vlnce_training import reference_map
                tdm = reference_map(self.env, tdm, self.base_maps)
            return self._render_map(tdm, show_goal=show_goal)
        return None

    def _render_map(self, tdm, show_goal=False):
        if not isinstance(tdm, dict) or "map" not in tdm:
            return None
        from habitat_topdown import agent_map_raster
        m = (np.asarray(tdm["map"], dtype=np.uint8) if show_goal
             else agent_map_raster(tdm["map"]))
        fog = tdm.get("fog_of_war_mask")
        if fog is not None:
            fog = np.asarray(fog)
        rgb = self.base_maps.colorize_topdown_map(m, fog, fog_of_war_desat_amount=0.82)
        coords = tdm.get("agent_map_coord")
        angles = tdm.get("agent_angle")
        if coords is not None:
            coords = np.atleast_2d(np.asarray(coords))
            angles = (np.atleast_1d(np.asarray(angles)) if angles is not None
                      else [0.0] * len(coords))
            for i, pos in enumerate(coords):
                rgb = self.base_maps.draw_agent(
                    image=rgb, agent_center_coord=(int(pos[0]), int(pos[1])),
                    agent_rotation=float(angles[i]),
                    agent_radius_px=max(4, min(rgb.shape[0], rgb.shape[1]) // 40))
        if rgb.shape[0] > rgb.shape[1]:
            rgb = np.rot90(rgb, 1)
        return rgb

    # --------------------------------------------------------------- close
    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    # ---------------------------------------------------------- internal
    def _check_actuation(self):
        """复用 vln_model.check_actuation 思路：防 actuation 被泄漏改动。"""
        if not self.BASE_AMOUNTS:
            return
        for name, want in self.BASE_AMOUNTS.items():
            act = self.A.actuation(self.env, name)
            if abs(float(act.amount) - want) > 1e-6:
                act.amount = want


if __name__ == "__main__":
    # 最低限度冒烟（不跑 habitat，只验 schema 可序列化）
    import json
    print(json.dumps(VLNEnv.action_space, ensure_ascii=False, indent=2)[:400])
    print("---")
    print(json.dumps(VLNEnv.observation_space, ensure_ascii=False, indent=2))
