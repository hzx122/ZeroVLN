# 3DRS 后端：3DGS 渲染 + 自研 navmesh 运动学

`vln_demo` 原来只能跑 habitat/MP3D。3DRS（`/workspace/agentic_sim/3DRS`，
143 个真实采集场景）**habitat 打不开**，所以这个目录把仿真器换掉了：视觉交给
ABot-Navigation 的 3D 高斯渲染服务，运动学自己按场景自带的 `.navmesh` 算，
再用一层 habitat 形状的 facade 包起来，让上层代码一行不改地跑。

## 为什么原仿真器用不了

一个场景目录长这样：

```
3DRS/<scene>/
  <scene>.gs.ply          # 3D 高斯（30-40 万 primitive），不是网格
  <scene>.mesh.ply        # 有几何但没纹理/材质
  <scene>.navmesh         # Recast/Detour，MSET **v2**
  <scene>_metainfo.json   # scene_type + 中文场景描述
  <scene>_bev.png         # 参考俯视图
```

三条都堵死：

1. **`.gs.ply` 是高斯，habitat 没有高斯光栅器**。habitat 只会画三角形。
2. **`.mesh.ply` 没纹理**，硬塞进 habitat 只能得到灰白几何体，VLN 模型看不出任何语义 —— 这个 demo 全靠 RGB 语义驱动，等于白跑。
3. **navmesh 是 MSET v2，habitat-sim 0.2.4 只认 v1**，`pathfinder.load_nav_mesh()` 直接 false。就算前两条解决了也走不了路。

所以不是「配一下就能用」，而是必须换。

## 换成什么

```
        agent_tools.py / run_agent  （完全没改）
                    │  env.step / env.sim.get_agent_state / action_space
                    ▼
              rs3d/env.py            habitat 形状的 facade
                ├── rs3d/nav.py      导航查询：地面高度、可走性、撞墙滑行
                │      └── rs3d/navmesh.py   MSET v2 解析（不依赖 habitat）
                └── rs3d/gs_client.py  HTTP → 3DGS 渲染服务（COLMAP 位姿）
```

- **视觉**：`render_sim_dynamic.py`（ABot-Navigation）常驻进程，GPU 上按 LRU 池着场景，
  POST 一组 COLMAP 外参就回 RGB（可选 depth）。单视角约 24 ms。
- **运动学**：`nav.py` 只用 navmesh 里 **可走** 多边形的 detail 三角形（这就是 habitat
  报告为 navigable 的那块面），按 1 m 网格分桶，点查询只碰几个三角形而不是全部 ~3 万个。
  前进按 0.05 m 逐步走（不会穿墙），每步重新贴地（所以能上坡上台阶），撞墙后把**剩余**
  位移投影到最近的边界边再试一次 —— 相当于 habitat 的 `allow_sliding`。
- **facade**：`env.py` 实现上层真正用到的那点接口（`reset/step/get_metrics/episode_over`、
  `sim.agents[0].agent_config.action_space`、`sim.get_agent_state`、
  `sim.previous_step_collided`），**全程不 import habitat**。
  动作量在 `act()` 里现取，所以 `agent_tools` 临时改 `actuation.amount` 那套照样生效。

## 怎么跑

```bash
# 1) 起渲染服务（常驻，占一张卡，跨多次 console 运行复用）
cd /workspace/agentic_sim
venv_render/bin/python ABot-Navigation/render_sim_dynamic.py \
    --scenes_root 3drs_gs_root --port 7036 --gpus 0 \
    --max_scenes_per_gpu 2 --sh_degree 0 --render_scale 1.0

# 3drs_gs_root 是 3DRS/<scene>/<scene>.gs.ply 的软链树，用这个生成：
#   python src/rs3d/make_scenes_root.py --gs-root .../agentic_sim/3drs_gs_root

# 2) 起 vLLM（和 habitat 后端同一套脚本）
bash ~/vllm/vllm-vln-35B.sh

# 3) 起控制台
cd src && python vln_model.py --backend 3drs --port 5039
#   --scenes A,B,C          只挂这几个场景（默认全部 143 个）
#   --episodes_per_scene N  每个场景采几个出生点（默认 1）
#   --render_url http://127.0.0.1:7036
```

浏览器打开 `http://<host>:5039`，和 habitat 后端同一个页面：场景/回合下拉、
W/A/S/D 手动驾驶、开始交给模型、第一视角 + 俯视图。

参数默认值从 `src/vln_demo.yaml` 里读（640x480、hfov 120、前进 0.25 m、转 30°、
sliding 开），所以两个后端的机器人走法和视野是一样的，结果可比。

## 和 habitat 后端的差别

| | habitat | 3drs |
|---|---|---|
| 场景 | MP3D 网格 | 3D 高斯，走渲染服务 |
| 回合 | R2R 数据集（含指令 + 目标点） | **没有回合**，每个场景默认 1 个可复现的出生点 |
| 指令 | 数据集原文 | metainfo 场景描述原文，预填在输入框里，可改可换 |
| 场景名 | MP3D id | metainfo 描述截出的中文标题 + id |
| `distance_to_goal` / `success` / `spl` | 真值 | **NaN** |
| 驱动 | 经典 / ReAct | **只有 ReAct**（选了经典会自动切回并提示） |
| 俯视图 | habitat `top_down_map` measure | 自己按 navmesh 栅格化（最大连通岛更亮） |

**目标点是故意没有的**：3DRS 出的是场景不是回合，没有 ground-truth 路径，
硬造一个目标点等于自己发明评测标准。所以 goal 相关指标一律 NaN，结束时报
「走了 X m · 本场景无目标点，不判定成功」而不是永远判失败；页面上距离显示为 `-`。
自由探索和指令跟随能跑，VLN-CE 打分不能。

**指令预填的是场景描述原文**：资产自带的唯一文本就是 metainfo 里那段 VLM 写的场景
描述，它不是导航命令、也没有东西拿它打分，但它是这个场景真实存在的说明，所以直接
原样填进指令框当起点（`default_instruction()` 不改写、不裁剪），你看着它改成想跑的
任务就行。没有描述的场景仍然是空的，留空点开始会被拦下。那段描述同时截出下拉框里的
中文场景名（`scene_title()` 取第一个分句），加载时也打进日志。

## 坐标与位姿约定

- 世界系 y-up，和 `.gs.ply` 一致；yaw=0 朝 −z，从上往下看逆时针增大 —— 和 habitat 一样。
- 渲染服务吃 COLMAP：`images.txt` 每行 `id qw qx qy qz tx ty tz camera_id name`，
  是 **world-to-camera**；相机 +x 右、+y 下、+z 前。`gs_client.look_at_w2c` 负责这个转换。
- 相机装在 `agent 位置 + [0, sensor_height, 0]`（默认 1.25 m）。
- **`/render_gs` 的两个字段各要一行纯文本**，不是整个文件体也不是 list：
  服务端自己会 `[cam_extrinsics]` 包一层再 `.split()`，多包一层就是 500。

## 已验过的点

- 渲染：143 个场景全部被服务发现；出图和 `_bev.png`、`metainfo` 描述对得上；
  roll/yaw 符号正确（不是镜像也不是倒的）。
- depth 是真米：相机抬 1.25 m，depth 同步 +1.25，斜率 1.0；navmesh 比地面恒定高 0.26 m。
- navmesh：140/143 场景的可走面积与 `_metainfo.json` 一致；
  采样点 0 个读错高度、0 个走出网格、0 个超长步；单步 0.3–0.7 ms。
- 俯视图和资产自带的 habitat 生成的 `_bev.png` 洞对洞一致。
- facade 驱动**未修改的** `agent_tools`：动作 id、动作量存取还原、子步动画、
  后退（负 `forward_m`）、跑完位姿仍在 navmesh 上，全部正常。
- 端到端：Qwen3.6-35B 在 `0U4zEqlqDJ` 跑完一个 ReAct 回合（7 轮 / 16 仿真步 / 2.9 m），
  描述与场景吻合，模型主动 `stop`；场景热切换 1.2 s；不存在的回合号被拒绝且控制台不挂。

## 文件

- `navmesh.py` — MSET v2（Recast/Detour tile）解析：多边形、detail 三角形、
  `POLYFLAGS_WALK`、连通岛。不依赖 habitat。
- `nav.py` — 空间索引 + 点查询（`is_navigable`/`snap`/`height_at`）、
  按面积均匀采样、边界边、`try_step` 逐步前进 + 滑行。
- `gs_client.py` — COLMAP 内外参生成 + 渲染服务 HTTP 客户端。
- `env.py` — habitat 形状的 `RS3DSim` / `RS3DEnv`、回合采样、俯视图。
- `model_client.py` — `stream_chat` 真正用到的那点 evaluator 接口
  （URL、几个 args、`_to_openai_messages`），好让 3DRS 后端不必 import habitat。
- `check_navmesh.py` — 拿 `_metainfo.json` 的面积校验解析器。
- `make_scenes_root.py` — 生成渲染服务要的 `.gs.ply` 软链树。
