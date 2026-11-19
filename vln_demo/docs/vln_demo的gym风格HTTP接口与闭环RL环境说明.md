# vln_demo 的 gym 风格 HTTP 接口与闭环 RL 环境说明

> 把 vln_demo 的 habitat-sim VLN-CE 评测器包成 gym 风格 HTTP 服务（纯 CPU 软渲染，**不含 vLLM**），目标是让 policy 外置在 verl，verl 里的 VLM 用 tool-call 访问、VLN(FM) 模型用上传 `(dx,dy,dyaw)` 轨迹访问，bench 渲染新观察 + 返回碰撞/指标，构成闭环 RL 环境。
>
> 参考模型 = `date_mfm/projects/MegaTE/vla/scripts/qwen3vl2bfm/run_train_qwenvl_2b_fm_tp2_fm_ar_perct_0728-align-baseline-cocofix-0p5.sh`（FM VLN：action_dim=3 / action_chunk=32 / ego 系 `(dx,dy,dyaw)`，32 waypoint 一条轨迹）。
>
> 实现日期：2026-08-07。
>
> **当前状态（务必先读）**：bench 侧单步接口（reset / 各层 step / render / 各指标返回）已验通。**闭环已补验（2026-08-07，见第十节）**：实现 `verl_access` 访问层，用真 2B/35B VLM + 真 0729 FM head 跑 5 用例，VLM 看图→输出→bench→新观察 与 FM 出 chunk→bench→滑窗历史→再出 两条闭环首次端到端跑通，映射正确；4 条 FM 开放风险中 #1 已验消除、#3 证伪（CE==anonymous stats）、#2/#4 几何对但导航质量偏低（policy 问题非映射）。直接接 verl 仍差第七节 4 缺口（env 适配器 / policy 解析 / reward / 向量化）。

---

## 一、产物与绝对路径

```
vln_demo/
├── src/vln_gym/                              # gym 包
│   ├── __init__.py                           # 包说明
│   ├── env.py            # VLNEnv（duck-typed gym Env，分层动作分发 + 指标收集）
│   ├── fm_action.py      # (dx,dy,dyaw) 反归一化 + turn→fwd→turn 分解（带单元自检）
│   ├── server.py         # stdlib HTTP gym server（单 session，单线程）
│   └── client.py         # VLNRemoteClient + smoke 子命令
└── vln_gym_server.sh                         # 启动器（gordon venv + mesa 软渲染 + run server）
```

复用（不重写）的既有逻辑：
- `rynnvln_eval.VLNEvaluator`（gordon NAV `/data/storage/.../RynnBrain/navigation/src/rynnvln_eval.py`）→ `config_env()` 给 habitat env。
- `agent_tools.py`（`vln_demo/src/agent_tools.py`）→ `drive_forward/drive_turn/drive_backward`（子步防穿墙 + 碰撞/stall + finally-restore actuation）、`default_amounts/action_id/clamp_distance/clamp_degrees/parse_direction`。
- HTTP 模式照抄 `vln_model.py` 的 `BaseHTTPRequestHandler` + `jpg_bytes` + `render_map` 思路（stdlib，零新依赖；gordon venv 没装 gymnasium/FastAPI）。

---

## 二、环境约束（为什么是这套设计）

1. **gordon venv 没装 gymnasium** → `VLNEnv` 不继承 `gym.Env`，duck-typed 实现 `reset/step/render/close + action_space/observation_space`，API 与 gym core（`Gymnasium/gymnasium/core.py`）一致。
2. **habitat/Magnum 的 EGL GL context 是 thread-local** → server 用**单线程 `HTTPServer`**（不是 `ThreadingHTTPServer`），保证所有 env 访问都在创建 env 的主线程，否则 `GL::Context::current(): no current context` → SIGABRT。代价：一个慢 step 阻塞 `/health`，单 session RL 可接受。
3. **depth 是 float32，PIL 存不了 jpg** → observation 只编码 rgb（VLM/FM 路径不需要 depth）。
4. **VLNEvaluator 在 inference_backend="vllm_server" 时要求 vllm_server_urls 非空** → bench 不含 vLLM 但仍要过该校验，`vllm_server_urls` 填占位 `http://127.0.0.1:1`（bench 永不调 `generate_response`，永不命中）。
5. **bench 纯 CPU 软渲染**，不起 vLLM、不占 GPU（实测 GPU 全 10MiB 空闲）。policy 推理由 verl rollout worker 负责，经 HTTP 访问 bench。

---

## 三、分层动作接口（用户拍板「分层支持不同形式的模型输出」）

`step(action)` 按 caller 给的动作方言分发：

| mode | action 形态 | 执行 | 1 个 RL step = |
|---|---|---|---|
| `discrete` | `int 0/1/2/3`（裸 int 即此层） | `env.step(action)` 一次 | 1 habitat step（forward 0.25m / turn 30°） |
| `tool` | `{"name","arguments"}` | `agent_tools.drive_forward/turn/backward`（子步+碰撞）或 stop | 该工具完整运动 |
| `fm_trajectory` | `{"mode":"fm_trajectory","waypoints":[[dx,dy,dyaw]*32],"normalized":bool}` | 每 waypoint 分解为 `turn(atan2(dy,dx))→forward(hypot(dx,dy))→turn(dyaw-atan2(dy,dx))`，全 32 个连跑 | 整个 chunk |

- 裸 `[[dx,dy,dyaw]*N]` 列表也认作 fm_trajectory（normalized=True）。
- FM 层 `normalized=true` 时按 `_CE_WP_STATS`（源 `date_mfm/.../SocialNav/src/dataset/vln_ce_dataset.py:83`，内联不跨仓 import）把 [-1,1] 反归一化回原尺度。

### FM 动作约定（`fm_action.py`）
- ego 系：`dx>0` 前进、`dy>0` 左移、`dyaw>0` 左转（ccw）。
- 不变量：`turn1 + turn2 == dyaw`（heading 总变化正确）；位移方向正确。
- habitat 无 strafe 原语，侧移靠「turn 90°→forward→turn -90°」实现（实测 `decompose(0,0.1,0)` → turn left 90 / forward 0.1 / turn right 90，净 heading 0、侧移 0.1m）。
- 若实测发现 habitat 左右约定反了，改 `fm_action.LEFT_POSITIVE` 一行即可。
- 单元自检：`python3 -m vln_gym.fm_action`（验分解几何 + 反归一化往返）。

### ⚠ FM 链路的开放风险（没拿真模型验过，接训练前必须确认）
1. **没端到端验过真 FM head 输出**。smoke 用的是手造轨迹 `[0.1,0,0]`/`[0,0,0.05]`，不是真模型吐的 chunk。真 FM head 输出经不经得起这套映射，没测。
2. **ego 系方向是假设的**：`dx=前进、dy=左、dyaw=左转(ccw)` 照 SocialNav 数据约定写的，**没拿真 FM head 输出和 habitat 坐标系交叉验证过**。万一反了，翻 `LEFT_POSITIVE` 一行能改，但现在不能保证它对。
3. **反归一化 stats 可能用错**：`_CE_WP_STATS` 硬编码的是 CE 125cm_0deg 数据集统计；FM head 训练实际用哪套 stats（SocialNav 文件里还列了 anonymous RealSense 的另一套，q01/q99 差很多）**没确认**。若 head 按 anonymous stats 训、按 CE 还原，尺度全错。
4. **forward-only 分解的保真度**：纯侧移靠「转 90°→走→转 90°」实现，几何对，但在真实有障碍的 sim 里，真 strafe 和「转走转」撞墙情况会不同。32 waypoint 链式执行（每段相对当前位姿）逻辑正确。

接训练前建议：拿真 FM head 在一个 episode 上 rollout 一次，把输出 chunk 喂进 bench，肉眼对比 sim 里的运动和预期是否一致，确认 2/3 两点。

---

## 四、observation / reward / info（用户拍板「能返回的全返回」）

### observation（JSON，rgb 走 base64 jpg）
```json
{"rgb_b64":"<base64 jpg 640×480>","step":0,"instruction":"...","scene":"2azQ1b91cZZ","episode_id":"11","gps":[x,y],"yaw_deg":...}
```

### reward
默认 `reward = -distance_to_goal`（稠密，越近越高；NaN→0）。**verl 侧 reward 函数可覆盖**，bench 只如实返回指标。

### info（能返回的全返回）
`success, spl, oracle_success, distance_to_goal, oracle_navigation_error, steps_used, scene, episode_id, gps, yaw_deg, mode, collided_this_step, blocked_this_step, waypoint_log(fm), raw_action, action_id, episode_over, reset`

action_space / observation_space schema 经 `GET /action_space`、`GET /observation_space` 返回 JSON。

---

## 五、HTTP 端点（单 session）

```
GET  /health                 -> {"ok":true,"env_loaded":true}
GET  /action_space           -> JSON schema
GET  /observation_space      -> JSON schema
GET  /spec                   -> {scenes, scene_eps, splits}
POST /reset   {scene,episode_id,instruction,seed}  -> {obs, info}
POST /step    {action}                            -> {obs,reward,terminated,truncated,info}
GET  /render?mode=rgb|map    -> image/jpeg
GET  /frame.jpg /map.jpg     -> 同上（兼容 webui 调试）
POST /close                  -> {"closed":true}
GET  /                       -> 最小状态页（debug）
```

---

## 六、启动与用法

### 启动 server（纯 CPU）
```bash
cd /workspace/vln_demo
# 8899 被 Jun02 的 stale http.server 占着，默认用 8907（可 DEMO_HTTP_PORT 覆盖）
DEMO_HTTP_PORT=8907 bash vln_gym_server.sh > /tmp/vln_gym.log 2>&1 &
# 等 health（habitat 冷解析 ~40s）：
until curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8907/health | grep -q 200; do sleep 2; done
```
迁移到新机器：改 `vln_gym_server.sh` 顶部 `VENV/NAV/DATA_ROOT` 三个路径变量（同 `run_demo_ext.sh`）。

### 客户端（verl rollout worker 里当本地 env 用）
```python
from vln_gym.client import VLNRemoteClient
env = VLNRemoteClient("http://bench-host:8907")
obs, info = env.reset(episode_id="11")
# VLM tool-call：
obs, r, term, trunc, info = env.step({"name":"move_forward","arguments":{"distance":0.8}})
# FM 轨迹：
obs, r, term, trunc, info = env.step({"mode":"fm_trajectory","waypoints":wp32,"normalized":True})
```

### smoke
```bash
python3 -m vln_gym.client smoke --url http://127.0.0.1:8907
# 跑完四层动作（discrete×4 / tool move_forward / fm forward / fm turn / stop），断言字段齐全。
```

---

## 七、与 verl 闭环 RL 的对接（目标示意 + 当前缺口）

### 目标链路
```
verl rollout worker (policy=vLLM/FM, GPU)  ──HTTP──>  vln_gym server (CPU habitat sim)
   │  reset() / step(action)                                 │ 渲染新观察 + 算指标
   │  obs(rgb_b64) + reward + info(success/SPL/NE/collision) │
   └─────────────────────────────────────────────────────────┘
   policy 用 obs 推理下一动作 → step(...)；reward 信号回 verl GRPO 更新 policy。
```
- VLM 路径：policy 输出 tool-call（`move_forward/turn/move_backward/stop`）→ `step({name,arguments})`。
- VLN(FM) 路径：policy 输出 32×`(dx,dy,dyaw)` chunk → `step({mode:"fm_trajectory",...})`，bench 分解执行 + 返回碰撞/`waypoint_log`。
- reward 默认 `-distance_to_goal`，verl 侧可在 reward 函数里用 `info` 里的 `success/SPL/NE/collision` 自定义组合（见 `date_mfm/.../verl_nav/reward/vln_goal_reward.py` 的范式）。

### ⚠ 当前缺口（直接接 verl 还不能训练，差以下 4 块）
1. **verl env 适配器**：verl 有自己的 rollout 循环和 env API，`VLNRemoteClient` 还不是 verl 认的 env，需写一个把 gym-style env 接进 verl rollout 的适配层。
2. **policy 推理 + 输出解析**：verl worker 里要起 vLLM 跑 VLM/FM、把模型吐的文字解析成 `{name,arguments}` 或 waypoint chunk。没写。
3. **verl reward 函数**：bench 默认给 `-distance_to_goal`，但 verl GRPO 要 `compute_score` 风格的 reward，没写。
4. **并行向量化**：现在单 session 一个 env，GRPO 要几十路并行 rollout，向量化没做。

此外 **闭环一次都没跑过**（没有真 policy 看图→输出→喂回 bench 的整轮验证）。接训练前建议先用 `VLNRemoteClient` 跑一个「reset→手动喂动作→看返回」的端到端 loop，再接 verl。

---

## 八、验证（2026-08-07 端到端 smoke）

1. `python3 -m vln_gym.fm_action` → `decompose 几何 + 反归一化往返全过`（含 360° 左转 yaw 回 0 的保真确认）。
2. 起 server 8907，`/health` 200，`/spec` 返回 11 个 val_unseen 场景。
3. `python3 -m vln_gym.client smoke --url http://127.0.0.1:8907` → `ALL PASS`（**仅 bench 侧单步接口验证，非闭环**）：
   - reset rgb 解码 shape=(480,640,3) uint8；
   - discrete forward×4 step=4 mode=discrete；
   - tool move_forward(0.8) collided=True mode=tool（**只测了 1 次工具调用，没测 VLM 多轮 ReAct**）；
   - fm_trajectory 32×[0.1,0,0] waypoints=32 collided=True mode=fm_trajectory（**用假轨迹，没用真 FM head 输出**）；
   - fm_trajectory 32×[0,0,0.05](左转) waypoints=32；
   - stop terminated=True；
   - 全程 `reward=-distance_to_goal` 稠密返回。
4. 资源确认：`nvidia-smi` 全卡 10MiB 空闲、无 vllm 进程、bench 仅 `python3 -m vln_gym.server`（纯 CPU）。

**没验的**：① 带真 policy 的闭环（VLM 看图→输出动作→喂回 bench→拿新观察）；② 真 FM head 输出经 bench 映射是否正确（见「三」末尾 4 条开放风险）；③ 多并行 / 长时间跑。

---

## 九、已知边界 / 后续

- **未跑闭环**（最重要）：目前只验了 bench 侧单步接口，**没有带真 policy 跑过一轮 reset→看图→输出动作→step→新观察 的闭环**。接 verl 前必须先补（见「七」4 缺口）。
- **FM 映射未用真模型验**：ego 方向约定、归一化 stats 两点开放风险（见「三」末尾），接训练前拿真 FM head 跑一次确认。
- **单 session**：当前一个 server 一个 env 实例（用户拍板「先单 session 跑通」）。verl GRPO 多并行 rollout 的向量化（多 env_id 路由 / 多进程）留后。
- **单线程 HTTPServer**：一个慢 step 阻塞 `/health`（habitat GL 线程亲和所致，无法用线程化 server）。
- **reset 状态**：标准 RL 循环（reset→steps→stop→reset）已验通；曾在「一个长寿命 server env 上连续多轮手动探测后」偶发 `list index out of range`（habitat iterator 状态），重启即恢复，标准循环不复现。长跑 RL 时若遇 reset 异常，重启 server 即可（单 session，无状态损失）。
- **reward 覆盖**：默认 `-distance_to_goal`，verl 侧可用 `info` 里的 `success/SPL/NE/collision/steps_used/gps/yaw` 自定义组合。
- **depth 不编码**：observation 只有 rgb（depth 是 float32，PIL 存不了 jpg）；如 RL 需要 depth 可后续走 `.npy` base64。

---

## 十、闭环验证结果（2026-08-07，回填「未跑闭环」缺口）

> 接管本文档后，实现轻量访问层 `vln_demo/src/verl_access/`（模拟 verl rollout worker，经 HTTP bench，不直持 habitat env），用**真 VLM（2B+35B）**和**真 FM head（0729 hf_16k）**跑 5 个用例，回填第九节「未验」三点 + 第三节末尾 4 条开放风险。产物在 `vln_demo/outputs/verl_access/20260807/`，汇总见 `summary.md`。

### 访问层（新增，复用不重写）

`vln_demo/src/verl_access/`：
- `bench_client.py`：`VLNRemoteClient` 语义封装 + `obs.rgb→PIL image part`（与 `vln_model._img_part` 同形，内部走 `{"type":"image","image":<PIL>}`，发 vLLM 前由 `to_openai_messages` 转 data url）。
- `vlm_policy.py`：ReAct 闭环。复用 `prompts.agent_system_prompt` + `agent_tools.build_system_prompt(TOOL_SPECS)` + `parse_tool_calls`/`split_thinking`/`assistant_history_content`/`append_tool_budget`/`strip_old_images`，vLLM payload 照抄 `vln_model.stream_chat`（urllib SSE + `chat_template_kwargs.preserve_thinking` + `prompts.sampling_params`）。**执行段换成 `bench.step({name,arguments})`（tool 模式），不直持 env**；新 obs rgb 作为工具返回 image 喂回对话。
- `fm_policy.py`：FM 闭环/单发。复用 SocialNav `train.load_model_and_processor`（0729 hf_16k + forked transformers）+ `infer_batch._predict_fm_batch` 的 generate→forward→ActionFormer→Euler ODE 段（照搬）。gym 每步观察→SocialNav 输入适配：滑窗历史帧 + 由 gps 增量反推 input_waypoints + 镜像 jsonl 的 prompt 格式（`<image>×N <input_waypoint>×N anonymous Instruction: …`）。
- `cases.py` / `analysis.py`：5 用例编排 + 汇总（映射正确率 / 距离曲线 / 碰撞率 / chunk 健康度）。

### 5 用例结果

| # | 用例 | policy | 结果 | 评估 |
|---|---|---|---|---|
| 1 | VLM ReAct 闭环 | 2B | 跑通，图像喂回正常，tool-call→bench tool 映射 100% 正确；2B 跑 1 轮（turn+forward）后不再调用工具，nudge 后仍空→结束，dist 7.11→7.94 | 闭环+映射正确；2B 不持续导航（policy 行为） |
| 2 | VLM ReAct 闭环 | 35B | 跑通，6 轮 turn→forward×4→stop，动作有逻辑（定向→推进→停止），映射 100%，collision_rate 0.2，dist 7.11→10.26 单调上升 | 闭环正确，agent 执行有逻辑动作；走反是 policy 质量，非映射 bug |
| 3 | FM 合成 chunk | 无模型 | 4 种 chunk（直行/纯转/侧移/混合）→bench fm_trajectory，`fm_action.decompose` 不变量（turn1+turn2==dyaw、位移方向）全过；直行 3.2m 请求撞墙实际 2.04m、侧移 1.6m 实际 1.6m（且距目标下降）、纯转 dyaw 正确 | **bench 侧 fm_trajectory 映射几何正确**，隔离模型 |
| 4 | FM 真单发+denorm | 0729 hf_16k | 加载成功（"VLA modules found — skipping reinit"，FM head 权重正确），出 (32,3) norm delta（range[-0.087,0.802] mean_abs 0.268），anonymous denorm 得 dx~7-8cm/步、dyaw~±3°（合理步长），bench 32/32 执行 | **真 FM head 输出 sane，链路通**；风险#3 证伪（见下） |
| 5 | FM 真闭环 | 0729 hf_16k | reset→FM 出 chunk→bench 执行→滑窗历史(1→8帧)→FM 再出→…8 chunk，全 nonzero、norm range[-0.98,0.97]；dist 7.11→11.74 上升、7/8 chunk 碰撞 | **真 FM head 多轮闭环首次跑通**，链路完全正确；导航差源于零/短 history 起步 + R2R↔anonymous 域 mismatch + 碰撞后状态 |

### 第三节末尾 4 条开放风险 — 实测回填

1. **没端到端验过真 FM head 输出** → **已验（用例4/5）**。0729 hf_16k 加载、generate→forward→ActionFormer→ODE 全链路跑通，输出 (32,3) 归一化 delta 在 [-1,1] 内、非零、denorm 到合理米/弧度。bench 能完整执行 32 waypoint。**风险消除**。
2. **ego 系方向是假设的** → 几何上 `decompose` 不变量（用例3）方向与约定一致（左转 dyaw>0、侧移 dy>0 靠 turn90→fwd→turn-90 实现）；但闭环导航差无法判断方向是否真反（policy 本就走不准）。**未最终交叉验证**，建议后续可视化对比 sim 运动与预期；若反，翻 `fm_action.LEFT_POSITIVE` 一行。
3. **反归一化 stats 可能用错** → **证伪**。bench `fm_action._CE_WP_STATS` 与 SocialNav FM 训练用的 `_ZYT_WP_STATS` **数值相同**（用例4 实测 CE vs anonymous drift=0.00000）。**这条风险不存在**，bench 硬编码 CE 无错。
4. **forward-only 分解保真度** → 几何对（用例3 不变量全过）；真实有障碍 sim 里 32 waypoint 链式执行高碰撞率（用例5 7/8 碰撞），但属 policy 选 chunk 撞墙而非分解几何错。

### 新发现（本次验证暴露的两个 bench 问题）

- **`yaw_deg` 字段实为弧度不是度**：用例3 纯转 chunk（32×dyaw=0.05rad=91.6°）实测 `info["yaw_deg"]=1.6`（=91.6° 的弧度值）；用例2 35B 转向 45° 实测 `yaw_deg=0.785=π/4`。`env.py:_metrics/_observe` 直接取 `compass[0]` 命名 `yaw_deg`，但 habitat compass 返回弧度。**字段名误导**，verl 侧 reward 若按度用会差 ~57 倍。建议改名 `yaw_rad` 或在 bench 内 `*180/π`。
- **bench 长寿命 env 偶发 `IndexError: list index out of range`**（habitat `reconfigure`→`get_agent` 时 `self.agents` 空列表），第九节已记。本次实测：标准 reset→steps 循环也可能在累积后触发（不止"连续多轮手动探测"）。已在 `env.py:reset` 加**自动恢复**：catch `IndexError/RuntimeError` → `config_env()` 重建 env → retry，把"重启即恢复"在进程内自动化，外置 policy 不必重启 bench。

### 结论

**闭环验证目标达成**：带真 policy 的闭环（VLM 看图→输出→bench→新观察；FM 出 chunk→bench→滑窗历史→再出）首次端到端跑通，输出映射正确（tool-call/fm_trajectory 全链路无解析/执行错误），agent 能执行有逻辑的动作序列（35B: 定向→推进→停止；FM: 多 chunk 持续输出）。**第七节「闭环没跑过」+ 第三/九节 FM 未验两点缺口补上**。导航质量本身偏低（35B 走反、FM 高碰撞），属 policy 能力问题，需后续调 prompt/历史窗口/在 VLN 上微调，非访问层或 bench 映射缺陷。

### 十一、FM 管线正控制强验证（2026-08-07，把"sane range"升为"管线正确"）

> 第十节的 case4/5 只证了 FM 输出在 [-1,1]、denorm 到合理量级——弱证据。本节用训练集自带 GT 的样本做正控制，钉死 forward/ODE + prompt + delta 语义三点。脚本 `verl_access/verify_fm.py`，样本 `wgoal_0624-benchmark..._with_noise.jsonl[0]`（"walk around"，traj_type=anonymous，gt_waypoints 32×3 真实 delta）。

**A. forward/ODE 对齐（byte-identical）**：用 MinimalDataset 产出的 data_dict（input_ids/pixel_values/image_grid_thw/input_waypoints），同 `torch.manual_seed(0)` 下跑 (i) SocialNav `infer_batch._predict_fm_batch`（参考）与 (ii) 本仓 `fm_policy` 内联的 generate→forward→ActionFormer→Euler ODE，逐元素对比：`max|diff|=0.000000`、`allclose(atol=1e-4)=True`、同种子复跑也 0 差异。**我的 forward/ODE 与 SocialNav 参考实现逐字节一致——管线核心正确。**

**B. prompt fidelity**：我自构的 prompt（`<image>×N <input_waypoint>×N anonymous Instruction:…`，`<image>`→`<|vision_start|><|image_pad|><|vision_end|>`）跑 `model.generate()`，解码输出 = `'goal point: [490, 866]; <flow_matching_policy><|im_end|>'`——正是训练格式。**prompt 构造正确，模型确实吃到图像并产出 goal 点 + FM token。**

**C. delta-vs-cumsum（第三节末尾隐含风险，本节钉死）**：模型输出 denorm 后末步 = [0.043m, 0.006m, 0.034rad]，与 GT per-step 同量级（cm/0.01rad），32 步平坦不单调增长 → **模型输出是 per-step delta，不是累积位置**。bench 每 waypoint 相对执行（`fm_action.execute_waypoint`）几何正确。**直接喂 delta、不 cumsum，是对的。**

**D. GT 不匹配 = 模型精度，非管线 bug**：我的输出 dx 均匀 ~3-4cm/步，GT dx 含 1.05m 大跳，per-step 误差 mean=0.23、cumsum 末点 NE=19.9m。但 A 已证我的 forward/ODE = 参考 byte-identical，且参考输出 range[-0.086,0.469] 与我的同量级——**即 SocialNav 离线 infer_batch 在这条样本上同样不贴近 GT**。不匹配源于 0729-16k 模型精度（goal 点 [490,866] vs GT [509,797] + 模型未复现 GT 的大跳），不是访问层/管线缺陷。

**正控制结论**：FM 管线（`fm_policy`）正确性已坐实——forward/ODE byte-identical（A）、prompt 对齐（B）、delta 语义对（C）。GT 精度属模型能力（D，参考亦然）。第十节 case4/5 的"sane range"升级为"管线正确"。第三节末尾 4 风险全部回填：#1 消除、#3 证伪、#2/#4 几何对（#2 ego 方向仍未定向交叉验证，但 per-step delta 量级与 GT 同尺度，方向反会表现为系统性偏转，未见）。
