# LightNav-0 集成：远端策略 + 远端 MuJoCo

这套集成不复制 LightNav 的模型实现，也不把 MuJoCo/CasADi 装进控制台环境。边界是两个公开协议：

```text
closed_loop 控制台
  ├─ 当前干净 RGB + 指令 ──WebSocket──> LightNav policy :8050
  │                                      └─ PPU vLLM 0.18 + 服务端 SlowFast 历史
  └─ turn / forward ───────WebSocket──> LightNav MuJoCo :18088
                                         └─ Python 3.11 + MuJoCo + CasADi
```

`lightnav` driver 可以驱动控制台已有的所有 simulator；`lightnav_mujoco` backend 则让 ReAct、
手动控制和 LightNav driver 都能驱动 LightNav 自带的 ProcTHOR/MuJoCo 场景。

## 启动

推荐使用仓库的幂等外挂入口：

```bash
bash vln_demo/start_lightnav_sidecars.sh start
```

它按顺序确保 `8050`（GPU 14）、`18088`（CPU/EGL）和主控制台 `5000` 可用；已经健康的
端口原样复用。`5000` 缺失时由原有 `start_all.sh` 主配置补起，默认模型和驱动不会改成
LightNav。`stop` 只停止两个 LightNav sidecar，不会停止主控制台或 GPU 0–13 的服务：

```bash
bash vln_demo/start_lightnav_sidecars.sh status
bash vln_demo/start_lightnav_sidecars.sh stop
```

下面是对应的手工协议。

先在 LightNav 仓库启动模型服务。当前机器已验证 PPU vLLM `0.18.0+ppu2.0.0` 与
Transformers `5.5.x`，不需要为了纯 vLLM 推理升级到 0.19：

```bash
cd /opt/external/user-d5121be6/closed_loop_3dgs_rl/third_party/LightNav-0
CUDA_VISIBLE_DEVICES=14 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLN_VLLM_ENFORCE_EAGER=1 VLN_KV_CACHE_GIB=2 PYTHONPATH=src \
python3 -m lightnav.serving.ws_server \
  --model_path /opt/external/user-d5121be6/LightNav-0/model \
  --backend vllm_local --task vln \
  --gpu_memory_utilization 0.4 --max_batch_size 1 \
  --host 127.0.0.1 --port 8050
```

再在 LightNav 的独立 Python 3.11 环境启动 MuJoCo（8088 在这台机器上被占用，所以示例用
18088）：

```bash
cd /opt/external/user-d5121be6/closed_loop_3dgs_rl/third_party/LightNav-0/mujoco_demo
MUJOCO_GL=egl /root/.local/bin/uv run vln-mujoco \
  --host 127.0.0.1 --port 18088 --vln-server ws://127.0.0.1:8050
```

如需单独开一个只挂 MuJoCo、默认选 LightNav 的调试控制台，可以另用 5091；生产 5000
仍由 `start_all.sh` 管理：

```bash
cd /opt/external/user-d5121be6/closed_loop_3dgs_rl
DEMO_BACKEND=lightnav_mujoco \
DEMO_DEFAULT_DRIVER=lightnav \
LIGHTNAV_SERVER_URL=ws://127.0.0.1:8050 \
LIGHTNAV_MUJOCO_URL=http://127.0.0.1:18088 \
bash scripts/demo/run_console.sh 5091
```

`--backend all` 也会尝试挂载它；远端服务不在线时和其它可选 backend 一样打印失败原因并跳过。

## 动作语义与损失边界

模型输出每条是 `[forward_m, lateral_m(+left), yaw_rad(+ccw)]`。统一 driver 严格复现
LightNav 原生 Habitat policy 的决策口径：

1. 跳过开头 forward/yaw 都为零的点，取首个有效 waypoint；
2. 只执行一个 waypoint 就重新观察、重新推理；
3. `dt=0.1s`，前进裁到 `[0, 0.25]m`，转向裁到 `[-30°, 30°]`；
4. lateral 丢弃，因为原生 Habitat velocity controller 同样是 2-DoF unicycle；
5. 控制台没有同时速度动作，使用对称的 `半转 → 弦长前进 → 半转`，在理想 SE(2) 中与恒定
   twist 的终点完全相同。碰撞/离散执行仍可能不同，因此每步日志保留 lateral、前进限幅、
   yaw 限幅和实走距离。

MuJoCo server 本身使用直接 qpos 积分，没有对外提供导航碰撞信号；这个 backend 不伪造碰撞、
目标点或成功率。右侧地图是 8 m 范围的里程计轨迹，不是 occupancy map。

## 原生与集成实测（同场景、同指令、同为 20 秒）

指令：`Move forward, then go to the trashcan on the right.`；每次从 `(6.5, 13.8, 0)` reset。

| 链路 | 20 秒内推理/控制 | 净位移 | yaw | 解释 |
|---|---:|---:|---:|---|
| LightNav 原生 MuJoCo + 完整轨迹 MPC | 94 次预测 | 0.337 m | -13.7° | 完整 10 点轨迹，控制与推理并行 |
| 统一 `lightnav` driver + MuJoCo backend | 3 次预测 / 10 primitive 步 | 0.307 m | -37.8°（MuJoCo 系） | 首点闭环，turn/forward 阻塞分解 |

模型协议层没有格式或数值损失：控制台收到同一公开协议的原始 token、pointing 和 10×3
waypoint；MuJoCo backend 还会透传原生相机 JPEG 字节，不做二次有损编码。行为差异集中在
控制器层：完整轨迹 MPC 被首点投影取代，导致反馈频率显著降低，
20 秒净位移接近（少约 9%），但朝向相差约 24°。因此当前集成适合跨 simulator 使用同一
LightNav policy、做可审计的 adapter 对照；它不能声称与 LightNav 原生 MuJoCo MPC 行为等价。

## 测试

```bash
PYTHONPATH=vln_demo/src pytest -q \
  vln_demo/src/test_lightnav_driver.py \
  vln_demo/src/lightnav_mujoco/test_env.py
bash scripts/test/web/run.sh
```

有服务时还应做 live smoke：reset 后左转 10°、前进 0.1 m；当前实测为 10.31° / 0.105 m，
RGB 为 `270×480`，动作量在每次调用后恢复。
