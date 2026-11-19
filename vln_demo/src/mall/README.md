# mall_floor2 后端（万科里商场 3DGS，local-user-06 资产）

给 closed-loop bench / console / DLC 链路新增的一个真实可评分环境：
1225 个真实起点 × 45 个商铺目标（「前往OCHIRLY」），渲染走 3DGS render
server（COLMAP `/render_gs`，degree-0 ply），运动学走**膨胀后的可走域
mask**，与 zekang 闭环训练侧 `mt_move_server_strict.py` 逐字同语义。

## 口径（与训练一致，勿单独改）

| 项 | 值 |
|---|---|
| 相机 | 1280×720，hfov 90°（fx=640），**绝对高度 1.5 m** |
| yaw | 0°=−Z，逆时针为正（habitat 系，场景原生，无坐标转换） |
| 前进 | 0.25 m；**中点+落点都在 mask 上才走**，否则整个动作拒绝（无滑动） |
| 转向 | 15°（08-24 起现行标准；历史 30°） |
| 抵达 | 到店门口 ≤2 m 且 STOP，POIBench 同款 |
| SPL | walk_grpo.json 2851 节点拓扑图 Dijkstra（端点按**坐标**最近 snap；walk.json 的 front_node 索引的是别的节点表，不能用），缺图退回欧氏 |

## 资产

`MALL_ASSETS_DIR`（默认
`/opt/external/user-9656ce61/grpo/locnav_grpo/exp0822/assets_arr5/mall`）：

- `walk.json` — 1225 starts `{idx,x,z,yaw_deg}` + 45 shops `{name,front_world}`
- `floor_meta.json` — `{px_per_m:12, x0:7.035261, y1:-6.400001, W:1136, H:1670}`
- `mall_walk_union_topo.npy` — **膨胀后的可走域**（= topo_poi 连线栅格化+8px 膨胀
  ∪ navmesh floor mask，+15.5% 可走像素）。★本后端碰撞判据必须用它，不是原
  navmesh（原 navmesh 走廊太窄，点可见地板常被拒）。
- `walk_grpo.json`（可选）— 拓扑图，SPL 专家距离用。

世界↔像素：`p=(x−x0)·ppm`（列）、`q=(z+y1)·ppm`（行），`mask[q,p]`。
世界→像素一律 `int(round())`，与训练侧 `gen_union_mask_mall.py` /
`gen_poi_trajs_ppu.py` 逐字一致（勿改 `floor`）。

另：仓库内自带平面图 `floorplan.png`（1136×1670，与 `floor_meta` 标定同源，
即 `floorplan_mall_thr_2_2_dedup.png`），供俯视/可视化参考；碰撞判据只认
`MALL_ASSETS_DIR` 里的膨胀 mask，不读这张图。

目标商铺 = 45 店里剔除设施（安全出口/电梯/扶梯/楼梯/服务台/导览屏/休息区，
共 9 个），剩 36 家；每个起点按 `(seed, start_idx)` 确定性抽一家，
回合表稳定可复现。指令 = `前往<店名>`。

## 渲染服务

与 3DRS 同协议同公式（`rs3d.gs_client` 直接复用）：

```
cd /workspace/agentic_sim && \
venv_render/bin/python ABot-Navigation/render_sim_dynamic.py \
  --scenes_root /opt/external/user-9656ce61/ppu_sim/3drs_gs_root \
  --port 7039 --gpus 3 --sh_degree 0 --max_scenes_per_gpu 2
```

场景布局 `<scenes_root>/mall_floor2/gs/block0/point_cloud.ply`（6.4G，
degree-0，与 3DRS 同款服务进程，但 scenes_root 不同，故独立进程独立卡）。
`MALL_RENDER_URL` 覆盖地址（默认 7039）。

> 默认由 `vln_demo/start_all.sh` 作为第 4 个渲染服务拉起（PPU3/7039，与 3DRS/
> ABotN/SAGE 三个一起"只补不杀"管理）。卡位连续约定见 `start_all.sh` 头注释：
> 渲染占 PPU0-3，sglang×8 占 4-11，vllm 35B×2 占 12-13。

★ bench 的 `close()` 默认**不**向渲染服务发 `/unload`：该 GS 进程常与训练
侧 move server 共享，unload 会强制对方 ~40s 冷重载。要开：`MALL_UNLOAD_ON_CLOSE=1`。

## 使用

- console：`DEMO_BACKEND=mall bash run_demo_ext.sh`（或 `--backend mall`）
- gym bench：`DEMO_BACKEND=mall [MALL_RENDER_URL=...] bash vln_gym_server.sh`
- DLC：`python3 verl/dlc/submit_bench_3dgs_dlc.py --backend mall ...`
  （`verl/dlc/backends.py` 里一条配置自动接 entry/submit）

指标系：`get_metrics()` 的 `position=[x, −z, 1.5]`、`heading_deg=habitat_yaw+90°`
是 FM chunk 系约定（`forward=(cos h, sin h)`，同 abotn 的两系边界设计）；
obs 的 gps/compass 仍是 habitat 形（经 `get_agent_state()`）。
