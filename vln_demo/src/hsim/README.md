# hsim — habitat_sim 资产集（PARTNR / HSSD、Habitat-HAB / ReplicaCAD）

`agentic_sim/benchmark` 下这两套资产**有场景、没有 VLN 任务**，所以它们不走
habitat-lab 的 task/dataset 那一层，直接用 habitat_sim 的 `Simulator`：
`hsim/env.py` 是共用仿真器，`hsim/datasets.py` 描述每套资产怎么读。

    ./scripts/demo/run_console.sh 5000                      # 四套资产全挂（默认）
    DEMO_BACKEND=partnr ./scripts/demo/run_console.sh 5000  # 只挂 PARTNR

页面上先选「资产集」，场景下拉只列所选集里的场景——MP3D 的 id、HSSD 的数字、
3DRS 的哈希混在一个列表里根本分不清谁是谁。

## 两套资产分别是什么

| 资产集 | 场景 | 回合 | 指令 |
| --- | --- | --- | --- |
| PARTNR / HSSD | HSSD 合成户型，val split 里 12 个 | PARTNR 官方回合，每场景默认取 3 个 | 数据集原文（英文重排任务） |
| Habitat-HAB / ReplicaCAD | hab2 基准那两个 stage | 每场景 1 个采样出生点 | 空，自己写 |

PARTNR 只暴露**在该 split 里真有回合的场景**：下载包里有 50 个 HSSD 场景，但 val
只访问其中 12 个，给另外 38 个编出生点就等于把假回合摆在真回合旁边。ReplicaCAD 那
两个 stage 自带的是 Fetch 机械臂的重排回合、不是导航回合，所以按 3DRS 的老规矩：
每个场景 1 个可复现的采样出生点，指令留空，不替资产发明任务。

出生点采样用 `(seed, scene, episode_id)` 播种，所以「回到起点」回到同一个位姿、
两次启动对「第 3 个回合是哪个」的看法一致。

## 没有路径指标

两套都没有 ground-truth 路径（PARTNR 评的是重排命题），所以
`distance_to_goal`/`success`/`spl` 一律 NaN、页面显示 `-`，driver 自动锁成 ReAct。
`get_metrics()` 仍给步数、碰撞数、位置、朝向、路径长度，俯视图由
`render_top_down()` 直接从 pathfinder 栅格化（habitat 的 `top_down_map` measure 只
存在于 habitat-lab 的 task 里）。

## 三个踩过的坑

1. **habitat_sim 的 `Simulator` 不是 habitat-lab 的 `Env.sim`**：没有
   `agents` / `get_agent_state()` / `previous_step_collided`，而碰撞是
   `Agent.act()` 的返回值。`env.py` 里的 `SimFacade` 只补这三个，其余全透传，
   这样 `agent_tools.py` 一行都不用改。
2. **HSSD 的 scene_dataset_config 在 0.2.4 上会整场景加载失败**：它把articulated
   object 挂在 `.json`（新的 `*.ao_config.json` 写法）路径键下，0.2.4 只认
   `.urdf`，拿不到 handle 就直接放弃整个场景。`datasets.py` 会在原文件旁边生成一份
   只多一个键的 `hssd-hab-partnr-h024.scene_dataset_config.json`（原文件不动，
   因为 habitat 按 config 自己所在目录解析相对路径）。个别 URDF 仍实例化失败，
   日志里那几行 `Articulated Object create failed` 是无害的，场景照常加载。
3. **两套资产都没有 navmesh 文件**：HSSD 的 scene instance 里写了
   `navmesh_instance` 但文件不在下载包里，hab2 的 stage 压根没有。
   `_ensure_navmesh()` 用 Recast 现场重建，不到一秒。

## 渲染是 CPU

跟 R2R 后端一样走 llvmpipe，`HABITAT_SIM_GPU_DEVICE_ID=-1`，不跟 vLLM 抢显存。
光照只能是 Phong：HSSD 自带的是 `pbr_ibl_configs`（PBR + IBL），0.2.4 不支持，
所以画面偏平。试过换成多光源灯组，结果比 0.2.4 的默认光更差（边缘细节 4.27 →
最好 2.58，且三成画面压黑），所以保留默认。
