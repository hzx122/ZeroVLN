# ObjectNav: mesh + HabitatGS

独立于 R2R/RxR 固定评估集。需要另行提供原始数据清单与渲染资产。

## 数据准备

```bash
PYTHONPATH=vln_demo/src python -m objectnav.catalog --output /CPFS/run/catalog.jsonl
PYTHONPATH=vln_demo/src python -m objectnav.materialize --manifest /CPFS/run/catalog.jsonl --output /CPFS/run/catalog-cpfs.jsonl --assets /CPFS/run/assets
```

必须检查同名 `.report.json` 中的拒绝记录，不能把 tracked 当作人工质检通过。
唯一身份为 category + scene + episode_id。HM3D/MP3D 的 LeRobot 数值 ID 不能直接当原始 ObjectNav ID；通过场景、初始位姿、朝向、指令回查源目标，歧义记录拒绝导入。
GS/Insight 的 frame.yaw 是前向向量 bearing，比 quaternion 的 Y 轴角多 pi；已用原始 RGB-D 对齐。
WAMDP observation.state 是 FLU，但 camera extrinsics 是 Habitat/OpenGL camera-to-world。
MP3D 使用原始 prebaked NavMesh；HM3D Train 使用原始 h1.50/r0.10 缓存；不擅自重新生成碰撞几何。

`materialize` 将 JFS 渲染资产复制到 CPFS，逐文件校验；不写源资产。生成清单保存原路径及内容哈希。

## 服务和本地调试

`start.sh` 默认 GPU15、7041、最多8个场景/相机缓存，取代 DLC 默认 27B。
使用兼容的 HabitatGS 构建 和独立 NumPy 2.2.6 依赖路径 `OBJECTNAV_PYTHON_DEPS`；不得降级宿主 NumPy。
GL 在单个专属线程，请求指定完整位姿，导航状态在每个 CPU 回合内。队列最多32，满返回503。
GS 使用 GPU CUDA rasterization + PPU host-copy；标准 mesh 使用 CPU llvmpipe。

```bash
OBJECTNAV_GPU=1 OBJECTNAV_PORT=17041 OBJECTNAV_MANIFEST=/CPFS/run/catalog-cpfs.jsonl bash scripts/demo/objectnav/start.sh
# 用现有 Habitat console Python（其 lib 在 LD_LIBRARY_PATH）：
PYTHONPATH=vln_demo/src python -m objectnav.probe --manifest /CPFS/run/pilot.jsonl --url http://127.0.0.1:17041 --output /CPFS/run/probe
```

唯一 DLC 入口仍为 `vln_demo/dlc/submit.py --objectnav-manifest ... --submit`。
旧服务的工作树不修改、不切分支；新服务使用独立代码目录、运行目录和手动 workspace。

## 控制台与批跑

控制台 `DEMO_BACKEND=objectnav`；指定 `OBJECTNAV_MANIFEST` 和 `OBJECTNAV_RENDER_URL`。
`all` 仅在提供清单时注册 ObjectNav，不影响普通 R2R/RxR。
批跑使用 `bash vln_demo/evaluate.sh --datasets objectnav --manifest <外部清单> ... --submit`，在同一个 DLC Pod 内完成隔离评估并自动退出。
CPU worker 自动连接指定新服务7041和9038；32回合、100次工具预算和隔离归档沿用原逻辑。

评分为源数据所选物体的可达 goal viewpoint / standoff 的0.5米测地距离，且必须STOP。
这是 selected-goal 诊断指标，不是官方 category-wide ObjectNav SR。指标中保留 scoring 标签。
源目标、SAM mask、未来轨迹不会作为图像输入；model map 只有可走区域、已走轨迹和当前位姿。
probe 在目标处设置位姿仅用于验证评分边界，不能当作模型导航成功率。

### 分类入口

默认服务按原始分类展示 `gary-hm3d-objectnav-train`、
`gary-mp3d-objectnav-train`、`gary-hm3d-insight`、
`gary-habitatgs-insight`、`gary-habitatgs-longrange`。各入口只包含对应
分类的场景与回合，保留原始指令、episode ID 和评分规则，共用 ObjectNav
渲染服务。内部 `objectnav` 入口仍供显式指定回合的批跑使用，不显示在下拉框。

匿名包不含默认清单；可通过环境变量或提交参数提供外部清单。
自行设置 `vln_demo/config/objectnav.json` 时，相对路径从提交仓库根目录解析。
新服务验收逐类检查数量、精确回合归属，并各自验证加载、图像和转向。
