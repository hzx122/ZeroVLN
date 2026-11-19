# Lepu 办公层资产

从 `lepu` 分支 `f92cf9f` 提取资产语义和平面图。仿真直接复用当前
`mall.env.MallEnv` / `MallSim`，不导入该分支的录制、环视图、移动加速或模型配置。

- 场景：`jW00XG5o3g`，7381 个起点、19 个房间与设施目标。
- 导航资产：`/opt/external/user-9656ce61/grpo/locnav_grpo/exp0822/assets_arr5/lepu`。
  使用 `walk.json`、`floor_meta.json` 和 `lepu_walk_union_g45.npy`；原始资产保持共享盘引用。
- 高斯资产：`/opt/external/user-9656ce61/ppu_sim/3drs_gs_root/jW00XG5o3g/gs/block0/point_cloud.ply`。
  与 `mall_floor2` 共用当前 DLC 的 **7039 / GPU 3**，无需新增 GPU 或渲染进程。
- 相机与运动学沿用 Mall：1280×720、90°、高度 1.5 m、前进 0.25 m、转向 15°；
  前进中点和终点都可走才执行。到目标 2 m 内 STOP 判成功。
- 会议室、安全出口、电梯等都是有效目标，仅过滤显式 `is_infra` 标记；不能套用商场的店铺名过滤。
- 没有 `walk_grpo.json` 拓扑边；视觉点导航沿可走域执行 A*，SPL 的专家距离仍为欧氏估计。

控制台使用 `DEMO_BACKEND=all` 或 `DEMO_BACKEND=mall,lepu`；Gym 支持 `lepu`。
可用 `LEPU_ASSETS_DIR` / `LEPU_RENDER_URL` / `LEPU_EPISODES` 调整路径、服务地址和起点数量。
默认渲染地址跟随 `MALL_RENDER_URL`，未设置时为 `http://127.0.0.1:7039`。
关闭环境不会卸载共享渲染服务中的场景。
