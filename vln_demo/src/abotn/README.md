# abotn — ABotN 三套高斯资产（POIBench、PointBench 室内/室外）

`zytoss/` 下的 ABotN benchmark：**高斯场景 + 真值轨迹**。视觉跟 3DRS 一样走 3DGS
渲染服务，但走法不一样——这些场景没有 navmesh，自带的是一张占据栅格图，所以
`abotn/occmap.py` 自己做运动学，`abotn/env.py` 把它包成 `habitat.Env` 的形状。

    ./scripts/demo/run_console.sh 5000                   # 七套资产全挂（默认）
    DEMO_BACKEND=poi ./scripts/demo/run_console.sh 5000  # 只挂 POIBench

页面上先选「资产集」再选场景。

## 三套分别是什么

| 资产集 | 场景 | 回合 | 指令 | 到达阈值 |
| --- | --- | --- | --- | --- |
| POI 导航 / ABotN-POIBench | 11 个真实街区 | 163 个官方回合 | 数据集原文中文指令（「前往111医药馆」） | 2.0 m（SPL 另减 2 m） |
| 室内点导航 / ABotN-PointBench | 16 个户型，38–100 ㎡ | 240 个 | 无，我们按坐标+直线距离生成一句 | 0.5 m |
| 室外点导航 / ABotN-PointBench | 15 个街区/公园/路口 | 225 个 | 同上 | 0.5 m |

**不合成一套「ABotN」**：POI 任务是一句话对一家店，point-goal 任务压根没有指令、
只有一个坐标，室内室外的尺度又差一个数量级——混进一个下拉框，正好把你挑场景时唯一
在意的区别藏掉。

**场景 id 带前缀**（`poi-` / `in-` / `out-`）。有四个场景目录同时出现在 POIBench 和
PointBench/Outdoor 里（`20260212160022`、`20260212163520`、`20260227165628`、
`20260227170203`），控制台按场景 id 索引「这个场景属于哪套资产」，裸 id 会让这四个
撞车，渲染服务也会把一个场景当成两套任务来伺候。

## 指标是真的

这三套都有真值轨迹，所以 `distance_to_goal` / `success` / `spl` 是真数字，不是 NaN，
driver 也不会被锁成 ReAct。判定规则按 ABotN 自己的：

* `distance_to_goal`：到 `label.extend.end_point` 的二维欧氏距离。
* 到达阈值：POI 2.0 m（店铺标在门脸上，所以给 2 m），point-goal 0.5 m。
* `SPL = success × 最短 / max(实走, 最短)`，最短取标注折线的长度（POIBench 再减掉
  它自己那 2 m 到达余量，跟它的 evaluator 一致）。`actual_len` 字段只精确到 0.1 m，
  所以宁可自己量折线，只在量不出来时才回退到它。

控制台不会把 STOP 真的 step 进仿真器（见 `vln_model.py` 模块注释），所以它拿距离
自己重算判定，用的半径就是上面这张表——R2R 那个 3 m 只属于 R2R。

## 两个坐标系

ABotN 的世界系是 **z 朝上**，x/y 水平，yaw 从 +x 转向 +y，相机在 `地面 z + 0.65`。
`agent_tools.py` 读的是 habitat 的形状：y 朝上、前方 −z。所以 `env.py` 用

    (x, y, z) -> (x, z, -y)      psi = yaw - pi/2

在边界上转一下。这是个刚体变换，`agent_tools` 量的距离和相对朝向一个都不变。渲染走
`/render_gs_xyz_euler` 而不是 `/render_gs`——前者收的就是 ABotN 这套原生位姿，
后者要 COLMAP 的 y-up 约定（那是 `rs3d` 用的）。

朝向约定是量过的：运动方向与标注 yaw 的偏差中位数 −0.2°（POI，n=1636）、0.0°
（Outdoor，n=5168），100% 落在 30° 以内。

## 必须是第二个渲染服务

`--sh_degree` 在渲染服务里是**按进程**定的，而 `load_gs_ply` 会断言球谐系数的个数正好
等于 `3*(degree+1)^2-3`。ABotN 的 ply 是 degree 3（45 个系数），3DRS 的是 degree 0，
一个进程装不下两套。所以先把场景软链成一个 `scenes_root`：

    PYTHONPATH=src python3 -m abotn.make_scenes_root     # 42 个软链，不拷贝，35 GB 留在原地

再单独起一个：

    cd /workspace/agentic_sim && \
      venv_render/bin/python repo/ABot-Navigation/render_server/render_sim_dynamic.py \
      --scenes_root abotn_gs_root --port 7037 --gpus 1 --sh_degree 3

7036 那个（3DRS，degree 0）继续跑，两个互不干扰。渲染在 GPU 上；控制台不负责起它，
它把高斯常驻显存，本来就该比控制台活得久。

## 走路是占据栅格，不是 navmesh

每个场景 `map/` 下三个文件：`occ_map.png`（≥128 是空地）、`occ_map_height.tiff`
（逐像素地面高度，米）、`occ_map_meta.txt`（四角世界坐标 + 图像尺寸）。分辨率一律
0.05 m/px。`occmap.py` 抄的是 ABotN 的 `convert_actual_to_pixel`，x/y 的方向由四角
顺序推出来（这 42 个场景都是 sx=−1、sy=+1）。

前进用「整段扫过去，取最远那个还空着的点」；撞墙时把两个轴向分量分别再试一次、取走得
更远的那个——这就是 habitat 沿墙滑行在栅格上的等价物。机器人半径按 0.10 m 的方形足迹。

标注本身有约 1–3% 的点落在 0.15 m 足迹的障碍里，所以 `reset()` 会把起点吸附到最近的
合法像素（`_snap`，上限 1.5 m）。实测 628 个回合的标注起点一个都没被挪动。

## 那个高度图的两个坑

**坑一：5 m 空洞守卫不能抬。** ABotN 自己的 `_adjust_pose_z` 里有一句
`if ground_z > 5: ground_z = 0.0`——把高度图里超过 5 m 的值当成空洞。POIBench 有两个
场景（麦当劳、太田）的高度图里确实有 6–7 m 和 19–21 m 的成片高值，看着像高架连廊，
于是这里曾经把 `hole_above` 按场景抬到「标注轨迹最高点 + 1 m」去迁就它。

**那是错的。** 拿高斯点云 `gs/*/point_cloud_rotated.ply` 对了一遍：这两个场景的地板
和别处一样在 z ≈ −0.4，那几片高值是楼体和噪声，不是可走地面。抬完的结果是相机被架到
7 m / 22 m 高空俯视屋顶。所以现在就用 ABotN 的 5.0，按点云地板核对全部 26 个
POI/室外场景，每个都落在自己地板上方 0.7–2.2 m。

顺带说明为什么「标注 z 恰好等于高度图读数 + 0.65」不能当成证据：628 个 task 里 610 个
正好是 0.65，剩下 18 个正是麦当劳(10) + 太田(8)——标注 z 本身就是 `_adjust_pose_z`
算出来的，不是人开出来的，所以它和高度图不是两个独立的信息源。

**坑二：`occ_map_height.tiff` 是 8 位的。** 它是 `mode="L"` 的 uint8，`occ_map_meta.txt`
里没有任何 scale/offset，也就是说 ABotN 的地面**量化到整米**，相机跟着有最多半米误差。
多数场景地板本来就在 0 附近，读数 0、眼高 0.65，看不出来；地板刚好压在量化台阶上就很
明显——飘宿 ep0 的真实地面在 +0.68 而高度图说 0，相机落在地板**下面** 0.03 m，一进去
整个画面下半部是路面内部的黑糊。

修法在 [`floorfit.py`](floorfit.py)：从高斯点云量出每 1 m 格子的地板，只保留它与高度图
的**差值**，限幅 ±1.5 m、补洞、平滑，缓存成几十 kB 的 npz，运行时加到 `ground_z` 上。
两个关键的保守选择：

- **只抬不压。** 点云的地板估计本身偏低（摄影测量在真实表面下留一层「裙边」溅射），
  带符号地用会把相机压穿地面——用渲染服务自己的深度图反投影独立量过：太田的出生点
  在原始高度图下离可见路面 0.70 m，带符号修正后只剩 0.25 m，等于把要修的毛病自己造了
  一遍。只抬的话太田保持 0.70，飘宿从 −0.03 抬到 +0.67，正好是标称的 0.65。
- **限幅 1.5 m。** 三个 PointBench 场景的点云只覆盖了五分之一的可走区域，只拍到雨棚或
  树冠的格子「地板」能差出好几米。超出限幅就退回 ABotN 自己的读数。

缓存目录默认 `$ABOTN_ROOT/.floor_cache`（42 个场景共 2.1 MB），冷缓存时首次进场景现算
1–20 s；`python -m abotn.floorfit` 可以一次性预热。`ABOTN_FLOOR_FIT=0` 关掉修正，回到
ABotN 的原始行为。

另外 `ground_z` 的像素换算按 ABotN 的 `get_z_value`，用**高度图自己的**尺寸而不是 meta
里的 `IMAGE_WIDTH`，因为有一个场景的 tiff 比 png 窄一个像素。

## 只用 numpy + PIL

跟 `rs3d` 一样：控制台跑在 gordon 那个 habitat venv 里，没有 cv2。ABotN 原版的
`dilate_free_space` 用的是 `cv2.dilate`，这里没用到——我们不膨胀空地，直接按足迹判。
