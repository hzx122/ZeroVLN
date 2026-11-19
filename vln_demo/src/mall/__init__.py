# -*- coding: utf-8 -*-
"""mall_floor2 backend — 万科里商场自采 3DGS 场景（local-user-06 资产）。

1225 个真实起点 × 商铺目标（「前往OCHIRLY」），膨胀 walk mask 严格运动学
（0.25 m 步进 / 15° 转向 / 撞墙整体拒绝，无滑动），相机 1280×720、hfov 90°、
绝对高度 1.5 m，渲染与 3DRS 同款 COLMAP /render_gs 协议（degree-0 ply）。

资产与约定见 ``mall/README.md``；运动学权威实现来自
``mt_move_server_strict.py``（zekang 闭环训练用的 /move 服务）。
"""
