#!/usr/bin/env python3
"""统一 3DGS 数据源配置（单一事实源）。

每个后端（rs3d/abotn/sage/hsim）一条配置项，集中管理 3DGS render server 的**加载参数**
+ bench 侧后端契约：scenes_root / sh_degree / render 端口段 / render-url env 变量名 /
max_scenes_per_gpu / DEMO_BACKEND / split 选择器(env+取值) / eps 限量 env / 相机
(height,hfov) / DLC 布局(render 端口、gpu 组、bench:render 比、preload)。

加新后端 = 在 ``BACKENDS`` 加一条，``bench_3dgs_entry.sh`` / ``submit_bench_3dgs_dlc.py``
/ ``run_local_<backend>.sh`` / SFT builder 全从这读，**不再各处加 case 分支**。

bash 消费（entry / run_local）：
  eval "$(python3 verl/dlc/backends.py <backend> --emit-sh)"   # 注入 SCENES_ROOT/SH_DEG/...
python 消费（SFT builder / submit）：
  from backends import BACKENDS, fields
  cfg = fields("sage")   # dict

emit-sh 输出变量：SCENES_ROOT / SH_DEG / MAX_SPG / RENDER_URL_ENV / DEMO_BACKEND /
BENCH_PER_RENDER / PRELOAD_SCENES / RENDER_PORTS(空格分隔) / GPU_GROUPS(空格分隔) /
BENCH_EXTRA(空格分隔的 K=V)。其中 BENCH_EXTRA 按**当前 env**（SAGE_SPLIT/ABOTN_SET/
SAGE_EPS_PER_SCENE）填，故 entry 须在 submit 脚本 export 这些 env 之后再 eval。
"""
from __future__ import annotations

import os
import shlex
import sys

# 后端 → 加载参数。scenes_root 全绝对路径（abotn/3drs 是 local-user-03 盘的 symlink，sage 在
# AutoVLA 授权盘 /data/storage 下，符合 CLAUDE.md 写入限制）。
BACKENDS = {
    "rs3d": {
        "scenes_root": "/workspace/agentic_sim/3drs_gs_root",
        "sh_degree": 0,
        "max_scenes_per_gpu": 2,
        "render_url_env": "RS3D_RENDER_URL",
        "demo_backend": "rs3d",
        "split_env": None, "split_default": None, "split_choices": None,
        "eps_env": None, "eps_default": 0,
        "camera_height": None, "hfov": None,     # rs3d free-roam 无 SFT，相机参数不用
        "dlc_render_ports": [7036, 7038, 7040, 7042],
        "dlc_gpu_groups": ["0,1,2,3", "4,5,6,7", "8,9,10,11", "12,13,14,15"],
        "dlc_bench_per_render": 4,
        "preload_sh": "",                        # 不预加载（free-roam，无 goal 无 reward，验链路用）
    },
    "abotn": {
        "scenes_root": "/workspace/agentic_sim/abotn_gs_root",
        "sh_degree": 3,
        "max_scenes_per_gpu": 11,
        "render_url_env": "ABOTN_RENDER_URL",
        "demo_backend": "abotn",
        "split_env": "ABOTN_SET", "split_default": "poi", "split_choices": ["poi", "in", "out"],
        "eps_env": None, "eps_default": 0,
        "camera_height": 0.65, "hfov": 120.0,
        "dlc_render_ports": list(range(7037, 7053)),      # 16 端口
        "dlc_gpu_groups": [str(i) for i in range(16)],     # 16 render × 1 GPU（★单卡/server 避 CUDA illegal access）
        "dlc_bench_per_render": 1,
        # 预加载该 set 全部场景（消除冷启动惊群）；entry 在 eval 后 SCENES_ROOT/ABOTN_SET 已就绪。
        "preload_sh": '"$(ls "$SCENES_ROOT" 2>/dev/null | grep "^${ABOTN_SET}-" | tr \'\\n\' \',\' | sed \'s/,$//\')"',
    },
    "sage": {
        "scenes_root": "/data/zerovln/SAGE-3D/sage_gs_root",
        "sh_degree": 3,
        "max_scenes_per_gpu": 5,                  # sage ply ~230MB/场
        "render_url_env": "SAGE_RENDER_URL",
        "demo_backend": "sage",
        "split_env": "SAGE_SPLIT", "split_default": "instr",
        "split_choices": [
            "train", "instr", "scene", "traj",
            "action_head_train_p0_unified",
            "official_gvln_action_head_p0_v3full827train",
        ],
        "eps_env": "SAGE_EPS_PER_SCENE", "eps_default": 0,   # train(929 场景)必设 >0 省 RAM
        "camera_height": 1.2, "hfov": 105.3,     # 对齐 sage/env.py: CAMERA_HEIGHT=1.2 / SAGE_HFOV=105.3
        "dlc_render_ports": list(range(7060, 7076)),       # 16 端口（与 abotn 7037 段错开，DLC pod 隔离本可重用但留区分）
        "dlc_gpu_groups": [str(i) for i in range(16)],      # 16 render × 1 GPU（sh3 同 abotn，单卡/server）
        "dlc_bench_per_render": 1,
        "preload_sh": "",                        # 944 场景太多不预加载，lazy load
    },
    "mall": {
        # local-user-06 自采万科里商场（mall_floor2）：1225 真实起点 × 商铺目标，
        # 膨胀 walk mask 严格运动学（见 vln_demo/src/mall/）。ply 是 degree 0（与
        # 3DRS 同系，但 scenes_root 在 zekang 盘 → 独立服务进程）。资产（walk.json/
        # floor_meta/膨胀 mask/拓扑图）由 bench 侧 MALL_ASSETS_DIR 定位，不经本配置。
        "scenes_root": "/opt/external/user-9656ce61/ppu_sim/3drs_gs_root",
        "sh_degree": 0,
        "max_scenes_per_gpu": 1,                 # 单场景 6.4G ply，约 8-10G 显存
        "render_url_env": "MALL_RENDER_URL",
        "demo_backend": "mall",
        "split_env": None, "split_default": None, "split_choices": None,
        "eps_env": None, "eps_default": 0,
        "camera_height": 1.5, "hfov": 90.0,      # mall/env.py: CAMERA_HEIGHT=1.5 / CAMERA_HFOV=90（1280×720, fx=640）
        "dlc_render_ports": [7053, 7054, 7055, 7056],      # 单场景 4 个 render 足够（避开 abotn 7037-7052 / sage 7060-7075 段）
        "dlc_gpu_groups": ["0", "1", "2", "3"],            # 单卡/server（避多卡切场景 CUDA illegal access 一类问题）
        "dlc_bench_per_render": 4,               # 16 bench : 4 render（同 rs3d 布局）
        "preload_sh": '"mall_floor2"',           # 唯一场景启动即预加载（冷加载 6.4G ~40s，避免惊群）
    },
}


def fields(backend: str) -> dict:
    """python 消费：返回某后端配置 dict（深拷贝免被改）。"""
    if backend not in BACKENDS:
        raise SystemExit(f"backends: unknown backend {backend!r} (known: {list(BACKENDS)})")
    return dict(BACKENDS[backend])


def emit_sh(backend: str) -> str:
    """bash 消费：emit 一段 shell 赋值，供 entry/run_local ``eval`` 注入变量。

    BENCH_EXTRA 按当前 env 填（SAGE_SPLIT/ABOTN_SET/SAGE_EPS_PER_SCENE），故调用方须先
    export 这些 env（submit 脚本会 export）。
    """
    c = fields(backend)
    lines = [
        f"SCENES_ROOT={shlex.quote(c['scenes_root'])}",
        f"SH_DEG={c['sh_degree']}",
        f"MAX_SPG={c['max_scenes_per_gpu']}",
        f"RENDER_URL_ENV={c['render_url_env']}",
        f"DEMO_BACKEND={c['demo_backend']}",
        f"BENCH_PER_RENDER={c['dlc_bench_per_render']}",
        f"RENDER_PORTS=({' '.join(str(p) for p in c['dlc_render_ports'])})",
        f"GPU_GROUPS=({' '.join(shlex.quote(g) for g in c['dlc_gpu_groups'])})",
        f"PRELOAD_SCENES={c['preload_sh']}",
    ]
    extra = []
    if c["split_env"]:
        extra.append(f"{c['split_env']}={shlex.quote(os.environ.get(c['split_env'], c['split_default']))}")
    if c["eps_env"]:
        extra.append(f"{c['eps_env']}={os.environ.get(c['eps_env'], str(c['eps_default']))}")
    lines.append(f"BENCH_EXTRA=({' '.join(shlex.quote(e) for e in extra)})")
    return "\n".join(lines) + "\n"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=list(BACKENDS))
    ap.add_argument("--emit-sh", action="store_true", help="输出 shell 赋值段供 eval")
    ap.add_argument("--field", help="只输出某字段值（python 消费单值，如 camera_height）")
    args = ap.parse_args()
    if args.field:
        print(fields(args.backend)[args.field])
    elif args.emit_sh:
        sys.stdout.write(emit_sh(args.backend))
    else:
        print(args.backend)


if __name__ == "__main__":
    main()
