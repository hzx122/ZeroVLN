#!/usr/bin/env bash
# 默认主方法：Trajectory-Level RLOO、完整奖励、联合训练；直接运行 GPU 训练器。
# --cfg job 查看YAML配置；run.mode=prepare 写计划；默认实际训练。
# 配置模型/数据/Ray/Gym地址后运行；多机拓扑保持 configs/gpu/rloo.yaml。
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$REPO/scripts/train:$REPO/scripts/eval:$REPO/verl:$REPO/vln_demo/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec "${PYTHON:-python3}" -B "$REPO/scripts/train/run_rloo.py" "$@"
