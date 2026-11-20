#!/usr/bin/env bash
# 轮次级（对照） RLOO：仅去掉首次到达奖励。
# 共用 configs/gpu/rloo.yaml；这里只传相对主实验默认值需要修改的字段。
# 查看生效配置：./scripts/train/rloo/turn/run_rloo_turn_no_arrival.sh --cfg job
# 运行：./scripts/train/rloo/turn/run_rloo_turn_no_arrival.sh run.output_dir="$RL_ARTIFACT_ROOT/my-run"
# 后续 key=value 参数优先，可覆盖下面的消融设置。
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
exec "$REPO/scripts/train/rloo/run_rloo.sh" rloo.credit_level=turn rloo.first_arrival_reward=0.0 "$@"
