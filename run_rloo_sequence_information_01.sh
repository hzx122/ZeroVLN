#!/usr/bin/env bash
# 序列级（默认主方法） RLOO：信息发现奖励系数从0.05提高至0.1。
# 共用 configs/gpu/rloo.yaml；这里只传相对主实验默认值需要修改的字段。
# 查看生效配置：./run_rloo_sequence_information_01.sh --cfg job
# 运行：./run_rloo_sequence_information_01.sh run.output_dir="$RL_ARTIFACT_ROOT/my-run"
# 后续 key=value 参数优先，可覆盖下面的消融设置。
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$REPO/run_rloo.sh" rloo.information_reward_coefficient=0.1 "$@"
