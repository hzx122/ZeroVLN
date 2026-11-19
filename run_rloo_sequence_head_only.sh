#!/usr/bin/env bash
# 序列级（默认主方法） RLOO：只训query和FM动作专家，冻结大脑优化器/调度器。
# 共用 configs/gpu/rloo.yaml；这里只传相对主实验默认值需要修改的字段。
# 查看生效配置：./run_rloo_sequence_head_only.sh --cfg job
# 运行：./run_rloo_sequence_head_only.sh run.output_dir="$RL_ARTIFACT_ROOT/my-run"
# 后续 key=value 参数优先，可覆盖下面的消融设置。
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$REPO/run_rloo.sh" rloo.train_modules=head "$@"
