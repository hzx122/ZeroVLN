#!/usr/bin/env bash
# Complete inlineable DLC entry. Rank is consumed by the Python coordinator
# before child model/Gym processes clear distributed-training environment.
set -euo pipefail
REPO=${ACTION_HEAD_SUITE_REPO:-/workspace/ZeroVLN}
: "${ACTION_HEAD_SUITE_PLAN:?Set the prepared immutable plan.json}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export ACTION_HEAD_SUITE_RUNTIME_BASE=${ACTION_HEAD_SUITE_RUNTIME_BASE:-/tmp/ahs}
case "$ACTION_HEAD_SUITE_RUNTIME_BASE" in
  /workspace/.runtime/*) unset RULE_COLLECTION_RUNTIME_ROOT ;;
  *) export RULE_COLLECTION_RUNTIME_ROOT=${RULE_COLLECTION_RUNTIME_ROOT:-/tmp/ahs} ;;
esac
mkdir -p "$ACTION_HEAD_SUITE_RUNTIME_BASE"
exec /usr/local/bin/python3.12 -B "$REPO/scripts/eval/action_head_benchmark_suite.py" run \
  --plan "$ACTION_HEAD_SUITE_PLAN"
