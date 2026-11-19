#!/usr/bin/env bash
# Run the shared Qwen3.6/FM-head SFT entry on one local node.
set -euo pipefail

ACTION_HEAD_AS_TOOL_VERSION=v1
while (($#)); do
  case "$1" in
    --action-head-as-tool-version) ACTION_HEAD_AS_TOOL_VERSION=${2:?}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$ACTION_HEAD_AS_TOOL_VERSION" == v1 || "$ACTION_HEAD_AS_TOOL_VERSION" == v2 || "$ACTION_HEAD_AS_TOOL_VERSION" == v3 || "$ACTION_HEAD_AS_TOOL_VERSION" == v4.3 || "$ACTION_HEAD_AS_TOOL_VERSION" == v6 ]] || exit 2

REPO=${REPO:-/workspace/ZeroVLN}
ENTRY=${ENTRY:-$REPO/verl/scripts/train/action_head_dualvln_sft/entry.sh}
LOCAL_PYTHON=${LOCAL_PYTHON:-/usr/local/bin/python3.12}
RUN_NAME=${RUN_NAME:-action-head-fm-formal-p0-local-$(date +%Y%m%d-%H%M%S)}
NNODES=1
NODE_RANK=0
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29518}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
GATE=${GATE:-train}
MAX_STEPS=${MAX_STEPS:-200}
LEARNING_RATE=${LEARNING_RATE:-3e-5}
MANIFEST=${MANIFEST:-/data/zerovln/action_head/sft_data/action-head-sft-data-p0-full-randomfast-20260827-r7/manifest.jsonl}
SLOW_MODEL=${SLOW_MODEL:-/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B}
HEAD_CHECKPOINT=${HEAD_CHECKPOINT:-/data/zerovln/action_head/integration/action_head_wan22_init_v1}
QUERY_CHECKPOINT=${QUERY_CHECKPOINT:-/data/zerovln/action_head/integration/qwen36_four_query_5001_migration_20260824/query}
OUTPUT_DIR=${OUTPUT_DIR:-/data/zerovln/action_head/sft_ckpt/$RUN_NAME}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

for path in "$ENTRY" "$LOCAL_PYTHON" "$MANIFEST" "$SLOW_MODEL" "$HEAD_CHECKPOINT" "$QUERY_CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "required path does not exist: $path" >&2
    exit 2
  fi
done
VISIBLE_COUNT=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
if [[ "$NPROC_PER_NODE" -le 0 || "$NPROC_PER_NODE" -gt "$VISIBLE_COUNT" ]]; then
  echo "NPROC_PER_NODE=$NPROC_PER_NODE exceeds visible device count $VISIBLE_COUNT" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" && "${ALLOW_EXISTING_OUTPUT:-0}" != "1" ]]; then
  echo "refusing existing OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 2
fi

# DSW does not expose the PPU runtime libraries to the base Python by default.
export PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
export PPU_HOME=${PPU_HOME:-$PPU_SDK}
export CUDA_PATH=${CUDA_PATH:-$PPU_SDK/CUDA_SDK}
export CUDA_HOME=${CUDA_HOME:-$CUDA_PATH}
export CUDA_TOOLKIT_ROOT=${CUDA_TOOLKIT_ROOT:-$CUDA_PATH}
export PATH="$CUDA_PATH/bin:$PPU_SDK/asight/bin:$PPU_SDK/bin:$PPU_SDK/ppu-smi/bin:$PATH"
export LIBRARY_PATH="$CUDA_PATH/lib64:$PPU_SDK/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES
LOCAL_RUNTIME_ROOT=${LOCAL_RUNTIME_ROOT:-$REPO/.local_runtime/$RUN_NAME}
export TMPDIR=${TMPDIR:-$LOCAL_RUNTIME_ROOT/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$LOCAL_RUNTIME_ROOT/xdg}
export HF_HOME=${HF_HOME:-$LOCAL_RUNTIME_ROOT/huggingface}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$LOCAL_RUNTIME_ROOT/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$LOCAL_RUNTIME_ROOT/triton}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$LOCAL_RUNTIME_ROOT/cuda}
export PYTORCH_KERNEL_CACHE_PATH=${PYTORCH_KERNEL_CACHE_PATH:-$LOCAL_RUNTIME_ROOT/kernels}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$LOCAL_RUNTIME_ROOT/extensions}
export RTC_CACHE_PATH=${RTC_CACHE_PATH:-$LOCAL_RUNTIME_ROOT/rtc}
export PPU_LIB_RTC_CACHE_PATH=${PPU_LIB_RTC_CACHE_PATH:-$LOCAL_RUNTIME_ROOT/ppu-rtc}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$LOCAL_RUNTIME_ROOT/numba}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$LOCAL_RUNTIME_ROOT/matplotlib}
export DG_CACHE_DIR=${DG_CACHE_DIR:-$LOCAL_RUNTIME_ROOT/deep-gemm}
export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-$LOCAL_RUNTIME_ROOT/deep-gemm-jit}
export ACTION_HEAD_LOCAL_RANK_CACHE_ROOT=${ACTION_HEAD_LOCAL_RANK_CACHE_ROOT:-$LOCAL_RUNTIME_ROOT/ranks}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$PYTORCH_KERNEL_CACHE_PATH" \
  "$TORCH_EXTENSIONS_DIR" "$ACTION_HEAD_LOCAL_RANK_CACHE_ROOT" \
  "$RTC_CACHE_PATH" "$PPU_LIB_RTC_CACHE_PATH" "$NUMBA_CACHE_DIR" \
  "$MPLCONFIGDIR" "$DG_CACHE_DIR" "$DG_JIT_CACHE_DIR"

export NNODES NODE_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE GATE MAX_STEPS
export LEARNING_RATE MANIFEST SLOW_MODEL HEAD_CHECKPOINT QUERY_CHECKPOINT OUTPUT_DIR
export PYTHON="$LOCAL_PYTHON"
if [[ "${CHECK_ENV_ONLY:-0}" == "1" ]]; then
  "$LOCAL_PYTHON" -c 'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}))'
  exit 0
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'local Action Head SFT dry-run:\n'
  printf '  RUN_NAME=%s\n  NPROC_PER_NODE=%s\n  CUDA_VISIBLE_DEVICES=%s\n' \
    "$RUN_NAME" "$NPROC_PER_NODE" "$CUDA_VISIBLE_DEVICES"
  printf '  MANIFEST=%s\n  OUTPUT_DIR=%s\n  MAX_STEPS=%s\n' \
    "$MANIFEST" "$OUTPUT_DIR" "$MAX_STEPS"
  printf '  LOCAL_PYTHON=%s\n  ENTRY=%s\n' "$LOCAL_PYTHON" "$ENTRY"
  printf '  ACTION_HEAD_AS_TOOL_VERSION=%s\n' "$ACTION_HEAD_AS_TOOL_VERSION"
  exit 0
fi

exec bash "$ENTRY" --action-head-as-tool-version "$ACTION_HEAD_AS_TOOL_VERSION"
