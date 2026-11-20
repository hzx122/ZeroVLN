#!/usr/bin/env bash
# One 16-PPU pod: render 0-3, Next 4-11, 35B 12-13, LightNav 14, Habitat-gs 15.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../vln_demo/dlc" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
source "$REPO/scripts/demo/dlc/defaults.sh"
case "${1:-start}" in
  start|--check) ;;
  *) echo "Usage: VLN_RUN_DIR=/mounted/unique/run bash $0 [start|--check]"; exit 2 ;;
esac
: "${VLN_RUN_DIR:?Set VLN_RUN_DIR to a unique directory on CPFS}"
source "$REPO/scripts/demo/dlc/attempt.sh"
prepare_vln_attempt
exec > >(tee -a "$VLN_RUN_DIR/logs/entry.log") 2>&1
export VLN_RUN_DIR
export VLN_SLEEP_PYTHON=${VLN_SLEEP_PYTHON:-/usr/local/bin/python3}
export VLN_SLEEP_SCRIPT=${VLN_SLEEP_SCRIPT:-/opt/external/user-d5121be6/sleep.py}
# SSH shells do not inherit the image's PPU compiler/tool environment.
export PPU_HOME=${PPU_HOME:-/usr/local/PPU_SDK}
export PPU_SDK=${PPU_SDK:-$PPU_HOME}
export PPU_PATH=${PPU_PATH:-$PPU_HOME}
export CUDA_PATH=${CUDA_PATH:-$PPU_HOME/CUDA_SDK}
export CUDA_HOME=${CUDA_HOME:-$CUDA_PATH}
export CUDA_SDK=${CUDA_SDK:-$CUDA_PATH}
export CUDA_TOOLKIT_ROOT=${CUDA_TOOLKIT_ROOT:-$CUDA_PATH}
export CUDACXX=${CUDACXX:-$CUDA_PATH/bin/nvcc}
export PATH="$CUDA_PATH/bin:$PPU_HOME/bin:$PPU_HOME/ppu-smi/bin:$PATH"
export VLN_NODE=${VLN_NODE:-/opt/external/user-d5121be6/tools/node/bin/node}
export VLN_PI_CLI="$REPO/third_party/pi-agent/packages/coding-agent/dist/cli.js"
export PI_CODING_AGENT_DIR="$VLN_RUN_DIR/pi-agent"
bash "$REPO/scripts/demo/dlc/prepare_pi_tools.sh"
export PI_SKIP_VERSION_CHECK=1
export DEMO_TERMINAL_CWD=${DEMO_TERMINAL_CWD:-$VLN_RUN_ROOT/manual-workspace}
# Independent deployments must not rotate/repoint the old server's workspace.
mkdir -p "$DEMO_TERMINAL_CWD"
[[ -e "$DEMO_TERMINAL_CWD/.pi" || -L "$DEMO_TERMINAL_CWD/.pi" ]] || ln -s "$REPO/vln_demo/config/pi/SpatialOnboarding" "$DEMO_TERMINAL_CWD/.pi"
[[ -e "$DEMO_TERMINAL_CWD/AGENTS.md" || -L "$DEMO_TERMINAL_CWD/AGENTS.md" ]] || ln -s "$REPO/third_party/pi-agent/.pi/workspaces/SpatialOnboarding/AGENTS.md" "$DEMO_TERMINAL_CWD/AGENTS.md"
# Keep compiler/download caches on the pod's local filesystem, as required by
# the main branch runtime contract. Logs and session artifacts remain on CPFS.
export TMPDIR=/tmp/vln-dlc-cache/tmp
export XDG_CACHE_HOME=/tmp/vln-dlc-cache/xdg
export HF_HOME=/tmp/vln-dlc-cache/huggingface
export TRITON_CACHE_DIR=/tmp/vln-dlc-cache/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/vln-dlc-cache/torchinductor
export CUDA_CACHE_PATH=/tmp/vln-dlc-cache/cuda
export PYTORCH_KERNEL_CACHE_PATH=/tmp/vln-dlc-cache/torch-kernels
export TORCH_EXTENSIONS_DIR=/tmp/vln-dlc-cache/torch-extensions
export VLLM_CACHE_ROOT=/tmp/vln-dlc-cache/vllm
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$PYTORCH_KERNEL_CACHE_PATH" \
    "$TORCH_EXTENSIONS_DIR" "$VLLM_CACHE_ROOT"
export LD_LIBRARY_PATH="/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
# This task requires a whole-node allocation; preflight checks all 16 devices.
unset CUDA_VISIBLE_DEVICES
export RYNNVLN_VLLM_SERVER_URLS=http://127.0.0.1:28900
export RYNNVLN_VLLM_MODEL_NAME=Qwen3.6-35B
cat > "$VLN_RUN_DIR/bin/pi" <<'PI'
#!/usr/bin/env bash
set -euo pipefail
export PI_SKIP_VERSION_CHECK=1
unset OPENAI_API_KEY
exec "$VLN_NODE" "$VLN_PI_CLI" --provider "${VLN_PI_PROVIDER:-vllm-next-local}" --model "${VLN_PI_MODEL:-qwen3.8-flash-next}" --thinking medium "$@"
PI
chmod +x "$VLN_RUN_DIR/bin/pi"
export PATH="$VLN_RUN_DIR/bin:$PATH"
echo "[entry] $(date -Is) host=$(hostname) repo=$REPO run_root=$VLN_RUN_ROOT attempt=$VLN_RUN_DIR"
if ! /usr/bin/python3 "$HERE/preflight.py"; then
  echo '[entry] dependency preflight failed; see preflight.json and logs/preflight-*'
  exit 1
fi
if [[ "${1:-start}" == --check ]]; then exit 0; fi
exec /usr/bin/python3 -u "$HERE/supervise.py"
