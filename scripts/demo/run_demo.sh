#!/usr/bin/env bash
# VLN demo runner: start one vLLM server for the RynnBrain checkpoint,
# then run src/vln_demo.py to dump RGB frame sequences for 3 episodes.
set -uo pipefail   # NOTE: no -e, we handle rc manually to always kill vLLM

NAV_DIR=/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation
DATA_ROOT=/data/shared/groups/mfm/common/anno/vlmevalkit/RynnBrain/data
VENV=/opt/navigation-runtime/venv
DEMO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../vln_demo" && pwd)
DEMO_OUT=$DEMO_DIR/outputs/run_$(date +%Y%m%d_%H%M%S)
VLLM_PORT=28899
VLLM_GPU=${VLLM_GPU:-13}

MODEL_PATH=${MODEL_PATH:-/data/shared/groups/mfm/vl/rl/pretrain_models/final_mix_mt_me6_05/hf_iter_0001250}

mkdir -p "$DEMO_OUT"
LOG_VLLM=$DEMO_OUT/vllm.log
LOG_DEMO=$DEMO_OUT/demo.log

echo "[runner] output dir: $DEMO_OUT"

source "$VENV/bin/activate"

# --- env setup (same tricks as run_rynnvln_benchmarks.sh) ---
ln -sfn "$DATA_ROOT" "$NAV_DIR/data"

export EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export GALLIUM_DRIVER=llvmpipe
export HABITAT_SIM_GPU_DEVICE_ID=-1
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
# libOpenGL.so.0 extracted from the libopengl0 deb into venv/lib
export LD_LIBRARY_PATH="$DEMO_DIR/lib:$VENV/lib:${LD_LIBRARY_PATH:-}"

export RYNNVLN_LANGUAGE=en
export RYNNVLN_TURN_ANGLE=30
export RYNNVLN_HFOV=120
export RYNNVLN_ALLOW_SLIDING=1
export RYNNVLN_LOG_RAW_OUTPUT=1
export RYNNVLN_VLLM_IMAGE_FORMAT=JPG
export RYNNVLN_VLLM_IMAGE_QUALITY=90
export MAX_NEW_TOKENS=2048

export MODEL_PATH
export DEMO_OUTPUT_PATH=$DEMO_OUT
export RYNNVLN_VLLM_SERVER_URLS=http://127.0.0.1:$VLLM_PORT
export RYNNVLN_VLLM_MODEL_NAME=RynnBrain
export RYNNBRAIN_NAV_DIR=$NAV_DIR

# --- start vLLM server ---
echo "[runner] starting vLLM on GPU $VLLM_GPU port $VLLM_PORT ..."
(
  cd "$NAV_DIR"
  CUDA_VISIBLE_DEVICES=$VLLM_GPU python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name RynnBrain \
    --host 127.0.0.1 --port $VLLM_PORT \
    --trust-remote-code --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --limit-mm-per-prompt '{"image":64}' \
    --no-enable-log-requests
) >"$LOG_VLLM" 2>&1 &
VLLM_PID=$!
echo "[runner] vLLM pid=$VLLM_PID log=$LOG_VLLM"

# --- wait for /health ---
ready=0
for i in $(seq 1 200); do
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "[runner] ERROR: vLLM died early, last log:"; tail -n 40 "$LOG_VLLM"; exit 1
  fi
  if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$VLLM_PORT/health" 2>/dev/null | grep -q '^[234]'; then
    ready=1; echo "[runner] vLLM ready after ~$((i*3))s"; break
  fi
  sleep 3
done
if [ "$ready" != "1" ]; then
  echo "[runner] ERROR: vLLM health timeout, last log:"; tail -n 40 "$LOG_VLLM"
  kill $VLLM_PID 2>/dev/null
  exit 1
fi

# --- run demo ---
echo "[runner] running demo ..."
cp -f "$DEMO_DIR/src/vln_demo.yaml" "$NAV_DIR/vln_demo.yaml"
(
  cd "$NAV_DIR"
  python3 "$DEMO_DIR/src/vln_demo.py"
) >"$LOG_DEMO" 2>&1
DEMO_RC=$?
echo "[runner] demo rc=$DEMO_RC"

# --- always stop vLLM ---
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "[runner] vLLM stopped"

echo "[runner] results in $DEMO_OUT"
exit $DEMO_RC
