#!/usr/bin/env bash
set -euo pipefail
REPO=/workspace/ZeroVLN
OUT=/data/zerovln/artifacts/rule-teacher-v6-local-20260930-r1
# Explicitly authorized by the user for this probe; remove after services stop.
CACHE=/tmp/anonymous-rule-teacher-20260930
for key in TMPDIR XDG_CACHE_HOME HF_HOME TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR CUDA_CACHE_PATH PYTORCH_KERNEL_CACHE_PATH TORCH_EXTENSIONS_DIR MPLCONFIGDIR RTC_CACHE_PATH PPU_LIB_RTC_CACHE_PATH VLLM_CACHE_ROOT VLLM_RPC_BASE_PATH NUMBA_CACHE_DIR DG_CACHE_DIR; do
  mkdir -p "$CACHE/${key,,}"
  export "$key=$CACHE/${key,,}"
done
# CPFS cannot host Unix socket files. Linux abstract IPC creates no filesystem
# entry and also stays below sockaddr_un's 107-byte path limit.
export VLLM_RPC_BASE_PATH=@rule_teacher30
export PYTHONDONTWRITEBYTECODE=1
export PPU_SDK=/usr/local/PPU_SDK CUDA_PATH=/usr/local/PPU_SDK/CUDA_SDK
export CUDA_HOME=$CUDA_PATH CUDA_TOOLKIT_ROOT=$CUDA_PATH
export PATH=$CUDA_PATH/bin:$PPU_SDK/bin:$PPU_SDK/ppu-smi/bin:$PATH
export LD_LIBRARY_PATH=/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO/verl:$REPO/vln_demo/src:${PYTHONPATH:-}
export SAGE_SPLIT=official_gvln_action_head_p0_v3full827train
export SAGE_PRIVILEGED_STATE=1
export SAGE_COLLISION=usd_rooms SAGE_FURNITURE_COLLISION=none SAGE_ROBOT_RADIUS=0.07
export SAGE_USD_COLLISION_MASK_ROOT=/data/zerovln/SAGE-3D/sage_usd_rooms_masks
export SAGE_CAMERA_HEIGHT=1.2 SAGE_HFOV=115 SAGE_LAZY_EPISODES=1 SAGE_RENDER_URL=http://127.0.0.1:7074
cd "$REPO"
case "${1:-}" in
 nlp)
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  exec /workspace/.runtime/rule-nlp-spacy38/bin/python -B "$REPO/scripts/eval/serve_rule_call_nlp.py" --model "$OUT/nlp-models/en_core_web_sm-3.8.0/en_core_web_sm/en_core_web_sm-3.8.0" --port 19074 ;;
 brain)
  export VLLM_GPUS=0,1 VLLM_TP_SIZE=2 VLLM_PORT=28970 VLLM_MTP_TOKENS=0
  export RULE_TEACHER_SAMPLING_PROFILE=v6_depth ACTION_HEAD_ENABLE_QUERY_TAIL=0
  export VLLM_ENFORCE_EAGER=0 VLLM_GPU_MEMORY_UTILIZATION=0.7 VLLM_ENABLE_PREFIX_CACHING=1
  unset VLLM_SCHEDULER_CLS
  export VLLM_MAX_MODEL_LEN=32768 VLLM_MAX_NUM_SEQS=0 VLLM_MAX_NUM_BATCHED_TOKENS=8192
  exec bash "$REPO/scripts/serve/vllm/vllm-vln-35B.sh" ;;
 render)
  RS=/workspace/agentic_sim/repo/ABot-Navigation/render_server
  export PYTHONPATH=$REPO/vln_demo/render_libs:$RS/submodules/diff-plane-rasterization:$RS/submodules/simple-knn:$RS
  exec /opt/navigation-runtime/venv/bin/python "$RS/render_sim_dynamic.py" --scenes_root /data/zerovln/SAGE-3D/sage_gs_root --port 7074 --gpus 2 --sh_degree 3 --max_scenes_per_gpu 4 ;;
 gym)
  /usr/local/bin/python3 -B - <<'PY'
import time, urllib.request
for attempt in range(120):
    try:
        with urllib.request.urlopen('http://127.0.0.1:7074/ping', timeout=2) as response:
            if response.status == 200:
                break
    except OSError:
        pass
    if attempt % 10 == 0:
        print('Waiting for the dedicated teacher renderer before Gym initialization', flush=True)
    time.sleep(1)
else:
    raise SystemExit('Teacher renderer not ready; Gym was not started')
PY
  exec /usr/local/bin/python3 -B -m vln_gym.server --port 18974 --backend sage --scene sage-840275 --episode_id official_840275_13_0 --split "$SAGE_SPLIT" --max_steps 1000 ;;
 *) shift; exec /opt/navigation-runtime/venv/bin/python -B "$@" ;;
esac
