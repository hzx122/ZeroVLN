#!/usr/bin/env bash
# Foreground processes: ownership and cleanup belong to the DLC supervisor.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
LIGHTNAV_ROOT="$REPO/third_party/LightNav-0"
case "${1:-}" in
  lightnav)
    cd "$LIGHTNAV_ROOT"
    export CUDA_VISIBLE_DEVICES=14 VLLM_ENABLE_V1_MULTIPROCESSING=0
    export VLN_VLLM_ENFORCE_EAGER=1 VLN_KV_CACHE_GIB=2 PYTHONPATH="$LIGHTNAV_ROOT/src"
    if [[ "${2:-}" == --check ]]; then
      exec /usr/local/bin/python3 -m lightnav.serving.ws_server --help
    fi
    exec /usr/local/bin/python3 -u -m lightnav.serving.ws_server \
      --model_path /opt/external/user-d5121be6/LightNav-0/model \
      --backend vllm_local --task vln --gpu_memory_utilization 0.4 \
      --max_batch_size 1 --host 0.0.0.0 --port 8050
    ;;
  mujoco)
    MUJOCO_ROOT="$LIGHTNAV_ROOT/mujoco_demo"
    MUJOCO_EXEC=${LIGHTNAV_MUJOCO_EXEC:-$MUJOCO_ROOT/.venv/bin/vln-mujoco}
    if [[ ! -x "$MUJOCO_EXEC" ]]; then
      MUJOCO_EXEC=/opt/external/user-d5121be6/LightNav-0/repo/mujoco_demo/.venv/bin/vln-mujoco
    fi
    cd "$MUJOCO_ROOT"
    export CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYTHONPATH="$MUJOCO_ROOT"
    if [[ "${2:-}" == --check ]]; then exec "$MUJOCO_EXEC" --help; fi
    exec "$MUJOCO_EXEC" --host 0.0.0.0 --port 18088 --vln-server ws://127.0.0.1:8050
    ;;
  *) echo 'Usage: start_extra.sh lightnav|mujoco' >&2; exit 2 ;;
esac
