#!/usr/bin/env bash
set -euo pipefail
SRC=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${RYNNBRAIN_VENV:?Set your simulation virtual environment}
NAV=${RYNNBRAIN_NAV_DIR:?Set your local RynnBrain navigation checkout}
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH:-}"
export EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe HABITAT_SIM_GPU_DEVICE_ID=-1
export LP_NUM_THREADS="${VLN_RENDER_THREADS_PER_WORKER:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS=1
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONPATH="$SRC:${PYTHONPATH:-}" PYTHONDONTWRITEBYTECODE=1
exec "$VENV/bin/python" -u -m remote_habitat.server --scenes-root "$NAV/data/scene_datasets/mp3d" "$@"
