#!/bin/bash
# Start the VLN console with every asset set mounted: R2R/MP3D (habitat-lab),
# RxR/MP3D (same simulator, see src/rxr.py), 自采 3DGS (3DRS), PARTNR/HSSD
# (habitat_sim direct), ABotN's three Gaussian benchmarks (POIBench, PointBench
# 室内/室外), and SAGE-3D GVLN (InteriorGS Gaussians, see src/sage/).
# hab2/ReplicaCAD is mounted only if you name it (DEMO_BACKEND=hab) -- its
# render is broken alongside the others, see BACKENDS["hab"] in vln_model.py.
# Pick a subset with DEMO_BACKEND, e.g. DEMO_BACKEND=habitat,partnr or
# DEMO_BACKEND=poi,in,out.
#
#   ./run_console.sh [port]        # default 5000
#
# Why a launcher rather than `python vln_model.py`:
#   * habitat lives in gordon's venv and its magnum bindings need libOpenGL.so.0
#     out of that venv's lib/ -- LD_LIBRARY_PATH is read by the dynamic loader at
#     exec time, so it cannot be fixed up from inside python.
#   * habitat's GL stack has no display here: it renders through llvmpipe with a
#     surfaceless EGL platform, and picks the GPU by HABITAT_SIM_GPU_DEVICE_ID=-1
#     (CPU) so it never fights the vLLM servers for PPU memory.
#   * cwd must be the RynnBrain navigation dir: that is where vln_demo.yaml's
#     relative data/ paths and the R2R split resolve from.
# The Gaussian backends need none of this -- they only need their render server
# up, which is deliberately NOT started here: those hold Gaussians on the GPU and
# are meant to outlive many console runs. There are three of them, on different
# ports: spherical-harmonic degree is fixed per server process (7036 for 3DRS at
# degree 0, see rs3d/README.md; 7037 for ABotN at degree 3, see abotn/README.md),
# and a server also pins one scenes_root, which is why SAGE gets its own 7038
# even though it is degree 3 like ABotN.
set -e
PORT="${1:-${DEMO_HTTP_PORT:-5000}}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV="${RYNNBRAIN_VENV:-/opt/navigation-runtime/venv}"
NAV_DIR="${RYNNBRAIN_NAV_DIR:-/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation}"

export PATH="$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$VENV/lib:$LD_LIBRARY_PATH"
export RYNNBRAIN_NAV_DIR="$NAV_DIR"

# headless software GL for habitat
export EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export HABITAT_SIM_GPU_DEVICE_ID=-1
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# model side
export RYNNVLN_LANGUAGE="${RYNNVLN_LANGUAGE:-en}"
export RYNNVLN_VLLM_IMAGE_FORMAT="${RYNNVLN_VLLM_IMAGE_FORMAT:-JPG}"
export RYNNVLN_VLLM_IMAGE_QUALITY="${RYNNVLN_VLLM_IMAGE_QUALITY:-90}"
export RYNNVLN_VLLM_MODEL_NAME="${RYNNVLN_VLLM_MODEL_NAME:-Qwen3.6-35B}"
export RYNNVLN_VLLM_SERVER_URLS="${RYNNVLN_VLLM_SERVER_URLS:-http://127.0.0.1:28899}"

cd "$NAV_DIR"
exec "$VENV/bin/python" "$SRC/vln_model.py" \
  --backend "${DEMO_BACKEND:-all}" --port "$PORT" "${@:2}"
