#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../vln_demo/src/objectnav" && pwd)
GARY_ROOT=${OBJECTNAV_GARY_ROOT:-/opt/external/user-e8e1af01/vln}
GS_ROOT=$GARY_ROOT/habitat-gs
PLATLIB=$GS_ROOT/build/cp312-cp312-linux_x86_64/install/platlib
RUNTIME=/data/shared/groups/mfm/common/anno/gary.zheng
export CUDA_VISIBLE_DEVICES=${OBJECTNAV_GPU:-15}
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/objectnav-cache/numba}
export EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export PYTHONPATH="${OBJECTNAV_PYTHON_DEPS:-/opt/external/user-d5121be6/tools/objectnav-py312}:$HERE/..:$RUNTIME/runtime_python/cp312:$PLATLIB:$GS_ROOT/src_python"
export LD_LIBRARY_PATH="$RUNTIME/runtime_libs/glvnd_x86_64:$PLATLIB/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR=${TMPDIR:-/tmp/objectnav-cache} XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/objectnav-cache}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$NUMBA_CACHE_DIR"
: "${OBJECTNAV_MANIFEST:?Set OBJECTNAV_MANIFEST to a normalized catalog}"
if [[ "${1:-}" == --check ]]; then
  "$GARY_ROOT/.venvs/wamdp_habitat/bin/python" -c 'import habitat_sim, numpy; from objectnav.render import Renderer; import os; r=Renderer(os.environ["OBJECTNAV_MANIFEST"]); assert hasattr(habitat_sim,"RenderInstanceHelper"); print("Habitat-gs renderer imports/catalog OK",len(r.rows))'
  exit 0
fi
exec "$GARY_ROOT/.venvs/wamdp_habitat/bin/python" -u -m objectnav.render --manifest "$OBJECTNAV_MANIFEST" --port "${OBJECTNAV_PORT:-7041}" --cache-size "${OBJECTNAV_SCENE_CACHE_SIZE:-8}"
