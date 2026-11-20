#!/usr/bin/env bash
# Local service wrapper: private RAM-backed caches under the workspace.
# CPFS cannot host UNIX sockets and may return ESTALE for compiled kernels.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../vln_demo/vllm" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
RUNTIME="${ACTION_HEAD_LOCAL_RUNTIME_DIR:-$REPO/.local_runtime/brain-tmpfs}"
case "$RUNTIME" in
  "$REPO"/.local_runtime/*|"$(dirname "$REPO")"/.runtime/*|/workspace/.runtime/*) ;;
  *) echo 'local runtime must be a dedicated workspace runtime directory' >&2; exit 2 ;;
esac
[[ $# -gt 0 ]] || { echo "usage: $0 command [args...]" >&2; exit 2; }
mkdir -p "$RUNTIME"
exec unshare --user --map-root-user --mount bash -s -- "$RUNTIME" "$@" <<'SH'
set -euo pipefail
runtime=$1
shift
mount -t tmpfs -o size=16G,mode=700 tmpfs "$runtime"
for key in TMPDIR XDG_CACHE_HOME HF_HOME TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR \
           CUDA_CACHE_PATH PYTORCH_KERNEL_CACHE_PATH TORCH_EXTENSIONS_DIR \
           DG_CACHE_DIR VLLM_CACHE_ROOT; do
    dir="$runtime/${key,,}"
    mkdir -p "$dir"
    export "$key=$dir"
done
# Abstract sockets avoid sockaddr_un's 107-character filesystem path limit.
export VLLM_RPC_BASE_PATH="@vln-brain-$BASHPID"
export PYTHONDONTWRITEBYTECODE=1
exec "$@"
SH
