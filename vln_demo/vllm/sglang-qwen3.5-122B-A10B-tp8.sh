#!/usr/bin/env bash
# TP8 批量推理：MTP 关闭；保留 Radix/Mamba 前缀缓存。
set -euo pipefail
# 完整16卡服务：GPU0–3渲染，GPU12–13底层控制；主模型独占4–11。
export CUDA_VISIBLE_DEVICES=4,5,6,7,8,9,10,11
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export PYTHONDONTWRITEBYTECODE=1
# 所有运行缓存放容器本地；DLC supervisor 会传入独立目录并在退出时清理。
cache_dir=${VLN_SGLANG_LOCAL_CACHE:-/tmp/vln-sglang/qwen35-122b}
for key in TMPDIR XDG_CACHE_HOME HF_HOME TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR CUDA_CACHE_PATH PYTORCH_KERNEL_CACHE_PATH TORCH_EXTENSIONS_DIR FLASHINFER_WORKSPACE_BASE SGLANG_CACHE_DIR; do
    export "$key=$cache_dir/${key,,}"
    mkdir -p "$cache_dir/${key,,}"
done

exec "${VLN_SGLANG_PYTHON:-python3}" -u -m sglang.launch_server \
  --model-path /opt/external/user-d5121be6/model/Qwen3.5-122B-A10B \
  --served-model-name qwen3.5-122b-a10b \
  --host 0.0.0.0 --port 9122 \
  --tp-size 8 \
  --context-length 262144 \
  --mem-fraction-static 0.8 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-metrics --enable-cache-report
