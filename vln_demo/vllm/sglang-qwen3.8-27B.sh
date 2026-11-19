#!/bin/bash
# 卡位约定：渲染占 PPU0-3，sglang×8 占 4-11，vllm 35B×2 占 12-13，14-15 备用。
# 往后移一张是给 mall 渲染腾出 PPU3（见 vln_demo/start_all.sh 头注释）。
export CUDA_VISIBLE_DEVICES=4,5,6,7,8,9,10,11

# 模型下载可继续使用本机代理，但服务自身对 0.0.0.0 的 warmup 请求必须直连。
# 同时维护大小写变量，兼容 urllib/httpx/requests 的不同读取方式。
LOCAL_NO_PROXY=localhost,127.0.0.1,0.0.0.0,::1
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$LOCAL_NO_PROXY"
export no_proxy="${no_proxy:+$no_proxy,}$LOCAL_NO_PROXY"

# 防 VLM 图片预处理 OOM：这是个 Mamba 混合 VLM，auto 的 mem_fraction_static≈0.784
# 会把卡吃到 ~98%，图片 stack 峰值没余量 → OOM → 流断 → pi-agent 报
# "Stream ended without finish_reason"。降到 0.7 每张卡多留约 6G 给图片预处理。
# 注意：不要加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ——它在 PPU 上和
# cuda graph capture 冲突（custom_all_reduce 捕获报 "invalid argument"），实测会起不来。
# 留的 6G+ 余量已足够吸收图片处理峰值，不需要再加 expandable_segments 防碎片。
# 详见 vln_demo更新日志 2026-08-27 条。

# NEXTN/MTP 当前关闭，以保留 Qwen3.5/Mamba 的 Radix prefix cache。
# 如需恢复，在下面的 launch_server 参数中加入：
#   --speculative-algo NEXTN \
#   --speculative-num-steps 3 \
#   --speculative-eagle-topk 1 \
#   --speculative-num-draft-tokens 4 \

python3 -m sglang.launch_server \
  --model-path /opt/external/user-d5121be6/model/Qwen3.8-27B \
  --tp-size 8 \
  --mem-fraction-static 0.7 \
  --host 0.0.0.0 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-metrics \
  --enable-cache-report \
  --port 9000
