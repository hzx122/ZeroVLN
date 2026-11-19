export PYTHONPATH=/workspace:${PYTHONPATH:-}
export VLLM_LOGGING_CONFIG_PATH=/workspace/vln_demo/vllm/vllm_log_hidden_id.json

# VLN demo 用: MoBot_VLN (final_mix_mt_me6 / hf_iter_0001250)。
# 网页上选「MoBot_VLN-2B」走这个。这是 VLN 任务上真正训过的模型, 用来和通用的
# 35B 对照。
# Dense 模型并行约束：只允许按可用 GPU 数调整 DP；TP 固定，禁止修改。
#
# 2.72B dense (Qwen3.5 架构, 24 层 hidden 2048): text 1.88B + embed 0.51B
# + vision 0.33B。bf16 权重 5.1 GB —— 那是**存储**大小不是参数量, 别按 5B 记。
# 单卡放得下, 所以 TP=1 DP=1 就占一张卡 —— 倒数第三张 (GPU 13)。
# 图像上限同 35B: ReAct 一次请求最多 16 张帧。
# served-model-name 用真名不用 default, 理由见 vllm-vln-35B.sh。
#
# --chat-template: ReAct 的 keep_thinking 需要它。本模型自带的模板会把每条
# assistant 消息按 </think> 切开, 只在「该轮位于最后一个用户提问之后」时才把
# reasoning 打出来 —— 工具循环里最后一条永远是工具返回, 所以历史里的思考 100%
# 被丢掉, 客户端塞原始回复也没用。35B 的模板多了个 preserve_thinking 开关,
# 2B 的没有, 这个文件就是给它补上同样那一行(逐字照抄 35B 的判断)。
# 已用真 tokenizer 验过: 不传 preserve_thinking 时渲染结果与自带模板逐字节相同,
# 所以经典单轮路径不受影响。
CUDA_VISIBLE_DEVICES=13 \
vllm serve "/data/shared/groups/mfm/vl/rl/pretrain_models/final_mix_mt_me6/hf_iter_0001250" \
    --allowed-local-media-path / \
    --served-model-name MoBot_VLN-2B \
    --chat-template /workspace/vln_demo/src/chat_template_mobot2b_keep_thinking.jinja \
    --port 28898 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 131072 \
    --limit-mm-per-prompt '{"image":64}' \
    --enable-log-requests
