# 本地模型服务

这里的启动脚本按职责拆分。网页控制台可以接任意兼容后端，下面只是当前机器的默认部署，不能据此在模型提示词或业务代码里假定具体参数规模。

## 当前 14 卡主策略（16 卡机器）

| 服务 | 入口 | 默认端口 | 默认设备 | 说明 |
|---|---|---:|---|---|
| Next vLLM（pi-agent 默认） | `vllm38next` | 9038 | 4-11 | Qwen3.8-Flash-Next，TP8/DP1，262144 上下文，served model `qwen3.8-flash-next` |
| 普通 vLLM（控制台默认） | `vllm-vln-35B.sh` | 28900 | 12,13 | 通用 Qwen3.6-35B；prefix cache + 可选 MTP |
| 四-query Action Head | `vllm-action-head-query.sh` | 28899 | 14,15 | 专用 worker/logits processor；固定不启用 MTP |

PPU 0-3 留给四个 3DGS render server，与 Next 的 8 卡和 35B 的 2 卡组成 14 卡主策略。14-15 是外挂槽：LightNav 使用 14；双卡 Action Head 需与同槽服务协调。`start_all.sh` 管理 render 和 console；模型服务单独启动，以免重启网页时误杀已预热模型。

## Next 模型

```bash
bash vln_demo/vllm/vllm38next
```

`vln_demo/vllm/vllm38next` 是唯一维护版本，与本目录的 SGLang、其他 vLLM 启动脚本一起纳入 Git。以下旧入口均为指向它的软链接：

- `/opt/external/user-d5121be6/vllm38next`
- `/opt/external/user-d5121be6/model/flagos-qwen38-public/runtime/vllm38next`

脚本沿用公开镜像 runtime、PPU 4-11、TP8/DP1 和端口 9038。原 SGLang 八卡槽由它接替；`sglang-qwen3.8-27B.sh` 仅保留为手动切回入口，不与 Next 同时启动。

Next 启用 `--enable-prefix-caching --mamba-cache-mode align`，并通过 `--enable-prompt-tokens-details` 在 API usage 中返回实际命中的 `cached_tokens`。相同历史前缀可以复用缓存，减少 agent 后续轮次的首 token 等待；新前缀仍需计算，生成阶段的速度不由 prefix caching 保证。MTP 暂不启用。

公开 PPU 镜像的原生 align 路径有两处 Next 兼容问题，由本目录 `next_prefix_cache.py` 修正：

- PLE 只有一个卷积状态，GDN 有卷积和循环两个状态。原复制缓冲区要求完整 `MambaSpec` 相等，会在首个请求触发 assert。插件按每组真实状态数分配描述符，沿用原有逐层复制逻辑，同时检查 token 块边界、复制函数和状态布局；原 `get_mamba_groups` 断言保持不变。
- 原页面统一逻辑按物理页面比例扩大压缩 QSA 的 token 块，与 align 调度器保存状态的边界错开，导致开启后仍不命中。插件利用 QSA 已有的 block-stride 寻址，对物理页面填充，保留逻辑块 400 和压缩块 100 行；代价是增加 KV 缓存占用。

`launch_next.py` 将插件安装到每次隔离启动的私有 tmpfs，通过 vLLM general-plugin 入口加载到各进程；所有维护文件都在仓库，公开镜像和宿主 Python 环境无需修改。其他模型继续使用原分配/页面统一逻辑。升级镜像或启用 MTP 时需要重新验证这些兼容条件。

CPU 回归测试在同一公开镜像内运行，覆盖异构状态复制、块间内存隔离、布局校验和 QSA 页面边界：

```bash
python3 - <<'PY'
from pathlib import Path
import subprocess
p = Path('vln_demo/vllm').resolve()
runtime = '/opt/external/user-d5121be6/model/flagos-qwen38-public/runtime'
subprocess.run([
    '/usr/bin/python3', '-I', str(p / 'launch_next.py'),
    runtime + '/scripts/run_official.py', 'python3', '-c',
    (p / 'test_next_prefix_cache.py').read_text(),
], check=True)
PY
```

2026-09-07 本机 TP8 单请求实测：每个输入连续请求三次，固定 temperature=0、seed=73、输出 128 token，以下取后两次中位数。开启前后输入一致，两个长度的输出文本也完全一致。

| 输入 token | 关闭缓存首 token | 命中缓存首 token | 命中 token | 完整请求耗时（关闭 → 命中） |
|---:|---:|---:|---:|---:|
| 3751 | 1.190 s | 0.147 s | 3600 | 5.528 → 4.873 s |
| 29651 | 8.678 s | 0.208 s | 29600 | 12.781 → 5.062 s |

本次生成阶段为约 26–27 token/s，关闭缓存时约 29–31 token/s；收益主要来自省去历史前缀计算。启动时报告 KV 容量 1,021,985 token，262144 上下文对应约 3.90 倍容量。上述是本机小样本，不能视为任意并发、前缀或输出长度的速度保证。

控制台默认模型为 `qwen35b`（28900，Qwen3.6-35B-A3B）。网页 pi-agent 仅在 SpatialOnboarding 项目扩展中注册 `vllm-next-local/qwen3.8-flash-next`（显示名 Qwen3.8-Flash-Next），使用 `http://127.0.0.1:9038/v1`，项目默认思考级别 medium。配置位于 `vln_demo/config/pi/`，通过 workspace 的 `.pi` 软链接生效；不修改 `/root/.pi/agent` 全局配置。

局部注册的 Next 和 27B 均为 256K（262144 token，pi 十进制显示 262.1K），支持 `off / low / medium / xhigh`。开启思考时通过 `chat_template_kwargs.reasoning_effort` 发送等级；关闭时发送 `enable_thinking=false` 并省略等级。模板自身未指定等级时默认 xhigh；minimal、high、max 不在原生模板支持范围内。

## 普通 vLLM

```bash
bash vln_demo/vllm/vllm-vln-35B.sh
```

常用覆盖变量：`VLLM_MODEL`、`VLLM_GPUS`、`VLLM_PORT`、`VLLM_MAX_MODEL_LEN`、`VLLM_GPU_MEMORY_UTILIZATION`、`VLLM_MTP_TOKENS`、`VLLM_ENABLE_PREFIX_CACHING`。

这个入口不读取任何 `ACTION_HEAD_*` 变量。带 head 的改动见下面的专用脚本。

## 四-query Action Head

```bash
ACTION_HEAD_QUERY_CHECKPOINT=/path/to/dualvln_query_embeddings.pt \
ACTION_HEAD_QUERY_CACHE=/dev/shm/action_head_queries \
bash vln_demo/vllm/vllm-action-head-query.sh
```

专用脚本负责 query token、节点本地编译缓存、自定义 worker 和 logits processor。vLLM 0.18 下 custom logits processor 与 speculative decoding 不兼容，因此该入口不接受 MTP 配置。普通基线必须继续调用 `vllm-vln-35B.sh`。

`vllm-vln-35B-hidden.sh` 是旧的全量 hidden 导出兼容入口，不是当前四-query 在线链路。

## 代理约定

外网下载可以使用 `http://127.0.0.1:3128`，但 localhost、`127.0.0.1`、`0.0.0.0` 和 `::1` 的服务请求必须写入 `NO_PROXY/no_proxy`。SGLang 脚本已经为自身 warmup 设置这组绕过地址。
