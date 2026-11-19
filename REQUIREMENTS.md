> 开源版本运行入口见根 README；使用 `configs/gpu/environment.example.sh` 配置自己的环境。DLC历史命令不作为本版本启动方式。

# 独立安装环境：PPU、NVIDIA 与 VLN Demo

本文从 **Ubuntu 基础系统 + 独立安装包/源码** 描述环境，不以已有业务镜像、其他人的 venv 或复制 site-packages 为安装前提。现有机器仅提供版本和运行证据。流程入口见 [README](README.md)，页面怎么玩见 [README 中的浏览器部署说明](README.md#browser-console)。

## 1. 两类硬件、两种用途

| 硬件 | 训练/模型环境 | Demo/仿真环境 |
|---|---|---|
| PPU-ZW810E | `train-ppu`：厂商 PyTorch/vLLM/Triton/算子 + 本项目 | `demo-ppu`：厂商 PyTorch + Habitat/Mesa + PPU 3DGS 扩展 |
| NVIDIA | `train-cuda`：官方 CUDA PyTorch + 本项目；官方 vLLM 独立 `serve-cuda` | `demo-cuda`：CUDA PyTorch + Habitat/Mesa + 重新编译的 CUDA 3DGS 扩展 |

**模型服务与仿真环境分开。** Habitat 使用 NumPy 1.x；模型服务的依赖可能要求 NumPy 2.x。v8 统一页面让模型服务和 Gym/renderer 各用自己的环境，网页不需要把所有包装进同一个 Python。

本次材料分为：

- [common.txt](requirements/common.txt)：公共 Python 依赖。
- [train.txt](requirements/train.txt)：SFT/FM/RL 上层依赖；不含硬件 wheel。
- [demo.txt](requirements/demo.txt)：网页、导航仿真和 3DGS Python 依赖；原生扩展另编译。
- [ppu-wheels.txt](requirements/ppu-wheels.txt)：**仅从厂商 wheelhouse 安装**的 PPU 组件版本。
- [ppu-serving.txt](requirements/ppu-serving.txt)：PPU vLLM声明的公共服务依赖，单独说明transformers冲突。
- [nvidia-serving.txt](requirements/nvidia-serving.txt)：官方 NVIDIA vLLM 独立服务环境。
- [PPU 构建来源](requirements/ppu-build-provenance.json)：SDK、PyTorch 源码版本与四个原安装 wheel 的 SHA256。
- [本机包盘点](requirements/ppu-observed-inventory.txt)：445 个包的版本证据，含历史/非本项目组件，**不是可直接安装的锁文件**。

安装清单按系统依赖、硬件组件、Python 包和源码扩展组织，分别给出 PPU 与 NVIDIA 的安装步骤。

## 2. 基础系统与公共工具

参考系统为 Ubuntu 24.04 x86_64、Python 3.12（当前运行节点 3.12.3）。以下命令只用于**新目标机器/新构建环境**，不修改正在运行的共享环境：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3.12 python3.12-dev python3.12-venv \
  git git-lfs ca-certificates curl pkg-config build-essential ninja-build cmake \
  libgomp1 libopenblas-dev libjpeg-dev libpng-dev zlib1g-dev libglm-dev \
  libegl1 libegl-mesa0 libgl1 libglx-mesa0 libgl1-mesa-dri libgbm1 \
  libopengl0 libegl1-mesa-dev libglvnd-dev xorg-dev freeglut3-dev \
  ffmpeg libglib2.0-0t64 libsm6 libxext6 libxrender1
```

Mesa/EGL/GBM/DRI 组件使用同一仓库发行版的一组兼容包，不手工混装不同构建后缀。已运行环境曾使用 Mesa `25.2.8-0ubuntu0.24.04.2`、GLVND `1.7.0-1build1`；这些是对照版本，新机器由其 Ubuntu 仓库解出配套包。CPU renderer 还需要 `g++`、OpenMP，首次使用在本地缓存编译 C++ 合成器。

创建环境时不用 `--system-site-packages`，避免暗中依赖旧镜像：

```bash
# 在当前工作区准备环境时的允许路径；DLC中改为容器本地/root/rl-code/<EXPR>/envs。
ENV_ROOT=/opt/venvs/closed-loop-independent
mkdir -p "$ENV_ROOT"
python3.12 -m venv "$ENV_ROOT/train-ppu"
python3.12 -m venv "$ENV_ROOT/demo-ppu"
# NVIDIA机器使用同样方式创建train-cuda、demo-cuda，必要时另建serve-cuda。
```

设置 `REPO` 为包含本次文档的 Git checkout 绝对路径。每套环境先安装构建工具：

```bash
: "${ENV_ROOT:?环境根}" "${REPO:?项目checkout绝对路径}"
"$ENV_ROOT/train-ppu/bin/python" -m pip install 'setuptools>=77,<81' wheel 'cmake<4' ninja packaging
"$ENV_ROOT/demo-ppu/bin/python" -m pip install 'setuptools>=77,<81' wheel 'cmake<4' ninja packaging
```

编译缓存、pip 临时目录放本机本地目录；需要保存的 wheel、构建日志和环境回执写 `/data/zerovln/artifacts/<EXPR_NAME>/`。以下各节安装时都使用目标环境的 `python -m pip`。

## 3. PPU 环境：从 SDK 和 wheel 重建

### 3.1 系统侧材料

宿主机安装与目标 PPU 匹配的驱动；容器/用户态安装厂商 SDK、编译器、数学库及通信库。它们是独立安装输入，不由 pip 提供，也不是从已有业务镜像复制出来。

| 组件 | 当前运行证据 | 新环境安装目标 |
|---|---|---|
| PPU SDK | `/usr/local/PPU_SDK/release.yaml`：`2.0.0-715aa1` | 厂商对应 SDK 安装包；提供 `envsetup.sh`、CUDA 兼容编译器和运行库 |
| CUDA 兼容接口 | PPU torch 报告 CUDA API 12.9 | 不等于应安装 NVIDIA CUDA 12.9；必须使用 PPU SDK |
| PCCL/ACCL-P | `/opt/accl-p/libpccl.so.2.0.0` | 厂商通信包及配套网络/驱动配置；该文件名不充分代表发行包版本 |
| 运行库 | `libhggcrt1.so`、SDK `CUDA_SDK/lib64`、SDK `lib`，适用时 `sailSHMEM/lib` | 来自同一套 SDK/通信包，不手工伪造软链接 |
| PyTorch | 2.9.0，源码 `880d7c8fb5e3faa2540153edd5a4f5fda38b062a` | 厂商 CPython 3.12、Linux x86_64 wheel |
| vLLM | `0.18.0+ppu2.0.0` | 厂商 PPU 构建，含其运行时算子依赖 |
| Triton/算子 | 见 `ppu-wheels.txt`；额外 `causal-conv1d==1.7.0` 原先在独立 overlay | 用目标 SDK/PyTorch 对应的 wheel，不复制旧 `.so` |

SDK、PCCL 和 PPU wheel 使用厂商/平台软件源的对应发行包；版本与原始 wheel SHA 见 [PPU 构建来源](requirements/ppu-build-provenance.json)。

拿到 SDK 后按该发行包的安装说明安装，再初始化：

```bash
export LIBRARY_PATH=${LIBRARY_PATH:-}
source /usr/local/PPU_SDK/envsetup.sh cuda
export CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK
export CUDA_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="/opt/accl-p:$CUDA_HOME/lib64:/usr/local/PPU_SDK/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
```

当前开源入口继承已激活的 SDK/CUDA 环境。PPU 按实际安装位置 source SDK；训练与模拟环境路径统一填写到 configs/gpu/environment.example.sh 的本机副本。

### 3.2 Python 安装次序

准备 `PPU_WHEELHOUSE`：包含上述厂商 wheel 及厂商发布的传递依赖。torch/torchvision 的名称和普通 CUDA 包相同，**不能省略 `--no-index` 后让 pip 从公共源回退下载**。

```bash
: "${PPU_WHEELHOUSE:?独立厂商wheel目录}" "${ENV_ROOT:?}" "${REPO:?}"
# 完整模型环境：只从厂商材料安装硬件组件，避免自动替换成NVIDIA包。
"$ENV_ROOT/train-ppu/bin/python" -m pip install \
  --no-index --find-links "$PPU_WHEELHOUSE" --no-deps -r "$REPO/requirements/ppu-wheels.txt"
"$ENV_ROOT/train-ppu/bin/python" -m pip install -r "$REPO/requirements/train.txt"
"$ENV_ROOT/train-ppu/bin/python" -m pip install -r "$REPO/requirements/ppu-serving.txt"

# 仿真环境仅需PPU torch/vision，不安装整套vLLM。
"$ENV_ROOT/demo-ppu/bin/python" -m pip install \
  --no-index --find-links "$PPU_WHEELHOUSE" --no-deps torch==2.9.0 torchvision==0.24.0
"$ENV_ROOT/demo-ppu/bin/python" -m pip install -r "$REPO/requirements/demo.txt"
```

`--no-deps` 用于避免硬件 wheel 的解析器更换平台；传递依赖由公共 profile 与厂商配套清单安装。torch 的 filelock、typing-extensions、sympy、networkx、Jinja2、fsspec，以及 vLLM 的 FastAPI、tokenizers、protobuf、openai、llguidance、xgrammar、compressed-tensors、FlashInfer/DeepGEMM 等，公共依赖已经按该wheel元数据列入 `ppu-serving.txt`；专用算子仍须按厂商该构建的完整依赖清单安装。`train.txt` 并不是 vLLM 全套传递依赖锁；独立 wheelhouse 应包含厂商配套清单，不能用盲目升级所有包替代。

实际环境存在需要保留说明的兼容差异：PPU vLLM 包声明 `transformers<5`、`aiohttp>=3.13.3`，而当前基础环境记录为 transformers 5.5.4、aiohttp 3.9.0；numba 0.61.2 也不支持本机基础环境的 NumPy 2.4.4。新公共 profile 使用符合 numba 的 NumPy 范围，并更新 aiohttp 下限。Qwen3.5 模型类依赖 transformers 5.x，不能为了让包检查变绿降回 4.x；vLLM/transformers 的组合需要厂商对应兼容构建与项目已有 `verl/action_head/serve/bootstrap` 补丁一起确认。新增安装不通过的冲突要如实保留。

旧 `FAST_KERNEL_OVERLAY` 位于 `/workspace/envs/qwen36_full_rl_fastkernels_native_py312/site-packages`，实际额外包是 causal-conv1d 1.7.0。独立安装后将 launch plan 的 `fast_kernel_overlay` 指向新环境对应目录；该包须在目标 PPU torch ABI 下重建或使用厂商匹配 wheel。

## 4. NVIDIA 环境：官方 CUDA 栈与独立模型服务

### 4.1 训练 / FM / 3DGS

宿主安装支持所选 CUDA 版本的 NVIDIA 驱动；容器运行另需 NVIDIA Container Toolkit。编译 rasterizer/FlashAttention/causal-conv1d 需要完整 CUDA Toolkit（含 nvcc），仅安装 PyTorch wheel 不提供整套编译工具链。

本项目对应 PyTorch 2.9.0，可选官方 CUDA 12.8 构建作为 NVIDIA 迁移基线：

```bash
: "${ENV_ROOT:?}" "${REPO:?}"
python3.12 -m venv "$ENV_ROOT/train-cuda"
python3.12 -m venv "$ENV_ROOT/demo-cuda"
for env_name in train-cuda demo-cuda; do
  "$ENV_ROOT/$env_name/bin/python" -m pip install 'setuptools>=77,<81' wheel 'cmake<4' ninja packaging
  "$ENV_ROOT/$env_name/bin/python" -m pip install \
    torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
done
"$ENV_ROOT/train-cuda/bin/python" -m pip install -r "$REPO/requirements/train.txt"
"$ENV_ROOT/demo-cuda/bin/python" -m pip install -r "$REPO/requirements/demo.txt"
# CUDA Toolkit 12.8已独立安装到该位置：
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
```

该 torch/vision/cu128 组合来自 [PyTorch 官方版本安装表](https://pytorch.org/get-started/previous-versions/)。不要加载 PPU SDK、PCCL 或 PPU Triton；Triton/NCCL 使用 NVIDIA PyTorch 的匹配依赖。FM 的 FLA、causal-conv1d、FlashAttention 按目标 torch/CUDA 版本安装/重编译；不能将 PPU 的同名 wheel 或 overlay 移来复用。下面按目标 torch/CUDA 环境从源码安装算子：

```bash
"$ENV_ROOT/train-cuda/bin/python" -m pip install flash-linear-attention==0.5.0
MAX_JOBS=4 "$ENV_ROOT/train-cuda/bin/python" -m pip install --no-build-isolation \
  causal-conv1d==1.7.0 flash-attn==2.7.4.post1
```

### 4.2 NVIDIA vLLM 与联合 RL 的配置

官方 vLLM 0.18.0 绑定 **torch 2.10.0、torchvision 0.25.0**，并声明 **transformers<5**；它与 PPU 的 torch 2.9.0/transformers 5.5.4 组合不同。来源：[官方 CUDA requirements](https://raw.githubusercontent.com/vllm-project/vllm/v0.18.0/requirements/cuda.txt)、[公共 requirements](https://raw.githubusercontent.com/vllm-project/vllm/v0.18.0/requirements/common.txt)。

普通 Demo 的模型服务可独立建环境，再通过 HTTP 连接：

```bash
python3.12 -m venv "$ENV_ROOT/serve-cuda"
"$ENV_ROOT/serve-cuda/bin/python" -m pip install -r "$REPO/requirements/nvidia-serving.txt"
```

模型服务同时配置 Qwen 模型、processor 和项目的 query-tail/native-context 扩展。不要将 `nvidia-serving.txt` 和 `train.txt` 合成一个要求集，也不要把两个环境的 site-packages 拼到一起。

**当前 RLOO 同卡共置训练/rollout 不能仅靠独立 HTTP 服务替代。** NVIDIA 全链迁移还需对齐训练器与 vLLM 的 torch/transformers 版本、项目 worker/权重同步接口、PPU专用路径与算子。

## 5. Habitat / Mesa：不用旧共享 venv

两种硬件均在自己的 `demo-*` 环境安装 Habitat。R2R/RxR 默认 CPU 软件 EGL，避免与大脑争用显存：

```bash
export EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export HABITAT_SIM_GPU_DEVICE_ID=-1
```

当前历史运行环境记录为 habitat-sim 0.2.4、habitat-lab `0.2.420230405`、Magnum Python 绑定。本次读取到历史 Habitat-Sim 源码 HEAD 为 `f179b584bcd713c5a2a998132211e2cae881d6d1`；Magnum/pybind11 等子模块的实际构建状态不能由 `habitat_sim==0.2.4` 一个版本号完整表达。

独立源码构建模板（在新机器的本地 Git 源码目录，`DEMO_PY` 是刚创建的 `demo-ppu/bin/python` 或 `demo-cuda/bin/python`）：

```bash
: "${DEPS_ROOT:?本地源码依赖根}" "${DEMO_PY:?仿真环境python绝对路径}"
git clone --recursive --branch v0.2.4 https://github.com/facebookresearch/habitat-sim.git "$DEPS_ROOT/habitat-sim"
cd "$DEPS_ROOT/habitat-sim"
git submodule update --init --recursive
HEADLESS=1 WITH_CUDA=0 "$DEMO_PY" -m pip install --no-build-isolation --no-deps .
```

方法来自 [Habitat-Sim 0.2.4 官方源码构建说明](https://raw.githubusercontent.com/facebookresearch/habitat-sim/v0.2.4/BUILD_FROM_SOURCE.md)。`WITH_CUDA=0` 指 Habitat 仿真器，**不关闭单独 3DGS renderer 的 GPU**。Python 3.12 构建使用兼容的 C++/pybind11 子模块，并记录其实际 Git 版本。

Habitat-Lab 使用与 RynnBrain 相容的源码，提供 `habitat` 模块及 Hydra 配置。现有工作区的 Habitat-Lab checkout 为 `094d6be2f9d057e4781a68ae792132895fd4d3d0`，其源码目录为 `agentic_sim/repo/habitat-lab/habitat-lab`；将选定源码 commit 接入独立 Git 依赖后，使用 `"$DEMO_PY" -m pip install --no-build-isolation --no-deps <本地checkout>/habitat-lab`。

## 6. 我们如何适配 PPU 3DGS 渲染

### 6.1 代码与真实适配范围

主仓维护的 renderer 在 `third_party/abot_renderer/`，来源记录为 [UPSTREAM.json](third_party/abot_renderer/UPSTREAM.json)：ABot-Navigation `2a0aefb56f1e2d315bba924239e9e8ad9dca9d92` 的 `render_server/`。

GPU 路径沿用 PyTorch `cuda:X` + `torch.utils.cpp_extension.CUDAExtension`，两个原生扩展为：

- `submodules/diff-plane-rasterization`：高斯投影、分桶、排序与光栅化，导出 `diff_plane_rasterization._C`。
- `submodules/simple-knn`：高斯几何初始化所用近邻计算，导出 `simple_knn._C`。

**我们的已有部署适配是 CUDA 兼容接口 + PPU 编译产物 + 运行库/服务接线**。历史记录明确复用了上游同事编译的 CPython 3.12 PPU `_C.so`，配合 PPU torch 2.9.0/CUDA API 12.9。

本仓新增/维护的是设备选择、HTTP 服务、相机/场景接入、场景生命周期与缓存、H00 静态边界/蓝色伪影处理，以及 CPU 后端。模型与环境仍通过原 `render_gs`/Gym 接口交互，PPU 与 NVIDIA 的业务接口相同。

### 6.2 两个平台各自编译，不共享 `.so`

GPU 扩展可使用与运行环境匹配的现有构建。如需自行编译，将 `ABOT_RENDER_SOURCE` 指向按上节版本完整检出的上游 `render_server`，包括 CUDA 源码、子模块和 GLM 头文件。`DEMO_PY` 指向目标硬件环境。

```bash
: "${ABOT_RENDER_SOURCE:?完整上游 render_server 路径}" "${DEMO_PY:?}" "${WHEEL_DIR:?数据盘环境构建目录}"
# PPU：先执行第3节SDK初始化；NVIDIA：先配置第4节的CUDA Toolkit。
# NVIDIA可按实际GPU设置TORCH_CUDA_ARCH_LIST；不要把sm_80/sm_90当作PPU ISA。
"$DEMO_PY" -m pip wheel --no-build-isolation --no-deps \
  "$ABOT_RENDER_SOURCE/submodules/diff-plane-rasterization" --wheel-dir "$WHEEL_DIR"
"$DEMO_PY" -m pip wheel --no-build-isolation --no-deps \
  "$ABOT_RENDER_SOURCE/submodules/simple-knn" --wheel-dir "$WHEEL_DIR"
"$DEMO_PY" -m pip install --no-index --find-links "$WHEEL_DIR" --no-deps diff-plane-rasterization simple-knn
```

上述命令通过 `CUDAExtension` 重建扩展。PPU 编译器/工具链的额外构建选项以对应 SDK 发行说明为准。历史 CMake 文件中出现的 NVIDIA architecture 列表不应照搬给 PPU。

Python 小版本、torch ABI、SDK/CUDA、设备 ISA 任一变化，都重新编译；`undefined symbol`、`invalid device function` 不能通过重命名文件或复制旧二进制解决。若从源码包目录导入，确认未被残留 `_C.so` 遮蔽新安装 wheel。正常使用以实际模块 `__file__` 记录溯源。

### 6.3 当前统一评测的 CPU 后端

统一主方法已有 CPU renderer，与旧 GPU 3DGS 路径并存：`gaussian_renderer/cpu.py` 负责投影/协方差/SH，`cpu_composite.cpp` 负责按深度合成。启动前 `SAGE_RENDER_DEVICE=cpu`，同时 `--device cpu`；两者必须一致。`SAGE_CPU_RENDER_CACHE` 指向容器本地缓存，首次使用由 g++ 编译，**不需要 PPU/NVIDIA rasterizer `.so`**。

这是推理渲染后端，不是训练用可微替代品。CPU 路径与 PPU 适配要分开理解：统一评测选择 CPU，是资源与并发配方；原 Demo 四路 3DGS 服务仍可走各平台的 GPU renderer。不能因 SAGE 使用 CPU 就删除大脑/FM 的 GPU 依赖。

## 7. 固定源码、模型与路径接线

独立依赖按 Git 检出：

| 依赖 | 当前版本 |
|---|---|
| 本项目 | 当前 ZeroVLN 仓库，用 `git rev-parse HEAD` 记录版本 |
| upstream VERL | 由 `VERL_SOURCE_ROOT` 指定。联合 RL 验证使用 `b213f118f8df230bc97f68cdc56043623c6babdb`，开环 SFT 验证使用 `1633e1f51545b16b4751f86481b9616fcd23cb64` |
| RynnBrain | `https://github.com/alibaba-damo-academy/RynnBrain.git`，`40f19547941454835ff5e63c22994da74605cfea`，使用 `navigation/` |
| GPU renderer | 已维护在本仓 `third_party/abot_renderer/`，来源见上一节 |

当前根目录启动器从 `environment.local.sh` 读取本机配置。按[环境配置](README.md#environment-setup)设置训练解释器、仿真环境和外部资源；多节点任务使用相同的绝对路径。

- 可视化与闭环评测使用 `RYNNBRAIN_VENV`、`RYNNBRAIN_NAV_DIR` 和系统 Mesa。自定义 Mesa 安装可设置 `MESA_ROOT`。模型、训练配置和评测资产通过 `configs/gpu/evaluation.json` 绑定。
- 开环训练使用 `configs/gpu/openloop.json` 中的初始化权重、三源数据与索引。联合 RL 的 `configs/gpu/rloo.yaml` 使用单独的训练清单、Megatron 源码、Ray 地址和 Gym 服务地址。
- 采集使用外部准备的 supervisor plan，其服务命令、运行环境和输入文件需要和目标机器一致。各入口的启动顺序、输出和检查方式见[分链路部署说明](README.md#workflow-deployment)。

## 8. 环境与运行记录

安装后的记录应包含系统/SDK/驱动、pip 元数据、Git commit、扩展构建来源、解释器和模块路径；这属于环境交付证据，不增加正式训练/伴评的 GPU 预检门槛。已有任务保持独立 stdout 心跳、真实进程身份与退出码，失败不自动回退旧镜像或旧源码。
