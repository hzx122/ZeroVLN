# vln_demo 迁移到 GPU 集群的完整依赖清单

> **2026-11-14 当前入口：** [导航可视化安装与操作手册](../docs/2026-11-14_安装并操作VLN导航可视化复现手册.md)集中说明自由Demo、v8/RLOO统一页面与归档回放；[PPU/NVIDIA独立环境及3DGS适配](../REQUIREMENTS.md)不依赖旧镜像或共享venv。下文保留历史实现与实测记录；旧SDK提交、复制venv/源码、旧IP/PID及预检步骤不作为新的正式启动规范。

本文档记录把 `vln_demo` 从本节点(anonymous)迁到另一台 GPU 机器所需的全部依赖：
文件、Python 环境、系统包、模型 ckpt、数据、环境变量、运行命令。**拷贝数据后按本清单对齐即可在新机器上跑通。**

本节点已实测跑通(2026-08-07)：3 episode 全完成，Ep12 success=1.0/SPL=0.71，产出 750 帧 jpg + 3 个 vis mp4 + result.json。MoBot_VLN-2B 模型(2.72B dense)单卡 PPU、软渲染 habitat、ReAct 多轮推理端到端 OK。

---

## 0. 一句话总览

vln_demo = **gordon 的 habitat/RynnBrain 评测 venv(只读复用)** + **系统 mesa 软 EGL 栈(apt 装)** + **PPU 版 vllm(系统装)起一个 VLM 服务** + **本目录的 src/vllm 脚本/lib**。迁移时前三项要在新机器上备齐(或换成等价物)，第四项随 vln_demo 目录一起拷过去。

---

## 1. 目录本体(随拷)

```
/workspace/vln_demo/
├── src/                          # vln_demo.py / vln_free.py / vln_interactive.py /
│                                 #   vln_model.py / agent_tools.py / prompts.py /
│                                 #   webui.py / vln_demo.yaml / chat_template_mobot2b_keep_thinking.jinja
├── vllm/                         # vLLM 启动脚本 + 日志配置
│   ├── vllm-vln-2B.sh            #   2B 后端(GPU13/28898/MoBot_VLN-2B/--chat-template)
│   ├── vllm-vln-35B.sh           #   35B 后端(GPU14,15/28899/Qwen3.6-35B)
│   └── vllm_log_hidden_id.json   #   vLLM logging dictConfig
├── lib/                          # 软渲染 EGL 库(系统未装 mesa 时的兜底, 见 §3)
│   ├── libEGL.so.1(.1.0)         #   glvnd libEGL 1.7.0-1build1
│   ├── libEGL_mesa.so.0(.0.0)    #   mesa 25.2.8-0ubuntu0.24.04.2 EGL ICD
│   ├── libgallium-25.2.8-0ubuntu0.24.04.2.so
│   ├── 50_mesa.json              #   glvnd EGL vendor ICD
│   └── dri/libdril_dri.so        #   mesa 25.2 dril 驱动
├── run_demo.sh                   # 自包含入口(自己起 vLLM, 默认 MODEL_PATH/chat-template 与 vllm-vln-2B.sh 不一致, 不推荐)
├── run_demo_ext.sh               # ★推荐入口: 外部 vLLM + 软渲染, 已验证可跑通
├── free_roam.sh / interactive.sh / model.sh   # 交互/webui 入口(同样复用 gordon venv)
└── README.md                     # 原作者说明
```

**路径已改写**: 原 `local-user-03/vln_demo` → `anonymous/vln_demo`(src/py + 4 个 *.sh + README)。`run_demo_ext.sh` 与 `vllm/vllm-vln-*.sh` 用 `$HERE`/`BASH_SOURCE` 自算目录, 拷到新机器改顶部路径变量即可。

**home 根还放了一个模块**(vllm 脚本依赖):
```
/workspace/vllm_log_filters.py   # vLLM 日志 filter, 被 vllm_log_hidden_id.json 引用
```
vllm 脚本设 `PYTHONPATH=<home根>`, 所以这个 .py 必须在 home 根(或改 PYTHONPATH 指到它所在处)。迁移时一起拷。

---

## 2. Python 环境(只读复用 gordon venv)

```
VENV=/opt/navigation-runtime/venv
```
- python 3.12.3, **--system-site-packages**(venv 自己的包很少, 主要靠系统 site-packages)。
- 关键包: `habitat_sim==0.2.4`、`vllm==0.18.0+ppu2.0.0`(PPU 版)、torch(PPU)。`habitat`(habitat-lab)未装在 venv, 但 `rynnvln_eval` 从 NAV/src 导入(见 §4)。
- venv/lib 里只放了 `libOpenGL.so.0`(habitat 软渲染需要, 系统没有)。
- **迁移**: 新机器若能访问 gordon 这个 venv 路径(共享 cpfs), 直接 `source $VENV/bin/activate` 用即可。若新机器不通 cpfs, 需复制整个 venv(注意 `--system-site-packages` 依赖系统 python3.12 的 vllm/torch, 单拷 venv 不够, 还得对齐系统的 vllm+PPU torch)。

`agent_tools.py:39` 有个 docstring 注释引用 `local-user-03/Agentic_MoBot`(parser 代码出处), 非运行依赖, 不用管。

---

## 3. 系统 apt 包(mesa 软 EGL 栈) ★

habitat-sim 用 **Magnum WindowlessEglApplication + surfaceless 软渲染**(无 /dev/dri、无显示)。本节点**原本缺 libegl1 + libegl-mesa0**, 导致 `eglGetDisplay` 失败。已用 dpkg 装了 mesa `.2` 全套:

```
libegl1          1.7.0-1build1              # glvnd libEGL.so.1
libegl-mesa0     25.2.8-0ubuntu0.24.04.2    # libEGL_mesa.so.0 + 50_mesa.json ICD
mesa-libgallium  25.2.8-0ubuntu0.24.04.2    # libgallium-*.so(软光栅化)
libgl1-mesa-dri  25.2.8-0ubuntu0.24.04.2    # dri/libdril_dri.so(.2 dril 统一驱动)
libgbm1          25.2.8-0ubuntu0.24.04.2    # (依赖)
libglx-mesa0     25.2.8-0ubuntu0.24.04.2    # (一致性)
# 已有(不用动): libglvnd0 1.7.0-1build1, libGL.so.1, libGLX_mesa.so.0
```

**版本必须同 build 后缀(全 .2 或全 .1)**, 否则 libEGL_mesa↔libgallium 的 versioned symbol 对不上(versioned symbol 不匹配, 软链接骗不过)。

**迁移到新机器**: 目标节点若有完整 mesa(多数 GPU 节点有), 直接用; 若没有, apt 装(注意 Ubuntu 24.04 noble 的 noble-updates 镜像 .1 版 deb 已被 .2 替代且 .1 会 404, 直接下 .2 全套):
```bash
# 在目标机器上(需 root):
cd /tmp && mkdir mesa2debs && cd mesa2debs
for p in libegl1 libegl-mesa0 mesa-libgallium libgl1-mesa-dri libgbm1 libglx-mesa0; do
  # libegl1 在 libglvnd 源, 其余在 mesa 源; 版本对齐到同一 build
  curl -fLO "https://service.example.invalid/ubuntu/pool/main/m/mesa/${p}_25.2.8-0ubuntu0.24.04.2_amd64.deb" 2>/dev/null \
    || curl -fLO "https://service.example.invalid/ubuntu/pool/main/libg/libglvnd/${p}_1.7.0-1build1_amd64.deb"
done
dpkg -i *.deb && ldconfig
```
或 `apt-get install -y libegl1 libegl-mesa0 mesa-libgallium libgl1-mesa-dri libgbm1 libglx-mesa0`(让 apt 解版本, 可能挑 .1=404, 那就 dpkg -i 上面下的 .2)。

`vln_demo/lib/` 里也带了一份 .2 的 libEGL/libEGL_mesa/libgallium/dril/ICD(系统没装时的兜底); 但**实测单靠 lib 目录的拼接栈跑不通**(glvnd↔mesa dispatch 需要 /usr/share/glvnd/egl_vendor.d/ 系统目录 + 系统级 matched 栈), 所以**必须系统装 mesa**, lib 目录里的只是冗余备份(LD_LIBRARY_PATH 带上无害)。验证 EGL 是否 OK 的唯一可靠办法是真跑 demo(本节点的 ctypes 探针会假报 BAD_DISPLAY, 别信)。

---

## 4. RynnBrain navigation 代码(只读复用 gordon)

```
NAV=/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation
```
- `src/rynnvln_eval.py` 提供 `VLNEvaluator`(vln_demo 子类化它)。`src/` 还有 args/dataset/habitat_extensions/utils 等。
- `config/` 有 hydra 配置基(`vln_r2r.yaml` 等), vln_demo.yaml 引用 `defaults: /habitat: habitat_config_base ...`。
- vln_demo.py 把 `src/vln_demo.yaml` cp 到 NAV 根, 然后 `cd $NAV && python3 vln_demo.py`(hydra 按 CWD 解析 config 路径)。
- runner 把 `$DATA_ROOT` 软链到 `$NAV/data`(RynnBrain 数据)。
- **迁移**: 通 cpfs 就直接用 gordon 的 NAV(只读); 不通就拷整个 navigation 目录 + 确认 config 里的 `scenes_dir`/`data_path` 相对路径能解析(默认 `data/scene_datasets/` + `data/datasets/r2r/{split}/{split}.json.gz`)。

---

## 5. 数据(RynnBrain R2R)

```
DATA_ROOT=/data/shared/groups/mfm/common/anno/vlmevalkit/RynnBrain/data
  ├─ scene_datasets/      # MP3D 场景(habitat 加载)
  ├─ datasets/r2r/        # R2R val_unseen 等 split 的 episode json.gz
  ├─ check_sim/ robotnav_skeleton_*/  # 其它
```
runner: `ln -sfn $DATA_ROOT $NAV/data`。**迁移**: 数据量大, 拷贝或挂载到新机器同路径, 或改 runner 的 `DATA_ROOT` 变量。

---

## 6. 模型 ckpt(vLLM 服务)

| 后端 | served-model-name | 端口 | GPU | MODEL_PATH |
|------|-------------------|------|-----|------------|
| 2B ★ | MoBot_VLN-2B | 28898 | 13 | `/data/shared/groups/mfm/vl/rl/pretrain_models/final_mix_mt_me6/hf_iter_0001250` |
| 35B  | Qwen3.6-35B    | 28899 | 14,15(TP=2) | `/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B` |

- 2B = MoBot_VLN(final_mix_mt_me6 / hf_iter_0001250), 2.72B dense(Qwen3.5 架构, bf16 权重 5.1GB), **单卡 TP=1**, 是 VLN 任务上真正训过的模型。
- 2B 必须传 `--chat-template`(指向 `vln_demo/src/chat_template_mobot2b_keep_thinking.jinja`, 给 2B 模板补 preserve_thinking 分支, 否则 ReAct 的 keep_thinking 是空操作)。35B 自带模板不用传。
- 两个 served-model-name 用真名(不是 default), 配错会 404 不会静默走错模型。
- run_demo.sh 默认 `MODEL_PATH=.../final_mix_mt_me6_05/...`(带 _05), 与 vllm-vln-2B.sh 的 `final_mix_mt_me6`(不带 _05)不同——**用 run_demo_ext.sh, 别用 run_demo.sh 的默认**。
- **迁移**: ckpt 拷到新机器同路径或改脚本里的 MODEL_PATH。vllm 是 PPU 版(0.18.0+ppu2.0.0, 系统装); 非 PPU 集群(如 NVIDIA)需换对应 vllm+torch 构建, 命令行基本不变。

---

## 7. 环境变量(run_demo_ext.sh 设的全集)

```bash
source $VENV/bin/activate
export LD_LIBRARY_PATH="$HERE/lib:$VENV/lib:$LD_LIBRARY_PATH"   # libEGL/libOpenGL
export EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 \
       MESA_LOADER_DRIVER_OVERRIDE=llvmpipe GALLIUM_DRIVER=llvmpipe \
       HABITAT_SIM_GPU_DEVICE_ID=-1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export RYNNVLN_LANGUAGE=en RYNNVLN_TURN_ANGLE=30 RYNNVLN_HFOV=120 RYNNVLN_ALLOW_SLIDING=1
export RYNNVLN_LOG_RAW_OUTPUT=1 RYNNVLN_VLLM_IMAGE_FORMAT=JPG RYNNVLN_VLLM_IMAGE_QUALITY=90 MAX_NEW_TOKENS=2048
export RYNNVLN_VLLM_SERVER_URLS=http://127.0.0.1:28898   # 2B; 35B 用 28899
export RYNNVLN_VLLM_MODEL_NAME=MoBot_VLN-2B              # 35B 用 Qwen3.6-35B
export RYNNBRAIN_NAV_DIR=$NAV
export MODEL_PATH=...   # 见 §6
export DEMO_OUTPUT_PATH=$OUT
# 选 episode 的开关(可选): DEMO_SCENE / DEMO_EPISODE_IDS / DEMO_NUM_EPISODES / DEMO_MAX_STEPS / DEMO_SPLIT
```
- vllm 脚本额外设 `PYTHONPATH=<home根>`(给 vllm_log_filters.py) + `VLLM_LOGGING_CONFIG_PATH=$HERE/vllm/vllm_log_hidden_id.json`。

---

## 8. 运行命令(已验证)

```bash
# 0) 系统 mesa 装好(§3) + gordon venv / NAV / 数据 / ckpt 在位
# 1) 起 vLLM(后台, 等 /health 200, 约 5min warmup)
bash /workspace/vln_demo/vllm/vllm-vln-2B.sh > /tmp/vllm.log 2>&1 &
# 轮询: while ! curl -s http://127.0.0.1:28898/health; do sleep 5; done

# 2) 跑 demo(3 episode, ~3-6min)
BACKEND=2B bash /workspace/vln_demo/run_demo_ext.sh
# 或 35B: BACKEND=35B bash run_demo_ext.sh (先起 vllm-vln-35B.sh)

# 3) 产出在 DEMO_OUTPUT_PATH(默认 outputs/run_ext_<ts>/):
#    result.json(每集 success/spl/os/ne/steps) + frames/<scene>_<ep>/*.jpg +
#    vis_0/<scene>_<ep>.mp4(官方第一视角+俯视) + check_sim_0/rgb_<ep>.jpg + demo.log/vllm.log

# 4) 跑完停 vLLM 释放 GPU
pkill -f 'vllm serve.*final_mix_mt_me6'
```

---

## 9. 迁移检查表(新机器上逐项对齐)

- [ ] 拷 `vln_demo/` 整目录(含 src/vllm/lib/run_demo_ext.sh)到新路径
- [ ] 拷 `vllm_log_filters.py` 到新 home 根(或改 vllm 脚本 PYTHONPATH)
- [ ] **系统装 mesa .2 全套**(§3), `ldconfig -p | grep libEGL_mesa` 能查到 + `/usr/share/glvnd/egl_vendor.d/50_mesa.json` 存在
- [ ] gordon venv 在位(或等价 habitat+PPU-vllm 环境), `source venv/bin/activate` 后 `python -c "import habitat_sim,vllm"` 不报错
- [ ] RynnBrain NAV 目录在位, `src/rynnvln_eval.py` 可 import
- [ ] RynnBrain 数据(`scene_datasets/` + `datasets/r2r/`)挂到 `$NAV/data`(或改 DATA_ROOT)
- [ ] 模型 ckpt 在位(2B: final_mix_mt_me6; 35B: local-user-02) 或改 MODEL_PATH
- [ ] vllm(PPU 版)可执行; 非 PPU 集群换成对应 vllm 构建
- [ ] 改 `run_demo_ext.sh`/`vllm/vllm-vln-*.sh` 顶部路径变量(VENV/NAV/DATA_ROOT/MODEL_PATH)
- [ ] 起 vLLM → /health 200 → 跑 run_demo_ext.sh → 出 result.json + frames + vis mp4

---

## 10. 已知坑(实测踩过)

1. **libegl1/libegl-mesa0 缺失**: 系统没装 mesa EGL → `eglGetDisplay EGL_BAD_PARAMETER`。必须系统装 .2 全套, lib 目录拼接栈单独不够(见 §3)。
2. **mesa 版本 build 后缀必须全一致**: .2 libegl-mesa0 配 .1 libgallium → versioned symbol 不匹配, 软链接无效。全 .2 或全 .1。
3. **vllm_log_filters.py**: vllm_log_hidden_id.json 引用 `vllm_log_filters.HidePromptTokenIds`, 必须在 PYTHONPATH(local-user-03 原作者的 home 根)。没它 vllm 起不来(Cannot resolve filter)。
4. **MODEL_PATH env**: vln_demo.py `run_demo()` 硬读 `os.environ["MODEL_PATH"]`, runner 必须设(否则 KeyError)。
5. **2B 必须传 --chat-template**: 2B 自带模板无 preserve_thinking 分支, ReAct keep_thinking 会 100% 丢思考。35B 自带不用传。
6. **run_demo.sh 默认 MODEL_PATH 是 me6_05**(带 _05), 与 vllm-vln-2B.sh 的 me6 不同; 用 run_demo_ext.sh。
7. **gordon venv 没有 vllm 命令行入口**(`venv/bin/vllm` 不存在), 但 `import vllm` 解析到系统 `/usr/local/lib/python3.12` 的 vllm(0.18.0+ppu)。vllm 脚本里 `vllm serve` 走系统的 `/usr/local/bin/vllm`。
8. **ctypes 直接调 eglGetDisplay/eglInitialize 会假报 BAD_DISPLAY**(glvnd dispatch 在裸 ctypes 下没正确初始化), 不能用来判断 EGL 是否可用——唯一可靠验证是真跑 demo。
9. vLLM warmup 约 5 分钟(PPU torch.compile), runner 要等 /health。
10. 软渲染不需要 /dev/dri(mesa surfaceless + llvmpipe 纯 CPU), 但需要系统 mesa 栈在位。
