# verl/dlc —— 两节点 DLC 部署（3DGS 16 卡 + RL 训练）

把本机 demo（README §7：1 render + 4 bench + trainer 全 localhost）扩成**两节点** DLC：
1 节点 16 PPU 部署 3DGS（4 render × 4 bench = 16 并发）+ 1 节点跑闭环 FM-GRPO trainer。
三层本就纯 HTTP 解耦，跨机只靠 `--bench-url`，天然支持两节点。

## 文件

| 文件 | 作用 |
|---|---|
| `_dlc_common.py` | 4 个共享 helper（safe_name/shell_quote/get_dlc_client/user_vpc）+ 常量（IMAGE/WORKSPACE/HC_DIR）。从 `submit_vln_goal_dlc.py` 内联版抽出，单点维护。 |
| `bench_3dgs_entry.sh` | bench 节点 entry：起 4 render + 16 bench → 写 ready → 等 trainer_done → pkill。 |
| `submit_bench_3dgs_dlc.py` | 提交 bench job（PyTorchJob，1 Worker 16 PPU）。 |
| `fm_grpo_entry.sh` | trainer 节点 entry：poll ready → 跑 `fm_grpo_env_trainer.py` → 写 done。 |
| `submit_fm_grpo_dlc.py` | 提交 trainer job（PyTorchJob，1 Worker 16 PPU，单进程只用 1 卡）。 |

## 运行（两步，顺序提交）

```bash
cd /workspace/ZeroVLN/verl/dlc
# 先提 bench（同 EXPR_NAME），等 ready 文件写出后再提 trainer（同名）
python3 submit_bench_3dgs_dlc.py    --display-name fmgrpo_smoke
python3 submit_fm_grpo_dlc.py       --display-name fmgrpo_smoke
```

**`--display-name` 两 job 必须同名** —— 握手文件按 `EXPR_NAME` 配对
（`bench_ready_<EXPR>.txt` / `trainer_done_<EXPR>`）。

查 job / tail 日志：

```bash
python3 /workspace/dlc_cli/dlc.py list
python3 /workspace/dlc_cli/dlc.py get <job_id>
tail -f /workspace/discussion_logs/bench_3dgs_fmgrpo_smoke.log
tail -f /workspace/discussion_logs/fm_grpo_fmgrpo_smoke.log
```

## 握手机制（跨 pod HTTP，同 VPC）

两 job 同 `user_vpc`（10.230.x.x extended_cidrs）→ trainer pod 直连 bench pod IP:port。
bench 把 `http://<podIP>:8914,...,http://<podIP>:8929` 写进 CPFS 的 `bench_ready_<EXPR>.txt`；
trainer poll 该文件拿到 `--bench-url`；trainer 跑完写 `trainer_done_<EXPR>`，bench poll 到即 pkill 退出。

握手目录（AutoVLA 授权盘，两 pod 都挂了 /data/storage）：
`/data/zerovln/vln_verl/dlc/`

## 训练超参

trainer 默认 = `docs/VLN闭环RL全链路启动手册.md` §4 同款（本机 smoke 跑通的那套）：
`episodes=0U4zEqlqDJ:0 opt_steps=30 group=8 sde=5 sigma=0.3 lr=1e-4 fm_head eval=5`。
`submit_fm_grpo_dlc.py` 全部可 CLI 覆盖（`--opt-steps 100 --lr 5e-5` 等）。

## 已知坑 / 首跑验证

1. **env 首跑需验证**：bench/render 用 gordon venv（CPFS），trainer 用 `.venv_fm`（`--system-site-packages`，
   靠 image 系统 torch）。image `ppu-training:...-zhouyang-0518` 是否能让 SocialNav vendored transformers 4.57.1
   接住系统 torch、FM 模型能否在 PPU `cuda:0` 跑 `generate` —— 首跑看 trainer 日志。
   若 PPU 报 cuda op 不支持，换 image 或 trainer 回本机 GPU 跑（bench 留 DLC）。
2. **pod IP 取 `hostname -I | awk '{print $NF}'`**（照 dlc_cli/submit.sh 同集群已知可用写法）。
3. **trainer 16 卡只用 1**：单进程 2B 模型，节点 shape 固定 16，pool 无更小，故仍请 16（浪费 15 卡，后续多卡分片再上）。
4. **场景名必须匹配 backend**：rs3d 用 3DRS 场景（`0U4zEqlqDJ:0`），传错 → `distance_to_goal=nan`。`curl :8914/spec` 查。
5. **DLC `user_command` 跑在 /bin/sh**：heredoc 只做 `export + exec bash entry`，复杂逻辑全进 bash entry（POSIX 兼容）。

## 不做（超出范围）

- 不改 `fm_grpo_env_trainer.py` / `vln_gym_server.sh` / render server 源码（零改复用）。
- 不动 `submit_vln_goal_dlc.py`（date_mfm goal-point 链路，另一回事）。
- 不做多 trainer 多卡分片。

详见 `docs/VLN闭环RL两节点DLC部署手册.md`。
