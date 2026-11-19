#!/usr/bin/env python3
"""Phase-2 Step-4b: closed-loop FM-GRPO with the vln_demo gym HTTP bench.

The open-loop imitation reward (Step-4) saturates on the SFT'd head and shows
an SDE-train / ODE-deploy gap. This trainer swaps the reward source: instead of
``similarity(traj, expert GT)`` it scores each sampled trajectory by EXECUTING
it in the habitat sim bench (``BenchClient.step_fm``) and reading the env's
terminal outcome (distance-to-goal / success / collision).

Why this should climb where imitation did not:
- Real headroom — the SFT FM head was trained to match expert deltas, NOT to
  minimize distance-to-goal or avoid collisions; the bench exposes that gap.
- Train/deploy closer — the reward is on the EXECUTED trajectory (the array the
  policy produced), so the policy is optimized on the same object that is
  deployed; no imitation-vs-array surrogate.

Loop (per opt step, per episode = one GRPO group):
  reset bench(scene, ep) -> obs (rgb, gps, instruction)         [once]
  cond = FMInfer.generate+forward+ActionFormer -> af (1, H)      [once, no-grad]
  for b in range(group):
    traj = sample_sde(predict_velocity, cond_rep, cfg)           [no-grad]
    bench.reset(scene, ep)  # same start, fair comparison
    _, _, term, trunc, info = bench.step_fm(denorm(traj), normalized=False)
    r = -distance_to_goal + 10*success - collision
  adv = grpo_advantages(rs, group)
  new_lp = recompute_logprob(predict_velocity, cond_rep, roll...)
  loss = -(adv * new_lp).mean(); loss.backward()
  opt.step()

Eval: deterministic ODE (sigma=0, pinned init_noise) traj -> reset -> step_fm ->
env reward; the deployment metric (vs init).

Run (.venv_fm env for FM model, bench server on 8907 — bench 可在另一台机器，
只需 BENCH_URL 指过去):
  cd /workspace/ZeroVLN
  source vln_demo/.venv_fm/bin/activate
  python3 verl/workers/fm_grpo_env_trainer.py \
    --episodes 0U4zEqlqDJ:0 --opt-steps 30 --group-size 8 \
    --num-sde-steps 5 --sigma 0.3 --accum-steps 1 --lr 1e-4 --train-params fm_head \
    --bench-url http://127.0.0.1:8914
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent                 # verl/workers
_BENCH = _HERE.parent / "bench"                          # verl/bench (纯 HTTP client)
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_HERE))                            # fm_policy, fm_sde (同目录)
# SocialNav train_clean: FMInfer.__init__ does `from train import load_model_and_processor`
# (train_clean/train.py). 新仓 vendor：third_party/SocialNav（relative to repo root）。
_SOCIAL = _HERE.parents[2] / "third_party" / "SocialNav"
sys.path.insert(0, str(_SOCIAL / "train_clean"))
sys.path.insert(0, str(_SOCIAL))

# 解耦后全本地：fm_policy / fm_sde 同目录（verl_nav/workers），bench_client 在
# verl_nav/bench（纯 HTTP，零 vln_gym import）。vln_demo 仅作 HTTP bench 服务。
from fm_policy import FMInfer, _denorm_delta, _ZYT  # noqa: E402
from bench_client import BenchClient  # noqa: E402  (单 URL 退化用)
from bench_pool import BenchPool  # noqa: E402  (多 bench 并发分发，照 verl judge_client)
from fm_sde import SDEConfig, sample_sde, recompute_logprob, grpo_advantages  # noqa: E402
from PIL import Image  # noqa: E402

FM_CKPT = ("/data/shared/groups/mfm/vl/exp/qwen3_vl_2b_fm/fm_baseline_fixed/"
           "qwen3vl_2b_fm_0708_V8_tp2_12nodes_mbs8_20260721_102900/checkpoint_tp2_hf_10k")
BENCH_URL = "http://127.0.0.1:8907"

REGIMES = {
    "fm_head": ["flow_matching_model"],
    "fm_head+actionformer": ["flow_matching_model", "actionformer"],
    "fm_head+actionformer+backbone": ["flow_matching_model", "actionformer", "backbone"],
}


def set_train_params(model, regime):
    for p in model.parameters():
        p.requires_grad_(False)
    parts = REGIMES[regime]
    if "flow_matching_model" in parts:
        for p in model.flow_matching_model.parameters():
            p.requires_grad_(True)
    if "actionformer" in parts:
        for p in model.query_multihead_multi_attn.parameters():
            p.requires_grad_(True)
        model.query_action.requires_grad_(True)
    if "backbone" in parts:
        for p in model.model.parameters():
            p.requires_grad_(True)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[params] regime={regime} trainable={n/1e6:.2f}M")
    return n


@torch.no_grad()
def extract_cond(fm: FMInfer, frames, wp_norm, instruction):
    """Mirror FMInfer.predict generate+forward+ActionFormer; return cond af (1, H).

    Reuses FMInfer._build_prompt_ids + the same generate/forward/ActionFormer
    chain so the cond is byte-identical to what the ODE deployment eats. Squeeze
    (1, num_queries=1, H) -> (1, H) for fm_sde's (B, H) contract.
    """
    input_ids, pixel_values, image_grid_thw, wp = fm._build_prompt_ids(
        frames, wp_norm, instruction)
    fm.model.rope_deltas = None
    generated = fm.model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.long),
        pixel_values=pixel_values, image_grid_thw=image_grid_thw,
        input_waypoints=wp, max_new_tokens=64, do_sample=False,
    )
    fm.model.rope_deltas = None
    outputs = fm.model(
        input_ids=generated,
        attention_mask=torch.ones_like(generated, dtype=torch.long),
        pixel_values=pixel_values, image_grid_thw=image_grid_thw,
        train=False, logits_to_keep=0, input_waypoints=wp,
    )
    hidden = outputs.hidden_states
    qa = fm.model.query_action.expand(hidden.shape[0], -1, -1)
    if getattr(fm.model, "query_action_layer", 2) == 1:
        af, _ = fm.model.query_multihead_attn(qa, hidden, hidden)
    else:
        af = fm.model.query_multihead_multi_attn(qa, hidden, hidden)
    af = af.to(hidden.dtype)           # (1, num_queries, H)
    B, Q, H = af.shape
    return af.reshape(B, Q * H)        # (1, H)


def env_reward(info):
    """Shaped reward from bench info. Real headroom: SFT head doesn't minimize
    distance-to-goal and collides; success is rare -> all three terms move."""
    d = info.get("distance_to_goal")
    try:
        d = float(d)
    except (TypeError, ValueError):
        d = 0.0
    if d != d:  # nan（rs3d 某 episode 的 metric 可能 nan）-> 当 0 处理
        d = 0.0
    success = 1.0 if info.get("success") else 0.0
    collided = 1.0 if info.get("collided_this_step") else 0.0
    return -d + 10.0 * success - 1.0 * collided


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def info_breakdown(infos):
    """Mean of success/distance/collision/spl/path_efficiency/arrived_within_2x
    over a list of bench info dicts. Missing/nan fields skipped per-metric
    (rs3d 无 goal 时 spl/success 都 nan -> 不崩，只记 reward/distance 项)。

    path_efficiency = walked / ref_short（≈1 走参考路径；>1 绕远；<1 抄障配 collision 看）。
    arrived_within_2x = distance ≤ 2×arrive_radius 的比例（success 太严时的弱成功信号）。
    """
    def mean(key):
        vs = [_safe_float(i.get(key)) for i in infos]
        vs = [v for v in vs if v == v]  # drop nan
        return sum(vs) / len(vs) if vs else float("nan")
    pes, aw2 = [], []
    for i in infos:
        w, s = _safe_float(i.get("path_length_m")), _safe_float(i.get("shortest_path_m"))
        if w == w and s == s and s > 1e-6:
            pes.append(w / s)
        d, r = _safe_float(i.get("distance_to_goal")), _safe_float(i.get("success_radius"))
        if d == d and r == r:
            aw2.append(1.0 if d <= 2 * r else 0.0)
    return dict(
        success=mean("success"), distance=mean("distance_to_goal"),
        collision=mean("collided_this_step"), spl=mean("spl"),
        path_efficiency=(sum(pes) / len(pes) if pes else float("nan")),
        arrived_within_2x=(sum(aw2) / len(aw2) if aw2 else float("nan")),
    )


def eval_ode_breakdown(fm, bench, eps, cfg_ode, gen, device):
    """ODE σ=0 deploy over eps -> dict of mean reward + breakdown分项。坏 ep reset
    无 rgb / 崩 -> skip（正式 140 ep 必然有坏 ep，robust 跳过不中断评测）。"""
    rewards, infos = [], []
    for (sc, ep) in eps:
        try:
            r, info = eval_ode_env_reward(fm, bench, sc, ep, cfg_ode, gen, device)
            rewards.append(r)
            infos.append(info)
        except Exception as e:
            print(f"  [eval skip {sc}:{ep}] {e}")
            continue
    bd = info_breakdown(infos)
    bd["reward"] = sum(rewards) / len(rewards) if rewards else float("nan")
    return bd


@torch.no_grad()
def eval_ode_env_reward(fm, bench, scene, ep, cfg_ode, gen, device):
    """Deterministic ODE (sigma=0) deployment: reset -> cond -> ODE traj ->
    reset -> step_fm -> env reward."""
    obs, info = bench.reset(scene=scene, episode_id=ep)
    instr = obs.get("instruction") or "walk to the goal"
    frame = Image.fromarray(obs["rgb"]).convert("RGB")
    wp_norm = np.zeros((1, 2), np.float32)  # no history at start
    cond = extract_cond(fm, [frame], wp_norm, instr)
    pin = torch.randn(1, cfg_ode.action_chunk, cfg_ode.action_dim,
                      device=device, generator=gen)
    roll = sample_sde(fm.model.flow_matching_model.predict_velocity, cond,
                      cfg_ode, device=device, generator=gen, init_noise=pin)
    traj = roll["traj"][0].float().cpu().numpy()
    real = _denorm_delta(traj, _ZYT)
    # reset+step 必须同 client（pool.round-robin 下分开调会打到不同 bench）。
    info = bench.reset_step_fm(scene, ep, real.tolist(), normalized=False,
                               render_final_only=True)
    return env_reward(info), info


def _parse_eps(s):
    return [(p.split(":")[0], p.split(":")[1]) for p in s.split(",") if p.strip()]


@torch.no_grad()
def run_episode_ode_multiturn(fm, client, scene, ep, cfg_ode, gen, device,
                              max_chunks=16, stop_on_arrival=True, verbose=False,
                              policy_lock=None, history_len=8):
    """Deterministic ODE 多轮 episode rollout（Phase A：验 SFT 多轮 success 非零）。

    与单轮 eval_ode_env_reward 的本质区别：那个只决策一次、开环执行整条 chunk、读
    终点算 reward，丢弃新观测；这里 **reset 一次后连续多轮交互**——每轮用滚动历史窗口
    重编码 cond → σ=0 ODE 采一条 32×3 chunk → step_fm 全 32 点执行（不 reset）→ 拿
    执行后的新 obs 再进入下一轮，直到到达半径/超时/episode_over。

    到达判定用 info 的 distance_to_goal ≤ success_radius（排除 rs3d 的 nan）。abotn
    的 success 必须**主动 STOP** 才置位（abotn/env.py:437），FM 动作无 STOP，所以到
    半径后由 trainer 代发 step_tool("stop") 注册 success（discussion §3 第一阶段解耦
    STOP）。

    cond 每轮用 **N 帧历史 + N 个 input_waypoint** 重编码（对齐带历史的 SFT：历史位姿
    变换到当前帧 ego 系的相对 xy），由 HistoryBuffer 维护滚动窗口（history_util）。
    history_len=1 时退化回单帧、input_waypoint=当前帧自身（≈零 history），兼容旧 SFT。

    client 必须是单个 BenchClient（reset/step_fm/step_tool 跨步共享 env 状态，不能
    round-robin 换 bench）。多个 episode 并发时共享同一个 fm.model（单卡），GPU 前向
    非线程安全 → policy_lock 串行化 policy 段（extract_cond+sample_sde+gen），bench 的
    HTTP step 留在锁外，仍可跨 episode 重叠。返回 episode 结果 dict。
    """
    import contextlib
    from history_util import HistoryBuffer, world_pose_from_info
    lock = policy_lock if policy_lock is not None else contextlib.nullcontext()

    obs, info = client.reset(scene=scene, episode_id=ep)
    if obs.get("rgb") is None:
        raise RuntimeError("no rgb in reset")
    instr = obs.get("instruction") or "walk to the goal"
    d0 = _safe_float(info.get("distance_to_goal"))
    # 历史窗口：reset 用首帧+首 pose 重复 N 份初始化（对齐 SFT 起步 linspace 补满）。
    # world_pose 缺失（后端没透 world_xy）时回退 (0,0,0)，input_waypoint 全 0 ≈ 旧口径。
    frame0 = Image.fromarray(obs["rgb"]).convert("RGB")
    p0 = world_pose_from_info(info) or (0.0, 0.0, 0.0)
    hist = HistoryBuffer(n=history_len)
    hist.reset(frame0, p0[0], p0[1], p0[2])
    n_chunks = 0
    collided_any = False
    arrived = False
    for t in range(max_chunks):
        if info.get("episode_over"):
            break
        # policy 段串行（GPU 前向 + generator 非线程安全）
        with lock:
            frames = hist.frames()
            wp_norm = hist.input_waypoints_norm()  # N 个历史位姿在当前帧 ego 系
            cond = extract_cond(fm, frames, wp_norm, instr)
            roll = sample_sde(fm.model.flow_matching_model.predict_velocity, cond,
                              cfg_ode, device=device, generator=gen)
            traj = roll["traj"][0].float().cpu().numpy()
        real = _denorm_delta(traj, _ZYT)
        obs, r, term, trunc, info = client.step_fm(real.tolist(), normalized=False,
                                                   render_final_only=True)
        n_chunks += 1
        if info.get("collided_this_step"):
            collided_any = True
        # 新观测入历史窗口（chunk 级：每轮 chunk 末尾一帧一 pose）
        if obs.get("rgb") is not None:
            ft = Image.fromarray(obs["rgb"]).convert("RGB")
            pt = world_pose_from_info(info)
            if pt is not None:
                hist.append(ft, pt[0], pt[1], pt[2])
        d = _safe_float(info.get("distance_to_goal"))
        sr = _safe_float(info.get("success_radius"))
        if verbose:
            print(f"    [chunk {t}] dist={d:.2f} sr={sr:.2f} "
                  f"collided={info.get('collided_this_step')} term={term}", flush=True)
        arrived = (d == d) and (sr == sr) and (d <= sr)
        if arrived and stop_on_arrival:
            # 代发 stop：abotn success 只在 _stopped 且在半径内时置位
            obs, r, term, trunc, info = client.step_tool("stop")
            break
        if term or trunc:
            break
    out = dict(info)
    out["n_chunks"] = n_chunks
    out["dist0"] = d0
    out["collided_any"] = collided_any
    out["arrived"] = bool(arrived)
    return out


def eval_multiturn_breakdown(fm, pool, eps, cfg_ode, gen, device, max_chunks=16,
                             history_len=8):
    """并发跑多轮 episode eval，聚合 info_breakdown + n_chunks 均值。

    每个 episode 独占一个 bench client（map_episodes 保证 1:1），episode 内串行多轮。
    坏 ep（reset 无 rgb / 崩）记 None 跳过，不中断整批。gen 单实例被并发共享——ODE
    σ=0 下 x0 只影响可忽略的浮点尾差，eval 不追求逐位复现，可接受。policy_lock 串行
    化各 episode 的 GPU policy 前向（共享单卡 fm.model，非线程安全），bench HTTP step
    仍跨 episode 重叠。
    """
    import threading
    policy_lock = threading.Lock()

    def _fn(client, sc, ep):
        return run_episode_ode_multiturn(fm, client, sc, ep, cfg_ode, gen, device,
                                         max_chunks=max_chunks, policy_lock=policy_lock,
                                         history_len=history_len)
    results = pool.map_episodes(eps, _fn)
    infos = [r for r in results if r is not None]
    bd = info_breakdown(infos)
    ncs = [r.get("n_chunks") for r in infos if r.get("n_chunks") is not None]
    bd["n_chunks"] = sum(ncs) / len(ncs) if ncs else float("nan")
    bd["n_eval"] = len(infos)
    bd["n_skip"] = len(results) - len(infos)
    return bd


# ============================================================================
# 训练用多轮 episode rollout（SDE 采样 + 逐 chunk 保留 logprob 材料）
# ============================================================================
@torch.no_grad()
def rollout_episode_sde_multiturn(fm, client, scene, ep, cfg, device, gen,
                                  max_chunks=8, history_len=8, stop_on_arrival=True,
                                  policy_lock=None):
    """一条**多轮 episode** 的 SDE 采样 rollout，返回 episode return + 每 chunk 的
    logprob 材料（供 loss 阶段 recompute_logprob 带梯度重算）。

    与 eval 的 run_episode_ode_multiturn 同骨架（reset→HistoryBuffer→多轮 step_fm→
    到半径代发 stop→max_chunks 上限），差别：
    - 用 cfg（sigma>0 随机 SDE）而非 cfg_ode（σ=0 确定）采样，探索出 GRPO 需要的
      轨迹多样性；
    - 每 chunk 保存 (cond, xs, x_nexts, ts, dt, sigmas)——rollout 全程 no_grad，材料
      是 GPU tensor，主线程 loss 阶段对 flow_matching_model.predict_velocity 重算
      logprob（cond detach 当常量，fm_head regime backbone 冻结）。

    episode return（同 env_reward 口径，但 collision 改**累积**）：
        return = -distance_to_goal + 10*success - collided_count
    到半径代发 step_tool("stop") 激活 abotn success（否则 success 恒 0、+10 项死）。

    policy 段（extract_cond+sample_sde+gen）由 policy_lock 串行（共享单卡 fm.model 非
    线程安全），bench HTTP step 留锁外，跨 rollout 重叠。gen 每条 rollout 独占（同起点
    不同轨迹，见 rollout_group_multiturn 的 per-rollout seed）。
    """
    import contextlib
    from history_util import HistoryBuffer, world_pose_from_info
    lock = policy_lock if policy_lock is not None else contextlib.nullcontext()

    obs, info = client.reset(scene=scene, episode_id=ep)
    if obs.get("rgb") is None:
        raise RuntimeError("no rgb in reset")
    instr = obs.get("instruction") or "walk to the goal"
    d0 = _safe_float(info.get("distance_to_goal"))
    frame0 = Image.fromarray(obs["rgb"]).convert("RGB")
    p0 = world_pose_from_info(info) or (0.0, 0.0, 0.0)
    hist = HistoryBuffer(n=history_len)
    hist.reset(frame0, p0[0], p0[1], p0[2])

    chunks = []            # 每 chunk 的 logprob 材料
    collided_count = 0
    arrived = False
    for t in range(max_chunks):
        if info.get("episode_over"):
            break
        with lock:
            frames = hist.frames()
            wp_norm = hist.input_waypoints_norm()
            cond = extract_cond(fm, frames, wp_norm, instr)     # (1,H) no_grad
            roll = sample_sde(fm.model.flow_matching_model.predict_velocity, cond,
                              cfg, device=device, generator=gen)
            traj = roll["traj"][0].float().cpu().numpy()
        real = _denorm_delta(traj, _ZYT)
        obs, r, term, trunc, info = client.step_fm(real.tolist(), normalized=False,
                                                   render_final_only=True)
        # 保留材料（cond detach 当常量；xs/x_nexts/ts/sigmas 供 recompute_logprob）
        chunks.append({"cond": cond.detach(), "xs": roll["xs"],
                       "x_nexts": roll["x_nexts"], "ts": roll["ts"],
                       "dt": roll["dt"], "sigmas": roll["sigmas"]})
        if info.get("collided_this_step"):
            collided_count += 1
        if obs.get("rgb") is not None:
            ft = Image.fromarray(obs["rgb"]).convert("RGB")
            pt = world_pose_from_info(info)
            if pt is not None:
                hist.append(ft, pt[0], pt[1], pt[2])
        d = _safe_float(info.get("distance_to_goal"))
        sr = _safe_float(info.get("success_radius"))
        arrived = (d == d) and (sr == sr) and (d <= sr)
        if arrived and stop_on_arrival:
            obs, r, term, trunc, info = client.step_tool("stop")
            break
        if term or trunc:
            break

    d = _safe_float(info.get("distance_to_goal"))
    if d != d:
        d = 0.0
    success = 1.0 if info.get("success") else 0.0
    episode_return = -d + 10.0 * success - 1.0 * collided_count
    out = {"return": float(episode_return), "chunks": chunks,
           "info": dict(info), "n_chunks": len(chunks), "dist0": d0,
           "collided_count": collided_count, "arrived": bool(arrived)}
    return out


def rollout_group_multiturn(fm, pool, scene, ep, group_size, cfg, device,
                            base_seed, max_chunks=8, history_len=8):
    """一个 GRPO group：同一 (scene,ep) 起点的 group_size 条独立多轮 episode rollout。

    map_episodes 独占 client 并发（env 有状态不能 round-robin），policy_lock 串行 GPU
    前向、bench HTTP 重叠。每条 rollout 独立 generator（base_seed 派生唯一 seed，保证
    同起点采不同轨迹，否则 K 条相关、advantage 退化）。返回 [rollout_dict or None]。
    """
    import threading
    policy_lock = threading.Lock()
    seed_lock = threading.Lock()
    seed_ctr = {"i": 0}

    def _fn(client, sc, e):
        with seed_lock:
            si = seed_ctr["i"]
            seed_ctr["i"] += 1
        gen = torch.Generator(device=device).manual_seed(int(base_seed) * 100003 + si)
        return rollout_episode_sde_multiturn(
            fm, client, sc, e, cfg, device, gen, max_chunks=max_chunks,
            history_len=history_len, policy_lock=policy_lock)

    eps_k = [(scene, ep)] * group_size
    return pool.map_episodes(eps_k, _fn)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=FM_CKPT)
    ap.add_argument("--bench-url", default=BENCH_URL,
                    help="bench HTTP URL；多 bench 并发用逗号分隔（照 verl judge_client "
                         "多 replica + ThreadPool 并发分发，~M× 加速）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--episodes", default="2azQ1b91cZZ:11",
                    help="comma-sep scene:episode_id list (each = a GRPO prompt), train pool")
    ap.add_argument("--val-episodes", default="",
                    help="comma-sep scene:episode_id held-out val list（空=无 val）")
    ap.add_argument("--opt-steps", type=int, default=30)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--num-sde-steps", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--train-params", default="fm_head", choices=list(REGIMES))
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-train-sample", type=int, default=10,
                   help="periodic train eval 抽 N 个 ep 省时（0=全 train eps）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tb-dir", default="")
    ap.add_argument("--eval-multiturn", action="store_true",
                    help="只跑 deterministic ODE 多轮 episode eval（不训练）：验 SFT "
                         "模型多轮闭环 success 是否非零（Phase A）")
    ap.add_argument("--max-chunks", type=int, default=16,
                    help="多轮 episode 每条最多决策轮数（每轮执行全 32 点 chunk ~3m）")
    ap.add_argument("--history-len", type=int, default=8,
                    help="闭环历史窗口 N（8=对齐 SocialNav；1=退化单帧≈零 history 兼容旧 SFT）")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    eps = _parse_eps(args.episodes)
    val_eps = _parse_eps(args.val_episodes)
    print(f"[env] device={device} regime={args.train_params} "
          f"train_eps={len(eps)} val_eps={len(val_eps)}")

    bench_urls = [u.strip() for u in args.bench_url.split(",") if u.strip()]
    bench = BenchPool(bench_urls, timeout=300)
    print(f"[bench] pool size={len(bench_urls)} urls={bench_urls} health={bench.health()}")

    fm = FMInfer(ckpt=args.ckpt, device=device)
    set_train_params(fm.model, args.train_params)
    chunk, dim = fm.action_chunk, fm.action_dim

    cfg = SDEConfig(action_chunk=chunk, action_dim=dim,
                    num_steps=args.num_sde_steps, sigma=args.sigma)
    cfg_ode = SDEConfig(action_chunk=chunk, action_dim=dim,
                        num_steps=args.num_sde_steps, sigma=0.0)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    # ---- Phase A：多轮 episode eval（验 SFT 多轮 success 非零，不训练）----
    if args.eval_multiturn:
        writer = None
        if args.tb_dir:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(args.tb_dir)
        # 抽子集省时（复用 --eval-train-sample 语义）
        eval_eps = eps
        if args.eval_train_sample and 0 < args.eval_train_sample < len(eps):
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(eps), size=args.eval_train_sample, replace=False)
            eval_eps = [eps[i] for i in idx]
        print(f"[eval-multiturn] episodes={len(eval_eps)} max_chunks={args.max_chunks} "
              f"benches={len(bench_urls)}")
        bd = eval_multiturn_breakdown(fm, bench, eval_eps, cfg_ode, gen, device,
                                      max_chunks=args.max_chunks,
                                      history_len=args.history_len)
        print(f"[eval-multiturn] n_eval={bd['n_eval']} skip={bd['n_skip']} "
              f"success={bd['success']:.3f} dist={bd['distance']:.2f} "
              f"spl={bd['spl']:.3f} n_chunks={bd['n_chunks']:.1f} "
              f"collision={bd['collision']:.3f} path_eff={bd['path_efficiency']:.3f} "
              f"arr2x={bd['arrived_within_2x']:.3f}")
        if writer:
            for k, val in bd.items():
                if isinstance(val, (int, float)):
                    writer.add_scalar(f"eval_multiturn/{k}", val, 0)
        if val_eps:
            vbd = eval_multiturn_breakdown(fm, bench, val_eps, cfg_ode, gen, device,
                                           max_chunks=args.max_chunks,
                                           history_len=args.history_len)
            print(f"[eval-multiturn-val] n_eval={vbd['n_eval']} skip={vbd['n_skip']} "
                  f"success={vbd['success']:.3f} dist={vbd['distance']:.2f} "
                  f"spl={vbd['spl']:.3f} n_chunks={vbd['n_chunks']:.1f} "
                  f"arr2x={vbd['arrived_within_2x']:.3f}")
            if writer:
                for k, val in vbd.items():
                    if isinstance(val, (int, float)):
                        writer.add_scalar(f"eval_multiturn_val/{k}", val, 0)
        if writer:
            writer.close()
        bench.close()
        return 0

    # 探测有效 ep（reset 有 rgb）。多轮 rollout 每轮 cond 随滚动历史变化，不能像单轮
    # 那样预缓存（cond 只在 backbone 冻结且**单帧零历史**时才是常量；多轮历史下每步不同）。
    valid_eps = []  # 跳过 reset 无 rgb 的坏 ep（正式 140 ep 必然有）
    print(f"[probe] checking {len(eps)} episodes reset...")
    for (sc, ep) in eps:
        try:
            obs, info = bench.reset(scene=sc, episode_id=ep)
            if obs.get("rgb") is None:
                print(f"  [{sc}:{ep}] SKIP: no rgb in reset")
                continue
            valid_eps.append((sc, ep))
            print(f"  [{sc}:{ep}] OK dist0={info.get('distance_to_goal')}")
        except Exception as e:
            print(f"  [{sc}:{ep}] SKIP: {e}")
            continue
    eps = valid_eps
    n_eps = len(eps)
    print(f"[probe] valid episodes = {n_eps}")

    # periodic train eval 子集（固定 seed，每 eval 同一批，省时）
    if args.eval_train_sample and 0 < args.eval_train_sample < len(eps):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(eps), size=args.eval_train_sample, replace=False)
        eval_train_eps = [eps[i] for i in idx]
    else:
        eval_train_eps = list(eps)
    print(f"[eval] train-subset={len(eval_train_eps)} val={len(val_eps)} every={args.eval_every}")

    trainable = [p for p in fm.model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr)

    writer = None
    if args.tb_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.tb_dir)

    # init ODE deploy reward（全 train eps，r0 稳定）+ val init breakdown
    r0bd = eval_ode_breakdown(fm, bench, eps, cfg_ode, gen, device)
    r0 = float(r0bd["reward"])
    print(f"[init] ODE-deploy train reward = {r0:.4f} success={r0bd['success']:.3f} "
          f"dist={r0bd['distance']:.2f} spl={r0bd['spl']:.3f}")
    if writer:
        writer.add_scalar("eval/train_reward", r0, 0)
    if val_eps:
        v0 = eval_ode_breakdown(fm, bench, val_eps, cfg_ode, gen, device)
        print(f"[init-val] reward={v0['reward']:.3f} success={v0['success']:.3f} "
              f"dist={v0['distance']:.2f} spl={v0['spl']:.3f}")
        if writer:
            for k, val in v0.items():
                writer.add_scalar(f"val/{k}", val, 0)

    n_eps = len(eps)
    pidx = 0
    for step in range(args.opt_steps):
        opt.zero_grad(set_to_none=True)
        step_returns, step_rstds, step_losses = [], [], []
        step_infos = []      # accum 内全部 episode 终态 info，step 末合一算分项
        step_nchunks = []
        t_rollout = t_loss = 0.0
        last_adv = None
        for _ in range(args.accum_steps):
            (sc, ep) = eps[pidx % n_eps]
            pidx += 1
            # 多轮 rollout：K 条同起点 episode（SDE 采样，每条内部多轮交互，收集每
            # chunk 的 logprob 材料 + episode return）。cond 每轮随历史实时算，不预缓存。
            t0 = time.time()
            rollouts = rollout_group_multiturn(
                fm, bench, sc, ep, args.group_size, cfg, device,
                base_seed=args.seed * 1000 + step * max(1, args.accum_steps) + pidx,
                max_chunks=args.max_chunks, history_len=args.history_len)
            t_rollout += time.time() - t0
            rollouts = [r for r in rollouts if r is not None]
            if len(rollouts) < 2:
                print(f"[step {step}] <2 valid rollouts ({len(rollouts)}), skip accum", flush=True)
                continue
            returns = torch.tensor([r["return"] for r in rollouts], device=device,
                                   dtype=torch.float32)
            # 整组归一化（K' 条当一个 group）：episode return 级 advantage
            adv = grpo_advantages(returns, group_size=len(rollouts))
            last_adv = adv
            step_returns.append(returns.mean().item())
            step_rstds.append(returns.std().item() if returns.numel() > 1 else 0.0)
            step_infos.extend(r["info"] for r in rollouts)
            step_nchunks.extend(r["n_chunks"] for r in rollouts)
            # episode advantage 归因到每条 rollout 的**每个 chunk** 的 logprob（multi-step
            # RL 标准做法）。除 total_chunks 让 opt.step 梯度量级与 episode 长度解耦。
            total_chunks = sum(len(r["chunks"]) for r in rollouts) or 1
            t0 = time.time()
            for k, ro in enumerate(rollouts):
                for ch in ro["chunks"]:
                    new_lp = recompute_logprob(
                        fm.model.flow_matching_model.predict_velocity, ch["cond"],
                        ch["xs"], ch["x_nexts"], ch["ts"], ch["dt"],
                        ch["sigmas"], cfg.var_floor)
                    loss = -(adv[k].detach() * new_lp).mean() / total_chunks / args.accum_steps
                    loss.backward()
                    step_losses.append(loss.item() * total_chunks * args.accum_steps)
            t_loss += time.time() - t0

        gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        opt.step()
        mr = float(np.mean(step_returns)) if step_returns else float("nan")
        bd = info_breakdown(step_infos)
        if writer:
            writer.add_scalar("train/env_reward", mr, step)   # 现在是 episode return 均值
            writer.add_scalar("train/reward_std",
                              float(np.mean(step_rstds)) if step_rstds else 0.0, step)
            writer.add_scalar("train/success_rate", bd["success"], step)
            writer.add_scalar("train/distance_to_goal", bd["distance"], step)
            writer.add_scalar("train/collision_rate", bd["collision"], step)
            writer.add_scalar("train/n_chunks",
                              float(np.mean(step_nchunks)) if step_nchunks else 0.0, step)
            writer.add_scalar("train/loss",
                              float(np.mean(step_losses)) if step_losses else 0.0, step)
            if last_adv is not None:
                writer.add_scalar("train/adv_mean", float(last_adv.mean()), step)
                writer.add_scalar("train/adv_std", float(last_adv.std()), step)
            writer.add_scalar("train/grad_norm", float(gnorm), step)
            writer.add_scalar("time/rollout", t_rollout, step)
            writer.add_scalar("time/loss", t_loss, step)
            writer.add_scalar("time/step_total", t_rollout + t_loss, step)
        if step % args.eval_every == 0 or step == args.opt_steps - 1:
            tbd = eval_ode_breakdown(fm, bench, eval_train_eps, cfg_ode, gen, device)
            vstr = ""
            if writer:
                for k, val in tbd.items():
                    writer.add_scalar(f"eval/train_{k}", val, step + 1)
            if val_eps:
                vbd = eval_ode_breakdown(fm, bench, val_eps, cfg_ode, gen, device)
                if writer:
                    for k, val in vbd.items():
                        writer.add_scalar(f"val/{k}", val, step + 1)
                vstr = (f" val_r={vbd['reward']:.3f} val_succ={vbd['success']:.3f} "
                        f"val_dist={vbd['distance']:.2f} val_spl={vbd['spl']:.3f}")
            print(f"[step {step:3d}] train_return(episode)={mr:.3f} "
                  f"grad_norm={float(gnorm):.2f} "
                  f"ODE_train={tbd['reward']:.3f}{vstr}")

    rfin = eval_ode_breakdown(fm, bench, eps, cfg_ode, gen, device)
    rf = float(rfin["reward"])
    print(f"\n[eval-ODE] init={r0:.4f} -> final={rf:.4f} (delta={rf - r0:+.4f}) "
          f"success={rfin['success']:.3f} dist={rfin['distance']:.2f} spl={rfin['spl']:.3f}")
    if writer:
        writer.add_scalar("eval/train_reward", rf, args.opt_steps + 1)
        if val_eps:
            vfin = eval_ode_breakdown(fm, bench, val_eps, cfg_ode, gen, device)
            for k, val in vfin.items():
                writer.add_scalar(f"val/{k}", val, args.opt_steps + 1)
            print(f"[eval-ODE-val] reward={vfin['reward']:.3f} success={vfin['success']:.3f} "
                  f"dist={vfin['distance']:.2f} spl={vfin['spl']:.3f} "
                  f"path_eff={vfin['path_efficiency']:.3f} arr2x={vfin['arrived_within_2x']:.3f}")
        writer.close()
    improved = rf > r0 + 0.05
    print(f"\n{'='*60}\nFM-GRPO CLOSED-LOOP ENV ({args.train_params}): "
          f"{'ODE DEPLOY REWARD IMPROVED ✓' if improved else 'NO IMPROVEMENT ✗'}\n{'='*60}")
    bench.close()
    # 干净跑完即 exit 0（改进与否只进 TB banner，不进退出码——避免 DLC 把"没改进"误标 Failed）。
    return 0


if __name__ == "__main__":
    sys.exit(main())
