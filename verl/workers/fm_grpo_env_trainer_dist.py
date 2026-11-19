#!/usr/bin/env python3
"""Phase-2 闭环 FM-GRPO 多卡分布式 trainer（torchrun 多进程 + 手动梯度 all_reduce）。

rank-aware 版 fm_grpo_env_trainer.py：每 rank 持本地 FMInfer(cuda:LOCAL_RANK)，
GRPO group 的 K 条 rollout 按 contiguous 切到 W rank（K_r=K//W），每 rank 跑自己
K_r 条（policy_lock 串行本 rank GPU 前向，跨 rank 并行）→ all_gather 全组 returns
→ 各 rank 算同一 advantage → 各 rank 对自己 chunks recompute_logprob+backward（grad
累积）→ all_reduce(grad, SUM) → clip → opt.step（同梯度同更新，参数保持同步）。

on-policy 正确性：rollout 用当前 θ（各 rank 参数由上一步 opt.step 同步保持一致）；
advantage 在全组 return 级（all_gather 后各 rank 同结果）；loss 分母用
global_total_chunks（all_reduce 后各 rank 同结果）→ 各 rank 本地 grad 之和 = 全局
grad（all_reduce SUM 还原），无重复计数。不套 DDP wrapper（VLM generate() no_grad
自回归与 DDP forward-hook 交互需 find_unused_params 绕路；手动 all_reduce 更直接）。

单卡退路：WORLD_SIZE 未设/==1 → world=1 rank=0，行为等价单进程 trainer（走本文件
sharded 逻辑，W=1 退化）。原 fm_grpo_env_trainer.py 保持单卡 reference 不动。

启动（torchrun，8+8 本地 / 1+1 DLC）：
  torchrun --nproc_per_node=8 verl/workers/fm_grpo_env_trainer_dist.py \
    --episodes ... --bench-url http://...,http://... --opt-steps 200 --group-size 8 \
    --fm-ckpt <sft_hist_ckpt>
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

_HERE = Path(__file__).resolve().parent                 # verl/workers
_BENCH = _HERE.parent / "bench"
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_HERE))
_SOCIAL = _HERE.parents[2] / "third_party" / "SocialNav"
sys.path.insert(0, str(_SOCIAL / "train_clean"))
sys.path.insert(0, str(_SOCIAL))

# 复用原 trainer 的所有 rank-无关 helper（extract_cond/env_reward/eval_ode_*/rollout_*）。
from fm_grpo_env_trainer import (  # noqa: E402
    FM_CKPT, BENCH_URL, REGIMES, set_train_params,
    env_reward, _safe_float, info_breakdown,
    eval_ode_env_reward, rollout_episode_sde_multiturn, _parse_eps,
)
from fm_policy import FMInfer, _denorm_delta, _ZYT  # noqa: E402
from bench_pool import BenchPool  # noqa: E402
from fm_sde import SDEConfig, sample_sde, recompute_logprob, grpo_advantages  # noqa: E402


def _is_dist() -> bool:
    return dist.is_available() and os.environ.get("WORLD_SIZE", "1") != "1"


def _gather_object(obj):
    """all_gather_object 跨 rank 收 list，flatten；W=1 直接返 [obj]。"""
    if not _is_dist():
        return [obj]
    world = dist.get_world_size()
    gathered = [None] * world
    dist.all_gather_object(gathered, obj)
    return gathered


def rollout_group_dist(fm, pool, scene, ep, k_r, cfg, device, base_seed,
                       seed_offset, max_chunks=8, history_len=8):
    """一个 GRPO group 的**本 rank 分片**：跑 k_r 条同起点多轮 episode（SDE 采样）。

    与原 rollout_group_multiturn 差异：只跑 K_r 条（本 rank 分片），per-rollout
    seed = base_seed*100003 + seed_offset + si（seed_offset=rank*K_r，保证全组 K 条
    跨 rank 种子唯一，否则 K 条相关、advantage 退化）。policy_lock 串行本 rank GPU
    前向，bench HTTP 重叠。返回 [rollout_dict or None]（保序，长度 k_r）。
    """
    import threading
    policy_lock = threading.Lock()
    seed_lock = threading.Lock()
    seed_ctr = {"i": 0}

    def _fn(client, sc, e):
        with seed_lock:
            si = seed_ctr["i"]
            seed_ctr["i"] += 1
        gen = torch.Generator(device=device).manual_seed(
            int(base_seed) * 100003 + int(seed_offset) + si)
        return rollout_episode_sde_multiturn(
            fm, client, sc, e, cfg, device, gen, max_chunks=max_chunks,
            history_len=history_len, policy_lock=policy_lock)

    eps_k = [(scene, ep)] * k_r
    return pool.map_episodes(eps_k, _fn)


def eval_ode_breakdown_dist(fm, bench, eps, cfg_ode, gen, device, rank, world):
    """ODE σ=0 deploy eval，episode 跨 rank strided 分片 + all_gather_object 聚合。
    坏 ep skip。各 rank 跑 eps[rank::world]，gather 后各 rank 同结果；rank0 聚合写 TB。"""
    my_eps = eps[rank::world] if world > 1 else eps
    rewards, infos = [], []
    for (sc, ep) in my_eps:
        try:
            r, info = eval_ode_env_reward(fm, bench, sc, ep, cfg_ode, gen, device)
            rewards.append(r)
            infos.append(info)
        except Exception as e:
            print(f"  [eval skip {sc}:{ep}] {e}")
            continue
    all_r = _gather_object(rewards)
    all_i = _gather_object(infos)
    flat_r = [r for sub in all_r if sub is not None for r in sub]
    flat_i = [i for sub in all_i if sub is not None for i in sub]
    bd = info_breakdown(flat_i)
    bd["reward"] = sum(flat_r) / len(flat_r) if flat_r else float("nan")
    return bd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=FM_CKPT)
    ap.add_argument("--fm-ckpt", default="", help="SFT hist ckpt（alias，覆盖 --ckpt）")
    ap.add_argument("--bench-url", default=BENCH_URL)
    ap.add_argument("--device", default="cuda:0", help="W=1 单卡用；torchrun 下由 LOCAL_RANK 决定")
    ap.add_argument("--episodes", default="2azQ1b91cZZ:11")
    ap.add_argument("--val-episodes", default="")
    ap.add_argument("--opt-steps", type=int, default=30)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--num-sde-steps", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--train-params", default="fm_head", choices=list(REGIMES))
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-train-sample", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tb-dir", default="")
    ap.add_argument("--max-chunks", type=int, default=8)
    ap.add_argument("--history-len", type=int, default=8)
    args = ap.parse_args()

    # ---- 分布式 init ----
    if _is_dist():
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, world, local_rank = 0, 1, 0
        device = args.device
    is_rank0 = (rank == 0)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    eps = _parse_eps(args.episodes)
    val_eps = _parse_eps(args.val_episodes)
    if is_rank0:
        print(f"[env] device={device} rank={rank}/{world} regime={args.train_params} "
              f"train_eps={len(eps)} val_eps={len(val_eps)}", flush=True)

    # ---- bench URL strided 分片（每 rank 拿 [rank::world]，约束 #url >= world）----
    bench_urls = [u.strip() for u in args.bench_url.split(",") if u.strip()]
    if world > 1:
        if len(bench_urls) < world:
            if is_rank0:
                print(f"[ERROR] bench urls {len(bench_urls)} < world {world}; need >= world", flush=True)
            return 1
        my_urls = bench_urls[rank::world]
    else:
        my_urls = bench_urls
    bench = BenchPool(my_urls, timeout=300)
    if is_rank0:
        print(f"[bench] world={world} total_urls={len(bench_urls)} per_rank_urls={my_urls}", flush=True)

    ckpt = args.fm_ckpt or args.ckpt
    fm = FMInfer(ckpt=ckpt, device=device)
    set_train_params(fm.model, args.train_params)
    chunk, dim = fm.action_chunk, fm.action_dim
    cfg = SDEConfig(action_chunk=chunk, action_dim=dim,
                    num_steps=args.num_sde_steps, sigma=args.sigma)
    cfg_ode = SDEConfig(action_chunk=chunk, action_dim=dim,
                        num_steps=args.num_sde_steps, sigma=0.0)
    gen = torch.Generator(device=device).manual_seed(args.seed + rank * 1000)

    # ---- group_size 必须是 world 的倍数（contiguous 切分；K_r=K//W 每 rank 条数）----
    K = args.group_size
    if K % world != 0:
        K_eff = (K // world) * world
        if K_eff < world:
            if is_rank0:
                print(f"[ERROR] group_size {K} < world {world}; set group_size>=world", flush=True)
            return 1
        if is_rank0:
            print(f"[warn] group_size {K} not multiple of world {world}, truncate to {K_eff}", flush=True)
        K = K_eff
    K_r = K // world
    if is_rank0:
        print(f"[grpo] group_size={K} per-rank K_r={K_r} max_chunks={args.max_chunks} "
              f"history={args.history_len}", flush=True)

    # ---- probe 有效 ep（strided 分片 + all_gather 保序）----
    # no-rgb 重试一次（8s）：预加载后通常一次成；重试兜底 render 短暂抖动（abotn_mc
    # 曾因冷启动惊群 reset 无 rgb 误弃 ep → init eval 样本不足 → reward=nan）。
    valid_pairs = []
    for i in range(rank, len(eps), world):
        sc, ep = eps[i]
        obs = None
        info = {}
        for attempt in (0, 1):
            try:
                obs, info = bench.reset(scene=sc, episode_id=ep)
            except Exception as e:
                if attempt == 0:
                    time.sleep(8); continue
                if is_rank0:
                    print(f"  [probe {sc}:{ep}] SKIP: {e}")
                obs = None
                break
            if obs.get("rgb") is not None:
                break
            if attempt == 0:
                time.sleep(8); continue
            if is_rank0:
                print(f"  [probe {sc}:{ep}] SKIP: no rgb (after retry)")
            obs = None
            break
        if obs is None or obs.get("rgb") is None:
            continue
        valid_pairs.append((i, (sc, ep)))
        if is_rank0:
            print(f"  [probe {sc}:{ep}] OK dist0={info.get('distance_to_goal')}")
    all_valid = _gather_object(valid_pairs)
    merged = sorted([x for sub in all_valid if sub is not None for x in sub], key=lambda t: t[0])
    eps = [t[1] for t in merged]
    n_eps = len(eps)
    if is_rank0:
        print(f"[probe] valid episodes = {n_eps}", flush=True)

    # periodic train eval 子集（固定 seed，各 rank 同结果）
    if args.eval_train_sample and 0 < args.eval_train_sample < len(eps):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(eps), size=args.eval_train_sample, replace=False)
        eval_train_eps = [eps[i] for i in idx]
    else:
        eval_train_eps = list(eps)
    if is_rank0:
        print(f"[eval] train-subset={len(eval_train_eps)} val={len(val_eps)} every={args.eval_every}", flush=True)

    trainable = [p for p in fm.model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr)

    writer = None
    if is_rank0 and args.tb_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.tb_dir)

    # ---- init ODE deploy reward（sharded eval，各 rank 同结果）----
    r0bd = eval_ode_breakdown_dist(fm, bench, eps, cfg_ode, gen, device, rank, world)
    r0 = float(r0bd["reward"])
    if is_rank0:
        print(f"[init] ODE-deploy train reward = {r0:.4f} success={r0bd['success']:.3f} "
              f"dist={r0bd['distance']:.2f} spl={r0bd['spl']:.3f}", flush=True)
        if writer:
            writer.add_scalar("eval/train_reward", r0, 0)
    if val_eps:
        v0 = eval_ode_breakdown_dist(fm, bench, val_eps, cfg_ode, gen, device, rank, world)
        if is_rank0:
            print(f"[init-val] reward={v0['reward']:.3f} success={v0['success']:.3f} "
                  f"dist={v0['distance']:.2f} spl={v0['spl']:.3f}", flush=True)
            if writer:
                for k, val in v0.items():
                    writer.add_scalar(f"val/{k}", val, 0)

    pidx = 0
    for step in range(args.opt_steps):
        opt.zero_grad(set_to_none=True)
        step_returns, step_rstds, step_losses = [], [], []
        step_infos = []
        step_nchunks = []
        t_rollout = t_loss = 0.0
        last_adv = None
        for _ in range(args.accum_steps):
            (sc, ep) = eps[pidx % n_eps]
            pidx += 1
            t0 = time.time()
            rollouts = rollout_group_dist(
                fm, bench, sc, ep, K_r, cfg, device,
                base_seed=args.seed * 1000 + step * max(1, args.accum_steps) + pidx,
                seed_offset=rank * K_r,
                max_chunks=args.max_chunks, history_len=args.history_len)
            t_rollout += time.time() - t0
            valid_rs = [r for r in rollouts if r is not None]
            if len(valid_rs) < 1:
                if is_rank0:
                    print(f"[step {step}] rank{rank} no valid rollouts, skip accum", flush=True)
                # 仍需 all_gather 形状一致：占位零 return
                local_returns = torch.zeros(K_r, device=device, dtype=torch.float32)
                chunks_local = [[] for _ in range(K_r)]
            else:
                rmean = float(np.mean([r["return"] for r in valid_rs]))
                returns_local = []
                chunks_local = []
                for r in rollouts:
                    if r is None:
                        returns_local.append(rmean)
                        chunks_local.append([])
                    else:
                        returns_local.append(r["return"])
                        chunks_local.append(r["chunks"])
                local_returns = torch.tensor(returns_local, device=device, dtype=torch.float32)
            # all_gather 全组 returns（各 rank 同结果）
            if _is_dist():
                gathered = [torch.zeros_like(local_returns) for _ in range(world)]
                dist.all_gather(gathered, local_returns)
                full_returns = torch.cat(gathered)
            else:
                full_returns = local_returns
            full_adv = grpo_advantages(full_returns, group_size=K)
            local_adv = full_adv[rank * K_r:(rank + 1) * K_r].detach()
            last_adv = full_adv
            step_returns.append(float(local_returns.mean().item()))
            step_rstds.append(float(local_returns.std().item()) if local_returns.numel() > 1 else 0.0)
            step_infos.extend(r["info"] for r in valid_rs)
            step_nchunks.extend(len(c) for c in chunks_local)
            # global_total_chunks（all_reduce SUM）
            local_tc = torch.tensor([float(sum(len(c) for c in chunks_local))],
                                    device=device, dtype=torch.float32)
            if _is_dist():
                dist.all_reduce(local_tc, op=dist.ReduceOp.SUM)
            global_tc = float(local_tc.item()) or 1.0
            t0 = time.time()
            for k, chunks in enumerate(chunks_local):
                for ch in chunks:
                    new_lp = recompute_logprob(
                        fm.model.flow_matching_model.predict_velocity, ch["cond"],
                        ch["xs"], ch["x_nexts"], ch["ts"], ch["dt"],
                        ch["sigmas"], cfg.var_floor)
                    loss = -(local_adv[k] * new_lp).mean() / global_tc / args.accum_steps
                    loss.backward()
                    step_losses.append(loss.item() * global_tc * args.accum_steps)
            t_loss += time.time() - t0

        # all_reduce grads（SUM → 全局 grad；各 rank 同结果）
        if _is_dist():
            for p in trainable:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        opt.step()
        if is_rank0:
            mr = float(np.mean(step_returns)) if step_returns else float("nan")
            bd = info_breakdown(step_infos)
            if writer:
                writer.add_scalar("train/env_reward", mr, step)
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
            do_eval = (step % args.eval_every == 0 or step == args.opt_steps - 1)
            if do_eval:
                tbd = eval_ode_breakdown_dist(fm, bench, eval_train_eps, cfg_ode, gen, device, rank, world)
                vbd = (eval_ode_breakdown_dist(fm, bench, val_eps, cfg_ode, gen, device, rank, world)
                       if val_eps else None)
                if writer:
                    for k, val in tbd.items():
                        writer.add_scalar(f"eval/train_{k}", val, step + 1)
                    if vbd:
                        for k, val in vbd.items():
                            writer.add_scalar(f"val/{k}", val, step + 1)
                vstr = ""
                if vbd:
                    vstr = (f" val_r={vbd['reward']:.3f} val_succ={vbd['success']:.3f} "
                            f"val_dist={vbd['distance']:.2f} val_spl={vbd['spl']:.3f}")
                print(f"[step {step:3d}] train_return(episode)={mr:.3f} "
                      f"grad_norm={float(gnorm):.2f} "
                      f"ODE_train={tbd['reward']:.3f}{vstr}", flush=True)
        elif _is_dist():
            # 非 rank0 也跑 eval（sharded 需要所有 rank 参与），但不写 TB/不打印
            if step % args.eval_every == 0 or step == args.opt_steps - 1:
                eval_ode_breakdown_dist(fm, bench, eval_train_eps, cfg_ode, gen, device, rank, world)
                if val_eps:
                    eval_ode_breakdown_dist(fm, bench, val_eps, cfg_ode, gen, device, rank, world)

    # ---- final eval（各 rank 同结果，rank0 写 TB/打印）----
    rfin = eval_ode_breakdown_dist(fm, bench, eps, cfg_ode, gen, device, rank, world)
    rf = float(rfin["reward"])
    vfin = (eval_ode_breakdown_dist(fm, bench, val_eps, cfg_ode, gen, device, rank, world)
            if val_eps else None)
    improved = rf > r0 + 0.05
    if is_rank0:
        print(f"\n[eval-ODE] init={r0:.4f} -> final={rf:.4f} (delta={rf - r0:+.4f}) "
              f"success={rfin['success']:.3f} dist={rfin['distance']:.2f} spl={rfin['spl']:.3f}", flush=True)
        if writer:
            writer.add_scalar("eval/train_reward", rf, args.opt_steps + 1)
            if vfin:
                for k, val in vfin.items():
                    writer.add_scalar(f"val/{k}", val, args.opt_steps + 1)
                print(f"[eval-ODE-val] reward={vfin['reward']:.3f} success={vfin['success']:.3f} "
                      f"dist={vfin['distance']:.2f} spl={vfin['spl']:.3f}", flush=True)
            writer.close()
        print(f"\n{'='*60}\nFM-GRPO DIST CLOSED-LOOP ({args.train_params}, W={world}): "
              f"{'ODE DEPLOY REWARD IMPROVED ✓' if improved else 'NO IMPROVEMENT ✗'}\n{'='*60}", flush=True)
    bench.close()
    if _is_dist():
        dist.barrier()
    # 干净跑完即 exit 0（改进与否只进 TB banner，不进退出码——避免 DLC 把"没改进"误标 Failed）。
    # 真正的崩溃走未捕获异常自然非零退出；上方配置错误（url<world/group<world）显式 return 1 保留。
    return 0


if __name__ == "__main__":
    sys.exit(main())
