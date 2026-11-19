# -*- coding: utf-8 -*-
"""fm_policy — 真 FM head 模型（Qwen3VLForConditionalGeneration + flow_matching_policy）
经 bench HTTP 执行的闭环/单发推理（用例 4/5）。

复用（不重写）SocialNav 的 FM 推理链路：
  - train_clean/train.py:load_model_and_processor（加载 0729 hf_16k + forked transformers）
  - train_clean/data.py:_ZYT_WP_STATS + _normalize_input_waypoints + denormalize_and_cumsum 的反归一化部分
  - train_clean/infer_batch.py:_predict_fm_batch 的 generate→forward→ActionFormer→ODE 段（照搬，B=1）

与 SocialNav 离线 infer_batch 的差异：输入不是 jsonl 文件，而是 gym 每步的内存 PIL 帧 + 指令 +
由 gps 增量反推的 input_waypoints 历史。prompt 构造镜像 jsonl 训练格式：
  <|im_start|>user\n<image>×N <input_waypoint>×N anonymous Instruction: {instr}
  Predict the goal point and trajectory.<|im_end|>\n<|im_start|>assistant\n
模型 generate 出 "goal point: [x,y]; <flow_matching_policy>"，再 forward 拿 hidden_states →
ActionFormer → Euler ODE → (32,3) 归一化 delta。denorm 用 anonymous stats（**不是 bench 硬编码
的 CE stats**——开放风险 #3 实锤），喂 bench 时 normalized=False 绕开 bench 的 CE。

⚠ 开放风险（文档点名，本用例即验之）：
  #1 真 FM head 输出未端到端验过；#2 ego 方向约定（dx前/dy左/dyaw左转ccw）未交叉验证；
  #3 归一化 stats（anonymous vs bench 的 CE）；delta-vs-cumsum 语义。闭环从 reset 起历史为空，
  FM 训练时带 8 帧 history + 8 input_waypoint——零 history 下模型是否退化，是本用例的发现。
"""
import os
import sys
import io
import json
import math
import base64
import argparse

# 解耦后路径：verl/bench（纯 HTTP bench_client，零 vln_gym import）+ SocialNav（新仓 third_party vendor）
_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE) + os.sep + "bench"   # verl/bench
_SOCIAL = os.path.dirname(os.path.dirname(_HERE)) + os.sep + "third_party" + os.sep + "SocialNav"  # 新仓 vendor
for p in (_BENCH, _HERE, _SOCIAL + os.sep + "train_clean", _SOCIAL):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

# 纯 HTTP bench client（verl_nav/bench/bench_client.py，零 vln_gym import ——
# vln_demo bench 服务可整体搬到另一台 GPU 服务器，仅靠 HTTP 交互）
from bench_client import BenchClient  # noqa: E402

FM_CKPT = ("/data/shared/groups/mfm/vl/exp/qwen3_vl_2b_fm/fm_baseline_fixed/"
           "qwen3vl_2b_fm_0729-baseline-cocofix-0p5-align_tp2_16nodes_mbs8_"
           "20260802_012647/checkpoint_tp2_hf_16k")

# SocialNav token 常量（src/constants.py）
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IMAGE_TOKEN = "<image>"            # jsonl 里用的占位
VISION_START = "<|vision_start|>"
IMAGE_PAD = "<|image_pad|>"
VISION_END = "<|vision_end|>"
WP_TOKEN = "<input_waypoint>"      # 历史轨迹点占位
FM_TOKEN = "<flow_matching_policy>"

# prompt 变体（与 verl/sft/build_abotn_sft_data._PROMPT_VARIANT 同款，env 切换；详见该文件注释）。
# legacy(默认)= 历史混合体结尾 `Predict the goal point and trajectory.`；base_c= 对齐 0729
# 变体 C 结尾 `Predict the goal point and keep moving.`。SFT 数据与部署须同 PROMPT_VARIANT，
# 否则 train/deploy prompt 错配。gpt 两变体均裸 <flow_matching_policy>（变体 A/C 的 goal-point
# 文本因画布约定未验而暂不加，见 build_abotn_sft_data._PROMPT_ENDINGS 注释）。
_PROMPT_VARIANT = os.environ.get("PROMPT_VARIANT", "legacy").lower()
_PROMPT_ENDINGS = {
    "legacy": "\nPredict the goal point and trajectory.",
    "base_c": "\nPredict the goal point and keep moving.",
}

# anonymous stats（data.py:_ZYT_WP_STATS，FM 实际训练用；非 bench 的 CE）
_ZYT = {
    "dx":   {"q01": -0.10273411870002747, "q99":  0.1031249612569809},
    "dy":   {"q01": -0.10309220850467682, "q99":  0.10309189558029175},
    "dyaw": {"q01": -0.5668500065803528,  "q99":  0.5747644901275635},
}
# bench 硬编码的 CE stats（fm_action.py），用于量化开放风险 #3 的漂移
_CE = {
    "dx":   {"q01": -0.10273411870002747, "q99":  0.1031249612569809},
    "dy":   {"q01": -0.10309220850467682, "q99":  0.10309189558049175},
    "dyaw": {"q01": -0.5668500065803528,  "q99":  0.5747644901275635},
}


def _denorm_delta(norm_deltas, stats):
    """(32,3) 归一化 delta -> 真实米/弧度 delta（不 cumsum，bench 要相对 delta）。

    quantile 反归一化（同 data.py denormalize_and_cumsum 的反归一化部分）：
        real = (norm + 1)/2 * (q99 - q01) + q01
    """
    q01 = np.array([stats["dx"]["q01"], stats["dy"]["q01"], stats["dyaw"]["q01"]], np.float32)
    q99 = np.array([stats["dx"]["q99"], stats["dy"]["q99"], stats["dyaw"]["q99"]], np.float32)
    return (np.asarray(norm_deltas, np.float32) + 1.0) / 2.0 * (q99 - q01) + q01


def _norm_input_wp(wp_xy, stats):
    """(N,2) 米 -> 归一化（同 data.py:_normalize_input_waypoints，按 dx/dy range 缩放）。"""
    rx = abs(stats["dx"]["q99"] - stats["dx"]["q01"])
    ry = abs(stats["dy"]["q99"] - stats["dy"]["q01"])
    arr = np.asarray(wp_xy, np.float32).copy()
    arr[:, 0] /= (rx + 1e-8)
    arr[:, 1] /= (ry + 1e-8)
    return arr


class FMInfer:
    """加载一次模型，多次推理。device=cuda, bf16。"""

    def __init__(self, ckpt=FM_CKPT, device="cuda"):
        from train import load_model_and_processor  # SocialNav/train_clean/train.py
        print(f"[fm] loading {ckpt}", flush=True)
        self.device = device
        self.model, self.processor = load_model_and_processor(
            ckpt, device=device, bf16=True)
        self.model.eval()
        self.num_flow_steps = int(getattr(self.model, "num_flow_steps", 3))
        self.action_chunk = int(getattr(self.model, "action_chunk", 32))
        self.action_dim = int(getattr(self.model, "action_dim", 3))
        print(f"[fm] loaded: action_chunk={self.action_chunk} action_dim={self.action_dim} "
              f"num_flow_steps={self.num_flow_steps}", flush=True)

    def _build_prompt_ids(self, frames, input_wp_norm, instruction):
        """镜像 MinimalDataset.__getitem__ 构造 prompt input_ids + pixel_values。

        frames: List[PIL.Image]（1 或 8）
        input_wp_norm: (N,2) 归一化历史轨迹点（N == len(frames)，零 history 传 zeros）
        """
        n = len(frames)
        # content 镜像 jsonl: <image>×N <input_waypoint>×N anonymous Instruction: ...\nPredict...
        # 结尾随 PROMPT_VARIANT 切换（legacy/base_c），与 SFT 数据同款。
        ending = _PROMPT_ENDINGS.get(_PROMPT_VARIANT, _PROMPT_ENDINGS["legacy"])
        content = (IMAGE_TOKEN * n + " " + WP_TOKEN * n +
                   f" anonymous Instruction: {instruction}{ending}")
        # <image> -> <|vision_start|><|image_pad|><|vision_end|>（同 replace_image_tokens）
        content = content.replace(IMAGE_TOKEN,
                                  f"{VISION_START}{IMAGE_PAD}{VISION_END}")
        user_text = f"{IM_START}user\n{content}{IM_END}\n{IM_START}assistant\n"
        inputs = self.processor(text=[user_text], images=frames,
                                do_resize=False, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)
        # input_waypoints: (1, N, 2)
        wp = torch.tensor(np.asarray(input_wp_norm), dtype=torch.float32,
                          device=self.device).unsqueeze(0)
        return input_ids, pixel_values, image_grid_thw, wp

    @torch.no_grad()
    def predict(self, frames, input_wp_norm, instruction, max_new_tokens=64):
        """返回 (32,3) 归一化 delta ndarray。"""
        input_ids, pixel_values, image_grid_thw, wp = self._build_prompt_ids(
            frames, input_wp_norm, instruction)
        prompt_len = input_ids.shape[1]
        # 1. generate（greedy，出 goal point + <flow_matching_policy>）
        self.model.rope_deltas = None
        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.long),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_waypoints=wp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        gen_ids = generated[0, prompt_len:].tolist()
        # 2. forward 拿 hidden_states
        self.model.rope_deltas = None
        outputs = self.model(
            input_ids=generated,
            attention_mask=torch.ones_like(generated, dtype=torch.long),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            train=False,
            logits_to_keep=0,
            input_waypoints=wp,
        )
        hidden = outputs.hidden_states  # (1, L, H)
        # 3. ActionFormer
        qa = self.model.query_action.expand(hidden.shape[0], -1, -1)
        if getattr(self.model, "query_action_layer", 2) == 1:
            af = self.model.query_multihead_attn(qa, hidden, hidden)
        else:
            af = self.model.query_multihead_multi_attn(qa, hidden, hidden)
        af = af.to(hidden.dtype)
        # 4. Euler ODE
        B = 1
        dt = 1.0 / self.num_flow_steps
        traj = torch.randn(B, self.action_chunk, self.action_dim,
                           device=self.device, dtype=torch.float32)
        for i in range(self.num_flow_steps):
            t = torch.full((B,), i * dt, device=self.device, dtype=torch.float32)
            v = self.model.flow_matching_model.predict_velocity(
                af, traj.to(hidden.dtype), t)
            traj = traj + v.float() * dt
        return traj[0].float().cpu().numpy(), gen_ids


def _save(out_dir, name, obj):
    fn = os.path.join(out_dir, f"{name}.jsonl")
    with open(fn, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[fm] saved {fn}")


def run_fm_case(case, scene="2azQ1b91cZZ", episode_id="11", closed_loop=False):
    from PIL import Image
    import time
    bench = BenchClient(os.environ.get("VLN_BENCH_URL", "http://127.0.0.1:8907"))
    out_dir = os.path.join(os.path.dirname(_VLN_SRC), "outputs", "verl_access",
                           time.strftime("%Y%m%d"))
    os.makedirs(out_dir, exist_ok=True)
    fm = FMInfer()

    if case == 4:
        print("\n===== 用例 4: FM 真模型单发 + denorm/range =====")
        obs, info = bench.reset(scene=scene, episode_id=episode_id)
        rgb = obs.get("rgb")
        instr = obs.get("instruction") or "walk to the goal"
        frame = Image.fromarray(rgb).convert("RGB")
        # 零 history：1 帧 + 1 个零 input_waypoint
        wp_zero = _norm_input_wp(np.zeros((1, 2), np.float32), _ZYT)
        norm_delta, gen_ids = fm.predict([frame], wp_zero, instr)
        real_zyt = _denorm_delta(norm_delta, _ZYT)
        real_ce = _denorm_delta(norm_delta, _CE)
        drift = np.abs(real_zyt - real_ce)
        rec = {
            "case": 4, "scene": scene, "episode_id": episode_id,
            "instruction": instr,
            "gen_token_ids_head": gen_ids[:20],
            "norm_delta_shape": list(norm_delta.shape),
            "norm_delta_range": {"min": float(norm_delta.min()),
                                  "max": float(norm_delta.max()),
                                  "mean_abs": float(np.abs(norm_delta).mean())},
            "real_zyt_range": {"dx": [float(real_zyt[:, 0].min()), float(real_zyt[:, 0].max())],
                               "dy": [float(real_zyt[:, 1].min()), float(real_zyt[:, 1].max())],
                               "dyaw_rad": [float(real_zyt[:, 2].min()), float(real_zyt[:, 2].max())]},
            "real_ce_range": {"dx": [float(real_ce[:, 0].min()), float(real_ce[:, 0].max())],
                               "dy": [float(real_ce[:, 1].min()), float(real_ce[:, 1].max())],
                               "dyaw_rad": [float(real_ce[:, 2].min()), float(real_ce[:, 2].max())]},
            "ce_vs_zyt_drift_mean": {"dx": float(drift[:, 0].mean()),
                                     "dy": float(drift[:, 1].mean()),
                                     "dyaw": float(drift[:, 2].mean())},
            "note": "anonymous=FM 实际训练 stats；CE=bench 硬编码 stats（开放风险#3漂移）;"
                    "喂 bench normalized=False 绕开 CE",
        }
        # 喂 bench 验可执行
        obs2, r, term, trunc, info2 = bench.step_fm(
            real_zyt.tolist(), normalized=False)
        rec["executed"] = {"reward": r, "terminated": term, "truncated": trunc,
                           "collided": info2.get("collided_this_step"),
                           "waypoints_executed": len(info2.get("waypoint_log") or []),
                           "dist_0": info.get("distance_to_goal"),
                           "dist_1": info2.get("distance_to_goal")}
        print(f"[case4] norm range [{rec['norm_delta_range']['min']:.3f},"
              f"{rec['norm_delta_range']['max']:.3f}] mean_abs="
              f"{rec['norm_delta_range']['mean_abs']:.3f}")
        print(f"[case4] anonymous real dx[{real_zyt[:,0].min():.3f},{real_zyt[:,0].max():.3f}] "
              f"dy[{real_zyt[:,1].min():.3f},{real_zyt[:,1].max():.3f}] "
              f"dyaw[{real_zyt[:,2].min():.3f},{real_zyt[:,2].max():.3f}]rad")
        print(f"[case4] CE vs anonymous drift(mean) dx={drift[:,0].mean():.5f} "
              f"dy={drift[:,1].mean():.5f} dyaw={drift[:,2].mean():.5f}")
        print(f"[case4] bench executed {rec['executed']['waypoints_executed']}/32 "
              f"collided={rec['executed']['collided']} dist "
              f"{rec['executed']['dist_0']}->{rec['executed']['dist_1']}")
        _save(out_dir, "case4_fm_single", rec)
        bench.close()
        return rec

    if case == 5:
        print("\n===== 用例 5: FM 真模型闭环 =====")
        max_chunks = 8
        traj = {"case": 5, "scene": scene, "episode_id": episode_id,
                "steps": [], "final": {}, "why": ""}
        obs, info = bench.reset(scene=scene, episode_id=episode_id)
        instr = obs.get("instruction") or "walk to the goal"
        history_frames = [Image.fromarray(obs["rgb"]).convert("RGB")]
        history_gps = [info.get("gps")]
        prev_gps = info.get("gps")
        for ci in range(max_chunks):
            if info.get("episode_over"):
                traj["why"] = "episode_over"; break
            # 构造 input_waypoints：历史 gps 相对当前帧的 ego 位移（取最近 N 个）
            N = min(len(history_frames), 8)
            wp_xy = []
            for g in history_gps[-N:]:
                if g is None or prev_gps is None:
                    wp_xy.append([0.0, 0.0])
                else:
                    wp_xy.append([g[0] - prev_gps[0], g[1] - prev_gps[1]])
            wp_xy = np.array(wp_xy, np.float32)
            wp_norm = _norm_input_wp(wp_xy, _ZYT)
            frames = history_frames[-N:]
            norm_delta, gen_ids = fm.predict(frames, wp_norm, instr)
            real = _denorm_delta(norm_delta, _ZYT)
            nonzero = bool(np.abs(norm_delta).max() > 1e-3)
            obs, r, term, trunc, info = bench.step_fm(
                real.tolist(), normalized=False)
            traj["steps"].append({
                "chunk_i": ci, "n_frames": N,
                "norm_delta_flat": norm_delta.flatten().tolist()[:12],
                "norm_range": [float(norm_delta.min()), float(norm_delta.max())],
                "nonzero": nonzero, "gen_head": gen_ids[:10],
                "reward": r, "terminated": term, "truncated": trunc,
                "collided": info.get("collided_this_step"),
                "distance_to_goal": info.get("distance_to_goal"),
                "gps": info.get("gps"), "yaw_deg": info.get("yaw_deg")})
            print(f"[case5] chunk{ci}: norm[{norm_delta.min():.2f},{norm_delta.max():.2f}] "
                  f"nonzero={nonzero} dist={info.get('distance_to_goal')} "
                  f"collided={info.get('collided_this_step')} term={term}")
            history_frames.append(Image.fromarray(obs["rgb"]).convert("RGB"))
            history_gps.append(info.get("gps"))
            prev_gps = info.get("gps")
            if term or trunc:
                traj["why"] = "terminated/truncated"; break
        traj["final"] = {k: info.get(k) for k in
                         ("success", "spl", "distance_to_goal", "steps_used",
                          "gps", "yaw_deg", "episode_over")}
        _save(out_dir, "case5_fm_closed", traj)
        bench.close()
        return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, choices=[4, 5], required=True)
    ap.add_argument("--scene", default="2azQ1b91cZZ")
    ap.add_argument("--episode_id", default="11")
    args = ap.parse_args()
    run_fm_case(args.case, args.scene, args.episode_id,
                closed_loop=(args.case == 5))


if __name__ == "__main__":
    main()
