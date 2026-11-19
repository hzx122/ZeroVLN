# -*- coding: utf-8 -*-
"""verify_fm — FM 管线正确性强验证（正控制）。

两路对照，把"sane range"升级为"管线正确"：

A. forward/ODE 对齐：用 MinimalDataset 产出的 data_dict（input_ids/pixel_values/
   image_grid_thw/input_waypoints），同 torch 种子下跑 (i) infer_batch._predict_fm_batch
   (ref) 与 (ii) 本模块内联的 forward+ODE，对比逐元素。→ 验我的 forward/ODE 与
   SocialNav 参考实现逐一致。

B. 端到端 + prompt fidelity + delta 语义：用样本原图+input_waypoints+指令跑我的
   fm.predict（我自己构的 prompt），(1) 解码 generate tokens 验是否产出
   "goal point: [x,y]; <flow_matching_policy>"（prompt 对齐）；(2) denorm 我的输出
   → 真实 delta，跟样本 gt_waypoints（真实 delta，32×3）比 per-step/末点误差。
   → 验 prompt→模型→输出→denorm→delta 语义整条对（含 delta-vs-cumsum：若模型
   输出本是累积位置而非 delta，denorm 后与 GT-delta 对不上，误差爆炸）。
"""
import os
import sys
import io
import json
import base64

_HERE = os.path.dirname(os.path.abspath(__file__))
_VLN_SRC = os.path.dirname(_HERE)
_SOCIAL = "/workspace/date_mfm/projects/MegaTE/vla/reference/SocialNav"
# fm_policy 已搬到 verl_nav/workers（解耦）；dev 验证脚本跨指 verl_nav。
_VERL_NAV = "/workspace/date_mfm/projects/MegaTE/vla/verl_nav"
for p in (_VLN_SRC, _SOCIAL, os.path.join(_SOCIAL, "train_clean"),
          os.path.join(_VERL_NAV, "workers"), os.path.join(_VERL_NAV, "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from PIL import Image

from fm_policy import FMInfer, _denorm_delta, _norm_input_wp, _ZYT, FM_CKPT  # verl_nav/workers

JSONL = ("/data/shared/groups/mfm/vl/anno/inner/2609_vln/benchmark/0709/"
         "wgoal_0624-benchmark_frames_expand_new_p1_full_all_with_noise.jsonl")


def _load_sample(idx=0):
    with open(JSONL) as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise RuntimeError("sample not found")


@torch.no_grad()
def my_forward_ode(model, input_ids, attention_mask, pixel_values,
                    image_grid_thw, input_waypoints, gen_max=64, seed=0):
    """内联 fm_policy 的 generate→forward→ActionFormer→ODE（同 FMInfer.predict 但接受
    外部 input_ids，用于 A 的对齐：data_dict 的 input_ids 直接喂进来）。"""
    prompt_len = input_ids.shape[1]
    g = torch.Generator(device="cpu")
    # generate 用模型内部 RNG；ODE 的 randn 用 torch.randn，受 torch.manual_seed 控制
    torch.manual_seed(seed)
    model.rope_deltas = None
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        input_waypoints=input_waypoints,
        max_new_tokens=gen_max, do_sample=False)
    gen_ids = generated[0, prompt_len:].tolist()
    torch.manual_seed(seed)  # ODE 前重置，保证 randn 可复现
    model.rope_deltas = None
    outputs = model(input_ids=generated,
                    attention_mask=torch.ones_like(generated, dtype=torch.long),
                    pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                    train=False, logits_to_keep=0, input_waypoints=input_waypoints)
    hidden = outputs.hidden_states
    qa = model.query_action.expand(hidden.shape[0], -1, -1)
    if getattr(model, "query_action_layer", 2) == 1:
        af = model.query_multihead_attn(qa, hidden, hidden)
    else:
        af = model.query_multihead_multi_attn(qa, hidden, hidden)
    af = af.to(hidden.dtype)
    B = 1
    dt = 1.0 / int(getattr(model, "num_flow_steps", 3))
    ac, ad = int(getattr(model, "action_chunk", 32)), int(getattr(model, "action_dim", 3))
    traj = torch.randn(B, ac, ad, device=hidden.device, dtype=torch.float32)
    for i in range(int(getattr(model, "num_flow_steps", 3))):
        t = torch.full((B,), i * dt, device=hidden.device, dtype=torch.float32)
        v = model.flow_matching_model.predict_velocity(af, traj.to(hidden.dtype), t)
        traj = traj + v.float() * dt
    return traj[0].float().cpu().numpy(), gen_ids


def main():
    fm = FMInfer()
    model, processor = fm.model, fm.processor
    device = fm.device
    from data import MinimalDataset, MinimalDataConfig
    from infer_batch import _collate_batch, _predict_fm_batch

    ds = MinimalDataset(JSONL, processor, model_id="Qwen3VL",
                        config=MinimalDataConfig(), max_samples=1)
    print(f"[verify] dataset records: {len(ds.records)}")
    data_dict = ds[0]
    raw = _load_sample(0)

    # ---- A. forward/ODE 对齐（手搓 batch，绕开 streaming-only 的 _collate_batch）----
    pad_id = processor.tokenizer.pad_token_id or 0
    plen = int(data_dict["inference_prompt_len"])
    prompt_ids = data_dict["input_ids"][:plen].unsqueeze(0).to(device).long()
    batch = {
        "B": 1,
        "prompt_ids": prompt_ids,
        "prompt_mask": torch.ones_like(prompt_ids, dtype=torch.long),
        "max_prompt_len": plen,
        "pixel_values": data_dict["pixel_values"].to(device),
        "image_grid_thw": data_dict["image_grid_thw"].to(device),
        "input_waypoints": data_dict["input_waypoints"].unsqueeze(0).to(device),
        "decode_max_new_tokens": 64,
    }
    from infer_batch import _predict_fm_batch
    torch.manual_seed(0)
    ref_norm_list, _ = _predict_fm_batch(model, batch, device, fm.num_flow_steps)
    ref_norm = ref_norm_list[0]
    my_norm_a, _ = my_forward_ode(
        model, batch["prompt_ids"], batch["prompt_mask"],
        batch["pixel_values"], batch["image_grid_thw"], batch["input_waypoints"],
        gen_max=64, seed=0)
    # 同种子再跑一次我的，验可复现（ODE 随机性唯一来源）
    torch.manual_seed(0)
    my_norm_a2, _ = my_forward_ode(
        model, batch["prompt_ids"], batch["prompt_mask"],
        batch["pixel_values"], batch["image_grid_thw"], batch["input_waypoints"],
        gen_max=64, seed=0)
    diff_a = np.abs(ref_norm - my_norm_a)
    diff_det = np.abs(my_norm_a - my_norm_a2)
    print("\n===== A. forward/ODE 对齐（同 data_dict、同 seed）=====")
    print(f"ref_norm  range[{ref_norm.min():.4f},{ref_norm.max():.4f}]")
    print(f"my_norm_a range[{my_norm_a.min():.4f},{my_norm_a.max():.4f}]")
    print(f"我 vs ref  逐元素 max|diff|={diff_a.max():.6e} mean={diff_a.mean():.6e} "
          f"allclose(1e-4)={np.allclose(ref_norm, my_norm_a, atol=1e-4)}")
    print(f"我可复现性  max|diff|={diff_det.max():.6e} "
          f"(应≈0；非0 则差异来自 RNG/非确定)")

    # ---- B. 端到端 + prompt + delta 语义 ----
    print("\n===== B. 端到端（我的 prompt）+ generate 解码 + GT 对比 =====")
    images = [Image.open(e["path"]).convert("RGB") for e in raw["image"]]
    instr = raw["conversations"][0]["value"]
    # 抽 "anonymous Instruction: ..." 后的指令
    import re
    m = re.search(r"anonymous Instruction:\s*(.+?)\n", instr)
    instr_text = m.group(1).strip() if m else instr[:80]
    print(f"instruction: {instr_text!r}")
    input_wp_real = np.array(raw["input_waypoints"], np.float32)  # (8,2) 真实
    input_wp_norm = _norm_input_wp(input_wp_real, _ZYT)
    torch.manual_seed(0)
    my_norm_b, gen_ids = fm.predict(images, input_wp_norm, instr_text)
    # 解码 generate tokens
    gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
    print(f"generate 解码: {gen_text!r}")
    has_fm_token = "<flow_matching_policy>" in gen_text or \
        any(processor.tokenizer.convert_tokens_to_ids("<flow_matching_policy>") == t
            for t in gen_ids)
    print(f"  含 <flow_matching_policy> token? {has_fm_token}")
    # denorm → 真实 delta
    my_real = _denorm_delta(my_norm_b, _ZYT)  # (32,3) 真实 delta
    gt_real = np.array(raw["gt_waypoints"], np.float32)  # (32,3) 真实 delta
    err = np.abs(my_real - gt_real)
    # cumsum 后末点误差（导航意义）
    my_cum = np.cumsum(my_real, axis=0)
    gt_cum = np.cumsum(gt_real, axis=0)
    ne = np.linalg.norm(my_cum[-1, :2] - gt_cum[-1, :2])  # xy 末点 NE
    print(f"my_real range dx[{my_real[:,0].min():.4f},{my_real[:,0].max():.4f}] "
          f"dy[{my_real[:,1].min():.4f},{my_real[:,1].max():.4f}] "
          f"dyaw[{my_real[:,2].min():.4f},{my_real[:,2].max():.4f}]")
    print(f"gt_real range dx[{gt_real[:,0].min():.4f},{gt_real[:,0].max():.4f}] "
          f"dy[{gt_real[:,1].min():.4f},{gt_real[:,1].max():.4f}] "
          f"dyaw[{gt_real[:,2].min():.4f},{gt_real[:,2].max():.4f}]")
    print(f"per-step delta 误差: mean={err.mean():.5f} max={err.max():.5f} (米/弧度)")
    print(f"cumsum 末点 NE(xy)={ne:.4f} m  (gt 末点 xy={gt_cum[-1,:2].tolist()})")
    # 若模型输出本是累积位置（非 delta），denorm 后量级会随步数累积远超 gt 的 per-step
    print(f"delta-vs-累积 诊断: my_real 末步量级={np.abs(my_real[-1]).tolist()} "
          f"应与 gt per-step 同量级(~0.01m/~0.01rad)；若 my 末步远大则输出是累积位置→喂 bench 错")

    out = {
        "A_forward_ode_parity": {
            "max_abs_diff": float(diff_a.max()),
            "mean_abs_diff": float(diff_a.mean()),
            "allclose_1e4": bool(np.allclose(ref_norm, my_norm_a, atol=1e-4)),
        },
        "B_end_to_end": {
            "instruction": instr_text,
            "generate_decoded": gen_text[:200],
            "has_fm_token": bool(has_fm_token),
            "my_norm_range": [float(my_norm_b.min()), float(my_norm_b.max())],
            "per_step_delta_err_mean": float(err.mean()),
            "per_step_delta_err_max": float(err.max()),
            "cumsum_end_ne_xy_m": float(ne),
            "gt_end_xy": gt_cum[-1, :2].tolist(),
            "my_last_step_abs": np.abs(my_real[-1]).tolist(),
        },
    }
    out_dir = os.path.join(os.path.dirname(_VLN_SRC), "outputs", "verl_access",
                           __import__("time").strftime("%Y%m%d"))
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, "verify_fm.jsonl")
    with open(fn, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[verify] saved {fn}")


if __name__ == "__main__":
    main()
