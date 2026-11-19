#!/usr/bin/env python3
"""Streaming FM-head SFT trainer for deployment-shaped nested bundles.

The slow Qwen backbone and visual encoder are frozen in this smoke stage.  The
manifest contains precomputed four-query outputs and visual tokens.  This is
therefore a head-only wiring/data test, not the final DualVLN-style joint query
training run.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.qwen36_fast_slow_action_head import Qwen36FastSlowActionHead


class BundleDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        item = self.rows[index]
        with np.load(item["sample"]) as data:
            result = {
                "query_hidden_states": data["query_hidden"].astype(np.float32),
                "reference_visual_tokens": data["reference_visual_tokens"].astype(np.float32),
                "current_visual_tokens": data["current_visual_tokens"].astype(np.float32),
                "target_actions": data["target_actions"].astype(np.float32),
                "action_mask": data["action_mask"].astype(np.bool_),
            }
        return result


def collate(samples):
    batch = {name: torch.from_numpy(np.stack([sample[name] for sample in samples]))
             for name in samples[0]}
    for prefix in ("reference_visual", "current_visual"):
        batch[prefix + "_mask"] = torch.ones(
            batch[prefix + "_tokens"].shape[:2], dtype=torch.bool)
    return batch


def move(batch, device, dtype):
    return {name: value.to(
                device=device,
                dtype=(torch.bool if value.dtype == torch.bool else dtype),
                non_blocking=True)
            for name, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, dtype, seed, repeats=1, max_batches=0):
    model.eval()
    losses = []
    for repeat in range(repeats):
        for index, batch in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            batch = move(batch, device, dtype)
            generator = torch.Generator(device=device).manual_seed(
                seed + repeat * 1_000_000 + index)
            losses.append(float(
                model(generator=generator, **batch)["action_loss"].float().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else None


def split_rows(rows):
    train = [row for row in rows if row.get("split", "train") == "train"]
    validation = [row for row in rows if row.get("split") == "validation"]
    train_scenes = {row.get("source_scene") for row in train}
    validation_scenes = {row.get("source_scene") for row in validation}
    overlap = (train_scenes & validation_scenes) - {None}
    if overlap:
        raise ValueError(f"scene leakage between train/validation: {sorted(overlap)}")
    if not train:
        raise ValueError("manifest contains no train samples")
    return train, validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-repeats", type=int, default=4,
                        help="Fixed Monte Carlo draws per validation checkpoint.")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="Use small deterministic defaults and require finite parameter updates.")
    args = parser.parse_args()
    if args.smoke:
        args.steps = min(args.steps, 20)
        args.batch_size = min(args.batch_size, 2)
        args.max_train_samples = args.max_train_samples or 8
        args.max_validation_samples = args.max_validation_samples or 4
        args.eval_every = min(args.eval_every, 5)
        args.save_every = 0
    if args.steps <= 0 or args.batch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("steps, batch-size and grad-accum must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest_path = Path(args.manifest).resolve()
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()
            if line.strip()]
    train_rows, validation_rows = split_rows(rows)
    rng = random.Random(args.seed)
    rng.shuffle(train_rows)
    if args.max_train_samples:
        train_rows = train_rows[:args.max_train_samples]
    if args.max_validation_samples:
        validation_rows = validation_rows[:args.max_validation_samples]

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    if device.type == "cpu" and dtype == torch.bfloat16:
        dtype = torch.float32
    model = Qwen36FastSlowActionHead.from_pretrained(
        args.checkpoint, map_location="cpu").to(device=device, dtype=dtype).train()
    first_parameter = next(model.parameters())
    initial_parameter = first_parameter.detach().float().cpu().clone()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, int(args.steps * args.warmup_ratio))

    def lr_scale(step):
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    train_loader = DataLoader(
        BundleDataset(train_rows), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=device.type == "cuda",
        drop_last=False, generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(
        BundleDataset(validation_rows), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate) if validation_rows else None
    iterator = iter(train_loader)
    losses, evaluations = [], []
    optimizer.zero_grad(set_to_none=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fixed_eval_seed = args.seed + 1_000_000
    initial_validation = evaluate(
        model, validation_loader, device, dtype, fixed_eval_seed,
        repeats=args.eval_repeats) if validation_loader else None
    best_validation = (float(initial_validation)
                       if initial_validation is not None else float("inf"))
    if initial_validation is not None:
        print(json.dumps({"step": 0, "validation_loss": initial_validation,
                          "eval_repeats": args.eval_repeats}), flush=True)
    for step in range(args.steps):
        accumulated = 0.0
        for micro in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = move(batch, device, dtype)
            generator = torch.Generator(device=device).manual_seed(
                args.seed + step * args.grad_accum + micro)
            loss = model(generator=generator, **batch)["action_loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step + 1}: {loss}")
            (loss / args.grad_accum).backward()
            accumulated += float(loss.detach().float().cpu())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip).float().cpu())
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        train_loss = accumulated / args.grad_accum
        losses.append(train_loss)
        record = {"step": step + 1, "train_loss": train_loss,
                  "grad_norm": grad_norm, "lr": scheduler.get_last_lr()[0]}
        if validation_loader and ((step + 1) % args.eval_every == 0
                                  or step + 1 == args.steps):
            value = evaluate(
                model, validation_loader, device, dtype, fixed_eval_seed,
                repeats=args.eval_repeats)
            record["validation_loss"] = value
            evaluations.append({"step": step + 1, "loss": value})
            if value is not None and value < best_validation:
                best_validation = value
                model.save_pretrained(output / "best_action_head")
        print(json.dumps(record), flush=True)
        if args.save_every and (step + 1) % args.save_every == 0:
            model.save_pretrained(output / f"checkpoint-{step + 1}" / "action_head")

    model.save_pretrained(output / "last_action_head")
    parameter_delta = float((first_parameter.detach().float().cpu()
                             - initial_parameter).norm())
    report = {
        "schema": "fm_head_sft_report_v1",
        "ok": bool(np.isfinite(losses).all() and parameter_delta > 0),
        "smoke": args.smoke,
        "manifest": str(manifest_path),
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "sampling_unit": "fast_window_uniform",
        "validation_noise_policy": "fixed_across_checkpoints",
        "validation_repeats": args.eval_repeats,
        "initial_validation_loss": initial_validation,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_first_5": float(np.mean(losses[:5])),
        "mean_last_5": float(np.mean(losses[-5:])),
        "best_validation_loss": None if not evaluations else best_validation,
        "parameter_delta_l2": parameter_delta,
        "evaluations": evaluations,
        "boundary": "head-only; four query outputs and visual tokens are precomputed/frozen",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
