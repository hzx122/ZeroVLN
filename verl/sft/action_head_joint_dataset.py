"""Anchor-grouped dataset/collator shared by single- and multi-node SFT."""
from __future__ import annotations

import json
import hashlib
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from sft.fm_training_seed import fm_training_seed
from sft.action_head_data_contract import BEAM_SAMPLING, sampling_contract

from sft.joint_fm_head_sft import (
    JointBundleDataset,
    build_slow_batch_inputs,
    joint_collate,
)


def iter_manifest_rows(manifest: str):
    """Read manifest metadata without retaining a second full decoded copy."""
    with Path(manifest).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_anchor_splits(manifest: str, validate_row=None, *, only_split=None) -> dict[str, list[list[dict]]]:
    """Validate and group train/validation metadata in one streaming pass."""
    splits = {}
    for index, row in enumerate(iter_manifest_rows(manifest)):
        if validate_row is not None:
            validate_row(row, index)
        if only_split is not None and row.get("split", "train") != only_split:
            continue
        groups = splits.setdefault(row.get("split", "train"), {})
        groups.setdefault(row["slow_anchor_id"], []).append(row)
    return {split: [groups[key] for key in sorted(groups)] for split, groups in splits.items()}


def load_anchor_groups(manifest: str, split: str = "train") -> list[list[dict]]:
    return load_anchor_splits(manifest, only_split=split).get(split, [])


def select_case_branch_paths(groups, *, seed, branches_per_case=2):
    """Select whole root-to-frontier paths, retaining shared-prefix exposure."""
    if branches_per_case <= 0:
        raise ValueError('branches_per_case must be positive')
    cases = {}
    for group in groups:
        row = group[0]
        case = (row['source_scene'], row['source_task_id'])
        branch = row['branch_id']
        nodes = cases.setdefault(case, {})
        if branch in nodes:
            raise ValueError(f'duplicate branch node: {case}/{branch}')
        identity = ('source_scene', 'source_task_id', 'branch_id', 'parent_branch_id',
                    'golden_anchor_id', 'slow_anchor_id', 'option_sequence_index')
        if any(any(item[key] != row[key] for key in identity) for item in group):
            raise ValueError(f'mixed branch windows: {case}/{branch}')
        nodes[branch] = group
    selected_groups, selections = [], []
    for case, nodes in sorted(cases.items()):
        depth = max(int(group[0]['option_sequence_index']) for group in nodes.values())
        frontier = sorted(branch for branch, group in nodes.items()
                          if int(group[0]['option_sequence_index']) == depth)
        stable = int.from_bytes(hashlib.sha256(
            json.dumps([seed, *case], ensure_ascii=False).encode()).digest()[:8], 'big')
        chosen = random.Random(stable).sample(frontier, min(branches_per_case, len(frontier)))
        paths = []
        for leaf in chosen:
            branch, expected_depth = leaf, depth
            path = []
            while expected_depth >= 0:
                if branch not in nodes:
                    raise ValueError(f'missing parent in case branch: {case}/{branch}')
                group = nodes[branch]
                row = group[0]
                if int(row['option_sequence_index']) != expected_depth:
                    raise ValueError(f'nonconsecutive case branch: {case}/{branch}')
                path.append(group)
                branch = row['parent_branch_id']
                expected_depth -= 1
            if branch != 'root':
                raise ValueError(f'case branch does not reach root: {case}/{leaf}')
            path.reverse()
            selected_groups.extend(path)
            paths.append(dict(leaf_branch_id=leaf,
                              slow_anchor_ids=[group[0]['slow_anchor_id'] for group in path]))
        selections.append(dict(source_scene=case[0], source_task_id=case[1],
                               final_depth=depth, available_branches=len(frontier), paths=paths))
    return selected_groups, dict(
        policy='fixed_two_complete_case_branches_v1', seed=seed,
        requested_branches_per_case=branches_per_case, cases=len(selections),
        selected_paths=sum(len(case['paths']) for case in selections),
        anchor_exposures=len(selected_groups),
        unique_anchor_contexts=len({group[0]['slow_anchor_id'] for group in selected_groups}),
        shared_prefix_policy='one_exposure_per_selected_path', selections=selections)


class ActionHeadAnchorDataset(Dataset):
    def __init__(self, manifest: str, split: str = "train",
                 limit: int = 0, seed: int = 0, *, groups=None):
        self.groups = load_anchor_groups(manifest, split) if groups is None else groups
        self.branch_selection = None
        if self.groups and self.groups[0][0].get('action_head_as_tool_version') == 'v6':
            contracts = {sampling_contract(row) for group in self.groups for row in group}
            if len(contracts) != 1:
                raise ValueError('mixed sampling contracts; materialize an explicit mixture first')
            if split == 'train' and contracts == {BEAM_SAMPLING}:
                self.groups, self.branch_selection = select_case_branch_paths(self.groups, seed=seed)
        if not self.groups:
            raise ValueError(f"manifest contains no {split!r} slow anchors")
        # Deterministic subset for cheap, frequent inline validation: sample
        # `limit` anchors with a fixed seed so every snapshot sees the same
        # subset (curve comparable across steps). 0 = use all anchors.
        if limit and len(self.groups) > limit:
            if self.branch_selection is not None:
                raise ValueError('anchor limit would truncate selected complete case branches')
            self.groups = random.Random(seed).sample(self.groups, limit)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> list[dict]:
        group = self.groups[index]
        dataset = JointBundleDataset(group)
        return [dataset[item] for item in range(len(dataset))]


class ActionHeadAnchorCollator:
    def __init__(self, processor, seed: int, dataset_size: int, *, fast_processor=None):
        self.processor = processor
        self.fast_processor = fast_processor
        self.seed = int(seed)
        self.dataset_size = int(dataset_size)
        self.calls = 0
        self.noise_seed_policy = os.environ.get("ACTION_HEAD_NOISE_SEED_POLICY", "legacy")

    def __call__(self, features: list[list[dict]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("collator received an empty slow-anchor batch")
        window_batches = []
        reference_images, current_images, visual_modes = [], [], set()
        messages_batch, completions, counts, seed_anchors = [], [], [], []
        for group in features:
            batch = joint_collate(group)
            visual_modes.add('live' if 'reference_image' in batch else 'preencoded')
            if 'reference_image' in batch:
                reference_images.extend(batch.pop('reference_image'))
                current_images.extend(batch.pop('current_image'))
            anchor_ids = batch.pop("slow_anchor_id")
            messages = batch.pop("slow_messages")
            group_completions = batch.pop("slow_completion")
            if len(set(anchor_ids)) != 1:
                raise RuntimeError("one slow-anchor group contains multiple anchor ids")
            messages_batch.append(messages[0])
            completions.append(group_completions[0])
            counts.append(len(anchor_ids))
            seed_anchors.append(anchor_ids[0])
            window_batches.append(batch)
        if len(visual_modes) != 1:
            raise ValueError("mixed live/preencoded anchor batch")
        slow = build_slow_batch_inputs(
            self.processor, messages_batch, completions, torch.device("cpu"))
        result = {
            key: torch.cat([batch[key] for batch in window_batches], dim=0)
            for key in window_batches[0]
        }
        result.update(slow)
        if reference_images:
            from sft.live_fast_visual import build_fast_inputs
            if self.fast_processor is None:
                backbone = os.environ.get("ACTION_HEAD_FAST_BACKBONE", "")
                if backbone:
                    from transformers import AutoProcessor
                    self.fast_processor = AutoProcessor.from_pretrained(backbone, trust_remote_code=True)
                else:
                    self.fast_processor = self.processor
            result.update(build_fast_inputs(self.fast_processor, reference_images, current_images))
        result["window_anchor_index"] = torch.repeat_interleave(
            torch.arange(len(features), dtype=torch.long),
            torch.tensor(counts, dtype=torch.long))
        result["window_index_within_anchor"] = torch.cat([
            torch.arange(count, dtype=torch.long) for count in counts
        ])
        result["noise_seed"] = torch.tensor([
            fm_training_seed(self.seed, self.calls + index, seed_anchors[index], self.noise_seed_policy)
            for index in range(len(features))
        ], dtype=torch.long)
        # DistributedSampler pads a dataset shorter than world size by
        # repeating its first indices. Those ranks must still participate in
        # ZeRO collectives/forward, but their repeated windows contribute zero
        # to the global loss and throughput count.
        result["anchor_valid"] = torch.ones(len(features), dtype=torch.float32)
        self.calls += len(features)
        return result
