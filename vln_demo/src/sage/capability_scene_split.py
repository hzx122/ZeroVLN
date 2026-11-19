"""Deterministic scene-level construction/evaluation holdout for capability data."""
from __future__ import annotations

import argparse
import hashlib
import json


DEFAULT_POOL = ("/data/zerovln/SAGE-3D/"
                "episode_filter/kept_train_r0.07.jsonl")
DEFAULT_SALT = "sage-capability-scene-holdout-v1"


def scene_partition(scene: str, salt: str = DEFAULT_SALT,
                    evaluation_percent: int = 20) -> str:
    value = int(hashlib.sha256((salt + ":" + str(scene)).encode()).hexdigest(), 16) % 100
    return "evaluation" if value < int(evaluation_percent) else "construction"


def partition_scenes(scenes, partition: str, limit: int = 0,
                     salt: str = DEFAULT_SALT, evaluation_percent: int = 20) -> list[str]:
    selected = sorted({str(x) for x in scenes
                       if scene_partition(str(x), salt, evaluation_percent) == partition},
                      key=lambda x: hashlib.sha256((salt + ":order:" + x).encode()).hexdigest())
    return selected[:limit] if limit > 0 else selected


def load_scene_pool(path: str) -> list[str]:
    scenes = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                scenes.add(str(json.loads(line)["scene"]))
    return sorted(scenes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=DEFAULT_POOL)
    parser.add_argument("--partition", choices=("construction", "evaluation"), required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--evaluation-percent", type=int, default=20)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    args = parser.parse_args()
    scenes = partition_scenes(load_scene_pool(args.pool), args.partition, args.limit,
                              args.salt, args.evaluation_percent)
    print(",".join(scenes) if args.format == "csv" else json.dumps(scenes))


if __name__ == "__main__":
    main()
