"""Bound the active scene working set while draining a fixed case selection."""
from __future__ import annotations

import hashlib
import threading
from collections import Counter


def renderer_index(scene: str, replicas: int) -> int:
    return int.from_bytes(hashlib.sha256(scene.encode()).digest()[:4], "big") % replicas


class SceneCaseScheduler:
    """FIFO among admissible cases; never drop or resample blocked cases.

    An actor holds a lease for the complete episode, including reset and the
    final observation. This bounds both live scenes and work per renderer.
    The server's cache capacity must still be configured separately.
    """

    def __init__(self, cases: list[dict], *, replicas: int = 3,
                 max_scenes: int = 48, max_cases: int = 64):
        if min(replicas, max_scenes, max_cases) <= 0:
            raise ValueError("renderer count and admission limits must be positive")
        self.pending = list(cases)
        self.replicas = replicas
        self.max_scenes, self.max_cases = max_scenes, max_cases
        self.active = [Counter() for _ in range(replicas)]
        self.peak_scenes = [0] * replicas
        self.peak_cases = [0] * replicas
        self.claimed = self.completed = 0
        self.cancelled = False
        self.condition = threading.Condition()

    def acquire(self, preferred_stage: str | None = None) -> dict | None:
        """Lease a case, retaining an actor's expensive backend when possible.

        ``preferred_stage`` is optional so established callers keep global FIFO.
        While preferred work remains, the actor waits for an admissible matching
        case instead of rebuilding its backend for an interleaved manifest. Once
        that stage is exhausted, it may drain any remaining work; this preserves
        the shared-pool tail behavior rather than reserving static shards.
        """
        with self.condition:
            while self.pending:
                if self.cancelled:
                    raise RuntimeError("case scheduling cancelled after actor failure")
                preferred = ([index for index, case in enumerate(self.pending)
                              if case.get("stage") == preferred_stage]
                             if preferred_stage is not None else [])
                candidates = preferred if preferred else range(len(self.pending))
                for index in candidates:
                    case = self.pending[index]
                    replica = renderer_index(case["scene"], self.replicas)
                    active = self.active[replica]
                    if sum(active.values()) >= self.max_cases:
                        continue
                    if case["scene"] not in active and len(active) >= self.max_scenes:
                        continue
                    self.pending.pop(index)
                    active[case["scene"]] += 1
                    self.claimed += 1
                    self.peak_scenes[replica] = max(self.peak_scenes[replica], len(active))
                    self.peak_cases[replica] = max(self.peak_cases[replica], sum(active.values()))
                    return case
                self.condition.wait()
            return None

    def cancel(self) -> None:
        with self.condition:
            self.cancelled = True
            self.condition.notify_all()

    def release(self, case: dict) -> None:
        with self.condition:
            active = self.active[renderer_index(case["scene"], self.replicas)]
            scene = case["scene"]
            if active[scene] <= 0:
                raise RuntimeError(f"scene lease was not acquired: {scene}")
            active[scene] -= 1
            if not active[scene]:
                del active[scene]
            self.completed += 1
            self.condition.notify_all()

    def stats(self) -> dict:
        with self.condition:
            return {"claimed": self.claimed, "completed": self.completed,
                    "pending": len(self.pending),
                    "active_cases": [sum(c.values()) for c in self.active],
                    "peak_active_scenes": list(self.peak_scenes),
                    "peak_active_cases": list(self.peak_cases),
                    "max_scenes_per_renderer": self.max_scenes,
                    "max_cases_per_renderer": self.max_cases}
