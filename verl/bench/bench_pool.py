# -*- coding: utf-8 -*-
"""bench_pool — 多 bench 后端的并发分发层（照 verl judge_client 的模式）。

verl 的 rollout→reward 流式交叠靠**多 judge replica + ThreadPool + round-robin**
（`tools/reward/judge_client.py` 的 `HealthGatedPicker`/`RoundRobin` +
`ThreadPoolExecutor`）。bench 这边等价：起 M 个 `vln_gym_server` 进程（不同端口），
每个单线程 env = 一个 replica；N 条轨迹经 ThreadPool 并发分发到 M 个 bench，
每个 bench 内 habitat 仍串行（GL 线程亲和），但**进程之间并行** → ~M× 加速。

"流式"= 产-消流水线交叠（SDE 采完一条即 submit 到 pool，早完早返回），不是 HTTP
chunked。bench 单进程内 GL 串行无法并行，并行在进程之间——和 verl 多 judge replica
一个道理。

用法（trainer）：
    pool = BenchPool(["http://127.0.0.1:8907","http://127.0.0.1:8908"], timeout=300)
    rs = pool.map_rewards(waypoints_list, scene, ep, reward_fn=env_reward)
    # N 条轨迹并发分发到 M 个 bench，返回 reward 列表（保序）
    # 单调用兼容 BenchClient: pool.reset / pool.step_fm / pool.health（round-robin 选一个）
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

from bench_client import BenchClient  # 同目录纯 HTTP client

# 默认重试：失败时轮换到下一个 bench URL，最多试 pool 大小次（照 judge_client
# chat_with_retry 的 URL 轮换重试，简化版——bench 无 /metrics 负载，纯 round-robin）。


class BenchPool:
    """多 bench URL 的并发池。round-robin 选 client + ThreadPool 并发分发。

    单 URL 退化为串行（ThreadPool 1 worker），与原 BenchClient 行为一致。
    """

    def __init__(self, urls: List[str], timeout: int = 300,
                 max_workers: Optional[int] = None):
        if not urls:
            raise ValueError("BenchPool needs >=1 url")
        self.urls = [u.strip() for u in urls if u.strip()]
        self.timeout = timeout
        self.clients = [BenchClient(u, timeout=timeout) for u in self.urls]
        self._n = len(self.clients)
        self._rr = 0
        self._lock = threading.Lock()
        # 一个 worker 配一个 bench：HTTP IO 密集，GIL 不挡，ThreadPool 正合适。
        self._ex = ThreadPoolExecutor(max_workers=max_workers or self._n)

    # ---- round-robin 选 client（线程安全）----
    def next_client(self) -> BenchClient:
        with self._lock:
            c = self.clients[self._rr % self._n]
            self._rr += 1
            return c

    # ---- 单调用兼容 BenchClient（round-robin，eval 用）----
    def health(self):
        return self.next_client().health()

    def spec(self):
        return self.next_client().spec()

    def reset(self, *, scene=None, episode_id=None, instruction=None, seed=None):
        return self.next_client().reset(scene=scene, episode_id=episode_id,
                                         instruction=instruction, seed=seed)

    def step_fm(self, waypoints, normalized=True, render_final_only=False):
        return self.next_client().step_fm(waypoints, normalized=normalized,
                                          render_final_only=render_final_only)

    def reset_step_fm(self, scene, ep, waypoints, normalized=False,
                      render_final_only=False):
        """Pick ONE client, reset to (scene, ep), step_fm on the SAME client.
        返回 info。reset 和 step 必须同 client（否则 step 打到没复位的 env 上，
        info 错乱）——eval/单发用这个，不要分开调 reset+step_fm。"""
        c = self.next_client()
        c.reset(scene=scene, episode_id=ep)
        _, _, term, trunc, info = c.step_fm(waypoints, normalized=normalized,
                                            render_final_only=render_final_only)
        return info

    def render(self, mode="rgb"):
        return self.next_client().render(mode)

    def close(self):
        self._ex.shutdown(wait=False)
        for c in self.clients:
            try:
                c.close()
            except Exception:
                pass

    # ---- 并发分发（训练用）----
    def exec_trajectory(self, scene: str, ep: str, waypoints: list,
                        reward_fn: Callable[[dict], float],
                        render_final_only: bool = False) -> tuple:
        """单条轨迹：round-robin 选 bench → reset 同起点 → step_fm → (reward, info)。

        返回 (reward_fn(info), info)，info 留给 trainer 拆 success/distance/
        collision/spl 分项（reward_fn 只合成标量会丢分项）。失败（HTTP 错/超时）
        轮换到下一个 bench URL 重试，最多试 self._n 次（照 judge_client
        chat_with_retry 的 URL 轮换重试）。全失败抛 RuntimeError。

        render_final_only 透传给 step_fm：单发 chunk 只需终点观测，开它把渲染从
        每 sub-step 降到每 chunk 一次。
        """
        last_err: Optional[Exception] = None
        for _ in range(max(1, self._n)):
            c = self.next_client()
            try:
                c.reset(scene=scene, episode_id=ep)  # 同起点（公平 GRPO）
                _, _, term, trunc, info = c.step_fm(
                    waypoints, normalized=False,
                    render_final_only=render_final_only)
                return reward_fn(info), info
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"bench exec_trajectory failed after {self._n} tries: {last_err}")

    def map_rewards(self, waypoints_list: List[list], scene: str, ep: str,
                    reward_fn: Callable[[dict], float],
                    render_final_only: bool = False) -> List[tuple]:
        """N 条轨迹并发分发到 M 个 bench。返回 [(reward, info), ...]（保序）。

        每条独立 reset+step_fm（bench 间无共享状态）。早完成的早返回，ThreadPool
        调度下一条 → 产-消流式交叠。失败条目记 (-1e9, {})（不崩整步，但梯度上
        会被 advantage 归一化压住；照 judge fallback 非惩罚性思路，这里用大负值
        显式标记）。info 留空 dict 让 trainer 的分项统计跳过该条。

        render_final_only 透传给每条 exec_trajectory。
        """
        futs = [self._ex.submit(self.exec_trajectory, scene, ep, w, reward_fn,
                                render_final_only)
                for w in waypoints_list]
        out = []
        for f in futs:
            try:
                out.append(f.result())
            except Exception:
                out.append((-1e9, {}))
        return out

    # ---- episode 级并发（多轮闭环 eval / RL 用）----
    def map_episodes(self, eps: List[tuple],
                     episode_fn: Callable[["BenchClient", str, str], dict],
                     ) -> List[Optional[dict]]:
        """N 个 episode 并发跑，每个 episode 独占一个 bench client 跑完整个多轮
        rollout（reset + 多次 step_fm/step_tool 必须同 client，env 有状态且非线程
        安全）。返回 [episode_dict or None, ...]（保序；失败/skip 记 None）。

        与 map_rewards 的差异：map_rewards 每条轨迹是单发 reset+step_fm（无状态跨
        step），可自由 round-robin；map_episodes 每个 episode 内部**跨多步共享同一
        env 状态**，所以必须把一个 client 独占给一个 episode 直到跑完，用 client
        队列保证任一时刻一个 client 最多被一个 episode 占用（并发度 = client 数）。

        episode_fn(client, scene, ep) -> dict：调用方封装多轮 rollout（见 trainer
        的 run_episode_ode_multiturn）。抛异常的 episode 记 None，不中断整批。
        """
        cq: "queue.Queue[BenchClient]" = queue.Queue()
        for c in self.clients:
            cq.put(c)

        def _run(item):
            sc, ep = item
            c = cq.get()  # 阻塞直到有空闲 client → 并发度自然 = client 数
            try:
                return episode_fn(c, sc, ep)
            except Exception as e:
                print(f"[map_episodes skip {sc}:{ep}] {e}", flush=True)
                return None
            finally:
                cq.put(c)

        futs = [self._ex.submit(_run, item) for item in eps]
        out = []
        for f in futs:
            try:
                out.append(f.result())
            except Exception:
                out.append(None)
        return out

