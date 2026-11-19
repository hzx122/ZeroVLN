"""Deterministic, threshold-triggered sampling of complete navigation turns.

Canonical episode messages are never changed. A fixed original-turn selection
is applied until the next compaction, so native prefixes stay append-only
inside each context segment. No model summarizer or privileged state is used.
"""
from __future__ import annotations

import copy

from .trajectory_map import MAP_CONTEXT_TEXT, CURRENT_VIEW_TEXT

REVISION = "v8_complete_turn_stride_v1"


def complete_turns(messages):
    """Keep assistant calls with all of their feedback, including tool roles."""
    starts = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    groups = []
    for start, end in zip(starts, starts[1:] + [len(messages)]):
        group = messages[start:end]
        if len(group) < 2 or any(m.get("role") not in ("user", "tool") for m in group[1:]):
            raise ValueError("context compaction requires complete assistant/feedback turns")
        calls = group[0].get("tool_calls") or []
        if not isinstance(calls, list) or any(not isinstance(c, dict) for c in calls):
            raise ValueError("context compaction found invalid tool calls")
        call_ids = [c.get("id") for c in calls]
        result_ids = [m.get("tool_call_id") for m in group[1:] if m.get("role") == "tool"]
        if (calls or result_ids) and (not all(call_ids) or len(set(call_ids)) != len(call_ids)
                                      or sorted(call_ids) != sorted(result_ids, key=str)):
            raise ValueError("context compaction found unmatched tool calls/results")
        groups.append((start, end))
    return (starts[0] if starts else len(messages)), groups


class TurnContext:
    def __init__(self, *, recent=8, frequency=4, reserve_tokens=8192):
        if any(type(x) is not int or x <= 0 for x in (recent, frequency, reserve_tokens)):
            raise ValueError("context recent/frequency/reserve must be positive integers")
        self.recent, self.frequency, self.reserve_tokens = recent, frequency, reserve_tokens
        self.segment = 0
        self.kept = set()
        self.through = 0
        self.removed_maps = set()
        self.audit = None
        self.notice = ""

    def active_messages(self, messages):
        if not self.segment:
            return messages
        prefix, groups = complete_turns(messages)
        if len(groups) < self.through:
            raise ValueError("provide the full append-only canonical history, not a sampled history")
        selected = set(range(prefix))
        for number, (start, end) in enumerate(groups, 1):
            if number > self.through or number in self.kept:
                selected.update(range(start, end))
        result = []
        for i, message in enumerate(messages):
            if i not in selected:
                continue
            m = copy.deepcopy(message)
            if isinstance(m.get("content"), list):
                m["content"] = [p for j, p in enumerate(m["content"])
                                if (i, j) not in self.removed_maps]
            result.append(m)
        if not result or result[0].get("role") != "system":
            raise ValueError("context compaction requires the original system message")
        content = result[0]["content"]
        if isinstance(content, str):
            result[0]["content"] = content + "\n\n" + self.notice
        else:
            content.append(dict(type="text", text=self.notice))
        return result

    def _compact(self, messages):
        """Stage a new selection; prepare_messages commits only after validation."""
        _, groups = complete_turns(messages)
        n = len(groups)
        kept = {i for i in range(1, n + 1)
                if i > n - self.recent or (i - 1) % self.frequency == 0}
        # Retain original indices. Never stride the already-strided list.
        if self.segment:
            kept &= self.kept | set(range(self.through + 1, n + 1))
        maps = []
        for i, message in enumerate(messages):
            parts = message.get("content")
            if not isinstance(parts, list):
                continue
            for j, part in enumerate(parts):
                if part.get("type") == "text" and part.get("text") == MAP_CONTEXT_TEXT:
                    if j + 1 >= len(parts) or parts[j + 1].get("type") not in ("image", "image_url"):
                        raise ValueError("trajectory map label has no matching image")
                    maps.append((i, j))
        for i, j in maps[:-1]:
            self.removed_maps.update(((i, j), (i, j + 1)))
            parts = messages[i]["content"]
            if j + 2 < len(parts) and parts[j + 2].get("text") == CURRENT_VIEW_TEXT:
                self.removed_maps.add((i, j + 2))
        self.segment += 1
        self.kept, self.through = kept, n
        self.notice = (
            "[Context history sampled by the harness]\n"
            f"Complete original turns retained through turn {n}: {','.join(map(str, sorted(kept)))}. "
            f"The most recent {self.recent} turns are complete; older turns are sampled every "
            f"{self.frequency} turns. Missing turns and their observations are unavailable. "
            "Do not infer their details or treat separated records as consecutive actions. "
            "Only the latest cumulative trajectory map at this boundary is retained. "
            "The environment, original task, and cumulative tool-call budget have not reset.")
        return self.active_messages(messages)

    def metadata(self):
        return copy.deepcopy(dict(revision=REVISION, segment_id=self.segment,
                    **({"compaction": self.audit} if self.audit else {})))

    def prepare_messages(self, messages, *, count_tokens, max_model_len=131072,
                         max_completion_tokens=512, input_tokens=None):
        """Standalone budget transaction; never calls a model or executes tools.

        count_tokens must use the target model's template and include visual
        tokens. input_tokens can supply an authoritative native count for the
        current *active* context. A successful rewrite requires a new context
        segment if a caller later integrates native token/KV/routing replay.
        Always pass the full canonical episode, including all original turns.
        """
        if (type(max_model_len) is not int or max_model_len <= self.reserve_tokens
                or type(max_completion_tokens) is not int or max_completion_tokens <= 0):
            raise ValueError("invalid context window/output budget")
        active = self.active_messages(messages)
        before = count_tokens(active) if input_tokens is None else input_tokens
        if type(before) is not int or before < 0:
            raise ValueError("token counter must return a nonnegative integer")
        triggered = before + max_completion_tokens + self.reserve_tokens > max_model_len
        after = before
        if triggered:
            staged = copy.deepcopy(self)
            active = staged._compact(messages)
            after = count_tokens(active)
            if type(after) is not int or after < 0:
                raise ValueError("token counter must return a nonnegative integer")
            if after >= before or after + max_completion_tokens + self.reserve_tokens > max_model_len:
                raise ValueError("sampled context cannot fit; recent complete turns are protected")
            staged.audit = dict(input_tokens_before=before, input_tokens_after=after,
                max_model_len=max_model_len, kept_original_turns=sorted(staged.kept),
                recent=staged.recent, frequency=staged.frequency,
                through_original_turn=staged.through)
            self.__dict__.update(staged.__dict__)
        audit = dict(status="compacted" if triggered else "unchanged",
            input_tokens_before=before, input_tokens_after=after, max_model_len=max_model_len,
            completion_tokens=max_completion_tokens, reserve_tokens=self.reserve_tokens,
            context=self.metadata(), requires_new_native_context=triggered)
        return copy.deepcopy(active), audit
