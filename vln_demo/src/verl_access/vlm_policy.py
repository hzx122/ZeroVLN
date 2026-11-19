# -*- coding: utf-8 -*-
"""vlm_policy — ReAct 工具调用策略，经 HTTP bench 执行（verl 端访问层）。

复用（不重写）vln_demo 既有的 ReAct 组件：
  - prompts.agent_system_prompt() / agent_task_text() / sampling_params() / MODES
  - agent_tools.build_system_prompt(TOOL_SPECS) / split_thinking() / parse_tool_calls()
    / assistant_history_content() / append_tool_budget() / strip_old_images() / count_images()
  - vln_model.stream_chat 的 vLLM payload（urllib SSE + chat_template_kwargs.preserve_thinking）

与 vln_model.run_agent 的唯一本质差异：**执行段不经直持 env，而经 bench HTTP step(tool)**。
bench 侧 env.step 的 tool 分支已内含 agent_tools.drive_forward/drive_turn（子步+碰撞+
finally-restore），所以这里不重造运动逻辑，只把模型 Invoke 翻译成 bench 的
{name,arguments} 动作方言，再把新观察 rgb 作为工具返回喂回对话。
"""
import os
import sys
import io
import json
import time
import base64
import copy
import hashlib
from collections import OrderedDict
from threading import Lock
import urllib.request
import urllib.error

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import prompts as P  # noqa: E402
import agent_tools as A  # noqa: E402
from verl_access.bench_client import (BenchClient, normalize_observation_contract,
                                      obs_image_part, tool_return_text)  # noqa: E402

# 与 vln_model 同口径的预算（vln_model.py:65-67）
MAX_TURNS = 100
MAX_TOOL_CALLS = 200
KEEP_LAST_IMAGES = 16
MODEL_WINDOW = 131072

# Runtime-only, bounded per collector process. Keys include current RGB pixels,
# so even an in-place PIL update cannot reuse an earlier observation's JPEG.
_IMAGE_CACHE_MAX_ENTRIES = 1024
_IMAGE_CACHE_MAX_BYTES = 32 * 1024 * 1024
_image_cache = OrderedDict()
_image_cache_bytes = 0
_image_cache_lock = Lock()


def _image_to_data_url(image, jpeg_quality):
    global _image_cache_bytes
    rgb = image.convert("RGB")
    key = (rgb.size, jpeg_quality, hashlib.sha256(rgb.tobytes()).digest())
    with _image_cache_lock:
        cached = _image_cache.get(key)
        if cached is not None:
            _image_cache.move_to_end(key)
            return cached
    # Preserve the existing Pillow encoder and every byte on the wire. Encode
    # outside the lock; independent actors need not serialize their new frames.
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    value = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    with _image_cache_lock:
        cached = _image_cache.get(key)
        if cached is not None:
            _image_cache.move_to_end(key)
            return cached
        if len(value) <= _IMAGE_CACHE_MAX_BYTES:
            _image_cache[key] = value
            _image_cache_bytes += len(value)
            while (len(_image_cache) > _IMAGE_CACHE_MAX_ENTRIES
                   or _image_cache_bytes > _IMAGE_CACHE_MAX_BYTES):
                _, discarded = _image_cache.popitem(last=False)
                _image_cache_bytes -= len(discarded)
    return value


def encode_canonical_images(messages, jpeg_quality=90):
    """Encode newly appended frames once in a harness-owned history.

    Reuse the existing exact JPEG encoder and image_url sender. Text, private
    image tags and ordering stay unchanged; subsequent deep copies contain
    immutable strings instead of repeatedly copying/encoding all past pixels.
    Stage every conversion before modifying the owned list.
    """
    pending = []
    for message in messages:
        if message.get('role') == 'assistant':
            continue  # The existing sender keeps assistant text only.
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get('type') == 'image':
                pending.append((part, _image_to_data_url(part['image'], jpeg_quality)))
    for part, value in pending:
        part.update(type='image_url', image_url={'url': value})
        part.pop('image', None)
    return len(pending)


def to_openai_messages(conversations, jpeg_quality=90):
    """内部消息（含 {"type":"image","image":<PIL>}）-> vLLM OpenAI 兼容 payload。

    照抄 rynnvln_eval._to_openai_messages + _image_to_data_url：assistant 段压成纯
    text，image 段转 data:image/jpeg;base64,...。bench obs rgb 走此路喂 vLLM。
    """
    import io
    from PIL import Image  # noqa: E402,F811
    messages = []
    for message in conversations:
        role = message["role"]
        content = message["content"]
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if role == "assistant":
            text = "".join(p.get("text", "") for p in content
                           if p.get("type") == "text")
            messages.append({"role": role, "content": text})
            continue
        openai_content = []
        for part in content:
            if part.get("type") == "text":
                openai_content.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image":
                openai_content.append({"type": "image_url",
                                       "image_url": {"url": _image_to_data_url(part["image"], jpeg_quality)}})
            elif part.get("type") == "image_url":
                # Archived requests already contain encoded observation/map
                # frames. Preserve their bytes instead of silently dropping
                # them or decoding and applying a second JPEG compression.
                openai_content.append({"type": "image_url",
                                       "image_url": copy.deepcopy(part["image_url"])})
        messages.append({"role": role, "content": openai_content})
    return messages


class ContextOverflow(Exception):
    def __init__(self, detail):
        super().__init__(detail)
        self.detail = detail


def _classify_http_error(exc):
    body = ""
    try:
        body = exc.read().decode("utf-8", "replace")
    except Exception:
        pass
    if exc.code == 400 and "context length" in body.lower():
        return ContextOverflow(body[:200])
    return RuntimeError(f"HTTP {exc.code}: {body[:200]}")


def chat_stream(vllm_url, model_name, messages, mode, temperature,
                preserve_thinking, timeout=300, retries=3, seed=None):
    """POST vLLM /v1/chat/completions (SSE)，返回 (full_text, finish_reason, usage)。

    payload 与 vln_model.stream_chat 一致：stream + include_usage +
    chat_template_kwargs{enable_thinking, preserve_thinking} + prompts.sampling_params。
    """
    params, thinking = P.sampling_params(mode, temperature)
    tpl_kwargs = {"enable_thinking": thinking}
    if preserve_thinking:
        tpl_kwargs["preserve_thinking"] = True
    payload = {
        "model": model_name,
        "messages": to_openai_messages(messages),
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": tpl_kwargs,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    payload.update(params)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    url = f"{vllm_url}/v1/chat/completions"

    last_err = None
    for attempt in range(1, max(1, retries) + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            raise _classify_http_error(exc) from None
        except Exception as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 8))
            continue
        chunks, finish, usage, got_any = [], None, {}, False
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except Exception:
                    continue
                if obj.get("usage"):
                    usage.update(obj["usage"])
                choices = obj.get("choices") or [{}]
                fr = choices[0].get("finish_reason")
                if fr:
                    finish = fr
                delta = choices[0].get("delta") or {}
                piece = (delta.get("reasoning_content") or "") + \
                        (delta.get("content") or "")
                if piece:
                    got_any = True
                    chunks.append(piece)
        except Exception as exc:
            if not got_any:
                last_err = exc
                try:
                    resp.close()
                except Exception:
                    pass
                time.sleep(min(2 ** attempt, 8))
                continue
        finally:
            try:
                resp.close()
            except Exception:
                pass
        return "".join(chunks), finish, usage
    raise RuntimeError(f"vLLM streaming failed after {retries} attempts: {last_err}")


class VLMAgent:
    """ReAct agent：vLLM 推理 + bench HTTP 执行的闭环驱动。

    用法：
        bench = BenchClient("http://127.0.0.1:8907")
        ag = VLMAgent(bench, vllm_url="http://127.0.0.1:28898",
                      model_name="MoBot_VLN-2B", mode="think")
        traj = ag.rollout(scene, ep, instruction, max_turns=30)
    """

    def __init__(self, bench, vllm_url, model_name, mode="think",
                 temperature=None, thinking_history="keep_thinking",
                 max_turns=MAX_TURNS, max_tool_calls=MAX_TOOL_CALLS,
                 keep_last_images=KEEP_LAST_IMAGES, verbose=True,
                 observation_contract=None):
        self.bench = bench
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.mode = mode
        self.temperature = temperature
        self.preserve_thinking = thinking_history == "keep_thinking"
        self.thinking_history = thinking_history
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.keep_last_images = keep_last_images
        self.verbose = verbose
        self.observation_contract = normalize_observation_contract(
            observation_contract)

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _init_msgs(self, instruction, obs):
        img_part = obs_image_part(obs)
        if img_part is None:
            raise RuntimeError(
                "reset/step 未返回 rgb 观察（bench env 可能进坏状态，需重启 bench）")
        return [
            {"role": "system",
             "content": [{"type": "text", "text": P.agent_system_prompt()}]},
            {"role": "user",
             "content": [img_part,
                         {"type": "text", "text": P.agent_task_text(instruction)}]},
        ]

    def rollout(self, scene, episode_id, instruction=None, max_turns=None,
                seed=None):
        max_turns = max_turns or self.max_turns
        obs, info = self.bench.reset(scene=scene, episode_id=episode_id,
                                     instruction=instruction, seed=seed)
        instruction = obs.get("instruction") or instruction or ""
        msgs = self._init_msgs(instruction, obs)
        prev_info = info
        calls_used = 0
        motion = {"path_dist": 0.0, "yaw": 0.0, "steps": 0}
        traj = {"scene": scene, "episode_id": episode_id, "instruction": instruction,
                "model": self.model_name, "mode": self.mode,
                "observation_contract": self.observation_contract,
                "turns": [], "final": {}, "why": "",
                "initial_info": {k: info.get(k) for k in
                                 ("gps", "yaw_deg", "world_xy", "world_yaw_deg",
                                  "distance_to_goal", "steps_used")}}
        stopped = False
        traj["invalid_tool_call_policy"] = "feedback"
        for turn in range(max_turns):
            if calls_used >= self.max_tool_calls:
                traj["why"] = f"达到工具调用上限 {self.max_tool_calls}"
                break
            if info.get("episode_over"):
                traj["why"] = "回合结束"
                break
            sent = A.strip_old_images(msgs, self.keep_last_images)
            self._log(f"[agent] t{turn+1}/{max_turns} msgs={len(sent)} "
                      f"imgs={A.count_images(sent)} calls={calls_used}/{self.max_tool_calls} "
                      f"step={info.get('steps_used')} dist={info.get('distance_to_goal')}")
            t0 = time.time()
            try:
                text, finish, usage = chat_stream(
                    self.vllm_url, self.model_name, sent, self.mode,
                    self.temperature, self.preserve_thinking,
                    seed=(None if seed is None else int(seed) + turn))
            except ContextOverflow as exc:
                traj["why"] = f"上下文超出窗口（t{turn+1}）{exc.detail}"
                break
            except Exception as exc:
                traj["why"] = f"模型请求失败：{exc}"
                self._log(f"[agent] model err: {exc}")
                break
            dt = time.time() - t0
            thinking, visible = A.split_thinking(text)
            calls = A.parse_tool_calls(visible)
            self._log(f"[agent] t{turn+1} <- pt={usage.get('prompt_tokens')} "
                      f"ct={usage.get('completion_tokens')} dt={dt:.1f}s "
                      f"calls={[c['name'] for c in calls]}")

            # 记 assistant 消息（keep_thinking 下保原始回复）
            msgs.append({"role": "assistant",
                         "content": [{"type": "text",
                                      "text": A.assistant_history_content(
                                          text, visible, self.thinking_history) or text}]})

            parsed = calls
            turn_record = {"turn": turn + 1, "assistant": (visible or "").strip(),
                           "reasoning": (thinking or "").strip(), "calls": [],
                           "prompt_tokens": usage.get("prompt_tokens"),
                           "completion_tokens": usage.get("completion_tokens"),
                           "dt_s": round(dt, 2)}
            calls = [c for c in calls if c["name"] in A.TOOLS_BY_NAME]
            format_error = A.tool_call_batch_error(visible, parsed, A.TOOLS_BY_NAME)
            if format_error:
                calls_used += 1
                raw = format_error
                turn_record["tool_call_error"] = raw
                traj["turns"].append(turn_record)
                msgs.append(A.tool_call_feedback_message(
                    raw, max(0, self.max_tool_calls - calls_used)))
                if calls_used >= self.max_tool_calls:
                    traj["why"] = f"达到工具调用上限 {self.max_tool_calls}"
                    break
                continue
            if len(calls) > P.MAX_CALLS_PER_TURN:
                calls = calls[:P.MAX_CALLS_PER_TURN]

            results = []  # list of (text, wants_img)
            for call in calls:
                if info.get("episode_over"):
                    break
                if calls_used >= self.max_tool_calls:
                    traj["why"] = f"达到工具调用上限 {self.max_tool_calls}"
                    break
                name, argd = call["name"], call["arguments"]
                calls_used += 1
                stop_rejection = A.rejected_stop_result(argd) if name == "stop" else None
                if stop_rejection is not None:
                    turn_record["calls"].append({"name": name, "args": argd,
                                                "ok": False, "result": stop_rejection})
                    results.append((A.compact_json(stop_rejection), False))
                    continue
                if name == "stop":
                    stopped = True
                    obs, r, term, trunc, info = self.bench.step_tool(name, argd)
                    turn_record["calls"].append(
                        {"name": name, "args": argd, "stop": True,
                         "reward": r, "terminated": term, "truncated": trunc,
                         "info": {k: info.get(k) for k in
                                  ("steps_used", "gps", "yaw_deg",
                                   "distance_to_goal", "world_xy",
                                   "world_yaw_deg", "success", "episode_over")}})
                    break
                # Reject model arguments before the HTTP/service boundary.
                normalized, note = A.normalize_primitive_call(name, argd)
                if normalized is None:
                    raw = A.rejected_tool_arguments_result(name, note)
                    turn_record["calls"].append({"name": name, "args": argd,
                                                "ok": False, "result": raw})
                    results.append((A.compact_json(raw), False))
                    continue
                obs, r, term, trunc, info = self.bench.step_tool(name, normalized)
                rtxt = tool_return_text(
                    name, info, prev_info, arguments=argd, motion=motion,
                    observation_contract=self.observation_contract)
                results.append((rtxt, True))
                turn_record["calls"].append({
                    "name": name, "args": argd,
                    "reward": r, "terminated": term, "truncated": trunc,
                    "info": {k: info.get(k) for k in
                             ("steps_used", "gps", "yaw_deg", "distance_to_goal",
                              "world_xy", "world_yaw_deg", "benchmark_intervention",
                              "collided_this_step", "blocked_this_step", "success",
                              "episode_over", "tool_result")}})
                prev_info = info
                if term or trunc:
                    break

            traj["turns"].append(turn_record)
            if stopped:
                traj["why"] = traj.get("why") or f"模型调用 stop"
                break
            if (info.get("episode_over") or
                    (turn_record["calls"] and
                     turn_record["calls"][-1].get("terminated"))):
                traj["why"] = traj.get("why") or "episode_over (success/collision)"
                break
            if not results:
                traj["why"] = traj.get("why") or "无可执行的工具调用"
                break

            # 工具返回消息：最后一条带新画面 image
            budget_left = max(0, self.max_tool_calls - calls_used)
            for i, (txt, wants_img) in enumerate(results):
                body = A.append_tool_budget(txt, budget_left) \
                    if i == len(results) - 1 else txt
                content = [{"type": "text", "text": "[工具返回]\n" + body}]
                if wants_img and i == len(results) - 1:
                    ip = obs_image_part(obs)
                    if ip:
                        content.insert(0, ip)
                msgs.append({"role": "user", "content": content})

            if calls_used >= self.max_tool_calls:
                traj["why"] = f"达到工具调用上限 {self.max_tool_calls}"
                break
        else:
            traj["why"] = f"达到轮数上限 {max_turns}"

        traj["final"] = {k: info.get(k) for k in
                         ("success", "spl", "distance_to_goal",
                          "oracle_navigation_error", "steps_used", "gps", "yaw_deg",
                          "world_xy", "world_yaw_deg",
                          "episode_over")}
        traj["calls_used"] = calls_used
        return traj
