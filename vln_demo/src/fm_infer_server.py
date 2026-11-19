#!/usr/bin/env python3
"""FM head HTTP sidecar（.venv_fm 起，给 gordon-venv 控制台调用）。

gordon venv（canonical `start_sage_usd.sh` 控制台）装不了 SocialNav forked
transformers（pin `huggingface-hub<1.0`，gordon 是 hub 1.11.0）→ 不能 in-process
加载 FM head（`fm_policy.FMInfer`）。本 sidecar 在 **.venv_fm** 里加载 fm_policy
（hub 0.36.2 + forked transformers 4.57.1，与 SFT/RL 同环境），暴露 HTTP `/predict`；
控制台 `_load_fm` 经 `FM_INFER_URL` 走 HTTP 调它——FM head 与控制台 venv 解耦
（同本仓 render/vllm/bench 都是 HTTP sidecar 的架构）。

契约（与 fm_policy.FMInfer.predict 对齐）：
  POST /predict  body {"images":["<b64 jpg>",...], "instruction":str,
                       "input_waypoint":[[x,y],...]}   # 归一化，N=len(images)
  → {"delta":[[32,3] 归一化], "real":[[32,3] 米/弧度 denorm], "gen_ids":[...]}
  real 用 anonymous stats denorm（fm_policy._ZYT），控制台直接拿去执行，不必本地 denorm。
"""
import os
import sys
import io
import json
import base64
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))            # vln_demo/src
_REPO = os.path.dirname(_HERE)                                 # vln_demo
_REPO_ROOT = os.path.dirname(_REPO)                            # closed_loop_3dgs_rl
# closed_loop 训练配置那份 fm_policy + SocialNav（与 vln_model._load_fm 同款路径）
for p in (os.path.join(_REPO_ROOT, "verl", "workers"),
          os.path.join(_REPO_ROOT, "verl", "bench"),
          os.path.join(_REPO_ROOT, "third_party", "SocialNav"),
          os.path.join(_REPO_ROOT, "third_party", "SocialNav", "train_clean")):
    if p not in sys.path:
        sys.path.insert(0, p)

_FM = {"infer": None, "denorm": None, "anonymous": None}


def load_fm():
    """Lazy/eager load FMInfer（.venv_fm 里 fm_policy，hub 0.36.2 兼容）。"""
    if _FM["infer"] is not None:
        return _FM
    from fm_policy import (FMInfer, _denorm_delta, _ZYT, FM_CKPT)  # verl/workers
    ckpt = os.environ.get("FM_CKPT") or FM_CKPT
    device = os.environ.get("FM_DEVICE", "cuda")
    print(f"[fm-sidecar] loading {ckpt} on {device}", flush=True)
    infer = FMInfer(ckpt=ckpt, device=device)
    _FM.update(infer=infer, denorm=_denorm_delta, anonymous=_ZYT)
    print(f"[fm-sidecar] loaded chunk={infer.action_chunk} dim={infer.action_dim} "
          f"steps={infer.num_flow_steps}", flush=True)
    return _FM


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ct, code=200):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/ping":
            self._send("pong", "text/plain")
        elif u.path == "/health":
            self._send(json.dumps({"loaded": _FM["infer"] is not None}),
                       "application/json")
        else:
            self._send("fm-infer sidecar", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/predict":
            self._send("not found", "text/plain", 404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            imgs = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
                    for b in req.get("images", [])]
            wp = np.array(req.get("input_waypoint", [[0.0, 0.0]]), np.float32)
            instr = req.get("instruction", "")
            if not imgs:
                raise ValueError("images 为空")
            fm = load_fm()
            norm_delta, gen_ids = fm["infer"].predict(imgs, wp, instr)
            real = fm["denorm"](norm_delta, fm["anonymous"])   # anonymous stats denorm
            self._send(json.dumps({
                "delta": norm_delta.tolist(),
                "real": real.tolist(),
                "gen_ids": list(gen_ids),
            }), "application/json")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._send(json.dumps({"error": str(exc)}), "application/json", 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("FM_INFER_PORT", "7039")))
    args = ap.parse_args()
    load_fm()                       # eager load at startup（失败直接退出，看得见）
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), H)
    print(f"[fm-sidecar] serving :{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
