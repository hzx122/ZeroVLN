# Interactive VLN-CE demo: step through the Habitat simulator manually.
# You either drive the agent yourself (w/a/d/x) or let the model move (n/N).
# The latest RGB observation (optionally with a top-down map) is served over
# HTTP so you can watch the agent walk in your browser via SSH port-forward.
import sys
import pathlib
import os
import io
import time
import json
import base64
import threading
import argparse
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NAV_DIR = os.environ.get(
    "RYNNBRAIN_NAV_DIR",
    "/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation",
)
DEMO_CONFIG = os.environ.get("DEMO_CONFIG", "vln_demo.yaml")
sys.path.insert(0, os.path.join(NAV_DIR, "src"))
os.environ.setdefault("RYNNVLN_LANGUAGE", "en")

import rynnvln_eval as base
from PIL import Image

# ---- action ids in this sim (see rynnlvn_eval actions2idx / EMBODIED_ACTION_TEXT) ----
ACTIONS = {"w": 1, "a": 2, "d": 3, "x": 0}
ACTION_NAME = {1: "forward 25cm", 2: "left 30deg", 3: "right 30deg", 0: "STOP"}

LATEST = {"jpg": None, "step": 0, "info": "", "map": False}

PAGE = """<!doctype html><meta http-equiv='refresh' content='1'>
<style>body{background:#111;color:#eee;font-family:monospace;margin:12px}
img{max-width:100%;border:1px solid #444}#i{font-size:14px;color:#8f8}</style>
<body><div id='i'>__INFO__</div><img src='/frame.jpg?ts=__TS__'></body>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path.startswith("/frame.jpg"):
            data = LATEST["jpg"]
            if data is None:
                self.send_response(204); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            body = (PAGE.replace("__INFO__", LATEST["info"])
                        .replace("__TS__", str(int(time.time())))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def jpg_bytes(img):
    b = io.BytesIO()
    img.convert("RGB").save(b, "JPEG", quality=88)
    return b.getvalue()


def compose_frame(rgb, info, with_map, step_id, action_seq):
    cur = Image.fromarray(rgb).convert("RGB")
    if with_map and info.get("top_down_map") is not None:
        m = info["top_down_map"]
        if isinstance(m, dict):
            m = m.get("map", m.get("top_down_map"))
        if m is None:
            return cur
        import numpy as np
        m = np.asarray(m)
        if m.dtype != 'uint8':
            m = m.astype('uint8')
        mp = Image.fromarray(m).convert("RGB")
        # scale map to 480 tall to match the rgb frame
        h = 480
        w = max(1, int(mp.width * h / mp.height))
        mp = mp.resize((w, h))
        canvas = Image.new("RGB", (cur.width + w, h), (20, 20, 20))
        canvas.paste(cur, (0, 0))
        canvas.paste(mp, (cur.width, 0))
        cur = canvas
    return cur


class InteractiveEvaluator(base.VLNEvaluator):
    pass  # reuse __init__ (env/top_down_map config) + conversation builders + vllm client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_id", default=os.environ.get("DEMO_EPISODE_ID", "11"))
    ap.add_argument("--scene", default=os.environ.get("DEMO_SCENE", "2azQ1b91cZZ"))
    ap.add_argument("--split", default=os.environ.get("DEMO_SPLIT", "val_unseen"))
    ap.add_argument("--instruction", default=os.environ.get("DEMO_INSTRUCTION", ""),
                    help="override the episode instruction")
    ap.add_argument("--port", type=int, default=int(os.environ.get("DEMO_HTTP_PORT", "5000")))
    ap.add_argument("--map", action="store_true", default=os.environ.get("DEMO_MAP", "1") == "1")
    ap.add_argument("--dump", action="store_true", default=os.environ.get("DEMO_DUMP", "1") == "1")
    ap.add_argument("--out", default=os.environ.get("DEMO_OUTPUT_PATH",
                    str(pathlib.Path(__file__).resolve().parents[1] / "outputs" / "interactive")))
    args = ap.parse_args()

    eval_args = argparse.Namespace(
        local_rank=0, model_path=os.environ["MODEL_PATH"],
        habitat_config_path=DEMO_CONFIG, eval_split=args.split,
        output_path=args.out, num_future_steps=4, num_frames=32, save_video=False,
        num_history=8, max_eval_episodes_per_rank=0,
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "2048")),
        prompt_style="embodied_r1", num_actions=4, max_history_frames=16,
        open_loop_steps=0, inference_backend="vllm_server",
        vllm_server_urls=os.environ["RYNNVLN_VLLM_SERVER_URLS"],
        vllm_model_name=os.environ.get("RYNNVLN_VLLM_MODEL_NAME", "RynnBrain"),
        vllm_timeout=300, vllm_retries=3, vllm_temperature=0.0, vllm_top_p=None,
        vllm_api_key="", vllm_image_format="JPG", vllm_image_quality=90,
        enable_thinking=os.environ.get("RYNNVLN_ENABLE_THINKING", "0") == "1",
        model_max_length=4096, world_size=1, rank=0, gpu=0, port="1111",
        dist_url="env://", device="cpu",
    )
    base.init_distributed_mode(eval_args)

    ev = InteractiveEvaluator(
        config_path=DEMO_CONFIG, split=args.split, env_num=1,
        output_path=args.out, model=None, processor=None, epoch=0, args=eval_args)
    env = ev.config_env()

    picked = [e for e in env.episodes
              if args.scene in e.scene_id and str(e.episode_id) == str(args.episode_id)]
    if not picked:
        sample = sorted({(str(e.episode_id), e.scene_id.split('/')[-2]) for e in env.episodes})[:8]
        raise SystemExit(f"episode not found scene={args.scene} id={args.episode_id}; sample={sample}")
    episode = picked[0]
    instruction = args.instruction or episode.instruction.instruction_text
    scene_id = episode.scene_id.split('/')[-2]

    dump_dir = os.path.join(args.out, f"{scene_id}_{episode.episode_id}")
    if args.dump:
        os.makedirs(dump_dir, exist_ok=True)

    # HTTP server for live view
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    LATEST["map"] = args.map

    env.current_episode = episode
    observations = env.reset()

    print("\n" + "=" * 70)
    print(f"scene={scene_id}  episode={episode.episode_id}")
    print(f"INSTRUCTION: {instruction}")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80))
        host = s.getsockname()[0]; s.close()
    except OSError:
        host = "127.0.0.1"
    print(f"live view -> http://{host}:{args.port}  "
          f"(本机: http://127.0.0.1:{args.port}, ssh -L {args.port}:127.0.0.1:{args.port} <host>)")
    print("keys: w=forward  a=left  d=right  x=stop  n=model next  N=model to-end  q=quit")
    print("=" * 70 + "\n")

    rgb_list, action_seq, action_list = [], [], []
    step_id = 0
    auto = False

    def refresh(info):
        frame = compose_frame(observations["rgb"], info, args.map, step_id, action_seq)
        LATEST["jpg"] = jpg_bytes(frame)
        LATEST["step"] = step_id
        LATEST["info"] = (f"step {step_id} | dist_to_goal {info.get('distance_to_goal', float('nan')):.2f} m "
                          f"| queued {len(action_seq)} | scene {scene_id} ep {episode.episode_id}")

    refresh(env.get_metrics())

    try:
        while not env.episode_over:
            if auto or not sys.stdin.isatty():
                cmd = 'n' if (auto or not sys.stdin.isatty()) else 'q'
                if not sys.stdin.isatty():
                    time.sleep(0.05)
            else:
                cmd = input(f"[step {step_id}] cmd> ").strip() or "n"
                cmd = cmd[:1]

            if cmd == 'q':
                break
            elif cmd == 'N':
                auto = True
                cmd = 'n'
            elif cmd in ACTIONS:
                action = ACTIONS[cmd]
                rgb_list.append(Image.fromarray(observations["rgb"]).convert("RGB"))
                observations = env.step(action)
                step_id += 1
                print(f"  manual {cmd} -> {ACTION_NAME[action]}  (step {step_id})")
                refresh(env.get_metrics())
                continue
            elif cmd != 'n':
                print("  unknown key, use w/a/d/x/n/N/q")
                continue

            # model-driven step (cmd == 'n')
            rgb_list.append(Image.fromarray(observations["rgb"]).convert("RGB"))
            if len(action_seq) == 0:
                curr = rgb_list[-1]
                conv = ev.construction_conversation_embodied(instruction, rgb_list[0:-1], curr)
                t0 = time.time()
                outputs = ev.generate_response(conv)
                action_seq = ev.parse_actions_embodied(outputs)
                if len(action_seq) == 0:
                    action_seq = [0]
                action_seq = action_seq[:ev.open_loop_steps]
                dt = time.time() - t0
                print(f"  model ({dt:.1f}s): {outputs.strip()[:120]}")
                print(f"    -> actions {action_seq}")
                if args.dump:
                    # dump the images of this inference step, prompt order
                    idx = 0
                    n_img = sum(p.get("type") == "image" for m in conv for p in m["content"])
                    for m in conv:
                        for p in m["content"]:
                            if p.get("type") == "image":
                                im = p["image"]
                                if not isinstance(im, Image.Image):
                                    im = Image.fromarray(im)
                                role = "CUR" if idx == n_img - 1 else f"h{idx:02d}"
                                im.convert("RGB").save(os.path.join(
                                    dump_dir, f"step{step_id:03d}_img{idx:02d}_{role}.jpg"), quality=92)
                                idx += 1
                    with open(os.path.join(dump_dir, "outputs.jsonl"), "a") as f:
                        f.write(json.dumps({"step": step_id, "raw": outputs, "actions": action_seq}) + "\n")
            action = action_seq.pop(0)
            action_list.append(action)
            observations = env.step(action)
            step_id += 1
            print(f"  step {step_id}: {ACTION_NAME.get(action, action)}")
            refresh(env.get_metrics())

        m = env.get_metrics()
        print("\n" + "-" * 70)
        print(f"episode over. steps={step_id} success={m['success']:.1f} "
              f"spl={m['spl']:.3f} dist_to_goal={m['distance_to_goal']:.2f} m")
        print("-" * 70)
    except (EOFError, KeyboardInterrupt):
        print("\ninterrupted")
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
