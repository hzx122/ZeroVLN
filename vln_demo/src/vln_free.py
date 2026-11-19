# Free-roam WASD controller for the Habitat VLN scene.
# Press w/a/s/d (or arrow keys) and the agent moves IMMEDIATELY (no Enter).
# The live first-person view + top-down map is served over HTTP so you can
# watch in a browser via SSH port-forward. Model is optional (m = one model step).
import sys
import pathlib
import os
import io
import time
import threading
import argparse
import socket
import termios
import tty
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NAV_DIR = os.environ.get(
    "RYNNBRAIN_NAV_DIR",
    "/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation",
)
DEMO_CONFIG = os.environ.get("DEMO_CONFIG", "vln_demo.yaml")
sys.path.insert(0, os.path.join(NAV_DIR, "src"))
os.environ.setdefault("RYNNVLN_LANGUAGE", "en")

import numpy as np
import rynnvln_eval as base
from habitat.utils.visualizations import maps as base_maps
from PIL import Image, ImageDraw

# action ids in this sim
FWD, LEFT, RIGHT, STOP = 1, 2, 3, 0
ACTION_NAME = {FWD: "forward", LEFT: "left", RIGHT: "right", STOP: "stop", 4: "back"}

LATEST = {"jpg": None, "info": "starting..."}

PAGE = (
    "<!doctype html><html><head><meta charset='utf-8'><title>VLN free roam</title>"
    "<style>body{background:#111;color:#eee;font-family:monospace;margin:10px}"
    "#info{color:#8f8;font-size:15px;padding:6px 0;white-space:pre-wrap}"
    "#view{border:1px solid #444;image-rendering:pixelated;max-width:100%;background:#000}"
    "</style></head><body>"
    "<div id='info'>connecting...</div>"
    "<img id='view' src='/frame.jpg'>"
    "<script>"
    "var img=document.getElementById('view'),info=document.getElementById('info');"
    "function tick(){"
    "  fetch('/info').then(r=>r.text()).then(t=>{info.textContent=t;}).catch(e=>{});"
    "  var im=new Image();"
    "  im.onload=function(){img.src=im.src;};"
    "  im.src='/frame.jpg?t='+(new Date().getTime());"
    "}"
    "setInterval(tick,400);tick();"
    "</script></body></html>"
)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/frame.jpg"):
            data = LATEST["jpg"]
            if data is None:
                self.send_response(204); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path.startswith("/info"):
            body = LATEST["info"].encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def jpg_bytes(img):
    b = io.BytesIO()
    img.convert("RGB").save(b, "JPEG", quality=85)
    return b.getvalue()


# How much to dim the fog-of-war (unexplored) areas. 1.0 = no fog effect at all,
# 0.0 = full black fog. 0.82 keeps the full floor plan visible but lightly shaded,
# so the whole map shows and only explored areas are at full brightness.
FOG_DIM = float(os.environ.get("DEMO_FOG_DIM", "0.82"))


def _render_full_map(tdm):
    """Colorize the FULL floor plan (no crop), overlay fog-of-war as a light dim,
    and draw the agent. tdm is the top_down_map dict from habitat."""
    if not isinstance(tdm, dict) or "map" not in tdm:
        return None
    m = np.asarray(tdm["map"])
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    fog = tdm.get("fog_of_war_mask")
    if fog is not None:
        fog = np.asarray(fog)
    # full map colorized; fog (unexplored) only slightly dimmed, not blacked out
    rgb = base_maps.colorize_topdown_map(m, fog, fog_of_war_desat_amount=FOG_DIM)
    # draw agent marker(s)
    coords = tdm.get("agent_map_coord")
    angles = tdm.get("agent_angle")
    if coords is not None:
        coords = np.atleast_2d(np.asarray(coords))
        angles = np.atleast_1d(np.asarray(angles)) if angles is not None else [0.0] * len(coords)
        for i, pos in enumerate(coords):
            rgb = base_maps.draw_agent(
                image=rgb,
                agent_center_coord=(int(pos[0]), int(pos[1])),
                agent_rotation=float(angles[i]),
                agent_radius_px=max(4, min(rgb.shape[0], rgb.shape[1]) // 40),
            )
    # habitat stores map as (row=y, col=x); match their display orientation
    if rgb.shape[0] > rgb.shape[1]:
        rgb = np.rot90(rgb, 1)
    return rgb


def compose(rgb, info, with_map, step_id):
    cur = Image.fromarray(rgb).convert("RGB")
    if with_map and info.get("top_down_map") is not None:
        mrgb = _render_full_map(info["top_down_map"])
        if mrgb is not None:
            mp = Image.fromarray(mrgb).convert("RGB")
            h = 480
            w = max(1, int(mp.width * h / mp.height))
            mp = mp.resize((w, h))
            dm = ImageDraw.Draw(mp)
            dm.rectangle([0, 0, 44, 16], fill=(0, 0, 0))
            dm.text((4, 2), "MAP", fill=(0, 255, 128))
            canvas = Image.new("RGB", (cur.width + w + 3, h), (0, 200, 120))
            canvas.paste(cur, (0, 0))
            canvas.paste(mp, (cur.width + 3, 0))
            cur = canvas
    return cur


class KB:
    """Single-key reader (no Enter). Uses cbreak on a TTY; falls back to plain
    char reads when stdin is a pipe/fifo (for scripted tests)."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.is_tty = os.isatty(self.fd)
        if self.is_tty:
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *a):
        if self.is_tty:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def key(self, timeout=0.1):
        import select
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return None
        ch = os.read(self.fd, 1).decode("utf-8", "replace")
        if not ch:
            return "eof"
        if ch == "\x1b":  # arrow keys: ESC [ A/B/C/D
            r, _, _ = select.select([sys.stdin], [], [], 0.001)
            if r and os.read(self.fd, 1).decode() == "[":
                r, _, _ = select.select([sys.stdin], [], [], 0.001)
                if r:
                    return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(
                        os.read(self.fd, 1).decode())
            return "esc"
        return ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_id", default=os.environ.get("DEMO_EPISODE_ID", "11"))
    ap.add_argument("--scene", default=os.environ.get("DEMO_SCENE", "2azQ1b91cZZ"))
    ap.add_argument("--split", default=os.environ.get("DEMO_SPLIT", "val_unseen"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("DEMO_HTTP_PORT", "5000")))
    ap.add_argument("--map", action="store_true", default=os.environ.get("DEMO_MAP", "1") == "1")
    ap.add_argument("--out", default=os.environ.get(
        "DEMO_OUTPUT_PATH",
        str(pathlib.Path(__file__).resolve().parents[1] / "outputs" / "free_roam")))
    args = ap.parse_args()

    eval_args = argparse.Namespace(
        local_rank=0, model_path=os.environ.get("MODEL_PATH", ""),
        habitat_config_path=DEMO_CONFIG, eval_split=args.split, output_path=args.out,
        num_future_steps=4, num_frames=32, save_video=False, num_history=8,
        max_eval_episodes_per_rank=0, max_new_tokens=2048, prompt_style="embodied_r1",
        num_actions=4, max_history_frames=16, open_loop_steps=0,
        # free-roam is pure CPU driving; use hf backend (no vLLM needed). We never
        # call generate_response, so model staying None is fine.
        inference_backend="hf",
        vllm_server_urls="",
        vllm_model_name="RynnBrain", vllm_timeout=300, vllm_retries=3,
        vllm_temperature=0.0, vllm_top_p=None, vllm_api_key="",
        vllm_image_format="JPG", vllm_image_quality=90, enable_thinking=False,
        model_max_length=4096, world_size=1, rank=0, gpu=0, port="1111",
        dist_url="env://", device="cpu",
    )
    base.init_distributed_mode(eval_args)

    ev = base.VLNEvaluator(
        config_path=DEMO_CONFIG, split=args.split, env_num=1, output_path=args.out,
        model=None, processor=None, epoch=0, args=eval_args)
    env = ev.config_env()

    picked = [e for e in env.episodes
              if args.scene in e.scene_id and str(e.episode_id) == str(args.episode_id)]
    if not picked:
        raise SystemExit(f"episode not found scene={args.scene} id={args.episode_id}")
    episode = picked[0]
    scene_id = episode.scene_id.split("/")[-2]

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    env.current_episode = episode
    obs = env.reset()

    step_id = 0
    running = True

    def refresh():
        info = env.get_metrics()
        LATEST["jpg"] = jpg_bytes(compose(obs["rgb"], info, args.map, step_id))
        LATEST["info"] = (f"step {step_id} | dist_to_goal {info.get('distance_to_goal', float('nan')):.2f} m "
                          f"| scene {scene_id} ep {episode.episode_id}")

    refresh()
    print("\n" + "=" * 66)
    print(f"FREE ROAM  scene={scene_id} ep={episode.episode_id}")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80))
        host = s.getsockname()[0]; s.close()
    except OSError:
        host = "127.0.0.1"
    print(f"live view -> http://{host}:{args.port}  "
          f"(本机: http://127.0.0.1:{args.port}, ssh -L {args.port}:127.0.0.1:{args.port} <host>)")
    print("WASD/arrows: w/Up=forward  a/Left=left  d/Right=right  s/Down=back(if avail)")
    print("x=stop  r=reset  q=quit   (keys are INSTANT, no Enter needed)")
    print("=" * 66 + "\n", flush=True)

    try:
        with KB() as kb:
            while running and not env.episode_over:
                k = kb.key(timeout=0.1)
                if k is None:
                    continue
                if k in ("q", "esc"):
                    break
                elif k in ("w", "up"):
                    obs = env.step(FWD); step_id += 1; act = "forward"
                elif k in ("a", "left"):
                    obs = env.step(LEFT); step_id += 1; act = "left"
                elif k in ("d", "right"):
                    obs = env.step(RIGHT); step_id += 1; act = "right"
                elif k in ("s", "down"):
                    # try backward if action 4 exists, else small forward-back hint
                    try:
                        obs = env.step(4); act = "back"
                    except Exception:
                        print("(no backward action in this sim)", flush=True)
                        continue
                    step_id += 1
                elif k == "x":
                    obs = env.step(STOP); act = "stop"
                elif k == "r":
                    obs = env.reset(); step_id = 0; act = "reset"
                    print("reset to start", flush=True)
                else:
                    continue
                refresh()
                info = env.get_metrics()
                print(f"[{step_id:4d}] {act:8s} dist={info.get('distance_to_goal', float('nan')):5.2f}m", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        m = env.get_metrics()
        print(f"\ndone. steps={step_id} dist_to_goal={m['distance_to_goal']:.2f}m success={m['success']:.1f}")


if __name__ == "__main__":
    main()
