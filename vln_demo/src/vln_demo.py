# VLN-CE demo runner: subclasses VLNEvaluator (keeps Habitat env + top_down_map
# config + vLLM client), but runs only a few episodes and dumps the full RGB
# frame sequence for every inference step (current view + history frames,
# in prompt order) as JPGs, alongside the official per-episode mp4.
import sys
import os

NAV_DIR = os.environ.get(
    "RYNNBRAIN_NAV_DIR",
    "/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation",
)
# Hydra resolves config paths relative to CWD; the runner cds to NAV_DIR and
# drops vln_demo.yaml there.
DEMO_CONFIG = os.environ.get("DEMO_CONFIG", "vln_demo.yaml")
sys.path.insert(0, os.path.join(NAV_DIR, "src"))

os.environ.setdefault("RYNNVLN_LANGUAGE", "en")

import argparse
import json

import tqdm

import rynnvln_eval as base
from PIL import Image

# ---- demo knobs (env-overridable) ----
NUM_EPISODES = int(os.environ.get("DEMO_NUM_EPISODES", "3"))
SCENE_CONTAINS = os.environ.get("DEMO_SCENE", "2azQ1b91cZZ")
WANT_EPISODE_IDS = {e for e in os.environ.get(
    "DEMO_EPISODE_IDS", "10,11,12").split(",") if e != ""}
MAX_STEPS = int(os.environ.get("DEMO_MAX_STEPS", "300"))
DEBUG = False


class DemoVLNEvaluator(base.VLNEvaluator):
    def _dump_conversation_images(self, conversations, dump_dir, step_id):
        os.makedirs(dump_dir, exist_ok=True)
        idx = 0
        n_img = sum(p.get("type") == "image" for m in conversations for p in m.get("content", []))
        for msg in conversations:
            for part in msg.get("content", []):
                if part.get("type") != "image":
                    continue
                img = part["image"]
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                img = img.convert("RGB").copy()
                role = "CUR" if idx == n_img - 1 else f"h{idx:02d}"
                img.save(os.path.join(dump_dir, f"step{step_id:03d}_img{idx:02d}_{role}.jpg"), quality=92)
                idx += 1

    def eval_action(self, idx) -> None:
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            scene_episode_dict.setdefault(episode.scene_id, []).append(episode)

        sucs, spls, oss, ones = [], [], [], []
        print(self.output_path)
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            process_bar = tqdm.tqdm(range(len(episodes)), desc=f"scene {scene_id}")
            for episode in episodes:
                episode_instruction = episode.instruction.instruction_text
                print("episode start", episode_instruction)
                episode_id = episode.episode_id
                env.current_episode = episode
                observations = env.reset()
                os.makedirs(os.path.join(self.output_path, 'check_sim_0'), exist_ok=True)
                Image.fromarray(observations['rgb']).save(
                    os.path.join(self.output_path, 'check_sim_0', f'rgb_{episode_id}.jpg'))

                dump_dir = os.path.join(self.output_path, 'frames', f'{scene_id}_{episode_id}')
                vis_frames = []
                step_id = 0

                if self.save_video:
                    os.makedirs(os.path.join(self.output_path, 'vis_0', f'{scene_id}_{episode_id}'), exist_ok=True)

                rgb_list = []
                action_seq = []
                action_list = []

                while not env.episode_over:
                    if step_id >= MAX_STEPS:
                        break
                    rgb = observations["rgb"]
                    x, y = observations["gps"]
                    camera_yaw = observations["compass"][0]

                    image = Image.fromarray(rgb).convert('RGB')
                    rgb_list.append(image)

                    info = env.get_metrics()
                    if self.save_video and info['top_down_map'] is not None:
                        frame = base.observations_to_image({'rgb': observations['rgb']}, info)
                        vis_frames.append(frame)

                    if len(action_seq) == 0:
                        curr_rgb = rgb_list[-1]
                        if self.prompt_style == "embodied_r1_interleaved":
                            conversations = self.construction_conversation_embodied_interleaved(
                                episode_instruction, rgb_list[0:-1], curr_rgb)
                        else:
                            conversations = self.construction_conversation_embodied(
                                episode_instruction, rgb_list[0:-1], curr_rgb)

                        outputs = self.generate_response(conversations)
                        self._dump_conversation_images(conversations, dump_dir, step_id)

                        action_seq = self.parse_actions_embodied(outputs)
                        action_list.extend(action_seq)
                        self.log_raw_generation({
                            "scene_id": scene_id,
                            "episode_id": episode_id,
                            "step_id": step_id,
                            "prompt_style": self.prompt_style,
                            "instruction": episode_instruction,
                            "raw_output": outputs,
                            "parsed_actions": action_seq,
                        })
                        if len(action_seq) == 0:
                            action_seq = [0]
                        action_seq = action_seq[:self.open_loop_steps]

                    action = action_seq.pop(0)
                    observations = env.step(action)
                    step_id += 1

                process_bar.update(1)
                metrics = env.get_metrics()
                if self.save_video:
                    base.images_to_video(
                        vis_frames, os.path.join(self.output_path, 'vis_0'),
                        f'{scene_id}_{episode_id}', fps=6, quality=9)
                vis_frames.clear()
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                print(f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, "
                      f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                      f"ne: {metrics['distance_to_goal']}, steps: {step_id}")
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction,
                }
                with open(os.path.join(self.output_path, 'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")


def run_demo():
    args = argparse.Namespace(
        local_rank=0,
        model_path=os.environ["MODEL_PATH"],
        habitat_config_path=DEMO_CONFIG,
        eval_split=os.environ.get("DEMO_SPLIT", "val_unseen"),
        output_path=os.environ["DEMO_OUTPUT_PATH"],
        num_future_steps=4,
        num_frames=32,
        save_video=True,
        num_history=8,
        max_eval_episodes_per_rank=0,
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "2048")),
        prompt_style=os.environ.get("RYNNVLN_PROMPT_STYLE", "embodied_r1"),
        num_actions=int(os.environ.get("RYNNVLN_NUM_ACTIONS", "4")),
        max_history_frames=int(os.environ.get("RYNNVLN_MAX_HISTORY_FRAMES", "16")),
        open_loop_steps=int(os.environ.get("RYNNVLN_OPEN_LOOP_STEPS", "0")),
        inference_backend="vllm_server",
        vllm_server_urls=os.environ["RYNNVLN_VLLM_SERVER_URLS"],
        vllm_model_name=os.environ.get("RYNNVLN_VLLM_MODEL_NAME", "RynnBrain"),
        vllm_timeout=float(os.environ.get("RYNNVLN_VLLM_TIMEOUT", "300")),
        vllm_retries=int(os.environ.get("RYNNVLN_VLLM_RETRIES", "3")),
        vllm_temperature=float(os.environ.get("RYNNVLN_VLLM_TEMPERATURE", "0")),
        vllm_top_p=None,
        vllm_api_key=os.environ.get("RYNNVLN_VLLM_API_KEY", ""),
        vllm_image_format=os.environ.get("RYNNVLN_VLLM_IMAGE_FORMAT", "JPG"),
        vllm_image_quality=int(os.environ.get("RYNNVLN_VLLM_IMAGE_QUALITY", "90")),
        enable_thinking=os.environ.get("RYNNVLN_ENABLE_THINKING", "0") == "1",
        model_max_length=4096,
        world_size=1,
        rank=0,
        gpu=0,
        port="1111",
        dist_url="env://",
        device="cpu",
    )
    base.init_distributed_mode(args)

    evaluator = DemoVLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=1,
        output_path=args.output_path,
        model=None,
        processor=None,
        epoch=0,
        args=args,
    )

    # Pick demo episodes BEFORE eval_action re-creates the env internally.
    # config_env() applies the RYNNVLN_LANGUAGE filter and returns env; we reach
    # the same episode set by pre-filtering on a probe env through evaluator.
    probe_env = evaluator.config_env()
    all_eps = list(probe_env.episodes)
    picked = [ep for ep in all_eps
              if SCENE_CONTAINS in ep.scene_id and str(ep.episode_id) in WANT_EPISODE_IDS]
    picked = picked[:NUM_EPISODES]
    print(f"[demo] picked {len(picked)} episodes: {[ep.episode_id for ep in picked]}")
    if not picked:
        sample = [(str(e.episode_id), e.scene_id.split('/')[-2]) for e in all_eps[:5]]
        raise SystemExit(f"[demo] no episodes matched scene={SCENE_CONTAINS!r} "
                         f"ids={sorted(WANT_EPISODE_IDS)}; sample={sample}")
    for ep in picked:
        print(f"[demo]   ep {ep.episode_id}: {ep.instruction.instruction_text[:140]}")
    probe_env.episodes = picked

    # eval_action() calls self.config_env() again, building a fresh env from the
    # same dataset; to keep OUR picked subset, monkeypatch config_env to return
    # the probe env the first time.
    orig_config_env = evaluator.config_env
    state = {"used": False}

    def config_env_once():
        if not state["used"]:
            state["used"] = True
            return probe_env
        return orig_config_env()

    evaluator.config_env = config_env_once
    evaluator.eval_action(0)
    print("[demo] done")


if __name__ == "__main__":
    run_demo()
