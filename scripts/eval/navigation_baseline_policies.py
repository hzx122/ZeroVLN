"""Official NaVILA / DualVLN policies; RGB+instruction is the only policy input.

Simulation, target lookup and success scoring deliberately live in the evaluator.
"""
from pathlib import Path
import hashlib
import re
import sys

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]


def navila_actions(text):
    """Native trainer's cm/degree quantization, with invalid outputs explicit."""
    text = text.lower()
    if re.search(r"\bstop\b", text):
        return [0]
    for action, phrase, unit, quantum, choices in (
        (1, "move forward", "cm", 25, (25, 50, 75)),
        (2, "turn left", "degree", 15, (15, 30, 45)),
        (3, "turn right", "degree", 15, (15, 30, 45)),
    ):
        if re.search(rf"\bis {phrase}\b", text):
            match = re.search(rf"{phrase} (\d+) {unit}", text)
            amount = int(match[1]) if match else quantum
            if amount % quantum:
                amount = min(choices, key=lambda x: abs(x - amount))
            return [action] * max(1, amount // quantum)
    raise ValueError(f"unrecognized NaVILA output: {text!r}")


class NaVILAPolicy:
    def __init__(self, model_path, **_):
        sys.path.insert(0, str(REPO / "third_party/NaVILA"))
        import torch
        from llava.model.builder import load_pretrained_model
        self.torch = torch
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            str(model_path), "navila-llama3-8b-8f", device_map={"": 0},
            attn_implementation="eager")
        self.model.eval()
        self.reset()

    def reset(self):
        self.frames = []

    def observe(self, rgb):
        self.frames.append(Image.fromarray(rgb).convert("RGB"))

    def predict(self, rgb, instruction, look_down=False):
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import (vlnce_frame_sampling, process_images,
                                    tokenizer_image_token, KeywordsStoppingCriteria)
        self.observe(rgb)
        # Match the evaluation trainer's 512px black padding while reusing
        # the package's identical temporal sampling helper.
        frames = [Image.new('RGB', (512, 512)) for _ in range(max(0, 8-len(self.frames)))] + self.frames
        images = vlnce_frame_sampling(frames, num_frames=8)
        historical = "<image>\n" * 7
        question = (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations {historical}, "
            f'and current observation <image>\n. Your assigned task is: "{instruction}" '
            "Analyze this series of images to decide your next action, which could be turning left or right by a specific degree, "
            "moving forward a certain distance, or stop if the task is completed.")
        conv = conv_templates["llama_3"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX,
                                    return_tensors="pt").unsqueeze(0).cuda()
        pixels = process_images(images, self.processor, self.model.config).half().cuda()
        stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        with self.torch.inference_mode():
            output = self.model.generate(
                ids, images=pixels, do_sample=False, max_new_tokens=32, use_cache=True,
                stopping_criteria=[KeywordsStoppingCriteria([stop], self.tokenizer, ids)],
                pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        if stop and text.endswith(stop):
            text = text[:-len(stop)].strip()
        return {"actions": navila_actions(text), "text": text, "source": "navila"}


class DualVLNPolicy:
    def __init__(self, model_path, depth_path, fast_view='legacy',
                 first_empty_recovery='refresh', slow_call_budget=None, **_):
        if fast_view not in ('legacy', 'down', 'horizontal'):
            raise ValueError(f'unknown S1 view: {fast_view}')
        if first_empty_recovery not in ('refresh', 'left'):
            raise ValueError(f'unknown first S1 recovery: {first_empty_recovery}')
        self.fast_view = fast_view
        self.first_empty_recovery = first_empty_recovery
        self.slow_call_budget = slow_call_budget
        sys.path.insert(0, str(REPO / "third_party/InternNav"))
        import torch
        from transformers import AutoProcessor, AutoTokenizer, PreTrainedModel
        from internnav.model.basemodel.internvla_n1 import internvla_n1_arch
        from internnav.model.basemodel.internvla_n1.internvla_n1 import (
            InternVLAN1ForCausalLM, InternVLAN1ModelConfig)
        from internnav.model.basemodel.internvla_n1.internvla_n1_policy import InternVLAN1Net
        internvla_n1_arch.MODEL_PATH_TO = str(depth_path)
        self.torch = torch

        class LocalNet(InternVLAN1Net):
            # Only loading differs: explicit local assets and portable attention.
            def __init__(net):
                PreTrainedModel.__init__(net, InternVLAN1ModelConfig.from_pretrained(model_path))
                net.model = InternVLAN1ForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa", device_map={"": 0}).eval()
                net.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                net.processor = AutoProcessor.from_pretrained(model_path)
                net.processor.tokenizer = net.tokenizer
                net.tokenizer.padding_side = "left"
                net.num_history = 8
                net.resize_w = net.resize_h = 384
                net.continuous_traj = True
                net.init_prompts()
                net.reset()

        self.net = LocalNet()
        self.bind_slow_counter()
        kind = self.net.model.get_system1_type()
        if "nextdit" not in kind or "async" not in kind:
            raise ValueError(f"expected RGB nextdit async checkpoint, got {kind}")
        self.reset()

    def bind_slow_counter(self):
        generate = self.net.model.generate

        def counted_generate(*args, **kwargs):
            self.check_slow_budget()
            self.slow_calls += 1
            return generate(*args, **kwargs)

        # generate_latents and generate_traj are deliberately not counted.
        self.net.model.generate = counted_generate

    def reset(self):
        self.net.reset()
        self.latent = None
        self.memory_rgb = None
        self.steps_since_s2 = 0
        self.slow_calls = 0

    def check_slow_budget(self):
        budget = getattr(self, 'slow_call_budget', None)
        if budget is not None and self.slow_calls >= budget:
            raise SlowBudgetExhausted('DualVLN slow-system budget exhausted')

    def observe(self, rgb):
        self.net.step_no_infer(rgb, None, None)

    def predict(self, rgb, instruction, look_down=False, fast_rgb_provider=None):
        # S2 history and S1 observations have independent camera contracts.
        # The provider renders the current S1 view lazily, only when S1 is used.
        new_subgoal = False
        # Official partial_async budget: refresh S2 after eight native actions.
        if self.latent is None or self.steps_since_s2 >= 8 or look_down:
            self.check_slow_budget()
            output = self.net.s2_step(rgb, None, None, instruction, None, look_down=look_down)
            self.steps_since_s2 = 0
            self.latent = output.output_latent
            if output.output_action is not None:
                actions = list(output.output_action)
                if not actions:
                    raise ValueError(f"unrecognized DualVLN output: {self.net.llm_output!r}")
                return {"actions": actions, "text": self.net.llm_output, "source": "s2"}
            new_subgoal = True
        else:
            self.observe(rgb)
        if self.latent is None:
            raise ValueError("DualVLN returned neither actions nor a latent")
        if self.fast_view == 'legacy':
            fast_rgb = rgb
        else:
            if fast_rgb_provider is None:
                raise ValueError('explicit S1 view requires a current-frame provider')
            fast_rgb = fast_rgb_provider()
        if new_subgoal:
            self.memory_rgb = fast_rgb.copy()
        pixels = [np.asarray(Image.fromarray(frame).resize((224, 224))) / 255.
                  for frame in (self.memory_rgb, fast_rgb)]
        pair = self.torch.from_numpy(np.stack(pixels)).unsqueeze(0).to(self.net.device)
        # Reuse official S1 trajectory -> 25cm/15degree action conversion and prefix4.
        result = self.net.s1_step_latent(pair, None, self.latent)
        actions = list(result.idx)
        recovery = None
        if not actions and new_subgoal and self.first_empty_recovery == 'left':
            # Habitat's first S1 STOP is a failed local proposal: turn once and
            # ask S2 again. Later empty chunks only refresh, without a turn.
            actions = [2]
            recovery = 'first_empty_s1_left'
        self.steps_since_s2 += len(actions)
        if len(actions) < 4:
            # Match the upstream agent's early subgoal completion refresh.
            # An empty displacement is not STOP.
            self.latent = None
        return {"actions": actions, "text": self.net.llm_output, "source": "s1",
                "s1_view": self.fast_view, "new_subgoal": new_subgoal,
                "recovery": recovery,
                "s1_anchor_rgb_sha256": hashlib.sha256(self.memory_rgb.tobytes()).hexdigest(),
                "s1_current_rgb_sha256": hashlib.sha256(fast_rgb.tobytes()).hexdigest()}


class SlowBudgetExhausted(RuntimeError):
    """A valid terminal condition, separate from model/environment failure."""
