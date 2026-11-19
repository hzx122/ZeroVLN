"""The slice of ``rynnvln_eval.VLNEvaluator`` that ``stream_chat`` actually uses.

``vln_model.stream_chat(ev, ...)`` only ever reads ``ev.vllm_server_url``,
``ev.args.{vllm_model_name,vllm_api_key,vllm_retries,vllm_timeout}`` and
``ev._to_openai_messages(conv)``.  Reproducing those three things here is what
lets the 3DRS backend run the ReAct console without importing habitat -- which
matters because habitat is what cannot render these scenes in the first place.

``_to_openai_messages`` mirrors the evaluator's version exactly (assistant
turns collapse to plain text, images become data URLs) so the rendered
conversation is byte-identical to the habitat backend's.
"""

from __future__ import annotations

import argparse
import base64
import io
import os


class ModelClient:
    """Duck-type of the evaluator, for the vLLM streaming path only."""

    def __init__(self, url: str = "", model_name: str = "", timeout: float = 300.0,
                 retries: int = 3, api_key: str = "", image_format: str = "JPG",
                 image_quality: int = 90):
        self.vllm_server_url = url
        self.args = argparse.Namespace(
            vllm_model_name=model_name,
            vllm_api_key=api_key,
            vllm_retries=int(retries),
            vllm_timeout=float(timeout),
            vllm_image_format=image_format,
            vllm_image_quality=int(image_quality),
            vllm_temperature=0.0,
            vllm_top_p=None,
        )

    # -- conversation rendering -------------------------------------------
    def _image_to_data_url(self, img) -> str:
        fmt = (self.args.vllm_image_format or "JPG").upper()
        pil_fmt, mime = ("PNG", "png") if fmt == "PNG" else ("JPEG", "jpeg")
        buf = io.BytesIO()
        rgb = img.convert("RGB")
        if pil_fmt == "JPEG":
            rgb.save(buf, pil_fmt, quality=self.args.vllm_image_quality)
        else:
            rgb.save(buf, pil_fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return "data:image/%s;base64,%s" % (mime, b64)

    def _to_openai_messages(self, conversations):
        messages = []
        for message in conversations:
            role = message["role"]
            content = message["content"]
            if role == "assistant":
                messages.append({"role": role, "content": "".join(
                    p.get("text", "") for p in content if p.get("type") == "text")})
                continue
            parts = []
            for p in content:
                if p.get("type") == "text":
                    parts.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "image":
                    parts.append({"type": "image_url",
                                  "image_url": {"url": self._image_to_data_url(p["image"])}})
            messages.append({"role": role, "content": parts})
        return messages


def from_env() -> ModelClient:
    """Build a client seeded from the same env vars the evaluator reads."""
    return ModelClient(
        url=os.environ.get("RYNNVLN_VLLM_SERVER_URLS", "").split(",")[0].strip(),
        model_name=os.environ.get("RYNNVLN_VLLM_MODEL_NAME", ""),
        timeout=float(os.environ.get("RYNNVLN_VLLM_TIMEOUT", "300")),
        retries=int(os.environ.get("RYNNVLN_VLLM_RETRIES", "3")),
        api_key=os.environ.get("RYNNVLN_VLLM_API_KEY", ""),
        image_format=os.environ.get("RYNNVLN_VLLM_IMAGE_FORMAT", "JPG"),
        image_quality=int(os.environ.get("RYNNVLN_VLLM_IMAGE_QUALITY", "90")))
