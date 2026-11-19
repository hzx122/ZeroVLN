"""Production-side composition for the fast/slow action expert Agent.

This module is intentionally owned by the Agent/VERL server.  The remote
``vln_demo`` process only renders observations and executes trajectory
prefixes; it never receives Qwen hidden states or action expert tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Optional

from .query_provider import VllmQueryStateProvider
from .policy import Qwen35VisualTokenEncoder, TorchActionHeadPolicy, TrajectoryTransform
from .render_client import RemotePrefixExecutor, RemoteRenderClient
from .runtime import (
    ActionHeadPolicy,
    ActionHeadRuntimeConfig,
    ActionHeadToolRuntime,
    RunActionHeadExecutor,
)


@dataclass(frozen=True)
class AgentRuntimeConfig:
    render_url: str
    render_timeout_s: float = 0.0
    option: ActionHeadRuntimeConfig = field(default_factory=ActionHeadRuntimeConfig)

    def validate(self) -> None:
        if not self.render_url.startswith(("http://", "https://")):
            raise ValueError("render_url must be an HTTP(S) URL")
        if not math.isfinite(self.render_timeout_s) or self.render_timeout_s < 0:
            raise ValueError("render_timeout_s must be finite and nonnegative (0 waits indefinitely)")
        self.option.validate()


class ActionHeadAgentRuntime:
    """VERL-side owner of hidden capture, native-HF policy and remote env.

    The slow Agent still owns conversation construction and vLLM generation.
    When that generation emits ``run_action_head``, it passes the same request
    and compact response snapshot here.  The runtime resolves the generated
    contract hidden state locally, calls the native PyTorch/HF action expert, and
    sends only trajectory prefixes to the render server.
    """

    def __init__(self, policy: ActionHeadPolicy,
                 hidden_provider: VllmQueryStateProvider,
                 config: AgentRuntimeConfig,
                 interrupted=None, progress_callback=None):
        config.validate()
        self.config = config
        self.query_count = hidden_provider.query_count
        self.requires_query_tail = getattr(hidden_provider, "requires_query_tail", True)
        self.render_client = RemoteRenderClient(
            config.render_url, timeout=config.render_timeout_s)
        option_runtime = ActionHeadToolRuntime(
            policy=policy,
            executor=RemotePrefixExecutor(self.render_client),
            config=config.option,
            interrupted=interrupted,
            progress_callback=progress_callback,
        )
        self.tool_executor = RunActionHeadExecutor(
            option_runtime, slow_condition_provider=hidden_provider)

    def reset_environment(self, *, scene: Optional[str] = None,
                          episode_id: Optional[str] = None,
                          instruction: Optional[str] = None,
                          seed: Optional[int] = None,
                          backend: Optional[str] = None):
        """Reset the remote environment and return its sanitized observation."""
        result = self.render_client.reset(
            scene=scene, episode_id=episode_id, instruction=instruction, seed=seed,
            **({"backend": backend} if backend is not None else {}))
        reset_provider = getattr(self.tool_executor.slow_condition_provider, 'reset', None)
        if reset_provider is not None:
            reset_provider()
        return result

    def render_map(self, show_goal=False):
        """Return the remote model-safe map; GT display requires explicit opt-in."""
        return self.render_client.render_map(show_goal=show_goal)

    def ui_metrics(self):
        """Return remote scoring fields for UI display, outside model input."""
        return self.render_client.ui_metrics()

    def execute_tool(self, *, option_id: str, instruction: str,
                     arguments: Dict[str, Any], current_observation: Any,
                     slow_request_snapshot: Any,
                     slow_response_snapshot: Any,
                     action_head_as_tool_version: str = "v1"):
        return self.tool_executor.execute(
            option_id=option_id,
            instruction=instruction,
            arguments=arguments,
            current_observation=current_observation,
            slow_request_snapshot=slow_request_snapshot,
            slow_response_snapshot=slow_response_snapshot,
            action_head_as_tool_version=action_head_as_tool_version,
        )

    def execute_turn(self, *, direction: str, degrees: float):
        """Run the v4 slow-system turn against the same remote environment."""
        return self.render_client.execute_turn(direction, degrees)

    def execute_recovery_turn(self, *, direction: str, degrees: float):
        """Run a bounded v4.1 recovery turn on the remote environment."""
        return self.render_client.execute_recovery_turn(direction, degrees)

    def execute_backoff(self, *, distance: float):
        """Retreat on the same remote environment, retaining its latest pose info."""
        return self.render_client.execute_backoff(distance)


def build_action_head_agent_runtime(*, checkpoint: str, backbone_model: str,
                                    query_cache_root: str, gym_url: str,
                                    device: str = "cuda", dtype=None,
                                    visual_device: str | None = None,
                                    visual_dtype=None,
                                    api_key: str = "", seed: int | None = None,
                                    trajectory_transform: TrajectoryTransform | None = None,
                                    option_config: ActionHeadRuntimeConfig | None = None,
                                    shared_policy: ActionHeadPolicy | None = None,
                                    shared_condition_provider=None,
                                    context_encoder_model: str | None = None,
                                    context_aggregator_checkpoint: str | None = None,
                                    context_image_cache: bool = False,
                                    interrupted=None,
                                    progress_callback=None) -> ActionHeadAgentRuntime:
    """Explicit production assembly entry for the Agent/VERL process."""
    import torch

    dtype = dtype or torch.bfloat16
    visual_device = visual_device or device
    visual_dtype = visual_dtype or dtype
    del api_key
    if shared_policy is None:
        visual_encoder = Qwen35VisualTokenEncoder.from_pretrained(
            backbone_model, device=visual_device, dtype=visual_dtype)
        policy_class = TorchActionHeadPolicy
        if context_encoder_model:
            from .context_encoder import NativeContextActionHeadPolicy
            policy_class = NativeContextActionHeadPolicy
        policy = policy_class.from_pretrained(
            checkpoint, visual_encoder, device=device, dtype=dtype, seed=seed,
            trajectory_transform=trajectory_transform)
    else:
        policy = shared_policy
    if shared_condition_provider is not None:
        hidden_provider = shared_condition_provider
    elif context_encoder_model:
        from .context_encoder import NativeContextProvider
        hidden_provider = NativeContextProvider(
            context_encoder_model, device=device, dtype=dtype, seed=seed or 20260917,
            aggregator_checkpoint=context_aggregator_checkpoint, image_cache=context_image_cache)
    else:
        hidden_provider = VllmQueryStateProvider(
            query_cache_root,
            hidden_size=policy.model.config.qwen_hidden_size,
            query_count=policy.model.config.query_num_queries)
    return ActionHeadAgentRuntime(
        policy=policy,
        hidden_provider=hidden_provider,
        config=AgentRuntimeConfig(
            render_url=gym_url,
            option=option_config or ActionHeadRuntimeConfig(),
        ),
        interrupted=interrupted,
        progress_callback=progress_callback,
    )
