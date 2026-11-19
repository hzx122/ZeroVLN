"""Independent small-brain assembly using the existing SFT condition encoder.

No model-server lifecycle, query-tail files, or brain endpoint mutation. Frozen
encoders are shared across checkpoint loads, all on the controller owner thread.
"""
import hashlib


def configure_local_caches():
    """Own runtime-only caches until this console process exits."""
    import atexit
    import os
    from pathlib import Path
    import shutil
    import tempfile
    root = Path(__file__).resolve().parents[2] / '.local_runtime' / f'manual-fm-{os.getpid()}'
    for key in ('TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TRITON_CACHE_DIR',
                'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'PYTORCH_KERNEL_CACHE_PATH',
                'TORCH_EXTENSIONS_DIR', 'DG_CACHE_DIR', 'DG_JIT_CACHE_DIR'):
        path = root / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    tempfile.tempdir = None
    atexit.register(shutil.rmtree, root, ignore_errors=True)


class IndependentQueryProvider:
    requires_query_tail = False

    def __init__(self, model, processor, query, device):
        from sft.joint_fm_head_sft import FrozenQwenQueryForward
        self.frozen = FrozenQwenQueryForward(model, query)
        self.processor, self.device = processor, device
        self.query_count = query.query_count

    def __call__(self, *, request, response, contract):
        import torch
        from action_head.query_provider import VllmQueryHidden
        from sft.joint_fm_head_sft import build_slow_inputs
        del contract
        # Use precisely the brain's sent image window plus its actual tool text.
        inputs = build_slow_inputs(self.processor, request['messages'],
                                   response['text'], self.device)
        if len(inputs.get('image_grid_thw', ())) > 11:
            raise ValueError('小脑条件超过与大脑共用的 8+3 图像窗口')
        with torch.no_grad():
            hidden = self.frozen.forward_full(inputs)[0].float().cpu().numpy()
        digest = hashlib.sha256(hidden.tobytes()).hexdigest()[:16]
        return VllmQueryHidden(hidden, f'independent-{digest}', '',
                              query_count=self.query_count)


class ManualRuntimeFactory:
    def __init__(self, interrupted, progress_callback, load_progress=None):
        self.interrupted, self.progress_callback = interrupted, progress_callback
        self.visuals = {}
        self.query_models = {}
        self.native_frontends = {}
        self.load_progress = load_progress or (lambda phase: None)
        self.cache_configured = False

    def _query_model(self, model_dir, device, dtype):
        from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
        key = (model_dir, device, dtype)
        if key not in self.query_models:
            self.load_progress('首次加载独立 Qwen 条件编码器，约 35B 参数（后续复用）')
            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
            model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
                model_dir, dtype=dtype, device_map=device, low_cpu_mem_usage=True,
                attn_implementation='sdpa', trust_remote_code=True).eval()
            model.requires_grad_(False)
            self.query_models[key] = model, processor
        return self.query_models[key]

    def prewarm(self, settings):
        """Prepare frozen encoders without selecting any trained checkpoint."""
        if not self.cache_configured:
            configure_local_caches()
            self.cache_configured = True
        import torch
        from action_head.policy import Qwen35VisualTokenEncoder
        device = settings.get('device', 'cuda:0')
        dtype = torch.float32 if str(device).startswith('cpu') else torch.bfloat16
        visual_device = settings.get('visual_device', device)
        key = (settings['backbone'], visual_device, dtype)
        if key not in self.visuals:
            self.load_progress('预加载小脑视觉编码器')
            self.visuals[key] = Qwen35VisualTokenEncoder.from_pretrained(
                key[0], device=visual_device, dtype=dtype)
        self._query_model(settings['query_encoder_model'], settings.get('query_device', device), dtype)

    def __call__(self, selection, settings):
        self.load_progress('初始化小脑运行环境')
        if not self.cache_configured:
            configure_local_caches()
            self.cache_configured = True
        import torch
        from action_head.agent_runtime import build_action_head_agent_runtime
        from action_head.policy import Qwen35VisualTokenEncoder, TorchActionHeadPolicy, TrajectoryTransform
        device = settings.get('device', 'cuda:0')
        visual_device = settings.get('visual_device', device)
        dtype = torch.float32 if str(device).startswith('cpu') else torch.bfloat16
        visual_key = (settings['backbone'], visual_device, dtype)
        if visual_key not in self.visuals:
            self.load_progress('首次加载视觉编码器（切换同类权重时复用）')
            self.visuals[visual_key] = Qwen35VisualTokenEncoder.from_pretrained(
                settings['backbone'], device=visual_device, dtype=dtype)
        visual = self.visuals[visual_key]
        self.load_progress('加载 FM Head 权重')
        transform = TrajectoryTransform.from_quantiles(
            [-0.10273411870002747, -0.10309220850467682, -0.5668500065803528],
            [0.1031249612569809, 0.10309189558029175, 0.5747644901275635],
            flip_y=True, flip_yaw=True)
        native = selection['condition_mode'] == 'v5_native'
        policy_class = TorchActionHeadPolicy
        if native:
            from action_head.context_encoder import NativeContextActionHeadPolicy
            policy_class = NativeContextActionHeadPolicy
        policy = policy_class.from_pretrained(selection['checkpoint'], visual,
                    device=device, dtype=dtype, seed=int(settings.get('seed', 20260827)),
                    trajectory_transform=transform)
        provider = None
        if not native:
            from models.dualvln_query_encoder import DualVlnQueryEncoder
            query_device = settings.get('query_device', device)
            model_dir = settings['query_encoder_model']
            model, processor = self._query_model(model_dir, query_device, dtype)
            self.load_progress('加载所选 checkpoint 的 query 权重（复用条件编码器）')
            from pathlib import Path
            query = DualVlnQueryEncoder.from_pretrained(
                Path(selection['condition']).parent).to(device=query_device, dtype=dtype).eval()
            if query.query_count != selection['query_count'] or query.hidden_size != selection['hidden_size']:
                raise ValueError('所选 snapshot 的 query 与 FM Head 维度不匹配')
            if int(model.config.text_config.hidden_size) != query.hidden_size:
                raise ValueError('独立条件编码器与 checkpoint query 维度不匹配')
            provider = IndependentQueryProvider(model, processor, query, query_device)
        else:
            from action_head.context_encoder import NativeContextProvider, NativeQwenFrontend
            key = (settings['query_encoder_model'], device, dtype)
            if key not in self.native_frontends:
                self.load_progress('首次加载 v5 图文编码器（后续复用）')
                self.native_frontends[key] = NativeQwenFrontend(key[0], device, dtype)
            self.load_progress('加载所选 checkpoint 的聚合器（复用 v5 图文编码器）')
            provider = NativeContextProvider(key[0], device=device, dtype=dtype,
                seed=int(settings.get('seed', 20260917)),
                aggregator_checkpoint=selection['condition'], image_cache=True,
                frontend=self.native_frontends[key])
        self.load_progress('组装小脑执行器')
        return build_action_head_agent_runtime(
            checkpoint=selection['checkpoint'], backbone_model=settings['backbone'],
            query_cache_root='', gym_url=settings['gym_url'], device=device, dtype=dtype,
            shared_policy=policy, shared_condition_provider=provider,
            context_encoder_model=(settings['query_encoder_model'] if native else None),
            context_aggregator_checkpoint=(selection['condition'] if native else None),
            context_image_cache=native,
            interrupted=self.interrupted, progress_callback=self.progress_callback)
