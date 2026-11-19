"""Ray-owned lifecycle for the production four-query vLLM wrapper."""
from __future__ import annotations

import os
import asyncio
import json
import hashlib
import http.server
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class VllmService:
    """Own one tensor-parallel vLLM server on the GPUs assigned by Ray."""

    def __init__(self, *, repo: str, model: str, query_checkpoint: str,
                 query_cache: str, log_file: str, port: int,
                 capture_full_routes: bool, startup_timeout: int = 1800,
                 bulk_route_capture: bool = False, max_num_seqs: int = 0,
                 max_num_batched_tokens: int = 0,
                 gpu_memory_utilization: float = 0.9,
                 enable_sleep_mode: bool = False,
                 tensor_parallel_size: int = 2,
                 cuda_visible_devices: str | None = None,
                 replica_rank: int = 0,
                 direct_weight_sync: bool = False, max_images: int = 64, continuous_context: bool = False,
                 v8_prefix_cache: bool = False, native_colocated_weight_sync: bool = False,
                 v8_cuda_graphs: bool = False):
        if max_images <= 0:
            raise ValueError("max_images must be positive")
        self.max_images = int(max_images)
        self.continuous_context = bool(continuous_context)
        self.v8_cuda_graphs = bool(v8_cuda_graphs)
        if self.v8_cuda_graphs and not (self.continuous_context and capture_full_routes):
            raise ValueError('v8 CUDA graphs require continuous context and full routes')
        self.v8_prefix_cache = bool(v8_prefix_cache)
        if self.v8_prefix_cache and not (self.continuous_context and capture_full_routes):
            raise ValueError("v8 prefix cache requires continuous full-route capture")
        self.repo = Path(repo)
        self.model = model
        self.query_checkpoint = query_checkpoint
        self.query_cache = query_cache
        self.log_file = log_file
        self.port = int(port)
        self.capture_full_routes = bool(capture_full_routes)
        self.startup_timeout = int(startup_timeout)
        self.bulk_route_capture = bool(bulk_route_capture)
        self.max_num_seqs = int(max_num_seqs)
        self.max_num_batched_tokens = int(max_num_batched_tokens)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1)")
        self.enable_sleep_mode = bool(enable_sleep_mode)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.cuda_visible_devices = cuda_visible_devices
        self.replica_rank = int(replica_rank)
        self.direct_weight_sync = bool(direct_weight_sync)
        self.native_colocated_weight_sync = bool(native_colocated_weight_sync)
        if self.native_colocated_weight_sync and (self.direct_weight_sync or not self.enable_sleep_mode):
            raise ValueError("native colocation requires sleep and excludes NCCL relay transport")
        self._native_sync_ack = None
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        self.weight_sync = os.environ.get("ACTION_HEAD_VLLM_WEIGHT_SYNC", "restart")
        if self.weight_sync not in ("restart", "inplace"):
            raise ValueError("ACTION_HEAD_VLLM_WEIGHT_SYNC must be restart or inplace")
        self.process: subprocess.Popen | None = None
        self._log_handle = None
        self._log_spool = None
        self._artifact_server = None
        self._artifact_thread = None
        self._rpc_root = None
        self._direct_checkpoint_engine = None
        self._direct_sender = None
        self._direct_sync_ack = None

    def _start_artifact_server(self) -> None:
        if self._artifact_server is not None:
            return
        root = Path(self.query_cache).resolve()
        retain_for_continuation = self.continuous_context

        class ArtifactHandler(http.server.BaseHTTPRequestHandler):
            def artifact_path(handler):
                name = urllib.parse.unquote(handler.path.removeprefix('/'))
                request_id = name.removesuffix('.safetensors')
                if (not name.endswith('.safetensors') or not request_id
                        or any(char not in
                               'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_'
                               for char in request_id)):
                    handler.send_error(400)
                    return None
                return root / name

            def do_GET(handler):
                source = handler.artifact_path()
                if source is None:
                    return
                try:
                    size = source.stat().st_size
                    handler.send_response(200)
                    handler.send_header('Content-Type', 'application/octet-stream')
                    handler.send_header('Content-Length', str(size))
                    handler.send_header('X-Action-Head-Artifact-Lifetime',
                        'continuous-parent-v1' if retain_for_continuation else 'consume-once')
                    handler.end_headers()
                    with source.open('rb') as stream:
                        while chunk := stream.read(1024 * 1024):
                            handler.wfile.write(chunk)
                    # v8's next request reads this same native R3 artifact on
                    # every TP worker. Download completion is not its lifetime
                    # boundary; the rollout owner releases it after the child
                    # request completes (or the episode ends).
                    if not retain_for_continuation:
                        source.unlink(missing_ok=True)
                except FileNotFoundError:
                    handler.send_error(404)

            def do_DELETE(handler):
                if not retain_for_continuation:
                    handler.send_error(405)
                    return
                source = handler.artifact_path()
                if source is None:
                    return
                source.unlink(missing_ok=True)
                handler.send_response(204)
                handler.end_headers()

            def log_message(self, format, *args):
                return

        self._artifact_server = http.server.ThreadingHTTPServer(('0.0.0.0', 0), ArtifactHandler)
        self._artifact_thread = threading.Thread(
            target=self._artifact_server.serve_forever,
            name='action-head-artifacts', daemon=True)
        self._artifact_thread.start()

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=3) as response:
                return response.status == 200
        except Exception:
            return False

    def start(self) -> str:
        if self.process is not None and self.process.poll() is None:
            return self.url()
        Path(self.query_cache).mkdir(parents=True, exist_ok=True)
        self._start_artifact_server()
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        # Match the formal bootstrap RPC budget. The upstream 300s default
        # can kill a healthy TP worker on a long multimodal sampling batch.
        # Keep explicit operator overrides and the separate control timeout.
        env.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
        # Ray propagates the driver's environment verbatim, while /tmp is
        # node-local.  Recreate inherited local paths on the actor node before
        # nvcc/clang, DeepGEMM, Triton, or TorchInductor tries to use them.
        for key in (
            "TMPDIR", "XDG_CACHE_HOME", "HF_HOME",
            "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
            "PYTORCH_KERNEL_CACHE_PATH", "TORCH_EXTENSIONS_DIR",
            "ACTION_HEAD_COMPILE_CACHE_ROOT",
            "ACTION_HEAD_PROCESS_COMPILE_CACHE_ROOT",
        ):
            value = env.get(key)
            if value:
                Path(value).mkdir(parents=True, exist_ok=True)
        # The driver may export a long experiment/cache path. AF_UNIX has a
        # 107-byte pathname limit, including vLLM's appended UUID. Each actor
        # therefore owns a short local root, independent of inherited paths
        # and other actors; startup retries only remove this private root.
        rpc_root = tempfile.mkdtemp(prefix=f"ahv-{self.port}-", dir="/tmp")
        self._rpc_root = Path(rpc_root)
        env["VLLM_RPC_BASE_PATH"] = rpc_root
        env["ACTION_HEAD_DIRECT_WEIGHT_SYNC_ROOT"] = rpc_root
        env["ACTION_HEAD_NATIVE_COLOCATED_WEIGHTS"] = str(int(self.native_colocated_weight_sync))
        if (self.weight_sync == "inplace" or self.enable_sleep_mode
                or self.direct_weight_sync):
            env["VLLM_SERVER_DEV_MODE"] = "1"
        assigned = (self.cuda_visible_devices
                    if self.cuda_visible_devices is not None
                    else env.get("CUDA_VISIBLE_DEVICES", ""))
        if len([item for item in assigned.split(",") if item]) != self.tensor_parallel_size:
            raise RuntimeError(
                "the four-query vLLM actor requires one assigned GPU per "
                f"tensor-parallel rank: tp={self.tensor_parallel_size}, got {assigned!r}")
        env.update({
            "CUDA_VISIBLE_DEVICES": assigned,
            "VLLM_GPUS": assigned,
            "VLLM_TP_SIZE": str(self.tensor_parallel_size),
            "VLLM_PORT": str(self.port),
            "VLLM_MODEL": self.model,
            "VLLM_MAX_IMAGES": str(self.max_images),
            "ACTION_HEAD_CONTINUOUS_CONTEXT": str(int(self.continuous_context)),
            "ACTION_HEAD_V8_CUDA_GRAPHS": str(int(self.v8_cuda_graphs)),
            "ACTION_HEAD_V8_PREFIX_CACHE": str(int(self.v8_prefix_cache)),
            "VLLM_MTP_TOKENS": "0",
            "ACTION_HEAD_VLLM_ENABLE_SLEEP_MODE": str(int(self.enable_sleep_mode)),
            "ACTION_HEAD_BULK_ROUTE_CAPTURE": str(int(self.bulk_route_capture)),
            "VLLM_MAX_NUM_SEQS": str(self.max_num_seqs),
            "VLLM_MAX_NUM_BATCHED_TOKENS": str(self.max_num_batched_tokens),
            # Ray actors do not reliably inherit arbitrary driver exports on
            # every DLC runtime. Pass this through the constructor explicitly.
            "VLLM_GPU_MEMORY_UTILIZATION": str(self.gpu_memory_utilization),
            "ACTION_HEAD_ENABLE_QUERY_TAIL": "1",
            "ACTION_HEAD_QUERY_CHECKPOINT": self.query_checkpoint,
            "ACTION_HEAD_QUERY_CACHE": self.query_cache,
            "ACTION_HEAD_CAPTURE_ROUTING": "1",
            "ACTION_HEAD_CAPTURE_FULL_ROUTING": (
                "1" if self.capture_full_routes else "0"),
            "VERL_REPLICA_RANK": str(self.replica_rank),
            "ACTION_HEAD_VLLM_DIRECT_WEIGHT_SYNC": (
                "1" if self.direct_weight_sync else "0"),
            "ACTION_HEAD_QUERY_HOT_RELOAD": "1",
            # Full R3 needs a route row for every prefix token on every
            # rollout. Prefix-cache hits bypass MoE execution and therefore
            # cannot produce a complete replay artifact.
            "VLLM_ENABLE_PREFIX_CACHING": (
                "1" if self.v8_prefix_cache or not self.capture_full_routes else "0"),
            "ACTION_HEAD_COMPILE_CACHE_ROOT": (
                env.get("ACTION_HEAD_COMPILE_CACHE_ROOT",
                        str(Path(self.log_file).parent / f"vllm_cache_{self.port}"))),
        })
        if self.continuous_context:
            # Only stable-address tensor replay can participate in a graph.
            # The default Python-segment replay must remain eager.
            env["VLLM_ENFORCE_EAGER"] = "0" if self.v8_cuda_graphs else "1"
        attempts = max(1, int(os.environ.get(
            "ACTION_HEAD_VLLM_START_ATTEMPTS", "2")))
        handshake_timeout = max(30, int(os.environ.get(
            "ACTION_HEAD_VLLM_HANDSHAKE_TIMEOUT", "120")))
        last_error = None
        for attempt in range(1, attempts + 1):
            # vLLM uses filesystem-backed ZMQ endpoints.  A failed handshake
            # can leave stale endpoint names even though EngineCore and the API
            # parent are both alive, so every attempt starts with a clean root.
            shutil.rmtree(rpc_root, ignore_errors=True)
            Path(rpc_root).mkdir(parents=True, exist_ok=True, mode=0o700)
            self._rpc_root = Path(rpc_root)
            active_log = Path(self.log_file)
            if env.get('V6_SERVICE_IO_ISOLATION') == '1':
                import importlib.util
                spec = importlib.util.spec_from_file_location('service_log_spool',
                    self.repo / 'scripts/train/service_log_spool.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self._log_spool = module.LogSpool(self.log_file,
                    env.get('V6_SERVICE_LOG_SPOOL_ROOT', '/tmp/v6-service-log-spool'))
                self._log_handle = self._log_spool.handle
                active_log = self._log_spool.path
                log_offset = 0
            else:
                log_offset = active_log.stat().st_size if active_log.exists() else 0
                self._log_handle = open(self.log_file, "ab", buffering=0)
            if attempt > 1:
                self._log_handle.write(
                    f"\n[action-head] retrying vLLM startup attempt "
                    f"{attempt}/{attempts}\n".encode())
            self.process = subprocess.Popen(
                ["bash", str(self.repo / "vln_demo/vllm/vllm-vln-35B.sh")],
                cwd=self.repo, env=env, stdout=self._log_handle,
                stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + self.startup_timeout
            engine_ready_at = None
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    last_error = RuntimeError(
                        f"vLLM exited with rc={self.process.returncode}")
                    break
                if self._healthy():
                    return self.url()
                if engine_ready_at is None:
                    try:
                        with active_log.open("rb") as handle:
                            handle.seek(log_offset)
                            suffix = handle.read()
                        if b"init engine (profile, create kv cache, warmup model) took" in suffix:
                            engine_ready_at = time.monotonic()
                    except OSError:
                        pass
                elif time.monotonic() - engine_ready_at >= handshake_timeout:
                    last_error = TimeoutError(
                        "vLLM EngineCore initialized but its API handshake "
                        f"did not complete within {handshake_timeout}s")
                    break
                time.sleep(3)
            else:
                last_error = TimeoutError(
                    f"vLLM did not become healthy within {self.startup_timeout}s")

            self.stop(preserve_artifact_server=True)
            if attempt == attempts:
                break
        raise type(last_error)(f"{last_error}; see {self.log_file}")

    def url(self) -> str:
        import ray

        return f"http://{ray.util.get_node_ip_address()}:{self.port}"

    def artifact_url(self) -> str:
        import ray
        if self._artifact_server is None:
            raise RuntimeError('artifact server is not running')
        port = self._artifact_server.server_address[1]
        return f"http://{ray.util.get_node_ip_address()}:{port}"

    def restart(self, model: str | None = None) -> str:
        if self.weight_sync == "inplace":
            return self.reload_weights(model or self.model)
        # Rollout actors retain artifact_url() for their lifetime. Keep its
        # listener alive while replacing only the model-serving subprocesses.
        self.stop(preserve_artifact_server=True)
        if model is not None:
            self.model = model
        return self.start()

    def _post(self, path: str, payload: dict | None = None):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", method="POST",
            data=json.dumps(payload or {}).encode(),
            headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.startup_timeout) as response:
                body = response.read()
                return json.loads(body) if body else None
        except urllib.error.HTTPError as error:
            # HTTPError holds an open response and cannot cross Ray; keep the
            # server's error text instead of an opaque RayError.
            detail = error.read().decode(errors='replace')[-4000:]
            raise RuntimeError(f'vLLM POST {path} failed with HTTP {error.code}: {detail}') from None

    def _get(self, path: str):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{self.port}{path}",
                         timeout=self.startup_timeout) as response:
            return json.loads(response.read())

    def pause(self) -> None:
        self._post("/pause?mode=wait&clear_cache=true")
        if not self.is_paused():
            raise RuntimeError("vLLM scheduler did not acknowledge pause")

    def is_paused(self) -> bool:
        return self._get("/is_paused").get("is_paused") is True

    def resume(self) -> None:
        self._post("/resume")

    def sleep(self, level: int = 1) -> None:
        if not self.enable_sleep_mode:
            raise RuntimeError("vLLM sleep was not enabled at startup")
        if level not in (1, 2):
            raise ValueError("vLLM sleep level must be 1 or 2")
        self.pause()
        # Match worker-only wake: core/Executor.sleep keeps a separate flag
        # that worker RPC wake cannot clear, causing the next sleep to no-op.
        result = self._post("/collective_rpc", {
            "method": "sleep", "kwargs": {"level": level},
            "timeout": self.startup_timeout})
        if (not isinstance(result, dict)
                or len(result.get("results", [])) != self.tensor_parallel_size):
            raise RuntimeError(
                f"incomplete TP{self.tensor_parallel_size} worker sleep: {result!r}")
        if not self.is_paused() or self._get("/is_sleeping").get("is_sleeping") is not True:
            raise RuntimeError("vLLM did not acknowledge sleep while paused")

    def wake_workers(self, tags: list[str] | None = None) -> None:
        """Wake TP allocations without core.wake_up's implicit scheduler resume."""
        if not self.enable_sleep_mode or not self.is_paused():
            raise RuntimeError("worker wake requires enabled sleep and a paused scheduler")
        result = self._post("/collective_rpc", {
            "method": "wake_up", "kwargs": {"tags": tags},
            "timeout": self.startup_timeout})
        if (not isinstance(result, dict)
                or len(result.get("results", [])) != self.tensor_parallel_size):
            raise RuntimeError(
                f"incomplete TP{self.tensor_parallel_size} worker wake: {result!r}")
        if not self.is_paused():
            raise RuntimeError("vLLM scheduler resumed during worker-only wake")

    def wake_reload_resume(self, model: str) -> str:
        """Restore a sleeping colocated replica and atomically publish a policy.

        The learner sharing this physical GPU must already be on CPU. Keeping
        the scheduler paused across wake and reload prevents a request from
        observing the previous policy version.
        """
        if not self.enable_sleep_mode or not self.is_paused():
            raise RuntimeError(
                "colocated weight update requires a sleeping, paused service")
        self.wake_workers()
        self.reload_weights(model, resume=False)
        self.resume()
        if self.is_paused():
            raise RuntimeError("vLLM scheduler remained paused after policy publication")
        return self.url()

    def native_weight_target(self):
        if not self.native_colocated_weight_sync or self._rpc_root is None:
            raise RuntimeError('native colocated weight transport is not active')
        import ray
        from verl.utils.device import is_support_ipc
        return dict(root=str(self._rpc_root), tp_size=self.tensor_parallel_size,
            cuda_visible_devices=self.cuda_visible_devices,
            node_id=ray.get_runtime_context().get_node_id(), use_shm=not is_support_ipc())

    def receive_native_weights(self):
        if (not self.native_colocated_weight_sync or self.process is None
                or self.process.poll() is not None or not self.is_paused()):
            raise RuntimeError('native weight receive requires paused colocated service')
        self._native_sync_ack = None
        target = self.native_weight_target()
        if self._get('/is_sleeping').get('is_sleeping') is not True:
            raise RuntimeError('native weight receive requires the completed sleep barrier')
        self.wake_workers(['weights'])
        result = self._post('/collective_rpc', dict(method='update_weights_from_ipc',
            kwargs=dict(use_shm=target['use_shm']), timeout=self.startup_timeout))
        replies = result.get('results', []) if isinstance(result, dict) else []
        if (len(replies) != self.tensor_parallel_size
                or sorted(row['tp_rank'] for row in replies) != list(range(self.tensor_parallel_size))):
            raise RuntimeError('incomplete native TP weight acknowledgements')
        if not self.is_paused():
            raise RuntimeError('native policy resumed before publication barrier')
        self._native_sync_ack = dict(pid=self.process.pid, replies=replies)
        return replies

    def finish_native_weights(self, expected, step):
        ack = self._native_sync_ack
        if (not ack or self.process is None or self.process.poll() is not None
                or self.process.pid != ack['pid'] or not self.is_paused()):
            raise RuntimeError('native completion requires the original paused acknowledged service')
        if any(any(row.get(key) != value for key, value in expected.items()) for row in ack['replies']):
            raise RuntimeError('native received policy differs from the complete Bridge stream')
        with Path(self.query_checkpoint).open('rb') as handle:
            query_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
        result = self._post('/collective_rpc', dict(method='reload_native_query',
            kwargs=dict(expected_sha256=query_sha), timeout=self.startup_timeout))
        queries = result.get('results', []) if isinstance(result, dict) else []
        if (len(queries) != self.tensor_parallel_size
                or sorted(row['tp_rank'] for row in queries) != list(range(self.tensor_parallel_size))
                or any(row['query']['checkpoint_sha256'] != query_sha or row['query']['pending_query_phases']
                       for row in queries)
                or len({row['query']['learned_sha256'] for row in queries}) != 1):
            raise RuntimeError('native query publication failed TP agreement')
        self.wake_workers(['kv_cache'])
        self._post('/reset_mm_cache')
        self.resume()
        if self.is_paused():
            raise RuntimeError('native policy remained paused after validated publication')
        # Each replica owns its log: concurrent cross-node O_APPEND on CPFS
        # can lose receipts even after all in-memory acknowledgements pass.
        with Path(self.log_file).with_suffix('.weight_updates.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(mode='native_colocated_bucketed', step=step,
                **ack, query_sha256=query_sha, transport=expected)) + '\n')
        self._native_sync_ack = None
        return self.url()

    def init_direct_weight_sync(self, bucket_size_mb: int) -> None:
        """Create VERL's NCCL receiver in this GPU-owning relay actor."""
        if self.tensor_parallel_size not in (1, 2):
            raise RuntimeError("direct IPC weight sync requires TP1 or TP2")
        if not self.direct_weight_sync or not self.enable_sleep_mode:
            raise RuntimeError("direct weight sync requires explicit direct and sleep mode")
        if int(bucket_size_mb) <= 0:
            raise ValueError("direct weight bucket must be positive")
        from action_head_rl.direct_nccl import SingleDeviceNCCLCheckpointEngine
        self._direct_checkpoint_engine = SingleDeviceNCCLCheckpointEngine(
            bucket_size=int(bucket_size_mb) << 20,
            group_name="action_head_qwen_direct", rebuild_group=False,
            is_master=False)

    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        if self._direct_checkpoint_engine is None:
            raise RuntimeError("direct weight sync is not initialized")
        if method == "prepare":
            # This relay has a CUDA context separate from the vLLM subprocess.
            import torch
            torch.cuda.empty_cache()
        return getattr(self._direct_checkpoint_engine, method)(*args, **kwargs)

    async def update_weights(self, global_steps: int | None = None) -> dict:
        """Consume one NCCL stream and load every TP worker before any resume."""
        if self._direct_checkpoint_engine is None:
            raise RuntimeError("direct weight sync is not initialized")
        if self._direct_sender is not None:
            raise RuntimeError("previous direct transfer was incomplete; teardown required")
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("direct update requires the original live vLLM service")
        if self._rpc_root is None or not self.is_paused():
            raise RuntimeError("direct weight update requires a private root and paused scheduler")
        pid, started = self.process.pid, time.monotonic()
        with Path(self.query_checkpoint).open("rb") as handle:
            query_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        self._direct_sync_ack = None
        # Allocate communicators with KV asleep; restore only model weights.
        if self.enable_sleep_mode and self._get("/is_sleeping").get("is_sleeping") is True:
            self.wake_workers(["weights"])
        from action_head_rl.direct_weight_sync import TensorParallelWeightSender, PROTOCOL
        sender = TensorParallelWeightSender(
            root=str(self._rpc_root), tensor_parallel_size=self.tensor_parallel_size,
            bucket_size_mb=self._direct_checkpoint_engine.bucket_size >> 20,
            timeout=self.startup_timeout)
        self._direct_sender = sender
        # Eager thread submission is necessary: IPC handshakes block before
        # the sender's first await. create_task(to_thread(...)) deadlocks here.
        rpc = asyncio.get_running_loop().run_in_executor(
            None, self._post, "/collective_rpc", {
                "method": "update_weights_from_ipc",
                "kwargs": {"timeout": self.startup_timeout,
                           "expected_query_sha256": query_sha},
                "timeout": self.startup_timeout})
        try:
            sent = await sender.async_send_weights(
                self._direct_checkpoint_engine.receive_weights())
            result = await rpc
            replies = result.get("results", []) if isinstance(result, dict) else []
            ranks = [reply.get("tp_rank") for reply in replies if isinstance(reply, dict)]
            if (len(replies) != self.tensor_parallel_size
                    or sorted(ranks) != list(range(self.tensor_parallel_size))):
                raise RuntimeError(f"incomplete direct TP{self.tensor_parallel_size} update: {result!r}")
            learned_query_hashes, worker_pids = set(), set()
            for reply in replies:
                if (reply.get("protocol") != PROTOCOL
                        or reply.get("loader") != "reload_weights_layerwise"
                        or reply.get("loaded_parameter_count", 0) <= 0
                        or any(reply.get(key) != value for key, value in sent.items())):
                    raise RuntimeError(f"direct TP acknowledgement differs from sent weights: {reply!r}")
                worker_pid = reply.get("pid")
                if (type(worker_pid) is not int or worker_pid <= 0
                        or worker_pid in worker_pids
                        or reply.get("device") != f"cuda:{reply['tp_rank']}"):
                    raise RuntimeError("direct TP acknowledgement has invalid process/device identity")
                worker_pids.add(worker_pid)
                query = reply.get("query", {})
                if (query.get("checkpoint_sha256") != query_sha
                        or query.get("pending_query_phases") != 0
                        or not query.get("learned_sha256")):
                    raise RuntimeError("direct query acknowledgement differs from current policy")
                learned_query_hashes.add(query["learned_sha256"])
            if len(learned_query_hashes) != 1:
                raise RuntimeError("TP ranks loaded different query embeddings")
            if self.process.pid != pid or self.process.poll() is not None or not self.is_paused():
                raise RuntimeError("vLLM changed/exited/resumed during direct publication")
            self._direct_sync_ack = {
                "mode": "direct_nccl_ipc", "global_steps": global_steps,
                "pid": pid, "replica_rank": self.replica_rank,
                "seconds_transfer": time.monotonic() - started,
                "tp_acknowledgements": replies, **sent}
            self._direct_sender = None
            return dict(self._direct_sync_ack)
        except BaseException:
            # Cancel only this future; HTTP/worker kernels may still be live.
            # Keep failed sender buffers owned until stop() kills consumers.
            rpc.cancel()
            raise

    def resume_after_direct_weight_sync(self) -> str:
        if self._direct_sync_ack is None or self._direct_sender is not None:
            raise RuntimeError("direct resume requires complete TP acknowledgements")
        if (self.process is None or self.process.poll() is not None
                or self.process.pid != self._direct_sync_ack["pid"]
                or not self.is_paused()):
            raise RuntimeError("direct weight sync lost its original paused service")
        if self.enable_sleep_mode and self._get("/is_sleeping").get("is_sleeping") is True:
            self.wake_workers(["kv_cache"])
        self._post("/reset_mm_cache")
        self.resume()
        if self.is_paused():
            raise RuntimeError("vLLM remained paused after direct publication")
        row = {**self._direct_sync_ack, "resumed": True, "mm_cache_reset": True}
        with (Path(self.log_file).parent / "weight_updates.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        self._direct_sync_ack = None
        return self.url()

    def wake_for_shutdown(self) -> None:
        """Map sleeping allocations before terminating the owned allocator."""
        if self.process is None or self.process.poll() is not None:
            return
        if not self.enable_sleep_mode:
            return
        sleeping = self._get("/is_sleeping").get("is_sleeping") is True
        paused = self.is_paused()
        if sleeping:
            if not paused:
                raise RuntimeError("sleeping vLLM scheduler is not paused")
            self.wake_workers()
        if paused:
            self.resume()

    def reload_weights(self, model: str, *, resume: bool = True) -> str:
        """Use the installed vLLM loader without recreating workers or graphs.

        This still reads a complete HF export. A failed RPC leaves generation
        paused, so partially updated weights can never serve the next rollout.
        """
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("in-place update requires a running vLLM service")
        root = Path(model).resolve()
        if not (root / "config.json").is_file():
            raise FileNotFoundError(root / "config.json")
        started = time.monotonic()
        pid = self.process.pid
        self._post("/pause?mode=wait&clear_cache=true")
        result = self._post("/collective_rpc", {
            "method": "reload_weights", "kwargs": {"weights_path": str(root)},
            "timeout": self.startup_timeout})
        if (not isinstance(result, dict)
                or len(result.get("results", [])) != self.tensor_parallel_size):
            raise RuntimeError(
                f"incomplete TP{self.tensor_parallel_size} weight update: {result!r}")
        if self.process.pid != pid or self.process.poll() is not None:
            raise RuntimeError("vLLM process changed during in-place weight update")
        # pause clears EngineCore caches but leaves the API renderer's MM
        # sender cache populated. Clear both ends through the public endpoint
        # before another request can reference a now-missing image hash.
        self._post("/reset_mm_cache")
        if resume:
            self._post("/resume")
        elif not self.is_paused():
            raise RuntimeError("vLLM scheduler resumed during paused weight update")
        self.model = str(root)
        with (Path(self.log_file).parent / "weight_updates.jsonl").open("a") as handle:
            handle.write(json.dumps({"model": self.model, "pid": pid,
                "seconds": time.monotonic() - started, "mode": "inplace"}) + "\n")
        return self.url()

    def stop(self, *, preserve_artifact_server: bool = False) -> None:
        consumers_stopped = False
        if self.process is not None and self.process.poll() is None:
            # vLLM TP workers may create their own sessions. Reuse the
            # bootstrap cleanup that snapshots descendants before terminating
            # the API parent; killpg(API) alone can leave almost all VRAM live.
            for relative in ("verl/scripts/train/action_head_bootstrap", "vln_demo/src"):
                dependency = str(self.repo / relative)
                if dependency not in sys.path:
                    sys.path.insert(0, dependency)
            from collection_runtime import stop_services

            stop_services(
                [self.process.pid], [],
                Path(self.log_file).with_suffix(".shutdown.json"),
                os.environ.get("CUDA_VISIBLE_DEVICES", ""))
            self.process.wait(timeout=10)
            consumers_stopped = True
        if self._direct_sender is not None and not consumers_stopped:
            raise RuntimeError("failed direct IPC buffers retained: consumer exit is unverified")
        self.process = None
        if self._direct_sender is not None:
            self._direct_sender._close()
            self._direct_sender = None
        self._direct_sync_ack = None
        if self._artifact_server is not None and not preserve_artifact_server:
            self._artifact_server.shutdown()
            self._artifact_server.server_close()
            self._artifact_server = None
        if self._artifact_thread is not None and not preserve_artifact_server:
            self._artifact_thread.join(timeout=5)
            self._artifact_thread = None
        if self._rpc_root is not None:
            value = str(self._rpc_root)
            if (value.startswith('/dev/shm/ahvllm_')
                    or value.startswith('/tmp/action_head_rl/')
                    or value.startswith(f'/tmp/ahv-{self.port}-')):
                shutil.rmtree(self._rpc_root, ignore_errors=True)
            self._rpc_root = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        if self._log_spool is not None:
            spool = self._log_spool
            if not spool.close(timeout=5.):
                # Do not start a second archive writer over an incomplete log.
                raise RuntimeError(f'vLLM log archive pending; local bytes retained: {spool.status()}')
            self._log_spool = None
