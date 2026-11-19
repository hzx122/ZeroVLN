"""Owned, disposable single-host policy exports; never resumable checkpoints.

No directory is created during preflight.  The driver creates its private root
inside its cleanup scope, then brackets publication and consumer RPCs explicitly.
A failed RPC is not proof that its remote reader/writer has stopped.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


LOCAL_RUNTIME_BASE = Path("/tmp/action_head_rl")
# Conservative current 35B budget (two BF16 exports plus overhead), not a
# capacity guarantee for arbitrary model sizes. Do not put this cache in shm.
MIN_FREE_BYTES = 160_000_000_000


def _start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def validate_local_transport_config(*, nnodes: int, ray_address: str,
                                    rollout_transport_dtype: str) -> None:
    if nnodes != 1 or ray_address != "local":
        raise ValueError("local policy transport requires nnodes=1 and ray_address=local")
    if rollout_transport_dtype != "bf16":
        raise ValueError("local policy transport requires BF16 rollout transport")


class LocalPolicyTransport:
    def __init__(self, *, output: Path, runtime_root: str, nnodes: int,
                 ray_address: str, rollout_transport_dtype: str):
        validate_local_transport_config(
            nnodes=nnodes, ray_address=ray_address,
            rollout_transport_dtype=rollout_transport_dtype)
        self.output = Path(output).resolve()
        expected = LOCAL_RUNTIME_BASE / (
            "l-" + hashlib.sha256(str(self.output).encode()).hexdigest()[:12])
        if not runtime_root or Path(runtime_root) != expected:
            raise ValueError("local policy transport requires this run's derived ACTION_HEAD_LOCAL_RUNTIME_ROOT")
        self.runtime = expected
        self.root = expected / "policy_transport"
        self.pid = os.getpid()
        self.start_ticks = _start_ticks(self.pid)
        self.owner = {
            "schema": "local_policy_transport_owner_v1", "pid": self.pid,
            "start_ticks": self.start_ticks, "uid": os.getuid(),
            "output": str(self.output), "root": str(self.root),
            "nonce": uuid.uuid4().hex,
        }
        self._root_identity = None
        self._parent_identity = None
        self._phase = "uncreated"
        self._step = None

    @staticmethod
    def _identity(path: Path):
        stat = path.lstat()
        return stat.st_dev, stat.st_ino, stat.st_uid

    def _paths_safe(self) -> None:
        if self.root.resolve() != self.root:
            raise ValueError("local policy transport path contains a symlink")
        for path in (LOCAL_RUNTIME_BASE, self.runtime, self.root):
            if path.exists() and (not path.is_dir() or path.stat().st_uid != os.getuid()):
                raise ValueError(f"local policy transport path is not an owned directory: {path}")

    def _check_free(self) -> int:
        probe = self.runtime
        while not probe.exists():
            probe = probe.parent
        free = shutil.disk_usage(probe).free
        if free < MIN_FREE_BYTES:
            raise OSError(f"local policy transport requires at least {MIN_FREE_BYTES} free bytes; available={free}")
        return free

    def preflight(self) -> int:
        """Read-only gate, including when invoked before Ray/GPU initialization."""
        self._paths_safe()
        if self.root.exists():
            raise FileExistsError(f"refusing to adopt an existing local policy transport: {self.root}")
        return self._check_free()

    def create(self) -> None:
        self.preflight()
        self.runtime.mkdir(parents=True, exist_ok=True)
        self._paths_safe()
        self._parent_identity = self._identity(self.runtime)
        self.root.mkdir(mode=0o700)  # Exclusive: an earlier owner must be reconciled first.
        self._root_identity = self._identity(self.root)
        try:
            with (self.root / "owner.json").open("x") as handle:
                json.dump(self.owner, handle, sort_keys=True)
                handle.write("\n")
        except BaseException:
            # No RPC has been issued. Only our exact newly created directory
            # can be removed; a replaced path is retained for inspection.
            if (self._identity(self.runtime) == self._parent_identity
                    and self._identity(self.root) == self._root_identity):
                shutil.rmtree(self.root)
                self._root_identity = None
            raise
        self._phase = "idle"

    def _assert_owner(self) -> None:
        self._paths_safe()
        if (os.getpid() != self.pid or _start_ticks(self.pid) != self.start_ticks
                or self._identity(self.runtime) != self._parent_identity
                or self._identity(self.root) != self._root_identity):
            raise ValueError("local policy transport owner identity changed")
        marker = self.root / "owner.json"
        if marker.is_symlink() or json.loads(marker.read_text()) != self.owner:
            raise ValueError("local policy transport owner marker changed")
        if not shutil.rmtree.avoids_symlink_attacks:
            raise RuntimeError("local policy transport requires fd-safe rmtree")

    def begin_publication(self, step: int) -> None:
        self._assert_owner()
        if self._phase != "idle" or int(step) <= 0:
            raise ValueError("local policy transport has an unfinished publication or invalid step")
        self._check_free()
        self._phase, self._step = "publishing", int(step)

    def publication_complete(self, step: int) -> None:
        self._transition(step, "publishing", "published")

    def begin_reload(self, step: int) -> None:
        self._transition(step, "published", "reading")

    def _transition(self, step: int, previous: str, following: str) -> None:
        self._assert_owner()
        if self._phase != previous or self._step != step:
            raise ValueError("local policy transport publication/ACK order mismatch")
        self._phase = following

    def consumers_complete(self, step: int) -> None:
        """Call only after every fast-policy, rollout/eval and vLLM RPC ACK."""
        self._transition(step, "reading", "acknowledged")
        path = self.root / f"global_step_{step:06d}"
        if path.is_symlink():
            raise ValueError("local policy transport step directory became a symlink")
        if path.exists():
            shutil.rmtree(path)
        self._phase, self._step = "idle", None

    def cleanup(self) -> dict:
        event = {"root": str(self.root), "phase": self._phase, "step": self._step}
        if self._root_identity is None:
            return {**event, "status": "not_created"}
        self._assert_owner()
        if self._phase in ("publishing", "reading"):
            return {**event, "status": "retained", "reason": "remote_io_not_acknowledged"}
        shutil.rmtree(self.root)
        self._root_identity = None
        self._phase = "removed"
        return {**event, "status": "removed"}


def publication_root(*, output: Path, shared_transport: Path,
                     local: LocalPolicyTransport | None, step: int,
                     checkpoint_due: bool) -> Path:
    if checkpoint_due:
        return output
    if local is not None:
        local.begin_publication(step)
        return local.root
    return shared_transport


@contextmanager
def cleanup_local_transport(local: LocalPolicyTransport | None):
    """Run cleanup even if shutdown raises; never hide its original error."""
    try:
        yield
    finally:
        if local is not None:
            try:
                event = local.cleanup()
            except Exception as exc:
                event = {"status": "retained", "root": str(local.root),
                         "reason": f"{type(exc).__name__}: {exc}"}
            print(json.dumps({"local_policy_transport_cleanup": event}), flush=True)
