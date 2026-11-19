"""Rotate live repository links into standalone timestamped backups."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
import uuid


def _process(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0], int(fields[1]), fields[19]
    except (OSError, ValueError, IndexError):
        return None


def terminate_tree(proc):
    """PTY job control gives pi/tools their own process groups; stop descendants too."""
    if proc is None:
        return
    processes = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            info = _process(int(entry.name))
            if info:
                processes[int(entry.name)] = info
    owned = {proc.pid} if proc.pid in processes else set()
    while True:
        expanded = owned | {pid for pid, info in processes.items() if info[1] in owned}
        if expanded == owned:
            break
        owned = expanded

    def live():
        proc.poll()
        return [pid for pid in owned if (now := _process(pid)) is not None
                and now[0] != 'Z' and now[2] == processes[pid][2]]

    for sig, timeout in ((signal.SIGTERM, 1), (signal.SIGKILL, 2)):
        for pid in live():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        while live() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not live():
            return
    raise RuntimeError('终端子进程未退出，工作区尚未清空')


def rotate_workspace(workspace, template, config, archives, stop):
    """Move the original directory intact, then publish a prepared replacement.

    The same-filesystem rename preserves large image collections without a copy
    or deletion window. A failure before publication restores the old directory.
    """
    workspace, template, config, archives = map(Path, (workspace, template, config, archives))
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError('工作区必须是现有普通目录，不能是符号链接')
    if config.resolve() == workspace.resolve() or workspace.resolve() in config.resolve().parents:
        raise ValueError('项目配置必须位于工作区之外')
    if archives.resolve() == workspace.resolve() or workspace.resolve() in archives.resolve().parents:
        raise ValueError('备份目录必须位于工作区之外')
    pi = workspace / '.pi'
    if not pi.is_symlink() or pi.resolve(strict=True) != config.resolve(strict=True):
        raise ValueError('工作区 .pi 不是预期的项目配置链接，已取消清空')
    data = template.read_bytes()
    if not data:
        raise ValueError('固定 AGENTS.md 模板为空，已取消清空')
    archives.mkdir(parents=True, exist_ok=True)
    if workspace.stat().st_dev != archives.stat().st_dev:
        raise ValueError('工作区与备份目录不在同一文件系统，已取消清空')
    with (workspace.parent / '.workspace-reset.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-' + uuid.uuid4().hex[:8]
        backup = archives / stamp
        backup.mkdir()
        fresh = Path(tempfile.mkdtemp(prefix='.workspace-reset-', dir=workspace.parent))
        moved = False
        try:
            (fresh / 'AGENTS.md').symlink_to(template.resolve(strict=True))
            (fresh / '.pi').symlink_to(config.resolve(), target_is_directory=True)
            fresh.chmod(workspace.stat().st_mode & 0o777)
            receipt = dict(backup=str(backup), workspace=str(workspace), created_at=stamp,
                           template=str(template), template_sha256=hashlib.sha256(data).hexdigest(),
                           preserved=['AGENTS.md', '.pi'], status='prepared')
            (backup / 'manifest.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
            stop()
            # Freeze the old AGENTS text even if the old workspace used a link.
            old_agents = workspace / 'AGENTS.md'
            if old_agents.is_file():
                (backup / 'AGENTS.before.md').write_bytes(old_agents.read_bytes())
            # Only the known project configuration is dereferenced. Its nested
            # extension/skill links must also become standalone backup files.
            shutil.copytree(config, backup / 'pi.snapshot', symlinks=False)
            workspace.rename(backup / 'workspace')
            moved = True
            try:
                fresh.rename(workspace)
            except BaseException:
                (backup / 'workspace').rename(workspace)
                moved = False
                raise
            # Archive the bytes, not a live reference to a changing template.
            archived_agents = backup / 'workspace/AGENTS.md'
            if archived_agents.is_symlink() and (backup / 'AGENTS.before.md').is_file():
                frozen_agents = backup / 'AGENTS.frozen.tmp'
                frozen_agents.write_bytes((backup / 'AGENTS.before.md').read_bytes())
                frozen_agents.replace(archived_agents)
            (backup / 'workspace/.pi').unlink()
            (backup / 'pi.snapshot').rename(backup / 'workspace/.pi')
            receipt['status'] = 'complete'
            (backup / 'manifest.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
            return receipt
        finally:
            if fresh.exists():
                shutil.rmtree(fresh)
            if not moved:
                # No user files were moved; remove only our empty/prepared backup.
                shutil.rmtree(backup)
