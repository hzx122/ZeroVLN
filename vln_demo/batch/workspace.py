"""One-way template snapshots and verified, append-only episode archives."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import uuid

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / 'third_party/pi-agent/.pi/workspaces/SpatialOnboarding/AGENTS.md'
PROJECT_CONFIG = REPO / 'vln_demo/config/pi/SpatialOnboarding'
LOGICAL_WORKSPACE = '/opt/external/user-d5121be6/SpatialOnboarding/workspace'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def freeze_template(run):
    """Read repository once; never write to it and never follow an episode seed."""
    run = Path(run)
    data = TEMPLATE.read_bytes()
    seed = run / 'template'
    seed.mkdir(exist_ok=False)
    (seed / 'AGENTS.md').write_bytes(data)
    (seed / 'AGENTS.md').chmod(0o444)
    version = subprocess.check_output(['git', '-C', str(REPO / 'third_party/pi-agent'),
                                       'rev-parse', 'HEAD'], text=True).strip()
    metadata = dict(source=str(TEMPLATE), git_commit=version, sha256=sha256(data), bytes=len(data))
    (seed / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def new_workspace(run, target):
    # Run snapshots are provenance only; live workspaces always use the repo.
    data = TEMPLATE.read_bytes()
    metadata = dict(source=str(TEMPLATE), sha256=sha256(data), bytes=len(data),
                    mode='repository-symlink')
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    # Absolute path survives the sandbox's private logical workspace mount.
    # Repository targets are outside that writable mount and stay read-only.
    (target / 'AGENTS.md').symlink_to(TEMPLATE.resolve(strict=True))
    (target / '.pi').symlink_to(PROJECT_CONFIG.resolve(strict=True), target_is_directory=True)
    if sorted(p.name for p in target.iterdir()) != ['.pi', 'AGENTS.md']:
        raise ValueError('workspace was not empty')
    return metadata


def inventory(root, *, followlinks=False):
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=followlinks):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            relative = str(path.relative_to(root))
            mode = (path.stat() if followlinks else path.lstat()).st_mode
            if stat.S_ISLNK(mode):
                result[relative] = {'symlink': os.readlink(path)}
            elif stat.S_ISREG(mode):
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                result[relative] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
            elif not stat.S_ISDIR(mode):
                raise ValueError(f'cannot archive special file: {relative}')
    return result


def archive(source, destination, result, *, agents_template=None, project_config=None):
    """Caller must stop all episode processes first. No merge or writeback path.

    Failure preserves local source and incomplete archive for investigation.
    Only explicitly supplied AGENTS/.pi links are frozen to files/directories.
    Other symlinks are copied as links, never traversed into other workspaces.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    for name in ('archive-manifest.json', 'result.json', '_SUCCESS.json'):
        if (source / name).exists() or (source / name).is_symlink():
            raise ValueError('reserved archive metadata path: ' + name)
    expected = inventory(source)
    snapshots = {}
    config_snapshots = {}
    if agents_template is not None:
        template = Path(agents_template).resolve(strict=True)
        for name in ('AGENTS.md', 'workspace/AGENTS.md'):
            path = source / name
            if not path.parent.is_symlink() and path.is_symlink() and path.resolve() == template:
                snapshots[name] = template.read_bytes()
    if project_config is not None:
        config = Path(project_config).resolve(strict=True)
        for name in ('.pi', 'workspace/.pi'):
            path = source / name
            if not path.parent.is_symlink() and path.is_symlink() and path.resolve() == config:
                config_snapshots[name] = inventory(config, followlinks=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name('.' + destination.name + '.partial-' + uuid.uuid4().hex)
    shutil.copytree(source, staging, symlinks=True)
    archived_expected = dict(expected)
    for name, data in snapshots.items():
        path = staging / name
        if path.parent.is_symlink():
            raise OSError('workspace changed to a symlink during archive')
        path.unlink()
        path.write_bytes(data)
        archived_expected[name] = {'sha256': sha256(data), 'bytes': len(data)}
    for name, files in config_snapshots.items():
        path = staging / name
        if path.parent.is_symlink():
            raise OSError('workspace changed to a symlink during archive')
        path.unlink()
        shutil.copytree(config, path, symlinks=False)
        archived_expected.pop(name)
        archived_expected.update({name + '/' + key: value for key, value in files.items()})
    actual = inventory(staging)
    after = inventory(source)
    changed_templates = [name for name, data in snapshots.items()
                         if template.read_bytes() != data]
    changed_templates += [name for name, files in config_snapshots.items()
                          if inventory(config, followlinks=True) != files]
    if actual != archived_expected or after != expected or changed_templates:
        changed = sorted(k for k in expected.keys() | actual.keys() | after.keys()
                         if archived_expected.get(k) != actual.get(k) or expected.get(k) != after.get(k))
        changed = sorted(set(changed) | set(changed_templates))
        (staging / 'verification-error.json').write_text(json.dumps(
            {'changed_files': changed, 'source_changed': after != expected}, indent=2) + '\n')
        raise OSError('archive byte/hash verification failed; local source retained; files=' + ', '.join(changed[:10]))
    (staging / 'archive-manifest.json').write_text(json.dumps(archived_expected, ensure_ascii=False, indent=2) + '\n')
    (staging / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (staging / '_SUCCESS.json').write_text(json.dumps({'archive_complete': True,
        'navigation_success': result.get('navigation_success'), 'files': len(archived_expected)}) + '\n')
    # Atomic publish within the same CPFS directory; each attempt has one owner.
    staging.rename(destination)
    return destination


def verified_result(directory):
    """Read back every archived byte before accepting a resume checkpoint."""
    directory = Path(directory)
    marker = json.loads((directory / '_SUCCESS.json').read_text())
    if not marker.get('archive_complete'):
        raise ValueError('incomplete archive')
    expected = json.loads((directory / 'archive-manifest.json').read_text())
    actual = inventory(directory)
    for name in ('archive-manifest.json', 'result.json', '_SUCCESS.json'):
        actual.pop(name, None)
    if actual != expected:
        raise ValueError('saved archive verification failed: ' + str(directory))
    return json.loads((directory / 'result.json').read_text())
