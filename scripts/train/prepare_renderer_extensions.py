"""Install two verified renderer binary dependencies into container-local cache.

Python renderer sources stay in the main Git checkout. This is an immutable
compiled-library input seed, not an experiment source bundle or writable cache.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
DATA = Path('/data/zerovln')
FILES = {
    'diff_plane_rasterization/_C.so': 'a518fedff2810672a29fccee624169a8fb399277cc639bcb6bbc98f128b5ec1a',
    'simple_knn/_C.so': '6bbc9159c9bdbfe627db6374cfd3bc6d4407a934616f5a13f57c212f3a768715',
}


def install(seed, local):
    spec = importlib.util.spec_from_file_location('_renderer_binary_common',
        REPO/'verl/scripts/eval/qwen36_4query_ckpt_sweep/common.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    seed, local = Path(seed).resolve(), Path(local).resolve()
    if not seed.is_relative_to(DATA):
        raise ValueError('compiled input seed must be on the authorized data disk')
    if local.is_relative_to('/data/storage') or local.is_relative_to('/data/storage'):
        raise ValueError('renderer binary runtime cache must be container-local')
    manifest = json.loads((seed/'manifest.json').read_text())
    if (manifest['schema'] != 'abot-renderer-ppu-binaries-v1'
            or manifest['python_version'] != list(sys.version_info[:2])
            or manifest['files'] != FILES):
        raise ValueError('renderer binary seed schema, Python ABI or file set mismatch')
    for name, digest in manifest['files'].items():
        source, target = seed/name, local/name
        if common.sha256_file(source) != digest:
            raise ValueError('renderer binary seed hash mismatch: '+str(source))
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or common.sha256_file(target) != digest:
            shutil.copy2(source, target)
        if common.sha256_file(target) != digest:
            raise ValueError('local renderer binary hash mismatch: '+str(target))
    return dict(seed=str(seed), local_root=str(local), files=manifest['files'], source_copied=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--local-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.seed, args.local_root)))
