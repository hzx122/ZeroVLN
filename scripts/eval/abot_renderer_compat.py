"""Apply the established PPU renderer compatibility in memory to pinned Git code.

The upstream checkout stays clean. Both the upstream source and the resulting
historically validated source are checked before compiling any patched module.
"""
import hashlib
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import sys

CONTRACT = json.loads(Path(__file__).with_suffix('.json').read_text())


def patched_source(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    record = CONTRACT['files'].get(str(path.relative_to(root)))
    source = path.read_text()
    if record is None:
        return source
    if hashlib.sha256(source.encode()).hexdigest() != record['base_sha256']:
        raise ValueError('renderer upstream Git source changed: ' + str(path))
    lines = source.splitlines(keepends=True)
    for block in reversed(record['blocks']):
        if ''.join(lines[block['start']:block['stop']]) != block['before']:
            raise ValueError('renderer compatibility context changed')
        lines[block['start']:block['stop']] = [block['after']]
    source = ''.join(lines)
    if hashlib.sha256(source.encode()).hexdigest() != record['result_sha256']:
        raise ValueError('renderer compatibility result differs')
    return source


def install(root):
    root = Path(root).resolve()

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            return compile(patched_source(self.path, root), self.path, 'exec')

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec and spec.origin:
                origin = Path(spec.origin).resolve()
                if origin.is_relative_to(root) and str(origin.relative_to(root)) in CONTRACT['files']:
                    spec.loader = Loader(fullname, str(origin))
                    return spec
            return None

    sys.meta_path.insert(0, Finder())
