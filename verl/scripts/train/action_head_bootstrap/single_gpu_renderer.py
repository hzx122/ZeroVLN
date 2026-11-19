"""Launch the existing renderer with serial HTTP execution, bootstrap only."""
import runpy
import sys
import os
from pathlib import Path
from flask import Flask

original_run = Flask.run
def serial_run(self, *args, **kwargs):
    kwargs['threaded'] = False
    kwargs['processes'] = 1
    return original_run(self, *args, **kwargs)

Flask.run = serial_run
source = sys.argv.pop(1)
sys.argv[0] = source
if os.environ.get('ACTION_HEAD_RENDER_REPO'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts/eval'))
    from abot_renderer_compat import install, patched_source
    root = os.environ['ACTION_HEAD_RENDER_REPO']
    install(root)
    exec(compile(patched_source(source, root), source, 'exec'),
         dict(__name__='__main__', __file__=source, __package__=None))
else:
    runpy.run_path(source, run_name='__main__')
