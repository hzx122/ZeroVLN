"""Install the repository's Next compatibility plugin in private tmpfs.

The public image and the host Python environment stay untouched. The vLLM
general-plugin entry point loads the fix in the API process and every worker.
"""

import os
from pathlib import Path
import sys


BOOTSTRAP = r'''
import os
from pathlib import Path
import sys

root = Path('/runtime-home/vln-next-prefix-cache')
root.mkdir()
(root / 'next_prefix_cache.py').write_text(sys.argv[1])
info = root / 'vln_next_prefix_cache-1.0.dist-info'
info.mkdir()
(info / 'METADATA').write_text('Metadata-Version: 2.1\nName: vln-next-prefix-cache\nVersion: 1.0\n')
(info / 'entry_points.txt').write_text('[vllm.general_plugins]\nvln_next_prefix_cache = next_prefix_cache:install\n')
os.environ['PYTHONPATH'] = str(root) + (':' + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
if os.environ.get('VLLM_PLUGINS'):
    os.environ['VLLM_PLUGINS'] += ',vln_next_prefix_cache'
os.execvp(sys.argv[2], sys.argv[2:])
'''


if __name__ == "__main__":
    runtime_launcher = sys.argv[1]
    plugin = Path(__file__).with_name("next_prefix_cache.py").read_text()
    os.execv(
        "/usr/bin/python3",
        ["/usr/bin/python3", "-I", runtime_launcher,
         "python3", "-c", BOOTSTRAP, plugin, *sys.argv[2:]],
    )
