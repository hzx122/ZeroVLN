import json
import os
import subprocess
import tempfile
from pathlib import Path
import unittest
from sglang_server import checked_info, command
from sglang_probe import cached_tokens


class SGLangServerTests(unittest.TestCase):
    def test_cold_null_usage_details_and_warm_cache_hits(self):
        self.assertEqual(cached_tokens({'usage': {'prompt_tokens_details': None}}), 0)
        self.assertEqual(cached_tokens({}), 0)
        self.assertEqual(cached_tokens({'usage': {'prompt_tokens_details': {'cached_tokens': 8192}}}), 8192)

    def test_each_profile_keeps_tp8_cache_and_no_mtp(self):
        profiles = json.loads(Path(__file__).with_name('sglang_profiles.json').read_text())
        for config in profiles.values():
            with tempfile.TemporaryDirectory() as tmp:
                fake = Path(tmp)/'capture-python'
                fake.write_text('#!/usr/bin/env python3\nimport json,sys,os\nprint(json.dumps(dict(args=sys.argv[1:],gpus=os.environ.get("CUDA_VISIBLE_DEVICES"))))\n')
                fake.chmod(0o755)
                env = dict(os.environ, VLN_SGLANG_PYTHON=str(fake), VLN_SGLANG_LOCAL_CACHE=tmp)
                captured = json.loads(subprocess.check_output(command(config), env=env, text=True))
                args = captured["args"]
                self.assertEqual(captured["gpus"], "4,5,6,7,8,9,10,11")
            self.assertEqual(args[args.index('--model-path')+1], config['model_path'])
            self.assertEqual(args[args.index('--served-model-name')+1], config['model_id'])
            self.assertEqual(args[args.index('--port')+1], str(config['port']))
            self.assertEqual(args[args.index('--tp-size')+1], '8')
            self.assertEqual(args[args.index('--context-length')+1], '262144')
            self.assertEqual(args[args.index('--mem-fraction-static')+1], '0.8')
            for removed in ('--max-running-requests', '--mamba-scheduler-strategy', '--dtype'):
                self.assertNotIn(removed, args)
            self.assertNotIn('--disable-radix-cache', args)
            self.assertFalse(any('speculative' in arg for arg in args))
            self.assertIn('--enable-cache-report', args)
            good = dict(served_model_name=config['model_id'], tp_size=8, context_length=262144,
                        disable_radix_cache=False, speculative_algorithm=None, enable_cache_report=True)
            checked_info(good, config)
            for key, value in [('disable_radix_cache', True), ('speculative_algorithm', 'EAGLE'),
                               ('served_model_name', 'wrong'), ('tp_size', 1)]:
                with self.assertRaises(ValueError):
                    checked_info(dict(good, **{key:value}), config)
