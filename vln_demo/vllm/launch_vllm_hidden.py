#!/usr/bin/env python3
"""Deprecated demo entry point; delegates to the Verl-owned launcher."""
import sys
from pathlib import Path

_VERL_ROOT = str(Path(__file__).resolve().parents[2] / "verl")
if _VERL_ROOT not in sys.path:
    sys.path.insert(0, _VERL_ROOT)

from action_head.serve.launch_vllm_hidden import run


if __name__ == "__main__":
    run()
