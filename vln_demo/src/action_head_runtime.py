"""Compatibility import for the web demo; production ownership is in Verl."""
import sys
from pathlib import Path

_VERL_ROOT = str(Path(__file__).resolve().parents[2] / "verl")
if _VERL_ROOT not in sys.path:
    sys.path.insert(0, _VERL_ROOT)

from action_head.runtime import *  # noqa: F401,F403
