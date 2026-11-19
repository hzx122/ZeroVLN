"""Select a fixed v4.3 prompt revision for the lifetime of this process.

The default is the original v4.3 used by bootstrap and open-loop training.
Prompt experiments explicitly select r2/r3/react before Python starts.
Child processes inherit the same selection. Both direct
imports and protocol routing therefore use the same strict identity checks.
"""
import os as _os

_revision = _os.environ.get("ACTION_HEAD_V43_PROMPT_REVISION", "legacy")
if _revision == "react":
    from .protocol_v4_3_react import *  # noqa: F401,F403
elif _revision == "r2":
    from .protocol_v4_3_r2 import *  # noqa: F401,F403
elif _revision == "r3":
    from .protocol_v4_3_r3 import *  # noqa: F401,F403
elif _revision == "legacy":
    from .protocol_v4_3_legacy import *  # noqa: F401,F403
else:
    raise ValueError("ACTION_HEAD_V43_PROMPT_REVISION must be react, r2, r3 or legacy")
