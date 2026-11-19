"""Share mall's simulator, camera, collision rules and metrics for Lepu assets."""
import os

from mall.env import MallEnv


class LepuEnv(MallEnv):
    def __init__(self, assets, render_url=None, **kwargs):
        url = render_url or os.environ.get(
            "LEPU_RENDER_URL", os.environ.get("MALL_RENDER_URL", "http://127.0.0.1:7039"))
        super().__init__(assets, render_url=url, **kwargs)
