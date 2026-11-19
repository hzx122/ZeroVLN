"""Lepu asset semantics ported from lepu@f92cf9f, without its UI/recording changes."""
import os

from mall.assets import MallAssets

DEFAULT_ASSETS_DIR = (
    "/opt/external/user-9656ce61/grpo/locnav_grpo/exp0822/assets_arr5/lepu")
SCENE_ID = "jW00XG5o3g"


class LepuAssets(MallAssets):
    scene_id = SCENE_ID
    scene_label = "Lepu 办公层 / jW00XG5o3g"
    target_label = "房间与设施"
    floorplan_path = os.path.join(os.path.dirname(__file__), "floorplan.png")
    unload_env = "LEPU_UNLOAD_ON_CLOSE"
    navigation_planner = "mask_astar"

    def __init__(self, root=None):
        super().__init__(root or os.environ.get("LEPU_ASSETS_DIR", DEFAULT_ASSETS_DIR))

    def mask_npy(self):
        return os.path.join(self.root, "lepu_walk_union_g45.npy")

    def goal_shops(self):
        # Exits/elevators are annotated goals here, unlike the mall's shop pool.
        if self._goals is None:
            self._goals = [s for s in self.shops
                           if s.get("name") and not s.get("is_infra")]
        return self._goals
