"""Per-session camera checks for RL matching the three-source open-loop data.

These classes only validate observations. Camera construction remains owned by
the frozen SAGE settings and native Habitat YAML. A process may keep SAGE and
native sessions alive together, so the expected profile is instance state.
"""
from __future__ import annotations

import os

import numpy as np

from .env import VLNEnv
from .observation_profile import actual_profile, assert_profile, normalize_profile
from .vlnce_action_head import VLNCEActionHeadEnv


OPENLOOP_PROFILES = {
    "sage": {"width": 640, "height": 480, "hfov_deg": 115,
             "height_m": 1.2, "pitch_deg": 0},
    "r2r": {"width": 640, "height": 480, "hfov_deg": 120,
            "height_m": 1.25, "pitch_deg": 0},
    "rxr": {"width": 640, "height": 480, "hfov_deg": 120,
            "height_m": 1.25, "pitch_deg": 0},
}


def _reject_global_profile():
    if os.environ.get("VLN_OBSERVATION_PROFILE"):
        raise ValueError("three-source RL uses per-session camera profiles; "
                         "unset VLN_OBSERVATION_PROFILE before starting the Gym")


class _ProfiledTrainingEnv:
    def __init__(self, *args, backend, **kwargs):
        _reject_global_profile()
        # VLNEnv.__init__ may call reset(), which dispatches to our _metrics.
        self._training_observation_profile = normalize_profile(OPENLOOP_PROFILES[backend])
        super().__init__(*args, backend=backend, **kwargs)
        self._check_training_observation(self._last_obs,
            allow_missing=backend == "sage" and self._initial_reset_deferred)

    def _check_training_observation(self, obs, *, allow_missing=False):
        _reject_global_profile()
        expected = self._training_observation_profile
        actual = assert_profile(actual_profile(self.env), expected)
        if obs is None and allow_missing:
            return actual
        if not isinstance(obs, dict):
            raise ValueError("training camera requires an actual RGB/depth observation")
        for name in ("rgb", "depth"):
            if name not in obs:
                raise ValueError("training camera observation is missing " + name)
            shape = np.shape(obs[name])
            if len(shape) < 2 or tuple(shape[:2]) != (expected["height"], expected["width"]):
                raise ValueError(f"training {name} image dimensions differ from camera profile: "
                                 f"{shape} != {(expected['height'], expected['width'])}")
        return actual

    def _metrics(self, obs, extra=None):
        actual = self._check_training_observation(obs)
        info = super()._metrics(obs, extra=extra)
        info["observation_profile"] = actual
        return info


class ProfiledSageTrainingEnv(_ProfiledTrainingEnv, VLNEnv):
    def __init__(self, *args, backend="sage", **kwargs):
        if backend != "sage":
            raise ValueError("ProfiledSageTrainingEnv requires backend=sage")
        super().__init__(*args, backend=backend, **kwargs)


class ProfiledNativeTrainingEnv(_ProfiledTrainingEnv, VLNCEActionHeadEnv):
    def __init__(self, *args, backend="r2r", **kwargs):
        if backend not in ("r2r", "rxr"):
            raise ValueError("ProfiledNativeTrainingEnv requires backend=r2r or rxr")
        super().__init__(*args, backend=backend, **kwargs)
