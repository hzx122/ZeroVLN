"""Versionable camera settings shared by training and suite evaluation.

The profile is evaluator telemetry, never part of the navigation prompt.
Native Habitat sensors are configured in a private YAML; SAGE reads the same
profile before constructing its renderer camera. Actual settings are checked
on every reset/step so an old service cannot silently ignore a new profile.
"""
from __future__ import annotations

import json
import math
import os

FIELDS = ('width', 'height', 'hfov_deg', 'height_m', 'pitch_deg')


def normalize_profile(value):
    if set(value) != set(FIELDS):
        raise ValueError(f'camera profile requires exactly {FIELDS}')
    result = {}
    for key in FIELDS:
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError('nonfinite or nonnumeric camera setting: ' + key)
        if key in ('width', 'height'):
            if int(number) != number or not 16 <= number <= 8192:
                raise ValueError('invalid camera resolution: ' + key)
            result[key] = int(number)
        else:
            result[key] = float(number)
    if not 0 < result['hfov_deg'] < 180 or not 0 < result['height_m'] < 10 or not -89 < result['pitch_deg'] < 89:
        raise ValueError('camera FOV, height or pitch out of range')
    return result


def requested_profile():
    value = os.environ.get('VLN_OBSERVATION_PROFILE')
    return normalize_profile(json.loads(value)) if value else None


def assert_profile(actual, expected):
    actual, expected = normalize_profile(actual), normalize_profile(expected)
    if any(not math.isclose(actual[k], expected[k], abs_tol=1e-5, rel_tol=0) for k in FIELDS):
        raise ValueError(f'actual camera differs from training profile: {actual} != {expected}')
    return actual


def actual_profile(env):
    sim = env.sim
    if hasattr(sim, 'z_offset') and hasattr(sim, 'client'):
        camera = sim.client.camera
        return normalize_profile(dict(width=camera.width, height=camera.height,
            hfov_deg=camera.hfov_deg, height_m=sim.z_offset,
            pitch_deg=math.degrees(getattr(sim, 'camera_pitch', 0.))))
    specs = getattr(sim, '_remote_specs', None)
    if specs is None:
        specs = [dict(uuid=s.uuid, resolution=list(s.resolution), hfov=float(s.hfov),
            position=list(s.position), orientation=list(s.orientation))
            for s in sim.sim_config.agents[0].sensor_specifications]
    cameras = [s for s in specs if s['uuid'] in ('rgb', 'depth')]
    if len(cameras) != 2:
        raise ValueError('RGB and metric depth sensors are required')
    profiles = [dict(width=s['resolution'][1], height=s['resolution'][0],
        hfov_deg=s['hfov'], height_m=s['position'][1],
        pitch_deg=math.degrees(s['orientation'][0])) for s in cameras]
    return assert_profile(profiles[0], profiles[1])


def validate_environment(env):
    expected = requested_profile()
    if expected is None:
        return None
    return assert_profile(actual_profile(env), expected)
