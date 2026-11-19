"""Render at tool boundaries in batch mode, without skipping simulation steps."""
from functools import wraps
import os


def mode():
    value = os.environ.get('VLN_RENDER_MODE', 'display')
    if value not in ('display', 'batch'):
        raise ValueError('VLN_RENDER_MODE must be display or batch')
    return value


def terminal_frame(function):
    """Decorate (env, ...) -> (observation, info) motion/tool functions.

    Nested primitives inside a visual waypoint do not render. Only the outer
    action produces fresh RGB/depth before returning to the policy/agent.
    """
    @wraps(function)
    def wrapped(env, *args, **kwargs):
        if mode() != 'batch' or not getattr(env.sim, '_remote_render_installed', False):
            return function(env, *args, **kwargs)
        sim = env.sim
        depth = getattr(sim, '_remote_defer_depth', 0)
        sim._remote_defer_depth = depth + 1
        before = getattr(sim, '_remote_deferred_steps', 0)
        error = None
        result = None
        try:
            result = function(env, *args, **kwargs)
        except Exception as exc:
            error = exc
        finally:
            sim._remote_defer_depth = depth
        if depth == 0 and getattr(sim, '_remote_deferred_steps', 0) != before:
            observation, info = result if result is not None else (None, None)
            fresh = sim._sensor_suite.get_observations(sim.get_sensor_observations())
            observation = dict(observation or {})
            observation.pop('_render_deferred', None)
            observation.update(fresh)
            if error is not None:
                error.terminal_observation = observation
            else:
                return observation, info
        if error is not None:
            raise error
        return result
    return wrapped
