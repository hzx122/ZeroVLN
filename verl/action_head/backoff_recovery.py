"""Bounded retreat availability for the v4.9 recovery protocol.

Only actual tool feedback controls availability. A model may still choose stop
or another action. This helper never moves the robot or changes call budgets.
"""
import math
from .protocol_v4_5 import successful_runtime_results

MIN_BACKOFF_M = .05
MAX_BACKOFF_M = .3
STALL_NET_M = .15
RESET_PROGRESS_M = .3


def backoff_distance(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not MIN_BACKOFF_M <= value <= MAX_BACKOFF_M:
        raise ValueError('backoff distance must be finite and in [0.05, 0.3] meters')
    return float(value)


def backoff_state(messages):
    stalled = 0
    used = False
    for raw in successful_runtime_results(messages):
        if raw.get('tool') == 'move_backward':
            used = True
        elif raw.get('tool') == 'run_action_head':
            net = raw.get('net_m')
            if type(net) not in (int, float) or not math.isfinite(net) or net < 0:
                stalled = 0
                continue
            if net >= RESET_PROGRESS_M:
                used = False
            stalled = stalled + 1 if raw.get('blocked') is True and net < STALL_NET_M else 0
    return dict(available=stalled >= 2 and not used,
                consecutive_stalled_options=stalled, used_since_progress=used)


def validate_backoff(messages, arguments):
    """Return (distance, rejection); rejected attempts must cost a call, no motion."""
    state = backoff_state(messages)
    reason = None
    distance = arguments.get('distance')
    if not state['available']:
        reason = 'backoff_not_available'
    else:
        try:
            distance = backoff_distance(distance)
        except ValueError:
            reason = 'invalid_backoff_distance'
    if reason is None:
        return float(distance), None
    return None, dict(schema='backoff_rejection_v1', tool='move_backward', ok=False,
                      return_reason=reason, steps_used=0, robot_moved=False,
                      available=state['available'],
                      allowed_distance_m=[MIN_BACKOFF_M, MAX_BACKOFF_M])
