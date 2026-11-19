"""Measured geometry only; semantic admission still requires an explicit review.

One novel turn/exploration and its first return are useful. Repeating a previously
travelled route adds no progress. First local holds and novel orientation traces are retained for semantic review;
repeated fragments are excluded under the user-approved export policy.
No route-efficiency, exact object-instance or final episode-success filter lives
here. Planned poses and the controller's accepted flag are never evidence.
"""
from __future__ import annotations

import hashlib
import json
import math

PROGRESS_SCHEMA = 'rule_teacher_actual_progress_v3'
MIN_TRANSLATION_M = 0.1


def _wrap(angle):
    return (angle + 180.) % 360. - 180.


def _xy(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


def measured_poses(execution):
    """Prefer full measured waypoints, otherwise actual committed endpoints."""
    state = execution.get('state') or {}
    poses = state.get('execution_world_poses')
    source = 'state.execution_world_poses'
    if poses is None:
        initial = state.get('initial_pose')
        endpoints = state.get('actual_poses')
        if endpoints is None:
            windows = state.get('training_windows') or []
            endpoints = [w.get('actual_end_pose') for w in windows if w.get('executed') is True]
            source = 'state.training_windows.executed.actual_end_pose'
        else:
            source = 'state.actual_poses'
        poses = [initial, *endpoints] if initial is not None else []
    if (not isinstance(poses, (list, tuple)) or len(poses) < 2 or
            any(not isinstance(p, (list, tuple)) or len(p) != 3 or
                any(not isinstance(v, (int, float)) or isinstance(v, bool) or
                    not math.isfinite(v) for v in p) for p in poses)):
        raise ValueError('missing_or_invalid_measured_poses')
    poses = [[float(v) for v in p] for p in poses]
    initial = state.get('initial_pose')
    if initial is not None and (_xy(poses[0], initial) > 1e-5 or
                               abs(_wrap(poses[0][2] - initial[2])) > 1e-3):
        raise ValueError('measured_start_pose_mismatch')
    return poses, source


def _segment_distance(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx*dx + dy*dy
    ratio = min(1., max(0., ((point[0]-a[0])*dx + (point[1]-a[1])*dy)/length2)) if length2 else 0.
    return math.hypot(point[0]-a[0]-ratio*dx, point[1]-a[1]-ratio*dy)


def _sample_xy(poses, limit=65):
    """Arc-length sampling makes overlap independent of waypoint spacing."""
    lengths = [_xy(a, b) for a, b in zip(poses, poses[1:])]
    total = sum(lengths)
    if total == 0:
        return [poses[0][:2]]
    count = min(limit, max(3, math.ceil(total/.1) + 1))
    samples, index, offset = [], 0, 0.
    for sample in range(count):
        position = total*sample/(count-1)
        while index < len(lengths)-1 and offset + lengths[index] < position:
            offset += lengths[index]
            index += 1
        ratio = (position-offset)/lengths[index] if lengths[index] else 0.
        samples.append([poses[index][axis] + ratio*(poses[index+1][axis]-poses[index][axis])
                        for axis in (0, 1)])
    return samples


def _route_overlap(current, earlier, radius):
    samples = _sample_xy(current)
    return sum(min(_segment_distance(p, a, b) for a, b in zip(earlier, earlier[1:])) <= radius
               for p in samples)/len(samples)


def execution_progress(execution, prior_executions=(), *, min_translation_m=MIN_TRANSLATION_M,
                       min_turn_deg=10., revisit_radius_m=.1, heading_tolerance_deg=10.):
    """Classify actual movement, keeping novel turns separate from spatial anchors.

    Supply chronological earlier executions from the *same real case*, including
    unreviewed calls. Prior calls provide geometry history, never semantic approval.
    The first return/exploration is retained; a later same-endpoint route with at
    least 90% overlap is marked repetition. Missing evidence is not zero motion.
    """
    thresholds = dict(min_translation_m=min_translation_m, min_turn_deg=min_turn_deg,
                      revisit_radius_m=revisit_radius_m, heading_tolerance_deg=heading_tolerance_deg,
                      repeated_route_overlap=.9)
    if any(not math.isfinite(v) or v <= 0 for v in thresholds.values()):
        raise ValueError('progress thresholds must be finite and positive')
    result = dict(schema=PROGRESS_SCHEMA, thresholds=thresholds,
                  spatial_progress=False, admissible_geometry=False, repeated_no_progress=False,
                  semantic_review_performed=False)
    try:
        poses, source = measured_poses(execution)
    except (ValueError, TypeError, IndexError) as exc:
        return dict(result, kind='missing_actual_evidence', reason=str(exc))
    distance = sum(_xy(a, b) for a, b in zip(poses, poses[1:]))
    excursion = max(_xy(poses[0], p) for p in poses)
    yaw = _wrap(poses[-1][2] - poses[0][2])
    result.update(evidence_source=source, measured_pose_count=len(poses),
        initial_pose=poses[0], final_pose=poses[-1], actual_path_m=distance,
        net_translation_m=_xy(poses[0], poses[-1]), max_excursion_m=excursion,
        net_yaw_deg=yaw, absolute_yaw_deg=sum(abs(_wrap(b[2]-a[2])) for a, b in zip(poses, poses[1:])),
        measured_pose_sha256=hashlib.sha256(json.dumps(poses, separators=(',', ':'),
                                                     allow_nan=False).encode()).hexdigest())
    history = []
    for previous in prior_executions:
        try:
            old, _ = measured_poses(previous)
        except (ValueError, TypeError, IndexError):
            continue
        history.append(old)
    if excursion >= min_translation_m:
        for index, old in enumerate(history):
            if (_xy(poses[0], old[0]) <= revisit_radius_m and
                    _xy(poses[-1], old[-1]) <= revisit_radius_m and
                    max(_xy(old[0], p) for p in old) >= min_translation_m):
                overlap = _route_overlap(poses, old, revisit_radius_m)
                if overlap >= .9:
                    return dict(result, kind='repeated_no_progress', repeated_no_progress=True,
                                reason='repeated_same_endpoint_route', matched_history_index=index,
                                route_overlap=overlap)
        return dict(result, kind='spatial_progress', reason='measured_spatial_exploration',
                    spatial_progress=True, admissible_geometry=True)
    # Do not reject the first return after looking aside or the first within-call
    # out-and-back observation. Only a repeated similar orientation trace counts.
    for index, old in enumerate(history):
        old_excursion = max(_xy(old[0], p) for p in old)
        if old_excursion >= min_translation_m:
            continue
        # For tiny translations use a finer matching radius, so a new 2 cm step
        # is not confused with another nearby 2 cm step.
        radius = min(revisit_radius_m, max(excursion, old_excursion)*.25 + 1e-5)
        if (_xy(poses[0], old[0]) > radius or _xy(poses[-1], old[-1]) > radius or
                abs(_wrap(poses[0][2]-old[0][2])) > heading_tolerance_deg or
                abs(_wrap(poses[-1][2]-old[-1][2])) > heading_tolerance_deg):
            continue
        old_yaw = sum(abs(_wrap(b[2]-a[2])) for a, b in zip(old, old[1:]))
        similar_length = abs(result['absolute_yaw_deg']-old_yaw) <= max(heading_tolerance_deg, old_yaw*.2)
        coverage = sum(any(abs(_wrap(p[2]-q[2])) <= heading_tolerance_deg for q in old)
                       for p in poses)/len(poses)
        if similar_length and coverage >= .9:
            return dict(result, kind='repeated_no_progress', repeated_no_progress=True,
                        reason='repeated_local_motion', admissible_geometry=False,
                        matched_history_index=index,
                        orientation_overlap=coverage)
    if result['absolute_yaw_deg'] >= min_turn_deg:
        return dict(result, kind='useful_turn', reason='first_measured_orientation_trace', admissible_geometry=True)
    if distance > 1e-6:
        return dict(result, kind='minor_motion', reason='measured_motion_below_spatial_count_threshold',
                    admissible_geometry=True)
    return dict(result, kind='stationary_first', reason='first_hold_requires_semantic_review',
                admissible_geometry=True)


def distinct_spatial_anchor_count(poses, min_separation_m=MIN_TRANSLATION_M):
    """Count distinct start positions after filtering for spatial_progress.

    Deterministic ordering avoids review-file order changing the count. Yaw alone
    never creates another spatial anchor. Input is a list of [x, y, yaw] poses.
    """
    if not math.isfinite(min_separation_m) or min_separation_m <= 0:
        raise ValueError('minimum separation must be finite and positive')
    representatives = []
    for pose in sorted(poses):
        if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
            raise ValueError('invalid spatial anchor pose')
        if all(_xy(pose, previous) >= min_separation_m for previous in representatives):
            representatives.append(pose)
    return len(representatives)


class DepthProgressTracker:
    """Count finalized depths, using actual executions rather than candidate waves."""
    def __init__(self, limit=3):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('no-progress limit must be a positive integer')
        self.limit = limit
        self.history = []
        self.consecutive = 0
        self.last_depth = 0

    def complete_depth(self, depth, executions=()):
        if depth != self.last_depth + 1:
            raise ValueError('progress depths must be finalized exactly once in order')
        measured, history = [], list(self.history)
        for execution in executions:
            progress = execution_progress(execution, history)
            if progress['kind'] == 'missing_actual_evidence':
                raise ValueError('progress_evidence_error:' + progress['reason'])
            measured.append(progress)
            history.append(execution)
        # First holds stay exportable but add no actual progress. Tiny new moves
        # and new turns reset the streak; repeating either does not.
        new_progress = any(not item['repeated_no_progress'] and
                           (item['actual_path_m'] > 1e-6 or item['absolute_yaw_deg'] > 1e-6)
                           for item in measured)
        self.history = history
        self.last_depth = depth
        self.consecutive = 0 if new_progress else self.consecutive + 1
        return dict(schema='rule_teacher_depth_progress_v1', depth=depth,
                    execution_progress=measured, action_executed=bool(measured),
                    new_progress=new_progress, consecutive_no_progress=self.consecutive,
                    limit=self.limit, terminate=self.consecutive >= self.limit,
                    terminal_reason='no_new_progress' if self.consecutive >= self.limit else None)
