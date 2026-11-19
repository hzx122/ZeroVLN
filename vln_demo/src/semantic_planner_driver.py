"""Visual tool-contract grounding + privileged collision-map local planner.

Experimental executor, not a semantic oracle: VLM grounding can be wrong.
Never reads an episode's reference route, destination, or success metrics.
"""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
import time
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import agent_tools as A
from action_head.render_client import RemoteRenderClient, RemotePrefixExecutor, _world_pose
from action_head.runtime import OptionContext, OptionRunResult
from rs3d.gs_client import CameraSpec
from sage.gs_client import XYZEulerClient
from sage.datasets import SageSet
from sage.occmap import OccMap
from sage.planner import (plan_path, smooth_path, _path_is_safe,
                          _catmull_rom_chain, _resample_polyline, PathSmoothingConfig)
from vln_gym.fm_action import world_to_chunk_pose


GROUNDING_PROMPT = """You execute a navigation controller's LOCAL tool call, not the overall task.
Use ONLY the supplied current first-person image and exact target/where/plan.
Return JSON: {"status":"ok|unsupported|not_visible|ambiguous", "reason":"short explanation",
"point_ids":[1,...]}. The first image is clean; the second marks candidate floor
points with numbered circles. Choose ONLY existing candidate IDs. These points
have metric-depth floor support and collision-map clearance; this does not mean
every connecting segment is clear. Choose the correct semantic side yourself.
For ok, give 1 to 4 ORDERED points on visible WALKABLE FLOOR tracing a short safe
path that obeys the requested direction and landmark relation. Select points on
the floor, never on an object, wall or doorway lintel. For approaching an object,
end on visible floor before it. For passing left/right, include the corresponding
side-of-object floor waypoint before any point beyond it; never cut across it.
For passing a doorway, trace floor through the visible opening. Only provide a
visible local prefix when the instruction extends beyond view; do not invent an
unseen route. For searching, follow the requested visible direction; do not guess
the hidden task destination. If no meaningful compliant visible prefix exists,
return not_visible. If a relational constraint cannot be grounded, return ambiguous.
Prefer ONE nearest useful point toward the requested target, up to 1.6m of local
progress. Use multiple ordered points only when needed to preserve an explicit
left/right/pass-through relation. Do not add a final point on the target object.
Return unsupported for commands that cannot be represented by a visible floor path.
An empty path is never ok. Do not claim task completion. No markdown or extra text."""


class GroundingRejected(ValueError):
    pass


def floor_point(point, depth, camera, pose):
    p = np.asarray(point, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1000).any():
        raise GroundingRejected("invalid image coordinate")
    u, v = p * np.array([camera.width - 1, camera.height - 1]) / 1000
    z = float(depth[int(round(v)), int(round(u))])
    if not math.isfinite(z) or not .1 <= z <= 8:
        raise GroundingRejected("invalid or distant metric depth")
    lateral = (u - camera.width / 2) / camera.fx * z
    height = 1.2 - (v - camera.height / 2) / camera.fy * z
    if abs(height) > .35:
        raise GroundingRejected("selected pixel is not supported as floor by metric depth")
    x, y, yaw_deg = pose
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [x + c*z + s*lateral, y + s*z - c*lateral]


def path_prefix(path, pose, max_steps=4, step_m=.1):
    """Fixed chunk-frame deltas; retain all polyline corners, cap at 0.4m."""
    current = np.asarray(pose[:2], float)
    samples = []
    for endpoint in np.asarray(path, float)[1:]:
        while np.linalg.norm(endpoint-current) > 1e-7 and len(samples) < max_steps:
            delta = endpoint-current
            distance = float(np.linalg.norm(delta))
            current = current + delta * min(1., step_m/distance)
            yaw = math.atan2(delta[1], delta[0])
            samples.append(world_to_chunk_pose(current, yaw, pose[:2], math.radians(pose[2])))
        if len(samples) >= max_steps:
            break
    if not samples:
        return np.zeros((1, 3))
    absolute = np.asarray(samples)
    absolute[:, 2] = np.unwrap(np.r_[0., absolute[:, 2]])[1:]
    return np.diff(np.vstack([np.zeros(3), absolute]), axis=0)


def smooth_ordered_path(occ, path, anchor_indices, radius=.07):
    """Golden spline with fixed ordered semantic anchors and safe fallback.

    Unlike the golden moving average, this interpolation preserves every
    anchor exactly. Blend only inside a 10cm tube around the original route.
    """
    raw = np.asarray(path, dtype=float)
    # Drop duplicate vertices, remapping anchors before spline evaluation.
    keep = np.r_[True, np.linalg.norm(np.diff(raw, axis=0), axis=1) > 1e-8]
    mapping = np.cumsum(keep)-1
    anchors = sorted(set([0, *[int(mapping[i]) for i in anchor_indices], int(mapping[-1])]))
    raw = raw[keep]
    if len(raw) < 2 or not _path_is_safe(occ, raw, radius):
        raise GroundingRejected('ordered path is empty or unsafe')
    cfg = PathSmoothingConfig()
    density = cfg.samples_per_segment
    curve = _catmull_rom_chain(raw, cfg.catmull_alpha, density)
    baseline = np.vstack([raw[0], *[
        a+(b-a)*t/density for a,b in zip(raw,raw[1:]) for t in range(1,density+1)]])
    minimum = min(occ.clearance(*p) for p in baseline)
    raw_length = np.linalg.norm(np.diff(raw,axis=0),axis=1).sum()
    for strength in cfg.blend_strengths:
        candidate = baseline + strength*(curve-baseline)
        if np.linalg.norm(candidate-baseline,axis=1).max() > .1:
            continue
        segments = [_resample_polyline(candidate[a*density:b*density+1],
                    step_m=.1, preserve_vertices=(strength == 0))
                    for a,b in zip(anchors,anchors[1:])]
        sampled = np.vstack([segments[0], *[s[1:] for s in segments[1:]]])
        if (np.linalg.norm(np.diff(sampled,axis=0),axis=1).sum() > raw_length*cfg.max_length_ratio+1e-6
                or not _path_is_safe(occ,sampled,radius)
                or min(occ.clearance(*p) for p in sampled) < max(radius,minimum-cfg.max_clearance_loss_m)-1e-6):
            continue
        return sampled, dict(blend_strength=float(strength), fixed_anchors=raw[anchors].tolist(),
                             input_points=len(raw), output_points=len(sampled))
    fallback = _resample_polyline(raw,step_m=.1,preserve_vertices=True)
    if not _path_is_safe(occ,fallback,radius):
        raise GroundingRejected('safe path could not be resampled for execution')
    return fallback, dict(blend_strength=0., fixed_anchors=raw[anchors].tolist(), fallback='original_safe_path')


def continuous_headings(path):
    """Arc-length central tangents, computed once for the entire option."""
    points = np.asarray(path, float)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    before = np.maximum(0., arc-.15)
    after = np.minimum(arc[-1], arc+.15)
    tangent = np.column_stack([np.interp(after,arc,points[:,i])-np.interp(before,arc,points[:,i]) for i in range(2)])
    return np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]))


def tracked_prefix(path, headings, cursor, pose, max_steps=4):
    """Consume a persistent global path cursor; never snap back across chunks."""
    yaw0 = math.radians(pose[2])
    absolute = []
    # A large initial heading error is a bounded in-place alignment, rather
    # than a one-frame turn/move/turn jump. Alignment consumes the same budget.
    if cursor == 1:
        bearing = math.atan2(path[cursor][1]-pose[1],path[cursor][0]-pose[0])
        error = (bearing-yaw0+math.pi)%(2*math.pi)-math.pi
        while abs(error) > math.radians(20) and len(absolute) < max_steps:
            amount = math.copysign(min(abs(error),math.radians(20)),error)
            relative = (absolute[-1][2] if absolute else 0.)+amount
            absolute.append((0.,0.,relative))
            error -= amount
    while len(absolute) < max_steps and cursor < len(path):
        absolute.append(world_to_chunk_pose(path[cursor],headings[cursor],pose[:2],yaw0))
        cursor += 1
    if not absolute:
        return np.zeros((0,3)), cursor
    absolute = np.asarray(absolute)
    absolute[:,2] = np.unwrap(np.r_[0.,absolute[:,2]])[1:]
    return np.diff(np.vstack([np.zeros(3),absolute]),axis=0), cursor


class SemanticPlannerRuntime:
    def __init__(self, *, gym_url, vlm_url, vlm_model, artifact_dir,
                 render_url="http://127.0.0.1:7038", interrupted=None, progress_callback=None):
        self.render_client = RemoteRenderClient(gym_url)
        # Preserve the v4.3 step7000 slow-model request, including query capture.
        # The expert does not consume these hidden states.
        self.query_count = 16
        self.prefix_executor = RemotePrefixExecutor(self.render_client)
        self.vlm_url, self.vlm_model = vlm_url.rstrip('/'), vlm_model
        self.camera = CameraSpec(width=640, height=480, hfov_deg=115)
        self.renderer = XYZEulerClient(render_url, camera=self.camera)
        self.dataset = SageSet('official_gvln_action_head_p0_v3test100')
        self.maps = {}
        self.interrupted = interrupted or (lambda: False)
        self.progress_callback = progress_callback
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.tool_executor = self

    def reset_environment(self, **kwargs):
        return self.render_client.reset(**kwargs)

    def render_map(self):
        return self.render_client.render_map()

    def ui_metrics(self):
        return self.render_client.ui_metrics()

    def execute_turn(self, **kwargs):
        return self.render_client.execute_turn(**kwargs)

    def ground(self, rgb, contract, candidates, image_prefix):
        clean = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        marked = clean.copy()
        draw = ImageDraw.Draw(marked)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
        for candidate in candidates:
            u,v = np.asarray(candidate['point'])*np.array([639,479])/1000
            draw.ellipse((u-10,v-10,u+10,v+10), fill='yellow', outline='black')
            draw.text((u,v), str(candidate['id']), fill='black', font=font, anchor='mm')
        clean.save(str(image_prefix)+'_rgb.jpg')
        marked.save(str(image_prefix)+'_candidates.jpg')
        images = []
        for image in (clean, marked):
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            images.append({'type':'image_url', 'image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode()}})
        body = dict(model=self.vlm_model, temperature=0, max_tokens=512,
                    chat_template_kwargs={"enable_thinking": False}, messages=[
            {"role": "system", "content": GROUNDING_PROMPT},
            {"role": "user", "content": images+[
                {"type": "text", "text": json.dumps({'tool_call':contract,'valid_ids':[c['id'] for c in candidates]}, ensure_ascii=False)}]}])
        request = urllib.request.Request(self.vlm_url+'/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=90) as response:
            text = json.load(response)['choices'][0]['message']['content']
        try:
            text = text.strip()
            if text.startswith('```'):
                text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
            result = json.loads(text)
        except (ValueError, AttributeError) as exc:
            raise GroundingRejected('grounder returned invalid JSON') from exc
        if not isinstance(result, dict):
            raise GroundingRejected('grounder returned a non-object')
        return result

    def execute(self, *, option_id, instruction, arguments, current_observation,
                slow_request_snapshot=None, slow_response_snapshot=None,
                action_head_as_tool_version='v4.3'):
        if action_head_as_tool_version != 'v4.3':
            raise ValueError('semantic planner currently supports v4.3 only')
        option = OptionContext(option_id, instruction, arguments, None,
                               current_observation, current_observation)
        audit = dict(time=time.time(), scene=current_observation.get('scene'),
                     episode_id=current_observation.get('episode_id'), option_id=option_id,
                     contract=arguments, driver='semantic_planner_v2_smooth')
        reason, error, ok = 'fast_step_budget_reached', '', True
        try:
            contract, error = A.validate_action_head_arguments(arguments)
            if contract is None:
                raise GroundingRejected(error)
            pose = _world_pose(self.render_client.last_info)
            if pose is None:
                raise RuntimeError('remote environment omitted world pose')
            scene = current_observation['scene']
            rgb, depth = self.renderer.render_one(scene, (*pose[:2], 1.2),
                math.radians(pose[2]), want_depth=True)
            rendered = np.asarray(rgb, dtype=float)
            observed = np.asarray(current_observation['rgb'], dtype=float)
            if rendered.shape != observed.shape:
                raise RuntimeError('expert camera dimensions differ from brain observation')
            audit['render_rgb_mae'] = float(np.abs(rendered-observed).mean())
            if audit['render_rgb_mae'] > 15:
                raise RuntimeError('expert rerender differs from current brain observation')
            if scene not in self.maps:
                self.maps.clear()
                self.maps[scene] = OccMap(self.dataset.scene_dir(scene),
                    collision_mode='usd_rooms', furniture_mode='none')
            occ = self.maps[scene]
            if depth is None:
                raise RuntimeError('renderer did not return metric depth')
            candidates = []
            for v in (570, 660, 750, 840, 930):
                for u in (80, 200, 320, 440, 560, 680, 800, 920):
                    try:
                        xy = floor_point([u,v], depth, self.camera, pose)
                    except GroundingRejected:
                        continue
                    if .25 <= math.dist(pose[:2],xy) <= 4. and occ.is_free(*xy,.07):
                        candidates.append(dict(id=len(candidates)+1, point=[u,v], world=xy))
            audit['candidates'] = candidates
            if not candidates:
                raise GroundingRejected('no depth-supported traversable floor candidates in view')
            images = self.artifact_dir/'grounding_images'
            images.mkdir(exist_ok=True)
            image_prefix = images/str(time.time_ns())
            audit['image_prefix'] = str(image_prefix)
            grounding = self.ground(rgb, contract, candidates, image_prefix)
            audit['grounding'] = grounding
            if grounding.get('status') != 'ok':
                raise GroundingRejected(str(grounding.get('status'))+': '+str(grounding.get('reason')))
            ids = grounding.get('point_ids')
            if not isinstance(ids, list) or not 1 <= len(ids) <= 4:
                raise GroundingRejected('expected 1..4 ordered floor points')
            if any(type(i) is not int or not 1 <= i <= len(candidates) for i in ids):
                raise GroundingRejected('grounder chose an unknown candidate ID')
            goals = [candidates[i-1]['world'] for i in ids]
            audit['world_waypoints'] = goals
            path = [pose[:2]]
            anchors = []
            for goal in goals:
                if not occ.is_free(*goal, .07):
                    raise GroundingRejected('grounded floor point is not traversable')
                raw = plan_path(occ, path[-1], goal, .07, mode='ecm')
                if not raw:
                    raise GroundingRejected('no collision-free path to grounded point')
                if not _path_is_safe(occ, raw, .07):
                    raise GroundingRejected('planned path fails collision verification')
                segment, _ = smooth_path(occ, raw, .07)
                length = sum(math.dist(a,b) for a,b in zip(segment, segment[1:]))
                if not _path_is_safe(occ, segment, .07) or length > 1.4*math.dist(path[-1], goal)+.3:
                    raise GroundingRejected('local path requires an ungrounded detour')
                path.extend(segment[1:])
                anchors.append(len(path)-1)
            path, audit['joint_smoothing'] = smooth_ordered_path(occ,path,anchors)
            path = path.tolist()
            headings = continuous_headings(path)
            audit['planned_headings'] = headings.tolist()
            audit['planned_path'] = path
            # Keep ordered waypoints across chunks; never replan a shortcut past them.
            cursor = 1
            for _ in range(4):
                if self.interrupted():
                    reason, ok = 'interrupted', False
                    break
                pose = _world_pose(self.render_client.last_info)
                if cursor == len(path):
                    reason = 'local_path_exhausted'
                    break
                if not occ.seg_free(*pose[:2], *path[cursor], .07):
                    reason = 'tracking_blocked'
                    break
                prefix, next_cursor = tracked_prefix(path, headings, cursor, pose)
                executed = self.prefix_executor.execute_prefix(prefix, option)
                option.chunks += 1
                option.moved_m += executed.moved_m
                option.sim_steps += executed.sim_steps
                option.current_observation = executed.observation
                option.blocked_count += int(executed.blocked)
                option.chunk_feedback.append(dict(chunk=option.chunks,
                    actual_end_pose=executed.actual_end_pose, path_m=executed.moved_m,
                    blocked=executed.blocked, collision=executed.collision,
                    no_progress=executed.moved_m < .001, observation=executed.observation))
                audit.setdefault('executed', []).append(dict(prefix=prefix.tolist(),
                    moved_m=executed.moved_m, pose=_world_pose(self.render_client.last_info)))
                if self.progress_callback:
                    self.progress_callback(option, executed)
                # Advance monotonically only after the requested endpoint was reached.
                endpoint = _world_pose(self.render_client.last_info)[:2]
                if executed.blocked or executed.collision or executed.episode_ended:
                    reason = 'episode_ended' if executed.episode_ended else 'blocked'
                    break
                if next_cursor > cursor and math.dist(endpoint,path[next_cursor-1]) > .03:
                    reason = 'tracking_deviation'
                    break
                cursor = next_cursor
        except GroundingRejected as exc:
            reason, error, ok = 'planner_rejected', str(exc), False
        except Exception as exc:
            reason, error, ok = 'error', str(exc), False
        result = OptionRunResult(ok, reason, option.current_observation,
            option.moved_m, option.chunks, option.sim_steps, option.blocked_count > 0,
            error=error, chunk_feedback=option.chunk_feedback).as_tool_result_v43(option_id, arguments)
        audit.update(result=result.raw)
        with (self.artifact_dir/'planner_options.jsonl').open('a') as file:
            file.write(json.dumps(audit, ensure_ascii=False)+'\n')
        return result
