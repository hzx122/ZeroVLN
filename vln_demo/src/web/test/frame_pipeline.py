#!/usr/bin/env python3
"""Regression checks for clean, same-frame first-person image publication."""

import io
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image

import vln_model as model


class _Env:
    def get_metrics(self):
        return {"distance_to_goal": 7.5, "top_down_map": None}


def check_turn_replay_matches_native_observation_camera():
    """Replay must use the same camera frame as each backend's observe()."""
    from abotn.env import ABotNSim
    from mall.env import MallSim, SCENE_ID
    from rs3d.env import RS3DSim
    from sage.env import SageSim

    class Client:
        def render_one(self, scene, eye, yaw, want_depth=False):
            self.last = (scene, np.asarray(eye).copy(), yaw)
            return Image.new("RGB", (8, 8)), None

    for backend in (ABotNSim, SageSim, RS3DSim, MallSim):
        client = Client()
        if backend in (ABotNSim, SageSim):
            # Nonzero ground and distinct axes expose both swapped coordinates
            # and accidental double application of the camera height.
            occ = SimpleNamespace(ground_z=lambda x, y: 2.3)
            sim = backend("scene-a", occ, client, z_offset=0.65)
            sim._x, sim._y = 4.2, -7.1
        elif backend is RS3DSim:
            sim = backend("scene-a", object(), client, sensor_height=1.25)
            sim._pos = np.array([4.2, 2.3, -7.1])
        else:
            sim = backend(SimpleNamespace(scene_id=SCENE_ID), client)
            sim._x, sim._z = 4.2, -7.1
        sim._yaw = 0.4
        env = SimpleNamespace(sim=sim, scene=SCENE_ID)
        before = sim.get_agent_state().position.copy()
        for direction, sign in (("left", 1), ("right", -1)):
            spec = model._async_turn_render_spec(env, direction, 35)
            assert spec is not None, backend.__name__
            sim.observe()
            scene, eye, yaw = client.last
            assert spec["scene"] == scene, backend.__name__
            np.testing.assert_allclose(spec["eye"], eye, err_msg=backend.__name__)
            assert spec["start_yaw"] == yaw
            assert np.isclose(spec["delta"], sign * np.deg2rad(35))
            np.testing.assert_array_equal(sim.get_agent_state().position, before)


def check_golden_keeps_episode_drivable():
    """A completed golden run must not consume the simulator episode."""
    obs = {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}

    class Sim:
        xy = [0.0, 0.0]
        yaw = 0.0

    class Occ:
        @staticmethod
        def is_free(x, y, radius):
            return True

        @staticmethod
        def seg_free(x0, y0, x1, y1, radius):
            return True

    class Env:
        sim = Sim()
        current_episode = type("Episode", (), {"scene": "scene-a"})()
        episode_over = False
        robot_radius = 0.07

        def __init__(self):
            self.steps = []

        def _golden_pts(self, ep):
            return [(0.0, 0.0), (0.1, 0.0)]

        def _occ_for(self, scene):
            return Occ()

        def reset(self):
            self.episode_over = False
            self.sim.xy[:] = [0.0, 0.0]
            self.sim.yaw = 0.0
            return obs

        def step(self, action):
            self.steps.append(action)
            self.episode_over = True
            return obs

        def get_metrics(self):
            return {"distance_to_goal": 0.0, "success_radius": 0.5,
                    "shortest_path_m": 0.1, "path_length_m": 0.1}

    def drive_turn(env, direction, degrees, **kwargs):
        sign = 1 if direction == "left" else -1
        env.sim.yaw += np.deg2rad(degrees) * sign
        return obs, {}

    def drive_forward(env, meters, **kwargs):
        env.sim.xy[0] += float(meters) * np.cos(env.sim.yaw)
        env.sim.xy[1] += float(meters) * np.sin(env.sim.yaw)
        return obs, {"moved_m": float(meters), "blocked": False}

    model.A.drive_turn = drive_turn
    model.A.drive_forward = drive_forward
    model.refresh = lambda *args, **kwargs: None
    model.capture_model_request = lambda *args, **kwargs: None
    model.CFG["golden_step"] = 0.1
    env = Env()
    args = type("Args", (), {"max_steps": 10, "step_delay": 0.0})()
    model.run_golden(env, None, "", args, obs, [])
    assert not env.episode_over, "golden run ended the simulator episode"
    assert env.steps == [], "golden run sent STOP to the simulator"


def main():
    # The HTTP guard must reject ABotN before the worker touches its simulator.
    model.STATE["locked"] = False
    model.STATE["backend"] = "in"
    model.CMD["walk_golden"] = False
    reply = model.Handler._cmd(object.__new__(model.Handler),
                               {"do": ["walk_golden"]})
    assert reply == model.GOLDEN_ONLY_MSG
    assert model.CMD["walk_golden"] is False

    # A second in-worker capability guard covers scene-switch races and stale
    # pages without changing the current observation or action trail.
    model.ACTIVE["backend"] = "in"
    marker_obs, marker_trail = object(), [1, 2]
    got_obs, got_trail = model.run_golden(
        object(), None, "", None, marker_obs, marker_trail)
    assert got_obs is marker_obs and got_trail is marker_trail
    assert model.STATE["status"] == "idle"

    assert model.relative_pose_motion((0, 0), 0, (0, -1), 0) == {
        "dist": 1.0, "bearing": 0.0, "yaw": 0.0,
    }
    assert model.relative_pose_motion((0, 0), 0, (1, -1), 90) == {
        "dist": 1.414, "bearing": 45.0, "yaw": 90.0,
    }
    assert model.relative_pose_motion((0, 0), 0, (-1, -1), -90) == {
        "dist": 1.414, "bearing": -45.0, "yaw": -90.0,
    }
    assert model.relative_pose_motion((0, 0), 90, (1, 0), 90) == {
        "dist": 1.0, "bearing": 0.0, "yaw": 0.0,
    }
    assert model.relative_pose_motion((0, 0), 0, (0, 0), 90) == {
        "dist": 0.0, "bearing": 0.0, "yaw": 90.0,
    }

    obs = {"rgb": np.full((48, 64, 3), (12, 200, 34), dtype=np.uint8)}
    model.set_state(scene="scene-a", episode_id="ep-1")
    model.refresh(_Env(), obs, 7)

    canonical = model.observation_image(obs)
    assert canonical.size == (64, 48)
    assert model._cur_img(obs).tobytes() == canonical.tobytes()
    assert model.CURRENT_FRAME["image"].tobytes() == canonical.tobytes()
    # The old step banner made this pixel black.
    assert model.CURRENT_FRAME["image"].getpixel((2, 2)) == (12, 200, 34)
    with Image.open(io.BytesIO(model.STATE["frame_jpg"])) as web_frame:
        assert web_frame.size == canonical.size
    assert model.STATE["display_frame_jpg"] == model.STATE["frame_jpg"]

    # A rejected point wakes the browser stream without advancing the
    # canonical frame token, and clearing it restores the display-only base.
    clean_jpg = model.STATE["frame_jpg"]
    model.CURRENT_FRAME["depth"] = np.ones((48, 64), dtype=np.float32)
    model.ACTIVE["env"] = object()
    assert model.STATE["frame_jpg"] == clean_jpg

    canonical_seq = model.STATE["seq"]
    display_seq = model.STATE["display_seq"]
    model.show_invalid_point([500, 500], seconds=60.0)
    marker_version = model.TRANSIENT_SCREEN_MARKER["version"]
    assert model.STATE["seq"] == canonical_seq
    assert model.STATE["display_seq"] == display_seq + 1
    assert model.STATE["frame_jpg"] == clean_jpg
    assert model.STATE["display_frame_jpg"] != clean_jpg
    model._clear_invalid_point(marker_version)
    assert model.STATE["display_frame_jpg"] == clean_jpg

    # Turn interpolation is replayed only through display_seq; the final model
    # frame and canonical token are already available while this worker runs.
    rendered_yaws = []

    class ReplayClient:
        @staticmethod
        def render_one(_scene, _eye, yaw, want_depth=False):
            rendered_yaws.append(yaw)
            return Image.new("RGB", (64, 48), (80, 90, 100)), None

    replay_spec = {
        "sim": type("ReplaySim", (), {"_lock": threading.Lock()})(),
        "client": ReplayClient(), "scene": "scene-a",
        "eye": np.zeros(3), "start_yaw": 0.0,
        "delta": np.deg2rad(15.0),
    }
    canonical_seq = model.STATE["seq"]
    replay = model.replay_turn_async(replay_spec, 15.0)
    replay.join(timeout=2.0)
    assert not replay.is_alive()
    assert len(rendered_yaws) == 2
    assert model.STATE["seq"] == canonical_seq
    assert model.STATE["frame_jpg"] == clean_jpg
    assert model.STATE["display_frame_jpg"] == model.STATE["display_base_jpg"]

    # A visual-tool marker is world anchored and affects only the human MJPEG
    # stream. The canonical frame used by vln_observe/model/bbox stays exact.
    marker_env = type("MarkerEnv", (), {})()
    marker_state = type("State", (), {
        "position": np.array([0.0, 1.25, 0.0]),
        "rotation": type("Quat", (), {"w": 1.0, "x": 0.0,
                                      "y": 0.0, "z": 0.0})(),
        "sensor_states": {},
    })()
    marker_env.sim = type("Sim", (), {
        "get_agent_state": lambda self: marker_state,
        "camera_z": 1.25,
    })()
    marker_env._geom = {"hfov": 90.0}
    marker_env.get_metrics = lambda: {
        "distance_to_goal": 7.5, "top_down_map": None}
    marked_obs = {"rgb": np.full((48, 64, 3), (12, 200, 34), dtype=np.uint8),
                  "depth": np.full((48, 64), 10.0, dtype=np.float32)}
    model.set_active_visual_target({
        "kind": "point", "world_xyz": (0.0, 0.0, -2.0),
        "goal_radius_m": 0.2,
    })
    model.refresh(marker_env, marked_obs, 7)
    model.set_active_visual_target(None)
    assert model.STATE["display_frame_jpg"] != model.STATE["frame_jpg"]
    assert model.CURRENT_FRAME["image"].tobytes() == canonical.tobytes()

    snapshot = model.frame_snapshot_payload()
    assert (snapshot["width"], snapshot["height"]) == canonical.size
    assert snapshot["source"] == {
        "scene": "scene-a", "episode_id": "ep-1", "seq": model.STATE["seq"],
    }
    meta = model.set_react_reference({
        "kind": "bbox", "bbox": [100, 100, 800, 800],
        "source": snapshot["source"],
    })
    marked, _ = model.react_reference_snapshot()
    assert marked.size == canonical.size
    assert marked.getpixel((0, 0)) == canonical.getpixel((0, 0))
    assert meta["seq"] == snapshot["source"]["seq"]
    assert meta["bbox"] == [100, 100, 800, 800]

    bbox_content = model.react_first_user_content(
        obs, "go", reference_image=marked, reference_meta=meta)
    assert bbox_content[0]["type"] == "image"
    bbox_text = bbox_content[1]["text"]
    assert "上面这张图片" in bbox_text
    assert "已作为红色矩形框直接绘制在图片上" in bbox_text
    assert "[100, 100, 800, 800]" in bbox_text
    assert "唯一图片" not in bbox_text

    edge_meta = model.set_react_reference({
        "kind": "bbox", "bbox": [0, 0, 1000, 1000],
        "source": snapshot["source"],
    })
    assert edge_meta["bbox"] == [0, 0, 1000, 1000]
    full_frame, _ = model.react_reference_snapshot()
    assert full_frame.size == canonical.size

    # Point references require only a current pixel with same-frame depth.
    model.CURRENT_FRAME["depth"] = np.ones((48, 64), dtype=np.float32)
    model.ACTIVE["env"] = object()
    point_meta = model.set_react_reference({
        "kind": "point", "point": [530, 720],
        "source": snapshot["source"],
    })
    point_frame, _ = model.react_reference_snapshot()
    assert point_meta["point"] == [530, 720]
    assert point_meta["bbox"] is None
    point_content = model.react_first_user_content(
        obs, "go", reference_image=point_frame, reference_meta=point_meta)
    point_text = point_content[1]["text"]
    assert "物体、墙面、家具或地面" in point_text
    assert "经典 ReAct" in point_text
    assert "move_to_point" not in point_text
    assert "[530, 720]" in point_text

    direct_text = model.direct_point_start_text([530, 720])
    assert direct_text == (
        "pi-agent 已给出视觉点 [530, 720]，直接执行视觉点导航")
    assert "27B" not in direct_text
    assert "35B" not in direct_text

    # A missing depth sample still gets display-only invalid-point feedback.
    model.CURRENT_FRAME["depth"][8, 14] = np.nan
    try:
        try:
            model.set_react_reference({
                "kind": "point", "point": [220, 180],
                "source": snapshot["source"],
            })
        except ValueError as error:
            assert "没有有效 metric depth" in str(error)
        else:
            raise AssertionError("point without depth was accepted")
        assert model.TRANSIENT_SCREEN_MARKER["point"] == [220.0, 180.0]
        assert model.STATE["frame_jpg"] == clean_jpg
        model._clear_invalid_point(
            model.TRANSIENT_SCREEN_MARKER["version"])
    finally:
        model.CURRENT_FRAME["depth"][8, 14] = 1.0

    # Browser double-clicks use the direct-point queue, without the
    # pi-agent-only header and without invoking the lower-level ReAct loop.
    model.CMD["point_move"] = None
    reply = model.Handler._cmd(object.__new__(model.Handler), {
        "do": ["screen_point_move"], "x": ["310"], "y": ["760"],
    })
    assert reply == "ok"
    assert model.CMD["point_move"]["point"] == [310.0, 760.0]
    assert model.CMD["point_move"]["source"] == (
        "scene-a", "ep-1", snapshot["source"]["seq"])
    model.CMD["point_move"] = None

    try:
        model.set_react_reference({
            "kind": "bbox", "bbox": [100.5, 100, 800, 800],
            "source": snapshot["source"],
        })
    except ValueError as error:
        assert "必须是整数" in str(error)
    else:
        raise AssertionError("fractional bbox coordinate was accepted")

    model.refresh(_Env(), obs, 8)
    try:
        model.set_react_reference({
            "kind": "bbox", "bbox": [100, 100, 800, 800],
            "source": snapshot["source"],
        })
    except ValueError as error:
        assert "已经变化" in str(error)
    else:
        raise AssertionError("stale bbox source was accepted")

    content = model.react_first_user_content(obs, "go")
    sent_image = next(part["image"] for part in content
                      if part.get("type") == "image")
    assert sent_image.size == canonical.size
    assert sent_image.tobytes() == canonical.tobytes()
    check_golden_keeps_episode_drivable()
    check_turn_replay_matches_native_observation_camera()
    print("clean RGB + matching dimensions + stale-frame rejection")


if __name__ == "__main__":
    try:
        main()
    finally:
        model.TERMINAL.close()
