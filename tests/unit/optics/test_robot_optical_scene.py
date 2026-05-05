from __future__ import annotations

import numpy as np

from benchmarks.robot_optical_scene import (
    build_robot_optical_scene,
    make_robot_camera_rays,
)
from optics import OpticalFrameInputs, OpticalSceneCache


def test_robot_optical_scene_builds_body_bound_instances():
    scene = build_robot_optical_scene(detail="proxy", num_robots=1)

    assert scene.num_bodies == 13
    assert scene.num_triangles == 1104
    body_bound = [instance for instance in scene.registry.instances if instance.body_index is not None]
    assert len(body_bound) == 13

    snapshot = OpticalSceneCache(scene.registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(scene.cpu_frame),
        acceleration="cpu_bvh",
    )

    assert snapshot.acceleration is not None
    assert snapshot.acceleration.triangles_world.shape[0] == scene.num_triangles


def test_robot_camera_rays_point_toward_scene():
    scene = build_robot_optical_scene(detail="proxy", num_robots=1)
    rays = make_robot_camera_rays(
        num_rays=256,
        frame_id=scene.cpu_frame.frame_id,
        sim_time=scene.cpu_frame.sim_time,
        num_robots=1,
    )

    assert rays.num_rays == 256
    assert rays.sensor_role == "depth"
    direction_norms = np.linalg.norm(rays.directions_world, axis=1)
    np.testing.assert_allclose(direction_norms, 1.0)
    assert float(np.mean(rays.directions_world[:, 2])) < 0.0
