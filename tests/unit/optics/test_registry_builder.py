from __future__ import annotations

import numpy as np

from optics import (
    CpuReferenceOpticalExecutor,
    OpticalBindingBuildResult,
    OpticalFrameInputs,
    OpticalSceneCache,
    OpticalSourceKey,
    OpticalWorldRegistry,
    build_optical_registry_from_robot_model,
)
from physics.geometry import BodyCollisionGeometry, BoxShape, ShapeInstance, SphereShape
from physics.joint import FreeJoint
from physics.publish import CpuPublishedFrame
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from robot.model import RobotModel
from sensing import OpticalRaySensorSpec


def _single_body_model(shape_instance: ShapeInstance) -> RobotModel:
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name="base link",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(1.0, np.eye(3) * 0.001, np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [shape_instance])],
    )


def _frame(*, frame_id: int = 5, sim_time: float = 0.05, X_world=None) -> CpuPublishedFrame:
    return CpuPublishedFrame(
        frame_id=frame_id,
        sim_time=sim_time,
        step_index=frame_id,
        env_mask=None,
        q=np.zeros(0, dtype=np.float64),
        qdot=np.zeros(0, dtype=np.float64),
        X_world=[] if X_world is None else X_world,
        v_bodies=[],
        contact_count=0,
        contacts=[],
        telemetry=None,
    )


class TestOpticalRegistryBuilder:
    def test_builds_collision_only_registry_from_box_geometry(self):
        model = _single_body_model(
            ShapeInstance(
                shape=BoxShape((1.0, 1.0, 0.2)),
                origin_xyz=np.array([0.0, 0.0, 0.5]),
            )
        )

        result = build_optical_registry_from_robot_model(model, model_name="robot 1")

        assert isinstance(result, OpticalBindingBuildResult)
        assert isinstance(result.registry, OpticalWorldRegistry)
        assert not result.diagnostics
        assert len(result.registry.instances) == 1
        instance = result.registry.instances[0]
        assert instance.instance_id == "robot_1/base_link/collision/0/instance"
        assert instance.geometry_id == "robot_1/base_link/collision/0/geometry"
        assert instance.material_id == "default_collision"
        assert instance.numeric_instance_id == 1
        assert instance.roles == frozenset({"depth", "lidar", "segmentation"})
        np.testing.assert_allclose(instance.X_body_geometry.r, [0.0, 0.0, 0.5])

        source_key = next(iter(result.source_to_instance_id))
        assert isinstance(source_key, OpticalSourceKey)
        assert source_key.model_name == "robot 1"
        assert source_key.body_name == "base link"
        assert source_key.body_index == 0
        assert source_key.geometry_role == "collision"
        assert source_key.shape_index == 0
        assert result.source_to_instance_id[source_key] == instance.instance_id
        assert result.instance_to_source[instance.instance_id] == source_key

    def test_builder_output_executes_against_reference_executor(self):
        model = _single_body_model(
            ShapeInstance(
                shape=BoxShape((1.0, 1.0, 0.2)),
                origin_xyz=np.array([0.0, 0.0, 0.5]),
            )
        )
        build = build_optical_registry_from_robot_model(model)
        frame = _frame(X_world=[SpatialTransform.identity()])
        snapshot = OpticalSceneCache(build.registry).snapshot_from_frame_inputs(
            OpticalFrameInputs.from_published_frame(frame)
        )
        spec = OpticalRaySensorSpec(
            frame_id=snapshot.frame_id,
            sim_time=snapshot.sim_time,
            env_idx=snapshot.env_idx,
            sensor_id="depth_probe",
            origins_world=[[0.0, 0.0, 2.0]],
            directions_world=[[0.0, 0.0, -1.0]],
        )

        optical = CpuReferenceOpticalExecutor().execute(snapshot, spec)

        np.testing.assert_array_equal(optical.channel("hit_mask"), [True])
        np.testing.assert_allclose(optical.channel("range_m"), [1.4])
        assert optical.channel("instance_id").tolist() == ["main/base_link/collision/0/instance"]
        np.testing.assert_array_equal(optical.channel("numeric_instance_id"), [1])

    def test_reports_unsupported_smooth_collision_shape(self):
        model = _single_body_model(ShapeInstance(shape=SphereShape(0.1)))

        result = build_optical_registry_from_robot_model(model)

        assert len(result.registry.instances) == 0
        assert len(result.diagnostics) == 1
        diagnostic = result.diagnostics[0]
        assert diagnostic.severity == "warning"
        assert diagnostic.code == "unsupported_collision_shape"
        assert "SphereShape" in diagnostic.message
        assert diagnostic.source_key is not None
        assert diagnostic.source_key.body_name == "base link"
