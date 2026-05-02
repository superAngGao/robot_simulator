"""Tests for RerunBackend.

Rerun is an optional dependency. All tests are skipped if it is not installed.
The .rrd file-output test requires a real rr.save() call; shape-dispatch tests
mock the rr module to avoid network connections.
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rendering.backends.rerun_backend import RerunBackend
from rendering.render_scene import ContactPoint, PositionedShape, RenderScene, RenderSensorData, TerrainInfo
from sensing.readings import (
    ContactStateReading,
    ForceSensorReading,
    IMUReading,
    JointStateReading,
    OpticalCameraReading,
)

rr_available = importlib.util.find_spec("rerun") is not None


def _make_scene(shape_types=None) -> RenderScene:
    shapes = []
    _params = {
        "box": {"size": (0.2, 0.2, 0.2)},
        "sphere": {"radius": 0.1},
        "capsule": {"radius": 0.05, "length": 0.2},
        "cylinder": {"radius": 0.05, "length": 0.2},
        "convex_hull": {
            "vertices": np.array(
                [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 2, 4], [0, 4, 3], [1, 2, 4]], dtype=np.int32),
        },
        "mesh": {
            "vertices": np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=np.float64),
            "faces": np.array([[0, 1, 2]], dtype=np.int32),
            "filename": "tri.obj",
        },
    }
    for i, st in enumerate(shape_types or []):
        shapes.append(
            PositionedShape(
                shape_type=st,
                params=_params.get(st, {}),
                position=np.zeros(3),
                rotation=np.eye(3),
                body_index=i,
                body_name=f"body_{i}",
            )
        )
    contacts = [
        ContactPoint(
            position=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            depth=0.01,
            body_i=0,
            body_j=-1,
        )
    ]
    return RenderScene(
        shapes=shapes,
        contacts=contacts,
        terrain=TerrainInfo(terrain_type="flat", params={"z": 0.0}),
        skeleton_links=[(np.zeros(3), np.array([0.1, 0.0, 0.0]))],
        body_positions=[np.zeros(3)],
        body_names=["body_0"],
    )


class TestRerunBackend:
    def test_constructor_rejects_invalid_config(self):
        with pytest.raises(ValueError, match="terrain_half_size"):
            RerunBackend(terrain_half_size=0.0)
        with pytest.raises(ValueError, match="max_sensor_array_scalars"):
            RerunBackend(max_sensor_array_scalars=-1)
        with pytest.raises(ValueError, match="unknown sensor_scalar_groups"):
            RerunBackend(sensor_scalar_groups=("contact", "camera"))

    def test_sensor_scalar_groups_accepts_single_string(self):
        RerunBackend(sensor_scalar_groups="contact")

    @pytest.mark.skipif(not rr_available, reason="rerun not installed")
    def test_set_output_saves_rrd(self, tmp_path):
        """open() with save_path -> .rrd file created."""
        out = str(tmp_path / "debug.rrd")
        b = RerunBackend(app_id="test_app", save_path=out)
        b.open()
        b.render_frame(_make_scene(["box"]), timestamp=0.0)
        b.close()
        assert os.path.exists(out)

    def test_all_supported_shapes_do_not_raise(self):
        """All supported shape types render without error (mocked rr)."""
        mock_rr = MagicMock()
        mock_rr.Quaternion = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend()
            b.open()
            scene = _make_scene(["box", "sphere", "capsule", "cylinder", "convex_hull", "mesh"])
            b.render_frame(scene, timestamp=0.0)
            b.close()

    def test_convex_hull_uses_precomputed_faces(self):
        """convex_hull branch reads params['faces'] — does NOT recompute hull."""
        mock_rr = MagicMock()
        mock_rr.Quaternion = MagicMock(return_value=MagicMock())
        faces = np.array([[0, 2, 4], [0, 4, 3]], dtype=np.int32)
        verts = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=np.float64,
        )
        shape = PositionedShape(
            shape_type="convex_hull",
            params={"vertices": verts, "faces": faces},
            position=np.zeros(3),
            rotation=np.eye(3),
            body_index=0,
            body_name="hull",
        )
        scene = RenderScene(
            shapes=[shape],
            contacts=[],
            terrain=TerrainInfo(terrain_type="flat", params={"z": 0.0}),
            skeleton_links=[],
            body_positions=[],
            body_names=[],
        )
        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend()
            b.open()
            b.render_frame(scene, timestamp=0.0)
        # Mesh3D must have been called with the pre-computed faces
        call_kwargs = mock_rr.Mesh3D.call_args
        assert call_kwargs is not None
        passed_faces = call_kwargs.kwargs.get("triangle_indices")
        assert passed_faces is not None
        assert passed_faces.shape == (2, 3)

    def test_mesh_missing_topology_skips_without_warning(self, caplog):
        mock_rr = MagicMock()
        mock_rr.Quaternion = MagicMock(return_value=MagicMock())
        scene = _make_scene([])
        scene.shapes.append(
            PositionedShape(
                shape_type="mesh",
                params={
                    "vertices": np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=np.float64),
                    "faces": None,
                    "filename": "tri.obj",
                },
                position=np.zeros(3),
                rotation=np.eye(3),
                body_index=0,
                body_name="mesh",
            )
        )

        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend()
            b.open()
            with caplog.at_level("WARNING", logger="rendering.backends.rerun_backend"):
                b.render_frame(scene, timestamp=0.0)

        assert "requires vertices + faces" not in caplog.text

    def test_mesh_malformed_faces_warns_and_skips(self, caplog):
        mock_rr = MagicMock()
        mock_rr.Quaternion = MagicMock(return_value=MagicMock())
        scene = _make_scene([])
        scene.shapes.append(
            PositionedShape(
                shape_type="mesh",
                params={
                    "vertices": np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=np.float64),
                    "faces": np.array([0, 1, 2], dtype=np.int32),
                    "filename": "tri.obj",
                },
                position=np.zeros(3),
                rotation=np.eye(3),
                body_index=0,
                body_name="mesh",
            )
        )

        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend()
            b.open()
            with caplog.at_level("WARNING", logger="rendering.backends.rerun_backend"):
                b.render_frame(scene, timestamp=0.0)

        assert "expected mesh vertices" in caplog.text
        mesh_entities = [
            call.args[0]
            for call in mock_rr.log.call_args_list
            if call.args and "shape_0_mesh" in call.args[0]
        ]
        assert mesh_entities == []

    def test_contacts_render_as_arrows(self):
        """Contacts are logged as rr.Arrows3D."""
        mock_rr = MagicMock()
        mock_rr.Quaternion = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend()
            b.open()
            b.render_frame(_make_scene([]), timestamp=0.0)
        mock_rr.Arrows3D.assert_called_once()

    def test_flat_terrain_logs_mesh3d(self):
        """Rerun consumes scene.terrain instead of dropping it."""
        mock_rr = MagicMock()
        mock_rr.Quaternion = MagicMock(return_value=MagicMock())
        mock_rr.Mesh3D.side_effect = lambda **kwargs: {"mesh_kwargs": kwargs}
        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend()
            b.open()
            b.render_frame(_make_scene([]), timestamp=0.0)

        terrain_calls = [
            call for call in mock_rr.log.call_args_list if call.args and call.args[0] == "env_0/terrain"
        ]
        assert len(terrain_calls) == 1
        mesh_kwargs = terrain_calls[0].args[1]["mesh_kwargs"]
        assert mesh_kwargs["vertex_positions"].shape == (4, 3)
        assert mesh_kwargs["triangle_indices"].shape == (2, 3)

    def test_terrain_half_size_is_configurable(self):
        mock_rr = MagicMock()
        mock_rr.Quaternion = MagicMock(return_value=MagicMock())
        mock_rr.Mesh3D.side_effect = lambda **kwargs: {"mesh_kwargs": kwargs}
        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend(terrain_half_size=2.5)
            b.open()
            b.render_frame(_make_scene([]), timestamp=0.0)

        terrain_calls = [
            call for call in mock_rr.log.call_args_list if call.args and call.args[0] == "env_0/terrain"
        ]
        mesh_kwargs = terrain_calls[0].args[1]["mesh_kwargs"]
        np.testing.assert_allclose(mesh_kwargs["vertex_positions"][0], [-2.5, -2.5, 0.0])

    def test_halfspace_terrain_logs_mesh3d(self):
        mock_rr = MagicMock()
        mock_rr.Quaternion = MagicMock(return_value=MagicMock())
        scene = _make_scene([])
        scene.terrain = TerrainInfo(
            terrain_type="halfspace",
            params={
                "normal": np.array([0.0, 1.0, 1.0], dtype=np.float64),
                "point": np.array([0.0, 0.0, 0.2], dtype=np.float64),
            },
        )
        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend()
            b.open()
            b.render_frame(scene, timestamp=0.0)

        terrain_calls = [
            call for call in mock_rr.log.call_args_list if call.args and call.args[0] == "env_0/terrain"
        ]
        assert len(terrain_calls) == 1

    def test_sensor_data_logs_scalar_timelines(self):
        mock_rr = MagicMock()
        mock_rr.Quaternion = MagicMock(return_value=MagicMock())
        mock_rr.Scalars.side_effect = lambda value: {"scalar": value}
        scene = _make_scene([])
        scene.sensor_data = RenderSensorData(
            frame_id=2,
            sim_time=0.02,
            env_idx=0,
            imu_readings=[
                IMUReading(
                    frame_id=2,
                    sim_time=0.02,
                    env_idx=0,
                    body_index=0,
                    orientation_world_R=np.eye(3),
                    angular_velocity_body=np.array([0.1, 0.2, 0.3]),
                    linear_acceleration_body=None,
                )
            ],
            joint_state=JointStateReading(
                frame_id=2,
                sim_time=0.02,
                env_idx=0,
                joint_pos=np.array([1.0, 2.0]),
                joint_vel=np.array([3.0, 4.0]),
            ),
            force=ForceSensorReading(
                frame_id=2,
                sim_time=0.02,
                env_idx=0,
                qfrc_applied=np.array([5.0, 6.0]),
                tau_smooth=None,
                body_force=None,
                contact_force=np.array([[7.0, 8.0, 9.0]]),
            ),
            contact=ContactStateReading(
                frame_id=2,
                sim_time=0.02,
                env_idx=0,
                contact_count=3,
                contact_mask=np.array([1, 0]),
            ),
        )

        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend()
            b.open()
            b.render_frame(scene, timestamp=0.02)

        logged_entities = [call.args[0] for call in mock_rr.log.call_args_list if call.args]
        assert "env_0/sensors/contact/contact_count" in logged_entities
        assert "env_0/sensors/contact/contact_mask/0" in logged_entities
        assert "env_0/sensors/joint/q/0" in logged_entities
        assert "env_0/sensors/joint/qdot/1" in logged_entities
        assert "env_0/sensors/force/qfrc_applied/1" in logged_entities
        assert "env_0/sensors/force/contact_force/norm" in logged_entities
        assert "env_0/sensors/imu/body_0/angular_velocity_body/2" in logged_entities

    def test_sensor_array_scalar_limit_is_configurable(self):
        mock_rr = MagicMock()
        mock_rr.Quaternion = MagicMock(return_value=MagicMock())
        mock_rr.Scalars.side_effect = lambda value: {"scalar": value}
        scene = _make_scene([])
        scene.sensor_data = RenderSensorData(
            frame_id=2,
            sim_time=0.02,
            env_idx=0,
            joint_state=JointStateReading(
                frame_id=2,
                sim_time=0.02,
                env_idx=0,
                joint_pos=np.array([1.0, 2.0, 3.0]),
                joint_vel=None,
            ),
        )

        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend(max_sensor_array_scalars=1)
            b.open()
            b.render_frame(scene, timestamp=0.02)

        logged_entities = [call.args[0] for call in mock_rr.log.call_args_list if call.args]
        assert "env_0/sensors/joint/q/norm" in logged_entities
        assert "env_0/sensors/joint/q/0" in logged_entities
        assert "env_0/sensors/joint/q/1" not in logged_entities
        assert "env_0/sensors/joint/q/truncated_size" in logged_entities

    def test_sensor_logging_can_be_disabled(self):
        mock_rr = MagicMock()
        mock_rr.Quaternion = MagicMock(return_value=MagicMock())
        mock_rr.Scalars.side_effect = lambda value: {"scalar": value}
        scene = _make_scene([])
        scene.sensor_data = RenderSensorData(
            frame_id=2,
            sim_time=0.02,
            env_idx=0,
            contact=ContactStateReading(frame_id=2, sim_time=0.02, env_idx=0, contact_count=3),
        )

        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend(log_sensor_data=False)
            b.open()
            b.render_frame(scene, timestamp=0.02)

        logged_entities = [call.args[0] for call in mock_rr.log.call_args_list if call.args]
        assert "env_0/sensors/contact/contact_count" not in logged_entities

    def test_sensor_scalar_groups_filter_logged_timelines(self):
        mock_rr = MagicMock()
        mock_rr.Quaternion = MagicMock(return_value=MagicMock())
        mock_rr.Scalars.side_effect = lambda value: {"scalar": value}
        scene = _make_scene([])
        scene.sensor_data = RenderSensorData(
            frame_id=2,
            sim_time=0.02,
            env_idx=0,
            imu_readings=[
                IMUReading(
                    frame_id=2,
                    sim_time=0.02,
                    env_idx=0,
                    body_index=0,
                    orientation_world_R=np.eye(3),
                    angular_velocity_body=np.array([0.1, 0.2, 0.3]),
                    linear_acceleration_body=None,
                )
            ],
            joint_state=JointStateReading(
                frame_id=2,
                sim_time=0.02,
                env_idx=0,
                joint_pos=np.array([1.0, 2.0]),
                joint_vel=np.array([3.0, 4.0]),
            ),
            contact=ContactStateReading(
                frame_id=2,
                sim_time=0.02,
                env_idx=0,
                contact_count=3,
                contact_mask=np.array([1, 0]),
            ),
        )

        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend(sensor_scalar_groups=("contact",))
            b.open()
            b.render_frame(scene, timestamp=0.02)

        logged_entities = [call.args[0] for call in mock_rr.log.call_args_list if call.args]
        assert "env_0/sensors/contact/contact_count" in logged_entities
        assert "env_0/sensors/contact/contact_mask/0" in logged_entities
        assert "env_0/sensors/joint/q/0" not in logged_entities
        assert "env_0/sensors/imu/body_0/angular_velocity_body/0" not in logged_entities

    def test_log_optical_camera_reading_logs_supported_image_channels(self):
        mock_rr = MagicMock()
        mock_rr.Image.side_effect = lambda image: {"image": np.asarray(image)}
        mock_rr.DepthImage.side_effect = lambda image, **kwargs: {
            "depth": np.asarray(image),
            "kwargs": kwargs,
        }
        mock_rr.SegmentationImage.side_effect = lambda image: {"segmentation": np.asarray(image)}
        reading = OpticalCameraReading(
            frame_id=4,
            sim_time=0.04,
            env_idx=2,
            sensor_id="cam/main",
            image_shape=(2, 2),
            channels={
                "rgb": np.array(
                    [
                        [[0.0, 0.5, 1.0], [2.0, -1.0, np.nan]],
                        [[0.25, 0.25, 0.25], [1.0, 1.0, 1.0]],
                    ],
                    dtype=np.float64,
                ),
                "depth_m": np.array([[1.0, np.inf], [2.0, 3.0]], dtype=np.float64),
                "range_m": np.array([[1.5, np.inf], [2.5, 3.5]], dtype=np.float64),
                "numeric_instance_id": np.array([[1, 0], [2, 3]], dtype=np.int64),
                "intensity": np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float64),
                "position_world": np.zeros((2, 2, 3), dtype=np.float64),
            },
        )

        with patch.dict("sys.modules", {"rerun": mock_rr}):
            b = RerunBackend()
            b.open()
            b.log_optical_camera_reading(reading)

        mock_rr.set_time.assert_called_with("sim_time", timestamp=0.04)
        logged = {call.args[0]: call.args[1] for call in mock_rr.log.call_args_list if call.args}

        prefix = "env_2/sensors/optical/cam_main"
        assert f"{prefix}/rgb" in logged
        assert f"{prefix}/depth_m" in logged
        assert f"{prefix}/range_m" in logged
        assert f"{prefix}/numeric_instance_id" in logged
        assert f"{prefix}/intensity" in logged
        assert f"{prefix}/position_world" not in logged

        rgb_payload = logged[f"{prefix}/rgb"]["image"]
        assert rgb_payload.dtype == np.uint8
        np.testing.assert_array_equal(rgb_payload[0, 0], [0, 128, 255])
        np.testing.assert_array_equal(rgb_payload[0, 1], [255, 0, 0])

        depth_payload = logged[f"{prefix}/depth_m"]["depth"]
        assert depth_payload.dtype == np.float32
        assert logged[f"{prefix}/depth_m"]["kwargs"] == {"meter": 1.0}

        segmentation_payload = logged[f"{prefix}/numeric_instance_id"]["segmentation"]
        assert segmentation_payload.dtype == np.uint32
        np.testing.assert_array_equal(segmentation_payload, [[1, 0], [2, 3]])

        intensity_payload = logged[f"{prefix}/intensity"]["image"]
        assert intensity_payload.dtype == np.uint8
        assert intensity_payload.shape == (2, 2)

    def test_log_optical_camera_reading_supports_explicit_prefix_and_channels(self):
        mock_rr = MagicMock()
        mock_rr.DepthImage.side_effect = lambda image, **kwargs: {
            "depth": np.asarray(image),
            "kwargs": kwargs,
        }
        reading = OpticalCameraReading(
            frame_id=4,
            sim_time=0.04,
            env_idx=0,
            sensor_id="cam",
            image_shape=(1, 2),
            channels={
                "depth_m": np.array([[1.0, 2.0]], dtype=np.float64),
                "range_m": np.array([[1.5, 2.5]], dtype=np.float64),
            },
        )

        with patch.dict("sys.modules", {"rerun": mock_rr}):
            RerunBackend().log_optical_camera_reading(
                reading,
                timestamp=0.5,
                channels=("range_m",),
                entity_prefix="custom/camera",
            )

        mock_rr.set_time.assert_called_with("sim_time", timestamp=0.5)
        logged_entities = [call.args[0] for call in mock_rr.log.call_args_list if call.args]
        assert logged_entities == ["custom/camera/range_m"]

    def test_log_optical_camera_reading_rejects_explicit_unsupported_channel(self):
        mock_rr = MagicMock()
        reading = OpticalCameraReading(
            frame_id=4,
            sim_time=0.04,
            env_idx=0,
            sensor_id="cam",
            image_shape=(1, 1),
            channels={"position_world": np.zeros((1, 1, 3), dtype=np.float64)},
        )

        with patch.dict("sys.modules", {"rerun": mock_rr}):
            with pytest.raises(ValueError, match="unsupported"):
                RerunBackend().log_optical_camera_reading(reading, channels=("position_world",))

    def test_log_optical_camera_reading_rejects_malformed_image_channel(self):
        mock_rr = MagicMock()
        reading = OpticalCameraReading(
            frame_id=4,
            sim_time=0.04,
            env_idx=0,
            sensor_id="cam",
            image_shape=(2, 2),
            channels={"depth_m": np.array([1.0, 2.0], dtype=np.float64)},
        )

        with patch.dict("sys.modules", {"rerun": mock_rr}):
            with pytest.raises(ValueError, match="metric"):
                RerunBackend().log_optical_camera_reading(reading)
