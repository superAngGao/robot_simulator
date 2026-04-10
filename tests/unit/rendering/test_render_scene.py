"""Tests for rendering.render_scene and rendering.scene_builder."""

from __future__ import annotations

import numpy as np
import pytest

from physics.geometry import (
    BodyCollisionGeometry,
    BoxShape,
    CapsuleShape,
    ConvexHullShape,
    CylinderShape,
    ShapeInstance,
    SphereShape,
)
from physics.joint import FreeJoint
from physics.merged_model import merge_models
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform
from physics.terrain import FlatTerrain, HalfSpaceTerrain
from rendering.scene_builder import build_render_scene, build_render_scene_from_tree
from robot.model import RobotModel

ATOL = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_body_model(shape, body_name="box"):
    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name=body_name,
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
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(shape=shape)])],
        contact_body_names=[body_name],
    )


def _make_merged(model, terrain=None):
    terrain = terrain or FlatTerrain()
    return merge_models({"r": model}, terrain=terrain)


# ---------------------------------------------------------------------------
# Builder tests
# ---------------------------------------------------------------------------


class TestBuildRenderScene:
    def test_single_box(self):
        model = _single_body_model(BoxShape((0.2, 0.3, 0.4)))
        merged = _make_merged(model)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        X = merged.tree.forward_kinematics(q)

        scene = build_render_scene(merged, X)
        assert len(scene.shapes) == 1
        s = scene.shapes[0]
        assert s.shape_type == "box"
        assert s.params["size"] == pytest.approx((0.2, 0.3, 0.4))
        assert "box" in s.body_name

    def test_single_sphere(self):
        model = _single_body_model(SphereShape(0.05))
        merged = _make_merged(model)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        X = merged.tree.forward_kinematics(q)

        scene = build_render_scene(merged, X)
        assert len(scene.shapes) == 1
        assert scene.shapes[0].shape_type == "sphere"
        assert scene.shapes[0].params["radius"] == pytest.approx(0.05)

    def test_multiple_shapes(self):
        """Body with sphere + cylinder → both appear in scene."""
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(
            Body(
                name="link",
                index=0,
                joint=FreeJoint("root"),
                inertia=SpatialInertia(1.0, np.eye(3) * 0.001, np.zeros(3)),
                X_tree=SpatialTransform.identity(),
                parent=-1,
            )
        )
        tree.finalize()
        model = RobotModel(
            tree=tree,
            geometries=[
                BodyCollisionGeometry(
                    0,
                    [
                        ShapeInstance(shape=SphereShape(0.03)),
                        ShapeInstance(shape=CylinderShape(0.02, 0.1)),
                    ],
                )
            ],
        )
        merged = _make_merged(model)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        X = merged.tree.forward_kinematics(q)

        scene = build_render_scene(merged, X)
        assert len(scene.shapes) == 2
        types = {s.shape_type for s in scene.shapes}
        assert types == {"sphere", "cylinder"}

    def test_capsule_params(self):
        model = _single_body_model(CapsuleShape(0.02, 0.15))
        merged = _make_merged(model)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        X = merged.tree.forward_kinematics(q)

        scene = build_render_scene(merged, X)
        s = scene.shapes[0]
        assert s.shape_type == "capsule"
        assert s.params["radius"] == pytest.approx(0.02)
        assert s.params["length"] == pytest.approx(0.15)

    def test_convex_hull_params(self):
        verts = (
            np.array(
                [
                    [-1, -1, -1],
                    [-1, -1, 1],
                    [-1, 1, -1],
                    [-1, 1, 1],
                    [1, -1, -1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [1, 1, 1],
                ],
                dtype=np.float64,
            )
            * 0.05
        )
        model = _single_body_model(ConvexHullShape(verts))
        merged = _make_merged(model)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        X = merged.tree.forward_kinematics(q)

        scene = build_render_scene(merged, X)
        s = scene.shapes[0]
        assert s.shape_type == "convex_hull"
        assert s.params["vertices"].shape == (8, 3)

    def test_world_pose_applied(self):
        """Body translated to (1,2,3) → shape position matches."""
        model = _single_body_model(SphereShape(0.05))
        merged = _make_merged(model)
        q = np.zeros(merged.nq)
        q[0] = 1.0  # identity quat
        q[4] = 1.0  # px
        q[5] = 2.0  # py
        q[6] = 3.0  # pz
        X = merged.tree.forward_kinematics(q)

        scene = build_render_scene(merged, X)
        np.testing.assert_allclose(scene.shapes[0].position, [1, 2, 3], atol=ATOL)

    def test_flat_terrain(self):
        model = _single_body_model(SphereShape(0.05))
        merged = _make_merged(model, terrain=FlatTerrain(z=0.5))
        q = np.zeros(merged.nq)
        q[0] = 1.0
        X = merged.tree.forward_kinematics(q)

        scene = build_render_scene(merged, X, terrain=FlatTerrain(z=0.5))
        assert scene.terrain.terrain_type == "flat"
        assert scene.terrain.params["z"] == pytest.approx(0.5)

    def test_halfspace_terrain(self):
        normal = np.array([0, 0, 1], dtype=np.float64)
        point = np.array([0, 0, 0.1], dtype=np.float64)
        terrain = HalfSpaceTerrain(normal=normal, point=point)

        model = _single_body_model(SphereShape(0.05))
        merged = _make_merged(model, terrain=terrain)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        X = merged.tree.forward_kinematics(q)

        scene = build_render_scene(merged, X, terrain=terrain)
        assert scene.terrain.terrain_type == "halfspace"
        np.testing.assert_allclose(scene.terrain.params["normal"], [0, 0, 1], atol=ATOL)
        np.testing.assert_allclose(scene.terrain.params["point"], [0, 0, 0.1], atol=ATOL)

    def test_skeleton_links(self):
        """Two-body chain → one skeleton link."""
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(
            Body(
                "parent",
                0,
                FreeJoint("root"),
                SpatialInertia(1, np.eye(3) * 0.001, np.zeros(3)),
                SpatialTransform.identity(),
                -1,
            )
        )
        tree.add_body(
            Body(
                "child",
                0,
                FreeJoint("j"),
                SpatialInertia(1, np.eye(3) * 0.001, np.zeros(3)),
                SpatialTransform(np.eye(3), np.array([0, 0, -0.2])),
                0,
            )
        )
        tree.finalize()
        model = RobotModel(tree=tree, geometries=[])
        merged = _make_merged(model)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        q[7] = 1.0
        X = merged.tree.forward_kinematics(q)

        scene = build_render_scene(merged, X)
        assert len(scene.skeleton_links) == 1
        assert len(scene.body_positions) == 2
        assert len(scene.body_names) == 2

    def test_contacts(self):
        """ContactInfo list → ContactPoint list."""
        from physics.engine import ContactInfo

        model = _single_body_model(SphereShape(0.05))
        merged = _make_merged(model)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        X = merged.tree.forward_kinematics(q)

        contacts = [
            ContactInfo(
                body_i=0, body_j=-1, depth=0.01, normal=np.array([0, 0, 1.0]), point=np.array([0, 0, 0])
            ),
        ]
        scene = build_render_scene(merged, X, contacts=contacts)
        assert len(scene.contacts) == 1
        c = scene.contacts[0]
        assert c.body_i == 0
        assert c.body_j == -1
        assert c.depth == pytest.approx(0.01)
        np.testing.assert_allclose(c.normal, [0, 0, 1])

    def test_empty_scene(self):
        """No geometries, no contacts → empty lists."""
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(
            Body(
                "b",
                0,
                FreeJoint("root"),
                SpatialInertia(1, np.eye(3) * 0.001, np.zeros(3)),
                SpatialTransform.identity(),
                -1,
            )
        )
        tree.finalize()
        model = RobotModel(tree=tree, geometries=[])
        merged = _make_merged(model)
        q = np.zeros(merged.nq)
        q[0] = 1.0
        X = merged.tree.forward_kinematics(q)

        scene = build_render_scene(merged, X)
        assert len(scene.shapes) == 0
        assert len(scene.contacts) == 0


class TestBuildFromTree:
    def test_convenience_function(self):
        """build_render_scene_from_tree runs FK internally."""
        tree = RobotTreeNumpy(gravity=9.81)
        tree.add_body(
            Body(
                "b",
                0,
                FreeJoint("root"),
                SpatialInertia(1, np.eye(3) * 0.001, np.zeros(3)),
                SpatialTransform.identity(),
                -1,
            )
        )
        tree.finalize()
        geoms = [BodyCollisionGeometry(0, [ShapeInstance(shape=BoxShape((0.1, 0.1, 0.1)))])]

        q = np.zeros(tree.nq)
        q[0] = 1.0
        q[6] = 0.5  # z=0.5

        scene = build_render_scene_from_tree(tree, q, geometries=geoms)
        assert len(scene.shapes) == 1
        assert scene.shapes[0].shape_type == "box"
        np.testing.assert_allclose(scene.shapes[0].position[2], 0.5, atol=ATOL)
