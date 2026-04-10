"""Smoke tests for rendering.shape_artists — verify no crashes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rendering.render_scene import ContactPoint, TerrainInfo
from rendering.shape_artists import (
    SHAPE_DRAWERS,
    draw_box,
    draw_capsule,
    draw_contacts,
    draw_convex_hull,
    draw_cylinder,
    draw_sphere,
    draw_terrain,
)

# Identity pose
_POS = np.zeros(3)
_ROT = np.eye(3)
# 45-degree rotation around Z
_ROT45 = np.array(
    [
        [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
        [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
        [0, 0, 1],
    ]
)


@pytest.fixture
def ax3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    yield ax
    plt.close(fig)


class TestShapeArtists:
    def test_draw_box(self, ax3d):
        artists = draw_box(ax3d, _POS, _ROT, size=(0.2, 0.3, 0.4))
        assert len(artists) > 0

    def test_draw_sphere(self, ax3d):
        artists = draw_sphere(ax3d, _POS, _ROT, radius=0.1)
        assert len(artists) > 0

    def test_draw_cylinder(self, ax3d):
        artists = draw_cylinder(ax3d, _POS, _ROT, radius=0.05, length=0.2)
        assert len(artists) > 0

    def test_draw_capsule(self, ax3d):
        artists = draw_capsule(ax3d, _POS, _ROT, radius=0.03, length=0.15)
        assert len(artists) > 0

    def test_draw_convex_hull(self, ax3d):
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
        artists = draw_convex_hull(ax3d, _POS, _ROT, vertices=verts)
        assert len(artists) > 0

    def test_draw_box_rotated(self, ax3d):
        artists = draw_box(ax3d, np.array([1, 0, 0.5]), _ROT45, size=(0.1, 0.1, 0.1))
        assert len(artists) > 0

    def test_draw_all_shapes_rotated(self, ax3d):
        """All shape types with rotation — no crash."""
        pos = np.array([0.5, 0.3, 0.2])
        draw_box(ax3d, pos, _ROT45, size=(0.1, 0.1, 0.1))
        draw_sphere(ax3d, pos, _ROT45, radius=0.05)
        draw_cylinder(ax3d, pos, _ROT45, radius=0.03, length=0.1)
        draw_capsule(ax3d, pos, _ROT45, radius=0.02, length=0.08)

    def test_dispatch_table_complete(self):
        assert set(SHAPE_DRAWERS.keys()) == {"box", "sphere", "cylinder", "capsule", "convex_hull"}


class TestContactArtists:
    def test_draw_contacts(self, ax3d):
        contacts = [
            ContactPoint(
                position=np.array([0, 0, 0]), normal=np.array([0, 0, 1]), depth=0.01, body_i=0, body_j=-1
            ),
            ContactPoint(
                position=np.array([0.1, 0, 0]), normal=np.array([1, 0, 0]), depth=0.005, body_i=0, body_j=1
            ),
        ]
        artists = draw_contacts(ax3d, contacts)
        assert len(artists) == 2  # scatter + quiver

    def test_draw_empty_contacts(self, ax3d):
        artists = draw_contacts(ax3d, [])
        assert len(artists) == 0


class TestTerrainArtists:
    def test_draw_flat_terrain(self, ax3d):
        terrain = TerrainInfo(terrain_type="flat", params={"z": 0.0})
        artists = draw_terrain(ax3d, terrain, floor_size=1.0)
        assert len(artists) > 0

    def test_draw_halfspace_terrain(self, ax3d):
        terrain = TerrainInfo(
            terrain_type="halfspace",
            params={"normal": np.array([0, 0, 1.0]), "point": np.array([0, 0, 0.1])},
        )
        artists = draw_terrain(ax3d, terrain, floor_size=1.0)
        assert len(artists) > 0

    def test_draw_unknown_terrain(self, ax3d):
        terrain = TerrainInfo(terrain_type="unknown", params={})
        artists = draw_terrain(ax3d, terrain)
        assert len(artists) == 0
