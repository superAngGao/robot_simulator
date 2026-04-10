"""
Matplotlib 3D drawing functions for collision shapes.

Each function takes an Axes3D, world-frame position/rotation, and
shape-specific parameters, and returns a list of matplotlib artists.
Stateless pure functions — easy to test and extend.
"""

from __future__ import annotations

from typing import List

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .render_scene import ContactPoint, TerrainInfo

# ---------------------------------------------------------------------------
# Color defaults
# ---------------------------------------------------------------------------

_SHAPE_COLOR = "#4C9BE8"
_SHAPE_ALPHA = 0.3
_SHAPE_EDGE = "#333333"
_CONTACT_COLOR = "#E84C4C"
_CONTACT_ARROW_COLOR = "#CC3333"
_GROUND_COLOR = "#CCCCCC"
_NORMAL_SCALE = 0.05  # arrow length for contact normals [m]

# ---------------------------------------------------------------------------
# Shape drawing functions
# ---------------------------------------------------------------------------


def draw_box(ax, position, rotation, size, color=_SHAPE_COLOR, alpha=_SHAPE_ALPHA, **kwargs) -> list:
    """Draw a 3D box as semi-transparent faces."""
    hx, hy, hz = np.asarray(size) / 2.0
    # 8 corners in local frame
    local = np.array(
        [
            [-hx, -hy, -hz],
            [-hx, -hy, hz],
            [-hx, hy, -hz],
            [-hx, hy, hz],
            [hx, -hy, -hz],
            [hx, -hy, hz],
            [hx, hy, -hz],
            [hx, hy, hz],
        ]
    )
    # Transform to world
    world = (rotation @ local.T).T + position
    # 6 faces (indices into corner array)
    faces = [
        [0, 1, 3, 2],  # -X
        [4, 5, 7, 6],  # +X
        [0, 1, 5, 4],  # -Y
        [2, 3, 7, 6],  # +Y
        [0, 2, 6, 4],  # -Z
        [1, 3, 7, 5],  # +Z
    ]
    polys = [[world[i] for i in f] for f in faces]
    pc = Poly3DCollection(polys, alpha=alpha, facecolor=color, edgecolor=_SHAPE_EDGE, linewidth=0.5)
    ax.add_collection3d(pc)
    return [pc]


def draw_sphere(
    ax, position, rotation, radius, n_u=12, n_v=8, color=_SHAPE_COLOR, alpha=0.2, **kwargs
) -> list:
    """Draw a wireframe sphere."""
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    # Apply rotation and translation
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            pt = rotation @ np.array([x[i, j], y[i, j], z[i, j]]) + position
            x[i, j], y[i, j], z[i, j] = pt
    wf = ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.4)
    return [wf]


def draw_cylinder(
    ax, position, rotation, radius, length, n_sides=16, color=_SHAPE_COLOR, alpha=_SHAPE_ALPHA, **kwargs
) -> list:
    """Draw a cylinder aligned with local Z axis."""
    half_l = length / 2.0
    theta = np.linspace(0, 2 * np.pi, n_sides + 1)
    # Top and bottom circles in local frame
    top_local = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, half_l)])
    bot_local = np.column_stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, -half_l)]
    )
    # Transform
    top = (rotation @ top_local.T).T + position
    bot = (rotation @ bot_local.T).T + position

    artists = []
    # Side faces as quad strips
    side_polys = []
    for i in range(n_sides):
        quad = [bot[i], bot[i + 1], top[i + 1], top[i]]
        side_polys.append(quad)
    pc = Poly3DCollection(side_polys, alpha=alpha, facecolor=color, edgecolor=_SHAPE_EDGE, linewidth=0.3)
    ax.add_collection3d(pc)
    artists.append(pc)

    # Top and bottom caps
    top_cap = Poly3DCollection(
        [top.tolist()], alpha=alpha, facecolor=color, edgecolor=_SHAPE_EDGE, linewidth=0.3
    )
    bot_cap = Poly3DCollection(
        [bot.tolist()], alpha=alpha, facecolor=color, edgecolor=_SHAPE_EDGE, linewidth=0.3
    )
    ax.add_collection3d(top_cap)
    ax.add_collection3d(bot_cap)
    artists.extend([top_cap, bot_cap])

    return artists


def draw_capsule(
    ax,
    position,
    rotation,
    radius,
    length,
    n_sides=12,
    n_hemi=6,
    color=_SHAPE_COLOR,
    alpha=_SHAPE_ALPHA,
    **kwargs,
) -> list:
    """Draw a capsule (cylinder + hemisphere caps) aligned with local Z."""
    artists = []
    half_l = length / 2.0

    # Cylinder body
    artists += draw_cylinder(
        ax, position, rotation, radius, length, n_sides=n_sides, color=color, alpha=alpha
    )

    # Hemisphere caps (parametric half-sphere)
    u = np.linspace(0, 2 * np.pi, n_sides + 1)
    for sign, v_range in [(+1, (0, np.pi / 2)), (-1, (np.pi / 2, np.pi))]:
        v = np.linspace(v_range[0], v_range[1], n_hemi)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones_like(u), np.cos(v)) + sign * half_l
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                pt = rotation @ np.array([x[i, j], y[i, j], z[i, j]]) + position
                x[i, j], y[i, j], z[i, j] = pt
        wf = ax.plot_wireframe(x, y, z, color=color, alpha=alpha * 0.8, linewidth=0.3)
        artists.append(wf)

    return artists


def draw_convex_hull(
    ax, position, rotation, vertices, color=_SHAPE_COLOR, alpha=_SHAPE_ALPHA, **kwargs
) -> list:
    """Draw a convex hull from vertices using scipy triangulation."""
    from scipy.spatial import ConvexHull

    # Transform vertices to world frame
    world_verts = (rotation @ vertices.T).T + position
    hull = ConvexHull(world_verts)

    faces = [[world_verts[idx] for idx in simplex] for simplex in hull.simplices]
    pc = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=_SHAPE_EDGE, linewidth=0.3)
    ax.add_collection3d(pc)
    return [pc]


# ---------------------------------------------------------------------------
# Contact visualization
# ---------------------------------------------------------------------------


def draw_contacts(ax, contacts: List[ContactPoint], arrow_scale=_NORMAL_SCALE, **kwargs) -> list:
    """Draw contact points as red dots + normal arrows."""
    if not contacts:
        return []

    artists = []
    positions = np.array([c.position for c in contacts])
    normals = np.array([c.normal for c in contacts])

    # Red dots
    sc = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=60,
        c=_CONTACT_COLOR,
        depthshade=True,
        zorder=5,
    )
    artists.append(sc)

    # Normal arrows
    q = ax.quiver(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        normals[:, 0] * arrow_scale,
        normals[:, 1] * arrow_scale,
        normals[:, 2] * arrow_scale,
        color=_CONTACT_ARROW_COLOR,
        arrow_length_ratio=0.3,
        linewidth=1.5,
    )
    artists.append(q)

    return artists


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------


def draw_terrain(ax, terrain: TerrainInfo, floor_size: float = 1.0, **kwargs) -> list:
    """Draw terrain based on type."""
    if terrain.terrain_type == "flat":
        return _draw_flat_terrain(ax, terrain.params.get("z", 0.0), floor_size)
    elif terrain.terrain_type == "halfspace":
        return _draw_halfspace_terrain(ax, terrain.params, floor_size)
    return []


def _draw_flat_terrain(ax, z: float, floor_size: float) -> list:
    xs = np.linspace(-floor_size, floor_size, 5)
    ys = np.linspace(-floor_size, floor_size, 5)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, z)
    surf = ax.plot_surface(xx, yy, zz, alpha=0.25, color=_GROUND_COLOR, linewidth=0, zorder=0)
    return [surf]


def _draw_halfspace_terrain(ax, params: dict, floor_size: float) -> list:
    normal = np.asarray(params["normal"])
    point = np.asarray(params["point"])
    # Build two tangent vectors
    if abs(normal[2]) < 0.99:
        t1 = np.cross(normal, [0, 0, 1])
    else:
        t1 = np.cross(normal, [1, 0, 0])
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    t2 = t2 / np.linalg.norm(t2)
    # 4 corners
    corners = np.array(
        [point + s * floor_size * t1 + t * floor_size * t2 for s, t in [(-1, -1), (-1, 1), (1, 1), (1, -1)]]
    )
    pc = Poly3DCollection(
        [corners.tolist()], alpha=0.25, facecolor=_GROUND_COLOR, edgecolor="#AAAAAA", linewidth=0.5
    )
    ax.add_collection3d(pc)
    return [pc]


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

SHAPE_DRAWERS = {
    "box": draw_box,
    "sphere": draw_sphere,
    "cylinder": draw_cylinder,
    "capsule": draw_capsule,
    "convex_hull": draw_convex_hull,
}
