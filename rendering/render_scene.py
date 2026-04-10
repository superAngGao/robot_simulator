"""
Backend-agnostic scene description for rendering.

RenderScene collects all renderable elements extracted from physics state.
It contains only plain dataclasses and numpy arrays — no rendering library
dependencies. Any backend (matplotlib, OpenGL, Vulkan) can consume it.

Design reference: Drake SceneGraph role-based geometry separation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PositionedShape:
    """A collision shape positioned in world frame.

    Attributes:
        shape_type : One of "box", "sphere", "cylinder", "capsule",
                     "convex_hull", "mesh".
        params     : Shape-specific parameters (see module docstring).
        position   : (3,) world-frame position.
        rotation   : (3,3) world-frame rotation matrix.
        body_index : Source body index (for color coding / selection).
        body_name  : Human-readable body name.
    """

    shape_type: str
    params: dict
    position: NDArray[np.float64]
    rotation: NDArray[np.float64]
    body_index: int
    body_name: str


@dataclass(frozen=True)
class ContactPoint:
    """A single contact point for visualization.

    Attributes:
        position : (3,) contact point in world frame.
        normal   : (3,) unit contact normal.
        depth    : Penetration depth [m] (positive = penetrating).
        body_i   : First body index.
        body_j   : Second body index (-1 = ground).
    """

    position: NDArray[np.float64]
    normal: NDArray[np.float64]
    depth: float
    body_i: int
    body_j: int


@dataclass(frozen=True)
class TerrainInfo:
    """Backend-agnostic terrain description.

    Attributes:
        terrain_type : One of "flat", "halfspace", "heightmap".
        params       : Type-specific parameters.
                       flat: {"z": float}
                       halfspace: {"normal": NDArray, "point": NDArray}
                       heightmap: {"data": NDArray, "resolution": float, ...}
    """

    terrain_type: str
    params: dict


@dataclass
class RenderScene:
    """Complete renderable scene snapshot — backend-agnostic.

    Produced by ``scene_builder.build_render_scene()`` from physics state.
    Consumed by rendering backends (matplotlib, OpenGL, Vulkan).

    The ``deformable_meshes`` and ``particles`` slots are empty for now,
    reserved for future multi-physics rendering (soft body, fluid, cloth).
    """

    shapes: list[PositionedShape]
    contacts: list[ContactPoint]
    terrain: TerrainInfo
    skeleton_links: list[tuple[NDArray[np.float64], NDArray[np.float64]]]
    body_positions: list[NDArray[np.float64]]
    body_names: list[str]
    # Future extension slots
    deformable_meshes: list = field(default_factory=list)
    particles: list = field(default_factory=list)
