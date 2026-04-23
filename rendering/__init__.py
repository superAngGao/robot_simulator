"""
rendering — Visualisation tools for the robot simulator.
"""

from .backends.base import RenderBackend
from .backends.matplotlib_backend import MatplotlibBackend
from .backends.rerun_backend import RerunBackend
from .render_scene import ContactPoint, PositionedShape, RenderScene, TerrainInfo
from .scene_builder import build_render_scene, build_render_scene_from_gpu, build_render_scene_from_tree
from .viewer import RobotViewer

__all__ = [
    "RobotViewer",
    "RenderScene",
    "PositionedShape",
    "ContactPoint",
    "TerrainInfo",
    "build_render_scene",
    "build_render_scene_from_tree",
    "build_render_scene_from_gpu",
    "RenderBackend",
    "MatplotlibBackend",
    "RerunBackend",
]
