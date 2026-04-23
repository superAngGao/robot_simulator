"""
Matplotlib 3D rendering backend.

Wraps the existing shape_artists drawing functions behind the RenderBackend
interface. Stores per-frame RenderScene snapshots and uses FuncAnimation
(remove-and-redraw pattern) for GIF export — consistent with RobotViewer.

Design reference: matplotlib FuncAnimation — update callback removes old
artists and adds new ones each frame, keeping artists attached to the figure.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers projection

from ..render_scene import RenderScene
from ..shape_artists import SHAPE_DRAWERS, draw_contacts, draw_terrain
from .base import RenderBackend


class MatplotlibBackend(RenderBackend):
    """Offscreen / file-output backend using matplotlib 3D.

    Args:
        figsize   : Figure size in inches (width, height).
        save_path : If set, ``close()`` saves an animated GIF here.
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (8.0, 6.0),
        save_path: Optional[str] = None,
    ) -> None:
        self._figsize = figsize
        self._save_path = save_path
        self._fig = None
        self._ax = None
        # Store (scene, timestamp) pairs for deferred animation export
        self._frames: List[Tuple[RenderScene, float]] = []

    # ------------------------------------------------------------------
    # RenderBackend interface
    # ------------------------------------------------------------------

    def open(self) -> None:
        self._fig = plt.figure(figsize=self._figsize)
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._frames = []

    def render_frame(self, scene: RenderScene, timestamp: float, env_index: int = 0) -> None:
        self._frames.append((scene, timestamp))
        # Also draw immediately so the axes reflect the latest frame
        self._draw_scene(self._ax, scene)

    def close(self) -> None:
        if self._save_path and self._frames:
            self._save_animation()
        if self._fig is not None:
            plt.close(self._fig)
        self._fig = None
        self._ax = None

    def set_output(self, path: str) -> None:
        self._save_path = path

    @property
    def supports_offscreen(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_scene(self, ax, scene: RenderScene) -> list:
        """Clear axes and draw scene; return list of new artists."""
        ax.cla()
        artists: list = []

        artists.extend(draw_terrain(ax, scene.terrain))

        for p0, p1 in scene.skeleton_links:
            (line,) = ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color="#FF8C00",
                linewidth=1.5,
            )
            artists.append(line)

        for shape in scene.shapes:
            drawer = SHAPE_DRAWERS.get(shape.shape_type)
            if drawer is None:
                continue  # mesh / unknown — skip silently
            artists.extend(drawer(ax, shape.position, shape.rotation, **shape.params))

        artists.extend(draw_contacts(ax, scene.contacts))
        return artists

    def _save_animation(self) -> None:
        from matplotlib.animation import FuncAnimation

        fig = self._fig
        ax = self._ax
        active_artists: list = []

        def update(frame_idx: int) -> list:
            nonlocal active_artists
            for a in active_artists:
                try:
                    a.remove()
                except Exception:
                    pass
            scene, _ = self._frames[frame_idx]
            active_artists = self._draw_scene(ax, scene)
            return active_artists

        anim = FuncAnimation(fig, update, frames=len(self._frames), interval=50, blit=False)
        anim.save(self._save_path, writer="pillow")
