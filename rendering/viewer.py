"""
3D visualiser for the robot simulator.

Renders the robot skeleton (bodies as spheres, joints connected by lines)
and the ground plane using matplotlib 3D. Supports both:
  - Static snapshot  : render a single pose.
  - Animated replay  : play back a recorded simulation trajectory.

This is intentionally lightweight — the goal is fast debugging and
trajectory inspection, not photorealistic rendering (that's Phase 3).

Usage
-----
    viewer = RobotViewer(tree)
    viewer.render_pose(q)           # show single frame

    viewer.animate(times, qs,       # replay trajectory
                   interval=20,
                   save_path="out.gif")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from physics.robot_tree import RobotTree
from physics.spatial import SpatialTransform

if TYPE_CHECKING:
    from .render_scene import RenderScene

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

PALETTE = {
    "body": "#4C9BE8",  # blue spheres for bodies
    "joint": "#E8854C",  # orange lines for links
    "contact": "#E84C4C",  # red sphere for foot in contact
    "ground": "#CCCCCC",  # light grey ground plane
    "origin": "#888888",  # axis ticks
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _body_world_pos(X: SpatialTransform) -> np.ndarray:
    """Return the world-frame origin of a body transform."""
    return X.r.copy()


# ---------------------------------------------------------------------------
# RobotViewer
# ---------------------------------------------------------------------------


class RobotViewer:
    """Interactive 3D viewer for a RobotTree.

    Args:
        tree          : The articulated body tree to visualise.
        ground_z      : Z-level of the ground plane.
        floor_size    : Half-size of the rendered floor grid [m].
        body_radius   : Visual radius of body spheres [m].
        contact_names : Set of body names to highlight as foot contacts.
    """

    def __init__(
        self,
        tree: RobotTree,
        ground_z: float = 0.0,
        floor_size: float = 1.0,
        body_radius: float = 0.03,
        contact_names: Optional[List[str]] = None,
    ) -> None:
        self.tree = tree
        self.ground_z = ground_z
        self.floor_size = floor_size
        self.body_radius = body_radius
        self.contact_names = set(contact_names or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_pose(
        self,
        q: Optional[np.ndarray] = None,
        title: str = "Robot pose",
        show: bool = True,
        save_path: Optional[str] = None,
        render_scene: Optional["RenderScene"] = None,
    ) -> plt.Figure:
        """Render a single robot pose.

        Args:
            q            : Generalised positions (not needed if render_scene given).
            title        : Window / figure title.
            show         : If True, call plt.show().
            save_path    : If given, save figure to this path.
            render_scene : If given, draw collision shapes + contacts
                           instead of body-sphere skeleton.

        Returns:
            The matplotlib Figure object.
        """
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        if render_scene is not None:
            self._draw_scene(ax, render_scene)
        else:
            self._draw_frame(ax, q)
        ax.set_title(title)
        self._configure_axes(ax)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def animate(
        self,
        times: np.ndarray,
        qs: Optional[np.ndarray] = None,
        interval: int = 20,
        title: str = "Simulation replay",
        show: bool = True,
        save_path: Optional[str] = None,
        render_scenes: Optional[list] = None,
    ) -> animation.FuncAnimation:
        """Animate a recorded trajectory.

        Args:
            times          : (N,) time stamps [s].
            qs             : (N, nq) generalised positions (not needed if
                             render_scenes is given).
            interval       : Delay between frames in milliseconds.
            title          : Figure title.
            show           : If True, call plt.show().
            save_path      : If given, save to .gif or .mp4 (requires ffmpeg/pillow).
            render_scenes  : List of RenderScene objects, one per frame.

        Returns:
            The FuncAnimation object.
        """
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        self._configure_axes(ax)

        time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, color="#333333")
        artists: list = []  # cleared each frame

        def update(frame_idx: int) -> list:
            nonlocal artists
            for a in artists:
                a.remove()
            artists = []
            ax.set_title(title)

            if render_scenes is not None:
                artists = self._draw_scene(ax, render_scenes[frame_idx])
            else:
                artists = self._draw_frame(ax, qs[frame_idx])
            time_text.set_text(f"t = {times[frame_idx]:.3f} s")
            return artists + [time_text]

        n_frames = len(render_scenes) if render_scenes is not None else len(qs)
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=interval,
            blit=False,
        )

        if save_path:
            writer = "pillow" if save_path.endswith(".gif") else "ffmpeg"
            anim.save(save_path, writer=writer, dpi=120)

        if show:
            plt.show()

        return anim

    # ------------------------------------------------------------------
    # Internal drawing helpers
    # ------------------------------------------------------------------

    def _draw_frame(
        self,
        ax: "Axes3D",
        q: np.ndarray,
    ) -> list:
        """Draw bodies, links, and ground for a given q. Returns artist list."""
        X_world = self.tree.forward_kinematics(q)
        artists = []

        # Ground plane (mesh grid)
        artists += self._draw_ground(ax)

        # Links (lines between parent and child body origins)
        artists += self._draw_links(ax, X_world)

        # Bodies (scatter spheres)
        artists += self._draw_bodies(ax, X_world)

        return artists

    def _draw_ground(self, ax: "Axes3D") -> list:
        s = self.floor_size
        xs = np.linspace(-s, s, 5)
        ys = np.linspace(-s, s, 5)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.full_like(xx, self.ground_z)
        surf = ax.plot_surface(
            xx,
            yy,
            zz,
            alpha=0.25,
            color=PALETTE["ground"],
            linewidth=0,
            zorder=0,
        )
        return [surf]

    def _draw_links(
        self,
        ax: "Axes3D",
        X_world: List[SpatialTransform],
    ) -> list:
        artists = []
        for body in self.tree.bodies:
            if body.parent < 0:
                continue
            p_pos = _body_world_pos(X_world[body.parent])
            c_pos = _body_world_pos(X_world[body.index])
            (line,) = ax.plot(
                [p_pos[0], c_pos[0]],
                [p_pos[1], c_pos[1]],
                [p_pos[2], c_pos[2]],
                color=PALETTE["joint"],
                linewidth=2.5,
                zorder=2,
            )
            artists.append(line)
        return artists

    def _draw_bodies(
        self,
        ax: "Axes3D",
        X_world: List[SpatialTransform],
    ) -> list:
        artists = []
        for body in self.tree.bodies:
            pos = _body_world_pos(X_world[body.index])
            color = PALETTE["contact"] if body.name in self.contact_names else PALETTE["body"]
            sc = ax.scatter(
                [pos[0]],
                [pos[1]],
                [pos[2]],
                s=120,
                c=color,
                depthshade=True,
                zorder=3,
            )
            artists.append(sc)
        return artists

    # ------------------------------------------------------------------
    # RenderScene drawing
    # ------------------------------------------------------------------

    def _draw_scene(self, ax: "Axes3D", scene: "RenderScene") -> list:
        """Draw a complete RenderScene with collision shapes and contacts."""
        from .shape_artists import SHAPE_DRAWERS, draw_contacts, draw_terrain

        artists = []

        # Terrain
        artists += draw_terrain(ax, scene.terrain, self.floor_size)

        # Skeleton links
        for p_pos, c_pos in scene.skeleton_links:
            (line,) = ax.plot(
                [p_pos[0], c_pos[0]],
                [p_pos[1], c_pos[1]],
                [p_pos[2], c_pos[2]],
                color=PALETTE["joint"],
                linewidth=1.5,
                alpha=0.5,
                zorder=1,
            )
            artists.append(line)

        # Collision shapes
        for ps in scene.shapes:
            drawer = SHAPE_DRAWERS.get(ps.shape_type)
            if drawer:
                artists += drawer(ax, ps.position, ps.rotation, **ps.params)

        # Contacts
        artists += draw_contacts(ax, scene.contacts)

        return artists

    # ------------------------------------------------------------------
    # Axes configuration
    # ------------------------------------------------------------------

    def _configure_axes(self, ax: "Axes3D") -> None:
        s = self.floor_size
        ax.set_xlim(-s, s)
        ax.set_ylim(-s, s)
        ax.set_zlim(self.ground_z, self.ground_z + 2 * s)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=25, azim=-60)
        ax.grid(True, linestyle="--", alpha=0.4)
