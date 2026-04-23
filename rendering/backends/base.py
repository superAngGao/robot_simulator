"""
Abstract base class for rendering backends.

Any backend (matplotlib, Rerun, OpenGL, …) must implement this interface.
Design reference: Drake AbstractValue / LeafSystem output-port pattern —
backends are stateful sinks that consume RenderScene snapshots.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..render_scene import RenderScene


class RenderBackend(ABC):
    """Pluggable rendering backend interface.

    Lifecycle::

        backend.open()
        for frame in frames:
            backend.render_frame(scene, timestamp)
        backend.close()

    ``set_output`` is optional; backends that do not support file output
    silently ignore it.
    """

    @abstractmethod
    def open(self) -> None:
        """Initialise the backend (create window, open file, etc.)."""
        ...

    @abstractmethod
    def render_frame(self, scene: RenderScene, timestamp: float, env_index: int = 0) -> None:
        """Render one frame from *scene* at *timestamp* seconds.

        Args:
            scene      : Backend-agnostic scene snapshot.
            timestamp  : Simulation time in seconds.
            env_index  : Which parallel environment to visualise (default 0).
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Flush and release all resources."""
        ...

    def set_output(self, path: str) -> None:
        """Set file output path (no-op for live-display backends)."""

    @property
    @abstractmethod
    def supports_offscreen(self) -> bool:
        """True if the backend can render without a display."""
        ...
