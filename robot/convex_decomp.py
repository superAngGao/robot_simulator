"""
Convex decomposition pipeline.

Converts a triangle mesh (trimesh.Trimesh) into a list of ConvexHullShape
objects suitable for collision detection.

Two backends, tried in order:
  1. coacd  — CoACD (Approximate Convex Decomposition), pip install coacd
  2. vhacdx — V-HACD via vhacdx, pip install vhacdx
  3. single — Falls back to a single convex hull of the whole mesh.

The single-hull fallback is always available (requires only trimesh).
Multi-piece decomposition requires at least one optional backend.

Usage:
    from robot.convex_decomp import decompose_mesh, decompose_file

    # From a trimesh.Trimesh already in memory:
    hulls = decompose_mesh(mesh)                    # list[ConvexHullShape]

    # From a file path:
    hulls = decompose_file("link.stl", scale=0.001) # mm → m

References:
    CoACD: Wei et al. (2022) "Approximate Convex Decomposition for 3D Meshes
        with Probably Correct Decomposition Guarantee", SIGGRAPH 2022.
    V-HACD: Mamou & Ghazali (2009), widely used in Bullet/PhysX/MuJoCo.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physics.geometry import ConvexHullShape

__all__ = ["decompose_mesh", "decompose_file", "DecompBackend"]


class DecompBackend:
    COACD = "coacd"
    VHACD = "vhacd"
    SINGLE = "single"


def _try_coacd(mesh) -> list[NDArray] | None:
    try:
        import coacd
    except ImportError:
        return None
    import trimesh

    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(coacd_mesh)
    hulls = []
    for verts, faces in parts:
        h = trimesh.Trimesh(vertices=verts, faces=faces).convex_hull
        hulls.append(np.asarray(h.vertices, dtype=np.float64))
    return hulls if hulls else None


def _try_vhacd(mesh) -> list[NDArray] | None:
    try:
        from vhacdx import compute_vhacd
    except ImportError:
        return None
    import trimesh

    parts = compute_vhacd(mesh.vertices, mesh.faces)
    hulls = []
    for verts, faces in parts:
        h = trimesh.Trimesh(vertices=verts, faces=faces).convex_hull
        hulls.append(np.asarray(h.vertices, dtype=np.float64))
    return hulls if hulls else None


def _single_hull(mesh) -> list[NDArray]:
    hull = mesh.convex_hull
    return [np.asarray(hull.vertices, dtype=np.float64)]


def decompose_mesh(
    mesh,
    backend: str | None = None,
    min_vertices: int = 4,
) -> list[ConvexHullShape]:
    """Decompose a trimesh.Trimesh into convex pieces.

    Args:
        mesh: trimesh.Trimesh instance.
        backend: Force a specific backend ('coacd', 'vhacd', 'single').
                 None = auto (coacd → vhacd → single).
        min_vertices: Discard pieces with fewer hull vertices than this.

    Returns:
        list[ConvexHullShape], at least one element.
    """
    if backend == DecompBackend.SINGLE:
        vert_lists = _single_hull(mesh)
    elif backend == DecompBackend.COACD:
        vert_lists = _try_coacd(mesh)
        if vert_lists is None:
            raise ImportError("coacd is not installed. pip install coacd")
    elif backend == DecompBackend.VHACD:
        vert_lists = _try_vhacd(mesh)
        if vert_lists is None:
            raise ImportError("vhacdx is not installed. pip install vhacdx")
    else:
        # Auto: try multi-piece backends first, fall back to single hull
        vert_lists = _try_coacd(mesh) or _try_vhacd(mesh) or _single_hull(mesh)

    shapes = []
    for verts in vert_lists:
        if len(verts) < min_vertices:
            continue
        shapes.append(ConvexHullShape(verts))

    if not shapes:
        # Guarantee at least one piece
        shapes = [ConvexHullShape(v) for v in _single_hull(mesh)]

    return shapes


def decompose_file(
    filepath: str,
    scale: float | tuple[float, float, float] = 1.0,
    backend: str | None = None,
) -> list[ConvexHullShape]:
    """Load a mesh file and decompose it into convex pieces.

    Args:
        filepath: Path to STL/OBJ/DAE/PLY file.
        scale: Uniform scale factor or (sx, sy, sz) tuple.
        backend: See decompose_mesh.

    Returns:
        list[ConvexHullShape].

    Raises:
        ImportError: If trimesh is not installed.
        FileNotFoundError: If filepath does not exist.
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required. Install with: pip install robot_simulator[mesh]") from None

    import os

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Mesh file not found: {filepath!r}")

    mesh = trimesh.load(filepath, force="mesh")

    scale_arr = np.asarray(scale, dtype=np.float64)
    if not np.allclose(scale_arr, 1.0):
        mesh.vertices = mesh.vertices * scale_arr

    return decompose_mesh(mesh, backend=backend)
