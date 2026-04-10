"""
Mesh file loading and convex hull conversion.

Loads STL/OBJ/DAE mesh files via trimesh (optional dependency) and
converts them to ConvexHullShape for use in the collision pipeline.

trimesh is imported lazily inside functions so that users who only use
primitive shapes are not forced to install it.
"""

from __future__ import annotations

import os

import numpy as np

from physics.geometry import ConvexHullShape


def resolve_mesh_path(mesh_filename: str, urdf_dir: str) -> str:
    """Resolve a mesh filename relative to the URDF file directory.

    Handles:
    - Absolute paths (returned as-is)
    - Relative paths (resolved relative to urdf_dir)
    - ``package://`` URIs (raises NotImplementedError)
    """
    if mesh_filename.startswith("package://"):
        raise NotImplementedError(
            f"package:// URIs are not yet supported: {mesh_filename!r}. "
            "Use a relative or absolute path instead."
        )
    if os.path.isabs(mesh_filename):
        return mesh_filename
    return os.path.normpath(os.path.join(urdf_dir, mesh_filename))


def load_mesh(
    filepath: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ConvexHullShape:
    """Load a mesh file and return its convex hull as a ConvexHullShape.

    Supports STL, OBJ, DAE, PLY via trimesh.

    Args:
        filepath: Path to mesh file.
        scale: Scale factors (x, y, z) applied to vertices before hull
            computation.

    Returns:
        ConvexHullShape with the convex hull vertices.

    Raises:
        ImportError: If trimesh is not installed.
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If the mesh produces fewer than 4 hull vertices.
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required for mesh loading. Install with: pip install robot_simulator[mesh]"
        ) from None

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Mesh file not found: {filepath!r}")

    mesh = trimesh.load(filepath, force="mesh")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    scale_arr = np.asarray(scale, dtype=np.float64)
    if not np.allclose(scale_arr, 1.0):
        vertices = vertices * scale_arr

    hull = trimesh.convex.convex_hull(trimesh.Trimesh(vertices=vertices))
    hull_vertices = np.asarray(hull.vertices, dtype=np.float64)

    if hull_vertices.shape[0] < 4:
        raise ValueError(
            f"Mesh {filepath!r} has only {hull_vertices.shape[0]} convex hull "
            f"vertices (need >= 4). The mesh may be degenerate."
        )

    return ConvexHullShape(hull_vertices)
