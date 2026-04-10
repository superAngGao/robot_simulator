"""
URDF loader — two-phase parser that translates a URDF file into a RobotModel.

Phase 1: _parse_urdf()  — XML → internal _URDFData dataclasses (no physics).
Phase 2: _build_model() — _URDFData → physics/ instances → RobotModel.

load_urdf() returns a RobotModel (pure robot description).
load_urdf_scene() returns a Scene wrapping the RobotModel with terrain
and collision configuration.

References:
  ROS URDF specification: https://wiki.ros.org/urdf/XML
  Featherstone (2008) §4 — joint models and tree construction.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from physics.geometry import (
    BodyCollisionGeometry,
    BoxShape,
    CylinderShape,
    MeshShape,
    ShapeInstance,
    SphereShape,
)
from physics.joint import FixedJoint, FreeJoint, Joint, PrismaticJoint, RevoluteJoint
from physics.robot_tree import Body, RobotTreeNumpy
from physics.spatial import SpatialInertia, SpatialTransform

from .model import RobotModel

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal dataclasses (Phase 1 output)
# ---------------------------------------------------------------------------


@dataclass
class _LinkData:
    name: str
    mass: float
    inertia_3x3: NDArray[np.float64]  # (3, 3)
    com_xyz: NDArray[np.float64]  # (3,)
    com_rpy: NDArray[np.float64]  # (3,)
    collision_shapes: list[ShapeInstance] = field(default_factory=list)


@dataclass
class _JointData:
    name: str
    jtype: str  # "revolute"|"prismatic"|"fixed"|"floating"|"continuous"
    parent: str
    child: str
    origin_xyz: NDArray[np.float64]  # (3,)
    origin_rpy: NDArray[np.float64]  # (3,)
    axis: NDArray[np.float64]  # (3,) unit vector
    limit_lower: float
    limit_upper: float
    damping: float
    friction: float
    effort: float = 0.0


@dataclass
class _URDFData:
    links: dict[str, _LinkData]
    joints: list[_JointData]
    root_link: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_rpy(elem):
    if elem is not None:
        return np.array([float(x) for x in elem.get("rpy", "0 0 0").split()], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def _parse_xyz(elem):
    if elem is not None:
        return np.array([float(x) for x in elem.get("xyz", "0 0 0").split()], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def _parse_collision_shapes(link_elem) -> list[ShapeInstance]:
    shapes = []
    for col_elem in link_elem.findall("collision"):
        origin = col_elem.find("origin")
        origin_xyz = _parse_xyz(origin)
        origin_rpy = _parse_rpy(origin)
        geom = col_elem.find("geometry")
        if geom is None:
            continue
        shape = None
        box = geom.find("box")
        if box is not None:
            size = np.array([float(x) for x in box.get("size", "0 0 0").split()])
            shape = BoxShape(tuple(size))
        sphere = geom.find("sphere")
        if sphere is not None:
            shape = SphereShape(float(sphere.get("radius", "0")))
        cyl = geom.find("cylinder")
        if cyl is not None:
            shape = CylinderShape(float(cyl.get("radius", "0")), float(cyl.get("length", "0")))
        mesh = geom.find("mesh")
        if mesh is not None:
            filename = mesh.get("filename", "")
            scale_str = mesh.get("scale", "1 1 1")
            scale = tuple(float(x) for x in scale_str.split())
            shape = MeshShape(filename, scale=scale)
        if shape is not None:
            shapes.append(ShapeInstance(shape=shape, origin_xyz=origin_xyz, origin_rpy=origin_rpy))
    return shapes


# ---------------------------------------------------------------------------
# Phase 1 — parse XML
# ---------------------------------------------------------------------------


def _parse_urdf(urdf_path: str) -> _URDFData:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links: dict[str, _LinkData] = {}
    for link_elem in root.findall("link"):
        name = link_elem.get("name")
        inertial = link_elem.find("inertial")
        if inertial is not None:
            mass = float(inertial.find("mass").get("value", "0"))
            origin = inertial.find("origin")
            com_xyz = _parse_xyz(origin)
            com_rpy = _parse_rpy(origin)
            ie = inertial.find("inertia")
            if ie is not None:
                ixx = float(ie.get("ixx", "0"))
                ixy = float(ie.get("ixy", "0"))
                ixz = float(ie.get("ixz", "0"))
                iyy = float(ie.get("iyy", "0"))
                iyz = float(ie.get("iyz", "0"))
                izz = float(ie.get("izz", "0"))
                inertia_3x3 = np.array(
                    [
                        [ixx, ixy, ixz],
                        [ixy, iyy, iyz],
                        [ixz, iyz, izz],
                    ],
                    dtype=np.float64,
                )
            else:
                inertia_3x3 = np.zeros((3, 3), dtype=np.float64)
        else:
            log.warning("Link %r has no <inertial>; using placeholder mass 1e-6 kg.", name)
            mass = 1e-6
            com_xyz = np.zeros(3, dtype=np.float64)
            com_rpy = np.zeros(3, dtype=np.float64)
            inertia_3x3 = np.eye(3, dtype=np.float64) * 1e-9

        collision_shapes = _parse_collision_shapes(link_elem)
        links[name] = _LinkData(name, mass, inertia_3x3, com_xyz, com_rpy, collision_shapes)

    joints: list[_JointData] = []
    child_links: set[str] = set()
    for j_elem in root.findall("joint"):
        jname = j_elem.get("name")
        jtype = j_elem.get("type")
        parent_link = j_elem.find("parent").get("link")
        child_link = j_elem.find("child").get("link")
        origin = j_elem.find("origin")
        origin_xyz = _parse_xyz(origin)
        origin_rpy = _parse_rpy(origin)

        axis_elem = j_elem.find("axis")
        if axis_elem is not None:
            axis_raw = np.array(
                [float(x) for x in axis_elem.get("xyz", "0 0 1").split()],
                dtype=np.float64,
            )
            norm = np.linalg.norm(axis_raw)
            axis = axis_raw / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        else:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        limit_elem = j_elem.find("limit")
        if limit_elem is not None:
            limit_lower = float(limit_elem.get("lower", "-inf"))
            limit_upper = float(limit_elem.get("upper", "inf"))
            effort = float(limit_elem.get("effort", 0.0))
        else:
            limit_lower = -np.inf
            limit_upper = np.inf
            effort = 0.0

        dynamics = j_elem.find("dynamics")
        damping = float(dynamics.get("damping", 0.0)) if dynamics is not None else 0.0
        friction = float(dynamics.get("friction", 0.0)) if dynamics is not None else 0.0

        if jtype == "continuous":
            jtype = "revolute"
            limit_lower = -np.inf
            limit_upper = np.inf

        joints.append(
            _JointData(
                name=jname,
                jtype=jtype,
                parent=parent_link,
                child=child_link,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=axis,
                limit_lower=limit_lower,
                limit_upper=limit_upper,
                damping=damping,
                friction=friction,
                effort=effort,
            )
        )
        child_links.add(child_link)

    root_candidates = [name for name in links if name not in child_links]
    if len(root_candidates) != 1:
        raise ValueError(f"Expected exactly one root link, found: {root_candidates}")

    return _URDFData(links=links, joints=joints, root_link=root_candidates[0])


# ---------------------------------------------------------------------------
# Phase 2 — build RobotModel (pure robot, no contact/collision models)
# ---------------------------------------------------------------------------


def _build_model(
    data: _URDFData,
    floating_base: bool,
    contact_links: Optional[list[str]],
    gravity: float,
    urdf_dir: str = "",
) -> RobotModel:
    # --- 1. BFS topological sort ---
    children_map: dict[str, list[_JointData]] = {name: [] for name in data.links}
    for jd in data.joints:
        if jd.parent in children_map:
            children_map[jd.parent].append(jd)

    bfs_order: list[str] = []
    queue: deque[str] = deque([data.root_link])
    while queue:
        link_name = queue.popleft()
        bfs_order.append(link_name)
        for jd in children_map[link_name]:
            queue.append(jd.child)

    link_to_joint: dict[str, _JointData] = {jd.child: jd for jd in data.joints}

    # --- 2. Build tree ---
    tree = RobotTreeNumpy(gravity=gravity)
    link_to_body_idx: dict[str, int] = {}

    for link_name in bfs_order:
        ld = data.links[link_name]
        is_root = link_name == data.root_link

        if is_root:
            joint: Joint = FreeJoint("root") if floating_base else FixedJoint("root")
            X_tree = SpatialTransform.identity()
            parent_idx = -1
        else:
            jd = link_to_joint[link_name]
            X_tree = SpatialTransform.from_rpy(
                jd.origin_rpy[0],
                jd.origin_rpy[1],
                jd.origin_rpy[2],
                r=jd.origin_xyz,
            )
            parent_idx = link_to_body_idx[jd.parent]
            jtype = jd.jtype
            if jtype in ("revolute",):
                joint = RevoluteJoint(
                    jd.name,
                    axis=jd.axis,
                    q_min=jd.limit_lower,
                    q_max=jd.limit_upper,
                    damping=jd.damping,
                    friction=jd.friction,
                )
            elif jtype == "prismatic":
                joint = PrismaticJoint(jd.name, axis=jd.axis, damping=jd.damping)
            elif jtype == "floating":
                joint = FreeJoint(jd.name)
            else:
                joint = FixedJoint(jd.name)

        inertia = SpatialInertia(ld.mass, ld.inertia_3x3, ld.com_xyz)
        body = Body(
            name=link_name,
            index=0,
            joint=joint,
            inertia=inertia,
            X_tree=X_tree,
            parent=parent_idx,
        )
        idx = tree.add_body(body)
        link_to_body_idx[link_name] = idx

    tree.finalize()

    # --- 3. Collision geometries ---
    geometries: list[BodyCollisionGeometry] = []
    for link_name in bfs_order:
        ld = data.links[link_name]
        if not ld.collision_shapes:
            continue
        resolved_shapes: list[ShapeInstance] = []
        for si in ld.collision_shapes:
            if isinstance(si.shape, MeshShape):
                from .mesh_loader import load_mesh, resolve_mesh_path

                mesh_path = resolve_mesh_path(si.shape.filename, urdf_dir)
                convex = load_mesh(mesh_path, scale=tuple(si.shape.scale))
                resolved_shapes.append(
                    ShapeInstance(
                        shape=convex,
                        origin_xyz=si.origin_xyz,
                        origin_rpy=si.origin_rpy,
                    )
                )
            else:
                resolved_shapes.append(si)
        if resolved_shapes:
            geometries.append(
                BodyCollisionGeometry(
                    body_index=link_to_body_idx[link_name],
                    shapes=resolved_shapes,
                )
            )

    # --- 4. Actuated joints ---
    actuated_joint_names = [
        b.joint.name for b in tree.bodies if b.joint.nv > 0 and not isinstance(b.joint, FreeJoint)
    ]

    # --- 5. Effort limits ---
    joint_effort: dict[str, float] = {jd.name: jd.effort for jd in data.joints}
    efforts = np.array([joint_effort.get(name, 0.0) for name in actuated_joint_names], dtype=np.float64)
    effort_limits = efforts if np.any(efforts > 0) else None

    # --- 6. Contact body names ---
    contact_body_names = []
    if contact_links:
        for ln in contact_links:
            if ln in link_to_body_idx:
                contact_body_names.append(ln)
            else:
                log.warning("contact_links: link %r not found; skipping.", ln)

    return RobotModel(
        tree=tree,
        actuated_joint_names=actuated_joint_names,
        contact_body_names=contact_body_names,
        geometries=geometries,
        effort_limits=effort_limits,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_urdf(
    urdf_path: str,
    floating_base: bool = True,
    contact_links: Optional[list[str]] = None,
    gravity: float = 9.81,
) -> RobotModel:
    """Parse a URDF file and return a RobotModel.

    The RobotModel contains the kinematic tree, collision geometries,
    and joint metadata. Contact/collision models are managed at the
    Scene level — use load_urdf_scene() for a ready-to-simulate Scene.
    """
    data = _parse_urdf(urdf_path)
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    return _build_model(data, floating_base, contact_links, gravity, urdf_dir=urdf_dir)


def load_urdf_scene(
    urdf_path: str,
    floating_base: bool = True,
    contact_links: Optional[list[str]] = None,
    collision_exclude_pairs: Optional[list[tuple[str, str]]] = None,
    gravity: float = 9.81,
    solver_type: str = "pgs",
    **solver_kwargs,
):
    """Parse a URDF and return a ready-to-simulate Scene.

    Convenience function that wraps load_urdf() + Scene.single_robot().
    """
    from scene import Scene

    model = load_urdf(urdf_path, floating_base, contact_links, gravity)
    scene = Scene(
        robots={"main": model},
        solver_type=solver_type,
        solver_kwargs=solver_kwargs,
    ).build()

    # Apply user-specified collision exclusions
    if collision_exclude_pairs:
        tree = model.tree
        for ln_a, ln_b in collision_exclude_pairs:
            idx_a = next((b.index for b in tree.bodies if b.name == ln_a), None)
            idx_b = next((b.index for b in tree.bodies if b.name == ln_b), None)
            if idx_a is None:
                log.warning("collision_exclude_pairs: link %r not found.", ln_a)
                continue
            if idx_b is None:
                log.warning("collision_exclude_pairs: link %r not found.", ln_b)
                continue
            scene.collision_filter.exclude_pair(idx_a, idx_b)

    return scene
