"""
URDF loader — two-phase parser that translates a URDF file into a RobotModel.

Phase 1: _parse_urdf()  — XML → internal _URDFData dataclasses (no physics).
Phase 2: _build_model() — _URDFData → physics/ instances → RobotModel.

References:
  ROS URDF specification: https://wiki.ros.org/urdf/XML
  Featherstone (2008) §4 — joint models and tree construction.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from physics.collision import AABBSelfCollision, NullSelfCollision, SelfCollisionModel
from physics.contact import ContactParams, ContactPoint, NullContactModel, PenaltyContactModel
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
    effort: float = 0.0  # from <limit effort="..."/>, 0 = unspecified


@dataclass
class _URDFData:
    links: dict[str, _LinkData]
    joints: list[_JointData]
    root_link: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_xyz(elem: Optional[ET.Element], attr: str = "xyz") -> NDArray[np.float64]:
    if elem is None:
        return np.zeros(3, dtype=np.float64)
    val = elem.get(attr, "0 0 0")
    return np.array([float(x) for x in val.split()], dtype=np.float64)


def _parse_rpy(elem: Optional[ET.Element]) -> NDArray[np.float64]:
    if elem is None:
        return np.zeros(3, dtype=np.float64)
    val = elem.get("rpy", "0 0 0")
    return np.array([float(x) for x in val.split()], dtype=np.float64)


def _parse_inertia_matrix(inertia_elem: Optional[ET.Element]) -> NDArray[np.float64]:
    """Parse <inertia ixx=... ixy=... .../> into a 3x3 symmetric matrix."""
    if inertia_elem is None:
        return 1e-6 * np.eye(3, dtype=np.float64)
    ixx = float(inertia_elem.get("ixx", 1e-6))
    ixy = float(inertia_elem.get("ixy", 0.0))
    ixz = float(inertia_elem.get("ixz", 0.0))
    iyy = float(inertia_elem.get("iyy", 1e-6))
    iyz = float(inertia_elem.get("iyz", 0.0))
    izz = float(inertia_elem.get("izz", 1e-6))
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)


def _parse_collision_shapes(link_elem: ET.Element) -> list[ShapeInstance]:
    shapes: list[ShapeInstance] = []
    for col in link_elem.findall("collision"):
        origin = col.find("origin")
        xyz = _parse_xyz(origin)
        rpy = _parse_rpy(origin)
        geom = col.find("geometry")
        if geom is None:
            continue
        shape = None
        box = geom.find("box")
        if box is not None:
            size = tuple(float(x) for x in box.get("size", "0.1 0.1 0.1").split())
            shape = BoxShape(size)  # type: ignore[arg-type]
        sphere = geom.find("sphere")
        if sphere is not None:
            shape = SphereShape(float(sphere.get("radius", 0.05)))
        cylinder = geom.find("cylinder")
        if cylinder is not None:
            shape = CylinderShape(
                float(cylinder.get("radius", 0.05)),
                float(cylinder.get("length", 0.1)),
            )
        mesh = geom.find("mesh")
        if mesh is not None:
            filename = mesh.get("filename", "")
            shape = MeshShape(filename)
        if shape is not None:
            shapes.append(ShapeInstance(shape=shape, origin_xyz=xyz, origin_rpy=rpy))
    return shapes


# ---------------------------------------------------------------------------
# Phase 1 — XML parsing
# ---------------------------------------------------------------------------


def _parse_urdf(path: str) -> _URDFData:
    """Parse a URDF file into internal dataclasses. No physics objects created."""
    tree = ET.parse(path)
    root = tree.getroot()

    # --- Links ---
    links: dict[str, _LinkData] = {}
    for link_elem in root.findall("link"):
        name = link_elem.get("name", "")
        inertial = link_elem.find("inertial")
        if inertial is None:
            log.warning("Link %r has no <inertial> element; using placeholder mass 1e-6 kg.", name)
            mass = 1e-6
            inertia_3x3 = 1e-6 * np.eye(3, dtype=np.float64)
            com_xyz = np.zeros(3, dtype=np.float64)
            com_rpy = np.zeros(3, dtype=np.float64)
        else:
            mass_elem = inertial.find("mass")
            mass = float(mass_elem.get("value", 1e-6)) if mass_elem is not None else 1e-6
            origin = inertial.find("origin")
            com_xyz = _parse_xyz(origin)
            com_rpy = _parse_rpy(origin)
            if np.any(np.abs(com_rpy) > 1e-9):
                log.warning(
                    "Link %r has non-zero <inertial><origin rpy>; inertia tensor used as-is (Q11).",
                    name,
                )
            inertia_3x3 = _parse_inertia_matrix(inertial.find("inertia"))

        collision_shapes = _parse_collision_shapes(link_elem)
        links[name] = _LinkData(
            name=name,
            mass=mass,
            inertia_3x3=inertia_3x3,
            com_xyz=com_xyz,
            com_rpy=com_rpy,
            collision_shapes=collision_shapes,
        )

    # --- Joints ---
    joints: list[_JointData] = []
    child_links: set[str] = set()
    for j_elem in root.findall("joint"):
        jname = j_elem.get("name", "")
        jtype = j_elem.get("type", "fixed")
        parent_elem = j_elem.find("parent")
        child_elem = j_elem.find("child")
        parent_link = parent_elem.get("link", "") if parent_elem is not None else ""
        child_link = child_elem.get("link", "") if child_elem is not None else ""

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

        # continuous → treat as revolute with no limits
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

    # Root link = the one that never appears as a child
    root_candidates = [name for name in links if name not in child_links]
    if len(root_candidates) != 1:
        raise ValueError(f"Expected exactly one root link, found: {root_candidates}")

    return _URDFData(links=links, joints=joints, root_link=root_candidates[0])


# ---------------------------------------------------------------------------
# Phase 2 — build physics objects
# ---------------------------------------------------------------------------


def _build_model(
    data: _URDFData,
    floating_base: bool,
    contact_links: Optional[list[str]],
    self_collision_links: Optional[list[str]],
    collision_method: str,
    contact_params: Optional[ContactParams],
    gravity: float,
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

    # Map link name → joint data (for non-root links)
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
                )
            elif jtype == "prismatic":
                joint = PrismaticJoint(jd.name, axis=jd.axis, damping=jd.damping)
            elif jtype == "floating":
                joint = FreeJoint(jd.name)
            else:  # fixed (and any unknown type)
                joint = FixedJoint(jd.name)

        inertia = SpatialInertia(ld.mass, ld.inertia_3x3, ld.com_xyz)
        body = Body(
            name=link_name,
            index=0,  # will be set by add_body
            joint=joint,
            inertia=inertia,
            X_tree=X_tree,
            parent=parent_idx,
        )
        idx = tree.add_body(body)
        link_to_body_idx[link_name] = idx

    tree.finalize()

    # --- 3. Collision geometries (skip MeshShape bodies) ---
    geometries: list[BodyCollisionGeometry] = []
    for link_name in bfs_order:
        ld = data.links[link_name]
        if not ld.collision_shapes:
            continue
        has_mesh = any(isinstance(s.shape, MeshShape) for s in ld.collision_shapes)
        non_mesh = [s for s in ld.collision_shapes if not isinstance(s.shape, MeshShape)]
        if has_mesh and not non_mesh:
            log.warning(
                "Link %r has only MeshShape collision geometry; skipping from collision model (Q7).",
                link_name,
            )
            continue
        if has_mesh:
            log.warning(
                "Link %r has mixed mesh/primitive collision; using only primitive shapes.",
                link_name,
            )
        shapes_to_use = non_mesh if non_mesh else ld.collision_shapes
        geometries.append(
            BodyCollisionGeometry(
                body_index=link_to_body_idx[link_name],
                shapes=shapes_to_use,
            )
        )

    # --- 4. Contact model ---
    params = contact_params or ContactParams()
    if contact_links:
        contact_model = PenaltyContactModel(params=params)
        for link_name in contact_links:
            if link_name not in link_to_body_idx:
                log.warning("contact_links: link %r not found in URDF; skipping.", link_name)
                continue
            cp = ContactPoint(
                body_index=link_to_body_idx[link_name],
                position=np.zeros(3, dtype=np.float64),
                name=link_name,
            )
            contact_model.add_contact_point(cp)
        contact_body_names = [ln for ln in contact_links if ln in link_to_body_idx]
    else:
        contact_model = NullContactModel()
        contact_body_names = []

    # --- 5. Self-collision model ---
    parent_list = [b.parent for b in tree.bodies]

    if collision_method != "aabb":
        raise ValueError(f"Unsupported collision_method {collision_method!r}; only 'aabb' is supported.")

    if self_collision_links is not None:
        sc_geoms = [
            g for g in geometries if any(tree.bodies[g.body_index].name == ln for ln in self_collision_links)
        ]
    else:
        sc_geoms = geometries  # all non-mesh bodies

    if sc_geoms:
        self_collision: SelfCollisionModel = AABBSelfCollision.from_geometries(sc_geoms, parent_list)
    else:
        self_collision = NullSelfCollision()

    # --- 6. Actuated joint names ---
    actuated_joint_names = [
        b.joint.name for b in tree.bodies if b.joint.nv > 0 and not isinstance(b.joint, FreeJoint)
    ]

    # --- 7. Effort limits (None if all zero / unspecified) ---
    joint_effort: dict[str, float] = {jd.name: jd.effort for jd in data.joints}
    efforts = np.array([joint_effort.get(name, 0.0) for name in actuated_joint_names], dtype=np.float64)
    effort_limits = efforts if np.any(efforts > 0) else None

    return RobotModel(
        tree=tree,
        contact_model=contact_model,
        self_collision=self_collision,
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
    self_collision_links: Optional[list[str]] = None,
    collision_method: str = "aabb",
    contact_params: Optional[ContactParams] = None,
    gravity: float = 9.81,
) -> RobotModel:
    """Parse a URDF file and return a fully-constructed RobotModel.

    Args:
        urdf_path            : Path to the .urdf file.
        floating_base        : If True, root link gets a FreeJoint (6 DOF).
                               If False, root link is fixed to the world.
        contact_links        : Link names to register as contact points.
                               None → no contact model (NullContactModel).
        self_collision_links : Link names to include in self-collision.
                               None → all links with non-mesh geometry.
        collision_method     : Broad-phase algorithm. Only "aabb" supported.
        contact_params       : ContactParams for the penalty model.
                               None → default ContactParams().
        gravity              : Gravitational acceleration [m/s²].

    Returns:
        RobotModel with tree, contact_model, self_collision, and metadata.
    """
    data = _parse_urdf(urdf_path)
    return _build_model(
        data,
        floating_base=floating_base,
        contact_links=contact_links,
        self_collision_links=self_collision_links,
        collision_method=collision_method,
        contact_params=contact_params,
        gravity=gravity,
    )
