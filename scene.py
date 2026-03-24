"""
Scene — simulation context containing robots, static geometry, and terrain.

A Scene is the top-level container passed to the Simulator. It holds:
  - One or more named robots (RobotModel instances)
  - Static collision geometry (walls, obstacles — no mass, no dynamics)
  - Terrain (ground plane or heightmap)
  - A unified CollisionFilter for all bodies

The BodyRegistry assigns globally unique indices to all bodies (from all
robots + static geometries) so the CollisionPipeline and contact solver
can work with a flat body list.

References:
  Isaac Lab InteractiveScene: dict[str, Articulation] + terrain
  Drake SceneGraph: unified geometry management
  MuJoCo mjModel: single body list including worldbody
"""

from __future__ import annotations

from dataclasses import dataclass, field

from physics.collision_filter import CollisionFilter
from physics.geometry import CollisionShape
from physics.spatial import SpatialTransform
from physics.terrain import FlatTerrain, Terrain
from robot.model import RobotModel


@dataclass
class StaticGeometry:
    """Fixed collision body — has shape and pose but no mass or dynamics.

    Used for walls, obstacles, ramps, and any geometry that does not move.
    Equivalent to PhysX PxRigidStatic or Bullet mass=0 btRigidBody.

    Attributes:
        name     : Human-readable label (e.g. "north_wall").
        shape    : Collision shape (Box, Sphere, Cylinder, Capsule).
        pose     : Fixed world-frame transform.
        mu       : Sliding friction coefficient.
        condim   : Contact dimension (1/3/4/6).
        mu_spin  : Torsional friction coefficient.
        mu_roll  : Rolling friction coefficient.
    """

    name: str
    shape: CollisionShape
    pose: SpatialTransform
    mu: float = 0.5
    condim: int = 3
    mu_spin: float = 0.0
    mu_roll: float = 0.0


class BodyRegistry:
    """Maps global body indices to (robot_name, local_body_idx) or static geometry.

    Layout:
        [0 .. n_robot_a-1] = robot_a bodies
        [n_robot_a .. n_robot_a+n_robot_b-1] = robot_b bodies
        ...
        [sum(n_robots) .. sum(n_robots)+n_static-1] = static geometries

    Static geometries have infinite mass (inv_mass=0, inv_inertia=zeros).
    """

    def __init__(self, robots: dict[str, RobotModel], n_static: int = 0) -> None:
        self.robot_names: list[str] = []
        self.robot_offset: dict[str, int] = {}
        self.robot_num_bodies: dict[str, int] = {}

        offset = 0
        for name, model in robots.items():
            self.robot_names.append(name)
            self.robot_offset[name] = offset
            nb = model.tree.num_bodies
            self.robot_num_bodies[name] = nb
            offset += nb

        self.static_offset = offset
        self.n_static = n_static
        self.total_bodies = offset + n_static

    def global_id(self, robot_name: str, local_idx: int) -> int:
        """Convert (robot_name, local_body_idx) to global index."""
        return self.robot_offset[robot_name] + local_idx

    def static_global_id(self, static_idx: int) -> int:
        """Convert static geometry index to global index."""
        return self.static_offset + static_idx

    def to_local(self, global_id: int) -> tuple[str, int] | tuple[None, int]:
        """Convert global index to (robot_name, local_idx) or (None, static_idx)."""
        if global_id >= self.static_offset:
            return None, global_id - self.static_offset
        for name in reversed(self.robot_names):
            off = self.robot_offset[name]
            if global_id >= off:
                return name, global_id - off
        return None, global_id

    def is_static(self, global_id: int) -> bool:
        return global_id >= self.static_offset


@dataclass
class Scene:
    """Simulation context: robots + environment + collision rules.

    Attributes:
        robots             : Named robots. Single robot: {"main": model}.
        static_geometries  : Walls, obstacles (fixed, no dynamics).
        terrain            : Ground surface for terrain contact queries.
        collision_filter    : Unified filter (auto-built if None).
        solver_type        : Default contact solver type ("pgs", "jacobi", "admm").
        solver_kwargs      : Forwarded to the solver constructor.
    """

    robots: dict[str, RobotModel]
    static_geometries: list[StaticGeometry] = field(default_factory=list)
    terrain: Terrain = field(default_factory=lambda: FlatTerrain(0.0))
    collision_filter: CollisionFilter | None = None
    solver_type: str = "pgs"
    solver_kwargs: dict = field(default_factory=dict)

    # Populated by build()
    _registry: BodyRegistry | None = field(default=None, repr=False)

    def build(self) -> "Scene":
        """Finalize the scene: build BodyRegistry and CollisionFilter.

        Must be called after all robots and static geometries are added.
        Returns self for chaining.
        """
        self._registry = BodyRegistry(self.robots, n_static=len(self.static_geometries))

        if self.collision_filter is None:
            self.collision_filter = CollisionFilter(self._registry.total_bodies)

        # Auto-exclude parent-child pairs within each robot
        for name, model in self.robots.items():
            offset = self._registry.robot_offset[name]
            parent_list = [b.parent for b in model.tree.bodies]
            for child_idx, parent_idx in enumerate(parent_list):
                if parent_idx >= 0:
                    self.collision_filter.exclude_pair(offset + child_idx, offset + parent_idx)

        return self

    @property
    def registry(self) -> BodyRegistry:
        if self._registry is None:
            raise RuntimeError("Call scene.build() before accessing registry.")
        return self._registry

    @classmethod
    def single_robot(
        cls,
        model: RobotModel,
        terrain: Terrain | None = None,
        static_geometries: list[StaticGeometry] | None = None,
        **solver_kwargs,
    ) -> "Scene":
        """Convenience: build a Scene with one robot named 'main'."""
        scene = cls(
            robots={"main": model},
            static_geometries=static_geometries or [],
            terrain=terrain or FlatTerrain(0.0),
            solver_kwargs=solver_kwargs,
        )
        return scene.build()

    def __repr__(self) -> str:
        n_robots = len(self.robots)
        n_static = len(self.static_geometries)
        n_bodies = self._registry.total_bodies if self._registry else "?"
        return f"Scene(robots={n_robots}, static={n_static}, total_bodies={n_bodies})"
