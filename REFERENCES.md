# Robot Simulator — Open-Source Reference Projects

> **How to use this file:**
> Before making any technical design decision (API shape, algorithm choice, data
> structure, abstraction boundary), scan the relevant rows below and check how
> established projects handle the same problem. Record the finding in
> REFLECTIONS.md under the decision.

---

## Quick-reference matrix

| Concern | Primary reference | Secondary |
|---------|-------------------|-----------|
| Rigid body dynamics (ABA / RNEA) | Pinocchio | Drake MultibodyPlant |
| Joint model & passive torques | Drake | MuJoCo |
| Collision geometry abstraction | Drake SceneGraph | Pinocchio GeometryModel |
| Collision algorithms (AABB/OBB/GJK) | hpp-fcl / coal | Bullet dispatcher |
| Ground contact model | MuJoCo | Drake |
| GPU parallel simulation | Isaac Lab | Warp examples |
| RL environment interface | Gymnasium | Isaac Lab |
| Domain randomization | Isaac Lab | MuJoCo randomize |
| URDF / robot description format | Pinocchio urdf_parser | Drake |
| VecEnv & training pipeline | Isaac Lab | Stable Baselines 3 |

---

## Project details

### Pinocchio
**Repo:** https://github.com/stack-of-tasks/pinocchio
**What it is:** The reference open-source implementation of Featherstone ABA/RNEA.
Used in most academic robotics research.

**Key patterns to study:**
- `Model` + `Data` separation: static model parameters vs. per-step computation buffers.
  Our `RobotTree` + future `SimulatorState` should follow this split.
- `GeometryModel` + `GeometryData`: collision geometry stored separately from dynamics.
  Shapes (`hpp-fcl` objects) are independent of the collision algorithm.
- `buildGeom()`: URDF → GeometryModel pipeline.

**Lessons learned (do NOT repeat):**
- Joint damping is stored in `model.damping` but **not applied automatically** in
  `aba()`/`rnea()`. Users must manually subtract `damping * v` from tau.
  This is a known usability defect (pinocchio#1291, pinocchio#2435). Our
  `passive_torques()` must be automatic.

---

### Drake (MultibodyPlant)
**Repo:** https://github.com/RobotLocomotion/drake
**What it is:** Google's robotics toolkit. Best-in-class API design. C++ core,
Python bindings.

**Key patterns to study:**
- `RevoluteJoint(damping=...)`: damping is a joint constructor parameter, applied
  automatically via `AddInDamping()` each step. Our model.
- `SceneGraph` + `Shape` hierarchy (`Box`, `Sphere`, `Cylinder`, `Capsule`,
  `Convex`, `Mesh`): geometry registration is separate from `QueryObject`
  (collision query). Exactly our `geometry.py` + `collision.py` split.
- `MultibodyPlant.Finalize()`: mirrors our `RobotTree.finalize()` pattern.
- Role-based geometry: same shape can have `collision` role, `proximity` role, and
  `illustration` (rendering) role simultaneously.

**Relevant API docs:**
- Joint damping: `drake::multibody::Joint::default_damping()`
- Shape: `drake::geometry::Shape` and subclasses
- SceneGraph: `drake::geometry::SceneGraph`

---

### MuJoCo
**Repo:** https://github.com/google-deepmind/mujoco
**What it is:** Industry-standard simulator for RL. MJCF format. Implicit integrator.

**Key patterns to study:**
- MJCF `<joint damping="..."/>`: damping is part of the joint element, applied
  implicitly in the integrator (solves `(M + dt·D)·v̇ = f` as a coupled system).
  Superior numerical stability for stiff damping vs. explicit subtraction.
- `<geom type="box|sphere|cylinder|capsule|mesh">`: geometry is a `geom` element
  attached to a body. Multiple geoms per body are natural and standard.
- Contact model: `solref` and `solimp` parameters control spring-damper contact.
  More sophisticated than our current penalty model — reference for Phase 2
  contact improvements.
- `<compiler>`: auto-infers inertia from geometry if `<inertial>` is missing.
  Useful UX feature for `load_urdf()` to consider.

---

### hpp-fcl / coal
**Repo:** https://github.com/humanoid-path-planner/hpp-fcl (now: coal)
**What it is:** Collision detection library used by Pinocchio. Implements GJK, EPA,
AABB trees, BVH.

**Key patterns to study:**
- Shape class hierarchy: `Box`, `Sphere`, `Cylinder`, `Capsule`, `Cone`,
  `ConvexBase`, `BVHModel` (triangle mesh). Exact template for our
  `CollisionShape` hierarchy.
- Broadphase (`DynamicAABBTreeCollisionManager`) + narrowphase (GJK/EPA) separation.
  Our `BVHGJKSelfCollision` Phase 2+ implementation should follow this split.
- `CollisionObject`: pairs a shape with a transform. Maps to our `ShapeInstance`.

---

### Bullet / PyBullet
**Repo:** https://github.com/bulletphysics/bullet3
**What it is:** Widely-used game/robotics physics engine.

**Key patterns to study:**
- `btCollisionShape` hierarchy: `btBoxShape`, `btSphereShape`, `btCylinderShape`,
  `btCapsuleShape`, `btCompoundShape` (multiple shapes per body),
  `btBvhTriangleMeshShape`.
- Double-dispatch collision dispatcher: a matrix indexed by `(shapeTypeA,
  shapeTypeB)` selects the narrowphase algorithm. Allows registering custom
  collision algorithms for any shape-pair without touching shape definitions.
- `btCompoundShape`: explicitly models multiple primitives per body as a single
  compound. Our `BodyCollisionGeometry.shapes: list[ShapeInstance]` is the
  data-only equivalent.

**Lessons learned:**
- Joint friction from URDF is loaded but **not applied automatically** — must
  simulate via motor control. Similar to Pinocchio damping issue. We avoid this.

---

### Isaac Lab (formerly Isaac Gym)
**Repo:** https://github.com/isaac-sim/IsaacLab
**What it is:** NVIDIA's GPU-accelerated RL training platform. Primary reference
for Phase 2.

**Key patterns to study:**
- `ArticulationView`: batched access to N robot instances on GPU. Our
  `VecEnv` / Warp backend should mirror this API.
- `EnvCfg` dataclass pattern: environment configuration (robot, rewards,
  observations, terminations) as a typed dataclass, not scattered kwargs.
  Adopt for our `rl_env/base_env.py`.
- Domain randomization API: `EventTermCfg` + `randomize_*` functions.
  Reference for `domain_rand/` module design.
- `ObservationManager` / `RewardManager`: named, composable observation and
  reward terms. Better than a monolithic `_get_obs()` method.

---

### Gymnasium
**Repo:** https://github.com/Farama-Foundation/Gymnasium
**What it is:** Standard RL environment interface. Every RL library expects this.

**Key patterns to study:**
- `Env.step(action) -> (obs, reward, terminated, truncated, info)`: exact
  signature our `base_env.py` must implement.
- `Env.reset(seed, options) -> (obs, info)`: our reset must match.
- `spaces.Box`, `spaces.Dict`: observation and action space definitions.
- `VectorEnv`: standard batched env interface (Phase 4).

---

### Stable Baselines 3
**Repo:** https://github.com/DLR-RM/stable-baselines3
**What it is:** Clean RL algorithm implementations. Defines what Gymnasium
compliance means in practice.

**Key patterns to study:**
- What SB3 actually calls on an env: confirms which Gymnasium methods we must
  implement and which are optional.
- `VecEnv` interface: `step_async` / `step_wait` pattern for parallel envs.
- Used as a smoke-test: if SB3 can train on our env without modification, our
  Gymnasium compliance is correct.
