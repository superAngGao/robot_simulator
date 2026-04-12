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
| Multi-physics interface / material architecture | SOFA (taxonomy only, reject machinery) | Drake hydroelastic, Genesis (anti-pattern) |
| Collision geometry abstraction | Drake SceneGraph | Pinocchio GeometryModel |
| Collision algorithms (AABB/OBB/GJK) | hpp-fcl / coal | Bullet dispatcher |
| Contact manifold generation (face clipping) | ODE dBoxBox2 / Jolt ManifoldBetweenTwoFaces | Bullet btBoxBoxDetector / Coal contact_patch |
| Persistent contact manifold (GPU) | PhysX 5 PCM | Bullet btPersistentManifold |
| Friction regularization (PGS) | MuJoCo (R diag) / ODE (slip1/slip2) | Bullet (warmstart=0) / PhysX (angDamp) |
| Ground contact model | MuJoCo | Drake |
| GPU parallel simulation | Newton (ex-warp.sim) | MuJoCo Warp (MJWarp) |
| GPU collision data layout | Newton / MuJoCo Warp | PhysX 5 / Bullet3 GPU |
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
- `ProximityProperties` as a typed property bag per geometry, with groups
  ("material", "hydroelastic", "wildcard"). Each backend declares the keys
  it consumes. Extensible without modifying a core Material class.

**Hydroelastic contact — interaction surface ≠ geometry boundary:**
Drake has two contact models: *point contact* (standard, interaction at shape
boundary) and *hydroelastic contact* (mathematically different). In hydroelastic,
each compliant geom carries a scalar pressure field `e(p)` defined over its
volume. When two bodies overlap, the `ContactSurface` is the equal-pressure
locus `{Q : e_M(Q) = e_N(Q)}` — a surface that lives *inside* both bodies'
volumes, not at any geometry boundary. This is the strongest mathematical
precedent for treating "interaction interface" as a runtime-computed entity
distinct from geometric shape boundaries.
- Files: `drake/multibody/hydroelastics/`, `drake/geometry/query_results/contact_surface.h`
- Paper: Elandt et al., "A pressure field model for fast, robust approximation
  of net contact force and moment between nominally rigid objects" (2019)

**Relevant API docs:**
- Joint damping: `drake::multibody::Joint::default_damping()`
- Shape: `drake::geometry::Shape` and subclasses
- SceneGraph: `drake::geometry::SceneGraph`
- ContactSurface: `drake::geometry::ContactSurface`

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

**Friction regularization (Q25 research):**
- `btSequentialImpulseConstraintSolver`: friction warmstart 每帧归零
  （法向用 `old * 0.85`，摩擦初始化为 0）。阻断跨帧噪声积累。
- `m_frictionCFM` 参数存在但默认 = 0（未启用）。
- `btContactSolverInfo.h` 定义 `m_restitutionVelocityThreshold = 0.2`
  （bounce 死区，非摩擦，但同一设计模式）。
- Sleeping/deactivation：低速体冻结，终极兜底。

---

### ODE (Open Dynamics Engine)
**What it is:** 经典开源物理引擎，CFM/ERP 概念的起源。

**Friction regularization (Q25 research):**
- 全局 CFM 加到所有约束行对角线（含摩擦），默认 `1e-5` (f32) / `1e-10` (f64)。
- **`slip1`/`slip2` 参数**：摩擦行专用 CFM，等价于 `v_slip = k × f_friction`。
  物理含义：表面有力相关的微小滑移。数学效果：
  `lambda_t = -v_t / (W_diag + slip)`，当 `v_t ~ 1e-7` 和 `slip ~ 1e-4` 时
  摩擦冲量被压制到正常值的 ~1e-3。
- 两阶段求解：先法向（假设无摩擦），再摩擦（用固定法向力），解耦。

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

### Newton (successor to warp.sim)
**Repo:** https://github.com/newton-physics/newton
**What it is:** GPU-accelerated physics engine built on NVIDIA Warp. Linux Foundation
project led by Disney Research, Google DeepMind, and NVIDIA. Supersedes `warp.sim`
(removed in Warp 1.10.0). Integrates MuJoCo Warp as its primary rigid body backend.

**Key patterns to study:**
- `Model` + `State` separation (immutable config vs mutable dynamics)
- Shape data layout: `shape_body[]`, `shape_type[]`, `shape_transform[]`,
  `shape_scale[]`, `shape_material[]` — flat parallel arrays, GPU-friendly
- `Contacts` class: pre-allocated parallel arrays (`rigid_contact_shape0/1[]`,
  `rigid_contact_point0/1[]`, `rigid_contact_normal[]`, `rigid_contact_distance[]`,
  `rigid_contact_count`). Only counter reset per frame (1 kernel launch).
- `CollisionPipeline.collide(state, contacts)`: broadphase→narrowphase pipeline
  decoupled from solver.
- Multi-world: `shape_world[]` / `body_world[]` for parallel env isolation.
- Narrowphase dispatch table: shape-type pairs → specialized kernels.

---

### MuJoCo Warp (MJWarp)
**Repo:** https://github.com/google-deepmind/mujoco_warp
**What it is:** GPU-optimized MuJoCo port on NVIDIA Warp. Newest GPU physics
engine (GTC 2025). Same mjModel/mjData semantics, Warp kernel implementation.

**Key patterns to study:**
- Broadphase: SAP (project→sort→binary-search→pair-generate) or N-squared
  with hierarchical filter (plane→sphere→AABB→OBB). Selectable via `m.opt.broadphase`.
- Contact buffer: `(nworld, naconmax)` pre-allocated. `nacon` via `wp.atomic_add()`.
  Overflow → silent discard. EPA workspace pre-allocated to avoid OOM in RL training.
- Multi-world: kernels launch `dim=(nworld, ngeom)`, one thread per (world, geom).
- Narrowphase: `MJ_COLLISION_TABLE` routes 32 type-pairs to PRIMITIVE/CONVEX/SDF/FLEX.
- Geom arrays: same layout as C MuJoCo (`geom_type`, `geom_bodyid`, `geom_pos`, etc.)
  but with `(nworld, ngeom)` batch dimension.

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

---

### SOFA Framework
**Repo:** https://github.com/sofa-framework/sofa
**What it is:** Academic C++ multi-physics framework (INRIA). Used in surgical
simulation, biomechanics, FEM deformables. The strongest open-source precedent
for explicit "interaction interface ≠ body DoF" architecture.

**Why we care:** SOFA's multi-model representation cleanly separates
*mechanical state* (where DoFs live) from *collision model* (where interactions
happen), connected by an explicit *Mapping* layer. This is the architecture
we need for future multi-physics (rigid + soft + fluid), where the
"interaction interface" may be a subset of a surface mesh, a set of particles,
or a runtime-computed region — not just a shape boundary.

**Conceptual taxonomy to borrow (vocabulary only):**
- `MechanicalObject<T>`: stores DoF (position, velocity, mass). Only mechanics.
- `BaseMapping` with `applyJ` (forward) / `applyJT` (transpose): maps between
  representations, e.g. `SubsetMapping`, `BarycentricMapping`, `RigidMapping`.
  **`applyJT` is exactly what our future `CouplingImpulse` needs to do** —
  transpose-map an interface-space impulse back to body DoF.
- `CollisionModel` (`TriangleCollisionModel`, `PointCollisionModel`, etc.):
  independent component with its own `contactStiffness`, `contactFriction`,
  `contactRestitution`, `contactResponse`, `d_contactDistance`. Attached to a
  MechanicalObject via a Mapping.
- `ForceField`: per-body internal forces (gravity, springs, hyperelastic).
- `InteractionForceField`: force fields *between* two objects — standalone
  scene graph node, not owned by either body. Lives at the interface.
- `ConstraintCorrection`: maps Lagrange multipliers back to body DoF.

**Key insight — one body, multiple interface regions:** A single
MechanicalObject can have *multiple* CollisionModels attached via different
Mappings (including `SubsetMapping` to select a DoF subset), each with its
own friction/material. This is how SOFA handles "robot's rubber foot pad
+ metal body" on one mechanical object.

**CRITICAL WARNING — DO NOT copy the execution model:**
SOFA's conceptual taxonomy is right, but its runtime machinery is structurally
incompatible with GPU-batched RL (thousands of parallel envs). Concrete issues:

1. **13-visitor step + virtual dispatch**: `DefaultAnimationLoop::step()` runs
   ~13 visitors per simulation step (`AnimateVisitor`, `CollisionVisitor`,
   `SolveVisitor`, `UpdateMappingVisitor`, etc.). Each visitor walks the
   scene graph and calls virtual methods on every component. `Visitor.h` has
   `isThreadSafe()=false` by default — traversal is fundamentally sequential.
2. **Scatter-add mapping with indirection, not BLAS**: `BarycentricMapper::applyJT`
   is a hand-rolled `out[element[j]] += inPos * baryCoef[j]` loop with
   per-element indirection. *Not* a CSR/BSR sparse matvec — the Jacobian is
   implicit in `(map, elements)`, never assembled. Cannot be fused across envs
   because indirection tables differ per env.
3. **SofaCUDA is component-level, not pipeline-level**: `CudaMechanicalObject`
   launches one CUDA kernel per vector operation (`vAssign`, `vAdd`, `vOp`).
   Scene graph traversal stays on CPU. No batched-env support — one CUDA
   Mapping per body, not across bodies.
4. **Empirical performance**: SofaGym paper (Ménager et al., HAL hal-03778189)
   reports 0.15× realtime for a small soft-gripper environment, single env,
   single process. "Parallelism" = Python `SubprocVecEnv` (OS multiprocess).
   Compare Isaac Gym / MJX: 4096 envs, one GPU, one process, >100k env-steps/sec.
5. **Absent from all major robotics benchmarks** (SimBenchmark, 2024 Review
   of Nine Physics Engines for RL). Not a throughput-RL contender.
6. **SOFA's own maintainers want to escape**: 2023 dev report wiki lists
   "POC for iterators on scene graph to avoid relying on visitors" — the
   visitor pattern is a known liability.

**Our takeaway:**
- **Borrow vocabulary**: `MechanicalState`, `Mapping(applyJ/applyJT)`,
  `ForceField`, `CollisionModel`, `InteractionForceField`, `ConstraintCorrection`.
  These abstraction boundaries are sound and should inform our future naming.
- **Reject execution model entirely**: no scene graph, no visitors, no
  per-component virtual dispatch in the hot loop. Follow Warp/Newton/MJWarp:
  fixed-phase pipeline, each phase = one Warp kernel on flat SoA buffers
  indexed by `env_id`.
- **Mappings as data, not virtual classes**: store as pre-assembled
  `(out_idx, in_idx, weight)` triplet buffers across all envs, dispatched by
  one kernel per mapping type. Compile-time-static, not runtime-dynamic.
- **Preserve multi-representation concept**: mechanical / collision / visual
  as distinct DoF spaces is genuinely good for soft body work. Sync step
  must be one batched kernel, not `UpdateMappingVisitor`.

**Files worth reading (as external reference, not to copy):**
- `Sofa/framework/Simulation/Core/src/sofa/simulation/DefaultAnimationLoop.cpp`
- `Sofa/framework/Simulation/Core/src/sofa/simulation/Visitor.h`
- `Sofa/framework/Core/src/sofa/core/CollisionModel.h`
- `Sofa/framework/Core/src/sofa/core/BaseMapping.h`
- `Sofa/Component/Mapping/Linear/.../BarycentricMapperTopologyContainer.inl`
- `applications/plugins/SofaCUDA/` (as cautionary example)

**Slogan:** SOFA is the right taxonomy, the wrong machinery. Read the
headers, copy the names, rewrite the implementation.

---

### Genesis (Embodied AI Simulator)
**Repo:** https://github.com/Genesis-Embodied-AI/Genesis
**What it is:** Recent multi-physics simulator (2024) built on Taichi,
supporting rigid, MPM, FEM, SPH, PBD, and hybrid bodies. Aimed at embodied
AI / robot learning.

**Why we care (as an anti-pattern):** Genesis targets the same problem
space we do (GPU-accelerated multi-physics for embodied AI), but it made
a specific architectural choice we should *not* repeat: no unified
interface abstraction, cross-physics coupling via hand-coded couplers.

**Architecture:**
- Material per entity type: `genesis/engine/materials/rigid/`, `MPM/`, `FEM/`,
  `SPH/`, `PBD/`, `SF/`. Each subdirectory has its own material class
  hierarchy. `base.py`, `hybrid.py`, `kinematic.py`, `tool.py` at root.
- `Material` is `Generic[EntityT]` — bound to entity type, not to surface
  or interface.
- Cross-physics coupling via explicit coupler classes: `RigidMPMCoupler`,
  `RigidSPHCoupler`, `RigidPBDCoupler`, `MPMSPHCoupler`, etc. Coupling logic
  is hard-coded in each coupler, not derived from a unified interface
  abstraction.

**The N² problem:** with N physics types, Genesis needs O(N²) coupler
classes. With rigid + MPM + FEM + SPH + PBD = 5 types, that's 10 coupler
pairs. Each new physics type requires writing N new couplers. This scales
badly and creates tight coupling between subsystems.

**Validation for our direction:** Genesis's coupler explosion is exactly
what our `PhysicsSubsystem + CouplingImpulse` architecture is designed to
avoid. The plan is: each subsystem publishes `InterfaceRegion`s carrying
`InterfaceMaterial`; the coupling layer reads both sides' interfaces and
computes a generic impulse response. Adding a new subsystem only requires
implementing its `InterfaceRegion.apply_interface_impulse(impulse)`, not
writing N new couplers.

**Takeaway:** Genesis proves both that (a) multi-physics on GPU is feasible
and worth pursuing, and (b) skipping the interface abstraction leads to
unmaintainable O(N²) coupling code. Our design should extract the lesson
without copying the mistake.
