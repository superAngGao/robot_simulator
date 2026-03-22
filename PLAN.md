# Robot Simulator — Project Plan

> Created: 2026-03-16
> Goal: Build a custom robot simulator with sim-to-real capability for legged robots.

---

## Background & Motivation

- Target robot type: **legged robots** (quadruped / biped)
- Primary goal: **Sim-to-Real transfer** (train in simulation, deploy on real hardware)
- Hardware: High-end NVIDIA GPU available
- Approach: Build from scratch for full customizability and deep understanding

### Why not NVIDIA Isaac Sim?
Isaac Sim is a powerful industrial tool, but we build our own because:
- Full control over physics model and contact dynamics
- Lightweight and embeddable in products
- Deep understanding of underlying algorithms
- Custom domain randomization strategies
- Research platform, not a production dependency

---

## Architecture Overview

### Layered Architecture (5 layers)

```
┌──────────────────────────────────────────────────────────┐
│  Layer 4: Application                                    │
│  rl_env/vec_env.py  —  parallel VecEnv for RL training   │
├──────────────────────────────────────────────────────────┤
│  Layer 3: Task / Environment                             │
│  rl_env/base_env.py  —  Gymnasium interface              │
│  domain_rand/        —  physics & visual randomization   │
├──────────────────────────────────────────────────────────┤
│  Layer 2: Simulator  (single-env step, auto-manages      │
│  passive forces; wraps Layer 1 + contact + self-collision)│
├──────────────────────────────────────────────────────────┤
│  Layer 1: Physics Core  (backend-agnostic algorithms)    │
│  physics/{joint, robot_tree, contact, self_collision,    │
│           integrator}  —  NumPy now, Warp in Phase 2     │
├──────────────────────────────────────────────────────────┤
│  Layer 0: Math Primitives                                │
│  physics/spatial.py  —  pure spatial algebra, no physics │
└──────────────────────────────────────────────────────────┘

Robot Description (orthogonal config axis):
  URDF / programmatic builder
       → robot/urdf_loader.py → RobotModel
             (bundles RobotTree + ContactModel + AABBSelfCollision)
       → feeds into Layer 2 Simulator
```

### External-Facing APIs (two primary entry points)

```
1. load_urdf("robot.urdf", ...)  →  RobotModel     # bring-your-own robot
2. Env(model, ...)               →  Gymnasium env  # RL training interface
```

Everything below these two interfaces is implementation detail.

### Module Map

```
robot_simulator/
├── simulator.py           # Layer 2: single-env step, auto passive forces
│                          #   (orchestrates physics + contact + integrator)
│
├── physics/               # Layer 0 + Layer 1
│   ├── spatial.py         # Layer 0: 6D spatial algebra (Plücker)
│   ├── joint.py           # Layer 1: joint kinematics + passive torques
│   │                      #   (RevoluteJoint: arbitrary axis + damping)
│   ├── _robot_tree_base.py  # Layer 1: RobotTreeBase(ABC) — shared interface
│   ├── robot_tree.py      # Layer 1: RobotTreeNumpy — FK, RNEA, ABA (NumPy)
│   ├── geometry.py        # Layer 1: CollisionShape + BodyCollisionGeometry
│   ├── terrain.py         # Layer 1: Terrain(ABC) + FlatTerrain + HeightmapTerrain
│   ├── contact.py         # Layer 1: ContactModel(ABC) + PenaltyContactModel
│   │                      #           + NullContactModel + TerrainPenaltyContactModel
│   ├── collision.py       # Layer 1: SelfCollisionModel(ABC) + AABBSelfCollision
│   │                      #   (replaces self_collision.py in Phase 2)
│   ├── integrator.py      # Layer 1: Semi-implicit Euler / RK4
│   └── backends/           # Layer 1 (Phase 2): GPU backends
│       ├── __init__.py     #   get_backend("warp"|"tilelang") factory
│       ├── warp/           #   NVIDIA Warp backend
│       │   ├── robot_tree_warp.py   # RobotTreeWarp(RobotTreeBase)
│       │   ├── contact_warp.py      # Warp contact kernels
│       │   └── collision_warp.py    # Warp self-collision kernels
│       └── tilelang/       #   TileLang backend
│           ├── robot_tree_tl.py     # RobotTreeTileLang(RobotTreeBase)
│           ├── contact_tl.py        # TileLang contact kernels
│           └── collision_tl.py      # TileLang self-collision kernels
│
├── robot/                 # Robot Description axis
│   ├── model.py           # RobotModel dataclass
│   └── urdf_loader.py     # URDF → RobotModel (two-phase: parse + build)
│
├── rendering/
│   ├── viewer.py          # matplotlib debug viewer (Phase 1)
│   ├── camera_sim.py      # camera noise model (Phase 3)
│   └── lidar_sim.py       # LiDAR point cloud (Phase 3)
│
├── domain_rand/           # Layer 3: physics & visual randomization
│   ├── physics_rand.py
│   ├── visual_rand.py
│   └── noise_models.py
│
├── rl_env/                # Layer 3 + Layer 4
│   ├── cfg.py             # ObsTermCfg, NoiseCfg, EnvCfg dataclasses
│   ├── obs_terms.py       # standard obs term functions (base_lin_vel, joint_pos, ...)
│   ├── reward_terms.py    # standard reward term functions (Phase 2+)
│   ├── managers.py        # TermManager(ABC) + ObsManager + RewardManager(stub)
│   │                      #   + TerminationManager(stub)
│   ├── base_env.py        # Layer 3: Gymnasium-compatible single env
│   └── vec_env.py         # Layer 4: parallel VecEnv (torch.Tensor on CUDA, backend-agnostic)
│
├── deploy/                # Phase 5 (deferred)
│   ├── policy_export.py
│   └── hardware_bridge.py
│
├── examples/
└── tests/
```

---

## Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| Phase 1 physics | Python + NumPy | Validate correctness first |
| Phase 2 physics | NVIDIA Warp + TileLang | Dual GPU backends, benchmarked against each other |
| Tensor format | PyTorch (torch.Tensor) | Unified data format; zero-copy interop with backends via DLPack |
| Rendering (early) | matplotlib 3D | Quick visualization |
| Rendering (later) | Vulkan + ray tracing | Sim-to-Real visual fidelity |
| RL training | PyTorch + RL Games / SB3 | |
| Real robot interface | ROS2 / vendor SDK | |

---

## Development Phases

### Phase 1 — Basic Physics + Simple Rendering ✅ DONE
Validate dynamics correctness for a single legged robot.

Key algorithms:
- **Spatial algebra** — 6D force/velocity vectors, Plücker coordinate transforms
- **Featherstone ABA** — Articulated Body Algorithm for forward dynamics  O(n)
- **Penalty method contact** — Spring-damper foot-ground contact
- **Semi-implicit Euler** — Stable integrator for contact-rich simulation
- **matplotlib 3D viewer** — Visualize robot skeleton and motion

Deliverables:
- [x] `physics/spatial.py` — Spatial algebra utilities
- [x] `physics/joint.py` — Revolute, Prismatic, Fixed, Free joint models
- [x] `physics/robot_tree.py` — Kinematic tree, FK, RNEA, ABA
- [x] `physics/contact.py` — Spring-damper contact model
- [x] `physics/integrator.py` — Semi-implicit Euler + RK4 + simulate()
- [x] `rendering/viewer.py` — Simple 3D visualization + animation export
- [x] `examples/simple_quadruped.py` — Quadruped drop-test validation

### Phase 2 — GPU Acceleration + Parallel Environments

Architecture decisions confirmed in REFLECTIONS.md (2026-03-17), updated 2026-03-22:
- **Dual GPU backends**: Warp + TileLang, both implementing `RobotTreeBase(ABC)`; NumPy baseline kept for correctness checks.
- **Unified data format**: `torch.Tensor` on CUDA throughout VecEnv/managers. GPU backends receive tensors via zero-copy (Warp: `wp.from_torch()`; TileLang: DLPack).
- **Backend selection**: `physics/backends/get_backend("warp"|"tilelang")` factory; upper layers are backend-agnostic.
- **VecEnv parallelism**: GPU kernels batched over N envs (`dim=N`), not Python-level for-loop.
- **Obs/Action space**: Manager + term-function pattern (Isaac Lab style); `ObsManager` fully implemented, Reward/Termination as stubs.
- **Simulator placement**: Top-level `simulator.py` (not inside `physics/`).

#### 2a — Layer 1 refactoring (prerequisite for robot/ and GPU)

- [ ] `physics/_robot_tree_base.py` — `RobotTreeBase(ABC)` defining `aba/fk/passive_torques` interface
- [ ] `physics/robot_tree.py` — rename existing class to `RobotTreeNumpy(RobotTreeBase)`
- [ ] `physics/joint.py` — `RevoluteJoint`: arbitrary rotation axis (3-vector); add `damping` param
- [ ] `physics/robot_tree.py` — replace `joint_limit_torques()` with `passive_torques()` (limits + damping unified)
- [ ] `physics/geometry.py` — `CollisionShape(ABC)` + `BoxShape / SphereShape / CylinderShape / MeshShape` + `BodyCollisionGeometry`
- [ ] `physics/terrain.py` — `Terrain(ABC)` + `FlatTerrain` + `HeightmapTerrain`
- [ ] `physics/contact.py` — `ContactModel(ABC)` + rename existing → `PenaltyContactModel` + `NullContactModel`; replace `ground_z` with `terrain: Terrain`
- [ ] `physics/collision.py` — `SelfCollisionModel(ABC)` + `AABBSelfCollision.from_geometries()` + `NullSelfCollision`; retire `self_collision.py`

#### 2b — Robot description layer

- [ ] `robot/model.py` — `RobotModel` dataclass (`tree`, `contact_model`, `self_collision`, `actuated_joint_names`, `contact_body_names`)
- [ ] `robot/urdf_loader.py` — two-phase design: `_parse_urdf() → _URDFData` then `_build_model() → RobotModel`

  `load_urdf` final signature:
  ```python
  def load_urdf(
      urdf_path: str,
      floating_base: bool = True,
      contact_links: list[str] | None = None,
      self_collision_links: list[str] | None = None,
      collision_method: str = "aabb",
      contact_params: ContactParams | None = None,
      gravity: float = 9.81,
  ) -> RobotModel:
  ```

#### 2c — Simulator (Layer 2)

- [ ] `simulator.py` — `Simulator(model, integrator, dt)`: auto-calls `passive_torques()`, contact, self-collision, integrator each step

#### 2d — RL environment (Layer 3/4)

- [ ] `rl_env/cfg.py` — `ObsTermCfg`, `NoiseCfg` (Gaussian + Uniform), `EnvCfg` (with top-level `device`)
- [ ] `rl_env/obs_terms.py` — standard term functions: `base_lin_vel`, `base_ang_vel`, `joint_pos`, `joint_vel`, `contact_mask`, …
- [ ] `rl_env/managers.py` — `TermManager(ABC)`, `ObsManager` (full), `RewardManager` (stub), `TerminationManager` (stub); `train()` / `eval()` noise switch
- [ ] `rl_env/base_env.py` — `Env(model, cfg)`, Gymnasium interface
- [ ] `rl_env/vec_env.py` — `VecEnv`: holds Warp arrays directly, no Python env-loop

#### 2e — GPU backends (Warp + TileLang)

**统一数据格式**：`torch.Tensor` on CUDA。VecEnv、ObsManager、RewardManager 全部持有 PyTorch tensor。
GPU 后端通过零拷贝接收 tensor（Warp: `wp.from_torch()`; TileLang: DLPack），避免 host-device 拷贝。

**后端接口**：`physics/backends/__init__.py` 提供 `get_backend(name)` 工厂函数，返回对应的
`RobotTree` 子类 + `ContactModel` + `CollisionModel`。上层代码（Simulator、VecEnv）不感知具体后端。

Warp 后端：
- [ ] `physics/backends/warp/robot_tree_warp.py` — `RobotTreeWarp(RobotTreeBase)`: batched ABA + FK (Warp kernel, `dim=N`)
- [ ] `physics/backends/warp/contact_warp.py` — Warp contact kernel
- [ ] `physics/backends/warp/collision_warp.py` — Warp self-collision kernel

TileLang 后端：
- [ ] `physics/backends/tilelang/robot_tree_tl.py` — `RobotTreeTileLang(RobotTreeBase)`: batched ABA + FK
- [ ] `physics/backends/tilelang/contact_tl.py` — TileLang contact kernel
- [ ] `physics/backends/tilelang/collision_tl.py` — TileLang self-collision kernel

共用：
- [ ] `physics/backends/__init__.py` — `get_backend()` 工厂函数 + 后端注册
- [ ] Numerical validation: Warp / TileLang output vs NumPy baseline (same input, tolerance check)
- [ ] Benchmark: steps/s throughput, NumPy vs Warp vs TileLang (1 / 100 / 1000 envs)

#### 2f — High-fidelity contact modeling

当前 penalty 单点接触仅适用于快速原型验证。精确仿真需要升级为：

- [ ] **多点接触**：每个接触体定义多个接触点（如足底 4 角点），替代当前单点模型
- [ ] **凸体碰撞检测**：实现 GJK/EPA 算法，支持凸形状（box, sphere, cylinder, convex mesh）之间的穿透深度和接触面片计算
- [ ] **约束求解器（LCP）**：替代 penalty 弹簧阻尼模型，用基于约束的接触求解（Signorini 条件 + Coulomb 摩擦锥），消除参数调优依赖
- [ ] **接触点离散化**：将 GJK/EPA 求得的接触面片离散为多个接触点，输入约束求解器
- [ ] GPU 加速：GJK/EPA + LCP 的 Warp kernel 实现

### Phase 3 — High-Fidelity Rendering + Sensor Simulation
- Vulkan renderer with ray tracing
- Realistic camera simulation (noise, distortion, motion blur)
- LiDAR point cloud simulation
- IMU noise models

### Phase 4 — Domain Randomization
- Physics parameter randomization (mass, friction, damping, joint stiffness)
- Visual randomization (textures, lighting, object placement)
- Structured randomization schedules (curriculum)

### Phase 5 — Sim-to-Real Transfer Validation
- Deploy trained policy to real hardware
- Measure sim-to-real gap
- Iterative refinement of simulation parameters (system identification)

---

## Key References

- Featherstone, R. — *Rigid Body Dynamics Algorithms* (2008)
- Spatial algebra: http://royfeatherstone.org/spatial/index.html
- Penalty-based contact: Mirtich & Canny (1995)
- Sim-to-Real: OpenAI "Learning Dexterous In-Hand Manipulation" (2019)
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
