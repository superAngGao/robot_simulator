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
| Phase 2 physics | NVIDIA Warp + TileLang + CUDA C++ | 4 GPU backends (NumPy/Warp/TileLang/CUDA), benchmarked |
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

Architecture decisions confirmed in REFLECTIONS.md (2026-03-17), updated 2026-03-23:
- **4 GPU backends**: NumPy (CPU fallback) + Warp + TileLang + raw CUDA C++, all implementing `BatchBackend(ABC)`.
- **Unified data format**: `torch.Tensor` on CUDA throughout VecEnv/managers. GPU backends receive tensors via zero-copy (Warp: `wp.from_torch()`; TileLang: DLPack; CUDA: `data_ptr<float>()`).
- **Backend selection**: `physics/backends/get_backend("numpy"|"warp"|"tilelang"|"cuda")` factory; upper layers are backend-agnostic.
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

#### 2e — GPU backends ✅ DONE

**架构**：`BatchBackend(ABC)` + `StepResult` dataclass。VecEnv 通过 `get_backend(name)` 工厂选择后端，
上层代码完全不感知具体后端。`StaticRobotData` 将 `RobotModel` 展平为连续数组供 GPU 使用。

四个后端已实现并通过数值验证（float32 vs float64，atol=1e-4）：

| 后端 | 实现方式 | N=1000 steps/s | vs NumPy |
|------|----------|----------------|----------|
| NumPy | Python for-loop (CPU) | 533 | 1x |
| TileLang | TileLang kernel (FK+ABA) + PyTorch (contact/integration) | 438,700 | 823x |
| Warp | 7 个 @wp.kernel (FK/ABA/contact/collision/integration) | 750,363 | 1,408x |
| **CUDA** | **单融合 CUDA C++ kernel** (全物理步) | **2,204,524** | **4,136x** |

文件结构：
```
physics/backends/
├── __init__.py              # get_backend() 工厂
├── batch_backend.py         # BatchBackend(ABC), StepResult
├── static_data.py           # StaticRobotData
├── numpy_loop.py            # NumpyLoopBackend
├── warp/                    # Warp: spatial_warp.py, kernels.py, warp_backend.py, scratch.py
├── tilelang/                # TileLang: kernels_tl.py (FK+ABA kernel), tilelang_backend.py
└── cuda/                    # CUDA C++: kernels.cu, cuda_backend.py
```

#### 2f — High-fidelity contact modeling

当前 penalty 单点接触仅适用于快速原型验证。精确仿真需要升级为：

- [ ] **多点接触**：每个接触体定义多个接触点（如足底 4 角点），替代当前单点模型
- [ ] **凸体碰撞检测**：实现 GJK/EPA 算法，支持凸形状（box, sphere, cylinder, convex mesh）之间的穿透深度和接触面片计算
- [ ] **约束求解器（LCP）**：替代 penalty 弹簧阻尼模型，用基于约束的接触求解（Signorini 条件 + Coulomb 摩擦锥），消除参数调优依赖
- [ ] **接触点离散化**：将 GJK/EPA 求得的接触面片离散为多个接触点，输入约束求解器
- [ ] GPU 加速：GJK/EPA + LCP 的 Warp kernel 实现

#### 2g — CRBA + Tensor Core 加速（密集矩阵前向动力学）

**动机**：当前所有 GPU 后端使用 ABA（O(n) 顺序算法），每步仅做 6×6 标量运算，
无法利用 tensor core。对于中大型机器人（nv > 20），CRBA 将前向动力学转化为
**密集矩阵运算**（nv × nv），可以充分发挥 tensor core 的吞吐优势。

**算法：Composite Rigid Body Algorithm (CRBA)**

```
1. 计算关节空间质量矩阵 H (nv × nv)
   - 通过 composite inertia 从叶到根传播
   - H 对称正定，只需计算上三角

2. 计算偏置力 C = RNEA(q, qdot, 0) — 科里奥利力 + 重力

3. 求解前向动力学
   qddot = H^{-1} @ (tau - C + J^T @ f_ext)

   通过 Cholesky 分解求解：
   L @ L^T = H
   qddot = L^{-T} @ L^{-1} @ rhs
```

**GPU 优势（对比 ABA）**：

| 维度 | ABA | CRBA |
|------|-----|------|
| 复杂度 | O(n) | O(n²) + O(nv³) |
| 运算类型 | 6×6 标量 (sequential) | nv×nv 密集矩阵 (parallel) |
| Tensor core | ✗ 无法使用 | ✓ Cholesky / matmul |
| 适用范围 | nv < 20 (四足) | nv > 20 (人形、多指手) |
| GPU 并行度 | 仅跨环境 N | 跨环境 N × 矩阵内并行 |

**分组策略（大型机器人优化）**：

对于 nv > 50 的复杂机器人，将运动学树分解为子树组，每组独立计算局部质量矩阵，
通过 Schur complement 处理组间耦合：

```
完整机器人 (nv=50+)
├── 躯干组 (nv_local=6)      → H_trunk (6×6)
├── 左臂组 (nv_local=7)      → H_LA (7×7)
├── 右臂组 (nv_local=7)      → H_RA (7×7)
├── 左腿组 (nv_local=6)      → H_LL (6×6)
├── 右腿组 (nv_local=6)      → H_RL (6×6)
└── 组间耦合 → 块稀疏 Schur complement
```

每组规模对齐 tensor core tile size（16/32），避免全矩阵 O(n²) 开销。
本质上是 Featherstone Divide-and-Conquer Algorithm 的 GPU-native 变体。

**实现计划**：

Phase 2g-1: 标准 CRBA（CPU NumPy）
- [ ] `physics/robot_tree.py` — `RobotTreeNumpy.crba(q) -> H`（质量矩阵计算）
- [ ] `physics/robot_tree.py` — `RobotTreeNumpy.forward_dynamics_crba(q, qdot, tau, ext) -> qddot`
- [ ] 数值验证：CRBA 输出 vs ABA 输出（相同输入，应完全一致）
- [ ] 测试：`tests/test_crba.py`

Phase 2g-2: Batched Cholesky GPU 加速
- [ ] `physics/backends/` — 在现有后端中添加 CRBA 路径
- [ ] 使用 `torch.linalg.cholesky_solve` batched 版本（cuSOLVER → tensor core on Hopper）
- [ ] Benchmark：ABA vs CRBA 在不同 nv（10/20/30/50）和 N（1/100/1000）下的性能对比
- [ ] 自动选择策略：nv < 阈值用 ABA，nv ≥ 阈值用 CRBA

Phase 2g-3: 分组 CRBA（大型机器人）
- [ ] 子树分组 API：用户指定或自动分割
- [ ] 块对角 H 计算 + 组间 Schur complement
- [ ] TileLang / CUDA kernel 实现组内 matmul（对齐 tensor core tile）

**参考文献**：
- Featherstone (2008) §6 — CRBA 算法
- Featherstone (1999) — Divide-and-Conquer Algorithm for O(n log n) parallel dynamics
- Carpentier et al. (2019) — Pinocchio: `crba()` 实现
- NVIDIA cuSOLVER batched Cholesky — Hopper tensor core 支持

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
  - Ch.6: CRBA (Composite Rigid Body Algorithm)
  - Ch.7: ABA (Articulated Body Algorithm)
- Featherstone, R. — "A Divide-and-Conquer Articulated Body Algorithm" (1999)
- Spatial algebra: http://royfeatherstone.org/spatial/index.html
- Carpentier et al. — "The Pinocchio C++ Library" (2019): `crba()`, `aba()` reference
- Penalty-based contact: Mirtich & Canny (1995)
- Sim-to-Real: OpenAI "Learning Dexterous In-Hand Manipulation" (2019)
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
- NVIDIA cuSOLVER: batched Cholesky on Hopper tensor cores
