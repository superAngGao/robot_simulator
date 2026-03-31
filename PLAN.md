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
│  Layer 1: Physics Core  (rigid body dynamics pipeline)    │
│  physics/{robot_tree, step_pipeline, solvers/,           │
│           joint, contact, collision}  —  NumPy + GPU     │
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

#### 2f — High-fidelity contact modeling 🔄

**已完成：**
- [x] **GJK/EPA 凸体碰撞检测**：`physics/gjk_epa.py`，支持 Box/Sphere/Cylinder
  - `support_point()` 方法 added to all convex shapes
  - `gjk_epa_query()` 形状间碰撞 + `ground_contact_query()` 地面碰撞
  - `ContactManifold` 数据结构（body pair、normal、depth、points[]）
- [x] **PGS LCP 约束求解器**：`physics/lcp_solver.py`
  - `ContactConstraint`（接触点 + 法线 + 切线 + 摩擦系数）
  - Signorini 条件（lambda_n ≥ 0）+ Coulomb 摩擦锥投影
  - Baumgarte 稳定化（ERP/CFM）
  - 对角 Delassus 近似（W_ii）
- [x] **关节 Coulomb 摩擦**：`RevoluteJoint.friction` 参数
  - 从 URDF `<dynamics friction="..."/>` 解析（此前被忽略）
  - `compute_friction_torque()` 平滑 tanh 近似
  - 集成到 `passive_torques()`

**待完成（Q18 差距清单）：**
- [ ] `LCPContactModel(ContactModel)` — 将 GJK/EPA + PGS 集成到 Simulator 管线
- [ ] **完整 Delassus 矩阵** `W = J M⁻¹ Jᵀ`（替代当前对角近似）
- [ ] **Warm starting** — 缓存上一步 lambda，帧间持久化
- [ ] **Capsule 形状** + `support_point()`
- [ ] **接触持久化 (manifold cache)**
- [ ] **Broad-phase 空间加速**（Dynamic AABB Tree / 空间哈希）
- [ ] **弹性碰撞 (restitution)**
- [ ] GPU 加速：GJK/EPA + LCP 的 CUDA kernel

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

对于分支结构机器人，H 矩阵有天然块结构——不同子树（limb）之间的耦合为零，
只有 root-limb 耦合存在。利用此结构可以用层次化 Schur complement 替代全矩阵 Cholesky。

**自动分支点检测**：从 `parent_idx[]` 自动找多子节点 body 作为切割点，
每个子树自成一组，无需用户指定。单链机器人（无分支）退化为标准 CRBA。

```
H 的块结构（四足为例）：
H = [H_rr   H_rl₁  H_rl₂  H_rl₃  H_rl₄]     r = root (nv=6)
    [H_l₁r  H_l₁l₁  0      0      0    ]     l₁ = FL leg (nv=3)
    [H_l₂r  0       H_l₂l₂  0      0    ]     l₂ = FR leg (nv=3)
    [H_l₃r  0       0      H_l₃l₃  0    ]     l₃ = RL leg (nv=3)
    [H_l₄r  0       0      0      H_l₄l₄]     l₄ = RR leg (nv=3)

关键性质：H_li_lj = 0 (i≠j) — 非祖先关节无耦合
```

**层次化 Schur complement 求解**：

```
Step 1: 每个 limb 独立 Cholesky — L_i L_iᵀ = H_lili     [K 个并行, O(nv_limb³)]
Step 2: Schur complement — S = H_rr - Σᵢ H_rli H_lili⁻¹ H_lir  [O(K × nv_root × nv_limb²)]
Step 3: 解 root — S @ qddot_r = rhs_r'                    [O(nv_root³), 通常 6×6]
Step 4: 回代每个 limb — qddot_li = H_lili⁻¹(rhs_li - H_lir @ qddot_r)  [K 个并行]
```

总复杂度：O(K × nv_limb³ + nv_root³) vs 全矩阵 O(nv³)。
四足例：4×3³ + 6³ = 108+216 = 324 FLOPs vs 18³ = 5832 FLOPs（**18x 减少**）。

**Tensor core 对齐（未来优化）**：组大小 pad 到 16 对齐 wgmma tile，
或合并小组（如两个 nv=3 的 limb 合并为 nv=6）。当 nv_limb ≥ 16 时
单组 Cholesky 可走 tensor core，此时层次 Schur 的每个 Step 1 并行任务都能用 wgmma。

**实现计划**：

Phase 2g-1: 标准 CRBA（CPU NumPy）✅
- [x] `physics/robot_tree.py` — `RobotTreeNumpy.crba(q) -> H`
- [x] `physics/robot_tree.py` — `RobotTreeNumpy.forward_dynamics_crba(q, qdot, tau, ext) -> qddot`
- [x] 数值验证：CRBA == ABA（atol=1e-10），Pinocchio 对比（atol=1e-8）
- [x] 测试：`tests/test_crba.py`（13 tests）

Phase 2g-2: GPU CRBA（多种实现）✅
- [x] `BatchedCRBA` — PyTorch batched（`torch.linalg.cholesky_solve`）
- [x] `physics_step_crba_kernel` — CUDA fused（标量 Cholesky in-kernel）
- [x] `crba_build_kernel` + cuSOLVER — split path（tensor core Cholesky）
- [x] Benchmark：8 种实现对比（见 PROGRESS.md）

Phase 2g-3: 分组 CRBA + 层次 Schur（⬜ 潜在优化）
- [ ] `auto_detect_groups(bodies)` — 自动分支点检测
- [ ] `grouped_crba(q)` + `forward_dynamics_grouped_crba()` — CPU 参考实现
- [ ] CUDA fused grouped CRBA kernel（Schur complement in-kernel）
- [ ] Benchmark：grouped vs monolithic vs ABA 在 nv=30/46/62, N=1024/4096/8192
- [ ] 当 nv_limb ≥ 16 时启用 tensor core 加速 Step 1（wgmma）

**参考文献**：
- Featherstone (2008) §6 — CRBA 算法
- Featherstone (1999) — Divide-and-Conquer Algorithm for O(n log n) parallel dynamics
- Carpentier et al. (2019) — Pinocchio: `crba()` 实现
- NVIDIA cuSOLVER batched Cholesky — Hopper tensor core 支持

#### 2h — Force system refactor (StepPipeline) 🔄

**动机**：力的来源散落（gravity 隐式、passive 手动加、contact 6 种 solver 格式各异），
聚合点分散，大量重复计算（FK 2-3 次、ABA 2 次），不可观测（无法回答"此步每个 body 受什么力"）。

**架构**：MuJoCo 两阶段管线（mj_step1 smooth + mj_step2 constraint）：
- `StepPipeline`: `ForceSource[]` → `tau_smooth` → `qacc_smooth` → `ConstraintSolver` → `qacc` → integrate
- `DynamicsCache`: FK/body_v/H/L 算一次，全链路复用
- `ForceState`: 每步力分解的可观测性（qfrc_passive, qfrc_actuator, qacc_smooth, qacc）
- 积分内联在管线中（semi-implicit Euler），不是独立抽象 — 不同物理子系统各自积分

**文件**：
- 新建：`dynamics_cache.py`, `force_source.py`, `constraint_solver.py`, `constraint_solvers.py`,
  `dynamics_utils.py`, `step_pipeline.py`
- 改名：`mujoco_qp.py` → `admm_qp.py` (`MuJoCoStyleSolver` → `ADMMQPSolver`)
- 修改：`simulator.py`, `robot_tree.py`, `integrator.py`, `__init__.py`
- 废弃：`implicit_contact_step.py`, `contact.py` (LCP/Null/Penalty ContactModel)

**求解器层级**（session 14 清理后）：
- 公开 API: `PGSSplitImpulseSolver` (CPU RL), `ADMMQPSolver` (CPU precision)
- 内部: `PGSContactSolver` (PGS-SI 委托)
- GPU: Warp kernel 直接实现 (solver_kernels.py + admm_kernels.py)
- 废弃: `LCPContactModel`, `PenaltyContactModel`, `NullContactModel`
- 已删除: `ADMMContactSolver`, `JacobiPGSContactSolver` (死代码，session 14)

#### 2i — GPU solver development ⬜

**两条 GPU 求解器路线**（对应 CPU 的 PGS-SI 和 ADMM-QP）：

| 求解器 | 算法 | GPU 策略 | 适用 |
|--------|------|---------|------|
| **Jacobi-PGS-SI** | Jacobi PGS + split impulse | 全行并行，无数据依赖 | 大规模 RL (N=1000+) |
| **ADMM-TC** | ADMM-QP + tensor core batched Cholesky | A=(M+ρJᵀJ) 预分解，迭代内三角求解 | 高精度 GPU 仿真 |

Jacobi-PGS-SI (GPU fast path): ✅ 已实现
- Warp kernel `solver_kernels.py`，全行并行 + split impulse
- GpuEngine `solver="jacobi_pgs_si"` (默认)

ADMM-TC (GPU precision path): ✅ 已实现
- Warp kernel `admm_kernels.py`，in-kernel scalar Cholesky + 锥投影
- GpuEngine `solver="admm"`
- 基于 `ADMMQPSolver` 的 solref/solimp compliance 模型
- `A = H + ρJᵀJ` 每步分解一次（N 个独立矩阵 batched Cholesky）
- 迭代内只做三角求解 `L⁻ᵀL⁻¹rhs`（O(nv²) per env）
- Phase 2g 已验证 batched Cholesky 在 GPU 上可行
- 固定 rho + warmstart（GPU 更友好，避免条件分支重分解）

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
