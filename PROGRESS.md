# Robot Simulator — Progress Tracker

> Last updated: 2026-04-05 (session 17)
> Reference plan: [PLAN.md](./PLAN.md)

---

## Overall Status

| Phase | 状态 | 完成度 |
|-------|------|--------|
| Phase 1 — Basic Physics + Simple Rendering | ✅ 完成（含修复） | 100% |
| Phase 2 — GPU Acceleration + Parallel Envs | ✅ 完成 | 100% (Q28+Q29 resolved) |
| Phase 3 — High-Fidelity Rendering          | ⬜ 未开始 | 0% |
| Phase 4 — Domain Randomization             | ⬜ 未开始 | 0% |
| Phase 5 — Sim-to-Real Validation           | ⬜ 未开始 | 0% |

---

## Phase 1 — Basic Physics + Simple Rendering ✅

### 核心交付物

| 文件 | 状态 | 说明 |
|------|------|------|
| `physics/spatial.py` | ✅ | 空间代数（6D 向量、Plücker 变换、SpatialInertia） |
| `physics/joint.py` | ✅ | Revolute / Prismatic / Fixed / Free 关节模型；含关节限位 |
| `physics/robot_tree.py` | ✅ | 运动学树 + FK + RNEA + ABA；含 `body_velocities()` 公开方法 |
| `physics/contact.py` | ✅ | Penalty spring-damper 地面接触模型 |
| `physics/integrator.py` | ✅ | 半隐式欧拉 + RK4 + `simulate()` |
| `physics/self_collision.py` | ✅ | AABB 自碰撞检测（**新增**，2026-03-16） |
| `rendering/viewer.py` | ✅ | matplotlib 3D 可视化 + GIF/MP4 导出 |
| `examples/simple_quadruped.py` | ✅ | 四足机器人 drop-test 验证（含完整修复） |

### 实现的算法

- **Featherstone ABA**：O(n) 前向动力学，已验证与解析解吻合（自由落体误差 < 5 mm @ t=1s）
- **RNEA**：逆动力学（关节力矩计算）
- **Penalty contact**：地面 spring-damper + 正则化 Coulomb 摩擦
- **Semi-implicit Euler**：接触丰富场景下的稳定积分器（dt = 2e-4 s）
- **RevoluteJoint 限位**：penalty spring-damper，单向阻尼，参数化（q_min/q_max/k/b）
- **AABB 自碰撞**：OBB→世界AABB 投影 + MTV penalty force，自动排除相邻体对

### Phase 1 期间修复的 Bug（按时间顺序）

| 日期 | Bug | 修复方式 |
|------|-----|---------|
| 2026-03-16 | ABA 重力方向反转 | root 加速度初始化改为 `−a_gravity` |
| 2026-03-16 | 接触点世界位置错误（转置 R） | 改为 `R @ pos + r` |
| 2026-03-16 | 接触力坐标系错误 | 改为 `X.inverse().apply_force()` |
| 2026-03-16 | 接触在首次触地时发散 | 接触点临时移至 calf 原点（zero moment arm） |
| 2026-03-16 | `standing_state` 站立高度错误 | 改用 FK 测量最低足端 z |
| 2026-03-16 | 初始关节角度设置用了 `v_idx` | 改为 `q_idx` |
| 2026-03-16 | **接触点偏离足尖 0.2 m** | 新增独立 foot body + FixedJoint（几何精确） |
| 2026-03-16 | **关节角度无限制** | RevoluteJoint 添加 q_min/q_max + penalty 限位力矩 |
| 2026-03-16 | **腿部可穿透躯干** | 新建 AABB 自碰撞模块 |

### 验证结果

| 测试 | 结果 |
|------|------|
| 自由落体精度（解析 vs 仿真，t=1s） | 误差 < 5 mm ✅ |
| 四足 drop-test：四脚着地时间 | ~0.08 s ✅ |
| 站立稳定性（2 s，10 000 步） | 无发散 ✅ |
| 足端几何精度（foot_z vs calf_z） | 差 0.2000 m ✅ |
| 限位有效性（超限时产生恢复力矩） | 验证通过 ✅ |
| 0.5 s 仿真后 max\|q\| | 0.059 rad（远低于所有限位） ✅ |

### 已知剩余限制（Phase 2 解决）

- ~~`_compute_body_velocities()` 每步重复前向递推，与 ABA 内部冗余，应缓存~~ ✅ Q2 已修复（`body_velocities()` 公开方法）
- matplotlib 渲染速度慢，无法实时；Phase 3 替换为 Vulkan
- AABB 以 body origin 为中心，CoM 偏心时精度下降（Q3，Phase 2 升级）
- 无传感器模型（IMU / 力传感器）

---

## Phase 2 — GPU Acceleration + Parallel Environments 🔄

> 目标：将 Phase 1 的 NumPy 后端移植到 NVIDIA Warp（GPU 原生 Python），实现
> 1000+ 并行环境供 RL 训练。架构决策已在 REFLECTIONS.md (2026-03-17) 确认。

### 2a — Layer 1 重构（GPU 和 robot/ 的前提）✅

- [x] `physics/_robot_tree_base.py` — `RobotTreeBase(ABC)` 接口
- [x] `physics/robot_tree.py` — 重命名为 `RobotTreeNumpy(RobotTreeBase)`
- [x] `physics/joint.py` — `RevoluteJoint` 支持任意旋转轴（3-vector）+ `damping` 参数
- [x] `physics/robot_tree.py` — `joint_limit_torques()` → `passive_torques()`（统一限位+阻尼）
- [x] `physics/geometry.py` — `CollisionShape(ABC)` + `BoxShape / SphereShape / CylinderShape / MeshShape` + `BodyCollisionGeometry`
- [x] `physics/terrain.py` — `Terrain(ABC)` + `FlatTerrain` + `HeightmapTerrain`
- [x] `physics/contact.py` — `ContactModel(ABC)` + 现有逻辑改名 `PenaltyContactModel` + `NullContactModel`；`ground_z` → `terrain: Terrain`
- [x] `physics/collision.py` — `SelfCollisionModel(ABC)` + `AABBSelfCollision.from_geometries()` + `NullSelfCollision`；退役 `self_collision.py`

### 2b — Robot description layer ✅

- [x] `robot/model.py` — `RobotModel` dataclass
- [x] `robot/urdf_loader.py` — 两阶段：`_parse_urdf()` + `_build_model()`；`load_urdf()` 完整实现
- [x] `tests/test_urdf_loader.py` — 6 个单元测试（floating/fixed base、revolute、任意轴、contact links、缺 inertial）

### 2c — Simulator (Layer 2) ✅

- [x] `simulator.py` — `Simulator(model, integrator)`：自动调用 `passive_torques()`、contact、self-collision、integrator
- [x] `examples/simple_quadruped.py` — 改用 `Simulator`，删除手动步骤循环和 `joint_limit_torques()` 调用
- [x] `tests/test_simulator.py` — 4 个单元测试（valid state、passive torques、manual loop 对比、swap integrator）

### 测试补全 第一轮 ✅（2026-03-19 session 2）

新增 52 个测试（总计 68 个），覆盖所有 physics/ 核心模块：

| 测试文件 | 测试数 | 覆盖内容 |
|----------|--------|----------|
| `test_contact.py` | 9 | 法向力方向/大小/线性比例、摩擦、阻尼、active_contacts |
| `test_joint_limits.py` | 14 | 限位内零力矩、超限恢复力矩符号/大小、单向阻尼、粘性阻尼、passive_torques 聚合 |
| `test_aba_vs_pinocchio.py` | 5 | ABA 与 Pinocchio 对比（单摆×3、双摆×2，atol=1e-8） |
| `test_self_collision.py` | 13 | 分离无力、相邻体排除、力方向/大小、Newton 第三定律、阻尼、build_pairs、旋转 AABB、NullSelfCollision |
| `test_integrator.py` | 11 | 无效 dt、输出形状、自由落体精度（解析对比）、RK4 精度优于半隐式、能量守恒、NaN 检测、simulate()、四元数归一化 |

**同期修复的两个 Bug（Pinocchio 对比发现）：**

| Bug | 修复 |
|-----|------|
| `SpatialTransform` 使用 Plücker 约定（r 在子坐标系），与 SE3 语义不符 | 统一改为 SE3 约定：`apply_velocity=[R.T@ω; R.T@(v+ω×r)]`，`apply_force=[R@τ+r×(R@f); R@f]`，`compose: r=r1+R1@r2` |
| ABA Pass 3 根节点重力未变换到 body frame | 改为 `a_p = Xup_i.apply_velocity(-a_gravity)` |

两个修复对现有测试向后兼容（所有已有 X_tree 均为 R=I，两种约定等价）。

### 测试补全 第二轮 ✅（2026-03-20 session 4）

新增 98 个测试（总计 166 个），覆盖 Layer 0 基础层和此前零覆盖的模块：

| 测试文件 | 测试数 | 覆盖内容 |
|----------|--------|----------|
| `test_spatial.py` | 39 | skew/rot_x/y/z/quat 转换（含 Pinocchio 对比）、SpatialTransform apply_velocity/apply_force/compose/inverse（含 Pinocchio SE3 对比）、matrix()一致性、SpatialInertia matrix/add（含 Pinocchio Inertia 对比）、spatial cross 恒等式、**非零旋转 X_tree 的 ABA 对比**（验证 matrix() 修复） |
| `test_joints.py` | 31 | PrismaticJoint（transform/S/damping/任意轴）、FixedJoint（0-DOF/offset）、FreeJoint（transform/integrate_q/norm 保持）、RevoluteJoint 任意轴 |
| `test_robot_tree.py` | 13 | FK（零位/单关节/链式组合）、RNEA（零重力/静平衡/ABA roundtrip/Pinocchio 对比/非零旋转 X_tree）、防御性检查（未 finalize/重复 add/body_by_name） |
| `test_terrain.py` | 6 | FlatTerrain height_at/normal_at、HeightmapTerrain NotImplementedError |
| `test_geometry.py` | 9 | Box/Sphere/Cylinder/Mesh half_extents、BodyCollisionGeometry 多 shape max/空 shape |

**同期修复的两个 Bug：**

| Bug | 修复 |
|-----|------|
| `spatial.py:matrix()` 用 R 而非 R.T，与 SE3 约定不一致 | 改为 `E = R.T`，使 `matrix() @ v == apply_velocity(v)` 和 `matrix().T @ f == apply_force(f)` 成立。修复前 ABA 惯量传递在 X_tree 有非零旋转时会算错（潜在 bug，此前所有 X_tree 均 R=I 因此未暴露） |
| `robot_tree.py:rnea()` 根节点重力符号错误 | `a_gravity` → `-a_gravity`（与 ABA 一致）。RNEA 输出一直是错的但此前无测试覆盖 |

**总测试数：166（全部通过，不含 rl_env 的 6 个需 gymnasium 依赖）**

### Q15 空间向量约定统一 + rl_env 测试补全 ✅（2026-03-21 session 5）

**Q15 — 空间向量顺序约定统一为 `[linear; angular]`**

将全部 6D 空间向量从 Featherstone 约定 `[angular(3); linear(3)]` 改为 Pinocchio / Isaac Lab 约定 `[linear(3); angular(3)]`。改动覆盖：

| 文件 | 改动 |
|------|------|
| `physics/spatial.py` | matrix(), apply_velocity(), apply_force(), SpatialInertia.matrix(), spatial_cross_velocity(), gravity_spatial() |
| `physics/joint.py` | RevoluteJoint._S, PrismaticJoint._S, FreeJoint.integrate_q() |
| `physics/contact.py` | ContactPoint.world_velocity(), compute_forces() 空间力构造 |
| `physics/collision.py` | 线性速度提取, 空间力构造 |
| `rl_env/obs_terms.py` | base_lin_vel, base_ang_vel 索引 |
| 所有测试文件 | 移除 `_P6` 置换矩阵，直接与 Pinocchio 对比；更新所有空间向量索引 |

**rl_env 测试补全（+20 个新测试）**

| 覆盖内容 | 测试数 |
|----------|--------|
| TorqueController（pass-through + effort clip） | 2 |
| PDController（zero state、damping） | 2 |
| obs_terms 各 term 函数（shape、dtype、零值） | 9 |
| Env action_clip / episode truncation / init_noise / reset step count | 4 |
| VecEnv reset shape | 1 |
| ObsManager + velocity terms / uniform noise | 2 |

**总测试数：192（全部通过）**

### 2d — RL environment (Layer 3/4) ✅

- [x] `robot/model.py` — 新增 `effort_limits: NDArray | None` 字段
- [x] `robot/urdf_loader.py` — 解析 `<limit effort="..."/>`，按 actuated_joint_names 顺序组装 `(nu,)` 数组
- [x] `rl_env/cfg.py` — `NoiseCfg`（Gaussian + Uniform）、`ObsTermCfg`、`EnvCfg`（含 kp/kd/action_scale/action_clip/init_noise_scale）
- [x] `rl_env/controllers.py` — `Controller(ABC)`、`PDController`（effort clip）、`TorqueController`
- [x] `rl_env/obs_terms.py` — 6 个标准 term 函数（base_lin_vel、base_ang_vel、base_orientation、joint_pos、joint_vel、contact_mask）
- [x] `rl_env/managers.py` — `TermManager(ABC)`、`ObsManager`（完整，train/eval 噪声开关）、`RewardManager`（stub）、`TerminationManager`（stub）
- [x] `rl_env/base_env.py` — `Env(gym.Env)`：预计算静态索引、`_update_cache()`、Gymnasium reset/step API
- [x] `rl_env/vec_env.py` — `VecEnv`：Python for-loop，Phase 2e 换 Warp kernel 时接口不变
- [x] `rl_env/__init__.py` — 导出 Env、VecEnv、EnvCfg、ObsTermCfg、NoiseCfg、Controller、PDController、TorqueController
- [x] `tests/test_rl_env.py` — 6 个测试（obs shape、train噪声、eval无噪声、step有限值、VecEnv shape、effort clip）

**总测试数：74（全部通过）**

### 2e — GPU backend ✅

- [x] `physics/backends/batch_backend.py` — `BatchBackend(ABC)` + `StepResult` dataclass
- [x] `physics/backends/static_data.py` — `StaticRobotData.from_model()` 将 RobotModel 展平为连续数组
- [x] `physics/backends/numpy_loop.py` — `NumpyLoopBackend(BatchBackend)` CPU for-loop 后备
- [x] `physics/backends/warp/spatial_warp.py` — 全部空间代数 `@wp.func` 设备函数（rodrigues、transform、cross 等）
- [x] `physics/backends/warp/kernels.py` — 7 个 `@wp.kernel`：FK、passive_torques、PD controller、contact、collision、ABA、integrate
- [x] `physics/backends/warp/warp_backend.py` — `WarpBatchBackend(BatchBackend)` 编排所有 kernel
- [x] `physics/backends/warp/scratch.py` — `ABABatchScratch` 预分配 GPU 缓冲区
- [x] `rl_env/vec_env.py` — 重构为使用 `BatchBackend`（默认 `backend="numpy"`，向后兼容）
- [x] `rl_env/vec_env.py` — `BatchedObsManager` 用于 `(N, obs_dim)` 批量观测
- [x] `physics/backends/tilelang/tilelang_backend.py` — `TileLangBatchBackend(BatchBackend)` PyTorch CUDA 张量操作
- [x] `physics/backends/tilelang/kernels_tl.py` — 空间代数辅助函数（PyTorch 批量化）
- [x] 数值验证：Warp/TileLang (float32) vs NumPy (float64)，单步 atol=1e-4，50 步 atol=5e-3
- [x] Benchmark（三后端对比）：

| Backend | N=1 | N=10 | N=100 | N=1000 |
|---------|-----|------|-------|--------|
| NumPy | 536 steps/s | 549 | 553 | 563 |
| TileLang (H200, TL kernel) | 507 | 4,964 | 47,991 | 438,700 |
| Warp (H200) | 1,908 | 18,734 | 156,827 | 750,363 |
| CUDA C++ (H200) | 6,114 | 59,721 | 446,915 | 2,204,524 |

| vs NumPy | N=1 | N=10 | N=100 | N=1000 |
|----------|-----|------|-------|--------|
| TileLang | 1.0x | 9.6x | 93x | **823x** |
| Warp | 3.7x | 34x | 295x | **1,408x** |
| **CUDA** | **12x** | **113x** | **840x** | **4,136x** |

CUDA 性能最优原因：全物理步融合为单 kernel launch，零 inter-kernel overhead。

**总测试数：251（全部通过）**

### 2f — High-fidelity contact modeling 🔄

**已完成：**

- [x] **GJK/EPA 凸体碰撞检测** — `physics/gjk_epa.py`
  - `support_point()` for Box/Sphere/Cylinder/Capsule
  - `gjk()` 交叉测试 + `epa()` 穿透深度/法线
  - `gjk_epa_query()` 形状间 + `ground_contact_query()` 地面
  - 23 tests（含旋转、边界情况、Capsule-Sphere）

- [x] **PGS LCP 约束求解器** — `physics/lcp_solver.py`
  - 完整 Delassus 矩阵 `W = J M⁻¹ Jᵀ`（非对角近似）
  - 接触 Jacobian `_compute_contact_jacobian_row()`
  - Warm starting（body-local 坐标匹配，2cm 阈值）
  - Signorini (λₙ ≥ 0) + Coulomb 摩擦锥投影
  - Baumgarte 稳定化 (ERP/CFM) + Newton 弹性碰撞 (restitution)
  - 11 tests（含 body-body 碰撞、PGS 收敛性）

- [x] **LCPContactModel** — `physics/contact.py`
  - `ContactModel` ABC 的约束求解实现，可替代 `PenaltyContactModel`
  - 集成 GJK/EPA + PGS，`add_contact_body(idx, shape, name)` API
  - 12 tests

- [x] **CapsuleShape** — `physics/geometry.py`
  - 球-线段 Minkowski sum，`support_point()` 支持 GJK
  - 6 tests

- [x] **关节 Coulomb 摩擦** — `physics/joint.py`
  - `RevoluteJoint.friction` 参数（从 URDF `<dynamics friction>` 解析）
  - `compute_friction_torque()` tanh 平滑近似
  - 集成到 `passive_torques()`
  - 6 tests + URDF 端到端验证

- [x] **Broad-phase AABB Tree** — `physics/broad_phase.py`
  - `BruteForceBroadPhase` O(n²) + `AABBTreeBroadPhase` O(n log n)
  - 顶层按最长轴分割，递归树-树查询
  - 13 tests

- [x] **集成测试** — `tests/test_contact_integration.py`
  - GJK→LCP 完整管线、多 body、Penalty vs LCP 对比、URDF friction 端到端
  - 5 tests

**Phase 2f 测试（窄相+求解器+形状+摩擦+broad-phase+集成）：64 个**

- [x] **LCPContactModel 接入 Simulator.step()** — `physics/contact.py` + `simulator.py`
  - `ContactModel.compute_forces()` ABC 新增 `dt`、`tree` 可选参数
  - LCPContactModel 从 tree 提取真实质量/惯量（含平行轴定理）
  - Simulator 自动传 `dt` 和 `tree` 给接触模型
  - `load_urdf()` 新增 `contact_method="lcp"` 选项
  - 12 tests（单步/多步/两体/URDF LCP/Penalty vs LCP）

- [x] **碰撞过滤掩码** — `physics/collision_filter.py`（新建）
  - `CollisionFilter`：auto-exclude（parent-child）+ bitmask（group/mask uint32）+ explicit exclude set
  - `should_collide(i, j)` 查询，三种机制取交集
  - 集成到 `AABBSelfCollision.build_pairs()` 和 `LCPContactModel`
  - `RobotModel.collision_filter` 字段
  - `load_urdf()` 新增 `collision_exclude_pairs` 参数
  - 21 tests（standalone + AABB 集成 + load_urdf 集成）

- [x] **condim 1/3/4/6 接触维度** — `physics/solvers/pgs_solver.py`
  - ContactConstraint 新增 condim/mu_spin/mu_roll 字段
  - PGS 改为 variable-width rows + per-condim 锥投影
  - 新增 `_compute_angular_jacobian_row()` 用于 spin/rolling（纯角速度）
  - LCPContactModel 支持 per-body condim 覆盖
  - 17 tests

- [x] **Jacobi PGS 求解器** — `physics/solvers/jacobi_pgs.py`（新建）
  - GPU-friendly 并行变体：全行读旧 buffer、写新 buffer（double buffer）
  - relaxation factor ω，与 serial PGS 收敛到同一解
  - 10 tests（含 PGS 一致性验证）

- [x] **ADMM 求解器** — `physics/solvers/admm.py`（新建）
  - 隐式接触兼容：Step 1 线性系统 Cholesky 预分解 + Step 2 锥投影 + Step 3 对偶更新
  - 圆锥摩擦投影（几何精确 Coulomb 锥，vs PGS 的 box clamp 近似）
  - 12 tests（含锥投影单元测试 + PGS 方向一致性）

- [x] **Reference 测试体系**
  - `test_solver_reference.py`：21 tests — 解析 LCP（手算 Delassus + 互补条件），3 求解器 × 7 场景
  - `test_trajectory_vs_bullet.py`：7 tests — PyBullet PGS 多步轨迹对比（L2 < 0.5mm）
  - `test_complex_scenarios.py`：7 tests — 斜抛球撞粗糙竖直墙 vs PyBullet（600 步）

**Phase 2f 测试：164 个，全部通过**

**待完成（Q18 剩余项）：**

- [ ] 同 body 多 geom 过滤
- [ ] GPU 接触求解器 kernel（Jacobi PGS / ADMM → Warp/CUDA）
- [ ] 通用接触管线（静态环境几何 + body-body LCP 接入 Simulator）

### 2g — CRBA + Tensor Core 加速 ⬜

**目标**：为中大型机器人（nv > 20）实现 CRBA 前向动力学，利用 tensor core 的密集矩阵运算优势。

**已完成（2g-1 + 2g-2）：**

- [x] `RobotTreeNumpy.crba(q) → H` — 质量矩阵（composite inertia，Pinocchio 对比验证 atol=1e-8）
- [x] `RobotTreeNumpy.forward_dynamics_crba()` — H⁻¹(τ-C) via Cholesky（CRBA == ABA, atol=1e-10）
- [x] `BatchedCRBA` — PyTorch GPU 批量 CRBA（`torch.linalg.cholesky_solve`）
- [x] `physics_step_crba_kernel` — CUDA fused CRBA（标量 Cholesky 在 kernel 内）
- [x] `crba_build_kernel` + `integrate_kernel` — split path for cuSOLVER tensor core Cholesky
- [x] 16 个 CRBA 测试 + Pinocchio 对比

**9 个前向动力学实现：**

| # | 方法 | 选择 | nv=30 N=8192 |
|---|------|------|-------------|
| 1 | NumPy ABA | `tree.aba()` | ~530/s |
| 2 | NumPy CRBA | `tree.forward_dynamics_crba()` | — |
| 2b | **NumPy Grouped CRBA** | `tree.forward_dynamics_grouped_crba()` | — |
| 3 | Warp ABA | `backend="warp"` | 750K/s |
| 4 | TileLang ABA | `backend="tilelang"` | 879K/s |
| 5 | **CUDA ABA fused** | `backend="cuda"` | **1,657K/s** |
| 6 | CUDA CRBA-scalar fused | `backend="cuda_crba"` | 1,583K/s |
| 7 | CUDA CRBA-TC (split+cuSOLVER) | `backend="cuda_crba_tc"` | 710K/s |
| 8 | BatchedCRBA (PyTorch) | `BatchedCRBA` module | ~130K/s |

**关键发现：**
- nv ≤ 64 时 **fused scalar Cholesky (#6) 接近 ABA (#5)**（0.96x @ nv=30）
- cuSOLVER tensor core (#7) 因 3 次 kernel launch + H 矩阵 global memory 访问反而更慢
- wgmma 的 M=64 最小维度对 nv < 64 的 H 矩阵不友好，需要 pad 浪费计算
- Tensor core 真正受益需要 nv ≥ 128 或分组策略（Phase 2g-3）

**2g-3 分组 CRBA — ✅ CPU 参考实现 (CUDA kernel 留作后续优化)**

- [x] `auto_detect_groups()` — 自动分支点检测（多子节点 body 为切割点）
- [x] `forward_dynamics_grouped_crba()` — 层次化 Schur complement（limb 并行 Cholesky → root Schur → 回代）
- [x] 验证：grouped == monolithic CRBA == ABA（四足/人形/链式，atol=1e-10）
- [ ] CUDA fused grouped CRBA kernel — 留作 nv > 50 分支机器人优化时实现

**Phase 2g 测试数：26（CRBA + grouped Schur）**

---

### Q23 修复 + GPU 多体测试 ✅（2026-03-27 session 11）

**Bug 修复：**

| Bug | 修复 |
|-----|------|
| `solver_kernels_v2.py` J_body_j 缺少取反 | 6 处赋值加 `-` 号，匹配 CPU PGS 的 `J=[J_i, -J_j]` |
| `static_data.py` body_collision_radius 硬编码 0.05 | 从 `collision_shapes[].half_extents_approx()` 读取实际值 |
| `cpu_engine.py` body-body 碰撞半径用不存在的 `half_extents` | 改用 `half_extents_approx()` 方法 |

**新增测试（`test_gpu_multibody.py`，6 个）：**
- CPU vs GPU 自由落体对比（两独立球）
- 第二根体角速度不发散（500 步稳定性）
- 相向运动两球碰撞反弹
- body-body 碰撞 500 步 NaN 检查
- CPU vs GPU body-body 一致性
- 两球地面着陆稳定

### 2j — GPU 解析碰撞 + MuJoCo 精度对标 ✅（2026-03-27 session 11 后半）

**GPU 解析碰撞检测：**

- [x] `analytical_collision.py` — 14 个 `@wp.func`：
  - Ground: sphere, capsule, box, cylinder vs 平面
  - Body-body: sphere-sphere, sphere-capsule, capsule-capsule, sphere-box
  - Helpers: segment-segment 最近点、box support point
- [x] `collision_kernels.py` — `batched_detect_analytical` kernel，shape type dispatch
- [x] `static_data.py` — `body_shape_type (int32[nb])` + `body_shape_params (float32[nb,4])` GPU 数组
- [x] `gpu_engine.py` — 切换到解析碰撞 kernel
- [x] `cpu_engine.py` — 地面碰撞改用 GJK/EPA（z 精度从 63mm → 3.8mm）
- [x] 20 个 GPU 解析碰撞对比测试

**MuJoCo 精度对标（两球两墙场景）：**

| 求解器 | vs MuJoCo x L2 | vs MuJoCo z L2 | 最终 Δz |
|--------|----------------|----------------|---------|
| PGS-SI (radius=0) | 18.5 mm | 62.9 mm | 212 mm |
| PGS-SI (GJK/EPA) | 18.5 mm | 3.8 mm | 0.8 mm |
| **ADMMQPSolver** | **0.8 mm** | **0.3 mm** | **0.4 mm** |

- [x] 16 个 MuJoCo 对比测试（11 PGS-SI + 5 ADMM 亚毫米精度验证）

### 2k — GPU ADMM 求解器 + solver dispatch ✅（2026-03-30 session 12）

- [x] `admm_kernels.py` — velocity-level ADMM Warp kernel
  - in-kernel scalar Cholesky（n_rows × n_rows，Phase 2g 证明最优）
  - MuJoCo solimp/solref compliance 模型
  - 锥投影（normal ≥ 0, ||tangent|| ≤ μ·normal）
  - 跨步 warmstart（f, s, u 持久化）
- [x] `solver_kernels_v2.py` — `batched_compute_v_current` kernel（v_c = J @ qdot）
- [x] `solver_scratch.py` — ADMM scratch 数组（AR_rho, L, R_diag, f/s/u, warmstart）
- [x] `gpu_engine.py` — `solver` 参数 dispatch（`"jacobi_pgs_si"` | `"admm"`）
- [x] 精确 rhs_const = dt·a_ref(v_c) - (v_free - v_c)（需要分离的 v_c 和 v_free）
- [x] 32 个测试（行为 + 组件验证 + MuJoCo 对标）

**MuJoCo 对标（单球落地, dt=2e-4, 5000 步）：**

| 指标 | GPU ADMM | CPU ADMM | MuJoCo |
|------|---------|---------|--------|
| z-L2 轨迹误差 | **0.0003 mm** | 0.0000 mm | — |
| 稳态位置误差 | -0.3671 mm | -0.3672 mm | -0.3672 mm |
| 最大穿透 | 14.63 mm | ~14.6 mm | 14.63 mm |

**关键 bug 修复历程**：
1. rhs_const = dt·(-b·v_free + k·d·depth) → 稳态差 3.85mm（`v_c ≈ v_free` 近似在平衡态失效）
2. rhs_const = -v_free + dt·(...) → 动态太硬，44x 过强（`-v_free` 主导了 compliance）
3. rhs_const = dt·a_ref(v_c) - (v_free - v_c) → **精确匹配**（需要独立的 v_c kernel）

**Phase 2 总测试数：644（全部通过）**

**Phase 2 实现总览（2026-03-30 session 13 更新）：**

| 类别 | 实现数 | 说明 |
|------|--------|------|
| 前向动力学 | 10 | ABA×5后端 + **CRBA+Cholesky GPU**(Q29) + CRBA×3 + Grouped Schur + CUDA CRBA-TC |
| 接触求解器 | **7** | PGS + PGS-SI + Jacobi PGS + ADMM-C + ADMM + ADMMQPSolver + GPU ADMM |
| 接触维度 | condim 1/3/4/6 | MuJoCo 风格，variable rows + per-condim 锥投影 |
| GPU Delassus | **joint-space** | W = J H⁻¹ Jᵀ via CRBA+Cholesky（替代 body-level） |
| GPU 碰撞 | 解析 + 球近似 | 4 shape × ground + 4 body-body 解析 @wp.func |
| CPU 碰撞 | GJK/EPA + 球近似 | ground: GJK/EPA，body-body: 球近似 |
| 碰撞形状 | 5 | Box + Sphere + Cylinder + Capsule + Mesh(stub) |
| 碰撞过滤 | 1 | CollisionFilter（bitmask + explicit exclude + auto parent-child） |
| 场景管理 | 1 | Scene + BodyRegistry + StaticGeometry + 多机器人 |
| GPU 后端 | 1 | GpuEngine (Warp)；NumPy/TileLang/CUDA 已删除 (Q31) |
| **验证测试** | **32** | CPU vs MuJoCo (单/双摆+四足+两四足碰撞) + GPU vs CPU (四足) |

### Session 13 — Q28/Q29 修复 + 端到端验证 (2026-03-30)

**Q28 修复**：`batched_impulse_to_gen_v2` 力矩双重计算（Plücker + 手动 cross）。
改用纯旋转。同时修复 Q25（摩擦力矩 2x → 1x）。

**Q29 修复**：GPU pipeline 从 ABA+body-level Delassus 全面切换到 CRBA+Cholesky+joint-space Delassus。
  - 新 kernel: `crba_kernels.py`（CRBA+RNEA+Cholesky、contact Jacobian、W build、impulse apply）
  - 一次 Cholesky(H) 三重复用：smooth dynamics + Delassus + impulse
  - FK 从 3 次降到 1 次，ABA 完全消除
  - 四足 GPU z=0.4198 vs CPU z=0.4197（0.1mm，之前 120mm）

**测试重构**：49 个测试文件迁移到 5 层目录（unit/integration/gpu/reference/validation）。

**验证测试**（32 个新测试）：
  - CPU 纯动力学 vs MuJoCo：单摆/双摆/四足自由落体（13 tests，关节角 atol < 1e-3）
  - CPU 接触 vs MuJoCo：四足落地 ADMM/PGS（12 tests，稳态差 0.86mm）
  - 两四足碰撞 vs MuJoCo：26 bodies, 45 collision pairs（7 tests）
  - GPU vs CPU 四足：自由落体 + 接触 + 10k 步稳定性（11 tests）

**Q30 分析**：CPU vs MuJoCo 0.86mm 差异来自 compliance R 公式差异（per-row vs per-contact），
源于 Todorov 2014 论文推导 vs MuJoCo 实现差异。我们的 per-row R 穿透更小（0.25 vs 1.1mm），
对刚体仿真更优。不修改。

### Session 14 — 遗留清理 + 求解器重构 + 精度排查 (2026-03-31)

**遗留代码清理：**
- [x] `examples/simple_quadruped.py` 迁移到 Scene API（修复失效导入 + 旧 RobotModel API）
- [x] `test_self_collision.py` 添加 docstring 说明（AABBSelfCollision 仍被 GPU backends 使用）

**求解器调度重构（方案 A — 保守清理）：**
- [x] 删除 `ADMMContactSolver` (physics/solvers/admm.py) — 死代码，GPU 用 Warp kernel
- [x] 删除 `JacobiPGSContactSolver` (physics/solvers/jacobi_pgs.py) — 死代码，GPU 用 Warp kernel
- [x] 重命名 `mujoco_qp.py` → `admm_qp.py`
- [x] Simulator 默认求解器从 PGS 改为 PGS-SI（与 CpuEngine 一致）
- [x] 删除 25 个死代码测试，更新跨求解器一致性测试

**GPU vs CPU 精度排查：**

之前报告的 "GPU vs CPU 0.176mm 差异" 经排查确认**不是精度问题**：

| 对比 | 差异 | 根因 |
|------|------|------|
| CpuEngine vs GpuEngine（同一 MergedModel 路径） | 0.001 mm | float32 精度正常 |
| Simulator vs Engine（不同求解器配置） | 0.175 mm | ADMMContactSolver vs ADMMQPSolver 的 compliance 不同 |
| 同一 ADMMQPSolver，Simulator vs CpuEngine | 0.000 mm | 完全一致 |

CPU f32 截断实验：CPU f64 vs CPU f32 只差 0.008mm，进一步确认差异来自求解器，非精度。

**清理后求解器矩阵：**

| 求解器 | 平台 | 文件 |
|--------|------|------|
| ADMMQPSolver | CPU (precision) | `solvers/admm_qp.py` |
| PGSSplitImpulseSolver | CPU (RL) | `solvers/pgs_split_impulse.py` |
| GPU ADMM kernel | GPU (precision) | `backends/warp/admm_kernels.py` |
| GPU Jacobi-PGS-SI kernel | GPU (RL, 默认) | `backends/warp/solver_kernels.py` |

**文档更新：**
- [x] `repo_list.md` 全面重写（从 Phase 1 的 5 模块更新到 30+ 模块完整 API 参考）
- [x] `MANIFEST.md` 修正文件名和测试数
- [x] `PLAN.md` 求解器矩阵更新
- [x] `PROGRESS.md` + `REFLECTIONS.md` 补 session 14

**Phase 2 总测试数：619（全部通过，较 session 13 的 644 减少 25 个死代码测试）**

### Session 15 — Q25 PGS 摩擦修复 + Q26-gpu 多 shape 碰撞 (2026-04-01)

**Q25 修复：PGS 摩擦行 per-row R 正则化**
- 摩擦行 `R_i = (1-d)/d × |W_ii|`（ADMM 同款 solimp），摩擦 warmstart 归零
- CPU PGS/PGS-SI + GPU solver_kernels/crba_kernels/solver_kernels_v2 全覆盖
- 7 新测试（球体静止稳定性、重球、减速验证）

**Q26-gpu：GPU 多 shape 碰撞 + 动态 N² broadphase**
- MuJoCo 式展平 shape 数组（shape_type/body/params/offset/rotation + body_shape_adr/num）
- 碰撞排除矩阵 (nb×nb) 上传 GPU（parent-child + CollisionFilter）
- `batched_detect_multishape` kernel：动态 N² broadphase + 多 shape narrowphase + atomic counter
- WarpBatchBackend 标记弃用（单 shape only）
- 7 新测试（多 shape 地面、动态 broadphase、碰撞过滤、稳定性）

**总测试数：649（全部通过）**

### Session 16-17 — GpuEngine API 扩展 + Q31 Backend 清理 (2026-04-03 ~ 2026-04-05)

**GpuEngine API 扩展 (session 16)：**
- [x] State accessors — `q_wp`, `qdot_wp`, `v_bodies_wp`, `x_world_R_wp`, `x_world_r_wp`（zero-copy Warp 数组）
- [x] Per-env reset — `reset_envs(env_ids)` + scatter kernel（支持 ADMM warmstart 清除）
- [x] Decimation — `step_n(n_substeps)` 避免重复 GPU→CPU 拷贝
- [x] StepOutput 现在填充 `X_world` 和 `v_bodies`（之前为 None）
- [x] Bug fix: `reset()` 更新 `_default_q`，`_scatter_zero_2d` 修复 2D/3D 维度错误
- [x] 29 新测试（state accessors + StepOutput + step_n + reset_envs + ADMM reset）

**Q31 Backend 清理 (session 17)：**
- [x] 删除 `BatchBackend(ABC)` + `StepResult` + `get_backend()` 工厂
- [x] 删除 `NumpyLoopBackend` + `TileLangBatchBackend` + `CudaBatchBackend`
- [x] 删除 `torch_solver.py` + `batched_crba.py` + `tilelang/` + `cuda/` 目录
- [x] 删除 `VecEnv` + `BatchedObsManager`（`rl_env/vec_env.py`）
- [x] 删除对应测试（test_tilelang_backend、test_cuda_backend、test_batched_crba）
- [x] 更新 PLAN.md、repo_list.md、PROGRESS.md

**删除统计：~4,100 行代码，12 个文件，3 个测试文件**

GpuEngine 是唯一的 GPU 物理引擎。下一步：Manager-based RLEnv。

---

## Phase 3 — High-Fidelity Rendering + Sensor Simulation ⬜

**待完成：**
- [ ] `rendering/vulkan_renderer/` — Vulkan + ray tracing 渲染器
- [ ] `rendering/camera_sim.py` — 相机模型（噪声、畸变、运动模糊）
- [ ] `rendering/lidar_sim.py` — LiDAR 点云仿真
- [ ] IMU 噪声模型

---

## Phase 4 — Domain Randomization ⬜

**待完成：**
- [ ] `domain_rand/physics_rand.py` — 质量、摩擦、阻尼、关节刚度随机化
- [ ] `domain_rand/visual_rand.py` — 纹理、光照、颜色随机化
- [ ] `domain_rand/noise_models.py` — 传感器噪声模型
- [ ] 结构化随机化课程（curriculum）

---

## Phase 5 — Sim-to-Real Transfer Validation ⬜

**待完成：**
- [ ] `deploy/policy_export.py` — ONNX / TorchScript 导出
- [ ] `deploy/hardware_bridge.py` — ROS2 / 硬件 SDK 接口
- [ ] 真实机器人部署 + sim-to-real gap 测量
- [ ] 系统辨识（system identification）迭代优化仿真参数
