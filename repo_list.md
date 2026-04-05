# Robot Simulator — API Reference

> Last updated: 2026-03-31 (session 14)

---

## Layer 0: `physics/spatial.py` — 空间代数

| 类 / 函数 | 说明 |
|-----------|------|
| `SpatialTransform(R, r)` | 刚体坐标变换；`apply_velocity`, `apply_force`, `inverse`, `compose`；工厂：`identity`, `from_rotation`, `from_translation`, `from_rpy` |
| `SpatialInertia(mass, inertia, com)` | 6x6 空间惯量；工厂：`from_box`, `from_cylinder`, `point_mass`；`matrix()` 返回矩阵；支持 `+` 合并 |
| `rot_x/y/z(angle)` | 3x3 基本旋转矩阵 |
| `skew(v)` | 3x3 反对称矩阵 |
| `quat_to_rot(q)` / `rot_to_quat(R)` | 四元数 (scalar-first) <-> 旋转矩阵 |
| `spatial_cross_velocity(v)` / `spatial_cross_force(v)` | 速度/力叉积算子 (6x6) |
| `gravity_spatial(g)` | 空间重力向量 `[0,0,-g, 0,0,0]` |

约定：`[linear(3); angular(3)]` (Pinocchio / Isaac Lab)

---

## Layer 1: `physics/` — 物理核心

### `joint.py` — 关节模型

| 类 | DOF | 构造参数 | 说明 |
|----|-----|---------|------|
| `RevoluteJoint` | 1 | `name, axis, q_min, q_max, k_limit, b_limit, damping, friction` | 转动关节；任意旋转轴 (3-vector)；penalty 限位 + 粘性阻尼 + Coulomb 摩擦 |
| `PrismaticJoint` | 1 | `name, axis` | 移动关节 |
| `FixedJoint` | 0 | `name` | 刚性连接 |
| `FreeJoint` | 6 | `name` | 浮动基；q=[qw,qx,qy,qz,px,py,pz]；`integrate_q()` 四元数积分 |

所有关节：`transform(q) -> SpatialTransform`，`motion_subspace() -> S`

### `_robot_tree_base.py` + `robot_tree.py` — 运动学树

| 类 / 方法 | 说明 |
|-----------|------|
| `Body` (dataclass) | `name, index, joint, inertia, X_tree, parent, children, q_idx, v_idx` |
| `RobotTreeBase(ABC)` | 接口：`aba()`, `forward_kinematics()`, `passive_torques()` |
| `RobotTreeNumpy(gravity)` | NumPy 实现 |
| `.add_body(body) -> int` | 添加 body，返回 index |
| `.finalize()` | 锁定树，分配 q/v 切片索引 |
| `.forward_kinematics(q) -> list[SpatialTransform]` | FK：各 body 世界位姿 |
| `.body_velocities(q, qdot) -> list[Vec6]` | 各 body 空间速度 |
| `.rnea(q, qdot, qddot, ext) -> tau` | 逆动力学 |
| `.aba(q, qdot, tau, ext) -> qddot` | 前向动力学 O(n) Featherstone ABA |
| `.crba(q) -> H` | 质量矩阵 (nv x nv)，composite rigid body |
| `.forward_dynamics_crba(q, qdot, tau, ext) -> qddot` | CRBA + Cholesky 前向动力学 |
| `.forward_dynamics_grouped_crba(q, qdot, tau, ext) -> qddot` | 层次化 Schur complement (分支机器人优化) |
| `.passive_torques(q, qdot) -> tau` | 关节限位 + 粘性阻尼 + Coulomb 摩擦 |
| `.default_state() -> (q, qdot)` | 零位状态 |
| `.integrate_q(q, qdot, dt) -> q_new` | 位形积分 (含四元数归一化) |

### `geometry.py` — 碰撞形状

| 类 | 构造参数 | 说明 |
|----|---------|------|
| `CollisionShape(ABC)` | — | `half_extents_approx() -> ndarray`, `support_point(dir)` |
| `BoxShape(size)` | `(lx, ly, lz)` | 长方体 |
| `SphereShape(radius)` | `float` | 球 |
| `CylinderShape(radius, length)` | — | 圆柱 |
| `CapsuleShape(radius, length)` | — | 胶囊 (球+圆柱+球) |
| `MeshShape(filename)` | `str` | 网格 (stub，碰撞跳过) |
| `ShapeInstance(shape, pose?)` | — | 形状 + 局部偏移 |
| `BodyCollisionGeometry(body_index, shapes)` | — | 一个 body 的碰撞几何集合 |

### `terrain.py` — 地形

| 类 | 说明 |
|----|------|
| `Terrain(ABC)` | `height_at(x, y)`, `normal_at(x, y)` |
| `FlatTerrain(z=0.0)` | 水平地面 |
| `HeightmapTerrain` | 高度图 (NotImplementedError) |

### `contact.py` — 接触模型

| 类 | 说明 |
|----|------|
| `ContactModel(ABC)` | `compute_forces(X, v, num_bodies, dt?, tree?) -> list[Vec6]` |
| `ContactParams(k_normal, b_normal, mu, ...)` | penalty 接触参数 |
| `ContactPoint(body_index, position, name)` | 离散接触点 |
| `PenaltyContactModel(params)` | Spring-damper 地面接触 |
| `NullContactModel()` | 无接触 (返回零力) |
| `LCPContactModel(collision_filter?)` | GJK/EPA + PGS LCP 求解 |

### `collision.py` — 自碰撞

| 类 | 说明 |
|----|------|
| `SelfCollisionModel(ABC)` | `compute_forces(X, v, num_bodies) -> list[Vec6]` |
| `BodyAABB(body_index, half_extents)` | body 包围盒 |
| `AABBSelfCollision(k_contact, b_contact)` | penalty AABB 自碰撞；`.from_geometries()` 工厂 |
| `NullSelfCollision()` | 无自碰撞 |

### `collision_filter.py` — 碰撞过滤

| 类 | 说明 |
|----|------|
| `CollisionFilter(n_bodies)` | 三层过滤取交集 |
| `.add_auto_exclude(parent_list)` | parent-child 自动排除 |
| `.set_group_mask(body_id, group, mask)` | bitmask 过滤 (uint32 双向) |
| `.add_exclude_pair(i, j)` | 显式排除对 |
| `.should_collide(i, j) -> bool` | O(1) 查询 |

### `gjk_epa.py` — 凸体碰撞检测

| 函数 | 说明 |
|------|------|
| `gjk_epa_query(shape_a, X_a, shape_b, X_b) -> ContactManifold?` | GJK 交叉测试 + EPA 穿透深度 |
| `ground_contact_query(shape, X, ground_z) -> ContactManifold?` | 形状 vs 地面 |
| `ContactManifold` | `body_i, body_j, normal, depth, points[]` |

### `broad_phase.py` — Broad-phase 加速

| 类 | 说明 |
|----|------|
| `BroadPhase(ABC)` | `query(aabbs) -> list[(i,j)]` |
| `BruteForceBroadPhase` | O(n^2) 全对 |
| `AABBTreeBroadPhase` | O(n log n) 递归分割 |

### `integrator.py` — 数值积分器

| 类 | 说明 |
|----|------|
| `SemiImplicitEuler(dt)` | symplectic Euler；NaN/Inf 检测；推荐主力 |
| `RK4(dt)` | 4 阶 Runge-Kutta；每步 4 次 ABA |
| `simulate(tree, q0, qdot0, ctrl_fn, contact_fn, dt, duration)` | 完整仿真循环 |

### `dynamics_cache.py` — 动力学缓存

| 类 | 说明 |
|----|------|
| `DynamicsCache` | FK + body_v 缓存，算一次全链路复用 |
| `.from_tree(tree, q, qdot, dt) -> DynamicsCache` | 工厂 |
| `.X_world: list[SpatialTransform]` | 世界位姿 |
| `.body_v: list[Vec6]` | body 空间速度 |
| `ForceState` | 力分解可观测性：`qfrc_passive, qacc_smooth, qacc` |

### `force_source.py` — 力源

| 类 | 说明 |
|----|------|
| `ForceSource(ABC)` | `compute(tree, q, qdot, cache) -> tau` |
| `PassiveForceSource` | 调用 `tree.passive_torques()` |

### `constraint_solver.py` + `constraint_solvers.py` — 约束求解适配

| 类 / 函数 | 说明 |
|-----------|------|
| `ConstraintSolver(ABC)` | `solve(tree, q, qdot, tau_smooth, contacts, cache) -> qacc` |
| `NullConstraintSolver` | 无约束 (qacc = ABA(tau_smooth)) |
| `AccelLevelAdapter` | 包装 ADMMQPSolver (加速度级) |
| `VelocityLevelAdapter` | 包装 PGS/PGS-SI (速度级) |
| `wrap_solver(solver) -> ConstraintSolver` | 自动检测并包装 |

### `step_pipeline.py` — 两阶段物理管线

| 类 | 说明 |
|----|------|
| `StepPipeline(dt, force_sources, constraint_solver)` | MuJoCo 风格两阶段管线 |
| `.step(tree, q, qdot, tau, contacts, cache?) -> (q_new, qdot_new)` | Stage1: smooth forces → Stage2: constraint → integrate |
| `.last_force_state -> ForceState?` | 上一步力分解 |

---

## Layer 1: `physics/solvers/` — 接触求解器

### `pgs_solver.py` — PGS (内部)

| 类 | 说明 |
|----|------|
| `ContactConstraint` (dataclass) | `body_i, body_j, point, normal, tangent1/2, depth, mu, condim, mu_spin, mu_roll` |
| `PGSContactSolver(max_iter, erp, cfm)` | Projected Gauss-Seidel；Signorini + Coulomb 锥；condim 1/3/4/6 |
| `.solve(contacts, v, X, inv_mass, inv_inertia, dt) -> list[impulse]` | 速度级求解 |

### `pgs_split_impulse.py` — PGS-SI (CPU RL 路线)

| 类 | 说明 |
|----|------|
| `PGSSplitImpulseSolver(max_iter, erp, slop, cfm)` | PGS + split impulse 位置修正 |
| `.solve(...)` | 委托 PGS(erp=0) 做速度；位置修正独立计算 |
| `.position_corrections -> list?` | 暴露给 Simulator 做 FreeJoint 位置修正 |

### `admm_qp.py` — ADMM-QP (CPU 精度路线)

| 类 | 说明 |
|----|------|
| `ADMMQPSolver(max_iter, rho, solref, solimp, warmstart, adaptive_rho)` | 加速度级 QP；R-regularization |
| `solref=(timeconst, dampratio)` | 弹簧-阻尼器阻抗参数 |
| `solimp=(d0, d_width, width, midpoint, power)` | 穿透-compliance 映射 |
| `.solve(contacts, tree, q, qdot, dt, cache) -> qacc` | CRBA → H, RNEA → C, Cholesky → 求解 |

别名: `MuJoCoStyleSolver = ADMMQPSolver`

---

## Layer 1: `physics/` — PhysicsEngine 统一接口

### `merged_model.py` — 多机器人合并

| 类 / 函数 | 说明 |
|-----------|------|
| `RobotSlice(q_slice, v_slice, body_slice)` | 单个 robot 在合并树中的切片 |
| `MergedModel` | `tree, robot_slices, collision_shapes, collision_filter, nq, nv, nb` |
| `merge_models(robots: dict[str, RobotModel]) -> MergedModel` | 多 robot 合并为单一多根树 |

### `engine.py` — PhysicsEngine ABC

| 类 | 说明 |
|----|------|
| `PhysicsEngine(ABC)` | `step(q, qdot, tau) -> StepOutput` |
| `StepOutput` (dataclass) | `q_new, qdot_new, contacts?, force_state?` |

### `cpu_engine.py` — CPU 引擎

| 类 | 说明 |
|----|------|
| `CpuEngine(merged, solver?, dt)` | GJK/EPA 地面 + 球近似 body-body；默认 PGS-SI |
| `.step(q, qdot, tau, dt?) -> StepOutput` | 完整物理步 |

### `gpu_engine.py` — GPU 引擎

| 类 | 说明 |
|----|------|
| `GpuEngine(merged, num_envs, dt, solver, device)` | Warp kernel 管线 |
| `solver="jacobi_pgs_si"` | 默认，Jacobi PGS + split impulse |
| `solver="admm"` | ADMM with solref/solimp compliance |
| `.step(q?, qdot?, tau?, dt?) -> StepOutput` | GPU 物理步；支持内部状态或外部传入 |

---

## Layer 1: `physics/backends/` — GPU 后端

| 后端 | 文件 | 主要内容 |
|------|------|---------|
| **Warp** | `warp/kernels.py` | 7 个 @wp.kernel: FK, passive_torques, PD, contact, collision, ABA, integrate |
| | `warp/crba_kernels.py` | CRBA + RNEA + Cholesky + contact Jacobian + Delassus |
| | `warp/solver_kernels.py` | Jacobi PGS-SI Warp kernel |
| | `warp/admm_kernels.py` | ADMM Warp kernel (in-kernel Cholesky, 锥投影, warmstart) |
| | `warp/collision_kernels.py` | 解析碰撞 kernel (shape type dispatch) |
| | `warp/analytical_collision.py` | 14 个 @wp.func: ground + body-body 解析碰撞 |
| | `warp/spatial_warp.py` | 空间代数 @wp.func (rodrigues, transform, cross) |
| | `warp/scratch.py` | ABABatchScratch GPU 缓冲区 |
| **共享** | `static_data.py` | StaticRobotData.from_model() / .from_merged() |
| ~~CUDA/TileLang/NumPy~~ | ~~已删除 (Q31)~~ | ~~BatchBackend ABC + 3 backends → GpuEngine 统一~~ |

---

## Layer 2: 场景与仿真

### `scene.py` — 场景容器

| 类 | 说明 |
|----|------|
| `StaticGeometry(name, shape, pose, mu, condim)` | 静态碰撞体 (墙壁/障碍物)；无质量/动力学 |
| `BodyRegistry(robots, n_static)` | 全局 body 索引管理；`global_id()`, `to_local()`, `is_static()` |
| `Scene(robots, static_geometries, terrain, ...)` | 顶层容器 |
| `.build() -> Scene` | 创建 registry + collision_filter |
| `.single_robot(model, ...) -> Scene` | 便捷工厂 (单机器人) |
| `.registry -> BodyRegistry` | build 后可用 |

### `collision_pipeline.py` — 统一碰撞检测

| 类 | 说明 |
|----|------|
| `CollisionPipeline(scene)` | 三阶段碰撞检测 |
| `.detect(all_X, all_v) -> list[ContactConstraint]` | 1. body vs terrain (ground_contact_query) |
| | 2. body vs static_geom (gjk_epa_query) |
| | 3. body vs body (collision_filter + gjk_epa_query) |
| `.gather_mass_properties() -> (inv_mass, inv_inertia)` | 全局质量属性 (static = inv_mass=0) |

### `simulator.py` — 多机器人编排

| 类 | 说明 |
|----|------|
| `Simulator(scene_or_model, integrator, solver?, engine?)` | 默认 PGS-SI；支持 RobotModel 自动包装 |
| `.step(states_dict, taus_dict) -> states_dict` | 多机器人：`{"a": (q,qdot)}` |
| `.step_single(q, qdot, tau) -> (q, qdot)` | 单机器人便捷接口 |
| `.step(q, qdot, tau) -> (q, qdot)` | 向后兼容 (检测到 3 参数自动走 step_single) |

Engine mode: `Simulator(scene, integrator, engine=GpuEngine(...))` 时走 MergedModel 路径。

---

## Robot 描述

### `robot/model.py` — RobotModel

```python
@dataclass
class RobotModel:
    tree: RobotTreeNumpy
    actuated_joint_names: list[str] = []
    contact_body_names: list[str] = []
    geometries: list[BodyCollisionGeometry] = []
    effort_limits: NDArray | None = None
```

纯描述对象，不含碰撞管理 (碰撞在 Scene 层)。

### `robot/urdf_loader.py` — URDF 加载

| 函数 | 说明 |
|------|------|
| `load_urdf(path, floating_base, contact_links, ...) -> RobotModel` | URDF -> RobotModel |
| `load_urdf_scene(path, ...) -> Scene` | URDF -> Scene (便捷接口) |

两阶段：`_parse_urdf() -> _URDFData` → `_build_model() -> RobotModel`

---

## RL 环境

### `rl_env/cfg.py` — 配置

| 类 | 说明 |
|----|------|
| `NoiseCfg(type, std?, low?, high?)` | Gaussian / Uniform 噪声 |
| `ObsTermCfg(func, noise?, scale?, params?)` | 单个观测 term |
| `EnvCfg(obs_terms, kp, kd, action_scale, action_clip, dt, ...)` | 完整环境配置 |

### `rl_env/controllers.py` — 动作控制器

| 类 | 说明 |
|----|------|
| `Controller(ABC)` | `compute(action, q, qdot) -> tau` |
| `PDController(kp, kd, default_q, effort_limits?)` | PD + effort clamp |
| `TorqueController(effort_limits?)` | 直接力矩 + effort clamp |

### `rl_env/obs_terms.py` — 观测 term 函数

| 函数 | 返回 | 说明 |
|------|------|------|
| `base_lin_vel(env)` | `(3,)` | 基座线速度 (body frame) |
| `base_ang_vel(env)` | `(3,)` | 基座角速度 |
| `base_orientation(env)` | `(4,)` | 基座四元数 |
| `joint_pos(env)` | `(nu,)` | 关节角度 |
| `joint_vel(env)` | `(nu,)` | 关节速度 |
| `contact_mask(env)` | `(n_contact,)` | 足端接地 0/1 |

### `rl_env/managers.py` — Term 管理器

| 类 | 说明 |
|----|------|
| `TermManager(ABC)` | term 函数注册 + 批量计算 |
| `ObsManager(obs_terms, env)` | `compute() -> obs_vector`；train/eval 噪声切换 |
| `RewardManager` | stub (返回 0.0) |
| `TerminationManager` | stub (返回 False) |

### `rl_env/base_env.py` — Gymnasium 环境

| 类 | 说明 |
|----|------|
| `Env(model, cfg)` | `gym.Env` 兼容；预计算静态索引 |
| `.reset() -> obs` | 重置到初始状态 + noise |
| `.step(action) -> (obs, reward, terminated, truncated, info)` | 标准 Gymnasium API |

### `rl_env/vec_env.py` — 并行环境

| 类 | 说明 |
|----|------|
| `VecEnv(model, cfg, n_envs, backend)` | N 并行环境 |
| `backend="numpy"` | Python for-loop (CPU) |
| `backend="warp"` | GPU Warp kernel |
| `.reset() -> obs (N, obs_dim)` | 批量重置 |
| `.step(actions) -> (obs, rewards, terms, truncs, infos)` | 批量步进 |

---

## `rendering/viewer.py` — 可视化

| 类 / 方法 | 说明 |
|-----------|------|
| `RobotViewer(tree, floor_size, contact_names)` | matplotlib 3D (调试用) |
| `.render_pose(q)` | 单帧 |
| `.animate(times, qs, interval, save_path)` | 轨迹回放 -> .gif/.mp4 |

---

## `examples/simple_quadruped.py` — 验证 demo

| 函数 | 说明 |
|------|------|
| `build_quadruped() -> RobotModel` | 17 体四足 (12 revolute + 4 fixed foot)；碰撞几何：torso Box + foot Sphere + calf Cylinder |
| `standing_state(tree) -> (q, qdot)` | FK 精确计算站立高度 |
| `main(save_path?)` | Scene + Simulator + PD 控制 + 动画 |

---

## 测试结构

```
tests/
├── unit/           # 单元测试 (spatial, joints, robot_tree, contact, collision, solvers, rl_env, robot)
├── integration/    # 集成测试 (simulator, contact_pipeline, static_data)
├── reference/      # 外部对标 (analytical LCP, Bullet 轨迹, MuJoCo 精度, Pinocchio ABA)
├── gpu/            # GPU 测试 (backends, collision, solvers, kernels)
└── validation/     # 端到端验证 (dynamics vs MuJoCo, contact vs MuJoCo, GPU vs CPU, 两四足碰撞)
```

619 个测试，全部通过。
