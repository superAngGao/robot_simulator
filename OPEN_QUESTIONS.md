# Robot Simulator — Open Questions

> **How to use:** Read this file at the start of every session.
> Add new items as they come up. Resolve items by moving them to
> REFLECTIONS.md with the decision recorded, then deleting from here.

---

## Physics / Algorithms

**Q1 — Joint friction (URDF `<dynamics friction="..."/>`)**
URDF joint friction is parsed and stored in `_URDFData` but not used.
True static/dynamic friction at the joint level requires a different model
from viscous damping (e.g., Coulomb friction with stiction zone).
- Current: parsed, silently ignored in `_build_model`
- Needed: decide model (Coulomb? LuGre?) and where it lives (joint or integrator)
- Blocking: nothing for now. Revisit in Phase 2.

**Q2 — Body velocity exposure from RobotTree** ✅ RESOLVED
Added `RobotTree.body_velocities(q, qdot) -> list[Vec6]` in `physics/robot_tree.py`.
Removed `_compute_body_velocities()` from `simple_quadruped.py`.
Covered by `tests/test_body_velocities.py` (4 tests).
→ Moved to REFLECTIONS.md.

**Q3 — AABB center at body origin, not CoM**
Current `AABBSelfCollision` uses the body frame origin as the AABB center.
If the CoM is far from the origin (offset-heavy links), the bounding box
is inaccurate.
- Current: acceptable for Phase 1 (links are roughly symmetric)
- Fix: use CoM-centered AABB, or switch to OBB. Revisit in Phase 2.

**Q4 — Contact/self-collision unification (long-term)**
Phase 1 uses explicit `ContactPoint` (discrete foot tips) for ground contact
and `BodyAABB` for self-collision — two separate geometry systems.
Phase 2+ should unify: ground contact generated from `BodyCollisionGeometry`
automatically (any geometry touching terrain → contact), not from manually
specified points.
- Blocking: nothing for Phase 2. Keep the two systems from diverging in API.
- Revisit: when implementing `TerrainPenaltyContactModel`.

---

## robot/ Layer

**Q5 — URDF with no `<inertial>` on a link**
Treated as point mass `1e-6 kg` at origin with a warning log.
Alternative: infer inertia from collision geometry (MuJoCo `inertiafromgeom`).
- Current decision: placeholder mass, log warning.
- Revisit: if users report unrealistic behaviour for sensor/virtual links.

**Q6 — Multiple `<collision>` elements per link**
All shapes are kept in `BodyCollisionGeometry.shapes: list[ShapeInstance]`.
Each collision algorithm decides how to merge or iterate them.
- Confirmed design, no action needed — but must verify AABB merge logic
  in `AABBSelfCollision.from_geometries()` handles multi-shape bodies correctly.

**Q7 — Mesh collision geometry (`<geometry><mesh/>`)**
`MeshShape` stores only `filename`, no geometry is loaded or processed.
- Current: silently skipped in collision model construction (logged as warning)
- Needed: convex hull or SDF baking. Phase 3.

**Q8 — Simulator (Layer 2) module location** ✅ RESOLVED
Decision: top-level `simulator.py` (Option B).
Rationale: physics/ is an algorithm library; Simulator is a consumer/orchestrator.
Consistent with Drake (Simulator separate from MultibodyPlant) and MuJoCo (mj_step
is not inside the physics model). Matches the "two external entry points" constraint:
`load_urdf()` and `Env()` — Simulator sits between them, not inside physics/.
→ Moved to REFLECTIONS.md.

---

## rl_env / Layer 3

**Q9 — Generic obs/action space for diverse robot types** ✅ RESOLVED
Full design decided. See REFLECTIONS.md.
→ Moved to REFLECTIONS.md.

**Q13 — RewardManager / TerminationManager term functions**
Phase 2d 留了 stub（返回 0.0 / False）。Phase 3 需要实现具体 term：
- Reward: forward velocity、energy penalty、alive bonus、foot clearance 等
- Termination: base height too low、base orientation too tilted、timeout
- 设计问题：term 函数是否与 obs_terms 共享同一签名 `fn(env, **params) -> Tensor`？
  还是 reward term 返回 scalar、termination term 返回 bool？
- 参考：Isaac Lab RewardManager 用 `fn(env) -> Tensor` 统一，scalar 由 weight 乘后 sum
- Blocking: nothing for Phase 2e. Revisit at Phase 3 start.

**Q14 — VecEnv auto-reset on episode termination**
当前 `VecEnv.step()` 不自动 reset 已结束的 sub-env（terminated 或 truncated）。
RL 训练通常需要 auto-reset（Isaac Lab / Gymnasium VectorEnv 均支持）。
- 当前：调用方负责检测 term/trunc 并手动 reset
- 选项 A：VecEnv 内部 auto-reset，返回 `final_obs` 在 info 里（Gymnasium 标准）
- 选项 B：保持当前行为，由 RL trainer 管理 reset
- Blocking: nothing until Phase 3 RL training loop is implemented.
- **注意**：Q31 架构重构将重新定义 RL 环境层，auto-reset 设计在重构中一并解决。

**Q31 — RL 环境层架构重构：VecEnv → Manager-based RLEnv** (2026-04-03)

**问题**：当前存在两条并行 GPU 物理路径：
  - **GpuEngine**（warp kernels）：Simulator 用，完整 CRBA+Cholesky+ADMM/PGS
  - **TileLang/CUDA backends**（VecEnv 用）：独立重写了 FK/ABA/PGS/积分，功能是 GpuEngine 的子集

6 个核心算法完全重复实现。每次加 feature（如 Prismatic joint、multi-shape 碰撞）
需要改两套代码。TileLang 缺少 ADMM、body-body 碰撞等 GpuEngine 已有能力。

**决策**：
  - **GpuEngine 是唯一的 GPU 物理引擎**，支持 num_envs=N 批量
  - **RL 环境层**改为 Isaac Lab 风格 Manager-based 架构，薄层包装 GpuEngine
  - 删除 TileLang/CUDA/Numpy backends 和 BatchBackend 抽象

**目标架构**：
```
RLEnv (Manager-based, Gymnasium 兼容)
  ├── ActionManager      — action scaling → tau
  ├── ObservationManager — q/qdot/contacts → obs tensor
  ├── RewardManager      — weighted sum of reward terms
  ├── TerminationManager — OR of done conditions
  ├── EventManager       — domain randomization
  └── CommandManager     — goal/velocity commands
  │
  └── GpuEngine(num_envs=N)  ← 唯一物理调用

Simulator (场景编排)
  └── GpuEngine(num_envs=1)  ← 同一个引擎
```

**GpuEngine 需补齐的 Gap**：
  1. StepOutput 暴露 X_world, v_bodies（低）
  2. Decimation 支持（低）
  3. Contact forces 便捷接口（低）
  4. Per-env reset `reset_envs(env_ids)`（中）
  5. Runtime 参数修改 — domain randomization（高，可后期做）

**命名**：环境类改名为 `RLEnv`（或待定），"Vec" 批量化由 GpuEngine 内部处理。

**参考**：Isaac Lab `ManagerBasedRLEnv`（7+ managers, config-driven）、
Brax `PipelineEnv`（函数式 + wrapper 组合）、Isaac Gym `VecTask`（模板方法）。
三者共同点：env 层不碰物理，只管 obs/reward/reset/action。

**Q11 — `<inertial><origin rpy>` 非零的处理**
URDF 允许惯量张量在任意旋转的 CoM frame 里定义（非零 rpy）。
几乎所有真实 URDF 的 inertial rpy 都是零，但规范上合法。
- 当前决策：零 rpy 正常处理；非零 rpy log warning，不报错，张量直接使用
- 待定：是否需要将张量旋转到 link frame（`I_link = R @ I_com @ R.T`）
- 参考：Pinocchio 和 Drake 都做了完整旋转变换

**Q15 — 空间向量顺序约定 `[angular; linear]` vs `[linear; angular]`** ✅ RESOLVED
已统一改为 `[linear(3); angular(3)]`，与 Pinocchio / Isaac Lab 对齐。
改动覆盖：`spatial.py`、`joint.py`、`contact.py`、`collision.py`、`obs_terms.py`
以及所有相关测试文件。166 个测试全部通过（含 Pinocchio 对比测试，不再需要 `_P6` 置换矩阵）。
→ Moved to REFLECTIONS.md.

**Q12 — Fixed joint 合并优化（未来）**
当前每个 link 保留独立 Body，fixed joint 不合并。
若未来做合并优化（减少 ABA 计算量），需注意平行轴定理的正确应用：
`I_A = I_B + m * (|r|²·I₃ - r·rᵀ)`，其中 r 是从 A origin 到 B CoM 的向量。
Pinocchio issue #1388 曾在此处有 bug。
- 当前：不合并，无风险
- 未来：合并前必须加单元测试验证惯量变换

---

## GPU Dynamics Algorithms

**Q16 — CRBA vs ABA：GPU 上的前向动力学算法选择** 🔄 部分解决

**已解决**：
- ABA/CRBA 自动切换阈值：实测结果 — fused scalar CRBA 在 nv=30 时达 ABA 的 0.96x，
  在 nv=62 时为 0.56x。**对于 nv ≤ 64 的机器人，fused ABA 仍是最优选择。**
- cuSOLVER tensor core（wgmma）路径：因 3 次 kernel launch + global memory H 访问，
  比 fused scalar Cholesky 更慢。wgmma M=64 最小维度对 nv < 64 不友好。
- 精度：float32 scalar Cholesky 在 nv=62 下稳定，与 ABA 吻合 atol=1e-4 (float32 vs float64)。

**未解决 — 分组策略（Phase 2g-3，潜在优化）**：
- 决定使用**自动分支点检测**（多子节点 body 为切割点，每个子树一组）
- 层次化 Schur complement 求解（limb 并行 Cholesky → root Schur → 回代）
- 理论 FLOPs 减少显著（四足 18x），但实际 GPU 上小矩阵 Cholesky 开销可能抵消
- **何时实现**：当目标机器人 nv > 30 且有明确分支结构时值得实现
- 当 nv_limb ≥ 16 时，各 limb 的 Cholesky 可走 tensor core（wgmma tile 对齐）

---

## Collision / Contact

**Q30 — GPU Multi-Shape Collision Data Layout** ✅ RESOLVED (2026-04-01)

Cross-engine research (8 engines) completed. Universal pattern: flat parallel
arrays indexed by shape ID (`shape_body[]`, `shape_type[]`, `shape_transform[]`),
pre-allocated contact buffers with atomic counters (GPU) or static shapes (JAX),
type-pair dispatch table for narrowphase. Forward index `body_shape_adr/num`
optional but useful. Broadphase: start N-squared + bounding sphere, add SAP later.
`warp.sim` removed in Warp 1.10, superseded by Newton (Linux Foundation project).
→ Full analysis in REFLECTIONS.md session 15b.

**Q23 — GPU 多体求解器角速度发散** ✅ RESOLVED (2026-03-27)

**根因**：`solver_kernels_v2.py` 中 body-body 接触的 `J_body_j` 缺少取反。
CPU 正确写 `J_body_j = -J_compute(...)`，GPU 错误写 `J_body_j = +J_compute(...)`。
约束变成了绝对速度而非相对速度，导致 PGS 解出错误的 lambda 方向。

**附带修复**：
- `static_data.py`：`body_collision_radius` 从 `collision_shapes` 读取实际半径（原先硬编码 0.05）
- `cpu_engine.py`：body-body 碰撞检测用 `half_extents_approx()` 而非不存在的 `half_extents` 属性

**测试**：`test_gpu_multibody.py` 新增 6 个测试（CPU vs GPU 对比、角速度不发散、body-body 碰撞、地面着陆）
→ Moved to REFLECTIONS.md.

---

**Q18 — 接触系统与主流项目的差距（Phase 2f 后续优化）**

Phase 2f 已实现 GJK/EPA + PGS LCP + 关节 Coulomb 摩擦。与 MuJoCo/Bullet/Drake 对比，
剩余的关键差距按优先级：

1. **完整 Delassus 矩阵** — 当前 PGS 只用对角 `W_ii`，忽略接触点间耦合。
   多点接触（如 box 四角着地）时收敛慢、精度低。
   需要构建完整 `W = J M⁻¹ Jᵀ`，其中 J 是接触 Jacobian。
   参考：MuJoCo `mj_makeConstraint` + CG solver、Bullet `btSequentialImpulseConstraintSolver`。

2. **Warm starting** — 上一步的 LCP 解作为下一步初始值。
   PGS 迭代从 ~30 降到 ~3-5。实现简单：缓存 `lambda[]`，按接触 ID 匹配。
   参考：Bullet `btPersistentManifold::m_appliedImpulse`。

3. **Capsule 形状** — 几乎所有腿部 URDF 用 capsule（球+圆柱+球）。
   `support_point()` 易实现：两端球心 + 半径。

4. **接触持久化 (manifold cache)** — 帧间保持接触点，避免抖动。
   当前用 Bullet 方案（body-local 坐标距离匹配，阈值 2cm）。
   未来可升级为 PhysX 方案（EPA 返回 feature index：面/边/顶点 ID，精确匹配无阈值）。
   需要改 EPA 记录穿透方向对应的 simplex feature pair。
   参考：PhysX `PxContactPair::extractContacts` feature index、Bullet `btPersistentManifold`。

5. **Broad-phase 空间加速** — 当前自碰撞是 O(n²) 全对检测。
   空间哈希 / Dynamic AABB Tree (DBVT) 可降到 O(n log n)。

6. **弹性碰撞 (restitution)** — PGS 中 `e * v_n_prev` 项已预留但未实现。

7. **隐式接触积分** — 当前是显式（先算力再积分），MuJoCo/Drake 用隐式
   （接触约束与动力学耦合求解），数值稳定性更好。

   7b. ~~**最大恢复速度限制 (max depenetration velocity)**~~ ✅ RESOLVED (2026-04-06 session 19)
   `PGSContactSolver` / `PGSSplitImpulseSolver` / `crba_kernels.batched_build_W_joint_space`
   均加了 `max_depenetration_vel` 钳位（PhysX 风格），默认 1 m/s（因为我们把 position
   correction 折进 velocity，钳位值即 post-solve 速度，不能设太大）。
   `StaticRobotData.max_depenetration_vel` 集中管理。
   6 个新单元测试覆盖 clamp 行为 + 深穿透不弹飞集成测试。

8. ~~**碰撞过滤掩码**~~ ✅ RESOLVED — `physics/collision_filter.py` 实现了三层过滤：
   auto-exclude（parent-child）、bitmask（group/mask uint32）、explicit exclude set。
   集成到 `AABBSelfCollision`、`LCPContactModel`、`load_urdf(collision_exclude_pairs=...)`。
   参考：Drake CollisionFilterDeclaration + MuJoCo contype/conaffinity。

9. **接触维度控制 — condim 4/6 的滚动/扭转摩擦** (2026-04-06 调研完成，2026-04-07 架构讨论完成，**延后到 Phase 3 完成后实装**)

   当前 condim 固定 3D。`ContactConstraint` 已有 `condim`/`mu_spin`/`mu_roll` 字段，
   PGS solver 框架已支持 1/3/4/6，但 angular Jacobian 行（spin/roll）未实现。

   **状态**：研究 + 架构设计都完成，实装延后。本问题在 session 20 讨论中从
   "加一个 rolling friction 约束行"扩展成了"多物理场界面/材料架构"的完整设计讨论，
   最终决定**与 `InterfaceMaterial` 重命名 + 材料 per-shape 附着一起做，而非孤立实装**。
   触发时机：Phase 2 (rigid body) 完成 + Phase 3 (rendering + RL) 脚手架搭好之后。
   理由：(1) 现在没有必须解决的 bug；(2) 架构收益要 Phase 5+ 多物理场才兑现；
   (3) 过早 refactor 等于在错的 phase 付实装成本。

   **完整设计线索**：
   - `REFLECTIONS.md` 2026-04-07 (session 20) — 讨论脉络和决定
   - `REFERENCES.md` 新矩阵行 "Multi-physics interface / material architecture"
     + 新 SOFA / Genesis 条目 + Drake hydroelastic 小节
   - `memory/project_multiphysics_architecture.md` session 20 update — MVP 清单
     和 "为什么不需要 RigidBodyMaterial 壳类" 的物理论证

   **实装前要重读的东西**（按顺序）：
   1. 上面三个文件的 session 20 相关段
   2. Session 18 斜面滚动实验结果（REFLECTIONS "Sphere Rolling Physics Validated"）
      — condim=3 下横向速度守恒，这是 rolling friction 要解决的唯一物理缺口
   3. 本条目下面的 "各项目方案" 和 "实现要点"

   **各项目方案**：
   - MuJoCo：condim=4 加 1 行扭转（ω·n），condim=6 再加 2 行滚动（ω·t₁, ω·t₂）。
     统一椭圆锥 `f_n² ≥ Σ(f_i/μ_i)²`。R 对角耦合 `R[j]·μ[j]² = const`。
   - Bullet：独立 box 约束，上限不依赖法向力（物理不准但简单）。
   - PhysX：仅扭转（torsionalPatchRadius），无滚动摩擦，且仅 TGS solver。
   - Drake：无显式滚动摩擦，hydroelastic contact 隐式提供。
   - ODE：Tasora & Anitescu (2013) 互补模型，可选 `ρ·f_n` 耦合。

   **实现要点**：
   - 行 0-2 用线速度 Jacobian `J_lin = d + (r×d)`；行 3-5 用纯角速度 `J_ang = d`
   - mu_spin 单位是长度（接触面片直径），mu_roll 单位是长度（变形深度）
   - 验证场景：球在斜面横向初速度，condim=6 时 v_cross 应衰减，condim=3 时守恒
   - 参考：Tasora & Anitescu (2013) Meccanica 48, pp.1643-1659

   **实装时的第一步**（不是 rolling friction 本身，而是先做地基）：
   1. `InterfaceMaterial` dataclass（union 字段）+ `ShapeInstance.interface` 附着
   2. `ContactConstraint.mu/mu_spin/mu_roll` 从 `ShapeInstance.interface` 读取，
      删除全局 override 通道
   3. 纯重命名 + 字段迁移，**不改求解器逻辑**，作为独立 PR
   4. 然后才开始 condim 4/6 angular Jacobian rows 的实装

10. **同 body 多 geom 过滤** — 一个 body 有多个 collision shape 时，
    同 body 的 shape 之间不应碰撞。当前 `BodyCollisionGeometry` 合并为单 AABB，
    升级为 per-shape 碰撞后需要此过滤。

**Q17 — BVH 三角 Mesh 碰撞（Phase 3 延后）**

Phase 2f 使用 GJK/EPA 处理凸形状（Box/Sphere/Cylinder/ConvexMesh），非凸 mesh
通过凸分解预处理。直接的三角 mesh 碰撞检测延后到 Phase 3：

- **BVH 加速结构**：AABB/OBB 层次包围盒树，加速三角 mesh 查询
- **三角-凸体 narrowphase**：GJK 求解凸 shape 与单个三角形的最近距离
- **三角-三角碰撞**：仅在双方都是 mesh 时需要（罕见，大部分机器人 link 是凸的）
- **参考实现**：Bullet `btBvhTriangleMeshShape`、coal/hpp-fcl BVH、PhysX GPU mesh
- **触发条件**：当用户加载带 `<mesh>` 碰撞几何的 URDF 且无凸分解可用时
- 当前：MeshShape 存 filename，碰撞检测跳过（Q7）

**Q32 — TriangleMeshTerrain：heightfield → 三角网格地形碰撞（Phase 3+）** (2026-04-05)

斜面/复杂地形的终极方案。Isaac Lab 做法：Python 生成 heightfield（2D numpy int16）
→ `convert_height_field_to_mesh()` 转为 `trimesh.Trimesh` → PhysX `PxTriangleMeshGeometry`
静态碰撞体。法线来自三角面片，不走 terrain 查询。

实现路径：
- heightfield → 三角网格生成（每格 2 三角形，含坡度阈值垂直壁修正）
- BVH 加速的 mesh-vs-convex narrowphase（复用 Q17 的 BVH 基础设施）
- GPU 端：mesh 数据上传为 flat arrays，broadphase AABB 筛选后逐三角形 GJK
- 当前简单场景（无限斜面）由 `HalfSpaceShape` 覆盖（方案 B），本条目仅针对非平面地形

参考：Isaac Lab `hf_terrains.py` `@height_field_to_mesh` decorator、
MuJoCo `<hfield>` geom type、Bullet `btHeightfieldTerrainShape`（内部转 mesh）。

---

## Solver Stability

**Q21 — 求解器稳定性与算法改进路线（2026-03-24 决策）**

**背景**：两球撞墙场景中，PGS + Baumgarte ERP 发散（vx→3480），
ADMM 稳定但误差大（L2~10cm vs Bullet）。两条路线都未达到生产可用。

**决策：两条路线都补齐，提供 5 个求解器供用户按场景选择。**

### 最终求解器矩阵

| 求解器 | 平台 | 算法改进 | 适用场景 | 参考项目 |
|--------|------|---------|---------|---------|
| **PGS** | CPU | baseline（当前） | 调试/参考 | — |
| **PGS-SI** | CPU+GPU | + split impulse | RL 训练（快速、够用） | Bullet |
| **Jacobi-PGS-SI** | GPU | Jacobi 并行 + split impulse | 大规模 RL（N=1000+） | PhysX |
| **MuJoCo-QP** | CPU | + warmstart + 自适应ρ | MuJoCo 对标仿真 | MuJoCo |
| **ADMM-C** | CPU | + 合规接触 + 自适应ρ | 高精度 CPU 仿真 | MuJoCo |
| **ADMM-TC** | GPU | + tensor core batched Cholesky | 高精度 GPU 仿真 | Drake SAP |

后续扩展（不在当前批次）：
- **ADMM-C + Newton 精化**：ADMM 收敛后 1-2 步 Newton，sim-to-real 精度
- Jacobi-PGS-SI 的 warm start 优化

### PGS 路线改进（P1: split impulse）

```
Pass 1: PGS 解速度约束（v_n >= 0），无 Baumgarte bias
Pass 2: 直接位置修正（q += normal * depth * erp），不走 force chain
两个 pass 解耦 → 无正反馈 → 无发散
```
参考：Bullet `btSequentialImpulseConstraintSolver` split impulse

### ADMM 路线改进（A1+A2: 合规接触 + 自适应ρ）

**A1 合规接触**：锥投影从硬约束改为软约束（MuJoCo solref/solimp 模型）
```
硬约束：proj_K(s) — 投影到 λ_n≥0 + 摩擦锥
合规：  proj_compliant(s, depth, v_n) — stiffness*depth + damping*v_n 编码进投影
```
效果：自带位置修正，不需要 Baumgarte。

**A2 自适应ρ**：Boyd et al. 2011 标准方案
```
if primal_residual > 10 * dual_residual: ρ *= 2, 重分解 A
if dual_residual > 10 * primal_residual: ρ /= 2, 重分解 A
```
效果：不同刚度的接触自动适配。

**ADMM-TC (tensor core)**：batched Cholesky on GPU
- A = M + ρJᵀJ 每步分解一次（N 个独立矩阵并行）
- 迭代内只做三角求解（L⁻ᵀL⁻¹ rhs）
- Phase 2g 已验证 batched Cholesky 在 H200 上可行

### 不做的方案及原因

| 方案 | 不做的原因 |
|------|-----------|
| PGS velocity clamping | hack，split impulse 后不需要 |
| PGS CFM 柔化 | ADMM 合规接触包含了，且更系统 |
| PGS TGS (Temporal GS) | 串行更新，和 GPU Jacobi 并行矛盾 |
| 位置级互补条件 | 重写量太大，Newton 精化是更实际的替代 |

### 实施顺序

1. ~~PGS + split impulse（~50 行，解决 PGS 发散）~~ ✅ 2026-03-25
2. ~~ADMM 合规接触 + 自适应ρ（~130 行，提升 ADMM 精度）~~ ✅ 2026-03-25
3. ~~MuJoCoStyleSolver warmstart + 自适应 rho~~ ✅ 2026-03-26
   - 50kg 球 ADMM 50 iter 收敛不足（L2=2.37mm），修复后通过（L2<0.1mm）
   - GPU 策略：warmstart 不影响并行；自适应 rho 可用 `where()` 消除分支，
     或 GPU 路径用 `adaptive_rho=False` + warmstart（更友好）
4. GPU Jacobi PGS + split impulse kernel
5. GPU ADMM-TC kernel（batched Cholesky + tensor core）
6. （长期）ADMM + Newton 精化

**已完成实现（2026-03-25）：**

- `physics/solvers/pgs_split_impulse.py` — `PGSSplitImpulseSolver`
  - 委托 PGS(erp=0) 做速度求解，位置修正独立计算
  - 位置修正通过 `position_corrections` 属性暴露给 Simulator
  - Simulator 自动对 FreeJoint body 应用位置修正
  - 26 个测试（含 ball-wall 不发散验证 + 解析 LCP 对比 + 位置修正单元测试）

- `physics/solvers/admm.py` — `ADMMContactSolver` 新增参数：
  - `contact_stiffness`/`contact_damping`：阻抗归一化弹簧-阻尼 bias，替代 Baumgarte ERP
  - `adaptive_rho`：Boyd 2011 方案，primal/dual 残差比触发 ρ 缩放 + Cholesky 重分解
  - 向后兼容：默认参数行为不变

---

## Architecture / Refactor

**Q19 — physics/ 接触模块重构（GPU contact 完成后）**

GPU Jacobi PGS + ADMM 求解器实现完成后，需要一次认真的重构：
- `contact.py`（486 行）拆分为独立文件或 `contact/` 子包（Penalty/LCP/Null 分离）
- GPU backends 各自独立实现的 penalty contact kernel 统一为可插拔接触管线
- `collision.py` + `collision_filter.py` + `broad_phase.py` 考虑合并为 `collision/` 子包
- `physics/solvers/` 接口验证（PGS / Jacobi PGS / ADMM 统一 ABC）
- **触发条件**：GPU ADMM kernel 完成且测试通过后

**Q32 — batched_build_W_joint_space 求解器 bias 分离**

当前 `batched_build_W_joint_space` 同时计算 W/v_free（共用）和 Baumgarte ERP bias（PGS 专用）。
ADMM 有自己的 compliance 模型（solref/solimp），不需要 Baumgarte。临时方案是 `erp_pos if
solver != "admm" else 0.0`，不干净。

正确架构：
1. `batched_build_W_joint_space` 只算纯净的 W、v_free、v_current（无 bias）
2. PGS 路径：单独 kernel 或 kernel 参数加 Baumgarte bias 到 v_free
3. ADMM 路径：在 ADMM kernel 内部用自己的 compliance 模型（已有）

**触发条件**：下次接触求解器重构时（Q19）一并处理。

**Q20 — 与主流项目的功能差距 → Scene 重构方案已确定**

与 MuJoCo/Bullet/Drake/Isaac Lab 对比审查后，确定以下重构方案：

**已决定的 Scene 架构（解决 P0 #1 和 P1 #6）：**

引入 `Scene` 容器 + `CollisionPipeline` + `BodyRegistry`，一步到位支持多机器人。

核心数据结构：
```
Scene
  ├─ robots: dict[str, RobotModel]        # 多个有名字的机器人
  ├─ static_geometries: list[StaticGeometry]  # 墙壁/障碍物（shape+pose，无质量）
  ├─ terrain: Terrain                      # 地面/地形
  ├─ collision_filter: CollisionFilter      # 统一过滤
  └─ _registry: BodyRegistry               # 全局索引 ↔ (robot_name, local_idx)
```

关键设计决策（参考 Isaac Lab `InteractiveScene`）：
- **Scene 包含 RobotModel，而非 RobotModel 包含碰撞** — RobotModel 回归纯粹（tree + geometries + metadata），碰撞管理在 Scene 层。`contact_model` 和 `self_collision` 字段从 RobotModel 移除。
- **静态几何是独立类型** `StaticGeometry`（shape + pose + friction），不是 mass=∞ 的 Body — 不参与 ABA，不需要关节/积分。参考 PhysX `PxRigidStatic`。
- **多机器人现在实现** — 动力学 per-robot 独立（各自 ABA），碰撞全局统一（一个 CollisionPipeline）。BodyRegistry 管理全局索引映射。
- **API 风格**：dict，`sim.step({"robot_a": (q, qdot)}, {"robot_a": tau})`。单机器人有便捷包装 `Simulator.from_model()` + `step_single()`。

碰撞管线流程：
```
CollisionPipeline.detect(scene, all_X, all_v) → list[ContactConstraint]
  1. robot-body vs terrain       (ground_contact_query)
  2. robot-body vs static_geom   (gjk_epa_query)
  3. robot-body vs robot-body    (broad_phase + collision_filter + gjk_epa_query)
```

文件变化：
- 新建：`scene.py`（Scene, StaticGeometry, BodyRegistry）、`collision_pipeline.py`
- 改：`simulator.py`（step 用 CollisionPipeline）、`robot/model.py`（移除 contact_model/self_collision）
- 改：所有引用旧字段的 test

**剩余 P0/P1/P2 项（Scene 不解决的）：**

2. 力/力矩传感器 — 从 ABA 关节力矩提取。
3. Heightmap 地形 — 实现 HeightmapTerrain。
4. Mesh 碰撞 — 实现 MeshShape + BVH。
5. 球关节 — 新关节类型。
7-10. 长期完善项（MJCF、状态快照、电机模型、腱）。

**Q24 — GpuEngine dispatch 重构** 🔄 部分解决

已实现 solver dispatch（`solver="jacobi_pgs_si" | "admm"`）。
剩余 dispatch 维度：
- dynamics: aba / crba
- collision: sphere / analytical / gjk_gpu (未来)
- backend: warp / cuda

**触发条件**：下一个需要新 dispatch 维度的功能开发时。详见 memory `project_gpu_engine_dispatch.md`。

**Q31 — BatchBackend 整套退役（VecEnv 改用 GpuEngine）**

`WarpBatchBackend` 已退役（session 16，FreeJoint 坐标系修复后不再维护）。
剩余 BatchBackend 实现（NumpyLoopBackend / TileLangBatchBackend / CudaBatchBackend）
与 PhysicsEngine（CpuEngine / GpuEngine）功能重叠，每个 backend 独立实现完整物理步进
导致算法重复 4 处。

**目标**：VecEnv 直接包装 GpuEngine（已支持 num_envs=N 并行），退役 BatchBackend ABC 及
所有实现。合并后只剩两条管线：CpuEngine（精度/调试）+ GpuEngine（RL 训练）。

**退役顺序**：
1. ~~WarpBatchBackend~~ ✅ 已退役 (2026-04-02)
2. VecEnv 改用 GpuEngine 接口（`step(tau, dt)` 替代 `step(actions)`）
3. 退役 TileLangBatchBackend + CudaBatchBackend + NumpyLoopBackend
4. 删除 BatchBackend ABC + `get_backend()` 工厂

**触发条件**：下一次需要给 VecEnv 添加功能时（如 auto-reset、domain randomization）。

**Q25 — PGS 摩擦力通过力臂产生假角速度** ✅ RESOLVED (2026-04-01)

**现象**：球体静止放在地面，零初速度。数千步后角速度持续增长 → NaN。
仅影响 PGS/PGS-SI（CPU + GPU），ADMMQPSolver 无此问题。

**正反馈环路（4 步）**：
```
Step 1: float32 噪声 → v_tangential ≈ 1e-7（四元数→旋转→叉积中间运算）
Step 2: PGS 无死区 → lambda_t = -v_t / W_diag ≈ 1e-7（非零）
         代码：pgs_solver.py:447, solver_kernels.py:343
Step 3: 力臂放大 → torque = r × lambda_t, alpha = torque / I
         球体 r_arm=[0,0,-r], cross 产生 |torque| = r * |lambda_t|
         代码：pgs_solver.py:105-109, solver_kernels.py:233-234
Step 4: omega += alpha * dt → 下一步 v_t = omega × r 更大 → 正反馈
```
float64 噪声 ~1e-16，需 ~10¹⁰ 步才达宏观量级（不会遇到）。
float32 噪声 ~1e-7，~10⁴ 步即可增长到问题水平 → GPU 必修。

**ADMM 为什么没问题——三重机制**：
1. **R 正则化**（根本性）：`R_i = (1-d)/d × A_ii`，给摩擦力加二次代价。
   因 A_tt 含力臂贡献（球体 A_tt = 7/(2m) vs A_nn = 1/m），
   摩擦行 R 自动 3.5× 大于法向行，精确抵消力臂放大效应。
2. **切向阻尼参考**：`a_ref_t = -b × v_t`，目标是衰减切向速度，v_t ≈ 0 时不驱动力。
3. **圆锥投影**：‖f_t‖ ≤ μ·f_n（正确 Coulomb 锥），vs PGS 方盒 clip（√2 倍过冲）。

其中只有 R 是根本性的。ADMM 无 R 但完全收敛时等价于精确 Coulomb LCP，
与 PGS 有相同问题。ρ 只是算法参数，不改变最优解（仅加速收敛）。

**主流引擎解法对比**：

| 引擎 | 主要机制 | 辅助机制 | 参数 |
|------|---------|---------|------|
| Bullet | 摩擦 warmstart 每帧归零 | sleeping + split impulse | m_frictionCFM=0（默认关） |
| ODE | 全局 CFM + slip1/slip2（摩擦专用 CFM） | 两阶段法向→摩擦 | CFM=1e-5 (f32) |
| PhysX | 角阻尼 + sleeping | 摩擦仅后几轮迭代 | angularDamping>0 |
| MuJoCo | R 对角正则化（始终存在） | impratio 控制摩擦/法向 R 比 | solimp → R |
| Box2D | 累积冲量 clamp | warmstart 衰减因子 0.2 | — |
| AGX | SPOOK 物理合规性 | 混合直接/迭代 | compliance>0 |

**三层正则化分类**：
- 积分器层：角阻尼 τ=-k·I·ω（PhysX）— 全局，影响所有旋转
- 求解器层：CFM/R 加到 W 对角线（ODE/MuJoCo）— 仅影响约束行
- 约束公式层：a_ref 阻尼目标（MuJoCo）— 仅影响优化目标

**决策：方案 D = 摩擦行 per-row R + 摩擦 warmstart 归零**

方案 A（摩擦行 per-row R，移植自 ADMM）：
```python
# PGS 当前：W[i,i] += cfm (1e-6，聊胜于无)
# 修复后：摩擦行用 ADMM 同款自适应 R
if is_friction_row:
    R_i = (1-d)/d × |W[i,i]|    # A_tt 含力臂，R 自动更强
else:
    W[i,i] += cfm                 # 法向保持小 cfm
```

方案 B（Bullet 式摩擦 warmstart 归零）：
- 法向冲量正常 warmstart
- 摩擦冲量每帧重置为 0，阻断跨帧积累

**不选硬死区的原因**：v_t < ε 时摩擦关闭 → 低速滚动球进入死区后永远漂移。
per-row R 是连续的，摩擦始终存在但被合规性软化，低速球仍可减速停止。

**我们的 per-row R vs MuJoCo 的 per-contact R**：
我们 `R_i = (1-d)/d × A_ii`（每行各自 A_ii），MuJoCo 用 per-contact 常数 R。
对 Q25 而言我们的方案更优：A_tt > A_nn（含力臂贡献），摩擦行自动获得更强正则化，
精确抵消力臂放大。MuJoCo 需要手动调 impratio 达到类似效果。

**已实现（session 15）：**
- CPU PGS/PGS-SI：摩擦行 per-row R = (1-d)/d × |W_ii|（共享 ADMM 的 solimp 参数）
- CPU PGS：摩擦 warmstart 每帧归零（Bullet 方案，`friction_warmstart=False` 默认）
- GPU：solver_kernels / crba_kernels / solver_kernels_v2 同步修改
- 7 个新测试（球体静止稳定性 + 重球 + 减速验证 + 参数传递）
→ Moved to REFLECTIONS.md.

**已知权衡——摩擦精度 vs 稳定性（P4，先不优化）**：
R 合规性让粘着摩擦不再完全归零切向速度。实测：2 m/s 入射时残余 0.02 m/s（1%），
介于 MuJoCo (~3-5%) 和 Bullet (~0.05%) 之间。如需更硬摩擦，可调 solimp d_0：
- d_0=0.95（当前默认）→ ratio=0.053, 残余 ~1%
- d_0=0.98 → ratio=0.020, 残余 ~0.4%
- d_0=0.99 → ratio=0.010, 残余 ~0.2%（Q25 保护力度降低）

若调 solimp 仍不满足精度需求，备选方案：学习 PhysX 在积分器层加角速度阻尼
（`omega *= 1 - k_damp * dt`）。优点是不影响摩擦精度（R 可以调小或去掉），
缺点是全局影响所有旋转运动（空中旋转体也被阻尼）。可组合使用：小 R + 小角阻尼。

**Q26 — 几何系统重构（凸分解前提）** 🔄 GPU 多 shape 已实现 (2026-04-01)

**已完成（session 14）：**
- ConvexHullShape(vertices) + support_point (argmax)
- MeshShape 支持预加载顶点（不再是空壳）
- ShapeInstance.world_pose(X_body) 组合 body + shape 偏移
- CollisionPipeline/CpuEngine 遍历所有 shapes（不再只用 shapes[0]）
- aabb_half_extents() 考虑 origin_xyz 偏移
- 同 body shape 过滤（gid_i == gid_j 跳过）

**已完成（session 15）：**
- GPU 多 shape：展平 shape 数组（MuJoCo 模式）+ body_shape_adr/num 正向索引
- 动态 N² broadphase：bounding sphere + collision_excluded 矩阵，替代静态 pairs
- 碰撞 kernel `batched_detect_multishape`：ground 多 shape + body-body atomic counter
- 7 个新测试（多 shape 地面、动态 broadphase、碰撞过滤、稳定性）
- 649 测试全部通过

**测试覆盖盲区（session 16 补充）：**

高风险：
- GPU PGS 球体静止角速度稳定性（Q25 GPU kernel 改了但没测）
- Shape offset 接触点精度（只验证 count，没验证接触点世界坐标反映 offset）
- 非 Sphere shape（Box/Capsule）多 shape dispatch 路径
- 接触深度精度（depth 值是否物理合理，不只是 count>0）

中风险：
- CPU vs GPU 多 shape 一致性（CpuEngine 也支持多 shape）
- 接触缓冲溢出（max_contacts < 实际接触数时 graceful discard）
- 多 shape body-body 碰撞（两个多 shape body 互碰）
- Shape rotation（非零 origin_rpy）

**待完成：**
- 凸分解管线（V-HACD/CoACD → list[ConvexHullShape]）
- STL/OBJ 文件加载（当前 MeshShape 需要调用方传入顶点）
- GPU GJK kernel（ConvexHullShape 的 GPU 碰撞检测）

**Q26-gpu 设计方案（2026-04-01 确定）：**

调研了 8 个引擎（Newton/PhysX 5/Bullet3 GPU/MuJoCo C/MJX/MuJoCo Warp/Brax/Isaac Gym），
确认业界共识：展平 geom 数组 + 预分配接触缓冲 + atomic counter。

*1. 数据布局（MuJoCo 模式）：*
```
shape_type[nshape], shape_body[nshape], shape_params[nshape,4],
shape_offset[nshape,3], shape_rotation[nshape,9]
body_shape_adr[nb], body_shape_num[nb]  — 正向索引
```
展平存储，body 通过 adr+num 索引自己的 shapes。所有引擎（8/8）用此模式。

*2. 动态 broadphase（方案 A：N² 过滤）：*
```
for bi in range(nb):
  for bj in range(bi+1, nb):
    if excluded(bi,bj): continue
    if bounding_sphere_separated(bi,bj): continue  — 最便宜的检测
    slot = wp.atomic_add(n_pairs_active, env, 1)
    active_pair_bi/bj[env, slot] = bi, bj
```
替代当前静态 collision_pairs。天然支持多机器人（跨 robot body 对自动发现）。
四足 N=13 → 78 对，人形 N=30 → 435 对，10 四足 N=130 → 8385 对，GPU < 20μs。

*3. 接触缓冲（MuJoCo Warp 模式）：*
```
contact_count: (N_envs,) atomic counter，每步 zero 后 narrowphase 动态递增
max_contacts: 预分配上界，溢出静默丢弃
```
替代当前静态 1:1 slot 映射。Narrowphase 内层循环遍历 shape 对。

*4. 地面接触改进：*
对每个 contact body 遍历所有 shapes（不再只检测一个代表 shape）。
非 contact body 仍不检测地面（未来可升级为全 geom 检测）。

**备选方案 B（SAP broadphase，N > 200 时升级）：**
GPU radix sort + binary search overlap，O(N log N)。MuJoCo Warp 已验证。
当前 N < 60 不需要，留作 Phase 3 多机器人大规模场景时升级。
接口不变（都是输出 active pairs + atomic counter），可平滑切换。

**Q27 — 多物理子系统接口**

ForceSource/ConstraintSolver 绑死 RobotTreeNumpy，无法接入柔体/布料/流体。
需要 PhysicsSubsystem ABC + CouplingImpulse 接口。
详见 memory `project_multiphysics_architecture.md`。
暂不实现——等第一个非刚体子系统需求出现时再做。

**Q28 — GPU ADMM 多体同时接触发散** ✅ RESOLVED (2026-03-30)

**根因**：`solver_kernels_v2.py:batched_impulse_to_gen_v2` 力矩双重计算。
手动 `cross(r_arm, F)` 后又用 Plücker `transform_force_wp(Rinv, rinv, wrench)` 多加了 `rinv × F`。
对 body 不在原点的情况（z ≠ 0），水平法向冲量产生虚假垂直力矩 ∝ z × F_horizontal。

**修复**：impulse kernel 改用纯旋转（R^T @ F, R^T @ torque），不做 Plücker 平移。
RNEA 回溯的子→父 Plücker 变换保持不变。

**附带修复**：Q25 PGS 摩擦假角速度减半（同一 bug 让地面摩擦力矩 2x）。

**测试**：`test_q28_friction_divergence.py` 4 个测试（两球稳定性 + 单球角速度 + MuJoCo 精度回归）
→ Moved to REFLECTIONS.md.

**Q29 — GPU ADMM body-level vs joint-space Delassus** ✅ RESOLVED (2026-03-30)

**问题**：body-level Delassus 无法将接触力耦合到铰接关节。当腿段偏移与接触力
共线时（r × F = 0），RNEA 回溯给关节零力矩 → 关节冻结 → 四足稳态高度差 120mm。

**修复**：用 CRBA+Cholesky 全局替换 ABA+body-level 管线。新流程：
  1. CRBA → H, RNEA → C, Cholesky(H) → L（一次分解）
  2. qacc_smooth = L⁻ᵀ L⁻¹(tau-C)（复用 L，替代 ABA）
  3. W = J L⁻ᵀ L⁻¹ Jᵀ（joint-space Delassus，复用 L）
  4. dqdot = L⁻ᵀ L⁻¹(Jᵀλ)（复用 L，替代 RNEA backward + ABA H⁻¹）

**结果**：GPU 四足 z=0.4198 vs CPU z=0.4197（0.1mm 差距，之前 120mm）。
FK 从 3 次降到 1 次，ABA 完全消除。644 测试通过。
→ Moved to REFLECTIONS.md.

**Q30 — CPU ADMM vs MuJoCo 稳态穿透深度差 0.86mm** (2026-03-30)

场景：简易四足（13 body, 8 revolute, 8.4kg）从 z=0.45m 落地，ADMM solver, dt=2e-4。
稳态 base z：我们 0.419749，MuJoCo 0.418893，差 856µm。

**验证结果**：Delassus A_nn、质量矩阵 H、接触 Jacobian J 三者完全一致（ratio=1.0）。
差异全部来自 compliance 正则化 R 的计算方式不同。

**R 的公式差异（根因）**：

| | 我们 (per-row) | MuJoCo (per-contact) |
|---|---|---|
| 公式 | `R_i = (1-d)/d × A_ii` | R = 常数（所有 condim 行共享） |
| R 值 (法向) | 0.016 | 0.141 |
| 穿透深度 | 0.25mm | 1.11mm |
| 穿透是否依赖结构 | **否**（A 在平衡方程中抵消） | 是（∝ 1/A_nn） |

**推导**（Todorov 2014, MuJoCo 论文）：

设计目标：让 d ∈ [0,1] 控制约束满足比例 `a_i = d × rhs_i`。
反推 R：`A_ii/(A_ii + R_i) = d` → `R_i = A_ii × (1-d)/d`。

per-row R 代入后：`(A+R)_ii = A_ii/d`，穿透深度 = `(1-d)g/(kd²)`，
A 被消掉 → 穿透不依赖机器人质量/结构，仅由 compliance 参数(d,k)决定。
对刚体仿真，这是合理的——穿透是数值产物，应尽量小且一致。

MuJoCo 实现与论文推导不同：实现用 per-contact 常数 R，论文推导是 per-row。
per-contact R 让穿透 ∝ 有效质量，更适合模拟柔性接触面（solref/solimp 的双重用途）。

**参考**：
- Todorov (2014): per-row R 推导（我们的实现忠于此）
- MuJoCo 实现：per-contact R（工程简化）
- Levenberg-Marquardt：λ×diag(JᵀJ)（相同的 diagonal preconditioning 思路）
- Bullet/ODE：常数 CFM（per-contact，不缩放）

**结论**：非 bug，是建模选择差异。对刚体仿真，我们的 per-row 方案穿透更小(0.25 vs 1.1mm)、
条件数更可控、忠于原始论文推导。不修改。
**优先级**：P4（已充分理解，不影响功能）

**Q33 — 链式 chaos 放大 Q30 正则化差异，需监控触发器** (2026-04-07 session 21)

**现象**：`test_two_quadruped_collision::test_early_phase_separation_vs_mujoco`
在 2500 步比较时跨过了 atol=0.02 阈值（实测 22.3mm 超出）。session 21 调查
排除了 implicit damping 假设（damping=0 反而更糟），最终确认根因是 Q30
**per-row R vs per-contact R** 的微小每步差异，被两四足倾倒这个混沌系统
经 ~1500 步指数放大（每 100 步 × 1.1）到肉眼可见。

**为什么这是个 open question 而不是 resolved bug**：
1. **Q30 决策不变**：我们的 per-row R 物理上更优，不修改。
2. **测试本身已修正**：N_COMPARE 2500 → 2000 + atol 0.02 → 0.015，
   把比较窗口收缩到混沌主导之前的"确定性相位"。这是当前的临时缓解。
3. **风险**：未来加入更多复杂场景（多机器人、复杂 mesh、长时间模拟）很可能
   也会撞上同样的 chaos 放大问题。任何 Q30 量级的每步差异在 chaos 放大下
   都会跨过任何固定容差。
4. **没有通用解法**：要么继续 case-by-case 收紧测试范围（蔓延式 workaround），
   要么找一种"chaos-robust"的对比指标（守恒量、能量、统计量），要么真的去改
   per-row R。

**触发"重新评估 Q30"的条件**（达到任一条则重开 Q30）：
- 同一类失败在 ≥ 3 个独立 validation 测试中出现
- 在物理上"应该确定"的场景（非 chaotic 系统）出现 > Q30 量级的 CPU vs MuJoCo 差异
- 客户/用户场景报告与 MuJoCo 结果有显著差异

**可能的长期方案**（按风险递增排序）：
1. **Chaos-robust 指标**：把 chaotic 场景的对比从"轨迹差"换成"能量守恒"
   或"接触脉冲总量"。问题：实现复杂度高，丢失 step-by-step 调试能力。
2. **加 per-contact R 作为可选模式**：保留 per-row 默认，加 `regularization=
   "per_row"|"per_contact"` 开关。validation 测试用 per_contact 模式与 MuJoCo
   完全对齐，生产用 per_row。问题：维护两套求解器路径，每次改 PGS/ADMM 都
   要双写。
3. **混合 R**：法向用 per-contact（与 MuJoCo 一致），摩擦用 per-row（保留 Q25
   保护）。问题：Q30 论文推导的内部一致性被打破，精确意义需要重新论证。

**当前优先级**：P3（已知风险，已有缓解，但需要监控）。每次新增 validation 测试时
检查是否撞到这个问题；累计计数。

**关联**：Q30（per-row R 决策本身），Q25（per-row R 在 Q25 修复中是关键），
session 21 REFLECTIONS（详细调查脉络）。

---

## Performance / Optimization

**Q34 — Restitution 作为机械阻抗失配：物理材料 → e 的统一映射** (2026-04-07 session 21，**研究完成**)

**核心洞察（session 21 与用户讨论中浮现）**：MuJoCo 风格 spring-damper compliance
contact 里的阻尼系数 `b` 在数学上**等于 2× 接触界面的特征机械阻抗**。

具体推导：
- 1D 弹性杆：`c = √(E/ρ)` 是声速，`Z = ρcA` 是特征阻抗（单位 N·s/m）
- 等效集总参数：`k = EA/L`、`m = ρAL`、`ω_n = √(k/m) = c/L`
- 临界阻尼：`b_crit = 2√(km) = 2A√(Eρ) = 2ρcA = 2Z`
- **`b_crit ≡ 2Z`** — 这不是巧合，是同一个物理量在两种描述里的表达

物理解释：
- ζ = 1（默认临界阻尼）→ b = 2Z → **完美阻抗匹配** → 入射波被界面完全吸收
  → 没有反弹（默认 ADMM-C 不反弹的本质原因）
- ζ < 1 → b < 2Z → **阻抗失配** → 部分波反射 → 反弹
- ζ > 1 → 内部耗散比波传播还快（过阻尼）

两材料碰撞时的 e ≈ 声学反射系数 `(Z_A − Z_B) / (Z_A + Z_B)` 的某种函数。这给
"为什么钢球落到橡胶比落到钢弹得更高"一个**纯物理**的解释（钢-橡胶阻抗失配大）。

**研究 agent 调查结果（2026-04-07，完整报告 `docs/restitution_impedance_research_2026-04-07.md`）**

调查 8 个引擎（MuJoCo / Newton / Bullet / PhysX / ODE / Drake / Chrono SMC + NSC）
+ 7 篇论文（Hunt-Crossley 1975 / Lankarani-Nikravesh 1990 / Marhefka-Orin 1999
/ Falcon 1998 / Schwager-Pöschel 2007/2008 / Zhang-Sharf 2019 / Stronge 2000）
+ DEM 校准文献。

**纠正一个术语错误**：我之前说的 `e ≈ exp(-π·ζ/√(1-ζ²))` **不是 Hunt-Crossley
公式**，而是 **Kelvin-Voigt 半周期模型**。Hunt & Crossley 在 1975 原文里**明确
拒绝**了这个公式："during impact... half of a damped sine wave... is shown to
be logically untenable, for it indicates that the bodies must exert tension on
one another just before separating"。所以我的实验"早期脱离接触"假设是对的，
而且 Hunt-Crossley 1975 年就发现并解决了。文献里把半周期公式叫"Hunt-Crossley
closed form" 是普遍的误用。

**关键引擎发现**：
- **MuJoCo / Newton / Drake** 都不暴露 user-facing `restitution`，只让用户调
  damping ratio 或 dissipation `d` (s/m)。Drake 文档原话："bounce velocity
  bounded by 1/d"。
- **Bullet / PhysX / ODE** 暴露 `restitution`，但它们都是 velocity-impulse
  Newton 模型，不是 spring-damper compliance，所以是直接 LCP 速度偏置，没有
  spring-damper 翻译问题。
- **Project Chrono SMC** 是**唯一**一个 spring-damper compliance solver 暴露
  user-facing `e` 的引擎，内部用 Lankarani-Nikravesh 翻译。但 LN 公式在
  e<0.7 时不准（文献已知）。所以 Chrono SMC 是 C.2 的最直接先例，但用了
  已知不准的公式。

**关键论文发现**：
- **Zhang & Sharf 2019** "Exact restitution and generalizations for the
  Hunt-Crossley contact model" — 第一个**精确解析**的 (e, λ) 闭式，用 inverse
  restitution coefficient 级数，**有效在高耗散区**。这是 C.2 选项 B（闭式映射）
  应该用的公式，**不是** Lankarani-Nikravesh（不准），更不是 Kelvin-Voigt
  半周期（错的术语）。
- **Schwager & Pöschel 2008** "delayed recovery" — 显式处理早期脱离接触边界
  的精确解，针对 Hertzian viscoelastic 接触。
- **Marhefka & Orin 1999** — 机器人圈最常引，能量自洽全 e 范围，显式 force→0
  分离边界。
- **Drake 的合并规则**：`d_combined = (k₂/(k₁+k₂))·d₁ + (k₁/(k₁+k₂))·d₂`
  （刚度加权平均），物理上比 Bullet 的 multiply / PhysX 的 average/min/max
  更对（这是串联弹簧的正确合并）。

**关键阻抗发现（Q3 的核心）**：
> "**No paper found explicitly equates phenomenological contact `b` with `ρcA`.**
> None of the surveyed engines bridges contact damping to characteristic
> acoustic impedance."

DEM 文献从 (E, ν, ρ, R) 算 (k, b)，但用的是 Hertzian/Lankarani-Nikravesh
公式，**不是**波阻抗解释。Drake hydroelastic 把 k 跟物理量挂钩（pressure
field），但把 d 留作 phenomenological。Hogan 1985 robot impedance control 是
**对偶问题**（prescribe 期望的 (M,B,K)），从来不去把 B 跟材料 ρcA 关联。

**结论：物理是熟知的（continuum mechanics 教科书 + 声学 + DEM），但作为
user-facing API 设计原则被显式叙述出来这件事没有先例**。所以"我们的项目第一个
把接触阻尼作为物理机械阻抗显式暴露"的卖点**是站得住脚的**。

**Session 21 决定**：

✅ **写**。先写一份 blog post，再扩展成 workshop paper 级别。
- Blog：早期发布，门槛低，可以快速 iterate 反馈
- Workshop paper：blog 成熟后扩展，目标 IROS / RSS / SimSI workshop
- 时间表 / 大纲 TBD（"anyway，可以再议"）

**对 C.2 实装的影响（暂不锁定）**：
- 如果 paper 是目标，C.2 的实装应当成为 **demo**，而不是只在 docstring 写一句
- 这意味着 ADMM-C 暴露 `e` 入口（决策 B 中的 B2/B3）从"诚实就好不暴露"变成
  "这是论文的核心 demo，必须实装"
- combine rule 应该走 Drake stiffness-weighted 风格（决策 A 中的 A2），而不是
  PhysX/Bullet 的 multiply
- 物理材料量入口 `(ρ, E, η)` 也应该作为 first-class API
- **但**：以上都暂不锁定，等 blog/paper 大纲定下来再决定 C.2 具体形态

**短期 C.2 实装方向（保留选项）**：
1. **A1 (PGS Newton restitution + per-shape e + multiply)**：能用、跟主流一致、
   不押宝在 paper 上
2. **A2 + B2 (Drake-style combine + Zhang-Sharf 闭式 ADMM-C 翻译 + 物理量
   入口)**：是 paper 的 demo
3. 在 C.2 启动前先写 blog 大纲，反过来约束 C.2 的设计

**架构约束（session 21 用户追加澄清）**：

C.2 不需要等 blog/paper 大纲。用户的原话："C2 可以有多种实装，只要有咱们
workshop paper 中的哪种求解 backend 就行"。

含义：C.2 应当**架构上支持多个 restitution backend**，并行存在，可在初始化
时切换。**至少一个 backend** 是 paper demo（阻抗 / Zhang-Sharf 版本），其他
backend 可以是主流风格（per-shape e + multiply combine + Newton 速度反转）。

这跟现有的 solver dispatch 架构（`PGSSplitImpulseSolver` / `ADMMQPSolver` /
GPU `solver="jacobi_pgs_si"|"admm"`）是同一种设计模式：dispatch 一个新维度
"restitution_model"。两种 model 共存，默认值待定（取决于哪种更稳健）。

**candidate restitution backends**（最少实装哪些 TBD）：
1. **`phenomenological`**：per-shape `e` + multiply combine + PGS Newton 行
   `v_n_post ≥ -e × v_n_pre`。**主流默认风格**，给不在乎物理的用户用。
2. **`impedance` (paper demo)**：per-shape `(ρ, E, η)` 物理材料属性 +
   Drake-style stiffness-weighted combine 算 Z_eff + Zhang-Sharf 2019 闭式
   反推 ADMM-C 的 dampratio (或直接 b)。这是 workshop paper 的核心 demo。
3. **`hybrid` (可选)**：per-shape `e` 入口（保持 UX 一致），但内部用
   Zhang-Sharf 翻译到 spring-damper 参数，combine 用 Drake 风格。介于 1 和
   2 之间，没有"物理材料量"入口但有"正确的 e ↔ damping 翻译"。

**对 C.2 实装时间的解锁**：
- 之前的依赖 `blog_outline → C.2_path` **解除**
- 现在的依赖 `C.2_architecture must support multiple backends → C.2 implementation`
- 具体哪几个 backend 先实装、哪个是默认，等到 C.2 启动那一刻再定
- 但**架构上必须支持 dispatch**，不能写死单一 model

**关联**：
- `docs/restitution_impedance_research_2026-04-07.md`（完整研究报告）
- session 21 REFLECTIONS（待补，下次 session 起头时一起写）
- session 20 InterfaceMaterial 设计（"interface vs bulk material"在这里被统一）
- Q33 chaos + Q30 正则化（同一 session 的另一条线）
- C.2 PGS Newton restitution（短期实装路径，待 blog 大纲约束）
- memory `project_impedance_restitution_insight.md`（项目记录）

**当前优先级**：blog 大纲先写，C.2 实装方向待定。**新颖性已确认**。

---

**Q22 — DynamicsCache 与 ABA/CRBA 的 FK 重复计算**

`DynamicsCache.from_tree()` 计算 `X_world`（FK）和 `body_v`，但 `tree.aba()` 和
`tree.crba()` 内部各自重算 FK（X_J, X_up, Pass 1）。当前每步有 2-3 次重复 FK。

- 当前：CPU 上 FK 是 O(n) 且 n 小（<30），不是瓶颈
- 优化方案：让 ABA/CRBA 接受预算好的 `X_up[]` 数组，跳过 Pass 1
- GPU 收益：减少 kernel launch 次数
- **触发条件**：GPU solver 开发时（Phase 2i），FK 成为热路径时实施

---

## Infrastructure

**Q10 — Unit tests are missing** ✅ RESOLVED
Tests added across Phase 2a/2b/2c + session 2 补全：
- `tests/test_free_fall.py` — 解析自由落体 vs ABA（2 tests）
- `tests/test_body_velocities.py` — body velocity API（4 tests）
- `tests/test_urdf_loader.py` — URDF loader（6 tests）
- `tests/test_simulator.py` — Simulator 编排（4 tests）
- `tests/test_contact.py` — PenaltyContactModel（9 tests）
- `tests/test_joint_limits.py` — 关节限位 + 阻尼（14 tests）
- `tests/test_aba_vs_pinocchio.py` — ABA vs Pinocchio（5 tests）
- `tests/test_self_collision.py` — AABB 自碰撞（13 tests）
- `tests/test_integrator.py` — SemiImplicitEuler + RK4（11 tests）
Total: 68 tests，全部通过。
→ Moved to REFLECTIONS.md.
