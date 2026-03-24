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

8. ~~**碰撞过滤掩码**~~ ✅ RESOLVED — `physics/collision_filter.py` 实现了三层过滤：
   auto-exclude（parent-child）、bitmask（group/mask uint32）、explicit exclude set。
   集成到 `AABBSelfCollision`、`LCPContactModel`、`load_urdf(collision_exclude_pairs=...)`。
   参考：Drake CollisionFilterDeclaration + MuJoCo contype/conaffinity。

9. **接触维度控制** — MuJoCo `condim` 可选 1D（仅法向）/3D（+摩擦）/4D（+扭转摩擦）/6D（+滚动摩擦）。
   当前我们固定 3D（normal + 2 tangent）。

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

---

## Architecture / Refactor

**Q19 — physics/ 接触模块重构（GPU contact 完成后）**

GPU Jacobi PGS + ADMM 求解器实现完成后，需要一次认真的重构：
- `contact.py`（486 行）拆分为独立文件或 `contact/` 子包（Penalty/LCP/Null 分离）
- GPU backends 各自独立实现的 penalty contact kernel 统一为可插拔接触管线
- `collision.py` + `collision_filter.py` + `broad_phase.py` 考虑合并为 `collision/` 子包
- `physics/solvers/` 接口验证（PGS / Jacobi PGS / ADMM 统一 ABC）
- **触发条件**：GPU ADMM kernel 完成且测试通过后

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
