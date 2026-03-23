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

**Q16 — CRBA vs ABA：GPU 上的前向动力学算法选择**

当前所有 GPU 后端使用 ABA（O(n)，顺序标量运算），tensor core 完全空闲。
CRBA 将前向动力学转化为密集矩阵问题（nv × nv），可利用 tensor core。

设计问题：
- **ABA/CRBA 自动切换阈值**：nv 多大时 CRBA 的 O(n²+nv³) 矩阵开销被 tensor core
  吞吐弥补？需要在不同 nv (10/20/30/50) 和 N (100/1000/4096) 下实测。
- **分组策略**：大型机器人的子树分割应自动还是手动？自动分割的启发式
  （平衡组大小 vs 最小化组间耦合边数）？
- **精度**：CRBA Cholesky 在 float32 下的数值稳定性？对于 nv=50 的 H 矩阵
  条件数是否需要 float64 或混合精度？
- 参考：Pinocchio 同时实现了 `aba()` 和 `crba()`，Drake 的 `CalcMassMatrix()` 也是 CRBA。

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
