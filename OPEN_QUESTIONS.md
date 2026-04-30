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

**Q7 — Mesh collision geometry (`<geometry><mesh/>`)** ✅ RESOLVED (session 26)
CPU path complete: URDF `<mesh>` → trimesh 加载 → ConvexHullShape → GJK/EPA 碰撞（地面+体-体）。
CPU body-body 碰撞也从球近似升级为 GJK/EPA per-shape。
- `robot/mesh_loader.py` 隔离 trimesh 依赖（选装 `pip install .[mesh]`）
- `physics/` 层不依赖 trimesh，只看到 ConvexHullShape(numpy array)
- **GPU 端已实现**：见 Q41 ✅ RESOLVED (session 29/32)。

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

**Q31 — RL 环境层架构重构：VecEnv → Manager-based RLEnv** (2026-04-03) 🔄 部分解决

**已完成**：
  - TileLang/CUDA/Numpy backends + BatchBackend ABC 全部退役（session 16）
  - VecEnv 从代码库删除；`rl_env/__init__.py` 留有占位注释
  - GpuEngine `reset_envs(env_ids)` 已实现（session 13）
  - `rl_env/obs.py` 已加入 RL observation schema 草案（2026-04-27）：
    `ObsSchema / ObsFieldSpec / locomotion_obs_schema()` 明确默认 field order 为
    `base_lin_vel_body -> base_ang_vel_body -> base_orientation_quat_wxyz -> joint_pos -> joint_vel -> optional contact_mask`

**当前 `Env` 类的定位（已明确，2026-04-16）**：
  - `rl_env/base_env.py` 中的 `Env` **不是 RL 训练环境**，而是：
    - **CPU 精度验证工具**：CpuEngine 路径，可对照 MuJoCo/Bullet 验证物理
    - **单步交互接口**：手动调试、可视化、快速原型
  - 不应删除；应在 docstring 中明确标注用途
  - 命名维持 `Env`（不改名 `RLEnv`）

**为什么不急于实现 Manager-based RLEnv（2026-04-16 决策）**：
  - GpuEngine 功能尚未完整：缺少 decimation、contact force 便捷接口、
    runtime 参数修改（domain randomization 必需）
  - 先补齐 GpuEngine GPU 后端能力，再做 RL 环境层包装
  - 包装层薄（Isaac Lab 模式），一旦 GPU 能力齐全可以快速实现

**目标架构（不变，等 GpuEngine 能力齐全后实施）**：
```
RLEnv (Manager-based, Gymnasium 兼容)            ← 待实现
  ├── ActionManager      — action scaling → tau
  ├── ObservationManager — q/qdot/contacts → obs tensor (CUDA)
  ├── RewardManager      — weighted sum of reward terms
  ├── TerminationManager — OR of done conditions
  ├── EventManager       — domain randomization
  └── CommandManager     — goal/velocity commands
  └── GpuEngine(num_envs=N)  ← 唯一物理调用

Env (CPU 调试路径，保留)                          ← 已有
  └── Simulator → CpuEngine
```

**GpuEngine 需补齐的 Gap（优先级排序，这是当前工作重点）**：
  1. Decimation 支持：`step_n(tau, n_substeps)` → ✅ 已有 `step_n`
  2. Contact forces 便捷接口：每 env 每接触对的力向量（GPU→obs tensor）（中）
  3. StepOutput v_bodies / X_world 零拷贝（Warp→Torch DLPack）（中）
  4. Runtime 参数修改：质量、摩擦系数等 per-env 随机化（高，domain rand 必需）
  5. 多机器人 merged model：N env × M robot（高，多智能体必需）

**触发条件**：GpuEngine 上述 gap 补齐后，立即着手 RLEnv 实现。

**RL obs schema 当前合同（2026-04-27）**：
- quaternion 统一使用 scalar-first `[w, x, y, z]`，字段名写成
  `base_orientation_quat_wxyz`，与 `FreeJoint` / `physics.spatial` 对齐。
- normalization 采用显式 scale 语义：`obs[field] = raw_term * field.scale`。
  默认 scale 保持 `1.0`，不在通用层写死 robot/task-specific range。
- contact mask 是 optional field，语义为 `contact_body_names` 顺序下的 binary
  `0.0/1.0`；2026-04-27 已通过 published `contact_mask` 接入
  `StateSampleView.contact_mask` / `ContactStateReading.contact_mask`。GPU/RLEnv
  路径应消费该 published field，不应从 private contact scratch 推断。
- 现有 CPU debug `Env` 可通过 `obs_cfg_from_schema()` 复用该 schema，但该 schema
  不是最终 Manager-based `RLEnv` 的实现本体。

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

**为什么 GPU 完全替换 ABA？（L 因子复用，根本原因）**：

GPU 上将 ABA 全部替换为 CRBA+Cholesky，根本原因**不是** CRBA 本身更快（nv≤30 时
CRBA 比 ABA 慢 0.96×），而是 **L 因子单次分解 + 三处复用**（见 Q29 RESOLVED）：

```
CRBA → H,  RNEA → C,  Cholesky(H) → L    ← 一次分解

1. smooth dynamics:  qacc = L⁻ᵀ L⁻¹(τ - C)          — 替代 ABA forward pass
2. Delassus 构建:    W = J L⁻ᵀ L⁻¹ Jᵀ               — 替代 body-level Delassus
3. 冲量应用:         dqdot = L⁻ᵀ L⁻¹(Jᵀ λ)          — 替代 RNEA backward + H⁻¹
```

有接触时三处复用同一个 L，总开销低于 3 次独立 ABA。ABA 的树形递归（前向+后向）
在 GPU 上并行度差；密集 H 矩阵也更 tensor core 友好（nv×nv 矩阵乘，无树依赖）。
详见 Q29 RESOLVED（session 13）和 REFLECTIONS.md session 24。

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
      - union 字段清单：mu_sliding, mu_rolling, mu_spin, restitution, compliance,
        **margin**（session 30 新增：convex margin 属于界面层，不同物理域各自解释）
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

**Q35 — GPU non-flat terrain support** (2026-04-08, session 23)

`GpuEngine.__init__` 现在 hard-fail 任何非 `FlatTerrain`（NotImplementedError）。
之前是 silent wrong physics——把 HalfSpaceTerrain 当 z=0 平地处理，contact normal
固定 (0,0,1)。修复见 session 23 commit。

GPU 要支持 HalfSpaceTerrain，需要：
- `static_data` 增加 `terrain_normal` (vec3) 和 `terrain_point` (vec3) 字段
- 调整 `analytical_collision.py` 的 `*_vs_ground` 函数为 `*_vs_halfspace`，
  接受任意 plane normal 而非硬编码 -z support direction
- collision_kernels 把固定的 `wp.vec3(0,0,1)` 法向改为从 static_data 读
- HeightmapTerrain 是另一个独立的 Q（需要 BVH，见 Q32）

**触发条件**：用户需要 GPU 上跑斜面 RL 任务时。当前 CpuEngine 已支持。

**Q36 — Multi-robot × multi-shape combinatorics test coverage** ✅ RESOLVED (session 25)

Done as B(5) staircase: 36 tests across 6 files covering 8 dimensions
(multi-body, multi-shape, 3 robots, custom filter, mixed shapes, ground,
CPU/GPU, multi-env). Added `GpuEngine.query_contacts()` API. No P0 bugs
found. Discovered CPU body-body uses body-level approximation (not per-shape).
→ See commits a7ba463–f8d2dc6, REFLECTIONS.md session 25.

**Q37 — CRBA Cholesky numerical conditioning test coverage** ✅ RESOLVED (2026-04-09 session 24)

Done in session 24 as B(6). 47 new tests in
`tests/integration/test_crba_cholesky_conditioning.py`. Adopted Wilkinson
backward error methodology as standing practice (see REFLECTIONS session 24
+ feedback memory `feedback_numerical_stability_wilkinson.md`).

Test classes shipped:
- Class 1 (13 tests): Wilkinson backward error on synthetic SPD via direct
  GPU `_chol_factor` / `_chol_solve` kernel calls. Sweep cond ∈ {1, 1e2, 1e4}
  + clamp activation test on diag(1,...,1,1e-10). Backward error confirmed
  κ-independent → kernel is backward stable.
- Class 2 (11 tests): CRBA H matrix structural properties (symmetry, PD
  margin, three-method cond cross-validation, alpha lever check).
- Class 4 (13 tests): CRBA vs ABA agreement at n_links ∈ {2, 4, 6, 8}
  with zero/non-zero qdot, Newton residual `‖H q̈ + C - τ‖`, 20-trial
  random battery.
- Class 3 (6 tests): quadruped near-singular fixture (cond ~6e3, the
  quadruped fixture is NOT actually near-singular) + chain high-cond
  regime (n=12 α=1.5 → cond 4e7).
- Class 5 (4 tests): GPU qacc accessor cross-check. Added
  `qacc_smooth_wp` and `qacc_total_wp` properties to GpuEngine for
  downstream RL acceleration penalty / sysID. Computed via new
  `_compute_qacc_total` kernel after impulse apply.

Surprising findings:
- Quadruped fixture cond surface is FLAT (2e3-6e3 across joint space).
  Calf=0 is the worst-case but only by 4x. "Near-singular" doesn't
  exist on real robot fixtures — synthetic SPD (Class 1) is the only
  way to test high-κ regimes.
- GPU `_chol_factor` is truly backward stable (Wilkinson Test 2 passes
  independent of κ from 1 to 1e4).
- Regularization clamp at reg=1e-6 is real but rarely-activated. Original
  concern that "cond > 1e6 = silent wrong physics" was overblown — silent
  wrong physics requires intentionally constructed matrices (e.g.,
  `diag(1,...,1,1e-10)`), not just high κ.

Full results and methodology: REFLECTIONS.md session 24.

---

**Q38 — Three Cholesky use-site disambiguation in GpuEngine** (2026-04-09 session 24)

Q29 architecture: one Cholesky factor of H per step is reused for three
solves: smooth dynamics (`H⁻¹(τ-C)`), Delassus build (`W = J H⁻¹ Jᵀ` via
per-row solve), and impulse apply (`Δq̇ = H⁻¹ Jᵀ λ`). The "three uses
share one L" property is what makes the GPU pipeline efficient.

To **verify** through testing that each solve actually used the same L
(not just that the final answer is consistent), you'd need inspection
buffers inside the solver kernels: `qdot_after_smooth`,
`qdot_after_delassus_solve_each_row`, etc. These don't exist — the three
solves happen inside fused kernel sequences that update qdot in-place
without checkpointing intermediate states.

Session 24 Class 5 ships only the partial check: `qacc_smooth_wp` ≡
`qacc_total_wp` when there's no contact, and `qacc_total - qacc_smooth =
dqdot/dt` when there is contact. This catches gross mistakes (e.g.,
qacc_total uses the wrong sign on dqdot) but cannot disambiguate the
three internal Cholesky solves.

**Decision**: deferred. Only worth implementing if we discover an actual
three-use-site divergence bug, since the fused-kernel design is correct
by construction. Adding inspection buffers is a real refactor (~50 LOC
across crba_kernels + scratch + gpu_engine), not a test.

**Trigger to revisit**: any GPU vs CPU divergence on contact-rich
scenarios that's NOT explained by f32 precision (which Class 5 covers).

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

**Q31 — BatchBackend 整套退役（VecEnv 改用 GpuEngine）** ✅ 全部完成

`WarpBatchBackend` 已退役（session 16）。
TileLangBatchBackend + CudaBatchBackend + NumpyLoopBackend + BatchBackend ABC + VecEnv
均已删除（session 16）。双轨管线已建立：CpuEngine（精度/调试）+ GpuEngine（RL 训练）。

RL 环境层包装（Manager-based RLEnv）待 GpuEngine GPU 能力补齐后实施 — 见上方主条目 Q31。

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

**测试覆盖盲区（session 16 补充）：** ✅ ALL RESOLVED (sessions 21–22)

实际状态：8 项盲区在 session 15 ede06cb 已经写了 `tests/gpu/collision/test_gpu_multishape_coverage.py`
（778 行，覆盖全部 8 项），但**作者忘了划掉清单**。Session 21 也没 grep 现状就在
独立文件 `test_q25_gpu_multibody.py` 重写了 B.1 的多体扩展。Session 22 才发现这个 lapse
并系统地把 11 个原有测试加固到 31 个（B.2/3/5/7/8），全部使用 discriminator math 而非
"count >= 1" / "no NaN" 弱断言。

| 盲区 | TestClass | 测试数 | 加固内容 |
|------|-----------|--------|---------|
| B.1 GPU PGS 球体角速度稳定性 | TestGpuQ25FrictionStability | 3 | session 15+session 21 多体扩展 |
| B.2 Shape offset 接触点精度 | TestShapeOffsetContactPrecision | 4 | +y 轴对称 + body R × offset 组合（session 22） |
| B.3 非 Sphere 多 shape | TestNonSphereMultiShape | 5 | tilted box / tilted capsule / shape grouping（session 22） |
| B.4 接触深度精度 | TestContactDepthAccuracy | 3 | 已强 |
| B.5 CPU vs GPU 一致性 | TestCpuGpuMultiShapeConsistency | 4 | per-contact + body-body 跨引擎对比（session 22） |
| B.6 Contact buffer 溢出 | TestContactBufferOverflow | 2 | 已合理 |
| B.7 多 shape body-body | TestMultiShapeBodyBody | 5 | sphere-sphere 几何 + 多对 shape filter（session 22） |
| B.8 Shape rotation | TestShapeRotation | 5 | origin_rpy 真实测试 + 组合 R + settling 几何（session 22） |

总计：11 → **31 测试**。Settling test 的轨迹诊断保存在 `tests/fixtures/rotated_box_settling.png`。
延后到 Phase 3 渲染就绪后由用户视觉验证的场景：B.5-c (CPU/GPU 长轨迹)、B.7-c (separation
增大动力学)。原因：chaos / 多解动力学不适合 numerical assertion。

**待完成（功能扩展，不是测试盲区）：**
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

**Q39 — Contact query API refactor** ✅ RESOLVED (2026-04-10, session 25)

Done. `ContactInfo` moved to `engine.py` (shared). `query_contacts()` added as
abstract method on `PhysicsEngine` ABC, implemented by both `CpuEngine` (converts
from internal `ContactConstraint`) and `GpuEngine` (reads warp buffers).
B(5) tests migrated from `cpu._detect_contacts()` to public `cpu.query_contacts()`.

Remaining items deferred to RL env phase:
- Batch NumPy return (structured arrays instead of Python list)
- Multi-env batch query (all envs at once)

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

**Q40 — Test suite execution time blocking development velocity** 🔄 部分解决 (2026-04-20)

**原始问题**：`python -m pytest tests/ -m "not slow"` 运行 ~1049 tests 需要 ~15 分钟，
严重拖慢 commit 节奏。

**已解决（2026-04-20）**：
- 4 个持续 ≥20s 的 GPU multi-step 测试加了 `@pytest.mark.slow`：
  - `TestMassSplitting::test_100_steps_no_nan`
  - `TestColoredPGS::test_100_steps_no_nan`
  - `TestCrossSolverAgreement::test_single_sphere_all_solvers_agree`
  - `TestStep5MixedShapesGround::test_simulation_stable_100_steps`
- commit gate (`not slow`) 从 ~15 分钟降到 ~3–5 分钟
- CLAUDE.md 更新：commit gate = `not slow`，push gate = 全量

**双 gate 规则（已写入 CLAUDE.md）**：
- commit 前：`python -m pytest tests/ -m "not slow" -v`（~3–5 min）
- push 前：`python -m pytest tests/ -v`（~21 min，全量）

**残余问题**：
- 20s 阈值是经验值，换慢 GPU 后边界测试可能需要重新评估
- `test_max_qdot_bounded`（50 steps，~10–30s）未标 slow，若未来超阈值需补标
- cross-solver agreement 不再进入 commit gate（合理 tradeoff，已显式承认）

**优先级**：P3（commit gate 已可接受，全量 push gate 保证覆盖）。

**Q42 — Contact Manifold Generation: Face Clipping + Edge-Edge** (2026-04-12, session 27)

**问题**：`gjk_epa_query()` 只生成单个接触点（两个 support point 的中点），
对 face-face、face-edge、edge-edge 接触不正确。Box-Box 碰撞尤其明显：
缺乏多点接触导致 box 无法稳定放置，力矩传递错误。

**决策**（session 27）：
- CPU：方案 B — GJK/EPA + 面识别 + Sutherland-Hodgman clipping + edge-edge 最近点
- GPU：方案 B (首次接触 / 大位移) + 方案 C (持久 manifold 增量更新)
- 新增 `FaceTopology` 接口：Box 6面预计算、ConvexHull 从 scipy 提取面拓扑

**子任务**：
1. ✅ 方案调研：7 引擎 (Bullet/MuJoCo/Coal/PhysX/ODE/Box2D/Jolt) 对比
2. ✅ CPU 实现：FaceTopology + build_contact_manifold + S-H clipping + coplanar face merging
3. ✅ GPU 方案 B：Box-Box SAT kernel (single-point, 15-axis, depth+normal+contact_point)
4. ✅ GPU box-box multi-point manifold：S-H face clipping on GPU (session 29)
   - `ClipPoly` / `BoxBoxManifold` structs + `box_box_manifold()` in analytical_collision.py
   - Produces 1-4 contacts: face-face (4), face-edge (2), edge-edge (1)
   - `box_ground_manifold()` function implemented + unit-tested, but NOT wired into kernel
     (solver instability with multi-point ground + body-body contacts; see Q45)
5. ⬜ GPU 方案 C：per-pair manifold cache (4 points, fixed buffer, temporal coherence)
6. ⬜ `find_support_face` O(log F) 优化 (Gauss map / hill climbing) — 当 ConvexHull >200 面时

**完整设计线索**：REFLECTIONS.md session 27

**Q43 — find_support_face Optimization for High-Poly ConvexHull** (2026-04-12, session 27)

当前 `FaceTopology.find_support_face()` 用暴力扫描 O(F)。对机器人 URDF
典型的 <50 面 ConvexHull 足够，但 >200 面时需要优化：

- **邻接图爬山** O(√F)：利用凸多面体上 dot product 的凸性，从上一帧最佳面
  开始 hill climb。需要预计算面邻接图。Jolt/Bullet 方案。
- **Gauss map 层次搜索** O(log F)：面法线映射到单位球，BSP 树搜索。PhysX 方案。
- **Edge-edge 轴剪枝**：Gauss map arc overlap test (Dirk Gregorius GDC 2013)
  可将 O(E_A × E_B) SAT 降为 O(E_A + E_B)，对 GPU SAT 路径尤其重要。

**触发条件**：使用 >200 面 ConvexHull 且碰撞检测成为性能瓶颈时。
**优先级**：P2（优化，非正确性）。

**Q44 — Convex Margin (Jolt-style convex radius)** ✅ RESOLVED (2026-04-20, sessions 31/32)

当前 GJK 检测到穿透后必须进入 EPA（expanding polytope），EPA 有已知的数值
退化问题（degenerate simplex → 错误深度/法线，见 session 27 EPA 诊断）。
Session 27 已通过修复 GJK 退化分支 + EPA hexahedron 初始化解决最严重的 case，
但 EPA 本质上是数值不稳定的算法。

**Session 30 新发现**：sphere(r=0.1) vs box-as-ConvexHull(half=0.2) 在 GJK
simplex 有面经过原点时，EPA 返回 depth=0.02（正确 0.05），normal 偏 45°+。
8 引擎调研确认：convex margin 是业界主要应对策略，但深穿透仍需 EPA 鲁棒性。
**两道防线缺一不可**。

**实施计划（session 30 确定）**：
1. EPA 鲁棒性：4-point simplex 退化面检测 → hexahedron 重建 + 主循环跳过退化面
2. Convex margin：新增 `gjk_distance()` closest-distance 模式，浅接触不走 EPA
3. 测试：EPA 鲁棒性 17 tests + margin 24 tests + MuJoCo 响应对比 10 tests
4. 详细 plan 在 `.claude/plans/floating-drifting-yeti.md`

**Convex margin 方案（Jolt / Bullet / PhysX）**：所有形状内缩小量 margin ε，
GJK 在"近分离"状态工作（closest-point 问题，不需要 EPA）。穿透深度 =
margin - gjk_distance。只有深穿透（distance > margin）时才进入 EPA。

**架构归属（session 30 决定）**：margin 概念上属于 `InterfaceMaterial`（界面层，
session 20 设计）。不同物理域各自解释 margin：
- 刚体凸-凸：GJK Minkowski erosion
- 刚体凸-mesh (BVH)：AABB 膨胀 + 三角面 margin
- FEM 柔体表面：变形 mesh 上的接触距离阈值
- 流体 SPH：交互半径（per-particle）
当前实现：全局默认 `CONTACT_CONVEX_MARGIN = 1e-3`（`contact_tolerances.py`），
InterfaceMaterial 实装时迁入为 per-shape 字段。

**优点**：
- GJK closest-point 数值稳定性远优于 EPA
- 法线 = closest-point 方向，帧间连续、不跳变
- 大幅减少 EPA 触发频率（Jolt 经验：>95% 接触在 GJK 阶段解决）
- 不会引入额外抖动（反而减少抖动）

**代价**：
- 形状棱角被"磨圆"（Minkowski erosion），margin 过大会影响物理行为
- 新增 `gjk_distance()` + `_support_shrunk()` 函数
- 接触检测阈值从 `depth > 0` 变为 `distance < margin`

**默认 margin 值**：1e-3 m（对 r=0.05 形状占 2%，仿真级可接受）

**参考**：Jolt `ConvexShape::GetSupportFunction` (convex radius = 0.05 default)、
Bullet `btConvexInternalShape::getMargin()`、PhysX contact offset。

**触发条件**：已触发（session 30 bug 复现）。
**优先级**：**P1**（从 P2 升级 — EPA 退化 bug 在 capsule-convexhull 路径已暴露）。

**Q41 — GPU ConvexHullShape 碰撞支持** ✅ RESOLVED (2026-04-20, sessions 29/32)

四项原始需求全部实现：
1. ✅ `SHAPE_CONVEXHULL = 5`（`static_data.py:41`，`analytical_collision.py:34`）
2. ✅ `hull_vertices / hull_vert_adr / hull_vert_count` flat arrays（`static_data.py:165-173`）
3. ✅ GPU GJK kernel：`_convexhull_support_local` + `gjk_closest_distance` + `gjk_epa_penetration`
4. ✅ narrowphase dispatch：CONVEXHULL × {SPHERE, BOX, CAPSULE, CONVEXHULL}（GJK/EPA + S-H face clipping）

额外实现（超出原始需求）：
- `hull_hull_manifold`：S-H face clipping，1–4 接触点（session 32, commit 211fdba）
- `convexhull_ground_manifold`：顶点枚举多点（session 29）
- 13 GPU 测试（`tests/gpu/collision/test_gpu_convexhull.py`）

→ 详细设计见 REFLECTIONS.md session 29/32。

**Q45 — Jacobi PGS divergence with clustered multi-point contacts** ✅ RESOLVED (2026-04-13, session 29)

**根因**：同 body 上 N 个同法向接触 → W 矩阵零特征值（力分配不定）→
Jacobi 迭代矩阵 ρ > 1。调研 MJX（Newton/CG）、PhysX（colored GS）、
Bullet3（mass splitting），确认纯 Jacobi 无法处理此场景。

**解决**：新增两个 solver backend：
- `jacobi_pgs_ms` — Mass splitting (Tonge 2012)：W_diag × N_contacts_per_body
- `colored_pgs` — Graph-colored GS (PhysX 方案)：图着色 + 异色串行

box-ground 多点已激活。原始 `jacobi_pgs_si` 不支持多点 ground（会发散）。
→ Moved to REFLECTIONS.md session 29.

**Q46 — Solver backend 大规模系统验证** (2026-04-13, session 29)

Session 29 新增的三个能处理多点接触的 solver（`jacobi_pgs_ms`、`colored_pgs`、`admm`）
在小规模 fixture（9 body, 16 contacts）上验证通过，但以下维度未经测试：

**待验证维度**：
1. **大规模 RL 训练**：num_envs=1000+，持续 10⁶ steps 的数值稳定性
2. **复杂机器人**：人形（30+ body）、多足（高 DOF）的接触密集场景
3. 🔄 **Solver 间物理一致性**（2026-04-20 短时基线已加，长时/RL 仍待验证）：`jacobi_pgs_ms`、`colored_pgs`、`admm`
   在 3-robot mixed-shape 场景（50 步）的 link0 z 偏差 < 0.05 m，max|qdot| < 50 rad/s。
   见 `tests/gpu/solvers/test_solver_backends.py::TestCrossSolverConsistency`。
   长时间轨迹（10⁶ steps）和 RL 场景仍待验证（维度 1/2）。此维度**未关闭**。
4. **Colored GS 性能**：960 kernel launch/step 的 overhead 在大 N_envs 下是否可接受
   （当前 254ms/step vs ADMM 66ms/step，需要优化：减少 MAX_COLORS、kernel 内循环）
5. **Mass splitting 收敛精度**：under-relaxation 导致的力分配误差对 RL reward 的影响
6. 🔄 **CPU vs GPU 一致性**（2026-04-21 部分澄清，契约未完全统一）：
   ~~CPU GJK/EPA 返回 1 个~~ — 此描述已过时（commit `35ca490`，session 33）。
   CPU `ground_contact_query()` 对 Box/ConvexHull 已走 `contact_vertices()` 顶点枚举，
   flat box 实测返回 4 个接触点，与 GPU 行为对称。
   **残余差异**：CPU 返回所有穿透顶点（无数量上限），GPU cap 到 4 个最深点。
   对 `BoxShape`（恰好 8 顶点）平地场景 count 一致；对任意凸包（>4 底面顶点）
   仍可能 count mismatch。语义级契约（body set / normal / per-body max depth）已对齐。
   count-level parity 留待专项 box-ground CPU/GPU parity test 正式固化。

**Benchmark 基线**（session 29, 3-robot 9-body fixture, 500 steps）：
```
jacobi_pgs_si:  DIVERGED step 1
jacobi_pgs_ms:  stable, 174 ms/step, max|qdot|=1~3
colored_pgs:    stable, 254 ms/step, max|qdot|=1~3
admm:           stable,  66 ms/step, max|qdot|=0~4
```

**触发条件**：Phase 3 RL 训练循环实装前。
**优先级**：P1（直接影响 RL 训练质量和速度）。

**Q47 — Smooth Shape 统一架构：InnerShape + ConvexRadius** (2026-04-16, session 31)

> ⚠️ **此问题需要进一步思考与讨论，暂不实施。**

**背景**：Session 31 为解决 sphere-sphere EPA 退化 bug，采用了**方案 A**（特殊形状解析 dispatch）：
在 `gjk_epa_query()` 中对 SphereShape 增加解析路径，完全绕过 GJK/EPA。
该方案修复了眼前的 bug，但属于 ad-hoc 补丁，形状类型越多维护负担越重。

**根本问题（PhysX / Bullet 论坛共识）**：
EPA 在 smooth shapes（sphere、capsule）上本质上不稳定——这类形状的 support function
连续可微，GJK simplex 在接近切触时会退化（三角面过原点），EPA 选错法向量。

**方案 B（Jolt ConvexRadius 架构）—— 工业界最佳实践**：

Jolt Physics 的做法（[JoltPhysics docs](https://jrouwe.github.io/JoltPhysics/class_shape.html)）：

```
每个 Shape 有 inner shape (polyhedral) + convex_radius (float)
  SphereShape  → inner = Point(0-dim) + radius
  CapsuleShape → inner = Segment(1-dim) + radius
  BoxShape     → inner = Box(half_extents - margin) + margin
  ConvexHull   → inner = ShrunkenHull + margin
```

1. GJK 运行在 inner shape（多面体）上 → 不会产生退化 simplex
2. 结果是 inner shape 之间的距离 d
3. 实际接触深度 = (r_a + r_b) - d（r_a/r_b 是各自 convex_radius）
4. 法向量 = inner GJK 输出，数值稳定，无需 EPA

**优势**：
- 完全消除 EPA 退化问题（EPA 仅在 inner shapes 深度穿透时触发，概率极低）
- 统一所有形状，无需 type-specific dispatch
- 与 FEM 切割（Q18.9 InterfaceMaterial）天然兼容：margin = interface boundary
- Jolt 实测：> 95% 接触在 inner GJK 阶段解决，EPA 极少触发

**需要讨论的设计决策**：
1. `CollisionShape` 基类是否增加 `inner_shape()` / `convex_radius()` 接口？
2. 向后兼容：现有 test suite 依赖当前几何行为，架构变更需要重新校准公差
3. `ShapeInstance.interface` (InterfaceMaterial Q18.9) 与 convex_radius 的关系
4. GPU 端 (`warp/analytical_collision.py`) 是否同步采用此架构？

**当前状态（方案 A 补丁）**：
- `gjk_epa_query()` 已对 SphereShape 做解析 dispatch（sphere-sphere、sphere-capsule）
- CapsuleShape 已有完整解析路径（capsule_capsule、capsule_box、capsule_cylinder、capsule_hull）
- 其余形状（Box vs Sphere、ConvexHull vs Sphere）走 GJK distance + normal，绕过 EPA

**触发条件**：下一次涉及 sphere/capsule EPA 不稳定 bug，或开始设计 InterfaceMaterial 时。
**优先级**：P2（方案 A 补丁已稳定；架构重构有较大工作量，需专项讨论后决定）。

**Q48 — CPU 碰撞检测完备性：session 31/32 已知 bug 与后续缺口** (2026-04-16, session 31; 更新 session 32/33)

**背景**：Session 31 在 CPU GJK/EPA 管线中发现并修复了多个 bug：
1. **EPA 退化 simplex**（sphere-sphere/sphere-capsule）：EPA 在 smooth shapes 上生成退化
   三角面，法向量指向错误。修复：`gjk_epa_query()` 增加解析 dispatch（方案 A）。
2. ✅ **sphere-box / sphere-cylinder 解析 dispatch**（session 32）：
   `gjk_epa_query()` 对 SphereShape vs BoxShape / CylinderShape 增加 PhysX 风格解析路径，
   完全绕过 GJK/EPA。sphere-cyl 的 `gjk_distance()` 早退出 workaround 已移除。
   测试：`tests/unit/collision/test_sphere_analytical.py`（13 tests，atol=1e-6）。
   MuJoCo 对比：`tests/integration/test_margin_vs_mujoco.py::TestSphereAnalyticalVsMuJoCo`（3 tests）。
3. **凸 margin 管线**（Q44 Phase 1/2）：`gjk_epa_query()` 增加 margin 参数；
   Phase 1（gjk_distance on 收缩形状）处理浅接触，Phase 2（EPA）处理深穿透。
   测试：`tests/unit/collision/test_convex_margin.py`（336 行，所有 10 个形状对）。

**当前残余缺口**：

1. **`gjk_distance()` box-cyl/box-hull 早退出**（sphere-cyl 已通过解析 dispatch 修复）：
   根本原因尚未修复，仅通过在 `test_convex_margin.py` 中增大穿透深度绕过（pen = 3×margin）。
   下一步：追查 GJK distance 内部终止条件（signed-distance simplex walk 在这两对上的退化），
   修复后去掉 workaround。**优先级：P4（低优先级，解析 dispatch 已覆盖最常见 pair）**。

2. ✅ **CPU 多点接触流形生成（ground contact path）**（session 33）：
   `ground_contact_query()` 已升级为顶点枚举多点流形（Box/ConvexHull），与 `halfspace_convex_query()`
   逻辑对齐。Sphere 仍走单点支撑点路径。验证：`TestMultiPointContactCount`（4 tests）。
   *Body-body 接触的多点流形（`gjk_epa_query()`路径）已有 face clipping，此处关闭的是
   FlatTerrain ground contact 路径的缺口。*

3. ✅ **CpuEngine 集成级别覆盖**（session 33）：
   新增 `tests/integration/test_cpu_engine_shapes.py`（12 tests，3 classes）：
   Class 1 — 5 形状类型逐一 drop + settle（Sphere/Box/Capsule/Cylinder/ConvexHull）；
   Class 2 — 多点接触数量验证（flat/tilted box, sphere, convexhull）；
   Class 3 — 两体碰撞（sphere-sphere/box-sphere/box-box）。
   1174 tests 全部通过（含原有全量）。

**触发条件**：修复 `gjk_distance()` box-cyl/box-hull 早退出（唯一残余缺口）。
**优先级**：P4（仅剩 gjk_distance 早退出，解析 dispatch 已覆盖最常见 pair；集成测试和多点地面接触已补全）。

---

**Q49 — CPU 多接触冲量爆炸（与已解决的 GPU Q45 同根问题）** (2026-04-16, session 31)

**背景**：Q45 已在 GPU 端解决（session 29，mass splitting + colored_pgs + admm）。
CPU 端存在**相同根本原因**，Q46 item 6 有一行提及但未展开。

**理论风险分析**：

```
CPU PGSSplitImpulseSolver / PGSContactSolver 构建完整 Delassus 矩阵：
  W = J M⁻¹ Jᵀ   （J: nc×nv 接触 Jacobian，M: nv×nv 质量矩阵）

当多个接触点聚集在同一刚体上时：
  W 的 i-j 块 = Jᵢ M⁻¹ Jⱼᵀ ≠ 0  （接触点 i, j 共享同一刚体）
  → W 的最小特征值趋近 0
  理论上可能导致迭代发散
```

**⚠ 实测结论（2026-04-16）：CPU GS-PGS 不发散，与 GPU Jacobi-PGS 根本不同**

使用与 Q45 完全相同的三机器人场景（`test_b5_d4d8_mixed_ground.py` 的 fixture）
在 `CpuEngine`（GS-PGS，非 Jacobi）上实测 1000 步，结果：

```
STABLE after 1000 steps
n_contact: max=11, mean=1.6
max|qdot|: initial=0.98, peak=5.20, final=4.65
```

**根本原因——迭代方式不同**：

| 维度 | GPU Jacobi-PGS（Q45，已爆炸）| CPU GS-PGS（本问题，稳定）|
|------|------------------------------|--------------------------|
| 更新顺序 | 所有接触点**并行**，用旧 λ | 接触点**顺序**，立即用新 λ |
| 谱半径 | ρ(I - D⁻¹W) > 1（爆炸）| W 严格对角占优条件更容易满足 |
| 稳定条件 | 需要 mass-splitting 或 coloring | Gauss-Seidel 天然更收敛 |

结论：**Q49 原始描述高估了 CPU 风险**，CPU GS-PGS 的收敛性不依赖 mass splitting。
Q45 的爆炸是 GPU **Jacobi** 迭代的特有问题，不是所有 PGS 的通病。

**仍需关注的情形**：
- 极端刚性接触（弹性系数 e≈1，多球链式碰撞）——GS 也可能慢速漂移
- 很大 nc（>50）时迭代次数（60）是否足够收敛
- 数值精度：box 落地后静止时穿透深度约 25mm（penalty-spring 软接触所致），
  不是发散，而是 erp/slop 参数导致的稳态穿透，与 MuJoCo 精度对比仍需关注

**GPU 解法参考（Q45，已实现）**：
- `jacobi_pgs_ms`（Tonge 2012 mass splitting）
- `colored_pgs`（graph-colored Gauss-Seidel）
- `admm`（ADMM 分解）

**优先级降级**：由 P1 → P3。CPU GS-PGS 稳定性足以支撑 MuJoCo 对比验证工作。
若未来出现真实发散（极端场景），再考虑 port mass-splitting（~100 行）。

**关联**：Q46 item 6（CPU/GPU 一致性），Q45 RESOLVED（GPU mass splitting）。

---

## Rendering

**Q50 — 渲染层架构完善：RenderBackend ABC + 多后端路线图** (2026-04-22)

**背景**：Phase 3 渲染层目前约 15% 完成。现有结构：
- `rendering/render_scene.py` — 后端无关的 `RenderScene` 数据类（✅ 设计良好）
- `rendering/scene_builder.py` — CPU `MergedModel` → `RenderScene` 桥接（✅ 仅 CPU）
- `rendering/viewer.py` + `shape_artists.py` — matplotlib 3D 后端（✅ 可用，速度慢）

**已确认的架构缺陷（2026-04-22 调研）**：

1. **无 `RenderBackend(ABC)` 接口**：`viewer.py` 是 matplotlib 具体实现，无抽象契约。
   加新后端只能并排堆放，没有共同接口。参考：Drake `SceneGraph` + 可插拔 renderer。

2. **无 GPU → RenderScene 桥接**：`scene_builder.py` 只接受 CPU 类型。
   `GpuEngine` 有 N 个并行 env 的 GPU 数组（`q`、`contact_point` 等），
   没有 `build_render_scene_from_gpu(engine, env_idx)` 路径。

3. **无传感器提取路径**：`RenderScene` 缺 IMU（body linear/angular vel）、
   关节力矩、力传感器字段——这些是 RL obs 的必要输入。

4. **无 multi-env 视图接口**：无法在训练中选取 env #k 查看。

**已确定的后端路线图（2026-04-22）**：

| 阶段 | 后端 | 目的 |
|------|------|------|
| **近期（Phase 3）** | **Rerun** (`rerun-sdk`) | 验证 `RenderBackend` 接口；headless CI；timeline scrubbing；RL 训练监控 |
| **中期（Phase 3）** | **Vulkan**（raw 或 vkguide 路线） | 实时高性能可视化，为 sim-to-real 准备 |
| **长期（Phase 4+）** | **自研渲染 / Isaac Sim RTX** | 光追、传感器噪声模型、域随机化视觉；`RenderScene` → USD 导出路径 |

**Rerun 选型依据**（调研结论）：
- 零系统依赖，`pip install rerun-sdk`
- 原生支持所有我们的形状：`rr.Boxes3D / Capsules3D / Cylinders3D / Mesh3D / Arrows3D`
- CI headless：`rr.save("debug.rrd")`，无需 DISPLAY
- `rr.set_time_seconds("sim_time", t)` 内置 timeline；0.24 起支持 file + live 双 sink 同时输出
- 直接对标训练监控用例：训练跑 N env 时，把 env #0 的 `RenderScene` 流入 Rerun

**最小 `RenderBackend(ABC)` 接口（调研建议）**：

```python
class RenderBackend(ABC):
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def render_frame(self, scene: RenderScene, timestamp: float, env_index: int = 0) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def set_output(self, path: str) -> None: ...   # headless/CI 模式，可选覆盖

    @property
    @abstractmethod
    def supports_offscreen(self) -> bool: ...
```

`timestamp` 为必填（Rerun 需要，matplotlib 忽略，代价为零）。

**实施计划**：

```
Step 1：引入 RenderBackend(ABC)，MatplotlibBackend 包装现有 viewer.py  ✅ (session 33)
Step 2：RerunBackend 实现（验证接口 + 实际可用）                        ✅ (session 33, rerun-sdk 0.31 fix session 34)
Step 3：GPU 桥接 build_render_scene_from_gpu(engine, env_idx)           ✅ (session 33)
Step 4：RenderScene 加 numeric/state 传感器字段（sensing phase-1 readings） ✅ (2026-04-27)
Step 4.5：render phase-2 terrain + triangle mesh backend parity          ✅ (2026-04-28)
Step 4.6：Rerun scalar timelines for RenderScene.sensor_data             ✅ (2026-04-28)
Step 4.7：Rerun sensor scalar group filtering                            ✅ (2026-04-28)
Step 5（中期）：VulkanBackend                                            ⬜
```

**当前 feature 语义澄清（2026-04-23）**：

1. **`terrain` 是 scene-level 字段，不是某个 backend 私有 feature**
   - `RenderScene` 用独立的 `terrain: TerrainInfo` 表达地形，而不是把地形塞进
     `PositionedShape` 列表。
   - `scene_builder._terrain_to_info()` 当前把 `FlatTerrain` / `HalfSpaceTerrain`
     提升为后端无关快照；`TerrainInfo` 预留了 `heightmap` 类型，但尚未端到端打通。
   - 结论：地形渲染属于所有 backend 都应消费的 scene contract。当前缺口是
     backend feature parity：matplotlib 已消费 `scene.terrain`，Rerun 尚未实现。

2. **`mesh` 需要区分 collision 语义和 rendering 语义**
   - collision 路径：URDF `<mesh>` 先解析为 `MeshShape(filename, scale)`，再通过
     `robot/mesh_loader.py` / 凸分解管线转成 `ConvexHullShape` 供当前碰撞检测使用。
     当前已解决的是“mesh 作为碰撞输入”的问题，本质上走的是 convex-hull /
     convex-decomposition 方案，而不是通用三角 mesh narrowphase。
   - rendering 路径：`scene_builder` 仍会把原始 `MeshShape` 保留为
     `shape_type="mesh"`（附 `vertices` / `filename`），为未来后端直接显示三角 mesh
     留出通道；这与 collision 使用 `ConvexHullShape` 是两件事。
   - 结论：当前 backends 对通用 `mesh` 的处理（skip + warning / silent skip）是
     rendering parity gap，不应与 Q7 的“mesh collision geometry 已解决”混为一谈。

3. **因此，`terrain` / 通用 `mesh` 都不是 RerunBackend 单点问题**
   - `terrain`：应由所有可视化 backend 基于 `RenderScene.terrain` 统一消费。
   - 通用 `mesh`：需要在 rendering 层先明确 raw triangle mesh 的显示 contract
     （例如：必须提供 triangles / faces，或明确只支持 convex_hull，不支持 raw mesh）。
   - 在该 contract 明确前，“RerunBackend complete” 只能理解为
     “基础 primitives + contacts + skeleton 可用”，不能理解为与 matplotlib 或未来 Vulkan
     在 scene feature 上完全对齐。

   **状态更新（2026-04-28）**：第一段 render phase-2 parity 已落地，见
   `collab/q50-render-phase2-terrain-mesh-parity__implementation-note__codex__v1.md`。
   - `RerunBackend` 已消费 `scene.terrain`，支持 `flat` / `halfspace` debug plane。
   - `MeshShape` 已支持可选 `faces: (F, 3)`；`scene_builder` 透传
     `vertices + faces + filename`。
   - Matplotlib/Rerun 均在 `mesh` 具备 `vertices + faces` 时渲染，缺 triangle
     topology 时继续跳过/警告。
   - 尚未解决 filename-only mesh loading、material/texture、heightmap terrain、
     retained/realtime render view。

4. **`RenderScene.sensor_data` 是窄口 debug/export payload，不是通用 sensor packet**
   - 2026-04-27 已通过 `RenderSensorData` 接入 sensing phase-1 的 numeric/state readings：
     `IMUReading / JointStateReading / ForceSensorReading / ContactStateReading`。
   - 数据来源是 `PublishedFrame -> StateSampleView -> sensing builders`，不读取 engine-private scratch。
   - 不包含 camera / LiDAR / surface query / imaging payload；这些继续按 Q53 的
     `sensor_rendering/` / `SurfaceQuerySpec` 方向处理。
   - 详见 `collab/q50-step4-render-scene-sensor-data__implementation-note__codex__v1.md`。

   **状态更新（2026-04-28）**：`RerunBackend` 已消费窄口 `RenderScene.sensor_data`
   并记录 selected scalar timelines，见
   `collab/q50-rerun-sensor-scalar-timelines__implementation-note__codex__v1.md`。
   当前覆盖 contact count/mask、joint q/qdot、force scalar/vector fields、
   IMU angular velocity；数组逐元素记录有上限，避免大模型刷爆 Rerun entity tree。
   这仍是 debug/export 可视化，不引入 camera/LiDAR/surface-query payload。

   **状态更新（2026-04-28）**：`RerunBackend(sensor_scalar_groups=...)`
   已支持按 `contact` / `joint` / `force` / `imu` 过滤 scalar timeline 输出，
   默认仍记录全部四类。见
   `collab/q50-rerun-sensor-scalar-groups__implementation-note__codex__v1.md`。

**参考项目**：
- Drake：`SceneGraph` role-based geometry + pluggable renderer（最直接参考）
- Isaac Lab 3.0：pluggable renderer system，tiled rendering for N envs
- Rerun 官方文档：operating-modes（headless/save），archetype reference（3D primitives）
- Genesis：三后端可选（PyRender / LuisaRender / Madrona）

**触发条件**：Phase 3 开始实装（当前 session 已到触发点）。
**优先级**：P1（大规模 RL 验证的前提；Phase 3 主线）。

**Q51 — Force sensor / torque telemetry contract（CPU/GPU 统一观测接口）** (2026-04-23)

**状态更新（2026-04-27）**：phase-1 numeric/state 观测合同已解除 Q50 Step 4 阻塞。
当前实现路径为：

```text
PublishedFrame -> TelemetrySnapshot -> StateSampleView -> sensing readings -> RenderSensorData
```

已明确：

- `ForceSensorReading` phase-1 暴露 `qfrc_applied` + `tau_smooth`，不再使用含混的
  `generalized_force`
- GPU `force_sensor_wp` 进入 `ForceSensorReading.contact_force`
- CPU/GPU 不对称字段保持 `None`，不伪造 parity
- `physics/telemetry.py` 仍是过渡 bridge，不作为最终 sensor package 归属定案

因此 Q51 对 Q50 Step 4 的阻塞已关闭；RL obs schema 已给 phase-2 压力点命名；
contact mask 已作为第一条 sensing phase-2 published contract 落地；剩余问题转为
telemetry parity / 传感器坐标系扩展时再处理。

**背景**：Q50 Step 4 希望把 IMU、关节力矩、力传感器数据接入 `RenderScene`，用于
Rerun 训练监控与后续 RL obs 复用。但当前仓库只有一半设计落地：
- CPU 路径已有 `ForceState` / `StepOutput.force_state`，能表达
  `qfrc_passive / qfrc_actuator / qfrc_applied / tau_smooth / qacc_*`
- GPU 路径已有 `qacc_smooth_wp` / `qacc_total_wp`、`contact_force_sensor_wp`
  等局部 accessor，但 `StepOutput.force_state` 仍为 `None`
- GPU 内部存在 `tau_passive / tau_total` scratch，但尚未作为公开 API 暴露

**当前判断（2026-04-23；2026-04-27 更新）**：
- 这**不是** Q50 Step 1–3（`RenderBackend` / `RerunBackend` / GPU 几何桥接）的阻塞项
- 对 Q50 Step 4 的 phase-1 numeric/state 合同已收敛并落地
- RL obs schema 草案已落地；contact mask 已接 published contract；训练期更完整
  传感器可视化仍需后续 scalar/tensor timeline 设计
- rendering 不读取 `GpuEngine` 私有 scratch，只消费 published-frame / sensing bridge

**phase-1 已决策点**：
1. **所有权**：传感器/力矩快照应由 `physics/` 定义统一 contract，还是由
   `rendering/` 定义 `SensorData` 后反向拉取？
   - phase-1 采用 `sensing/` readings；`RenderSensorData` 只作为 debug/export 容器。
2. **数据粒度**：要暴露的是 `(nv,)` generalized force（MuJoCo 风格 `qfrc_*`），
   还是 per-joint torque，还是 per-body wrench / contact-force sensor？
   - phase-1 明确暴露 `qfrc_applied`、`tau_smooth`、可选 `contact_force`。
3. **API 形式**：走 `StepOutput.force_state` 的 CPU/GPU 语义对齐，还是新增
   `GpuEngine` zero-copy accessors（如 `tau_total_wp`），还是单独做
   `build_sensor_snapshot_from_gpu()`？
   - phase-1 走 `PublishedFrame -> TelemetrySnapshot -> StateSampleView`。
4. **命名与坐标系**：`qfrc_actuator / qfrc_passive / tau_smooth` 是否直接沿用；
   contact force sensor 用 world frame 还是 body-local；multi-env 如何切片。
   - phase-1 沿用当前 telemetry 命名；GPU contact force 维持当前 world-frame bridge 语义。
5. **模块归属 / 迁移路径**：当前 `physics/telemetry.py` 作为 phase-1 的
   published-frame telemetry bridge 先保留在 `physics/` 下；若后续 `TelemetrySnapshot`
   持续长出更强的 sensor-facing 语义，是否应迁移到新的 `sensing/` 包，并仅保留
   `ForceState / PublishedFrame` 作为 `physics/` 的真值层契约？

**建议的暂时策略**：
- Q50 Step 4 已接 numeric/state sensor data；保持窄口，不扩成 camera/LiDAR packet
- `physics/telemetry.py` 目前视为过渡层，不把它当成最终 sensor 模块归属的定案
- RL 观测接口已明确 quaternion / contact mask / normalization 的 phase-2 合同：
  wxyz quaternion、显式 scale、optional published contact mask；contact mask 已落地

**触发条件**：需要把 GPU 训练信号接入更完整的 Rerun scalar/tensor timeline，或
需要扩展 RL obs 到 force/contact-force 字段时。
**优先级**：P2（Q50 Step 4 阻塞已解除；RL obs phase-1 schema 与 contact-mask
published contract 已收敛）。

**Q52 — Physics publish pipeline implementation（GpuPublishedFrame / PublishedRing / QoS / reclaim）** (2026-04-24)

**背景**：2026-04-24 围绕“物理层如何把每步结果正式发布给渲染/传感器/日志消费者”进行了较完整的架构收敛，并形成了审查文档：

- `collab/render-physics-pipeline__proposal__codex__v1.md`

本轮结论已经不再停留在哲学层面，而是进入接近实现的接口与控制平面设计，包括：

- `PhysicsModel / PhysicsState / DerivedPhysicsCache / PublishedFrameCore`
- `PublishPolicy -> PublishPlan -> kernel launches`
- `GpuPublishedFrame / PublishedSlot / PublishedRing`
- `best_effort` vs `lossless`
- `borrow` vs `snapshot`
- `ConsumerState / AckPolicy / SlotReclaimer`

**状态更新（2026-04-27）**：

- phase-1 control plane 与同步 published-frame runtime 已经落地：
  `PublishPolicy / PublishPlan / ConsumerState / AckPolicy / SlotReclaimer /
  BorrowedFrameLease / SnapshotHandle / CpuPublishedFrame / GpuPublishedFrame`
- `GpuEngine` 已经同步写 dedicated published slot buffers，`latest_published_frame()`
  不是 stub
- Q50/Q51 已经让 debug export / render / telemetry / sensing / `RenderScene.sensor_data`
  消费 `PublishedFrame`
- `PublishedRing` 已抽为 `GpuEngine` 内部持有的 physics-runtime 控制组件，slot
  buffers 仍由 `GpuEngine` 分配和写入
- `GpuPublishedFrame` stale guard 已加固：除 `invalidated` 外，也检查
  `slot_meta.frame_id == frame.frame_id`
- published `contact_mask` 已作为轻量 summary block 接入 CPU/GPU published frame，
  不需要默认打开 dense `RigidBlock`
- `lossless + snapshot` 已接入 future-aware host staging：GPU lossless host
  snapshot 通过 `SnapshotHandle` 的 staged-completion 点推进 ack
- consumer/backpressure 语义已拆成四条轴：
  `consumer_kind / consumer_location / access_mode / qos_mode`
- `ConsumerState.consumer_location` 已显式加入，默认 `"host"`；device
  consumer 的 event/fence 状态类仍推迟到真实 device consumer 出现时
- `ConsumerState.device_completed_frame_id` / `device_done_event` 与
  `PublishedRing.mark_device_consumer_complete(...)` 已作为 device consumer
  reclaim 控制面落地；`GpuEngine` 已用真实 Warp/CUDA event 完成第一版
  device-side handoff（publish_event wait + done_event record + slot reuse wait）
  这里的“阻塞”分成两层：control plane 上 slot 被 lossless device consumer
  pin 住；device timeline 上用 stream wait/event 排序。热路径不应使用
  `wp.synchronize_event` / `cudaEventSynchronize`，后续 Warp→CUDA 迁移时直接
  映射为 `cudaEventRecord` + `cudaStreamWaitEvent`
- device lossless consumer stall detection / `max_lag_frames` enforcement 已
  落第一版：默认 `None` 保持 lossless 原语义；显式配置上限后，超出 lag
  budget 会抛 `DeviceConsumerStalledError`，并进入 `publish_stats()`
  监控快照
- `GpuEngine.publish_stats()` 已提供第一版 ring monitor：slot state/pin、
  consumer lag/blocker/stalled、以及 backpressure/skip/wait/stall/raise
  counters；并加入 host-observed rolling publish FPS / interval。该 FPS
  来自 CPU `mark_ready()` 时间戳，只衡量 materialized publish cadence，不读
  CUDA event elapsed time，也不引入 GPU 同步
- `PublishedRing` 不应成为 RL 训练热路径的必经入口；RL obs kernel 可直接读
  current GPU buffers / scratch，ring 主要服务外部或异步消费者的稳定 slot 生命周期

因此 Q52 的下一阶段不是“从零落控制平面”，而是 phase-2 runtime hardening：
host/device consumer 边界、stream/event staging、typed slot/block、device
consumer event/fence、以及更丰富的 compact
contact-pair published contract。

**历史判断（2026-04-24）**：

- 该方向已经形成了足够稳的 implementation-ready 草案
- 当时尚未真正落地到代码骨架；该判断已被 2026-04-25 至 2026-04-27 的
  phase-1/early phase-2 实现取代
- 这是后续 realtime rendering / high-fidelity rendering / sensor export / host logging 共用的数据发布底座
- 应优先从 GPU path 实装，再让 CPU path 对齐语义做简化 reference 版本

**当前已收敛的关键结论**：

1. **`PublishedRing` 默认大小取 `3`**
   - 默认 triple buffering
   - `best_effort` consumer 不应阻塞 physics
   - `lossless` consumer 必须形成硬 backpressure

2. **`PublishedFrame` 不能直接引用 mutable scratch**
   - 必须引用 dedicated published slot buffers
   - 否则 physics 下一步覆盖会破坏 frame 语义

3. **QoS 与读取方式是正交维度**
   - QoS:
     - `best_effort`
     - `lossless`
   - access mode:
     - `borrow`
     - `snapshot`

4. **`borrow` 与 `snapshot` 语义必须强区分**
   - `borrow`：ephemeral lease，只适合短时消费
   - `snapshot`：复制/转存到私有 staging，后续不依赖 slot 生命周期

5. **`lossless + snapshot` 的 ack 点**
   - 不是“copy 被 enqueue 到 stream”
   - 而是“staging 中已经拥有完整、自持副本；若涉及 async copy，则 copy completion event 已 signal”

6. **slot 回收只看启用中的 `lossless` consumer**
   - `best_effort` 只影响能看到哪些帧，不参与 reclaim
   - 多个 `lossless` consumer 并存时，最慢者决定 backpressure

7. **`lossless` QoS 不允许系统静默降级**
   - 默认行为应是：监控 + 报警 + 阻塞等待
   - 若要退化为 `best_effort`，必须用户显式 opt-in，并伴随清晰日志/事件

8. **dense `RigidBlock` 第一版可预分配，但写入不应默认每步开启**
   - `max_contacts` 是 per-env 上限，也是显存主要放大器
   - `contact_count` 更接近 core 边界信息，可作为轻量 core 条目
   - dense `RigidBlock` 的实际写入应由 `PublishPlan.do_rigid_block_write` 控制

**建议的第一阶段实施范围**：

1. 增加最小控制平面数据结构：
   - `PublishedSlotMeta`
   - `ConsumerState`
   - `AckPolicy`
   - `SlotReclaimer`

2. 增加最小 publish 配置与计划层：
   - `PublishPolicy`
   - `PublishPlan`

3. 在 GPU path 先落最小 published frame：
   - core:
     - `q`
     - `qdot`
     - `X_world_R`
     - `X_world_r`
     - `v_bodies`
     - `contact_count`
   - `TelemetryBlock`
   - `RigidBlock` 预分配，但按 plan 条件写入

4. 暂不在第一阶段做：
   - compaction-based compact contacts
   - full realtime renderer integration
   - full sensor stack integration
   - CPU/GPU 两套完全对齐的高层 API 美化

**待实施前仍建议重点复核的点**：

1. `max_contacts` 的默认值与用户调参指引
2. `PublishedRing=3` 在实际目标任务下的显存压力
3. `lossless` consumer 的最大时延与 ring sizing 关系
4. `borrow` API 的 context-manager / ephemeral lease 具体实现方式
5. `HostExportQueue` 的 staging ownership 与 copy completion 信号时机
6. RL 训练热路径是否应始终绕过 `PublishedRing`，直接读 current GPU
   buffers / scratch 或 current-frame device pointer

**建议的实施顺序**：

1. ✅ 控制平面（policy/plan/consumer/reclaimer）已落地
2. ✅ GPU 同步 `publish_core` 已落地为 dedicated slot buffer copy
3. ✅ `PublishedRing` 已成为 `GpuEngine` 内部控制组件
4. ✅ `lossless + snapshot` 已具备 future-aware host staging / completion ack
5. ✅ `ConsumerState.consumer_location="host"` 默认字段已落地
6. ✅ host-only `on_ring_full="block"` 的真实等待语义已落地
7. ✅ RL obs / sensing phase-2 的 per-body contact mask published contract 已落地
8. ✅ device consumer completion 控制面已落地（`device_completed_frame_id`）
9. ✅ `GpuEngine` 第一版 Warp/CUDA stream-event handoff 已落地：
   published slot 写完 record `publish_event`，device consumer stream wait 后
   record `done_event`，slot 复用前 physics stream wait `done_event`
10. ✅ device consumer stall detection / max-lag enforcement 已落地
11. ✅ `GpuEngine.publish_stats()` 第一版 ring/consumer lag monitor 已落地
12. ✅ host-observed FPS/rolling monitor buffer 已落地（不在默认路径做 CUDA timing）
13. 后续按需要补 compact contact-pair published contract
14. 后续把 host staging 从 Python future 升级为 Warp stream/event + bounded queue
15. 未来 GPU render-backed sensing 使用 stream event/fence，不走 CPU condition wait

**触发条件**：开始把 2026-04-24 这轮 design proposal 转成代码时。
**优先级**：P1（已接近实现，且是后续渲染/传感器主线的基础设施）。

**Q53 — Sensing / Rendering boundary（ImagingView ownership + SurfaceQuery execution boundary）** (2026-04-26)

**状态更新（2026-04-27）**：依赖方向与归属已形成 phase-2 决策，见
`collab/q53-sensing-rendering-boundary__decision__codex__v1.md`。

**状态更新（2026-04-29）**：第一版 `SurfaceQuerySpec` / executor 骨架已落地：

- `sensing.surface_query.SurfaceQuerySpec` 描述 world-frame ray batch；
  directions 在 spec 层统一 normalize，因此 result distance 保持米制语义
- `SurfaceQueryResult` 承载 hit mask、distance、hit position、normal
- `SurfaceQueryExecutor` protocol 明确 query execution 是显式 runtime 层
- `CpuPlaneSurfaceQueryExecutor` 支持 `FlatTerrain` / `HalfSpaceTerrain`
  的无限平面 ray query；`HeightmapTerrain` / mesh / body geometry 明确延后
- `RangeSensorReading` / `build_range_sensor_reading(...)` 已作为
  `SurfaceQueryResult -> sensor-facing reading` 的薄转换层落地；ray pattern /
  sensor pose builder 仍延后
- 该实现不 import `rendering`，不把 query result 塞进 `RenderScene`，符合
  Q53 的 sensing/rendering 边界

当前决定：

- `sensing/` 继续负责 sensor-facing specs/readings/views，不直接依赖 `rendering/`
- `rendering/` 继续负责 `RenderScene` / render backend，不承载通用 sensor packet
- future camera / RGB / segmentation / render-backed depth 走集成层，暂定
  `sensor_rendering/`，该层允许依赖 `sensing/` 和 `rendering/`
- `SurfaceQueryView` 属于 `sensing/`，但 builder 只构造 query spec/view；
  CPU/GPU query 结果由显式 executor/runtime 产生
- 已优先命名为 `SurfaceQuerySpec`，第一版 CPU plane executor 已落地
- 第一版 depth image 倾向先走 surface query / ray-cast depth；RGB / segmentation
  再走 `sensor_rendering/`
- `RenderScene` 可用于 debug overlay，不作为 LiDAR/camera 的 canonical execution contract
- `PublishPolicy.sensor_render` 已重命名为 `render_backed_sensing`

**背景**：随着 published-frame phase-1 consumer integration 完成，下一步开始讨论独立的
`sensing/` 模块。当前已基本收敛：

- `physics/` 负责真值与 `PublishedFrame`
- `rendering/` 负责 `RenderScene` 与 backend
- `sensing/` 负责 sensor-facing view / reading / builder

但在继续往 `SurfaceQueryView` / `ImagingView` 推进时，出现了两个结构性边界问题：

1. **`ImagingView` 归属问题**
   - 若 `ImagingView` 需要几何 + 语义 + 材质/光照，它天然会碰 `rendering/`
   - 若仍放在 `sensing/`，就等于默认允许 `sensing -> rendering`
   - 当前依赖图尚未正式允许或禁止这条边

2. **`SurfaceQueryView` 的 CPU/GPU 执行边界**
   - CPU query 多半走 host-side 几何查询 / BVH / numpy 路径
   - GPU query 需要单独的 Warp/kernel 执行路径
   - 差异不只是“字段是否存在”，而是“整个 query 执行方式不同”
   - 因此应尽早决定：`SurfaceQueryView` builder 是只产出 query scene，还是直接产出 query result

**当前判断（2026-04-26；2026-04-27 更新）**：

- `StateSampleView` 已足够稳，可以作为 `sensing/` 第一阶段正式落地方向
- `SurfaceQueryView` 可以进入 `sensing/` 设计，但 query execution 必须独立于 builder
- `ImagingView` 不直接进入 `sensing/`，后续通过 `sensor_rendering/` 这类集成层处理

**已决策点（phase-2 边界）**：
1. `ImagingView` / camera execution 不直接放进 `sensing/` 或 `RenderScene`；
   后续由 `sensor_rendering/` 集成层拥有。
2. 不引入正式依赖边 `sensing -> rendering`。
3. `SurfaceQueryView` builder 只产出 query spec/view，不直接产出 sensor query result。
4. CPU/GPU query 差异放在显式 executor/runtime 层，而不是藏进 builder。

**仍待实现前细化**：
1. camera reading schema
2. GPU realtime renderer 和 camera pipeline 是否共享 surface cache
3. GPU / mesh / body-geometry `SurfaceQueryExecutor` 的具体执行路径
4. ray pattern / sensor pose builder（body attachment、scan angle convention、
   noise/clipping policy）

**建议的暂时策略**：
- 继续保持 `StateSampleView` / numeric sensing 主线
- 下一步若实现 LiDAR / range finder，可在现有 `SurfaceQuerySpec + executor`
  上扩展 query origin/direction builders 和 GPU/mesh executor，不要复用
  `RenderScene` 当 query scene
- 下一步若实现 camera / render-backed depth，先创建集成层设计，不让 `sensing/`
  直接 import `rendering/`

**触发条件**：开始实现 `LiDAR / depth probe / camera` 这类非纯 numeric sensor 时。
**优先级**：P1（归属阻塞已解除；具体执行 schema 仍需在实现前细化）。

**Q54 — Optical computation workflow（PublishedFrame -> executable optical scene -> results）** (2026-04-29)

**背景**：Q53 已经把 `sensing/` / `rendering/` / future integration layer 的依赖边界守住，
并落地了第一版 `SurfaceQuerySpec` / `SurfaceQueryExecutor`。但进一步讨论光源、材质、
反射/折射、camera/RGB/optical sensors 时发现：真正缺的不是给 `PublishedFrame`
再加几个字段，而是一条明确的 optical scene synchronization pipeline。

详见：
`collab/q54-optical-computation-workflow__discussion__codex__v1.md`。

**真正需求**：

刚体仿真完成 frame `N` 后，系统需要能把该帧的物理状态与光学世界数据组合起来，
执行光学计算，并把结果交给 sensor readings / RL obs / Rerun / debug exporters 消费：

```
PublishedFrame N
  + optical geometry/material/light/medium/sensor registry
  -> OpticalSceneSnapshot N
  -> OpticalExecutor
  -> OpticalComputeResult
  -> sensing / RL / Rerun / export consumers
```

**当前 gap**：

1. `PublishedFrame` 只有物理时间线状态：`q/qdot/X_world/contact/telemetry` 等；
   它不拥有光源、材质、render/query mesh、texture、medium 或 acceleration structure。
2. `RenderScene` 是 debug/inspection snapshot，不是物理真实的 optical world model。
3. `RerunBackend` 是 visualization/logging sink，不是 optical transport executor。
4. `SurfaceQuerySpec` 只描述几何 ray batch；它适合 first-hit range/depth probe，
   不包含光照、材质、反射、折射、曝光或 camera response。
5. 缺少 `OpticalWorldRegistry / OpticalSceneCache / OpticalSceneSnapshot /
   OpticalExecutor / OpticalComputeResult` 这一层，把物理帧和光学资产变成可执行场景。

**当前倾向设计**：

- 不把 optical state 塞进 `PublishedFrame`。
- 不把 `RenderScene` 升级成 canonical optical scene。
- 不把 Rerun 当 optical executor；Rerun 只消费已经算好的 scene/result/debug artifacts。
- 新增独立 integration layer，命名为 `optics/`。`sensor_rendering/` 太窄，
  容易暗示这层只是 camera/render-backed sensing；Q54 实际覆盖 material/light/
  medium/acceleration structure/light transport contracts。
- 该层拥有：
  - `OpticalWorldRegistry`：materials/lights/media/geometry bindings；
  - `OpticalSceneCache`：CPU/GPU geometry buffers、material/light buffers、BVH/TLAS/dirty flags；
  - `OpticalSceneSnapshot`：frame-aligned immutable executable view；
  - `OpticalExecutor`：CPU/GPU/external renderer execution boundary；
  - `OpticalComputeResult`：host/device/external result buffers + readiness/fence metadata。
- optical sensor specs 留在 `sensing/`，不进入 `OpticalWorldRegistry`；
  registry 描述世界状态，sensor spec 描述每次要问的问题，二者生命周期不同。
- `OpticalSceneSnapshot` 不持有 Python-level `GpuPublishedFrame` borrow lease；
  GPU/device path 通过 Q52 device-consumer event 管 slot reclaim。

**执行后端当前建议**：

- 自己拥有 `OpticalExecutor` / `OpticalComputeResult` 合同和生命周期；
- 不立刻绑定某个完整 renderer，也不从零承诺自研完整光追；
- 先实现一个 tiny in-repo reference executor：first-hit depth + material-id
  segmentation，不做 direct-light intensity；用于验证
  `OpticalSceneSnapshot -> OpticalComputeResult`、material binding、result ownership、
  Rerun/logging 消费、未来 RL hot path 语义；
- Rerun 只作为 result/debug visualization sink，不作为 optical executor；
- 后续后端作为 adapter 接入：
  - Embree：CPU ray/intersection acceleration；
  - OptiX：未来 NVIDIA GPU ray tracing；
  - Warp/CUDA custom executor：简单传感器 kernel；
  - Mitsuba：offline/high-fidelity/reference 或 differentiable rendering。
- phase-2 minimal material schema 倾向：
  `material_id: str`、`albedo_rgb: tuple[float, float, float]`、`extension: dict`；
  PBR 字段等 executor 需要时再加。
- phase-2 minimal light schema 倾向：
  point/directional、position_or_direction、intensity、color_rgb、enabled；
  但 first reference executor 暂不消费 lights。
- multi-env batching semantics 尚未定：Phase A 可用 one-env CPU snapshot，
  Phase C 前必须决定 all-env snapshot、selected-env snapshot，还是 one snapshot per env。

**Multi-physics scene producer 补充决策**：

- 当前 `CpuPublishedFrame.X_world` 只是 Phase A rigid-body producer，不应成为
  `OpticalSceneCache` 的唯一长期输入模型。
- 后续 scene/cache 应消费 frame-aligned producer streams：
  - rigid body：frame 提供 body transforms，scene/cache 组合
    `X_world_geometry`；
  - cloth：frame 提供动态 vertices/normals，topology version 变化时 rebuild；
  - soft body：frame 提供当前 surface mesh / boundary representation；
  - fluid：frame 提供 particles / level-set / volume / solver-published surface
    mesh，registry 提供 identity/material/medium 语义。
- scene/cache 可以做 sensor-independent executable geometry preparation：
  buffer packing、dynamic vertex update、BVH/BLAS/TLAS refit/rebuild、可选
  fluid surface realization。
- scene/cache 仍不能做 ray traversal、shading、RGB/depth/intensity result、
  sensor noise/response 或 result-to-reading conversion。
- `OpticalInstanceSpec` 后续应从 rigid-only `body_index` 扩展到 binding/source
  kind：`world_static`、`rigid_body`、`deformable_mesh`、`particle_set`、
  `surface_mesh`、`volume_field`、`procedural`。
- 已引入 `OpticalFrameInputs` aggregate 作为 `OpticalSceneCache` 主入口；
  Phase A 只填 `rigid: CpuPublishedFrame`。
- `snapshot_from_published_frame(...)` 保留为便利包装，内部构造
  `OpticalFrameInputs.from_published_frame(...)`。

**Executor contract 补充决策**：

- `OpticalSceneSnapshot` 只回答 frame `N` 有什么、在哪里、如何编码；
  不做 ray traversal、shading、segmentation resolve 或 sensor response。
- `OpticalExecutor.execute(snapshot, spec)` 才做 sensor/query-specific
  optical computation，并返回 `OpticalComputeResult`。
- executor 不修改 registry，不重新取 frame，不做 asset binding，不强制 host reading，
  不把 Rerun/debug sink 作为计算路径。
- canonical channels 先固定语义：`hit_mask`、`depth_m`、`range_m`、
  `position_world`、`normal_world`、`instance_id`、`numeric_instance_id`、
  `material_id`、`numeric_material_id`、`semantic_id`、`rgb`、`intensity`。
- `range_m` 是沿 normalized ray 的真实 first-hit 距离；`depth_m` 是 camera /
  optical-axis sensor 语义下的投影深度。当前 `OpticalRaySensorSpec` reference
  executor 输出 `range_m`，不输出 `depth_m`。
- misses 语义属于 contract：`hit_mask=False`，距离为 `np.inf`，
  position/normal 为 NaN，human-readable ids 为 `None`，numeric background id
  需文档化。
- executor 分阶段：
  1. reference CPU first-hit depth + material/instance id；
  2. 内部分层 `validate / prepare_workload / intersect / resolve_channels /
     build_result`；
  3. Embree/simple CPU acceleration；
  4. direct-light/RGB capability；
  5. device result buffers + Q52 device-consumer completion；
  6. Mitsuba/offline adapter。

**2026-04-30 Phase A 落地状态**：

- 已新增 `sensing.OpticalRaySensorSpec`，保持 sensor spec 在 `sensing/`。
- 已新增 `optics/` package：
  - `OpticalWorldRegistry`
  - `OpticalMaterialSpec`
  - `OpticalLightSpec`
  - plane / triangle-mesh geometry handles
  - `OpticalInstanceSpec`
  - `OpticalFrameInputs`
  - `OpticalSceneCache`
  - `OpticalSceneSnapshot`
  - `OpticalExecutor`
  - `OpticalComputeResult`
  - `CpuReferenceOpticalExecutor`
  - `OpticalBindingBuildResult`
  - `OpticalSourceKey`
  - `build_optical_registry_from_robot_model(...)`
- Phase A snapshot 只支持 CPU one-env `CpuPublishedFrame`；
  body-bound geometry 通过 `PublishedFrame.X_world` 组合 transform。
- registry 在 `add_instance(...)` 时分配稳定 `numeric_instance_id`；cache 只携带
  和打包，不重新编号。
- `OpticalInstanceSpec.roles` 已作为最小 visibility/role 字段加入，executor
  按 `OpticalRaySensorSpec.sensor_role` 过滤 instances。
- reference executor 只做 first-hit `range_m`、`material_id`、`instance_id`、
  `numeric_instance_id`、hit position/normal；明确不做 direct-light intensity 或
  camera-style projected `depth_m`。
- 已新增 Phase-A registry builder：
  `build_optical_registry_from_robot_model(..., source_policy="collision_only")`。
  该 builder 从 `RobotModel.geometries` 生成 `OpticalWorldRegistry`、
  `OpticalInstanceSpec`、source/instance provenance maps 和 diagnostics。
- collision-derived builder 默认 roles 为 `{"depth", "lidar", "segmentation"}`，
  不默认进入 RGB。
- builder 支持可三角化 polyhedral shapes、带 faces 的 `MeshShape` 和
  `HalfSpaceShape`；对 sphere/capsule/缺 faces mesh 等暂不支持形状输出 warning
  diagnostic，不偷偷生成低保真近似。
- 新增 `tests/unit/optics/test_optics_phase_a.py` 覆盖 registry、snapshot transform、
  plane first-hit、triangle first-hit、frame mismatch。

**待 review 问题**：

1. `optics/` 命名和依赖方向是否足够清晰？
2. sensor specs 留在 `sensing/`，由 `OpticalExecutor.execute(snapshot, spec)` 消费，
   是否是正确生命周期边界？
3. first-hit depth + material-id segmentation 作为第一版 reference executor 是否足够？
4. Q52 device-consumer event 是否足以覆盖 optical snapshot 的 GPU slot 生命周期？
5. minimal material/light schema 是否还应更小？
6. 后端分阶段是否合理：reference executor -> Embree CPU -> Warp/CUDA/OptiX GPU ->
   Mitsuba offline/reference？
7. Phase C 前 multi-env batching 应选择什么语义？

**建议优先级**：

先做 Q54 decision，不急着写 reflection/refraction。确认 layer split、package name、
result ownership、device/host result lifecycle、executor adapter contract 后，再实现最小
`OpticalWorldRegistry + OpticalSceneCache + OpticalComputeResult + reference executor`
骨架。
