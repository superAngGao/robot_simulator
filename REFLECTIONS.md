# Robot Simulator — Reflections Log

A running record of decisions, lessons learned, and issues encountered.
Updated at the end of each development session.

---

## 2026-03-24 (session 7) — LCP Simulator integration + CollisionFilter

### LCPContactModel 接入 Simulator.step()

**问题**：LCPContactModel 之前使用占位质量（inv_mass=1.0, inv_inertia=I）和硬编码 dt=1e-3，
无法通过 Simulator 管线获取真实物理参数。

**设计决策**：扩展 `ContactModel.compute_forces()` ABC 新增 `dt` 和 `tree` 可选 kwargs。
PenaltyContactModel / NullContactModel 接受并忽略。LCPContactModel 从 `tree.bodies[i].inertia`
提取真实质量和惯量张量，使用平行轴定理将 CoM 惯量转换为 body origin 惯量。

**替代方案考虑**：
- 方案 A：`pre_step(tree, dt)` hook — 过度设计，两个信息完全可以在 compute_forces 传入
- 方案 B：LCPContactModel 构造时绑定 tree 引用 — 但 dt 仍需每步传，不如统一

**load_urdf 集成**：新增 `contact_method="lcp"` 参数。LCP 模式自动从 BodyCollisionGeometry
查找碰撞形状注册到 LCPContactModel；无碰撞几何的 link 降级为 0.01m 小球。

### CollisionFilter 碰撞过滤掩码

**设计决策**：独立 `physics/collision_filter.py`，三层过滤取交集：
1. Auto-exclude：parent-child 自动排除（kinematic tree 遍历一次性计算）
2. Bitmask：per-body group/mask uint32，双向检查 `group_i & mask_j != 0`
3. Explicit exclude：用户声明的 (i,j) pair set

**参考对标**：
- Drake `CollisionFilterDeclaration`（auto + explicit），最系统化
- MuJoCo `contype/conaffinity`（bitmask 双向），最简洁
- Bullet `filter group/mask`（bitmask 单向），我们改为双向

**集成点**：
- `AABBSelfCollision.build_pairs(collision_filter=...)` — pair generation 阶段
- `LCPContactModel(collision_filter=...)` — 为 body-body 接触预留
- `load_urdf(collision_exclude_pairs=[("arm_L","arm_R")])` — 用户 API

**性能考虑**：filter 是静态的（build 时一次性计算），`should_collide()` 是 O(1) 查询
（set lookup + 两次位运算），不影响运行时性能。

---

## 2026-03-23 (session 6) — Phase 2e GPU + Phase 2g CRBA + Phase 2f Contact

### Phase 2f: 高保真接触系统

**实现的组件**：GJK/EPA 碰撞检测、PGS LCP 约束求解器（完整 Delassus + warm starting）、
LCPContactModel（ContactModel ABC 实现）、CapsuleShape、关节 Coulomb 摩擦、AABB Tree broad-phase。

**Bug 发现与修复**：Delassus 矩阵 `W = J M⁻¹ Jᵀ` 构建中，Jacobian 行按 contact 索引存储，
但在计算 W 时跨 contact 的行引用了错误 body 的 Jacobian。修复为按 body 分组 Jacobian 行，
仅在同 body 行之间计算 W 贡献。测试覆盖补全时发现此 bug（多 body LCP 测试）。

**Warm starting 设计决策**：采用 Bullet 方案（body-local 坐标匹配，2cm 阈值），
而非 PhysX 方案（EPA feature index 精确匹配）。原因：当前 GJK/EPA 不返回 feature info，
local 坐标方案对所有碰撞类型通用。Feature index 升级路径记录在 Q18。

**接触系统 vs 主流项目差距分析**：与 MuJoCo/Bullet/Drake/PhysX 对比，
识别出 10 项差距（Q18），已完成 6 项（Delassus、warm start、Capsule、持久化、broad-phase、restitution），
剩余 4 项（碰撞过滤、接触维度、隐式积分、同 body geom 过滤）为后续优化。

---

### Phase 2e/2g:

### Decision: BatchBackend ABC + 4 GPU 后端架构

VecEnv 通过 `BatchBackend(ABC)` 委托给后端，`get_backend(name)` 工厂选择。
`StaticRobotData` 将 RobotModel 展平为连续数组供 GPU 使用。

4 个后端实现并 benchmark：NumPy (CPU fallback) → TileLang → Warp → CUDA (fused)。
CUDA fused kernel 最快（4136x vs NumPy @ N=1000），因为全物理步在单 kernel 内完成。

### Decision: CRBA 作为 ABA 的替代前向动力学

实现了 8 种前向动力学路径（见 PROGRESS.md）。核心发现：

**Fused scalar Cholesky 在 nv ≤ 64 时接近 ABA**（0.96x @ nv=30），
但 **cuSOLVER tensor core 路径反而更慢**（3 次 kernel launch + H 矩阵 global memory
读写的开销大于 tensor core 在 30×30~64×64 矩阵上的加速）。

**Tensor core (wgmma) 不适用于小矩阵的原因：**
1. wgmma M 维度最小 64，nv < 64 需要 pad，浪费计算
2. wgmma 需要 128 threads (warp group) 协作，但树遍历是 1 thread/env
3. 改为 128 threads/env 会导致 FK/RNEA 阶段 127 线程空闲，occupancy 灾难

**结论：** 对于当前目标（四足/人形，nv=10-30），CUDA ABA fused kernel 是最优选择。
CRBA + tensor core 的真正价值在 nv ≥ 128 或分组策略下才能体现。

### Lesson: TileLang DSL 的关键限制

1. `T.alloc_fragment` 不支持任意元素索引 → 改用 `T.alloc_local`
2. `T.Serial` 循环内的标量不可变 → 用 `T.alloc_local([1])` 包装累加器
3. `@T.prim_func` 注解在 global scope 求值 → 模块全局变量注入
4. `@T.macro` 无 `T.Kernel` = inline 辅助函数（类似 `@wp.func`）

### Lesson: Kernel fusion 的定量影响

同一算法 (CRBA)，不同 fusion 程度的性能差异（nv=30, N=8192）：

| Fusion 程度 | 实现 | steps/s |
|-------------|------|---------|
| 无 fusion（PyTorch 逐操作） | BatchedCRBA | ~130K |
| 部分 fusion（TileLang FK+ABA kernel + PyTorch） | TileLang | ~879K |
| **完全 fusion（单 CUDA kernel）** | CUDA CRBA-scalar | **1,583K** |

完全 fusion 比无 fusion 快 **12x**，主要消除了 Python loop overhead 和
intermediate tensor 分配/释放。

---

## 2026-03-20 (session 4) — 测试补全与 SE3 规范统一

### Bug: `spatial.py:matrix()` 与 SE3 约定不一致

`matrix()` 使用 `R`（child→parent）构造 6×6 矩阵，但 SE3 约定下速度变换矩阵应使用
`E = R.T`（parent→child）。修复后满足两个关键性质：
- `matrix() @ v == apply_velocity(v)`
- `matrix().T @ f == apply_force(f)`

此 bug 在 ABA Pass 2 惯量传递 `X^T @ I @ X` 中使用，当 X_tree 有非零旋转时会算错。
此前所有测试和 URDF 均使用 R=I 的 X_tree（PROGRESS.md 已注明），因此未暴露。
新增 `test_spatial.py::TestABAWithRotatedXTree` 验证修复（对比 Pinocchio，atol=1e-8）。

### Bug: `robot_tree.py:rnea()` 根节点重力符号错误

RNEA Pass 1 根节点加速度初始化使用 `a_gravity` 而非 `-a_gravity`。
Featherstone §5.3 明确要求初始化为 `-a_gravity`（等效于地面以 g 加速上升）。
ABA 中已正确使用 `-a_gravity`，但 RNEA 从未被独立测试因此一直是错的。
新增 `test_robot_tree.py::TestRNEA` 验证修复（含 ABA-RNEA roundtrip 和 Pinocchio 对比）。

### 发现: Pinocchio 使用 `[linear; angular]` 向量顺序

Pinocchio 的 `Motion.np` 和 `Force.np` 返回 `[linear(3); angular(3)]`，
Isaac Lab 也使用相同顺序。我们使用 Featherstone 原著约定 `[angular(3); linear(3)]`。
测试中通过 `_P6` 置换矩阵转换。已记录为 Q15（OPEN_QUESTIONS.md），
建议在测试充分后统一改为 `[linear; angular]` 与工业标准对齐。

---

## 2026-03-19 (session 3) — Phase 2d RL Environment Layer

### Decision: term 函数接收整个 env，从缓存属性读状态

Isaac Lab 的 obs term 签名是 `fn(env, **params) -> Tensor`，而不是直接传 `q`/`qdot`。
好处：term 函数可以读任何缓存属性（`X_world`、`v_bodies`、`active_contacts`），
不需要重算 FK；Phase 2e 换 Warp array 时只需改 env 的缓存属性，term 函数签名不变。

参考：Isaac Lab `ObservationManager`（`isaaclab/envs/mdp/observations.py`）。

### Decision: action clip 在 Env.step() 入口，effort limit 在 PDController 出口

两者语义不同：
- `action_clip`：训练超参数，限制神经网络输出范围，防止早期训练发散。
- `effort_limits`：物理约束，来自 URDF `<limit effort>`，硬件电机的实际力矩上限。

分开处理使两者可以独立配置，也符合 Isaac Lab 的惯例。

### Decision: __init__ 预计算静态索引（actuated_q_indices、actuated_v_indices）

用 `np.array` 存 fancy index，而不是每步重新遍历 tree.bodies。
原因：Phase 2e 批量化时这些索引直接用于 Warp array 切片，预计算是必要条件。

### Decision: VecEnv Phase 2d 用 Python for loop

Phase 2d 目标是"可运行的接口"，不是性能。for loop 实现简单、易调试，
Phase 2e 换 Warp kernel 时 `reset()`/`step()` 的输入输出签名完全不变。
参考：Isaac Lab 在 CPU 模式下也用 Python loop 作为 fallback。

### 测试覆盖现状（74 tests）

| 模块 | 测试文件 | 数量 |
|------|----------|------|
| physics/spatial + joint + robot_tree | test_aba_vs_pinocchio, test_body_velocities, test_joint_limits, test_free_fall | 24 |
| physics/contact | test_contact | 9 |
| physics/collision | test_self_collision | 13 |
| physics/integrator | test_integrator | 11 |
| robot/urdf_loader | test_urdf_loader | 6 |
| simulator | test_simulator | 4 |
| rl_env | test_rl_env | 6 |
| **合计** | | **74** |

---



### Bug: SpatialTransform 使用了 Plücker 约定而非 SE3 约定

Pinocchio ABA 对比测试发现双摆加速度不匹配，根因是 `SpatialTransform` 的三个方法
（`apply_velocity`、`apply_force`、`compose`）使用了 Featherstone Plücker 约定
（r 在子坐标系），但 `X_tree` 的构造语义是 SE3（r 在父坐标系，R 为 child→parent）。

**修复（SE3 约定统一）：**
```python
apply_velocity: [R.T@ω;  R.T@(v + ω×r)]      # r 在父坐标系
apply_force:    [R@τ + r×(R@f);  R@f]
compose:        r = self.r + self.R @ other.r
```

向后兼容：所有已有 `X_tree` 均为 `R=I`，两种约定在 R=I 时等价，39 个已有测试全部继续通过。

### Bug: ABA Pass 3 根节点重力未变换到 body frame

根节点的 `a_p` 应将世界系重力变换到 body frame：
```python
a_p = Xup_i.apply_velocity(-a_gravity)   # 正确
# 原来: a_p = Xup_i.inverse().apply_velocity(-a_gravity)  # 错误（SE3 修复前的临时补丁）
```

### Decision: 用 Pinocchio 作为 ABA 的外部基准

Pinocchio 是工业级多体动力学库，ABA 实现经过大量验证。用它作为 cross-validation
基准比手写解析解更可靠（解析解只适用于简单拓扑）。atol=1e-8 对 float64 是合理容差。

### 测试覆盖现状（68 tests）

| 模块 | 测试文件 | 数量 |
|------|----------|------|
| `physics/spatial.py` (间接) | test_body_velocities | 4 |
| `physics/contact.py` | test_contact | 9 |
| `physics/joint.py` | test_joint_limits | 14 |
| `physics/robot_tree.py` (ABA) | test_aba_vs_pinocchio + test_free_fall | 7 |
| `physics/collision.py` | test_self_collision | 13 |
| `physics/integrator.py` | test_integrator | 11 |
| `robot/urdf_loader.py` | test_urdf_loader | 6 |
| `simulator.py` | test_simulator | 4 |

**剩余缺口：** RNEA vs Pinocchio 对比、FreeJoint 浮动基座 ABA 旋转项、`forward_kinematics` 位姿正确性独立测试。

---



### Decision: Simulator.step() orchestration order

`passive_torques` is added to `tau` before the integrator call, not inside the
integrator. This keeps `physics/integrator.py` unaware of the passive-torque
concept (single responsibility) and matches the Drake pattern where
`CalcGeneralizedForces` is called by the System, not by the integrator.

### Decision: debug print in simple_quadruped.py calls FK twice per logged step

The step loop calls `sim.step()` (which runs FK internally) and then the debug
print calls `tree.forward_kinematics(q)` again for `active_contacts()`. This is
a 0.2% overhead at the 1-in-200 logging rate — acceptable for a debug example.
A production loop would cache `X_world` from the last step; deferred to Phase 2e
when Simulator gains optional state caching.

### Decision: RobotModel constructed inline in simple_quadruped.py

`build_quadruped()` returns `(tree, contact_model, self_collision)` — the
pre-Phase-2b signature. Rather than changing `build_quadruped()`, we wrap the
three objects into a `RobotModel` in `main()`. This is the minimal change and
keeps the builder function reusable for tests that don't need a full model.

---

## 2026-03-16 — Phase 1 kickoff & completion

### Decisions
- Chose **custom simulator** over Isaac Sim for research flexibility and full control over physics.
- Phase 1 uses **pure Python + NumPy** to validate correctness before any GPU optimization.
- Used **Featherstone ABA** (O(n) Articulated Body Algorithm) as the forward dynamics solver — the industry standard for multi-body robot dynamics.
- Used **penalty method (spring-damper)** for foot-ground contact. Simpler than LCP but sufficient for Phase 1.
- Used **semi-implicit Euler** as the primary integrator — better energy conservation than explicit Euler for contact-rich simulation.

### Bugs found and fixed

| Bug | Root cause | Fix |
|---|---|---|
| Gravity direction reversed | ABA Pass 3 initialized root acceleration as `+a_gravity` instead of `-a_gravity` | Set `a_p = -a_gravity` for root body (Featherstone §7.3) |
| Contact point world position wrong | Used `R.T @ pos` instead of `R @ pos` (transposed rotation) | Fixed to `R @ pos + r` |
| Contact force frame wrong | Used `X.apply_force()` (body→world direction) to transform world forces to body frame | Changed to `X.inverse().apply_force()` |
| Contact divergence on first touch | Moment arm (foot tip 0.2m from calf origin) created huge torque on lightweight calf link (0.4 kg, I~1.35e-3 kg·m²) | Placed contact point at calf body origin (zero moment arm); added PD stance controller |
| FK standing_state height wrong | Simple 2D leg geometry formula ignored lateral hip rotation (X-axis joint) | Used actual FK to measure lowest foot z, then set torso height accordingly |
| Joint angle set via wrong index | Used `v_idx` instead of `q_idx` when setting initial joint positions | Fixed to `q_idx` |

### Lessons learned
- **Coordinate frame conventions must be established and verified first.** A single transposed R caused cascading bugs in contact, FK, and force application. Write a unit test for `apply_velocity` and `apply_force` immediately in Phase 2.
- **Penalty contact is numerically stiff.** Even with semi-implicit Euler, dt must satisfy `dt < sqrt(m/k)`. For k=3000, m=0.4 kg, dt < 1.15e-2 s — but with articulation, effective mass is much smaller, requiring dt ~2e-4 s.
- **Passive (zero-torque) legged robots collapse immediately.** A PD stance controller is the minimum needed for any meaningful drop test. In Phase 2, add a proper whole-body controller.
- **Contact point placement matters enormously.** Placing contact at the link origin rather than the foot tip sacrifices geometric accuracy but gains numerical stability. A proper foot body (separate rigid body at the foot tip with realistic mass/inertia) is the correct long-term fix.
- **Validate physics with unit tests before integration.** The free-fall test (analytic vs simulated) caught the gravity sign bug early. Add more unit tests (single pendulum, energy conservation) in Phase 2.

### Simulation results (Phase 1 validation)
- Free-fall accuracy: |z_simulated − z_analytic| < 5mm at t=1s ✅
- Quadruped drop test: 4-foot contact established at t≈0.08s, stable stance maintained for 2s ✅
- No numerical divergence over 10,000 integration steps ✅

### Known limitations (to address in Phase 2)
- Contact point at calf origin, not at foot tip — geometry inaccurate
- No joint limits or collision between bodies
- Velocity approximation in contact_fn recomputes forward pass (inefficient)
- matplotlib animation is slow; unsuitable for real-time visualization

---

## 2026-03-16 — Phase 1 精度修复：足端几何 + 关节限位 + 自碰撞检测

### 背景

Phase 1 验证通过后，发现三个影响仿真物理精度的问题：
1. 接触点放在了小腿连杆原点，而非真实足尖（几何误差 0.2 m）
2. 关节角度无限制，可以无限旋转（测试中曾出现 3.57 rad 的膝关节角度）
3. 腿部可以穿透躯干（无自碰撞检测）

### 修复内容

#### Fix 1 — 独立 foot body（几何精度）

**问题根因：** Phase 1 为了绕开一个接触力矩不稳定的 bug，将接触点放在了 calf body
原点而非足尖。代价是接触点在几何上偏差了整个小腿长度（0.2 m）。

**正确做法：** 为每条腿在小腿末端添加一个独立的 `{leg}_foot` Body，通过
`FixedJoint`（位移偏移 `[0, 0, −CALF_LENGTH]`）挂接。接触点置于 foot body 原点
（= 真实足尖），力矩臂为零，消除了 Phase 1 中迫使我们妥协的那个不稳定问题。

**结构变化：** 树的 body 数量从 13 → 17；foot body 的小质量（0.05 kg）和小惯量
使其不影响整体动力学，但提供了精确的几何位置。

**关键数字验证：**
```
foot_z = 0.001 m（1 mm 离地间隙），calf_z − foot_z = 0.200 m ✓
```

#### Fix 2 — 关节限位（penalty spring-damper）

**设计选择：** 采用 **penalty 弹簧-阻尼** 而非硬约束（clamp + 速度反射），因为
penalty 方法与现有 ABA + 半隐式欧拉框架无缝兼容，不需要修改积分器。

**实现层次：**

| 层次 | 修改 | 内容 |
|------|------|------|
| `physics/joint.py` | `RevoluteJoint` 新增 `q_min/q_max/k_limit/b_limit` | 参数化限位 |
| `physics/joint.py` | `compute_limit_torque(q, qdot)` | 穿越限位时产生弹簧 + 阻尼恢复力矩 |
| `physics/robot_tree.py` | `joint_limit_torques(q, qdot)` | 全树一次性计算限位力矩 |
| `examples/simple_quadruped.py` | 每个 `RevoluteJoint` 加入限位参数 | 在 `controller()` 中叠加 `joint_limit_torques()` |

**限位设置（参考真实四足机器人）：**

| 关节 | 轴 | q_min | q_max |
|------|----|-------|-------|
| Hip（外展/内收）  | X | −0.61 rad (−35°) | +0.61 rad |
| Thigh（屈/伸）   | Y | −1.57 rad (−90°) | +1.57 rad |
| Calf（膝关节）    | Y | −2.62 rad (−150°) | +0.52 rad (30°) |

**阻尼逻辑细节：** 限位阻尼只作用于加深穿越的速度方向（下限处 ω < 0，上限处
ω > 0），避免对弹回方向的速度施加额外阻力，从而保持物理正确性。

#### Fix 3 — AABB 自碰撞检测（新模块 `physics/self_collision.py`）

**问题：** 腿部可以完全穿透躯干（无任何几何约束），这在实际机器人中物理上不可能。

**设计选择：** 选 AABB（轴对齐包围盒）而非精确碰撞，原因：
- 四足机器人的自碰撞主要场景是腿/躯干大体积干涉，不需要精确几何
- AABB 每帧开销是 O(n²) 简单乘法，对 Phase 1 的 NumPy 后端完全够用
- 为 Phase 2 GPU 化保留了接口设计空间

**OBB → 世界 AABB 投影：**
```python
world_half[i] = Σ_j |R[i,j]| * local_half[j]
```
这是将旋转 OBB 转换为保守世界 AABB 的标准公式，每步重算，无需缓存。

**碰撞对筛选：** 自动排除运动树中的直接父子对（它们在几何上相互接触），避免
spurious 碰撞力。当前注册：躯干 + 4 条小腿 = 5 bodies，10 个候选对。

**力的施加方式：** 沿最小穿透轴（MTV）对两个 body 施加等大反向的 penalty 弹簧力，
力作用点为各 body 原点（零力矩臂），这是一个合理的一阶近似。

### 经验教训

- **正确的几何结构比绕过稳定性问题更重要。** Phase 1 用"接触点放在 calf 原点"绕过
  了接触力矩不稳定，但这引入了 0.2 m 的几何误差。正确做法（独立 foot body）其实
  代码量并不大，且同样稳定。*教训：几何妥协会积累成系统性误差，应尽早偿还。*

- **penalty 方法的统一性。** ground contact、joint limit、self-collision 三者全部
  用 penalty spring-damper 实现，统一集成到 `ext_forces` + `tau` 中，无需修改 ABA
  内核。这证明 penalty 框架的可扩展性很强。

- **阻尼设计的方向性。** 关节限位的阻尼必须是单向的（只阻止继续穿越，不阻止弹回），
  否则会在限位附近引入人为的能量耗散，导致关节"粘"在限位上。

- **自碰撞的 AABB 足够用于 Phase 1。** 用精确网格碰撞检测过于重量级；AABB 的误差
  （过于保守的包围盒）在关节限位已经防止了极端配置的情况下可以接受。在 Phase 2/3
  可以升级为 GJK 或 SDF。

### 仍需解决（留给后续 Phase）

- 每步 `_compute_body_velocities()` 重新跑了一遍前向运动学 pass，与 ABA 内部重复
  计算。Phase 2 应将 body velocity 作为 ABA 的副产品缓存并直接复用。
- matplotlib 动画渲染速度慢，无法实时可视化，需要 Phase 3 的 Vulkan 渲染器。
- AABB 使用 body origin 作为包围盒中心，若 CoM 与 origin 偏差大（如大质量偏心体），
  精度会下降；Phase 2 可改为以 CoM 为中心的 AABB。

<!-- Add new entries below in the same format -->

---

## 2026-03-17 — physics/ 独立库定位 & 单向依赖规则

**决策：** `physics/` 的长期目标是成为独立的社区贡献（类似"GPU 版 Pinocchio"），
但当前阶段留在同一个 repo，待 Warp 后端 API 稳定后再提取。

**约束（写入 CLAUDE.md）：**

```
rl_env/  →  simulator.py  →  robot/  →  physics/
```

`physics/` 内部**严禁**反向 import 上层模块。这是硬规则，违反即 blocking review。

**理由：**
- Phase 2 物理层和 RL 层仍在共同演化（Warp 批量布局由 VecEnv 需求驱动），过早拆 repo 产生跨 repo 协调成本
- 但保持单向依赖边界，确保将来提取时物理层零修改可独立发布
- 参考：Pinocchio 是独立库，Isaac Lab 在其之上构建；我们的终态目标与此一致

**提取时机（未来判断标准）：**
- `RobotTreeBase` ABC 及 Warp 后端 API 已稳定
- 有外部用户场景（非 RL 的控制/动画/学术研究）需要单独使用物理层

---

## 2026-03-17 — Q9 决策：Obs/Action Space 设计

### 完整设计决策

| 项目 | 决策 |
|------|------|
| term func 签名 | `fn(env, **params) -> torch.Tensor` |
| obs 定义方式 | `dict[str, ObsTermCfg]`，提供预定义标准 cfg 函数（如 `QuadrupedObsCfg()`） |
| obs group | 先单 group（policy），critic 留 Phase 3+ |
| noise 类型 | Gaussian + Uniform |
| noise 位置 | Manager 层统一加，term 函数本身保持纯净 |
| noise 开关 | `manager.train()` / `manager.eval()`，模仿 PyTorch 风格 |
| Manager 体系 | `ObsManager`、`RewardManager`、`TerminationManager` 共享 `TermManager(ABC)`；Phase 2 只完整实现 `ObsManager`，其余留骨架 |
| tensor device | `EnvCfg` 顶层统一指定 `device: str`，所有 term 输出自动 `.to(device)` |
| obs 输出类型 | `torch.Tensor` |

### 参考

- Isaac Lab `ObservationManager`：term 是函数，cfg 是数据，Manager 是执行引擎——三者分离
- noise 在 Manager 层加，domain rand 时只改 cfg，不动 term 函数
- `device` 统一在顶层指定（Isaac Lab `sim_device` 同款做法）

### 模块结构

```
rl_env/
├── base_env.py        # Env(model, cfg) — Gymnasium 接口
├── vec_env.py         # VecEnv — N 个并行 env
├── managers.py        # TermManager(ABC) + ObsManager + RewardManager(stub) + TerminationManager(stub)
├── obs_terms.py       # 标准 obs term 函数（base_lin_vel, joint_pos, contact_mask, ...）
├── reward_terms.py    # 标准 reward term 函数（Phase 2+ 实现）
└── cfg.py             # ObsTermCfg, NoiseCfg, EnvCfg dataclasses
```

---

## 2026-03-17 — Q2 修复：body_velocities() 公开方法

**问题：** `_compute_body_velocities()` 在 `simple_quadruped.py` 里独立实现了一遍
前向速度递推，与 ABA Pass 1 内部的速度计算完全重复。每步实际跑了 3 次相同的递推
（ABA 内部 1 次 + contact 1 次 + self-collision 1 次）。

**修复：**
- `physics/robot_tree.py`：新增 `body_velocities(q, qdot) -> list[Vec6]` 公开方法
- `examples/simple_quadruped.py`：删除 `_compute_body_velocities()`，改用 `tree.body_velocities()`

**参考项目做法（一致）：**
- Pinocchio：`data.v[i]` — ABA 后直接从 `Data` 对象读，算一次缓存
- MuJoCo：`mjData.cvel` — `mj_step` 后所有 body velocity 存在 `mjData`，不重算
- Drake：`Context` 缓存所有运动学量，按需计算但只算一次

**测试：** `tests/test_body_velocities.py`（4 个测试，全部通过）

---

## 2026-03-17 — Warp 后端切换方式

**决策：** Option B — 两个独立类，共享抽象基类接口。

```
physics/
├── robot_tree.py          # RobotTreeNumpy（现有，Phase 1 baseline）
├── _robot_tree_base.py    # RobotTreeBase(ABC)：定义 aba/fk/passive_torques 等接口
└── warp_kernels/
    └── robot_tree_warp.py # RobotTreeWarp(RobotTreeBase)：GPU 实现
```

**理由：**
- NumPy 版本保留作为正确性基准，Warp 版本输出须与之对齐（数值误差在容忍范围内）。
- 共享 ABC 提供编译期接口一致性保证，避免两套实现悄悄偏离。
- 实现完全分离，Warp kernel 代码不污染 NumPy 路径，也不引入运行时 if/else。
- 参考：Isaac Lab `ArticulationView` 背后采用相同思路（不同后端实现同一接口）。

**影响：**
- 现有 `robot_tree.py` 改名或提取基类，改动量小。
- 测试可直接实例化两个类，对同一输入比较输出。

---

## 2026-03-17 — VecEnv 并行粒度决策

**决策：** Warp kernel 内部批量处理 N 个机器人（真正 GPU 并行），不用 Python 层 for loop。

**理由：**
- Python for loop 方案：N 个独立 `RobotTree` 实例，每步串行调用，GPU 利用率极低，无法达到 1000+ env 的吞吐量目标。
- Warp kernel 方案：ABA、FK、contact 全部写成 `wp.kernel`，kernel launch 时 `dim=N`，N 个机器人在 GPU 上真正并行执行，这是 Isaac Lab `ArticulationView` 的核心思路。

**影响：**
- Warp kernel 需要批量化数据布局：`q[N, nq]`、`qdot[N, nv]`、`tau[N, nv]` 等，而非单个向量。
- `RobotTree` 的 NumPy 实现保留作为正确性基准（Phase 1 baseline），Warp 版本结果须与之对齐。
- `VecEnv` 直接持有 Warp 数组，不经过 Python 层逐 env 循环。

---

## 2026-03-17 — Q8 决策：Simulator (Layer 2) 模块位置

**决策：** `simulator.py` 放在顶层包（Option B），不放在 `physics/` 内。

**参考项目调研：**
- **Pinocchio / MuJoCo**：没有 Simulator 类，物理算法直接暴露（`aba()`、`mj_step()`），用户自己写积分循环。
- **Drake**：`drake::systems::Simulator` 在独立的 `systems` 包，与物理核心 `drake::multibody::MultibodyPlant` 完全分开，通过 `DiagramBuilder` 连接。
- **Isaac Lab**：物理资产（`ArticulationView`）和 RL 环境逻辑（`isaaclab.envs`）在不同包里。

**规律：** 没有一个主流项目把 Simulator 放进物理核心包。物理算法层不知道"仿真循环"的存在。

**理由：**
1. `physics/` 是算法库（ABA、FK、contact），Simulator 是消费者/编排者，不是物理的一部分。
2. Simulator 的职责是胶水：调 `passive_torques()`、contact、integrator——属于 Layer 2，不属于 Layer 1。
3. 与"两个外部入口"约束一致：`load_urdf()` 和 `Env()`，Simulator 夹在中间，`from robot_simulator import Simulator` 比 `from robot_simulator.physics import Simulator` 更自然。

---

## 2026-03-17 — load_urdf 内部实现设计（✅ 已完成）

### 两阶段设计（已确认）

```
阶段 1: _parse_urdf(path) → _URDFData       纯 XML 解析，无物理对象
阶段 2: _build_model(_URDFData, ...) → RobotModel   构建物理对象
```

分两阶段的理由：`_URDFData` 可独立测试（不跑物理）；未来支持 SDF / MJCF
只需新增 `_parse_xxx()`，`_build_model` 复用。

### `_URDFData` 中间结构（已确认）

内部 dataclass，不对外暴露：
`_URDFInertial`, `_URDFCollision`, `_URDFLink`, `_URDFJoint`, `_URDFData`。
关键设计：
- `_URDFLink.collisions: list[_URDFCollision]` — 保留全部 `<collision>` 元素
- `_URDFJoint.friction: float` — 解析后暂存，`_build_model` 忽略（见 OPEN_QUESTIONS Q1）
- `_URDFData.root_link` — 自动探测：没有被任何 joint 引用为 child 的 link

### `_build_model` 流程（已确认到步骤 3，步骤 4 以后待续）

```
1. 探测 root link
2. 拓扑排序（BFS）→ 有序 link 列表（保证父节点先于子节点）
3. 逐 link 构建 Body（joint + inertia + X_tree）→ RobotTree   ← 已设计
4. floating_base 处理                                          ← 待续
5. 逐 link 构建 BodyCollisionGeometry
6. 用 contact_links 构建 ContactPoint 列表
7. 选 collision_method → SelfCollisionModel
8. 打包成 RobotModel
```

### 步骤 3：URDF → Body 映射（已确认）

参考：Pinocchio `buildModel()`、Drake `Parser().AddModels()`，两者做法一致。

**joint origin → X_tree：**
```python
X_tree = SpatialTransform.from_rpy(*joint.origin_rpy, r=joint.origin_xyz)
```
直接映射，无中间换算。

**inertial origin → SpatialInertia：**
CoM 偏移只影响惯量表示，**不影响 X_tree**（Pinocchio 和 Drake 均如此）。
```python
SpatialInertia(mass=link.inertial.mass,
               inertia=I_com,   # 在 CoM frame 里定义的张量
               com=link.inertial.origin_xyz)
```

**`<inertial><origin rpy>` 非零的情况：**
几乎所有真实 URDF 的 inertial rpy 都是零（主轴对齐）。
决策：先实现零 rpy，遇到非零时 log warning，不报错。
（见 OPEN_QUESTIONS 新增 Q11）

**无 `<inertial>` 的 link：**
用极小占位质量 `SpatialInertia.point_mass(1e-6, zeros)`，log warning。

**Pinocchio 坑（fixed joint 合并）：**
Pinocchio issue #1388：fixed joint 两侧 link 合并时惯量变换曾有 bug。
我们不做 fixed joint 合并（保留独立 Body），不受影响，但若未来做合并优化须
注意平行轴定理的正确应用。（见 OPEN_QUESTIONS 新增 Q12）

### `load_urdf` 最终签名（已确认）

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

- `contact_links=None` → 不设置任何接触点（显式，不自动探测）
- `self_collision_links=None` 的默认策略 → 待定（步骤 7 时讨论）

### 实现结果（2026-03-17）

全部步骤已完成并提交（commit 879f2c2）：

- floating_base=True → root body 持有 `FreeJoint("root")`，`X_tree = identity`（Pinocchio 方式 A）
- `BodyCollisionGeometry` 从 URDF `<collision>` 元素构建；MeshShape-only body 跳过并 log warning（Q7）
- `ContactPoint` 从 `contact_links` 参数构建，`position=zeros`（body origin）
- `self_collision_links=None` → 使用所有有非 Mesh 碰撞几何的 link；`collision_method="aabb"` → `AABBSelfCollision.from_geometries()`
- `RobotModel` 打包完成，6 个单元测试全部通过

### 背景

在讨论 `robot/` 层设计时，发现 `contact.py` 和 `self_collision.py` 都是具体类，
没有抽象接口，无法支持多种接触算法（LCP、SDF 等）或地形类型。

### 设计决策

#### ContactPoint 归属：公开数据结构

**决策：** `ContactPoint` 作为公开数据结构，存入 `RobotModel`，由 `PenaltyContactModel` 消费。

**理由：** `ContactPoint` 表达的是"机器人身上哪些位置可能接触环境"，属于 robot
description 的一部分，不是算法实现细节。类比 `BodyCollisionGeometry` 里的
`ShapeInstance`——几何描述是数据，算法是消费者。若藏入 `PenaltyContactModel`，
`load_urdf()` 就被迫直接依赖具体类而非 ABC。

#### 地形：独立 Terrain ABC，不进 geometry.py

**决策：** 新建 `physics/terrain.py`，定义 `Terrain(ABC)` + `FlatTerrain` +
`HeightmapTerrain`。地形不作为 `CollisionShape` 的子类。

**理由：** `CollisionShape` 是静态几何描述（查 AABB/OBB 半尺寸）；地形是
**可查询的运行时函数**（给定世界坐标返回高度和法向）。两者接口根本不同：

```python
# CollisionShape：静态
box.half_extents → [0.1, 0.05, 0.02]

# Terrain：动态查询
terrain.height_at(x, y) → 0.3
terrain.normal_at(x, y) → [0, 0.1, 0.99]
```

Drake 把地形也放进 `Shape` 体系，但代价是 `HalfSpace` 等类型和普通形状的使用
方式完全不同，造成接口混乱。独立 `Terrain` ABC 更准确地反映地形的本质，且
RL reset 时换地形（`update_terrain()`）也更自然。

#### ContactModel 抽象层

```
ContactModel(ABC)          ← 新增，compute_forces() + active_contacts()
  PenaltyContactModel      ← 现有 ContactModel 改名，逻辑不变
  NullContactModel         ← 新增，调试用
  TerrainPenaltyContactModel ← Phase 2，持有 HeightmapTerrain
```

`PenaltyContactModel` 内部的 `ground_z: float` 替换为 `terrain: Terrain`，
`FlatTerrain(z=0.0)` 行为与原来完全等价。

#### SelfCollisionModel 抽象层

```
SelfCollisionModel(ABC)    ← 新增，compute_forces() + from_geometries()
  AABBSelfCollision        ← 现有实现，加 from_geometries() 工厂方法
  OBBSelfCollision         ← Phase 2
  BVHGJKSelfCollision      ← Phase 2+
  NullSelfCollision        ← 调试用
```

### 重构成本评估

**低风险**，原因：绝大多数改动是加法，核心物理算法（ABA、FK、接触力公式）完全不动。

| 改动 | 类型 | 风险 |
|------|------|------|
| 新增 `terrain.py` | 纯新增 | 零 |
| 新增 `geometry.py` | 纯新增 | 零 |
| `ContactModel` → `PenaltyContactModel` + ABC | 改名 + 加法 | 低 |
| `ground_z` → `Terrain` | 一行改动 | 低（FlatTerrain 等价） |
| `AABBSelfCollision` 加 `from_geometries()` | 加法 | 低 |
| `simple_quadruped.py` 改类名 | 改名 | 低 |

验证方式：每步改完跑现有 drop-test，物理结果不变即通过。

### 待实施（Phase 2 前）

新文件结构：
```
physics/
├── geometry.py    ← CollisionShape 体系 + BodyCollisionGeometry（新增）
├── terrain.py     ← Terrain ABC + FlatTerrain + HeightmapTerrain（新增）
├── contact.py     ← ContactModel ABC + PenaltyContactModel + NullContactModel
├── collision.py   ← SelfCollisionModel ABC + AABBSelfCollision（替换 self_collision.py）
└── ...
```

### Context

Before starting Phase 2 (GPU + RL), we conducted a top-down requirements and
architecture analysis to prevent costly refactors later.

### User & requirements

| User type | Key need |
|-----------|----------|
| RL researcher | URDF import, Gymnasium interface, GPU parallel envs, domain rand |
| Physics researcher | Transparent/inspectable physics, pluggable models, unit-testable |
| Student / learner | Clean code, examples, documentation |

**Target scope (decided):**
- Open-source community (not single-user tool)
- General robot types: legged, manipulators, wheeled (not quadruped-only)
- Near-term: training only — hardware deployment (Phase 5) deferred

**Two and only two external-facing APIs:**
```
load_urdf("robot.urdf", ...)  →  RobotModel     # entry point for robot description
Env(model, ...)               →  Gymnasium env  # entry point for RL
```
Everything below is implementation detail. This constraint drives all layer decisions.

### Architecture: 5-layer model (adopted)

```
Layer 4: Application      (VecEnv, training scripts)
Layer 3: Task/Environment (Gymnasium env, domain rand)
Layer 2: Simulator        (single-env step, auto passive forces)
Layer 1: Physics Core     (ABA, FK, contact — backend-agnostic)
Layer 0: Math             (spatial algebra — pure math, no physics)
```

Robot description is an orthogonal configuration axis:
`URDF → robot/urdf_loader.py → RobotModel → Layer 2 Simulator`

### robot/ layer design (decided)

**`RobotModel` dataclass** bundles:
- `tree: RobotTree`
- `contact_model: ContactModel`
- `self_collision: AABBSelfCollision`
- `actuated_joint_names: list[str]`  (→ action space dimension `nu`)
- `contact_body_names: list[str]`

**`load_urdf()` API:**
```python
load_urdf(
    urdf_path: str,
    floating_base: bool = True,
    contact_links: list[str] | None = None,   # explicit, not auto-detected
    self_collision_links: list[str] | None = None,
    contact_params: ContactParams | None = None,
) -> RobotModel
```

**Design decision — explicit `contact_links`:**
Auto-detecting terminal links was rejected: robots have diverse morphologies
(cameras, IMUs, gripper fingers as leaves) and silent mis-detection is worse than
requiring an explicit argument. Since `load_urdf` is a high-exposure interface,
clarity beats convenience.

### Joint damping (decided)

Surveyed Pinocchio, Drake, MuJoCo, PyBullet:
- **All four** store damping as a joint property (Drake: constructor param,
  MuJoCo: MJCF attribute, PyBullet: `changeDynamics`).
- **Drake, MuJoCo, PyBullet** apply it automatically; Pinocchio requires manual
  addition and this is widely reported as a usability defect.

**Decision:** Damping is a property of `RevoluteJoint` (and `PrismaticJoint`).
Applied via `passive_torques(q, qdot)` on `RobotTree`, which Layer 2 Simulator
calls automatically. Users never touch it.

### Prerequisites in physics/ before robot/ can be implemented

The following changes to Layer 1 are required first:

| Change | File | Reason |
|--------|------|--------|
| Support arbitrary rotation axis (3-vector, not `Axis` enum) | `joint.py / RevoluteJoint` | URDF `<axis xyz="..."/>` allows any direction |
| Add `damping` parameter | `joint.py / RevoluteJoint, PrismaticJoint` | URDF `<dynamics damping="..."/>` |
| Replace `joint_limit_torques()` with `passive_torques()` | `robot_tree.py` | Unify limits + damping; called by Simulator |
| Add `body_velocities(q, qdot)` public method | `robot_tree.py` | Currently recomputed externally in `simple_quadruped.py`; should be first-class |

### Pending decisions (next session)

- URDF collision geometry (`<box>`, `<sphere>`, `<cylinder>`) → auto-derive
  `BodyAABB` half-extents? (seems straightforward, but `<mesh>` deferred to Phase 3)
- ~~Where does `Simulator` (Layer 2) live in the module tree?~~ → **Resolved: top-level `simulator.py`** (see below)
- How does Layer 3 Gymnasium env specify observation/action spaces generically
  enough for both legged and manipulator robots?
