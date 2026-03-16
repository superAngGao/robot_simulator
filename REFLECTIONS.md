# Robot Simulator — Reflections Log

A running record of decisions, lessons learned, and issues encountered.
Updated at the end of each development session.

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
