# MuJoCo 风格接触求解器 — 原理与实现

> 本文档描述 `physics/solvers/mujoco_qp.py` 和 `physics/implicit_contact_step.py` 的
> 数学原理、架构决策，以及实现过程中发现和修复的 bug。
>
> 代码对应：
> - `physics/solvers/mujoco_qp.py` — MuJoCoStyleSolver
> - `physics/implicit_contact_step.py` — ImplicitContactStep
> - `physics/robot_tree.py` — contact_jacobian()

---

## 1. 问题：旧 ADMM-C 为什么匹配不了 MuJoCo？

旧 ADMM-C（`physics/solvers/admm.py`）对标 MuJoCo 的软接触模型，但实测 z-L2 = 2.3mm
（与 Bullet 到 MuJoCo 的距离 2.6mm 几乎一样）。诊断发现四个根本原因：

| # | 问题 | 影响 |
|:--|:--|:--|
| 1 | **往返损耗** | ADMM 算出 v_new → 转 impulse → /dt → ABA 重算 → 再积分，信息丢失 |
| 2 | **Body-space 坐标** | 块对角 M 忽略关节耦合，30-DOF 机器人误差显著 |
| 3 | **Bias 偏移合规** | 合规进约束右端（bias），非严格凸 QP，无唯一平衡态 |
| 4 | **离散接触跳变** | depth > 0 才有力，导致 on-off 振荡 |

### 关键调研发现

对 MuJoCo 源码的调研揭示了一个颠覆性事实：

> **MuJoCo 的接触力也是显式的** — 在当前状态求解，不耦合进隐式积分器。
> 隐式积分（`W_hat = M - h*D`）只处理平滑力（阻尼、致动器、科里奥利力）。

这意味着差距不是"显式 vs 隐式"的架构问题，而是 QP 公式和求解流程的技术差异。

---

## 2. MuJoCo 的接触 QP

### 2.1 对偶问题（MuJoCo 的 Newton/CG 实际求解的）

```
f* = argmin  1/2 f^T (A + R) f  +  f^T (a_u_c - a_ref)
     s.t.    f in Omega  (摩擦锥)
```

各项含义：

| 符号 | 定义 | 物理含义 |
|:--|:--|:--|
| f | 约束空间接触力 | 法向力 + 切向摩擦力 |
| A | J H^{-1} J^T | 约束空间逆惯量（关节空间） |
| R | (1-d)/d * diag(A) | 合规正则化（自适应） |
| a_u_c | J * a_u | 无约束约束空间加速度 |
| a_ref | -b*v_c + k*d*depth | 弹簧-阻尼参考加速度 |

### 2.2 关节空间 vs Body 空间

```
旧 ADMM (body-space):
  M = diag(m1*I3, I1, m2*I3, I2, ...)    块对角，6*nb 维
  忽略关节耦合

MuJoCo (joint-space):
  H = CRBA(q)                              稠密，nv 维
  A = J * H^{-1} * J^T                     完整约束空间惯量
  捕获所有关节耦合
```

双摆示例：link2 上的接触力影响 joint1 的加速度，但 body-space 块对角 M 看不到这个耦合。
joint-space 的 H 是稠密的，H_{12} != 0 精确表达了这个耦合。

### 2.3 R 正则化 vs Bias 偏移

这是最关键的数学差异。同样是"软接触"，位置不同：

| | 旧 ADMM (Bias) | MuJoCo (R 正则化) |
|:--|:--|:--|
| 数学位置 | 约束右端 Jv + bias in K | 目标函数 1/2 f^T **R** f |
| 物理含义 | "速度应该是 bias" | "使用接触力是有代价的" |
| 凸性 | 不改变 QP 凸性 | **使 QP 严格凸**（R > 0） |
| 唯一解 | 不保证 | **保证**唯一稳定平衡 |
| 平衡态 vz | != 0（振荡） | = 0（精确） |

**R 自适应**：R_ii = (1-d)/d * A_ii。分子 (1-d)/d 来自 solimp 阻抗参数（用户可调），
分母 A_ii 来自机器人的关节空间惯量（自动计算）。重机器人 A_ii 小 → R 小 → 接触硬；
轻机器人 A_ii 大 → R 大 → 接触软。无需用户按机器人手动调参。

### 2.4 solref/solimp 参数模型

**solref = (timeconst, dampratio)** — 弹簧-阻尼参考加速度：

```
b = 2 / (d_width * timeconst)         阻尼系数
k = 1 / (d_width^2 * timeconst^2 * dampratio^2)   刚度系数

a_ref = -b * v_constraint + k * d(r) * depth

安全机制: timeconst = max(timeconst, 2*dt)  防止弹簧频率高于积分器
```

**solimp = (d_0, d_width, width, midpoint, power)** — 阻抗函数 d(r)：

```
d(r) 在 [d_0, d_width] 范围内，随穿透深度从 d_0 过渡到 d_width
  d_0 = 0.9     刚接触时 90% 硬约束
  d_width = 0.95 深穿透时 95% 硬约束
  width = 0.001  1mm 内完成过渡

过渡曲线: 分段幂律样条（MuJoCo engine_core_constraint.c 中的 getimpedance）
  x = |depth| / width, clamped to [0, 1]
  if x <= midpoint: y = x^power / midpoint^(power-1)
  else: y = 1 - (1-x)^power / (1-midpoint)^(power-1)
  d(r) = d_0 + y * (d_width - d_0)
```

---

## 3. 实现：五步修改

### S1. 关节空间接触雅可比 — `contact_jacobian()`

**文件**: `physics/robot_tree.py`

**原理**: 将 qdot 映射到接触点世界线速度。沿运动链从 body 到 root 遍历：

```
v_point = v_lin + omega x r

对每个关节 j:
  S_j = 关节运动子空间 (6, nv_j)，body 坐标系
  R_j = body-to-world 旋转矩阵
  r = point_world - joint_origin_world

  J[:, v_idx_j] = R_j @ S_j[:3] + cross(R_j @ S_j[3:], r)
                  ——————————————   ————————————————————————
                  线速度贡献          角速度通过力臂产生的线速度
```

**验证**: 数值微分（`(FK(q+eps) - FK(q))/eps`）+ Pinocchio `computeFrameJacobian`。

### S2. MuJoCo 风格 QP 求解器 — `MuJoCoStyleSolver`

**文件**: `physics/solvers/mujoco_qp.py`

**流程**:
1. 构建关节空间 Jacobian J（使用 S1 的 contact_jacobian）
2. 构建 A = J @ H^{-1} @ J^T（H 来自 CRBA）
3. 构建 R = (1-d)/d * diag(A)（自适应正则化）
4. 计算 a_ref（solref 弹簧-阻尼模型）
5. ADMM 求解对偶 QP

**ADMM 迭代**:
```
f = (A+R+rho*I)^{-1} * (rhs_const + rho*(s-u))
s = proj_cone(f + u)
u = u + f - s
```

### S3. 直连积分器 — `ImplicitContactStep`

**文件**: `physics/implicit_contact_step.py`

**核心改进**: 消除 impulse -> force -> ABA 往返。

```
旧流程（有损）:
  ADMM → v_new → impulse = M*(v_new-v) → force = impulse/dt → ABA → 再积分

新流程（无损）:
  a_u = ABA(q, qdot, tau)                    # 无接触加速度
  f, J = solver.solve(contacts, ...)          # 接触力（加速度级）
  a_c = ABA(q, qdot, tau + J^T @ f)          # 受约束加速度
  qdot_new = qdot + dt * a_c                  # 积分

数学精确性:
  H * a_c = tau + J^T*f - C                   (牛顿方程)
  H * a_u = tau - C                           (无接触)
  → a_c = a_u + H^{-1} * J^T * f             (精确，无近似)
```

两次 ABA 调用，零信息损失。第二次 ABA 隐式完成了 H^{-1} 运算。

### S4. 接触边距 — `ground_contact_query(margin=...)`

**文件**: `physics/gjk_epa.py`

在 depth > -margin 时即生成 ContactManifold。depth 可为负（间隙，尚未穿透）。
求解器使用 max(0, depth) 作为弹簧压缩量。消除接触 on/off 离散跳变。

### S5. 集成

**文件**: `simulator.py`

Simulator 检测 solver 类型：MuJoCoStyleSolver 走 ImplicitContactStep 路径，
旧求解器（PGS/PGS-SI/ADMM）走原有 impulse -> force -> ABA 路径。两路径共存。

---

## 4. 实现过程中发现的 Bug

### Bug 1: a_ref 符号错误

```python
# 错误: a_ref = -b*v - k*d*depth   (depth 正值时向下推！)
# 正确: a_ref = -b*v + k*d*depth   (depth 正值时向上推)

# 原因: MuJoCo 用 r = signed_distance (负值=穿透), 我们用 depth (正值=穿透)
# MuJoCo: a_ref = -b*v - k*d*r,  r = -depth
#        = -b*v - k*d*(-depth)
#        = -b*v + k*d*depth
```

**影响**: 球掉穿地面（634mm 穿透）。修复后穿透降到 25mm。

### Bug 2: ADMM 单次迭代退出（最关键的 bug）

**现象**: ADMM 在第 1 次迭代后就停止。输出的力只有正确值的 51%。

**根本原因**: 收敛判据只检查 primal residual `||f - s||`，没检查 dual residual。

```
ADMM 迭代 1:
  f = (A+R+ρI)^{-1} * rhs = 221.4 / 2.053 = 107.86    (正确值 210.3 的 51%)
                                        ^
                                   ρI 稀释了分母

  s = proj_cone(f) = max(0, 107.86) = 107.86           (f > 0，已在锥内)

  primal = ||f - s|| = 0    ← 触发退出！
  dual   = ||ρ(s_new - s_old)|| = 107.86    ← 巨大，不应退出
```

**为什么 primal = 0**: 球落地时接触力为正（向上），已满足 f_n >= 0 的锥约束。
所以 s = f（投影不改变任何东西），primal residual = 0。但 f 不是 QP 最优解——
它被 ρI 稀释了。ADMM 需要继续迭代让 s 追上 f，通过 ρ*(s-u) 项补偿 ρI。

**数学**: ADMM 在此场景下的收敛过程：

```
f_k = (1 - alpha^k) * f*     其中 alpha = rho/(A+R+rho)

rho = 1.0, A+R = 1.053:  alpha = 1/2.053 = 0.487

k=1:  f = 107.9  (51% of f*=210.3)
k=5:  f = 199.8  (95%)
k=10: f = 209.9  (99.8%)
k=50: f = 210.3  (精确)
```

**修复**:

```python
# 修复前: 只检查 primal
if primal_res < tolerance:
    break

# 修复后: primal AND dual 都必须小于 tolerance
if primal_res < tolerance and dual_res < tolerance:
    break
```

**效果**: z-L2 从 5.3mm 降到 **0.000mm**（与 MuJoCo 完美匹配）。

### Bug 3: b/k 公式中 impedance 参数错误

```python
# 错误: 用 d(0) = d_0 = 0.9 (表面阻抗)
d_imp = self._impedance(0.0)
b = 2.0 / (d_imp * timeconst)

# 正确: 用 d_width = solimp[1] = 0.95 (MuJoCo 文档明确规定)
d_w = self.solimp[1]
b = 2.0 / (d_w * timeconst)
```

影响较小（b 差 5%），但为匹配 MuJoCo 的精确参数行为必须修正。

---

## 5. 验证结果

| 指标 | 旧 ADMM-C | 新 MuJoCoStyleSolver | MuJoCo |
|:--|:--:|:--:|:--:|
| 全程 z-L2 vs MuJoCo | 2.3 mm | **0.000 mm** | 0 |
| 静止 vz | 0.003 m/s | **0.0** | 0.0 |
| 最大穿透 | 1.3 mm | **14.13 mm** | 14.13 mm |
| 安定步数 | 328 (不完全) | **371** | 371 |
| 碰撞时 vz@step203 | -0.010 | **-1.7811** | -1.7811 |

穿透从 1.3mm 增大到 14.13mm 是**正确的**——这意味着接触从"几乎硬"变为"MuJoCo 风格软"，
碰撞响应从"瞬间反弹"变为"弹簧-阻尼渐进减速"。

---

## 6. 为什么 MuJoCo 能做到 vz = 0（精确零）

R 正则化保证了唯一稳定平衡态。在平衡态：

```
a_c = 0 (零加速度), v = 0 (零速度)

a_u = -g (重力)
a_ref = +k*d*depth_eq (弹簧，与穿透深度成正比)

QP 最优力: f_eq = (A+R)^{-1} * (a_ref_eq + g)
受约束加速度: a_c = -g + A * f_eq = 0    ← 精确

v_new = v + dt * 0 = 0                    ← 永远精确
```

(A+R) 正定 → QP 有唯一极小值 → 平衡态唯一 → 一旦到达就不再偏离。

旧 ADMM-C 做不到这一点，因为 bias 偏移在 depth > 0 时总是非零，
驱动求解器推 v > 0（远离接触面），导致永久振荡。

---

## 7. 参考文献

1. Todorov (2014) — Convex and analytically-invertible dynamics with contacts and constraints
2. Boyd et al. (2011) — Distributed Optimization via ADMM, Section 3.3 (convergence criteria)
3. MuJoCo Computation Documentation — https://mujoco.readthedocs.io/en/stable/computation/
4. MuJoCo Source: engine_core_constraint.c — https://github.com/google-deepmind/mujoco
5. Featherstone (2008) — Rigid Body Dynamics Algorithms, Chapter 6 (CRBA), Chapter 7 (ABA)
6. Siciliano et al. (2009) — Robotics: Modelling, Planning and Control, Section 3.3 (Geometric Jacobian)
