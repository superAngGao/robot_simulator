# GPU ADMM 求解器 — 算法文档

> 速度级 ADMM 约束求解器，运行在 GpuEngine 管线 step 8。
> 实现文件：`physics/backends/warp/admm_kernels.py`

## 概览

GPU ADMM 是 CPU `ADMMQPSolver`（加速度级）的速度空间等价实现。
对 FreeJoint 自由体（球/箱），GPU 结果与 MuJoCo 匹配到亚微米（z-L2 = 0.3µm）。

```
CPU ADMMQPSolver (acceleration-level):
  min  ½ f^T (A+R) f + (a_uc - a_ref)^T f    s.t. f ∈ K
  变量: f (力), 单位 [N]

GPU batched_admm_solve (velocity-level):
  min  ½ λ^T (W+R) λ + c^T λ                 s.t. λ ∈ K
  变量: λ (冲量), 单位 [N·s]
  等价关系: λ = f·dt,  W = A,  c = dt·(a_uc - a_ref)
```

## 管线数据流

```
Step 2:  FK(q, qdot)          → v_bodies         (当前体速度)
Step 3:  ABA(q, qdot, τ)      → qacc_smooth      (无约束加速度)
Step 4:  v_predicted = qdot + dt·qacc_smooth
Step 5:  FK(q, v_predicted)   → v_bodies_pred     (预测体速度)
Step 6:  Collision detect     → contact_{normal, point, bi, bj, depth, active}
Step 7:  W-build(v_bodies_pred) → v_free, W, W_diag, J_body
Step 7b: v_current(v_bodies)  → v_current         ← ADMM 专用
Step 8:  ADMM(W, v_free, v_current, depth) → lambdas
Step 9:  impulse_to_gen(lambdas)            → gen_impulse
```

## 五个关键量

| 符号 | 名称 | 定义 | 来源 | 维度 |
|------|------|------|------|------|
| **v_c** (`v_current`) | 当前约束速度 | J @ qdot | Step 7b: 用 step-2 `v_bodies` 投影 | (max_rows,) |
| **a_uc** | 无约束约束加速度 | J @ qacc_smooth | 隐含: (v_free - v_c) / dt | (max_rows,) |
| **v_free** | 预测约束速度 | J @ v_predicted = v_c + dt·a_uc | Step 7: 用 step-5 `v_bodies_pred` 投影 | (max_rows,) |
| **a_ref** | 参考加速度 | -b·v_c + k·d·depth | ADMM kernel 内计算 | (max_rows,) |
| **rhs_const** | ADMM 常数右端 | dt·(a_ref - a_uc) | ADMM kernel 内计算 | (max_rows,) |

**关系**:
```
v_free = v_c + dt · a_uc           (定义)
rhs_const = dt·a_ref - (v_free - v_c)   (展开)
          = dt·(-b·v_c + k·d·depth) - v_free + v_c
```

## rhs_const 公式推导

CPU ADMM 的 QP（加速度空间）：
```
min  ½ f^T (A+R) f + (a_uc - a_ref)^T f
```
最优条件: (A+R) f = a_ref - a_uc = rhs_const_accel

令 λ = f·dt 转换到速度空间：
```
min  ½ λ^T (W+R) λ + dt·(a_uc - a_ref)^T λ
```
rhs_const_vel = dt · rhs_const_accel = dt · (a_ref - a_uc)

展开（**精确公式，无近似**）：
```
rhs_const = dt·a_ref(v_c) - dt·a_uc
          = dt·(-b·v_c + k·d·depth) - (v_free - v_c)
```

### 平衡态验证 (v_c = 0)

```
v_free = 0 + dt·(-g) = -dt·g
rhs = dt·(k·d·depth) - (-dt·g) + 0
    = dt·k·d·depth + dt·g

求解: (W+R)·λ = dt·k·d·depth + dt·g
      λ = m·dt·g  (平衡时恰好补偿重力)
      depth_eq = m·R·g / (k·d) ≈ 0.37 mm  ✓ (匹配 MuJoCo)
```

### 碰撞态验证 (v_c = -2 m/s, depth = 5mm)

```
a_ref = 105.26 × 2 + 2770 × 0.9 × 0.005 = 223.0 m/s²
a_uc ≈ -9.81 m/s²
rhs = dt·(223 - (-9.81)) = 2e-4 × 232.8 = 0.0466
λ = 0.0466 / 2.111 = 0.0221
Δv = 0.0221 m/s/step → ~91 步停止 (18ms)  ✓ (匹配 MuJoCo)
```

## Compliance 模型 (MuJoCo solimp/solref)

### Impedance 函数

```python
d = impedance(depth, solimp)  # [0, 1], piecewise power-law sigmoid
# solimp = (d_0, d_width, width, midpoint, power)
# d_0=0.9 (浅穿透阻抗), d_width=0.95 (深穿透阻抗)
```

### 合规矩阵 R

```
R_diag[i] = (1 - d) / d × |W[i,i]|
```

R 使接触变"软"——允许一定穿透换取稳定性。

### Spring-damper 参数

```python
# solref = (timeconst, dampratio)
b = 2 / (d_width × timeconst)          # 阻尼率 [1/s]
k = 1 / (d_width² × timeconst² × dampratio²)  # 弹簧刚度 [1/s²]
```

## ADMM 迭代

```
预处理（每步一次）:
  AR_rho = W + diag(R) + ρI
  L L^T = AR_rho                    # in-kernel scalar Cholesky

迭代（固定 admm_iters 次，GPU 无 early exit）:
  f-update:  rhs = rhs_const + ρ(s - u)
             f = L^{-T} L^{-1} rhs   # 三角求解
  s-update:  s = proj_K(f + u)       # 锥投影
  u-update:  u += f - s              # 对偶更新

输出:  lambdas = s  (已在锥内)
```

### 锥投影 (Friction Cone)

```
对每个 active contact (3 行: normal + tangent1 + tangent2):
  s_n = max(0, z_n)                  # 法向 ≥ 0
  if ||z_t|| > μ·s_n:                # 切向超出摩擦锥
    s_t = z_t × (μ·s_n / ||z_t||)   # 缩放到锥边界
  else:
    s_t = z_t
```

## Warmstart

跨时间步持久化 (f, s, u, n_active)：
- 接触数不变 → 复用上步的 f, s, u 作为初始值
- 接触数改变 → 冷启动（全零）
- `GpuEngine.reset()` 清零 warmstart 状态

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rho` | 1.0 | ADMM 惩罚参数（固定，GPU 不做自适应） |
| `admm_iters` | 30 | ADMM 迭代次数（固定，GPU 不做 early exit） |
| `warmstart` | True | 跨步 warmstart |
| `solref` | (0.02, 1.0) | (timeconst, dampratio) |
| `solimp` | (0.9, 0.95, 0.001, 0.5, 2.0) | (d_0, d_width, width, midpoint, power) |
| `reg` | 1e-4 | Cholesky 正则化 |

## 已知限制

1. **多体同时接触发散**：两球同时着地+互碰时 ~1000 步后 NaN（Q28）
2. **Body-level Delassus**：W 用单体逆质量，不含关节耦合。
   FreeJoint 精确，铰接体近似。后续可升级 joint-space 路径
3. **固定 ρ**：GPU 不做 adaptive rho（避免条件分支重分解）

## 文件

| 文件 | 内容 |
|------|------|
| `physics/backends/warp/admm_kernels.py` | 主 kernel + Cholesky/投影/impedance 辅助函数 |
| `physics/backends/warp/solver_kernels_v2.py` | `batched_compute_v_current` kernel |
| `physics/backends/warp/solver_scratch.py` | ADMM scratch 数组分配 |
| `physics/gpu_engine.py` | solver dispatch (`solver="admm"`) |
| `physics/solvers/mujoco_qp.py` | CPU 参考实现 (`ADMMQPSolver`) |
