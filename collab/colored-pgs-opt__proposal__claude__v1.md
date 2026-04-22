Initiative: colored-pgs-opt
Stage: proposal
Author: claude
Version: v1
Date: 2026-04-22
Status: draft
Related Files: physics/backends/warp/colored_pgs_kernels.py, physics/gpu_engine.py, physics/backends/static_data.py
Owner Summary: colored_pgs 当前比 ADMM 慢 4x，根因是 Python 侧 960 次 kernel launch overhead。
  本文提出四个优化方案（A～D），以 B+D 为推荐组合，预期可达到 ADMM 同级别性能。
  请 Codex 重点审查：方案 B 的 fused kernel 设计是否会引入平台级约束；
  方案 D 的 warm starting 接触匹配策略是否足够鲁棒。

---

## Problem

`colored_pgs` solver 在 3-robot 9-body fixture（500 steps）的 benchmark 中：

```
colored_pgs:  254 ms/step
admm:          66 ms/step
jacobi_pgs_ms: 174 ms/step
```

colored_pgs 比 ADMM 慢约 4x，慢于 jacobi_pgs_ms 约 1.5x。

**根因：Python 侧 kernel launch overhead。**

当前控制流（`physics/gpu_engine.py:897-915`）：

```python
for _ in range(s.solver_max_iter):          # 60 次
    for color in range(16):                 # MAX_COLORS = 16（固定）
        wp.launch(batched_colored_pgs_step, ...)
```

每个 `wp.launch` 都有 Python→CUDA driver 的同步开销（约 50-200 µs/次）。
60 × 16 = **960 次 launch/step**，即使每次 kernel 运行时间极短（真实接触场景
通常只有 4-8 个颜色，其余 8-12 次 launch 是空 pass），overhead 依然主导。

`solver_max_iter` 默认值：60（`physics/backends/static_data.py:213`）。

**与 ADMM 对比**：ADMM 每步固定 ~2-4 次 launch，与接触数/颜色数无关，
所以其 launch overhead 可忽略不计。

---

## Goal

将 `colored_pgs` 的 step 时间从 254ms 降低到与 ADMM（66ms）同量级，
同时保持 Gauss-Seidel 收敛特性和物理正确性。

**不改变**：
- 图着色算法（`batched_greedy_coloring` 保持不变）
- 物理语义（同色无 body 冲突 → 并行安全）
- 外部 API（`GpuEngine(solver="colored_pgs")`）

---

## Scope

变更仅限于 GPU solver 路径，不涉及 CPU solver、接触检测、动力学 kernel。

受影响文件：
- `physics/backends/warp/colored_pgs_kernels.py` — 新增 fused kernel（方案 B）
- `physics/gpu_engine.py` — dispatch 逻辑修改
- `physics/backends/static_data.py` — 可能新增 `warm_start: bool` 字段（方案 D）
- `tests/gpu/solvers/test_solver_backends.py` — 新增性能回归测试

---

## Affected Files / Layers

```
physics/backends/warp/colored_pgs_kernels.py    核心 kernel 新增/修改
physics/gpu_engine.py                           dispatch 循环修改
physics/backends/static_data.py                 warm start 参数（方案 D）
tests/gpu/solvers/test_solver_backends.py        性能 + 物理一致性测试
```

依赖方向：不引入新跨层依赖，符合 CLAUDE.md 依赖规则。

---

## Proposed Design

### 方案 A — 修 MAX_COLORS 空迭代（5 行，零近似）

**原理**：当前 `for color in range(16)` 无论实际颜色数是多少都跑 16 次。
典型刚体场景颜色数为 4-8，其余全是空 pass。读取 `n_colors_out` 的最大值
后只迭代实际颜色数。

**实现**：

```python
# gpu_engine.py — coloring 后加一次同步读取
n_colors_host = sol.n_colors.numpy()          # shape (N,)，开销约 5-10 µs
max_colors_this_step = int(n_colors_host.max())

for _ in range(s.solver_max_iter):
    for color in range(max_colors_this_step):  # 不再是 16
        wp.launch(batched_colored_pgs_step, ..., inputs=[..., color, ...])
```

**预期效果**：960 → 60×6 = 360 launches（假设 6 色）→ **1.6-2.5x 加速**。

**风险**：
- 需要 GPU→CPU 的 `numpy()` 同步（约 10 µs）。但 `n_colors` 是每步都跑的
  coloring kernel 的输出，读它不会打破已有的 CUDA stream 顺序。
- 若某步骤所有 env 的 n_colors 均为 0（无接触），max=0，循环不执行 → 正确。

**缺点**：launch 数仍与 `solver_max_iter` 成正比；高迭代次数场景仍慢。

---

### 方案 B — 融合迭代循环到单个 kernel（中等工作量，零近似）

**原理**：把 Python 侧 `for iter × for color` 两层循环搬入 Warp kernel 内部。
每个线程对应一个 env，在 GPU 线程内串行完成所有迭代和颜色遍历。

**实现**：新增 `batched_colored_pgs_all_iters` kernel：

```python
@wp.kernel
def batched_colored_pgs_all_iters(
    W, W_diag, v_free,
    lambdas,                       # read-write，GS in-place 更新
    contact_active, contact_color,
    n_colors,                      # (N,) per-env 实际颜色数
    mu: float,
    max_iter: int,
    nc: int,
    max_rows: int,
):
    env = wp.tid()
    n_col = n_colors[env]

    for _iter in range(max_iter):
        for color in range(n_col):
            # --- 与现有 batched_colored_pgs_step 完全相同的逻辑 ---
            for c in range(nc):
                if contact_active[env, c] == 0:
                    continue
                if contact_color[env, c] != color:
                    continue
                # normal + friction update（与现有 kernel 相同）
                ...
```

控制流变为：

```python
wp.launch(batched_greedy_coloring, ...)      # 1 launch
wp.launch(batched_colored_pgs_all_iters, ...,
          inputs=[..., sol.n_colors, ..., s.solver_max_iter, ...])  # 1 launch
```

每步合计 **2 次 launch**（vs 当前 961 次）。

**预期效果**：launch overhead 从 ~960×100µs = 96ms 降至 ~2×100µs = 0.2ms，
理论加速比 ~480x（launch overhead 部分）。总 step 时间预期落在 30-80ms 区间
（取决于每步实际计算量）。

**风险**：
1. **kernel 运行时间变长**：单次 kernel 运行 max_iter=60 × n_colors=6 × nc=16
   次内循环。约 5760 次标量操作/env，nc=64 场景约 46080 次。单 kernel 时间
   ~10-50ms，不触发 CUDA watchdog（上限 ~30s）。
2. **Warp kernel 不支持动态循环上限中的分支预测优化**：`for _iter in range(max_iter)`
   是静态整数参数传入，Warp 可以静态展开。`for color in range(n_col)` 是 per-env
   动态值，不能展开，但 n_col≤16，分支代价小。
3. **代码重复**：contact update 逻辑需要从 `batched_colored_pgs_step` 复制到
   `batched_colored_pgs_all_iters`。可通过提取 `@wp.func` helper 避免重复。

---

### 方案 C — CUDA Graph Capture（低 effort，零近似，有条件限制）

**原理**：用 `wp.ScopedCapture` 将 coloring + 全部 PGS pass 捕获为一个 CUDA graph，
之后每步只需 `wp.capture_launch(graph)` 一次调用。Newton（linux-foundation）采用此路线。

**限制**：CUDA graph 拓扑在 capture 时固定。我们的问题是 `n_colors` 每步可能变化
（接触 topology 动态改变），导致需要按 `max_colors_seen` 捕获最大图，或在
`n_colors` 变化时 rebuild graph（开销约 5-20ms/次）。

**实现草图**：

```python
# 只有 n_colors 不变时可复用 graph
if self._cpgs_graph is None or n_colors_changed:
    with wp.ScopedCapture(device=self._device) as capture:
        for color in range(max_colors_this_step):
            wp.launch(batched_colored_pgs_step, ..., inputs=[..., color, ...])
    self._cpgs_graph = capture.graph

wp.capture_launch(self._cpgs_graph)
```

**预期效果**：若 graph 稳定复用（静态场景），接近方案 B 效果。
若每步都 rebuild（动态接触），无净收益。

**结论**：方案 C 适合静态/准静态场景（manipulation、固定机构），
不适合 RL 训练（接触 topology 高频变化）。作为 **可选优化**，不作为主线。

---

### 方案 D — Warm Starting（减少迭代次数，与 A/B 正交组合）

**原理**：缓存上一步的 `lambdas` 作为当前步的初始值，PGS 从热启动出发，
收敛到同等残差所需迭代次数从 60 降到约 5-10（Bullet/PhysX 经验值）。

**接触匹配**：使用 `(bi, bj)` body pair 作为 key 匹配跨步接触。
若 body pair 存在于上步，复用其 lambda；否则初始化为 0。

**实现**：
- `StaticRobotData` 新增 `warm_start: bool = False`（默认关闭，逐步验证后开启）
- `SolverState` 新增 `lambdas_prev: wp.array` 缓存
- coloring 前：按 body pair 做匹配，复制 prev→curr

**预期效果**：
- 方案 A + D：960×(6/16)×(8/60) = 48 launches → **20x 加速**
- 方案 B + D：2 launches + 内核时间减少 6x → **总计 ~40x 于当前**

**风险**：
1. **接触切换时的抖动**：新接触 lambda=0，接触消失时残差跳变。
   PhysX 用 `exp(-dt * ω)` 衰减旧 lambda 降低抖动；可选实现。
2. **匹配开销**：每步 O(nc) GPU kernel 做 body-pair 查表，开销可忽略。
3. **验证要求高**：warm start 影响物理轨迹，需与无 warm-start 对比
   (body position 误差 < 1mm/step)。

---

## Recommended Implementation Order

```
阶段 1（本轮）：方案 A + B
  - A 是一行安全修复，先做
  - B 是主要加速，新增 fused kernel + 修改 dispatch

阶段 2（验证后）：方案 D
  - 在 A+B 基础上叠加 warm starting
  - 需单独验证物理一致性

方案 C：留作静态场景专项优化，不进入主线。
```

---

## Test Plan

### 性能回归
- `tests/gpu/solvers/test_solver_backends.py` 新增 `test_colored_pgs_step_time`
- fixture：3-robot 9-body，500 steps
- 断言：`step_time < 100ms`（vs 当前 254ms）

### 物理一致性（关键）
- 与现有 `TestCrossSolverConsistency` 扩展：方案 B 实现的 fused kernel
  与原 `batched_colored_pgs_step` 串行版的 `lambdas` 输出逐元素 atol=1e-5 对比
- 50 步轨迹：`colored_pgs`（fused）vs `colored_pgs`（原始）的 body position 误差 < 0.1mm

### Warm starting 专项（方案 D）
- 1000 步静止 box：接触冲量误差 < 5%（vs 无 warm start）
- 100 步碰撞场景（box 落地弹跳）：不引入额外速度振荡

---

## Tradeoffs

| 方案 | 加速比（估计） | 代码改动量 | 物理近似 | RL 兼容性 |
|------|--------------|----------|--------|---------|
| A（空颜色修复）| 1.5-2.5x | ~5 行 | 无 | 完全兼容 |
| B（fused kernel）| 8-15x | ~80 行 | 无 | 完全兼容 |
| C（CUDA graph）| 2-15x | ~30 行 | 无 | 仅静态场景 |
| D（warm start）| 与 A/B 叠加 6-10x | ~60 行 | 极微量（热启动抖动）| 需验证 |

推荐：**A + B 本轮实施**，D 单独验证后再合并。

---

## References

1. **PhysX GTC 2013 — Richard Tonge**：colored GS + mass splitting on GPU；
   原始 960-launch 问题的工业实证；TGS 替代方案。
   `NVIDIA-Omniverse/PhysX`: `physx/source/gpusolver/src/PxgTGSCudaSolverCore.cpp`

2. **Bullet3 GPU (Coumans)**：local shared-memory 原子锁，颜色判断在 kernel 内部；
   `src/Bullet3OpenCL/RigidBody/` — batch impulse solver。

3. **XPBD (Macklin 2016)**：全 Jacobi，1 launch/iter，零颜色循环；
   接受更慢收敛换取更少 launch。

4. **Newton (linux-foundation)**：`wp.ScopedCapture` + CUDA graph 路线；
   `newton-physics/newton` repo，Warp XPBD integrator。

5. **Mass Splitting (Tonge et al., SIGGRAPH 2012)**：
   `jacobi_pgs_ms` 已实现，是方案 B 的对照基线。

6. **Andrews et al. MIG 2022**："Parallel Block Neo-Hookean XPBD using Graph Clustering"；
   supernode coloring 把颜色数从 ~16 压到 3-5（未纳入本轮，作为后续研究项）。

7. **CUDA Graph kernel batching (arXiv 2501.09398, 2025)**：
   iteration-batch unrolling into CUDA graphs；实测 1.4-1.5x speedup on small workloads。
