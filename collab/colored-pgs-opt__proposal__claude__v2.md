Initiative: colored-pgs-opt
Stage: proposal
Author: claude
Version: v2
Date: 2026-04-22
Status: draft
Related Files: physics/backends/warp/colored_pgs_kernels.py, physics/gpu_engine.py,
  physics/backends/static_data.py, collab/colored-pgs-opt__proposal__claude__v1.md
Owner Summary: v2 修正 Codex 四条 finding：(1) 方案 D warm-start key 从 body pair 升级到
  body pair + 局部坐标最近邻点匹配；(2) 取消对 Warp loop unrolling 的错误断言；
  (3) 方案 C 的 RL 适用性判断修正为"收益不如 B 但可叠加"；
  (4) 性能断言移出主测试集改为独立 benchmark 脚本。
  主线推荐不变：A + B 本轮实施，D 单独验证后再合并。

---

## v2 Change Log（相对 v1）

| # | 来源 | 修改内容 |
|---|------|--------|
| 1 | Codex High | 方案 D warm-start key 重新设计：body pair + 局部坐标最近邻点 |
| 2 | Codex Medium | 删除方案 B 关于 Warp loop unrolling 的错误描述 |
| 3 | Codex Medium / 对话 | 方案 C 的 RL 排除理由修正，改为"可叠加但收益不如 B" |
| 4 | Codex Medium | 性能门槛测试从主测试集移至独立 benchmark 脚本 |

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
- `tests/gpu/solvers/test_solver_backends.py` — 物理一致性测试（不含 wall-clock 断言）
- `benchmarks/bench_colored_pgs.py` — 独立性能基准脚本（新建）

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
- 需要 GPU→CPU 的 `numpy()` 同步（约 10 µs）。`n_colors` 是 coloring kernel
  的输出，读它不会打破已有 CUDA stream 顺序。
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
    lambdas,
    contact_active, contact_color,
    n_colors,        # (N,) per-env 实际颜色数
    mu: float,
    max_iter: int,   # 运行时参数，非 wp.static
    nc: int,
    max_rows: int,
):
    env = wp.tid()
    n_col = n_colors[env]

    for _iter in range(max_iter):
        for color in range(n_col):
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
wp.launch(batched_greedy_coloring, ...)           # 1 launch
wp.launch(batched_colored_pgs_all_iters, ...)     # 1 launch
```

每步合计 **2 次 launch**（vs 当前 961 次）。

**关于 kernel 性能预期（v2 修正）**：

`max_iter` 是运行时整数参数，`range(max_iter)` **不会**被 Warp 静态展开
（静态展开需要 `wp.static()`，见 Warp 官方文档 codegen.html#example-static-loop-unrolling）。
内层循环的展开行为和寄存器占用需要在目标 GPU 上 profiling 验证，不应预先假设。

**已知的加速来源**（确定）：
- Python dispatch overhead 从 ~960×100µs≈96ms 降至 ~2×100µs≈0.2ms
- 单 kernel 内 GS 迭代避免 CPU-GPU 往返同步

**待 profiling 确认**：
- 单 kernel 运行时间（取决于 nc、n_col、max_iter 及寄存器压力）
- 总 step 时间的净加速比（预计 3-8x，需实测）

**其他风险**：
1. **代码重复**：contact update 逻辑需从 `batched_colored_pgs_step` 复制到新 kernel。
   通过提取 `@wp.func` helper 避免重复维护。
2. **kernel 运行时间变长**：单 kernel 60×16×nc 次内循环，约 15360 次（nc=16）到
   61440 次（nc=64）标量操作/env，不触发 CUDA watchdog（上限 ~30s）。

---

### 方案 C — CUDA Graph Capture（低 effort，零近似，可叠加）

**v2 修正**：v1 将方案 C 排除为"RL 不适用"，判断有误，在此纠正。

**实际情况**：当前 `colored_pgs` 的 launch 序列拓扑是**静态的**——
`coloring_kernel` 后接固定 `solver_max_iter × 16` 次 `pgs_step` launch，
与 `n_colors` 的数值无关（`n_colors` 变化只是数据变化，不改变 kernel 序列）。
因此 CUDA graph 可以在第一步 capture 一次，之后每步 replay，
包括 RL 训练的动态接触场景。

**与方案 B 的关系**：
- 方案 B 把 launch 数从 960 降到 2（消除 launch 次数）
- 方案 C 把每次 launch 的 Python dispatch overhead 压到接近 0（消除 per-launch 开销）
- 两者不互斥，B 之后还可叠加 C

**方案 C 单独的价值**：若不做 B，仅做 C，可以消除 960 次 Python dispatch 开销，
但不减少 GPU 上的 kernel 执行次数（空 pass 仍然执行）。

**限制**：
- 需要 Warp 版本支持 `wp.ScopedCapture`（Warp ≥ 0.11）
- CUDA graph 内不能有 Python 侧的动态分支（如 early-exit 残差检查）
- 若用 `wp.capture_while`（Warp ≥ 1.9，实验性），可支持条件节点

**结论**：方案 C 作为方案 B 之后的**可叠加优化**，本轮不作为主线。

---

### 方案 D — Warm Starting（减少迭代次数，与 A/B 正交）

**v2 核心修正：warm-start key 重新设计**

v1 用 `(bi, bj)` 作为跨步匹配 key，在多点 manifold 下有根本缺陷：
同一 body pair 上的 4 个接触点（box-ground、box-box）会被映射到同一历史 lambda，
把不同约束行的冲量错误合并，等于用"最深点"的历史冲量初始化其余三个点。
这不是边角 case——box-ground 4 点 manifold 是当前仓库的主测试场景之一
（`tests/gpu/solvers/test_solver_backends.py`、`tests/gpu/collision/test_b5_d4d8_mixed_ground.py`）。

**正确的匹配策略：body pair + 局部坐标最近邻点**

对于当前步骤中 env `e` 的接触点 `c`（body pair `(bi, bj)`，接触位置 `p_world`）：

1. 将 `p_world` 转换到 body_i 的局部坐标系：`p_local = R_i^T (p_world - t_i)`
2. 在上一步所有属于同一 `(bi, bj)` 的历史接触点中，找局部坐标距离最近的点
3. 若距离 < 阈值（推荐 5mm）→ 复用其历史 lambda；否则初始化为 0

这与 Bullet `btPersistentManifold` 的匹配逻辑一致，但无需完整 persistent manifold 结构。

**实现要点**：

```python
# SolverState 新增
lambdas_prev:     wp.array2d(float32)   # 上步 lambdas，(N, max_rows)
contact_pos_prev: wp.array2d(float32)   # 上步接触点位置，(N, max_contacts, 3)
contact_bi_prev:  wp.array2d(int32)
contact_bj_prev:  wp.array2d(int32)
contact_active_prev: wp.array2d(int32)
```

新增 `warm_start_init` kernel（在 coloring 之后、lambdas.zero_() 之前调用）：
- 对每个当前接触点，在上步同 body pair 点中找最近邻（O(nc²) per env，nc 小）
- 距离 < 5mm → 复用 lambda；否则置 0

**StaticRobotData 新增**：

```python
warm_start: bool = False          # 默认关闭，验证后开启
warm_start_threshold: float = 0.005  # 5mm
```

**预期效果**：
- 与方案 A 组合：360 × (60→8) = 48 launches
- 与方案 B 组合：2 launches，内核时间减少约 7x

**风险**：
1. **接触切换抖动**：新接触 lambda=0 正确；旧接触消失时历史 lambda 自然丢弃。
   若接触点在 5mm 范围内滑动，复用 lambda 可能引入微量误差，
   可选加 `exp(-dt/τ)` 衰减系数（PhysX 做法）。
2. **位置转换开销**：需要存储并传入 body transform（已有 `q` 数组，从中提取）。
3. **验证要求高**：warm start 影响物理轨迹，必须独立验证后再合并主线。

---

## Recommended Implementation Order

```
阶段 1（本轮）：方案 A + B
  - A：5 行安全修复，先做，立即可量化收益
  - B：新增 fused kernel + 修改 dispatch，主要加速来源

阶段 2（独立验证后）：方案 D
  - warm-start key 按 v2 重新设计（最近邻点匹配）
  - 需独立测试物理一致性，与方案 B 基线做对比

方案 C：方案 B 之后可考虑叠加，非本轮主线。
```

---

## Test Plan

### 物理一致性（放入主测试集）

1. **Fused kernel 等价性验证**（方案 B 核心）
   - fixture：box-ground 4 点 manifold + box-box，50 步
   - 断言：fused kernel 与原 `batched_colored_pgs_step` 串行版的
     `lambdas` 输出逐元素 atol=1e-5
   - 目的：确认方案 B 不改变数值结果

2. **Cross-solver 一致性**（已有，扩展）
   - 现有 `TestCrossSolverConsistency` 覆盖方案 B 后的 `colored_pgs`
   - 50 步轨迹 body position 偏差 < 0.05m，max|qdot| < 50 rad/s

3. **Warm starting 物理一致性**（方案 D，独立验证）
   - 1000 步静止 box：与无 warm start 的 body position 偏差 < 1mm/step
   - 100 步弹跳场景：不引入额外速度振荡（max|qdot| 不超过无 warm start 的 110%）
   - box-ground 4 点 manifold：每个接触点的 lambda 历史独立，无跨点污染

### 性能基准（独立脚本，不进主测试集）

新建 `benchmarks/bench_colored_pgs.py`：
- fixture：3-robot 9-body，500 steps，各 solver
- 输出：ms/step 对比表（不做 assert，仅记录）
- 在 OPEN_QUESTIONS.md Q46 条目的 benchmark 基线里引用

---

## Tradeoffs

| 方案 | 预期加速（估计） | 代码改动量 | 物理近似 | RL 兼容性 |
|------|--------------|----------|--------|---------|
| A（空颜色修复）| 1.5-2.5x | ~5 行 | 无 | 完全兼容 |
| B（fused kernel）| 3-8x（需 profiling）| ~80 行 | 无 | 完全兼容 |
| C（CUDA graph）| B 之后再 +10-30% | ~30 行 | 无 | 完全兼容（见 v2）|
| D（warm start）| 与 A/B 叠加 5-8x | ~100 行 | 微量热启动抖动 | 需验证 |

方案 B 的加速比从 v1 的"8-15x"降级为"3-8x（需 profiling）"，
因 loop unrolling 假设已移除。

---

## References

1. **PhysX GTC 2013 — Richard Tonge**：colored GS + mass splitting on GPU；
   `NVIDIA-Omniverse/PhysX`: `physx/source/gpusolver/src/PxgTGSCudaSolverCore.cpp`

2. **Bullet3 GPU (Coumans)**：local shared-memory 原子锁，颜色判断在 kernel 内部；
   `src/Bullet3OpenCL/RigidBody/` — batch impulse solver。

3. **Bullet `btPersistentManifold`**：warm starting 匹配逻辑参考，
   用局部坐标最近邻点（而非 body pair 聚合）维护接触历史。

4. **XPBD (Macklin 2016)**：全 Jacobi，1 launch/iter，零颜色循环。

5. **Newton (linux-foundation)**：`wp.ScopedCapture` + CUDA graph 路线。

6. **Warp 官方文档**：
   - Loop unrolling: `codegen.html#example-static-loop-unrolling`（需 `wp.static()`）
   - Conditional graph nodes: `user_guide/runtime.html#conditional-execution`（`wp.capture_while`）

7. **Mass Splitting (Tonge et al., SIGGRAPH 2012)**：`jacobi_pgs_ms` 对照基线。

8. **Andrews et al. MIG 2022**："Parallel Block Neo-Hookean XPBD using Graph Clustering"；
   supernode coloring（未纳入本轮）。
