Initiative: colored-pgs-opt
Stage: implementation-note
Author: claude
Version: v1
Date: 2026-04-22
Status: implemented
Related Files: collab/colored-pgs-opt__proposal__claude__v2.md
Commit: 7edff42
Owner Summary: 方案 A+B 已实装。colored_pgs 每步 kernel launch 从 960 次降到 2 次。
  1057 个 fast 测试全部通过，物理一致性测试（跨 solver 轨迹对比）无退步。
  方案 D（warm starting）延后，待独立验证。

---

## Open Questions Addressed

- **Q46 item 4 — Colored GS 性能**：部分推进。原描述"960 kernel launch/step 的 overhead
  在大 N_envs 下是否可接受"的根因已消除（launch 数从 960 → 2）。
  实际 ms/step 加速比仍需在目标 GPU 上 benchmark 验证（状态：P1 open → 实装完成，待 benchmark）。

## REFLECTIONS.md / PROGRESS.md Impact

不需要更新。本次是 solver 内部性能优化，不改变对外 API 或物理能力。
PROGRESS.md 的 Phase 2 GPU 状态不变。

---

## What Changed

### `physics/backends/warp/colored_pgs_kernels.py`

新增 `batched_colored_pgs_all_iters` kernel：

- 每个 GPU 线程对应一个 env
- 线程内部循环：`for _iter in range(max_iter): for color in range(n_colors[env_id]):`
- 内层是与原 `batched_colored_pgs_step` 完全相同的 normal + friction update 逻辑
- `n_colors[env_id]` 是 per-env 动态值，各 env 只迭代自己的实际颜色数（方案 A 内化）
- GS 语义保持：同色内并行安全（不共享 body），异色间串行（线程内顺序循环）

原 `batched_colored_pgs_step` 保留（未删除），供参考或未来 CUDA graph 路线使用。

### `physics/gpu_engine.py`

`colored_pgs` dispatch 路径（原 `gpu_engine.py:881-915`）改为：

```
before: 1 launch (coloring) + solver_max_iter × 16 launches (pgs_step) = 961 次
after:  1 launch (coloring) + 1 launch (pgs_all_iters)                 = 2 次
```

import 由 `batched_colored_pgs_step` 换为 `batched_colored_pgs_all_iters`。

---

## Files Touched

| 文件 | 改动类型 |
|------|--------|
| `physics/backends/warp/colored_pgs_kernels.py` | 新增 kernel（+70 行），更新 module docstring |
| `physics/gpu_engine.py` | dispatch 循环替换（-26 行，+19 行）|
| `collab/colored-pgs-opt__proposal__claude__v1.md` | 新增（proposal 原稿）|
| `collab/colored-pgs-opt__proposal__claude__v2.md` | 新增（proposal 修订版，含 Codex v1 challenge 反馈）|

---

## Tests Added / Updated

本次未新增测试——改动路径已被现有测试覆盖：

- `tests/gpu/solvers/test_solver_backends.py::TestColoredPGS`（3 项）：
  `test_100_steps_no_nan`、`test_max_qdot_bounded`、`test_coloring_no_body_conflict`
  — 验证 fused kernel 下稳定性和着色约束不变

- `tests/gpu/solvers/test_solver_backends.py::TestCrossSolverConsistency`（2 项）：
  50 步轨迹 body height 偏差 < 0.05m，max|qdot| < 50 rad/s
  — 验证 fused kernel 与 jacobi_pgs_ms / admm 物理一致

- 全 fast 套件（`not slow`）：1057 passed, 1 skipped

**未覆盖的验证维度**（proposal v2 已记录）：
- wall-clock 加速比（需独立 `benchmarks/bench_colored_pgs.py`，待建）
- 大规模 RL 场景（num_envs=1000+，10⁶ steps）

---

## Known Limitations

1. **加速比未实测**：launch overhead 消除的理论效益（~96ms → ~0.2ms）在本机未 benchmark。
   总 step 时间取决于 fused kernel 运行时（nc、max_iter、n_col 乘积），需 profiling。

2. **方案 D 延后**：warm starting 未实现。key 设计（body pair + 局部坐标最近邻点）
   见 proposal v2；实装时需新增 `contact_pos_prev` 缓冲（`wp.array(ndim=3)`）到
   `physics/backends/warp/solver_scratch.py`。

3. **方案 C 未实装**：CUDA graph capture 可在方案 B 基础上叠加，
   进一步消除剩余的 2 次 Python dispatch 开销，留作后续优化。

---

## 关键思考

**技术决策：为什么选"fused kernel"而非"CUDA graph capture"作为主线**

两者都能消除 launch overhead，但 fused kernel 更简单：
- CUDA graph 需要处理 `n_colors` 可能跨步变化的场景（即使固定 16 色也需要验证
  Warp capture API 在当前版本的行为）
- fused kernel 只是把现有逻辑搬入线程内部，没有新的 API 依赖

代价是：fused kernel 单次运行时间更长，GPU occupancy 可能略低于多次短 kernel。
这个 tradeoff 需要 profiling 才能量化，目前 launch overhead 主导时 fused 肯定更快。

**调试过程中无需特殊处理**

`batched_colored_pgs_all_iters` 的 `for color in range(n_colors[env_id])` 使用了
per-env 动态上界。Warp 支持这种写法（运行时整数上界），不需要 `wp.static()`。
proposal v2 已修正了 v1 中关于"静态展开"的错误描述。

**Codex challenge 对设计质量的影响**

v1 proposal 的三个 Medium 问题（Warp unrolling 误断、方案 C RL 排除理由、
方案 D key 设计缺陷）在 v2 都修正了，最终实装的设计比初稿更保守、更准确。
Codex 的"方案 C 在 RL 下也可用"观察与我们在对话中的独立推导一致，
增强了对这个结论的信心。
