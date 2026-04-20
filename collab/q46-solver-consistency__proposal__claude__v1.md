Initiative: q46-solver-consistency
Stage: proposal
Author: claude
Version: v1
Date: 2026-04-20
Status: draft
Related Files:
- OPEN_QUESTIONS.md (Q46 dim 3, dim 6)
- tests/gpu/solvers/test_solver_backends.py
- tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py
- physics/backends/warp/analytical_collision.py
- physics/backends/warp/collision_kernels.py
Owner Summary: 本轮聚焦 Q46 的两个可立即验证的维度：(A) 多 solver 在复杂场景下的物理一致性量化；(B) CPU vs GPU box-ground 接触点差异的明确文档化与测试覆盖。不涉及 RL 训练或性能优化。

---

## Problem

Q46 有 6 个待验证维度。其中维度 1/2 依赖 RL 训练循环（Phase 3），维度 4/5 是性能优化专项。
本轮只处理现在可以做的两个：

**维度 3 — Solver 间物理一致性**
现有 `test_single_sphere_all_solvers_agree` 只验证单球落地，场景过于简单。
在多点接触（box-ground manifold）+ body-body 接触的复杂场景下，4 个 solver 的轨迹偏差
从未被量化。Phase 3 RL 训练前必须知道：solver 选择对物理结果的影响有多大。

**维度 6 — CPU vs GPU 接触点差异**
GPU `box_ground_manifold` 返回最多 4 个接触点（顶点枚举），CPU 走 GJK/EPA 返回 1 个。
现有测试（`test_b5_d6d7_cpugpu_multienv.py:285`）用 `>=` 断言绕开了这个差异，
没有明确文档化这是设计决策还是待修复的 bug。

---

## Goal

1. 为 4 个 solver 在复杂场景（多点 box-ground + body-body）下建立定量一致性基线
2. 明确 CPU vs GPU box-ground 接触点差异的处置方式，并用测试固化

---

## Scope

**In scope:**
- 新增 cross-solver 一致性测试（复杂场景，量化轨迹偏差）
- 明确 CPU/GPU box-ground 差异：文档化为已知行为差异，更新相关测试断言
- 更新 Q46 状态

**Out of scope:**
- CPU box-ground 改为多点（工作量大，需独立 thread）
- Colored PGS 性能优化（维度 4）
- Mass splitting 收敛精度分析（维度 5）
- RL 训练验证（维度 1/2，Phase 3）

---

## Affected Files / Layers

- `tests/gpu/solvers/test_solver_backends.py` — 新增 cross-solver 复杂场景测试
- `tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py` — 更新 CPU/GPU 差异断言
- `OPEN_QUESTIONS.md` — 更新 Q46 dim 3/6 状态

---

## Proposed Design

### Part A: Cross-Solver Consistency Test

新增 `TestCrossSolverConsistency` 类，使用已有的 3-robot 9-body mixed-shape fixture
（与 `test_b5_d4d8_mixed_ground.py` 相同的场景，但通过 `jacobi_pgs_ms` 已验证稳定）。

**测试逻辑：**
1. 对 3 个稳定 solver（`jacobi_pgs_ms`、`colored_pgs`、`admm`）各跑 50 步
2. 记录每个 solver 的最终 q（位置）和 qdot（速度）
3. 断言：
   - 所有 solver 保持有限值（无 NaN/Inf）
   - solver 间位置偏差 `|q_a - q_b|` < 容差（建议 `atol=0.05`，50 步短时间内偏差应很小）
   - 所有 solver 的 max|qdot| 在合理范围（< 50 rad/s，防止隐性发散）

**为什么排除 `jacobi_pgs_si`：**
已知在多点接触场景下 step 1 即发散（Q46 benchmark 记录），加入会使测试无意义。

**容差选择依据：**
50 步 × 2e-4s = 0.01s 仿真时间，solver 间轨迹偏差主要来自收敛精度差异。
`atol=0.05` 对应 5cm 位置偏差，对 0.01s 仿真是保守上界。
如果偏差超过此值，说明某个 solver 在该场景下数值不稳定。

### Part B: CPU vs GPU Box-Ground Difference

**处置方式：明确文档化为已知行为差异（不修复）**

理由：
- GPU 多点 manifold 是正确的物理行为（更稳定的接触）
- CPU 单点是历史遗留，修复需要独立 thread（Q46 dim 6 后续）
- 现在的目标是固化这个差异，防止未来误判为 regression

**具体改动：**
1. `test_b5_d6d7_cpugpu_multienv.py` 中的 `>=` 断言改为精确断言：
   - CPU box-ground: 每个 box 形状产生 1 个接触点
   - GPU box-ground: 每个 box 形状产生最多 4 个接触点
   - 断言 `gpu_box_ground_count >= 4 * cpu_box_ground_count`（或类似精确表达）
2. 在测试文件顶部 docstring 中明确记录这个差异及其原因

---

## Test Plan

| 测试 | 类型 | 预期结果 |
|------|------|---------|
| `TestCrossSolverConsistency::test_3solver_50steps_agree` | gpu, slow | 3 solver 位置偏差 < 0.05，无 NaN |
| `TestCrossSolverConsistency::test_3solver_qdot_bounded` | gpu | max\|qdot\| < 50 for all solvers |
| `test_b5_d6d7_cpugpu_multienv.py` 更新断言 | gpu | CPU=1/box, GPU=4/box 精确断言 |

---

## Tradeoffs

| 选项 | 优点 | 缺点 |
|------|------|------|
| `atol=0.05` 容差 | 保守，不会因 GPU 调度抖动误报 | 可能漏掉细微的 solver 偏差 |
| 精确 CPU/GPU 断言 | 固化已知差异，防止 regression | 如果 CPU 端后续改为多点，需要更新测试 |
| 排除 jacobi_pgs_si | 测试聚焦稳定 solver | 不覆盖 si 的一致性（但 si 在此场景已知发散） |

---

## References

- Q46 benchmark: session 29, 3-robot 9-body fixture, 500 steps
- `test_b5_d4d8_mixed_ground.py`: 已验证 jacobi_pgs_ms 在 mixed-shape 场景稳定
- `analytical_collision.py:1534`: GPU `box_ground_manifold` 顶点枚举实现
- `test_b5_d6d7_cpugpu_multienv.py:285`: 现有 `>=` 断言（待精确化）
