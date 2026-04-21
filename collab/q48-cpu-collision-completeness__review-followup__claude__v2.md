---
Initiative: q48-cpu-collision-completeness
Stage: review-followup
Author: claude
Version: v2
Date: 2026-04-21
Status: implemented
Related Files:
- collab/q48-cpu-collision-completeness__implementation-note__claude__v1.md
- collab/q48-cpu-collision-completeness__review-followup__claude__v1.md
- physics/gjk_epa.py
- tests/integration/test_cpu_engine_shapes.py
Owner Summary: 补充 v2，供 Codex 写 review__codex__v1 时参考。v1 review-followup 记录了三轮 Codex 口头反馈的修复，但 Codex 从未写过正式 review 文件。本文档将当前代码状态整理为 Codex 可直接 review 的形式。
---

## 当前状态摘要

q48 的三个子问题：

| 子问题 | 状态 | Commit |
|--------|------|--------|
| Q48.1 gjk_distance box-cyl/box-hull 早退出 | 🔄 P4 deferred | — |
| Q48.2 ground_contact_query 多点 | ✅ 关闭 | 35ca490 |
| Q48.3 CpuEngine 集成测试 | ✅ 关闭 | 35ca490 |

## v1 review-followup 记录的三轮修复

### Round 1 — depth 语义 bug（High）

**问题**：`max_depth = 0.0` 初始化在所有顶点均为 margin-zone（depth < 0）时返回 0.0 而非真实负深度。

**修复**（`physics/gjk_epa.py`）：删除 `max_depth = 0.0` 初始化和 in-loop `if d > max_depth` 分支，改为循环后 `max_depth = max(point_depths)`。

**验证**：box 在 margin 内未穿透时，`manifold.depth` 现在返回正确的负值（gap 大小）。

### Round 2 — ConvexHull skip guard 检查了错误依赖（Medium）

**问题**：`HAS_CONVEXHULL` 只 import `ConvexHullShape`，未 import `scipy.spatial.ConvexHull`。scipy-free 环境会在运行时 fail 而非 skip。

**修复**（`tests/integration/test_cpu_engine_shapes.py`）：guard 加入 `scipy.spatial.ConvexHull` import；skip reason 改为 "scipy required"。

### Round 3 — body-body 断言弱（Medium，三轮迭代）

**v1**（rejected）：`sep_x >= 2r * 0.9`，初始分离就已满足阈值，无区分力。

**v2**（rejected）：drop-on-top 场景，500 步。接触发生在 step 714（0.143s at DT=2e-4），500 步时上球仍在空中，断言在无接触状态通过。

**v3**（current）：1200 步。接触后约 486 步用于落定。三个测试的判别量：
- `test_sphere_sphere_collision`：`z_upper > z_lower` AND `sep_z >= 2r * 0.9`
- `test_box_sphere_collision`：`z_sphere > 2*half * 0.9`（sphere 在 box 上方）
- `test_box_box_stacking`：`z_upper > 2*half * 0.9`（上 box 在下 box 上方）

**当前结果**：`pytest tests/integration/test_cpu_engine_shapes.py` — 12 passed in 22s。

## 已知残余限制（不变）

1. **Q48.1** gjk_distance box-cyl/box-hull 早退出：P4 deferred，`test_convex_margin.py` workaround（pen = 3×margin）仍在。
2. **body-body 接触点数量**：Class 3 不断言 body-body 接触点数，只检查几何判别量。
3. **N_SETTLE = 1500 steps**：单测约 1.5s，未标 `@pytest.mark.slow`。
4. **FlatTerrain only**：HalfSpaceTerrain 集成测试未加。

## 供 Codex review 的重点问题

1. v3 sphere-sphere 判别量（`sep_z >= 2r * 0.9`）在 1200 步后是否足够稳定？是否存在能量耗散不足、两球在 step 1200 时仍未落定的情况？
2. depth 语义修复（`max(point_depths)` vs 旧 `max_depth = 0.0`）：near-contact case 的负深度语义与 CPU engine 的 contact depth 过滤阈值（`1e-10`）是否完全对齐？
3. N_SETTLE = 1500 在 CI 下是否值得标 slow？
