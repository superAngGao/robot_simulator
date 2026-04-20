Initiative: q48-cpu-collision-completeness
Stage: review-followup
Author: claude
Version: v1
Date: 2026-04-20
Status: implemented
Related Files: physics/gjk_epa.py, tests/integration/test_cpu_engine_shapes.py
Owner Summary: Three issues from Codex review of v1 implementation-note are fixed. Codex should verify the depth-semantics fix is correct and that the new body-body discriminators are strong enough.

## Issues Addressed

### High — depth semantics in polyhedral ground_contact_query

**Root cause**: `max_depth` was initialized to `0.0` before iterating over
penetrating vertices.  When all vertices are in the margin zone (depth < 0,
i.e. near-contact but not yet penetrating), the loop never updated `max_depth`,
so `manifold.depth` was returned as `0.0` instead of the correct negative gap.

**Fix** (`physics/gjk_epa.py`): removed the `max_depth = 0.0` initializer and
the in-loop `if d > max_depth` branch; replaced with `max_depth = max(point_depths)`
after the loop.  This is equivalent for the penetrating case and correct for the
near-contact case.

**Invariant now holds**: `manifold.depth == max(point_depths)` for all vertices,
matching the sphere fallback path and the docstring contract ("depth can be
negative when within margin but not yet penetrating").

### Medium — ConvexHull skip guard checked wrong dependency

**Root cause**: `HAS_CONVEXHULL` only tried `from physics.geometry import
ConvexHullShape`.  `ConvexHullShape.__init__` calls `scipy.spatial.ConvexHull`
at construction time, so in a scipy-free environment the tests would not skip —
they would fail at runtime.  Skip reason also said "trimesh required" (wrong).

**Fix** (`tests/integration/test_cpu_engine_shapes.py`): guard now also imports
`scipy.spatial.ConvexHull` in the try block; skip reason updated to "scipy required".

### Medium — TestBodyBodyContact assertions were too weak

**Root cause**: All three body-body tests only checked NaN and `z > 0`.  Bodies
could fall independently to the ground without ever interacting and still pass.

**Fix (v1 attempt — rejected by Codex)**: horizontal separation `|x_b - x_a| >= 2r * 0.9`
for sphere-sphere.  Flaw: initial separation was already exactly `2r`, so the
threshold `0.9 * 2r` was trivially satisfied without any contact.

**Fix (v2 — rejected by Codex round 2)**: 500 steps was not enough — contact
first occurs at step ~714 (0.143s at DT=2e-4), so assertions ran before any
body-body interaction.  Diagnosed by running a Python script that tracked
`query_contacts()` per step: `z_lower≈0.050, z_upper≈0.201, sep_z≈0.151` at
step 500, with only ground contact (body_j=-1) present.

**Fix (v3 — current)**: Increased to 1200 steps.  Contact occurs at step 714,
leaving ~486 steps for settling.  After 1200 steps: `sep_z ≈ 0.10 ≈ 2r`.
A no-contact trajectory gives `sep_z → 0`, which fails `sep_z >= 2r * 0.9`.

Discriminators (all three tests):
- `test_sphere_sphere_collision`: `z_upper > z_lower` AND `sep_z >= 2r * 0.9`.
- `test_box_sphere_collision`: sphere rests above box top face `z_sphere > 2*half * 0.9`.
- `test_box_box_stacking`: upper box rests above lower box top face `z_upper > 2*half * 0.9`.

## Verification

`pytest tests/integration/test_cpu_engine_shapes.py` — 12 passed in 21s.

## Remaining Known Limitations (unchanged from v1)

1. gjk_distance box-cyl/box-hull early exit — P4, deferred.
2. Body-body contact count not validated (Class 3 still does not assert contact
   point counts for body-body pairs, only geometry).
3. FlatTerrain only; HalfSpaceTerrain integration tests not added.
4. N_SETTLE = 1500 not marked slow.

## 关键思考

### depth 语义 bug 的诊断过程

`max_depth = 0.0` 的初始化问题在代码里不显眼——穿透场景下循环一定会更新它，
所以所有正常仿真测试都通过了。Codex 是通过构造边界场景（box 在 margin 内但未穿透）
才暴露出来的。这类"只在边界条件下错"的 bug 很难被集成测试捕获，因为集成测试
通常让物体充分落地穿透，不会停在 margin 区。

修复方案选择 `max(point_depths)` 而非恢复 `max_depth = float('-inf')` + 循环，
是因为 `point_depths` 列表在此时已经填好，`max()` 更简洁且语义更清晰。

### sphere-sphere 测试场景的三次失败

**第一次（v1，side-by-side）**：断言 `sep_x >= 2r * 0.9`，但初始 `sep_x` 就是 `2r`，
阈值 `0.9 * 2r < 2r`，无接触时也满足。错误根因：阈值方向写反，应断言分离量*增大*。

**第二次（v2，drop-on-top，500 步）**：场景设计正确，但步数不够——接触发生在
step 714（0.143s），500 步时上方球还在空中（`z_upper ≈ 0.20`），断言在无接触
状态下通过。诊断方法：逐步调用 `query_contacts()`，发现 500 步内只有地面接触
（`body_j=-1`），无 body-body 接触。

**第三次（v3，1200 步）**：接触发生在 step 714，1200 步后 `sep_z ≈ 2r`，
无接触终态 `sep_z → 0`，断言有区分力。

**核心教训**：body-body 测试必须先用 `query_contacts()` 实测确认接触确实发生，
再定步数。"步数够用"不能靠估算，要靠实测。
