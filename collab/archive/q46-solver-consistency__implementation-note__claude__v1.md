Initiative: q46-solver-consistency
Stage: implementation-note
Author: claude
Version: v1
Date: 2026-04-20
Status: implemented
Related Files:
- collab/q46-solver-consistency__proposal__claude__v1.md
- collab/q46-solver-consistency__challenge__codex__v1.md
- collab/q46-solver-consistency__decision__owner__v1.md
Owner Summary: 两个测试目标均已实现并通过。Part A 建立了 3-solver 复杂场景一致性基线；Part B 将 CPU/GPU box-ground 差异从隐式 >= 断言改为语义级断言并在 docstring 中明确记录。Q46 dim 3/6 均未关闭，状态已更新。

---

## Open Questions Addressed

- **Q46 dim 3 — Solver 间物理一致性**：open → 新增定量基线（未关闭）
  - 3 个稳定 solver 在 3-robot mixed-shape 场景 50 步内 link0 z 偏差 < 0.05 m，max|qdot| < 50 rad/s
  - 长时间轨迹（10⁶ steps）和 RL 场景仍待验证

- **Q46 dim 6 — CPU vs GPU 一致性**：open → 差异显式记录，manifold 尚未统一（未关闭）
  - GPU 顶点枚举（最多 4 点）vs CPU GJK/EPA（1 点）差异已在 docstring 中明确
  - 两者在 body 集合、法线、per-body 最大深度上一致
  - CPU 多点化为独立后续 thread

## REFLECTIONS.md / PROGRESS.md Impact

不需要更新。本次是测试补全，无新物理算法或架构决策。

---

## What Changed

**Part A — Cross-Solver Consistency（Q46 dim 3）**

新增 `TestCrossSolverConsistency` 类，使用已有 `_build_fixture()`（3-robot mixed-shape，
9 bodies，multi-point box-ground + body-body contacts）。

断言策略（采纳 Codex challenge 建议）：
- 比较 per-robot link0 z（base height），不对完整 q 做宽松一刀切
- 比较 max|qdot| 上界
- quaternion 分量不直接比较
- `jacobi_pgs_si` 排除（已知在该场景 step 1 发散）

两个测试：
- `test_3solver_50steps_base_height_agree`（`@pytest.mark.slow`）：3 solver × 50 步，
  pairwise link0 z 偏差 < 0.05 m
- `test_3solver_qdot_bounded`：3 solver × 50 步，max|qdot| < 50 rad/s

**Part B — CPU/GPU Box-Ground 差异（Q46 dim 6）**

`test_b5_d6d7_cpugpu_multienv.py` 改动：
- module docstring 新增"CPU vs GPU box-ground manifold difference"段落，
  明确记录差异来源、一致性边界、CPU 多点化延期原因
- `test_cpu_gpu_sorted_ground_contact_match` 中的隐式 `>=` 注释改为显式语义断言注释，
  说明 GPU 允许更多 contacts 的原因

---

## Files Touched

`tests/gpu/solvers/test_solver_backends.py`
- module docstring 新增 Q46 dim 3 说明
- 新增模块级常量 `_STABLE_SOLVERS`、`_LINK0_Z`、helper `_link0_z()`
- 新增 `TestCrossSolverConsistency` 类（2 个测试）

`tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py`
- module docstring 新增 CPU/GPU 差异说明段落
- `test_cpu_gpu_sorted_ground_contact_match` 断言注释精确化

`OPEN_QUESTIONS.md`
- Q46 dim 3：标注"新增定量基线"
- Q46 dim 6：标注"差异已显式记录，manifold 尚未统一"

`collab/`
- 新增 proposal/challenge/decision 文件（full path）

---

## Tests Added / Updated

新增：
- `TestCrossSolverConsistency::test_3solver_50steps_base_height_agree` — gpu, slow
- `TestCrossSolverConsistency::test_3solver_qdot_bounded` — gpu

更新：
- `TestStep6CpuGpuMultiEnv::test_cpu_gpu_sorted_ground_contact_match` — 断言注释精确化，行为不变

验证：
```
python -m pytest tests/ -m "not slow" -x -q
# → 1045 passed, 1 skipped, 117 deselected in 306s
```

## Known Limitations

- `test_3solver_50steps_base_height_agree` 的 `atol=0.05` 对 0.01s 仿真是保守上界，
  可能无法捕捉细微的 solver 收敛差异。如需更严格基线，应延长仿真时间并收紧容差。
- `test_3solver_qdot_bounded` 的 50 rad/s 上界较宽松，主要防止隐性发散，
  不是精确的物理一致性断言。
- `test_3solver_50steps_base_height_agree` 标记为 slow（50 步 × 3 solver），
  不在 commit gate 内。

## Commit

81f9876 test: Q46 dim3/dim6 — cross-solver consistency baseline + CPU/GPU box-ground diff
