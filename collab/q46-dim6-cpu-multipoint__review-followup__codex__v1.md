---
Initiative: q46-dim6-cpu-multipoint
Stage: review-followup
Author: codex
Version: v1
Date: 2026-04-21
Status: draft
Related Files:
- collab/q46-dim6-cpu-multipoint__review__codex__v1.md
- collab/q46-dim6-cpu-multipoint__implementation-note__claude__v1.md
- tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py
Owner Summary: 上一轮 review 的两个 medium finding 均已关闭。重复的 body 1 断言已删除，implementation-note 也已改回 repo 模板结构。我重新跑了 Step 6 GPU 测试，当前无需额外代码变更。
---

## Findings Closed

### Finding 1 — body 1 flat-box 4-point 断言重复

已关闭。当前 `test_cpu_gpu_sorted_ground_contact_match` 中只保留了一处 body 1
count parity 断言，位于 per-body max-depth 比较之后：

- [tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py](/home/ga/robot_simulator/tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py:344)

先前位于 body-set comparison 后的重复断言已删除，因此测试意图现在与
implementation-note 描述一致。

### Finding 2 — implementation-note 未按 repo 模板组织

已关闭。当前 implementation-note 已恢复到 repo 规定的结构，包含：

- `Open Questions Addressed`
- `REFLECTIONS.md / PROGRESS.md Impact`
- `What Changed`
- `Files Touched`
- `Tests Added / Updated`
- `Known Limitations`
- `Commit`

对应文件：

- [collab/q46-dim6-cpu-multipoint__implementation-note__claude__v1.md](/home/ga/robot_simulator/collab/q46-dim6-cpu-multipoint__implementation-note__claude__v1.md:15)

## Verification

我在 `env_tilelang_20260119` 中复跑了当前 Step 6 GPU 测试：

```bash
PYTHONPATH=/home/ga/robot_simulator \
  /home/ga/anaconda3/bin/conda run -n env_tilelang_20260119 \
  pytest -q tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py -q
```

结果：

- `5 passed in 15.07s`

另外，文件级检查也确认 body 1 断言当前只出现一次。

## Remaining Gaps

- `BoxShape` 平地场景的 count parity 现在有 fixture 级保护，但仍没有独立的
  box-ground CPU/GPU parity test。
- `ConvexHull > 4` 底面顶点时的 CPU all-vertices vs GPU cap-4 差异仍然只有
  文档化，没有专项回归测试。

## Conclusion

上一轮 review 的两个 medium finding 都已修复，我没有看到新的阻塞问题。
这轮可以视为 review 闭环完成；后续如果继续推进，应转向专项 parity coverage，
而不是再修改当前这份 Step 6 集成测试的语义边界。
