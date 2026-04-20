Initiative: q46-solver-consistency
Stage: review
Author: codex
Version: v1
Date: 2026-04-20
Status: draft
Related Files:
- collab/q46-solver-consistency__implementation-note__claude__v1.md
- collab/q46-solver-consistency__challenge__codex__v1.md
Owner Summary: 这轮实现方向正确，按 challenge 收窄成了“派生物理量一致性 + 语义级 CPU/GPU 差异说明”。没有需要退回重做的阻塞问题，但有两处文案需要对齐：Q46 dim 3 在 OPEN_QUESTIONS 里看起来像已完成，而 Step 6 测试顶部 docstring 还残留旧的“contact count agreement”表述。

## Findings
1. [medium] [tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py](/home/ga/robot_simulator/tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py:8) — module docstring still says this test catches "CPU vs GPU contact count agreement", but the new documented behavior in the same file now explicitly says exact contact counts do **not** agree for box-ground manifold generation. The lower docstring/body is correct; the top summary is stale and now contradicts the intended semantics of the test.

2. [medium] [OPEN_QUESTIONS.md](/home/ga/robot_simulator/OPEN_QUESTIONS.md:1245) — Q46 dim 3 is now prefixed with `✅`, which reads like "resolved/completed", while the implementation note explicitly says the dimension is still open and only a short-horizon baseline was added. The surrounding text also says long-run and RL-facing validation remain pending. This should be phrased as partially addressed rather than visually closed.

## Coverage Gaps
- The new consistency baseline compares per-robot `link0 z` plus a `max|qdot|` bound. That is a reasonable first step, but it still leaves room for solver differences in lateral motion, orientation, and contact-force distribution to go unnoticed.
- `test_3solver_qdot_bounded` is better interpreted as a stability sentinel than a true cross-solver agreement test. It protects against hidden divergence, but it does not measure agreement between solvers.
- I could not execute the new GPU-targeted tests in this environment because they were skipped (`Warp or CUDA not available`). The review therefore verified code shape and collection logic, not runtime numeric behavior.

## Residual Risks
- The chosen tolerances (`atol=0.05`, `max|qdot| < 50`) are intentionally loose and may miss subtle solver drift. That is acceptable for a first baseline, but it should not be mistaken for a high-precision consistency guarantee.
- The CPU/GPU difference is now documented at the right semantic level, but until CPU multi-point box-ground is implemented, readers may still over-interpret "known difference, not a bug" as "fully settled design". It is better treated as an acknowledged temporary asymmetry.
