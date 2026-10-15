Initiative: admm-regularization
Stage: implementation-note
Author: codex
Version: v3
Date: 2026-10-15
Status: in_review
Related Files: collab/admm-regularization__code-review__codex__v2.md; docs/admm_regularization.md
Owner Summary: 按第二轮独立审阅补齐下降阻抗精度、R 原生范围和宽度阈值转换问题。216 项真实环境回归通过；最终候选待独立复审。正则化接口、普通法向验收及摩擦排除范围均保持不变。

## Provenance and scope

See implementation notes v1/v2 for architecture, ordinary numerical measurements
and exact first-round F1/F2/F3 repair provenance. Independent code review v2 of
5f51ffc8f9dac32105b41bc88874832b51a89348 closed F1/F2, confirmed the exact original
F3 repro rejected, but found further decreasing-impedance and width-switch failures.
Its complete report is archived verbatim as code-review v2, including paths and
hashes of all original outside-checkout repros. It remains Request changes for
that historical candidate; new tests do not retrospectively alter that verdict.

Production/test code for this round is 91ddcba4 (full identity in Git), built on
483a6bf7; subsequent candidate note/log/link changes are documentation only. Final
frozen review HEAD is supplied in the code-review v3 packet/report. Same independent
physics branch and local/remote checkouts; no main merge, push or rendering work.

## F3 completion: decreasing impedance and native R bounds

Previous interpolation d0+y*(dw-d0) lost significant digits near a small decreasing
endpoint. Forming y=1-tail also discarded the small complementary weight in the
upper branch. This caused ordinary 2 kg GPU R to miss the agreed tolerance and
allowed R beyond the endpoint-derived native bound, including Inf on CPU/GPU.

CPU and GPU now retain the piecewise complementary weight directly, combine
weighted endpoints, return exact endpoints for x<=0/x>=1, and clamp arithmetic
roundoff to the endpoint interval. The underlying impedance function is unchanged
mathematically. per_row retains its prior implementation. Native R bounds also
reserve 16 machine epsilons of relative headroom from overflow/subnormal limits;
the conservative restriction is documented and keeps rounding at the admitted
limit from escaping the native domain.

The exact former CPU mass8.556863841944613e-305 and GPU two-body
mass4.89730620518322e-35 edge cases now reject before frame/cache/state mutation.
Nearby CPU mass1e-304 and GPU mass5e-35 cases produce finite analytically correct R.
GPU checks cover endpoint/interior depths, ground/body-body sums and condim1/3.
Ordinary-mass decreasing curves compare directly to actual MuJoCo at actual GPU
depth with the original rtol3e-5+atol1e-6 budget; no tolerance increase.

## F4: preserve the model branch before float32 conversion

The just-above-threshold width nextafter(1e-15,+Inf) previously rounded onto the
GPU endpoint-average branch, giving R=.5 instead of the CPU/MuJoCo .125 for
solimp=(.2,.8,width,.5,2), mass2. Host normalization now canonicalizes only the
original width<=1e-15 branch to exact zero. The native kernel tests that canonical
zero, so a just-above-threshold width retains the nonzero-width curve. Below,
equal and immediately above cases are tested against actual MuJoCo on CPU and H200.
This preserves the discontinuous model decision rather than relying on how a
particular threshold rounds in float32.

## Executed evidence

At code HEAD 91ddcba4, the same nine-file command documented in note v1 completed
**216 passed, 1 existing pytest fixture-deprecation warning, 70.36 s**. Raw output:
admm-regularization__test-log__codex__v3.txt. Environment unchanged: real H200,
CUDA_VISIBLE_DEVICES=3, Python3.12.13, NumPy2.5.1, MuJoCo3.15.0, Warp1.12;
/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python with OPENBLAS_NUM_THREADS=1.
Local analytic/domain tests:17 passed. Changed Python files pass ruff and diff check.

The run contains 63 actual MuJoCo reference cases, four complete CPU trajectories,
31 GPU cases and 17 independent CPU analytic/domain cases, plus 101 relevant
existing regressions. Ordinary GPU sample maximum R relative error is now
6.6159592e-7 (well within the original combined budget); normal impulse maximum
absolute error remains 6.4867140e-8 N s. Matched trajectory errors remain at the
v1 roundoff scale, and both engines retain the documented ~12 mm soft-model sphere
peak penetration. This does not become a material calibration result.

An intermediate test initially expected admission of the exact old GPU upper edge;
the newly documented headroom correctly rejected it. The final test explicitly
checks atomic rejection there and adds the adjacent supported case with an
analytic R comparison. The failed intermediate run is not counted as a pass.

All F1/F2 admission/coefficient regressions and previous F3 boundary tests remain
in the final run. The next independent review must close F3/F4 and verify those
earlier repairs remain intact. No condim3 force accuracy, friction projection,
default-30-iteration convergence, arbitrary collision/integration correctness,
extreme-scale force convergence, private-buffer mutation, material fidelity,
performance or full MuJoCo compatibility is approved by these results.
