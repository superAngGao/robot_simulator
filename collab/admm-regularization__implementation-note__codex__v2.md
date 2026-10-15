Initiative: admm-regularization
Stage: implementation-note
Author: codex
Version: v2
Date: 2026-10-15
Status: in_review
Related Files: collab/admm-regularization__code-review__codex__v1.md; docs/admm_regularization.md
Owner Summary: 修复首轮代码审阅 F1/F2/F3，保留 v1 结果及完整审阅。普通方程、默认策略与摩擦范围未改变；新增数值边界与拒绝前不修改状态的验证。最终修订仍需新会话独立审阅。

## Provenance and retained scope

Prior frozen candidate 554133373595206ea2153a7f3cadd8f365451eec received Request
changes. Complete report archived verbatim as code-review v1. Original independent
repro remains outside the candidate at
C:/Users/AG/新建文件夹/admm_regularization_code_review_v1_probe.py
(SHA256 8DC93C827A99F5742E65E5C9E3716BB9B02BBA374608C7C46FF5BAC44A9CE3A1).
v1 implementation note and logs remain historical evidence, not final approval.

Production/test revision ee1f2d49 (full identity in Git), same local/remote branch
and audit interpreter as v1. Code-review v2 packet provides final frozen HEAD
including this note/logs. No default-policy, material, friction-projector,
iteration-count, main-branch or rendering changes.

## Findings addressed

**F1:** GPU mujoco host admission now validates finite float32-converted q/qdot/tau
before public step uploads. step_n validates before substeps; reset and reset_envs
validate initial states before reset writes. Invalid parameters are checked before
an otherwise valid external-state upload. Explicit zero, negative, nonfinite or
unrepresentable dt is rejected; None alone selects the default in this mode.
per_row retains its legacy default-selection behavior. Engine-owned device state
has no new host readback; public input admission is the boundary, not arbitrary
mutation of private device buffers. All invalid-input tests snapshot state, tau,
contacts, ADMM cached variables/R, default q, run identity and publication counters,
then verify equality and subsequent successful valid execution. Tests use two
environments, full/partial reset, step and batched step_n.

**F2:** CPU uses normalized standard-solref values and the exact validated b/k
coefficients in assembly. Constructor validation no longer claims dt-dependent
coefficient representability using an invented dt=1. Actual solve-time validation
occurs before frames/cache. Both Python-float and NumPy-array solref=(.02,1e155)
produce finite native coefficients/solve output; 1e160 is rejected before contact,
cache or diagnostic mutation. Native coefficients must be finite and normal;
extreme-scale force convergence is still excluded.

**F3:** CPU and GPU now validate conservative native R bounds before assembly.
For normalized impedance over all depths, possible pair mobility is bounded by
min(body weights) and 2*max(body weights). Bounds include Rn's floor and tangent
division. R and nonzero converted weights must be finite normal native values,
avoiding CUDA flush-to-zero dependence. GPU retains the host weight snapshot to
check mutable parameters each step without downloading device arrays. This
deliberately conservative range can reject an extreme model even if only its
particular current contacts fit; docs state the restriction explicitly. It is
a numerical boundary, not a physical/material admissibility assertion.

Exact F3 mass1e-35, d=1e-4 real-CUDA repro now rejects before cache/state mutation.
Mass1e-34 near-boundary cases retain finite analytically correct R, including
body-world and body-body sums and condim1/3. Separate native tangent overflow and
underflow repros reject at construction; nearby supported ratios produce expected
normal/tangent R on H200. CPU float64 and float32 host bounds are tested too.

## Verification

Same real H200, MuJoCo3.15.0, Warp1.12 and audit environment as v1. Commands are
the v1 nine-file suite with the expanded tests (no substituted mocks).

- At d4dbb9e3: **174 passed**, one existing fixture-deprecation warning, 69.30 s.
  Raw output: admm-regularization__test-log__codex__v2.txt.
- Final small delta ee1f2d49 consumes the normalized CPU reference tuple and
  rejects non-integer dimension types. Re-executed all four new verification files
  at this revision: **73 passed**. Raw output:
  admm-regularization__targeted-log__codex__v2.txt. The independent re-review must
  check final candidate behavior; the earlier full run is not mislabeled as a run
  of the later delta.
- Changed Python files pass ruff; local pure analytic/domain tests pass (14).

The four trajectory and ordinary GPU R/impulse measurements reproduce v1. Added
tests exercise native-domain/rejection behavior; they make no claim that extreme
mass/ratio forces converge. All prior model/trajectory/friction/material exclusions
remain as documented. A fresh independent code review is the remaining gate.
