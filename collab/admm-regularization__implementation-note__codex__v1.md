Initiative: admm-regularization
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-15
Status: in_review
Related Files: collab/admm-regularization__proposal__codex__v2.md; docs/admm_regularization.md
Owner Summary: 实现两种可选 R，先以解析及 MuJoCo 无摩擦交叉验证解耦正则化问题；保留默认 per_row 和现有摩擦行为。最终代码候选待独立审阅，不能用设计批准替代。

## Scope and provenance

Approved predecessor: proposal v2 at 4af0ca35e28c9b978b8c96f79b4fe19b2f4c2e3b,
with verbatim independent approval archived in design-review v2 at 6c66f4b.
Code diff starts at 6c66f4b. Executed production/test code HEAD:
1bbe9e99 (full identity available in Git); candidate also includes this note,
raw log, API documentation and progress update. Frozen review HEAD is provided
in the review packet and subsequent report. Branch physics/correctness-audit-20261012.

Local checkout C:/Users/AG/新建文件夹/robot_physics_correctness.
Primary server tileops-dev, physics checkout
/data2/shared/ang.gao/robot_sim_physics_correctness. Original server main remained
at f771e612beaeea126e26926738097512cd2d809c; no main checkout changes or CUDA
rendering work. Candidate commits transferred via bundles and fast-forwarded in
the independent physics checkout only; no GitHub push/merge.

## Changes and bounded gap scan

- ADMMQPSolver and GpuEngine expose immutable regularization/impratio selections;
  default per_row, opt-in mujoco. CPU/GPU contact_condim=1 enables normal-only
  engine validation. No new material/interface layer or solver backend.
- A focused shared host helper computes q0 COM mobility with existing articulated
  dynamics, implements the simple-slider exception, validates/normalizes model
  parameters and checks native coefficient range. Production has no MuJoCo import.
- CPU recomputes body weights and restores FK; GPU snapshots weights once with its
  static model. R is independent of rho. CPU last_R_diag is a defensive copy.
- GPU per_row now reads raw W instead of the PGS-modified W_diag. PGS is unchanged.
  Normal-only rows/columns, rhs, warm cache, projection/output and inactive slots
  are all masked together. This is necessary model selection plumbing.
- GPU positive standard-solref coefficients use the unfloored expression for
  mujoco mode; native overflow/underflow raises ValueError. Extreme impratio has
  R assembly tests, not force-convergence approval. High-power impedance uses
  bounded ratios to avoid overflowing intermediate power denominators.
- Existing condim3 projection and iteration/rho policy are unchanged. Default
  30 iterations, contact cache identity/rho scaling, material mapping, broader
  narrowphase/manifold correctness and physical calibration remain gaps.
- CPU's adapter skips empty contact frames, so solver diagnostics describe the
  last solve. Measurement code must check frame contacts; no new diagnostic
  lifecycle claim is made for the engine adapter.

## Execution and measurements

Environment: Python3.12.13, NumPy2.5.1, SciPy1.18.0, MuJoCo3.15.0, Warp1.12,
real H200 through CUDA_VISIBLE_DEVICES=3. Audit interpreter:
/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python.
Raw output: admm-regularization__test-log__codex__v1.txt.

Final combined run: **161 passed**, one existing pytest class-fixture deprecation
warning, 71.64 s. Includes 10 independent CPU analytic/domain cases, 38 actual
MuJoCo assembly/force cases, four complete CPU trajectories, eight real GPU cases,
and the relevant existing CPU/GPU ADMM, rigid-body metrics and articulated PGS
tests. Local analytic tests also passed (10); changed Python files pass ruff.

2 kg simple z-slider: Rn=.0555555555556 and f=67.658 N, matching the scalar oracle
and actual MuJoCo, closing the design-review trace/3 counterexample. Neighbour
classification cases include fixed parent, nested fixed parent, fixed child,
COM offsets and axis cutoffs. Non-diagonal ambiguous slider frames are rejected.
Free bodies, pendulum and quadruped weights match actual body_invweight0.

Frozen articulated four-contact tests compare actual J, A, R, smooth acceleration,
force and total acceleration at straight and bent poses. Off-diagonal mobility
is required to be nonzero. Positive unconstrained optima independently certify
the nonnegative optimum. Separate analytic two-contact cases have an active zero
force and satisfy KKT, so all-positive cases are not the sole optimization oracle.

Real GPU tests use unequal 2/5 kg independent bodies with both ground and
body-body contacts. Across both policies, dimensions, zero-width/clipped/high-power
solimp and impratio=1e-16/1/10: maximum R relative error **3.3250521e-7**;
maximum normal impulse absolute error **6.4867140e-8 N s** against an independent
active-set solve. Condim1 force checks use 1500 iterations with warmstart disabled.
Unused rows/columns and cached tangents are asserted exactly zero, followed by
an empty frame to test cache clearing. These are float32 numerical budgets,
not evidence that default iterations suffice or that condim3 forces are correct.

CPU real-engine results (max across the complete trajectory):

| Fixture / dt | q error | velocity error | per-step impulse error (N s) | penetration error (m) |
| --- | ---: | ---: | ---: | ---: |
| sphere / .001 | 1.39e-17 | 9.99e-16 | 5.00e-16 | 1.39e-17 |
| sphere / .0005 | 1.39e-17 | 1.33e-15 | 2.50e-16 | 1.39e-17 |
| quadruped / .001 | 4.44e-16 | 3.11e-14 | 9.75e-16 | 1.15e-16 |
| quadruped / .0005 | 6.66e-16 | 3.29e-14 | 7.82e-16 | 1.11e-16 |

All frame contact counts match; cumulative impulse differences <=2.02e-15 N s.
Sphere run=.5 s, bent quadruped=.12 s. Gates (q/depth1e-10, velocity1e-9,
step impulse1e-10, cumulative1e-9) allow accumulated roundoff/1e-12 solve error
for <=1000 matched-discretization steps. They express numerical equivalence in
these fixtures, not a universal mm bound or correctness of arbitrary trajectories.
Both engines' sphere peak penetration is .01236544 m at dt=.001 and .01262978 m
at half dt. This substantial soft-model penetration is not hidden by the matching
result; actual material fidelity and temporal convergence are separate gates.

Early test-harness failures were corrected before the full rerun: missing contact
tangent constructor arguments; stale last-solve forces incorrectly counted on
empty trajectory frames; an existing direct Warp helper call missing the newly
explicit dimension argument. None were reclassified as passes. No physical
parameter was tuned to mask a discrepancy. GPU R correction is an actual change
in the existing GPU model, documented explicitly.

## Reproduction

From the remote physics checkout:

```sh
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 \
/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python -m pytest \
  tests/reference/mujoco/test_admm_regularization.py \
  tests/reference/mujoco/test_admm_regularization_trajectories.py \
  tests/gpu/solvers/test_gpu_admm_regularization.py \
  tests/unit/solvers/test_contact_regularization.py \
  tests/unit/solvers/test_mujoco_qp.py \
  tests/gpu/solvers/test_gpu_admm.py \
  tests/gpu/solvers/test_gpu_admm_validation.py \
  tests/validation/rigid_body/test_physics_metrics.py \
  tests/integration/contact_pipeline/test_articulated_pgs_inertia.py -q -s
```

## Gates

Author evidence supports optional R assembly and the bounded normal-only
snapshots/trajectories. Independent code approval is still required at this note's
revision. The known .9 N versus 1.8 N condim3 counterexample remains in
admm-bounded-verification analysis v1; this work neither modifies nor closes it.
R compatibility does not approve friction, material calibration, full MuJoCo
compatibility, general high-speed impact, GPU performance or overall rigid-body
correctness. The next design stage remains friction constraint/projection analysis.
