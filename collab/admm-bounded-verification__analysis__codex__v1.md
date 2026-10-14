Initiative: admm-bounded-verification
Stage: analysis
Author: codex
Version: v1
Date: 2026-10-14
Status: in_review
Related Files: collab/admm-method-audit__analysis__codex__v2.md; collab/rigid-admm-projection__proposal__codex__v1.md; collab/admm-method-evidence/admm_bounded_verification.py; collab/admm-method-evidence/admm_bounded_verification_20261014.json
Owner Summary: 先诊断、再提方案。纯法向与摩擦接触呈现不同失败机制；真实 CPU/GPU 调用确认错误投影不能靠增加迭代修复。仅提交诊断和建议顺序，没有修改生产算法。

## Scope and provenance

This is a bounded diagnosis and repair recommendation, not an implementation
approval or a complete contact-model design. The owner's instruction to analyze
and propose before implementation remains in force. The earlier projection v1
proposal at 5d80e87 was paused, not independently approved; its existence does
not authorize implementation. This report supplies additional evidence before
the owner chooses the next change. Neither approval of this analysis nor an old
analysis approval selects a final physical friction/material model.

Local base: 5d80e87ee9514a72e7007c0f1552cc30d212333f, branch
physics/correctness-audit-20261012, repository robot_physics_correctness.
Verified server main remains clean f771e612beaeea126e26926738097512cd2d809c.
Executed production checkout: /data2/shared/ang.gao/robot_sim_physics_correctness,
HEAD 171a6fc2e71910f22bcf079681700dc3742bc968. The difference to local base is
only the old proposal. At execution the sole untracked file in that checkout was
the uploaded diagnostic. Production physics and tests are unchanged by this work.

Executed diagnostic SHA256:
0869838e90f0960e52c91905875abfaa7ac94cdb6901df3aff857f23bb9362e9.
Raw JSON records this hash and the executed checkout revision, which intentionally
does not claim to contain the then-untracked diagnostic. Raw stdout is the sibling
admm_bounded_verification_20261014.log. Python 3.12.13, NumPy 2.5.1,
SciPy 1.18.0, Warp 1.12.0, real NVIDIA H200; no GPU mock.

Reproduce from the isolated server checkout with this diagnostic present:

```sh
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 \
/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python \
collab/admm-method-evidence/admm_bounded_verification.py --gpu \
--output /tmp/admm_bounded_verification_reproduction.json
```

The command completed with exit 0: 24 frozen CPU cases, 18 CPU trajectories,
8 GPU first-contact cases. Exit 0 certifies execution and the diagnostic's
reference checks, NOT successful production accuracy. This is an evidence script,
not a green regression suite; after a repair, promote the relevant counterexamples
to asserting tests and retain this script as historical reproduction evidence.

## Reference problem and independence

Use the currently declared convex problem, with force f on CPU:

    minimize 0.5 f^T Q f - c^T f,  f in K
    Q = A + R, A = J H^-1 J^T, R_ii = (1-d)/d A_ii
    K_i = {n >= 0, ||t|| <= mu*n}; scalar nonnegative ray for condim1

The GPU uses impulse p=dt*f and rhs=dt*c. Its actual assembled Q is measured
separately from whether its R agrees with the intended formula. Relative GPU
error below means ||p-p_reference||/||p_reference|| for that actual Q/rhs;
it is not a measurement of real-world collision error.

All runs disable warmstart; CPU uses fixed rho, tolerance 1e-11 and constant
solimp d=.9, solref=(.02,1). This isolates the frozen solve from known cache and
adaptive-rho issues, and does not claim to validate the default warmstart path.
GPU also disables warmstart and uses constant d=.9, existing solref/rho defaults.

For frozen CPU fixtures: independently form J from [direction, arm cross direction]
and H from a 1 kg, radius .1 m sphere inertia. Check assembled J,Q,c against that
construction. The two-contact fixtures use synthetic points at (+/-.1,0,-.1):
they test coupled algebra, not realizable sphere-ground contact geometry.
An independent SLSQP solution uses two starting points, with positive-definite Q
checked and cone feasibility/normal-cone stationarity checked independently of
the production projector. Both optima must agree; the single-contact optima also
have analytic checks. No optimizer success flag alone certifies the answer.

Reference self-check budgets are feasibility <1e-8 and stationarity
<1e-7*(1+||c||), with two starts agreeing within 2e-6 N. These are O(1) fixture
numerical certification budgets, not physical product tolerances. Strong convexity
gives a force-error bound from a feasible point's stationarity defect divided by
lambda_min(Q); no statement extends these absolute budgets to arbitrary scales.
The classifier's 1e-8 apex/boundary cutoff is used only on these O(1) fixtures.

For single-contact trajectory/GPU snapshots, use the analytic diagonal-Q solution,
after checking zero off-diagonals and equal tangential diagonals. If the free
solution is outside K and c_n+mu*||c_t||>0, the boundary optimum is
n=(c_n+mu*||c_t||)/(Q_nn+mu^2 Q_tt), t=mu*n*c_t/||c_t||; the other regions
are the feasible free solution or zero. None of these reference forces is fed
back into a trajectory. Thus this report does not measure a repaired trajectory.

## Measured failures and controls

CPU frozen results at rho=1, cap=1000 (solver may stop sooner):

| Case | Returned force | Reference force | Stationarity defect |
| --- | --- | --- | --- |
| Single normal | .8999999999917 N | .9 N | 9.26e-12 |
| Single sliding | (.9,.900000000009,0) N | (1.8,1.8,0) N | 3.182 |
| Coupled normals | .418604651162 N each | .418604651163 N each | 2.18e-12 |
| Coupled sliding | (.418604651163,.418604651169,~0) N each | (.895027624309,.895027624309,~0) N each | 3.388 |

For single sliding, primal residual is 8.96e-12 and production exits after 111
iterations, yet the force error is .9 N. Coupled sliding similarly stops at 206
iterations with primal residual 9.25e-12 and force error .476423 N. The rho sweep
(.1,1,10) preserves this large friction error at a high cap; some slow cases have
not met the configured stopping criterion even at 1000. Conversely single normal
at rho=10 has .0046384 N error after 50 iterations, falling to 8.45e-12 N with a
1000 cap (stops at 241). These are distinct wrong-operator and finite-budget issues.

Root cause in CPU _project_all and Warp _cone_project_all: clamp n to nonnegative,
then clip t to the disk of radius mu*n. This is not the Euclidean projection onto
K required by the implemented scalar-rho ADMM split. Small primal/dual residuals
cannot certify this incorrectly projected QP. Feasibility alone also cannot.

GPU actual first-contact solve, dt=.2 ms, radius .1 m, initial penetration 10 um,
vx=0 or 1 m/s, mu=.8:

| Mass | vx | Relative impulse error, cap 30 | Relative impulse error, cap 1000 |
| --- | --- | --- | --- |
| 1 kg | 0 | 2.32e-8 | 2.32e-8 |
| 50 kg | 0 | .517177 | 1.78e-6 |
| 1 kg | 1 | .676003 | .676003 |
| 50 kg | 1 | .843567 | .676005 |

Separately, measured GPU tangential R_ii/W_ii=.1169581 rather than 1/9=.1111111.
The kernel builds R from W_diag, which already includes the PGS diagonal
regularization, while its dense Q uses W. This is a model-assembly mismatch,
not an effect that rho should absorb. The oracle above deliberately uses the
actual Q so this issue cannot explain away the projection counterexample.

## Trajectories: what simple checks miss

Real CpuEngine sphere-ground runs: radius .1 m, mass 1 kg, initial center z=.13 m,
vz=-1 m/s, vx=0/1/5 m/s, mu=.8, duration .15 s, dt=1/.5/.25 ms,
iteration caps 50/1000, no warmstart. Every detected contact step is compared to
its own analytic frozen-step optimum. The actual ground detector/integrator run.

For vx=0 and 1, all sampled force errors stay below 9.1e-12 N: mild oblique
contact is not an adequate counterexample. For vx=5, the first step is sliding:

| dt | First normal force | Frozen-step reference | Maximum component error |
| --- | --- | --- | --- |
| 1 ms | 137.08645 N | 166.9308649 N | 29.8444149 N |
| .5 ms | 134.8493188 N | 165.0924845 N | 30.2431657 N |
| .25 ms | 135.8023406 N | 165.8497361 N | 30.0473954 N |

Both iteration budgets produce these discrepancies. This is error against the
declared instantaneous QP, not proof that its reference normal force is the
physically correct collision force. Correct cone projection couples tangent and
normal response; adopting this convex associated law is a physical modeling
choice distinct from correcting its implementation.

All 18 runs satisfy the recorded linear impulse balance to <6.9e-15 kg m/s
(mass=1 kg, gravity included). That only verifies that the computed force is
consistently applied. It cannot validate the force law or the solve. Maximum
penetration is approximately 9.076/8.957/9.394 mm for the three step sizes,
unchanged by raising the iteration budget. No geometrically penetrating step
lacks a detected contact in these runs. These three nonmonotonic samples do not
establish temporal convergence, nor establish that penetration is an ADMM defect.
No measured restitution/contact-time or real-material validation is claimed.

## Proposed correction sequence, pending owner decision

1. **Restore the explicitly declared convex solve first.** Recommend retaining
   the current QP for this bounded repair and replacing CPU/GPU normal-first disk
   clipping by its true Euclidean cone projection. For input (a,b), r=||b||:
   mu=0 gives (max(a,0),0); feasible input is unchanged; a+mu*r<=0 gives zero;
   otherwise n=(a+mu*r)/(1+mu^2), t=mu*n*b/r. Preserve condim1, support condim3;
   do not represent the zero angular Jacobian rows of condim4/6 as supported.
   The previous proposal describes local implementation/test boundaries, but
   remains an unapproved draft. Do not silently change to a nonassociated Coulomb
   NCP, add material APIs, or tune R/solref to preserve old sliding outputs.
2. **Correct GPU R assembly as a separate, attributable change.** Derive ADMM R
   from the raw dense W diagonal rather than the PGS-preconditioned W_diag.
   Require the same d and independent raw A to produce the declared R on CPU/GPU;
   compare impulses using p=dt*f. Keep PGS's own regularization path intact.
   R remains a model parameter dependent on impedance and inertia, not a material
   table value; material calibration cannot be used to hide an assembly defect.
3. **Then make finite-budget and cached solves accountable.** Specify scaled
   optimality/feasibility checks, termination reporting and feasible output policy;
   test mass/scaling dependence. Specify contact identity/frame matching and
   scaled-dual rescaling before validating warmstart/adaptive rho. These policies
   need a separate concrete design; merely raising a fixed cap is insufficient.
4. **Assess the physical law and discretization with the corrected solver.**
   Repeat trajectory sweeps and compare contact impulse, penetration, rebound and
   time-step convergence against the chosen model. Decide whether the associated
   convex model is acceptable or a nonassociated law is required before claiming
   realistic friction. Material/geometry calibration and held-out experiments are
   later evidence, not implied by QP agreement. Interface abstractions stay deferred;
   material inputs must not overwrite model mass or inertia.

For the first proposed repair, use the analytic single-contact failures above,
independent certified coupled optima, projection optimality (Moreau/normal-cone
conditions), rotated tangents and mu=0/polar/negative-normal/small-scale cases.
Require converged solutions to be invariant to rho within numerical error; test
actual CPU/GPU entry paths, not just a replacement projector in a script.
CPU force error 1e-8 N on these O(1) fixtures at stopping tolerance 1e-11 is a
proposed numerical regression budget, not a physical-accuracy requirement; check
KKT and conditioning as well. GPU float32 budgets in the old proposal similarly
require actual device evidence and cannot transfer to material accuracy. Preserve
the failing original cases so a successful exit can no longer mask these defects.

First-patch completion would close only the wrong-projection defect for supported
dimensions. It would not close GPU R, finite-budget behavior, cache identity,
angular friction, general contact detection, physical material fidelity or tensor
core/GPU performance work. This analysis itself closes none of those defects.
