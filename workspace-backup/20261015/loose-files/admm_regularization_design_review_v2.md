Initiative: admm-regularization
Stage: design-review
Author: codex
Version: v2
Date: 2026-10-15
Status: accepted
Related Files: collab/admm-regularization__proposal__codex__v2.md; collab/admm-regularization__design-review__codex__v1.md; admm_regularization_design_review_v2_probe.py
Owner Summary: Approve with comments。独立重新执行上一轮反例，并检查 simple-slider 分类边界和实际四接触关节模型；F1/F2 在设计层面关闭。批准按 v2 实现可选 R 策略及其解析、无摩擦验证，不批准尚不存在的生产实现、摩擦投影正确性或完整轨迹兼容性。

## Verdict and provenance

**Approve with comments.** No unresolved blocking design findings. The two prior findings are closed against the exact v2 snapshot. Comments below are implementation and verification follow-ups, explicitly non-blocking for design approval; they do not waive any implementation gate.

Reviewer session: `/root/admm_regularization_design_v2`, fresh independent task with no implementation-conversation fork. This reviewer did not author the candidate. Read the owner-authorized protocol in full from `C:/Users/AG/新建文件夹/c4_worker_completion/AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md`.

Reviewed repository: `C:/Users/AG/新建文件夹/robot_physics_correctness`.

- Base: `119ec72e6a3e9b9ceea740e0c266bb9a1829c3e4`.
- HEAD: `4af0ca35e28c9b978b8c96f79b4fe19b2f4c2e3b`.
- Branch: `physics/correctness-audit-20261012`.
- Candidate: `collab/admm-regularization__proposal__codex__v2.md`.
- Candidate SHA256: `359DCBDAE5C5A1E796C41BAB472BBCDF2E565075C1CED34AAC45E0849E750A67`.
- Diff from base contains the v2 proposal and archived v1 report only, 254 added lines; no production implementation.
- Working tree clean before and after review; `git diff --check base..HEAD` passed. Final hash matches the frozen candidate.
- Remote execution: `tileops-dev:/data2/shared/ang.gao/robot_sim_physics_correctness`, clean at `fe7f3d6448f0d3c148d1c5a494129a8b3c5e4127`, same branch. Python: `/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python`; observed MuJoCo 3.15.0.

## Prior findings and independent closure

### F1 — Closed for design: simple translational bodies

The new construction distinguishes the general COM mobility from the simple-slider inverse-mass branch. Its root/fixed-root-parent, leaf, COM, axis, and canonical inertial-frame conditions match the relevant compiler rules for the existing one-joint-per-body representation. Canonical XML with `diaginertia`, identity inertial rotation, retained bodies and default simple optimization makes the comparison well-defined. Rejecting otherwise eligible sliders with non-diagonal inertia avoids inventing an inertial quaternion that the production model does not retain. This rejection remains to be demonstrated by implementation tests.

Independently re-ran the original v1 probe: the old trace-only counterexample still gives weight 1/6, R=.01851851851851851 and exact scalar force 72.49071428571432 N, versus actual MuJoCo weight .5, R=.05555555555555554 and force 67.65800000000007 N. A new independent probe applying the v2 classification matches MuJoCo:

| Fixture | v2 simple branch | Reference body_simple | v2/reference weight |
| --- | --- | --- | ---: |
| Independent 2 kg slider | yes | 2 | .5 |
| Slider under fixed root | yes | 2 | .5 |
| Slider under two nested fixed bodies | no | 0 | 1/6 |
| Slider with a 1 kg fixed child | no | 0 | 1/9 |
| Slider with COM offset .01 | no | 0 | 1/6 |
| COM offset .999e-6 | yes | 2 | .5 |
| COM offset 1e-6 | no | 0 | 1/6 |
| Tilted axis (.1,0,1) | no | 0 | 1/6 |
| Secondary axis component .999e-14 | yes | 2 | .5 |
| Secondary axis component 1.001e-14 | no | 0 | 1/6 |

Each fixture also compared actual `efc_R` with the derived v2 R. All assertions passed. The simple slider reproduces 67.658 N. Fixed bodies have zero weight, and the fixed child inherits nonzero articulated mobility. This covers the nearby cases that would defeat an unconditional 1/m replacement.

The supporting pinned rules were independently inspected in [MuJoCo's compiler](https://github.com/google-deepmind/mujoco/blob/3.15.0/src/user/user_model.cc#L2654-L2756) and [reference-weight construction](https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_setconst.c#L842-L869).

### F2 — Closed for design: impratio floor

The explicit `max(1e-15, impratio)` denominator and required boundary tests remove the mismatch without silently changing the accepted positive domain. The per_row option still requires impratio==1.

Independently compared actual condim3 elliptic `efc_R` for a 2 kg free body:

| impratio | Actual/v2 tangent R |
| ---: | ---: |
| 1e-16 | 55555555555555.53 |
| 1e-15 | 55555555555555.53 |
| 2e-15 | 27777777777777.766 |
| 1 | .05555555555555554 |
| 10 | .005555555555555554 |

Both tangents matched at every value. These were assembly checks, not claims of converged frictional forces at extreme scales. The rule matches [MuJoCo's contact regularization](https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_core_constraint.c#L2088-L2106).

## Whole-design feasibility and inspected high-risk areas

The existing production dynamics provide H0 and body velocity Jacobians with consistent body-frame COM shifts. General-branch comparison against actual MuJoCo body weights passed for the two-body double pendulum (maximum absolute error 4.996e-16) and 13-body quadruped (2.220e-15). The proposal uses existing model facts and does not require production MuJoCo or caller-supplied R.

Independently assembled a frozen, frictionless four-contact quadruped problem using production tree dynamics, current-state contact Jacobians, reference-state body weights, and separately calculated R/reference acceleration. The reference disables joint limits, damping and exact diagonals, uses Euler, and has no warmstart dependency. Assertions compared J, A, R, reference acceleration, complete rhs, force and generalized acceleration against actual MuJoCo. All passed. Maximum off-diagonal magnitude in A was .19741964730574726, so this exercises coupling rather than independent scalar contacts. R was approximately .141422931293942 per contact; all forces were 20.14460631294247 N, with maximum absolute force difference 3.553e-15. The analytic unconstrained solution was strictly positive, hence also the nonnegative QP optimum. This proves a feasible design target; it is not a test of the proposed unimplemented solver route.

Read the existing CPU ADMM assembly, CpuEngine contact creation, GpuEngine construction/dispatch, Warp ADMM kernel and W assembly. The public keyword-only options can preserve positional calls. Current CPU contacts are explicitly created with condim3, so the requested engine option has a concrete narrow insertion point. GPU contact identity is already available, and raw W exists separately from the PGS-adjusted W_diag; selecting raw diagonal data and uploading construction-time body weights are feasible. Keeping the existing three-slot allocation permits the specified zero-tangent condim1 subproblem without introducing a new solver resource lifecycle.

Reference-state weight lifetime is explicit: CPU recomputation accommodates mutable model inertia; GPU follows existing static-model ownership and requires rebuild after edits. New solver instances separate model-option selection from warmstart reuse. The design preserves the default per_row route, pins exclusions, keeps friction-projection work separate, and leaves trajectory thresholds gated on actual evidence rather than an invented universal tolerance.

## Non-blocking comments and required implementation follow-through

**C1 — Preserve the full condim1 invariant in the actual GPU entry point.** The design already requires no tangent equations, zero tangent R and output, and rho on unused zero rows. Code review should inspect matrix rows and columns, rhs, projection, warmstart loading/saving and slot reuse together. A test that only zeroes final impulses cannot establish this invariant. Retain the proposed inactive-slot and distinct-body tests with real CUDA execution. Disposition: implementation/code-review verification under the existing v2 contract, no design amendment required.

**C2 — Exercise native numerical boundaries without expanding compatibility claims.** Finite host values can overflow or underflow after float32 conversion, and high-power impedance expressions can create intermediate nonfinite values even when their mathematical result is bounded. Reference-acceleration coefficient floors also matter for extreme positive solref. During implementation, check normalization and converted/native assembly, keep CPU/GPU conformance evidence, and make any unsupported numerical-domain decision explicit before claiming broader rhs/force compatibility. Do not infer full positive-parameter force accuracy from the impratio R-only boundary checks. Disposition: retained numerical risk for implementation tests and code review; no claim of extreme-scale force convergence is approved here.

**C3 — Keep trajectory approval distinct.** The frozen quadruped success establishes a meaningful coupled verification fixture, not moving-contact, collision, integration or J-dot correctness. Execute the specified real-engine sphere and articulated comparisons at dt and dt/2, retain full state/impulse/penetration errors, and diagnose discrepancies. A blocked trajectory gate must remain visibly open even if assembly and snapshot gates pass. Disposition: required delivery evidence already present in v2.

## Executed verification and reproducibility

Read v2, archived v1 review and original repro; examined the local diff and the relevant production paths. Read the pinned MuJoCo sources through the web tool. Inspected the supplied `/tmp/admm_R_reference_probe.py` and `/tmp/admm_R_quad_probe.py`; their author assertions were not treated as approval evidence. Re-executed the latter: it reported four contacts and J error zero, with existing per_row force 25.30061228 N versus actual reference 20.14460631 N. The independent probe then checked the complete v2 frozen equations as described above.

New reviewer-owned repro, outside the candidate:

`C:/Users/AG/新建文件夹/admm_regularization_design_review_v2_probe.py`

SHA256: `4AFCA752BA712DB4A781740E0444E5733C0A1B56709290FF9507DC2EE550A334`.

Commands executed include:

```powershell
git status --porcelain
git rev-parse HEAD
git rev-parse --abbrev-ref HEAD
git diff --stat 119ec72e6a3e9b9ceea740e0c266bb9a1829c3e4..HEAD
git diff --check 119ec72e6a3e9b9ceea740e0c266bb9a1829c3e4..HEAD
Get-FileHash -Algorithm SHA256 collab/admm-regularization__proposal__codex__v2.md
Get-Content -LiteralPath 'C:/Users/AG/新建文件夹/admm_regularization_design_review_v1_probe.py' -Raw -Encoding UTF8 | ssh tileops-dev /data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python -
Get-Content -LiteralPath 'C:/Users/AG/新建文件夹/admm_regularization_design_review_v2_probe.py' -Raw -Encoding UTF8 | ssh tileops-dev /data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python -
ssh tileops-dev '/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python /tmp/admm_R_quad_probe.py'
```

The final independent probe exited 0 and printed PASS. An intermediate probe used a nonexistent Python disable-bit enum for diagexact and exited before the quadruped assertions; replacing that scratch-only setup with the documented XML flag fixed it. This was a reviewer test-harness issue, not a candidate defect. All final assertions ran anew after that correction. No candidate or remote tracked files were modified; Python import caches, if created, are ordinary untracked runtime side effects, not reused proof.

No production tests, full trajectory run or GPU kernel execution were performed: there is no candidate production implementation. CPU algebra and real MuJoCo were sufficient for this design re-review's decisive questions. Real CUDA evidence, invalid-input/pre-mutation behavior, non-diagonal slider rejection, empty solves, model-edit recomputation, default-route regressions and all promised implementation checks remain unexecuted gates for the next phase. No performance claim is made.

## Exact scope approved and gates remaining closed

Approval binds only proposal v2 at the HEAD/digest above. It permits implementing the optional per_row/mujoco R strategies, specified condim1 engine support, native CPU/GPU construction and stated verification plan. F1/F2 are closed for this design, not presumed closed in nonexistent code.

This does not approve production code, activation or default-model change, condim3 projection correctness, friction trajectories, extreme-scale force convergence, default iteration accuracy, warmstart/rho redesign, material calibration, full MuJoCo compatibility, GPU performance or overall simulator correctness. The implementation must receive a fresh independent code review with new execution evidence; a later friction design remains separate.
