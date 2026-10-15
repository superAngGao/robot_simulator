Initiative: admm-regularization
Stage: review
Author: codex
Version: v1
Date: 2026-10-15
Status: in_review
Related Files: collab/admm-regularization__proposal__codex__v1.md; admm_regularization_design_review_v1_probe.py
Owner Summary: Request changes。总体先 R、后摩擦的边界可行，但通用 COM trace 公式漏掉 MuJoCo 的纯平移 simple-body 分支，已在真实 condim1 模型上复现 R 三倍差异和约 4.83 N 力误差。任意正 impratio 的契约也遗漏参考实现的分母下限。修订这两项后需新独立设计复审；本报告不批准生产实现或摩擦正确性。

## Verdict and provenance

**Request changes.** Two blocking design findings below. The first affects an ordinary supported joint model, independently of friction, iteration count, or precision.

Reviewer session: `/root/admm_regularization_design_v1`, fresh isolated reviewer task; candidate not authored by this reviewer. Applicable owner-authorized rules read in full from `C:/Users/AG/新建文件夹/c4_worker_completion/AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md`.

Exact reviewed repository: `C:/Users/AG/新建文件夹/robot_physics_correctness`.

- Base: `fe7f3d6448f0d3c148d1c5a494129a8b3c5e4127`.
- HEAD: `119ec72e6a3e9b9ceea740e0c266bb9a1829c3e4`.
- Sole candidate: `collab/admm-regularization__proposal__codex__v1.md`, version v1.
- SHA256: `3CA6B83BF031723B9115C27B26A36334E5B6856798FDDF5C8A95E8D27E5153BF`.
- Actual diff: one proposal file, 133 added lines; no production implementation.
- Local working tree clean on initial and final inspection; `git diff --check base..HEAD` passed.
- Remote production-code inspection/runtime: `tileops-dev:/data2/shared/ang.gao/robot_sim_physics_correctness`, clean at base `fe7f3d6448f0d3c148d1c5a494129a8b3c5e4127`.

## Blocking findings

### F1 — High: unconditional trace/3 does not reproduce MuJoCo weights for simple translational bodies

Location: proposal lines 54–58, with affected claim at 20–25 and missing explicit coverage at 90–101.

Trigger: an independent 2 kg body with a single z-axis prismatic joint, spherical geometry, zero COM offset, standard positive solref, constant impedance .9, Euler, diagexact disabled, and a condim1 plane contact. This is supported by existing `physics/joint.py:241` `PrismaticJoint` and is not excluded by the proposal.

MuJoCo uses an explicit `body_simple == 2` branch setting translational inverse weight to inverse mass; the general COM trace formula applies in a different branch. See [MuJoCo 3.15.0 engine_setconst.c, lines 842–869](https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_setconst.c#L842-L869).

Independent execution with actual MuJoCo 3.15.0 and our production tree helpers produced:

| Quantity | Proposal formula | Actual MuJoCo |
| --- | ---: | ---: |
| H | 2 | 2 |
| COM linear Jacobian | (0, 0, 1)^T | same |
| Body inverse weight | 0.16666666666666666 | 0.5 |
| Normal R at d=.9 | 0.01851851851851851 | 0.05555555555555554 |
| Normal force, identical A=.5 and rhs=37.587777777777795 | 72.49071428571432 N | 67.65800000000007 N |

The force in the proposal column is the exact positive scalar QP optimum, not an execution of unimplemented code. MuJoCo force is actual `efc_force`. Contact penetration was .01 m, solref (.02,1), timestep .002 s. Thus tighter ADMM convergence cannot repair this mismatch. An additional two-slider MuJoCo check gave trace weight 1/3 versus stored weight .5; a free-body control matched at .5.

Why proposed checks can miss it: isolated free masses, revolute chains, floating quadrupeds, fixed children, and static bodies need not enter the pure-translation special branch. The author's double-pendulum/quadruped probe covers the general branch and does not contradict this counterexample.

Required direction: freeze the reference-compatible body classification and weight construction, including the simple translational branch, using production model facts and no MuJoCo runtime dependency. Alternatively explicitly narrow and enforce the supported model boundary, with rejection before mutation; a documentation-only exception that still returns purportedly compatible R is insufficient. Do not blindly replace all articulated translation weights by 1/m.

Proof after repair: include an isolated single prismatic root and nearby non-simple articulated/welded configurations, compare actual `body_invweight0` and `efc_R`, and retain the condim1 scalar-force comparison above. Test the implemented production CPU and GPU assembly paths when implementation follows design approval.

### F2 — Medium: accepted impratio domain and frozen formula disagree with the reference floor

Location: proposal line 44 and line 72.

Trigger: accepted finite positive `impratio=1e-16` with condim3. The proposal requires division by that value. The reference divides by `max(1e-15, impratio)`; see [MuJoCo 3.15.0 engine_core_constraint.c, lines 2088–2106](https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_core_constraint.c#L2088-L2106).

Independent free-body fixture execution gives Rn=.05555555555555554 and actual tangent R=55555555555555.53, whereas the proposed formula gives 555555555555555.4 (tenfold larger). Controls at impratio 1 and 1e-15 match. This finding concerns assembly only and does not depend on fixing the deferred friction projection.

Why proposed checks can miss it: the validation list mentions impratio variation and invalid values but does not require the first positive values below the reference floor. A finite-positive validator accepts this counterexample by design.

Required direction: either freeze MuJoCo's denominator floor for the mujoco route, or define and enforce a narrower accepted lower bound. Preserve the documented per_row impratio==1 rule. Add boundary checks just below, at, and above 1e-15 and verify finite CPU/GPU assembly or the explicitly chosen rejection behavior. This is a small contract correction but should not be left to implementer discretion under an exact formula.

## Feasibility and non-blocking observations

The intended sequencing and bounded scope are sensible: validate R and frictionless dynamics before changing the known condim3 projector. No default-model change, friction correctness claim, mass modification, or runtime MuJoCo dependency is authorized by this proposal. CPU and GPU implementations are not present and have not been approved.

The general COM construction has available production building blocks: `body_velocity_jacobians` maps generalized velocity into each body frame, the COM shift is available from `SpatialInertia.com`, and `RobotTreeNumpy.crba` provides H0. For world-frame construction, `contact_jacobian` and forward kinematics also exist. The trace is invariant under body/world rotation when the Jacobian and inertia coordinates are consistent. Per-solve CPU recomputation and construction-time GPU upload fit existing ownership. Restoring current FK after reference computations is important and explicitly required.

The GPU engine already carries both contact body indices into Jacobian assembly and retains raw W separately from PGS W_diag; adding the body-weight array and routing indices to ADMM is feasible. Condim1 masking must include matrix rows and columns, rhs, warmstart state, projection/output, and unused slots; merely zeroing final tangential impulses would not satisfy the stated no-tangent-subproblem contract. The design's zero-row plus rho approach provides a viable implementation direction. Code review must verify actual kernel wiring rather than a helper-only test.

For ordinary inputs, the specified contact-weight summation, normal R floor, solimp normalization, width-zero behavior and isotropic tangent relation otherwise follow the pinned [constraint source](https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_core_constraint.c). Exact reference comparisons must continue to pin model options and effective contact parameters. Numerical extremes in finite positive solref and high solimp power should be covered or bounded during implementation; native float32 arithmetic must not silently introduce nonfinite assembled data.

Trajectory thresholds are intentionally unresolved until matched-discretization or measured convergence evidence exists. Treat that as a still-closed gate: assembly/snapshot success must not be presented as successful trajectory equivalence. If a newly discovered dynamics or geometry defect prevents the required articulated comparison, record it and seek a bounded design adjustment rather than silently relaxing the test.

## Executed verification and reproducibility

Read candidate, prior bounded diagnosis, existing CPU ADMM, CPU/GPU engine integration, Warp ADMM and solver kernels, tree/joint/inertia/Jacobian helpers, and paired-model validation builders. Read the pinned MuJoCo 3.15.0 sources directly, including body-weight construction, contact diagonal assembly, parameter normalization and impedance construction.

Executed git status/revision/diff/stat/hash/whitespace checks locally and revision/status remotely. Initial `rg` calls used several nonexistent guessed filenames; these returned ordinary missing-file errors and were followed by actual path discovery. An initial SSH `python -c` version check lost quoting in the PowerShell-to-SSH path and failed before running Python; subsequent stdin-based scripts executed successfully. Neither issue is a production-code failure.

Independent probe is saved outside the candidate at:

`C:/Users/AG/新建文件夹/admm_regularization_design_review_v1_probe.py`

Reproduction from PowerShell:

```powershell
Get-Content -LiteralPath 'C:/Users/AG/新建文件夹/admm_regularization_design_review_v1_probe.py' -Raw -Encoding UTF8 | ssh tileops-dev /data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python -
```

Observed exit 0, MuJoCo version 3.15.0. Assertions independently confirm the threefold prismatic-weight/R mismatch and the actual impratio denominator floor. The script uses current production tree routines to calculate the proposed reference weight, and actual MuJoCo to construct contacts/forces; it does not feed reference R into a production solver. It imports code from the clean remote base checkout. No candidate files were edited and the remote probe ran via stdin without writing repository files.

No GPU runtime or full regression suite was executed in this design review: there is no candidate production implementation, and the decisive design counterexamples are CPU scalar assembly facts. Real CUDA evidence and all promised implementation tests remain required for subsequent code approval. Existing historical logs and the author's supplied probe are not substituted for new execution evidence.

## Exact scope and remaining gates

This verdict binds only the candidate proposal v1 at the revision and digest above. It requests design changes before implementation under the applicable workflow. It does not approve the old projection-first proposal, any new friction law, condim3 solve correctness, warmstart/adaptive-rho behavior, default-iteration accuracy, trajectories, material calibration, full MuJoCo compatibility, or overall simulator correctness. A revised design must receive a fresh independent review, carrying these reproductions forward.
