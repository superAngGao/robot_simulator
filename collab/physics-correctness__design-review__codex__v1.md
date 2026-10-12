Initiative: physics-correctness
Stage: review
Author: codex
Version: v1
Date: 2026-10-12
Status: in_review
Related Files: collab/physics-correctness__proposal__codex__v1.md; physics/constraint_solvers.py; physics/dynamics_utils.py; physics/solvers/pgs_solver.py; physics/solvers/pgs_split_impulse.py
Owner Summary: Request changes. The generalized-inertia repair addresses a reproduced CPU engine defect, but the proposed PGS settling oracle uses the wrong timestep dependence for the actual ERP=0.8 fixture. Correct the oracle before implementation. This is design review only.

## Identity, candidate, and isolation

- Reviewer session: `/root/physics_design_review`, fresh independent session; did not author the candidate.
- Review kind: bounded design review of the energy/settling audit and CPU PGS generalized-inertia correction.
- Local candidate: `C:/Users/AG/新建文件夹/robot_physics_correctness`.
- Base: `f771e612beaeea126e26926738097512cd2d809c`.
- Verified HEAD: `a9b970b2406010ef7f488fb6fc10d35812d861c8`.
- Verified proposal SHA256: `036014A35BF0685299CE260AD42B0FC7BC80AF5D65EAB07D0163CB49611C59DA`.
- Actual candidate delta: one tracked proposal, 97 added lines; no physics implementation changes.
- Actual local status: untracked `physics_audit_probe.py`, declared scratch evidence and excluded from the frozen implementation.
- Remote execution worktree: `/data2/shared/ang.gao/robot_sim_physics_correctness`, verified at the base revision above, with the same untracked probe. This is a baseline runtime checkout, not a checkout of the proposal commit. Its physics code is the candidate's unchanged physics code.
- Read the owner handoff and the archived AGENTS.md, collab/README.md, and collab/REVIEW_PRINCIPLES.md. No candidate file was modified by the reviewer. Test-generated Python/pytest caches are possible; no implementation edits were made.

## Verdict

**Request changes.** R1 is a blocking numerical-oracle error. The proposed repair direction is otherwise coherent with the actual data flow: solve using generalized mobility, then apply the body impulses through the dual kinematic map and the same mass inverse. The engine-level defect is independently reproduced. Design approval and implementation/code approval remain pending.

## Findings

### R1 — High: PGS settling formula is wrong for the proposed ERP=0.8 case

Location: proposal line 83; `physics/solvers/pgs_solver.py` normal regularization and bias/iteration construction, especially lines 326 and 372–377.

The proposal gives `depth = slop + CFM*(M*g/4)*dt/ERP`. This is only the formula for the solver's `ERP > 1` inverse-time branch. Its settling fixture uses `ERP=0.8`, for which the solver constructs a velocity bias `ERP*(depth-slop)/dt`. The solver unknown is impulse, not force.

For four symmetric contacts, zero restitution, no active depenetration clamp, and stationary normal motion, each impulse is `lambda_n=(M*g/4)*dt`. Since `v_pred + W_physical*lambda = 0`, the regularized normal equation reduces to `CFM*lambda_n = ERP*(depth-slop)/dt`. Therefore the correct excess depth is:

- `CFM*(M*g/4)*dt**2/ERP` when `0 < ERP <= 1`;
- `CFM*(M*g/4)*dt/ERP` when `ERP > 1` under this code's inverse-time interpretation.

The ERP=0 case has no penetration-restoring equilibrium oracle of this form. Clamped bias or restitution also requires separate assumptions.

Trigger and consequence: implementing the proposed formula at ERP=0.8 gives a false expected equilibrium and wrong dt scaling. With current very small CFM this could be hidden by a height tolerance, defeating the purpose of replacing the 10 cm assertion. At a resolvable nonzero CFM it rejects correct behavior or encourages changing the solver to match a wrong oracle.

Independent runtime counterexample, using the existing direct PGS solver with a single 2 kg body, normal through its origin, `g=9.81`, `CFM=0.001`, `slop=0.005`, `ERP=0.8`, predicted velocity `-g*dt`, condim=1, and no clamp activation:

| dt | depth from proposal | post-solve vz from proposal | corrected depth | post-solve vz corrected |
|---|---|---|---|---|
| 0.01 s | 0.00524525 m | +0.0193850299401 m/s | 0.0050024525 m | -1.39e-17 m/s |
| 0.005 s | 0.005122625 m | +0.0194829341317 m/s | 0.005000613125 m | -6.94e-18 m/s |

Corrected impulses were respectively 0.1962 and 0.0981 N s, equal to `m*g*dt`. ERP=10 controls at both timesteps validated the one-dt branch and returned exactly zero post-solve normal velocity. The single-contact test substitutes `m*g` for the four-contact per-foot load; the dimensional error is identical.

Why existing checks miss this: the old cross-solver height test does not assert force/impulse balance or timestep scaling, and the proposal has no dt/CFM-sensitive test of its formula.

Required repair/proof: freeze the piecewise ERP semantics and equilibrium assumptions in the revised proposal. Verify both ERP branches at two timesteps with a CFM large enough to resolve excess depth without activating the clamp. For the symmetric four-foot scenario, independently check per-foot impulse `M*g*dt/4`, zero normal velocity, and excess-depth ratios 4 or 2 under timestep halving, as appropriate. Do not establish a new height tolerance that masks this formula error.

## Non-blocking comments and implementation review obligations

### C1 — Explicitly account for the retained legacy articulated entrypoint

`physics/implicit_contact_step.py::_step_velocity` still calls the PGS solver with independent-body inverse masses and subsequently applies impulses using CRBA. It duplicates the original mismatch. The historical reflection describes this class as retired, but the implementation remains importable, and tests still use the class (currently observed callsites use the acceleration-level path).

The proposal specifically repairs `VelocityLevelAdapter`, so this does not invalidate that bounded repair. Record explicitly that the legacy articulated velocity path is outside the initial guarantee, or route it through shared corrected context construction if retaining it as a supported PGS entrypoint. Do not call it an independent-body caller: it accepts articulated trees. No repository production caller of this class was found in the inspected Python callsites. A claim covering all CPU PGS entrypoints requires closing or removing this path as well.

### C2 — Concrete acceptance boundary for generalized body Jacobians and mobility

The existing `qdot_to_body_v` recursion, cached Cholesky factor, PGS body rows, and backward impulse propagation provide the dependencies needed for the proposed design. Implementation review should require these identities and counterexamples, without a second PGS iteration implementation:

1. Every body Jacobian has shape `(6,nv)`, in the existing `[linear, angular]` body-frame convention. At nonzero/rotated configurations, `B_i @ qdot` equals independently evaluated body velocity. Fixed children inherit their parents' motion; independent roots have disjoint generalized columns.
2. For arbitrary body impulses, the backward pass equals `sum(B_i.T @ impulse_i)`. Include off-axis points and rotated fixed offsets so a missing transpose or moment arm cannot pass.
3. Generalized row assembly uses `Ji*Bi + Jj*Bj`, with the already-signed second-body rows. Check linear and angular rows for condim 1/3/4/6, contacts between distinct roots, and revolute articulation. Use analytic fixture geometry or independently constructed geometric Jacobians, not the new helper as its own oracle.
4. The unregularized matrix is `Jg @ H^-1 @ Jg.T`; the existing CFM/friction regularization, projection, warmstarting, and iteration loop remain the single implementation. Both PGS classes forward the same optional context. Other legacy solver signatures must not receive unsupported keywords.
5. The Cholesky factor used to form mobility is the same factor used to apply impulses. No stale q/configuration cache or independent second inverse may silently substitute a different mobility.
6. Partial context input is rejected before changing contacts/warmstart state. Keep the old no-context direct-call behavior and test that compatibility explicitly. Specify ordinary dimension/finite-value preconditions at the new API boundary; do not silently fall back from malformed articulated context to independent masses.

These points make the proposed acceptance cases actionable; they do not authorize a new solver variant, a broad capability framework, or GPU algorithm changes.

### C3 — Interpret energy convergence and model-specific equilibrium narrowly

The potential-datum-invariant normalization and full-trajectory step-halving study are improvements over the existing endpoint ratio. A first-order ratio band establishes numerical convergence for the frozen fixture; it does not itself bound the error coefficient or certify arbitrary trajectories. Retain measured absolute errors and the independent acceleration comparison in the evidence. Freeze the exact duration/timesteps, residual definitions, and equilibrium comparison tolerances before code approval, with their numerical basis. The ADMM four-row coupled equilibrium must remain independent of simply fitting its observed terminal height.

## Verification performed

Static inspection covered CpuEngine -> StepPipeline -> DynamicsCache -> VelocityLevelAdapter -> PGS/PGS-SI -> impulses_to_generalized; body row construction, normal/friction regularization, restitution/bias branches, warmstart and output mapping; ADMM generalized Jacobians/per-row R and reference acceleration; the legacy implicit step; both weak validation tests and model builders. Targeted historical reading included the force-system decision, the Q30 penetration argument, Q48 shallow-contact workaround, and the contact-solver design document. Historical claims were not treated as approvals.

Commands executed independently:

- Local `git status --short`, `git rev-parse HEAD`, `git diff f771e612beaeea126e26926738097512cd2d809c..HEAD --stat`, and `Get-FileHash ... -Algorithm SHA256`, with the results recorded above.
- SSH `tileops-dev`, remote `git rev-parse HEAD` / `git status --short`, confirming baseline execution provenance.
- From the remote worktree, `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -m pytest -q tests/reference/analytical tests/reference/pinocchio tests/integration/contact_pipeline tests/unit/collision -rs`: **356 passed, 5 skipped in 21.72 s**. All five skips are missing Pinocchio in that original environment. This was a new execution, not reuse of the author's log.
- From that same worktree, `PYTHONPATH=. /home/ang.gao/miniforge3/envs/robot_sim/bin/python /tmp/physics_inertia_repro.py`, after inspection of its content: both PGS and PGS-SI give post-impact vz of -0.95, -0.5, and approximately -0.05 m/s for fixed-child mass fractions 0.05, 0.5, and 0.95 with total mass 1 kg, initial vz=-1 m/s, zero gravity, ERP=CFM=0, one detected sphere/ground contact. The physically rigid composite should stop for all three mass splits. This independently confirms the proposed defect and minimal regression.
- An independent Python script passed over SSH stdin to the baseline interpreter constructed the one-row equilibrium fixture described in R1. It instantiated `ContactConstraint(0,-1, origin, +z, ..., depth, mu=0, condim=1)`, `PGSContactSolver(max_iter=100,tolerance=1e-14,erp=erp,cfm=.001,slop=.005)`, identity world transform, inverse mass 0.5, identity inverse inertia, and body velocity `[0,0,-9.81*dt,0,0,0]`. It compared depths from both formulas and measured `v_z + impulse_z/2`. Results are recorded in R1, including the ERP=10 controls. No repository source file was written for this script.

Read, but did not count as independent new execution: `/tmp/physics_probe_baseline.jsonl` and `/tmp/physics_baseline_tests.log`. Their energy/settling/geometry values agree with the proposal's stated baseline; this review did not independently rerun the full double-pendulum trajectory or GJK scan. The subsequent isolated MuJoCo/Pinocchio installation and `/tmp/physics_extra_baseline.jsonl` were reported by the author; this review does not certify their results.

## Scope and remaining gates

No implementation is approved by this design report. The reviewed intended scope remains CPU-entrypoint energy evidence, model-specific settling evidence, and generalized inertia in the two PGS variants via the engine adapter. R1 must be revised and independently re-reviewed before implementing the design checkpoint.

Actual CUDA tests, the isolated-environment MuJoCo/Pinocchio comparisons, and the post-fix CPU regression matrix remain required implementation evidence. Their absence here is an unexecuted design-stage check, not a code failure or a successful GPU result. The passing baseline subset coexists with a demonstrated engine defect and cannot certify contact fidelity.

GJK shallow overlaps, CPU/GPU contact manifold differences and solver convergence, friction-cone anisotropy, ADMM angular rows, and the legacy articulated PGS path remain separately bounded gaps. Rendering, RL framework expansion, release/merge/production activation, and claims of real-material accuracy remain closed. Physical foot/ground calibration still needs real held-out force/displacement data.
