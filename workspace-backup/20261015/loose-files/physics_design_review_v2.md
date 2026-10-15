Initiative: physics-correctness
Stage: design-review
Author: codex
Version: v2
Date: 2026-10-12
Status: accepted
Related Files: collab/physics-correctness__proposal__codex__v2.md; collab/physics-correctness__design-review__codex__v1.md; physics/constraint_solvers.py; physics/dynamics_utils.py; physics/implicit_contact_step.py; physics/solvers/pgs_solver.py; physics/solvers/pgs_split_impulse.py
Owner Summary: Approve the exact v2 design for the bounded CPU PGS generalized-inertia repair and energy/settling evidence changes. R1 is closed by the corrected impulse equation and fresh one-contact/four-contact experiments. No implementation or general physics-fidelity claim is approved.

## Identity and frozen candidate

- Reviewer session: `/root/physics_design_review_v2`, fresh independent reviewer, not the candidate author.
- Review kind: bounded design re-review, including independent closure of prior R1 and inspection of C1/C2/C3 dispositions.
- Local candidate: `C:/Users/AG/新建文件夹/robot_physics_correctness`.
- Base: `f771e612beaeea126e26926738097512cd2d809c`.
- Verified HEAD: `b5291c172c63820f242017fb2c1df380667f9505`.
- Proposal: `collab/physics-correctness__proposal__codex__v2.md`.
- Verified SHA256: `C31C4BFDF67E645601A87265E9ED6D66586000CE6BE5EBE7E85BE1D928EC7823`.
- Tracked delta from base: v1 proposal, v1 review, and v2 proposal only; 337 added lines across three documents. No physics implementation changes.
- Local working-tree status: only untracked `physics_audit_probe.py`, excluded scratch evidence. No candidate source or document was modified by this reviewer.
- Remote runtime checkout: `/data2/shared/ang.gao/robot_sim_physics_correctness`, independently verified at the base revision, with only untracked `physics_audit_probe.py`. Thus runtime checks exercise the unchanged baseline physics, not an implemented repair.
- Read the archived AGENTS.md, complete collab/README.md and REVIEW_PRINCIPLES.md, the owner handoff, proposal v2, and prior full report. This report is saved outside the candidate for the coordinator to record unchanged.

## Verdict

**Approve.** No unresolved substantive design finding remains within this exact scope. This authorizes implementation of the reviewed design and its tests; it does not approve any implementation, release, merge, production activation, or broader numerical or physical certification.

## Findings and prior-review dispositions

### R1 — Closed at design level

Proposal lines 81–91 and 104–109 now match the solver's actual impulse-level equations. The normal row adds CFM to mobility; at equilibrium its regularization contribution is `CFM * lambda_n`, where `lambda_n = M*g*dt/N` for N symmetric supporting contacts. The existing bias is `ERP*(depth-slop)/dt` for `0 < ERP <= 1`, and `ERP*(depth-slop)` for `ERP > 1`. Equating these gives the revised `dt**2` and `dt` formulas respectively.

The proposal expressly excludes restitution, active depenetration clamping, and ERP=0 from this restoring-height oracle. It also requires resolvable-CFM fixtures, impulse/velocity checks, both ERP branches, and timestep scaling. Those conditions prevent the original dimensional error from being hidden by the micrometre drop-height tolerance.

Fresh direct-solver experiments used M=2 kg, g=9.81 m/s², CFM=.001, slop=.005, incoming predicted velocity `[0,0,-g*dt,0,0,0]`, condim=1, identity body transform, inverse mass .5, identity inverse rotational inertia, zero restitution, tolerance 1e-14, and max_iter=30000. Four-contact points were `(±.5, ±.5, 0)` on the same rigid body, so they share translational and rotational mobility; they are not four independent bodies. Contact depths were initialized from the analytic equilibrium, without warm-cache seeding.

| Contacts | ERP | dt (s) | Excess depth (m) | Max absolute post-solve spatial velocity component | Max per-contact impulse error (N s) |
|---|---:|---:|---:|---:|---:|
| 1 | .8 | .01 | 2.4525e-6 | 1.39e-17 | 2.78e-17 |
| 1 | .8 | .005 | 6.13125e-7 | 6.94e-18 | 1.39e-17 |
| 1 | 10 | .01 | 1.962e-5 | 0 | 0 |
| 1 | 10 | .005 | 9.81e-6 | 0 | 0 |
| 4 | .8 | .01 | 6.13125e-7 | 4.99e-15 | 4.99e-12 |
| 4 | .8 | .005 | 1.5328125e-7 | 4.99e-15 | 5.00e-12 |
| 4 | 10 | .01 | 4.905e-6 | 4.99e-15 | 4.99e-12 |
| 4 | 10 | .005 | 2.4525e-6 | 4.99e-15 | 5.00e-12 |

The excess-depth ratios are exactly 4 and 2 as proposed. Every case passes the proposed 1e-10 speed and impulse limits. Repeating the original wrong single-contact formula at ERP=.8 produces +.0193850299401 m/s at dt=.01 and +.0194829341317 m/s at dt=.005, independently retaining the original counterexample. The generous iteration ceiling proves this equilibrium acceptance is feasible; it does not establish default-iteration convergence or performance.

### C1 — Addressed in the design; implementation proof remains required

Proposal lines 111–116 retain the legacy `ImplicitContactStep` velocity entrypoint by delegating to the same adapter/cache rather than maintaining a second articulated PGS algorithm. Inspection confirms the current legacy method receives `a_u` after `tree.aba(q,qdot,tau,ext_forces)`, so passing that acceleration into `DynamicsCache.qacc_smooth` preserves the intended external-force contribution. The shared adapter does not need to re-evaluate smooth forces. The available dynamics helpers have the required semantics for thin compatibility delegates.

The legacy path is still defective in the inspected baseline; the document commits to repairing it, rather than claiming it has already changed. Implementation review must verify the delegated impact case, preservation of precomputed smooth acceleration including external forces, and unchanged acceleration-level behavior.

### C2 — Addressed in the design; implementation proof remains required

Proposal lines 37–55 and 118–125 specify the body-frame `(num_bodies,6,nv)` map, finite lower-triangular Cholesky context, joint presence of both inputs, validation before contact-frame/warm-cache mutation, no malformed-context fallback, and forwarding only to the two compatible PGS solver classes. This is implementable with the current interfaces.

`SpatialTransform.matrix()` agrees with the velocity transform, and its transpose with backward force propagation. `DynamicsCache` already computes H and L on the contact path; the adapter already applies generalized impulses using L. Therefore a single `Jg H^-1 Jg.T` assembly can replace only the independent-body mobility block while retaining existing rows, friction regularization/projection, restitution, warmstart, and output mapping. The PGS-SI wrapper already delegates its solve to PGS, so no second iteration algorithm is necessary.

The required rotated/fixed/revolute/multiple-root and condim 1/3/4/6 tests, independent geometry, B*v identity, and impulse dual identity make the critical conventions falsifiable. Passing finite/shape checks alone will not be accepted as proof of a correct kinematic map or correct context provenance. The same per-step L must actually form and apply mobility.

### C3 — Addressed in the design; implementation proof remains required

Proposal lines 71–79 and 127–138 freeze the 2 s energy duration, three timesteps including the endpoint, full-path and endpoint error reporting, potential-datum-invariant scale, and narrowly interpreted convergence band. The existing test instead bypasses CpuEngine, samples before each step, compares only endpoints, and uses a potential-zero-sensitive denominator; the specified replacement directly targets those weaknesses.

The settling conditions now include a micrometre height tolerance and a final-window speed/range check with stated numerical rationale. For ADMM, actual code forms the coupled A matrix and diagonal per-row R. At stationary symmetric support, `A*f + a_u_c = 0`, so the remaining equilibrium equation is `R*f = a_ref`; the diagonal A entry cannot generally be substituted for the coupled row sum. The revised independently formed geometric Jacobian/CRBA/scalar-root method accounts for that coupling. It does not inherit the historical Q30 claim that structure cancels in every multi-contact configuration, nor fit the observed terminal height.

## Verification and evidence provenance

Inspected high-risk flow: CpuEngine contact detection/cache -> StepPipeline smooth acceleration -> VelocityLevelAdapter -> PGS/PGS-SI row construction, mobility, CFM/friction regularization, bias, warmstart, impulse output -> backward generalized impulse mapping and integration. Also inspected legacy implicit integration and its duplicate helpers, spatial velocity/force transforms, ADMM per-row R/reference acceleration, both weak validation tests, and relevant force-system/per-row-R/Q30 historical design sections. Historical explanations were treated as claims, not proof.

Commands and fresh results:

- Local `git status --short`, `git rev-parse HEAD`, `git diff --stat f771e612beaeea126e26926738097512cd2d809c HEAD`, and `Get-FileHash -Algorithm SHA256 ...v2.md`: identity and tree results above.
- Local `git diff --check f771e612beaeea126e26926738097512cd2d809c HEAD`: passed, no output.
- `ssh tileops-dev 'cd /data2/shared/ang.gao/robot_sim_physics_correctness && git rev-parse HEAD && git status --short && cat /tmp/physics_inertia_repro.py'`: verified baseline and inspected the reproduction before executing it.
- Fresh Python scripts sent over SSH stdin to `/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python -`, explicitly changing directory to the remote worktree and prepending it to sys.path: executed the equilibrium cases and wrong-formula controls described above. Per-contact lambdas were read from the solver's just-produced warm cache; post-solve velocity was computed from the actual returned body impulse. No implementation file was written.
- The same isolated Python invocation imported MuJoCo 3.15.0 and Pinocchio 4.1.0, then executed the inspected `/tmp/physics_inertia_repro.py`. Both PGS and PGS-SI returned post-impact vz -0.95, -0.5, and approximately -0.05 m/s for fixed-child mass fractions .05, .5, and .95 with total mass 1 kg. Each had one real detected contact. Correct rigid-composite behavior would stop for every split. This confirms the common mobility defect remains present and the proposed engine regression is necessary.
- `ssh tileops-dev 'cd /data2/shared/ang.gao/robot_sim_physics_correctness && /data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python -m pytest -q tests/reference/pinocchio -rs'`: **5 passed in 0.15 s**, no skips.

An initial combined SSH command failed in shell parsing around a quoted Python `-c` expression before executing its intended checks. This was a command-quoting failure, not a repository/dependency failure; corrected separate provenance checks and stdin scripts above completed successfully.

## Coverage limits and remaining gates

This was a bounded design re-review. No implementation exists in the candidate, so repaired engine behavior, malformed-context rejection, legacy equivalence, and the full acceptance matrix remain unverified implementation obligations. No GPU test, full energy trajectory, complete settling trajectory, or full MuJoCo comparison was rerun by this reviewer. The proposal's historical energy/settling numbers are not promoted to fresh reviewer measurements. Imported reference packages and five Pinocchio tests do not certify all independent-engine comparisons. Python/pytest caches may have been generated in the remote checkout; no tracked source was modified.

The approved scope is the specified CPU PGS/PGS-SI generalized-inertia correction through the shared adapter, legacy velocity-entrypoint delegation, targeted acceptance regressions, and revised energy/model-specific settling evidence. It preserves the existing ERP and friction model choices. Subsequent implementation requires fresh independent code review and the promised relevant analytical, reference, contact, collision, and real CUDA evidence, with failures classified by cause.

GJK shallow-overlap correctness, CPU/GPU manifold differences and GPU solver convergence, friction-box anisotropy, and ADMM angular-condim support remain outside this repair and open. Rendering, complete RL integration, release/merge/activation, and real-material or general contact-fidelity claims remain closed. Foot/ground calibration still requires real force/displacement data and held-out validation; agreement between simulators is not that evidence.
