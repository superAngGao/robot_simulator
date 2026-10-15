Initiative: physics-correctness
Stage: code-review
Author: codex
Version: v1
Date: 2026-10-12
Status: accepted
Related Files: collab/physics-correctness__proposal__codex__v2.md; collab/physics-correctness__proposal__codex__v3.md; collab/physics-correctness__design-audit__codex__v1.md; physics/constraint_solvers.py; physics/dynamics_utils.py; physics/implicit_contact_step.py; physics/solvers/pgs_solver.py; physics/solvers/pgs_split_impulse.py; tests/integration/contact_pipeline/test_articulated_pgs_inertia.py; tests/validation/rigid_body/physics_metrics.py
Owner Summary: Approve with comments the exact bounded CPU PGS inertia repair and revised numerical evidence at 70841b9. Fresh independent tests close the original inertia counterexamples, validate the corrected settling reference and complete energy convergence test, and reproduce the deliberately unresolved interaction defects. This is not general physics-fidelity approval.

## Identity, candidate and verdict

Reviewer session: `/root/physics_code_review_v1`, a fresh independent code reviewer who did not author this candidate. Read the archived `c4_worker_completion/AGENTS.md`, complete `collab/README.md` and `collab/REVIEW_PRINCIPLES.md`, accepted proposals v2/v3, design reviews v1/v2/v3, implementation note, audit, and evidence descriptions. Inspected actual production/test diffs, full changed production functions, engine/cache/pipeline data flow, and relevant unchanged solver/GPU/geometry source. Prior approvals and author logs were treated as claims, not code approval.

- Base: `f771e612beaeea126e26926738097512cd2d809c`.
- Exact reviewed HEAD: `70841b9c509a51c70ca0fe6b1f094597e29026ce`.
- Local candidate: `C:/Users/AG/新建文件夹/robot_physics_correctness`.
- Runtime candidate: SSH `tileops-dev`, `/data2/shared/ang.gao/robot_sim_physics_correctness`.
- Both candidates were clean at the beginning and end, with the exact HEAD above.
- Baseline runtime: `/data2/shared/ang.gao/robot_simulator`, independently verified clean at the base before and after the baseline reproductions.
- No tracked candidate or baseline file was edited. Scratch script and this report are outside the candidate.

**Verdict: Approve with comments.** No blocking code finding or required evidence gap remains in the reviewed bounded scope. C1 below is explicitly non-blocking. Approval binds this exact revision; it does not approve ADMM/GPU/geometry repairs, general physical fidelity, release, merge, or production activation.

## Findings and non-blocking follow-up

No substantive production defect was found in the new CPU articulated-mobility path or the corrected fixture-specific numerical references.

### C1 — P3, non-blocking: retain an oblique contact regression with an independent geometric oracle

Location: `tests/integration/contact_pipeline/test_articulated_pgs_inertia.py`, especially `test_two_roots_conserve_momentum_with_angular_rows` and `test_body_jacobians_match_velocities_and_virtual_work`.

The permanent multi-root condim tests deliberately isolate normal/angular motion at coincident origins and set tangential friction to zero. They are good original-defect regressions, but do not by themselves strongly challenge rotated tangential rows with nonzero moment arms between articulated roots. The B/velocity and transpose tests also compare related recursion implementations. A future sign/frame regression can be easier to miss in these symmetric fixtures.

This is not a demonstrated candidate failure. I independently checked 12 cases using central finite differences of world-space point positions and rotations, then compared the resulting J against the actual unregularized W captured from PGS and against nonzero generalized impulses applied through the adapter. All cases passed (details below). Together with the permanent analytical composite, revolute, multi-root, and malformed-context cases, this supplies the required present-revision evidence.

Recommended follow-up: retain the external repro as review evidence now; promote a compact, stable version into the permanent suite in a subsequent reviewed test change. Keep condim 1/3/4/6, nonzero contact moment arms, both root orientations and revolute angles, and a nonzero normal impulse. Do not use the new B helper or PGS row helper as the geometric oracle. Proof after that follow-up is the new independent geometric regression passing without weakening the analytical cases. No change to this frozen implementation is required by C1.

## Production correctness and design conformance

The repair implements the accepted design faithfully:

1. `body_velocity_jacobians` follows the body-frame spatial velocity transform and joint-column recursion, including fixed descendants and disjoint root columns. The existing backward body-impulse propagation is the dual map.
2. `VelocityLevelAdapter` creates B at `cache.q`, supplies the same `cache.L` subsequently used to apply the generalized impulse, and only adds the new keywords for PGS/PGS-SI. Engine contact detection and StepPipeline supply a per-step configuration/cache with H/L computed for the contact path. This avoids the original independent-mass solve followed by articulated-mass application.
3. PGS builds already-signed first/second-body rows into Jg, solves `L * weighted_J = Jg.T`, and forms `weighted_J.T * weighted_J`. This is `Jg H^-1 Jg.T`. Cross-contact and cross-body coupling are retained. Independent-body callers still use the original mobility assembly.
4. Regularization, free velocities, ERP/restitution bias, warmstarting, box projection, iteration, and output body impulses share one `_solve_rows` implementation. Inspection of the extraction found no unintended numerical change to that common row logic. PGS-SI forwards context to the same PGS implementation.
5. Paired-context checks occur before contact-frame or warm-cache mutation. Finite shapes, square lower-triangular L with positive diagonal, missing pairs, and malformed input are tested. PGS-SI publishes its zero position-correction list after successful inner execution, preserving previous state on rejected context. No malformed-context fallback to independent masses was added.
6. The legacy implicit velocity entrypoint now delegates to the common adapter/cache and explicitly supplies its already-computed `a_u`; its external-force contribution is preserved. The acceleration branch is unchanged. Compatibility helper methods delegate to common dynamics utilities rather than maintaining duplicate formulas.

The permanent 50-case regression file covers real CpuEngine impacts, legacy stepping, mass partition, offset/rotated composite response from an analytical rigid-body formula, revolute response, multiple roots/angular rows, B/velocity and virtual-work identities, rejected context state, both ERP unit branches, one/four-contact prescribed load equilibrium at two timesteps, and legacy external-force prediction. The original 20 outside-repo cases independently fail on base and pass on this candidate, so they actually discriminate the repaired behavior.

## Numerical references and evidence truthfulness

The energy replacement runs the actual CpuEngine, records every state including t=0 and t=2 s, checks finite energy, normalizes by initial energy above hanging equilibrium, verifies invariance under an added 1000 J datum, and compares three equal-duration timesteps. The [1.8,2.2] ratio is correctly described as a first-order fixture regression, not a universal absolute-error guarantee. The complete candidate energy test passed freshly in this review. The implementation note's absolute errors agree with the preserved baseline measurements; this review's pytest run did not print newly measured absolute values, so those tabulated decimal values remain separately identified historical measurements.

The corrected PGS drop reference is computed from the semi-implicit first state satisfying the detector's strict depth > 1e-10 threshold. It is not fitted to terminal measurements. The default result is n=391, z=.4199280336 m, and the arithmetic/boundary tests independently check it. Slop is correctly treated as an allowed interval, not an attraction target. The CFM-only drift estimate is explicitly separate from finite-iteration error and below 5 nm for the default fixture. The prescribed above-slop ERP/CFM equilibrium tests remain and test distinct behavior.

ADMM's reference forms geometric normal rows independently from its contact Jacobian, uses their coupled CRBA mobility, verifies the four-row gravity balance, and solves the scalar soft equilibrium. It is accurately called a model-consistency reference rather than an independent dynamics implementation. The revised default settling test freshly passed both solvers' 1 micrometre height/range, 1e-5 m/s tail-speed conditions and ADMM load balance. Recorded drop arrays sample t=0 through T-dt; the evidence README states this limitation.

The invalid v2 439-pass/one-failure run and reference fields remain explicitly marked invalid. The audit now emits `reference_z` and `reference_kind`, separating PGS discrete arrest from ADMM equilibrium. Optional MuJoCo skipping is localized to calls actually using MuJoCo, and broad engine-import exception skipping is removed. The tightened analytical friction checks validate the configured compliant model instead of claiming hard no-slip with a loose envelope.

The overall interaction architecture is reasonable: separate geometry, smooth forces, contact equations, numerical solve, and integration, with dual J/J-transpose maps and common articulated inertia. Current code only partially conforms to the broader intended interaction equations. PGS and ADMM selectors change the contact model as well as the numerical algorithm; box friction and circular friction are different admissible regions; ERP's two unit branches and material parameter meanings remain contract issues. The audit correctly states these limits and does not present this CPU repair as their resolution.

Fresh diagnostic execution and source inspection support the explicitly deferred defects:

- CPU ADMM returned approximately [.9,.9,0] with objective -1.8 for the stated cone QP whose exact feasible minimizer [1.35,1.35,0] has objective -2.025. The normal-first projection returns [1,1,0] for [1,2,0], instead of the Euclidean cone projection [1.5,1.5,0]. GPU uses the same defective projection form by source inspection; this review did not perform a separate GPU QP-optimality experiment.
- Real H200 Jacobi and colored PGS tangent impulses were -.5714285374 and -.5714285970, agreeing with the hard fixed point -.5714285714 rather than the intended compliant -.5657142857. Source confirms R is present in the update denominator but absent from the residual matrix.
- Real GPU ADMM R/A tangent ratio was .053163223 versus intended .052631579. Source shows ADMM deriving its R from the already regularized W_diag.
- An elevated FlatTerrain requested z=.5; GPU static height remained 0 and detected zero contacts while CPU detected one. `from_merged` omits that height.
- Fresh geometry queries reproduced all three box-hull misses at .01/.1/1 mm and the box-cylinder miss at 1 mm, with detection at 3 mm. Smaller cylinder cases did detect contacts. The audit qualifies implementation weaknesses separately from an exclusive root-cause claim, appropriately.

These are existing defects retained in scope as factual diagnostic evidence, not newly introduced blockers in the authorized CPU-only repair. No global claim that interaction code conforms to all its intended equations is approved.

## Independent executions

Runtime interpreter: `/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python`, using the isolated audit environment with NumPy/SciPy, MuJoCo and Pinocchio. CUDA diagnostic reported Warp 1.12.0, toolkit 12.9, driver 13.2 and NVIDIA H200. `CUDA_VISIBLE_DEVICES=3` selected the available physical device. Ordinary Warp compiled-kernel cache was reused; launches/results were new.

From the remote candidate root, with `OPENBLAS_NUM_THREADS=1`:

1. `$PY -m pytest -q tests/reference/analytical tests/reference/pinocchio tests/integration/contact_pipeline tests/unit/collision tests/validation/rigid_body/test_physics_metrics.py -rs`: **419 passed in 22.15 s**, no skips.
2. `$PY -m pytest -q tests/validation/rigid_body/test_contact_vs_mujoco.py::TestQuadrupedDropPgsSi::test_pgs_si_vs_admm_settling_height tests/integration/test_cpu_engine_shapes.py -rs`: **13 passed in 89.43 s**, no skips.
3. `$PY -m pytest -q tests/validation/rigid_body/test_dynamics_vs_mujoco.py::TestDoublePendulumDynamics::test_energy_conservation -rs`: **1 passed in 21.22 s**, no skips. This executed the complete three-timestep, two-second energy study.
4. `CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 $PY -m tests.validation.rigid_body.audit_solver_contracts`: completed with the fresh results above. Exit zero signifies diagnostic completion, not passing those unresolved contracts.

These are **433 passing candidate tests** across the three disjoint pytest selections, in addition to the reproductions/diagnostics below.

Original repro, read before execution: `C:/Users/AG/新建文件夹/test_physics_inertia_repro.py`, remote `/tmp/test_physics_inertia_repro.py`. From each baseline/candidate root:

`OPENBLAS_NUM_THREADS=1 PYTHONPATH=. $PY -m pytest -q /tmp/test_physics_inertia_repro.py --tb=no`

Results: clean base **20 failed in 1.22 s**; candidate **20 passed in 1.18 s**. These are fresh executions; the failures are expected baseline code failures, not dependency or toolchain failures.

Additional independent geometric repro is preserved outside the repository at:

`C:/Users/AG/新建文件夹/physics_review_fd.py`

Reproduce from PowerShell:

```powershell
Get-Content -Raw 'C:/Users/AG/新建文件夹/physics_review_fd.py' | ssh tileops-dev 'OPENBLAS_NUM_THREADS=1 /data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python -'
```

It explicitly selects the remote candidate root. Three deterministic nonzero configurations of a quadruped plus a welded second root, each at condim 1/3/4/6, use central FK differences with epsilon=1e-6 for point and angular velocities. A PGS subclass only captures the unregularized matrix and delegates to the unchanged real row solver; the actual adapter applies the result. Final checks require a nonzero normal impulse and compare H*delta-v with finite-difference J-transpose*lambda. All 12 cases passed at atol=rtol=2e-8; largest absolute W entry discrepancy was 2.3560e-8 (relative allowance also applies), largest generalized impulse discrepancy 1.1084e-10. An initial exploratory run had zero impulses; it was strengthened with an approaching velocity and explicit lambda_n>0.1 before the final reported run. The final saved script contains that strengthened check.

A separate SSH-stdin script freshly ran the eight geometry queries from the inspected audit fixture, producing the detection/miss results above; it wrote no source files.

Local static checks:

- `git diff --check BASE HEAD -- physics tests`: passed with no output.
- Ruff check on all five changed production modules, articulated regression, physics_metrics, both audit scripts and metric tests: **All checks passed**.
- Ruff format check on those same ten files: **10 files already formatted**.
- Whole-delta `git diff --check BASE HEAD` reports trailing spaces only in preserved raw failure logs (`physics_inertia_before.log`, `v2-oracle-tests.log`). This is disclosed raw evidence formatting, not a production/test whitespace failure; do not silently rewrite the preserved logs.

Some initial read-only commands used nonexistent paths for the archived principles, test file and GPU helper modules, and one Windows rg wildcard path was rejected. Correct locations were found and inspected. These were search/command errors, not failed repository tests. No failed physics execution is represented as a passing check.

## Provenance, limitations and approved boundary

Inspected author evidence includes the preserved baseline JSON/logs, invalid v2 oracle run, and implementation note. The coordinator separately reported an exact-candidate 432-pass targeted run; this report's independent 433 tests are its own executions. Prior design v3's four fresh PGS trajectory sweeps and author real-GPU 35-test/one-settling-test logs remain prior evidence, not new runs by this reviewer. I did not rerun the entire long MuJoCo suite, all four PGS sweeps, the full GPU regression suite, real-material tests, or broad manifold/friction studies. The exact production delta is CPU-only; fresh GPU contract diagnostic execution plus unchanged GPU source and retained prior GPU evidence are proportionate here.

Tests may generate ignored Python/pytest caches in the candidate and baseline and use the existing external Warp cache. The only deliberately written reviewer artifacts are the outside-repository scratch repro and this report. Final `git status --short` for local candidate, remote candidate and original server main was empty; final HEADs remain candidate 70841b9 and base f771e61 as stated.

Approved: CPU PGS/PGS-SI articulated inertia through the shared adapter, legacy velocity delegation with external forces preserved, compatibility of the independent-body path, accepted regression matrix, CPU energy/convergence and model-specific settling evidence, and the bounded factual audit/deferral at this exact revision. C1 is a nonblocking test-hardening follow-up, not a demand to change the frozen candidate.

Remaining closed: ADMM optimization-contract repair or model choice, GPU regularization changes, nonzero terrain propagation fix, GJK/margin repair, CPU/GPU manifold equivalence, engine-level condim/restitution/material capability expansion, claims of default-iteration convergence for arbitrary contacts, general energy accuracy, real material/sim-to-real calibration, rendering work, release/merge and product activation. Agreement on a normal symmetric drop cannot close the independently reproduced friction and geometry defects.
