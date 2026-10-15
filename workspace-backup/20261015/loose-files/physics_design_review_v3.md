Initiative: physics-correctness
Stage: design-review
Author: codex
Version: v3
Date: 2026-10-12
Status: accepted
Related Files: collab/physics-correctness__proposal__codex__v3.md; collab/physics-correctness__design-audit__codex__v1.md; collab/physics-correctness__proposal__codex__v2.md; collab/physics-correctness__design-review__codex__v2.md
Owner Summary: Approve with comments the bounded v3 PGS drop-oracle amendment and the companion audit's scoped conclusions. The discrete-crossing formula is correct, but its printed default-height number has a 4.8 nm arithmetic error. Fresh CPU and H200 experiments substantiate the reported implementation discrepancies. This is not implementation approval or general physics certification.

## Identity, scope and verdict

Reviewer session: `/root/physics_design_review_v3`, independent of the author. Read the complete archived `c4_worker_completion/AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md`, the accepted predecessor proposal/review, and both candidate documents. Inspected the current source and production delta, not just historical reflections.

Local candidate: `C:/Users/AG/新建文件夹/robot_physics_correctness`.

- Base: `f771e612beaeea126e26926738097512cd2d809c`.
- HEAD: `40325bb89a8426705d0b1980fd09d4e6864a7686`, verified before and after review.
- v3 proposal SHA256: `AC138D052505DA60D09FC0A900395447489D6DDB31FD1CE14AF3B7360228F2BC`.
- Companion audit SHA256: `7EE09B6706285E696EEDB4BA84D94EFA21B8B984374AD67D6A59ACA4E970F922`.
- Actual local tree: only untracked `physics_audit_probe.py`; no tracked changes. The reviewer did not change candidate files.

**Verdict: Approve with comments.** There is no blocking finding against this bounded design amendment. C1 below is a non-blocking numerical transcription correction: the operative formula, scope, and unchanged acceptance tolerances are sound. Approval authorizes implementing that formula and renaming the diagnostic references as specified; it does not approve the existing implementation, its currently failing v2 drop test, release, merge, product activation, or general CPU/GPU fidelity.

The audit is accepted as a bounded assessment and reproducible defect record. Its explicit deferral of ADMM model/projection changes, GPU regularization changes, geometry/manifold fixes, and material calibration is essential. An overall declaration that the interaction implementation conforms to its intended equations would be false.

## Findings and comments

### C1 — P3, non-blocking: correct the printed discrete crossing height

Location: proposal v3, “Replacement drop oracle,” the default `dt=.0002` reference and penetration numbers.

For the fixture's `G=9.81`, `z0=.45`, and `n=391`, the document's own formula gives:

`z_391 = .45 - 9.81 * .0002**2 * 391 * 392 / 2 = .4199280336 m`.

The corresponding penetration is `.0719664 mm`, not `.0719712 mm`; `.4199280288 m` is 4.8 nm too low. A fresh CpuEngine run independently found the first four contacts at input step 391, input height `.4199280335999999 m`, matching the formula. The incoming velocity was `-.7671420000000047 m/s`.

Consequence: hard-coding the prose value would create a small avoidable oracle error, although it is far below the 1 micrometre gate. Existing trajectory tolerances cannot catch this numerical typo. Direction: compute the crossing from the stated formula and the strict detector threshold, correct the explanatory constants, and retain a direct arithmetic check. Do not fit a reference to a terminal trajectory sample. This does not require changing any production solver parameter or tolerance and does not block this design approval.

### C2 — Non-blocking evidence qualification: a reproduced miss is stronger evidence than an exclusive root-cause claim

The companion audit correctly identifies a reproducible shallow-contact geometry defect. Inspection also confirms `gjk_distance` returns the norm of the last simplex vertex and truncates support-point lists rather than maintaining exact simplex correspondence (`physics/gjk_epa.py:961`, `:986`, `:991`). Those are concrete implementation weaknesses. This review did not perform an isolated repair experiment proving either weakness is the unique cause of every observed miss, and it did not validate the margin erosion construction. Retain the proven misses and the inspected weaknesses as separate facts; treat a complete causal explanation and repair sufficiency as future geometry work. This qualification does not undermine the audit's defect classification or authorize a geometry change now.

## Independent assessment of the v3 oracle

CpuEngine detects contacts using the current state before StepPipeline applies smooth acceleration and integrates the new velocity. The detector requires depth greater than `1e-10 m` (`physics/cpu_engine.py:283`). With zero initial velocity and symmetric straight legs, the contact-free semi-implicit recurrence is therefore exactly the proposed `z_n=z0-g*dt^2*n*(n+1)/2`. Use the smallest n for which `LEG_LENGTH-z_n > 1e-10`; mere equality/tangency must not count.

Once a detected contact is inside slop, the CPU PGS Baumgarte term is zero: `_solve_rows` uses `max(depth-slop,0)`. The inertia repair makes the constraint mobility and applied impulse use the same articulated H. In the ideal hard-contact limit the solver removes the incoming normal motion at that first detected state. There is no attraction toward 5 mm penetration. Positive CFM leaves a tiny inward velocity and finite iterations add a separate error. Consequently the prescribed above-slop stationary equilibrium remains valid while the old drop-to-that-equilibrium assertion is invalid. These statements are compatible, not a relaxation of the physical contract.

Fresh one-second runs used the actual `_run_cpu`/CpuEngine path and independently calculated n and reference height. M is 8.4 kg. The diagnostic's recorded states run from t=0 through T-dt; it executes all T/dt steps but does not return the final state. That endpoint convention does not affect the stated last-window bounds here and should remain explicit in evidence.

| dt (s) | PGS iterations | n | Computed reference z (m) | Tail mean z (m) | Mean error (nm) | Tail range (nm) | Maximum tail abs(vz) (m/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| .0004 | 60 | 196 | .4196973024 | .41969729933116295 | -3.06884 | 1.85843 | 3.74865e-8 |
| .0002 | 60 | 391 | .4199280336 | .4199280300387738 | -3.56123 | 3.06806 | 8.97060e-8 |
| .0002 | 120 | 391 | .4199280336 | .41992802907279836 | -4.52720 | 2.24247 | 6.29753e-8 |
| .0001 | 60 | 782 | .4199663907 | .41996638109864964 | -9.60135 | 2.61439 | 1.29641e-7 |

Every run independently passed the unchanged 1e-6 m reference error, 1e-6 m range, and 1e-5 m/s speed conditions. The proposed symmetric-model CFM bound at dt=.0002 and T=1 is `4.44239964e-9 m`, below 5 nm. It is `8.88644736e-9 m` at doubled dt and `2.22119982e-9 m` at halved dt: “below 5 nm here” applies to the default case. The total measured error can exceed the CFM-only bound, particularly at halved dt, so the proposal correctly keeps finite iteration effects separate. Doubling iterations need not monotonically reduce terminal height error; it changes the accumulated residual.

At the first contact step with 60 iterations, output vz was `+4.739272747467105e-6 m/s` and output z `.4199280345478545 m`. This small nonzero impact residual demonstrates why an ideal discrete crossing is a reference with tolerance, not an assertion of bit-exact arrest. The later tail remains within all frozen bounds.

The unchanged accepted v2 prescribed-state tests also ran freshly as part of `test_articulated_pgs_inertia.py`: **50 passed in 0.63 s**. They include one/four-contact equilibrium, dt scaling and both ERP branches, alongside the articulated impact checks. Thus the amended trajectory oracle does not discard the distinct load-balance obligations.

## Does the interaction design make sense, and does code implement it?

The architectural skeleton is reasonable: separate geometry, smooth dynamics, interaction law, numerical solve, and state integration; use the dual J/J-transpose maps and a common articulated inertia for response. The current implementation only partially satisfies that design. Choosing PGS versus ADMM currently changes both numerical algorithm and contact law; treating the selector as an algorithm-only comparison obscures what is being validated. Material-response accuracy and agreement between engines are also distinct questions.

The companion audit's following conclusions were checked against current source and, where specified, fresh runtime evidence:

| Claim | Verification and disposition |
|---|---|
| CPU articulated inertia mismatch | The base used independent-body mobility; current `PGSContactSolver.solve` forms `Jg H^-1 Jg.T` using the supplied L, while `VelocityLevelAdapter` applies impulses using that same cached L. `body_velocity_jacobians` includes fixed descendants and joint columns. Legacy velocity stepping delegates to the common adapter. Fresh 50-test run passes the simple and offset impact fixtures and equilibrium fixtures. This supports the repair direction; full code approval is deferred. |
| ADMM fails its declared convex cone QP | CPU module explicitly declares the constrained quadratic objective. `_project_all` clamps normal first, then tangents with normal fixed. Fresh solver gives `[.8999999999994869,.8999999999996376,0]` with objective `-1.7999999999996374`, whereas the feasible exact optimum `[1.35,1.35,0]` has `-2.0250000000000004`. For Q=(10/9)I this follows independently by projecting Q-inverse rhs onto the mu=1 cone. Fresh projection `[1,2,0] -> [1,1,0]` differs from the Euclidean projection `[1.5,1.5,0]`. CPU defect is experimentally proven; GPU `_cone_project_all` has the identical defective normal-first projection by source inspection, not by a separate GPU QP-optimality experiment in this review. |
| GPU PGS regularization is only in its denominator | Current production builder is `batched_build_W_joint_space` in `crba_kernels.py:725`, invoked by GpuEngine. It preserves physical W and writes CFM/friction R only to W_diag. Jacobi and colored kernels evaluate residuals with W. Fresh real-H200 results are tangent impulses `-.5714285373687744` and `-.5714285969734192`, respectively, versus soft-model `-.5657142857142857`; these match hard-model `-2/3.5`. This is proven failure to implement the stated compliant residual, not merely insufficient iteration count. The same structural issue also affects normal CFM; the fresh numerical counterexample here isolates tangential compliance. |
| GPU ADMM derives R from already regularized diagonal | GpuEngine passes W_diag into `batched_admm_solve`; `admm_kernels.py:258` multiplies it by the ADMM ratio. Fresh H200 engine output has physical tangential A=3.450249671936035 and R=.1834263950586319, giving R/A=.05316322296857834 rather than .052631578947368474. Proven parameter/data-flow mismatch. |
| Nonzero flat terrain height is lost | `StaticRobotData.from_merged` omits the terrain height and retains contact_ground_z=0; GpuEngine only rejects non-flat terrain. Fresh elevated-plane case requested .5 m and put the sphere at .599 m: CPU detected one contact; GPU static height was 0 and GPU detected zero. Proven current implementation defect, independent of the supported-terrain-type restriction. |
| Shallow geometry misses | Fresh query sweep reproduced box-hull misses at .01/.1/1 mm, detection at 3 mm; box-cylinder detection at .01/.1 mm, miss at 1 mm, detection at 3 mm. These exact fixtures establish a geometry defect. They do not establish that every shallow cylinder contact fails. See C2 about causal attribution. |
| CPU/GPU ground manifold difference | CPU `halfspace_convex_query` enumerates penetrating vertices. GPU `convexhull_ground_manifold` keeps at most four deepest. This difference is proven by source, while resulting support-torque/load discrepancies remain a model/coverage risk not numerically quantified in this review. |
| condim/restitution and material interface gaps | CPU engine construction fixes condim=3, global/terrain mu, and default restitution=0; low-level PGS row support is broader. CPU ADMM fills only linear normal/tangent Jacobian rows even for larger condim; GPU uses three rows. No implemented InterfaceMaterial was found. These are current routing/implementation limits, not evidence that every exposed engine supports low-level fields. |
| Pipeline reuse and energy claims | StepPipeline builds H/cache on the contact path; acceleration-level ADMM recomputes CRBA and adapter ABA. This contradicts a literal once-per-step reuse goal, but duplication alone does not prove wrong numerical results. Energy's full-trajectory and datum-invariant metrics are materially better checks; this review inspected them and read prior raw logs but did not rerun the full two-second energy study. |

The PGS box bounds imply a square tangential admissible region, unlike the ADMM circular bound. The ERP switch at 1 uses different units on either side. These are directly visible design choices needing explicit contracts, not automatically bugs to change in this bounded repair. Likewise, whether an associated convex cone QP is the intended friction law is a future model decision; blindly replacing the projection would select different coupled normal behavior. The audit correctly avoids authorizing that change now.

The historical claims about material fidelity, MuJoCo superiority, constitutive-parameter mappings, and chaos explaining every discrepancy have not been proven here. A smaller penetration or a passing symmetric drop cannot supply that evidence. The audit's proposed separation of bulk constitutive behavior, interface law and geometry is a sensible future direction, but remains a design direction, not an implemented capability.

## Commands, provenance and limitations

Execution host: SSH `tileops-dev`, runtime checkout `/data2/shared/ang.gao/robot_sim_physics_correctness`; interpreter `/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python`. Remote HEAD is the base with uncommitted author implementation/tests, as disclosed. The reviewer compared normalized text for 64 physics/validation files: 59 exactly matched the local candidate, five differed. Downloaded the five to the external `physics_v3_review_scratch` directory and inspected each diff: PGS and PGS-SI differences are line wrapping only; physics_metrics and dynamics tests are formatting only; contact tests differ in import ordering/spacing only. No production numeric difference was found. The coordinator confirmed pre-freeze formatting as the cause. The checked contract-audit script matches the frozen local content. This provenance is source equivalence for the inspected paths, not a claim that remote Git HEAD is the candidate commit.

Executed:

- Local identity, tree, SHA256 and diff checks, including `git diff --check BASE HEAD`: no whitespace errors.
- SSH provenance and reads of `/tmp/physics_solver_contracts.log` and `/tmp/physics_audit_final.log`. Those pre-existing files were inspected as author evidence, never counted as fresh execution.
- Fresh `CUDA_VISIBLE_DEVICES=3 .../python -m tests.validation.rigid_body.audit_solver_contracts`: completed, reproducing all reported QP/projection, GPU regularization, and elevated-terrain outputs. Warp 1.12.0, CUDA toolkit 12.9, driver 13.2, NVIDIA H200. Standard Warp compiled-kernel cache was used; numerical launches and returned arrays were new.
- Fresh SSH-stdin scratch scripts, with explicit runtime cwd/sys.path: independent crossing calculation and four one-second PGS trajectories; first-contact engine instrumentation; eight shallow-geometry cases. Scripts did not change tracked implementation files.
- Fresh `.../python -m pytest -q tests/integration/contact_pipeline/test_articulated_pgs_inertia.py`: 50 passed in 0.63 s.

Some initial read-only searches targeted nonexistent filenames (`REVIEW_PRINCIPLES.md` outside collab, `warp/model.py`, `halfspace_collision.py`); corrected locations were found and inspected. These were path/search errors, not repository test or runtime failures.

No fresh full energy, full MuJoCo/Pinocchio suite, exhaustive GPU suite, general friction/rotation/contact-manifold study, or real-material experiment was performed by this reviewer. The coordinator's separately reported 439-pass/one-known-oracle-failure run is not promoted to reviewer execution evidence. Normal Python/pytest/Warp caches may have been touched remotely. Local scratch downloads and this report are outside the frozen candidate; final candidate identity/digests/tree are unchanged.

## Approved boundary and remaining gates

Approved: the fixture-specific discrete first-penetration reference, unchanged PGS numerical tolerances, retention of prescribed-state CFM/load-balance tests and ADMM coupled equilibrium oracle, and the companion audit's bounded diagnosis/deferral. Retain the failed v2-oracle evidence and correct the printed constant as C1's disposition.

The current tests and `audit_physics.py` still contain the v2 settling expression; this is known pending work rather than evidence that v3 is already implemented. Fresh independent code review must examine the final oracle helper, callers, reference labels, production inertia repair and complete promised regression evidence at an exact frozen implementation revision.

ADMM objective/projection policy, GPU regularization equations, nonzero terrain propagation, GJK repair, manifold policy, material interfaces, real calibration, broader product capability, and release/merge/activation all remain outside this approval. The confirmed existing defects must remain visible and must not be converted into passing correctness claims through this CPU-only repair.
