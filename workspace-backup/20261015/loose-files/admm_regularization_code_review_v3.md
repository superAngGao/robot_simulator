Initiative: admm-regularization
Stage: code-review
Author: codex
Version: v3
Date: 2026-10-15
Status: accepted
Related Files: collab/admm-regularization__proposal__codex__v2.md; collab/admm-regularization__design-review__codex__v2.md; collab/admm-regularization__code-review__codex__v1.md; collab/admm-regularization__code-review__codex__v2.md; collab/admm-regularization__implementation-note__codex__v3.md; docs/admm_regularization.md
Owner Summary: Approve for the bounded optional ADMM regularization and normal-only verification scope at 52f14d5911468f4824f4e254ceb20e58973e3352. Fresh independent execution: 216 tests pass on the real H200; original F1/F2 controls still pass; F3 overflow/cancellation and F4 width-branch cases are independently closed with additional CPU/MuJoCo/CUDA assertions. No unresolved blocking finding. Friction projection, material truth, general simulator correctness, activation and main merge remain outside this approval.

## Verdict and provenance

**Approve.** No substantive new finding or unresolved blocker in the reviewed bounded scope. This verdict follows actual code/state-flow inspection and new runtime evidence, not predecessor approvals or author test summaries.

Reviewer session: `/root/admm_regularization_code_review_v3`, fresh independent review task, with no implementation-conversation fork. This reviewer did not author or edit the candidate. Read the current owner instructions in full from `C:/Users/AG/新建文件夹/c4_worker_completion/AGENTS.md`, that checkout's `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md`. The candidate contains an older collaboration README; the current owner-directed protocol governs this review. Initial attempts to find REVIEW_PRINCIPLES in the candidate failed because it is absent there; the specified current protocol files were then read successfully in UTF-8.

- Local checkout: `C:/Users/AG/新建文件夹/robot_physics_correctness`.
- Remote checkout: `tileops-dev:/data2/shared/ang.gao/robot_sim_physics_correctness`.
- Branch on both: `physics/correctness-audit-20261012`.
- Original implementation base: `6c66f4b12c62686626db7069953a838dacd61104`.
- Previous rejected candidate: `5f51ffc8f9dac32105b41bc88874832b51a89348`.
- Exact reviewed and executed HEAD on both: **`52f14d5911468f4824f4e254ceb20e58973e3352`**.
- Both working trees clean before and after review. No candidate tracked files modified.
- Inspected original-base production delta and current-round delta, accepted design/review, both archived code reviews, relevant tests, scoped documentation and progress/reflection claims. The original-base diff has 22 files, 2238 insertions and 31 deletions; the current-round diff has 10 files, 483 insertions and 10 deletions.
- Runtime: `/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python`, Python 3.12.13, NumPy 2.5.1, SciPy 1.18.0, MuJoCo 3.15.0, Warp 1.12.0. `CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1`; visible actual NVIDIA H200, CUDA toolkit 12.9 / driver 13.2.

Scratch, logs and this report are outside the candidate. Python/pytest/Warp caches may be updated by normal execution. Kernels loaded from their normal compilation cache but executed anew; this is no performance measurement. No mock substitutes were used for the CPU/MuJoCo/CUDA evidence.

## Findings and independent closure

### F1 — Closed; no regression in public GPU admission

The original unchanged v1 script again rejects q=NaN, qdot=Inf, tau=NaN, q=1e100, explicit zero dt, step_n invalid tau/zero dt, reset NaN and partial reset NaN. Source inspection verifies that step admission runs before external q/qdot uploads, and step_n admission precedes its first substep. Reset entry points check float32 conversion before reset mutation. Only None selects the default in the opt-in mode; the legacy per_row behavior is explicitly retained.

Fresh checked-in tests additionally assert unchanged q/qdot/tau, contacts, force/slack/dual caches, R, default q, run identity and publication counters, then successful recovery. They include two environments, NaN/Inf/conversion overflow, nonpositive/nonfinite/underflow dt, and invalid model parameters concurrent with otherwise valid external-state upload. This closes the finite-input admission contract at the public boundary; it is not protection against arbitrary private device-buffer changes or a proof that every admitted finite state has representable downstream dynamics.

### F2 — Closed; CPU validation and coefficient execution remain consistent

The unchanged original solref=(.02,1e155) reproduction executes without the former Python square OverflowError. Observed b=105.26315789473685 and k=2.7700831024930746e-307, with finite output. CPU assembly consumes the same normalized `reference_coefficients` result used for admission. The fresh suite covers Python tuple and NumPy array inputs at construction and mutation, and rejects 1e160 before contact/cache/diagnostic mutation. No extreme-force convergence claim follows from these coefficient checks.

### F3 — Closed for native R admission and ordinary-scale conformance

The current CPU and Warp interpolators explicitly return endpoints, retain the upper branch's complementary weight directly, interpolate weighted endpoints, and clamp to the endpoint interval. This removes both the old endpoint cancellation and the interior cancellation caused by constructing 1-small first. The native-domain check uses converted endpoints/weights, includes pair sums and tangent division, and reserves the documented 16 native epsilons at normal/overflow limits. The conservative restriction is explicit in the API document; it does not silently widen the numerical budget or claim material admissibility.

The original mass1e-35, d=1e-4 GPU case rejects. Fresh tests cover the original positive/negative adjacent masses, body-world/body-body, condim1/3 and tangent lower/upper boundaries.

The reviewer-owned v3 probe reproduces the precise decreasing-curve masses and widths from v2, with repaired assertions:

- GPU two-ball mass **4.89730620518322e-35**, solimp **(.9999,.00012,width,.5,2)**, width **.001/.010001**, condim **1/3**: ValueError before changed state, contacts, force/slack/dual caches, R, lambdas, previous-active count or publication counters. These checks start with valid populated state/cache, and perform a successful same-engine recovery after restoring valid parameters.
- GPU adjacent mass **5e-35**, same widths/dimensions: three real contacts, including ground and body-body, all finite R and agreement with an independent 80-digit Decimal polynomial oracle at actual GPU depth. Endpoint normal R is approximately **1.6664667640e38** for each ground contact and **3.3329335281e38** for the pair; interior pair R is **3.3323834691e38**.
- CPU two-ball mass **8.556863841944613e-305**, solimp **(.9999,.00013,width,.5,2)**, width **.001/.010001**, condim **1/3**: ValueError before frame/cache/diagnostic mutation, followed by successful same-solver recovery with ordinary impedance.
- CPU adjacent mass **1e-304**: finite R **1.5382615384615384e308** at the endpoint and **1.5380249902045627e308** at the interior, matching the independent Decimal oracle, for both dimensions.

For ordinary 2 kg decreasing curves, independent checks compare production CPU with actual MuJoCo, then production CUDA with both actual MuJoCo and the Decimal oracle at actual CUDA contact depth. Both condim1 and condim3 pass the original budget `3e-5*abs(R) + 1e-6` (inverse-mass scale .5). Representative values:

| width | expected R at GPU depth | actual H200 R | absolute error | original budget |
| --- | ---: | ---: | ---: | ---: |
| .001 | 4166.166666666667 | 4166.1669921875 | .000325521 | .124986 |
| .010001 | 4165.469731177287 | 4165.4697265625 | .000004615 | .124965092 |
| .0101 | 1581.6570386009814 | 1581.6630859375 | .006047337 | .047450711 |
| .02 | .49997958327617237 | .4999796152114868 | 3.1935e-8 | 1.59994e-5 |

The .0101 interior case was not added to the checked-in GPU parameter list but was independently rerun here; it now passes too. The additional probe also passes reversed endpoint direction, zero width, linear power=1, and high-power/small-midpoint controls. This is assembly evidence, not approval of extreme-scale QP convergence, A+R representability at every admitted R edge, or friction forces.

### F4 — Closed; the host width branch survives float32 conversion

Normalization canonicalizes only original widths <=1e-15 to exact zero. The Warp implementation branches on that canonical zero; a host width immediately above the threshold stays on the ordinary curve. This preserves the declared discontinuous model decision before conversion without moving the cutoff.

The independent v3 probe checks below, equal, nextafter-above and 1.000001e-15 widths, plus zero-width control, with actual MuJoCo, CPU and H200, both unequal endpoint directions, and both condim1/3:

| endpoints | below/equal/zero width: CPU/MuJoCo/H200 R | nextafter-above: CPU/MuJoCo R | nextafter-above: H200 R |
| --- | ---: | ---: | ---: |
| (.2,.8) | .5 | .125 | .1249999925494194 |
| (.8,.2) | .5 | 2 | 2 |

The former factor-four ordinary-mass discrepancy is gone. Checked-in reference/GPU threshold regressions also pass unchanged.

## Call, data and state-flow audit

CPU inspection covered policy construction/read-only properties; parameter and contact validation before frame/cache writes; q0 CRBA and body-frame COM Jacobian construction; Cholesky mobility weights; simple-slider classification and ambiguous nondiagonal inertia rejection; restoration of current FK in finally; raw current-state A; selected R; normalized reference coefficients; rhs, ADMM warmstart and diagnostic-copy publication. The body-velocity helper's linear/angular body-frame convention supports the COM shift used by the weight calculation. Mutable CPU mass changes recompute weights; GPU owns a static uploaded snapshot and documents rebuilding after model edits.

GPU inspection covered construction and scalar/body-weight admission, public step/reset entry points, current/reference velocity separation, W assembly, dispatch, all normal-only row and column masks, rho on otherwise zero rows, rhs, projection, warmstart load/save, inactive/empty slots and output impulses. `batched_build_W_joint_space` constructs raw W separately from the PGS-adjusted W_diag. ADMM now consumes raw W; PGS continues consuming its existing adjusted diagonal. In the ADMM route, the W-builder's positional correction is disabled and the solver forms its own reference rhs. The original multi-environment stale-cache and nonzero normal/tangent velocity scalar-oracle checks passed for both policies.

No new runtime MuJoCo dependency, material layer, default-policy change, public strategy-switch cache path, unrelated renderer change or main merge was introduced. The new host helper owns common validation/constants; native CPU/Warp arithmetic is separately implemented as required by the backend and cross-checked. The same CPU coefficient implementation is used for validation and execution. R does not consume rho. The existing friction projector and iteration/cache identity policy remain visibly outside the accepted scope.

## Verification commands and results

Fresh complete focused suite at the exact reviewed HEAD: **216 passed, 1 warning in 69.15 s**. The warning is the existing class-scoped pytest fixture deprecation in `tests/unit/solvers/test_mujoco_qp.py`; no tests skipped in this run. This includes the actual MuJoCo, CPU analytic/domain, real CUDA, four full trajectories and relevant existing regressions.

From the remote checkout:

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

Fresh suite log: `C:/Users/AG/新建文件夹/admm_regularization_code_review_v3_tests.log`.

The simple-slider reference/analytic force, neighbouring slider classifications, articulated weights, frozen coupled normals, model mutation, rho independence, read-only selections, empty-solve diagnostic and default-route regressions all pass. The ordinary checked-in GPU cases report maximum R relative error **6.615959164455971e-7** and maximum normal impulse error **6.48671395486744e-8 N s**.

All four full CPU trajectories ran again:

| fixture / dt | maximum q error | maximum velocity error | maximum step impulse error | maximum penetration error |
| --- | ---: | ---: | ---: | ---: |
| sphere / .001 | 1.3878e-17 | 1.1102e-15 | 4.9960e-16 | 1.3878e-17 |
| sphere / .0005 | 1.3878e-17 | 1.3323e-15 | 2.4980e-16 | 1.3878e-17 |
| quadruped / .001 | 4.4409e-16 | 3.0198e-14 | 9.7838e-16 | 1.1449e-16 |
| quadruped / .0005 | 6.6613e-16 | 3.1974e-14 | 7.8063e-16 | 1.4225e-16 |

No contact-count discrepancies. Maximum cumulative impulse error **2.0990e-15 N s**. Sphere peak penetration remains **.01236544039384005 m** and **.0126297776807347 m**: matching the reference soft-contact model does not establish material truth. These measurements close only the accepted matched-fixture trajectory gate.

Independent probe commands used this PowerShell pattern for both the unchanged v1 probe and new v3 probe:

```powershell
Get-Content -Raw -Encoding UTF8 'C:/Users/AG/新建文件夹/SCRIPT.py' |
  ssh tileops-dev 'CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 /data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python - 2>&1' |
  Tee-Object -FilePath 'C:/Users/AG/新建文件夹/OUTPUT.log'
```

- `SCRIPT=admm_regularization_code_review_v1_probe`, `OUTPUT=admm_regularization_code_review_v3_original_probe`: exit 0, original script unchanged. This script prints some admission outcomes; the newly executed checked-in assertions supply stronger atomicity/recovery checks for F1/F2.
- `SCRIPT=admm_regularization_code_review_v3_probe`, `OUTPUT=admm_regularization_code_review_v3_probe`: exit 0; all 40 ordinary CPU/MuJoCo/CUDA parameter/dimension combinations and 16 CPU/GPU upper-boundary combinations pass; final PASS printed. Independent high-precision R expectations do not call the production impedance/weight helper.

V2's scripts assert historical failure behavior, so they were read and preserved, not misreported as current passing tests. Their exact principal configurations were reconstructed in the new assertion probe. SHA256 hashes checked against the archived report:

| artifact under C:/Users/AG/新建文件夹/ | SHA256 |
| --- | --- |
| admm_regularization_code_review_v1_probe.py | 8DC93C827A99F5742E65E5C9E3716BB9B02BBA374608C7C46FF5BAC44A9CE3A1 |
| admm_regularization_code_review_v2_boundary_repro.py | 708C0CF86E89DCE5C4C7216EA09BCE0FC84A534F531F0C1B5FC6CBCC9AAE5583 |
| admm_regularization_code_review_v2_decreasing_probe.py | 40E5476719A51A61817A470560EC1ECC70159CDD1B938BEEB50C0581F5E28025 |
| admm_regularization_code_review_v2_cpu_edge_probe.py | BA93D33B76E6A5C51893669B8F8E443812718B27870AC571C14BC1FF7A31B6E1 |
| admm_regularization_code_review_v2_width_probe.py | B8A0D45E27F4480EA074136133D317E987270957F5C6181BF1DB87CB6C7CC973 |
| admm_regularization_code_review_v3_probe.py | 57E70A054D75CA9C0D45EEA5C4EA83FAE54630AB326E9D724C7DDACA5A9613F0 |

Remote `python -m ruff check` passed for the ten changed production/test Python files: admm_kernels.py, cpu_engine.py, gpu_engine.py, admm_qp.py, contact_regularization.py, and the five changed/added test files. `git diff --check 6c66f4b12c62686626db7069953a838dacd61104 HEAD -- physics tests docs` passed. Whole-diff `git diff --check` reports only the already identified trailing blank line at archived code-review v2 line192. That historical report remains verbatim under the recording rule; this is not a production/test whitespace failure or a blocker.

Other executed commands included local/remote git status, HEAD and branch checks; original/current-round stats and diffs; rg searches and UTF-8 reads; and Get-FileHash. No execution failure was reclassified as a pass. The only failed read attempts were the absent candidate REVIEW_PRINCIPLES paths described above.

## Coverage gaps, residual risks and bounded approval

No mandatory evidence gap remains for this implementation scope. The focused suite and independent probes are proportional to the change. No full repository suite, exhaustive parameter sweep, arbitrary concurrent/multi-device use or unrelated CUDA publication lifecycle audit was run or claimed. Some useful expanded direction/dimension/interior controls are reviewer-owned evidence rather than permanent checked-in regressions; the checked-in tests already cover the actual reported failure mechanisms.

Approval binds only **52f14d5911468f4824f4e254ceb20e58973e3352** and the accepted optional CPU/real-Warp per_row/mujoco R implementation, default per_row, raw-W correction, reference/simple-slider weights, documented numerical admission, condim1 engine plumbing, analytic/reference assembly, bounded normal-only snapshots, and the four matched CPU trajectories. F1/F2/F3/F4 are closed at this revision. This does not alter the historical Request changes verdicts for v1/v2.

Still excluded: condim3 cone/projector and friction-force correctness, friction trajectories, default-30-iteration accuracy, rho/cache identity redesign, extreme-scale force convergence, arbitrary/high-speed or moving-contact correctness, material calibration/truth, full MuJoCo compatibility, performance, private device-buffer mutation, renderer integration, product activation, publication, main merge and overall simulator correctness. The known condim3 projection defect remains unresolved. A later stage requires its own scope and review.

Recording this report unchanged and its execution evidence in a separate evidence-only commit is permitted by the owner protocol; such a commit must not change code, tests, requirements, verdict or approved scope. Any subsequent operative change requires review of that delta.
