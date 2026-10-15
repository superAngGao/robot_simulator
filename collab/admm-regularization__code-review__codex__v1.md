Initiative: admm-regularization
Stage: code-review
Author: codex
Version: v1
Date: 2026-10-15
Status: in_review
Related Files: collab/admm-regularization__proposal__codex__v2.md; collab/admm-regularization__design-review__codex__v2.md; docs/admm_regularization.md; admm_regularization_code_review_v1_probe.py
Owner Summary: Request changes. The 161-test reference/CPU/real-CUDA suite passes independently, including all four bounded trajectories. Three reproducible validation/numerical-boundary gaps remain: GPU invalid state admission, CPU coefficient validation differing from execution, and native GPU R overflow. Ordinary assembly and condim1 masking checks pass; this is not approval of the frozen candidate.

## Verdict and provenance

**Request changes.** Three blocking findings below concern explicit v2 input/domain requirements and design-review C2. They do not demand extreme-scale force convergence or expansion into the excluded friction work.

Reviewer session: `/root/admm_regularization_code_review_v1`, independent review-only agent; did not author or edit the candidate. Read the current owner protocol in full at `C:/Users/AG/新建文件夹/c4_worker_completion/AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md`. The initially attempted root `REVIEW_PRINCIPLES.md` did not exist; the specified collab file was then read completely, with UTF-8 decoding. Historical protocol in the candidate was not substituted for the owner-directed workflow.

- Local repository: `C:/Users/AG/新建文件夹/robot_physics_correctness`.
- Branch: `physics/correctness-audit-20261012`.
- Base: `6c66f4b12c62686626db7069953a838dacd61104`.
- HEAD: `554133373595206ea2153a7f3cadd8f365451eec`.
- Actual base..HEAD diff: 15 files, 1174 additions, 24 deletions. Inspected production changes, added/changed tests, documentation, implementation note, test-log claims, progress/reflection edits, accepted v2 design and design-review comments.
- Local and remote tracked trees clean before and after execution; both HEADs independently checked at the exact candidate. `git diff --check 6c66f4b..HEAD` passed.
- Remote: `tileops-dev:/data2/shared/ang.gao/robot_sim_physics_correctness`.
- Interpreter: `/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python`.
- Observed Python 3.12.13, NumPy 2.5.1, SciPy 1.18.0, MuJoCo 3.15.0, Warp 1.12.0; `CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1`; actual visible device NVIDIA H200, CUDA toolkit 12.9 / driver 13.2.

No tracked file was changed locally or remotely. Reviewer-owned probe/report/log are outside the checkout. Python import caches, pytest caches and normal Warp kernel caches are runtime side effects; the tests and probes themselves executed anew. No performance inference is drawn from cached compilation.

## Findings

### F1 — Medium, blocking: GPU mujoco mode does not enforce the promised finite-state boundary

Locations: `physics/gpu_engine.py:1034-1051` (`step` uploads), `1386-1405` (`_step_physics` validates only model parameters, then uploads tau); related public entry points `reset:943`, `reset_envs:965`, `step_n:1017`.

Trigger: construct an ordinary ball engine with `solver='admm', regularization='mujoco', contact_condim=1`, take a valid contacting step, then call public `step` with any of: q height NaN; normal velocity Inf; tau normal component NaN; or finite float64 q height `1e100` that overflows on float32 conversion. Every case returned without rejecting input, produced nonfinite integrated state, and changed the ADMM previous-force cache. This was a real H200 engine execution, not a direct kernel call. `step_n(tau=NaN)` similarly returned nonfinite state. `reset(q0=NaN)` and `reset_envs(...,q0=NaN)` accept invalid state and their subsequent public step also proceeds to nonfinite output. Those reset observations establish alternate admission paths; the mandatory solve boundary is still missing even if reset is treated separately from solve.

Related explicit-parameter boundary: `step(dt=0)` and `step_n(dt=0)` silently advance by the configured default because `dt = dt or self._dt` removes zero before the positive-dt validator sees it. This differs from the CPU mujoco solve, which rejects zero. None of these repros requires manipulating private GPU buffers. Only construction/configuration of the opt-in policy is needed.

Consequence: the new mode's specified rejection-before-contact/cache-mutation property does not hold. Invalid upstream state can contaminate contacts, state and warmstart history. Finite host inputs are insufficient if the converted representation is nonfinite.

Why existing tests miss it: the new GPU negative test calls private `_step_physics()` with invalid `_admm_solref`; it does not exercise public state/tau uploads or alternate public entries. Host native-range validation covers solref/solimp/dt/impratio only. The CPU negative tests cannot prove the GPU boundary.

Direction: validate the effective state, forces and relevant native representation for the opt-in mode before they can influence contact or cache mutation; preserve meaningful behavior through all public admission paths. Distinguish `dt is None` from an explicit zero. Avoid solving this only inside one `step` overload while `step_n` or reset-supplied state bypasses it. Validation need not redesign cache identity or rho.

Required repair proof: reproduce each public case with rejection; assert state/contact/cache/publication counters remain appropriately unchanged on the rejected solve, then execute a valid step successfully. Cover float64-to-float32 overflow as well as already-nonfinite values; `dt=None`, zero, negative, NaN, Inf; reset/partial reset and batched entries. Preserve the existing per_row compatibility scope explicitly.

### F2 — Medium, blocking: CPU validates one coefficient expression and executes another after mutating contacts

Locations: `physics/solvers/contact_regularization.py:47-62`; `physics/solvers/admm_qp.py:148` and `240-246`.

Trigger: after creating an ordinary mujoco solver, set its public mutable `solref=(.02, 1e155)` and solve at dt=.002. `reference_coefficients` accepts the parameters and returns finite positive `b=105.26315789473685`, `k=2.7700831024930746e-307` with the default d_width=.95. The CPU solve then evaluates Python `dampratio**2`, raises `OverflowError (34, 'Numerical result out of range')`, and has already replaced the contact tangent frame. The probe marks tangent1 with 7 before the call and observes that it changed.

This is a supported mutable solver field, not mutation of the read-only regularization/impratio selection. The existing invalid-parameter tests themselves exercise post-construction public solref mutation. Constructor validation with an invented dt=1 can reject this value initially, but does not make the solve-time guard correct or immutable.

Consequence: the documented native-domain validator does not validate the actual executed expression and fails its pre-mutation purpose. No extreme-force accuracy claim is needed: the implementation must either use a representable evaluated coefficient consistently or reject the domain before mutation.

Why tests miss it: existing extreme solref tests only choose values the helper already rejects. They do not exercise finite helper results whose differently grouped/retyped execution overflows. `normalize_parameters` returns normalized float64 reference data, but the solve discards that reference tuple and recomputes from `self.solref`.

Direction: have validation and CPU assembly consume one coefficient implementation and normalized type; alternatively explicitly validate the exact actual expression before any contact changes. Avoid a second independently drifting numerical definition. Consider constructor validation separately from actual dt-dependent solvability.

Required repair proof: retain this exact repro and check either finite consistently computed coefficients or documented ValueError before frame/cache/diagnostic mutation; include Python floats and NumPy parameter arrays/scalars, representable near-boundary values and rejected out-of-domain values. No requirement to converge the resulting extremely scaled QP is imposed.

### F3 — Medium, blocking: native GPU R can overflow after all host range checks pass

Locations: `physics/gpu_engine.py:295`, `1392-1398`; `physics/backends/warp/admm_kernels.py:292-302`.

Trigger: a free sphere of finite mass `1e-35` kg, ordinary sphere inertia produced by the existing `_ball_model` fixture, mujoco condim1 engine, dt=.002, and normalized `solimp=(1e-4,1e-4,.001,.5,2)`. Constructor reference weights are finite in float32. The public `reset` and `step` complete, but the production `admm_R_diag` contains Inf. Probe output: `weights_finite True R_finite False R_max inf`. All model/parameter checks pass before this occurs.

The native expression `(1-d)/d*w` overflows even though each admitted input is finite. The example deliberately probes representability, not plausible material scale or force convergence. The design-review C2 explicitly calls for checking converted/native assembly and stating unsupported domains. Current docs only describe rejection of unrepresentable reference-acceleration coefficients; the R domain is not rejected or constrained.

Consequence: an admitted model creates nonfinite physical regularization and proceeds into factorization/cache updates. Checking weights individually cannot establish that their sum, impedance scaling, or tangent division is representable.

Why tests miss it: impratio tests stress the small-ratio floor with O(1) masses, and coefficient tests exercise solref. They do not test the derived native R range. Existing float32 absolute tolerances also are not a proof of positive representability across the full admitted parameter domain.

Direction: define and enforce a bounded native R domain, including summed body mobility, normalized impedance scaling and tangent ratio. Fail before invalid assembly mutates solver state, or use a demonstrably safe representation with equivalent semantics. Apply corresponding reasoning to CPU derived values rather than claiming that finite operands guarantee finite R. This is compatible with explicitly excluding extreme-scale convergence.

Required repair proof: the exact real-CUDA repro must reject clearly before cache mutation or produce finite correctly represented R; test near the chosen boundary, both body-world and summed body-body weights, both condim1 and condim3, overflow and underflow cases. Document any intentional native-domain restriction.

## Verification and positive evidence

The complete nine-file suite claimed by the implementation note was rerun independently on the clean frozen remote checkout:

```sh
cd /data2/shared/ang.gao/robot_sim_physics_correctness
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

Observed **161 passed, 1 warning in 68.62s**. The warning is the existing class-scoped fixture deprecation in `test_mujoco_qp.py`, not a candidate failure.

Inspected the CPU solve from validation through frame mutation, current/reference FK, CRBA/Jacobians, R/rhs, factorization, warmstart and diagnostics; body-COM mobility and simple-slider exception; CPU engine contact dimension propagation; GPU construction/static ownership, dispatch and raw W use; ADMM inactive/tangent masking, Cholesky extent, warmstart load/store, projection and output. Read shared body-velocity Jacobian semantics to verify the COM shift's frame convention. Actual source references were inspected at the pinned [MuJoCo setconst](https://raw.githubusercontent.com/google-deepmind/mujoco/3.15.0/src/engine/engine_setconst.c) and [constraint assembly](https://raw.githubusercontent.com/google-deepmind/mujoco/3.15.0/src/engine/engine_core_constraint.c); runtime MuJoCo comparisons provide the decisive numerical evidence.

Positive observations:

- Simple-slider, neighbouring non-simple classifications, articulated weights, actual efc_R, coupled normal forces, model mutation, defensive diagnostic copy and empty-solve checks pass in the rerun.
- Both GPU policies use raw W; the PGS adjusted diagonal is no longer passed into ADMM. The PGS implementation is unchanged.
- Independent additional CPU NaN/Inf state/acceleration/depth and invalid-body probes reject before contact tangent/cache/diagnostic mutation.
- Independent two-environment real-CUDA test seeds f/s/u caches with 7, then steps normal-only contacts. Inactive and tangent slots in R, rhs, output and saved f/s/u are exactly zero. Combined with inspection and the existing row/column tests, this supports design comment C1.
- Independent two-environment nonzero normal and tangent velocity scalar-oracle test exercises both policies through public `step`; normal rhs and impulse agree with analytic mass/solref/gravity equations. This avoids relying solely on zero-velocity author fixtures or a production-computed rhs as the oracle.
- R and impulse maxima in the existing GPU run reproduce the author's reported scale: max R relative error 3.3250521e-7 and max normal impulse error 6.4867140e-8 N s.

All four real CPU trajectory tests executed anew and agree with the author measurements:

| Fixture / dt | max q error | max velocity error | max per-step impulse error | max penetration error |
| --- | ---: | ---: | ---: | ---: |
| sphere / .001 | 1.3878e-17 | 9.9920e-16 | 4.9960e-16 | 1.3878e-17 |
| sphere / .0005 | 1.3878e-17 | 1.3323e-15 | 2.4980e-16 | 1.3878e-17 |
| quadruped / .001 | 4.4409e-16 | 3.1086e-14 | 9.7491e-16 | 1.1449e-16 |
| quadruped / .0005 | 6.6613e-16 | 3.2863e-14 | 7.8236e-16 | 1.1102e-16 |

No contact-count discrepancies; max cumulative impulse error 2.0123e-15 N s. Sphere penetration is .01236544 m and .01262978 m at the two steps, as documented. These are legitimate matched-equation trajectory results for the stated fixtures and support C3; they do not establish broad collision/integration or material accuracy. Candidate-wide approval remains blocked by F1-F3.

## Reviewer probe and execution record

Probe: `C:/Users/AG/新建文件夹/admm_regularization_code_review_v1_probe.py`.
SHA256: `8DC93C827A99F5742E65E5C9E3716BB9B02BBA374608C7C46FF5BAC44A9CE3A1`.
Raw final probe output: `C:/Users/AG/新建文件夹/admm_regularization_code_review_v1_probe.log`.

```powershell
Get-Content -Raw -Encoding UTF8 'C:/Users/AG/新建文件夹/admm_regularization_code_review_v1_probe.py' |
  ssh tileops-dev 'CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 /data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python - 2>&1' |
  Tee-Object -FilePath 'C:/Users/AG/新建文件夹/admm_regularization_code_review_v1_probe.log'
```

Final remote exit 0. The probe intentionally catches and prints bad-boundary behavior; exit 0 means the reproductions and positive assertions ran, not that the candidate passed review. It reports CPU OverflowError/contact mutation, all GPU invalid admissions, and Inf native R, alongside positive cache/scalar-oracle results.

Other commands: local/remote `git status --porcelain`, `git rev-parse HEAD`, `git branch --show-current`; local `git diff --stat`, `git diff --name-only`, per-file actual `git diff 6c66f4b..HEAD`, `git diff --check`, `rg` searches and UTF-8 file reads; `Get-FileHash -Algorithm SHA256` for the probe. Version inspection used a Python script piped over SSH.

Two harness issues were resolved without candidate edits: PowerShell rendered a Python warning on SSH stderr as a native-command error despite the probe finishing; rerunning with remote `2>&1` recorded unambiguous remote exit 0. A nested-quote `python -c` environment query failed in the remote shell before Python execution; piping the version script fixed it. Neither was classified as candidate behavior or a passing test.

## Scope, residual risks and next gate

No substantive defect was found in ordinary R equations, simple-slider/general-body routing, model-option immutability through public properties, default per_row selection, native raw-W correction or full condim1 masking. No material operative contradiction was identified in the new scoped API document; historical broader documentation is not evidence of current approval. Reference coefficient and R native-domain claims need tightening with the fixes above.

This review does not approve the candidate until a fresh reviewer independently closes F1-F3 against a frozen revision, retaining these exact repros and checking affected ordinary paths. Approval of code, product activation, main merge and publication is not granted here. The original exclusions remain: condim3 cone/projector correctness, friction trajectories, default-30-iteration convergence, cache identity/rho redesign, full MuJoCo compatibility, arbitrary moving/high-speed contacts, material calibration, GPU performance, rendering and overall simulator correctness.

No full repository suite, multi-device/concurrent execution, CUDA publication lifecycle expansion or friction-force acceptance was attempted; none is claimed. The focused suite plus actual H200 independent probes is proportional to this bounded change. Extreme input probes test admission and native assembly only, not physical fidelity or extreme QP convergence.
