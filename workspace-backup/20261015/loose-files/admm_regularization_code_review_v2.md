Initiative: admm-regularization
Stage: code-review
Author: codex
Version: v2
Date: 2026-10-15
Status: in_review
Related Files: collab/admm-regularization__proposal__codex__v2.md; collab/admm-regularization__design-review__codex__v2.md; collab/admm-regularization__code-review__codex__v1.md; docs/admm_regularization.md
Owner Summary: Request changes. Independently reran the exact original probe and the full focused CPU/MuJoCo/H200 suite (174 passed). F1 and F2 close; the exact F3 repro rejects, but F3's native-domain guarantee remains incomplete. Decreasing impedance interpolation can round below its minimum endpoint, admitting Inf R on both CPU and GPU. Ordinary 2 kg GPU R also exceeds the accepted assembly error budget for this curve. An additional adjacent width-threshold conversion failure makes ordinary GPU R four times the actual MuJoCo/CPU value. Two blocking findings are documented below.

## Verdict and provenance

**Request changes.** Two blocking findings remain: F3 continued (cancellation defeats native bounds and ordinary assembly accuracy), and F4 (float32 conversion changes the width-threshold branch). CPU/GPU and endpoint/interior manifestations of F3 are one root cause, not duplicate findings. No other substantive blocking finding was identified in the inspected scope.

Reviewer session: `/root/admm_regularization_code_review_v2`, a fresh independent review agent that did not author or edit the candidate. Read the owner-directed `C:/Users/AG/新建文件夹/c4_worker_completion/AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` in full. The initial root REVIEW_PRINCIPLES path was absent; the collab file was read completely with UTF-8 decoding. Candidate historical protocol was not substituted for the current owner workflow.

- Local checkout: `C:/Users/AG/新建文件夹/robot_physics_correctness`.
- Remote checkout: `tileops-dev:/data2/shared/ang.gao/robot_sim_physics_correctness`.
- Branch: `physics/correctness-audit-20261012`.
- Original implementation base: `6c66f4b12c62686626db7069953a838dacd61104`.
- Rejected v1 candidate: `554133373595206ea2153a7f3cadd8f365451eec`.
- Reviewed frozen HEAD: `5f51ffc8f9dac32105b41bc88874832b51a89348`, independently verified locally and remotely before and after execution.
- Actual base-to-HEAD diff: 19 files, 1765 insertions, 31 deletions. Re-review delta: 11 files, 607 insertions, 23 deletions. Inspected actual production changes, relevant tests, scoped API document, accepted design/review, prior report, implementation notes and evidence claims.
- Local and remote working trees clean before and after review. `git diff --check` passed. No candidate tracked files were changed.
- Interpreter: `/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python`. Observed Python 3.12.13, NumPy 2.5.1, SciPy 1.18.0, MuJoCo 3.15.0, Warp 1.12.0. Real NVIDIA H200 with CUDA toolkit 12.9 / driver 13.2, `CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1`.

All review scripts/logs/report remain outside the candidate. Normal Python/pytest/Warp caches may be touched by runtime execution. Tests and probes executed anew; cached kernel compilation is not substituted for execution or used as performance evidence.

## Prior findings and independent closure

**F1 — Closed for the stated public finite-input admission boundary.** The exact original GPU q=NaN, qdot=Inf, tau=NaN, finite q=1e100, zero dt, step_n bad tau/zero dt, reset NaN and reset_envs NaN cases now reject with ValueError. Source inspection confirms public step validates all supplied inputs and model parameters before q/qdot upload, step_n validates before its first substep, and both reset paths validate before reset writes. None alone selects default dt in mujoco mode; legacy per_row fallback is retained. The fresh final-candidate suite asserts unchanged q/qdot/tau, contacts, warmstart arrays/R, default q, run identity and publication counters, with subsequent successful step/partial reset/step_n. It includes two environments, Inf and float32 overflow, negative/nonfinite/underflowing dt and a configuration failure concurrent with valid external-state input. No private-buffer tampering protection or new device-state readback is inferred.

**F2 — Closed.** The exact original mutable `solref=(.02,1e155)` probe now returns finite force output and no Python square OverflowError. The normalized tuple and the same `reference_coefficients` result are consumed by CPU assembly. The observed coefficients remain b=105.26315789473685 and k=2.7700831024930746e-307. Fresh final-candidate tests cover Python tuple and NumPy array construction/mutation and reject 1e160 before frame/cache/R mutation. Constructor normalization is separate from actual-dt coefficient admission. This closes the validation/execution inconsistency, not an extreme-force convergence gate.

**F3 — Exact original case fixed, general closure incomplete.** The original mass=1e-35, d=1e-4 real-engine case now rejects with ValueError before assembly. Fresh tests verify the stated mass1e-34 positive and mass1e-35 negative cases for body-world/body-body and condim1/3, plus tangent underflow/overflow and nearby finite cases. However, those tests use equal endpoints for the overflow boundary and do not test native decreasing-curve evaluation. The new counterexample below defeats the guard with both endpoint and interior contact depths.

## Blocking finding: F3 continued — Native impedance evaluation escapes the bounds assumed by R admission

Severity: **Medium, blocking**.

Locations:

- `physics/solvers/contact_regularization.py:78-81`: R bounds use converted minimum/maximum solimp endpoints.
- `physics/solvers/contact_regularization.py:113-124`: CPU impedance returns `d0 + y * (dw - d0)`.
- `physics/backends/warp/admm_kernels.py:25-37`: GPU impedance uses the same cancellation-prone expression, including at x=1.
- `physics/solvers/admm_qp.py:239` and `physics/backends/warp/admm_kernels.py:299`: actual R multiplication consumes that impedance.

The bound is conservative for exact interpolation inside the endpoint interval. Native arithmetic does not preserve that premise: when d0 is near 1 and dw is small, subtraction and re-addition lose the small endpoint. Near x=1 the upper curve also forms y=1-small, losing the small complementary contribution. This is not a claim that arbitrary finite operands guarantee a converged solve: it directly violates the new finite/normal R admission claim and the existing explicit assembly budget.

### Real H200 endpoint counterexample

Create two free balls with equal mass **4.89730620518322e-35**, using the existing `_ball_model` and `merge_models`. Construct a real `GpuEngine(..., solver='admm', regularization='mujoco', contact_condim=1, dt=.002)`. Set solimp to **(.9999, .00012, .001, .5, 2)**. Reset their heights to .09 and horizontal separation to .19, producing two ground contacts and one ball-ball contact at depth about .01, then call public `step()`.

No private device buffer is changed. `_admm_solimp` is the established solver-parameter configuration used by the implementation's own R tests and prior finding. Both the shared host validator and the engine step admission accept this configuration.

Observed values:

| Quantity | Value |
| --- | ---: |
| Each uploaded body weight | 2.0419388906938637e34 |
| Native endpoint dw | .00011999999696854502 |
| Interpolated native d at x=1 | .00011998414993286133 |
| Host maximum-R bound | 3.4028232635611926e38, finite |
| Exact endpoint pair R | 3.4028230967116344e38, below float32 max |
| Actual ground R, each | 1.7016363608790047e38 |
| Actual body-body R | **Inf** |

The call returns, and R storage has changed. The integrated q happens to remain finite in this fixture; that does not make the admitted Inf regularization valid. The same result was independently reproduced for **condim3**. The standalone repro explicitly asserts both finite exact expected values and actual Inf in the body-body row.

### Interior and ordinary-scale evidence

Replacing width with **.010001** makes the actual GPU x approximately **.9998998014**, strictly inside the curve. Both the overflow and ordinary-scale accuracy failure persist, so adding an endpoint special case alone would not close the finding. A second interior width **.0101**, x approximately **.9900988034**, also exceeds the approved R budget at ordinary mass.

A real **2 kg** single-ball engine removes any need to interpret extreme-scale forces. Actual MuJoCo, production CPU, production GPU and an independent reference expression evaluated at the GPU's actual contact depth give:

| Width | MuJoCo R | CPU R | GPU R | Expected R at GPU depth | GPU absolute error | Approved absolute budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| .001, x=1 | 4166.166666666667 | 4166.166666666353 | 4166.71728515625 | 4166.166666666355 | .55061849 | .124986 |
| .010001, x=.9998998 | 4165.472629461296 | 4165.472629461294 | 4166.71728515625 | 4165.469731174173 | 1.24755398 | .12496509 |
| .0101, x=.9900988 | 1581.6979849118884 | 1581.697984911888 | 1581.8751220703125 | 1581.6570386008502 | .21808347 | .04745071 |

Budget is the accepted `3e-5 * abs(expected R) + 2e-6 * inverse_mass`, with inverse_mass=.5. Evaluating the reference at actual GPU depth removes small collision-depth differences as an explanation. The .02-width midpoint control passes: error1.5114e-7 versus budget1.59994e-5. CPU matches actual MuJoCo in all four ordinary-mass cases.

### CPU native-bound counterexample

The float64 path has the same finite-domain hole at its own numerical boundary. With two balls of mass **8.556863841944613e-305**, solimp **(.9999, .00013, .001, .5, 2)**, and one normal body-body contact at depth .01, `validate_R_domain` accepts:

- Each computed body weight: 1.168652462480626e304.
- Host upper bound: **1.7976931348623133e308**, finite and below float64 max.
- Actual interpolated d: **.00012999999999996348**, below .00013.
- Production CPU `last_R_diag`: **[Inf]** after solve.

A second dw=.00015 case also overflows. The scratch run reports the R multiplication overflow warning at `admm_qp.py:239`. A control dw=.00012 produces finite R but can overflow A+R at this extreme scale; that separate downstream conditioning/representability limitation is not asserted as an additional finding or a newly required extreme-force gate.

### Why existing checks miss it, repair direction and required proof

The guard checks endpoint values and possible pair weights, but not the numerical range of the actual interpolation. Existing overflow tests use d0=dw=1e-4, so they cannot expose cancellation. Added normal-scale curve tests cover increasing/clipped/high-power cases but no decreasing near-endpoint case. The whole 174-test suite therefore passes despite this failure.

Use endpoint-exact, numerically stable and bounded native impedance evaluation on CPU/GPU, retaining meaningful small complementary curve terms near either endpoint. Ensure the guard's bounds actually cover the values the native assembly consumes. Simply accepting Inf, widening test tolerances, or only special-casing x=1 would not address the interior reproductions. If native rounding requires a small conservative admission margin, make it explicit and justified rather than weakening the finite/normal requirement.

Required repair proof: preserve and rerun these endpoint/interior CPU and real-H200 repros. Show finite correct R or clearly documented rejection before frame/cache/state mutation, including recovery. Add decreasing-curve conformance at ordinary masses against actual MuJoCo and independent native-depth expectations, as well as both endpoint directions, near-endpoint interior values, width-zero behavior and relevant midpoint/power normalization. Recheck condim1/3, body-world/body-body and the original F1-F3 cases. No friction-projector repair or extreme QP convergence is required to close this finding.

## Blocking finding F4 — Float32 conversion changes the defined width-threshold branch

Severity: **Medium, blocking**. This is an adjacent numerical-boundary issue with a distinct trigger from F3's cancellation. The coordinator suggested checking this adjacent boundary while the candidate remained frozen; the reviewer independently confirmed it in actual production GPU execution and actual MuJoCo.

Locations: `physics/solvers/contact_regularization.py:53-62` (`native_parameters` conversion/admission), `physics/backends/warp/admm_kernels.py:28-29` (`width <= 1e-15`), and GPU launch of normalized solimp at `physics/gpu_engine.py:1669-1673`.

For an ordinary 2 kg free ball, depth about .01, condim1, dt=.002 and solimp **(.2, .8, np.nextafter(1e-15, np.inf), .5, 2)**, normalized host width is **1.0000000000000003e-15**, strictly above the contract's cutoff. CPU and MuJoCo therefore use the regular curve and its saturated endpoint d=.8. Both yield **Rn=.125**. GPU casts that width to **1.0000000036274937e-15**, the same float32 value as its cutoff literal. Its `<=` branch uses the endpoint average d=.5 and produces **Rn=.5**, four times the reference.

The configuration passes admission and public step returns. This is a large ordinary-scale R discrepancy, not merely a rounding allowance or an extreme-force convergence concern. The independent probe checked the adjacent values:

| Host width | CPU / actual MuJoCo R | Actual H200 R |
| --- | ---: | ---: |
| nextafter(1e-15, 0) = 9.999999999999999e-16 | .5 | .5 |
| 1e-15 | .5 | .5 |
| nextafter(1e-15, +Inf) = 1.0000000000000003e-15 | .125 | **.5** |
| 1.000001e-15 | .125 | .1249999925494194 |

Existing width tests use zero and ordinary .001/.1 values, so they do not cross this native conversion threshold. Finite/positive checks do not establish preservation of a discontinuous branch decision.

Repair direction: preserve the defined host normalization decision when communicating the parameters to the native kernel, or explicitly reject a narrowly documented ambiguous conversion domain before mutation. Do not silently reinterpret a host value above the declared cutoff as the averaged-width branch. Keep any representation choice centralized and avoid moving the public cutoff merely to satisfy the examples.

Required proof: rerun the exact nextafter fixture against actual MuJoCo and real H200, asserting the defined branch and R budget, with below/equal/just-above cutoff controls, unequal endpoints in both directions, and condim1/3. Check that the impedance fix retains width-zero and ordinary curve behavior. This can be addressed alongside the F3 impedance repair, but must have explicit evidence because an interpolation-only fix would leave it unchanged.
## Verification and positive evidence

Fresh complete focused suite at the exact final HEAD: **174 passed, 1 warning in 71.44 s**. The warning is the existing pytest class-scoped fixture deprecation in `test_mujoco_qp.py`. Ruff checks of all changed production/test Python files passed; `git diff --check` passed.

Inspected CPU validation-to-frame/cache ordering, reference/current FK restoration, CRBA/COM Jacobian convention, simple-slider routing, normalized parameters and coefficients, R and rhs construction, diagnostic copies and warmstart flow. Inspected GPU construction-time body-weight ownership, public step/reset entry points, raw-W dispatch, native impedance/coefficient/R arithmetic, all condim1 rows/columns/rhs/cache/projection/output masks, inactive/empty slots and unchanged PGS use. No runtime MuJoCo dependency, strategy-switch cache path, material layer, default-policy change or unrelated rendering/main-merge work was introduced.

The original independent probe was executed unchanged against final HEAD. Besides the fixed failure cases, it again passed CPU invalid state/data pre-mutation checks, seeded stale tangent/inactive cache masking with two GPU environments, and nonzero normal/tangent velocity scalar-oracle checks for both policies. The full suite independently exercises actual MuJoCo weights/efc_R, simple and non-simple sliders, coupled normal active-set/positive optima, model edits, empty solves and both GPU strategies. Ordinary GPU fixtures reproduce maximum R relative error3.3250521e-7 and normal-impulse error6.4867140e-8 N s; the newly discovered decreasing fixtures are outside those covered cases, not evidence that those measured cases failed.

All four full real CPU trajectory comparisons executed anew:

| Fixture / dt | max q error | max velocity error | max per-step impulse error | max penetration error |
| --- | ---: | ---: | ---: | ---: |
| sphere / .001 | 1.3878e-17 | 1.1102e-15 | 4.9960e-16 | 1.3878e-17 |
| sphere / .0005 | 1.3878e-17 | 1.3323e-15 | 2.4980e-16 | 1.3878e-17 |
| quadruped / .001 | 4.4409e-16 | 3.1086e-14 | 9.7491e-16 | 1.1449e-16 |
| quadruped / .0005 | 6.6613e-16 | 3.2863e-14 | 7.8236e-16 | 1.1102e-16 |

Contact counts agree; maximum cumulative impulse error2.0123e-15 N s. Sphere peak penetration remains .01236544 m and .01262978 m. These close the bounded matched-fixture trajectory evidence requirement, not general integration/collision/material correctness.

## Reproduction commands and artifacts

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

Fresh suite log: `C:/Users/AG/新建文件夹/admm_regularization_code_review_v2_tests.log`.

Original unchanged probe: `C:/Users/AG/新建文件夹/admm_regularization_code_review_v1_probe.py`, verified SHA256 `8DC93C827A99F5742E65E5C9E3716BB9B02BBA374608C7C46FF5BAC44A9CE3A1`. Fresh output: `admm_regularization_code_review_v2_original_probe.log` in the same outside-checkout directory.

New reviewer-owned artifacts, all under `C:/Users/AG/新建文件夹/`:

- `admm_regularization_code_review_v2_boundary_repro.py`: precise real-H200 endpoint failure with assertions, condim1/3 and environment provenance. SHA256 `708C0CF86E89DCE5C4C7216EA09BCE0FC84A534F531F0C1B5FC6CBCC9AAE5583`.
- `admm_regularization_code_review_v2_decreasing_probe.py`: actual MuJoCo/CPU/GPU ordinary-mass comparisons, native-depth oracle, interior overflow and midpoint control. SHA256 `40E5476719A51A61817A470560EC1ECC70159CDD1B938BEEB50C0581F5E28025`.
- `admm_regularization_code_review_v2_cpu_edge_probe.py`: float64 native-bound counterexamples/control. SHA256 `BA93D33B76E6A5C51893669B8F8E443812718B27870AC571C14BC1FF7A31B6E1`.
- `admm_regularization_code_review_v2_width_probe.py`: ordinary-mass actual MuJoCo/CPU/H200 nextafter width-threshold comparison. SHA256 `B8A0D45E27F4480EA074136133D317E987270957F5C6181BF1DB87CB6C7CC973`.
- Each script has a same-stem `.log` containing the fresh complete output. Initial exploratory `admm_regularization_code_review_v2_edge_probe.py/.log` are also preserved.

Each script was executed with this PowerShell pattern (replace SCRIPT and LOG with the respective filenames):

```powershell
Get-Content -Raw -Encoding UTF8 'C:/Users/AG/新建文件夹/SCRIPT.py' |
  ssh tileops-dev 'CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 /data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python - 2>&1' |
  Tee-Object -FilePath 'C:/Users/AG/新建文件夹/LOG.log'
```

All final probe commands exited 0. For the counterexample scripts, that means the repro assertions/measurements ran successfully; it does not mean candidate approval. No original repro was edited. The CPU exploratory script prints outcomes rather than asserting its intentionally bad outcome; the captured Inf result and source location are explicit.

Other commands included local/remote `git status --porcelain`, `git rev-parse HEAD`, `git branch --show-current`; both actual base and v1-to-HEAD diffs; local `rg` and UTF-8 file reads; SHA256 checks; and remote `python -m ruff check` on the ten changed production/test Python files. An initial nested-quoted remote `python -c` version query failed in shell parsing before execution; the later piped probe obtained versions successfully. This was a reviewer command issue, not a candidate failure or a passing check.

## Scope and remaining gates

No approval is granted for this frozen candidate until both blocking numerical issues are repaired and independently re-reviewed. F1/F2 are closed at this revision; original F3 is repaired narrowly but remains open at the declared numerical-domain/assembly level; F4 separately affects the declared width normalization. Conservative native-domain restrictions are documented and can be appropriate; this report does not require supporting every finite model or force scale. It requires admitted assembly to satisfy those restrictions and the stated ordinary-scale budget.

The successful evidence covers optional per_row/mujoco R construction, raw-W correction, reference mobility with simple-slider exception, actual condim1 engine plumbing, bounded normal-only snapshots and the four stated frictionless trajectories. It does not approve condim3 projection correctness, frictional forces/trajectories, default-30-iteration accuracy, rho/cache identity redesign, private device-buffer mutation, extreme-scale QP convergence, arbitrary moving/high-speed contacts, material truth, full MuJoCo compatibility, GPU performance, rendering, product activation, publication, main merge or overall simulator correctness.

No full-repository suite, multi-device/concurrent execution or unrelated CUDA publication lifecycle audit was performed or claimed. The focused real-environment checks are proportional to this bounded change. F3 is independently reproduced in production CPU and real GPU assembly; F4 is independently reproduced in real GPU assembly against actual MuJoCo/CPU. Both remain code-review gates.

