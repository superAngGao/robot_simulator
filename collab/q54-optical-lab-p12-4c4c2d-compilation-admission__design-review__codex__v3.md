# Independent design re-review: bounded CUDA compilation admission v3

Initiative: Q54 / P12.4 / C4c2 / remaining parent C2d
Stage: independent design review
Author: codex, independent reviewer /root/compilation_admission_design_v3
Version: v3
Date: 2026-10-09 (Asia/Shanghai)
Status: accepted — design only
Related Files: compilation-admission design-request v3 and unchanged design-review v1/v2; runtime-integration design-request/review v3; remaining-setup gap-scan v1; compilation-job gap-review v1; runtime-adoption design-request v2; guarded-custody repair-design v2; supervised-waits design-request v3
Owner Summary: Approve the exact bounded v3 design. Successful generated roles now require distinct actual guardian adoptions tied to recipe, role, named entry, active guard and occurrence order; root-only retention remains available for partial returns and exceptions. The earlier wire and post-worker-exit deadline repairs remain sound. This approval permits bounded implementation and CPU proof, not new native compilation, checkpoint acceptance or any later product gate.

## Identity, isolation and verdict

**Verdict: Approve.** No substantive blocking or non-blocking finding remains in this full bounded design re-review. This is not approval of an implemented compilation collector/parser, compiler supervisor or native checkpoint: those implementations do not yet exist in the candidate.

Fresh reviewer session `/root/compilation_admission_design_v3` received the bounded review packet and repository access without the implementation conversation. I did not author or modify the candidate and did not delegate review or implementation. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully, including explicit UTF-8 rereads. The full candidate, its v2 delta, both prior reports and original probes were inspected; relevant governing contracts and actual API/resource paths were checked independently. Author claims and previous verdicts were not treated as proof.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Actual base: `4597d51e3e001e0f7a2c5c2fd657489a11314dfc`.
- Frozen reviewed HEAD: `b3c4110cbf34aaf7512aceb722609fade69f85f0`.
- Sole base-to-HEAD change: `collab/q54-optical-lab-p12-4c4c2d-compilation-admission__design-request__codex__v3.md`, 741 added lines.
- Candidate working-byte SHA256: `028b52c466ef46433235d4ec2495cd47d00200822cc64dfaea95614fd5bf26a2`.
- Candidate Git object: `6ca3866079f41bf01857c68902c43f9905b8e45d`.
- Initial and final local porcelain status empty; HEAD unchanged; base-to-HEAD `git diff --check` passed.

Accepted whole C2d3 remains accepted within its original scope: code/test `4da1fa75634d716d4018d616327c84096e047432`, production `f1ca45a`, original native execution `fc8618268dd8f3fb10e29e32626219500751c41a`, acceptance archive `4c20cac`. The earlier gap report's production wording does not replace the later source-pin identity. This review does not reopen unrelated C2d3 acceptance.

## Prior findings and successful-role repair

**R1 remains closed at design level.** Sections 2 and 6 require the actual wire sequence input0..5, compile6/7/8, terminal9, one immutable production phase table and rejection of missing/duplicate/reordered/unknown phases. The unchanged production core independently accepts nine normal guards followed by emergency9. The retained v1 probe still rejects ARM7 immediately after six guards, demonstrating the API constraint that v3 now respects. Implementation must still test the integrated COMPLETE interception; a legal standalone core trace is not that proof.

**R2 remains closed at design level.** Section 6 records the earliest operation or pending-ACK obligation independently before the core's actual worker-exit observation clears it. The outer job expires and acts only on verified adopted children after worker exit, leaving the core byte-stable. Section 7.4 supplies explicit outer expiry, sticky failure, source/deadline, single draining/expiry/identity-attempt rules and clock/observation errors. Both operation-deadline and earlier pending-ACK modeled-exit probes confirm the actual core behavior. No core event or worker signal is invented after EXITED. The later real Linux cases remain mandatory.

**R3 and R3-continuation are now closed at design level.** Sections 7.1–7.3 preserve one `_RuntimeObservation` registry, immutable upstream v2 bytes and J/L cut, globally indexed suffix occurrences, exact new object projection, full validated lifecycle prefix, component/source cuts and separate downstream checkpoint selection. Old v1/v2 parsers are not widened to accept compilation kinds. The literal source continuation and external phase rules connect this record to the real input-validation boundary.

The new lines 508–560 remove the v2 ambiguity. A successful spec, standard delegate, returned module, each specifically named entry and compiled owner must have a distinct direct `owner_adopted` anchor with the exact table role and `recipe_id`-derived operation. Each anchor must use literal `HOST_OWNER`, the same active setup guard, and precede the corresponding successful step. The collector checks the actual owner in the guardian array against the token's actual object; the parser joins the projected sequence to the independently supplied lifecycle. Repeated steps and derived recipe rows cannot redefine those bindings. A spec/loader exchange now fails different roles and operations; same-recipe entries fail their distinct `.entry.<name>` operations. Input objects retain their original adoption facts and cannot become these newly generated roles. A root-held unrelated object has no required role anchor.

This is feasible with current APIs. `lifecycle.py:57` permits the literal operation strings within its 128-character identifier domain; `:306` rejects duplicate exact-owner adoption and records distinct sequences; `:628` uses `HOST_OWNER`. `guarded_lifecycle.py:99` attaches the current guard to occurrence-time adoption events. `_RuntimeObservation.token` and `freeze_adoptions` at `native_bootstrap.py:2102,2143` preserve exact object identity and match actual guardian owner/record pairs. Existing `runtime_custody.py:332` already has the relevant direct-adoption role/scope/operation/time join pattern, which can be shared without admitting new kinds to the old grammar.

The repair does not force fabricated success on partial failures. The root holds each new object before direct adoption, including the delegate-create return before importlib's metadata initialization. If adoption fails, the object remains held, the recipe fails, and no successful step binds that failed role. Earlier successful anchors remain unchanged; partially adopted entry sets can remain retention-only until the all-entry step succeeds. Exceptions and diagnostics may have null direct anchors. A CPU execution through the actual guarded lifecycle and observation registry confirmed that a deterministic direct-adoption rejection preserves the original returned object through the guardian-held root, produces sticky guard failure, and records no successful role/direct anchor for that object.

The added role-join probe is explicitly a schema-slice model using actual collected CPU adoption facts. It rejects the prior swaps with lifecycle/retention facts fixed, but does not claim to be a production v3 parser or native-module identity test. The design correctly requires those integrated collector/source/parser tests in section 8. Neither descriptive parsing nor direct host adoption creates a CompiledCudaOwner capability, proves native scope, or authenticates an entirely rewritten consistent execution history.

## Full bounded design assessment

The resource and state flow is closed at the design level:

1. The accepted input cut remains upstream and immutable. A guardian-held root is established before new source/tool imports. Missing root adoption leaves only truthful input evidence and failed-guard/raw diagnostics, rather than a fabricated compilation record.
2. Each fixed recipe runs under its original ordinary setup guard. New source bytes are captured, external imports remain under the frozen environment and failure semantics, and returned roots are held before subsequent fallible work.
3. The one direct-load adapter is tied to the exact admitted cpp_extension module and original helper. It rejects overlap/reentry/thread/recipe drift and checks exact hook restoration. Successful build generation does not itself admit a module or create readiness.
4. Exact spec/delegate/module/entry/compiled-owner identities, lifecycle anchors and native before/after facts establish the successful worker prefix. Compiler quiescence is checked before core COMPLETE acceptance and DISARMED. Failure retains partial roots and the original failed guard without another emergency or ordinary unload.
5. Healthy checkpoint disposal uses the existing terminal9 target-sync/exit path. Compiler-job closure separately requires worker and adopted-child reaping, ECHILD under the verified policy, and actual pipe EOF. Neither parsed evidence, kill success nor an empty scan grants closure.

Specific high-risk checks:

- **Recipes and sole implementation.** Existing `cuda_scene.py:923`, `cuda_lbvh.py:1112` and `cuda_direct_light.py:1688` match the fixed names, entry order and flags. Factoring scene/direct-light source strings once into pure source modules and retaining LBVH's existing source owner avoids importing unrelated renderer/timeline compatibility graphs. Legacy wrappers and the guarded compiler share one recipe definition. Byte-for-byte source preservation, wrapper behavior and unchanged entry ABI remain code-review checks. No CUB workspace query or compiled optical entry runs at this foundation checkpoint.
- **Pinned Torch route.** The inspected source inserts headers and optionally generates pybind bindings, writes main.cpp/cuda.cu, builds synchronously and calls `_import_module_from_library`. That helper directly performs spec/module creation and exec, so the ordinary finder does not cover it. A private exact helper adapter and direct ledger address this actual gap. The wrapped delegate must retain its returned module before fallible importlib metadata work, consistent with current external admission at `native_bootstrap.py:544–575`. The wrapper and standard delegate remain distinct held objects with audited relationships; loader type strings are not authority.
- **Provenance and filesystem effects.** Fresh private directories, unsuffixed names, no earlier cache/module owner, frozen environment/tool paths and full compiler/header/library inventory are explicit prerequisites. Generated bytes are downstream observations, not permission to load arbitrary matching artifacts. The PCH absence/no-concurrent-writer rule is necessary because pinned `use_pch=False` calls deletion of shared Torch PCH paths. V3 forbids deleting those files to manufacture a pass. Emitted Ninja inputs and sampled executed process observations are correctly distinguished; verbose=False is not claimed to provide complete successful-command output.
- **Process ownership.** A dedicated one-tree, one-thread, one-waiter supervisor owns its subreaper setting and pidfds. Linux documents adoption by the nearest living ancestor subreaper and subsequent waitability; the ECHILD argument is plausible under the frozen no-escape/no-extra-child model and correct SIGCHLD policy. Actual policy readback and ABI correctness remain preflight obligations, not assumptions justified by Python getsignal. [Linux subreaper interface](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html), [Linux wait interface](https://man7.org/linux/man-pages/man2/waitpid.2.html).
- **Separate PGIDs and closure.** Ninja's pinned source assigns ordinary commands their own process groups, supporting the rejection of single-killpg cleanup. V3 instead signals positively identified owned direct/adopted children, retains earliest deadlines, handles new adoption after expiry, and distinguishes attempt/error/reaping. Persistent OS failure leaves DRAINING/incomplete. Normal compile quiescence uses the returned synchronous builder and blocked worker; whole-job ECHILD closure is a different boundary. [Ninja 1.13.1 source](https://raw.githubusercontent.com/ninja-build/ninja/v1.13.1/src/subprocess-posix.cc).
- **Native scope and failure.** HOST_OWNER describes Python custody only. Compilation/import may touch native state; the proposed three native-scope sites must bind real source-backed observations and lifecycle compile_or_jit operations, including nested create/exec within compilation. Unsupported scope stays UNKNOWN. No target safety is inferred from ordinal arguments, successful import or host adoption. Pre-return native exception safety remains in the accepted trusted dependency boundary.
- **Maintainability and scope.** One recipe, one original token registry, shared old validators with a bounded new grammar, one transport/core and one private direct adapter are appropriate. No offline publisher, general build service, daemon, cgroup provisioner, arbitrary callbacks or second renderer is introduced. Adapter removal and losing-lane cleanup have explicit future conditions. The unchanged 120-second setup/1-second ACK budgets are operational bounds, not benchmark gates or permission to restart a slow build.

## Verification, commands and results

Local environment: Windows PowerShell, `C:/Python314/python.exe`, bytecode disabled with `-B`. No backend imported. Read-only inspection used UTF-8 Get-Content/bounded ranges, rg/rg --files, Git status/rev-parse/branch/diff/hash-object and Get-FileHash. Initial root `REVIEW_PRINCIPLES.md` lookup and a Windows wildcard rg path failed; corrected reads used `collab/REVIEW_PRINCIPLES.md` and exact paths. Large combined outputs were narrowed to the relevant candidate/contracts/source ranges. `git diff --no-index` reported expected v2/v3 differences. None of these read errors is a candidate failure.

Executed from the repository:

```text
C:/Python314/python.exe -B ../c4_assets/compilation_admission_design_review_v1_probe.py .
C:/Python314/python.exe -B ../c4_assets/compilation_admission_design_review_v2_probe.py .
C:/Python314/python.exe -B ../c4_assets/compilation_admission_design_review_v3_probe.py .
git diff --check 4597d51e3e001e0f7a2c5c2fd657489a11314dfc HEAD
```

The unchanged v1 probe exited0 and reproduced both old API counterexamples. Its printed candidate digest is explicitly the retained v1 document. The unchanged v2 probe exited0, verified legal core wire0..9, preserved operation and earlier ACK deadlines around modeled worker exits with immutable exited-core bytes, and reproduced v2's role-free schema-slice swaps. Those old schema results are not assertions about v3.

The new v3 probe initially reached all role-join results but failed in its final assertion because the scratch test used a nonexistent summary key `failed`. I corrected only the outside-repository probe to the actual `guarded_segment_failed` key. The subsequent complete execution exited0:

```text
V3 ROLE-JOIN MODEL rejected: spec/loader swap
V3 ROLE-JOIN MODEL rejected: same-recipe named-entry swap
V3 ROLE-JOIN MODEL rejected: unrelated root substitution
V3 ROLE-JOIN MODEL rejected: wrong-recipe module substitution
V3 ROLE-JOIN MODEL rejected: root-only successful module
ACTUAL CPU API: all three recipes' exact role/operation/HOST_OWNER adoptions fit guards 6/7/8 and one registry.
ACTUAL CPU API: failed direct adoption keeps original root-held object, null direct anchor, sticky failure, no successful role.
LIMIT: no compilation collector/parser exists; modeled joins are design illustration only. Both peer threads joined.
```

This probe uses the actual controller, guardian, observation registry, core/client/framed transport and local socket peer. Its objects are CPU identity sentinels, not fabricated native modules. Its new suffix/join checks are an explicitly limited design model, not the actual future continuation/collector/parser and not a substitute for section 8's production-boundary tests. No child or signal is used by the probe.

Probe SHA256s:

| File under c4_assets | SHA256 |
| --- | --- |
| compilation_admission_design_review_v1_probe.py | d719a7a8fb5788242a85b8423e8a9aabc2bce3264544a6fb09feeab89f1974a4 |
| compilation_admission_design_review_v2_probe.py | 79ee58260c9431ab223e550378a5d51b0ebf6440a6a694e06efa22fa9aac8b93 |
| compilation_admission_design_review_v3_probe.py | ba8aa749ded92bfdd342a00b3bc03bf8d656d133648b70a2e383a719b6ccff17 |

Primary source-only SSH command used `ssh -o BatchMode=yes -o ConnectTimeout=8 tileops-dev` with `git -C <tree> rev-parse HEAD`, `git -C <tree> status --porcelain`, and `sha256sum` on installed cpp_extension. Transport succeeded; no fallback was used. Independently observed:

- Runtime tree `/data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip`: clean `fc8618268dd8f3fb10e29e32626219500751c41a`.
- CPU tree `/data2/shared/ang.gao/robot_sim_c2d3_array_join_wip`: clean `4da1fa75634d716d4018d616327c84096e047432`.
- Installed `/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages/torch/utils/cpp_extension.py` and local `remaining_c2d_pinned_cpp_extension.py`: SHA256 `52ba4de5deedd894e2bdafc9b7f40526dbc2cc1e4c560bbae3b38d4a5dcab0c6`.

I read the separate toolchain-path and sigaction-header preparation notes. Their path/header observations and inferred layout are preparation only; this review does not promote them to a complete toolchain inventory, ABI execution proof or host capability test.

## Limits, hygiene and exact approved scope

No compiler/Ninja execution, remote interpreter, backend import, native observer/getter, native-OS probe, GPU query/reset, process signal, installation, remote write or product operation occurred. All shell/SSH/probe calls completed synchronously. The v3 probe closed its sockets and joined both peer threads, including the initially failed run's finally paths. A final read-only Win32_Process query restricted to Python command lines naming the three review probes returned no residual process. No reviewer worker/supervisor/compiler/background job was created. Utility PIDs were not individually sampled; this is bounded hygiene, not exhaustive OS/GPU attestation. Only the v3 scratch probe and this report were written outside the repository.

Remaining required evidence is substantial and explicit in the approved design: actual integrated continuation/collector/parser counterexamples; exact source/policy/catalog and fixed-runner code preflight; complete frozen toolchain/environment closure; independently verified libc ABI/readback; real Linux pidfd/subreaper, separate-PGID, leader-first-exit, double-fork, fork-during-cleanup, inherited-pipe, stale-identity and OS failure cases; all three cold builds/direct admissions; target1/ambient0 restoration at every new boundary; PCH lifetime invariants; partial native create/exec/adoption/restoration failures; actual owned process/GPU residual closure; and fresh independent whole-checkpoint review. None was waived or represented as already passed. There was no toolchain or environment failure in this design review; the scratch assertion correction was reviewer test code only.

Approval binds only v3 at the HEAD/digest above and authorizes its bounded implementation and proportionate CPU proof. Exact source/policy/catalog/runner preflight still precedes new native compilation or native observer execution. The CUDA `compiled_cuda` foundation itself is not accepted implementation or full G1 completion across both lanes.

Warp compilation admission, one-slot construction, R_setup/R_final, ten warmups, graph/final barrier, C2e/composition, full C2d/C4 acceptance, product PREPARE/submission, sample/result/measurement authority, a3b, R2/S/host payload, shipping, performance and backend-selection gates remain closed. Accepted C2d3 retains its original scope. No new permission question or unrelated policy expansion is required by this report.

The complete report may be archived verbatim as review evidence. Archival must not change this verdict, reviewed identities, approval scope, required proofs or remaining gates.
