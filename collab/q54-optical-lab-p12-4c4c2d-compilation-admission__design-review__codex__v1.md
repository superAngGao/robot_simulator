# Independent design review: bounded CUDA compilation admission v1

Initiative: Q54 / P12.4 / C4c2 / remaining parent C2d
Stage: independent design review
Author: codex, independent reviewer /root/compilation_admission_design_v1
Version: v1
Date: 2026-10-09 (Asia/Shanghai)
Status: in_review — review complete; changes requested
Related Files: compilation-admission design-request v1; runtime-integration design-request/review v3; remaining-setup gap-scan v1; compilation-job gap-review v1; runtime-adoption design-request v2; guarded-custody repair-design v2; supervised-waits design-request v3; source locations below
Owner Summary: Request changes for the exact compilation design. The selected worker-JIT route and dedicated subreaper approach are plausible within this scope, but the fixed wire sequences contradict the existing protocol, the post-worker-exit timeout branch conflicts with the terminal core, and the new token evidence lacks its defined custody projection. These are bounded design repairs, not grounds to reopen accepted C2d3 or substitute an unrelated process service. No compilation or native execution is approved.

## Identity and verdict

**Verdict: Request changes.** Three blocking P2 design findings follow. This report approves neither implementation of this revision nor compiler/native execution.

Fresh independent session `/root/compilation_admission_design_v1` received the review packet and repository access without the author conversation. I did not author the candidate, modify implementation/design files, or delegate review. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully. The workflow explicitly authorizes this independent review.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Actual base: `8cad0e20ab6d97545013daea73e45965a90a89d0`.
- Frozen reviewed HEAD: `cc2f637e21dcf640343ff3d743fdebc0b27d094e`.
- Candidate: `collab/q54-optical-lab-p12-4c4c2d-compilation-admission__design-request__codex__v1.md`, 412 added lines, the sole base-to-HEAD change.
- Candidate working-byte SHA256: `0af9d96d9b97b36c8d9a8e59925b84b1eacf05005f00b723de36405a1fede6db`.
- Git candidate blob and working-file Git object both: `09a6d01396f1ec1d9bb3315feb8a8fcfcead7769`.
- Initial and final local porcelain status empty; HEAD and candidate digest unchanged; base-to-HEAD whitespace check passed.

Accepted whole C2d3 remains accepted at code/test `4da1fa75634d716d4018d616327c84096e047432`, production `f1ca45aa0e6b178804828acedb2ad2539712a5ef`, with whole review v4 archived at `4c20cac`. Original native execution provenance remains `fc8618268dd8f3fb10e29e32626219500751c41a`. I checked the current whole review's identity/scope rather than treating the earlier gap report's mislabeled production paragraph as controlling. This is not a new whole-C2d3 review.

## Findings

### R1 — P2: fixed guard sequences are shifted by one

Location: candidate lines 47–54 and 307–313 (sections 2 and 6).

The candidate says the six accepted input segments retain sequences `1..6`, compilation uses `7,8,9`, and terminal disposal uses `10`. Actual guarded-wait v1 begins at zero. `GuardWaitClient.__init__` initializes next sequence to 0 (`wait_protocol.py:194`); the supervisor starts at 0 as well. The existing fixed runtime driver invokes exactly six setup operations, in order, at `tests/support/native_runtime_process.py:879–897`. Their production bodies each open one segment (`native_bootstrap.py:2410,2517,2567,2672,2833,2888`). Existing fault selectors also use zero-based phase indices and terminal sequence 6. There is no preceding hidden guard.

Trigger: preserve the accepted input path, then start the first compiler segment. Its actual next wire sequence is 6. If the implementer follows the frozen `7/8/9` supervisor table, that first compiler COMPLETE either bypasses the special quiescence check or is rejected, depending on how table membership is enforced. Attempting to send ARM 7 instead is rejected by the unchanged core. Adding a dummy guard or renumbering the protocol would violate the stated preservation of the accepted path.

Independent CPU reproduction through the production core completed six normal guards `0..5`, observed next sequence 6, and rejected ARM 7 with `ARM is out of order`. This is an actual core execution, not a simulated assertion about the candidate text.

Required repair: state the actual wire values consistently: input `0..5`, compilation `6,7,8`, terminal disposal `9`, unless a separately justified real additional operation is explicitly designed. Distinguish any human ordinal from wire sequence. Derive the driver's phase table and compiler interception mapping from one definition; do not create a second numbering source.

Proof after repair: a CPU integrated driver/transport trace traverses the six unchanged input guards, all three compiler COMPLETE interceptions and sole terminal ARM with the exact expected values. Demonstrate that the first compiler segment is checked, all three successful DISARMED records bind the right recipe, and shifted/skipped/duplicate sequences fail before following work. This does not require a compiler run.

### R2 — P2: leader-first exit has no legal specified timeout-recording branch

Location: candidate lines 242–284 and 343–355 (job lifecycle, timeout actions and evidence grammar).

The design correctly requires saving the original deadline independently and immediately calling `core.observe_exit` on actual worker exit. It then unconditionally directs: on timeout, record the existing worker kill request and signal the held worker pidfd. These instructions cannot both describe the leader-first-exit path that motivated this work.

Trigger: while a compile guard is armed, the worker exits before its original deadline, leaving a still-live compiler that becomes an adopted direct child. The core transitions to EXITED and its trace must end with `process_exited`. At the saved deadline the outer job must terminate the adopted compiler, but there is no still-live worker to terminate and no legal existing-core kill event to append. `wait_supervision.py:136` rejects every event after EXITED; `:549–552` appends the worker kill request through that reducer. Its `tick` returns immediately after exit (`:470–474`). The independent CPU probe preserved a modeled original deadline, called `observe_exit(0)`, reached that deadline, and confirmed `record_kill_request()` rejects with `no event may follow observed exit`. The modeled observation does not claim a real worker was launched or reaped.

The outer record is the right place to solve this, but the frozen schema currently has only `draining {reason,deadline_ns}` and signal events, not a separately defined post-drain expiry transition. DRAINING already began with reason `worker_exit`; the schema does not specify whether/how later expiry becomes sticky failure, how the promised `deadline_expired` fact is replayed, or which existing-core events are permitted when only descendants remain. A late second draining event versus signal-implies-expiry are materially different possible implementations. Neither the same-run join nor merely retaining a deadline selects one. An exception while trying the forbidden core call could prevent the descendant termination attempt.

Required repair: freeze two explicit cases. While the worker remains live, keep the existing core expiry/kill-request semantics and mirror actual actions in the job record. After its exit, append no further core events and do not invent a worker signal; expire the independently retained job obligation and act on the positively identified adopted children. Define an explicit replayable outer expiry/attempt/failure transition (or equally precise existing-event grammar), sticky failure rules, deadline source/nullability, duplicate behavior and clock/error handling. Preserve the earlier pending-ACK deadline when that was the earliest obligation; do not extend time after worker exit. Do not change `process_exited` to mean whole-job exit.

Proof after repair: CPU replay covers leader exit before operation deadline and before an earlier pending-ACK deadline, with live adopted children at expiry; guard evidence stays byte-stable after its real exit event, outer evidence records expiry and child-only attempts, duplicate observations do not renew deadlines or signals, and no closed/successful outcome exists before real reaping/ECHILD/pipe closure. Later real isolated Linux tests must demonstrate these properties, including worker-exit-before-timeout, rather than testing only timeout-then-worker-exit.

### R3 — P2: new compilation tokens have no defined exported custody table

Location: candidate lines 175–189 and 316–362 (compiled owner and exact new schemas/aggregate joins).

The new compilation record refers to module/spec/loader/compiled-owner tokens, entry tokens and exception tokens, requires rejection of dangling references, and correctly calls for reuse of the existing live object-token registry. Its exact root fields contain neither an object table nor a reference to an immutable post-input object-table projection. The referenced `input_evidence_sha256` is upstream of compilation and cannot contain owners that do not yet exist. The proposed compiler-job record supplies process identities, not worker object custody.

Actual collector behavior makes this a concrete integration gap: `_RuntimeObservation.token` allocates a live token, but `_RuntimeObservation.payload` exports only tokens referenced in its own journal (`native_bootstrap.py:2099–2107,2110–2138,2170–2173`). Merely calling that registry from a new compilation collector does not add a row to the existing input evidence. Conversely, inserting new `build_entered`/`owner_registered` events into the old runtime journal fails its fixed kind grammar (`runtime_custody.py:647–648`). The old parser also rejects unreferenced object rows and validates original first occurrence/adoption links (`:328–341,953–956`). The current collector emits runtime-custody **v2** and preserves v1 compatibility; section 7's reference to accepted "v1 evidence" should not imply reverting or silently widening that accepted grammar.

Trigger: the first successful generated module receives a token after the immutable input cut. A strict consumer following the stated input-record join has no object row for it and must reject the new success record; a consumer accepting any positive token instead loses the promised dangling-reference and guardian-adoption checks. Failure exception tokens have the same problem. Naming an unspecified test/report aggregate does not define where those rows come from, their cut, or how they bind actual retained resources.

Required repair: freeze a bounded descriptive projection for new tokens from the **same** live registry, without creating a second live identity map. Define where it is serialized, exact row fields, first-occurrence and lifecycle-adoption references, how reused upstream tokens versus newly created tokens resolve, and the immutable observation cut/digest used by joins. Define event token meaning/nullability and reference rules, including partial-return/exception prefixes. Specify how the compiler source-session phase, existing source/import audits and runtime checkpoint selection compose after `validate_native_input` reaches `input_validated`; merely appending a separate success record cannot bypass those checks. Keep the original input record and accepted v1/v2 semantics intact; a new downstream table/record or explicit versioned projection can carry the additional facts. No serialized table is live authority.

Proof after repair: independently specified legal positive and partial-failure fixtures resolve every new/reused token to its exact recorded owner/adoption occurrence. With input/lifecycle facts fixed, missing object rows, wrong first occurrence, wrong guardian sequence, cross-session references, module/entry swaps, reused exception tokens and invented post-cut tokens must reject. Test the real collector projection, parser and source-session continuation from the actual input-validation boundary, not a hand-built table accepted solely by itself.

## High-risk areas checked and nonblocking implementation obligations

The design's core route is appropriately bounded. The three names, source owners, entry orders and `-O2` flags agree with actual loaders at `cuda_scene.py:923`, `cuda_lbvh.py:1112` and `cuda_direct_light.py:1688`. It retains original kernel bodies/generator, limits initial eligibility to private fresh outputs and keeps later cache reuse an explicit future decision. Pinning `TORCH_CUDA_ARCH_LIST=9.0` avoids the installed loader's default visible-device architecture discovery. Scope observation and target/ambient restoration remain necessary new native proofs; no existing input-only result proves them.

The direct-loader adaptation addresses a real bypass: installed `_import_module_from_library` constructs a spec, calls `module_from_spec`, executes and returns without ordinary finder admission (`cpp_extension.py:2628–2642`). A distinct retained direct ledger is appropriate. Implementation must retain the delegate's returned module **inside the create boundary before any subsequent fallible importlib metadata initialization**, as the current external wrapper already does; simply retaining after `module_from_spec` returns is insufficient. Hook restoration must preserve any earlier exception and cannot restore session health. These are applications of the candidate's existing retention requirements, not a request for a second generic loader.

The PCH absence policy matches the pinned removal branch (`:1855–1863`): it can be a no-op only under the explicit lifetime no-writer premise, including symlink handling. Implementation must not manufacture that premise by deleting shared files. The full compiler/header/library/environment inventory remains a pre-execution packet obligation, not already verified by the Torch label or these source reads. Successful `verbose=False` Ninja stdout is captured internally and discarded by pinned `_run_ninja_build`; when implementing the promised command evidence, distinguish emitted build.ninja rules from actually observed executed-command output and explicitly provide the required capture. Do not label inferred rules as executed subprocess observations.

Subreaping is a relevant, process-local addition to the dedicated supervisor, not an unrelated service. Linux documents reparenting to the nearest living ancestor subreaper and makes those orphans waitable by it. Under the stated one-tree/no-escape/no-extra-child model, original-worker reaping plus ECHILD with correct SIGCHLD behavior is a plausible closure argument stronger than an empty process snapshot. The proposed separation of direct-child waits from descendants normally reaped by their own parent is correct. The documented API does not itself establish the actual host/toolchain premises. [Linux subreaper interface](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html), [Linux wait interface](https://man7.org/linux/man-pages/man2/waitpid.2.html).

The libc adapter is explicitly proposed here; it is not assumed authorized by the older gap route. Confining it to a dedicated supervisor leaves captured-worker ctypes policy intact. Code preflight must freeze the actual prctl ABI, errno handling and the supported means of verifying SIGCHLD/SA_NOCLDWAIT, rather than treating `signal.getsignal` alone as a readback of all native sigaction flags. Python 3.12 documents pidfd/wait primitives; actual host availability, proc visibility and packaged-tool descendant behavior still require the stated Linux tests. [Python process APIs](https://docs.python.org/3.12/library/os.html#os.pidfd_open).

Normal compiler COMPLETE interception occurs before core acceptance/DISARMED, and the contract forbids unknown residuals from passing. A corrected fixed sequence table is essential to make that control real. The ECHILD proof applies to whole-job exit, not the live-worker normal segment check; the candidate appropriately distinguishes them. Incomplete OS outcomes retain custody and cannot become cleanup permission. Keep scans and output draining nonblocking relative to the existing deadline loop; inherited-pipe hangs and observation errors must not starve termination attempts. No native destructor cleanup or second emergency is admitted on failed compilation.

No additional blocking scope or maintainability finding was identified. The dedicated supervisor, fixed recipes and one private admission adapter can remain proportionate to G1. Preserve one transport/core and one object registry; do not turn the repair into arbitrary command execution, a daemon, cgroup provisioning, an offline publisher plus worker JIT, or a copied renderer/build generator. The stated losing-lane and adapter removal conditions are appropriate.

## Executed verification and limits

Read-only local verification used explicit UTF-8 Get-Content, rg/rg --files, Git status/rev-parse/diff --stat/diff --check/hash-object and Get-FileHash. Required governing documents and relevant actual source chains were inspected; large output was revisited in smaller ranges. An initial guessed root REVIEW_PRINCIPLES.md path and a guessed runtime_observation.py path did not exist; corrected reads used collab/REVIEW_PRINCIPLES.md and the collector in native_bootstrap.py. No result depends on those failed reads.

I executed this outside-repository probe with local Python 3.14 and bytecode disabled:

```text
C:/Python314/python.exe -B ../c4_assets/compilation_admission_design_review_v1_probe.py .
```

It imports only actual CPU protocol/core modules, launches no child, sends no signal, and asserts no NumPy/Torch/Warp import. Exit code 0. Output:

```text
Six unchanged input guards consume 0..5; next compilation guard is 6.
Candidate first compilation ARM 7 rejects: ARM is out of order
Original deadline after modeled worker exit cannot append core kill request: no event may follow observed exit
Candidate working bytes SHA256: 0af9d96d9b97b36c8d9a8e59925b84b1eacf05005f00b723de36405a1fede6db
PASS: two CPU state counterexamples; no native/process-tree proof claimed.
```

Probe SHA256: `d719a7a8fb5788242a85b8423e8a9aabc2bce3264544a6fb09feeab89f1974a4`. The probe's `observe_exit` is intentionally a modeled OS observation used only to test the actual state API; it is not proof of real reaping, descendants, pidfds or subreaper behavior.

Source-only primary SSH (`ssh -o BatchMode=yes -o ConnectTimeout=8 tileops-dev`) used Git, sha256sum, readlink and uname. Independently observed:

- Runtime tree `/data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip`: clean `fc8618268dd8f3fb10e29e32626219500751c41a`.
- CPU tree `/data2/shared/ang.gao/robot_sim_c2d3_array_join_wip`: clean `4da1fa75634d716d4018d616327c84096e047432`.
- Host kernel `6.8.0-85-generic`; gcc/g++ resolved names match the proposed `/usr/bin/x86_64-linux-gnu-{gcc,g++}-13` paths. No compiler was invoked.
- Installed cpp_extension source matches local snapshot SHA256 `52ba4de5deedd894e2bdafc9b7f40526dbc2cc1e4c560bbae3b38d4a5dcab0c6`.
- Local pinned Warp context snapshot SHA256 `eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee` verified; no Warp execution or admission review is claimed.

No fallback host, remote interpreter, compiler/Ninja invocation, backend import, native observer/getter, GPU query/reset, process signal, package install or remote write occurred. Primary documentation reads covered the Linux subreaper/wait interfaces, Python 3.12 process APIs and pinned Ninja 1.13.1 subprocess source; the latter uses a separate process group for ordinary command children. [Pinned Ninja source](https://raw.githubusercontent.com/ninja-build/ninja/v1.13.1/src/subprocess-posix.cc).

The coordinator also supplied `c4_assets/remaining_c2d_host_toolchain_paths_v2.md` during review. I read it as source-inventory preparation only: its gcc helper roots, Conda lib fallback and nvcc.profile observations do not constitute an independently completed toolchain manifest or change this verdict.

All tool commands returned synchronously; no reviewer-owned worker/supervisor/compiler/background job was created. The local CPU probe exited 0 and cannot create descendant jobs. Shell/SSH/git utility PIDs were not separately inventoried, so this is bounded process hygiene, not exhaustive OS/GPU residual attestation. Only this report and its CPU probe were written, both under `C:/Users/AG/新建文件夹/c4_assets`, outside the candidate.

Real process-tree tests, actual compiler/toolchain closure and command capture, all three cold builds, direct native create/exec/adoption failures, PCH lifetime isolation, target1/ambient0 restoration and final GPU/OS residual proof were not executed and are not waived. They belong after repaired design approval and exact native code preflight, with the required independent whole checkpoint review. The current findings are design/code-interface conflicts, not toolchain/environment failures.

## Scope and next checkpoint

Repair R1–R3 in a versioned design and submit it to a fresh independent review, preserving this report unchanged. Do not change accepted protocol semantics or input evidence merely to accommodate the shifted table or missing downstream projection. The report itself authorizes no implementation/native run.

The intended future checkpoint remains CUDA G1 only, ending at `compiled_cuda`; Warp admission still needs its own concrete definition. R_setup, one-slot optical construction, ten actual warmups, R_final, final graph/seal/barrier, C2e/composition and full parent C2d/C4 acceptance must continue separately. Product PREPARE/submission, measurement/sample/result authority, a3b, R2/S/host, shipping, performance/backend selection and other later gates remain closed. Accepted C2d3 and its original narrow scope remain intact.
