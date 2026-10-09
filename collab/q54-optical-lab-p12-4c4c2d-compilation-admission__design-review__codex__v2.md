# Independent design re-review: bounded CUDA compilation admission v2

Initiative: Q54 / P12.4 / C4c2 / remaining parent C2d
Stage: independent design review
Author: codex, independent reviewer /root/compilation_admission_design_v2
Version: v2
Date: 2026-10-09 (Asia/Shanghai)
Status: in_review — review complete; changes requested
Related Files: compilation-admission design-request v2 and unchanged design-review v1; runtime-integration design-request/review v3; remaining-setup gap-scan v1; compilation-job gap-review v1; runtime-adoption design-request v2; guarded-custody repair-design v2; supervised-waits design-request v3
Owner Summary: Request changes for one remaining P2 evidence-contract ambiguity. The corrected wire table and independent post-worker-exit obligation close R1/R2 at design level. R3 now supplies the missing registry projection, observation cuts and source continuation, but does not freeze an independent role binding for successfully admitted generated objects. Root retention is appropriate for partial failure custody; it cannot by itself supply the promised successful module/entry substitution checks. No native or compilation execution is approved.

## Identity and verdict

**Verdict: Request changes.** One blocking P2 finding follows. This is a design verdict, not a failure of an implemented v2 parser or a demonstrated live native bypass.

Fresh reviewer session `/root/compilation_admission_design_v2` received the bounded packet and repository access without the author conversation. I did not author or modify the candidate, edit implementation, or delegate review. AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md were read in full; the Chinese principles were reread explicitly as UTF-8 after the shell's initial default-decoding display was garbled. I reviewed the full candidate and its v1 delta, the prior findings/repro, governing contracts and affected source paths. A later coordinator message highlighted the successful-role versus partial-retention question; it was treated as a question to verify, not evidence or a requested verdict.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Actual base: `739c03edb76509eb93bacf8caff93185685157e8`.
- Frozen reviewed HEAD: `0012ca067f0e9e6a7f613826331b86ed75674fd2`.
- Sole base-to-HEAD change: `collab/q54-optical-lab-p12-4c4c2d-compilation-admission__design-request__codex__v2.md`, 673 added lines.
- Candidate working-byte SHA256: `442cc1d585c7a7da9da7c945dc80e1729f73a79ac59bd4c6eef687f0afab5da4`.
- Candidate Git object: `5aa4b5d0c660dffaaeb8a86cab9ee8bd093346cd`.
- Preserved v1 review SHA256: `ad73ef8b9498a909e1d177ddb9833b13d46b246ef9a2b2e95ce817bcbca03d5b`.
- Initial and final local porcelain status empty; HEAD remained unchanged; base-to-HEAD `git diff --check` passed.

Accepted whole C2d3 retains its original bounded acceptance: code/test `4da1fa75634d716d4018d616327c84096e047432`, production `f1ca45a`, original native execution `fc8618268dd8f3fb10e29e32626219500751c41a`, review-v4 archive `4c20cac`. The older gap report's production-identity wording is historical, not authority to replace the later source-pin repair. This review does not reopen whole C2d3.

## Finding

### R3-continuation — P2: successful object roles lack the independent binding needed by the required substitution checks

Location: candidate lines 474–499 and 528–550, especially 488–499; required proof at 624–628.

The added projection correctly establishes token existence, first occurrence and retention through the exact adopted root. However, `guardian_sequence` may be null for **any** continuation object with a root-retention edge. Only the root has mandatory direct adoption. The text does not restrict that null case to partial/failure-only objects, require role/recipe-specific direct adoption for successful objects, or define another independently anchored role record. `compile_owner_retained` contains only root token, owner token and slot. It has no recipe, spec/loader/module role, entry name, or successful-admission identity.

The later `compile_step` is both the first declaration of these roles and the source from which the recipe snapshot is derived. The phrase "role-specific events below" does not add an independent check: these are precisely the fields being checked. The design also leaves the "expected role/operation/scope" values unfrozen for non-null successful-object adoptions. It therefore allows materially different implementations: one makes successful generated owners pass distinct lifecycle-backed role checks; another treats any correctly retained token as a legal successful binding.

Concrete countermodel: retain a real spec as token101 in slot0 and a real loader as token102 in slot1 before `create_entered`; both object rows have null guardian_sequence, legal first-occurrence indices and valid root links. A baseline `create_entered` binds spec101/loader102. Swapping those two fields in that step and all its repeated/derived bindings still satisfies every stated independent retention, first-occurrence, distinctness, phase and guard fact. Input, complete lifecycle, source facts, object rows and retention occurrences need not change. Likewise, after two real scene entries are retained as104/105, exchange their tokens under the two fixed names in `entries_bound` and its repetitions/derived entries. The ordered names remain correct and both owners remain retained, but no separately anchored entry-name relation distinguishes the result. A similar problem exists for substituting a role-free retained object into a successful module/compiled-owner claim.

The outside-repository v2 probe models this exact projection/step slice and accepts the baseline and both swaps with fixed independent facts. It is deliberately **not** a complete record parser, production-code regression, or proof that v2 code currently has this bug: no v2 implementation exists. It demonstrates the missing discriminating information in the frozen schema. The design's separate live identity audits can still be implemented correctly; this finding does not claim that parsed bytes can mint authority. Nor does it demand authentication against a wholly rewritten consistent execution history. The requirement at issue is the explicit fixed-input/fixed-lifecycle mutation test and the claim at lines497–499 that a retained but unrelated token cannot serve as a generated role.

Why the stated checks miss it: root slots prove retention, not role; first occurrence proves chronology, not role; repeated bindings and derived recipe rows agree with the same substituted declaration; source/environment digests and guard joins carry no association from the retained token to an entry name. Checking collector identity against the same root slot confirms which object is held but does not export a second fact that the parser can join to its asserted role.

Required repair: freeze successful admission's exact role/recipe/entry-name anchor and the parser join to it, while preserving root-only retention for failed or not-yet-admitted returns. A small compatible choice is mandatory direct guardian adoption for successful spec, delegate, module, each named entry and CompiledCudaOwner, with explicit distinct role/operation/scope mappings tied to the recipe and fixed entry name; require those adoptions before each corresponding successful step. Another precise existing-mechanism solution is acceptable if it supplies an independently fixed relation and its exact schema/checks. Do not add a second live identity registry, infer roles from Python type strings, or require successful direct adoption merely to keep a partial returned object alive. State what may remain null in failure prefixes and how reused upstream objects differ from newly generated roles.

Proof after repair: keep independently collected input, lifecycle and retention/admission anchors fixed. Demonstrate a legal success and partial-create/exec/adoption failure, then reject coherent spec/loader and named-entry substitutions even when all later repetitions and derived recipe rows are updated consistently. Include substitution of a valid unrelated retained/input token, wrong recipe/role/entry operation, wrong direct-adoption sequence and cross-session owner. Exercise the actual collector/source continuation/parser together when implemented. Test that failed adoption still holds the original returned object through the root and creates no successful admission event. These are bounded CPU custody/parser obligations, not a reason to run a compiler before code preflight.

## Prior findings and full-design assessment

**R1 is closed at design level.** Sections2/6 now use input0..5, compilation6/7/8 and terminal9, one shared immutable table, and rejection of unknown/skipped/reordered phases. The actual fixed input driver still has exactly six setup operations (`tests/support/native_runtime_process.py:879–897`). An independent execution through the production core accepted nine normal guards0..8 followed by emergency9. The preserved v1 probe still rejects the old skipped ARM7 after six guards, as expected. New integrated interception tests remain implementation work; a standalone core trace does not prove a future wrapper intercepts each COMPLETE.

**R2 is closed at design level.** Sections6/7.4 explicitly preserve the earliest operation or pending-ACK obligation independently, record outer expiry, and distinguish live-worker core termination from child-only termination after `process_exited`. The latter leaves the core immutable. The new event grammar has sticky failure, source/deadline, no repeated draining/expiry/attempt, clock faults, failed OS observations and actual closure prerequisites. Production-core probes confirm that both saved operation and earlier ACK deadlines disappear from the core on modeled exit while its final evidence remains byte-stable; the outer record is therefore necessary and is now specified. These are modeled exit observations, not real subreaping proof.

**R3 is substantially repaired but not closed.** The same `_RuntimeObservation.token` registry is retained; the immutable v2 input bytes/cut are upstream; global journal indices, new projection rows, full lifecycle prefix, components and source-trace cuts are defined. The downstream parser no longer attempts to feed new kinds into the old fixed grammar, and final selection does not mislabel old input evidence as current compilation evidence. Source phases follow `input_validated` and compile DISARMED boundaries. The remaining successful-role issue is the sole blocking finding above.

Other high-risk areas inspected:

- The three recipe names, flags and entry order match existing CUDA loaders. Moving scene/direct-light constants into sole pure source modules, retaining LBVH's existing source module and delegating compatibility wrappers to one recipe avoids importing `cuda_direct_light`'s timeline/device compatibility graph just to retrieve strings. Byte-preservation and unchanged legacy behavior remain mandatory code checks. No kernel/generator copy or extra rendering implementation is proposed.
- Pinned Torch inserts headers and generates the requested direct-light pybind body, uses FileBaton/ExtensionVersioner, performs a synchronous Ninja build and then calls its direct loader helper. Intercepting that one admitted function is a bounded response to the actual finder bypass. The expected unsuffixed name, fresh private directory, no prior module/cache owner, exact original hook and single-entry session constraints prevent ordinary cache/hook paths from silently supplying admission. The adapter restoration and swallowed-exception rules preserve poison.
- Returning from `module_from_spec` is too late for first custody. V2 now correctly retains the delegate-create return before fallible importlib metadata initialization, as the existing external wrapper does at `native_bootstrap.py:544–575`. Create-without-return preserves available roots/exception and UNKNOWN scope; later exec/entry/adoption/restoration failures retain partial owners without a second sync or ordinary unload. The root mechanism is useful for these failures; the finding does not remove it.
- PCH absence/no-writer policy matches the pinned shared-path deletion branch. Fresh build directories alone would not isolate those files; v2 does not claim otherwise. The trusted manifest must include actual compiler/helper/include/link roots, environment and `nvcc.profile` effects. The outside-repository toolchain-path note remains preparation, not complete closure. The corrected distinction between emitted Ninja rules and sampled executed-process observations matches `verbose=False` discarding successful captured output.
- The job remains a dedicated one-tree, one-waiter supervisor with process-local subreaping, verified pidfds, conservative identity acquisition, no PID-only fallback, and no unrelated process authority. Normal compile COMPLETE is checked while the trusted synchronous builder has returned and the worker is blocked, before core acceptance. Whole-job closure instead requires worker/adopted-child reaping, ECHILD and real pipe EOF; kill success or empty proc scans do not suffice. Missing capability or persistent OS failure leaves an incomplete retained job.
- Linux documents orphan reparenting to the nearest living ancestor subreaper and subsequent waitability; wait semantics make the SIGCHLD/SA_NOCLDWAIT premise material. V2 names the fixed libc prctl/sigaction boundary and requires installed-ABI validation before native use. This is plausibility under the frozen trusted descendant model, not demonstrated host behavior. Sources: [Linux subreaper interface](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html), [Linux wait interface](https://man7.org/linux/man-pages/man2/waitpid.2.html), [Python3.12 process APIs](https://docs.python.org/3.12/library/os.html#os.pidfd_open). The separate-PGID premise agrees with [Ninja1.13.1 subprocess source](https://raw.githubusercontent.com/ninja-build/ninja/v1.13.1/src/subprocess-posix.cc).
- Exact-target/ambient restoration is still a new real native proof obligation. The proposed three native-scope sites must integrate with existing nested before/after and leaf-descriptor rules, including the outer compile call's inner create/exec calls. Existing input-only observations do not prove this. No unsupported native boundary may become target-safe by argument alone.
- Scope, timeout, migration and cleanup remain proportionate: one worker JIT route, no offline publisher/daemon/cgroup service, unchanged120s operation and1s ACK budgets, no optical execution, no cache reuse across sessions, no source closure widened to the entire compatibility graph. The direct adapter and losing lane have stated removal conditions. No additional blocking scope or resource-lifetime issue was identified.

## Executed verification and reproducibility

Local inspection used UTF-8 Get-Content with bounded ranges, rg/rg --files, Git status/rev-parse/branch/hash-object/diff --stat/diff --check and Get-FileHash. Large outputs were narrowed for the governing sections and candidate/source areas used in the conclusions. `git diff --no-index` between v1/v2 returned the expected status1 for differing documents and an LF/CRLF notice, not a failed test or a mutation. No source checks imported backend modules.

Commands executed from the repository:

```text
C:/Python314/python.exe -B ../c4_assets/compilation_admission_design_review_v1_probe.py .
C:/Python314/python.exe -B ../c4_assets/compilation_admission_design_review_v2_probe.py .
git diff --check 739c03e..HEAD
```

The original probe is unchanged, SHA256 `d719a7a8fb5788242a85b8423e8a9aabc2bce3264544a6fb09feeab89f1974a4`. It exited0 and reproduced the actual old skipped-sequence and forbidden post-exit core-event counterexamples. Its printed candidate digest is explicitly the retained **v1** candidate, not v2. The actual implementation is unchanged, so these old API counterexamples correctly still reproduce; v2 repairs the proposed usage.

The new probe is `C:/Users/AG/新建文件夹/c4_assets/compilation_admission_design_review_v2_probe.py`, SHA256 `79ee58260c9431ab223e550378a5d51b0ebf6440a6a694e06efa22fa9aac8b93`. Exit0, output:

```text
ACTUAL CORE: input0..5 / compile6..8 / terminal9 is a legal wire trace.
ACTUAL CORE + MODELED EXIT: saved operation deadline 121000000000 ; exited trace remains immutable.
ACTUAL CORE + MODELED EXIT: saved ack deadline 122000000000 ; exited trace remains immutable.
DESIGN MODEL: baseline satisfies role-free projection/step slice with fixed input/lifecycle/retention.
DESIGN MODEL: spec/loader swapped satisfies role-free projection/step slice with fixed input/lifecycle/retention.
DESIGN MODEL: same-recipe entries swapped satisfies role-free projection/step slice with fixed input/lifecycle/retention.
No v2 production parser, compiler, process-tree, ABI, signal or GPU proof claimed.
```

The probe imports only the actual CPU protocol/core plus stdlib. It asserts no Torch/Warp/NumPy import, launches no subprocess and sends no signal. The role model is a consciously limited ambiguity witness, not a test implementation offered as replacement production evidence. No pass count from an unrelated suite is claimed.

Source-only primary SSH used `ssh -o BatchMode=yes -o ConnectTimeout=8 tileops-dev` with Git, sha256sum, readlink and uname. Transport succeeded, so no fallback was used. Independently observed:

- Runtime tree `/data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip`: clean `fc8618268dd8f3fb10e29e32626219500751c41a`.
- CPU tree `/data2/shared/ang.gao/robot_sim_c2d3_array_join_wip`: clean `4da1fa75634d716d4018d616327c84096e047432`.
- Installed cpp_extension and local `remaining_c2d_pinned_cpp_extension.py` match SHA256 `52ba4de5deedd894e2bdafc9b7f40526dbc2cc1e4c560bbae3b38d4a5dcab0c6`.
- gcc/g++ resolve to `/usr/bin/x86_64-linux-gnu-gcc-13` and `/usr/bin/x86_64-linux-gnu-g++-13`; host kernel reports `6.8.0-85-generic`. No compiler was invoked.

## Limits, hygiene and exact scope

No compiler/Ninja execution, remote interpreter, backend import, native observer/getter, native-OS probe, GPU query/reset, process signal, installation, remote write or product operation occurred. No reviewer worker/supervisor/compiler/background job was created. Inspection shells, SSH utilities and the two local CPU probes returned synchronously; the probes do not create children. A final read-only Win32_Process query restricted to Python command lines naming these two exact review probes found zero residuals. Utility PIDs were not individually sampled, so this is bounded process hygiene, not exhaustive OS/GPU attestation. Candidate status remained clean. Only the v2 scratch probe and this report were written, outside the repository.

Full installed toolchain/header/library closure, libc ABI/readback, actual pidfd/subreaper behavior, separate-PGID/double-fork/leader-first-exit/pipe-hang process cases, packaged compiler cold builds, direct native loading, target1/ambient0, PCH lifetime invariants, partial native failures and actual owned OS/GPU residual proofs remain unexecuted. The candidate explicitly gates them on exact code preflight and later whole-checkpoint review; their absence here is not a claim of runtime success and is not an additional design blocker.

This report requests only the bounded successful-role/custody evidence repair and fresh independent design re-review. It approves no implementation of this exact candidate and no native/compilation execution. Accepted C2d3 remains accepted only in its original scope. CUDA compiled_cuda foundation itself is not accepted code; Warp admission, one-slot construction, C2e, ten warmups, R_setup/R_final, final graph/barrier, full C2d/C4, optical sample execution, a3b/result/R2/S/host payload, product activation, performance and backend-selection gates remain closed.

The complete report may be archived verbatim as evidence. Archival cannot alter this verdict, the inspected identities, the finding, or the closed gates.
