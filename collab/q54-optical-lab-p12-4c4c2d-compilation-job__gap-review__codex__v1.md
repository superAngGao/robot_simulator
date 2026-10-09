# Compilation job ownership and generated-module admission: bounded G1 refinement

Initiative: Q54 / P12.4 / C4c2 / remaining parent C2d
Stage: independent read-only gap review; next design definition only
Author: codex, independent session /root/compilation_job_gap_v1
Version: v1
Date: 2026-10-09 (Asia/Shanghai)
Status: in_review — gap review complete; no design or implementation approval
Related Files: runtime-integration design-request/review v3; remaining-setup gap-scan v1; runtime-adoption design-request v2; guarded-custody repair-design v2; supervised-waits design-request v3; source locations below
Owner Summary: G1 remains a bounded design prerequisite. Ninja invalidates single-process-group cleanup. An offline producer does not remove compiler ownership or producer custody obligations, and its proposed success-without-COMPLETE needs an explicit producer outcome policy. On the inspected facts, worker JIT is the smaller next design candidate: preserve normal guarded completion, define one generated-artifact/direct-loader admission path and session-member supervision. This is a recommendation to draft and independently review, not permission to compile.

## Identity and verdict

**Verdict: Request changes, restricted to readiness of the next compilation checkpoint definition.** There is enough evidence to write its bounded supplement; neither route is approved for implementation or execution. This does not reject or reopen accepted whole C2d3.

Local clean HEAD: `4c20cac91214a60bc191e85272a1cb5055a2114d`, branch `implementation/c4-supervised-waits`, repository `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`. The packet's accepted whole-C2d3 identity is `4da1fa75634d716d4018d616327c84096e047432`, production `fc8618268dd8f3fb10e29e32626219500751c41a`, acceptance archive `e5299e3`, unchanged whole-code-review v4 archived in this local HEAD. I did not repeat that acceptance review. Source-only SSH independently found the primary runtime-custody tree clean at `fc8618268dd8f3fb10e29e32626219500751c41a` and array-join tree clean at `4da1fa75634d716d4018d616327c84096e047432`.

I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully (rereading the Chinese principles as UTF-8), then the relevant governing sections and actual affected sources. No candidate authoring, implementation delegation, compiler/backend/native execution or process signal occurred. Author preparation/outline scratch is not an approval candidate and was not used as authority.

## Source-backed facts that change G1

1. **Torch PCH is outside the build directory.** Pinned `cpp_extension.py:1855–1863,1883,1990–2002` defaults to `use_pch=False` and calls removal of `_TORCH_PATH/include/torch/extension.h.gch` and `.sign`. The removal checks existence first. `use_pch=True` uses the same shared paths (`:1761–1762,1816–1853`). A new build directory does not isolate those effects. Independent read-only host checks found **both paths absent**, including a symlink-presence check. This is a current observation, not a lifetime guarantee.

   The smallest viable PCH policy may therefore be explicit `use_pch=False` plus frozen absence of exactly these two paths throughout the admitted job, under the existing trusted immutable-deployment boundary. Reject presence before entering `load_inline`; do not remove shared files to manufacture the condition. Also prohibit concurrent writers to those paths in the admitted deployment policy. This permits the existing false branch to be a no-op without a PCH patch. If that absence guarantee cannot be maintained, a reviewed narrowly scoped no-PCH adaptation or isolated deployment root is needed. Neither toggling `use_pch` nor merely setting `build_directory` solves it.

2. **Direct loading remains a genuine admission gap.** `cpp_extension.py:2628–2642` constructs the file spec, calls `module_from_spec` and `exec_module`, then returns the module. The fixed repository CUDA loaders (`cuda_scene.py:923`, `cuda_lbvh.py:1112`, `cuda_direct_light.py:1688`) use this route through `load_inline`. `_ExternalAdmission.find_spec` (`native_bootstrap.py:1433–1512`) admits inventoried files via the finder and holds spec/delegate before create. A generated direct load does not pass through it; scanning `sys.modules` afterwards neither intercepts create nor proves available-root custody. Setting `is_python_module=False` still calls `torch.ops.load_library`; it is not a build-only switch.

3. **Ninja's ordinary commands have separate process groups.** Ninja 1.13.1 `subprocess-posix.cc:94–97` sets `POSIX_SPAWN_SETPGROUP` for non-console commands; `:123–125` spawns their shell. Its interruption cleanup targets those groups, but depends on Ninja executing that cleanup. Killing the Python worker or one inherited PGID cannot substitute for supervision of all compiler members. This conclusion comes from the [pinned Ninja source](https://raw.githubusercontent.com/ninja-build/ninja/v1.13.1/src/subprocess-posix.cc), not compiler execution.

4. **Existing exit state is one process.** `tests/support/guarded_custody_process.py:334–375` starts a worker, polls it and calls `core.observe_exit(code)` or `process.kill()`. `wait_supervision.py:227–241` makes the observed process exit terminal and clears deadlines. That is correct for its existing meaning; it cannot establish compiler-job extinction. Leader-first exit is specifically dangerous: descendants can keep running/writing after that transition and after inherited-pipe EOF assumptions fail.

5. **Warp stays distinct.** Pinned Warp `build.py:40–115` uses its native CUDA compiler entry and device-context loader in process. `context.py` retains module execution owners and cache variants. No Ninja tree should be invented for that CUDA path. Its required additions are exact source/cache/architecture/configuration provenance and retained module/entry owners; native-internal exception safety stays in the accepted dependency boundary.

## Minimal route comparison

| Route | Necessary additions | State-flow consequence |
| --- | --- | --- |
| Worker JIT | Fixed recipe/toolchain/build policy; generated artifact admission; one bounded integration at Torch's direct-loader boundary; compiler-member ownership and exit evidence | Successful compile/admission remains ordinary setup ARM/ARMED → operation/audit success → COMPLETE/DISARMED. Partial failure retains available roots and uses the original failed guard; no second emergency. |
| Offline immutable artifacts | The same producer recipe/PCH/descendant obligations; producer loading custody or an explicitly build-only path; publication after observed producer/job closure; frozen worker inventory/import binding | Worker can reuse fixed-file admission. A producer that emits bytes then exits while still armed needs an explicit producer terminal-outcome policy; it is not existing normal optical setup success. |

**Recommendation:** draft the worker-JIT supplement first, using the existing kernel strings/ABI and normal guarded controller. It avoids introducing a deployment publication lifecycle and special producer ending solely to avoid one bounded loader adaptation. This is a minimal-flow judgment, not a claim that offline artifacts are invalid. Choose offline instead if immutable reusable deployment artifacts are an actual requirement or the bounded direct-loader integration cannot be made maintainable. Do not implement both routes speculatively.

The essential policy choice is **where executable-artifact authority is created**: a deployment publisher before worker startup, or a private worker admission step after supervised generation. The producer location alone does not settle process ownership, PCH, cache eligibility, module identity or failure disposition.

### The proposed offline hard-exit trace

It is physically possible to retain all available roots, emit only descriptive compiler-success bytes, and hard-exit before the original armed deadline without COMPLETE or another ARM. Such bytes prove neither exit nor readiness. But this is **not presently a successful lifecycle ending**: `guarded_lifecycle.py:194–237` completes successful segment bodies, while `:404–424` treats termination with a non-IDLE guard as failed/poisoned. Repair-design v2 does not add a successful compile-only terminal state.

A supplement could explicitly permit a producer-only retained-root terminal outcome and bind artifact eligibility to separately observed whole-job closure. That need not require a new wire role: the existing setup arm and deadline can mechanically protect exit. It does require a defined private transition/outcome and evidence interpretation; a new role is an alternative policy change, not an automatic necessity. Do not fabricate a failed compiler operation to obtain quarantine, count this as normal_complete, reinterpret emergency as a compiler-success role, or mint R_setup/R_final. If the producer instead uses normal COMPLETE/DISARMED and the already applicable terminal path, it must satisfy that path's real scope/coverage rules. That is a different proposed trace, not success under the original still-armed guard.

## Compiler ownership: plausible narrow mechanism, with exact remaining obligations

A dedicated, supervisor-created session is a useful ownership domain because Ninja changes PGID without creating a new session. An anchored SID plus identity-checked pidfds is a reasonable candidate on this Linux host; it is not yet a finished algorithm. Freeze the trusted toolchain's no-daemonization/no-session-escape assumption and actual command closure (including compiler/version probes), not a hostile-toolchain sandbox. No cgroup, daemon, installer or broad process-name/UID kill is justified by the inspected facts.

The next design must settle these interfaces:

1. **Launch/anchor:** establish the exact leader/SID before authorizing any spawn; retain its PID/start identity and handle. Observe leader exit without consuming its wait status until member closure is settled. Python exposes `waitid(..., WEXITED|WNOHANG|WNOWAIT)`; WNOWAIT leaves the child waitable. Freeze ownership of reaping so `Popen.poll/wait`, finalizers or another waiter cannot consume the anchor first. These are documented capabilities, not a host runtime test. [Python process APIs](https://docs.python.org/3.12/library/os.html#os.waitid)

2. **Member identity and closure:** discover only members of that anchored session, validate identity across pidfd acquisition and retain actual handles before acting. Specify handling of exit/reuse during discovery, newly spawned children during cleanup, process-table access failure and handle exhaustion. A pidfd solves addressing of an acquired process; it does not enumerate descendants or prove a snapshot complete. A single empty `/proc` scan or one snapshot followed by signals is insufficient. The draft must explain when spawning has ceased and why its final observation covers all remaining members under the frozen toolchain model.

3. **Signals, exit and reaping:** use only the approved termination policy against verified owned members; preserve separate attempt/error/observed-exit facts. Non-child pidfds do not make their targets waitable children. Distinguish supervisor-reaped leader/direct children, descendant exit observation, and reaping by the trusted OS parent; if complete residual/reaping proof is required, retain observation until it exists rather than inventing descendant return codes. The documented pidfd interface supports exit polling and child waits, not adoption of arbitrary grandchildren. [Linux pidfd manual](https://man7.org/linux/man-pages/man2/pidfd_open.2.html), [Python pidfd signal API](https://docs.python.org/3.12/library/signal.html#signal.pidfd_send_signal)

4. **Deadline through leader-first exit:** retain the original job obligation and earliest absolute deadline independently of the single-worker core's EXITED transition. An outer compiler-job record can truthfully record leader exit while continuing member supervision; do not relabel `process_exited` as “all descendants exited”, delay recording known facts without explanation, or renew the budget after leader exit/EOF. At expiry, termination attempts are not successful closure. Keep incomplete evidence and owned handles if the OS has not delivered exit. This matches the existing no-fabricated-exit rule.

5. **Normal completion boundary:** worker JIT must finish its compiler children and artifact/module audits before admitting the normal completion that permits further setup. Specify how the trusted worker's build return and supervisor's member observations compose, including an unexpected residual at COMPLETE. Do not add arbitrary guard fields or disarm first and check children later. Scope the job envelope to compiler-capable setup; preserve the existing worker protocol and optical guard meanings unless a separately reviewed delta is truly necessary.

The host reports kernel `6.8.0-85-generic`; the pinned environment is Python 3.12. API availability, proc access and the actual membership/exit algorithm still need later bounded validation. The primitives make a no-provisioning design plausible; they do not justify claiming it already works. Any missing capability should be reported, not replaced silently with broad kills.

## Minimal supplementary design surface

Freeze only the following new decisions, reusing existing source custody and lifecycle mechanics:

- **Recipe and filesystem effects:** literal extension names/entry sets, source-string owners and generated header/pybind transformations; architecture and flags; exact compiler, linker, Ninja, headers/runtime roots and relevant environment selection. Pin the intended Conda CUDA 12.8 toolchain instead of the system CUDA 13.2 default. The packet's binary hashes are planning observations, not a completed closure. Define PCH absence or its chosen alternative, build/cache write roots, and eligibility of fresh versus reused artifacts. For the first checkpoint, fresh private build outputs simplify provenance but do not by themselves fix PCH.
- **Generated-module admission:** one production-private adapter at the pinned direct-loader boundary, not caller-injected callbacks or a general import exemption. Reuse standard ExtensionFileLoader custody mechanics: hold exact intent/spec/delegate before create; retain returned module before exec; retain exception/partial owners on failure; bind actual module and fixed entries before returning to a GPU leaf. Define a distinct direct-load ledger if no sys.modules publication occurs, and audit it by exact object identity. Audit the specific adaptation and dependency identity so a cache hit, swallowed import failure or replaced hook cannot bypass it. Do not copy CUDA kernels or create a second general loader.
- **Compiler-job owner and evidence:** the five interfaces above, with a versioned descriptive record separating generation success, admission, normal guard completion, leader exit, member exit/reaping and residual closure. Parsed bytes create no live custody, admission or readiness authority. Publication is an additional interface only if offline is selected.

The already settled mechanisms remain settled: guardian-first imports; held repository source consumption; fixed phase closure; actual target/stream and conservative UNKNOWN scope; pre-return custody within the trusted native boundary; sticky failed armed segments; the sole emergency target-sync terminal; two immutable readiness objects; ten observed warmups; one seal before final barrier. They need integration and tests where affected, not a new architecture decision in this supplement.

## Later proof and limits of this review

Before code approval, prove the chosen path with cold build/admission and any allowed cache positive; wrong recipe/artifact/entry/cache origin negatives; PCH-presence rejection with unchanged shared files; partial create/exec and owner-adoption failures; fixed target/architecture; compiler/lock/pipe hangs and original-deadline enforcement. Process fixtures must include Ninja-style separate PGIDs, leader-first exit, fork during member discovery/termination, reused/disappeared PID and inaccessible-member failures, plus an unrelated sentinel that remains untouched. Observe actual exits/reaping and no continuing output/cache writes. Test missing/failed COMPLETE and producer output-write/exit failure if offline is chosen. These are future proofs, not executions performed here.

Read-only verification used UTF-8 Get-Content/Select-Object, rg, Git status/rev-parse, Get-FileHash, source-only SSH with Git/uname/test/ls/sha256sum, and primary-source web reads. Large initial output was revisited in bounded chunks. One unnecessary `/dev/null` search failed on Windows; no conclusion depends on it. Independently verified local snapshots and installed copies:

| Snapshot | SHA-256 |
| --- | --- |
| remaining_c2d_pinned_cpp_extension.py | 52ba4de5deedd894e2bdafc9b7f40526dbc2cc1e4c560bbae3b38d4a5dcab0c6 |
| remaining_c2d_pinned_warp_build.py | 5993e58b44a916fc12f35afde1cededc59ab1db7df7a7eecd75551191d9197df |
| native_scope_pinned_warp_context.py | eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee |

No test, interpreter on the remote host, compiler/Ninja command, native getter, backend import, install, GPU probe, signal or remote write occurred. All inspection calls completed synchronously; no reviewer-owned runtime/compiler/background job was created. Utility PIDs were not individually sampled, so this is not exhaustive OS/GPU residual attestation. The packet's finished author jobs and clean residuals were not rerun or independently re-certified. Final local and both remote Git checks remained clean at the exact revisions above. Only this outside-repository report was written.

All compilation/render/warmup execution and product/later gates remain closed. Accepted whole C2d3 remains accepted within its original scope. The next authorized action supported by this review is drafting and independently reviewing the bounded compilation supplement; this report itself grants no implementation or execution approval.
