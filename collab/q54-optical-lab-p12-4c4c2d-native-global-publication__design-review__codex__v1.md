# C2d3 native-global publication: independent design review v1

Initiative: q54-optical-lab-p12-4c4c2d-native-global-publication
Stage: design-review
Author: codex, fresh independent reviewer
Version: v1
Date: 2026-10-08 (Asia/Shanghai)
Status: accepted
Related Files: native-global-publication design-request v1 and gap-scan v1; runtime-adoption design-request v2/design-review v2; native-self-alias design-request v1; runtime-integration design-request v3; optical_worker/native_bootstrap.py; optical_worker/runtime_environment.py
Owner Summary: Approve the exact bounded design for implementation. It specifies attainable native-interval observation and direct custody without asserting an unavailable creator relation. No substantive blocking design finding was established. This is not code approval, captured-worker admission proof, whole C2d3 acceptance, or authority to open a later gate.

## Independent identity, snapshot and verdict

Reviewer session: /root/native_global_design_review_v1. This fresh session received the task packet and evidence locations, not the author's implementation conversation. I did not author or modify the candidate, and did not delegate this review. Same-model independent context is not a claim of a separate security boundary or an external Claude review.

Verdict: **Approve**, exclusively for implementation of this design supplement at the exact snapshot below. No substantive design findings or unresolved blocking design gaps were established. The earlier gap scan's Request changes was a request for a representable contract, not an approval inherited by this review.

- Repository: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design.
- Actual branch: implementation/c4-supervised-waits.
- Implementation base: 62cf8ff0b1492163d166e243ffde7347fd4e03d9.
- Frozen and independently checked HEAD: 80387e8d8199e9dc6741924b95ff1aeeae047b45.
- Candidate: collab/q54-optical-lab-p12-4c4c2d-native-global-publication__design-request__codex__v1.md.
- Raw candidate SHA-256: 5c29fad420b9d31a848f8b53df07450b46f0fee0b4e2abe9f38aad7997be8b2c, checked initially and finally.
- Archived gap report raw SHA-256: 2bc5cdbd78947faec562b82d4d10d4ce2c4655425b9812cf9d314012e97590e4, independently matching the original supplied digest.
- Initial and final local working tree: clean. Base-to-HEAD delta contains only the new design and unchanged archived gap report. Protected implementation/tests/AGENTS/shared review documents have no delta. git diff --check passed.
- Primary remote: tileops-dev, 183.166.183.33:40031; /data2/shared/ang.gao/robot_sim_c2d3_binding_wip remained clean at detached baseline 62cf8ff0b1492163d166e243ffde7347fd4e03d9. No fallback host was used.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely. Inspected the candidate and prior gap report, authoritative runtime-adoption v2 contract/approval, native-self-alias supplement, relevant parent runtime-integration v3 setup/evidence/failure/cleanup restrictions, owner P12.4 memory decision v2 and cleanup-ledger follow-up. Inspected actual native_bootstrap loader/finder, external admission state and failure intake, publication reconciliation, file identity and audit, source-session failure and external configuration; inspected runtime_environment's active schema, collision/root/parent and native-selector checks. Author conclusions were checked against original code and raw evidence.

## Prior representational gaps and design closure

**Gap F1, exact-owner relation:** runtime_environment.py:255-287 and native_bootstrap.py:1088-1164 currently require raw creator attributes or an exact creator-self alias. The three globals are distinct fileless modules, and ordinary direct-attribute/GC inspection does not establish those relations. The final audit at native_bootstrap.py:1226-1250 correctly rejects unrecorded modules; admitting matching names alone would not repair provenance.

Candidate sections 1, 3 and 4 instead define a separate native_global outcome, with one finite frozen name/observer/exec/phase table and direct session/guardian custody. The deployment inventory remains the sole file/hash authority. No fictitious file loader, attribute path, creator or hidden native owner is asserted. Rows cannot collide with other admission outcomes, use a Python/package observer, widen from discovery or enter through worker IPC. Version v4 explicitly rejects old active profiles while preserving historical files. This closes F1 at the design-contract level; it does not prove the proposed deployment rows in a captured worker.

**Gap F2, ambiguous nested return deltas:** the fresh CPU reproduction below reconfirms that cython_runtime is visible at the zlib request during _bounded_integers.exec, and _cython_3_2_8 at the _common request during the nested bit_generator.exec. Assigning everything to the next child return or requiring all not-yet-admitted keys to remain absent at every child entry would fail.

Candidate section 4 explicitly distinguishes first observation from later admission. It adds a common request-entry checkpoint before the existing stdlib/external/repository split, tracks identity-bound external delegated frames including Python bodies, records the preceding observer-exec absence, and allows first observation only at the specified request-entry/native-exec-return grammar. Matching ancestors cannot substitute for the current frame. Pending identity persists across nested work and becomes admitted only at its own successful exec return after wrapper/publication/sticky-failure checks. This closes F2 at the design-contract level without converting an enclosing-interval observation into instruction-level causation.

## High-risk areas reviewed

1. **Implementability and one production path.** The existing captured finder resolves stdlib before _ExternalAdmission.find_spec, so the common finder entry is the necessary location for the observed zlib request. The design selects that location and extends the existing admission and file wrappers; no second finder, materializer, native interposer or dependency patch is needed. Current name-only admission.stack does not already supply the proposed frame authority. The design requires replacing/extending that execution bookkeeping with exact issued frame bindings, rather than treating current names as sufficient evidence.

2. **Wrapper versus delegated execution.** Sections 4's frame states, suspension on finder handling and wrapper validation, normal-return marking before fallible checks, and rejection of first observation during wrapper work prevent a wrapper-triggered import from borrowing native execution status. Normal nested external imports remain legal. The normal-return observation is the explicitly allowed endpoint of the just-returned exact native call; it is not permission to treat later wrapper checks as executing native code. These distinctions need direct production-entry tests.

3. **Custody precedes eligibility and validation.** All newly available finite-key values, including invalid/replacement values, must enter the already-owned graph before fallible metadata validation/adoption/recording. Original objects remain held too. Exact ModuleType/raw metadata checks are only format checks. Returned native roots are retained before later wrapper work, and pending records bind actual object/frame identities. The private registry and immutable binding checks address same-metadata copies, removed/rebound records and loader/spec/rule drift. Parsed rows and tokens remain descriptive, not constructors of authority.

4. **Successful completion and audits.** Required rows of an executed observer must be present and bound to that frame before file completion. An unexecuted observer may leave its rows absent; it cannot gain an observation at audit. Empty stack/no pending, persistent object and metadata identity, exact admitted records and read-only repeated audits are explicit. Existing unknown-publication rejection and attribute/self-alias relations are preserved. No extra module permission follows from another extension reusing an unchanged admitted object.

5. **Failure and terminal reporting.** Existing _ExternalAdmission.fail at native_bootstrap.py:648-684 retains late deltas on every intake while notifying terminal failure once. Source-session failure marks the guard and selects termination. Sections 4/5 preserve that separation: swallowed nested failure cannot admit late objects, retry, COMPLETE, ARM, synchronize or release. Native failure remains conservative. The stdlib branch currently deliberately remains usable after poison; the candidate explicitly requires the new common checkpoint to preserve terminal stdlib reporting without normal adoption. This is a critical regression obligation, not a new liveness exemption.

6. **Concurrency and trust limits.** The configured CPython/import-thread identity and rejection of observable foreign-thread entry/checkpoint reentrancy are stated. Nested imports are distinct from checkpoint recursion. Background mutations or transient replace/restore between checkpoints remain trusted dependency obligations. The design claims an observed enclosing external interval and subsequent custody, not identification of every Python/native callback or instruction executed inside that interval. It does not claim that the GIL proves native safety or that quarantine reverses unsafe native cleanup. Evidence of such cleanup still blocks admission under the parent contract.

7. **Evidence and replay.** Section 6 specifies exact component keys, calls and events, operation/checkpoint enums, absence facts, pending/admitted identity continuity and issuer-owned object tokens. A call's terminal snapshot is descriptive; completed means delegate return plus wrapper success. Failed/started calls cannot authorize admitted events. Whole-stage custody must additionally bind actual file call/return history, guards, session/lane/target and terminal outcome. This component alone is deliberately insufficient to prove a completed runtime. Negative replay must apply the live-state invariants too: deleting an observer's required events, leaving pending under a completed observer, admitting a parent before its nested child has returned, or fabricating a successful aggregate cannot pass. These are implementation obligations from sections 4/6/7, not permission to accept a self-consistent event list in isolation.

8. **Scope and cleanup.** New work is untimed setup custody. No measured sample, R_setup/R_final, sealed optical graph, active arena lease, renderer result or product IPC is introduced. Environment profiles/selectors/pins/ledger paths must be versioned together. Parent P12.4e selection/removal obligations remain in force; the extension does not authorize a permanent second importer or backend bridge. No numerical resource/performance gate is invented or relaxed.

## Independent observations and commands

Local inspection used Get-Content with explicit UTF8, Select-Object ranges, rg, Get-FileHash and these Git checks:

    git status --porcelain=v1
    git branch --show-current
    git rev-parse HEAD
    git diff --name-status 62cf8ff0b1492163d166e243ffde7347fd4e03d9 HEAD
    git diff --check 62cf8ff0b1492163d166e243ffde7347fd4e03d9 HEAD
    git diff --exit-code 62cf8ff0b1492163d166e243ffde7347fd4e03d9 HEAD -- optical_worker tests AGENTS.md collab/README.md collab/REVIEW_PRINCIPLES.md

All revision, protected-path and diff-check results matched the packet. Initial default-encoding Chinese text was reread as UTF8. Large combined reads were narrowed where truncated. One rg glob-style Windows path was invalid and was replaced by rg --files filtering; this was reviewer tooling, not a product failure.

All remote Python used /home/ang.gao/miniforge3/envs/robot_sim/bin/python, -I -S -B, timeout 30, over ssh -o BatchMode=yes -o ConnectTimeout=10 tileops-dev. No timeout fired. No site.main, package install, GPU runtime initialization or candidate edits occurred.

**Fresh CPU ordering reproduction:** inspected the preserved probe source before executing it. Source C:/Users/AG/新建文件夹/c4_assets/cython_gap_review_probe_v1.py has SHA-256 a827cdf4affd89b8b5444786f39de6d22597995834204dd0b649e6e6ac603dad. It adds only the fixed existing site-packages root, observes request entry, wraps the installed standard extension delegate, strongly retains observed globals and asserts Torch/Warp were not imported.

    Get-Content -Raw -Encoding UTF8 ../c4_assets/cython_gap_review_probe_v1.py | ssh -o BatchMode=yes -o ConnectTimeout=10 tileops-dev 'timeout 30 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -'

Fresh output was preserved at C:/Users/AG/新建文件夹/c4_assets/native_global_design_review_v1_probe_output.txt. Actual interpreter CPython 3.12.13, PID 4093086, timeout parent 4093085, successful exit, success_no_backend printed. It observed:

- Sequence 155, finder_request_entry(zlib): cython_runtime first visible in numpy.random._bounded_integers.exec_module.
- Sequence 164, finder_request_entry(numpy.random._common): _cython_3_2_8 first visible in numpy.random.bit_generator.exec_module, nested inside _bounded_integers.exec.
- Sequence 306, finder_request_entry(scipy._cyutility): _cython_3_2_5 first visible in scipy._lib._ccallback_c.exec_module.
- Earlier observed identities remained unchanged at subsequent printed boundaries. Each queried direct owner attribute list was empty and direct GC-referent match false; neither is a proof that native internal ownership does not exist.

Fresh binary SHA-256 values matched the historical probe: _bounded_integers 5b981bd3ee5d46dcfa3f1972838b3a5482d2c28d09b5126e976ebdbda9d8cd44; bit_generator f7466ade9efe0c03f7c9f568130ea0297e651107d7920dc0d335ba23107a1278; _ccallback_c 1fcb0dee747e2cd6e0db04f7ffa29c23d848fbd1ab5cda491c46eb5088af899b.

This is a fresh ordinary CPU observer reproduction. It is not the captured production importer, admission evidence, full canonical materialization or a GPU proof. It supports feasibility and ordering only.

**Historical captured evidence:** independently hashed and inspected selected original JSON fields under /data2/shared/ang.gao/c2d3-c8b404c-go2-960-guarded-author-v8. All seven supplied historical digests matched:

- external_failure.json: 93fa4f97c92dd324d190f7839e3213dd048020b82b061f2b92f7cad2af432a1d
- failure.json: e95ad9c6c1efbdc031f9c8a92aa2bb676f95a77c87fc9bef075f8e6309a62f9a
- worker.json: fe3a7556df870d287b5291f672ddf60d8fdd0a3de2ed9cbeaefd9a7d305349a2
- custody_artifact.json: 42c76f61e9a083b8a1820c0ee5ec2c4e4329f11d6f28204409bfaf29d956d0e1
- supervisor_stdout.json: cde7946e60b6bacc46d7a0bf38e11bb56368f9805fd7d849e0666aa38fa8a9a9
- request.json: af4eee4c2a97fa7605c1032badace1307e0a9cdc886a75aa6f742eed6e633adb
- identity.json: a7a600a53f892e86ac453018a27514575280a15372dcdac4f0d8dbd9839a225e

The request records c8b404c54e3135d615727ee6e4a43c63822ba257, Warp target1, real Go2 960x640, 120000ms operation/1000ms ACK. External event4552 completed PIL.ImageFile; event4553 failed audit with unlisted module publication during canonical_materialization. The unrecorded diagnostic has 113 rows, truncated=false; all three named globals have null file/spec/package, no loader, and held_in_failure_graph=true. This is original diagnostic evidence of exact-held membership, not resurrection of historical live objects or creator proof.

Worker summary records full_quarantine, guardian TRANSFERRED_TO_QUARANTINE, failed guard sequence2, release not_authorized, zero emergency sync/terminal ARM and dirty ordinals [1]. Supervisor trace records process_exited returncode74, EXITED/complete, deadline_expired=false and kill_requested=false. Earlier normal completions do not complete failed segment2. These facts substantiate rejection and terminal handling, not successful upload/readback. An initial overly broad JSON print was truncated; the relevant fields were then reread with explicit bounded projection. No truncated dump was relied on as complete inspection.

## Limitations, hygiene and mandatory follow-through

No candidate implementation exists for this supplement, so no new production state machine or parser was tested. No pytest suite, captured positive, new guarded Go2 run, backend initialization/upload/readback, native fault injection, hang/dual-device matrix, performance measurement or complete C2d3 code review was performed. Such checks are not replaced by old artifacts or CPU probes. The candidate expressly makes them mandatory before code approval.

Before whole-stage approval, require the real captured production importer and guardian/supervisor for both lanes, canonical synthetic/dynamic and both real Go2 workloads, actual input allocations/upload/validation-readback, cold/ambient/foreign/UNKNOWN dual-device behavior, partial-return/adoption/reconciliation/recording faults, hangs and actual exit/reap. Freeze the full runtime-custody schema and replay negatives in the same implementation. Repin exact Git LF bootstrap/policy/catalog bytes and cold selectors, preserve prior attribute/self-alias/absence/disabled/stdlib/C0 regressions, and update exact cleanup paths. Any different actual required interval returns for a reviewed versioned table, not runtime discovery.

Only scratch output and this report outside the repository were written. No package change, process signal, GPU reset, push, merge or unrelated-work manipulation occurred. Remote inspector/timeout PID pairs were 4094889/4094888, 4095294/4095293 and 4095943/4095942; the fresh native CPU probe was 4093086/4093085. Final scoped ps for all eight PIDs returned the header only. Read-only nvidia-smi showed only existing MPS server2238410, 60MiB; this review launched no GPU work. This is a scoped residual check, not a global no-hang assertion. Short-lived SSH/Git utility PIDs were not exhaustively tracked. The final local check PowerShell was PID7816, start 2026-10-08 10:13:33, and completed. Final local/remote tree and revision checks remained unchanged and clean.

Approval binds exactly HEAD80387e8d8199e9dc6741924b95ff1aeeae047b45 and candidate digest5c29fad420b9d31a848f8b53df07450b46f0fee0b4e2abe9f38aad7997be8b2c. It permits implementing only the finite native-global observation/direct-custody supplement, v4 environment migration and descriptive evidence integration specified here. It grants no standalone CPU-subset C2d3 acceptance.

Still CLOSED: whole C2d3/C2d/C4 acceptance, C2e/f, C4c3/c4, C4d/B3, product GPU preparation/activation, a3b, result/baseline, host payload, R2 and shipping. No final optical graph/readiness, active sample lease or renderer result is authorized. C4 is not complete.

The author may archive this unchanged report under the evidence-only exception. That archival cannot amend the verdict, approved snapshot, requirements, scope or gates.