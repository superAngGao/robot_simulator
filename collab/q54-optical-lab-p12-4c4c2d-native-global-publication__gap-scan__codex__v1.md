# C2d3 Cython global publication: independent bounded gap review v1

Initiative: q54-optical-lab-p12-4c4c2d-cython-publication
Stage: gap-scan
Author: codex, fresh independent reviewer
Version: v1
Date: 2026-10-08 (Asia/Shanghai)
Status: draft
Related Files: cython-publication test-evidence v1; cython-owner-probe test-evidence v1; native-module-name implementation-note/design-request/design-review v1; runtime-adoption design-request/design-review v2; native-self-alias design-request v1; optical_worker/runtime_environment.py; optical_worker/native_bootstrap.py; tests/support/native_runtime_process.py; tests/unit/optics/test_runtime_failure_observations.py
Owner Summary: Request changes to the bounded publication contract before implementation. Actual captured rejection and retention are substantiated. A narrow native-interval publication observation with direct session custody is implementable in the existing loader/finder, without inventing a creator attribute or exact creator. A separately reviewed design must freeze its state/evidence rules and literal deployment rows. This is a GAP REVIEW, not C2d3 design or code approval.

## Identity, revision and scope

Reviewer session: /root/cython_gap_review. Started as a fresh independent session with the review packet, not the implementation conversation; independently inspected code and evidence. I authored no candidate implementation, changed no candidate files and spawned no further agents. Initial independence acknowledgement was sent to the coordinator.

Frozen base = HEAD = 62cf8ff0b1492163d166e243ffde7347fd4e03d9. This is a gap scan at a snapshot, not a claimed implementation diff review.

Local repository C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design, branch implementation/c4-supervised-waits. Primary remote SSH tileops-dev (183.166.183.33:40031), repository /data2/shared/ang.gao/robot_sim_c2d3_binding_wip, detached HEAD. Both initial and final working trees were clean at that exact revision. Final local check was 2026-10-08T10:02:35.9007682+08:00. git diff --check passed. No fallback host was used.

Runtime-adoption v2 file SHA-256 independently matches 1a0d3eed4dd182ddd5ea4ad0c19344cad735aabbdd317d77a14aad1932cc0224. Its approved native trust boundary and exact creator-attribute contract are requirements, not approval of any proposed extension here. AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md were read completely; the principles were reread with explicit UTF-8 after a default-encoding display issue. Current PROGRESS.md C4/C2d2/C2d3 entries and the required four handoffs were read. Runtime-adoption v2 and its complete review v2, native-self-alias design v1, native-module-name design and review v1 were inspected for applicable authority and restrictions.

Verdict: **Request changes**, for the representational/design gap below. The current safe rejection should remain. This verdict neither approves a future design nor characterizes deliberate fail-closed behavior as a reason to relax validation. The review found a feasible bounded direction; no unavailable fundamental Python capability prevents direct custody once an actual published object is observable. Exact native creator causation remains unproven and is unnecessary if the new contract truthfully claims only the narrower observation.

## F1 — P2: current exact-owner contract cannot express these observed globals

Locations: runtime-adoption v2 section 4.2; runtime_environment.py:255-287; native_bootstrap.py:1088-1164 and audit at :1226-1250.

Trigger: the existing real Go2 canonical materialization imports the pinned CPU native dependencies, which leave cython_runtime, _cython_3_2_5 and _cython_3_2_8 in sys.modules. They have no file/spec/loader provenance and are distinct objects from the native extension roots.

The production schema permits created_module or module_alias with a fixed raw attribute path from an admitted file-module creator. Empty paths are only creator-self aliases. Reconciliation actually walks vars(value), checks identity against sys.modules, and retains creator/publisher identity. Filling in one of these names, selecting an empty path, substituting a convenient parent, or using matching version/hash metadata would not satisfy that relation. Ordinary direct-attribute and direct-GC probes do not demonstrate any such path; they also do not prove that opaque native ownership is absent. Searching indefinitely for an undocumented Python owner is not a justified implementation prerequisite.

Consequence: the current final audit correctly rejects the three unrecorded globals. A table-only workaround would either fail or falsely assert ownership. Existing diagnostics prove available-object retention on failure, not permitted provenance or successful admission. Existing creator-attribute/self-alias tests cannot close a genuinely different relation.

Required direction: independently review a separately versioned, explicitly discriminated relation for a published interpreter-global object observed during a fixed admitted native interval. State plainly that the session/guardian becomes a direct strong holder of the actual object; it does not discover an exported Python creator path. Require a literal deployment entry, exact active source-session/native-operation binding, observed absence-to-identity transition at supported checkpoints, and successful close of the declared native interval before admission. Preserve the existing relations intact. The detailed minimum below is a proposed direction, not authorization to implement.

Proof: schema/authority/identity negatives, real captured production positives for each required workload/lane, original failure regression and sticky partial-failure custody. No name/version prefix, metadata-shaped module, hash, parsed receipt or ordinary import output can mint the relation.

## F2 — P2: return deltas alone cannot identify the correct nested interval

Locations: historical ordinary CPU native_boundary_new_key events; the proposed extension points around native_bootstrap.py:483-544 and _CapturedSourceFinder.find_spec at :366-388. This is a design trap exposed by new evidence, not a claim that the present code already implements this faulty algorithm.

The historical v5 observer reports _cython_3_2_8 at both bit_generator.exec return and outer _bounded_integers.exec return. It reports cython_runtime at the latter return. These overlapping deltas cannot establish exclusive creation by either extension. Fresh probes additionally show that these globals are visible before nested native calls begin. Assigning the object to whichever child returns next is wrong; requiring every nested entry to see all undeclared globals absent would reject ordinary execution.

Fresh saved probe results (ordinary cold CPU process, not captured admission):

| First observation checkpoint | New exact key | Active enclosing native interval |
| --- | --- | --- |
| finder_request_entry(zlib), sequence 155 | cython_runtime | numpy.random._bounded_integers / exec_module |
| finder_request_entry(numpy.random._common), sequence 164 | _cython_3_2_8 | numpy.random.bit_generator / exec_module, inside outer _bounded_integers.exec |
| finder_request_entry(scipy._cyutility), sequence 306 | _cython_3_2_5 | scipy._lib._ccallback_c / exec_module |

At outer _bounded_integers.exec entry, sequence 154, all three keys were absent. At bit_generator.exec entry, sequence 163, cython_runtime already existed and _cython_3_2_8 did not. At _ccallback_c.exec entry, sequence 305, the two NumPy-observed globals already existed and _cython_3_2_5 did not. Subsequent returns retained those same objects; no replacement was observed. These are temporal facts within the observed admitted call chain, not instruction-level creation or Python owner relationships.

Implementability: current _CapturedSourceFinder is first in the captured import topology and sees missing-module find_spec requests before its stdlib dispatch. A bounded check there can observe zlib's request without wrapping zlib or other stdlib loaders. _ExternalAdmission.find_spec alone is insufficient because the captured finder resolves stdlib first. The two other requests likewise reach the common finder entry. Thus no general import-tracing subsystem is required for these concrete checkpoints. A finder request is not a stdlib create/exec observation and must never be recorded as one. Cached imports do not invoke find_spec; exact checkpoint expectations must be frozen/tested for the actual cold captured environment, not copied from this ordinary probe. If the captured path differs, it fails closed and returns for explicit evidence/design reconciliation.

Required direction/proof: distinguish the suspended enclosing interval, the request/return where an object was first observed, and the later interval success that admits it. Nested deltas must not duplicate records or relabel children as creators. Test a parent insertion before child entry, a child insertion, a nested child failure, and identical repeated outer observations. A new pending observation may need to be held while its parent native call is still executing and trusted dependencies use it; this is not successful admission, nor permission to escape to repository GPU work. A design that requires complete admission before such nested dependency use is incompatible with this actual ordering.

## Minimally sufficient honest design direction

1. Keep one production loader/finder/session and one source of literal rules. Add a distinct relation/outcome rather than overloading creator_attribute, creator_self, file_module or namespace. Each frozen rule binds the exact global key, deployment file-row/native delegate operation, phase, and allowed first-observation checkpoint/interval. The actual native module/spec/delegate/session/controller identities come from the registry, not a caller string. The three associations above are candidates to confirm through captured execution; this review does not approve their table rows. Existing inventory hashes bind the deployment file; they do not prove who allocated the global.

2. Separate custody from eligibility. A useful state shape is absent -> observed-held/pending -> admitted, with sticky failed/quarantined reachable from every started state. All configured keys must be absent at the initial cold baseline, including None entries. At each supported observation point, obtain and retain actual new/replacement values in the already-held admission graph before any fallible reconciliation, adoption callback or metadata check. A pending record cannot count as a completed external outcome. The captured session itself is the Python strong owner; available loader/spec/module roots and the process remain owners of unreturned native state under the existing trusted dependency obligation.

3. Observe only fixed configured globals for this new relation at existing common finder request entry, native delegate entry/normal return/error intake and final audits. Ordinary full-delta failure retention and unknown-publication rejection remain separate mechanisms. Capture the actual enclosing native operation identity and its start observation before delegation. At a nested request entry, new objects belong to the observed still-open parent interval, never to the not-yet-entered requested child. At a child return, objects first seen there belong to that observed interval; previously held identities stay previously held. Track unique operation instances and nesting, not just a list of module names. A wrapper's checks/request handling outside the delegate should not silently count as the delegate executing.

4. A minimal candidate grammar can permit first observation at either an existing finder request entry while the exact declared native exec is active, or that exact exec return. Freeze the enclosing native module, exec operation and phase per global; record the actual requested child name descriptively rather than granting permission from an ordinary-probe-only request order. No fresh create-time global need be permitted for the three observed cases; create entry/return still check absence or already-held identity. The fresh probes show no required case outside this grammar, but the captured cold workloads have not demonstrated it and must do so before code acceptance. Reject a different first-publication interval rather than inferring a new rule. Freeze the preceding absence checkpoint. Admit only after the associated declared native interval returns successfully, required dependency/file checks succeed and the key still points to the same held object. Keep pending objects held across nested trusted work, and do not claim a successful segment while any required publication is pending/missing. Audit can verify existing records but never create a missing record, reinterpret a failure, or learn a new permission. No second create/exec event is emitted for a global that has no loader invocation.

5. Later extensions can encounter the same held global as an already-admitted dependency. Identity persistence is checkable; invisible internal reads/references are not. Do not emit a reuse/ownership event just because another extension ran. No separate consumer allowlist is necessary merely to permit an unchanged already-admitted global across the existing trusted file closure; if stronger consumer causation is promised, it requires separate mechanical evidence. A later native call may populate helper types inside a shared module; immutable full module-dictionary contents must not be invented as an ownership invariant. Identity/metadata expectations and any permitted contents checks must be explicit.

6. Reject early/preloaded values, wrong phase/interval/checkpoint, undeclared keys, object replacement/removal at any supported observation, cross-session records, repeated admission and inconsistent nesting. Hold a replacement as failure evidence while retaining the original. Failure after native work starts, including a swallowed inner exception or failure while checking/adopting/recording the observed global, is monotone: retain original error plus available graph, no COMPLETE/new ARM/retry/ordinary release, quarantine under the original deadline. Any late enclosing-body publication must still enter failure custody. Do not fabricate prior observation or successful admission at final failure scan.

7. State the limits precisely. The contract establishes a published object first observed during a bounded native interval plus direct subsequent custody. It cannot establish the exact C instruction or distinguish all Python/native activity invoked inside that interval. Transient create/remove or replace/restore entirely between supported checkpoints is not observed. Stronger claims would need dependency-specific native instrumentation/build attestation, a separately scoped change. Pin the supported interpreter/import-thread model and fail on observable unsupported concurrent/reentrant admission; native background mutation/exception safety remains a trusted deployment obligation. The GIL alone is not a general proof of no concurrency. Do not introduce a native sandbox claim.

8. Version the environment contract and descriptive runtime-custody schema explicitly; preserve old artifacts. Records should distinguish rule ID, session/phase, exact held object record, native interval ID and operation, prior absence checkpoint, first-observation checkpoint/request, pending-versus-admitted state, successful interval close and later audits. Name an interval/observer, not an exact creator. Exact field order/enums, cross-record checks and failure transitions must be frozen in the actual design. Serialized records cannot register objects or authorize release. Repin source-policy/catalog bytes only during the subsequently approved implementation.

This is a bounded extension inside the existing admission path. It does not justify permitting every unrecorded object or relaxing unknown-publication audit. If reviewers require exact native creator ownership rather than the above explicitly narrower interval observation plus direct custody, the exact blocker is the missing mechanically demonstrable relation/build-level provenance for these installed extensions. The available evidence cannot meet that stronger claim.

## Required proof before treating the gap as closed

- CPU parser/state tests discriminate the new relation from file, alias and attribute ownership; reject old schema, collisions, missing root/operation/checkpoint, None preload, wrong phase, foreign session and caller-fabricated records before execution.
- Production-entry fault tests retain exact originals and replacement objects when adoption/reconciliation/recording fails; then remove mutable sys.modules references and check sentinels remain alive until process-terminal disposition. Swallowed nested failure and late parent publication never restore admission or issue a second terminal action.
- Nested positives/negatives distinguish parent insertion before child request, child insertion at return, premature insertion, wrong requested checkpoint, failed declared parent after successful child, duplicate outer delta and wrong interval with identical module metadata. Audit must reject missing records, removals and same-metadata replacements without re-admission.
- Existing creator-attribute/self-alias, generated stdlib, optional absence/disabled adapter, file identity, native name, baseline/topology, custody and C0 behavior remain regression obligations. No runtime discovery widens the literal table.
- Real captured cold executions must establish checkpoint order and exact held identities for the required canonical asymmetric/dynamic and both real Go2 materializations in both lanes, with no substitute materializer or package changes. Ordinary observer output here is feasibility evidence only. Continue through actual guarded upload/readback/storage and required failure/hang/dual-device/reap matrix; passing the publication boundary alone is not C2d3 acceptance.
- Freeze and validate the final versioned runtime-custody schema with invented import, duplicate/late admission, wrong interval and forged-success replay negatives. Descriptive evidence must agree with actual supervised terminal state.

## Verified raw evidence

Historical guarded root: /data2/shared/ang.gao/c2d3-c8b404c-go2-960-guarded-author-v8. Independently read and hashed, not rerun or represented as fresh execution:

| File | SHA-256 |
| --- | --- |
| external_failure.json | 93fa4f97c92dd324d190f7839e3213dd048020b82b061f2b92f7cad2af432a1d |
| failure.json | e95ad9c6c1efbdc031f9c8a92aa2bb676f95a77c87fc9bef075f8e6309a62f9a |
| worker.json | fe3a7556df870d287b5291f672ddf60d8fdd0a3de2ed9cbeaefd9a7d305349a2 |
| custody_artifact.json | 42c76f61e9a083b8a1820c0ee5ec2c4e4329f11d6f28204409bfaf29d956d0e1 |
| supervisor_stdout.json | cde7946e60b6bacc46d7a0bf38e11bb56368f9805fd7d849e0666aa38fa8a9a9 |
| request.json | af4eee4c2a97fa7605c1032badace1307e0a9cdc886a75aa6f742eed6e633adb |
| identity.json | a7a600a53f892e86ac453018a27514575280a15372dcdac4f0d8dbd9839a225e |

The request binds historical revision c8b404c54e3135d615727ee6e4a43c63822ba257, Warp, target 1, 120000 ms operation and 1000 ms ACK, frozen 960x640 Go2 case and environment digest d2b1edc48c5a28a4ef71237fdf16d92d089f02f3fe7926c858d3e0c651146d3b. There are 113 unrecorded diagnostic rows, truncated=false; all three Cython keys report exact held_in_failure_graph=true and null file/spec/package with no loader. The diagnostic uses identity membership in actual retained owners. This is not independent resurrection of those historical Python objects and not creator proof. Other diagnostic rows include legitimate stdlib imports, so 113 is not a forbidden-module count.

Event 4552 completes PIL.ImageFile; 4553 fails audit with SourceBootstrapError/unlisted module publication during canonical_materialization. Worker: full_quarantine, quarantined_process_exit, guardian transferred, release not_authorized, failed setup sequence 2, zero emergency sync and terminal ARM, possible dirty ordinals [1], unknown_device_scope=false. Two earlier normal completions do not complete failed sequence 2. Supervisor: EXITED/complete, observed return code 74, no deadline expiration and no kill. Identity records parent 3669870, supervisor 3669902, worker 3669903. No successful materialization/upload/readback is inferred.

Historical ordinary artifact /data2/shared/ang.gao/c2d3-go2-960-cpu-dependencies-v5.json matched 17f559fb58dfffc9d7fee34a82e42363cadf86bc3e403d55cd478f128b7584d1. Inspected 4759-event list and the overlapping native_boundary_new_key rows described above, plus 913 module records. This old trace is neither new execution nor captured admission.

## Fresh commands, probe provenance and results

Read-only local commands used Get-Content -Raw/-Encoding UTF8, targeted Select-Object ranges, rg, git status --short, git branch --show-current, git rev-parse HEAD, git diff --check and Get-FileHash. Inspected loader create/exec order, exact registry/delegate checks, stack/failure retention, publication validation/reconciliation, final audit, captured stdlib/finder branch and diagnostic identity projection/tests. Large truncated reads were followed by narrower reads. An initial raw JSON summarizer wrongly assumed unrecorded_modules['entries']; it exited 1 with KeyError, then was corrected to actual ['modules'] and completed. This was reviewer tooling, not a candidate failure.

All remote Python diagnostics used the exact /home/ang.gao/miniforge3/envs/robot_sim/bin/python, CPython 3.12.13, via SSH BatchMode=yes/ConnectTimeout=10 and timeout 30 with -I -S -B stdin. No timeout fired. No site.main(), dependency installation or candidate edit occurred. Fresh CPU native probes explicitly added only the existing site-packages path, delegated to the installed standard ExtensionFileLoader, and asserted neither Torch nor Warp was imported. They are ordinary observer processes, not a captured-worker substitute.

First native-boundary-only probe ran over stdin, PID 4077297 / timeout parent 4077296, exit 0. It first saw cython_runtime at zlib.create entry, _cython_3_2_8 at numpy.random._common.create entry, _cython_3_2_5 at scipy._cyutility.create entry, with the enclosing intervals above. Its source existed in the tool transcript, not a persisted script. A second expanded probe added request-entry observation to prove the existing finder has sufficient checkpoints.

Preserved second probe: C:/Users/AG/新建文件夹/c4_assets/cython_gap_review_probe_v1.py.
Exact file SHA-256: a827cdf4affd89b8b5444786f39de6d22597995834204dd0b649e6e6ac603dad.
Output: C:/Users/AG/新建文件夹/c4_assets/cython_gap_review_probe_v1_output.txt (PowerShell Tee-Object capture; report/probe source are UTF-8).
Invocation:

    Get-Content -Raw -Encoding UTF8 <probe-path> | ssh -o BatchMode=yes -o ConnectTimeout=10 tileops-dev 'timeout 30 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -'

PID 4079273 / timeout parent 4079272, exit 0. Checkpoint grammar: finder_request_entry(fullname) before returning None to existing finders; native_delegate_entry(name, create_module|exec_module) before stack push; native_delegate_returned or native_delegate_raised before stack pop. The observer prints monotonically numbered observations, actual active native stack, current exact object IDs and identity changes; its dict holds observed objects strongly. It is deliberately not a production receipt issuer. This probe did not assert all conceivable between-checkpoint mutations detectable.

Both fresh native probes reconfirmed empty direct raw owner-attribute matches and no direct GC referent for every present queried global in the three inspected native modules. Binary SHA-256s freshly measured:

| Native file module | SHA-256 |
| --- | --- |
| numpy.random._bounded_integers | 5b981bd3ee5d46dcfa3f1972838b3a5482d2c28d09b5126e976ebdbda9d8cd44 |
| numpy.random.bit_generator | f7466ade9efe0c03f7c9f568130ea0297e651107d7920dc0d335ba23107a1278 |
| scipy._lib._ccallback_c | 1fcb0dee747e2cd6e0db04f7ffa29c23d848fbd1ab5cda491c46eb5088af899b |

No reference-count threshold or complete deep/native owner-graph claim is made.

Fresh bounded test invocation in remote candidate:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 90 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_runtime_failure_observations.py

Result: 2 passed in 0.03s, exit 0. These exercise diagnostic exact-hold versus same-metadata copy, None/recorded exclusions, no registry mutation and 256-entry truncation. They do not prove admission/creator provenance or GPU safety. No broad regression suite was necessary for a non-implementation gap scan.

Primary source corroboration was read directly from the official Cython repository. [ModuleNode.py 3.2.8](https://github.com/cython/cython/blob/3.2.8/Cython/Compiler/ModuleNode.py#L3484-L3488) emits a module-state reference obtained through AddModuleRef for cython_runtime. [ModuleSetupCode.c 3.2.8](https://github.com/cython/cython/blob/3.2.8/Cython/Utility/ModuleSetupCode.c#L2976-L2983) implements its older-interpreter branch with PyImport_AddModule plus an incremented reference. [ModuleSetupCode.c 3.2.5](https://github.com/cython/cython/blob/3.2.5/Cython/Utility/ModuleSetupCode.c#L1268-L1270) forms an ABI module name. These support the plausibility of internal/shared native state and show why a Python-exported attribute cannot be assumed. They do not attest which exact generated source/build options produced the installed binaries. The version-like key alone proves neither build provenance nor ownership.

## Hygiene, limitations and remaining scope

Only scratch probe/output/report files outside the repository were written. Both candidate trees stayed clean. Final scoped ps for 4079273/4079272/4077297/4077296 returned the header only; earlier scoped checks also found no remaining 3669870/3669902/3669903. ps exit 1 is the normal no-match result, not a failed cleanup. One command chained nvidia-smi after ps with &&, so the no-match status prevented that query; it was then executed independently. Read-only nvidia-smi reported only pre-existing MPS server 2238410. No process was signaled, no GPU reset issued and no unrelated work touched. Short-lived read utility PIDs were not all recorded; no complete process ledger is claimed.

Not executed: a new guarded Go2 run, captured positive for any proposed publication relation, GPU runtime initialization/upload/readback, fault/hang/dual-device acceptance matrix, dependency rebuild or native allocation/creator instrumentation, full C2d3 review, performance/memory comparisons. Historical captured artifacts are explicitly historical; fresh ordinary CPU probes establish feasibility and ordering only. Build-level creator provenance and invisible native failure safety remain limitations, not inferred passes.

The next concrete deliverable is a bounded versioned design resolving F1/F2, with fresh independent design review before capability-changing implementation. It must not use this gap report as prior approval. Current rejection remains closed until that workflow and its required evidence are complete.

No whole C2d3, C2d or C4 approval. C2e/f, C4c2 runtime acceptance, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, R2/host payload and shipping remain closed. No final readiness, active arena lease, optical sample or result is authorized. Fair CUDA/Warp evaluation and eventual one selected backend remain the parent objective; this dependency-custody contract belongs in the existing shared admission path and does not authorize a permanent second importer/backend bridge. Losing-lane fixtures and temporary drivers retain the approved P12.4e removal obligations.
