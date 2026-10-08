# Independent native-scope aggregate design review

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: bounded independent design-review
Author: Codex, fresh independent reviewer /root/native_scope_aggregate_design_v1
Version: v1
Date: 2026-10-08 (Asia/Shanghai)
Status: in_review — Request changes for the exact bounded design
Related Files: native-scope-aggregate design-request v1 and gap-review v1; runtime-custody implementation-plan/gap-review v3; runtime-adoption design-request v2; cuda-ambient design-request/review v1; optical_worker/runtime_custody.py; native_bootstrap.py; native_setup.py; native_readback.py; guarded_lifecycle.py
Owner Summary: The v2 journal direction and Warp ScopedDevice readback repair are feasible and preserve the existing authorities. One blocking specification defect remains: the operation reference is bounded against the journal index instead of its lifecycle cut. Correct that exact occurrence rule and its negative proofs before implementation. This is no whole-C2d3 approval or permission for a new positive native observer.

## Verdict and inspected revision

**Request changes**, limited to F1 below. No preset verdict, candidate authorship, inherited implementation conversation, subdelegation or candidate edit was used. The repair direction itself does not need a new architecture or broader product decision.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Base: `14accac054605459966fe37af51fd7258efc073d`.
- Frozen HEAD: `71566e0ffd51283071f56903bfd414291348f8b1`.
- Candidate: `collab/q54-optical-lab-p12-4c4c2d-native-scope-aggregate__design-request__codex__v1.md`.
- Candidate SHA256: `1237f5d2b7cb5a12ba125b6f256dd7bf9d49c36cc024a793c982ea64d10003c3`.
- Governing gap report SHA256: `5e264476a233c42aa55929f70d4735e2258059df2481b32d4ee87f92f9d8dc66`.
- Initial and final local working tree clean; HEAD and candidate hash unchanged. Base-to-HEAD adds only the 318-line candidate. `git diff --check` passed. Production `optical_worker` diff against `9b5702b4016824ea7fc5b5e7c108c5adf02a1bd1` is empty.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely first. The first PowerShell read garbled Chinese; the principles were reread as explicit UTF-8. Inspected the candidate, governing gap, operative predecessor contracts and their bounded approval rules; oversized reads were followed by focused rereads. Prior reports and author updates were treated as claims and scope information, not proof of this candidate.

## F1 — P2: operation references use the wrong index domain and leave occurrence ordering ambiguous

Location: candidate lines 111–122, especially “operation_event_index names an actual lifecycle event at or before the envelope index”; related required replay proof at lines 249–258 and 295–301. Existing producer: `native_bootstrap.py:2112` (`_RuntimeObservation.append`). Existing replay: `runtime_custody.py:559` (`apply`) and `:633` (`returned_owner`).

The envelope has two different counters. `index` is a journal position. `lifecycle_event_index` is the latest actual lifecycle event at append. The producer derives them from different lists. The new field is expressly a lifecycle-event reference, so bounding it by envelope `index` is not an execution-order check. The existing returned_owner parser correctly compares its leaf reference against `self.lci` instead.

Independent archive inspection establishes this is not merely a hypothetical equal-counter convention. In an actual Warp record, runtime_binding is journal 1340 / lifecycle 433; in the retained CUDA asymmetric record it is journal 9169 / lifecycle 2421. In the Warp record `test_actual_cold_canonical_war0/runtime_custody.json`:

- binding_consumed is journal 1342, lifecycle 438, setup guard sequence 4;
- lifecycle 438 is guard_armed;
- indexed_upload_scope enters at lifecycle 441;
- the first allocation leaf enters at lifecycle 447;
- the first copy leaf enters at lifecycle 451.

A proposed native row at journal index 1343, with envelope lifecycle_event_index 438 and operation_event_index 447, satisfies the literal `447 <= 1343` check. It references a real allocation leaf under that same setup guard, yet that leaf had not entered at the row's cut (`447 > 438`). A before/after pair placed in this early interval can also satisfy pair order and the “no intervening different entered leaf” rule. A matching scope pair can be transplanted early in the same way. The factual counterexample is the independent-counter relationship; no nonexistent v2 parser was executed or claimed to accept a full forged record.

Consequence: an implementation following the named numeric bound can accept a snapshot transplanted before its operation. Another implementation can silently substitute lifecycle_event_index and reject it. The design is intended to freeze the exact joins and requires wrong-leaf/reordered mutation rejection; these are materially different contracts. The later promise to wrap actual calls is an important producer constraint but does not resolve the contradictory parser rule. Same guard, matching descriptor, pair nesting and counts do not establish that the referenced operation had happened at this observation.

Required direction:

1. State the namespaces explicitly: journal references such as before_index address journal entries; operation_event_index addresses the lifecycle trace. Require `0 <= operation_event_index <= envelope.lifecycle_event_index < observation_cut.lifecycle_event_count` at each native row. Never compare a lifecycle reference to journal index.
2. Freeze the before-occurrence rule as well as the within-pair rule. For a helper/runtime_stream before row, its operation reference must be the eligible most recent entered leaf at that lifecycle cut, with the exact original descriptor, phase and guard. A stale same-guard leaf cannot be reused after a different leaf has entered. Scope after rows intentionally retain the opening scope leaf while admitting its actual nested helper leaves. Preserve the separately defined binding-adoption and emergency-sync-start relationships.
3. Keep the envelope's latest-actual-event rule, including intervening adoptions; do not require envelope lifecycle_event_index to equal operation_event_index on every after row.

Proof required after repair: a revised exact design and fresh review; then CPU mutations on the implemented codec with trusted launch inputs fixed. Include the concrete future-leaf example above, an out-of-cut reference, a stale allocation reference after the next copy leaf has entered, a different same-kind occurrence under the same guard, crossed pairs and duplicate pair reuse. They must reject. Genuine before/after helper rows spanning returned-owner/adoption events and outer scope pairs spanning nested helpers must pass. Verify normal and emergency namespaces separately. These tests must distinguish the wrong cross-domain comparison from the corrected one, rather than only exercising grossly out-of-range integers.

## Assessment of the remaining design and prior gaps

The original gap F1 is addressed in direction, but cannot be closed at exact design scope until the reference rule above is corrected. Adding a discriminated native_scope journal kind under v2 is a reasonable bounded change. Keeping the old v1 grammar, immutable old bytes, unchanged components/summary and explicit nonacceptance of missing native coverage avoids relabeling earlier executions. The existing selection reader currently pins one runtime version; the requested shared codec dispatch is a real required implementation change, not an assumption that it already supports v2.

The schema distinguishes Torch logical/raw device state from Warp default metadata/native context and distinguishes the explicit held stream from current stream selection. Tokens remain descriptive identities tied to existing held objects and actual import/adoption records; the canonical source is not prematurely introduced. The producer remains the fixed driver and the collector remains CPU-only. There is no new callback, live token, uploader, source authority or lifecycle state machine. PID/TID and externally observed process identity are acceptance evidence, not another self-reported token authority.

The prior gap F2 has a feasible repair at design scope. Actual local code has Warp upload under ScopedDevice but no such outer scope in `_readback_prepared_native_input`. The shared readback checks source/prepared/backend/target/stream and storage identity first, adopts a retained host-owner list, then issues D2H/event leaves and invokes the shared CPU validator. Factoring that body beneath one new Warp scope can retain the same owners, helper functions and validator. Returning only after scope exit prevents successful validation from hiding failed restoration. The proposed explicit validation scope leaf accounts for its context selection without another ARM or deadline. Existing lifecycle work counting increments on each entered leaf, so the requested additional runtime_import_or_init count has an actual implementation basis.

Installed-source inspection supports that assessment:

- `_src/utils.py:1267–1284`: ScopedDevice saves default_device, selects the target, enters its context guard, then exits that guard and restores metadata.
- `_src/context.py:3153–3167`: a CUDA ContextGuard uses native context push/pop. The design need not introduce a test setter or change the global native restore policy.
- `native/warp.cu:894–910`: nonempty D2H uses native `ContextGuard guard(context)` around cudaMemcpyAsync. `native/cuda_util.cpp:146` sets its default restoration policy false; `native/cuda_util.h:253–270` restores only when need_restore is true. This explains why the unscoped body is a meaningful negative control and why metadata-only checks miss the difference.
- `_src/context.py:3638–3653`: first target context initialization creates its streams and restores the previously saved context on successful return. `_init_streams` calls set_stream; set_stream calls `wp_cuda_context_set_stream`. `warp.cu:2285–2303` obtains/populates get_context_info. The cached branch at `warp.cu:374–376` returns without the unseen-context workaround. Thus the proposed post-return native stream getter has a concrete feasible audited path. Its admission still requires the promised exact code preflight and successful original stream initialization; a cache-miss fallback is not approved.

The failure and cut direction is conservative. Real returned owners are retained before observation/adoption. Failed getters produce held diagnostics, not fabricated successful null values. An incomplete pair remains incomplete. A contradiction is recorded before sticky failure, and no later good row repairs it. Terminal native rows receive only the narrow emergency exception, with a real binding and validation, the actual sync-start event, and no prior compromised history. `guarded_lifecycle._synchronize` calls the callback inside its try block, records failed synchronization and quarantines on callback failure, and releases only after successful return. Therefore a post-sync observer failure can use that existing path without a retry or ordinary release. This does not authorize running observers after a failed original sync or after sticky failure.

Retained cuts and record-before-sidecar persistence remain descriptive and monotonic. A prior record does not reopen execution, newer raw diagnostics do not extend it, and evidence failure after actual authorized release cannot erase that release. Full terminal trace comparison and unavailable/killed-prefix status remain necessary at acceptance. None of the available old records closes the new native coverage requirement.

## Non-blocking comment C1 — native scope errors are not necessarily Python exceptions

Location: candidate lines 52–60, 233–241 and 267–269; installed `warp.cu:2165–2170` and `context.py:3157–3167`.

The native push/pop wrappers return void and invoke check_cu; Python ContextGuard does not consume a CUresult from them. Do not interpret “successful actual exit” solely as “__exit__ returned without a Python exception.” The required direct post-exit context query and acceptance check are what can expose failed restoration despite a normal Python return. This is non-blocking because the proposal already requires those observed-state checks, no success on contradiction, and the trusted native dependency obligation.

Disposition: carry this source fact into the required code preflight and failure proof. A CPU observer test should model an original exit that returns while the observed context remains target1: retain the contradictory after row, fail the original segment, and admit no next normal leaf or release. Also test a getter CUresult error versus a genuine null context. Actual positive native restoration still requires the reviewed real-device run; this CPU model is not its substitute. After an earlier sticky body failure, unwind the original scope once but obey the prohibition on further native observation; do not manufacture a successful after row merely because exit returned.

## Commands, results, evidence and limits

Executed read-only local PowerShell/Get-Content UTF-8, rg/rg --files, Get-FileHash and Git status/revision/branch/diff commands. `git diff --stat 14accac054605459966fe37af51fd7258efc073d HEAD` showed only the candidate; `git diff --check` and the production comparison described above passed. An exploratory Windows `rg tests/support/*` invocation failed on that wildcard spelling; subsequent searches used the actual directory. This was an inspection-command error, not a candidate failure.

Local `C:/Python314/python.exe -I -S -B -` scripts used only json/pathlib/hashlib/tarfile to verify source-text hashes, inspect exact source spans and read the archive in place without extraction. Archive SHA256 independently matched `eb08d4ed62f4271596a7b8a1e982f6cd7c2b7f3ab8ec7488319c6bb9aeab5510`; twelve runtime_custody.json members were enumerated. The script printed the actual Warp events 438–451 and the illustrative relation result: journal 1343, lifecycle cut 438, operation 447, wrong-domain check true, correct-domain check false. This was fresh inspection/arithmetic over historical execution evidence, not fresh GPU execution or v2 parsing. I did not rerun all twelve old CPU parser/attachment validations; the previous report's rerun remains its own evidence.

The inspected retained source files were `c4_assets/native_scope_warp_source_read_v1.jsonl` and `native_scope_warp_utils_read_v1.json`. Every embedded source string matched its recorded SHA256. A fresh primary-host command piped a stdlib-only reader into `ssh tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -` and independently matched installed file hashes:

| Source | SHA256 |
| --- | --- |
| warp/_src/context.py | eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee |
| warp/_src/utils.py | 01bd20f279f5762881c50cf8b6d766734fe21bd5f9b1ec286b34b1f3fcb27f24 |
| warp/native/warp.cpp | b9feea467eb11e0d9016ca33fceb7b1a4358fc21c65513a692378155f1d05fc7 |
| warp/native/warp.cu | 7c7e6e7a02db7b285ca1800d0ec52e28d2cf8a704435d925c70a8c5ccd8f176b |
| warp/native/cuda_util.cpp | 21d9dfe57b861ada6b98d0fa2ccfb3b8ed0b14972ed840eb04d0ad7e5f70a89f |
| warp/native/cuda_util.h | 59207856d17907bca935612ccb3ce7cde0952d8491011e1d45e8bb2d5b9ed1b8 |
| warp/native/warp.h | 0bb164157668ad83ce5fe78d3a583c68258ee333ce37d97e746b5d85a2e94511 |

The last two headers remain read source rather than a claim of membership in the current pinned environment inventory. No library was loaded or rebuilt by this review; source/binary correspondence retains the existing trusted dependency boundary.

The same read-only remote process reported Python 3.12.13 and an empty numpy/torch/warp module set. Git used `--no-optional-locks`. At inspection, runtime-custody worktree was clean detached `71566e0ffd51283071f56903bfd414291348f8b1`; original binding worktree was clean detached `a3d750bbaf79d760a6727e56146d277e1cb3b770`. The author had explicitly reported advancing the former from the packet's bcc1bac while the local design stayed frozen. I did not change either tree or inspect/signal the author's running matrix. Author-reported matrix/replay progress is not independently verified by this report and was not used as native-scope acceptance.

Consulted primary API documentation: [NVIDIA context management](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__CTX.html) distinguishes the calling thread's current context and API result status; [NVIDIA primary-context management](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__PRIMARY__CTX.html) exposes primary-context state independently of retaining it. The proposed explicit result checks and separate active-state fields are appropriate. [Linux dlopen documentation](https://man7.org/linux/man-pages/man3/dlopen.3.html) supports using RTLD_NOLOAD to avoid newly loading an absent object, but that flag alone does not certify getter side effects or library provenance.

No backend import, native query/GPU call, package installation, device reset, process signal, worker launch, remote file mutation, candidate modification or private instruction edit occurred. No pytest suite was run: the submitted change is a design document and the blocking finding is an exact-schema ordering ambiguity demonstrated directly against actual distinct index domains. Mandatory implementation mutation tests and actual native proofs cannot be replaced by unrelated existing tests. This report is the only new reviewer artifact; it is outside the candidate.

## Scope and remaining gates

Fix and independently re-review this bounded design before implementing it. The proposed bounded readback/schema/driver work remains unimplemented and unapproved by this verdict. After design approval, the exact implementation, pinned closure/policy and new observer ABI/call graph still require independent code/native-scope preflight before positive Warp GPU attempts. Both-lane actual asymmetric/replay/both-Go2 runs, nonempty Warp D2H, required observer/foreign/UNKNOWN/stream-only/missing-scope negatives, persistence/replay tests, the complete existing native/ACK/EOF/hang/cleanup/exit matrix, C0/source regressions and fresh whole-C2d3 review remain required.

No full C2d3/C2d/C4 approval, final graph seal, R_setup/R_final, prepared-sample authority, measured optical work, performance claim or backend decision is issued. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload, R2 and shipping remain CLOSED. C4 remains active. The author may archive this complete unchanged report using the evidence-only exception without enlarging its verdict or scope.
