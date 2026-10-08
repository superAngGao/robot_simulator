# Independent native-scope aggregate design re-review

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: bounded independent design-review
Author: Codex, fresh independent reviewer /root/native_scope_aggregate_design_v2
Version: v2
Date: 2026-10-08 (Asia/Shanghai)
Status: accepted for the exact bounded design scope below
Related Files: native-scope-aggregate design-request v2, design-review v1 and gap-review v1; runtime-custody implementation-plan/gap-review v3; runtime-adoption design v2; cuda-ambient design/review v1; optical_worker/runtime_custody.py, native_bootstrap.py, native_setup.py, native_readback.py, guarded_lifecycle.py
Owner Summary: Approve the exact v2 design for bounded implementation. The prior blocking index-domain/occurrence ambiguity is closed at design scope. The native-return comment is incorporated explicitly. No substantive design finding remains; the new codec, observer and Warp readback scope still require implementation, independent code/native preflight and actual native proof.

## Verdict and exact inspected revision

**Approve**, limited to the frozen v2 native_scope journal, occurrence joins, Warp readback scope and associated observer/failure/verification contract. This is design approval only. It neither approves the current implementation as satisfying those requirements nor authorizes a positive Warp GPU attempt before the specified independent code/native-scope preflight.

- Fresh independent reviewer: `/root/native_scope_aggregate_design_v2`. No candidate authorship, inherited implementation conversation, subdelegation or candidate edits.
- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Frozen HEAD: `07a76ff1ecd788408eab0c5c14c8673faccd6e6e`.
- Independently resolved immediate parent/base: `d9f889c08e4d0e61ba5d2b743437a12ee5f7feec`.
- Candidate: `collab/q54-optical-lab-p12-4c4c2d-native-scope-aggregate__design-request__codex__v2.md`.
- Candidate SHA256: `9e678c7aaffa1147b49a6f1f3b946619f0c316e3bc68883ff0fa9a4d1ed9a0ad`.
- Prior complete design-review v1 SHA256: `c8b51355c7a331d09a2711a03c9f34aa7494937a8713e34f56093a0c6255c348`.
- Governing native-scope gap-review v1 SHA256: `5e264476a233c42aa55929f70d4735e2258059df2481b32d4ee87f92f9d8dc66`.
- Initial and final tree clean; HEAD/candidate digest unchanged. Base-to-HEAD adds only the 367-line v2 design. `git diff --check` passed. The `optical_worker` comparison against `9b5702b` is empty.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely before substantive review. The first PowerShell read garbled the Chinese principles; an explicit UTF-8 reread corrected it. Read the complete candidate and prior review, governing gap report and relevant predecessor contracts, with focused rereads after oversized output. Prior verdicts and author descriptions were treated as claims and scope information, not approval evidence.

## Findings and disposition of the prior review

No substantive blocking or new non-blocking design finding was established.

### F1 closed at design scope: two index domains and exact occurrences

Candidate lines 121-151 now expressly distinguish journal indices (`index`, `before_index`) from lifecycle indices (`operation_event_index`, `lifecycle_event_index`). Every native row requires `0 <= operation_event_index <= envelope.lifecycle_event_index < observation_cut.lifecycle_event_count`. This removes the wrong comparison to journal position.

The repair is supported by actual code, not just the prior report. `_RuntimeObservation.append` at native_bootstrap.py:2112 independently uses journal length for `index` and guard-event length minus one for `lifecycle_event_index`. Runtime replay at runtime_custody.py:559 advances these domains separately; the existing returned_owner check at :633 already bounds its lifecycle leaf against `self.lci`.

Fresh in-place inspection of the actual retained archive independently reproduced the original counterexample. In `test_actual_cold_canonical_war0/runtime_custody.json`, binding_consumed is journal 1342 / lifecycle 438, under setup guard sequence 4. Lifecycle 438 is guard_armed; indexed_upload_scope enters at 441, allocation at 447 and copy at 451. A hypothetical native row at journal 1343 / lifecycle 438 referencing operation 447 satisfies `447 <= 1343` but fails `447 <= 438`. The revised contract therefore excludes the original future-leaf transplant. This was arithmetic over actual historical evidence, not execution of a nonexistent v2 parser.

The new most-recent-entered-leaf BEFORE rule also closes stale same-guard and same-kind occurrence reuse. It applies to runtime_stream, helpers and opening scopes, under the actual phase/guard and expected descriptor. Helper/runtime_stream AFTER repeats its opening operation and permits no intervening different entered leaf. The enclosing scope exception correctly retains its opening operation while admitting actual nested helper leaves and nested pairs. The producer's intervening adoption events remain legal: actual archive lifecycle 442 adopts the upload context after scope leaf 441, and 448 adopts the returned allocation after leaf 447. Requiring envelope and operation indices always to equal would reject these real windows; v2 explicitly avoids that mistake.

Binding POINT and terminal pairs have separate meaningful rules. Binding refers to the exact binding adoption and actual immediately preceding runtime_binding journal fact, with no later leaf or phase change. Terminal rows refer to the original emergency_sync_started under its live emergency guard, before success/failure. These do not borrow the normal leaf rule or a setup guard. The unchanged setup restrictions still apply to all other kinds. Candidate lines 335-346 require concrete distinguishing mutations, including this actual unequal-index case, stale/repeated leaves, pair crossing/reuse, lawful adoption/nesting positives and emergency boundary substitutions.

### C1 incorporated: returning from Python exit is not restoration proof

Candidate lines 65-70, 288-291 and 347-350 explicitly require direct observed post-exit state and distinguish a successful null context from a getter error. If exit returns but target1 remains current, the completed contradictory AFTER row is retained before sticky failure; no subsequent normal leaf, COMPLETE, target sync or ordinary release is admitted. After an earlier sticky body failure, the original scope unwinds once and no further native query or successful AFTER row is manufactured.

Independent retained-source inspection and fresh installed-file hash verification confirm the premise. `warp/_src/utils.py:1267-1284` saves/restores default-device metadata and delegates context entry/exit. `_src/context.py:3153-3167` delegates CUDA context push/pop. `native/warp.cu:2165-2170` defines void native wrappers using check_cu; Python exit does not receive a CUresult from them. Therefore the required direct query and sticky failure tests are necessary. C1 is now explicit design text and a mandatory implementation proof obligation, not an unresolved design comment.

## Affected invariants and feasibility

The Warp correction is feasible within the existing graph. Current native_readback.py validates source/prepared/lane/target/stream identities and storage, adopts a host-retention list, then performs the array loop, event/create/record/wait, storage recheck and shared CPU validation. It currently lacks the proposed outer Warp scope. Factoring that body under one indexed_validation_scope after the existing checks/retention setup preserves the existing helper and validator authority. The report is retained/adopted inside that body and returned to SourceSession only after scope exit. The additional target leaf accounts for native context selection without another ARM or deadline; work counts derive from actual declared operations and must increase by exactly the intended one for Warp.

Actual Warp upload already encloses its helpers in ScopedDevice at native_setup.py:337, while terminal target synchronization uses its own ScopedDevice at :125. The proposed interior target-context checks and outer default0/null checks therefore describe different real boundaries. CUDA retains its existing logical/raw policy and nested device/stream copy helper; the design does not generalize that exception to Warp.

The original missing-scope negative has a concrete source basis. `_warp_copy_exact` calls wp.copy; `_src/context.py:8827-8829` dispatches nonempty D2H to wp_memcpy_d2h. `warp.cu:894-910` creates a native ContextGuard with its default policy, while cuda_util.cpp:146 sets always_restore false and cuda_util.h:253-270 restores only when need_restore is true. An enclosing Python push/pop scope supplies the separate restoration boundary. This is source-based feasibility, not a new real-device restoration result.

The proposed poststream getter remains conditional on a real cache preflight. Device.context initializes streams before restoring the previous context; `_init_streams` calls set_stream, which calls wp_cuda_context_set_stream. The native setter consults get_context_info. That function's cached branch returns existing state, whereas its cache-miss branch can switch context and allocate/free a byte. V2 explicitly requires proving the same original stream path established the cache before allowing the observer getter, with no fallback. An implementation that merely assumes getters are harmless does not satisfy this approval.

The aggregate remains descriptive and has no new live authority. V1's exact grammar and bytes retain their old meaning; v2 without coverage cannot pass native acceptance. Current runtime_custody.py pins one version in both record and durable selection validation, so shared explicit v1/v2 dispatch is real implementation work, not already-completed behavior. Tokens remain tied to actual held module/runtime/target/stream/binding objects and upstream environment/source facts; a version string or copied hash cannot establish execution. External PID/start/ancestry evidence remains required beyond the repeated worker/TID fields.

Returned-owner ordering is implementable at the stated existing hooks: runtime stream owners enter runtime_owners and returned_owner before fallible metadata/adoption; helper wrappers must retain the actual result before observing or returning it for adoption. No production observer callback, extra uploader, backend import, storage walk or sync is introduced into CPU collection. Keeping raw diagnostics and immutable validated cuts avoids fabricating rows after getter/append/persistence failure.

The terminal exception also fits the actual controller. guarded_lifecycle.py:426 records emergency_sync_started, invokes the sync callback inside try, records failure and quarantines on an exception, and records success/releases only after the callback returns. A post-sync observer failure can therefore use the original failure path without a second sync or premature release. It must not run after original sync failure or compromised/sticky history. Lifecycle remains the synchronization authority; native rows do not claim completion themselves.

## Verification, commands and limits

Executed local UTF-8 Get-Content reads, rg/rg --files searches, Git revision/branch/status/diff checks and local `C:/Python314/python.exe -I -S -B -` stdlib-only scripts. Compared the complete v1/v2 document delta. No candidate implementation or tests were changed.

The actual parent is d9f889c08e4d0e61ba5d2b743437a12ee5f7feec, confirmed by `git log -3 --format='%H %P %s'`. An initial parsing of the compact packet omitted its leading `d`, producing an unknown-revision command error; the coordinator clarified the intended base and independent Git history agreed. An initial piped Python script containing the Chinese absolute path suffered PowerShell encoding substitution and failed before reading files; rerunning with `Path.cwd().parent` succeeded. These are inspection-command errors, not candidate failures. `git diff --no-index` returned expected exit 1 for differing v1/v2 documents.

The local stdlib script verified the three document digests above and archive SHA256 `eb08d4ed62f4271596a7b8a1e982f6cd7c2b7f3ab8ec7488319c6bb9aeab5510`. It enumerated twelve runtime_custody.json members and read the named actual record without extraction. It printed the concrete lifecycle/journal facts and wrong-bound true/correct-bound false result. I did not rerun all twelve old parser/attachment validations; their previous review evidence remains historical, and none is a v2 execution.

All seven embedded source strings in `c4_assets/native_scope_warp_source_read_v1.jsonl` and `native_scope_warp_utils_read_v1.json` matched their recorded hashes. A fresh read-only command piped a stdlib pathlib/hashlib/json/sys/subprocess reader through `ssh tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -`. Installed file hashes independently matched:

| Warp file | SHA256 |
| --- | --- |
| _src/context.py | eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee |
| _src/utils.py | 01bd20f279f5762881c50cf8b6d766734fe21bd5f9b1ec286b34b1f3fcb27f24 |
| native/warp.cpp | b9feea467eb11e0d9016ca33fceb7b1a4358fc21c65513a692378155f1d05fc7 |
| native/warp.cu | 7c7e6e7a02db7b285ca1800d0ec52e28d2cf8a704435d925c70a8c5ccd8f176b |
| native/cuda_util.cpp | 21d9dfe57b861ada6b98d0fa2ccfb3b8ed0b14972ed840eb04d0ad7e5f70a89f |
| native/cuda_util.h | 59207856d17907bca935612ccb3ce7cde0952d8491011e1d45e8bb2d5b9ed1b8 |
| native/warp.h | 0bb164157668ad83ce5fe78d3a583c68258ee333ce37d97e746b5d85a2e94511 |

This verifies source identity, not a new binary rebuild or proof of proprietary driver internals. Header inspection does not claim those headers are already members of the captured inventory. The admitted source/binary native dependency trust boundary remains binding.

The remote reader reported Python 3.12.13 and no numpy/torch/warp modules loaded. Git commands used `--no-optional-locks`. The runtime-custody worktree was clean at `71566e0ffd51283071f56903bfd414291348f8b1`; the original binding worktree was clean at `a3d750bbaf79d760a6727e56146d277e1cb3b770`. The author-owned active matrix was not inspected for results, modified, signaled or used as acceptance evidence.

No backend import, library load, GPU/native query, worker launch, reset, package installation, process signal, remote mutation or private instruction edit occurred. No pytest suite was run: the submitted change is a design document, and the new v2 codec/observer/readback implementation does not yet exist. Source inspection and the arithmetic counterexample are not substitutes for its required implementation tests or native validation. This report is the only new reviewer artifact, outside the candidate.

## Mandatory implementation proof and remaining scope

Before any new positive Warp attempt, implement only the approved bounded changes, repin the captured source closure/catalog/policy, freeze the result, and obtain fresh independent code/native-scope preflight. That review must inspect actual ABI/result handling, already-loaded library identity, cached getter path, original delegation exactly once, return retention, failure unwinding and the unchanged guard/deadline authority.

Required CPU proof remains the actual implemented codec mutations and positive cuts in candidate lines 328-350: fixed trusted launch inputs; wrong-version/field/type/token/root/guard/phase rejection; concrete unequal-index/stale/same-kind occurrence rejection; proper adoption/nesting positives; crossed/duplicate/reordered pair rejection; separate emergency interval checks; contradictory factual rows retained but rejected at acceptance; getter error distinct from null; exit-return-without-restoration; owner/append/serialization/persistence failures and stale durable pairs versus full terminal traces. Do not report an illustrative predicate as a passing production parser test.

After exact code preflight, required native proof remains both-lane asymmetric/replay/both-Go2 positives, actual nonempty Warp D2H restoration, observer return/failure cases, foreign stream and UNKNOWN pre-return negatives, CUDA stream-only and Warp missing-scope counterexamples, actual exit/reap/residual evidence, the full native/ACK/EOF/hang/cleanup/exit matrix and C0/source regressions. Then obtain fresh independent whole-C2d3 review. Historical records and the author's current matrix cannot fill the new same-execution native_scope coverage requirement.

This approval does not open whole C2d3/C2d/C4, final graph seal, R_setup/R_final, prepared-sample authority, measured optical work, performance or backend choice. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload, R2 and shipping remain CLOSED. C4 remains active. The author may archive this complete unchanged report under the evidence-only exception without enlarging its verdict or scope.
