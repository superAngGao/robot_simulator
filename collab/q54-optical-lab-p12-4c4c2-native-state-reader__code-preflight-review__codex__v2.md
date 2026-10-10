Initiative: Q54 / P12.4 / C4c2
Stage: Fresh independent exact-code/native-call PREFLIGHT review, review only
Author: Codex, independent reviewer /root
Version: v2
Date: 2026-10-10, Asia/Shanghai
Status: in_review
Session: 01a1251b-f2f2-77f1-9b5c-2c815bedf4db
Related Files: native-state-reader design-request/design-review v1; code-preflight-review v1; worker-intake gap-review v1; native-scope-aggregate v2; native-scope-cache v1; runtime-custody implementation-plan v3; runtime-integration v3; fixed-CUDA-artifacts v2; source-custody v1; runtime-adoption v2
Owner Summary: Request changes. The 13 original counterexamples now reject correctly, and 439 committed CPU tests pass. Two additional caught-failure paths remain: upload-context source rejection can still COMPLETE, and raw reader getter failure can still query again and ordinarily release. This revision may not proceed to new native validation.

**Verdict: Request changes.** Exact HEAD `5d804f73b5ae00e89fc18455080b1e339d20dade` may **not** proceed to bounded native setup/framed-worker validation. The findings below are independently reproduced CPU contract failures. They are not observed GPU failures, dependency failures, or hardware acceptance results.

**Identity, independence and preservation**

| Item | Verified value |
| --- | --- |
| Candidate | `C:\Users\AG\新建文件夹\c4_native_reader_code_review_v2` |
| HEAD | `5d804f73b5ae00e89fc18455080b1e339d20dade` |
| Prior reviewed candidate; verified merge-base | `4dcf73d78ef36dbfd40ac3025a228cf9be1b21cd` |
| Original reader base; verified merge-base | `868225db0ac5794bd7089cd73357b5c920d0c0e9` |
| Branch | Detached HEAD |
| Initial/final tree | Clean, including untracked files |
| Delta from prior candidate | 13 files; 647 insertions, 67 deletions |
| Full reader delta | 22 files; 1,799 insertions, 616 deletions |
| Whitespace | `git diff --check 868225d HEAD` passed |
| Approved design LF SHA-256 | `0b6dab55c496e730219dfbb253b7d9530dff40faa6c90676c71a2b90e0f41ca5` |
| Prior full review LF SHA-256 | `604e8711e6d71395f4f272c368bc8c46e6319b2bf400d4af9c8f8f9fd29f259e` |
| Current reader LF SHA-256 | `58f473ac75dc3031493689ca2df0be5752184292c4c175658de16c7150c260c1` |
| Bootstrap LF SHA-256 | `5a8f54ce74f9f991f11ad554cf0a7fa1a5f24793c48be4738a2444aa49fff277` |
| Catalog v18 LF SHA-256 | `c73965d822269b82a48547ad301c2faf42b0d88e69ae2c4d920b4c5f84f6a86a` |
| AST policy digest | `c92b24b367e393fd8c4732be86f0bd97f94f47c6f0e6d09809de5894a698d21c` |
| Integrated runtime contract LF SHA-256 | `cff830f90fca7c966916ceb43b7ad0dfebd621011df66b9aa66d313653c7ebe2` |

This fresh session received the review packet, not the implementation conversation, and did not author the candidate. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully in UTF-8, the approved design, design-review C1, the full prior code review, predecessor requirements and worker-intake F1/F2/F3. Author notes and prior verdicts were not treated as proof.

No agents, candidate edits, installations, resets, signals, GPU queries, backend execution or native-library handle acquisition occurred. SSH was read-only on `tileops-dev`; no fallback or host switch occurred. The optional host-routing decision file is absent in this checkout. All new scripts, tests, logs and this report are confined to the requested v2 scratch. All 14 existing files under v1 scratch, including the original `test_independent_identity.py`, were rehashed and remain unchanged. See `identity_hashes.json` and `final_audit.json`.

**Disposition of the original findings**

- **R1 original triggers: closed for this revision.** `_ticket` now rejects failed/compromised/terminal-failed state; normal reads require the existing armed setup; terminal reads require TERMINATING, the original started target sync, retained custody, unauthorized release and PARTIAL_FROZEN guardian. Live emergency ticket lifetime alone is no longer sufficient. Checks precede `_state`'s first getter and apply equally to raw calls. Original post-release and explicit emergency `_fail` probes now reject without incrementing the modeled stream getter. Committed tests also cover successful active sync, failed sync, post-sync/pre-release and cleanup. Original ticket/deadline, one sync and no retry are preserved.
- **R2 original triggers: closed, but caught rejection is not complete; see R2-followup.** Guardian-held source selection rejects removed/replaced controller links. Registered upload consumption checks the retained source. Mandatory input entry rejects wrong phase. Reader copies replacing the source record and binding copies fail exact guardian custody/record comparisons. Caught `require_reader`/`_ticket`/input-entry rejections now poison the guard. Both-lane original readback, copied-reader, copied-binding and caught-rejection probes pass after inversion; the original Warp upload allocation sentinel is no longer reached.
- All **13 original parametrized defect cases** were adapted only for v2 fixture custody/record changes and inverted into repair assertions. They pass in the final independent run. This does not erase the distinct counterexamples below.

**R2-followup — P2, blocking: registered upload-context source rejection remains non-sticky.**

Locations: `optical_worker/native_setup.py:1412-1430` (`_consume_upload_context`), particularly the source-session identity check at line 1424 and raise at line 1426; called before reader admission by `_upload_cuda_native_input:1548` and `_upload_warp_native_input:1739`.

Trigger: during an armed captured-input setup segment, create the actual registered upload context; change its `source_session` projection to None; invoke the actual lane upload entry and catch the rejection. The existing registry still holds the original controller and source. This is the supported review's local identity/rebinding counterexample, not a claim of hostile-Python sandboxing or an IPC exploit.

Observed result, both lanes: the source mismatch is correctly rejected before native work, and its registration is consumed, but `lifecycle._replay.failed` remains false. A replacement context can be created and consumed on the same lifecycle, and the segment records `normal_disarmed` after COMPLETE. The observed result is `{failed: false, normal_complete: true, replacement_admitted: true}`. No allocation/copy was needed to demonstrate this failure.

Cause: `_UPLOAD_CONTEXTS.pop` and the tuple identity checks, including the newly added source identity, run **before** the `_observe_public_operation` block. The block covers only the later controller/source comparison. Burning one context prevents retrying that particular object; it does not mark the original armed operation failed.

Consequence: captured-source admission failure can be caught and converted into a successful normal guard, contrary to R2's requested sticky caught-rejection rule. This is narrower than a claim that the rejected context itself reaches a native call; it does not.

Reproduction: `test_independent_identity.py::test_caught_upload_context_source_rejection_is_sticky[cuda]` and `[warp]`. Both invoke the real upload entry, assert the intended pre-call rejection, then independently inspect failed state, renewed context admission and normal completion. Both fail their repaired-behavior assertions in `pytest_review_probes.log`.

Why existing checks miss it: new integration tests mutate `lifecycle._native_source_session` or session phase while leaving the registered context's own source field intact. Those reach the guarded comparison. Existing upload-context rebinding tests prove registration burn on simpler contexts, not sticky failure of a live captured guarded segment.

Direction: for an existing registered context, use its retained original controller as failure authority while validating the context's identity fields, including source, so a caught rejection inside the original active guard is recorded before raising. Preserve consumption/burn, pre-native rejection and C0 behavior. Do not trust a substituted context.lifecycle for poisoning, or add another registry/guardian.

Required repair proof: both lanes must reject at the original upload entry with zero scope entry/allocation/copy/query, poison the original active guard even when caught, reject continuation/replacement admission and emit no COMPLETE. Preserve zero emergency retry and no ordinary release after that failed segment. Include the original v1 cases in re-review.

**R3 — P2, blocking: the shared raw-state query does not poison on its own getter failure.**

Locations: `optical_worker/native_scope.py:356-432` (`_state`); `_result:271-275`, `_getter:277-297`; callers `tests/support/native_runtime_process.py:549-550` (`_NativeScopeProbe._state`) and `:571-594` (later shared-helper diagnostics). `snapshot:434` has the failure handling that direct raw access lacks.

Trigger: in an otherwise healthy armed setup or active original emergency sync, call the actual `_state(False)` through its raw diagnostic interface; have the modeled `cuCtxGetCurrent` return error 999; catch that error; then attempt another state query. The probe establishes the original pinned-Device cache path first. Only the native getter is modeled; actual reader, guard peer, lifecycle and terminal release logic execute.

Observed result in **both** normal and terminal cases:

`{failed: false, rejected: false, more_getters: 1, release: "completed"}`

The first call preserves an error-code raw row and emits no successful snapshot. It does not mark failed/UNKNOWN or terminal_failed. After restoring the getter double to successful behavior, the second `_state` returns and reaches the modeled native stream getter once more. The normal case completes its setup guard and subsequently releases through one modeled emergency sync; the active-sync case completes that original sync and releases. Assertions are made outside the synchronization callback, so lifecycle exception catching cannot turn an assertion failure into a spurious test pass.

Cause: `_ticket` now handles admission failures correctly, but `_state`'s query body is outside any try/_fail boundary. `_result` and `_getter` record/raise without poisoning. Only `snapshot` catches those failures and calls `_fail`. The retained diagnostic facade calls `_state` directly, including at later BVH/shared-helper sites, so those calls bypass `snapshot`'s failure boundary.

Consequence: a reader failure in the explicitly retained raw route can be caught without stopping later queries or ordinary release. Successful input snapshots remain protected; this finding does not allege that their `snapshot` getter error path leaks through. It identifies an inconsistent failure contract in the one shared query implementation, not a second fact producer.

Reproduction: `test_independent_identity.py::test_raw_query_caught_getter_error_is_sticky[False]` and `[True]`; both fail the expected sticky/reject/no-release assertions in `pytest_review_probes.log`. The error is deterministic CPU evidence, not a GPU fault result. The `_state`, `_result`, `_getter` and `snapshot` bodies are AST-identical to v1; this was an uncovered residual in the original extraction, not a new native ABI change.

Why existing checks miss it: normal getter-failure tests invoke `snapshot`, whose catch poisons correctly. R1 terminal tests explicitly call `_fail` before reading again; they verify the new admission predicate, not whether a caught raw getter failure sets it. The later-BVH receipt test checks that no input journal rows are emitted, without failing the raw getter facade.

Direction: put the common query failure/retention boundary around the shared raw-state implementation so both snapshot and diagnostic callers have the same sticky semantics. Preserve the original exception and ordered prefix; no retry or successful complete row after failure. Keep the query algorithm single-sourced and add no wait, registry or guardian.

Required repair proof: caught CUresult failure and caught Python/C10/Warp getter exception must fail setup/active terminal authority before another query; assert no later getter, normal completion, new emergency attempt or ordinary release. Exercise the actual diagnostic facade as well as successful raw later-helper calls, which must still emit no input receipts. Preserve successful-null/error distinction and existing snapshot/cache/retention tests.

**Code, state and resource assessment**

The following checks support the parts of the repair that work; they do not override the blocking findings.

- **One resource graph:** SourceSession retains the reader and its held/raw containers before acquisition/query. The existing guardian also adopts the exact reader. Libraries/functions are retained before metadata configuration/calls. Stream return remains in runtime_owners and its returned-owner journal before reader cache/after work; binding is held/adopted/journaled before binding_returned. The extended seven-slot record freezes stream and binding identities. Exact owner checks reject replacing that record's reader or binding with an unadopted copy.
- **Source authority:** `_bound_source_session` derives the original source from existing guardian owners/roles and compares the mutable controller link. It introduces no new source registry, sync, guardian or release authority. Scanning the existing setup-owned graph adds linear CPU setup work; it does not enter measured acquire/enqueue/resolve paths or support a performance claim. The remaining failure boundary should be repaired locally rather than by adding another authority layer.
- **Actual call placement:** I independently inspected bootstrap stream/cache/binding placement; CUDA file-frame/leaf/before/delegate/retained-create-return/scope-exit/after/frame-complete ordering; and upload/readback empty/copy/event-create/record/wait and Warp outer-scope placement. Returned device/event owners are retained before after-query, metadata, append/adoption and outer exit can fail. CPU integration tests cover returned event and created module retention through getter, append, adoption and scope-exit failures. Complete contradictory rows are appended before expected-scope rejection; failed bodies skip successful after rows and unwind the original scope once.
- **Lifecycle:** R1 uses existing RUNNING/setup and started terminal-sync state. Emergency guard stays armed through cleanup/exit under the original deadline, but queries no longer inherit that whole lifetime. No extra ARM, renewal, synchronization or retry was added. Normal UNKNOWN/failure forbids emergency cleanup; terminal reader failure quarantines. R3 describes the missing event that can prevent these correct predicates from being set.
- **C1 phase boundary:** mandatory source-bound input entry now rejects phase drift; later shared upload/readback helpers return no input reader and produce no input-phase receipts. The diagnostic facade uses the sole production raw query and preserves its later context expectations. C0 does not construct a reader. The outstanding raw failure issue is separate from the correct phase/receipt distinction.
- **Consolidation and replay:** no duplicate ABI/query/state producer was found in the changed test support. `_NativeScopeProbe` delegates successful observations to production and retains named faults/raw diagnostics. `_RuntimeObservation` and runtime_custody.py are unchanged by the full reader delta; historical v1/v2 meanings and v3 six-pair CUDA file/return/binding joins were not weakened. The shared input coverage validator uses actual completed input guard occurrences and exact unique site/edge/event sets, including zero-size calls. Sacrificial one-sync validation and integrated zero-emergency-sync/reclaimed/ACK-close validation retain different terminal requirements. This is code inspection, not a new execution of either native campaign.
- **Historical upload comparison:** normalization strips exact reader imports/admission statements, exact guarded observation wrappers and exact hold_returned statements, with counted prefixes/holds/sites. Warp's shared array body is still inlined for comparison against commit `956d1fa8d51b85c25ac17825adef17640f0e69b5`; allocation/copy/adoption nodes remain compared. Four independent AST mutation tests demonstrate that swapping copy operands or deleting device-owner adoption is still rejected, in both lanes. Observation ordering is additionally reviewed directly; normalization alone does not prove that ordering.
- **Source release:** all 91 committed LF source hashes match catalog v18 and pass the actual AST validator; both lane inventories and AST policy digest match. The reader and bootstrap full-source pins match. The integrated contract continues to reference v18 and has the exact digest above. Source-policy tests reject modified symbols/paths/modes and transferred exceptions despite recomputed caller hashes. No Torch/Warp/NumPy/ctypes module was imported by the independent catalog audit. The packed-render snapshot exception remains unchanged.

**Native-call preflight evidence and limits**

I reused unchanged v1 source/ABI analysis only after checking exact identities. `check_remote.py` performed read-only SSH with `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B`, reading/hashing files without backend imports, dlopen or native queries. All **16** inspected installed source/header/library byte identities match v1, including Warp context/utils/core/error sources, Torch CUDA/stream/lazy-seed sources, cuda_runtime_api.h/cuda.h and both query-library files. Exact hashes are in `remote_hashes.jsonl`; v1 source excerpts remain untouched in `remote_sources.jsonl` and `remote_native.jsonl` under old scratch. The old remote evidence checkout remains clean/detached at `26542543d067d20c64670d375065535420ede203`; this v2 candidate was not deployed there.

AST comparison also verifies that `_initialize_queries`, `_warp_cache_precondition`, `_state`, `_result`, `_getter` and `snapshot` have not changed since the exact v1 candidate. Changed admission and stream/binding call placement were inspected independently, rather than inferred from that equality.

The retained source analysis supports: CUDA `cudaGetDevice(int*) -> int`; the separate Torch count/primary cold paths and warming logical getter restriction; Warp's five checked driver signatures; exact error-buffer ABI; and the conditional cache proof before `wp_cuda_context_get_stream`. Library path/bytes/mapped-path checks still precede RTLD_NOLOAD|RTLD_LOCAL acquisition. No fallback load, setter, primary retain/release, allocation, memory query or extra sync was added to reader queries. Warp's getter is only conditionally readonly: initially absent raw context/stream, original retained get_stream return, exact empty error bytes on both sides and the positive returned/cache identity are required before enabling it. Void scope exit still requires the actual post-exit observation.

These are pinned source/ABI and CPU call-order conclusions. They do not prove binary/source rebuild equivalence, proprietary driver internals, actual device/error-buffer state or real restoration. No new GPU/native reader execution has occurred in this review or been inferred from prior evidence.

**Executed verification**

Environment: Windows PowerShell, Python **3.14.0** at `C:/Python314/python.exe`; `-B`, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, pytest cache disabled. `ROBOT_SIM_WARP_CONTEXT_SOURCE=C:\Users\AG\新建文件夹\c4_assets\worker_intake_gap_scratch_v1\pinned_warp_context.py`; fixture independently hashes these pinned bytes. All pytest basetemp directories are under v2 scratch.

Main regression command (paths below are relative to the candidate; output `pytest_regressions.log`):

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider --basetemp=<v2 scratch>/pytest_regressions
  tests/unit/optics/test_native_scope_observer.py
  tests/unit/optics/test_native_scope_integration.py
  tests/unit/optics/test_native_upload_core.py
  tests/unit/optics/test_guarded_native_readback.py
  tests/unit/optics/test_guarded_lifecycle.py
  tests/unit/optics/test_native_source_policy.py
```

Result: **439 passed in 22.28s**, no skips. This is an independent CPU run, not reuse of the author's 155-pass claim or the separately running source/lifecycle suite.

Final independent probe command (output `pytest_review_probes.log`, observed pytest exit **1**):

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider --basetemp=<v2 scratch>/pytest_review_probes
  <v2 scratch>/test_independent_identity.py
  <v2 scratch>/test_ast_normalization.py
```

Result: **17 passed, 4 failed in 3.57s**. Passes comprise all 13 inverted original cases and four historical-AST mutation checks. Failures are exactly the two raw getter cases and two upload-source caught-rejection cases described above. They remain failing expected-repair assertions, not tests reworded to declare the defects acceptable.

Additional reproducible commands/artifacts:

- `git status --porcelain=v1 --untracked-files=all`, `git rev-parse HEAD`, `git branch --show-current`, both merge-base checks, diff/stat and diff/check: exact clean candidate.
- `C:/Python314/python.exe -B <v2 scratch>/audit.py`: identity/design/prior-report hashes, all 91 LF source/AST entries and both lane inventories pass; `audit.log`, `identity_hashes.json`.
- `C:/Python314/python.exe -B <v2 scratch>/check_remote.py`: SSH exit 0, all 16 exact identities match; remote command/source preserved in `remote_inspection.py`; no fallback.
- `C:/Python314/python.exe -B <v2 scratch>/finish_audit.py`: original evidence preserved, unchanged native-query AST bodies and clean HEAD verified; remote prior inspection PIDs absent.
- Scoped local Win32_Process descendant/scratch checks: no owned Python/SSH/SCP residual; `process_closure.json`. Remote closure result: `remote_closure.json`.

Inspection/tool issues are not candidate failures: some large combined reads were truncated and followed by focused reads; the optional routing file was absent. The initial adaptation redundantly adopted a reader already held by the v2 fixture, causing two fixture failures. That duplication was removed only in new scratch. An initial terminal diagnostic assertion could be swallowed by terminate's exception handler; the final repro records observations and asserts outside it. An initial `-k` selection deselected AST probes; the final unfiltered run includes all four. Earlier logs remain preserved (`pytest_independent.log`, `pytest_independent_corrected.log`, `pytest_additional.log`, `pytest_final_probes.log`); only the final results above define the repro conclusions.

**Unexecuted work, process closure and exact scope**

No fresh native setup/framed-worker case, six-pair CUDA runtime acceptance, Warp physical restoration/failure case, Go2 native run, final-sample numerical computation, full C2f matrix, Linux process hang/kill campaign or formal performance measurement was executed. No GPU process inventory was queried, per this packet. The full historical artifact/source-custody/C0 integration suites were not all rerun; the actual selected six files, mutation probes and catalog checks above bound CPU coverage. No result from the author's concurrently running suite is claimed. Real release/destructor/device behavior remains unproved by these doubles.

Reviewer CLI PID **42964**, parent **20732**, is recorded with creation identity. All test/tool sessions returned. Read-only remote inspection PID **2467971** and its parent **2467903** were absent at closure; the closure probe itself completed synchronously with SSH exit 0. Final scoped local descendant and scratch scans found no owned Python/SSH/SCP residual. No unrelated process was signaled. These are scoped process observations, not exhaustive OS/GPU attestation.

F1 now has the intended production fact producer structurally, but real CUDA intake still requires hardware evidence after a repaired exact-revision preflight. F2 final-sample physical/numerical proof remains open. F3 predecessor compact reply/journal recovery is outside this review and is not implicitly approved. C1's input/later/C0 phase distinction is substantially implemented; R2-followup and R3 prevent approval of the complete failure contract.

**Approved scope: none for new native execution at this HEAD.** Repair the two substantiated boundaries without broadening authority, preserve these counterexamples and old evidence, then obtain a fresh independent exact-revision preflight. A later approved preflight would permit only bounded native setup/framed-worker validation; actual native acceptance would still require its hardware evidence.

Complete C2f/C4, final-sample physical/numerical acceptance, product activation, formal performance, result/baseline, backend selection, R2/S/host delivery, a3b and shipping remain closed. No subset approval or CPU pass is parent C4 acceptance. This review stops at the bounded preflight.
