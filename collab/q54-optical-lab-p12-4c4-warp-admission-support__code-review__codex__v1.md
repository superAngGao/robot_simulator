# Warp post-compilation admission test support: independent code review

Initiative: q54-optical-lab-p12-4c4-warp-admission-support
Stage: code-review
Author: codex
Version: v1
Date: 2026-10-10
Status: accepted
Related Files: tests/support/native_runtime_process.py; tests/support/cuda_artifact_runtime_acceptance.py; collab/q54-optical-lab-p12-4c4-warp-module-preparation__implementation-note__codex__v7.md; Warp module preparation design-request v2; runtime-integration design-request v3 sections 5 and 8
Owner Summary: Approve the frozen test-support delta for bounded supervised native execution of its post-compilation mutation selectors. This is not native acceptance, compilation acceptance, or completion of C4.

Reviewer session: /root/warp_admission_support_v1, fresh independent review session supplied with the review packet and repository access. The reviewer did not author or modify the candidate and did not delegate.

Verdict: **Approve**.

## Exact revision and scope

Repository: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design

Actual branch: implementation/c4-supervised-waits.
Base: 63051811cc84495862d7da43d5049bc4297180c8.
Reviewed HEAD: 4fe6b3ae71ac6aa1586cad5f4f7ac66c6717a2ec.
HEAD was independently read at the beginning and end. Working-tree porcelain output was empty at both checks. The actual diff contains exactly the two support files and implementation note above: 146 insertions, 10 deletions. No production file changes are part of this delta.

The Warp design-request v2 digest was independently verified as 194b3c25130df77954f5b2de2b556da81009b98f94efb861848ae06566a41a5b. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md in full, the Warp supplement, the implementation note, and parent runtime-integration v3 sections 5 and 8. Earlier verdicts and the author's test claims were not treated as proof of this candidate.

Approval covers the twelve added post-compilation selectors and the strengthened common checking of all fifteen admission selectors, using the existing supervised native test route. No remote update or native execution was performed by this reviewer. This approval does not itself record any native case as passed.

## Findings

No substantive findings or blocking evidence gap for this bounded support-execution review.

High-risk paths checked against actual code:

- The mutation branch is reached after the production preparation entry has returned and after an unchanged binding/module-set validation within setup guard sequence 10. Each mutation is followed by the production binding check, held-source audit, external audit and module-set validator; it does not substitute fixture rejection logic. If the mutation is admitted, the explicit AssertionError differs from every permitted error tuple and therefore fails the runner.
- The expected-error table matches the production predicates: configuration and cache identity/permissions in validate_warp_cache; options in definition_owners; artifact symlink containment in _artifact_bytes; bytes/metadata comparison in validate_module_set; executable mapping identity; registered module-set identity and projections; and exact compilation-record registry identity. Target/stream substitution reaches _validate_runtime_binding_resources before stream introspection or a native operation. The source-cache mutation reaches the captured loader's exact linecache-content audit.
- The options and backend mutations change values whose literal accepted defaults are false. The mapping mutation replaces the dict while preserving its contents, so identity rejection is exercised. NamedTuple _replace creates a distinct record while the original remains in the issued tuple. The copied set preserves projections but lacks registry identity. The changed-session case is accurately described as projection corruption, not a real second-session provenance test.
- Mutations do not discard original native resources: the session/guardian already holds the binding's original target and stream, compiled records, executables and mappings; the issued registry also holds the original record tuple and module set. Replacing an exposed mapping or record does not remove these original owners. Copied sets/records and replacement CPU objects create no new native handle. Artifact mutations operate on the known generated first CUBIN under the worker's private cache. Original bytes are saved with exclusive creation before alteration; the redirect points to that saved evidence copy. No shared repository or installed dependency bytes are changed.
- The failure occurs within the next original setup guard. The existing segment/session failure paths preserve the failed guard, transfer custody to quarantine and use hard exit. The delta adds no synchronization, guard, retry, ordinary unload, new process manager or cleanup destructor. Source audit can initiate failure before the segment unwinds; the existing termination-evidence checks prevent a second guard failure or disposal.
- The runner still performs production custody-artifact replay and runtime-custody selection/attachment validation. It additionally requires exit 74, ten normal completions, eleven arms, exact error type/message, one guard_failed event for sequence 10, full_quarantine, no low_level_leaf_entered after the recorded mutation marker, and no successful warp_module_binding.json. The surrounding code requires zero emergency sync attempts, terminal supervisor state, actual worker PID disappearance and absence of ordinary atexit cleanup. A prior preparation failure cannot pass because the intended marker has not yet been written.
- The implementation note keeps native evidence and later gates explicitly unfinished and describes the target/stream and changed-session cases within their actual evidentiary limits. No production dispatcher, result authority, backend selection or performance scope is added.

## Independent verification

Environment: local Windows PowerShell, C:/Python314/python.exe, Python 3.14; bytecode disabled. No Warp/CUDA dependency execution or remote connection was used.

Executed from the candidate repository:

    C:/Python314/python.exe -B -m pytest tests/unit/optics/test_warp_module_preparation.py tests/unit/optics/test_warp_failure_diagnostics.py tests/unit/optics/test_cuda_artifact_runtime.py -q -p no:cacheprovider

Observed result: **60 passed in 10.14s**, terminal exit 0, tool session 83002. An earlier invocation of the same command was included in an overlarge aggregate tool response; its result is not counted separately as verification. The recorded standalone invocation above supplies the test result.

Executed reviewer scratch probe outside the tracked tree:

    C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/warp_admission_support_review_v1_probe.py

Observed result: terminal exit 0; 15 valid synthetic branches accepted and 135 corrupt variants rejected. The probe extracts and executes the actual Warp admission runner branch from the candidate AST. For each selector it checks unrelated error text, the explicit admitted-mutation AssertionError, wrong failed guard, wrong custody, a later native-leaf event, a success-binding file, wrong worker exit, extra completion, and extra arm. This tests those runner assertions only; its synthetic rows are not native evidence and do not test the surrounding full artifact replayers. The probe also confirmed 86 unique selectors, 15 admission selectors, and compiled both support files in memory without writing bytecode.

Other verification: git diff --check over the exact base/HEAD succeeded; exact changed-file inventory, branch, final HEAD and clean status were rechecked. Scratch temporary directories were removed by their TemporaryDirectory contexts. The retained probe script and this report are outside the candidate tree.

## Limits and remaining work

No new selector has been run on a native device in this review. CPU unit doubles and synthetic runner rows do not establish CUDA context/restoration, installed Warp behavior, compilation success, actual target1/ambient0 observations, native custody or supervisor behavior. Real execution may expose environment/toolchain failures or assumptions requiring a revised candidate and independent review. Exact production predicate matching here is static code evidence, not a claim that native outputs have already matched.

The new source-cache case checks rejection after mutation; it does not prove held source was consumed despite stale linecache or changed on-disk source. The changed-session selector is not a genuine foreign live session. The target/stream cases substitute CPU objects and do not show another GPU being used. Pre-compilation cache/executable admission, source-consumption evidence, required input/fixed-CUDA regressions, the complete compilation acceptance matrix and independent native acceptance remain outstanding as described in the implementation note.

One-slot graph, Warp timeline/arena, ten actual warmups, new final readiness and final recording/seal/barrier, rendering correctness/lifecycle matrices, C4c3/c4, C4d/B3, whole-C4 systematic refactor and fresh whole review remain unfinished. Product PREPARE/submission, performance/benchmark results, baselines, backend selection and later product gates remain closed. Approval of this support delta changes none of those gates.

## Owned process terminal state

The recorded pytest invocation and scratch probe both reached exit 0. At the final local residual-process check, Win32_Process returned no Python process whose command line matched the reviewed pytest entrypoint or the scratch probe. No reviewer-owned local test process was left running. No native worker, supervisor or remote process was started, signaled, altered or reaped by this reviewer.

The separately running remote batch at reviewed revision 552d1b7 (packet identifiers parent 3074957 / tool 82019) was deliberately untouched. Its status and terminal outcome were not inspected or claimed by this review; its existing owner remains responsible for its scoped terminal-process check.
