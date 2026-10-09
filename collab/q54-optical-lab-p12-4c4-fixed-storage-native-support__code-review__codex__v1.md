# Independent fixed-storage native-support code review v1

Initiative: Q54 / P12.4 / C4 fixed render storage
Stage: independent code review, bounded native validation support
Author: codex, independent reviewer /root/fixed_storage_native_support_v1
Version: v1
Date: 2026-10-10 (Asia/Shanghai)
Status: accepted
Related Files: runtime-integration design-request codex v3 sections 5, 8, 9; remaining-setup gap-scan codex v1 G2; fixed-render-storage code-review codex v1; fixed-storage-native-support implementation-note codex v1
Owner Summary: Approve this four-file test-support delta at 3d6692bb31145d70b4b841dbbc82f1183cd6a58f. No substantive finding. Independent focused regressions: 150 passed, 41 skipped in 16.90 seconds. This approves support for the subsequent native validation; it does not establish native allocation correctness, close G2, or approve C4 or later product gates.

## Identity and exact scope

Reviewer session: `/root/fixed_storage_native_support_v1`, started with a fresh review packet, without the implementation conversation. I authored no candidate code and delegated no work. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md, the requested parent sections and G2, the prior bounded storage review, and the current implementation note. Prior reports and author test results were claims to check, not evidence for approval of this delta.

Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
Branch: `implementation/c4-supervised-waits`.
Base: `219d7d4c2d381020ce5855c29c28128ad0776315`.
Reviewed HEAD: `3d6692bb31145d70b4b841dbbc82f1183cd6a58f`.
HEAD tree: `ac0e9bc7ecdea313e356a99f4321bb23b7891922`.
Initial and final Git status were clean; HEAD stayed unchanged.

**Verdict: Approve**, solely for:

- `tests/support/native_runtime_process.py`
- `tests/support/cuda_artifact_runtime_acceptance.py`
- `tests/unit/optics/test_prepared_fixed_render_inputs.py`
- `collab/q54-optical-lab-p12-4c4-fixed-storage-native-support__implementation-note__codex__v1.md`

The base-to-HEAD diff contains exactly these four files. An independent `git diff --name-only 12c1619 HEAD -- optical_worker optics` returned no paths: production code in those directories remains unchanged from approved `12c161969a6393722510cb3c373e2a0e3c25a637`. That prior production approval is neither reopened nor expanded here.

## Findings and inspected behavior

No blocking or substantive non-blocking defect was found in this bounded support delta.

1. **Actual entry and selection.** The five literal selectors are included in `RUNTIME_FAULTS`, which the existing supervisor imports for child dispatch and argument validation. They bypass unrelated legacy injection. The child uses the original verified SourceSession, runtime binding, upload/validation path and lane-specific factory/module preparation, then calls `SourceSession.prepare_fixed_render_storage`. Warp retains its existing compile and same-session executable reuse observations. CUDA retains its captured factory binding checks. No new production callback, supervisor, controller or source-policy exception is introduced. The acceptance function supports explicit new-selector calls; the script's existing default CASES inventory is not silently expanded.

2. **Retention is production-owned.** `_prepare_observed_fixed_storage` replaces the runtime allocator only around the production storage call. After `original_empty` returns, its wrapper immediately returns the result without metadata inspection or another native query. Production `_output_array_allocation` appends that result to its pre-adopted `slot_array_custody` list before device metadata and adoption. The support adoption hook checks that actual guardian-held list, then records only IDs, shape, dtype, ordinal and operation strings. It does not insert output objects into an observer-owned list. The second-allocation fault therefore observes one returned array; the second pre-adoption fault observes two returned and retained arrays. The acceptance check requires exactly those counts and matching unique retained IDs. Success checks all 14 ordinary or 23 diagnostic arrays and exact prepared-input identity. Output ID sets deliberately avoid imposing CUDA allocation order on its accessor order.

3. **Original failure and failed-ticket discipline.** The allocator's existing public-operation failure handling fails the guard before the wrapper's catch runs. That catch performs only CPU root inspection and JSON diagnostics. It makes no native restoration query and no live-ticket request. A diagnostic write exception is retained separately, followed by a bare re-raise of the primary error. The outer finally restores all three replaced callables. Tests exercise both lanes and allocation/adoption failures, including failed diagnostic writes; their native-state doubles require a live guard, so a post-failure native query would fail those cases. There is no failure-path restoration claim.

4. **Scope observations have bounded meaning.** Allocation observations reuse the established driver/runtime getters under the live setup guard. They check the held target stream, target ordinal 1 and exactly the target primary context active. Warp additionally observes default device and current context device; CUDA observes logical and raw device. Success compares complete before/after snapshots and requires the original ambient state (Warp default 0 with no current context; CUDA logical 0/raw 1, reflecting its established deferred-device behavior). These are downstream raw storage observations, not fabricated input-readback receipts. The Warp scope wrapper explicitly avoids classifying storage scopes as input readback. CPU doubles establish the hook and checks, not the native meaning of the getters; actual-device execution is still required.

5. **Exact guards and terminal behavior.** The expected zero-based storage setup guard is 11 for Warp and 7 for CUDA. Successful storage brings normal completion counts to 12 and 8 respectively; the existing one-sync terminal disposal contributes the additional ARM without a normal COMPLETE. Allocation/adoption failures must retain the original error in both diagnostics, exit 74, record exactly one failed storage guard, retain full quarantine, and have no subsequent low-level leaf, additional ARM/COMPLETE or emergency sync. The runner combines these checks with the existing custody artifact replay and runtime-custody selection/attachment validation.

6. **Hang evidence stays partial.** The second-allocation hang writes its boundary observations before the unchanged `_stall` loop. Acceptance requires the original supervisor deadline expiry, kill request, observed exit -9, unchanged completed-guard count, and no terminal `worker.json`, `failure.json` or successful storage binding. The supervisor may retain the earlier `prefix.json`; that is partial worker evidence, not synthesized terminal success. The original parent-liveness handling and owned child reaping remain unchanged. No process-deadline execution was performed in this review.

7. **Compatibility and scope.** Existing selector conditions are unchanged except for explicit inclusion of the new storage family. The extra CUDA observer condition preserves the pre-existing CUDA-only artifact restriction. Production allocation/layout, source catalog, lifecycle and product gates are untouched. The implementation note accurately distinguishes support/CPU evidence from outstanding native acceptance. Hardcoded zero sample/final-readiness diagnostic fields are descriptive support metadata, not independent proof of product readiness; the inspected production path still does not create those capabilities.

## Fresh verification

Environment: Windows PowerShell and `C:/Python314/python.exe`; CPU arrays, guard peers and native-state doubles. Executed from the candidate repository:

```text
C:/Python314/python.exe -B -m pytest -q -rs -p no:cacheprovider tests/unit/optics/test_prepared_fixed_render_inputs.py tests/unit/optics/test_native_output_allocation.py tests/unit/optics/test_native_scope_observer.py tests/unit/optics/test_warp_failure_diagnostics.py tests/unit/optics/test_cuda_artifact_runtime.py tests/unit/optics/test_runtime_native_scope_codec.py
150 passed, 41 skipped in 16.90s
```

Tool exec session `19467` was polled to exit code 0. All reported results above are from this fresh reviewer invocation. The skips comprise nine pinned Warp source-text CPU-model checks and 32 actual POSIX deployment record-replay cases. They are unavailable checks, not successful native or replay acceptance. The focused suite checks original storage behavior, the extracted shared fixture, both new positive modes, allocation/adoption failure, diagnostic I/O failure, restoration of replaced callables, and rejection of tampered retention/stream/scope/restoration observations. The fifth selector's actual stall/kill branch was inspected, not executed on this Windows CPU run.

`git diff --check 219d7d4 HEAD` passed without output. I checked full revision/tree identity, changed-path scope, clean status and the empty production delta described above. Inspection initially used a nonexistent root REVIEW_PRINCIPLES.md and nonexistent src/ paths; subsequent reads used the actual collab/ and repository-root paths. Some oversized tool reads were truncated and the relevant sections were read again. These were inspection-tool issues, not candidate defects.

## Limitations, gates and process hygiene

No remote command, GPU allocation, native compilation, rendering, CUDA query or sacrificial native worker was executed by this reviewer. In particular, this report does not establish real target1/ambient0 storage scope, successful native restoration, native partial allocation/adoption custody, or deadline kill behavior for either lane. Those remain required in the subsequent source-frozen native round. Separate earlier compilation evidence and the parent's prepared remote checkout/batch script were not used as evidence for this support verdict, and that external batch script is outside this reviewed scope.

Whole G2, R1 mutable storage, complete resource/command/event graph, per-frame ABI binding, warmups, final readiness/origin, recording/seal/barrier, C2e/f acceptance, full C4 validation/refactor/independent review, product PREPARE/native sample activation, R2/S/host delivery, performance/results/baselines and shipping remain closed under their own requirements. Approval here permits using the reviewed support for validation; it does not waive those requirements.

I made no candidate edits. Only this report was written outside the candidate; pytest used its normal temporary directories, and bytecode/cacheprovider were disabled. The reviewed test entrypoints use in-process guard-peer threads, whose teardown joins and checks them; no peer teardown failed. The tool exposed a session identifier rather than the OS PID of the pytest process, so I do not claim a recorded PID/start-time ledger for that invocation. Its exit was observed. A post-exit scoped Win32_Process query found no Python process matching this focused pytest invocation. I launched no GPU worker/supervisor, signaled no process, and did not touch unrelated author or remote work. This is bounded owned-process hygiene, not an exhaustive OS or GPU attestation.

This complete report may be archived verbatim as evidence for the exact revision and scope above.
