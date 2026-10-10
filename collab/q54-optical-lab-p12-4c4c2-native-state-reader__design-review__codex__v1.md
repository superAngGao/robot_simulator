```text
Initiative: Q54 / P12.4 / C4c2
Stage: Fresh independent DESIGN review — native-state reader
Author: Codex, independent reviewer /root
Version: v1
Date: 2026-10-10, Asia/Shanghai
Status: accepted
Session: 01a124e2-d69d-72d3-a86c-b59248c84055
Related Files: native-state-reader design-request v1 and referenced predecessor designs
Owner Summary: Approve with comments for this bounded implementation.
Native execution, complete C2f/C4 acceptance and product gates remain closed.
```

**Verdict: Approve with comments.** No blocking design finding or unresolved policy decision was identified. Approval binds the exact design and revision below and permits implementation only. Fresh exact-code/native-call preflight remains required before executing the new native reader.

**Identity and preservation**

| Item | Verified result |
|---|---|
| Candidate | `C:\Users\AG\新建文件夹\c4_native_reader_design_review` |
| HEAD | `1472b5fd2eba898bc53ef1c0ef2b5dbcbb80b056` |
| Base; verified merge-base | `7c8c3ec3c6fa5b57eeae6a6676b2eb6d03250823` |
| Branch | Detached HEAD |
| Initial/final working tree | Clean, including untracked files |
| Base-to-HEAD diff | One design file, 257 inserted lines; no implementation changes |
| Design committed LF SHA-256 | `0b6dab55c496e730219dfbb253b7d9530dff40faa6c90676c71a2b90e0f41ca5` |
| Previous gap-report committed LF SHA-256 | `b78661032b29b7e84e03a59c5f6a5c5642bd155fc63c7f5265669730f1853c34` |
| Whitespace verification | `git diff --check` passed |

I read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` in UTF-8 first. This review received the task packet, not an implementation conversation. No additional agents were started. No tracked files were changed.

**Design assessment**

The proposed sequencing is coherent. It repairs the production observation mechanism required by F1 while explicitly leaving final-sample physical-scope evidence and complete C2f integration unfinished. That is a valid prerequisite stage, not full acceptance.

The following high-risk decisions are sufficiently specified:

- **Artifact ordering:** The existing loader retains each created module and journals its return inside the indexed contexts. The proposed observations surround that real path and preserve frame-start, entered-leaf, returned-owner, scope-exit and frame-completion ordering. The six mandatory pairs and original v3 joins remain intact.
- **Reader custody:** One source-owned reader and its containers enter the existing retained graph before library acquisition or queries. Immediate library/function and returned-object retention covers failures in metadata inspection, observation, append and adoption. No second guardian or release authority is introduced.
- **Native query restrictions:** Exact environment rows, resolved mapped paths, file bytes, `RTLD_NOLOAD | RTLD_LOCAL`, fixed symbols and ABI declarations provide a feasible narrow boundary. Missing prerequisites fail without a fallback load or alternate query.
- **Cold CUDA and Warp cache proof:** The design preserves conditional cold logical-device derivation and avoids warming CUDA getters before stream creation. Warp’s getter remains conditional on the original stream path, exact empty error-buffer checks, raw object identities and the first native-stream comparison.
- **Failure behavior:** Complete contradictory snapshots are retained before expected-state rejection. Getter failures preserve their ordered diagnostic prefix without fabricating a complete state. Sticky failures stop further queries; unsuccessful scope restoration or emergency observation cannot authorize ordinary release.
- **Source policy and imports:** A separately captured module with an exact source-pinned exception is compatible with the current loader/catalog architecture. Updating the bootstrap pins, catalog and frozen runtime contract together preserves the release boundary. The CPU collector remains separate from native execution.
- **Consolidation:** Removing the test-owned successful-fact producer while retaining a diagnostic/fault facade avoids a second query/state implementation. Historical evidence retains its original producer meaning.
- **Acceptance boundaries:** Setup coverage must match actual eligible lifecycle occurrences. Sacrificial termination and integrated clean close retain different terminal requirements. Neither can supply final-sample physical proof from setup-only observations.

**C1 — P2, non-blocking implementation comment: preserve the phase distinction at shared helper callers.**

**Location:** Design lines 69–78, 162–178 and 197–213; `native_setup.py`’s `_upload_warp_array` and upload-context consumers; `native_readback.py:_readback_native_arrays`; `optics/device_bvh.py:668,694`; `tests/support/native_scope_acceptance.py`.

**Trigger:** During extraction, instrumenting shared helper bodies indiscriminately, or copying the existing acceptance checker’s runtime-identity scan across the entire integrated lifecycle.

**Consequence:** Later BVH preparation can be mistaken for input upload/readback. This could emit out-of-phase v3 facts, apply ambient-state expectations inside an already-entered target scope, or incorrectly demand input-stage pairs for later operations. An optional-reader fallback could instead silently omit required input-stage observations.

**Why prior checks miss it:** The existing observer explicitly switches later helper calls to raw diagnostics. The old success checker also assumes sacrificial termination after input validation. Those conditions do not establish correct behavior after extraction into the integrated path.

**Direction:** Implement the design’s existing phase boundary explicitly at the source-bound orchestration and shared coverage validator. Keep one helper implementation, preserve C0’s reader-free path, require the registered reader for the captured input path, and retain later raw diagnostics without relabeling them as input-phase facts.

**Proof at implementation review:** Exercise a captured input call, a later BVH caller and C0. Verify omission/substitution fails before the captured input native call; later calls produce no input-stage receipts; coverage joins every eligible input occurrence, including zero-size operations; and C0 acquires no reader.

This comment requires no design expansion. The design already establishes the necessary distinction; the implementation must preserve it.

**Prior findings**

- **F1:** Addressed at the design level. Direct production observations can satisfy the retained artifact contract without test mutation or weakened replay. The current code remains unfinished; this review does not claim the runtime gap is already repaired.
- **F2:** Remains open. Deferring final-sample/resolve/close physical-scope coverage is acceptable here because the design expressly retains it as required C2f work and grants no whole-integration acceptance.
- **F3:** Outside this approval. I did not approve the separately corrected reply-depth change.

**Executed verification**

- Inspected the actual design, predecessor requirements, bootstrap loader/retention paths, runtime binding, shared upload/readback callers, source policy, v3 parser, observer, acceptance checker and guarded emergency synchronization.
- Ran the existing CPU suites using Windows `C:/Python314/python.exe -B -m pytest`, plugin autoload disabled, pytest cache disabled and scratch-local temporary output:
  - `test_native_scope_observer.py`
  - `test_cuda_artifact_replay.py`
  - **Result: 45 passed in 6.75 seconds; no skips.**
- The observer tests used the hash-checked copied Warp source text for exact-Device CPU models. These tests exercise the existing mechanism and parser, not a new reader implementation.
- Independently checked all **64 copied raw files, 4,690,753 bytes**, against the supplied inventory’s sizes and SHA-256 values. This was a local inventory comparison, not a fresh remote campaign attestation.
- Direct raw inspection confirmed CUDA’s selected record contains two journal entries and zero native-scope rows; Warp’s contains 1,383 journal entries and zero native-scope rows. Neither was treated as new execution evidence.
- Used read-only SSH on `tileops-dev`. The remote frozen repository remained clean and detached at `26542543d067d20c64670d375065535420ede203`.
- Re-read pinned Warp source without importing Warp or executing native functions. Confirmed:
  - `context.py` SHA-256 matches `eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee`.
  - `wp_cuda_context_get_stream` declares `[ctypes.c_void_p] → ctypes.c_void_p`.
  - The getter enters `get_context_info`; its cache-miss branch can select context and allocate.
  - The original setter, error-buffer implementation, scope restoration and cache-erasure code support the design’s stated proof obligations.

Some initial inspection commands encountered Windows quoting, Unicode-pipe or path errors. Corrected commands completed successfully; these were inspection-tool errors, not candidate failures.

**Limits, process closure and approved scope**

No GPU workload, library acquisition/query, installation, reset, signal or implementation edit occurred. CUDA cold-query semantics and changed native call placement still require the promised exact-code preflight. Full source-policy/C0 regression acceptance, new-reader fault tests, real-device restoration and the complete integrated workload/failure matrix remain unexecuted here.

Scratch output is confined to:

`C:\Users\AG\新建文件夹\c4_assets\native_reader_design_scratch_v1`

Reviewer CLI PID was `1956`, started at `2026-10-10 16:16:48`. The final scoped local process check found no Python/SSH/SCP descendant residual. Recorded remote inspection PIDs `2138072` and `2147023` were absent; the final synchronous read-only probe exited successfully. Unrelated processes were untouched.

Approval covers only implementation of this bounded preparation reader, its direct observation placement, exact source-policy exception and test consolidation. It does **not** authorize new native execution, approve unfinished C2f/C4, or open activation, performance/result/baseline, backend-selection, R2/S/host-delivery, a3b or shipping gates. This bounded review is complete.