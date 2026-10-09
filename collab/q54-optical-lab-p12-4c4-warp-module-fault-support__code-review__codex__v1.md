# Warp module guarded-fault acceptance support — independent review v1

Initiative: Q54 / P12.4 / C4 Warp module preparation
Stage: bounded native acceptance support code review
Author: Codex independent reviewer
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: tests/support/native_runtime_process.py; tests/support/cuda_artifact_runtime_acceptance.py; collab/q54-optical-lab-p12-4c4-warp-module-preparation__implementation-note__codex__v5.md
Owner Summary: Request changes for one localized diagnostic exception-preservation defect. The five fixed fault boundaries otherwise follow the approved production flow. CPU checks pass; the new selectors have not been executed natively and this is not native acceptance.

Reviewer session: `/root/warp_module_fault_support_v1`, fresh independent task session. I did not author the candidate and did not spawn additional agents.

Verdict: **Request changes**.

## Exact revision and scope

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Exact delta base: `9787594249c84e00bb5776f51ed298667deda646`.
- Exact inspected HEAD: `8648067ba08bb826c50b1322a75bdff5c4db75d5`.
- Working tree was clean before and after inspection/testing; HEAD remained unchanged.
- Diff has precisely the three packet paths: 225 insertions and 11 deletions.
- Production remains at approved `beedab6143e702a1a50ef0e2686fd0138e194376`; `git diff --name-only beedab6143e702a1a50ef0e2686fd0138e194376 HEAD -- optical_worker optics tools` returned no paths. This review does not reapprove the earlier production implementation.
- Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` in full. A first attempted root `REVIEW_PRINCIPLES.md` read failed because the file is under `collab`; the correct file was subsequently read fully. Only root `AGENTS.md` exists in the candidate.
- Read the Warp module preparation design-request v2 and parent runtime-integration design-request v3 sections 4, 5, 8, and 9. Independently verified supplement SHA256 `194b3c25130df77954f5b2de2b556da81009b98f94efb861848ae06566a41a5b`.

The bounded question is suitability of the added support for later native execution: import-before-execution, load-before-return, executable publication before public return, returned executable adoption failure, and original-deadline hang; native restoration observations and actual partial-root checks. No new production policy or process framework is included.

## Finding F1 — P2, preserve the primary exception if post-quarantine diagnostics fail

Location: `tests/support/native_runtime_process.py:1129`, especially the exception handler at lines 1131–1156 and `_write(..., "warp_failure_roots.json", ...)` at line 1141.

Trigger: `session.prepare_warp_modules()` raises the intended production/fixed fault and completes quarantine, then a field inspection or the new roots evidence write raises. An ordinary diagnostic I/O error is sufficient; `_write` performs file creation, flushing, fsync and publication.

Observed consequence: the new handler does not guard its diagnostic work. Its exception escapes instead of reaching its final bare `raise`, so the outer driver catches the diagnostic exception and writes that as the primary `failure.json` type/message. The original compilation exception remains only as `__context__`. The runner then diagnoses loss of the expected original exception rather than preserving that original error as primary and separately reporting absent diagnostic evidence.

This does not establish a production custody escape or a false acceptance: quarantine has already occurred, the runner rejects, and the exception chain may still contain the original. It does violate the explicitly required original-exception preservation in the acceptance support. The adjacent `observed_compile` diagnostic handler already implements the needed separation correctly; the new roots diagnostic does not.

Independent proof: the scratch reproduction executes the candidate's exact AST `try` block rather than a rewritten equivalent. A minimal CPU session raises the original `RuntimeError`; all root inspections succeed; the injected diagnostic writer raises one `OSError`. The propagated exception is exactly that `OSError`, with the original only in its context. A control running the exact `observed_compile` wrapper under the same diagnostic write failure preserves the original exception and retains the diagnostic.

Why current checks miss it: the 57 existing CPU regressions exercise production preparation/compile custody, not this new native-driver roots observer or evidence-writing failure. Happy diagnostic writes in later native fault runs would also miss this branch.

Repair direction: guard the entire post-quarantine observation/serialization/write section with a nested diagnostic exception handler; retain diagnostic failure without adding or repairing device custody, then always re-raise the original exception. Missing/invalid roots evidence must continue to fail acceptance; do not manufacture a successful roots record or hide the diagnostic failure. This is a test-support-only fix.

Proof needed after repair: run an isolated CPU check against the actual observer block with (1) successful roots observation, (2) an inspection failure, and (3) a roots-write failure; each must propagate the identical original exception. In the diagnostic failure cases the observation must remain missing/failed and acceptance must still reject. Repeat the relevant existing CPU regressions and fresh independent review of the repair before new native execution.

## Inspected behavior and negative-test soundness

- Captured source loader records `started`, adopts the actual module, and only then executes the source. The installed host-adoption observer calls the original adoption first and verifies a live setup ticket. The import fault therefore occurs after genuine adoption and before the first optical module body. Expected guard 6 and no compile/load follow directly from this call flow.
- Production `compile_module` holds the module, definitions and exact `execs` mapping before entering its native leaf and public Warp load. The before-load fault is after entering the real target/stream contexts; the after-load and hang faults occur after actual executable publication but before the wrapper returns to production. The adoption fault occurs before original guardian adoption, after production has already placed the returned executable in its session owner list.
- The observer no longer adds an executable or compiled record to `session.runtime_owners`. Call records contain scalar observations/identities rather than executable owners. Ordinary stack/traceback references still exist and are not claimed absent. Post-failure checks test actual guardian identity membership and the executable found in the real module mapping; they do not infer custody solely from a success boolean.
- `observed_compile` samples native state after the real contexts unwind and before the original guarded segment unwinds. For these fixed injections the segment is still ARMED. Its diagnostic catch preserves the original exception, unlike F1. No native observation is performed by the post-quarantine roots block on the fixed paths: target context is already initialized and the remaining reads are Python object/registry inspection. I also inspected the pinned Warp `Device.context` getter, which returns its existing `_context` directly here.
- Runner requires exact fault/guard, 6 or 7 completed earlier segments, one additional ARM for the failed segment, one matching `guard_failed`, zero subsequent marked native leaves, zero emergency synchronization, worker exit 74, original error text/type, full quarantine, zero completed module records and no issued module set/binding. Published-executable failures compare the actual retained executable identity with the injection marker; adoption failure additionally requires session ownership.
- Hang uses the same configured 120000 ms operation deadline and existing supervisor. It requires timeout/kill, worker -9, preheld mapping and no fabricated worker/failure terminal JSON. It does not assert restoration after killing a hung worker. No new timeout or retry route was added.
- Existing positive and IDLE branch handling remains distinct. Positive compilation no longer receives observer-added compiled-record/executable ownership; production already retains those. Import observations do not enter another guard. Existing selectors continue through their prior runner branch. These are static/code-flow observations plus CPU regression evidence, not device execution claims.
- No additional substantive finding was established in the bounded delta.

## Executed checks and reproducible evidence

Environment: Windows PowerShell, `C:/Python314/python.exe`, bytecode writing disabled. Scratch and pytest temporary data were outside the candidate under `C:/Users/AG/新建文件夹/c4_assets`.

1. `git status --short`, `git rev-parse HEAD`, `git diff --stat 9787594249c84e00bb5776f51ed298667deda646 HEAD`, exact path diff inspection, and final status/HEAD recheck: clean, exact requested revision and scope.
2. `Get-FileHash -Algorithm SHA256 collab/q54-optical-lab-p12-4c4-warp-module-preparation__design-request__codex__v2.md`: matches the supplied digest.
3. `git diff --check 9787594249c84e00bb5776f51ed298667deda646 HEAD`: exit 0, no output.
4. `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider --basetemp C:/Users/AG/新建文件夹/c4_assets/warp_fault_support_review_v1_pytest tests/unit/optics/test_warp_module_preparation.py tests/unit/optics/test_cuda_artifact_runtime.py`: **57 passed in 10.16s**, exit 0; tool session 1697 observed complete.
5. `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/warp_fault_support_review_v1_repro.py`: final run exit 0. It executes extracted candidate AST for F1 and the exception-preserving control, compiles both support modules in memory, and checks 74 runtime fault names are unique.

Final scratch output:

```json
{"case": "post_quarantine_roots_write_failure", "propagated_type": "OSError", "original_preserved_as_primary": false, "original_only_in_context": true}
{"case": "native_scope_write_failure_control", "original_preserved_as_primary": true, "diagnostic_retained": true}
{"compile": "pass", "fault_count": 74, "fault_names_unique": true, "pid": 27708, "parent_pid": 39764}
```

The first scratch run reproduced F1 and the passing control, then its separate constant-uniqueness harness failed with `NameError: RUNTIME_PHASES` because it extracted only `*FAULTS` assignments. I repaired the scratch extraction to include the dependency assignments and reran successfully. This was a review-harness error, not a candidate failure. The candidate was not modified.

## Process hygiene, limits, and closed gates

No SSH connection, GPU command, worker/supervisor launch, remote mutation, takeover, restart, broad signaling, process cleanup or cache deletion was performed by this reviewer. No native test was run. The existing remote campaign remains owned by the coordinating agent; I did not reuse its results as proof for this delta.

The CPU pytest tool session 1697 exited 0. A scoped local process query afterward found no matching pytest process (only the query's own PowerShell). The pytest PID was not captured before its short run ended; its exit is evidenced by the tool session, not an invented PID ledger. The final scratch Python process 27708 had PowerShell parent 39764, returned exit 0, and a subsequent exact-PID query found neither process. Shared reviewer tool host `codex.exe` PID 10808, parent 34516, creation time 2026-10-07 19:15:47 remained active as expected. No owned child residual was observed. No remote/GPU process hygiene claim is made for another owner's live batch.

The implementation note's CPU result claim is independently corroborated by this fresh run, with its own timing. Its native selectors remain unexecuted; static inspection and CPU mocks cannot prove native CUDA scope restoration, compiler behavior, hardware custody or supervisor native hang behavior. The prior remote batch status supplied by the parent is outside this review's executed evidence.

F1 prevents approval of this exact support revision for the intended preserved-exception evidence contract. After repair/review, native execution still needs its own real dual-GPU observations and terminal/process-hygiene evidence. Source/cache/options/artifact/target/stream/foreign-record negatives and fixed-CUDA/input regressions remain separate acceptance obligations where not yet completed.

No approval is given for product PREPARE, performance/results/baselines/selection, native compilation acceptance, or whole C4. Whole C4 remains incomplete: graph/timeline/arena; ten real warmups and final readiness/recording/seal/barrier; rendering correctness and lifecycle; C4c3/c4; C4d/B3; systematic refactor and fresh whole review.
