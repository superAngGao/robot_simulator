# Warp module guarded-fault acceptance support — independent re-review v2

Initiative: Q54 / P12.4 / C4 Warp module preparation
Stage: bounded native acceptance support code re-review
Author: Codex independent reviewer, session /root/warp_module_fault_support_v2
Version: v2
Date: 2026-10-10
Status: accepted (bounded support code only)
Related Files: tests/support/native_runtime_process.py; tests/support/cuda_artifact_runtime_acceptance.py; tests/unit/optics/test_warp_failure_diagnostics.py; warp-module-preparation design-request v2 and implementation-note v5/v6; warp-module-fault-support code-review v1; runtime-integration design-request v3
Owner Summary: Approve the bounded support code at the exact frozen revision. F1 is closed: inspection/write failures retain the diagnostic and propagate the identical original exception. Fresh CPU verification passes, and missing/invalid roots evidence still rejects. The five new native selectors remain unexecuted; this is not complete native acceptance or whole-C4 approval.

## Verdict, identity and scope

**Verdict: Approve**, limited to the five fixed guarded-fault acceptance selectors and associated observations, assertions and exception-preservation repair at `552d1b78f90d3b358710174bed2c58b6ccdd571f`.

No substantive blocking or non-blocking code finding remains in this bounded review. Prior support F1 is closed. This verdict establishes suitability of the inspected support code for subsequent native execution; it does not establish the unexecuted real-device properties.

This was a fresh independent reviewer session with the task packet, without the implementation conversation. I did not author or edit candidate files, spawn subagents, or change the original reviewer repro. The report and new scratch scripts are outside the candidate.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Original support base: `9787594249c84e00bb5776f51ed298667deda646`.
- Repair base / prior reviewed HEAD: `8648067ba08bb826c50b1322a75bdff5c4db75d5`.
- Exact reviewed HEAD: `552d1b78f90d3b358710174bed2c58b6ccdd571f`.
- Initial and final working-tree status were empty; HEAD remained unchanged.
- Repair delta: four files, 216 insertions and 20 deletions. Cumulative support delta: six files, 421 insertions and 11 deletions.
- `git diff --name-only beedab6143e702a1a50ef0e2686fd0138e194376 HEAD -- optical_worker optics tools` returned no paths. The prior production approval remains scoped to its exact code revision; this review does not reopen or broaden it.
- Independently verified supplement SHA256 `194b3c25130df77954f5b2de2b556da81009b98f94efb861848ae06566a41a5b` and prior support review SHA256 `1be6b15ad4a51a8d538ec830c2451a0606c57c38085d540ed7cdcdaeeac8c8d5`.
- Original repro remained SHA256 `15ee1be3a8e8b1f8263737a042734ead5e942c9a637b8da165d46e71e8c86d5d` before and after verification.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully; complete supplement v2, prior support review v1, implementation-notes v5/v6 and bounded production code-review v4; parent runtime-integration v3 sections 4, 5 (including 5.1/5.2), 8 (including 8.1/8.2), and 9. The initial root-level principles lookup failed because that file is under collab; the correct file was then read fully. Broad combined output was followed by focused reads for truncated material. Only root AGENTS.md was found.

## Prior F1 closure and independent counterexamples

At `tests/support/native_runtime_process.py:1135`, the entire post-quarantine inspection/record construction/write now executes inside a nested `try`. Its `except BaseException` retains the diagnostic in the existing session list. The outer bare `raise` at line 1160 propagates the original preparation error, rather than the diagnostic. The repair adds no native leaf, adoption, retry, guard, synchronization or fabricated roots record.

The new three-case test executes the exact AST try block selected by its sole `prepare_warp_modules` call. Successful diagnostics, module-inspection failure and evidence-write failure all preserve object identity of the original RuntimeError; the failure cases retain the injected OSError and record no successful write. This is a direct test of the actual support block, not a parallel implementation.

I independently read the unchanged original v1 repro and executed it through a new external wrapper. Only its old expected diagnostic-as-primary assertions were adapted in memory to require the original exception and retained diagnostic; the original file was neither edited nor overwritten. Its exact candidate AST extraction remains at line 1129. The former counterexample now reports:

```json
{"case":"post_quarantine_roots_write_failure","propagated_type":"RuntimeError","original_preserved_as_primary":true,"original_only_in_context":false}
{"case":"native_scope_write_failure_control","original_preserved_as_primary":true,"diagnostic_retained":true}
```

The separate existing native-observer control still preserves the original under diagnostic write failure. Both support modules compile in memory, and all 74 runtime fault names remain unique.

Missing evidence is not treated as success. I additionally executed the actual AST body of the runner's `if compile_fault` branch with a valid CPU fixture, then with a missing roots file and an invalid roots object. The valid fixture passed; missing evidence raised FileNotFoundError; invalid evidence raised KeyError. These checks exercise the actual unconditionally required roots read and validation at `tests/support/cuda_artifact_runtime_acceptance.py:201`. They establish fail-closed CPU evidence handling, not a native run or a full replay-validator test.

## Inspected high-risk flow

- Captured loader marks the module `started` and invokes real host adoption before `super().exec_module`. The observer calls original adoption before checking the live ticket and injecting import failure. Thus the first optical body is prevented from executing while its real owner is retained, under setup guard 6. The observation is not a caller-supplied ARMED flag.
- Production holds the actual Module, definitions and exact `execs` mapping before the native compile/load leaf. Before-load injection occurs inside real target/stream contexts. After-load and hang injection occur after original public load published the actual executable into that mapping but before the observer returns to production.
- The adoption injection acts on the actual returned executable after production has appended it to session ownership and before original guardian adoption. It cannot manufacture the retention property it then checks.
- The load/compile observers do not append executables or compiled records to an observer retention list. Recorded calls contain scalar native observations and identities. Ordinary local/traceback references still exist and are not claimed absent. Actual guardian membership and the real mapping/executable identity are checked after quarantine.
- The outer compile observer reads native state after production ScopedDevice/ScopedStream unwind and while the original setup guard remains active. Its diagnostic exception handling still preserves the original failure. Native state includes context/device/primary-context/stream observations, rather than just logical labels.
- The post-quarantine block is CPU-only for these fixed paths. Target context is already established; the pinned Warp Device.context getter directly returns its existing `_context`. The remaining operations inspect Python objects, session registries and guardian membership. There is no custody repair or native query after quarantine.
- Runner checks exact expected guard and original failure, six or seven prior completions and one additional ARM, exactly one matching failed guard, no subsequent marked native leaf, zero emergency synchronization, full quarantine, no completed module record/module set/binding, and worker exit 74. Published-owner cases compare marker and actual retained executable identity; adoption additionally requires session membership.
- Hang retains the original configured 120000 ms guard deadline and existing supervisor route. Runner requires deadline expiration/kill, worker -9, preheld mapping and absence of fabricated worker/failure terminal records. It makes no restoration claim after a killed hang. No new process manager, extended deadline, retry or emergency path is introduced.
- Existing positive and IDLE branches remain separate. Production source/runtime policy and product gates are unchanged. The cumulative diff was inspected as well as the repair; observations above are code-flow facts and CPU checks, not hardware evidence.

## Commands and observed verification

Environment: Windows PowerShell; `C:/Python314/python.exe` with `-B`; pytest cache provider disabled. Scratch scripts, generated fixture evidence and basetemp are outside the candidate under `C:/Users/AG/新建文件夹/c4_assets`.

1. `git status --porcelain=v1`, `git rev-parse HEAD`, `git branch --show-current`, stat/path diff inspection, and final status/HEAD recheck: clean and exact requested revision.
2. `git diff --check 8648067ba08bb826c50b1322a75bdff5c4db75d5 HEAD` and `git diff --check 9787594249c84e00bb5776f51ed298667deda646 HEAD`: exit 0, no output.
3. `Get-FileHash -Algorithm SHA256` on design v2, prior review v1 and unchanged original repro: matches digests above.
4. `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/warp_fault_support_review_v2_verify.py`: final exit 0; adapted original counterexample/control pass, in-memory compile/unique-selector checks pass, then launches the exact following test command:

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider --basetemp C:/Users/AG/新建文件夹/c4_assets/warp_fault_support_review_v2_pytest tests/unit/optics/test_warp_failure_diagnostics.py tests/unit/optics/test_warp_module_preparation.py tests/unit/optics/test_cuda_artifact_runtime.py
```

Observed **60 passed in 10.13s**, exit 0, tool session 78039. These are fresh results at the frozen candidate, independently corroborating the author's result with this run's own timing.

5. `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/warp_fault_support_review_v2_runner.py`: exit 0; actual runner branch accepts valid fixture and rejects missing/invalid roots evidence.

The first scratch-wrapper invocation passed the adapted original repro/control, then its own hash check failed with AttributeError because executing the original script in the wrapper's namespace replaced a local path variable with the repro's RuntimeError variable. I isolated the exec namespace in the scratch wrapper and reran successfully. This was a reviewer-harness error, not a candidate defect; no tests had been launched in that failed invocation. No candidate test failure remains.

## Process hygiene and limitations

Reviewer-launched CPU process ledger: first scratch Python 40928, PowerShell parent 23124, exited 1 for the harness-only issue above; successful wrapper Python 41404, PowerShell parent 38772; pytest child 20744, parent 41404, exited 0; runner-check Python 10960, PowerShell parent 38304, exited 0. Subsequent exact-PID CIM queries returned none of these seven process IDs. All tool sessions completed. Shared reviewer host codex.exe PID 10808, parent 34516, creation time 2026-10-07 19:15:47 remains active as expected. This is scoped local child-exit evidence, not a claim about all machine processes.

No SSH, native/GPU test, remote mutation, takeover/restart, signaling, cache deletion or GPU reset was performed by this reviewer. No native worker or supervisor was launched. I did not inspect another owner's GPU-process state or infer absence of remote residuals from local queries.

During review the coordinator reported its existing beedab6 batch terminal: tool 10124 exit 0; v11-v16 passed; parent 3048547 and twelve worker/supervisor PIDs absent; only unrelated MPS PID 2238410 in its GPU census. It also reported local archive `c4-warp-module-author-evidence-v3.tar.gz`, SHA256 `0cdea96fa3467aea46fc0ff9a5d3b1d7dc079e8bfb72dc3d57fa827a0c75329c`. These are coordinator-reported context, not independently validated by this support review and not evidence that any new selector executed. The remote checkout was reported unchanged at beedab6.

The five added selectors remain unexecuted natively. Static inspection and CPU mocks cannot prove actual compiler behavior, target1/ambient0 restoration, native failure custody, original-deadline hang termination or unrelated-process preservation. Subsequent native execution must record these facts and scoped process hygiene. Broader source/cache/options/artifact/target/stream/foreign-record negatives and fixed-CUDA/input regressions remain separate acceptance obligations where incomplete.

## Closed gates and remaining work

This support code approval does not approve complete native compilation acceptance, product PREPARE, rendering, benchmark/performance/results/baselines or backend selection. Whole C4 remains incomplete: one-slot graph/timeline/arena integration; ten real warmups; R_setup/R_final and final readiness/recording/seal/barrier; CPU-oracle rendering correctness and lifecycle matrices; C4c3/c4; C4d/B3; systematic refactor and a fresh whole review. No child-stage approval opens those parent gates.
