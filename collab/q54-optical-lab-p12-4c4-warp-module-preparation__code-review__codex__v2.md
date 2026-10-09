# Warp module preparation: independent code re-review

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: bounded code re-review before integrated native acceptance
Author: Codex, independent reviewer session /root/warp_module_code_v2
Version: v2
Date: 2026-10-10
Status: in_review
Related Files: warp-module-preparation design-request v2, code-review v1 and test-evidence v1; runtime-integration design-request v3; optical_worker/warp_setup.py; optical_worker/native_bootstrap.py; changed unit and native support tests
Owner Summary: The prior ABI/entry snapshot finding is repaired at the CPU code boundary, including the legitimate hash-initialization transition. A separate reproducible IDLE-entry defect remains: unknown scope with otherwise valid input reaches a lifecycle API that fabricates a guard failure before the intended IDLE failure routing. Quarantine and zero synchronization still hold. Integrated native acceptance and whole C4 remain closed.

## Identity, scope and verdict

**Verdict: Request changes.** Prior F1 is closed for this bounded code re-review. F2 below is a substantiated P2 finding in the affected preparation integration. It is not a claim that the pending native acceptance matrix has been completed or that its absence is another newly discovered code defect.

This fresh reviewer received the packet without the implementation conversation, did not author or change the candidate, and did not spawn subagents. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md in full, the complete supplement v2, required parent sections 4, 5, 5.1, 5.2, 6 and 8, and the archived full v1 review. I inspected the repair delta and affected call, resource and state flows. The coordinator supplied one additional counterexample during this review; I checked it independently and strengthened it with state assertions and a control, rather than adopting the author's conclusion.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Original bounded integration base: `0356108a958f34c7d17064a92d2cc7a339b07d0b`.
- Previous reviewed candidate / re-review delta base: `4400cfd3adc1eee09c5d1f71f2dc711f0d68a189`.
- Frozen candidate HEAD: `0194cb2442c52300b06fd33f7042ed291e723af3`.
- Initial and final local `git status --porcelain=v1` were empty; HEAD was unchanged. The repair delta has 8 files, 381 insertions and 16 deletions. Both original-base and repair-base `git diff --check` passed.
- Read-only SSH observed the same clean HEAD in `/data2/shared/ang.gao/robot_sim_warp_modules_wip`; remote `git branch --show-current` was empty (detached HEAD).
- Supplement SHA256 matched `194b3c25130df77954f5b2de2b556da81009b98f94efb861848ae06566a41a5b`. Archived v1 report SHA256 matched `3a3f3c9cfbdf80b00549513de6a22c28a7199a2acc633cc4ddef6b7546e02af2`.
- The v15 source-catalog row matches the revised `warp_setup.py` SHA256 `c318e655b955bafec4ecaf0727369d1818bdaee05ba370d0eae194e037b1d43f`.

## F1 closure: immutable ABI values and initialized kernel entry

`warp_setup.py:214–239` now keeps exact annotation/type owners while additionally capturing Warp type codes, array dimension and recursively captured dtype, nested struct fields, scalar storage type, and kernel key/hash/mangled entry. The pinned `get_type_code` encodes the array class/dimension/dtype and scalar/vector/matrix type information; struct field values are separately traversed because its struct result is the native name. This removes the old shared-reference-only comparison for array and struct annotations.

`compile_module` now computes `get_module_hash(256)` before collecting entry facts. Pinned Warp initializes kernel hashes during module hashing, so this accepts the legitimate initialization instead of comparing a None hash to an initialized entry. The same facts are checked after loading and at later module-set admission. The check no longer relies on a cached module hash to notice argument or entry changes.

I read the unchanged original repro, author adaptation and its generator, and the exact pinned metadata definitions. Local pinned `review_types.py` and `context.py` SHA256 matched read-only remote installed files: respectively `9224fb234ff61f4a1c3338b7171d545171ab051fb625679568ea80a6af8321bd` and `eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee`.

My execution of the inspected adapted original repro produced **4 passed in 2.49s**: dtype, dimension and kernel-entry-hash mutations reject through actual production later admission; descriptor replacement remains rejected. The fixture first performs successful unchanged admission. It uses AST-extracted pinned `_ArrayAnnotationBase`, `get_type_code`, cached hash and mangled-name methods, real production definition inventory/issuance/validation and lifecycle, with CPU dependency/artifact doubles. This is an independently executed CPU comparison proof, not native Warp execution.

I also wrote `c4_assets/test_review_warp_compile_transition_v2.py` outside the candidate. Its **4 passed in 1.60s** exercise the actual compile helper with a kernel initially lacking a hash. The hash initializer runs before the snapshot, unchanged compilation succeeds, and dtype/ndim/hash mutation during the load rejects after publication while preserving executable/mapping custody, issuing no successful record, completing no failed segment and attempting no emergency sync. These tests use pinned metadata routines but simulated native loading/scopes. The new repository tests additionally cover struct-field and nested-array mutation and unchanged repeated admission. No remaining substantive F1 defect was found in this bounded repair.

## F2 — P2, blocking: unknown-scope IDLE entry invents a failed guard

Location: `optical_worker/native_bootstrap.py:3269–3272` in `prepare_warp_modules`, its call through `_validate_runtime_binding_resources` at `:3083`, and `optical_worker/guarded_lifecycle.py:154–160,294–298`. The existing regression at `tests/unit/optics/test_warp_module_preparation.py:157–177` misses the trigger.

Trigger: a live Warp session is `input_validated`, its input report and exact runtime binding are valid, the previous normal guard completed and the client is genuinely IDLE with no active reference or prior failed segment, but sticky device scope has become unknown. Call `prepare_warp_modules`.

The entry validates runtime resources before checking the controller's compromised state. `_validate_runtime_binding_resources` invokes `controller.require_adopted`, which invokes `_check_running`. That method calls `_fail_guard` for compromised scope even while IDLE. The recorded event is:

```text
operation=guard_failed, guard=null, details={failure_kind: operation}
```

The client becomes FAILED. `_fail_warp_preparation` then sees a failed replay/non-IDLE client and follows `self.fail`, bypassing its intended compromised-IDLE route. This falsely records a guard failure when no import/compile segment was entered. The supplement explicitly requires initial CPU checks and failure disposition from the actual IDLE state, preserving completed guard history without fabricating a failed normal segment.

Observed consequence is incorrect guard state/failure evidence and incorrect routing classification. The graph still transfers to quarantine, no compilation begins, the previous completion remains recorded, and no synchronization occurs. I do not claim unsafe resource release or an observed native device effect.

Why existing tests miss it: the unknown-scope test first sets `validation_report = None`. That causes an earlier pure CPU rejection, so `_fail_warp_preparation` sees the original IDLE state and correctly applies its compromised-scope precedence. Leaving the report valid exposes the intervening state mutation. Production `check_live` checks failed/terminal custody but not compromised scope; inspection confirms the fixture's simplified live check is not hiding a production rejection that would eliminate this trigger.

The coordinator's scratch `test_warp_unknown_entry_author.py` independently failed here: **1 failed in 0.89s**, exit 1. My stronger outside-candidate repro is `C:/Users/AG/新建文件夹/c4_assets/test_review_warp_unknown_idle_v2.py`. Result: **1 failed, 1 passed in 1.11s**, exit 1. Before calling preparation both cases assert IDLE, no guard reference and no replay failure. With a valid report, one null-reference `guard_failed` appears and the client becomes FAILED. With an invalid report, no guard failure appears and the client stays IDLE. Both cases retain the original completion, compile no records, take no new normal segment, attempt zero synchronization and quarantine the graph. Only the assertion forbidding invented guard failure fails.

Repair direction: validate preparation's actual entry guard/scope state with a pure CPU check before invoking adopted-resource APIs that mutate failed state on compromised custody. Route this rejection through the existing IDLE handler; do not weaken the lifecycle's protection for actual work or broaden controller policy to solve this local ordering defect. Preserve single-use entry, original exception retention and unknown/incomplete scope precedence. After repair, the valid-report repro and invalid-report control must both reject with no fabricated failed guard, no next leaf/new guard/sync, retained roots and unchanged successful history. Keep active/pending/failed-guard paths and healthy exact-target IDLE emergency cases covered, then independently re-review the new exact revision. Native counterparts remain an acceptance obligation.

## Other inspected high-risk areas

- The repair does not introduce compiler/loading entrypoints or a new generic framework. It uses the already admitted pinned `warp._src.types` module. Kernel owners remain strongly held alongside immutable facts.
- Exact module/definition registries, module options, mapping variant, executable/handle/device identity, artifact bytes and metadata checks still surround the revised facts. The compile mapping is held before the load, the published executable before adoption/inspection, and successful records remain tied to successful DISARMED guard identities. Aggregate issuance remains outside the three completed compile guards.
- Existing captured source checks, runtime binding validation and external audit still compose around compilation. Initial source/cache/configuration policy, in-process serial Warp load path, and the literal header inventory are unchanged by this repair; this is not a fresh exhaustive compiler-closure audit.
- The support driver mutates real compiled kernel dtype/ndim/hash before the actual `validate_module_set` call and demands its ValueError plus the expected guarded terminal counts. An assertion after the validator prevents accidental acceptance from looking successful.
- The new outer observer samples before/after the complete `compile_module` call, while the inner observer samples around `load_module`. Thus successful outer observations can establish ambient restoration after ScopedDevice/ScopedStream exits, separately from input-phase receipts. It currently samples successful returns only: failure restoration, native load/publication/adoption errors and hang remain separate pending coverage, not proven by these additions.
- Documentation accurately distinguishes historical success at 4400cfd, repaired CPU tests and pending revised native acceptance. No readiness, warmup, graph, rendering, arena, timeline, product PREPARE or result/backend-selection authority is added.

## Executed verification and limitations

Local environment: PowerShell, `C:/Python314/python.exe`, CPython 3.14. All pytest commands used `-B` and `-p no:cacheprovider`. Scratch repros and pytest basetemps are outside the candidate.

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_warp_module_preparation.py tests/unit/optics/test_guarded_runtime_binding.py tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_custody.py --basetemp=C:/Users/AG/新建文件夹/c4_assets/review_warp_modules_v2_pytest
```

Observed **158 passed, 9 skipped in 269.80s (0:04:29)**, exit 0. Unified exec session **48511** reached terminal completion. The Windows/platform skips do not satisfy mandatory POSIX/native coverage. This independently reran the four requested suites at the revised HEAD; it is not the prior reviewer's 152-test result. Temporary copied releases and intentional source/bytecode mutation by bootstrap tests stayed in their outside-candidate basetemp.

```text
C:/Python314/python.exe -B -m pytest -q -s -p no:cacheprovider C:/Users/AG/新建文件夹/c4_assets/test_author_warp_abi_repair.py
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider C:/Users/AG/新建文件夹/c4_assets/test_review_warp_compile_transition_v2.py --basetemp=C:/Users/AG/新建文件夹/c4_assets/review_warp_transition_v2_pytest
C:/Python314/python.exe -B -m pytest -q -s -p no:cacheprovider C:/Users/AG/新建文件夹/c4_assets/test_warp_unknown_entry_author.py
C:/Python314/python.exe -B -m pytest -q -s -p no:cacheprovider C:/Users/AG/新建文件夹/c4_assets/test_review_warp_unknown_idle_v2.py
```

Results are reported above, each with terminal exit status. The last two failures are the reproduced candidate defect, not environment failures. The original unadapted v1 script is unchanged; its obsolete one-argument `_definition_facts` call was not presented as a meaningful current-candidate test.

Other read-only commands: Git status/branch/revision/log/stat/numstat/diff/diff-check; UTF-8 Get-Content and rg; Get-FileHash; structured source-catalog JSON inspection; SSH Git and installed-file sha256sum. The first root-level REVIEW_PRINCIPLES.md lookup failed because the file is under collab; the correct file was then fully read. Some batched displays were truncated and were replaced by focused reads. One rg included a nonexistent separate native_scope_observer.py; the actual observer lives in native_runtime_process.py and was inspected there. These inspection issues did not modify the candidate.

The previous reviewer and author native success records at 4400cfd are historical context only; I did not reuse them as new execution proof. During review the coordinator reported terminal revised v4 success and v5 dtype rejection, and a live v6 ndim campaign. I did not independently inspect those new native artifacts and do not credit them as reviewer verification. I did not launch, signal or take over any author campaign. No GPU campaign, installation, cache deletion or remote mutation occurred. Local pytest fixtures own and close their peer/subprocess helpers. All reviewer test/tool sessions reached terminal results; the candidate remained clean and frozen. This report and reviewer scratch tests are outside the repository for the author to archive verbatim.

No bounded code approval is issued until F2 is repaired and the revised candidate is independently reviewed. Native positives at both resolutions, hostile source/cache/configuration/artifact/ABI/foreign-record cases, load/publication/adoption failures and hang, ARMED-before-import, IDLE entry/issuance faults, actual ambient restoration, partial-root custody, zero next leaf/extra sync/guard, observed worker/supervisor exit and unrelated-process preservation remain distinct acceptance obligations. CPU mocks cannot satisfy those real-device properties.

Whole C4 remains incomplete: graph/timeline/arena, R_setup/R_final, ten warmups, final recording/seal/barrier, CPU-oracle rendering correctness, lifecycle matrices, C4c3/c4, C4d/B3, systematic refactor and fresh whole review. Product PREPARE, benchmarks/results/baselines and backend selection remain closed.
