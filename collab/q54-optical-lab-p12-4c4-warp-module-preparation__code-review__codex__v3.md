# Warp module preparation: independent code re-review v3

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: bounded code re-review before integrated native acceptance
Author: Codex, independent reviewer session /root/warp_module_code_v3
Version: v3
Date: 2026-10-10
Status: in_review
Related Files: warp-module-preparation design-request v2, code-review v1/v2, implementation-note v3 and test-evidence v1; runtime-integration design-request v3; optical_worker/native_bootstrap.py; optical_worker/warp_setup.py; source policy/catalog; module unit tests and native support driver
Owner Summary: The original unknown-scope IDLE entry counterexample is repaired, and the prior ABI/entry repair continues to pass independently executed CPU regressions. A separate post-final-DISARMED issuance boundary still fabricates a failed normal guard on unknown scope. The graph is quarantined and no synchronization occurs, but the recorded state and failure route violate the accepted aggregate-issuance contract. Native acceptance and whole C4 remain incomplete.

## Identity, scope and verdict

**Verdict: Request changes.** Prior F1 remains closed at the bounded CPU code boundary; prior F2's original entry counterexample is closed. F3 below is a blocking P2 finding in the same bounded module-preparation implementation. It is not a claim of unsafe native resource release or a relabeling of the outstanding native acceptance matrix as a code defect.

This is a fresh review session without the implementation conversation. I did not author or modify the candidate and did not spawn subagents. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully, the complete supplement v2, parent sections 4, 5, 5.1, 5.2, 6 and 8 including 8.1/8.2, both complete prior code reviews, implementation-note v3 and test-evidence v1. Initial PowerShell decoding and oversized display issues were addressed by focused UTF-8 reads. I inspected the actual repair diff and affected call, ownership and state flows rather than accepting author summaries.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Original integration base: `0356108a958f34c7d17064a92d2cc7a339b07d0b`.
- Re-review delta base: `0194cb2442c52300b06fd33f7042ed291e723af3`.
- Frozen reviewed HEAD: `f83edafa8c30d448127d230a5fb670c89da00194`.
- Delta: 9 files, 220 insertions, 14 deletions. The production behavior change is six added lines and one removed line in `native_bootstrap.py`; controller/lifecycle policy is unchanged.
- Supplement SHA256 independently matched `194b3c25130df77954f5b2de2b556da81009b98f94efb861848ae06566a41a5b`.
- Both `git diff --check 0194cb2442c52300b06fd33f7042ed291e723af3 HEAD` and the equivalent original-base check passed.
- Independently compared the old/new source-catalog JSON. The only changed source rows are bootstrap and source policy; their recorded SHA256 values match actual bytes. Bootstrap is `8448dc8aab0b99d16830db3bb10243f0beb71f0cfa72fc649e85da52e6310877`; policy is `c6f4b5eb8bfac30a809d7815edcd9d102a84f2f2f1be453a5f820b4da62a1f6d`. `warp_setup.py` remains `c318e655b955bafec4ecaf0727369d1818bdaee05ba370d0eae194e037b1d43f`.

## Prior findings: original counterexamples checked

F2 entry closure: `prepare_warp_modules` burns its single-use flag and now checks actual IDLE, no context, no guard reference and uncompromised replay before validating uploaded resources or calling `require_adopted`. The failure therefore reaches `_fail_warp_preparation` while still IDLE. Its existing compromised-scope check records custody tightening and routes to quarantine without calling the mutating adopted-resource APIs. No lifecycle protection for real guarded work was weakened.

I read and ran the unchanged original `c4_assets/test_review_warp_unknown_idle_v2.py`. Both valid-report and invalid-report cases now raise `SourceBootstrapError("Warp module preparation requires healthy IDLE target scope")`; both retain client IDLE, the original completion, no new guard failure/arm/compile/checkpoint, quarantine and zero synchronization. This independently closes the exact prior F2 trigger rather than testing only the earlier invalid-report control. The repository active-guard regression also checks that an actual existing guard remains the failed guard and cannot use IDLE emergency disposal.

F1 continued closure: I inspected the immutable type-value/entry facts and the ordering of `get_module_hash(256)` before the snapshot. The previously adapted original ABI repro still rejects in-place dtype/ndim and kernel-hash changes at production `validate_module_set`, while its unchanged-set admission and descriptor-replacement control behave correctly. The original compile-transition repro still accepts legitimate initially absent kernel-hash initialization and rejects dtype/ndim/hash changes during loading, retaining the published executable and mapping with no successful record or emergency sync. These are CPU tests with pinned AST-extracted metadata and simulated dependency/native scopes, not GPU execution. Local pinned `review_types.py` and `context.py` hashes match the recorded values `9224fb234ff61f4a1c3338b7171d545171ab051fb625679568ea80a6af8321bd` and `eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee`; I did not refetch or independently verify the current remote installation in this round.

## F3 — P2, blocking: actual IDLE aggregate issuance still invents a failed guard

Location: `optical_worker/warp_setup.py:313–323`, especially `controller.require_adopted(binding, cache, *records)` at line 323; invoked by `native_bootstrap.py:3304` and routed through `_fail_warp_preparation` at line 3228. The unchanged underlying mutation occurs in `guarded_lifecycle.py` `_check_running` / `_fail_guard`.

Trigger: all three compile segments have successfully completed and DISARMED; the controller is truly IDLE with no context/reference and no failed replay; all three records remain held. Record sticky unknown scope at this explicitly required CPU aggregate-issuance fault boundary, then call the actual `issue_module_set`.

The issuer checks IDLE and `not replay.failed`, but does not check compromised scope before its adopted-resource API. `require_adopted` calls `_check_running`, which records a new `guard_failed` with `guard: null` and changes the client to FAILED. The outer handler consequently enters `self.fail` instead of its actual-IDLE disposition branch. This is the same class of state/evidence defect as F2, at the supplement's separately specified post-final-DISARMED aggregate boundary. The new entry precheck cannot protect a later failure boundary after four successful additional guards.

Observed facts from the independent repro:

```text
actual issuer: WorkerSetupProtocolError("compromised scope forbids further normal work")
new event: operation=guard_failed, guard=null, details={failure_kind: operation}
client=FAILED; normal_complete_count=5; emergency_sync_attempt_count=0
guardian=TRANSFERRED_TO_QUARANTINE; three records retained; no module set/checkpoint issued
```

The matched control records the same unknown scope at the same IDLE point but raises a plain CPU exception before calling the issuer. That control remains IDLE with no `guard_failed`, preserves all five completions and all three records, and quarantines without sync. The counterexample asserts all these custody and completion facts before its final assertion fails; the failure is specifically the fabricated guard record. The five completions in the CPU fixture are its modeled upload, optical import and three compiles, not a claim about the six real input phases.

Why existing checks miss it: the permanent issuance test injects an exception directly and exercises healthy exact-target emergency disposal; it never lets actual `issue_module_set` consume compromised IDLE scope. The new valid-report test covers initial preparation entry only. Production `check_live` does not reject compromised scope, and the issuer's own first condition permits the demonstrated state. The requirement explicitly distinguishes initial and aggregate CPU failures from active guard failures, requires unknown/incomplete precedence, and forbids relabeling completed guards as failed. The unknown-scope fault is injected through the actual lifecycle API, without fabricating a guard state or modifying the production comparison.

I independently identified and ran this case before receiving the coordinator's message about its separately authored `test_warp_unknown_issuance_author.py`. I subsequently read that author repro as corroborating context; its reported output is not substituted for my execution. My stronger two-case repro is saved at `C:/Users/AG/新建文件夹/c4_assets/test_review_warp_unknown_issuance_v3.py`.

Repair direction: reject compromised/non-healthy aggregate entry with a pure CPU check before any adopted-resource API can mutate normal-work state, and preserve the existing actual-IDLE failure handler and original guard history. Do not weaken lifecycle protections for active work or add a new emergency guard. Keep single-use issuance, original exception retention and the original held graph. Re-run this actual-issuer counterexample and its control plus original entry cases, healthy exact-target issuance emergency, incomplete custody, and active/pending/failed guard regressions. Native aggregate-entry fault/disposition remains a distinct acceptance obligation. Freeze and independently re-review the exact repair.

## Verification and other inspected risks

Environment: local Windows PowerShell; `C:/Python314/python.exe`, CPython 3.14.0. All pytest executions used `-B -p no:cacheprovider`; scratch files and basetemps were outside the candidate.

Executed original entry plus prior ABI/transition repros:

```text
C:/Python314/python.exe -B -m pytest -q -s -p no:cacheprovider C:/Users/AG/新建文件夹/c4_assets/test_review_warp_unknown_idle_v2.py C:/Users/AG/新建文件夹/c4_assets/test_author_warp_abi_repair.py C:/Users/AG/新建文件夹/c4_assets/test_review_warp_compile_transition_v2.py --basetemp=C:/Users/AG/新建文件夹/c4_assets/review_warp_repros_v3_pytest
```

Observed **10 passed in 3.87s**, exit 0: 2 original F2 cases, 4 F1 repair cases and 4 hash-transition cases.

Executed the new issuance counterexample and matched control:

```text
C:/Python314/python.exe -B -m pytest -q -s -p no:cacheprovider C:/Users/AG/新建文件夹/c4_assets/test_review_warp_unknown_issuance_v3.py --basetemp=C:/Users/AG/新建文件夹/c4_assets/review_warp_unknown_issuance_v3_control_pytest
```

Observed **1 failed, 1 passed in 1.38s**, exit 1. The earlier single-case version also failed in 1.35s. These are candidate-defect failures, not environment/toolchain failures.

Executed proportional repository regressions:

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_warp_module_preparation.py tests/unit/optics/test_guarded_runtime_binding.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_source_custody.py tests/unit/optics/test_guarded_lifecycle.py --basetemp=C:/Users/AG/新建文件夹/c4_assets/review_warp_modules_v3_pytest
```

Observed **363 passed in 143.65s (0:02:23)**, exit 0; unified exec session **99247** reached terminal completion. This is a fresh execution at the candidate. I did not rerun the prior broad `test_native_bootstrap.py` suite or claim its earlier results as new evidence. The selected source-policy/custody and runtime-binding suites directly cover the revised pins and affected composition; unchanged full-bootstrap and POSIX/native coverage remain separate.

Inspected high-risk composition includes initial entry ordering; original-guard import/compile failure handling; actual-IDLE unknown/incomplete custody versus exact-target emergency; aggregate issuance after successful guards; exact runtime binding and prepared-owner validation; ABI/entry initialization and later drift checks; published executable/mapping retention; single-use preparation/issuance; source-policy pins and catalog changes; and native fault insertion/expected terminal checks. The unchanged controller still rejects failed/pending clients for emergency admission; this delta does not modify its state policy.

The new native support case preserves a valid real input report and marks unknown at IDLE before calling production preparation. Its runner requires worker exit 74, six normal completions/six arms, the specific entry error, zero sync and no failed guard, through the existing custody artifact validator. The native ABI cases and successful outer restoration observer retain their prior structure. I inspected this wiring but did not execute native support in this round. Successful-return outer snapshots do not prove failure restoration. The prior compiler/cache/header closure was not exhaustively reaudited; those implementation bytes are outside the current behavior delta.

After the coordinator reported candidate v8 terminal, I independently read its raw local archived evidence with Python `tarfile`/`json`, without extraction, and ran the current production `validate_custody_artifact` on its `custody_artifact.json`. This verification exited 0. Archive `C:/Users/AG/新建文件夹/c4_assets/c4-warp-module-author-evidence-v2.tar.gz` independently hashes to `f5dc01cec70068c944429d2f68856ef332ad52525bf7828cf3d82a0c1a9d5877`. The `c4-warp-module-author-v8/request.json` names exact candidate `f83edafa8c30d448127d230a5fb670c89da00194`, Warp and Go2 960x640. Its fault marker names validated-input IDLE with valid report and no guard. Failure is the expected healthy-IDLE-scope `SourceBootstrapError`. The replay-validated worker has IDLE/no guard reference, `guarded_segment_failed=False`, full quarantine, six completions, zero emergency sync and zero `guard_failed` events. Its supervisor has six arms/six completions, `EXITED`, observed return code 74, no pending ACK, no active wait, no deadline expiration or kill request. Recorded worker/supervisor PIDs are 2988790/2988789. This independently checked author-produced terminal artifact supports native entry closure only; it is not a reviewer-launched campaign or proof of the separate aggregate boundary. I did not independently inspect `/proc` for those PIDs or revalidate the complete runtime-custody/input chain. The first archive-read attempt encountered PowerShell pipeline encoding of the Chinese literal path; deriving the path from the actual working directory corrected this inspection-only error.

## Native limits, process hygiene and gates

I did not launch, signal or take over any native/GPU campaign, change the remote worktree, install dependencies, delete caches or mutate candidate files. The coordinator's predecessor v4–v7 native results remain author evidence whose raw records I did not independently inspect in this round. Candidate unknown-entry v8 received the bounded independent archive verification described above. No statement here approves full native acceptance or asserts a global remote process census; the coordinator's assertion that no native jobs remain is not my own census.

Local reviewer tools consisted of read-only Git/source/hash/archive inspection, CPU pytest, and writing the new repro/report outside the candidate. Pytest fixtures own their guard peers and local process helpers. All reviewer tool/test sessions reached terminal results. Initial and final `git status --porcelain=v1` were empty and HEAD remained the exact frozen candidate. No tracked candidate file, private agent instruction or campaign state was edited. The new counterexample and complete report remain outside the repository for verbatim archival by the author.

Remaining native obligations include current-revision cold positives at both resolutions and distinct caches, source consumption and mutation, hostile cache/options/artifact/target/stream/ABI/foreign-record admission, original-guard load/publication/adoption failures and hang, ARMED-before-import, actual-IDLE entry and aggregate faults with exact-target emergency versus unknown/incomplete precedence, retained partial roots, no next leaf/extra guard/sync, scope restoration on failure, terminal worker/supervisor observation and unrelated-process preservation. CPU doubles cannot satisfy these properties.

No bounded implementation approval is issued until F3 is repaired and the exact revised candidate is independently reviewed. Even later bounded code approval is separate from integrated native acceptance. Whole C4 still requires one-slot graph/timeline/arena integration, ten real warmups, R_setup/R_final and final readiness/recording/seal/barrier, CPU-oracle rendering correctness and lifecycle matrices, C4c3/c4, C4d/B3, systematic refactor and fresh whole review. Product PREPARE, performance/benchmark/results/baselines and backend selection remain closed.
