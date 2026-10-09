# Warp module preparation: independent code re-review v4

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: bounded code re-review before integrated native acceptance
Author: Codex, independent reviewer session /root/warp_module_code_v4
Version: v4
Date: 2026-10-10
Status: accepted (bounded code scope only)
Related Files: warp-module-preparation design-request v2, code-review v1/v2/v3, implementation-note v4 and test-evidence v2; runtime-integration design-request v3; optical_worker/warp_setup.py; optical_worker/native_bootstrap.py; guarded lifecycle, source catalog, unit and native support tests
Owner Summary: The actual aggregate issuer now rejects unknown scope before a resource API can fabricate a failed guard. The original F3 counterexample and control pass unchanged, prior F1/F2 regressions continue to pass, and the affected CPU suites pass at the frozen candidate. Bounded code approval does not approve complete native acceptance or whole C4.

## Identity and verdict

**Verdict: Approve, for the bounded module-preparation code scope at the exact revision below.** No substantive blocking or non-blocking code finding remains from this re-review. Prior F1, F2 and F3 are closed at the CPU code boundary covered here. Required complete native acceptance remains a distinct unapproved gate; this verdict is not a GPU acceptance claim.

This is a fresh independent reviewer session without the implementation conversation. I did not author or modify the candidate, and did not spawn subagents. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully; the complete supplement v2; parent sections 4, 5, 5.1, 5.2, 6 and 8 including 8.1/8.2; all three complete prior code reviews; implementation-note v4 and test-evidence v2. The first root-level principles lookup was corrected to collab/REVIEW_PRINCIPLES.md. Truncated broad displays were replaced by focused reads.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Original integration base: `0356108a958f34c7d17064a92d2cc7a339b07d0b`.
- Repair/re-review base: `f83edafa8c30d448127d230a5fb670c89da00194`.
- Exact reviewed HEAD: `beedab6143e702a1a50ef0e2686fd0138e194376`.
- Initial and final `git status --porcelain=v1` were empty; HEAD remained unchanged.
- Repair delta: 8 files, 364 insertions and 22 deletions. Production behavior changes only the issuer precondition (four added/two removed lines); guarded lifecycle policy is unchanged.
- Both repair-base and original-base `git diff --check` passed.
- Supplement SHA256: `194b3c25130df77954f5b2de2b556da81009b98f94efb861848ae06566a41a5b`, independently matched.
- Structured old/new catalog comparison found exactly one changed source row, `optical_worker/warp_setup.py`, matching actual SHA256 `40fddec15b483fca513c708aeaba2288e5c033926162d71eb047e2b2e4451458`. No rows were removed; all other catalog fields were equal. Catalog SHA256 is `68ecaeb15bd365aaf46db99a142da45de2e8393848db944d915ca3d567ad0f16`.

## Prior findings and actual counterexample closure

F3: `warp_setup.issue_module_set` now requires actual client IDLE, no failed replay, no active context/reference and no compromised scope before cache validation or `controller.require_adopted`. Unknown or foreign sticky scope therefore produces a plain CPU ValueError. The existing `_fail_warp_preparation` sees the original IDLE history, retains the exception and available roots, tightens unknown custody and delegates to unchanged terminal policy. The mutating `_check_running` rejection is never reached in the original trigger. No completed guard is relabeled as failed and no emergency admission is manufactured.

I read and ran the unchanged original `test_review_warp_unknown_issuance_v3.py`, including its matched direct CPU-exception control. Actual issuer observation: `ValueError("Warp module issuance requires healthy IDLE target scope")`; client IDLE; no new `guard_failed`; five original fixture completions; three records still held; no module-set registry entry or checkpoint; whole-graph quarantine; zero emergency synchronization. The control retains the same state/history/custody with its original RuntimeError. The five fixture completions model upload, optical import and three compiles, not the six real input phases.

F2 remains closed: both original entry cases, with valid and invalid input report, reject through the pure CPU initial healthy-IDLE precheck. Each preserves IDLE, its one original fixture completion, zero failed/new normal guards, zero compile/checkpoint/sync and quarantine. The repository active-guard case still preserves and fails the actual original guard rather than entering IDLE emergency disposal.

F1 remains closed: immutable type-value and entry facts are still computed after legitimate module-hash initialization and compared after load and during later admission. The adapted original production-validator cases reject in-place dtype, dimension and kernel hash drift, with unchanged admission and descriptor replacement controls. The original compile-transition tests still permit initially absent hash initialization and reject load-time dtype/dimension/hash mutation while holding the published executable/mapping and issuing no successful record. These use pinned AST-extracted metadata and CPU dependency/native doubles, not real GPU execution. Local pinned types/context SHA256 values matched the recorded `9224fb234ff61f4a1c3338b7171d545171ab051fb625679568ea80a6af8321bd` and `eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee`; no new remote installation verification is claimed.

## Affected flow and native support inspection

I inspected initial checks, guarded imports, serial compile/adoption, post-DISARMED issuance, source-session failure retention, runtime-binding resource validation and unchanged guarded-lifecycle emergency/quarantine policy. The session and original mapping retain partial roots before fallible registration; published executables are held before inspection/adoption. Module-set issuance still requires the three exact cache records and successful distinct compile guard identities. Later admission still checks registered set/record/definition/executable identities, ABI/options/hash/metadata/artifacts and mapping variants. The revised precondition adds no compiler route, retry, synchronization, readiness or execution authority.

Captured-source publication/audit, existing private-cache/configuration policy and source admission still compose around the same path. The installed compiler/header closure and unchanged full bootstrap were not exhaustively reaudited or rerun in this round. The source-policy, source-custody and runtime-binding suites were rerun for the changed pin and affected integration.

The six fixed native selectors are wired through the existing supervisor/worker, not a new process manager. Entry faults occur after otherwise valid real input. Issuance faults occur after all three real compile guards complete; the wrapper asserts actual IDLE/no context/reference/no failure, valid input and retained record identities. Unknown issuance marks real sticky scope and calls the actual production issuer. Healthy issuance deliberately raises a CPU error at that boundary. Incomplete issuance removes stream guardian membership while retaining the actual stream through the source-session graph, then raises a CPU error; the production failure handler must discover the custody failure. Healthy/incomplete entry invalidate the report after the same initial IDLE assertions; unknown entry keeps it valid.

The runner first validates the existing custody replay and terminal supervisor artifact, then requires worker code 74, exact errors, six or ten original normal completions, corresponding arm counts, zero fabricated `guard_failed`, and zero synchronization for unknown/incomplete faults. Healthy exact-target CPU failures require one emergency sync, `aborted_synced_exit` and completed release; unknown/incomplete require full quarantine. Runtime-custody selection/attachment validation remains in the same runner.

These are inspected test semantics, not executed native acceptance. The fault marker's held-record assertion is taken before fault injection. Successful-return outer scope observations do not prove arbitrary native failure restoration. Full acceptance still needs its own retained-root and failure-restoration proof; the marker or summary alone is not substituted for those properties.

## Independently executed verification

Environment: Windows PowerShell; `C:/Python314/python.exe`, CPython 3.14.0. Pytest used `-B -p no:cacheprovider`; repros and basetemps were outside the candidate. Commands ran from the candidate root.

```text
C:/Python314/python.exe -B -m pytest -q -s -p no:cacheprovider ../c4_assets/test_review_warp_unknown_issuance_v3.py ../c4_assets/test_review_warp_unknown_idle_v2.py ../c4_assets/test_author_warp_abi_repair.py ../c4_assets/test_review_warp_compile_transition_v2.py --basetemp=../c4_assets/review_warp_repros_v4_pytest
```

Observed **12 passed in 4.77s**, exit 0: original F3/control, original F2/control, four ABI admission cases and four compile-transition cases.

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_warp_module_preparation.py tests/unit/optics/test_guarded_runtime_binding.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_source_custody.py tests/unit/optics/test_guarded_lifecycle.py --basetemp=../c4_assets/review_warp_modules_v4_pytest
```

Observed **365 passed in 143.81s (0:02:23)**, exit 0; unified exec session 47254 reached terminal completion. These are current-candidate executions, not the previous reviewer or author's totals. They include healthy IDLE emergency, unknown/incomplete custody, active/failed/pending guard behavior, original guarded partial failure and published-owner custody coverage.

Other read-only checks: Git status/revision/branch/stat/numstat/diff/diff-check, UTF-8 source reads/searches, file hashes and structured catalog comparison. An initial inline Python catalog inspection lost quotes through Windows PowerShell argument passing and produced a SyntaxError; the corrected stdin script exited 0. This was an inspection/tool invocation error, not a candidate defect. No test failure or unresolved local tool session remains.

The predecessor native archive independently hashes to `f5dc01cec70068c944429d2f68856ef332ad52525bf7828cf3d82a0c1a9d5877`. I did not rerun its artifact validators or credit its older positives/mutations/f83edaf entry case as new-candidate execution proof.

Before the final report cut the coordinator supplied terminal v9. I independently read the complete local archive through stdlib `tarfile`/`json` without extraction and ran current `validate_custody_artifact` on its actual custody artifact (exit 0). Archive `C:/Users/AG/新建文件夹/c4_assets/c4-warp-module-author-v9.tar.gz` independently matches SHA256 `add2066ba7d3e62e9bd870134f3cd7966164f70aa87d2dae582410999f137188`. Request names exact `beedab6143e702a1a50ef0e2686fd0138e194376`, Warp, Go2 960x640. Marker identifies actual post-DISARMED IDLE issuance with valid report and three held records; failure is the expected healthy-IDLE-scope ValueError. Replay-validated worker facts show IDLE/no guard reference, guarded_segment_failed=false, ten completions, no guard_failed events, zero emergency sync, full quarantine and no release authorization. Supervisor facts show ten arms/ten completions, EXITED/complete, observed return code 74, no pending ACK/active wait, no deadline expiration or kill request. This independently checked author-produced terminal artifact supports the native original F3 boundary closure only. I did not launch this campaign, independently check its PIDs in /proc, or revalidate its full runtime-custody/input attachment chain. The coordinator's PID-absence checks and ongoing later cases are not my own observations or accepted full native evidence.

## Limits, process hygiene and closed gates

I did not launch, signal, restart or take over any native/GPU campaign; no remote mutation, dependency installation, cache deletion or remote process census occurred. All reviewer-launched tool/test sessions completed. Local fixtures own and close their guard peers and process helpers. No candidate file or private agent instruction was edited. This report is saved outside the candidate for verbatim archival.

Approval is limited to this module-preparation implementation and repair at `beedab6143e702a1a50ef0e2686fd0138e194376`. Native acceptance remains separate: current-revision cold positives at both resolutions/distinct caches; held-source consumption/mutation; hostile cache/configuration/artifact/target/stream/ABI/foreign-record cases; original-guard load/publication/adoption failures and hang; ARMED-before-import; real healthy/unknown/incomplete IDLE entry and issuance disposition; failure restoration and partial-root custody; no next leaf/extra guard/sync; observed worker/supervisor exit and unrelated-process preservation. CPU mocks do not establish these real-device properties.

Whole C4 still requires one-slot graph/timeline/arena integration, ten real warmups, R_setup/R_final and final readiness/recording/seal/barrier, CPU-oracle rendering correctness and lifecycle matrices, C4c3/c4, C4d/B3, systematic refactor and fresh whole review. Product PREPARE, performance/benchmark/results/baselines and backend selection remain closed.

