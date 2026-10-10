```text
Initiative: Q54 / P12.4 / C4
Stage: Independent follow-up code review — R0 final preparation and fault disposal
Author: Codex, fresh independent reviewer /root
Version: v2
Date: 2026-10-10, Asia/Shanghai
Status: accepted
Related Files: runtime-integration design-request/review v3; systematic-refactor gap-review v1; code-review v1/v2/v3; implementation-note v4/v5
Owner Summary: Approve with comments for the exact bounded candidate. F1 is closed. M1 remains open and nonblocking. Product execution and full C4 remain closed.
```

**Verdict: Approve with comments.**

No new substantive blocking defect was established. The repair consumes acknowledged final quiescence without another GPU synchronization, while retaining failure precedence, sticky scope history, terminal custody and supervised cleanup.

This approval covers the complete R0 final-preparation delta described below. It does not establish final-sample execution, product activation or full C4 acceptance.

**Independence and candidate identity**

This was a fresh independent review invocation. I did not author the implementation, inherit its implementation conversation, implement fixes or spawn agents. The available reviewer identity is `/root`, with scratch record `r0_final_quiescence_review_v2`; no separate CLI session UUID was exposed.

I read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` fully. I inspected the accepted design and review, preceding findings, original repro, implementation claims, actual delta and surrounding control/resource paths.

| Item | Observed value |
|---|---|
| Candidate | `C:/Users/AG/新建文件夹/c4_final_graph` |
| Branch | `implementation/c4-final-graph` |
| Base | `3d77448291643ebc1a512fe8c4a9d1be2e78341d` |
| Base tree | `778383b42451cdc788d034682a81433fea6560fb` |
| Reviewed HEAD | `640cf9f24490e9bf7d259c37ec9adfb0412ac1ba` |
| HEAD tree | `278cb20ce095bdb49ae293c29bda4c47fcf2de15` |
| Prior reviewed code | `8247cd453492c26ead81fcc663db1186bf67a6e5` |
| Repair/native code | `47114b0af297e27c4e54fa64b83975909ef68252` |
| Last approved code | `99eb1f3aff0436134346d08b2be4f235a910b19d` |
| Complete delta | 27 files, 2,915 insertions, 201 deletions |
| Working tree | Initial, intermediate and final checks clean |

The `47114b0`→HEAD delta changes only `PROGRESS.md` and implementation-note v5. The prior complete review remains unchanged from `90fd43c`, with committed-byte SHA256 `13e020482da10e5d88b0db2acc008b14fd565a263e68cc1b3df76c8d3abbaaf3`.

**F1 — P2 blocking finding: closed**

Relevant implementation: [guarded_lifecycle.py:494](C:/Users/AG/新建文件夹/c4_final_graph/optical_worker/guarded_lifecycle.py:494) and [guarded_evidence.py:146](C:/Users/AG/新建文件夹/c4_final_graph/optical_worker/guarded_evidence.py:146).

The original trigger was successful final graph preparation, exact-target synchronization and DISARMED, followed by acquire/cancel and EOF termination. Historical dirty scope caused another emergency synchronization despite current final quiescence.

The repaired behavior matches design v3 §8:

- The existing v4 reducer recognizes final quiescence only while its completed barrier remains the latest normal segment, with IDLE guard state and no failed, terminal-failed or compromised scope.
- Before using that classification, the live controller checks its retained proof, `SETUP_QUIESCENT` state, closed submission latch, final graph/admission identity, sealed guardian, epoch, target, operation count, coverage and graph digest.
- Invalid live proof records failure and selects quarantine.
- Failed/active/pending guard state and compromised scope retain precedence over cancellation.
- Historical dirty devices remain recorded.
- Cancellation still requires the existing emergency-role ARM/ARMED handshake for bounded cleanup and exit. It performs no GPU synchronization and no emergency COMPLETE/DISARMED.
- A later normal segment invalidates the earlier cancellation proof and retains the conservative synchronization path.
- Historical v3 interpretation remains unchanged.

I reran the unchanged original repro, whose SHA256 matches the supplied `0689f43d4bca2a6fd329a8c299e0e38474d55d68c434c006ad847b3a2388347c`. All four old-bug assertions failed at the expected comparison:

```text
calls = ['final_sync']
emergency_sync_attempt_count = 0
disposition = cancelled_before_submission_exit
```

These expected failures demonstrate the repaired behavior; they are not new regressions.

Independent additional probes covered both lanes with an acquired but unsubmitted sample, a later failed normal segment, cleanup throwing and reentrant termination during cleanup. They verified original exception retention, zero additional synchronization, full versus remaining quarantine, and rejection of duplicate termination without repeated destructive work.

**Other high-risk behavior inspected**

No additional substantive finding was established in these areas:

- Final readiness consumes locally registered observed warmups and retains the exact prepared input and compiled owners. It creates new readiness/factory identities without mutating provisional readiness.
- Final frame/C3/arena plans reuse warmed outputs and terminal/diagnostic events. The origin remains upstream of the final graph and barrier.
- Production row admission precedes materialization/runtime initialization.
- Coverage seals once before the final target barrier. Admission follows successful DISARMED and post-barrier graph, scope, handle and coverage validation.
- Barrier failure, lost DISARMED and binding drift do not grant quiescence or admission.
- Captured acquisition checks final admission and compromised scope. The common lease enqueue path reaches the still-closed submission latch.
- Warp shares recorded phase/reset/event execution between setup and arena transactions and delegates slot/generation/hold transitions to `NativeOpticalArena`.
- Shared guarded resolution preserves validation before waiting and after DISARMED, before dropping transaction ownership.
- The relocated arena-spec construction body is AST-equivalent except for obtaining `case_contract` from its argument.
- Lifecycle, runtime-custody and combined replay use the explicit nested lifecycle version. Input-custody readiness retains its earlier meaning.

These conclusions respect the accepted trusted-process boundary; they do not claim containment of arbitrary hostile Python reflection or native calls.

**M1 — retained P2 structural follow-up, nonblocking here**

Location: [native_bootstrap.py:3480](C:/Users/AG/新建文件夹/c4_final_graph/optical_worker/native_bootstrap.py:3480), together with existing stage orchestration and failure routing.

The final-preparation entry repeats guard setup, uploaded-binding lookup, audits, phase publication and generic failure handling. Existing lane-specific failure routing also remains.

The consequence is continued maintenance risk: later stages can diverge in failure classification or publication ordering. Current tests establish particular paths, not the absence of duplicated orchestration.

Retain the existing direction: consolidate sequencing and state-based failure routing within the current owners, preserving original exceptions, guard outcomes, custody and late-publication rejection. Do not introduce another lifecycle/workflow engine. Future closure needs the existing healthy-IDLE, active/pending/failed-guard, source-integrity and exception-retention counterexamples, plus native scope evidence wherever entry boundaries change.

This review does not close M1 or the whole-C4 simplification obligation.

**Executed verification**

Local environment: Windows PowerShell, Python 3.14.0 at `C:/Python314/python.exe`, pytest 9.1.1 and NumPy 2.5.2. Warp and trimesh are unavailable.

Tests used `-B`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider` and scratch-only `--basetemp`.

| Check | Actual result |
|---|---|
| All nine requested suites | **569 passed, 4 skipped**, 250.38 seconds |
| Unchanged original reviewer file | Six other probes passed; four old-bug assertions failed as expected |
| Corrected independent additional probes | **8 passed**, 33.15 seconds |
| Exact committed LF source hashes and AST policy | All **90** catalog sources passed |
| Arena-spec relocation AST comparison | Passed |
| Production/test/benchmark `git diff --check` | Passed |
| Full-delta `git diff --check` | Seven trailing-space notices in the unchanged archived report’s Markdown header |
| Final revision/tree/status | Unchanged and clean |

The nine suites were `test_native_final_graph.py`, `test_guarded_lifecycle.py`, `test_guarded_evidence_fixtures.py`, `test_native_sources.py`, `test_native_source_policy.py`, `test_runtime_custody.py`, `test_warp_arena.py`, `test_native_optical_arena.py` and `test_cuda_arena_resolution.py`, all under `tests/unit/optics`.

Actual skips:

- Two real-Go2 cases: asset path not configured locally.
- Two POSIX release-byte checks: Windows uses the explicit LF fixture.

Catalog SHA256 independently matched `b0ba5a8bc58dc4299f844afe224062ea40d1bfee9716fd5113c0163cb4a4f375`. Production verification was not changed to normalize source bytes.

The first combined probe execution had four scratch-harness failures because I attempted to patch a read-only guardian instance method. I corrected the scratch injection to patch the class method and reran all eight additional cases successfully. The initial log is retained. Initial PowerShell quoting and static-audit API/key lookup mistakes were also corrected; these were inspection-tool errors, not candidate failures.

Evidence:

- [CPU suite log](C:/Users/AG/新建文件夹/c4_assets/r0_final_quiescence_review_v2/cpu-core.log)
- [Original repro and initial probe log](C:/Users/AG/新建文件夹/c4_assets/r0_final_quiescence_review_v2/probes.log)
- [Corrected probe log](C:/Users/AG/新建文件夹/c4_assets/r0_final_quiescence_review_v2/probes-corrected.log)
- [Independent probes](C:/Users/AG/新建文件夹/c4_assets/r0_final_quiescence_review_v2/test_independent.py)
- [Archive audit script](C:/Users/AG/新建文件夹/c4_assets/r0_final_quiescence_review_v2/audit_native.py)
- [Archive replay log](C:/Users/AG/新建文件夹/c4_assets/r0_final_quiescence_review_v2/archive-replay.log)

**Native evidence and its adequacy**

All three supplied archive hashes, sizes and member counts matched. There were no duplicate names or unsafe/link members.

| Archive | Bytes | Members | SHA256 |
|---|---:|---:|---|
| v1 | 28,937,648 | 189 | `3f88013319638b1b47d60e49c598a9cc97e66b2c76366819365b1375f6539c57` |
| v2 | 11,635,180 | 79 | `46b2f786f60efbd53ead89fae9e230683a3abf019efddefa336bdf963a933829` |
| v3 | 11,595,877 | 75 | `3b60c683f01de96855477b7dca5681ca89243595f319cda7a043d938791d3371` |

Read-only SSH confirmed the primary-host checkout clean and detached at `47114b0`. I replayed the archives through its unmodified production validators using `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -`. This was fresh CPU verification of archived native evidence, not a fresh GPU campaign.

For both v3 runs, I verified revision/catalog/environment bindings, runtime-custody selection and attachment, lifecycle/combined-supervisor replay, ten distinct warmup transactions/completions, final readiness count ten, three plans, graph/quiescence bindings, acquire/cancel, zero sample submissions and PID joins.

| Native case at `47114b0` | Normal completions | Cleanup guards | Extra syncs | Observed worker exit |
|---|---:|---:|---:|---:|
| Warp, injected `channel_eof` | 27 | 1 | 0 | 0 |
| CUDA, injected `parent_lost` | 23 | 1 | 0 | 0 |

Both records show completed release, `cancelled_before_submission_exit`, no terminal-sync entry marker, and supervisor `EXITED` without deadline expiry or kill request. The terminal guard remains armed through worker cleanup; no completion/disarm exchange is manufactured.

The native barrier observations use target device 1 and the held stream, with exact recorded ambient restoration. CUDA’s baseline is logical device 0 but raw driver device 1; this is not evidence of raw-device-0 restoration.

These real native states and production disposal paths adequately address F1’s redundant synchronization. The fault categories are injected into termination. They do **not** demonstrate externally killed parents, actual transport failure detection or product request-loop IPC.

Historical evidence remains bounded:

- Old positive terminal traces fail tightened v4 replay with `wrong terminal purpose`; they are not repaired successes.
- The original `c59edc7` acceptance failure remains historical failure.
- The `f05235f` final-sync failure records replay with quarantine and no emergency retry.
- Both `8247cd4` diagnostic runs retain their historical revision and F1 limitation.
- Archived warmup RGB bytes match between primary and diagnostic runs; this does not prove final-sample correctness or performance.

**Unexecuted obligations and exact approved scope**

Approval binds **`640cf9f24490e9bf7d259c37ec9adfb0412ac1ba`**, including its documentation/evidence delta, for:

Observed R0 warmups → new final readiness/origin/factories → actual C3 and dormant arena plans over warmed storage → immutable coverage seal → exact-target barrier and DISARMED → acquire/cancel → state-appropriate supervised fault disposal.

The following remain unproved or closed:

- Captured final enqueue, final-sample rendering, native final-arena resolve/reclaim and final-output correctness.
- Repaired native diagnostic termination and the complete cross-lane/fault/mode matrix.
- Actual parent death, transport loss, unavailable ACK and cleanup/exit-hang behavior for this final stage.
- All required rows/profiles/shadow combinations and broader native failure/numerical matrices.
- R1 composition, C2f tokens, post-reclaim close and ACK-before-cleanup integration.
- M1 and systematic whole-C4 structural closure.
- Product PREPARE/activation, R2/S/host delivery, C4c3/c4, C4d/B3, external holds/depth two, performance, backend selection, results/baselines/a3b and shipping.

**Process hygiene**

No tracked files, private agent context or remote files were changed. No GPU jobs, dependency installations, resets, process signals or agents were launched. I did not manually write `report.md`.

Observed local test PIDs included `42432`/parent `5932` and `44708`/parent `1512`. Their sessions completed; final scoped checks found neither those processes/direct children nor a Python pytest process matching this review’s scratch paths. The corrected probe session also returned exit zero; its PID was not separately sampled.

All six recorded native parent/supervisor/worker PIDs were absent on the primary host: `538794`, `539099`, `539108`, `569566`, `570865`, `570881`. Archived GPU-process records likewise contain none of those PIDs. Unrelated GPU activity was present and untouched.

All reviewer tool sessions reached terminal results. No reviewer-owned unresolved hang was observed. These are scoped checks, not an exhaustive process or GPU-idleness attestation.