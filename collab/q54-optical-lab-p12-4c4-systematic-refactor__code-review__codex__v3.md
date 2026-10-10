Initiative: Q54 / P12.4 / C4  
Stage: Independent code review — R0 final graph admission  
Author: Codex, fresh reviewer session `/root`  
Version: v1  
Date: 2026-10-10, Asia/Shanghai  
Status: in_review  
Related Files: runtime-integration design-request/review v3; systematic-refactor gap-review v1; code-review v1/v2; implementation-note v4  
Owner Summary: Request changes. Final graph construction and admission have substantial positive evidence, but fault termination does not consume the newly established quiescence proof and unnecessarily synchronizes again.

**Verdict: Request changes.**

One P2 blocking finding remains. No captured measured-submission bypass or premature final-barrier admission was established. Existing approvals retain their original scope; this report grants no new checkpoint approval.

**Independence and candidate identity**

This was a fresh independent review invocation. I did not author the implementation, receive its implementation conversation, implement fixes, or spawn agents. Available session identity is `/root`, with review record `r0_final_graph_review_v1`. No provider/CLI session UUID was exposed; none is invented.

I read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` completely, and inspected the supplied authority, prior reviews, implementation claims, complete delta and relevant surrounding code.

| Item | Observed value |
|---|---|
| Candidate | `C:/Users/AG/新建文件夹/c4_final_graph` |
| Branch | `implementation/c4-final-graph` |
| Base | `3d77448291643ebc1a512fe8c4a9d1be2e78341d` |
| Reviewed HEAD | `8247cd453492c26ead81fcc663db1186bf67a6e5` |
| HEAD tree | `4ab1a109b39bbef3703754f84037d10ffddeaa2e` |
| Last approved code | `99eb1f3aff0436134346d08b2be4f235a910b19d` |
| Delta | 25 files; 2,450 insertions, 194 deletions |
| Working tree | Initial, intermediate and final status clean |

All reviewer-created files are under the designated scratch directory. I did not write `report.md`.

**F1 — P2, blocking: fault termination ignores acknowledged final quiescence**

Locations:

- [guarded_lifecycle.py:540](C:/Users/AG/新建文件夹/c4_final_graph/optical_worker/guarded_lifecycle.py:540), especially terminal-purpose selection at line 548.
- [guarded_evidence.py:528](C:/Users/AG/新建文件夹/c4_final_graph/optical_worker/guarded_evidence.py:528), which preserves the same classification during replay.
- Newly reachable through [guarded_lifecycle.py:350](C:/Users/AG/新建文件夹/c4_final_graph/optical_worker/guarded_lifecycle.py:350), which now establishes final quiescence and admission.

**Trigger:** Successfully prepare the final graph, synchronize the exact target, receive DISARMED, acquire and cancel a sample without enqueueing, then terminate for `channel_eof`. The controller is `SETUP_QUIESCENT`, its guard is IDLE, coverage is complete, and no sample was submitted.

**Actual behavior:** `terminate()` selects `target_sync` whenever historical dirty scope exists. It does not consult the newly recorded final quiescence. The reducer likewise permits cancellation only when no dirty history exists. Both therefore require another emergency synchronization despite the acknowledged final barrier.

Independent production-controller probes observed:

| Lane | Additional sync outcome | Sync calls | Terminal disposition |
|---|---|---|---|
| Warp | Success | Final + emergency | `aborted_synced_exit` |
| CUDA | Success | Final + emergency | `aborted_synced_exit` |
| Warp | Injected failure | Final + emergency | `quarantined_process_exit` |
| CUDA | Injected failure | Final + emergency | `quarantined_process_exit` |

Before termination, the probes checked IDLE guard state, `SETUP_QUIESCENT`, zero terminal sample records, and zero transaction/result holds after cancellation.

This contradicts runtime-integration design v3 §8: healthy IDLE, complete coverage, proven final setup quiescence and no sample possible require cancellation, whole-graph release and exit. Emergency synchronization belongs to the branch with possible outstanding target work.

The consequence is an unnecessary blocking native operation and a new failure opportunity: its failure quarantines an already-quiescent graph. I did not establish unsafe release or memory corruption.

This is within the newly reachable final-preparation fault lifecycle. It does not require implementing the excluded C2f product CLOSE operation.

**Why existing checks miss it:** The final-graph unit test stops after cancellation and later scope rejection. The fixed native driver deliberately invokes the existing termination path, and its acceptance checks expect an emergency segment. Both archived `e31b83a` success runs record one emergency synchronization after final quiescence. Successful replay confirms internal consistency with the current reducer, not compliance with the approved classification.

**Repair direction:** Make the existing terminal authority and v4 reducer recognize exact current final quiescence plus no possible sample submission. Preserve sticky dirty history, failure/unknown-scope precedence, custody and bounded cleanup/exit supervision. Preserve historical v3 interpretation. Do not add another lifecycle engine or open product clean-close.

**Required proof:** Both-lane post-barrier EOF/parent-loss and acquire/cancel cases must avoid another synchronization. Failed or missing DISARMED, compromised scope and genuinely outstanding work must retain their existing conservative paths. Add replay counterexamples and native evidence for the repaired terminal path at its actual revision.

Reproduction: [test_reviewer_probes.py](C:/Users/AG/新建文件夹/c4_assets/r0_final_graph_review_v1/test_reviewer_probes.py), `test_quiescent_cancel_still_invokes_emergency_sync`. Results: [terminal-probes.log](C:/Users/AG/新建文件夹/c4_assets/r0_final_graph_review_v1/terminal-probes.log).

**Maintainability assessment**

The change makes useful progress toward the requested simplification:

- Warp setup and arena execution share one recorded phase/reset/event loop.
- Warp delegates slot, FIFO, generation and hold transitions to `NativeOpticalArena`.
- Both adapters use the common guarded terminal observation.
- Final construction reuses the frame binder and warmed physical storage.
- The benchmark spec wrapper delegates to the production constructor. Independent AST comparison found the relocated construction body unchanged except for obtaining `case_contract` from its argument.
- Lifecycle/runtime/combined replay use the explicit nested lifecycle version.

M1 remains open. [native_bootstrap.py:3480](C:/Users/AG/新建文件夹/c4_final_graph/optical_worker/native_bootstrap.py:3480) adds another orchestration method repeating guard entry, uploaded-binding resolution, audits, phase publication and generic failure routing. The existing lane-specific failure distinction remains elsewhere in that class.

This is a retained structural follow-up, not a second newly reproduced blocking defect. Further stages could repeat the same routing drift. Follow the existing M1 direction: consolidate orchestration and state-based failure routing within existing owners, preserving original exceptions and guard/custody semantics. This checkpoint must not be recorded as completing M1 or whole-C4 refactoring.

**Other high-risk behavior inspected**

No additional substantive finding was established in these areas:

- Observed warmups are consumed through local identity registries. Final readiness is new; provisional readiness remains unchanged.
- Final factories and frame/plan identities are new, while outputs and terminal/diagnostic events reuse warmed storage.
- Frozen R0 row admission precedes materialization/runtime initialization in the production source-session path.
- Final coverage is sealed before synchronization. Admission follows successful DISARMED and post-barrier graph, scope, handle and coverage checks.
- Failed final synchronization or lost DISARMED does not grant quiescence or admission.
- Captured acquisition checks final admission and compromised scope. Common lease enqueue reaches the controller’s closed submission latch.
- Shared resolution retains pre-wait and post-DISARMED binding validation before dropping the transaction hold.
- Independent additional probes rejected copied final graphs, diagnostic handle drift during DISARMED and post-seal setup work.
- The new Warp adapter’s CPU tests exercise authority copies, cancellation, generations, failure holds, resolution and equal acquire work counts for 1/120/600-frame catalogs. These are not native timing or execution proof.

**Executed verification**

Local environment: Windows PowerShell; `C:/Python314/python.exe`, Python 3.14.0; pytest 9.1.1; NumPy 2.5.2. Warp and trimesh are unavailable.

Tests used `-B`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider` and scratch-only `--basetemp`.

| Verification | Result |
|---|---|
| All nine requested suites | **533 passed, 4 skipped**, 160.75 seconds |
| Independent probes, corrected harness | **8 passed**, 31.21 seconds |
| Expanded terminal counterexample | **4 passed, 6 deselected**, 16.28 seconds; assertions document the defective behavior |
| All 90 catalog source bodies | Exact committed LF digests and AST policy passed |
| Spec-constructor relocation comparison | Passed; only case-contract lookup changed |
| `git diff --check 3d77448 HEAD` | Passed |
| Final HEAD/tree/status checks | Unchanged and clean |

The nine suites were:

```text
tests/unit/optics/test_native_final_graph.py
tests/unit/optics/test_warp_arena.py
tests/unit/optics/test_native_optical_arena.py
tests/unit/optics/test_cuda_arena_resolution.py
tests/unit/optics/test_guarded_lifecycle.py
tests/unit/optics/test_guarded_evidence_fixtures.py
tests/unit/optics/test_native_sources.py
tests/unit/optics/test_native_source_policy.py
tests/unit/optics/test_runtime_custody.py
```

Actual skips were two cases requiring the configured real Go2 asset path and two POSIX release-byte checks. No additional native-fixture skips occurred in this invocation. CPU doubles do not acquire native evidentiary status merely because these tests passed.

Catalog SHA-256 independently matched:

`618eb6a69be829f7d153e1a4680a25c22ab9ae4de7b71527ff8fdc1dceffba6e`

The initial scratch-probe run had six harness failures: an invalid fault enum, an incorrect assumption that the shared fixture used production `session.fail`, and overly narrow expected exceptions. I corrected only scratch code and retained the initial log. The copied-graph probe therefore establishes rejection, not production source-session disposal.

Logs and scripts:

- [cpu-core.log](C:/Users/AG/新建文件夹/c4_assets/r0_final_graph_review_v1/cpu-core.log)
- [probes.log](C:/Users/AG/新建文件夹/c4_assets/r0_final_graph_review_v1/probes.log)
- [audit_archive.py](C:/Users/AG/新建文件夹/c4_assets/r0_final_graph_review_v1/audit_archive.py)
- [archive-linux-replay.log](C:/Users/AG/新建文件夹/c4_assets/r0_final_graph_review_v1/archive-linux-replay.log)

Final probe-script SHA-256: `0689f43d4bca2a6fd329a8c299e0e38474d55d68c434c006ad847b3a2388347c`.

**Native archive verification and limits**

Independently verified:

- Size: **28,937,648 bytes**
- Members: **189**, with no duplicate names or unsafe/link members
- SHA-256: `3f88013319638b1b47d60e49c598a9cc97e66b2c76366819365b1375f6539c57`
- Archived drivers, archiver and closure JSON match adjacent originals.
- Requests bind to the source catalog at their actual Git revision.
- Worker/supervisor/parent PID joins agree.
- Lifecycle, runtime-custody selection, runtime record and available combined artifacts replay.
- Warmups record ten distinct transactions and terminal completions, indices 0–9 and ten completed guard sequences.
- Successful final records bind readiness count ten, frames 1–3, generations 1–3, final graph, quiescence and acquire/cancel observations.

| Actual revision | Lane/case | Archived outcome |
|---|---|---|
| `c59edc74d7d766eb71bd4b7f81d94045440cb889` | Warp success attempt | Original combined replay failed; preserved error confirmed |
| `e31b83a7b03bc84df8cf63ba7ad6ff64898b3ee3` | Warp/CUDA primary preparation | Worker exit 0; 27/23 normal completions; one subsequent emergency sync each |
| `f05235f14c75c0c75b2bef7f473d66df8d1ec8b8` | Warp/CUDA injected final-sync failure | Worker exit 74; quarantine; no final quiescence; zero emergency syncs |

Current replay of the `c59edc7` records does not retroactively turn its original failed acceptance into success.

The native success records show target device 1 and the held stream during the barrier, with exact before/after restoration. CUDA’s recorded baseline is **logical device 0 but raw driver device 1**; I do not claim raw-device-0 restoration.

Initial local manifest parsing encountered Windows handling of Linux absolute paths. A lexical path adapter enabled local descriptive replay. I then repeated and passed archive verification on Linux using the unmodified production validators in the clean remote `f05235f` checkout. Relevant replay/environment source files are unchanged between that revision and candidate HEAD. This was read-only CPU replay, not native execution.

The `e31b83a`→`f05235f` diff changes only the two support-driver files. Native evidence does not cover the later `8247cd4` admission/latch delta. No additional author diagnostic evidence was supplied during this review.

**Unexecuted obligations and exact scope**

This review does not establish:

- Native execution of the latest admission/latch delta.
- Native diagnostic final preparation.
- Final-sample rendering, native guarded final-arena resolution or output correctness at HEAD.
- Full row/profile/shadow/native failure matrices.
- Actual lost-ACK, hang, parent-loss or cleanup-exit behavior for the new final stage.
- R1 composition, C2f token dispatch, post-reclaim close or product activation.
- Performance, backend selection, results/baselines, R2/S/host delivery, shipping or whole C4.

No new scope is approved while F1 remains unresolved. The positive evidence supports substantial R0 final-graph construction and admission behavior, with native observations confined to the revisions above. Prior approval at `99eb1f3` remains bounded to its original consolidation/repair scope.

**Process hygiene**

No candidate edits, Git mutations, dependency installations, GPU jobs, GPU resets, signals or agents were launched.

Read-only SSH confirmed the remote checkout clean at `f05235f`; its branch output was empty, consistent with detached HEAD. All 15 recorded archive PIDs were absent when checked:

`75699/75759/75765; 126947/127296/127308; 102476/102798/102800; 160220/160601/160605; 170404/170845/170851`.

Observed local pytest processes included PID 29252, parent 34824, and PID 23472, parent 38780. Their tool sessions completed. Final scoped checks found neither those PIDs/direct children nor a Python pytest process matching this review’s scratch paths. Later short-lived process identities were not all sampled; no exhaustive OS attestation is claimed.

All reviewer tool sessions returned terminal results. Remote replay used stdin and `-B`, without remote scratch files. No reviewer-owned unresolved hang was observed. The candidate remained frozen and clean.