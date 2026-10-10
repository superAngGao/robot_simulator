Initiative: Q54 / P12.4 / C4
Stage: R0 final graph F1 repair
Author: Codex, implementation session
Version: v5
Date: 2026-10-10
Status: in_review
Related Files: runtime-integration design/request and review v3; systematic-refactor code-review v3; implementation-note v4
Owner Summary: Reuse acknowledged final quiescence when terminating before sample submission. Remove the redundant GPU wait while preserving supervised cleanup, uncertain-state quarantine and historical dirty scope. Full C4 and speed comparison remain incomplete.

## Requirement and repair

Independent code-review v3 requested changes at 8247cd453492c26ead81fcc663db1186bf67a6e5.
Its F1 reproduces a second target synchronization after the final acknowledged
barrier and acquisition/cancellation. Runtime-integration design v3 section 8
requires release without another GPU wait in that state.

Repair code is 47114b0af297e27c4e54fa64b83975909ef68252. The existing lifecycle
and v4 reducer share the classification through their existing replay state.
Current final quiescence requires its acknowledged guard to remain the latest
normal segment, with no failed guard or compromised scope. The live controller
also checks its base-issued proof against the retained final graph, epoch,
target, operation count, sealed coverage and closed submission latch before
poisoning the state. Cancellation keeps historical dirty devices and uses the
existing supervised CPU cleanup/exit guard. No second lifecycle or new receipt
schema is introduced. Later normal work invalidates the pre-sample proof.

Historical v3 interpretation is unchanged. The previously unapproved v4 positive
traces containing redundant synchronization remain historical evidence at their
own code revision; they are not conformant final termination traces under this
repair. The input runtime-custody schema retains its original meaning.

The native driver now accepts state-appropriate cancellation after final setup,
checks zero extra synchronization and absence of the native sync-entry marker,
and verifies the injected fault reaches the original production controller.
Its EOF/parent-loss selectors inject the terminal fault into the real native
post-barrier state. They do not simulate process death or integrate the still
closed product request loop. CPU tests exercise both faults, both lanes, both
pass modes, and with/without acquire/cancel. Do not label these as live product
IPC or measured sample execution.

## Verification

Local interpreter: C:/Python314/python.exe, Windows, no native Warp/trimesh.

- Final graph, guarded lifecycle and historical evidence fixtures: 239 passed
  in 155.63 seconds before adding the conservative-state cases.
- New both-lane/both-mode conservative-state cases: 20 passed, 48 deselected in
  55.60 seconds. Missing proof, a latched possible sample, unknown scope and
  missing cleanup ACK quarantine without sync/release; a later normal guard
  selects the original target-sync path.
- Guarded lifecycle, evidence fixtures, source/catalog policy and runtime custody:
  420 passed, 4 skipped in 51.49 seconds. After import formatting and final
  catalog regeneration, source/catalog policy: 206 passed, 4 skipped in 50.27
  seconds. Skips are the real Go2 fixture and POSIX release-byte checks.
- Original reviewer script is unchanged, SHA256
  0689f43d4bca2a6fd329a8c299e0e38474d55d68c434c006ad847b3a2388347c.
  Its four assertions of the old redundant-sync behavior now fail at the
  expected assertion: both lanes show only final_sync, zero emergency attempts,
  cancelled_before_submission_exit. This is a reproduced behavior change, not
  a passing regression suite. Output: c4_assets/final-quiescence-original-repro-47114b0.log.
- Ruff on the changed production modules and final graph test passes; git diff
  --check passes. All 90 prospective committed LF source bodies pass the
  existing AST policy. Final catalog SHA256:
  b0ba5a8bc58dc4299f844afe224062ea40d1bfee9716fd5113c0163cb4a4f375.

## Prior independent report and supplemental diagnostic evidence

The full report was archived unchanged in evidence-only commit 90fd43c:
collab/q54-optical-lab-p12-4c4-systematic-refactor__code-review__codex__v3.md,
SHA256 13e020482da10e5d88b0db2acc008b14fd565a263e68cc1b3df76c8d3abbaaf3.
The CLI JSONL records fresh session 01a123d7-f3c6-7310-825d-844b51ee188a;
this externally recorded identity supplements the report without editing it.
Verdict remains Request changes until fresh independent review of this repair.

Supplemental native archive v2 was obtained at actual clean revision 8247cd4:
c4-final-graph-native-evidence-v2.tar.gz, 11,635,180 bytes, 79 members,
SHA256 46b2f786f60efbd53ead89fae9e230683a3abf019efddefa336bdf963a933829.
Both lanes completed real Go2 960x640 diagnostic final preparation; their old
termination still exhibits F1. All six owned PIDs were absent at archive.
Warmup frame 10 RGB was bitwise equal between primary e31b83a and diagnostic
8247cd4, separately in both lanes: float32, 7,372,800 bytes, SHA256
7e0ca5304509b45656c6995cb4b5f3c92f4e6780caf241796d94d0ddcb7c1562.
This does not establish final sample execution, all-channel CPU correctness,
performance, or a conformant termination at the repaired revision.

## Native repair evidence

Both actual47114b0 runs completed through the unchanged production supervisor
in /data2/shared/ang.gao/robot_sim_final_quiescence_47114b0 (clean detached HEAD),
using /home/ang.gao/miniforge3/envs/robot_sim/bin/python 3.12.13 and existing
frozen environment contracts/assets. Target1 H200, existing Go2 960x640 R0 row,
primary pass, ten setup warmups and three dormant final plans:

- Warp runtime_final_graph_channel_eof: parent538794, supervisor539099,
  worker539108; normal setup segments27, cleanup guard1, worker exit0.
- CUDA runtime_final_graph_parent_lost: parent569566, supervisor570865,
  worker570881; normal setup segments23, cleanup guard1, worker exit0.

Each acquires/cancels a final plan without enqueue, then injects the indicated
fault into production termination. Both acknowledge final quiescence, release
the graph and terminate with cancelled_before_submission_exit, zero emergency
sync attempts and no terminal_sync_entered.json. The fixed acceptance driver
replays the combined lifecycle/supervisor artifact and full runtime-custody
attachment, checks target/stream and ambient restoration, and observes/reaps
actual worker exit. This is native disposal evidence after simulated fault
notification, not a claim of an externally killed parent or working product IPC.

Raw archive: c4-final-graph-native-evidence-v3.tar.gz, 11,595,877 bytes,
75 members, SHA256
3b60c683f01de96855477b7dca5681ca89243595f319cda7a043d938791d3371.
It contains both original run directories, run_final_graph_r0_v4.py, archiver,
environment reference and closure record. Local copy is under c4_assets;
remote copy under /data2/shared/ang.gao. All six owned PIDs were absent when
archived. No install, reset, broad signal or performance measurement occurred.

## Continuation

Fresh review must close F1 against the actual repair and native evidence.
M1 remains open: consolidate repeated optical-stage orchestration and failure
routing within existing owners. Do not introduce another workflow engine.
Then continue the approved R1 update/refit-or-rebuild/render composition and
C2f token/resolve/reclaim/close integration, required numerical/native failure
matrices, C4c3/c4 and C4d/B3, and systematic whole-C4 review. No product PREPARE,
sample submission, results/baselines, speed verdict or backend selection is
enabled by this repair or by its bounded approval.
