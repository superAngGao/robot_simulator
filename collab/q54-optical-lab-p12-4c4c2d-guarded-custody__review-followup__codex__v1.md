# C2d1 design approval record

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: review-followup
Author: codex
Version: v1
Date: 2026-10-03
Status: accepted
Related Files: design-request v1/v2; design-review v1/v2; PROGRESS.md
Owner Summary: Record exact scoped design approval; no implementation or gate
change. C4 remains unfinished.

Reviewed proposal commit: `f1f4394a3f84aea1107daa3a09bb8d06cee99f59`.
Proposal v2 SHA-256:
`00aa16cfef0dfd83b7f44081f845c9a3397d79435b4b3cf319e27c467d840983`.
Fresh independent session:
`/root/c4_c2d1_guarded_custody_design_review_v2`.
Verdict: **Approve with comments**, design only.
Complete report preserved unchanged, SHA-256:
`62093885dabc84525ee20b6b81fb189990484f83d75489fe84bd5da6f79191d1`.

The prior independent v1 report rejected the CPU-only clean-cancel branch's
missing bounded cleanup/exit supervision and the underspecified v3 replay
grammar. V2 freezes a terminal ARM/ACK before CPU cancellation cleanup with
zero GPU sync, and exact trace/schema/partial-tail/custody rules. The reviewer
independently reran the old dependency counterexample and a real-socket bounded
CPU feasibility probe. Both returned exit0; no implemented controller, actual
subprocess watchdog kill/reap or real-device safety is claimed by those probes.

Non-blocking C1: v2 line260 contains a trailing space. Preserve the reviewed
bytes/digest in this evidence-only commit. Remove the space at the next scoped
document change and report that formatting delta/check explicitly. The proposal
revision's whitespace check therefore has this known failure, not a clean pass.

Primary183 alias/port40031 was retried twice and timed out. Authorized fallback
101 connected. Exact remote proposal worktree:
`/data3/ang.gao/robot_sim_c2d1_design_v2`, clean at the reviewed revision.
Read-only hardware query showed indices0/1 A100-SXM4-40GB, index2 H200 NVL.
The original dirty worktree retained its two modified and two untracked files;
no reset, environment install, GPU reset, merge or push occurred.

Author ran no GPU/test processes this round. Independent report records owned
CPU probe PIDs and their later absence. Author final process-name query found
only unrelated long-lived Python processes2568 and626988, left untouched;
GPU compute query returned no rows. One earlier scoped grep command failed
from SSH quoting; the process-name query above replaced it. These observations
are not a diagnosis of the old unidentified101 hang or proof of GPU quiescence.

Next authorized work: implement the single guarded lifecycle subtype, exact
ticket-history observation and specified v3 evidence with fixed supervised CPU
tests, then fresh independent code review. Captured-source runtime adoption,
real Warp/Torch setup/fault matrix, final graph/readiness/warmup and C2e/f remain
separate outstanding work. C0 semantics stay unchanged. Product GPU preparation,
host/shipping/B3, a3b, result/baseline and C4-complete claims remain closed.
