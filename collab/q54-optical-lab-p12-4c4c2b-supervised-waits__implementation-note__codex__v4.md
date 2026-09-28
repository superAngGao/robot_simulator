# C4c2b terminal publication requires the completed write fence

Initiative: q54-optical-lab-p12-4c4c2b-supervised-waits
Stage: implementation-note
Author: codex
Version: v4
Date: 2026-09-29
Status: in_review
Related Files: implementation-note v3; wait_supervision.py;
tests/unit/optics/test_guarded_wait_core.py
Owner Summary: Close the independently reproduced second-signal interruption
between terminal publication and socket shutdown. New candidate requires its
own clean Linux validation and fresh independent review; no approval inferred.

## Independent session outcome and evidence limits

The fresh session `/root/c4_c2b_wait_code_review_v4` inspected candidate
8592902240e99474aa7223f6470e6d9e64ddfcf8. It reported an independent Linux run
of 173 passed in 16.67s, then the concrete counterexample below. The session
ended with a tool error before delivering a final report. There is no final
verdict or full v4 report to archive, and this interrupted review is not an
approval. This paragraph records its interim communication, not a substitute
review report. Completed v1/v3 reports remain archived unchanged.

The reviewer reproduced on Windows 3.14 and Linux 3.12: suspend transport at
its actual send; expire its deadline in a first handler; interrupt after the
deadline-expired record commits but before `_fence_output` in a second public
core call. The second call raises; the first handler catches it. The outer
transport then sends the complete ARMED or DISARMED frame after TERMINATING.
Deadlines and replay remain internally consistent, but cease-output is false.
The external scratch reproduction is
`C:/Users/AG/新建文件夹/c4-c2b-fence-interruption-repro-v4.py`, SHA-256
59f700363c907b45bd3f39b9264861ece3db627783e0c939f9b0cb0705d4cded.

## Bounded repair

`_append` now completes the OS write fence before publishing any transition
whose resulting state is TERMINATING. Only then may the single dictionary
insertion publish the immutable state/event pair. An interrupted terminal
attempt cannot publish a claim whose required write-side shutdown has not
happened. A nested terminal transition must satisfy the same prerequisite;
stale outer publication still cannot replace the already occupied record.

This supersedes v3's explicitly post-commit fence ordering. The state/event
grammar, canonical evidence, deadline and counter contracts are unchanged.
The fence remains a bounded control-socket operation, not device completion,
process exit, resource release, or permission to discard guardian custody.

Eight added negative cases cover ARMED/DISARMED and interruption before the
fence condition, at fence invocation, before record insertion and immediately
after it. Linux exercises real SIGUSR1/SIGUSR2; Windows uses same-thread
scheduling. They require actual peer EOF with zero newly transmitted bytes,
terminal state, the original deadline, zero completed guards, one expiry
record, and valid replay. They do not replace real subprocess hang/backpressure
regression or future GPU lifecycle validation.

## Verification and retained boundary

Author local Python 3.14 core test: 120 passed in 2.14s; Ruff checks passed.
Clean Linux candidate tests and a fresh full independent review are pending.
Historical 8592902 author Linux log: 173 passed in 16.83s, SHA-256
73bea3cf0d1c8c22fe28ae1fed3669d3f981f982ba51adcfce94d84a87e87a55.
That older run is not evidence for this delta.

183 SSH timed out; the authorized 101 fallback retains separate clean revision
worktrees. No dependency installation or existing worktree replacement.
Shared collaboration policy and all C0/C1 runtime protocols remain untouched.
C4c2 runtime integration still needs its complete separate design; C4c3/c4,
C4d/B3, product GPU preparation, host/shipping, a3b and result/baseline stay
closed. This CPU dependency slice does not complete C4.
