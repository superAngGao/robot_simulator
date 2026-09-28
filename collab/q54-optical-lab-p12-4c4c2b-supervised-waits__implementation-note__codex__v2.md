# C4c2b single-commit transition repair

Initiative: q54-optical-lab-p12-4c4c2b-supervised-waits
Stage: implementation-note
Author: codex
Version: v2
Date: 2026-09-29
Status: in_review
Related Files: implementation-note v1; code-review v1; wait_supervision.py;
test_guarded_wait_core.py
Owner Summary: Supersedes v1's deferred CPU restoration mechanism. A later
return-window reentry still lost termination at9b8bc3b. Publish each state/event
pair at one append-only commit point instead; fresh review remains required.

## Exact history and review status

Overall implementation base fcb9608. Initial candidate5dd04f8 received Request
changes from independent code-review v1, preserved verbatim in collab. The first
repair9b8bc3b passed author Linux153 tests in16.46s, including realSIGUSR1 and
the subprocess matrix. Its log /data3/ang.gao/c4-c2b-9b8bc3b-cpu.log has SHA256
ea19c7adbc7fc561443bfba39e1476677b0668c7349b4920fab0a662b4bbb8b6.

Fresh session /root/c4_c2b_wait_code_review_v2 terminated with a tool-side
content-safety error before producing a report. No approval or completed
independent test run is attributed to that interrupted session. The author
separately reproduced the following remaining problem and sent it to the
session before termination. It is an author finding, not an invented review.

## Return-window counterexample and repair

After ordinary ARM/ARMED/COMPLETE, schedule core.fault("parent_lost") at the
`_operation_active = False` line of `_invoke` while recording DISARMED. Catch
the nested GuardProtocolError in the signal/callback. In9b8bc3b the finalizer
already tested its latch; the later latch was missed and the outer call returned
IDLE, deadlineNone, completion_count1, terminal_reasonNone. Evidence remained
parseable but omitted the fault. The candidate therefore was not acceptable.

Current implementation removes deferred trace/state restoration entirely.
An exact builtin dict owns append-only integer-indexed `(state, event)` records.
Each transition calculates against one committed predecessor. `dict.setdefault`
is the single publication point; the exact inserted record identity must win.
Nested public entry immediately records terminal custody at that same sequence
position. A stale outer transition cannot overwrite its record, split state
from trace, or clear its deadline. No normal nested operation is executed.
This implementation relies on the indivisible builtin insertion used by the
supported CPython interpreters, not a sequence of Python check/append/assign.
It is supported-process concurrency control, not arbitrary memory protection.

No I/O, process kill, device action or owner release occurs inside a transition.
Once a DISARMED commit has already won, a later parent-loss fault is a subsequent
transition and gets the specified terminal grace deadline. Before that commit,
the original active deadline survives. Both orderings must be terminal, with a
valid trace; neither may end IDLE or claim unobserved process exit. A previously
written ACK whose commit fails still cannot be recalled; the guard remains.

The nine earlier precommit tests remain. Additional source-line scheduling
tests cover every normally executed line in the public scope, append, ACK
handler, clock reader and state reader during DISARMED, explicitly including
the finalizer-release line and builtin publication line. They assert terminal
state, deadline presence, no pending output, correct first reason and replay.
POSIX runs actualSIGUSR1; Windows runs the same-thread callback directly. This
is finite targeted validation, not a universal asynchronous-interleaving proof.

Local Python3.14 core93 passed in0.83s. Linux new-candidate regression and fresh
independent review remain required; prior153 passes do not validate this repair.
All exclusions/gates in v1 and approved designv3 remain unchanged. C4 completion,
GPU preparation/adoption, C4c2 runtime integration, C4c3/c4/d/B3, host/shipping,
a3b and result/baseline remain unapproved/closed.
