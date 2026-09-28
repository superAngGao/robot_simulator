# C4c2b guarded-wait CPU core

Initiative: q54-optical-lab-p12-4c4c2b-supervised-waits
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-09-29
Status: in_review
Related Files: design-request v3; code-review v1; optical_worker/wait_protocol.py;
optical_worker/wait_supervision.py; tests/support/guarded_wait_process.py
Owner Summary: Inert CPU protocol implementation; first independent review
found a same-thread reentry defect. This revision repairs and tests that defect,
but requires fresh independent review before this dependency slice is accepted.

## Scope and predecessors

Design v3 was approved at ff3b4db and archived at fcb9608. Initial code is
5dd04f85731ebe4089bdfc01f507bd75a67761ec. The candidate containing this note
includes the bounded R1 repair and the complete first review, not approval.
The review archive's SHA256 is
43cc543a96d9d9c1e86d1025af87366bdd7c706bfc6ce93bcbdce54fe7e58aae.

Two stdlib-only modules provide exact versioned configuration/wire records,
bounded framing, an exact-object client ticket, the supervisor transition core,
nonblocking transport and replayable CPU evidence. The fixed subprocess harness
runs these actual components. No production consumer adopts them in this slice.

Normal COMPLETE retains the original operation deadline through pending
DISARMED output. Partial progress does not restart budgets. Kill request,
kill failure and observed process exit are distinct. Evidence consistency never
creates a live ticket or proves GPU/resource safety. Sequence/count/trace capacity
uses the encoding-derived constants in v3, not an invented product threshold.

## R1 repair

Every externally callable core operation now enters a non-reentrant scope;
internal composition uses distinct private implementations. Nested public calls
cannot commit state. Detection is sticky even if the callback catches its error.
Before the interrupted outer operation returns any response/commit success, it
discards its tentative CPU trace/state changes, restores the pre-call guard and
records the terminal failure with the original deadline as an upper bound.
This is CPU transition atomicity, not rollback of a device submission or socket
write: the core performs neither. A socket ACK already written before its commit
fails cannot be recalled; the original guard remains terminal and armed.

Clock callbacks cannot turn swallowed nested errors into ARMED. Transport
service also rejects reentry and preserves terminal state/output cancellation.
Public configuration properties are read-only, with frozen exact dataclass
content. This remains supported-path control, not an arbitrary-Python security
boundary or native executable-leaf capability.

Tests schedule reentry at the actual transition-to-trace commit boundary, both
with propagated and swallowed exceptions, across ARM, ARMED, COMPLETE, DISARMED,
terminal entry, expiry, kill request/failure and observed exit. POSIX executes a
real SIGUSR1 handler; Windows uses the same direct same-thread callback. Tests
verify no success return, no normal completion count, preserved deadline,
replayable evidence and subsequent bounded kill/observed-exit progression.

## Verification recorded so far

Initial clean 5dd04f8 on permitted fallback101, Linux robot_sim Python3.12.13:

```text
python -B -m pytest -q -p no:cacheprovider \
 tests/unit/optics/test_guarded_wait_core.py \
 tests/unit/optics/test_guarded_wait_process.py \
 tests/unit/benchmarks/test_closed_worker_protocol.py \
 tests/unit/benchmarks/test_optical_closed_worker.py \
 tests/unit/optics/test_sacrificial_watchdog.py
```

132 passed in16.14s, plus cleanup-ledger check1 passed. Log at
/data3/ang.gao/c4-c2b-5dd04f8-cpu.log, SHA256
8ec425366a73cf85893239f630af33bb765d392652a7c16b59eb5e7b6f4bd1d3.
Independent first review separately ran133 tests and reproduced R1; its full
record is authoritative for those reviewer observations. These initial passes
are NOT proof that R1 was repaired.

Current repair on Windows Python3.14: core92 passed in1.57s; Ruff check/format
and diff check passed. Linux real-signal/process and C0/C1 regressions on the
new clean candidate remain to run; record them with the reviewed revision.
An earlier dirty draft caused seven C0 manifest tests to refuse the working
tree. That was not bypassed; clean committed worktrees are used for regression.

## Gates and cleanup

C0 A/K, C1 S/V/E, product worker/session/launcher/supervisor, native setup,
lifecycle, source-admission/request surface and frozen gate/runtime JSON are
unchanged. The shared collab README, review principles and CLAUDE.md are unchanged.
No installs, push, default changes, GPU preparation, sample or benchmark result.
The cleanup ledger classifies this shared safety infrastructure pending actual
selected-runtime adoption, removing it if no selected consumer remains.

C4c2 runtime integration still needs its separate complete design, including
workload/camera materialization, final warmup/readiness/guardian ordering,
native timeline, exact R0/R1 lease graph and real GPU failure custody. C4c3/c4,
C4d/B3, host payload/shipping, a3b, results/baselines and product activation
remain closed. This slice does not complete C4.
