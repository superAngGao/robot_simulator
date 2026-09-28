# C4c2b client publication and terminal write fence

Initiative: q54-optical-lab-p12-4c4c2b-supervised-waits
Stage: implementation-note
Author: codex
Version: v3
Date: 2026-09-29
Status: in_review
Related Files: implementation-note v2; code-review v3; wait_protocol.py;
wait_supervision.py; test_guarded_wait_core.py
Owner Summary: Repair the independent v3 review's client FAILED resurrection
and post-terminal socket output. Retain the approved supervisor single-commit
direction. New exact candidate still requires clean Linux tests and fresh review.

## Review facts

Candidate f74fd9bf548bbeda8d1accdbd105e2c6d4524ced received Request changes.
Full independent report is archived verbatim as __code-review__codex__v3.md,
SHA256763b54539771aa26acf61272e315b5eaff8b36d94d971bb605d25f9cb6de1fc1.
It independently ran Linux154 tests and Windows93, closed the original
supervisor findings, then reproduced two additional client/transport defects.
Passing counts did not close these counterexamples.

## R1: client state publication

The client now commits each `(phase, active ticket, next sequence)` tuple once
under an append-only integer key, using the same exact builtin insertion
principle as the supervisor. A nested failure occupies the successor first;
an interrupted success cannot overwrite it. Public ARM/COMPLETE are scoped
against nesting; repeated fail is idempotent and a failed exchange cannot start
another I/O wait. No state/ticket/sequence is independently assigned afterward.

Publication is the linearization point. Failure before ARMED publication means
no ticket returns. Failure after publication retires that ticket; neither it
nor any other ticket can complete, and no subsequent normal wait is accepted.
This is not a promise that an asynchronously retired immutable metadata object
can never be returned by a previously committed call. Such an object has no
live authority once its client's state has failed. State history owns metadata
only, not device/resource owners; the client's complete history is reclaimed
with the client and cannot reactivate a retired ticket.

Tests use the actual client/socket/core/transport and schedule nested wait,
explicit fail and close at both success-publication and finalizer windows for
ARM and COMPLETE. POSIX uses SIGUSR1, Windows direct same-thread scheduling.
They verify persistent FAILED, no pre-publication return, no reusable ticket
or subsequent normal request. Existing exact-ticket and clock tests remain.

## R2: irreversible transport write fence

A transport binds its exact private socket once to a fresh IDLE core before
any protocol operation. At terminal record commitment the core shuts down that
socket's write side using the OS endpoint. If shutdown raises, it closes that
socket rather than retaining a potentially writable endpoint. It never
releases process/resource custody or resets a deadline. No arbitrary callback
or backend handle is registered.

The transition/replay function remains pure. This terminal side effect occurs
after the terminal record's commit and before the initiating public call
returns. Thus this supersedes v2's broad statement that the live core performs
no I/O at all: it performs this bounded terminal control-socket fence, not
normal output, process kill, GPU work or resource release. The frozen wire,
state/event enums, timeout rules and schemas are unchanged.

Closing the write side preserves already-enqueued prefixes but blocks future
send syscalls, including an outer Python frame suspended immediately before
send. It does not rely on another Python check-then-send condition. Tests inject
both direct parent loss and nested service at the actual send line for ARMED
and DISARMED, require zero peer-visible bytes followed by EOF, and retain the
original deadline/count/trace. A one-byte delegate around the real socket send
tests the already-written-prefix case separately; natural OS backpressure
remains covered by the fixed process harness. No new device-safety assertion
is inferred from socket closure.

## Current verification and exclusions

Local Python3.14 core112 passed in2.09s; report archive SHA matches its source.
New clean Linux regression and independent review are pending. Prior f74fd9b
author Linux154/16.61s log remains historical, not proof for these changes.
Ruff/diff checks will run before candidate freeze. No C0/C1 protocol/adoption,
worker/setup/lifecycle/gate/source-policy or shared collaboration-policy change.
No GPU preparation, result/baseline or performance measurement. C4c2 integration
still requires separate complete design; C4c3/c4/d/B3, host/shipping, a3b and all
product activation remain closed. This dependency slice does not complete C4.
