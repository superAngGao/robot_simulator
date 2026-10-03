# C2d1: retain exception custody before inner abort termination

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: implementation-note
Author: codex
Version: v4
Date: 2026-10-03
Status: in_review
Related Files: approved custody/repair designs v2; code-review v3; implementation-note v3
Owner Summary: Repair F5's real inner-abort boundary without reopening normal
work. Fresh independent review is required; CPU success does not open C4 gates.

Base: `6f87d23a5ce2c1acefc8795caf79709fa7ea33e2`. The complete v3 independent
Request changes report is archived verbatim, SHA256
`add59aab2abab30d7da299c123b58fcba174bd224797225ed1a1da96d5c4d128`.
It closed F1–F4 but found F5 despite independently passing730 tests. Previous
author730-pass/process evidence is archived separately as test-evidencev3.

## Bounded repair

On the exact owner thread, before termination, a failed/compromised controller
can still adopt a BaseException through native_setup._abort's existing
setup_exception_graph/setup_exception retention operation. It first retains
that exception and failure through existing terminal custody, then calls the
sole inherited host adoption implementation. It grants no leaf/normal-work,
COMPLETE, sync, release, retry or readiness authority. Non-exception objects or
ordinary role/operation labels still reject after failure. The labels describe
retention only and cannot mint an executable capability. Reentrant/foreign-
thread terminal calls still fail; no sealed guardian is reopened.

Unchanged native_setup._abort can therefore immediately reach the overridden
terminate, freeze/quarantine and return NativeWorkerPreparationAborted carrying
exact terminal evidence before outer context unwinding. For an already sealed
guardian, its existing skip-adoption path plus retained original exception/
traceback remains sufficient; the guardian stays sealed until normal terminal
freeze. No C0 source, reducer schema, product gate or shared policy is changed.
Only the changed lifecycle source digest is repinned.

## Tests and observed proof before remote freeze

New unit matrix invokes the actual _abort after foreign/unknown descriptor,
foreign/unknown returned owner, explicit unknown, invalid permit and sealed
invalid permit, each with caught versus propagated abort. Inside the context
it checks exact abort type/evidence, original exception/owners, terminal guardian
state, zero sync/COMPLETE/release and one termination/custody. Outer unwinding
cannot add a second transition. Three non-exception/wrong-label adoption
negatives prove ordinary work is still closed. A direct F3 test now ends the
explicit-unknown body without another public operation.

Three fixed POSIX cases add foreign/unknown/invalid-permit inner abort to the
real supervisor harness. Their pre-unwind snapshot must equal the final worker
record; actual returncode74, no destructor/atexit/leaf/sync and observed reap
are required. These are CPU fixtures, not GPU safety proof.

Local Python3.14 lifecycle/evidence/wait-core297 passed5.75s; with source-policy
462 passed. Ruff check/format passes. An initial combined shell included an
empty Get-Process lookup after pytest and returned1 despite462 passes; pytest
was rerun separately for an unambiguous actual exit. No test was hung.

Unmodified external c2d1-code-v3-independent-repros.py: actual exit0,
runner36532(parent36624), child20548(parent36532), both absent afterwards.
All four F5 records now say returned_abort=true and terminal evidence present
BEFORE context exit. Its F3 and16 F4 cases still pass. This original repro
prints F5 facts; the new unit matrix supplies assertions, not merely exit0.

Remote broad regression and fresh independent review will execute the new
frozen candidate. No results are preclaimed. Still closed: integrated source-
frozen guarded runtime/module custody, real Warp/Torch and multi-GPU faults,
complete graph/barrier/warmup/readiness, C2e/f arenas, product preparation,
a3b/results/baseline, R2/host/S, later C4/B3/shipping and C4 completion.
