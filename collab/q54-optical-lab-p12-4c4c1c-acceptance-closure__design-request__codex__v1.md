# C4c1c retained acceptance: fixed replay schedule and leaf audit

Initiative: q54-optical-lab-p12-4c4c1c-acceptance-closure
Stage: design-request
Author: codex
Version: v1
Date: 2026-09-28
Status: in_review
Related Files: C4c1 ABI design v1 sections 14.3/14.4; v2/v3;
GPU-validation independent review v4; tests/support/sacrificial_lifecycle.py
Owner Summary: Clarify the test-only two-pass lifetime before implementing
review findings F3/F4/F5. No requirement is waived and no product gate changes.

Code base: `55365c528cbdb5208804be9e46220e4bdf6a7335`.
The independent v4 report is preserved verbatim alongside this request.
Author's clean-101 full selection passed 98 tests; the independent reviewer
passed 181 CPU/POSIX and 165 selected GPU/regression tests. These results do
not cover the three missing acceptance properties and do not approve C4c1c.

## 1. Scope and retained rules

Implement only the missing recorded-bundle reuse, runtime leaf contamination
and execution-context checks. All bundle execution remains inside the fixed,
supervised, one-shot C1c child. C0's production A/K protocol, source allowlist,
GPU-preparation gate, C4 slot/lease authority and result activation stay intact.
No arbitrary callback, Python object, module name or native handle is accepted
through the test request. This is supported-path verification in trusted code,
not a hostile-Python or native-memory security boundary.

The existing one-pass fault matrix remains. New two-pass validation uses one
resource graph, one setup-quiescence proof, one guardian seal, one irreversible
submission history and one terminal V drain. It does not reopen a guardian,
clear poison/quarantine, create a reusable session or add an intermediate
completion event. S/V/E semantics and ACK-before-blocking-call stay unchanged.

## 2. Fixed two-pass schedule

Before setup quiescence, allocate and adopt:

- the original scene/input/output/command graph;
- an independent GPU snapshot destination for every physical output array;
- both sets of host inspection storage and CPU views; and
- all audit/context observer owners required by the fixed driver.

The camera fixture includes origins/directions among outputs. Resident rays
remain inputs, not writable output sentinels; zero-ray reuse exercises the
scalar resets and has no nonempty ray input to overwrite.

Seal freezes `render_command_count` (existing 2/4/5), `repetition_count` (1 or
2), the exact bundle/command tuple and a fixed ordered auxiliary schedule
derived from physical output names. One pass has no auxiliary work. Two passes
execute exactly:

```text
setup fills sentinel A -> S ACK/barrier/disarm -> one seal
-> render commands 0..N-1
-> same-stream D2D copies to retained first-pass snapshot arrays
-> same-stream fills of original output arrays with distinct sentinel B
-> the same render commands 0..N-1 (same objects and parameter/storage binding)
-> V ACK -> target drain -> retained host readbacks -> target drain
-> inspect both -> release whole graph -> exit
```

Copies and fills are test-only verification work, not native benchmark timing.
No host inspection or host/device synchronization occurs between passes.
Fills use the exact execution stream with scoped restoration and no implicit
cross-stream wait. Copies name that stream explicitly. First-pass snapshots
cannot be overwritten by second-pass commands. Host destinations are never
allocated during the final drain.

The guardian's state sequence remains SETUP, SETUP_QUIESCENT, SEALED,
SUBMITTING, VALIDATED, INSPECTED, RELEASED. In SUBMITTING, its frozen schedule
distinguishes render leaves from auxiliary copy/fill leaves. Before **each**
leaf, the appropriate attempted counter and active-leaf position commit;
quiescence becomes false. Normal-return counters advance only after return.
No next leaf or V drain is allowed with an active/missing/out-of-order leaf.
The second render pass cannot start before every auxiliary leaf returned.
Duplicate positions, extra passes, skipped copies/fills, wrong role/order and
early drain fail before entering their operation. A public snapshot cannot
authorize progress. No increment or reuse of a C4 generation is claimed.

Any auxiliary or second-pass exception is post-submit poison even if that
particular API may have failed before enqueue. The original full graph stays
held; the existing exact-target E-or-quarantine rules apply. Every V failure
still goes directly to quarantine under V without a second sync. The two-pass
schedule never resets the possible-submission latch or releases a subset.

## 3. Real recorded-leaf audit

Prepare and install fixed runtime hooks outside the guarded interval. Reuse
the existing forbidden API inventory where applicable, with explicitly
test-only additions for copies, Warp array materialization, module loading,
event creation/record/wait and ordinary non-recorded launch. Event work is
forbidden here because C1 render bundles have no completion event; this does
not change the later C4 terminal/diagnostic event policy.

Within the interval only the exact frozen `wp.Launch.launch` operation and its
expected native kernel leaf may execute. Hook observations come from actual
wrapped entry calls; the fixed driver cannot satisfy them by filling a flag.
Record actual native launch count and identity, and reject forbidden calls
before their original body. Validate hook installation/restoration by callable
identity. Restore in `finally` on normal and exceptional exit before any
legitimate setup/auxiliary/drain/inspection operation.

Fixed negative requests inject one enumerated real operation inside a launch
wrapper: allocation, copy/upload, NumPy/host materialization, interop/DLPack,
module loading, event operation or synchronization. No callback is transported.
Tests require the specific bottom-level hook observation, zero native render
launches for that rejected entry, no terminal event/completion/result, safe
post-latch disposition and no early finalizer. They must not merely raise a
synthetic contamination exception without attempting the runtime API.

This audit is correctness instrumentation, not an eligible benchmark sample,
performance measurement or claim to intercept arbitrary Python/C++ heap use.

## 4. Device and current-stream identity without a foreign setup touch

The fixed Warp 1.12 device registry initially has `_stream=None` on untouched
devices (independently inspected on 101). Reading `get_stream` can lazily
initialize that state; a test must not conceal that as CPU-only metadata.

During allowed target-only setup, allocate/adopt a nondefault current-stream
fixture on the target device, distinct from the explicit execution stream.
Set it as target-current without adding a cross-stream wait, then freeze its
exact object and native handle. Capture ambient device identity and its
already-existing current-stream state by CPU metadata only: either the exact
existing owner/handle or the explicit UNINITIALIZED sentinel. Retain every
existing owner. Do not create or query a CUDA stream on an untouched foreign
device merely to obtain a comparison value.

Immediately before and in `finally` after every recorded leaf, compare:

- ambient device identity;
- target current-stream object/handle (the distinct nondefault fixture);
- ambient device's existing stream object/handle, or unchanged UNINITIALIZED;
- explicit execution stream identity and native handle.

Two mandatory real-device rows cover target 0 / ambient 0 with a nondefault
current stream, and target 1 / ambient 0 with a nondefault target-1 stream and
the exact ambient-0 preexisting state. This does not claim coverage of a
nondefault foreign GPU-0 stream when that fresh child's state is UNINITIALIZED.
The latter is an observed exact state to preserve, not permission to ignore
ambient stream drift. Any unsolicited initialization or replacement fails.

Normal and injected-exception paths perform the same comparisons before drain
when safe. Context mismatch is unknown-scope quarantine/hard exit, not a
target-only sync authorizing release. No foreign-device setup or multi-device
recovery exception is introduced. A deliberate metadata/current-stream drift
negative test must show the observer fails, rather than only testing assertions
against an unchanged runtime. Such a negative is contained in the disposable
child and never changes another process's device/stream state.

## 5. Acceptance and cleanup

CPU tests cover the complete one/two-pass schedule, duplicate/skip/order errors,
pre-call versus return accounting, auxiliary and second-pass failures, retained
owners, forbidden seal/reopen, early drain/release and unchanged S/V/E behavior.

Real GPU tests cover all four fixed shade variants with the same bundle twice,
distinct full sentinels, both snapshots against CPU truth and each other,
zero-ray repeated resets, audit positive/negative operations, and actual
single/two-device current-stream restoration on success and failure. Preserve
the existing setup/readback/partial-submit/transport/watchdog fault matrix and
run it again on the revised clean candidate. No skip counts as real-device proof.

Archive independent reports unchanged. Correct stale progress descriptions to
the actual scoped acceptance status; do not alter shared review policy. All new
helpers remain selection/validation-only and follow the existing losing-lane
cleanup ledger, not a new permanent product protocol.

Design approval authorizes this bounded C4c1c repair only. A fresh code review
must approve the implemented scope and runtime evidence before C4c2 starts.
C4c2-c4, C4d/B3, product GPU preparation, a3b, baseline/result, host/shipping and
backend selection remain closed.
