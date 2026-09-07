# Q54 Optical Lab P12.4c4a Arena Control Contract (Codex v2)

Author: Codex

Date: 2026-09-07

Status: P1 review fixes submitted for Claude re-review; C4b/C4c/C4d, B3,
P12.4a3b result activation, benchmark samples, and baseline access remain
closed

Base implementation: `a4799186966155fe0c0cb6073d704a5c55043a29`

This note is the authoritative amendment to the C4a v1 implementation note.

## P1: prepared plans are speculative, not reservations

`prepare_plan()` no longer rejects or reserves a shared candidate generation.
It accepts a non-negative `planned_acquisition_offset`, from which it derives
the exact future round-robin slot and generation without changing arena state.
This permits all samples in a depth-one or depth-two batch to finish validation,
hashing, and object construction before the measured interval.

Multiple plans may target the same `(slot_id, expected_generation)`. The first
matching `acquire()` performs the sole state transition and wins atomically.
Every competitor then fails as stale before state mutation. Prepared plans are
held by a `WeakKeyDictionary`; dropping a plan drops the registry entry and its
dormant lease graph, so abandoned preparation cannot lock a generation.

The depth-two host-payload *control plane* may now be constructed for planning
and lifecycle tests, but both prepared and active authorities remain explicitly
`result_capable=False`. There is still no B3 payload stage, runner acceptance,
or result activation.

## P1: lifecycle evidence is state-machine proof

The lifecycle contract now freezes:

- an explicit operation enum and shutdown-disposition enum;
- the arena ring depth;
- an exact final snapshot for every slot, including generation, state,
  completion flags, payload readiness, and all hold counts;
- full canonical replay beginning with exactly one `arena_opened` transition.

Deserialization replays each transition and enforces legal arena/slot source
and target states, FIFO acquire/enqueue/resolve/reclaim ordering, generation
increments, event-record uniqueness, hold changes, cancel semantics, poison
fan-out, drain, close, and final state. It derives and compares:

- final arena and slot state;
- next round-robin slot;
- backpressure, completion, resolution, and reclaim counters;
- final drain/reclaim flags;
- external hold count, poison-time hold count, and poisoned-slot count.

A recomputed public SHA-256 cannot make an invented operation, illegal
transition, fabricated summary, unknown disposition, or one-way terminal-state
claim valid. Shutdown disposition, waiting state, and release authorization are
checked bidirectionally.

The versioned C4a operation enum contains only transitions C4a can truthfully
produce. C4d must extend and review the enum/replay semantics when it implements
emergency synchronization and external-hold discharge; it cannot bypass the
replayer by filling summary fields directly.

Because both P1 fixes change serialized semantics, the amended contract uses
new identities: gate v12, result v12, row-semantics v7, native-output-arena
contract v2, and prepared-plan/active-binding/lifecycle schema v2.

## Verification

Executed in the server `robot_sim` environment:

```text
ruff check (all changed Python files)                         passed
C4a focused tests                                             23 passed
all tests/unit/optics + tests/unit/benchmarks                422 passed
git diff --check                                              passed
```

Required negative coverage includes:

- dropped plan does not block later preparation;
- same-generation competing plans have one winner and one non-mutating stale
  failure;
- both depth-two slot plans are built before either measured acquire;
- invented transition with recomputed hash is rejected;
- unknown/mismatched shutdown disposition is rejected;
- arbitrary summary counters and truncated/rehashed traces are rejected.

## Review request

Please review the two P1 fixes against C4a only. All C4b/B3/result/baseline
surfaces remain closed.
