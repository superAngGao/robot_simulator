# Q54 Optical Lab P12.4c4a Arena Control Contract (Codex v3)

Author: Codex

Date: 2026-09-07

Status: second P1/P2 review fixes submitted for Claude re-review; C4b/C4c/C4d,
B3, P12.4a3b result activation, benchmark samples, and baseline access remain
closed

Base fixes: `a4799186966155fe0c0cb6073d704a5c55043a29` and
`36aac2422e3300548f0974afb25e5e8ec17fc419`

This note supersedes the C4a v1/v2 implementation notes where their behavior
differs.

## P1: cross-ring generation projection

Future generation increments are now based only on complete cycles in the
relative `planned_acquisition_offset`:

```text
slot_id = (cursor + offset) % ring_depth
generation = slot_generations[slot_id] + 1 + offset // ring_depth
```

The cursor no longer contributes a second wrap. The calculation lives in the
CPU-safe pure `plan_native_optical_acquisition()` helper used directly by
`prepare_plan()`. Tests cover a non-zero depth-two cursor with offsets 0, 1,
and 2, including the exact cross-ring boundary. A depth-one three-generation
batch is prepared entirely in advance, then acquired and cancelled in order;
each cancellation burns exactly one generation without making later plans
stale.

## P1: B3 host-payload sealing gate restored

Production `NativeOpticalArena` construction again rejects `rgb`, `full`, and
`rgb8` payload arenas until B3 and joint activation review. The frozen gate
again states:

```text
host_payload_arena_seal = blocked_until_B3_and_joint_activation_review
```

Depth-two arithmetic is tested through the same pure planning helper used by
production, not by constructing a placeholder host-payload arena. No separate
control-only arena type or unreviewed storage contract is introduced.

## P2: exact non-idempotent transition predecessors

Lifecycle replay now requires `slot_reclaimable` to have the exact source
state `PAYLOAD_READY`; an existing `RECLAIMABLE` slot cannot repeat the
transition. A parameterized negative matrix duplicates and rehashes every
non-idempotent C4a operation:

```text
arena_opened
acquired
cancelled_generation_burned
enqueue_started
completion_recorded
completion_resolved
device_result_hold_released
slot_reclaimable
slot_reclaimed
drain_started
final_drain_complete
arena_closed
arena_poisoned
```

Every duplicate is rejected by canonical replay. `acquire_backpressure` is
intentionally excluded because it is the sole non-mutating repeatable
observation.

## Verification

Executed in the server `robot_sim` environment:

```text
ruff check (all changed Python files)                         passed
C4a focused tests                                             39 passed
all tests/unit/optics + tests/unit/benchmarks                438 passed
git diff --check                                              passed
```

## Review request

Please review the cross-ring planner, restored host-payload seal, and complete
duplicate-transition negative matrix. C4b and all result/baseline surfaces
remain closed.
