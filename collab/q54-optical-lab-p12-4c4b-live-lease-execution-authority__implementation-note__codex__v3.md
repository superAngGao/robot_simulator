# Q54 Optical Lab P12.4 C4b Live-Lease Execution Authority

- Initiative: Q54 optical computation workflow / P12.4 backend selection
- Stage: corrective implementation note
- Author: Codex
- Version: v3
- Date: 2026-09-09
- Status: ready for short re-review
- Corrects commit: `cc05de3445b9c3108dff806e1e8749b7d3e04f35`

## Outcome

The remaining C4b execution-authority P1 is closed. A template capability now
has exactly one power: constructing a dormant single-use transaction. It can
no longer authorize a stage or terminal-event submission.

Successful `CudaNativeOpticalArena.acquire()` activates a separate one-shot
authority bound to all of the following exact objects and identities:

- the dormant transaction;
- the active C4 lease;
- arena slot and generation;
- selected replay frame;
- terminal event;
- C3 binding, ordered-stage evidence, and dataflow digests;
- the owning active CUDA arena sample.

No such authority exists for a transaction obtained directly from the private
executable. Therefore the previously reproduced
`executable.activate() -> executable.authorize()` path is impossible:
`_CudaExecutableTarget` has no execution-authorize method, and passing its
static template capability to the transaction is rejected.

## Lifecycle-coupled authorization

The live authority does more than inspect a lease before entry. It owns the
linearization of transaction work with the C4 control state machine:

```text
live enqueue authorization
  -> validate transaction + lease + slot + generation
  -> C4 begin_enqueue()
  -> authorize exact registered stages
  -> verify every stage enqueued
  -> active sample becomes ENQUEUED

live finish authorization
  -> revalidate the same live lease in ENQUEUING
  -> authorize the unique transaction finish()
  -> record the terminal CUDA event
  -> C4 record_completion()
  -> active sample becomes COMPLETION_RECORDED
```

The authorization context revalidates the live lease both when authorization
begins and inside each controlled `enqueue_stage()`/`finish()` operation. If a
body or lifecycle commit fails after enqueue starts, the C4 arena is poisoned
instead of manufacturing completion or returning the slot to `FREE`.

This also makes direct access to the private live authority fail safe: a
successful direct transaction submission necessarily performs the matching C4
lifecycle transitions. It cannot record a terminal event while leaving
`terminal_event_record_count` at zero.

## Invalidated authorities

The authority is single-use and rejects execution:

- before `acquire()`;
- when given only the static template/construction capability;
- after pre-enqueue cancellation;
- after completion, device-result release, and reclaim;
- after submitted failure or arena poison;
- after transaction, lease evidence, slot, generation, target, or active-sample
  identity drift.

## Tests

The real-GPU negative coverage now includes:

- the original private-executable bypass shape;
- direct static-capability authorization;
- dormant prepared authority before acquire;
- the same live authority after cancel;
- the same live authority after reclaim;
- the same live authority after poison;
- direct use of the legitimate live authority, proving both GPU completion and
  the C4 completion counter advance together.

Verification on `192.168.80.101` in `robot_sim`:

- CUDA arena GPU tests: 18 passed;
- C4b/C2/C4a focused set: 75 passed;
- combined C2/C3/C4 set: 124 passed;
- repository-wide pytest: 1818 passed, 115 skipped, with one pre-existing
  pytest fixture deprecation warning;
- Ruff lint/format and `git diff --check`: passed.

## Scope remains closed

This correction does not change a2 provenance, O(1) acquire, arena layouts,
delivery, result schemas, gate activation, or benchmark artifacts. a3b,
baseline generation, host-payload arenas, S, C4d, and B3 remain closed.

## Review request

Please rerun the original bypass and verify:

1. a private executable can create only a dormant transaction;
2. no static executable/template capability grants execution;
3. execution requires the exact currently live C4 lease and generation;
4. even direct live-authority use cannot separate terminal-event recording from
   the C4 completion transition;
5. cancel, reclaim, and poison permanently invalidate that authority.
