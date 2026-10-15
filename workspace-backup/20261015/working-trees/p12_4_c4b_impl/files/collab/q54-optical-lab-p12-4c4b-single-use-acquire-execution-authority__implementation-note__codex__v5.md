# Q54 Optical Lab P12.4 C4b Single-Use Acquire Execution Authority

- Initiative: Q54 optical computation workflow / P12.4 backend selection
- Stage: corrective implementation note
- Author: Codex
- Version: v5
- Date: 2026-09-09
- Status: ready for short re-review
- Corrects commit: `06aab90a72c7ad6cbfc49cab462d537aab08e0e1`

## Outcome

The remaining real-lease remint P1 is closed. Static transaction construction
capability can only create dormant transactions. It can no longer bind a live
lease or choose the execution binding.

The C4 path now freezes and verifies three one-to-one relationships:

- controlled template and transaction to exact slot/frame/terminal-event plan;
- single-use prepared capability to exact transaction, exact active owner, and
  fixed C3 execution binding;
- live `C4ActiveOutputLease` to exactly one execution authority, transaction,
  and active owner.

## Frozen transaction identity

When the arena seals a target, the controlled template receives a one-way plan
identity containing:

- slot ID;
- transform frame index;
- terminal event token and exact event object identity;
- C4 plan-binding digest;
- ordered runtime-stage evidence digest.

The identity must exist before a controlled template can activate. Every
dormant transaction inherits that exact registered identity. The old
`_bind_arena_execution_lease()` API is deleted.

## Prepare and acquire authority split

Untimed `prepare_sample()` creates the exact active owner and asks the timeline
to mint one `_CudaArenaPreparedExecutionCapability`. The timeline derives the
fixed lease binding from the registered transaction identity plus the verified
C3 provenance. The binding is a fixed typed record; callers cannot supply an
empty or partial `(name, value)` collection.

Only after `NativeOpticalArena.acquire()` has successfully produced a real
`LEASED` generation does `CudaNativeOpticalArena.acquire()` consume that
prepared capability. Consumption atomically mints the live authority and
registers:

```text
lease identity -> exact authority -> exact transaction -> exact active owner
```

The static construction capability is not an argument to this operation.
Prepared capability reuse fails, transaction remint fails, and any lease that
already has a registered authority fails before stage enqueue.

## Exact owner and binding checks

Duck-typed active owners are rejected. Preparation requires the exact
`CudaActiveArenaSample` type, and the precise owner object is retained in the
prepared and live-authority registries. Every authorization rechecks this
object identity together with the transaction and lease identities.

The live binding is derived mechanically from the sealed transaction and C3
provenance and must match the active lease evidence for all six fields:

- slot ID;
- terminal event token;
- transform frame index;
- C3 one-shot binding digest;
- ordered stage-evidence digest;
- dataflow binding digest.

Consequently, a frame-0 lease cannot authorize a frame-1 transaction even when
the slot, shapes, stage graph, and terminal event object are otherwise shared.

## Failure and retirement

If authority mint or activation fails after control acquire but before enqueue,
the still-`LEASED` generation is cancelled and burned. No terminal event or C4
completion is recorded. Normal cancel and reclaim retire the live lease
registration; a retained poisoned generation keeps its registration and cannot
be rebound.

## Tests

New real-GPU negative tests cover:

- a valid frame-0 lease reminted with a frame-1 prepared transaction;
- unchanged lifecycle state and zero terminal-event records after rejection;
- duck-typed active-owner rejection;
- empty C3 binding digest rejection;
- removal of the old static-capability lease-binding API.

Verification on `192.168.80.101` in `robot_sim`:

- CUDA arena GPU tests: 22 passed;
- C4b/C2/C4a focused set: 79 passed;
- combined CUDA C2/C3/C4 GPU set: 56 passed;
- broad `tests/unit/optics` plus `tests/unit/benchmarks`: 446 passed;
- repository-wide pytest: 1822 passed, 115 skipped, with one pre-existing
  pytest fixture deprecation warning;
- Ruff lint/format and `git diff --check`: passed.

## Scope remains closed

This correction does not change gate activation, result capability, benchmark
rows, output payloads, host delivery, arena layout, C4d, or B3. a3b, baseline
generation, host-payload arenas, S, C4d, and B3 remain closed.

## Review request

Please rerun the real-lease remint reproduction and verify:

1. static construction capability cannot mint live execution authority;
2. one prepared capability binds only its exact registered transaction/owner;
3. one real C4 lease can have only one registered live authority;
4. the binding is derived from the sealed transaction identity and cannot be
   empty, partial, or caller-selected;
5. frame-0 lease plus frame-1 transaction fails before first enqueue and does
   not change completion accounting;
6. ordinary acquire, enqueue, finish, resolve, release, and reclaim remain
   unchanged.
