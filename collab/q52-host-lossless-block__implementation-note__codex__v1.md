Initiative: q52-publish-pipeline
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-27
Status: ready-for-review
Related Files: OPEN_QUESTIONS.md#Q52, physics/publish.py, physics/gpu_engine.py, tests/unit/physics/test_publish.py
Owner Summary: Implements host-only `PublishPolicy(on_ring_full="block")` for lossless published-ring backpressure. Host lossless blockers wait on a ring condition variable and are released when borrow/snapshot ack advances. Device lossless blockers are rejected with a clear `NotImplementedError` because they require stream/event reclaim semantics.

---

## 1. Scope

This pass implements only host lossless blocking.

Implemented:

- `PublishedRing` owns a host condition variable.
- `PublishedRing.acquire(...)` waits when the target slot is pinned by host
  lossless consumers and policy is `on_ring_full="block"`.
- `PublishedRing.acknowledge_consumer(...)` advances `acked_frame_id` and wakes
  blocked publishers.
- `GpuEngine` lossless borrow/snapshot ack paths call
  `PublishedRing.acknowledge_consumer(...)`.
- device lossless blockers under `on_ring_full="block"` raise
  `NotImplementedError`.

Not implemented:

- device consumer stream/event reclaim
- host staging queue capacity or timeout policy
- cancellation/shutdown semantics for long-running export queues

## 2. Blocking Rule

The host rule is unchanged:

```text
slot reclaimable iff
  slot.frame_id <= min(acked_frame_id for enabled lossless consumers)
```

When the target slot is not reclaimable:

```text
on_ring_full="skip":
  return None

on_ring_full="raise":
  raise RuntimeError with blocker ids

on_ring_full="block":
  if all blockers are host consumers:
    wait until ack/unregister/policy change wakes the ring
  if any blocker is device consumer:
    raise NotImplementedError
```

## 3. Device Consumer Boundary

`on_ring_full="block"` is intentionally host-only in this pass.

Device consumers should not be handled by a CPU condition wait. They need a
separate event/fence design, likely:

```text
physics stream records publish_event[slot]
sensor/render stream waits publish_event[slot]
sensor/render stream records done_event[slot]
physics stream waits done_event[slot] before reuse
```

That is a separate device-consumer phase.

## 4. Threading Constraint

Host-only `block` assumes the blocking publish/control path is not the only
execution context that can advance the blocking consumer's ack.

Valid release sources include:

- a host snapshot staging future callback
- another host consumer thread
- an external caller unregistering the consumer
- an external caller changing the publish policy

Invalid usage:

```text
single host thread enters PublishedRing.acquire(...)
same host thread is also responsible for completing the snapshot/borrow ack
```

That shape can self-deadlock. Single-threaded loops that cannot rely on a
separate ack source should use `on_ring_full="raise"` or `on_ring_full="skip"`
instead of `block`.

## 5. Verification

Command:

```bash
PYTHONPATH=. pytest \
  tests/unit/physics/test_publish.py \
  tests/unit/physics/test_cpu_publish_runtime.py \
  -q
```

Result:

```text
33 passed
```
