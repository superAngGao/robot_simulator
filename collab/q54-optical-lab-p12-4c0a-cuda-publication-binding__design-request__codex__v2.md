# Q54 Optical Lab P12.4c0a CUDA Publication Binding

Author: Codex

Date: 2026-08-13

Status: P1 poisoned-shutdown resource-safety revision for Claude re-review; no
C0b implementation is authorized by this commit

Approved parent contracts:

- P12.4 backend gate v5;
- P12.4a3 dependency order v3;
- frozen publication contract
  `q54.p12.4.publication-cuda-binding.v1`.

## Outcome

C0 adds one physics-owned, same-device CUDA consumer binding. It closes the
existing Warp-on-CUDA physics publication lifecycle for a future Torch/CUDA
optical consumer without making the optical renderer depend on Warp objects and
without putting Torch events into `PublishedRing`.

This design deliberately does not enable the CUDA optical product path. C0b
ends with a tested publication primitive. C2-C4/B3 and the later joint
ownership review remain required before any dynamic CUDA product path exists.

The P1 revision also freezes the only two legal resource dispositions after a
poisoned CUDA timeline: a shutdown-only target-device synchronization must prove
all device work drained before ordinary resource release; if that synchronization
does not succeed, the entire engine publication resource graph is quarantined
until CUDA context/process teardown. Logical unregister is never device-safety
evidence.

## Current code and the gap

`GpuEngine` already publishes Warp CUDA arrays into a three-slot
`PublishedRing`. A slot owns its publish event. A legacy device consumer waits
that event on a caller-provided Warp stream, records a done event, and advances
`ConsumerState.device_completed_frame_id`.

That API is insufficient for the approved CUDA lane:

- each borrow and completion may currently receive a different stream;
- `mark_device_consumer_complete()` uses `max()` and cannot reject duplicate or
  out-of-order completion;
- there is no active-borrow, binding, stream-lifetime, close, or poison state;
- only the latest done event is retained, with no explicit evidence that the
  producer established a reuse dependency before an older event was released;
- replacing or unregistering a consumer and resetting the ring do not account
  for an external CUDA binding.

C0 preserves the legacy Warp device-consumer methods for the Warp product lane,
but a consumer ID may use either that legacy API or the new binding, never both.

## Public API

The public API is physics-owned and accepts both the native handle and its
lifetime owner:

```python
binding = engine.bind_cuda_device_consumer(
    consumer_id,
    device_ordinal=torch_stream.device.index,
    cuda_stream_handle=torch_stream.cuda_stream,
    cuda_stream_owner=torch_stream,
)

with binding.borrow(frame_id) as lease:
    frame = lease.frame
    # The later physics-to-optical adapter may create the approved same-device
    # zero-copy transform views and enqueue the final physics-buffer reader.
    lease.complete(last_reader_operation="cuda_scene_update")

binding.close()
engine.unregister_consumer(consumer_id)
```

The extra `cuda_stream_owner` is required. A bare integer cannot prove that a
destroyed CUDA stream has not been reused. The binding holds a strong reference
to the owner until close and validates on bind, borrow, complete, and close that:

- `owner.cuda_stream` still equals the frozen integer handle;
- `owner.device` still identifies the frozen device ordinal;
- that ordinal equals the `GpuEngine` CUDA device.

This is a non-owning CUDA-stream contract: Torch still creates and owns the
stream, while the strong reference prevents its ordinary destruction during the
binding lifetime. Manual native destruction or mutation underneath the owner
is unsupported; the next CUDA/identity failure poisons the binding and fails
closed.

The final implementation may use a small structural protocol rather than
importing Torch. Production requires an owner object exposing `cuda_stream` and
`device`; test fakes may implement the same protocol. The physics module does
not import the CUDA optical executor or benchmark package.

## Runtime mapping

Inside physics publication only, C0 constructs:

```python
wp.Stream(engine_device, cuda_stream=cuda_stream_handle)
```

The wrapper is non-owning. Every Warp stream/event operation runs inside
`wp.ScopedDevice(engine_device)`. The scope restores the prior CUDA context and
Warp default device on success and failure. GPU tests additionally assert the
caller's `torch.cuda.current_device()` before and after bind, borrow, complete,
close, and injected failures.

The hot path contains no event query, host polling, `torch.cuda.synchronize`,
`wp.synchronize`, `wp.synchronize_event`, D2D fallback, or host staging.

## Ownership

| Resource | Owner | Lifetime |
| --- | --- | --- |
| published slot arrays | physics publication ring | engine/ring lifetime |
| publish event | published slot metadata | until the slot is republished |
| external CUDA stream | CUDA optical session | binding holds a non-owning wrapper and strong owner reference until close |
| consumer done event | physics publication binding | from successful record until a covering producer wait is enqueued |
| logical consumer state | `PublishedRing` | registered consumer lifetime |
| optical scene/output/readback buffers | later CUDA optical lane | explicitly outside the physics-input lease |
| poisoned engine publication graph | process-level quarantine owner | until CUDA context/process teardown, unless emergency synchronization succeeded |

No Torch event enters the ring. `ConsumerState.device_done_event` continues to
refer only to a Warp-owned event. No optical caller may mutate consumer, slot,
or ring internals.

## State machines

### Binding

```text
OPEN --CUDA/event failure--> POISONED --abort_poisoned--> ABORTING
  |                                                    /        \
  +--close, with zero active leases--> CLOSED      ABORTED_SYNCED QUARANTINED
```

- one consumer ID has at most one binding;
- binding a disabled, non-lossless, host, or snapshot consumer is rejected;
- a second bind, including an identical rebind, is rejected;
- `register_consumer()` may not replace a bound consumer;
- legacy `borrow_device_frame()` / `complete_device_consumer()` reject a bound
  consumer;
- `unregister_consumer()` rejects an open, poisoned, or aborting binding. It is
  allowed only after normal close, `ABORTED_SYNCED`, or after the process-level
  quarantine owner has atomically accepted the entire engine resource graph;
- unregister removes only logical consumer accounting. It does not release a
  stream, event, slot, or engine allocation and is never evidence that device
  work stopped;
- ring reset rejects any open/poisoned binding and any poisoned slot. A
  poisoned run is not recoverable through `reset()` in C0.

### Lease

```text
BORROWED --complete at queue head--> COMPLETED
BORROWED --ordinary exception at queue head--> COMPLETED, then original exception propagates
BORROWED --CUDA/event failure--> INVALIDATED and binding POISONED
```

The binding keeps a FIFO of active leases. Published frame IDs may contain gaps,
but borrows must be strictly increasing and unique. Multiple frames may be
borrowed, while completion must match FIFO borrow order. Therefore a direct
attempt to complete a later lease first is observably rejected without changing
logical completion.

`lease.complete()` is explicit and required on a normal context exit. A normal
exit that forgot completion first records a safe done event when it is the FIFO
head, advances logical completion, and then raises a completion-required error.
This exposes the programming error without leaking or falsely pinning the slot.
An ordinary exception in the body receives the same safe completion but the
original exception is preserved.

If exception unwinding reaches a non-head lease, ordered completion cannot be
honored. The binding and every active slot are poisoned and the run aborts; the
implementation does not silently skip the earlier lease.

Duplicate completion, completion of a fabricated/non-active lease, and
out-of-order completion are rejected. These lifecycle errors do not alter
consumer completion. CUDA wait, stream access, event allocation, event record,
or ring-commit failures use the poison path below.

## Borrow transaction

`binding.borrow(frame_id)` performs the following ordered transaction:

1. require `OPEN` and validate owner, stream, and device identity;
2. atomically obtain the ready, non-poisoned published frame through a public
   ring/engine operation;
3. require a strictly newer, not-already-active frame ID;
4. under `wp.ScopedDevice`, enqueue
   `external_stream.wait_event(frame.ready_event)`;
5. only after the wait enqueue succeeds, register the active lease and advance
   `latest_seen_frame_id`;
6. record lane-visible evidence that the publish wait was enqueued. This is
   timeline evidence, not a claim that the event has physically completed.

A wait failure poisons the consumer binding and the attempted slot. It does not
advance either logical completion field.

## Completion transaction

`lease.complete(last_reader_operation=...)` performs:

1. require the lease to be active and at the FIFO head;
2. revalidate owner, frozen stream handle, and device ordinal;
3. create a physics-publication-owned, non-timing Warp event;
4. under `wp.ScopedDevice`, record it on the bound external stream after all
   previously enqueued physics-buffer readers;
5. append an immutable completion record retaining that event;
6. atomically advance the ring's logical device completion for the exact frame;
7. mark the lease completed and remove it from the active FIFO.

The ring commit is exact and monotonic for the binding's FIFO; it does not use
an unchecked `max()` to accept a duplicate or skipped completion. A later event
on the one stable stream covers earlier stream work, but earlier event objects
remain retained until a producer wait on that event or a later covering event
has actually been enqueued.

## Reuse and event retention

When the producer selects a slot for reuse, physics publication asks each bound
consumer for the earliest retained completion record that covers that slot's
old frame. It then:

1. validates that logical completion covers the old frame;
2. enqueues a wait for the covering done event on the physics producer stream;
3. marks all covered completion records as having a reuse dependency;
4. only then releases superseded event references and overwrites the slot.

The existing legacy-consumer path continues to wait its current done event.
The new path supplies explicit retention/reuse evidence. Logical reclaim means
only that the producer may select the slot and enqueue this wait. C0 never uses
an event query or synchronization to report physical completion.

## Close, removal, and final drain

Normal `binding.close()`:

- rejects any active lease;
- revalidates the stream owner;
- enqueues, on the physics stream, a wait for the newest retained done event,
  which covers all prior work on the stable external stream;
- marks all retained completion records reuse-safe and releases them;
- releases the non-owning Warp wrapper and strong stream-owner reference;
- changes the binding to `CLOSED` without synchronizing the host.

This device-side final drain makes subsequent consumer removal safe even when a
previously borrowed slot has not yet cycled through normal reuse.
`unregister_consumer()` is a separate explicit operation after close. No binding
implicitly removes a logical consumer.

The product/session owner must close the binding before intentionally releasing
the Torch stream. Because C0 introduces an externally owned resource lifetime,
C0b also adds an idempotent explicit `GpuEngine.close()` with deliberately narrow
responsibility: close/drain publication bindings, shut down the existing host
snapshot executor, and mark the engine closed. It does not individually free
physics arrays. Normal close rejects active leases. A poisoned close delegates
to the shutdown-only `abort_poisoned()` transaction below before any owner,
event, slot, or engine allocation reference may be released.

## Poison and shutdown

CUDA wait/record/stream failure makes the external timeline untrustworthy. C0
uses an abort-wide policy for that binding:

- set the binding and `ConsumerState` to poisoned;
- poison the attempted slot and every slot held by an active lease;
- retain all events and payloads already owned by publication;
- do not advance logical completion for the failed lease;
- reject future borrow, completion, publish, normal close, and reset;
- cause later producer publication/acquire to abort even if it targets a
  different slot.

Other consumers retain their own state, but the current engine run is
abort-only because one lossless dependency is no longer provable. C0 does not
attempt in-place CUDA recovery.

`binding.abort_poisoned()` is shutdown-only and is the only operation allowed to
move a poisoned binding toward resource disposal. It never records a synthetic
done event, advances logical completion, establishes a producer reuse wait, or
makes the run reusable. It executes exactly once under the binding state lock:

1. atomically change `POISONED` to `ABORTING` and retain every active lease,
   external-stream wrapper/owner, publish/done event, ring slot payload, and
   engine allocation;
2. under the target-device guard, perform an emergency target-device
   synchronization;
3. record one of the two immutable dispositions below before any logical
   unregister is allowed.

The target-device synchronization is an explicit exception to the no-host-sync
hot-path rule. It is legal only in poisoned shutdown, is counted in lifecycle
evidence, and must never appear in borrow, completion, normal reuse, normal
close, S timing, or benchmark result generation.

### Disposition A: `emergency_sync_succeeded`

Successful device synchronization proves all previously enqueued work on the
target device, including an un-fenced reader, has completed. Only then may C0:

- invalidate active Python lease objects;
- release retained events and the external-stream wrapper/owner reference;
- transition the binding to `ABORTED_SYNCED`;
- permit logical unregister and ordinary `GpuEngine.close()` teardown.

Slots and the run remain semantically poisoned and cannot be reset or reused;
the synchronization establishes memory-release safety, not a valid consumer
completion or successful frame transaction.

### Disposition B: `quarantined_until_process_exit`

If emergency synchronization raises, returns an error, or cannot be started,
C0 must not release any object that may own or reach device memory. Before
returning control, a process-level quarantine owner atomically takes strong
ownership of the complete `GpuEngine` publication resource graph, including:

- the engine and `PublishedRing`;
- all published slot payload arrays and slot metadata;
- all publish and retained consumer done events;
- the external-stream wrapper and stream-owner object;
- active lease records and consumer/binding state;
- all other engine CUDA allocations, because kernels may hold pointers beyond
  the published transform arrays.

The quarantine registry is append-only, has no release/drain API, is not
cleared by `GpuEngine.close()` or normal application cleanup, and performs no
proactive CUDA cleanup from object finalizers. Dropping its final references is
allowed only as part of terminating the process. A quarantined device makes the
current process/run terminal: no new engine may be created on that device and
the session owner must terminate the worker/process so CUDA context/process
teardown reclaims the resources as a unit. If the surrounding application
cannot guarantee process-level isolation, a failed emergency synchronization
is a fatal process condition, not a recoverable Python exception.

After quarantine acceptance, logical unregister may be requested only to remove
control-plane accounting from callers that are already unwinding. The
quarantine capsule still owns the original consumer/binding/ring objects, and
unregister must not mutate or release that retained resource graph. Neither an
unregister return nor `GpuEngine.close()` return may be reported as GPU drain,
physical completion, or memory release.

The first `abort_poisoned()` call returns an immutable disposition record.
Repeated calls after `ABORTED_SYNCED` or `QUARANTINED` are idempotent and return
the same record without a second synchronization. Calling it before poison is
rejected. Re-entrant calls while `ABORTING` fail closed and cannot race resource
release.

Explicit `GpuEngine.close()` drives every poisoned binding through this
transaction before normal teardown. `GpuEngine.__del__` never performs emergency
synchronization: host synchronization from an unpredictable finalizer is
forbidden. If explicit close was omitted while a CUDA binding is open, active,
poisoned, or aborting, the finalizer transfers the entire engine resource graph
to the same process-level quarantine owner and emits a resource warning; it
does not unregister consumers or free CUDA resources. This makes explicit close
the only non-quarantining shutdown path.

Binding construction is transactional. Identity and consumer validation occurs
before a non-owning Warp wrapper or registry entry is created. A partial failure
before any publish wait/read is enqueued may release its local wrapper normally.
After the first successful publish wait, any initialization/registration failure
uses the same poison and abort disposition rules. A partially initialized
`GpuEngine` that has not created a binding follows its existing cleanup; one
that has published or bound external CUDA work is owned by explicit close or
quarantine.

An ordinary Python exception after successful reader enqueue is not a CUDA
timeline failure. The context manager records the real done event and advances
completion before propagating the exception. If that safety record itself
fails, the poison path overrides normal recovery and raises a publication
poison error chained from the original exception.

## Production evidence boundary

C0b exposes immutable physics-publication evidence without importing
`benchmarks.optical`:

- binding ID and hashed stream token;
- frame/slot/step/device identity;
- hashed publish/done event tokens;
- borrow count and publish-wait-enqueued state;
- last physics-buffer reader operation;
- consumer-completion-recorded and logical-reclaim state;
- done-event-retained and producer-reuse-wait-enqueued states;
- close/final-drain state;
- poison state/reason category;
- emergency-sync attempted/succeeded counts and failure category;
- teardown disposition, quarantine acceptance, and process-terminal state.

These facts are written only by state transitions in the publication binding.
They are not copied from a gate row and do not rely on event polling. The later
shipping lane may map them into `ShippingPublicationReceipt`; it cannot claim
zero-copy transform binding, payload drain, or optical-output reclaim from C0
evidence alone.

Raw CUDA handles are not serialized. Stable tokens hash the run-local binding
identity, device ordinal, resource kind, and native handle.

## C0b implementation boundary

Expected production changes are limited to:

- a small physics-owned CUDA publication binding/lease module;
- public ordered/poison operations and metadata in `physics.publish`;
- `GpuEngine` bind, reuse-wait, reset, replace/remove, and legacy-conflict
  integration;
- explicit engine close plus process-level quarantine ownership for unsafe
  finalization and failed emergency synchronization;
- exports needed by product callers;
- CPU state-machine tests and real one-/two-device GPU lifecycle tests;
- a C0b implementation note and cleanup-ledger update if new temporary test
  scaffolding is introduced.

C0b does not add Torch optical scene ownership, CUDA LBVH, renderer changes,
output arenas, delivery rings, benchmark submission, baseline artifacts, or a
backend selector.

## Acceptance tests

CPU/fake-runtime tests must cover:

- consumer/device/owner/handle validation and rebinding rejection;
- publish wait success and wait failure;
- strict borrow and completion order, duplicate completion, and completion
  without an active borrow;
- ordinary exception safety completion;
- event allocation/record/ring-commit poison paths;
- binding/consumer/slot poison scope, successful emergency shutdown abort, and
  failed-sync quarantine disposition;
- active-lease close rejection, device-side final drain, unregister ordering,
  reset rejection, and evidence transitions;
- idempotent abort, re-entrant abort rejection, explicit engine close,
  finalizer/partial-initialization handoff, and proof that unregister alone
  releases no device resource;
- proof that no lifecycle code calls host polling or synchronization.

The last rule excludes the explicitly named poisoned-shutdown emergency
synchronization. Tests must prove it is unreachable from every normal path and
is invoked at most once after poison.

Real GPU tests must cover:

- a Torch-owned CUDA stream wrapped by physics publication;
- publish-event wait, zero-copy read enqueue, done-event record, slot reuse, and
  data consistency without host synchronization in the C0 hot path;
- caller ambient-device restoration on success and injected failure;
- target-device/stream mismatch with CUDA current device deliberately set to a
  different GPU;
- two-device mismatch/restoration where available;
- event retention through producer wait and close/final-drain behavior;
- injected wait/record failure proving the slot payload, stream owner, events,
  and engine allocations stay strongly owned until emergency synchronization
  succeeds or the full resource graph is accepted by quarantine;
- successful emergency synchronization before resource release, plus an
  injected synchronization failure that selects process-terminal quarantine
  without a subsequent CUDA cleanup call;
- legacy Warp consumer behavior remaining intact.

The tests may synchronize only after the lifecycle transaction, to verify final
data or clean up the test. Such synchronization is outside borrow, completion,
reuse dependency, and close operations.

Full unit, GPU, formatting, and `git diff --check` validation remains mandatory
before the C0b review commit. No benchmark result or baseline is produced.

## Review questions for Claude

1. Does the required stream-owner object close the destroyed/reused-handle gap
   without transferring stream ownership to physics?
2. Are FIFO borrow/completion, forgotten-completion, and ordinary-exception
   rules both safe and unambiguous?
3. Is retaining completion records until a covering producer wait sufficient
   to prove slot reuse safety while avoiding host observation?
4. Does shutdown-only emergency synchronization establish the only ordinary
   release proof after poison, with whole-engine process quarantine as the sole
   fail-closed fallback?
5. Is device-side close/final-drain sufficient before unregistering the logical
   consumer?
6. Does the evidence boundary provide C0-owned S provenance without allowing it
   to overclaim zero-copy optical binding or physical event completion?
7. Is the C0b file/test scope narrow enough to avoid prematurely implementing
   C2-C4/B3 or a CUDA optical product lane?

## Recommendation

Approve this P1-revised C0a state machine before C0b implementation. If approved, Codex
will implement exactly this public physics-publication contract, run the
lifecycle/GPU regression matrix, write the C0b implementation note, and return
that code commit for Claude review.
