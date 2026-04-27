Initiative: q52-publish-pipeline
Stage: discussion
Author: codex
Version: v1
Date: 2026-04-27
Status: ready-for-review
Related Files: OPEN_QUESTIONS.md#Q52, physics/publish.py, physics/gpu_engine.py, collab/q52-async-host-staging__implementation-note__codex__v1.md
Owner Summary: Reorganizes the Q52 consumer/backpressure model after the async host staging pass. The main clarification is that consumer purpose, access mode, QoS, and execution location are separate axes. Host consumers can use host-side ack/backpressure; GPU-resident render/sensing consumers should use stream/event/fence semantics rather than CPU condition waits.

---

## 1. Why This Note Exists

The current Q52 implementation has a host-side `PublishedRing` and
future-aware `SnapshotHandle`. That is enough for host consumers such as debug
export and dataset logging.

However, render-backed sensing and future RL observation paths may be
GPU-resident. For those paths, making CPU participate in fine-grained per-frame
blocking would be the wrong long-term shape. Device-side consumers should use
GPU ordering primitives.

This note separates those concerns before implementing
`PublishPolicy(on_ring_full="block")`.

## 2. Orthogonal Axes

Do not collapse these four concepts into one enum.

```text
consumer_kind:
  realtime_render
  render_backed_sensing
  host_export

consumer_location:
  host
  device

access_mode:
  borrow
  snapshot

qos_mode:
  best_effort
  lossless
```

`consumer_kind` says what the consumer is for.

`consumer_location` says where the consumption work actually executes.

`access_mode` says whether the consumer reads the published slot directly
within a short lease (`borrow`) or creates an owned copy/staging artifact
(`snapshot`).

`qos_mode` says whether the consumer may drop frames.

## 3. Current Consumer Classes

### Realtime Render

Typical first version:

```text
consumer_kind = realtime_render
consumer_location = host or device
access_mode = borrow
qos_mode = best_effort
```

Semantics:

- reads the latest available frame
- may skip frames
- should not block physics
- should keep any borrow scope short

For a host viewer, the existing host `borrow_latest_frame()` path is sufficient.

For a GPU viewer, the future target should be stream/event ordering, not a CPU
wait loop.

### Render-Backed Sensing

Typical future versions:

```text
consumer_kind = render_backed_sensing
consumer_location = device
access_mode = borrow or device_snapshot
qos_mode = best_effort or lossless
```

Semantics:

- uses published physics state to drive camera/depth/LiDAR/surface-query work
- may produce downstream observation tensors
- should stay on GPU when the downstream pipeline is GPU-resident

If sensing output is part of a training-critical observation stream, it may need
lossless semantics. But the release condition should be a GPU event/fence or a
device-visible queue state, not a host condition variable.

### Host Export

Typical current version:

```text
consumer_kind = host_export
consumer_location = host
access_mode = snapshot
qos_mode = best_effort or lossless
```

Semantics:

- creates host-owned copies for debug, logging, or datasets
- uses `SnapshotHandle`
- for `lossless + snapshot`, ack happens when staging owns the requested fields
- participates in host-side ring reclaim/backpressure

This is the path implemented by Q52 async host staging.

## 4. Reclaim Rules By Location

### Host Consumer Reclaim

Current rule:

```text
slot is reclaimable iff
  slot.frame_id <= min(acked_frame_id for enabled host lossless consumers)
```

Best-effort host consumers are not blockers.

Host lossless consumers advance ack at explicit lifecycle points:

```text
borrow:
  lease exit -> ack frame_id

snapshot:
  staged payload owns requested fields -> ack frame_id
```

This is where a host-side `on_ring_full="block"` can make sense:

```text
GpuEngine wants to reuse slot k
slot k is pinned by host lossless consumer
policy is block
host publish/control thread waits until host ack advances
```

### Device Consumer Reclaim

Device consumers should not make the CPU wait for every frame if the whole
pipeline is GPU-resident.

Preferred shape:

```text
physics stream writes published slot k
record publish_event[k]

render/sensor stream waits publish_event[k]
render/sensor kernels read slot k
record consumer_done_event[k]

physics stream reuses slot k only after required done events are satisfied
```

The CPU may enqueue dependencies and manage coarse policy, but the ordering
belongs to the GPU runtime.

## 5. Where GPU Primitives Fit

GPU primitives are relevant, but not for host export backpressure.

```text
CUDA/Warp events + stream waits:
  best fit for cross-kernel / cross-stream producer-consumer ordering

global atomics:
  useful for device queues, counters, persistent kernels, or compacted worklists

mbarrier:
  useful for tightly coupled device-side pipelines, especially within or around
  cooperative kernel structures

host condition variables:
  useful only for host consumers and host-side backpressure
```

So the likely long-term split is:

```text
host lossless consumer:
  acked_frame_id + host-side wait/notify

device lossless consumer:
  device event/fence/queue state

mixed host/device consumer:
  device event completes first, then host staging or callback advances host ack
```

## 6. Implication For on_ring_full="block"

`PublishPolicy(on_ring_full="block")` should not be treated as one universal
blocking mechanism.

Recommended interpretation:

```text
on_ring_full="block" for host lossless blockers:
  block the host publish/control path until host ack advances

on_ring_full="block" for device lossless blockers:
  insert/wait GPU stream dependencies or consult device completion state
  avoid CPU sleep-waiting on GPU-only render/sensing work
```

This implies `PublishedRing` may need either:

- separate host/device blocker accounting, or
- a pluggable blocker interface:

```text
HostAckBlocker
DeviceEventBlocker
```

The current `ConsumerState.acked_frame_id` is adequate for host consumers, but
not expressive enough for device event/fence completion.

## 7. RL Hot Path Is Not A Ring Consumer

The main RL training loop should not be forced through `PublishedRing`.

Typical GPU training loop:

```text
physics kernels -> obs kernels -> policy network -> action writeback -> next physics step
```

If this loop runs on one CUDA/Warp stream, kernel launch order already gives the
required read-after-write ordering. If it spans multiple streams, explicit
stream events are the right dependency primitive.

`PublishedRing` has a different job:

```text
provide stable slot lifetime for external or asynchronous consumers
```

Examples:

- host debug export
- dataset logging
- realtime viewers
- lower-rate render/sensor side channels
- consumers that need a frame to remain stable after physics moves on

RL observation schema can still define a published contract. The important
distinction is:

```text
contract:
  field names, shapes, ordering, units, and optionality

PublishedRing:
  retained slot lifetime for consumers outside the training hot path
```

For example, `contact_mask` now has a stable contract. A host snapshot can read
it from a published ring slot, while a GPU obs kernel can read the current
device buffer directly if it is executing in the main training stream.

The bypass applies to numeric state observations such as:

- `q`
- `qdot`
- `X_world`
- `v_bodies`
- `contact_mask`

Render-backed sensors are different. Camera, depth, LiDAR, and similar
render/surface-query sensors need stable frame lifetime while their render or
query pass executes. Those paths should consume through the ring or an
equivalent retained-slot/event mechanism, not directly through transient
scratch buffers.

This avoids making Python ring acquire/mark-ready bookkeeping a mandatory cost
at high physics rates such as 5000 Hz.

## 8. Best-Effort Borrow Safety

Best-effort consumers do not pin slots.

That means a borrowed GPU frame can become stale if physics reuses the slot
while the consumer still holds a descriptor. The stale-slot guard will raise
`SlotReclaimedError` on guarded field access, but it does not make the borrow a
retention guarantee.

Correct usage for best-effort borrow is:

```text
enter borrow scope
read/copy the required fields immediately
leave borrow scope
```

Realtime renderers should either finish their read within the borrow scope or
explicitly snapshot/copy the fields they need. A long-lived renderer reference
to a borrowed frame is a misuse.

If `SlotReclaimedError` is raised during a best-effort borrow, the consumer
should skip that frame and try again with the latest available frame on its next
attempt. Retrying the same slot is never correct because the slot lifetime has
already been lost.

## 9. Proposed Near-Term Decision

Do not implement generic `on_ring_full="block"` yet as a plain host condition
wait without naming its scope.

Instead:

1. rename/clarify the current pending work as **host lossless block**;
2. use `ConsumerState.consumer_location` as the explicit host/device marker;
3. keep host export on the existing `SnapshotHandle` / `acked_frame_id` path;
4. design GPU render-backed sensing around stream/event dependencies;
5. keep the RL training hot path free to read current GPU buffers directly;
6. only then implement a unified reclaim decision that can consider both host
   ack and device event/fence blockers.

## 10. Review Targets

Questions for Claude/review:

1. Should `consumer_location` be explicit in `ConsumerState`, or should host vs
   device be modeled by separate consumer state classes?
2. Should device consumers share `ConsumerState`, or use a separate
   `DeviceConsumerState` with event/fence handles?
3. Is `on_ring_full="block"` too overloaded? Should we split host policy and
   device policy?
4. For GPU render-backed sensing, should the first implementation use
   Warp/CUDA stream events rather than mbarrier/atomics?
5. Should host export and device sensing share one `PublishedRing`, or should
   device pipelines eventually use a separate device-visible ring/queue?

## 11. Review Response

Current response after review:

1. `consumer_location: Literal["host", "device"] = "host"` has been added to
   `ConsumerState`.
2. Do not split `DeviceConsumerState` yet; introduce it when there are real
   device event/fence handles to store.
3. Treat `on_ring_full="block"` as host-only until device consumer semantics
   exist. If a device consumer is registered, a host condition wait should not
   be used as the device blocking mechanism.
4. Use CUDA/Warp stream events for GPU render-backed sensing. mbarrier and
   global atomics are useful primitives, but they are not the first tool for
   cross-stream physics-to-sensor ordering.
5. Keep a shared host-side `PublishedRing` for now. Consider a separate
   device-visible ring only if the full GPU pipeline needs to run without CPU
   control-plane involvement.

## 12. Current Recommendation

For Q52 phase-1:

```text
Host export:
  keep current SnapshotHandle + acked_frame_id design

Host-side block:
  implemented as host lossless backpressure only

GPU render/sensing:
  defer to a separate device-consumer design using event/fence ordering

RL hot path:
  bypass PublishedRing and consume current device buffers / scratch directly
```

This avoids baking a CPU-centric blocking model into GPU-only render/sensing
paths or the main training loop.
