Initiative: q52-publish-pipeline
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-27
Status: ready-for-review
Related Files: OPEN_QUESTIONS.md#Q52, physics/publish.py, physics/gpu_engine.py, tests/unit/physics/test_publish.py
Owner Summary: Refreshes Q52 after Q50/Q51/Q53/RL obs work changed the implementation baseline. The publish control plane was already phase-1 real, so this pass starts phase-2 runtime hardening: stale-slot guards now detect slot reuse, and `PublishedRing` is a dedicated physics-runtime control component owned internally by `GpuEngine`.

---

## 1. Updated Baseline

The old Q52 text treated the publish pipeline as mostly unimplemented. That is
no longer accurate.

Already real in phase-1:

- `PublishPolicy / PublishPlan / ViewPolicy`
- `ConsumerState / AckPolicy / SlotReclaimer / RingPressureStats`
- `BorrowedFrameLease / SnapshotHandle`
- `CpuPublishedFrame / GpuPublishedFrame`
- synchronous GPU publish into dedicated slot buffers
- CPU/GPU `latest_published_frame()`
- published-frame consumers for debug export, render scene, telemetry, sensing,
  and `RenderScene.sensor_data`

Still phase-2 work:

- Warp stream/event host staging / bounded export queue
- `on_ring_full="block"`
- typed slot/block dataclasses
- compact contact-pair published contract
- stronger retained-frame/runtime safety beyond this synchronous ring pass

Partially landed after this note:

- `SnapshotHandle` is now future-aware.
- GPU `lossless + snapshot` host snapshots stage through a host-side future.
- `on_snapshot_staged` ack advances from the handle's staged-completion point,
  not from the call site.

---

## 2. Stale-Slot Guard

Before this pass, `GpuPublishedFrame` guarded payload access only with:

```text
slot_meta.invalidated
```

That misses the stable failure mode where a slot has already been reused and
marked ready for a newer frame. The old frame still points at the same
`PublishedSlotMeta`, but `invalidated` is false again.

The guard now checks both:

```text
slot_meta.invalidated
slot_meta.frame_id != frame.frame_id
```

This catches both in-progress reclaim/rewrite and completed slot reuse.

---

## 3. PublishedRing Ownership

Decision:

```text
PublishedRing belongs to physics runtime and is owned internally by GpuEngine.
```

`PublishedRing` is not externalized to `Simulator` and is not owned by
`rendering/` or `sensing/`.

Rationale:

- the ring publishes physics truth, not render or sensor-owned data
- ring pressure can affect physics-step publish behavior
- slot payload allocation depends on `GpuEngine` runtime dimensions and device
- `GpuEngine` should keep the public API stable and delegate internal
  meta/reclaim/latest bookkeeping to the ring

---

## 4. Control/Payload Split

Implemented split:

```text
physics.publish.PublishedRing
  owns: slot metadata, consumer list, reclaim decision, latest-frame pointer
  holds: references to slot payload buffers

physics.gpu_engine.GpuEngine
  owns: slot buffer allocation and wp.copy writes from scratch to slot buffers
```

This keeps `physics.publish` CPU-only and free of Warp allocation logic, while
still centralizing the atomic acquire/reclaim decision inside the ring.

---

## 5. Public API Stability

No consumer-facing GPU API was changed:

- `set_publish_policy(...)`
- `register_consumer(...)`
- `unregister_consumer(...)`
- `ring_pressure_stats()`
- `latest_published_frame()`
- `borrow_latest_frame(...)`
- `snapshot_frame_to_host(...)`

These now delegate to the internal ring where appropriate.

---

## 6. Verification

Command:

```bash
PYTHONPATH=. pytest \
  tests/unit/physics/test_publish.py \
  tests/unit/physics/test_cpu_publish_runtime.py \
  tests/unit/physics/test_telemetry_snapshot.py \
  tests/unit/rendering/test_published_frame_bridge.py \
  tests/unit/sensing \
  -q
```

Result:

```text
45 passed
```

Extended command:

```bash
PYTHONPATH=. pytest \
  tests/unit/rendering \
  tests/integration/test_published_frame_render_backend_integration.py \
  tests/unit/sensing \
  tests/unit/physics/test_telemetry_snapshot.py \
  tests/unit/physics/test_cpu_publish_runtime.py \
  tests/unit/physics/test_publish.py \
  -q
```

Result:

```text
74 passed
```
