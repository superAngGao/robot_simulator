Initiative: q52-publish-pipeline
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-28
Status: implemented
Related Files: OPEN_QUESTIONS.md#Q52, physics/publish.py, physics/gpu_engine.py, tests/unit/physics/test_publish.py, tests/gpu/test_gpu_engine_api.py
Owner Summary: Adds the first device-consumer reclaim hook to Q52 and wires it to Warp/CUDA stream events in `GpuEngine`. Host consumers still reclaim through `acked_frame_id`; device consumers reclaim through `device_completed_frame_id` backed by a recorded `done_event`.

## Summary

This pass separates the reclaim progress counters:

- host consumers: `acked_frame_id`
- device consumers: `device_completed_frame_id`

`ConsumerState.reclaim_frame_id` now chooses the correct counter based on
`consumer_location`.

`SlotReclaimer` and `PublishedRing` use `reclaim_frame_id` when deciding
whether a slot is still pinned by lossless consumers.

`GpuEngine` now records real Warp events for device-side handoff:

```text
physics/current stream:
  write published slot k
  record publish_event[k]

render/sensor stream:
  wait publish_event[k]
  read slot k
  record consumer_done_event[k]

physics/current stream before slot reuse:
  wait consumer_done_event[k]
```

The important distinction is that "blocking" is split in two:

- control-plane blocking: `PublishedRing` keeps the slot pinned until a
  lossless device consumer advances `device_completed_frame_id`
- device-timeline blocking: the physics/current stream enqueues a wait on the
  consumer's `done_event` before overwriting that slot

This must not become host blocking in the hot path. In Warp terms, the intended
primitive is `Stream.wait_event(...)`; in CUDA terms, the future migration should
map directly to:

```text
cudaEventRecord(publish_event, physics_stream)
cudaStreamWaitEvent(consumer_stream, publish_event)
... render/sensor kernels read the published slot ...
cudaEventRecord(done_event, consumer_stream)
cudaStreamWaitEvent(physics_stream, done_event)
```

Avoid `wp.synchronize_event(...)`, `cudaEventSynchronize(...)`, or elapsed-time
queries in the default render/publish path. Those belong only in explicit debug
or profiler workflows, because they force CPU-visible synchronization.

## New API

`PublishedRing` now has:

```python
mark_device_consumer_complete(consumer, frame_id, done_event=None)
```

This advances `consumer.device_completed_frame_id` and stores the latest
`device_done_event` object.

`GpuEngine` now exposes:

```python
frame = engine.borrow_device_frame("camera", frame_id, stream=render_stream)
done_event = engine.complete_device_consumer("camera", frame.frame_id, stream=render_stream)
```

`borrow_device_frame(...)` makes the device consumer stream wait on the frame's
`ready_event`. `complete_device_consumer(...)` records a done event on that
consumer stream and passes it to `PublishedRing.mark_device_consumer_complete(...)`.

Device borrows must be paired with `complete_device_consumer(...)`. The host
borrow path can ack on Python context-manager exit, but the device path cannot:
the reclaim point is the recorded device done event after consumer-stream work
has been enqueued. Forgetting completion leaves the slot pinned until
`max_lag_frames` stall detection fires, if configured.

Host consumers are rejected by this method with `ValueError`.

`ConsumerState.max_lag_frames` is now enforced for lossless device consumers.
The default `None` preserves the original lossless behavior. When a device
consumer sets a finite budget and the producer reaches a pinned slot with
`next_frame_id - consumer.reclaim_frame_id > max_lag_frames`, `PublishedRing`
raises `DeviceConsumerStalledError` with the consumer id, lag, reclaim frame,
producer frame, and target slot metadata.

`max_lag_frames=0` is legal but strict. Since a never-completed device consumer
starts at reclaim frame `-1`, it can stall on the first pinned-slot
backpressure check.

`GpuEngine.publish_stats()` returns a lightweight monitor snapshot:

- ring size, latest frame, next frame, target slot
- per-slot state and pinning consumers
- per-consumer reclaim id, lag, max lag, blocker/stalled flags
- backpressure / skip / host-wait / stall / raise counters
- materialized publish count plus host-observed rolling publish interval / FPS

The stats path is metadata only. It does not query CUDA event timing or force a
GPU synchronization point. `is_stalled` is a static lag-budget flag; acquire
raises `DeviceConsumerStalledError` only when a stalled device consumer is also
blocking the target slot being reused. Rolling FPS uses CPU `mark_ready()`
timestamps, so it measures publish cadence rather than GPU kernel completion.

## Boundary

Still not implemented:

- async device event polling
- device-visible queue state
- realtime renderer or render-backed sensing kernels

The existing protection remains for pending device blockers:
`PublishPolicy(on_ring_full="block")` with a device lossless blocker that has
not recorded completion still raises `NotImplementedError`. If the device
consumer has recorded completion, the slot can be reused and the physics stream
enqueues a device-side wait on the recorded done event before writing the slot.

Device lossless consumers can still pin a ring forever when `max_lag_frames` is
left as `None`. This is intentional: unbounded lossless remains the default
semantic. Production render/sensor consumers should set a finite lag budget if
they want stall detection instead of indefinite backpressure.

## Verification

```bash
PYTHONPATH=. pytest tests/unit/physics/test_publish.py tests/unit/physics/test_cpu_publish_runtime.py -q
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_gpu_engine_api.py -q
ruff check physics/publish.py physics/gpu_engine.py physics/__init__.py tests/unit/physics/test_publish.py tests/gpu/test_gpu_engine_api.py
python -m compileall physics/publish.py physics/gpu_engine.py physics/__init__.py tests/unit/physics/test_publish.py tests/gpu/test_gpu_engine_api.py
```

Results:

- CPU publish: `43 passed`
- GPU API: `40 passed`
- `ruff` passed
- `compileall` passed
