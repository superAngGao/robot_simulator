Initiative: q52-publish-pipeline
Stage: handoff-for-review
Author: codex
Version: v1
Date: 2026-04-29
Status: ready-for-review
Related Files: OPEN_QUESTIONS.md#Q52, physics/publish.py, physics/gpu_engine.py, tests/unit/physics/test_publish.py, tests/gpu/test_gpu_engine_api.py
Reviewer Prompt: Review the Q52 device-consumer runtime changes, especially stream/event semantics, stall detection policy, and stats API boundaries.

# Q52 Device Consumer Runtime Handoff

This pass moves Q52 from a device-event skeleton to a runtime with data
consistency coverage, stall protection, and a lightweight monitor snapshot.

## Implemented

### 1. Device consumer stream-event handoff

`GpuEngine` now exposes:

```python
frame = engine.borrow_device_frame("camera", frame_id, stream=consumer_stream)
done_event = engine.complete_device_consumer("camera", frame.frame_id, stream=consumer_stream)
```

Runtime ordering:

```text
physics/current stream:
  write published slot k
  record publish_event[k]

consumer stream:
  wait publish_event[k]
  read published slot k
  record done_event[k]

physics/current stream before slot reuse:
  wait done_event[k]
```

The code uses explicit stream waits:

```python
(stream or self._current_stream()).wait_event(event)
```

The intended CUDA migration mapping is:

```text
cudaEventRecord(publish_event, physics_stream)
cudaStreamWaitEvent(consumer_stream, publish_event)
cudaEventRecord(done_event, consumer_stream)
cudaStreamWaitEvent(physics_stream, done_event)
```

Important semantic boundary:

- control-plane pinning: `PublishedRing` keeps a slot unreclaimable until a
  lossless device consumer advances `device_completed_frame_id`
- device-timeline wait: stream/event dependencies order GPU work without host
  synchronization

The default publish/render hot path should not use `wp.synchronize_event`,
`cudaEventSynchronize`, or CUDA elapsed-time queries.

### 2. Fake device consumer data consistency test

`tests/gpu/test_gpu_engine_api.py` now includes a fake device consumer kernel
that runs on an independent Warp stream.

The test:

1. Publishes frame 0.
2. Borrows that frame for a device consumer.
3. Launches a kernel that reads published `q_wp` and writes a checksum to a
   device buffer.
4. Records the consumer done event on the consumer stream.
5. Advances physics until slot 0 is reused.
6. Synchronizes only at assertion time and checks that the checksum still
   matches frame 0.

This verifies that slot reuse waits for the consumer's GPU work, rather than
passing only because of CPU-side ordering.

Post-review follow-ups added:

- `borrow_device_frame(frame_id=None)` defaults to the latest frame.
- borrowing with no latest frame raises `KeyError`.
- `_wait_for_device_consumers_before_reuse(...)` has mock-level tests for
  waiting recorded done events, skipping absent events, and rejecting incomplete
  lossless device consumers.

### 3. Device stall detection / `max_lag_frames`

`ConsumerState.max_lag_frames` is now enforced for lossless device consumers.

Default behavior:

```python
max_lag_frames=None
```

This preserves unbounded lossless semantics.

`max_lag_frames=0` is legal but strict. Since a never-completed consumer starts
at reclaim frame `-1`, it can stall on the first pinned-slot backpressure
check.

When a finite budget is configured, `PublishedRing.acquire(...)` checks:

```text
next_frame_id - consumer.reclaim_frame_id > max_lag_frames
```

If exceeded, it raises `DeviceConsumerStalledError`.

The exception carries:

- `consumer_id`
- `producer_frame_id`
- `reclaim_frame_id`
- `lag_frames`
- `max_lag_frames`
- `target_slot_id`
- `target_slot_frame_id`

This gives the future renderer/runtime a concrete recovery hook for downgrade,
restart, or fail-fast behavior.

### 4. Publish stats API

`GpuEngine.publish_stats()` and `PublishedRing.publish_stats(...)` return a
metadata-only snapshot. The stats path does not query CUDA event timing and does
not force GPU synchronization.

`is_stalled` is a static lag-budget observation for each consumer. It does not
mean the next `acquire(...)` call must raise: acquire raises only when that
stalled device consumer is also blocking the target slot being reused.

Top-level fields:

- `ring_size`
- `latest_frame_id`
- `next_frame_id`
- `target_slot_id`
- `min_lossless_reclaim_frame_id`
- `blocking_consumer_ids`
- `stalled_consumer_ids`
- `backpressure_count`
- `skip_count`
- `block_wait_count`
- `stall_count`
- `raise_count`
- `materialized_publish_count`
- `rolling_publish_window_size`
- `rolling_publish_sample_count`
- `last_publish_host_time_s`
- `rolling_publish_interval_s`
- `rolling_publish_fps`

Rolling FPS is host-observed cadence. It uses CPU timestamps captured in
`PublishedRing.mark_ready(...)`, so it reports materialized publish frequency
without querying CUDA event elapsed time or synchronizing GPU work.

Per-slot fields:

- `slot_id`
- `frame_id`
- `step_index`
- `sim_time`
- `state`
- `invalidated`
- `pinned_by_consumer_ids`

Per-consumer fields:

- `consumer_id`
- `consumer_kind`
- `consumer_location`
- `qos_mode`
- `access_mode`
- `enabled`
- `latest_seen_frame_id`
- `acked_frame_id`
- `device_completed_frame_id`
- `reclaim_frame_id`
- `lag_frames`
- `max_lag_frames`
- `is_blocking_target_slot`
- `is_stalled`

## Key Files

- `physics/publish.py`
  - `DeviceConsumerStalledError`
  - `PublishedSlotStats`
  - `ConsumerPublishStats`
  - `PublishRuntimeStats`
  - `PublishedRing.publish_stats(...)`
  - `max_lag_frames` enforcement

- `physics/gpu_engine.py`
  - `borrow_device_frame(...)`
  - `complete_device_consumer(...)`
  - explicit stream event waits
  - `publish_stats()`

- `tests/gpu/test_gpu_engine_api.py`
  - Warp event type checks
  - fake device consumer checksum test
  - GPU stall detection test

- `tests/unit/physics/test_publish.py`
  - ring stall tests
  - stats snapshot tests
  - negative `max_lag_frames` validation

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

## Not Implemented Yet

1. CUDA event elapsed timing. Recommendation: keep this in profiler/debug paths
   only, not default stats.
2. Real render-backed sensing kernels.
3. Async device event polling / device-visible queue state.
4. Host staging migration from Python `Future` to Warp/CUDA stream-event plus
   bounded queue.

## Current Judgment

Q52 device consumer runtime now has a closed control-plane/device-timeline
mechanism, a GPU data consistency test, explicit stall protection, and a
metadata-only monitor snapshot with host-observed rolling publish FPS. The next
reasonable step is real render-backed sensing integration.
