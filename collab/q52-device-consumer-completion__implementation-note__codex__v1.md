Initiative: q52-publish-pipeline
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-28
Status: implemented
Related Files: OPEN_QUESTIONS.md#Q52, physics/publish.py, tests/unit/physics/test_publish.py
Owner Summary: Adds the first device-consumer reclaim hook to Q52. Host consumers still reclaim through `acked_frame_id`; device consumers now reclaim through an explicit `device_completed_frame_id`, giving future CUDA/Warp event callbacks a control-plane landing point.

## Summary

This pass does not pretend to implement real GPU stream/event wiring yet.
Instead, it separates the reclaim progress counters:

- host consumers: `acked_frame_id`
- device consumers: `device_completed_frame_id`

`ConsumerState.reclaim_frame_id` now chooses the correct counter based on
`consumer_location`.

`SlotReclaimer` and `PublishedRing` use `reclaim_frame_id` when deciding
whether a slot is still pinned by lossless consumers.

## New API

`PublishedRing` now has:

```python
mark_device_consumer_complete(consumer, frame_id, done_event=None)
```

This advances `consumer.device_completed_frame_id` and stores the latest
`device_done_event` object. Today `done_event` is opaque; a future Warp/CUDA
integration can pass the real event/fence handle here after the render/sensor
stream has finished consuming a slot.

Host consumers are rejected by this method with `ValueError`.

## Boundary

Still not implemented:

- real Warp/CUDA event creation
- stream wait insertion
- async device event polling
- device-visible queue state
- device consumer stall detection / max-lag enforcement
- realtime renderer or render-backed sensing kernels

The existing protection remains: `PublishPolicy(on_ring_full="block")` with a
pending device lossless blocker still raises `NotImplementedError` until a real
event/fence path exists. If a device consumer has already marked the frame
complete, the slot can be reused.

Device lossless consumers can still pin a ring forever if the downstream
device pipeline stalls and never calls `mark_device_consumer_complete(...)`.
`ConsumerState.max_lag_frames` exists as a policy field, but this pass does not
yet enforce it for device consumers.

## Verification

```bash
PYTHONPATH=. pytest tests/unit/physics/test_publish.py tests/unit/physics/test_cpu_publish_runtime.py -q
ruff check physics/publish.py tests/unit/physics/test_publish.py
python -m compileall physics/publish.py tests/unit/physics/test_publish.py
```

Results:

- `36 passed`
- `ruff` passed
- `compileall` passed
