Initiative: q52-publish-pipeline
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-27
Status: ready-for-review
Related Files: OPEN_QUESTIONS.md#Q52, physics/publish.py, physics/gpu_engine.py, physics/cpu_engine.py, tests/unit/physics/test_publish.py, tests/unit/physics/test_cpu_publish_runtime.py, tests/gpu/test_gpu_engine_api.py
Owner Summary: Moves `lossless + snapshot` from a purely synchronous bridge to a future-aware staging contract. `SnapshotHandle` can now represent not-yet-staged snapshots; GPU lossless host snapshots use a host staging future and only ack once the staged payload owns all requested fields.

---

## 1. Scope

This pass implements the first async-staging boundary for Q52 without changing
the public engine API:

- `SnapshotHandle` supports immediate and future-backed payloads.
- `SnapshotHandle.result()` waits for future-backed staging and returns the
  owned payload.
- `SnapshotHandle.staged` and `SnapshotHandle.is_ready` reflect future
  completion.
- `on_staged` callbacks run exactly once.
- CPU snapshots remain synchronously staged.
- GPU `lossless + snapshot` consumers stage through a host-side future.
- GPU best-effort snapshots remain synchronous to avoid letting a reclaimable
  slot be read by a background worker.

## 2. Ack Semantics

The key semantic change is that `lossless + snapshot` ack has moved to the
snapshot handle's staged-completion point.

Before:

```text
snapshot_frame_to_host(...)
  copy to host
  ack consumer
  return ready handle
```

Now for GPU lossless snapshot:

```text
snapshot_frame_to_host(...)
  submit host staging future
  return not-yet-staged handle

future completes
  staged payload owns requested fields
  handle runs on_staged
  consumer.acked_frame_id advances
```

This matches the Q52 rule that a lossless snapshot must not ack when work is
only enqueued. It acks when staging owns a complete copy.

## 3. Ring Interaction

Lossless consumers pin published slots until their `acked_frame_id` advances.
Because GPU lossless snapshot ack now waits for staging completion, the ring
cannot reclaim the source slot while the background staging future still needs
to read it.

Best-effort snapshots do not participate in reclaim accounting, so they remain
synchronous in this pass. Async best-effort export would need a separate
retention/drop policy.

## 4. Deferred Work

Still deferred after this pass:

- real Warp stream/event-based device-to-host copies
- explicit host staging queue capacity and overflow policy
- `PublishPolicy(on_ring_full="block")` wait semantics
- cancellation/shutdown behavior for long-running export queues
- typed host snapshot payloads instead of `dict[str, object]`

Known risk in the current future-backed bridge:

- worker-thread staging errors surface through the `SnapshotHandle` future.
  For example, if a slot is reclaimed unexpectedly before a lossless snapshot
  finishes staging, `SlotReclaimedError` will be raised from `handle.result()`,
  not from `snapshot_frame_to_host(...)`.
- this is acceptable for phase-1 because lossless ack should pin the source
  slot until staging completes, but phase-2 should add explicit error-state
  reporting and queue-level failure accounting.

## 5. Verification

Commands:

```bash
PYTHONPATH=. pytest \
  tests/unit/physics/test_publish.py \
  tests/unit/physics/test_cpu_publish_runtime.py \
  -q
```

Result:

```text
29 passed
```

Extended command:

```bash
PYTHONPATH=. pytest \
  tests/unit/physics/test_publish.py \
  tests/unit/physics/test_cpu_publish_runtime.py \
  tests/unit/physics/test_telemetry_snapshot.py \
  tests/unit/rendering/test_debug_exporter.py \
  tests/unit/rendering/test_published_frame_bridge.py \
  tests/unit/sensing \
  tests/unit/rl_env \
  -q
```

Result:

```text
81 passed
```

GPU target command:

```bash
PYTHONPATH=. conda run -n env_tilelang_20260119 python -m pytest \
  tests/gpu/test_gpu_engine_api.py::TestStepOutputFields::test_lossless_snapshot_acks_after_staging \
  -q
```

Result:

```text
1 passed
```
