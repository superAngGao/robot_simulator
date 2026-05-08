Initiative: q54-gpu-optical-readback-delivery-policy
Stage: review-request
Author: codex
Version: v1
Date: 2026-05-08
Status: pending-review
Related Files: optics/execution.py, optics/device.py, optics/warp_execution.py, examples/mujoco_menagerie_gpu_preview.py, collab/q54-gpu-optical-output-profile-api__review-request__codex__v1.md, collab/q54-gpu-optical-readback-materialization-microbench__review-request__codex__v1.md
Owner Summary: GPU render latency is now low enough that blocking RGB readback is the dominant video-preview cost. Existing interfaces separate GPU output profile from host readback mode, but they do not yet model delivery/synchronization policy. This proposal adds a third axis: sync blocking, async ordered/no-drop, async latest/drop, and device-only delivery.

# Q54 GPU Optical Readback Delivery Policy Review Request

## 1. Background

The current GPU optical video path has three partially separated concepts:

```text
output_profile:
  what the executor writes on device

video_readback:
  which result channels the host reads

delivery / synchronization:
  currently implicit and blocking
```

The first two are already implemented or partially implemented:

```python
executor.execute_camera(
    snapshot,
    bvh,
    camera,
    output_profile=OpticalOutputProfile.RGB_PREVIEW,
)
```

```bash
--video-readback full|rgb|none
```

The third concept is not yet explicit. Today, video readback is synchronous:

```text
render_i
wait result.ready_event
stage selected channels via Warp .numpy()
host materialization/copy
consume_i
render_i+1
```

This means there is no latency hiding between render and readback.

## 2. Recent Measurements

Measured on `cuda:1`, Go2 Menagerie, `960x640`, CUDA LBVH, GPU camera raygen,
30 frames:

```text
shadows, readback=none:
  render_execute_mean ~= 2.55 ms
  frame_total_mean    ~= 2.65 ms

no shadows, readback=none:
  render_execute_mean ~= 1.20 ms
  frame_total_mean    ~= 1.29 ms

shadows, readback=rgb:
  render_execute_mean ~= 2.57 ms
  readback_host_mean  ~= 5.07 ms
  frame_total_mean    ~= 7.77 ms
```

The `readback=rgb` frame time is approximately:

```text
render + blocking readback
```

not:

```text
max(render, readback)
```

So selected RGB readback reduces payload relative to full readback, but does
not hide latency behind later frames.

## 3. Existing Interface Layer 1: Output Profile

`OpticalOutputProfile` answers:

```text
Which channels are guaranteed to exist?
Which device buffers should be allocated?
Which channels should kernels write as public result channels?
```

Current profiles:

```text
GEOMETRY_FULL
  first-hit geometry query
  depth/range/normal/id/parity/debug

DIRECT_LIGHT_FULL
  full direct-light semantic result
  debug/parity/full preview

RGB_PREVIEW
  fast RGB preview/video path
  rgb + hit_mask + diagnostics

RENDER_ONLY
  device render timing / GPU-only downstream
  diagnostics only, no host readback implied
```

This layer should remain executor-facing. It should not encode host delivery
semantics such as drop policy or ring depth.

## 4. Existing Interface Layer 2: Readback Mode

The video benchmark currently exposes:

```bash
--video-readback full|rgb|none
```

Current mapping:

```text
full:
  output_profile=direct_light_full
  stage_optical_compute_result_to_host(...)
  blocking full/canonical host result

rgb:
  output_profile=rgb_preview
  stage_optical_channels(..., canonical_dtypes=False)
  blocking selected RGB + diagnostics result

none:
  output_profile=render_only
  no host readback
```

This layer answers:

```text
Which channels does the host consumer request?
```

It should not encode whether the readback is sync, async ordered, or latest/drop.

## 5. Missing Interface Layer 3: Delivery Policy

We should introduce an explicit delivery/synchronization policy:

```python
class OpticalDeliveryPolicy(Enum):
    SYNC_BLOCKING = "sync_blocking"
    ASYNC_ORDERED = "async_ordered"
    ASYNC_LATEST = "async_latest"
    DEVICE_ONLY = "device_only"
```

This layer answers:

```text
When is the host allowed to block?
Can frames be dropped?
Are results consumed in frame order?
How are device buffers and host staging slots released?
```

The key design point:

```text
readback mode says "what to read"
delivery policy says "how to deliver it"
```

## 6. Scenario Matrix

### 6.1 Correctness / Parity / Debug

Recommended policy:

```text
output_profile = direct_light_full or geometry_full
readback       = full
delivery       = sync_blocking
drop frames    = no
```

Rationale:

```text
Tests should be deterministic and fail close to the source of an error.
Immediate host materialization is acceptable and desirable.
```

### 6.2 Pure Render Benchmark

Recommended policy:

```text
output_profile = render_only
readback       = none
delivery       = device_only
drop frames    = not applicable
```

Rationale:

```text
Measures device render cost only.
Does not claim to represent final video or host-consumed sensor cost.
```

### 6.3 Video Recording / Offline Benchmark

Recommended policy:

```text
output_profile = rgb_preview
readback       = rgb or future rgb8
delivery       = async_ordered
drop frames    = no
```

Rationale:

```text
The video file or benchmark should contain every frame in order.
If the readback ring fills, the producer blocks on the oldest pending frame.
This preserves frame completeness while allowing render_i+1 to overlap
with readback_i during steady state.
```

### 6.4 Robot Sensor Pipeline

Recommended policy:

```text
output_profile = geometry_full, rgb_preview, or a future sensor-specific profile
readback       = consumer-required channels
delivery       = async_ordered
drop frames    = no, unless the user explicitly chooses a lossy sensor mode
```

Rationale:

```text
Sensor results are frame-aligned simulation data.
Silent frame dropping would violate the sensor contract.
If host-side consumers cannot keep up, backpressure is correct.
```

This matches the existing Q52/Q54 direction where lossless snapshot staging
acks only after the staged copy is self-contained.

### 6.5 Real-Time Preview UI

Recommended policy:

```text
output_profile = rgb_preview
readback       = rgb8 preferred, rgb acceptable initially
delivery       = async_latest
drop frames    = yes
```

Rationale:

```text
Interactive preview wants low display latency and the newest completed frame.
It does not need to show every simulation frame.
```

This should not be used for parity, benchmark completeness, or sensor logs.

## 7. Proposed Dataflow

### 7.1 Current Blocking Path

```text
render_i enqueue
wp.synchronize_event(result.ready_event)
stage_optical_channels(result, ...)
  -> synchronize ready_event again
  -> value.numpy()
  -> np.asarray(...).copy()
consume_i
render_i+1 enqueue
```

### 7.2 Async Ordered Path

```text
render stream:
  render_i ---------------- render_i+1 ---------------- render_i+2

copy stream:
           wait render_i -> copy_i
                              wait render_i+1 -> copy_i+1

host:
                         consume_i
                                               consume_i+1
```

The important rule:

```text
submit_readback(...) must not block.
consume/wait_next_ordered(...) may block.
```

## 8. Proposed Objects

### 8.1 DeviceFrameResult

This is effectively the current device `OpticalComputeResult`:

```text
device channels
render_ready_event
output_profile
resources retaining device buffers
frame metadata
```

It must remain alive until any async readback copy using its device buffers has
completed.

### 8.2 ReadbackSlot

A reusable host-side staging slot:

```text
host buffers for requested channels
copy_ready_event
state: free | copying | ready | consuming
frame metadata currently occupying the slot
```

First implementation can use normal host arrays if pinned host memory is too
large a step, but the design should leave room for pinned buffers.

Preferred first useful slots:

```text
rgb float32 [H, W, 3]
hit_mask int32 or bool [H, W]
diagnostic scalars int32
```

Future:

```text
rgb8 uint8 [H, W, 3]
```

### 8.3 ReadbackJob

A handle returned by non-blocking submit:

```text
frame_id / frame_index / sim_time
slot
requested channels
copy_ready_event
retained device resources
delivery sequence number
```

The job owns enough references to keep device buffers alive until copy
completion.

## 9. Synchronization And Release Rules

### 9.1 Render Completion

```text
render_ready_event means:
  all guaranteed device channels for result.output_profile have been written
```

This is already the intended `OpticalComputeResult.ready_event` meaning.

### 9.2 Readback Submission

On the copy stream:

```text
copy_stream.wait_event(render_ready_event)
enqueue D2H copies into ReadbackSlot buffers
copy_ready_event = copy_stream.record_event()
```

`submit_readback` returns immediately after enqueue/record.

### 9.3 Host Consumption

A host consumer may access a slot only after:

```text
copy_ready_event has completed
```

For `async_ordered`, consumption order must follow frame sequence even if a
later frame completes first.

For `async_latest`, the consumer may skip older completed or pending frames and
display the newest completed frame.

### 9.4 Slot Release

A slot returns to the free pool only after:

```text
copy_ready_event completed
host consumer has finished using the host buffers
```

The first API can make this explicit with a context manager:

```python
with completed_frame.borrow_host() as host_frame:
    ...
```

or use an explicit:

```python
completed_frame.release()
```

The context-manager shape is safer.

### 9.5 Device Resource Release

The async readback job must retain:

```text
DeviceFrameResult.resources
device channel arrays being copied
```

until:

```text
copy_ready_event completed
```

After copy completion, device resources may be released even if the host slot is
still being consumed.

## 10. Backpressure And Drop Semantics

### 10.1 SYNC_BLOCKING

No ring required:

```text
render
wait render
copy/materialize
return host result
```

### 10.2 ASYNC_ORDERED

Ring behavior:

```text
if no free slot:
  wait oldest pending job
  consume/release or return completed job to caller
  then submit new readback
```

No frames are dropped.

This is the recommended first async policy.

### 10.3 ASYNC_LATEST

Ring behavior:

```text
if no free slot:
  either skip readback for current frame
  or drop/cancel the oldest not-yet-consumed job if safe
```

Open issue:

```text
CUDA copies already enqueued cannot be trivially canceled.
```

Therefore V1 should defer `async_latest` implementation and only define its
semantics.

### 10.4 DEVICE_ONLY

No host slots:

```text
render and optionally wait for ready_event for timing
```

This is today's `--video-readback none` meaning.

## 11. Proposed Public Surface

### 11.1 Executor Stays Focused

Do not overload executor APIs with delivery policy.

Keep:

```python
result = executor.execute_camera(
    snapshot,
    bvh,
    camera,
    output_profile=OpticalOutputProfile.RGB_PREVIEW,
)
```

Executor returns a device result and a render-ready event.

### 11.2 Add A Readback Scheduler

New component:

```python
scheduler = OpticalReadbackScheduler(
    device=device,
    copy_stream=copy_stream,
    policy=OpticalDeliveryPolicy.ASYNC_ORDERED,
    ring_depth=3,
)
```

Submit:

```python
job = scheduler.submit(
    result,
    channels=("rgb", "hit_mask", "bvh_stack_overflow_count", "shadow_stack_overflow_count"),
)
```

Consume:

```python
completed = scheduler.poll_completed()
```

or:

```python
completed = scheduler.wait_next_ordered()
```

For sync behavior, either:

```python
stage_optical_channels(...)
```

continues to be the reference blocking helper, or the scheduler exposes:

```python
scheduler.readback_blocking(result, channels=...)
```

Recommendation:

```text
Keep existing blocking helpers for tests/debug.
Add scheduler for async video/sensor paths.
```

### 11.3 CLI Shape

Current:

```bash
--video-readback full|rgb|none
```

Proposed addition:

```bash
--video-delivery sync|async-ordered|async-latest|device-only
--readback-ring-depth 3
```

Validation:

```text
--video-readback none requires --video-delivery device-only
--video-delivery device-only requires --video-readback none
--video-delivery async-latest initially errors as design-only
--fail-on-overflow with async delivery checks delayed diagnostics, not current-frame diagnostics
```

Default recommendation for now:

```text
--video-delivery sync
```

because it preserves current behavior. Benchmark experiments can opt into:

```text
--video-delivery async-ordered
```

## 12. First Implementation Scope

### V1: Async Ordered RGB Readback For Video Benchmark

Implement:

```text
OpticalDeliveryPolicy enum
OpticalReadbackScheduler skeleton
async_ordered ring with blocking backpressure
sync helper remains unchanged
video benchmark support for --video-delivery async-ordered
CSV columns:
  delivery_policy
  readback_submit_ms
  readback_wait_ms
  readback_lag_frames
  readback_ring_depth
```

Initial channel support:

```text
rgb float32
hit_mask int32
diagnostic int32 scalars
```

Do not implement yet:

```text
pinned host memory if Warp/Torch interop makes it too large
rgb8 GPU pack
async_latest/drop
arbitrary channel lists in public CLI
```

### Expected V1 Result

For current 960x640 Go2 numbers:

```text
render ~= 2.55 ms
blocking rgb readback ~= 5.07 ms
```

Async ordered steady state should approach:

```text
~max(render, readback) ~= 5.1 ms/frame
```

It will not reach pure render speed until payload/readback is reduced, likely
via GPU `rgb8` pack.

## 13. Follow-Up Scope

### V2: GPU RGB8 Pack

Add a GPU preview-pack channel:

```text
rgb8 uint8 [H, W, 3]
```

This reduces readback bytes:

```text
float32 RGB: ~7.0 MiB at 960x640
uint8 RGB:   ~1.8 MiB at 960x640
```

If readback drops below render time, async ordered throughput should approach
the render kernel cost.

### V3: Async Latest Preview

Implement only when a real-time viewer exists.

Key extra decisions:

```text
drop pending readback vs skip current readback when ring full
how to report dropped frame count
whether diagnostics are required for dropped frames
```

### V4: Sensor Pipeline Integration

Integrate with the formal sensor pipeline rather than the example script.

This needs Q52-style backpressure and ack semantics:

```text
lossless sensor result ack occurs after host staging copy is complete
lossy sensor mode must be explicit
```

## 14. Open Questions For Claude

1. Is the three-axis split correct?

```text
output_profile = what device kernels produce
readback mode  = what host channels are requested
delivery policy = how/when results are delivered
```

Or should readback mode and delivery policy be collapsed for a smaller first
API?

2. Should `OpticalReadbackScheduler` live in `optics/device.py`, a new
`optics/readback.py`, or under a broader sensor/runtime module?

3. For V1, is `async_ordered` enough, or should `sync_blocking` also be routed
through the scheduler immediately?

4. For `async_ordered`, should ring-full behavior block inside `submit(...)`,
or should `submit(...)` return a backpressure status so the caller decides where
to wait?

5. Should diagnostics be staged with RGB for video benchmark by default, or
should diagnostics be a separate optional readback group to reduce per-frame
overhead?

6. Is a normal NumPy host-buffer ring acceptable for V1, or should pinned host
memory be part of the first async implementation?

7. For robot sensor semantics, is `async_ordered/no-drop` sufficient, or do we
need an explicit `lossless`/`lossy` sensor QoS field separate from delivery
policy?

8. Should `--fail-on-overflow` be disallowed for async delivery in V1, or should
it become a delayed check on the next completed frame?

9. Is `async_latest` worth defining now as design-only, or should we defer the
name until a real-time preview UI exists?

10. Does this design preserve the Q52 lifetime rule that published/device
resources are not released before the downstream consumer has a self-contained
staged copy?

