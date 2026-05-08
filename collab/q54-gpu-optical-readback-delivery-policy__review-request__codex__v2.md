Initiative: q54-gpu-optical-readback-delivery-policy
Stage: review-request
Author: codex
Version: v2
Date: 2026-05-08
Status: pending-review
Related Files: optics/execution.py, optics/device.py, optics/warp_execution.py, examples/mujoco_menagerie_gpu_preview.py, collab/q54-gpu-optical-readback-delivery-policy__review-request__codex__v1.md
Owner Summary: This revision corrects the v1 framing. Output profile, readback payload, and delivery policy are not equal public axes. The user-facing API should be consumer-contract first, while implementation should be organized delivery-first because synchronization, resource lifetime, ring buffers, and backpressure are delivery concerns.

# Q54 GPU Optical Readback Delivery Policy Review Request v2

## 1. Why v2

The v1 proposal described:

```text
output_profile
readback mode
delivery policy
```

as three separate axes. That was useful for naming the moving parts, but it is
not the right public architecture.

These concepts are not peers:

```text
consumer intent / contract
  dominates delivery semantics
    which dominates lifetime and synchronization
      which selects output/readback representation
```

The corrected split is:

```text
User-facing API:
  consumer-first

Implementation organization:
  delivery-first
```

## 2. Core Principle

Users should not start by choosing:

```text
sync vs async
ordered vs latest
drop vs no-drop
ring depth
copy stream behavior
ready event ownership
```

Most users know the consumer they are building:

```text
parity/debug
video recording
real-time preview
robot sensor stream
render benchmark
```

So the public API should expose consumer contracts.

Internally, however, code should be organized around delivery policy because
delivery owns the hard mechanics:

```text
where synchronization happens
who keeps device buffers alive
when host staging slots can be reused
what happens when the ring is full
how backpressure is applied
whether dropping is legal
```

## 3. Layered Model

Recommended conceptual layering:

```text
Layer 1: Consumer Contract
  What semantic guarantee does the caller want?

Layer 2: Delivery Runtime
  How are rendered device results delivered to the consumer?

Layer 3: Representation
  What device outputs and host payloads are needed?

Layer 4: Kernel/Copy Implementation
  Which kernels, buffers, copy streams, and staging slots implement it?
```

This replaces the v1 "three equal axes" framing.

## 4. Layer 1: Consumer Contract

Introduce a user-facing mode enum, names still open:

```python
class OpticalConsumerMode(Enum):
    PARITY_DEBUG = "parity_debug"
    RENDER_BENCH = "render_bench"
    VIDEO_ORDERED = "video_ordered"
    REALTIME_PREVIEW = "realtime_preview"
    SENSOR_LOSSLESS = "sensor_lossless"
```

Each mode defines defaults for:

```text
delivery policy
drop/backpressure behavior
default output profile
default readback payload
diagnostics policy
host consumption ordering
```

Advanced callers may eventually override representation details, but the normal
entry point should be consumer mode.

## 5. Consumer Modes

### 5.1 PARITY_DEBUG

Purpose:

```text
unit tests
parity checks
debug captures
single-frame inspection
```

Contract:

```text
current frame result is complete
host result is immediately available
no frame drop
errors surface close to the call site
```

Defaults:

```text
delivery_policy = sync_blocking
output_profile  = direct_light_full or geometry_full
readback_payload = full
diagnostics = required
drop = no
```

Implementation expectation:

```text
existing stage_optical_compute_result_to_host(...)
existing stage_optical_channels(...)
remain valid reference paths
```

### 5.2 RENDER_BENCH

Purpose:

```text
measure device render cost without host readback
kernel regression tracking
GPU-only downstream smoke
```

Contract:

```text
no host frame result
ready_event is meaningful
diagnostics may exist on device but are not synchronously read unless requested
```

Defaults:

```text
delivery_policy = device_only
output_profile  = render_only
readback_payload = none
diagnostics = device-side only by default
drop = not applicable
```

Implementation expectation:

```text
this is current --video-readback none semantics
```

### 5.3 VIDEO_ORDERED

Purpose:

```text
offline video recording
video benchmark with every frame included
throughput benchmark that still preserves a complete sequence
```

Contract:

```text
every requested frame is eventually delivered
frame order is preserved
latency of individual frames may increase
producer may block under backpressure
no silent frame drop
```

Defaults:

```text
delivery_policy = async_ordered
output_profile  = rgb_preview
readback_payload = rgb, hit_mask, diagnostics
diagnostics = required, but checked on completed frames
drop = no
backpressure = block when ring full
```

Expected performance model:

```text
blocking:
  frame ~= render + readback

async ordered steady-state:
  frame ~= max(render, readback)
```

For current Go2 960x640 numbers:

```text
render ~= 2.55 ms
blocking rgb readback ~= 5.07 ms

async ordered should approach ~= 5.1 ms/frame
```

It will not approach pure render time until readback payload is reduced, likely
via GPU `rgb8` pack.

### 5.4 REALTIME_PREVIEW

Purpose:

```text
interactive viewer
operator preview
low-latency visual feedback
```

Contract:

```text
show newest completed frame
old frames may be skipped
low display latency is more important than complete sequence
```

Defaults:

```text
delivery_policy = async_latest
output_profile  = rgb_preview
readback_payload = rgb8 preferred
diagnostics = optional / sampled
drop = yes
backpressure = drop or skip readback, not block render loop
```

Implementation status:

```text
define semantics now
do not implement V1 unless a real-time viewer needs it
```

Open issue:

```text
already-enqueued CUDA copies cannot be trivially canceled, so the drop strategy
must be chosen carefully:
  skip submitting new readback when no slot is free
  or ignore old completed frames
```

### 5.5 SENSOR_LOSSLESS

Purpose:

```text
robot sensor simulation
training data
frame-aligned optical observations
```

Contract:

```text
sensor result corresponds to a specific simulation frame
no silent frame drop
ordered delivery
backpressure is allowed and expected if consumers fall behind
ack point must mean staged result is self-contained
```

Defaults:

```text
delivery_policy = async_ordered
output_profile  = sensor-specific, geometry_full, or rgb_preview depending on sensor
readback_payload = consumer-required channels
diagnostics = required
drop = no unless user explicitly opts into lossy sensor QoS
backpressure = block / signal pressure
```

Relationship to Q52:

```text
This should preserve the existing lossless snapshot principle:
ack only after host staging has a complete self-contained copy.
If async copy is involved, copy_ready_event must have completed before ack.
```

## 6. Layer 2: Delivery Runtime

Although public API is consumer-first, implementation should be delivery-first:

```python
class OpticalDeliveryPolicy(Enum):
    SYNC_BLOCKING = "sync_blocking"
    DEVICE_ONLY = "device_only"
    ASYNC_ORDERED = "async_ordered"
    ASYNC_LATEST = "async_latest"
```

Delivery policy owns:

```text
event waits
copy stream usage
readback slot ownership
ring full behavior
device resource retention
host buffer release
drop legality
backpressure
```

This is why development should start with delivery primitives rather than with
public consumer presets.

## 7. Layer 3: Representation

Representation is selected by the consumer mode but remains separately modeled
for implementation.

### 7.1 OutputProfile

Executor-facing:

```text
GEOMETRY_FULL
DIRECT_LIGHT_FULL
RGB_PREVIEW
RENDER_ONLY
```

Answers:

```text
what device kernels write as public channels
which device result channels are guaranteed
```

Does not answer:

```text
whether host readback blocks
whether frames can drop
whether a ring is used
```

### 7.2 ReadbackPayload

Host-facing:

```text
none
full
rgb
rgb8 future
sensor-specific channel groups future
```

Answers:

```text
which channels are copied to host
what host dtype/layout the consumer receives
```

Does not answer:

```text
whether copy is sync or async
whether frames can drop
```

## 8. Layer 4: Implementation Objects

### 8.1 DeviceFrameResult

Equivalent to current device `OpticalComputeResult`:

```text
device channels
output_profile
render_ready_event
resources retaining device buffers
frame metadata
```

Rule:

```text
must remain alive until all async copies from its device buffers complete
```

### 8.2 ReadbackSlot

Reusable host-side staging slot:

```text
host buffers
copy_ready_event
state: free | copying | ready | borrowed
frame metadata
```

V1 can start with normal host arrays if pinned memory is too much, but the
design should allow pinned host buffers.

### 8.3 ReadbackJob

Returned by non-blocking submit:

```text
frame metadata
sequence number
slot
requested payload
copy_ready_event
retained device resources
```

The job bridges device lifetime and host slot lifetime.

### 8.4 CompletedFrame

Host-consumable result:

```text
frame metadata
host channel views
diagnostics
release/borrow protocol
```

Prefer a context-manager borrow:

```python
with completed.borrow_host() as host_frame:
    ...
```

so host slots cannot be accidentally reused while a consumer still reads them.

## 9. Synchronization And Release Semantics

### 9.1 Render Ready

```text
render_ready_event means all guaranteed device channels for output_profile have
been written.
```

### 9.2 Submit Async Readback

On copy stream:

```text
copy_stream.wait_event(render_ready_event)
enqueue D2H copies into slot
copy_ready_event = copy_stream.record_event()
```

Submit should not wait for copy completion unless applying backpressure.

### 9.3 Consume

Host consumer may read a slot only after:

```text
copy_ready_event completed
```

For ordered delivery:

```text
return completed frames in sequence order
```

For latest delivery:

```text
return newest completed frame
older frames may be discarded by policy
```

### 9.4 Release Device Resources

Device resources retained by the job may be released after:

```text
copy_ready_event completed
```

They do not need to stay alive until host consumption finishes because the host
slot then has its own copy.

### 9.5 Release Host Slot

Host slot may be reused only after:

```text
copy_ready_event completed
host consumer has released/closed/returned the completed frame
```

## 10. Backpressure Semantics

### 10.1 sync_blocking

No ring:

```text
render
wait render
readback/materialize
return host result
```

### 10.2 device_only

No readback:

```text
render
optional wait for timing
return or discard device result
```

### 10.3 async_ordered

Ring:

```text
if free slot exists:
  submit immediately
else:
  block on oldest pending job
  make its completed frame available / release consumed slot
  then submit
```

No drop.

Open implementation decision:

```text
Should submit(...) block on ring full,
or should it return NEEDS_DRAIN / BACKPRESSURE so caller decides where to wait?
```

Recommendation for V1:

```text
blocking submit is acceptable for the video benchmark,
but the lower-level scheduler should expose enough state to support explicit
drain later.
```

### 10.4 async_latest

Ring:

```text
if free slot exists:
  submit readback
else:
  do not block render loop
  skip submitting current readback or mark old unconsumed frame as dropped
```

V1 should not implement this, only reserve the semantics.

## 11. Proposed Public API Shape

### 11.1 High-Level Consumer-First API

Future sensor/video runtime should prefer:

```python
consumer = OpticalConsumerConfig(
    mode=OpticalConsumerMode.VIDEO_ORDERED,
    width=960,
    height=640,
)
```

or:

```python
optical_runtime.attach_consumer(
    sensor_id="front_rgb",
    mode=OpticalConsumerMode.SENSOR_LOSSLESS,
)
```

The consumer mode selects defaults.

Advanced override should be possible but explicit:

```python
OpticalConsumerConfig(
    mode=OpticalConsumerMode.VIDEO_ORDERED,
    payload=ReadbackPayload.RGB8,
    ring_depth=3,
)
```

### 11.2 Low-Level Delivery Runtime

Implementation can expose:

```python
scheduler = OpticalReadbackScheduler(
    device=device,
    copy_stream=copy_stream,
    delivery_policy=OpticalDeliveryPolicy.ASYNC_ORDERED,
    payload=ReadbackPayload.RGB,
    ring_depth=3,
)
```

Submit:

```python
job = scheduler.submit(device_result)
```

Drain:

```python
completed = scheduler.poll_completed()
completed = scheduler.wait_next_ordered()
```

### 11.3 Existing Blocking Helpers

Keep:

```python
stage_optical_channels(...)
stage_optical_compute_result_to_host(...)
```

as sync/debug/test helpers.

Do not force all sync paths through the scheduler in V1.

## 12. CLI Shape For The Current Video Benchmark

Current:

```bash
--video-readback full|rgb|none
```

For experiments, add:

```bash
--video-consumer render_bench|video_ordered|parity_debug
```

or the more explicit low-level flag:

```bash
--video-delivery sync|async-ordered|device-only
```

Recommendation:

```text
Use --video-delivery first in the benchmark script because it is an engineering
benchmark, not the final user-facing API.

Reserve --video-consumer for the later formal optical runtime.
```

Validation:

```text
readback=none requires delivery=device-only
delivery=device-only requires readback=none
delivery=async-ordered supports readback=rgb initially
delivery=async-latest is design-only and errors if requested
fail-on-overflow with async delivery becomes delayed-frame diagnostics
```

## 13. First Implementation Plan

### V1: Delivery Runtime Skeleton + Async Ordered Video Benchmark

Implement:

```text
OpticalDeliveryPolicy enum
ReadbackPayload enum or narrow payload identifiers
OpticalReadbackScheduler
ReadbackSlot ring
ReadbackJob
CompletedFrame borrow/release
async_ordered delivery for RGB preview payload
video benchmark --video-delivery async-ordered
CSV fields for delivery timing
```

Do not implement:

```text
async_latest
rgb8 pack
sensor runtime integration
arbitrary public channel lists
full parity path through scheduler
```

Initial supported payload:

```text
RGB_PREVIEW device result:
  rgb float32 [H, W, 3]
  hit_mask int32 [H, W]
  diagnostics int32 scalars
```

CSV additions:

```text
consumer_mode
delivery_policy
readback_payload
readback_submit_ms
readback_wait_ms
readback_lag_frames
readback_ring_depth
readback_ring_block_count
completed_frame_index
```

Expected behavior:

```text
sync rgb:
  frame ~= render + readback

async_ordered rgb:
  steady-state frame ~= max(render, readback)
```

Current target:

```text
Go2 960x640:
  render ~= 2.55 ms
  blocking rgb readback ~= 5.07 ms
  expected async ordered ~= 5.1 ms/frame before rgb8
```

### V2: GPU RGB8 Payload

Add:

```text
ReadbackPayload.RGB8
GPU clip/gamma/uint8 pack
host uint8 RGB buffer
```

Expected benefit:

```text
float32 RGB payload ~= 7.0 MiB at 960x640
uint8 RGB payload   ~= 1.8 MiB at 960x640
```

If readback falls below render time:

```text
async_ordered throughput should approach render cost
```

### V3: Formal Consumer API

Once delivery runtime is validated in the benchmark:

```text
introduce OpticalConsumerMode
move benchmark flags toward consumer presets
connect sensor/runtime code to SENSOR_LOSSLESS semantics
```

### V4: Realtime Preview Latest Mode

Implement only when a viewer exists:

```text
async_latest
drop/skip semantics
newest completed frame selection
dropped frame counters
```

## 14. Risks

### 14.1 Naming Confusion

Risk:

```text
output_profile, readback_payload, delivery_policy, consumer_mode may feel like
too many knobs.
```

Mitigation:

```text
public docs lead with consumer_mode
benchmark/internal docs may expose delivery_policy
output_profile/readback_payload documented as representation details
```

### 14.2 Async Readback Without True Async Copy

Risk:

```text
Warp .numpy() is synchronous and cannot implement async_ordered.
```

Mitigation:

```text
V1 scheduler must use a copy mechanism that can enqueue D2H copy on a copy
stream, or explicitly report that it is a structural mock with no expected
overlap.
```

This is the biggest implementation spike.

### 14.3 Pinned Host Memory

Risk:

```text
normal NumPy host buffers may not support true async D2H copy.
```

Mitigation:

```text
Evaluate pinned host memory in the scheduler spike.
If pinned memory is required, make it internal to ReadbackSlot.
```

### 14.4 Diagnostics Break Overlap

Risk:

```text
fail-on-overflow or per-frame diagnostics can force immediate scalar readback.
```

Mitigation:

```text
async policies check diagnostics on completed frames, not current in-flight
frames.
For strict immediate failure, use PARITY_DEBUG / sync_blocking.
```

### 14.5 Lifetime Bugs

Risk:

```text
device result GC before async copy finishes
host slot reused while consumer still reads
```

Mitigation:

```text
ReadbackJob retains device resources until copy_ready_event completes.
CompletedFrame borrow/release controls host slot reuse.
Tests should explicitly stress delayed release.
```

## 15. Review Questions For Claude

1. Is the corrected two-level framing right?

```text
public API consumer-first
implementation delivery-first
```

2. Are the proposed consumer modes the right set?

```text
PARITY_DEBUG
RENDER_BENCH
VIDEO_ORDERED
REALTIME_PREVIEW
SENSOR_LOSSLESS
```

3. Should the current video benchmark expose `--video-delivery` as a low-level
engineering flag, or jump directly to `--video-consumer` presets?

4. Is it correct to keep the existing blocking staging helpers outside the
scheduler for V1?

5. Should `async_ordered` ring-full behavior block inside `submit(...)`, or
return an explicit backpressure status?

6. Is `CompletedFrame.borrow_host()` the right lifetime API, or is explicit
`release()` simpler for the first implementation?

7. Should diagnostics be mandatory in `VIDEO_ORDERED`, or optional to reduce
payload and avoid delayed overflow checks?

8. Is normal host memory acceptable for a first structural scheduler, or should
pinned host memory be required for V1 because otherwise overlap cannot be
measured honestly?

9. Which copy backend should be spiked first for async D2H?

```text
Warp-native copy if available
Torch tensor pinned-memory copy
custom CUDA extension copy
```

10. Does the proposed `SENSOR_LOSSLESS` mode preserve the Q52/Q54 staged-copy
ack semantics strongly enough?

11. Should `REALTIME_PREVIEW` be defined now as design-only, or omitted until a
viewer exists?

12. Are `output_profile` and `readback_payload` sufficiently separated, or is
there still a risk that users will treat representation knobs as semantic
consumer guarantees?

