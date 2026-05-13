# Q54 RenderSession / Workspace Stage I Plan

Date: 2026-05-13
Author: Codex
Status: review-request

## Context

The original stage skeleton put the runtime work before full dynamic geometry:

```text
Stage F: Minimal RenderSession / Workspace Skeleton
Stage G: Async Ordered Delivery For Video Export
Stage H: GPU RGB8 Pack
Stage I: Render Session / Workspace Refactor
Stage J: Dynamic Geometry Pipeline
```

We have now completed most of the lab-local proof points that Stage F/G/H
needed:

```text
Go2RenderSession
Go2RenderPipeline / Go2RenderFrameContext
RenderRequest / RenderResult
VideoDeliveryFacade
OpticalDeliveryRuntime protocol
DeliveryResult bridge
RenderedVideoFrame envelope
go2_video_delivery_smoke GPU validation
```

The next question is not whether dynamic Go2 visual motion matters. It does.
The question is whether to first harden the runtime ownership boundary so the
future dynamic geometry work does not pile more stream/ring/workspace ownership
into `tools/optical_pipeline_lab`.

Recommendation: start Stage I with a small workspace/session ownership plan,
not with a broad production API move.

## Current Ownership Inventory

### Go2RenderSession

Currently owns:

```text
scene
device
stream
base gpu_frame
DeviceOpticalSceneCache
static snapshot
static BVH
direct-light executor
bvh_backend / split_strategy
RGB8 pack helper
```

Good:

- long-lived resource bundle is explicit;
- scene/cache/snapshot/BVH/executor are no longer scattered through the loop;
- dynamic `begin_frame(frame_inputs)` can create frame-specific snapshot/BVH.

Still lab-local:

- class lives in `tools/optical_pipeline_lab/go2_backend.py`;
- construction reads lab/example args;
- no reusable workspace object exists;
- only one Warp stream is clearly owned;
- no formal render/copy stream split.

### Go2RenderFrameContext

Currently owns per-frame:

```text
session reference
env_idx
optional frame-specific snapshot
optional frame-specific BVH
prepare_timing
```

Good:

- static and dynamic frame boundaries are explicit;
- `render(RenderRequest) -> RenderResult` is already the right internal shape;
- frame-specific snapshot/BVH references survive until render completes.

Still missing:

- no context lifetime protocol beyond Python object lifetime;
- no future accumulation/workspace hooks;
- no explicit event dependency object.

### RenderedVideoFrame

Currently owns video-loop envelope data:

```text
frame_index
camera
result
camera_rays_ms
render_execute_ms fallback
render_profile_row
include_shadow_traversal_stats
geometry_mode
prepare_timing
render: RenderResult | None
```

Good:

- lab video metadata stays out of `RenderResult`;
- production path carries the runtime `RenderResult`;
- `render_execute_ms_value()` now reads runtime render timing first.

Still transitional:

- `result` is still a stored compatibility field;
- `render_profile_row` is still a stored flattened CSV dict;
- this is intentionally a lab envelope, not a runtime API type.

### VideoDeliveryFacade

Currently owns:

```text
DeliveryRequest
delivery_policy_label
frame_dir
RGB8 pack callback
synchronize_event callback
TorchAsyncReadbackRing
pending async job
ready queues
first/last frame completion timing
latest rendered frame index
```

Good:

- `submit + complete_available + flush` exposes blocking points;
- RGB8 pack, readback, image/write timing are delivery-owned;
- sync/async CSV rows share one loop body;
- async completion identity is explicit.

Still lab-local:

- depends on lab `RenderedVideoFrame`;
- writes PNGs through lab/consumer adapter logic;
- owns progress/CSV-adjacent concepts indirectly;
- Torch async ring is not yet a generic runtime ring/workspace.

### Matrix / Lab Reporting

Currently owns:

```text
frame_timing.csv schema
matrix_summary.csv schema
progress lines
rolling FPS
overflow fail policy
smoke suites
```

This should remain lab-local. It is not a candidate for `optics/`.

## Stage I Goal

Centralize runtime resource ownership without freezing public API:

```text
render/copy streams
workspace allocation hooks
readback ring ownership hooks
frame context lifecycle
delivery runtime creation
```

Do this while keeping:

```text
lab CLI stable
frame_timing.csv stable
go2_video_delivery_smoke stable
dynamic synthetic smoke stable
public OpticalCameraStream deferred
```

## Proposed Stage I Slices

### I0 — Ownership Inventory And Names

This document.

Review the current ownership map and decide names before moving code.

Candidate terms:

```text
OpticalRenderWorkspace:
  buffers, streams, reusable scratch/rings

OpticalRenderSession / Pipeline:
  scene/cache/accel/executor lifecycle

OpticalDeliveryRuntime:
  submit/complete/flush delivery execution
```

### I1 — Extract Go2 Session Module Boundary

Move the concrete Go2 session/pipeline/frame context out of the large backend
file into a nearby lab module, for example:

```text
tools/optical_pipeline_lab/go2_session.py
```

Move:

```text
Go2RenderSession
Go2RenderFrameContext
Go2RenderPipeline
```

Keep in `go2_backend.py`:

```text
CLI parsing
lab scenario orchestration
video loop
matrix-facing backend helpers
row/progress behavior
```

Why this first:

- reduces `go2_backend.py` size before deeper ownership work;
- creates a clearer import boundary for future workspace extraction;
- still keeps everything lab-local.

Non-goal:

- do not move these classes to `optics/` yet.

### I2 — Introduce Lab Render Workspace Object

Add a small lab-local workspace object:

```python
@dataclass
class Go2RenderWorkspace:
    device: object
    render_stream: object
    copy_stream: object | None = None
```

Initial conservative behavior:

```text
render_stream == existing wp.Stream
copy_stream == None
Torch async readback still owns its own mechanics
```

The point is not immediate performance. The point is to stop passing unnamed
stream/device resources around as loose session fields.

Later this workspace can grow:

```text
readback ring slots
temporary render buffers
RGB8 pack scratch
copy stream/event dependencies
allocator policy
```

### I3 — Delivery Runtime Factory Hook

Implement a lab-local pipeline hook:

```python
Go2RenderPipeline.create_delivery_runtime(request) -> VideoDeliveryFacade-like object
```

But do not force `VideoDeliveryFacade` to implement generic
`OpticalDeliveryRuntime` yet unless the video envelope question is resolved.

Conservative option:

```python
def create_video_delivery(...):
    return VideoDeliveryFacade.create(...)
```

or:

```python
def create_delivery_runtime(...):
    raise NotImplementedError
```

until workspace/ring ownership is clearer.

Review decision needed: whether I3 should create a real object or stay deferred
after I1/I2.

### I4 — Workspace-Aware Delivery Ring

Only after I2/I3:

- decide whether `TorchAsyncReadbackRing` belongs to delivery runtime or
  workspace;
- make ring construction accept workspace-owned stream/copy policy if needed;
- keep CSV output unchanged.

This is the first slice that may affect async delivery mechanics, so it should
come after module/workspace boundaries are explicit.

## Recommended First Implementation

Start with I1 only:

```text
extract Go2RenderSession / Go2RenderFrameContext / Go2RenderPipeline
to tools/optical_pipeline_lab/go2_session.py
```

Do not change behavior.

Do not add workspace yet.

Do not fill `create_delivery_runtime(...)`.

Rationale:

- It is a pure boundary cleanup with low semantic risk.
- It makes the later workspace diff easier to review.
- It keeps all Go2-specific code in `tools/optical_pipeline_lab`.
- It avoids mixing module extraction with resource ownership changes.

## Non-Goals

- No public `OpticalRenderSession`.
- No public `OpticalCameraStream`.
- No dynamic Go2 visual importer change.
- No new path tracing API.
- No CSV schema changes.
- No delivery policy label changes.
- No async scheduler redesign in the first slice.
- No `result` / `render_profile_row` envelope migration in the same patch.

## Validation

For I1:

```text
python -m pytest -q \
  tests/unit/optics/test_render_api.py \
  tests/unit/optics/test_optical_pipeline_lab.py

ruff check tools/optical_pipeline_lab \
  tests/unit/optics/test_render_api.py \
  tests/unit/optics/test_optical_pipeline_lab.py
```

GPU validation does not need to run for a pure module extraction unless review
requests it. The recent R4 GPU smoke already validates behavior after the
delivery/envelope changes.

For I2/I3/I4:

```text
go2_video_delivery_smoke
```

should be rerun on GPU if stream/ring/workspace behavior changes.

## Risks

### Import Cycles

`go2_session.py` will need helpers currently in `go2_backend.py`, such as scene
construction, RGB8 pack, render profile row construction, and dynamic BVH
builders.

Mitigation:

- either keep helper functions in `go2_backend.py` and pass callbacks into the
  session;
- or move only the truly session-owned helpers with the classes;
- avoid importing all of `go2_backend.py` from `go2_session.py`.

This is the main reason I1 should be reviewed carefully.

### Premature Generic Runtime

Moving Go2 classes to a separate module can look like promotion to production.
It is not. Keep module under `tools/optical_pipeline_lab` and document it as
lab-local.

### Workspace Overreach

A workspace object can become a dumping ground. I2 should start with stream and
device only, then grow only when a concrete resource needs ownership.

### Dynamic Geometry Pull-Forward

Do not mix Go2 body-bound importer work into Stage I. That belongs to Stage J
after the session/workspace boundary is stable.

## Review Questions

1. Is I1 module extraction the right first Stage I slice, or should we introduce
   `Go2RenderWorkspace` before splitting files?

2. Should the extracted module be named `go2_session.py`,
   `go2_pipeline.py`, or something else?

3. Which helpers should move with the classes during I1?

4. Should `create_delivery_runtime(...)` remain a stub until after workspace
   exists, or should I1/I2 make it create a lab delivery facade?

5. Is `Go2RenderWorkspace` the right name for the lab-local workspace object,
   or should this stay unnamed until stream/copy split is real?

6. Should R3b deferred envelope cleanup stop here while Stage I starts?

## Codex Recommendation

```text
Q1: start with I1 module extraction
Q2: use go2_session.py if it contains session + pipeline + frame context
Q3: move only helpers needed to avoid import cycles; leave video loop helpers
    in go2_backend.py
Q4: keep create_delivery_runtime(...) as a stub for I1
Q5: introduce Go2RenderWorkspace in I2, not I1
Q6: yes, stop R3b cleanup for now
```

The key posture: make ownership boundaries reviewable before changing
ownership behavior.
