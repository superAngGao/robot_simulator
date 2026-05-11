# Q54 Optical Render API Architecture Plan

Date: 2026-05-10
Author: Codex
Status: review-request

## Goal

Design the next internal optical render API boundary before adding path tracing
or more scene-specific shadow policy.

The immediate goal is not a polished public simulator API. It is a stable
runtime boundary that can carry:

- current direct-light rendering;
- RGB/RGB8/full/render-only output profiles;
- sync and ordered async delivery;
- frame timing/profile diagnostics;
- future path tracing as a new render backend.

## Scope Decision

This plan is about the internal runtime boundary only.

In scope for the next implementation slice:

- request/result dataclasses and enums;
- an internal pipeline/frame entrypoint shape;
- adapting the existing Go2 lab path to the new vocabulary;
- preserving current CLI behavior, CSV schema, and benchmark paths.

Out of scope for the next implementation slice:

- public simulator-facing API freeze;
- path tracing implementation;
- shadow policy / quality profile API;
- generic dynamic-scene session abstraction;
- moving all delivery machinery out of the lab in one step.

## Current State

Existing useful pieces:

- `OpticalOutputProfile` defines compute-side channel contracts:
  `GEOMETRY_FULL`, `DIRECT_LIGHT_FULL`, `RGB_PREVIEW`, `RENDER_ONLY`.
- `OpticalComputeResult` already models device/host result location, channels,
  `ready_event`, and resource ownership.
- GPU executors support direct-light execution from either host rays or a
  pinhole camera.
- The lab now has a `Go2RenderSession` that owns scene/device/stream/frame/cache,
  snapshot, BVH, and executor.
- Delivery experiments exist: sync staging, torch async ring, RGB8 pack, ordered
  completion, timing CSV.

Current friction:

- `Go2RenderSession` is Go2-specific and lives in lab tooling.
- Render settings and delivery settings are still encoded as CLI strings and
  `argparse.Namespace` fields.
- `output_profile`, `readback_payload`, `delivery_policy`, and `write_policy`
  are conceptually separate but still wired together ad hoc.
- `render_profile` is a mutable list of timing tuples rather than an explicit
  diagnostics request/result.
- Async delivery ownership is outside the render session, which is correct for
  now, but the boundary is implicit.
- Path tracing has no natural slot yet; adding it directly to current executor
  calls would mix accumulation semantics with direct-light preview semantics.

## Design Principle

Keep these concepts separate:

```text
render backend
  direct_light | future path_tracing

compute/output profile
  geometry_full | direct_light_full | rgb_preview | render_only
  future path_tracing_preview | path_tracing_accumulation

delivery payload
  none | rgb | rgb8 | full | future encoded/hdr

delivery policy
  device_only | sync_host | torch_async_ordered | future realtime_drop

write policy
  none | png_sequence | future video_encoder
```

Short version:

```text
render precision/output profile != delivery payload precision
```

## Proposed Internal Types

These should start as internal runtime dataclasses in `optics/render_api.py`.
That module should be CPU-safe and dependency-light: dataclasses/enums only, no
Warp, Torch, or Go2 lab imports. The names are deliberately plain; public API
names can be nicer later.

A1 should not re-export these names from `optics/__init__.py` yet. Keep imports
explicit:

```python
from optics.render_api import RenderRequest
```

```python
class RenderBackend(Enum):
    DIRECT_LIGHT = "direct_light"
    PATH_TRACING = "path_tracing"  # reserved, not implemented yet


@dataclass(frozen=True)
class RenderRequest:
    frame_id: int
    sim_time: float
    env_idx: int
    camera: OpticalPinholeCameraSpec | None = None
    rays: OpticalRaySensorSpec | None = None
    use_gpu_raygen: bool = True
    backend: RenderBackend = RenderBackend.DIRECT_LIGHT
    output_profile: OpticalOutputProfile = OpticalOutputProfile.RGB_PREVIEW
    diagnostics: RenderDiagnosticsRequest = RenderDiagnosticsRequest()
    accumulation_id: int | None = None  # reserved for future PATH_TRACING


@dataclass(frozen=True)
class RenderDiagnosticsRequest:
    profile_timing: bool = False
    traversal_counters: bool = False
    fail_on_overflow: bool = True


@dataclass(frozen=True)
class RenderResult:
    compute: OpticalComputeResult
    timing: Mapping[str, float]
    diagnostics: Mapping[str, int | float]
```

Validation rule: exactly one of `camera` or `rays` must be provided. Do not split
`CameraRenderRequest` and `RayRenderRequest` yet; the current lab path needs
both forms, and a single request type keeps A1/A2 small.

Delivery:

```python
class ReadbackPayload(Enum):
    NONE = "none"
    RGB = "rgb"
    RGB8 = "rgb8"
    FULL = "full"


class DeliveryPolicy(Enum):
    DEVICE_ONLY = "device_only"
    SYNC_HOST = "sync_host"
    TORCH_ASYNC_ORDERED = "torch_async_ordered"


@dataclass(frozen=True)
class DeliveryRequest:
    payload: ReadbackPayload
    policy: DeliveryPolicy
    ring_depth: int = 2
    write_policy: WritePolicy = WritePolicy.NONE


@dataclass(frozen=True)
class DeliveryResult:
    frame_index: int
    host_channels: Mapping[str, object]
    device_result: OpticalComputeResult | None
    timing: Mapping[str, float]
    lag_frames: int = 0
    dropped: bool = False
```

`RGB8` should be treated as a delivery payload/post-process bridge, not a render
backend. The render result remains linear device RGB; RGB8 conversion is the
delivery-side preview representation.

Session:

```python
class OpticalRenderSession:
    def render(self, request: RenderRequest) -> RenderResult:
        ...

    def deliver(
        self,
        rendered: RenderResult,
        request: DeliveryRequest,
    ) -> DeliveryResult:
        ...
```

The session owns render resources: device, streams, scene cache, current device
snapshot, BVH/workspace, executors, and render-side scratch buffers. Delivery
resources such as async readback rings may be owned by a separate delivery
object or by a session-owned delivery manager, but the conceptual boundary
should remain visible.

## API Shape For Lab vs Public Users

Lab/internal runtime:

```python
request = RenderRequest(
    camera=camera,
    backend=RenderBackend.DIRECT_LIGHT,
    output_profile=OpticalOutputProfile.RGB_PREVIEW,
    diagnostics=RenderDiagnosticsRequest(profile_timing=True),
)
rendered = session.render(request)
delivered = session.deliver(
    rendered,
    DeliveryRequest(
        payload=ReadbackPayload.RGB8,
        policy=DeliveryPolicy.TORCH_ASYNC_ORDERED,
        ring_depth=2,
    ),
)
```

Future public/simulator-facing API should be policy-oriented:

```text
VIDEO_ORDERED
REALTIME_PREVIEW
SENSOR_ORDERED
PARITY_DEBUG
RENDER_BENCH
```

Those public modes should map onto internal request fields only after the
runtime boundary has proven stable.

## Entry Point

The top-level entry point should be a small facade around session lifecycle, not
the low-level executor itself. Recommended internal name:

```text
OpticalRenderPipeline
```

Do not use `OpticalCameraStream` / `OpticalSensorStream` yet. Those names are
better candidates for the later public API because they imply consumer-facing
modes and policy defaults.

Recommended internal shape:

```python
pipeline = OpticalRenderPipeline.create(
    registry,
    device="cuda:0",
    backend=RenderBackend.DIRECT_LIGHT,
    acceleration=AccelerationConfig(kind="cuda_lbvh"),
    defaults=RenderDefaults(
        output_profile=OpticalOutputProfile.RGB_PREVIEW,
        delivery=DeliveryRequest(
            payload=ReadbackPayload.RGB8,
            policy=DeliveryPolicy.TORCH_ASYNC_ORDERED,
            ring_depth=2,
        ),
    ),
)
```

Per frame:

```python
frame = pipeline.begin_frame(
    OpticalFrameInputs.from_published_frame(published_frame),
    env_idx=0,
)

rendered = frame.render_camera(
    camera,
    diagnostics=RenderDiagnosticsRequest(profile_timing=True),
)

delivered = pipeline.deliver(rendered)
```

Equivalent explicit form:

```python
frame = pipeline.begin_frame(frame_inputs, env_idx=0)
rendered = frame.render(
    RenderRequest(
        camera=camera,
        backend=RenderBackend.DIRECT_LIGHT,
        output_profile=OpticalOutputProfile.RGB_PREVIEW,
        diagnostics=RenderDiagnosticsRequest(profile_timing=True),
    )
)
delivered = pipeline.deliver(
    rendered,
    DeliveryRequest(
        payload=ReadbackPayload.RGB8,
        policy=DeliveryPolicy.TORCH_ASYNC_ORDERED,
        ring_depth=2,
    ),
)
```

This gives three clear lifecycle layers:

```text
OpticalRenderPipeline:
  long-lived runtime: registry binding, device, streams, cache, delivery manager

RenderFrameContext:
  frame-scoped state: frame inputs, device snapshot, BVH/refit state

RenderRequest / DeliveryRequest:
  per-camera or per-ray work item and output transport choice
```

Why not expose executor calls directly?

- executors know how to compute, not how to update scene state or deliver;
- path tracing will need session/frame state for accumulation;
- async delivery needs ordered lifecycle independent from render compute;
- callers should not manually pair snapshot/BVH/frame IDs for normal use.

The frame-scoped object name is intentionally not final. Claude review preferred
`RenderFrameContext` or `FrameHandle` over `OpticalRenderFrame`; use
`RenderFrameContext` as the A2 working name unless implementation pressure says
otherwise.

The lab can initially implement this facade by wrapping the existing
`Go2RenderSession`; a later generic implementation can move to `optics/` once
dynamic scenes and multiple presets prove the boundary.

Minimal public-ish call shape later:

```python
camera_stream = optical.create_camera_stream(
    registry,
    device="cuda:0",
    mode=OpticalMode.VIDEO_ORDERED,
    resolution=(1920, 1080),
)

reading = camera_stream.render(published_frame, camera)
```

That public mode would expand internally into render/delivery requests. It
should wait until the internal pipeline facade has survived the lab workloads.

## Path Tracing Slot

Path tracing should enter as a render backend, not as a delivery mode:

```text
RenderBackend.PATH_TRACING
PathTracingSettings(samples_per_frame, max_bounces, accumulation_id, reset)
OpticalOutputProfile.PATH_TRACING_PREVIEW
OpticalOutputProfile.PATH_TRACING_ACCUMULATION
```

Delivery remains reusable:

```text
path tracing accumulation -> rgb8 preview delivery
path tracing debug        -> full host delivery
direct light preview      -> rgb8 preview delivery
```

This keeps async readback, RGB8 pack, ordering, and future encoders shared
between direct-light and path-tracing paths.

Path-tracing accumulation state should be session-owned but addressed by an
explicit `accumulation_id` in a future `PathTracingSettings`. That avoids hiding
long-lived GPU buffers inside individual requests while still letting callers
reset or switch accumulation streams intentionally.

For A1, include `accumulation_id: int | None = None` as a reserved field on
`RenderRequest` with documentation that it has no effect for
`RenderBackend.DIRECT_LIGHT`.

## Proposed Implementation Stages

### A0: Design Only

Land this architecture note and review it. Do not implement public API yet.

### A1: Internal Dataclasses In `optics/render_api.py`

Add internal dataclasses/enums in `optics/render_api.py`. Keep them
dependency-light and do not move existing executors yet.

Acceptance:

- imports are CPU-safe;
- no `optics/__init__.py` re-export yet;
- dataclasses cover current lab fields;
- unit tests validate default mappings and invalid combinations.
- `RenderRequest` validates exactly one of `camera` or `rays`;
- `RenderRequest.accumulation_id` exists as a path-tracing-reserved field;
- `DeliveryRequest` validates ring depth and payload/policy compatibility.

### A2: Adapter Around `Go2RenderSession`

Teach lab `Go2RenderSession` to accept `RenderRequest` internally while keeping
CLI behavior unchanged. This can be a thin adapter method first; do not promote
`Go2RenderSession` to a generic session yet.

Acceptance:

- existing lab runs produce equivalent frame CSV rows;
- no GPU benchmark regression beyond noise;
- `render_execute_ms`, `pack_rgb8_ms`, traversal counters still work.
- old CLI flags still map to the same readback/output behavior.

### A3: Delivery Boundary Adapter

Move sync staging, RGB8 pack, and torch async ring orchestration behind
`DeliveryRequest` / `DeliveryResult`.

Acceptance:

- sync and torch_async code paths share request/result vocabulary;
- ring ownership remains explicit;
- no change to output files or CSV schema except intentional metadata.
- RGB8 pack remains measured as `pack_rgb8_ms`.

### A4: Generic Session Candidate

Only after A2/A3, decide whether to promote `Go2RenderSession` into a generic
`OpticalRenderSession` or keep a Go2-specific adapter.

Do not force this too early; scene lifecycle and dynamic geometry will shape the
right abstraction.

## Default Answers For Review

Codex recommendation:

```text
module:
  optics/render_api.py
  do not re-export from optics/__init__.py in A1

request shape:
  one RenderRequest with exactly-one-of camera/rays validation

diagnostics:
  RenderDiagnosticsRequest belongs in core render_api.py
  implementation-specific counters may remain optional and experimental

RGB8:
  delivery/post-process bridge, not render backend

entrypoint:
  OpticalRenderPipeline for internal facade
  OpticalCameraStream / OpticalSensorStream deferred to public API stage

path tracing accumulation:
  session-owned state addressed by explicit accumulation_id later
  RenderRequest includes reserved accumulation_id in A1

next implementation:
  A1 + A2 only
```

## Interaction With E2 Shadow Work

The E2.2 traversal-counter probe is intentionally separate from this API plan.
It can remain as lab diagnostics while API work proceeds.

Do not turn the current Go2 shadow observations into `ShadowPolicy` yet. More
scene coverage should arrive first through the pipeline/API work; then shadow
quality policy can be designed from multiple scenes instead of one Go2 setup.

## Review Questions

1. Do you agree that A1 should live in `optics/render_api.py`, not under
   `tools/optical_pipeline_lab`?
2. Do you agree with one `RenderRequest` plus exactly-one-of `camera`/`rays`,
   rather than separate request classes now?
3. Do you agree that `RenderDiagnosticsRequest` belongs in the internal core
   API, while traversal counters remain optional implementation diagnostics?
4. Do you agree RGB8 should be modeled as delivery/post-process bridge rather
   than render backend?
5. Do you agree the top-level internal entrypoint should be
   `OpticalRenderPipeline`, with `OpticalCameraStream` deferred to public API?
6. Do you agree path-tracing accumulation should later be session-owned and
   addressed by explicit `accumulation_id`?
7. Do you agree the next implementation should stop at A1/A2, leaving A3
   delivery refactor and path tracing for later?

## Review Result

Claude review: PASS / green light for A1.

Accepted follow-ups:

```text
1. Do not re-export render_api names from optics/__init__.py in A1.
2. Keep one RenderRequest with exactly-one-of camera/rays validation.
3. Include accumulation_id: int | None = None as a reserved path-tracing field.
4. Keep traversal diagnostics in RenderResult.diagnostics Mapping, not typed API.
5. Treat RGB8 as delivery bridge.
6. Use OpticalRenderPipeline for the internal entry point.
7. Prefer RenderFrameContext / FrameHandle naming over OpticalRenderFrame.
8. Stop at A1/A2; defer A3 delivery refactor and path tracing.
9. Track CPU-safety of normalize_output_profile as a render_api dependency.
10. Document A1/A2's exact sim_time comparison as internal-only for now.
11. Track render_profile argument plumbing as an A3 cleanup item.
12. Track traversal_counters/profile_timing coupling as a future session cleanup.
```

## Recommendation

Start with A1/A2 only:

```text
internal request/result dataclasses
Go2RenderSession adapter
OpticalRenderPipeline shape documented but not fully generic yet
no public API freeze
no path tracing yet
```

This gives path tracing a clean future slot without committing today to the
final simulator-facing API.
