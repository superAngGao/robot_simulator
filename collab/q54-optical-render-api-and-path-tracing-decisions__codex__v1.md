# Q54 Optical Render API + Path Tracing Decisions

Date: 2026-05-10
Author: Codex
Status: decision-note

## Decision 1 — Path Tracing Placement

Path tracing is not part of the current direct-light optimization loop.

Keep the current E-series focused on:

- RenderSession/workspace boundary
- render timing/profile semantics
- buffer allocation/reuse
- shade and first-hit optimization
- ordered RGB8 delivery

Start path tracing later as a Q54-PT branch after the RenderSession/workspace
boundary can carry multiple render backends.

Recommended staging:

```text
PT0 CPU reference path tracer
PT1 GPU stochastic single-bounce / AO-style preview
PT2 progressive accumulation buffers and reset semantics
PT3 small multi-bounce diffuse path tracing
PT4 materials, denoise, and temporal/progressive delivery
```

Path tracing should be a render backend/output-profile family:

```text
render_backend = direct_light | path_tracing
compute payload = linear rgb / radiance / accumulation
delivery payload = rgb8 / rgb / future hdr/encoded video
```

This keeps path tracing from changing the delivery API. RGB8, async ordered
readback, rings, and future encoders should remain reusable delivery mechanisms.

## Decision 2 — API Design Timing

After E0/E1, API work is appropriate, but it should target the internal
RenderSession/Delivery boundary first. Do not freeze the public simulator-facing
camera/render API yet.

Near-term internal vocabulary:

```text
RenderSession:
  owns device, streams, workspace, scene cache, snapshot, BVH, executor

RenderRequest:
  camera/rays, output_profile, render_backend, diagnostics/profile flags

RenderResult:
  device-side result, ready_event, output_profile, resources

DeliveryRequest:
  readback_payload, delivery_policy, write_policy, ring depth

DeliveryResult:
  host/device delivered payload, timing, lag/drop/backpressure metadata
```

Candidate internal shape:

```python
session = OpticalRenderSession.create(scene_config, device="cuda:0")
rendered = session.render(request)
delivered = session.deliver(rendered, delivery_request)
```

The key API principle:

```text
render precision/output profile != delivery payload precision
```

Examples:

```text
direct-light compute -> linear float RGB -> rgb8 delivery
future path tracing -> radiance/accumulation -> rgb8 preview delivery
sensor debug -> geometry/full diagnostics -> ordered host delivery
render bench -> render_only -> device_only/no readback
```

Public/user-facing API should remain consumer-first later:

```text
VIDEO_ORDERED
REALTIME_PREVIEW
SENSOR_ORDERED
PARITY_DEBUG
RENDER_BENCH
```

but those names should map onto internal `output_profile`, `readback_payload`,
`delivery_policy`, and `write_policy` fields only after the internal runtime
boundary has proven stable.

## Immediate Consequence

The next optimization slice should continue on the direct-light pipeline:

```text
shadow/shade profile analysis
no-shadow vs shadow GPU measurements
shade kernel split or shadow any-hit specialization
workspace/buffer reuse if profile shows allocation cost
```

Do not start PT0/PT1 until the direct-light RenderSession and delivery contracts
are clearer.
