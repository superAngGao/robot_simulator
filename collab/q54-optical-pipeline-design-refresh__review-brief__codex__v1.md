# Q54 Optical Pipeline Design Refresh Review Brief

Date: 2026-05-14
Author: Codex
Status: review-brief

## Summary

We refreshed the repo-level plan in:

```text
GPU_OPTICAL_PIPELINE_DESIGN.md
```

This is now the active source of truth for the current render foundation work.
The key correction is:

```text
Go2 is not the render pipeline.
Go2 is one scene/source that feeds the render pipeline.
```

Earlier collab plans were useful during discussion, but they had started to
accumulate around `Go2Render*` names. The root design doc now consolidates that
work into a source-driven Optical Pipeline Lab render foundation.

## What Changed In The Design

### 1. Canonical Entrypoint

The intended caller shape is now source-driven:

```python
source = OpticalLabRenderSource(
    registry=registry,
    base_frame=current_gpu_frame,
    bounds_min=bounds_min,
    bounds_max=bounds_max,
)

pipeline = OpticalLabRenderPipeline.create(
    source=source,
    options=OpticalLabRenderOptions(
        device="cuda:0",
        bvh_backend="cuda_lbvh",
        bvh_split_strategy="sort",
        shadows=True,
    ),
    timings=timings,
)

frame = pipeline.begin_frame(frame_inputs=sim.publish_gpu_frame())
result = frame.render(request)
```

This is the intended path for a physics simulation. It should provide a render
source, not a backend-shaped adapter.

### 2. Layer Boundaries

The refreshed design separates:

```text
Scene/source adapter:
  converts an external world to OpticalLabRenderSource

Render foundation:
  creates workspace/session/cache/snapshot/BVH/executor from source/options
  exposes begin_frame(...).render(...)

Video/delivery/reporting:
  consumes RenderResult and frame metadata
  handles readback/write/CSV/matrix policy
```

### 3. Naming Direction

Current implementation names are transitional:

```text
tools/optical_pipeline_lab/go2_session.py
Go2RenderWorkspace
Go2RenderSession
Go2RenderFrameContext
Go2RenderPipeline
```

Target names:

```text
tools/optical_pipeline_lab/render_session.py
OpticalLabRenderWorkspace
OpticalLabRenderSession
OpticalLabRenderFrameContext
OpticalLabRenderPipeline
```

The design explicitly keeps Go2 names only where they describe actual Go2
scenarios or Menagerie adapter behavior:

```text
go2_backend.py
go2_menagerie_static
go2_video_ordered_static
go2_video_ordered_baseline
go2_video_delivery_smoke
go2_video_ordered_legacy_960
model_dir/model_xml defaults
```

## Current Implementation Context

Already completed before this design refresh:

```text
I1:
  extracted Go2RenderSession / Go2RenderFrameContext / Go2RenderPipeline
  from go2_backend.py into go2_session.py

I2:
  introduced Go2RenderWorkspace(device, stream)
  session.device/session.stream are compatibility properties over workspace
```

The design refresh does not change code behavior. It updates the durable plan so
future code does not keep adding generic responsibilities under Go2 names.

## Active Roadmap In The Root Design Doc

```text
C1 rename-only:
  go2_session.py -> render_session.py
  Go2Render* -> OpticalLabRender*
  keep Go2 aliases temporarily

C2 source/options:
  add OpticalLabRenderSource
  add OpticalLabRenderOptions
  add a source/options construction path

C3 Go2 source builder:
  make go2_backend.py build an OpticalLabRenderSource
  call the generic OpticalLabRenderPipeline entrypoint

C4 workspace frame preparation:
  move dynamic snapshot/refit/rebuild GPU execution into workspace

C5 video-loop split:
  later move generic video render/export helpers out of go2_backend.py
```

The important ordering is: rename first, then source/options, then Go2 migration,
then workspace dynamic frame preparation.

## Review Questions

1. Is the root design correction right: Go2 should be a scene/source, not the
   render pipeline?

2. Are `OpticalLabRenderSource` and `OpticalLabRenderOptions` the right
   vocabulary for the lab-local canonical entrypoint?

3. Is `OpticalLabRender*` the right target prefix, or should the lab concrete
   classes use a shorter name like `LabRender*`?

4. Is C1 rename-only the right next implementation slice before source/options?

5. Should `bounds_min` / `bounds_max` belong on `OpticalLabRenderSource`, or
   should they stay in adapter metadata until camera construction is generalized?

6. Should C2 add a new `create_from_source(...)` while keeping the callback-based
   `create(...)`, or should it replace `create(...)` immediately?

7. Is it acceptable to leave generic video-loop helpers inside `go2_backend.py`
   temporarily while the render entrypoint is fixed?

## Codex Recommendation

Proceed with C1 after review:

```text
rename go2_session.py/classes to render_session.py / OpticalLabRender*
keep Go2 compatibility aliases
do not change behavior
do not introduce source/options in the same patch
```

This keeps the next diff mechanical and makes C2 source/options easier to
review.
