# Q54 Optical Lab Render Foundation Consolidated Plan

Date: 2026-05-14
Author: Codex
Status: review-request

## Purpose

This consolidates the recent RenderSession / workspace / video-envelope plans
into one current direction.

The older plans were useful, but the naming and entrypoint assumptions are now
lagging behind the architecture:

```text
Go2 is no longer the render pipeline.
Go2 is one scene/source that feeds the render pipeline.
```

The next work should therefore stop growing generic infrastructure under
`Go2Render*` names and should define a canonical lab render pipeline entrypoint.

## Supersedes

Use this document as the active plan over:

```text
q54-optical-render-session-workspace-stage-i-plan__review-request__codex__v1.md
q54-optical-render-video-current-design-and-r3-plan__review-request__codex__v1.md
q54-optical-render-video-envelope-r3b-plan__review-request__codex__v1.md
```

Those documents remain useful history. This one is the current integration
plan.

## Current State

Completed:

```text
R1/R2: runtime delivery contract + lab bridge
R3/R3b: video envelope carries RenderResult and uses runtime render timing
R4: GPU delivery smoke
I1: Go2RenderSession / FrameContext / Pipeline extracted to go2_session.py
I2: Go2RenderWorkspace(device, stream) introduced
```

Current concrete names:

```text
tools/optical_pipeline_lab/go2_session.py
Go2RenderWorkspace
Go2RenderSession
Go2RenderFrameContext
Go2RenderPipeline
```

Current problem:

```text
These are now generic lab render infrastructure names wearing Go2 clothes.
```

`go2_backend.py` still mixes two categories:

```text
1. actual Go2/Menagerie scene adapter and CLI compatibility
2. generic lab video render/export loop helpers
```

That split is real, but should not all be solved in one patch.

## Target Architecture

### Scene Source Layer

Scene sources adapt an external world into the render pipeline vocabulary.

Examples:

```text
Menagerie Go2 source
synthetic body triangle source
future physics simulation source
```

A source should answer:

```text
What optical registry should be rendered?
What base/current GpuPublishedFrame should bind geometry?
What optional scene metadata/camera hints exist?
```

It should not own:

```text
Warp stream lifecycle
BVH lifecycle
executor lifecycle
delivery/readback
CSV/reporting
video loop policy
```

Candidate type:

```python
@dataclass
class OpticalLabRenderSource:
    registry: object
    base_frame: GpuPublishedFrame
    frame_id: int
    sim_time: float
    bounds_min: object | None = None
    bounds_max: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
```

Open question: whether `frame_id/sim_time` should be separate fields or always
read from `base_frame`.

### Render Foundation Layer

These are lab-local concrete runtime classes, not public `optics/` API:

```text
OpticalLabRenderWorkspace
OpticalLabRenderSession
OpticalLabRenderFrameContext
OpticalLabRenderPipeline
```

Their responsibilities:

```text
Workspace:
  device, render stream, future copy stream/scratch/event resources

Session:
  source-derived scene identity
  DeviceOpticalSceneCache
  base GpuPublishedFrame
  static snapshot/BVH
  executor
  acceleration/backend configuration

FrameContext:
  env_idx
  frame-specific snapshot/BVH for dynamic frames
  prepare_timing
  render(RenderRequest) -> RenderResult

Pipeline:
  begin_frame(frame_inputs=...)
  create session from source/options
  high-level frame orchestration
```

### Delivery / Video Layer

These are separate and should not be pulled into workspace/session naming:

```text
RenderedVideoFrame
DeliveredVideoFrame
VideoDeliveryFacade
VideoFrameTimingRowBuilder
FrameTimingRecorder
matrix suites
```

The video loop can later move from `go2_backend.py` to a generic module, but it
is not the immediate rename slice.

### Go2 / Menagerie Adapter Layer

Go2-specific names should remain where they describe actual Go2 scenarios:

```text
go2_backend.py
go2_menagerie_static
go2_video_ordered_static
go2_video_ordered_baseline
go2_video_delivery_smoke
go2_video_ordered_legacy_960
model_dir / model_xml defaults
build_menagerie_example_args(...)
```

The adapter should eventually do this:

```python
source = build_go2_render_source(args)
pipeline = OpticalLabRenderPipeline.create(
    source=source,
    options=options,
    timings=timings,
)
```

not this:

```python
pipeline = Go2RenderPipeline.create(args, scene_factory=..., ...)
```

## Canonical Entrypoint Direction

The lab render pipeline should have a canonical construction path that does not
require a backend-specific adapter per caller:

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

This is what a physics simulation should call. It should not need a
`go2_backend`-style adapter; it only needs to provide the source vocabulary.

## Proposed Slices

### C0 — Consolidated Plan

This document.

No code changes.

### C1 — Rename Generic Render Infrastructure

Target:

```text
tools/optical_pipeline_lab/go2_session.py
  -> tools/optical_pipeline_lab/render_session.py

Go2RenderWorkspace
  -> OpticalLabRenderWorkspace
Go2RenderSession
  -> OpticalLabRenderSession
Go2RenderFrameContext
  -> OpticalLabRenderFrameContext
Go2RenderPipeline
  -> OpticalLabRenderPipeline
```

Compatibility:

```python
# tools/optical_pipeline_lab/go2_session.py
from .render_session import (
    OpticalLabRenderWorkspace as Go2RenderWorkspace,
    OpticalLabRenderSession as Go2RenderSession,
    OpticalLabRenderFrameContext as Go2RenderFrameContext,
    OpticalLabRenderPipeline as Go2RenderPipeline,
)
```

`go2_backend.py` may continue re-exporting the old names temporarily.

Tests:

- migrate generic tests to import/use `render_session`;
- keep Go2 preset/matrix tests named Go2.

Non-goal:

- no source type yet;
- no behavior changes;
- no video loop split.

### C2 — Introduce Source And Options Types

Add lab-local types:

```python
@dataclass
class OpticalLabRenderSource:
    registry: object
    base_frame: GpuPublishedFrame
    bounds_min: object | None = None
    bounds_max: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

@dataclass
class OpticalLabRenderOptions:
    device: str = "cuda:0"
    bvh_backend: str = "cpu"
    bvh_split_strategy: str = "sort"
    shadows: bool = True
```

Then add:

```python
OpticalLabRenderPipeline.create_from_source(source, options, timings, ...)
```

or change `create(...)` to take source/options if review agrees.

Non-goal:

- do not delete callback-based `create(...)` until Go2 backend is migrated.

### C3 — Migrate Go2 Backend To Source Builder

Change Go2 backend construction to:

```python
source = _build_render_source_for_preset(scene_preset, args, device=device)
pipeline = OpticalLabRenderPipeline.create_from_source(source, options, timings)
```

Go2-specific code remains in `go2_backend.py`:

```text
MJCF import
model_dir/model_xml
synthetic scene preset
base GPU frame creation
camera defaults
CLI compatibility
```

Generic render session construction moves behind source/options.

### C4 — Move Dynamic Frame Preparation Toward Workspace

After source/options are stable, move the GPU execution part of dynamic
begin-frame into workspace:

```python
prepared = session.workspace.prepare_dynamic_frame(
    cache=session.cache,
    frame_inputs=frame_inputs,
    env_idx=env_idx,
    base_bvh=session.bvh,
    bvh_backend=session.bvh_backend,
    bvh_split_strategy=session.bvh_split_strategy,
)
```

Return:

```python
@dataclass
class PreparedFrameResources:
    snapshot: object
    bvh: object
    timing: FramePrepareTiming
```

Pipeline still owns:

```text
static/dynamic decision
Go2/current source identity
FrameContext creation
```

Workspace owns:

```text
device/stream use
snapshot/refit/rebuild execution
event synchronization for frame preparation
```

### C5 — Split Generic Video Loop Later

Possible later move:

```text
go2_backend.py generic video helpers
  -> video_loop.py
```

This is lower priority than fixing the render entrypoint. Do not mix it with C1
or C2.

## Naming Inventory

### Rename Now / Soon

```text
go2_session.py -> render_session.py
Go2RenderWorkspace -> OpticalLabRenderWorkspace
Go2RenderSession -> OpticalLabRenderSession
Go2RenderFrameContext -> OpticalLabRenderFrameContext
Go2RenderPipeline -> OpticalLabRenderPipeline
generic tests named test_go2_pipeline_* -> test_lab_render_pipeline_*
```

### Keep Go2 Names

```text
go2_backend.py
go2_menagerie_static
go2_video_ordered_static
go2_video_ordered_baseline
go2_video_delivery_smoke
go2_video_ordered_legacy_960
build_menagerie_example_args
Go2-specific GPU smoke output paths
```

### Watch Later

```text
build_menagerie_example_args
```

This is still transitional. It is Go2/Menagerie-specific today, but once the
lab runner no longer mimics example CLI args, it may become:

```text
build_lab_render_run_args
```

or disappear behind source/options.

## Risks

### Too Much Rename Churn

Mitigation:

- C1 is rename-only;
- keep aliases;
- do not mix source/options with rename.

### Premature Public API

`OpticalLabRender*` names are intentionally lab-local. Do not move them to
`optics/` yet.

### Source Type Too Vague

The source type must be just rich enough for real callers:

```text
physics simulation
Menagerie Go2
synthetic dynamic smoke
```

It should not become a catch-all backend object.

### Video Loop Still In go2_backend.py

This remains awkward after C1/C2, but acceptable. Moving video loop first would
be a larger diff and does not solve the canonical render entrypoint.

## Review Questions

1. Should the target class prefix be `OpticalLabRender*`, or is
   `LabRender*` / `RenderLab*` better?

2. Should the module be `render_session.py`, `lab_render_session.py`, or
   `render_runtime.py`?

3. Is C1 rename-only the right immediate next slice before adding source/options?

4. Should `OpticalLabRenderSource` own `frame_id` / `sim_time` explicitly, or
   should those always be read from `base_frame`?

5. Should `bounds_min` / `bounds_max` belong to source, or stay in adapter scene
   metadata until camera construction is generalized?

6. Should C2 add `create_from_source(...)` alongside callback-based `create(...)`,
   or replace `create(...)` immediately?

7. Is `go2_backend.py` acceptable as the temporary home for generic video loop
   helpers until after source/options land?

8. Should workspace dynamic frame preparation wait until after source/options
   (Codex recommendation), or proceed immediately after rename?

## Codex Recommendation

```text
C1 next: rename go2_session.py/classes to OpticalLabRender* with Go2 aliases.
C2 after that: add OpticalLabRenderSource + OpticalLabRenderOptions.
C3 migrate Go2 backend to source/options.
C4 move dynamic frame preparation execution into workspace.
C5 split generic video loop only after the render entrypoint is clean.
```

The key principle:

```text
External systems should provide a render source, not a backend adapter.
```
