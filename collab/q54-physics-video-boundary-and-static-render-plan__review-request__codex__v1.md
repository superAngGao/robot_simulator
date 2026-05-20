# Q54 Simulation Frame Runtime / Video / Static Render Boundary Plan

Author: Codex  
Status: Review request  
Scope: architecture only; the earlier `run_physics_render_loop(...)` draft was reverted and should not be treated as an implementation plan

## Why this document exists

We are about to connect physics-driven frames to the Optical Lab render/video path. This is the highest-risk boundary in the current Q54 series because four concepts meet here:

- physics owns simulation time and published-frame lifetime
- render owns `OpticalLabRenderFrameContext` and `RenderRequest`
- video/delivery owns camera sequencing, readback, timing rows, and output files
- future RL/sensor consumers need a stable frame workflow interface that is not video-specific

If these layers are mixed casually, we will accumulate technical debt quickly:

- `video_loop.py` may start knowing about `GpuEngine` / physics leases
- `physics_loop.py` may duplicate delivery/readback/timing logic
- `runner.py` may become an orchestration dumping ground
- static asset rendering may get distorted to look like a fake physics path
- RL may end up calling video helpers directly just to obtain observations

The goal is to define a small set of ownership rules before the next code slice,
and to reserve the right high-level interface for a future full simulation frame
workflow.

## Current state

The canonical render pipeline entrypoints are:

```python
OpticalLabRenderPipeline.create_from_source(...)
OpticalLabRenderPipeline.create_from_source_factory(...)
```

The per-frame render entrypoint is:

```python
frame_context = pipeline.begin_frame(...)
render_result = frame_context.render(RenderRequest(...))
```

Go2 static asset rendering currently reaches this through:

```text
runner.run_scenario(...)
→ go2_backend.render_many_views(...)
→ OpticalLabRenderPipeline.create_from_source_factory(...)
→ video_loop.render_video_frame(...)
→ pipeline.begin_frame(...)
→ frame_context.render(...)
→ VideoDeliveryFacade
```

Physics smoke tests currently reach the same render pipeline through:

```text
runner.create_physics_render_runtime_for_config(...)
→ physics_source.create_physics_render_runtime(...)
→ PhysicsLabRenderRuntime.begin_frame(...)
→ PhysicsLabFrameLease.frame_context.render(...)
→ lease.complete()
```

Important current mismatch:

`video_loop.render_video_frame(...)` still owns `pipeline.begin_frame(...)`. That is acceptable for static/synthetic video today, but it is the wrong abstraction for physics runtime. In physics runtime, `begin_frame` must be paired with a physics borrow/complete lifecycle.

## Design philosophy

### 1. Physics is the time source

Physics should not be called by the render pipeline as an implementation detail.

The owner of the simulation loop advances physics:

```text
GpuEngine.step(...)
→ published frame
→ render/sensor consumer borrows that frame
```

The render pipeline consumes a frame that already exists. It does not decide when physics advances.

### 2. Render consumes frame contexts

The video layer's per-frame render input should be:

```python
OpticalLabRenderFrameContext
```

not:

```python
OpticalLabRenderPipeline
```

This is the key boundary. If video consumes a frame context, then static, synthetic dynamic, and physics runtime can all provide frame contexts without video knowing which source produced them.

### 3. Delivery is render/output ownership, not physics ownership

Delivery/readback/timing belongs to the render/video output layer:

```text
RenderResult
→ optional RGB8 pack
→ sync or async readback
→ timing row
→ progress/output
```

Physics should not implement readback policy, PNG output, frame timing CSV rows, or async delivery rings.

### 4. Borrow completion and delivery completion are different things

Physics published-frame completion means:

```text
all GPU work that reads the borrowed physics frame has been enqueued/ordered
```

Delivery completion means:

```text
the render result has become visible to the requested consumer/output policy
```

These are not the same lifecycle. In the conservative initial design, the physics lease should cover at least:

```text
runtime.begin_frame(...)
→ frame_context.render(...)
→ lease.complete()
```

For the current implementation, `frame_context.render(...)` returns only after
the render compute result is ready. After that point the result is render-owned,
so the physics borrow can be released before delivery submit as long as delivery
only reads render-owned buffers.

Preferred initial ordering:

```text
with physics_provider.begin_frame(...) as frame_context:
    rendered = render_video_frame_from_context(frame_context, plan)
    # provider exit / lease.complete() happens here
delivery.submit(rendered)
```

This keeps slow sync readback from unnecessarily pinning physics publish-ring
slots. If render becomes fully async in the future, this contract must be
revisited.

### 5. Backpressure is policy, not an accident

For `FrameSourceKind.PHYSICS_RUNTIME` with the current default consumer:

```text
qos_mode="lossless"
access_mode="borrow"
consumer_location="device"
```

the expected behavior is:

- if physics is slow, render waits for the next published frame
- if render is slow, the physics publish ring sees backpressure through the pinned borrowed frame until `complete_device_consumer(...)`
- if delivery is sync, the single-threaded runner may also wait on host readback before stepping the next frame
- if delivery is async ordered, render/readback can overlap while preserving ordered completion

Future realtime preview modes can choose different policy, such as latest-frame/drop behavior, but that must be explicit and not hidden inside the video loop.

### 6. The high-level interface is a frame runtime, not a small scheduler

The eventual outer layer should represent a complete simulation-frame workflow:

```text
optional action/control input
→ physics step or static frame selection
→ frame-context acquisition
→ render/sensor computation
→ delivery/readback or device observation production
→ timing/backpressure/accounting
→ frame result
```

Calling this layer a scheduler is too small. Scheduling is one internal concern;
the public role is runtime/workflow orchestration.

Long-term target naming:

```text
SimulationFrameRuntime
```

Acceptable shorter internal filename:

```text
frame_runtime.py
```

This layer should wrap physics, render, video/delivery, and future RL/sensor
products, but it should not implement their internals. It coordinates phases and
policy.

Comparable patterns in other simulators:

- Isaac Lab has high-level environment/simulation contexts that order action
  processing, physics stepping, optional rendering, scene updates, and
  observation computation.
- Gazebo Sim expresses similar orchestration through simulation phases:
  `PreUpdate` for controls, `Update` for physics, and `PostUpdate` for
  sensors/rendering/readback of the finished world state.
- MuJoCo exposes lower-level primitives (`mj_step`, render calls), leaving this
  outer workflow to the application.

Q54 should grow a small version of the Isaac/Gazebo-style outer workflow, but
only after the render/video frame-context boundary is clean.

### 7. RL is a future frame-runtime consumer, not a video-loop consumer

RL should not depend on `video_loop.py`.

An RL path may need:

```text
action
→ physics step
→ camera/range/force/contact observations
→ device tensors or host arrays
→ reward/termination/extras
```

This overlaps with video/export in the physics and sensor phases, but not in the
final delivery shape. Video wants PNG/CSV/progress/readback products. RL often
wants device-resident observation tensors and deterministic step results.

Therefore the future `SimulationFrameRuntime` should support multiple
consumers/products:

```text
video/export product
debug host-readback product
RL observation product
device-only sensor product
```

Video delivery is one consumer of the frame runtime, not the runtime itself.

Consumer registration should prefer construction-time registration for
fail-fast validation and stable result shape. Per-frame enable/disable policy is
allowed, but it must not change the result dataclass fields. A disabled consumer
should produce `None` or an explicit absent sentinel in its typed result field,
not remove that field.

## Static asset rendering

Static rendering must remain first-class. It should not be represented as fake physics.

Static asset rendering means:

```text
static asset builder
→ OpticalLabRenderSource
→ OpticalLabRenderPipeline
→ per-frame OpticalLabRenderFrameContext
```

There is no physics step, no physics publish ring, and no device consumer complete.

This is not an exception to the frame-context rule. Static rendering still
provides an `OpticalLabRenderFrameContext`; it simply obtains that context from
`pipeline.begin_frame(frame_inputs=None)` instead of from a physics borrow.

The clean model is to treat static rendering as a frame-context provider with a no-op lifetime:

```text
StaticFrameContextProvider.begin_frame(frame_index, env_idx=plan.camera.env_idx)
→ pipeline.begin_frame(frame_inputs=None, env_idx=env_idx)
→ context manager exits with no borrow completion
```

For static providers, `env_idx` should still be accepted and passed through even
when it is effectively a no-op for the current single-env static path. This
keeps the provider call shape aligned with synthetic and physics providers.

For static video/orbit, only the camera changes per video frame. The scene frame and acceleration data are stable:

```text
camera_i changes
scene frame stays fixed
pipeline.begin_frame(frame_inputs=None)
frame_context.render(request_i)
delivery.submit(...)
```

For a single still render, the same model degenerates to one frame:

```text
with static_provider.begin_frame(0) as frame_context:
    frame_context.render(request)
```

This preserves the design intent from the Go2 rename work:

- Go2/Menagerie is a static asset builder
- it is not a render pipeline
- it is not a physics runtime
- it should not pretend to publish physics frames

## Proposed abstraction

Introduce two levels of internal vocabulary:

1. frame-context providers, which acquire one `OpticalLabRenderFrameContext`
2. a later simulation frame runtime, which coordinates providers, video/sensor
   consumers, delivery, and frame results

The provider vocabulary does not need to be public API yet.

Conceptually:

```python
@dataclass(frozen=True)
class FrameIdentity:
    frame_id: int
    sim_time: float
    env_idx: int = 0


class RenderFrameContextLease(Protocol):
    frame_context: OpticalLabRenderFrameContext
    done_event: object | None

    def __enter__(self) -> OpticalLabRenderFrameContext: ...
    def __exit__(self, exc_type, exc, tb) -> None: ...
```

Provider variants:

```text
Static provider:
  pipeline.begin_frame(frame_inputs=None)
  no-op completion

Synthetic sequence provider:
  pipeline.begin_frame(frame_inputs=prebuilt_frame_i)
  no-op completion

Physics provider:
  physics step or supplied published frame
  runtime.begin_frame(published_frame=...)
  __exit__ completes physics device consumer
```

Note: the existing `PhysicsLabFrameLease.__enter__()` returns the lease object.
A future physics frame-context provider can wrap that lease and return
`lease.frame_context` from the provider's context manager without changing the
existing lease API.

The video layer then consumes only:

```python
OpticalLabRenderFrameContext
```

and optionally receives metadata from the lease after exit:

```text
done_event
frame_id
sim_time
```

## Future simulation frame runtime

After P1/P2 split the video and provider boundaries, add a runtime/workflow layer.

The runtime should own phase order and backpressure policy:

```text
for each frame:
  receive optional action/control
  advance or select the frame source
  acquire a frame context through a provider
  run one or more frame consumers
  close the provider/lease lifecycle
  submit delivery/readback products
  collect completed products and timing
  return a frame result
```

It should not own implementation details:

- no BVH construction details
- no Go2 static asset building logic
- no direct RGB8 pack implementation
- no torch async ring internals
- no reward function or policy implementation
- no low-level render kernels

Long-term target module:

```text
tools/optical_pipeline_lab/frame_runtime.py
```

Long-term target class:

```text
SimulationFrameRuntime
```

Do not rush this name into P4. If the first implementation only coordinates a
provider-backed video benchmark path, use a narrower internal name such as:

```text
FrameWorkflowRunner
```

or keep the class private. Promote to `SimulationFrameRuntime` only when the
action/observation/RL-facing shape is better understood.

Possible future result shape:

```python
@dataclass
class SimulationFrameResult:
    frame_index: int
    frame_id: int
    sim_time: float
    products: Mapping[str, object]
    timing: Mapping[str, float]
    backpressure: Mapping[str, object]
```

This shape is intentionally illustrative, not a P4 implementation contract.
`products: Mapping[str, object]` is too loose for the first concrete slice. P4
should prefer a narrow, typed result for the product it actually supports, for
example a video-focused field. Future RL observation/reward fields should be
added once their shape is known, instead of prematurely hiding them behind
string keys.

## Required video loop split

Before adding a real physics video runner, split `video_loop.render_video_frame(...)` so `pipeline.begin_frame(...)` is no longer fused to camera/request/delivery logic.

Proposed pieces:

### `VideoRenderPlan`

Contains:

- camera
- optional materialized rays
- `RenderRequest`
- `camera_rays_ms`
- geometry mode label
- include-shadow-traversal flag

It does not contain a pipeline or frame context.

It also should not retain `frame_inputs`. The plan builder may consume frame
identity inputs to resolve `camera.frame_id` / `camera.sim_time`, but dynamic
frame acquisition remains provider/frame-context ownership.

The concrete input name should be:

```python
frame_identity: FrameIdentity | None = None
```

not `frame_inputs`. `FrameIdentity` is intentionally minimal
(`frame_id`, `sim_time`, `env_idx`) and has no physics dependency. Existing
static/synthetic wrappers may derive it from their local frame inputs before
calling the plan builder.

### `build_video_render_plan(...)`

Inputs:

- scene
- args/options
- frame index
- optional ray cache
- optional `FrameIdentity`
- `build_video_camera`

Output:

- `VideoRenderPlan`

This helper builds camera/rays/request only. It must not call `pipeline.begin_frame(...)`.

`env_idx` should come from `plan.camera.env_idx`. Providers should receive or be
called with that value; they should not infer it independently.
If `frame_identity` is `None`, the plan builder should fall back to an explicit
args/options env value when one exists, otherwise `0`.

`geometry_mode` should be an explicit `VideoRenderPlan` field. P1 may preserve
the current behavior by deriving it from existing args/frame inputs, but later
physics modes should not rely on the old `frame_inputs is not None` heuristic.

The resolved camera inside the plan should already carry the frame identity used
for the render request. For physics paths, that identity must ultimately come
from the current `OpticalLabRenderFrameContext` / borrowed frame, not from a
base scene frame.

### `render_video_frame_from_context(...)`

Inputs:

- `OpticalLabRenderFrameContext`
- `VideoRenderPlan`
- frame index

Output:

- `RenderedVideoFrame`

This helper calls:

```python
frame_context.render(plan.request)
```

and packages the existing `RenderedVideoFrame` envelope.

It must preserve:

```python
prepare_timing = dict(frame_context.prepare_timing)
```

in `RenderedVideoFrame.prepare_timing`. This is required so physics paths keep
`snapshot_ms`, `accel_refit_ms`, and `accel_rebuild_ms` in the video timing row.

Timing semantics:

- `NaN` means not applicable or not measured
- `0.0` means measured and effectively zero

Static paths should not fabricate `0.0` frame-preparation timings just to make
CSV rows dense. Summary/aggregation code must preserve this distinction.
If aggregation already computes means over finite values while ignoring `NaN`,
no special handling is needed. Any `fillna(0)`-style logic must be avoided or
changed before frame-preparation timing is summarized.

### Existing `render_video_frame(...)`

Keep it for current static/synthetic callers, but implement it as:

```text
frame_inputs = video_frame_inputs(args, frame_index)
frame_identity = frame_identity_from_inputs(frame_inputs)  # None for static path
plan = build_video_render_plan(..., frame_identity=frame_identity)
frame_context = pipeline.begin_frame(frame_inputs=frame_inputs, env_idx=plan.camera.env_idx)
return render_video_frame_from_context(frame_context, plan)
```

No behavior change in this slice.

## Delivery placement

`VideoDeliveryFacade` should remain generic and unaware of physics.

Static/synthetic outer loop:

```text
frame_identity = frame_identity_from_static_or_synthetic_source(i)
plan = build_video_render_plan(..., frame_identity=frame_identity)
with static_or_synthetic_provider.begin_frame(i, env_idx=plan.camera.env_idx) as frame_context:
    rendered = render_video_frame_from_context(frame_context, plan)
delivery.submit(rendered)
delivery.complete_available(...)
```

Physics outer loop:

```text
published = step_physics(i)
with physics_provider.begin_frame(i, published, env_idx=requested_env_idx) as frame_context:
    frame_identity = FrameIdentity(frame_context.frame_id, frame_context.sim_time, frame_context.env_idx)
    plan = build_video_render_plan(..., frame_identity=frame_identity)
    rendered = render_video_frame_from_context(frame_context, plan)
delivery.submit(rendered)
delivery.complete_available(...)
```

The difference is provider lifecycle, not delivery behavior.

For static/synthetic paths, frame identity can be known before provider entry.
For physics paths, frame identity must be read from the current borrowed context
inside provider entry. `requested_env_idx` for physics comes from scenario/camera
configuration, defaulting to 0 for current smoke tests.

## Async warmup placement

Current `build_torch_async_warmup_result(...)` calls `pipeline.begin_frame(...)`
directly. That is acceptable only for the existing static/synthetic path.

Use the term **provider-backed warmup** for the replacement behavior: warmup
acquires an `OpticalLabRenderFrameContext` through the same provider lifecycle
as normal frames, instead of directly calling `pipeline.begin_frame(...)`.

Before any physics provider path supports `video_readback_delivery="torch_async"`,
warmup must also go through the frame-context provider lifecycle, or the physics
path must explicitly reject/skip async warmup.

Mechanism: do not use a magic `frame_index=-1`. The provider-backed benchmark
should perform one ordinary warmup acquisition before the main loop using a real
or otherwise explicitly resolved `FrameIdentity`. If a provider cannot safely
produce such a warmup frame, torch async delivery must fail fast for that
provider.

Recommended staging:

- P1 leaves existing async warmup unchanged.
- P3 adds provider-backed warmup for the new frame-context benchmark path.
- Physics provider paths should reject `torch_async` as soon as the provider is
  introduced, until provider-backed warmup exists.
- P5 physics video smoke should start with sync delivery unless provider-backed
  warmup is already implemented.

## Proposed implementation plan

### P0: Keep the reverted runner-loop draft out of the series

The reverted draft had:

- adds `PhysicsRuntimeLoopFrame`
- adds `run_physics_render_loop(...)` directly to `runner.py`
- adds two tests and bumps MANIFEST to 242

Recommendation:

- do not reintroduce that draft as-is
- replace it with the provider/video split described here

Reason:

- it puts loop ownership in `runner.py`
- it encourages callback-generic orchestration before video/context boundaries are clean

### P1: Split video frame construction from frame-context ownership

Files:

- `tools/optical_pipeline_lab/video_loop.py`
- `tests/unit/optics/test_optical_pipeline_lab.py`

Add:

- `VideoRenderPlan`
- `build_video_render_plan(...)`
- `render_video_frame_from_context(...)`

Keep:

- `render_video_frame(...)` public behavior unchanged
- current Go2/static tests passing

Focused tests:

- plan builds camera/request without calling `pipeline.begin_frame`
- plan builder may consume `FrameIdentity`, but `VideoRenderPlan` does not retain `frame_inputs`
- rendering from a fake `OpticalLabRenderFrameContext` returns existing `RenderedVideoFrame` fields
- `render_video_frame_from_context(...)` preserves `frame_context.prepare_timing`
- `VideoRenderPlan.geometry_mode` is explicit and used by `RenderedVideoFrame`
- existing `render_video_frame(...)` still delegates and preserves behavior

### P2: Introduce frame-context providers

Possible module:

```text
tools/optical_pipeline_lab/frame_contexts.py
```

or, if we want to keep it lab-runner scoped:

```text
tools/optical_pipeline_lab/render_frames.py
```

Add:

- static provider
- synthetic frame sequence provider
- physics provider wrapping `PhysicsLabRenderRuntime`

Keep provider API narrow:

```python
with provider.begin_frame(frame_index) as frame_context:
    ...
```

Tests:

- static provider calls `pipeline.begin_frame(frame_inputs=None)`
- synthetic provider passes frame-specific `frame_inputs`
- physics provider wraps `PhysicsLabFrameLease` and returns `OpticalLabRenderFrameContext` from provider `__enter__`
- physics provider completes lease on exit
- physics provider path rejects `torch_async` until provider-backed warmup exists
- `BaseException` cleanup remains covered by existing `PhysicsLabFrameLease` tests

### P3: Teach video benchmark to accept a frame-context provider

Options:

1. Add a new function:

```python
run_video_benchmark_with_frame_contexts(...)
```

2. Or extend `run_video_benchmark(...)` with an optional provider while preserving current behavior.

Recommended first slice:

- add a new function to avoid destabilizing current Go2 path
- later fold if duplication becomes obvious

The new function should reuse:

- `build_video_render_plan(...)`
- `render_video_frame_from_context(...)`
- `VideoDeliveryFacade`
- `VideoFrameTimingRowBuilder`
- `record_delivered_video_frame(...)`

It should not duplicate delivery/readback/timing internals.

It must also handle async warmup through the provider lifecycle before physics
providers are allowed to use torch async delivery.

The new provider-backed benchmark path should not inherit the old
`render_frame(pipeline, args, frame_index, ray_cache)` callback. That callback is
coupled to `pipeline.begin_frame(...)`; new customization should happen through
plan builders or frame consumers.

### P4: Introduce a narrow frame workflow runner

Possible module:

```text
tools/optical_pipeline_lab/frame_runtime.py
```

Add a narrow internal runner that coordinates:

- frame-context provider
- video/render consumer callback
- delivery facade
- timing/backpressure metadata

Do not export the long-term `SimulationFrameRuntime` name unless the interface
already covers non-video consumers. Prefer a narrower name like
`FrameWorkflowRunner` for this first concrete slice.

Initial scope should still be lab-internal and test-only. It should not enable
`runner.run_scenario(...)` for physics yet.

Tests should verify:

- static provider and physics provider can both be driven by the same runtime
- provider lifecycle closes before delivery completion is required
- video delivery is one product path, not hard-coded as the only product type
- runtime lives outside `runner.py`
- result type is narrow/typed for the currently supported product, not
  `Mapping[str, object]`
- future RL/device-observation products can be represented without importing
  `video_loop.py`

### P5: Add physics video smoke path behind tests, not CLI

Use:

- `create_physics_render_runtime_for_config(...)`
- physics frame-context provider
- narrow frame workflow runner, e.g. `FrameWorkflowRunner`
- video provider/consumer path from P3

Do not yet enable:

- `runner.run_scenario(...)` for `FrameSourceKind.PHYSICS_RUNTIME`

GPU smoke should prove:

```text
GpuEngine.step()
→ physics provider
→ OpticalLabRenderFrameContext
→ video render plan
→ delivery facade
→ timing row
```

Camera/request construction for this path must use the current frame identity
from `OpticalLabRenderFrameContext` / borrowed frame. It must not rely on a
`PhysicsLabRenderScene` base frame that may be stale relative to the current
physics step.

Ordering constraint for physics paths:

```text
with physics_provider.begin_frame(...) as frame_context:
    frame_identity = FrameIdentity(
        frame_id=frame_context.frame_id,
        sim_time=frame_context.sim_time,
        env_idx=frame_context.env_idx,
    )
    plan = build_video_render_plan(..., frame_identity=frame_identity)
```

The `FrameIdentity` for physics must be obtained inside the provider context,
after the current borrowed frame exists. Tests should mock this ordering so a
base scene frame cannot accidentally be used.

### P6: Enable `FrameSourceKind.PHYSICS_RUNTIME` in runner only after P5

Once the test path is real:

- relax `OpticalLabScenarioConfig.validate_implemented()` for one tiny physics runtime preset
- route `run_scenario(...)` based on `frame_source`
- keep static asset path unchanged

## Open design questions for Claude

1. Is `OpticalLabRenderFrameContext` the right per-frame boundary for video, or should video consume a slightly richer lease object?

2. Should physics `complete_device_consumer(...)` happen immediately after `frame_context.render(...)` is enqueued, before delivery submit, assuming delivery reads render-owned buffers only?

3. Do we need a formal `RenderFrameContextLease` protocol, or is a small provider convention enough while this stays lab-internal?

4. Should `run_video_benchmark(...)` be extended in place, or should we add `run_video_benchmark_with_frame_contexts(...)` first to keep the current static path stable?

5. Is `frame_contexts.py` / `render_frames.py` a better home than `runner.py` for providers?

6. Is the static provider model correct for Go2/Menagerie static asset rendering, including orbit video where only the camera changes?

7. Was reverting the direct `runner.py` loop draft the right call, or should any of its ordering test expectations be reused after the provider split?

8. Is `SimulationFrameRuntime` the right name/level for the outer workflow, or should this stay as a smaller scheduler/provider abstraction for now?

9. Should future RL observation production be modeled as another frame-runtime product/consumer, rather than as part of video delivery?

10. Should `frame_runtime.py` be introduced only after P1/P2/P3, or should we create an empty/narrow runtime vocabulary earlier to anchor the architecture?

11. Is the proposed ordering "render from context → release physics borrow →
    submit delivery" correct for the current render contract?

12. Should physics provider paths reject `torch_async` until provider-backed
    warmup exists?

13. Should frame consumers be registered at runner construction time or passed
    dynamically on each step? The current recommendation is construction-time
    registration for fail-fast validation and stable result shape, with per-frame
    enable/disable policy if needed.

14. What absent-value sentinel should typed workflow results use for a
    construction-registered consumer that is disabled on a given frame? Is
    `None` sufficient for the first video-focused result type? Current
    recommendation: `None` is sufficient for the first typed video result.

## External architecture references

- Isaac Lab `SimulationContext.step/render` and `ManagerBasedEnv.step` show a
  high-level runtime ordering physics, optional rendering, scene updates, and
  observations:
  https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/sim/simulation_context.html
  and
  https://isaac-sim.github.io/IsaacLab/v2.0.0/_modules/isaaclab/envs/manager_based_env.html
- Gazebo Sim system plugins separate phase ownership: `PreUpdate` for control,
  `Update` for physics, and `PostUpdate` for read-only sensors/rendering:
  https://gazebosim.org/api/sim/7/createsystemplugins.html
- MuJoCo exposes lower-level `mj_step` / render primitives and leaves the outer
  application loop to the caller:
  https://mujoco.readthedocs.io/en/2.1.5/programming.html

## Recommended next action

Ask Claude to review this boundary plan before more code lands.

If approved, first implementation slice should be P1 only:

```text
split video_loop into plan + render-from-context
preserve current render_video_frame behavior
no physics runner yet
no CLI behavior change
```
