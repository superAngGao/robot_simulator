# Q54 P12.5 Design Request: System Boundary Consolidation

## Context

P9 through P12 progressively moved the Optical Pipeline Lab from ad hoc render
runner paths toward physics-owned, product-based, preset-driven workflows:

```text
P9   multi-product tick/runtime orchestration
P10  product workflow helpers and explicit runtime-owner path
P11  preset workflow surface and Go2 static path exit from legacy backend naming
P12  direct-light backend family: CPU / Warp / CUDA
```

This produced real capability, but it also exposed a broader architecture
question: the system now spans physics ownership, render backend selection,
device-channel ownership, product workflows, examples, benchmarks, and public
calling surfaces. P12.5 should be a deliberate system-level consolidation pass,
not a narrow "add backend registry" refactor.

## Core Position

P12.5 should treat the Optical Lab as a three-layer system:

```text
Physics ownership
    -> published frame / simulation tick stream
Render ownership
    -> scene snapshot / acceleration / render backend / device channels
User workflow ownership
    -> presets / products / artifacts / examples / benchmarks
```

The goal is to make the boundaries explicit, unique, and documented. The goal
is not to add small adapters everywhere. Any adapter/refactor introduced in
P12.5 must live at a necessary boundary between large components and should be
the only place that boundary is translated.

## Why P12.5 Comes After P12.3/P12.4

P12.5 should not be designed in a vacuum. It should wait until the direct-light
backend family has enough real behavior:

```text
cpu_direct_light          implemented reference path
warp_bvh_direct_light     existing GPU/Warp path
cuda_direct_light         implemented minimal CUDA path
```

P12.4 should then establish parity and benchmark reporting across those
backends. With real backend behavior and real benchmark rows, P12.5 can
consolidate actual boundaries instead of prematurely abstracting guessed ones.

Recommended sequence:

```text
P12.3  CUDA direct-light backend
P12.4  backend parity + benchmark/matrix contract
P12.5  system boundary consolidation
P12.6  optional public backend override + examples UX
P13    path tracing / accumulation / OptiX / non-direct-light rendering
```

## Goals

1. Re-express the physics/render/user workflow boundaries from a higher level.
2. Consolidate device-channel materialization into one system boundary.
3. Consolidate backend compatibility and selection rules into one reviewed
   location.
4. Clarify the recommended user-facing API tiers after P10/P11/P12.
5. Update README/MANIFEST and architecture docs so the repo describes the
   system that actually exists.
6. Identify legacy/transitional entry points and document their status without
   forcing premature deletion.

## Non-Goals

P12.5 should not:

- implement new CUDA kernels;
- add path tracing, accumulation, OptiX, denoising, or multisample semantics;
- add RL/training control loops;
- introduce a broad public API redesign;
- delete legacy tools unless replacement paths and tests already exist;
- make adapter layers for every local type mismatch.

## Design Principles

### 1. Boundary Components Must Be Unique

An adapter is acceptable only when it represents a necessary connection between
large components.

Good:

```text
executor device channels -> staging/readback/delivery
```

Bad:

```text
cuda delivery adapter
warp delivery adapter
video product adapter
rgb adapter
workflow adapter
```

If conversion logic starts appearing in multiple products, runners, examples,
and delivery helpers, P12.5 should pull it back into one boundary component.

### 2. Systematic Beats Small

The adapter or refactor does not have to be tiny. It has to be coherent. A
larger boundary module is better than many small scattered conversion helpers
if it gives the system one stable place to reason about ownership and
materialization.

### 3. Implementation Should Follow Real Backends

P12.5 should not invent abstractions ahead of implementation. It should extract
only the boundaries proven by CPU/Warp/CUDA direct-light paths and benchmark
reporting.

### 4. User APIs Should Describe Workflows, Not Internals

Users should select presets, products, artifacts, and eventually backend
intent. They should not need to assemble scene snapshots, render sessions,
delivery facades, or frame providers unless they are explicitly working at an
advanced/internal layer.

## Physics Ownership Boundary

P12.5 should document and lock the current ownership model:

```text
Physics runtime owns:
  - simulation clock
  - step/action advancement
  - state and telemetry
  - published frame production

Render runtime owns:
  - consuming a published frame
  - scene snapshot preparation
  - acceleration structure preparation
  - rendering device channels

Product workflow owns:
  - ordered product execution per tick
  - artifact lifecycle
  - debug/video/observation product orchestration
```

Important rule:

```text
render products consume published frames; they do not reach backward and own
physics stepping internals.
```

Future RL/training should fit as:

```text
action -> physics.step_tick(...) -> observation/render/debug products -> policy
```

P12.5 does not implement that loop, but it should not block it.

## Render Ownership Boundary

P12.5 should describe the render stack as:

```text
OpticalLabRenderSource
  -> scene snapshot
  -> acceleration backend
  -> render backend
  -> OpticalComputeResult channels
  -> staging/readback/delivery
```

Key vocabulary:

```text
accel_backend:
  cpu_bvh
  cuda_lbvh
  future optix

render_backend:
  cpu_direct_light
  warp_bvh_direct_light
  cuda_direct_light
  future path_tracer / optix variants

output_profile:
  direct_light_full
  rgb_preview
  render_only

delivery_policy:
  sync
  device_only
  torch_async

write_policy:
  none
  png_sequence
  video_encoder
```

P12.5 should distinguish these axes clearly. In particular:

- `cuda_lbvh` is an acceleration backend, not a render backend;
- `cuda_direct_light` is a render backend, not an RGB8 pack shortcut;
- `path_tracer` belongs to P13 because it changes accumulation/sample
  semantics.

## Device Channel Boundary

P12.5 should introduce or finalize a single device-channel materialization
boundary after CUDA direct-light has landed.

Purpose:

```text
OpticalComputeResult(device channels)
    -> torch tensor view / host numpy staging / async readback
```

It should support the real backend outputs:

```text
Warp array        from warp_bvh_direct_light
torch CUDA tensor from cuda_direct_light
host numpy array  from cpu_direct_light
```

Possible module:

```text
optics/device_channel.py
```

Possible functions:

```python
def channel_is_device(value: object) -> bool: ...
def channel_to_torch(value: object): ...
def channel_to_numpy(value: object): ...
def stage_channels_to_host(result: OpticalComputeResult, channels: Sequence[str]): ...
```

Design constraints:

- Torch and Warp imports must be optional/lazy.
- CPU-only imports must remain safe.
- The boundary must not know about Optical Lab products or preset names.
- The boundary must not know about `cuda_direct_light` specifically.
- RGB8 pack can use this boundary or an adjacent derived-channel boundary, but
  it should not force executor outputs to become Warp-owned.

Open decision:

```text
Should device-channel materialization live in optics/device_channel.py, or
should optics/device.py absorb it once torch optional-import safety is proven?
```

Recommended initial answer:

```text
Use optics/device_channel.py so the boundary is explicit and does not overload
the existing device workload/staging module.
```

## Backend Compatibility Boundary

P12.5 should consolidate compatibility validation currently distributed across
scenario config and run options.

Candidate module:

```text
tools/optical_pipeline_lab/render_backends.py
```

Candidate responsibilities:

- supported render backend list;
- render backend -> allowed acceleration backends;
- render backend -> allowed geometry/frame-source modes;
- render backend -> run option restrictions;
- reserved/future backend fail-fast messages.

Example compatibility table:

```text
cpu_direct_light:
  accel_backend = cpu_bvh
  geometry = static first
  raygen = host
  delivery = sync

warp_bvh_direct_light:
  accel_backend = cpu_bvh | cuda_lbvh
  geometry = static | dynamic supported where existing paths support it
  raygen = host | gpu
  delivery = sync | device_only | torch_async where warmup supports it

cuda_direct_light:
  accel_backend = cuda_lbvh
  geometry = static first
  raygen = host first, gpu after P12.3 camera raygen
  delivery = sync first, torch_async after channel boundary supports it
```

P12.5 should decide whether this table becomes executable code, documentation,
or both.

Recommended answer:

```text
Make it executable in a small backend compatibility module and mirror it in
architecture docs.
```

## User Workflow Boundary

P12.5 should define the intended API tiers:

### Tier 1: Preset User

Goal: run reviewed presets and produce artifacts.

```python
result = run_optical_lab_preset(
    "go2_video_ordered_static",
    frames=120,
    products=("video", "debug"),
    out=Path("runs/go2"),
)
```

### Tier 2: Explicit Config User

Goal: use a reviewed workflow with customized backend/config.

```python
config = replace(
    get_preset("go2_video_ordered_static"),
    render_backend=RenderBackend.CUDA_DIRECT_LIGHT,
    accel_backend=AccelBackend.CUDA_LBVH,
)

runtime = create_static_asset_lab_runtime_for_config(config, output=output)
result = run_optical_lab_products(
    config=config,
    output=output,
    runtime=runtime,
    products=("video", "debug"),
    owns_runtime=True,
)
```

### Tier 3: Advanced Runtime Owner

Goal: caller owns physics runtime or custom runtime and only attaches products.

```python
with create_physics_body_triangle_lab_runtime(...) as runtime:
    result = run_optical_lab_products(
        config=config,
        runtime=runtime,
        products=[video_spec, observation_spec, debug_spec],
        output=output,
    )
```

### Tier 4: Future RL/Training User

Goal: action/policy loop owns stepping and consumes observation/render-backed
products.

```python
tick = runtime.step_tick(action)
obs = observation_product.consume(tick)
action = policy(obs)
```

P12.5 should document Tier 4 but not implement it.

## Documentation Deliverables

P12.5 should produce a coherent documentation pass:

### 1. Architecture Doc

Candidate:

```text
docs/OPTICAL_LAB_SYSTEM_ARCHITECTURE.md
```

Sections:

- physics ownership;
- render backend taxonomy;
- acceleration vs render vs delivery;
- device channel ownership;
- product workflow lifecycle;
- user API tiers;
- examples and benchmark roles;
- legacy/transitional paths.

### 2. Optical Lab README

Candidate:

```text
tools/optical_pipeline_lab/README.md
```

Sections:

- recommended entry points;
- how to run a preset;
- how to run explicit backend config;
- artifact layout;
- environment requirements;
- backend support matrix;
- known limitations.

### 3. Examples README

Candidate:

```text
examples/optical_lab/README.md
```

Sections:

- available examples;
- why Go2 static is a preset workflow example;
- why physics body triangle is the physics-owned example;
- how backend selection is exposed or not exposed yet;
- which examples are smoke tests vs benchmark examples.

### 4. MANIFEST

Update `MANIFEST.md` to reflect:

- P9 multi-product runtime;
- P10 workflow helpers;
- P11 preset workflow;
- P12 backend family;
- P13 pending path tracing;
- legacy/transitional static runner status.

## Code Deliverables

P12.5 code changes should be document-driven. Possible deliverables:

1. `optics/device_channel.py`
   - unified channel materialization boundary.

2. `tools/optical_pipeline_lab/render_backends.py`
   - backend compatibility and selection helpers.

3. `create_static_asset_lab_runtime_for_config(...)`
   - explicit config runtime helper for backend experiments.

4. Matrix/benchmark naming cleanup
   - ensure CSV summaries record `render_backend`, `accel_backend`, and user
     visible backend labels consistently.

5. Legacy status markers
   - document or deprecate transitional helpers without deleting active paths.

## Tests

P12.5 tests should focus on contracts, not new rendering behavior:

- device-channel boundary accepts Warp arrays, torch CUDA tensors, and host
  arrays where appropriate;
- CPU-only import safety;
- backend compatibility table accepts/rejects expected combinations;
- P11/P12 workflow entry points still write stable `scenario_config.json`;
- matrix/benchmark summary rows include backend fields consistently;
- README examples dry-run successfully;
- legacy/transitional paths still work or fail with documented messages.

## Review Questions

1. Is P12.5 correctly scoped as a system boundary consolidation pass rather than
   a narrow backend registry refactor?
2. Should P12.5 wait until after P12.4 benchmark/parity rows, or should some
   boundary cleanup begin immediately after P12.3?
3. Is `optics/device_channel.py` the right home for the device-channel
   materialization boundary?
4. Should backend compatibility live as executable code in
   `tools/optical_pipeline_lab/render_backends.py`, or remain documented inside
   `scenarios.py` until more backends exist?
5. Is `create_static_asset_lab_runtime_for_config(...)` the right internal
   helper for Tier 2 explicit backend users?
6. Which legacy paths should be documented as transitional rather than removed?
7. Should P12.5 include public `run_optical_lab_preset(..., render_backend=...)`,
   or should that remain P12.6 after docs and examples settle?
