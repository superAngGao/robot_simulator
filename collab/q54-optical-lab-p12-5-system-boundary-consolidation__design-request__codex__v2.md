# Q54 P12.5 Design Request V2: System Boundary Consolidation

## Context

P9 through P12 progressively moved the Optical Pipeline Lab from ad hoc render
runner paths toward physics-owned, product-based, preset-driven workflows:

```text
P9   multi-product tick/runtime orchestration
P10  product workflow helpers and explicit runtime-owner path
P11  preset workflow surface and Go2 static path exit from legacy backend naming
P12  direct-light backend family: CPU / Warp / CUDA
```

P12.5 should not be a narrow "add a backend registry" refactor. It should be a
system-level consolidation pass across:

```text
Physics ownership
    -> published frame / simulation tick stream
Render ownership
    -> scene snapshot / acceleration / render backend / device channels
User workflow ownership
    -> presets / products / artifacts / examples / benchmarks
```

The goal is to make boundaries explicit, unique, executable where appropriate,
and documented. Any adapter/refactor introduced in P12.5 must live at a
necessary boundary between large components and should be the only place that
boundary is translated.

## V2 Changes From V1

V2 incorporates review feedback:

1. Split P12.5 into two implementation moments:
   - P12.5a Boundary Foundations, after P12.3;
   - P12.5b Documentation Consolidation, after P12.4.
2. Make `optics/device_channel.py` the preferred home for the device-channel
   materialization boundary.
3. Treat backend compatibility as executable code plus mirrored documentation.
4. Promote `create_static_asset_lab_runtime_for_config(...)` as the Tier 2
   explicit config helper.
5. Record RGB8 torch/Warp compatibility as a planned derived-channel problem,
   while keeping early `cuda_direct_light + rgb8` fail-fast acceptable.
6. Avoid naming legacy paths without confirming they exist in the repo.

## Timing and Slices

P12.5 should follow real backend behavior, but not every boundary foundation
must wait until all documentation is ready.

Recommended sequence:

```text
P12.3   CUDA direct-light backend
P12.5a  Boundary foundations needed by CUDA/direct-light family
P12.4   backend parity + benchmark/matrix contract
P12.5b  documentation + system consolidation
P12.6   optional public backend override + examples UX
P13     path tracing / accumulation / OptiX / non-direct-light rendering
```

Rationale:

- `optics/device_channel.py` and explicit-config runtime construction may be
  needed as soon as P12.3 returns native torch CUDA tensors.
- Complete backend support docs and MANIFEST state should wait until P12.4
  produces stable benchmark/parity rows.

## Goals

1. Re-express the physics/render/user workflow boundaries from a higher level.
2. Consolidate device-channel materialization into one system boundary.
3. Consolidate backend compatibility and selection rules into one reviewed
   executable location.
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

If conversion logic appears in multiple products, runners, examples, and
delivery helpers, P12.5 should pull it back into one boundary component.

### 2. Systematic Beats Small

The adapter or refactor does not have to be tiny. It has to be coherent. A
larger boundary module is better than many small scattered conversion helpers
if it gives the system one stable place to reason about ownership and
materialization.

### 3. Implementation Should Follow Real Backends

P12.5 should not invent abstractions ahead of implementation. It should extract
only boundaries proven by CPU/Warp/CUDA direct-light paths and benchmark
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

P12.5 documents this loop but does not implement it.

Architecture documentation should include a dataflow diagram showing ownership
and data direction across these three layers.

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

P12.5 should distinguish these axes clearly:

- `cuda_lbvh` is an acceleration backend, not a render backend;
- `cuda_direct_light` is a render backend, not an RGB8 pack shortcut;
- `path_tracer` belongs to P13 because it changes accumulation/sample
  semantics.

## Device Channel Boundary

P12.5a should introduce or finalize a single device-channel materialization
boundary.

Purpose:

```text
OpticalComputeResult(device channels)
    -> torch tensor view / host numpy staging / async readback / derived channels
```

It should support the real backend outputs:

```text
Warp array        from warp_bvh_direct_light
torch CUDA tensor from cuda_direct_light
host numpy array  from cpu_direct_light
```

Preferred module:

```text
optics/device_channel.py
```

Candidate functions:

```python
def channel_is_device(value: object) -> bool: ...
def channel_to_torch(value: object): ...
def channel_to_numpy(value: object): ...
def stage_channels_to_host(result: OpticalComputeResult, channels: Sequence[str]): ...
```

Possible later telemetry helper:

```python
def channel_device_name(value: object) -> str: ...
```

This is useful for debug/telemetry but should not block the first boundary
implementation.

Design constraints:

- Torch and Warp imports must be optional/lazy.
- CPU-only imports must remain safe.
- The boundary must not know about Optical Lab products or preset names.
- The boundary must not know about `cuda_direct_light` specifically.
- It should handle value representation, not backend policy.

Affected call sites:

- `optics.device.stage_optical_compute_result_to_host(...)`
- `optics.device.stage_optical_channels(...)`
- `tools.optical_pipeline_lab.async_readback.TorchAsyncReadbackRing`
- RGB8 pack or derived-channel paths, once torch tensor support is enabled.

### RGB8 Pack Boundary

Early `cuda_direct_light + rgb8` may remain fail-fast. The validation layer
should reject it with a clear "not yet supported" message rather than letting a
Warp pack kernel fail later.

The planned compatibility path is:

```python
if rgb is torch CUDA tensor:
    rgb_wp = wp.from_torch(rgb)  # zero-copy view for existing Warp RGB8 pack
```

Using `wp.from_torch(...)` inside the derived-channel pack helper is acceptable:
it does not reverse CUDA executor ownership. It is a local staging adaptation
for an existing Warp RGB8 kernel.

P12.5 should record this path, but it does not have to implement RGB8 support
before the core channel boundary and CUDA direct-light parity are stable.

## Backend Compatibility Boundary

P12.5 should consolidate compatibility validation currently distributed across
scenario config and run options.

Preferred module:

```text
tools/optical_pipeline_lab/render_backends.py
```

Responsibilities:

- supported render backend list;
- render backend -> allowed acceleration backends;
- render backend -> allowed geometry/frame-source modes;
- render backend -> run option restrictions;
- reserved/future backend fail-fast messages.

Compatibility should be executable code and mirrored in user-readable docs.

Candidate shape:

```python
@dataclass(frozen=True)
class BackendSpec:
    accel: frozenset[str]
    geometry: frozenset[str]
    raygen: frozenset[str]
    delivery: frozenset[str]

BACKEND_SPECS = {
    "cpu_direct_light": BackendSpec(
        accel=frozenset({"cpu_bvh"}),
        geometry=frozenset({"static"}),
        raygen=frozenset({"host"}),
        delivery=frozenset({"sync"}),
    ),
    "warp_bvh_direct_light": BackendSpec(
        accel=frozenset({"cpu_bvh", "cuda_lbvh"}),
        geometry=frozenset({"static", "dynamic_rigid"}),
        raygen=frozenset({"host", "gpu"}),
        delivery=frozenset({"sync", "device_only", "torch_async"}),
    ),
    "cuda_direct_light": BackendSpec(
        accel=frozenset({"cuda_lbvh"}),
        geometry=frozenset({"static"}),
        raygen=frozenset({"host", "gpu"}),
        delivery=frozenset({"sync", "torch_async"}),
    ),
}
```

The exact `cuda_direct_light` allowed values should reflect what P12.3 actually
implements. For example, if P12.3 initially supports host raygen and sync only,
the table should say so until camera raygen and async readback are real.

Preset compatibility should be validated in workflow entry points, not
necessarily inside `get_preset(...)`.

Reason:

- `get_preset(...)` should remain a lightweight config factory.
- Runtime support can depend on current backend state and run options.
- Workflow validation is the right place to combine config and output options.

Open decision:

```text
Should get_preset(...) ever validate backend compatibility for implemented
presets, or should compatibility validation stay entirely in workflow/run
entry points?
```

Recommended answer:

```text
Keep get_preset(...) pure for now. Validate compatibility in workflow/run
entry points and add tests that every built-in preset validates there.
```

## User Workflow Boundary

P12.5 should define the intended API tiers.

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

P12.5 should make this tier explicit through:

```python
def create_static_asset_lab_runtime_for_config(
    config: OpticalLabScenarioConfig,
    *,
    output: ArtifactOutput,
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> StaticAssetLabRuntime:
    """Create runtime from explicit config for backend experiments."""
```

Example:

```python
config = replace(
    get_preset("go2_video_ordered_static"),
    render_backend=RenderBackend.CUDA_DIRECT_LIGHT,
    accel_backend=AccelBackend.CUDA_LBVH,
)

output = ArtifactOutput(root=Path("runs/cuda-direct"), frames=120)
runtime = create_static_asset_lab_runtime_for_config(config, output=output)

result = run_optical_lab_products(
    config=config,
    output=output,
    runtime=runtime,
    products=("video", "debug"),
    owns_runtime=True,
)
```

`create_static_asset_lab_runtime(preset, ...)` should delegate to the explicit
config helper after resolving the preset.

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

P12.5 documents Tier 4 but does not implement it.

## Documentation Deliverables

P12.5b should produce a coherent documentation pass.

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
- legacy/transitional paths;
- dataflow diagram.

### 2. Backend Support Matrix

Candidate:

```text
docs/OPTICAL_LAB_BACKEND_SUPPORT_MATRIX.md
```

This should mirror `tools/optical_pipeline_lab/render_backends.py` in
user-readable form. It can be generated later; manual mirroring is acceptable
while the table is small.

### 3. Optical Lab README

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

### 4. Examples README

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

### 5. MANIFEST

Update `MANIFEST.md` to reflect:

- P9 multi-product runtime;
- P10 workflow helpers;
- P11 preset workflow;
- P12 backend family;
- P13 pending path tracing;
- legacy/transitional static runner status.

### 6. Backend Contribution Note

Preferred location:

```text
docs/OPTICAL_LAB_SYSTEM_ARCHITECTURE.md
```

as a section at the end of the architecture doc:

```text
Adding a new render backend
```

Reason:

- this is Optical Lab-specific architecture knowledge, not a general project
  contribution rule;
- the architecture doc should be self-contained for backend authors;
- readers should not need to jump between system docs and global contribution
  docs to add one backend correctly.

Checklist:

1. Update `tools/optical_pipeline_lab/render_backends.py`.
2. Update backend support matrix docs.
3. Add backend-specific tests.
4. Add parity/benchmark rows.
5. Update MANIFEST if this is a new backend family.

## Code Deliverables

P12.5 code changes should be document-driven.

### P12.5a Boundary Foundations

1. `optics/device_channel.py`
   - unified channel materialization boundary.

2. `create_static_asset_lab_runtime_for_config(...)`
   - explicit config runtime helper for backend experiments.

3. Initial `tools/optical_pipeline_lab/render_backends.py`
   - executable compatibility table for implemented P12 direct-light backends.

4. Tests for:
   - CPU-only import safety;
   - channel materialization with host arrays, Warp arrays, and torch CUDA
     tensors where available;
   - compatibility validation;
   - explicit config runtime helper.

### P12.5b Documentation Consolidation

1. `docs/OPTICAL_LAB_SYSTEM_ARCHITECTURE.md`
2. `docs/OPTICAL_LAB_BACKEND_SUPPORT_MATRIX.md`
3. `tools/optical_pipeline_lab/README.md`
4. `examples/optical_lab/README.md`
5. `MANIFEST.md`
6. "Adding a new render backend" section in the architecture doc.
7. legacy/transitional status markers for confirmed legacy paths.

## Legacy and Transitional Paths

P12.5 should document legacy paths only after confirming they exist in the repo.

Known likely candidates to examine:

```text
tools/optical_pipeline_lab/menagerie_static_runner.py
tools/optical_pipeline_lab/runner.py::run_scenario(...)
any old direct runner paths discovered by rg
```

Do not copy path names from review notes without verifying them.

Suggested statuses:

```text
Active:
  P10/P11 product workflows
  P11 preset workflows
  physics-owned product paths

Active but narrow:
  examples used as smoke tests
  explicit physics body triangle demos

Transitional:
  legacy runner/CLI paths retained for benchmark compatibility
```

P12.5 should document status before deleting anything.

## Tests

P12.5 tests should focus on contracts, not new rendering behavior:

- device-channel boundary accepts Warp arrays, torch CUDA tensors, and host
  arrays where appropriate;
- CPU-only import safety;
- backend compatibility table accepts/rejects expected combinations;
- built-in presets validate through workflow/run validation;
- P11/P12 workflow entry points still write stable `scenario_config.json`;
- matrix/benchmark summary rows include backend fields consistently;
- README examples dry-run successfully;
- legacy/transitional paths still work or fail with documented messages.

## Review Questions

1. Is the V2 split into P12.5a Boundary Foundations and P12.5b Documentation
   Consolidation correct?
2. Is `optics/device_channel.py` the right home for the device-channel
   materialization boundary?
3. Should `get_preset(...)` stay a pure factory, with backend compatibility
   validation kept in workflow/run entry points?
4. Should `tools/optical_pipeline_lab/render_backends.py` become the executable
   source of backend compatibility truth in P12.5a, or should that wait until
   P12.5b after benchmark rows?
5. Is `create_static_asset_lab_runtime_for_config(...)` the right internal
   helper for Tier 2 explicit backend users?
6. Is it acceptable to keep `cuda_direct_light + rgb8` fail-fast until the
   derived-channel pack path supports torch tensors?
7. Should P12.5 include public `run_optical_lab_preset(..., render_backend=...)`,
   or should that remain P12.6 after docs and examples settle?
