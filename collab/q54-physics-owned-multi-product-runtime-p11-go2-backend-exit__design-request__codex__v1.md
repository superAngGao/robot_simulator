Initiative: q54-physics-owned-multi-product-runtime-p11
Stage: design-amendment-request
Author: Codex
Version: v1
Date: 2026-07-20
Status: in_review
Related Files: collab/q54-physics-owned-multi-product-runtime-p11__design-request__codex__v1.md, collab/q54-static-asset-source-naming-cleanup__implementation-note__codex__v1.md, collab/q54-optical-lab-render-foundation-c1-rename__implementation-note__codex__v1.md, collab/q54-optical-lab-render-foundation-alias-deletion__implementation-note__codex__v1.md, tools/optical_pipeline_lab/go2_backend.py, tools/optical_pipeline_lab/preset_products.py
Owner Summary: P11 should not preserve `go2_backend.py` as a hidden backend dependency. Go2/Menagerie is a scene/preset/example, not a system backend. P11 should finish the earlier Go2 naming refactor by moving the remaining generic runtime/video/reporting responsibilities into named lab components and making Go2 exercise the public P11 preset workflow as an example.

# Q54 P11 Design Amendment: Go2 Backend Exit

## Why This Amendment Exists

The original P11 design correctly adds a user-facing preset workflow above P10,
but P11.3 exposed a remaining architectural mismatch:

```text
preset_products.py
  -> go2_backend._build_video_camera
  -> go2_backend._pack_video_rgb8
  -> go2_backend.wp.synchronize_event
```

That is not a good long-term boundary. The current `go2_backend.py` name is a
historical artifact from the earliest Menagerie Go2 video benchmark. Since
then, the generic responsibilities have already been migrated into separate
lab components:

```text
render_session.py       # OpticalLabRender*
static_asset_source.py  # build_static_asset_render_source(...)
video_loop.py           # generic video render/delivery loop helpers
frame_contexts.py       # render frame providers
product_workflow.py     # P10 product workflow
product_specs.py        # P10 declarative products
```

P11 is now the main user-facing workflow layer. It should not depend on a module
named after one concrete robot model. Go2 should be a preset/example that calls
P11, not a backend that P11 calls.

## Prior Decisions To Preserve

Earlier naming work made three important decisions:

1. Go2/Menagerie is not the render pipeline.
2. Generic lab responsibilities should not use `Go2*` names.
3. `go2_backend.py` was temporarily allowed to remain only because it still
   owned concrete Menagerie CLI, camera, video, and reporting wrapper behavior.

This amendment does not contradict those decisions. It completes them.

The previous static asset cleanup explicitly said:

```text
Go2 remains a concrete Menagerie asset instance and CLI/reporting wrapper,
not the name of the static render source component.
```

P11 changes the context: once the official public workflow is
`run_optical_lab_preset(...)`, even CLI/reporting wrapper behavior should be
owned by P11-facing examples or generic lab modules. Keeping `go2_backend.py` as
a central dependency would keep the old benchmark-shaped architecture alive.

## Updated Completion Standard

The Go2 naming refactor is not complete until this is true:

```text
No core Optical Pipeline Lab workflow module depends on go2_backend.py.
Go2/Menagerie appears as preset metadata, static asset source data, and examples.
Go2 examples call the same P11 public workflow users call.
```

Concretely:

- `preset_products.py` must not import or reference `go2_backend`.
- `preset_workflows.py` must not import or reference `go2_backend`.
- `product_workflow.py`, `product_specs.py`, `preset_runtime.py`, and
  frame/product/runtime modules must not import or reference `go2_backend`.
- `runner.py` may keep a temporary legacy CLI bridge only until the P11 example
  and CLI migration are done.
- tests should stop using `go2_backend` for generic render/video behavior.
  Tests that specifically cover legacy Menagerie behavior may remain during the
  transition, but should be labeled as legacy.

## Revised Module Boundary

The remaining `go2_backend.py` responsibilities should be split into generic
components plus Go2 example/preset code.

Recommended structure:

```text
tools/optical_pipeline_lab/
  camera_presets.py     # fixed/orbit camera builders over scene bounds
  video_products.py     # reviewed VideoProductSpec factories for presets
  preset_products.py    # string/spec/frame-product resolver
  preset_runtime.py     # preset -> live runtime factory
  preset_workflows.py   # run_optical_lab_preset(...)
  static_asset_source.py
  render_session.py
  video_loop.py
```

Possible example structure:

```text
examples/optical_lab/
  README.md
  physics_body_triangle_video_debug.py
  physics_body_triangle_observation.py
  go2_video_ordered_static.py
```

The exact module names can change during review. The important boundary is that
generic components are named by responsibility, while Go2 remains a scene/preset
consumer of those components.

## What Moves Out Of `go2_backend.py`

### Camera Builders

The existing `_build_video_camera(...)` is not fundamentally Go2-specific. It
builds fixed/orbit cameras from scene bounds and frame identity. It should move
to a generic camera module.

Target shape:

```python
def build_lab_video_camera(scene, args, frame_index):
    ...
```

or, if the argument shape should be narrowed:

```python
def build_video_camera_for_scene_bounds(
    *,
    bounds_min,
    bounds_max,
    width: int,
    height: int,
    frame_id: int,
    sim_time: float,
    mode: str,
    frame_index: int,
    frame_count: int,
    view: str,
) -> OpticalPinholeCameraSpec:
    ...
```

The second form is cleaner but may create a larger diff. P11 can start with the
first form if tests keep the boundary explicit.

### RGB Packing / Synchronization

`_pack_video_rgb8` is already just a private alias to
`video_loop.pack_video_rgb8`. P11 product factories should use the generic
function directly.

Warp synchronization should be behind a small lab helper instead of reaching
through `go2_backend.wp`:

```python
from typing import Any


def synchronize_ready_event(event: Any) -> None:
    ...
```

Use `Any` rather than a concrete Warp type at this boundary so importing the
helper does not require Warp to be present. The implementation can lazily import
Warp and call `wp.synchronize_event(event)` when available.

This helper can live in `video_products.py`, `delivery.py`, or a small GPU/Warp
utility module. The goal is to avoid giving `go2_backend` ownership of GPU sync
for P11 products while keeping the public product factory importable in CPU-only
test environments.

### Video Product Factories

P11 should expose stable reviewed video spec factories:

```python
def create_physics_body_triangle_video_product_spec() -> VideoProductSpec:
    ...
```

`preset_products.py` should register these factories:

```python
_PRODUCT_STRING_FACTORIES = {
    "physics_body_triangle_video_smoke": {
        "video": create_physics_body_triangle_video_product_spec,
        "debug": DebugProductSpec,
    },
}
```

Do not show or register a Go2 video factory until `go2_video_ordered_static` is
actually runnable through the P11 workflow. If static Go2 needs a static-asset
workflow slice first, keep Go2 out of the P11 product registry until that
support lands.

### CLI / Example Wrapper

The current `go2_backend.main()` and `render_many_views(...)` path should not be
the durable public workflow. It can remain as a legacy bridge during migration,
but the target official example should call:

```python
run_optical_lab_preset(
    preset="go2_video_ordered_static",
    frames=120,
    products=("video", "debug"),
    out=Path("runs/examples/go2_video_ordered_static"),
)
```

If static Go2 cannot yet run through P11 because P11 initially supports only
physics-owned runtimes, then the example should be deferred or marked as a
legacy comparison path. Do not fake P11 support by keeping a hidden
`go2_backend.render_many_views(...)` bypass under the new API.

## P11 Scope Update

Original P11 scope was:

```text
physics_body_triangle_video_smoke preset
  -> runtime factory
  -> product resolver
  -> preset workflow
  -> examples
```

Updated P11 scope:

```text
public preset workflow
  -> physics-owned preset support first
  -> no new dependency on go2_backend
  -> Go2 becomes example/preset consumer when it can use the same workflow
```

This keeps physics-owned stepping as the primary dynamic path while making sure
the user-facing API is not shaped by the historical Go2 benchmark module.

## Revised Implementation Slices

### P11.3a: Remove P11 Product Resolver Dependency On Go2 Backend

Implement before P11.4:

- add `camera_presets.py` or equivalent generic camera helper module;
- add `video_products.py` or equivalent preset video spec factory module;
- move/copy the reviewed fixed/orbit video camera builder out of
  `go2_backend.py`;
- have `preset_products.py` depend on `video_products.py`, not `go2_backend.py`;
- update tests to assert `preset_products.py` does not import `go2_backend`;
- keep Go2 legacy tests only where they specifically cover legacy wrapper
  behavior.

Required tests:

- `"video"` for `physics_body_triangle_video_smoke` resolves without importing
  `tools.optical_pipeline_lab.go2_backend`;
- returned `VideoProductSpec` uses the generic camera builder and generic RGB
  pack helper;
- `"observation"` still fail-fast requires explicit spec;
- unknown strings and unsupported presets still fail fast.

### P11.4: Public Preset Workflow

Implement:

- `tools/optical_pipeline_lab/preset_workflows.py`;
- `run_optical_lab_preset(...)`;
- runtime ownership and cleanup;
- output/out handling by reusing `ArtifactOutput`;
- product selection through `resolve_lab_product_specs(...)`;
- execution through P10 `run_optical_lab_products(...)`.

Important rule:

```text
preset_workflows.py must not import go2_backend.py.
```

### P11.5: Examples

Add examples that call the P11 API directly:

```text
examples/optical_lab/README.md
examples/optical_lab/physics_body_triangle_video_debug.py
examples/optical_lab/physics_body_triangle_observation.py
```

Add Go2 example only when it can call P11 without a legacy backend bypass:

```text
examples/optical_lab/go2_video_ordered_static.py
```

If Go2 static still requires a different runner because it is not
physics-owned, then record that as a separate follow-up:

```text
P12: static-asset preset workflow through the same P11 surface
```

Do not call that P11 complete.

### P11.6: Legacy Go2 Backend Retirement

After P11 examples and any required static-asset workflow support are complete:

- remove `go2_backend.py`, or shrink it to a deprecated compatibility shim;
- remove generic tests that import `go2_backend`;
- update `runner.py` legacy `run_scenario(...)` delegation if still present;
- update `MANIFEST.md` and `GPU_OPTICAL_PIPELINE_DESIGN.md`.

P11.6 can be deferred if the team wants P11.4/P11.5 first, but the completion
standard should stay explicit: the refactor is not finished while
`go2_backend.py` remains a core dependency.

## Non-Goals

This amendment does not require:

- moving Menagerie asset loading into production `optics`;
- deleting Go2 presets or benchmark names immediately;
- making static Go2 physics-owned;
- adding a generic static-asset runtime unless P11 chooses to support static
  presets through the same workflow;
- changing render backend implementations;
- changing physics stepping ownership;
- inferring product defaults from arbitrary scenario config fields.

## Review Questions

1. Should the new video spec factory module be named `video_products.py`,
   `preset_video.py`, or something else?

2. Should camera builders move to `camera_presets.py` now, or should
   `video_products.py` own the first extracted camera builder until more camera
   presets exist?

3. Should P11 include static Go2 in the first `run_optical_lab_preset(...)`
   implementation, or should static-asset preset support become P12?

4. Should `go2_backend.py` be deleted in P11.6, or kept briefly as a deprecated
   shim for `examples/mujoco_menagerie_gpu_preview.py`?

5. How should generic tests migrate away from `go2_backend` without creating a
   noisy one-shot rewrite: move test sections alongside each extracted module,
   keep a temporary legacy test block, or split the current monolithic optical
   lab test file first?

## Codex Recommendation

Adopt this amendment before implementing P11.4.

Recommended binding decisions:

- do P11.3a before P11.4;
- add `video_products.py` for reviewed product spec factories;
- add `camera_presets.py` only if the extracted camera helper can be named
  cleanly without broad churn; otherwise keep the first camera helper in
  `video_products.py` and split it later;
- keep initial `run_optical_lab_preset(...)` focused on the physics-owned
  preset;
- do not add a Go2 P11 example until Go2 can call the same public workflow
  without `go2_backend.render_many_views(...)`;
- treat `go2_backend.py` retirement as the actual completion criterion for the
  Go2 naming refactor, even if deletion happens after P11.4.
